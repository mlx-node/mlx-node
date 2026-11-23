//! Engram table rows (F8_E4M3, `head_dim` bytes per row) read by `pread`
//! through the load's one shard set (`F_NOCACHE` + `F_RDAHEAD 0`; never
//! mapped, never a Metal buffer). Misses are coalesced into 16 KiB-aligned
//! page spans up to `MAX_COALESCE_BYTES`, served by a persistent worker pool
//! with one reusable `AlignedBuf` per worker and a bounded raw-F8 LRU cache.
//!
//! The file says type 43, which mlx-node parses as MXFP8. The table is read
//! as F8_E4M3 only when its bytes to the end of the shard are exactly
//! `rows · head_dim`; the MXFP8 size (`rows · head_dim / 32 · 33`) and any
//! other size are refused.

use std::collections::{HashMap, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
#[cfg(test)]
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::Instant;

use napi::{Error, Result};

use super::f8::decode_f8_rows;
#[cfg(test)]
use super::scan::Scan;
use super::scan::{ENGRAM_TABLE, Located};
use crate::models::engram_hash::EngramConfig;
use crate::moe_stream::io::{AlignedBuf, Extent, ShardSet};
use crate::utils::gguf::GgufTensorType;

const ALIGN: u64 = 16 << 10;
pub(crate) const MAX_COALESCE_BYTES: u64 = 256 << 10;
pub const DEFAULT_ENGRAM_THREADS: usize = 16;
pub const ENGRAM_CACHE_ENV: &str = "MLX_HY4_ENGRAM_CACHE_MB";
pub const DEFAULT_ENGRAM_CACHE_MB: usize = 256;

const NONE: u32 = u32::MAX;
/// Per-entry index budget alongside `head_dim` raw bytes and `LruNode` (12 B):
/// at `head_dim = 96`, 128 B/entry fits 2,097,152 rows inside 256 MiB.
const CACHE_INDEX_OVERHEAD: usize = 20;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 Engram rows: {msg}"))
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Decoded rows `[ids.len(), head_dim]` (f32, exact) for global row ids of
/// the one table, laid out `[pos][col]` for Engram layer `layer` (its index
/// in the Engram layer list).
pub trait EngramRows {
    fn rows(&self, layer: usize, ids: &[u32]) -> Result<Vec<f32>>;

    fn prefetch(&self, _layer: usize, _ids: &[u32]) -> Result<Option<PrefetchHandle>> {
        Ok(None)
    }

    fn take_prefetched(&self, handle: PrefetchHandle) -> Result<Vec<f32>> {
        handle.wait()
    }
}

/// Accepts `extent` (bytes from the table start to the shard end) only as an
/// F8_E4M3 table of `n_rows × head_dim`.
pub fn prove_f8_extent(extent: u64, n_rows: u64, head_dim: usize) -> Result<()> {
    if n_rows == 0 || head_dim == 0 {
        return Err(bad(format!("empty table [{n_rows}, {head_dim}]")));
    }
    let f8 = n_rows
        .checked_mul(head_dim as u64)
        .ok_or_else(|| bad("table size overflows u64"))?;
    if extent == f8 {
        return Ok(());
    }
    let mxfp8 = head_dim
        .is_multiple_of(32)
        .then(|| n_rows.checked_mul(head_dim as u64 / 32 * 33))
        .flatten();
    if mxfp8 == Some(extent) {
        return Err(bad(format!(
            "{extent} bytes is an MXFP8 table [{n_rows}, {head_dim}], not F8_E4M3"
        )));
    }
    Err(bad(format!(
        "{extent} bytes to the end of the shard; F8_E4M3 [{n_rows}, {head_dim}] is {f8}"
    )))
}

/// One coalesced 16 KiB-aligned page read covering one or more sorted unique
/// miss row IDs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PageSpan {
    pub(crate) start: u64,
    pub(crate) end: u64,
    pub(crate) rows: Vec<u32>,
}

/// Coalesces unique miss row IDs into 16 KiB-aligned windows clipped at
/// `file_len`, merging consecutive windows whenever `next_start <= cur_end`
/// and the merged length is at most `MAX_COALESCE_BYTES`.
pub(crate) fn coalesce_spans(
    miss_ids: &[u32],
    base: u64,
    head_dim: usize,
    file_len: u64,
) -> Vec<PageSpan> {
    if miss_ids.is_empty() || head_dim == 0 {
        return Vec::new();
    }
    let mut sorted = miss_ids.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let hd = head_dim as u64;
    let mut spans: Vec<PageSpan> = Vec::new();
    for id in sorted {
        let off = base + u64::from(id) * hd;
        let win_start = off & !(ALIGN - 1);
        let win_end = (off + hd)
            .checked_next_multiple_of(ALIGN)
            .unwrap_or(file_len)
            .min(file_len);
        if let Some(cur) = spans.last_mut()
            && win_start <= cur.end
        {
            let merged_end = cur.end.max(win_end);
            if merged_end - cur.start <= MAX_COALESCE_BYTES {
                cur.end = merged_end;
                cur.rows.push(id);
                continue;
            }
        }
        spans.push(PageSpan {
            start: win_start,
            end: win_end,
            rows: vec![id],
        });
    }
    spans
}

struct LruNode {
    key: u32,
    prev: u32,
    next: u32,
}

/// Bounded raw-F8 LRU cache backed by a flat byte slab + intrusive doubly-linked
/// list. Only NaN-validated raw rows are inserted.
pub(crate) struct F8RowCache {
    head_dim: usize,
    cap: usize,
    head: u32,
    tail: u32,
    nodes: Vec<LruNode>,
    slab: Vec<u8>,
    index: HashMap<u32, u32>,
}

impl F8RowCache {
    pub(crate) fn new(cap_rows: usize, head_dim: usize) -> Self {
        let cap = if head_dim == 0 {
            0
        } else {
            cap_rows.min((NONE - 1) as usize)
        };
        Self {
            head_dim,
            cap,
            head: NONE,
            tail: NONE,
            nodes: Vec::new(),
            slab: Vec::new(),
            index: HashMap::new(),
        }
    }

    pub(crate) fn from_budget_bytes(budget_bytes: usize, head_dim: usize) -> Self {
        if budget_bytes == 0 || head_dim == 0 {
            return Self::new(0, head_dim);
        }
        let entry_bytes = head_dim + std::mem::size_of::<LruNode>() + CACHE_INDEX_OVERHEAD;
        Self::new(budget_bytes / entry_bytes, head_dim)
    }

    pub(crate) fn capacity(&self) -> usize {
        self.cap
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.index.len()
    }

    #[cfg(test)]
    pub(crate) fn contains(&self, row_id: u32) -> bool {
        self.index.contains_key(&row_id)
    }

    fn move_to_front(&mut self, slot: u32) {
        if self.head == slot {
            return;
        }
        let prev = self.nodes[slot as usize].prev;
        let next = self.nodes[slot as usize].next;
        if prev != NONE {
            self.nodes[prev as usize].next = next;
        }
        if next != NONE {
            self.nodes[next as usize].prev = prev;
        } else {
            self.tail = prev;
        }
        let old_head = self.head;
        self.nodes[slot as usize].prev = NONE;
        self.nodes[slot as usize].next = old_head;
        if old_head != NONE {
            self.nodes[old_head as usize].prev = slot;
        }
        self.head = slot;
    }

    pub(crate) fn get(&mut self, row_id: u32) -> Option<&[u8]> {
        if self.cap == 0 {
            return None;
        }
        let &slot = self.index.get(&row_id)?;
        self.move_to_front(slot);
        let off = slot as usize * self.head_dim;
        Some(&self.slab[off..off + self.head_dim])
    }

    pub(crate) fn insert(&mut self, row_id: u32, raw: &[u8]) {
        if self.cap == 0 || raw.len() != self.head_dim {
            return;
        }
        if let Some(&slot) = self.index.get(&row_id) {
            let off = slot as usize * self.head_dim;
            self.slab[off..off + self.head_dim].copy_from_slice(raw);
            self.move_to_front(slot);
            return;
        }
        if self.nodes.len() < self.cap {
            if self.nodes.len() == self.nodes.capacity() {
                // Bounded geometric growth avoids both eager allocation on small tables
                // and Vec 2x overshoot past the configured MiB cap.
                let next_cap = (self.nodes.capacity().max(64) * 2).min(self.cap);
                let add = next_cap - self.nodes.len();
                self.nodes.reserve_exact(add);
                self.slab.reserve_exact(add * self.head_dim);
            }
            let slot = self.nodes.len() as u32;
            let old_head = self.head;
            self.nodes.push(LruNode {
                key: row_id,
                prev: NONE,
                next: old_head,
            });
            self.slab.extend_from_slice(raw);
            if old_head != NONE {
                self.nodes[old_head as usize].prev = slot;
            } else {
                self.tail = slot;
            }
            self.head = slot;
            self.index.insert(row_id, slot);
            return;
        }
        let slot = self.tail;
        let evicted = self.nodes[slot as usize].key;
        self.index.remove(&evicted);
        self.nodes[slot as usize].key = row_id;
        let off = slot as usize * self.head_dim;
        self.slab[off..off + self.head_dim].copy_from_slice(raw);
        self.index.insert(row_id, slot);
        self.move_to_front(slot);
    }
}

pub(crate) fn parse_cache_mb(raw: Option<&str>) -> Result<usize> {
    let Some(v) = raw.map(str::trim).filter(|v| !v.is_empty()) else {
        return Ok(DEFAULT_ENGRAM_CACHE_MB);
    };
    v.parse::<usize>().map_err(|_| {
        bad(format!(
            "{ENGRAM_CACHE_ENV}={v:?}: expected a non-negative integer (MiB)"
        ))
    })
}

/// Which global rows each (layer, column) may address.
struct Columns {
    cols: usize,
    n_rows: u64,
    offsets: Vec<u64>,
    sizes: Vec<u64>,
}

impl Columns {
    fn new(cfg: &EngramConfig, n_rows: u64) -> Result<Self> {
        let cols = cfg.n_cols();
        let want = cols * cfg.layers.len();
        if cols == 0 || cfg.head_offsets.len() < want || cfg.head_vocab_sizes.len() < want {
            return Err(bad("head offsets / sizes do not cover every column"));
        }
        if n_rows > u64::from(u32::MAX) + 1 {
            return Err(bad(format!("{n_rows} rows exceed u32 row ids")));
        }
        Ok(Self {
            cols,
            n_rows,
            offsets: cfg.head_offsets.clone(),
            sizes: cfg.head_vocab_sizes.clone(),
        })
    }

    fn check(&self, layer: usize, ids: &[u32]) -> Result<()> {
        if !ids.len().is_multiple_of(self.cols) {
            return Err(bad(format!("{} ids for {} columns", ids.len(), self.cols)));
        }
        for (i, &id) in ids.iter().enumerate() {
            let g = layer * self.cols + i % self.cols;
            let (lo, n) = match (self.offsets.get(g), self.sizes.get(g)) {
                (Some(&lo), Some(&n)) => (lo, n),
                _ => return Err(bad(format!("Engram layer {layer} has no column {g}"))),
            };
            let id = u64::from(id);
            if id < lo || id >= lo + n || id >= self.n_rows {
                return Err(bad(format!(
                    "row {id} outside column {g}'s [{lo}, {}) (table {} rows)",
                    lo + n,
                    self.n_rows
                )));
            }
        }
        Ok(())
    }
}

#[derive(Default)]
struct QueueState {
    jobs: VecDeque<SpanJob>,
    shutdown: bool,
}

struct TableShared {
    shards: Arc<ShardSet>,
    shard: usize,
    /// File offset of row 0.
    base: u64,
    file_len: u64,
    head_dim: usize,
    cache_enabled: bool,
    cache: Mutex<F8RowCache>,
    queue: Mutex<QueueState>,
    ready: Condvar,
    bytes: AtomicU64,
    ns: AtomicU64,
    #[cfg(test)]
    next_done_seq: AtomicU64,
}

struct BatchState {
    remaining: usize,
    output: Vec<f32>,
    miss_raw: Vec<u8>,
    err: Option<Error>,
    bytes_read: u64,
    wait_ns: u64,
    #[cfg(test)]
    done_seq: u64,
}

struct EngramBatch {
    cancelled: AtomicBool,
    state: Mutex<BatchState>,
    done: Condvar,
    shared: Arc<TableShared>,
    ids: Vec<u32>,
    uniq: Vec<u32>,
    miss_ids: Vec<u32>,
    ids_are_uniq_order: bool,
}

/// Handle for an in-flight asynchronous Engram row fetch.
/// Dropping without calling [`PrefetchHandle::wait`] sets `cancelled` without
/// blocking; workers hold an `Arc` to the batch and safely drop or skip work.
pub struct PrefetchHandle {
    batch: Arc<EngramBatch>,
    waited: bool,
}

impl PrefetchHandle {
    /// Waits for all coalesced miss spans to finish, inserts validated raw F8
    /// misses into the LRU cache, and returns `[ids.len(), head_dim]` f32 rows.
    pub fn wait(mut self) -> Result<Vec<f32>> {
        self.waited = true;
        let t0 = Instant::now();
        let mut st = lock(&self.batch.state);
        while st.remaining > 0 {
            st = self
                .batch
                .done
                .wait(st)
                .unwrap_or_else(PoisonError::into_inner);
        }
        if let Some(err) = st.err.take() {
            let wait_ns = t0.elapsed().as_nanos() as u64;
            st.wait_ns = wait_ns;
            self.batch.shared.ns.fetch_add(wait_ns, Ordering::Relaxed);
            return Err(err);
        }
        let uniq_dec = std::mem::take(&mut st.output);
        let miss_raw = std::mem::take(&mut st.miss_raw);
        drop(st);

        let hd = self.batch.shared.head_dim;
        if self.batch.shared.cache_enabled && !self.batch.miss_ids.is_empty() {
            let mut cache = lock(&self.batch.shared.cache);
            for (i, &row_id) in self.batch.miss_ids.iter().enumerate() {
                cache.insert(row_id, &miss_raw[i * hd..(i + 1) * hd]);
            }
        }

        let out = if self.batch.ids_are_uniq_order {
            uniq_dec
        } else {
            let mut out = vec![0f32; self.batch.ids.len() * hd];
            for (&id, o) in self.batch.ids.iter().zip(out.chunks_mut(hd)) {
                let k = self
                    .batch
                    .uniq
                    .binary_search(&id)
                    .map_err(|_| bad("lost row"))?;
                o.copy_from_slice(&uniq_dec[k * hd..(k + 1) * hd]);
            }
            out
        };

        let wait_ns = t0.elapsed().as_nanos() as u64;
        lock(&self.batch.state).wait_ns = wait_ns;
        self.batch.shared.ns.fetch_add(wait_ns, Ordering::Relaxed);
        Ok(out)
    }

    #[cfg(test)]
    pub(crate) fn done_seq(&self) -> u64 {
        let mut st = lock(&self.batch.state);
        while st.remaining > 0 {
            st = self
                .batch
                .done
                .wait(st)
                .unwrap_or_else(PoisonError::into_inner);
        }
        st.done_seq
    }
}

impl Drop for PrefetchHandle {
    fn drop(&mut self) {
        if !self.waited {
            self.batch.cancelled.store(true, Ordering::Relaxed);
        }
    }
}

struct SpanJob {
    batch: Arc<EngramBatch>,
    start: u64,
    end: u64,
    /// `(miss_idx, uniq_idx, row_id)` for every row covered by this span.
    targets: Vec<(usize, usize, u32)>,
}

fn next_job(shared: &TableShared) -> Option<SpanJob> {
    let mut q = lock(&shared.queue);
    loop {
        if let Some(job) = q.jobs.pop_front() {
            return Some(job);
        }
        if q.shutdown {
            return None;
        }
        q = shared.ready.wait(q).unwrap_or_else(PoisonError::into_inner);
    }
}

fn run_span(
    shared: &TableShared,
    job: &SpanJob,
    buf: &mut AlignedBuf,
) -> Result<(Vec<u8>, Vec<f32>, u64)> {
    let span_len = job.end - job.start;
    let e = Extent {
        shard: shared.shard,
        offset: job.start,
        len: span_len,
    };
    let page_bytes = shared.shards.read_aligned(e, buf)?;
    shared.bytes.fetch_add(span_len, Ordering::Relaxed);
    if job.batch.cancelled.load(Ordering::Relaxed) {
        return Ok((Vec::new(), Vec::new(), span_len));
    }
    let hd = shared.head_dim;
    let n = job.targets.len();
    let mut raw_out = if shared.cache_enabled {
        Vec::with_capacity(n * hd)
    } else {
        Vec::new()
    };
    let mut dec_out = vec![0f32; n * hd];
    for (i, &(_miss_idx, _uniq_idx, row_id)) in job.targets.iter().enumerate() {
        let off = shared.base + u64::from(row_id) * hd as u64;
        let rel = (off - job.start) as usize;
        let raw = &page_bytes[rel..rel + hd];
        decode_f8_rows(raw, &mut dec_out[i * hd..(i + 1) * hd])
            .map_err(|err| bad(format!("row {row_id}: {}", err.reason)))?;
        if shared.cache_enabled {
            raw_out.extend_from_slice(raw);
        }
    }
    Ok((raw_out, dec_out, span_len))
}

fn worker(shared: &TableShared) {
    let mut buf = AlignedBuf::new();
    let hd = shared.head_dim;
    while let Some(job) = next_job(shared) {
        if job.batch.cancelled.load(Ordering::Relaxed) {
            let mut st = lock(&job.batch.state);
            st.remaining -= 1;
            if st.remaining == 0 {
                #[cfg(test)]
                {
                    st.done_seq = shared.next_done_seq.fetch_add(1, Ordering::Relaxed) + 1;
                }
                job.batch.done.notify_all();
            }
            continue;
        }
        let res = catch_unwind(AssertUnwindSafe(|| run_span(shared, &job, &mut buf)))
            .unwrap_or_else(|_| Err(bad("reader panicked")));
        let mut st = lock(&job.batch.state);
        match res {
            Ok((raw_out, dec_out, bytes_read)) => {
                st.bytes_read += bytes_read;
                if !job.batch.cancelled.load(Ordering::Relaxed) && st.err.is_none() {
                    for (i, &(miss_idx, uniq_idx, _row_id)) in job.targets.iter().enumerate() {
                        st.output[uniq_idx * hd..(uniq_idx + 1) * hd]
                            .copy_from_slice(&dec_out[i * hd..(i + 1) * hd]);
                        if shared.cache_enabled {
                            st.miss_raw[miss_idx * hd..(miss_idx + 1) * hd]
                                .copy_from_slice(&raw_out[i * hd..(i + 1) * hd]);
                        }
                    }
                }
            }
            Err(e) => {
                st.err.get_or_insert(e);
                job.batch.cancelled.store(true, Ordering::Relaxed);
            }
        }
        st.remaining -= 1;
        if st.remaining == 0 {
            #[cfg(test)]
            {
                st.done_seq = shared.next_done_seq.fetch_add(1, Ordering::Relaxed) + 1;
            }
            job.batch.done.notify_all();
        }
    }
}

/// The real table, read by coalesced 16 KiB page spans through a persistent
/// worker pool with a bounded raw-F8 LRU cache.
pub struct F8Table {
    shared: Arc<TableShared>,
    columns: Columns,
    workers: Vec<JoinHandle<()>>,
}

impl F8Table {
    #[cfg(test)]
    pub fn open(scan: &Scan) -> Result<Self> {
        Self::open_in(
            Arc::new(ShardSet::open(&scan.paths)?),
            &scan.engram_table,
            &scan.cfg.engram,
        )
    }

    /// `loc` is `per_layer_token_embd.weight` in the shard at `path`.
    #[cfg(test)]
    pub(crate) fn open_at(path: &Path, loc: &Located, cfg: &EngramConfig) -> Result<Self> {
        Self::on_shard(Arc::new(ShardSet::open(&[path])?), 0, loc, cfg)
    }

    /// `loc` is `per_layer_token_embd.weight` in shard `loc.shard` of
    /// `shards`, the descriptors every other reader of the load uses.
    pub fn open_in(shards: Arc<ShardSet>, loc: &Located, cfg: &EngramConfig) -> Result<Self> {
        Self::on_shard(shards, loc.shard, loc, cfg)
    }

    fn on_shard(
        shards: Arc<ShardSet>,
        shard: usize,
        loc: &Located,
        cfg: &EngramConfig,
    ) -> Result<Self> {
        let cache_mb = parse_cache_mb(std::env::var(ENGRAM_CACHE_ENV).ok().as_deref())?;
        let budget_bytes = cache_mb
            .checked_mul(1 << 20)
            .ok_or_else(|| bad(format!("{ENGRAM_CACHE_ENV}={cache_mb}: overflows usize")))?;
        let cache = F8RowCache::from_budget_bytes(budget_bytes, cfg.head_dim);
        Self::build(shards, shard, loc, cfg, DEFAULT_ENGRAM_THREADS, cache)
    }

    #[cfg(test)]
    pub(crate) fn open_with_options(
        shards: Arc<ShardSet>,
        shard: usize,
        loc: &Located,
        cfg: &EngramConfig,
        threads: usize,
        cache_rows: usize,
    ) -> Result<Self> {
        let cache = F8RowCache::new(cache_rows, cfg.head_dim);
        Self::build(shards, shard, loc, cfg, threads, cache)
    }

    fn build(
        shards: Arc<ShardSet>,
        shard: usize,
        loc: &Located,
        cfg: &EngramConfig,
        threads: usize,
        cache: F8RowCache,
    ) -> Result<Self> {
        if threads == 0 {
            return Err(bad("0 engram worker threads"));
        }
        let n_rows = table_rows(loc, cfg)?;
        let columns = Columns::new(cfg, n_rows)?;
        let (file, path) = shards
            .file(shard)
            .zip(shards.path(shard))
            .ok_or_else(|| bad(format!("no shard {shard} of {}", shards.len())))?;
        let file_len = file
            .metadata()
            .map_err(|e| bad(format!("stat '{}': {e}", path.display())))?
            .len();
        let base = loc.abs_offset()?;
        let extent = file_len.checked_sub(base).ok_or_else(|| {
            bad(format!(
                "'{}': table starts at {base}, past the end ({file_len})",
                path.display()
            ))
        })?;
        prove_f8_extent(extent, n_rows, cfg.head_dim)
            .map_err(|e| bad(format!("'{}': {}", path.display(), e.reason)))?;
        let shared = Arc::new(TableShared {
            shards,
            shard,
            base,
            file_len,
            head_dim: cfg.head_dim,
            cache_enabled: cache.capacity() > 0,
            cache: Mutex::new(cache),
            queue: Mutex::new(QueueState::default()),
            ready: Condvar::new(),
            bytes: AtomicU64::new(0),
            ns: AtomicU64::new(0),
            #[cfg(test)]
            next_done_seq: AtomicU64::new(0),
        });
        // Construct Self before spawning so a failed spawn drops Self and joins
        // workers already running.
        let mut table = Self {
            shared,
            columns,
            workers: Vec::with_capacity(threads),
        };
        for i in 0..threads {
            let shared = Arc::clone(&table.shared);
            let handle = std::thread::Builder::new()
                .name(format!("hy4-engram-{i}"))
                .spawn(move || worker(&shared))
                .map_err(|e| bad(format!("spawn engram worker {i}: {e}")))?;
            table.workers.push(handle);
        }
        Ok(table)
    }

    pub fn n_rows(&self) -> u64 {
        self.columns.n_rows
    }

    /// Real aligned bytes read from disk and foreground nanoseconds spent, since open.
    pub fn read_stats(&self) -> (u64, u64) {
        (
            self.shared.bytes.load(Ordering::Relaxed),
            self.shared.ns.load(Ordering::Relaxed),
        )
    }

    #[cfg(test)]
    pub(crate) fn cache_len(&self) -> usize {
        lock(&self.shared.cache).len()
    }

    #[cfg(test)]
    pub(crate) fn cache_capacity(&self) -> usize {
        lock(&self.shared.cache).capacity()
    }

    /// Submits coalesced 16 KiB-aligned misses to the persistent worker pool
    /// after serving raw-F8 LRU hits under a single lock.
    pub fn prefetch_rows(&self, ids: &[u32]) -> Result<PrefetchHandle> {
        let t0 = Instant::now();
        if let Some(&id) = ids.iter().find(|&&id| u64::from(id) >= self.columns.n_rows) {
            return Err(bad(format!(
                "row {id} outside the table ({} rows)",
                self.columns.n_rows
            )));
        }
        if ids.is_empty() {
            let batch = Arc::new(EngramBatch {
                cancelled: AtomicBool::new(false),
                state: Mutex::new(BatchState {
                    remaining: 0,
                    output: Vec::new(),
                    miss_raw: Vec::new(),
                    err: None,
                    bytes_read: 0,
                    wait_ns: 0,
                    #[cfg(test)]
                    done_seq: 0,
                }),
                done: Condvar::new(),
                shared: Arc::clone(&self.shared),
                ids: Vec::new(),
                uniq: Vec::new(),
                miss_ids: Vec::new(),
                ids_are_uniq_order: true,
            });
            return Ok(PrefetchHandle {
                batch,
                waited: false,
            });
        }

        let hd = self.shared.head_dim;
        let ids_are_uniq_order = ids.windows(2).all(|w| w[0] < w[1]);
        let uniq = if ids_are_uniq_order {
            ids.to_vec()
        } else {
            let mut u = ids.to_vec();
            u.sort_unstable();
            u.dedup();
            u
        };

        let mut uniq_dec = vec![0f32; uniq.len() * hd];
        let mut misses: Vec<(usize, u32)> = Vec::new();
        if self.shared.cache_enabled {
            let mut cache = lock(&self.shared.cache);
            for (u_idx, &id) in uniq.iter().enumerate() {
                if let Some(raw) = cache.get(id) {
                    decode_f8_rows(raw, &mut uniq_dec[u_idx * hd..(u_idx + 1) * hd])
                        .map_err(|err| bad(format!("row {id}: {}", err.reason)))?;
                } else {
                    misses.push((u_idx, id));
                }
            }
        } else {
            misses.extend(uniq.iter().copied().enumerate());
        }

        let miss_ids: Vec<u32> = misses.iter().map(|&(_, id)| id).collect();
        let spans = coalesce_spans(&miss_ids, self.shared.base, hd, self.shared.file_len);
        let n_spans = spans.len();
        let miss_raw = if self.shared.cache_enabled && !miss_ids.is_empty() {
            vec![0u8; miss_ids.len() * hd]
        } else {
            Vec::new()
        };

        let batch = Arc::new(EngramBatch {
            cancelled: AtomicBool::new(false),
            state: Mutex::new(BatchState {
                remaining: n_spans,
                output: uniq_dec,
                miss_raw,
                err: None,
                bytes_read: 0,
                wait_ns: 0,
                #[cfg(test)]
                done_seq: 0,
            }),
            done: Condvar::new(),
            shared: Arc::clone(&self.shared),
            ids: if ids_are_uniq_order {
                Vec::new()
            } else {
                ids.to_vec()
            },
            uniq,
            miss_ids,
            ids_are_uniq_order,
        });

        if n_spans > 0 {
            let mut jobs = Vec::with_capacity(n_spans);
            let mut cursor = 0usize;
            for span in spans {
                let count = span.rows.len();
                let mut targets = Vec::with_capacity(count);
                for (k, row_id) in span.rows.into_iter().enumerate() {
                    let miss_idx = cursor + k;
                    let (uniq_idx, expected_id) = misses[miss_idx];
                    debug_assert_eq!(row_id, expected_id);
                    targets.push((miss_idx, uniq_idx, row_id));
                }
                cursor += count;
                jobs.push(SpanJob {
                    batch: Arc::clone(&batch),
                    start: span.start,
                    end: span.end,
                    targets,
                });
            }
            {
                let mut q = lock(&self.shared.queue);
                if q.shutdown {
                    return Err(bad("engram worker pool is shut down"));
                }
                q.jobs.extend(jobs);
            }
            if n_spans == 1 {
                self.shared.ready.notify_one();
            } else {
                self.shared.ready.notify_all();
            }
        }

        self.shared
            .ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(PrefetchHandle {
            batch,
            waited: false,
        })
    }

    /// Decoded rows `[ids.len(), head_dim]` for any global row ids, in order.
    pub fn read_rows(&self, ids: &[u32]) -> Result<Vec<f32>> {
        self.prefetch_rows(ids)?.wait()
    }
}

impl Drop for F8Table {
    fn drop(&mut self) {
        {
            let mut q = lock(&self.shared.queue);
            q.shutdown = true;
        }
        self.shared.ready.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

impl EngramRows for F8Table {
    fn rows(&self, layer: usize, ids: &[u32]) -> Result<Vec<f32>> {
        self.columns.check(layer, ids)?;
        self.read_rows(ids)
    }

    fn prefetch(&self, layer: usize, ids: &[u32]) -> Result<Option<PrefetchHandle>> {
        self.columns.check(layer, ids)?;
        self.prefetch_rows(ids).map(Some)
    }

    fn take_prefetched(&self, handle: PrefetchHandle) -> Result<Vec<f32>> {
        handle.wait()
    }
}

/// The whole table held in memory (tiny fixtures; the real 63 GB table is
/// read by row).
pub struct ResidentF8Table {
    codes: Vec<u8>,
    head_dim: usize,
    columns: Columns,
}

impl ResidentF8Table {
    pub fn new(codes: Vec<u8>, cfg: &EngramConfig) -> Result<Self> {
        let n_rows = cfg.head_vocab_sizes.iter().sum::<u64>();
        prove_f8_extent(codes.len() as u64, n_rows, cfg.head_dim)?;
        Ok(Self {
            codes,
            head_dim: cfg.head_dim,
            columns: Columns::new(cfg, n_rows)?,
        })
    }
}

impl EngramRows for ResidentF8Table {
    fn rows(&self, layer: usize, ids: &[u32]) -> Result<Vec<f32>> {
        self.columns.check(layer, ids)?;
        let hd = self.head_dim;
        let mut out = vec![0f32; ids.len() * hd];
        for (&id, o) in ids.iter().zip(out.chunks_mut(hd)) {
            let at = id as usize * hd;
            decode_f8_rows(&self.codes[at..at + hd], o)
                .map_err(|err| bad(format!("row {id}: {}", err.reason)))?;
        }
        Ok(out)
    }
}

/// Name, type 43 and ggml dims `[head_dim, Σ head_vocab_sizes]`; the row
/// count.
fn table_rows(loc: &Located, cfg: &EngramConfig) -> Result<u64> {
    let info = &loc.info;
    if info.name != ENGRAM_TABLE {
        return Err(bad(format!("'{}' is not {ENGRAM_TABLE}", info.name)));
    }
    if info.tensor_type != GgufTensorType::MXFP8 {
        return Err(bad(format!(
            "{}: type {} is not GGUF type 43",
            info.name,
            info.tensor_type.name()
        )));
    }
    let n_rows = cfg.head_vocab_sizes.iter().sum::<u64>();
    if info.dims != [cfg.head_dim as u64, n_rows] {
        return Err(bad(format!(
            "{}: ggml dims {:?}, expected [{}, {n_rows}]",
            info.name, info.dims, cfg.head_dim
        )));
    }
    Ok(n_rows)
}
