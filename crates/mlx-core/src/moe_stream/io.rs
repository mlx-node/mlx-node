//! Shard descriptors and the persistent `pread` pool.
//!
//! Every shard is read through one `O_RDONLY` descriptor with `F_NOCACHE` and
//! `F_RDAHEAD 0` (macOS); nothing here maps a file. The reader that finishes
//! an MXFP4 extent repacks it straight into its slot. A failed or short read
//! fails its ticket and never runs the repack, so the slot's bytes stay as
//! they were and the caller does not publish it.

use std::collections::{HashSet, VecDeque};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::Instant;

use napi::{Error, Result};

use super::repack::SlotRepack;
use crate::array::alloc::SendPtr;
use crate::cold_tier::ShardIdentity;
use crate::utils::gguf::GgufTensorInfo;

pub const THREADS_ENV: &str = "MLX_MOE_STREAM_IO_THREADS";
pub const DEFAULT_THREADS: usize = 12;
pub const MIN_THREADS: usize = 4;
pub const MAX_THREADS: usize = 18;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("moe_stream io: {msg}"))
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// `O_RDONLY` + `F_NOCACHE` 1 + `F_RDAHEAD` 0.
pub fn open_nocache(path: &Path) -> Result<File> {
    let file = File::open(path).map_err(|e| bad(format!("open '{}': {e}", path.display())))?;
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        for (cmd, arg, what) in [
            (libc::F_NOCACHE, 1, "F_NOCACHE"),
            (libc::F_RDAHEAD, 0, "F_RDAHEAD"),
        ] {
            // SAFETY: `file` owns an open descriptor; both commands only
            // change caching / readahead.
            if unsafe { libc::fcntl(file.as_raw_fd(), cmd, arg) } != 0 {
                return Err(bad(format!(
                    "{what} on '{}': {}",
                    path.display(),
                    std::io::Error::last_os_error()
                )));
            }
        }
    }
    Ok(file)
}

struct Shard {
    path: PathBuf,
    file: File,
    len: u64,
    stat: ShardIdentity,
}

/// The shards one model reads, indexed as given to [`ShardSet::open`].
pub struct ShardSet {
    shards: Vec<Shard>,
    #[cfg(test)]
    fault: Mutex<Option<(usize, u64)>>,
}

impl ShardSet {
    pub fn open<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        let mut shards = Vec::with_capacity(paths.len());
        for p in paths {
            let path = p.as_ref().to_path_buf();
            let file = open_nocache(&path)?;
            let meta = file
                .metadata()
                .map_err(|e| bad(format!("stat '{}': {e}", path.display())))?;
            shards.push(Shard {
                len: meta.len(),
                stat: ShardIdentity::from_metadata(&meta),
                path,
                file,
            });
        }
        Ok(Self {
            shards,
            #[cfg(test)]
            fault: Mutex::new(None),
        })
    }

    pub fn len(&self) -> usize {
        self.shards.len()
    }

    pub fn is_empty(&self) -> bool {
        self.shards.is_empty()
    }

    pub fn path(&self, shard: usize) -> Option<&Path> {
        self.shards.get(shard).map(|s| s.path.as_path())
    }

    /// The shard's descriptor (header parsing; data goes through the pool).
    pub fn file(&self, shard: usize) -> Option<&File> {
        self.shards.get(shard).map(|s| &s.file)
    }

    /// Fails when any shard changed (size, inode, mtime) since `open`.
    pub fn recheck(&self) -> Result<()> {
        for s in &self.shards {
            let meta = s
                .file
                .metadata()
                .map_err(|e| bad(format!("stat '{}': {e}", s.path.display())))?;
            if ShardIdentity::from_metadata(&meta) != s.stat {
                return Err(bad(format!(
                    "'{}' changed since it was opened",
                    s.path.display()
                )));
            }
        }
        Ok(())
    }

    fn check(&self, e: &Extent) -> Result<()> {
        let s = self
            .shards
            .get(e.shard)
            .ok_or_else(|| bad(format!("shard {} of {}", e.shard, self.shards.len())))?;
        if e.len == 0 || e.offset.checked_add(e.len).is_none_or(|end| end > s.len) {
            return Err(bad(format!(
                "extent {}+{} outside '{}' ({} bytes)",
                e.offset,
                e.len,
                s.path.display(),
                s.len
            )));
        }
        Ok(())
    }

    /// Fills `buf` from `offset`; the file may end early, but not before
    /// `need` bytes.
    fn read_at_least(
        &self,
        shard: usize,
        buf: &mut [u8],
        offset: u64,
        need: usize,
    ) -> std::io::Result<()> {
        #[cfg(test)]
        if let Some((s, at)) = *lock(&self.fault)
            && s == shard
            && (offset..offset + buf.len() as u64).contains(&at)
        {
            return Err(std::io::Error::from_raw_os_error(libc::EIO));
        }
        let file = &self.shards[shard].file;
        let mut done = 0;
        while done < buf.len() {
            match file.read_at(&mut buf[done..], offset + done as u64) {
                Ok(0) => break,
                Ok(n) => done += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        if done < need {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        Ok(())
    }
}

/// `len` bytes at `offset` of shard `shard`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extent {
    pub shard: usize,
    pub offset: u64,
    pub len: u64,
}

impl Extent {
    /// Expert `expert` of a 3-D expert tensor (`[n_experts, out, in]`, expert
    /// outermost) whose shard data section starts at `data_offset`.
    pub fn expert(
        shard: usize,
        data_offset: u64,
        info: &GgufTensorInfo,
        expert: usize,
    ) -> Result<Self> {
        let shape = info.mlx_shape();
        if shape.len() != 3 || shape[0] <= 0 {
            return Err(bad(format!(
                "{}: expected a 3-D expert tensor, got {shape:?}",
                info.name
            )));
        }
        let n = shape[0] as u64;
        let size = info.data_size()?;
        if !size.is_multiple_of(n) || expert as u64 >= n {
            return Err(bad(format!(
                "{}: expert {expert} of {n} ({size} bytes)",
                info.name
            )));
        }
        let per = size / n;
        let offset = data_offset
            .checked_add(info.offset)
            .and_then(|o| o.checked_add(per * expert as u64))
            .ok_or_else(|| bad(format!("{}: offset overflows", info.name)))?;
        Ok(Self {
            shard,
            offset,
            len: per,
        })
    }
}

enum Sink {
    Slot { repack: SlotRepack, slot: usize },
    Raw(SendPtr<u8>),
}

/// One extent and where its bytes go.
pub struct Read {
    pub extent: Extent,
    sink: Sink,
}

impl Read {
    /// MXFP4 blocks of one expert, repacked into `slot` of `repack`.
    pub fn slot(extent: Extent, repack: &SlotRepack, slot: usize) -> Self {
        Self {
            extent,
            sink: Sink::Slot {
                repack: *repack,
                slot,
            },
        }
    }

    /// Bytes copied as they are to `dst` (`extent.len` bytes). A failed read
    /// may leave `dst` partly written.
    pub fn raw(extent: Extent, dst: *mut u8) -> Self {
        Self {
            extent,
            sink: Sink::Raw(SendPtr(dst)),
        }
    }

    fn check(&self) -> Result<Option<(usize, usize)>> {
        match &self.sink {
            Sink::Slot { repack, slot } => {
                if *slot >= repack.slots() || self.extent.len != repack.slot_bytes() {
                    return Err(bad(format!(
                        "slot {slot} of {} with {} bytes, the slot takes {}",
                        repack.slots(),
                        self.extent.len,
                        repack.slot_bytes()
                    )));
                }
                Ok(Some(repack.key(*slot)))
            }
            Sink::Raw(p) if p.0.is_null() => Err(bad("null raw destination")),
            Sink::Raw(_) => Ok(None),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Priority {
    High,
    Low,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IoStats {
    pub reads: usize,
    pub bytes: u64,
    /// Submit to the last read done.
    pub ns: u64,
    /// Longest submit-to-done of one read (queue wait included).
    pub max_latency_ns: u64,
    /// Summed over reads.
    pub read_ns: u64,
    pub repack_ns: u64,
}

struct BatchState {
    remaining: usize,
    stats: IoStats,
    failures: Vec<String>,
}

struct Batch {
    start: Instant,
    state: Mutex<BatchState>,
    done: Condvar,
}

impl Batch {
    fn finish(&self, out: Result<(u64, u64, u64)>) {
        let at = self.start.elapsed().as_nanos() as u64;
        let mut st = lock(&self.state);
        match out {
            Ok((bytes, read_ns, repack_ns)) => {
                st.stats.bytes += bytes;
                st.stats.read_ns += read_ns;
                st.stats.repack_ns += repack_ns;
            }
            Err(e) => st.failures.push(e.reason.clone()),
        }
        st.stats.max_latency_ns = st.stats.max_latency_ns.max(at);
        st.remaining -= 1;
        if st.remaining == 0 {
            st.stats.ns = at;
            self.done.notify_all();
        }
    }

    fn wait(&self) -> MutexGuard<'_, BatchState> {
        let mut st = lock(&self.state);
        while st.remaining > 0 {
            st = self.done.wait(st).unwrap_or_else(PoisonError::into_inner);
        }
        st
    }
}

/// A submitted batch. Dropping it without [`Ticket::wait`] still blocks
/// until every read of the batch is done.
pub struct Ticket {
    batch: Arc<Batch>,
}

impl Ticket {
    /// Blocks until the batch is done; any failed read fails the batch.
    pub fn wait(self) -> Result<IoStats> {
        let st = self.batch.wait();
        if st.failures.is_empty() {
            return Ok(st.stats);
        }
        Err(bad(format!(
            "{} of {} reads failed: {}",
            st.failures.len(),
            st.stats.reads,
            st.failures
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join("; ")
        )))
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        drop(self.batch.wait());
    }
}

struct Job {
    batch: Arc<Batch>,
    read: Read,
}

#[derive(Default)]
struct Queues {
    high: VecDeque<Job>,
    low: VecDeque<Job>,
    shutdown: bool,
}

struct Shared {
    shards: Arc<ShardSet>,
    queue: Mutex<Queues>,
    ready: Condvar,
}

/// Persistent reader threads over one [`ShardSet`].
pub struct IoPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

/// Thread count from the env value: [`DEFAULT_THREADS`] when unset.
pub fn threads_from(value: Option<&str>) -> Result<usize> {
    let Some(raw) = value.map(str::trim).filter(|v| !v.is_empty()) else {
        return Ok(DEFAULT_THREADS);
    };
    match raw.parse::<usize>() {
        Ok(n) if (MIN_THREADS..=MAX_THREADS).contains(&n) => Ok(n),
        _ => Err(bad(format!(
            "{THREADS_ENV}={raw:?}: expected {MIN_THREADS}..={MAX_THREADS}"
        ))),
    }
}

impl IoPool {
    pub fn from_env(shards: Arc<ShardSet>) -> Result<Self> {
        Self::new(
            shards,
            threads_from(std::env::var(THREADS_ENV).ok().as_deref())?,
        )
    }

    pub fn new(shards: Arc<ShardSet>, threads: usize) -> Result<Self> {
        if threads == 0 || threads > MAX_THREADS {
            return Err(bad(format!("{threads} reader threads")));
        }
        let shared = Arc::new(Shared {
            shards,
            queue: Mutex::new(Queues::default()),
            ready: Condvar::new(),
        });
        let mut pool = Self {
            shared,
            workers: Vec::with_capacity(threads),
        };
        for i in 0..threads {
            let shared = Arc::clone(&pool.shared);
            let handle = std::thread::Builder::new()
                .name(format!("moe-io-{i}"))
                .spawn(move || worker(&shared))
                .map_err(|e| bad(format!("spawn reader {i}: {e}")))?;
            pool.workers.push(handle);
        }
        Ok(pool)
    }

    pub fn threads(&self) -> usize {
        self.workers.len()
    }

    pub fn shards(&self) -> &Arc<ShardSet> {
        &self.shared.shards
    }

    /// Queue `reads`; nothing is queued when any read is invalid. High
    /// priority reads run before every queued low priority read.
    ///
    /// # Safety
    /// Every destination stays valid, and no one else reads or writes it,
    /// until the ticket is waited or dropped. Raw destinations of one batch
    /// do not overlap each other or any slot (two reads into one slot are
    /// refused).
    pub unsafe fn submit(&self, reads: Vec<Read>, priority: Priority) -> Result<Ticket> {
        let mut seen = HashSet::new();
        for r in &reads {
            self.shared.shards.check(&r.extent)?;
            if let Some(key) = r.check()?
                && !seen.insert(key)
            {
                return Err(bad(format!("slot {} read twice in one batch", key.1)));
            }
        }
        let batch = Arc::new(Batch {
            start: Instant::now(),
            state: Mutex::new(BatchState {
                remaining: reads.len(),
                stats: IoStats {
                    reads: reads.len(),
                    ..IoStats::default()
                },
                failures: Vec::new(),
            }),
            done: Condvar::new(),
        });
        if !reads.is_empty() {
            let mut q = lock(&self.shared.queue);
            if q.shutdown {
                return Err(bad("pool is shut down"));
            }
            let dq = match priority {
                Priority::High => &mut q.high,
                Priority::Low => &mut q.low,
            };
            dq.extend(reads.into_iter().map(|read| Job {
                batch: Arc::clone(&batch),
                read,
            }));
            drop(q);
            self.shared.ready.notify_all();
        }
        Ok(Ticket { batch })
    }
}

impl Drop for IoPool {
    fn drop(&mut self) {
        lock(&self.shared.queue).shutdown = true;
        self.shared.ready.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

fn next_job(shared: &Shared) -> Option<Job> {
    let mut q = lock(&shared.queue);
    loop {
        if let Some(job) = q.high.pop_front().or_else(|| q.low.pop_front()) {
            return Some(job);
        }
        if q.shutdown {
            return None;
        }
        q = shared.ready.wait(q).unwrap_or_else(PoisonError::into_inner);
    }
}

/// Slot reads cover whole `ALIGN` pages into an `ALIGN`-aligned buffer:
/// measured on this Mac, an unaligned `F_NOCACHE` pread of a 6.3 MB expert
/// left ~83 % of its pages in the page cache.
const ALIGN: usize = 16 << 10;

struct AlignedBuf {
    ptr: *mut u8,
    cap: usize,
}

impl AlignedBuf {
    fn new() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            cap: 0,
        }
    }

    fn get(&mut self, len: usize) -> Result<&mut [u8]> {
        if len > self.cap {
            self.release();
            let cap = len
                .checked_next_multiple_of(ALIGN)
                .ok_or_else(|| bad("read buffer size overflows"))?;
            let layout = std::alloc::Layout::from_size_align(cap, ALIGN)
                .map_err(|e| bad(format!("read buffer layout: {e}")))?;
            // SAFETY: `layout` has a non-zero size (`cap >= len > 0`).
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                return Err(bad(format!("read buffer of {cap} bytes: out of memory")));
            }
            (self.ptr, self.cap) = (ptr, cap);
        }
        // SAFETY: `ptr` holds `cap >= len` bytes owned by this buffer; the
        // reader writes them before anyone reads them.
        Ok(unsafe { std::slice::from_raw_parts_mut(self.ptr, len) })
    }

    fn release(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: allocated in `get` with this exact layout.
            unsafe {
                std::alloc::dealloc(
                    self.ptr,
                    std::alloc::Layout::from_size_align_unchecked(self.cap, ALIGN),
                )
            };
            (self.ptr, self.cap) = (std::ptr::null_mut(), 0);
        }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        self.release();
    }
}

fn worker(shared: &Shared) {
    let mut buf = AlignedBuf::new();
    while let Some(job) = next_job(shared) {
        let out = catch_unwind(AssertUnwindSafe(|| {
            run(&shared.shards, &job.read, &mut buf)
        }))
        .unwrap_or_else(|_| Err(bad("reader panicked")));
        job.batch.finish(out);
    }
}

/// `(bytes, read_ns, repack_ns)`.
fn run(shards: &ShardSet, read: &Read, buf: &mut AlignedBuf) -> Result<(u64, u64, u64)> {
    let e = read.extent;
    let len = usize::try_from(e.len).map_err(|_| bad("extent length exceeds usize"))?;
    let t0 = Instant::now();
    let io_err = |err: std::io::Error| {
        let what = if err.kind() == std::io::ErrorKind::UnexpectedEof {
            "short read (file ended)".to_string()
        } else {
            err.to_string()
        };
        bad(format!(
            "read {len} bytes at {} of '{}': {what}",
            e.offset,
            shards
                .path(e.shard)
                .map(|p| p.display().to_string())
                .unwrap_or_default()
        ))
    };
    match &read.sink {
        Sink::Raw(dst) => {
            // SAFETY: `submit`'s contract: `dst` holds `len` bytes owned by
            // this read until the ticket is done.
            let out = unsafe { std::slice::from_raw_parts_mut(dst.0, len) };
            shards
                .read_at_least(e.shard, out, e.offset, len)
                .map_err(io_err)?;
            Ok((e.len, t0.elapsed().as_nanos() as u64, 0))
        }
        Sink::Slot { repack, slot } => {
            let start = e.offset - e.offset % ALIGN as u64;
            let end = (e.offset + e.len)
                .checked_next_multiple_of(ALIGN as u64)
                .ok_or_else(|| bad("extent end overflows"))?
                .min(shards.shards[e.shard].len);
            let window = usize::try_from(end - start).map_err(|_| bad("read window size"))?;
            let head = (e.offset - start) as usize;
            let bytes = buf.get(window)?;
            shards
                .read_at_least(e.shard, bytes, start, head + len)
                .map_err(io_err)?;
            let read_ns = t0.elapsed().as_nanos() as u64;
            let t1 = Instant::now();
            repack.write(*slot, &bytes[head..head + len])?;
            Ok((e.len, read_ns, t1.elapsed().as_nanos() as u64))
        }
    }
}

#[cfg(test)]
mod tests;
