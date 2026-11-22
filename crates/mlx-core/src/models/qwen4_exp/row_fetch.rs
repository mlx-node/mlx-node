//! Off-thread reads of single table rows (token and PLE embeddings). The
//! decoder knows a token's 17 rows as soon as the token is known; a few
//! reader threads `pread` them from descriptors kept open while the model
//! thread encodes the layers before the rows are consumed. The model thread
//! takes a row's bytes when it imports the row: ready, or read by itself
//! when still queued, or awaited when a reader holds it.
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;

use crate::utils::gguf_read::ReadAt;

/// `(tensor name, row)`.
pub(super) type Key = (String, usize);

pub(super) struct Job {
    pub key: Key,
    pub file: Arc<File>,
    /// Optional checkpoint path so Windows worker threads can open per-thread
    /// file handles instead of serializing on a single shared `FileObject->Lock`.
    #[cfg_attr(not(windows), allow(dead_code))]
    pub path: Option<Arc<PathBuf>>,
    pub offset: u64,
    pub len: usize,
}

/// Rows a request may leave behind (never taken: a cancelled turn, a
/// replayed token) before the next request drops everything pending. Above
/// the 557,056 rows of a whole 32,768-token prompt (17 per token), which a
/// layer-major prefill requests before its first layer; a later request
/// must never drop those while they wait.
const PENDING_LIMIT: usize = 1 << 20;

struct BatchWait {
    wanted: HashMap<Key, usize>,
    ready: Vec<(usize, std::io::Result<Vec<u8>>)>,
}

struct State {
    queue: VecDeque<Key>,
    /// Queued jobs by key for O(1) taker removal without scanning `queue`.
    /// A key is pending at most once across queued, running and done: a second
    /// job for it would complete after the one take and be handed to a later
    /// request for the key.
    queued: HashMap<Key, Job>,
    running: HashSet<Key>,
    done: HashMap<Key, std::io::Result<Vec<u8>>>,
    done_waiters: usize,
    /// Optional O(1) completion sink for `take_many` so workers wake the model
    /// thread once when the full batch lands instead of once per row.
    batch: Option<BatchWait>,
}

struct Shared {
    state: Mutex<State>,
    work_cv: Condvar,
    done_cv: Condvar,
}

pub(super) struct Fetcher {
    shared: Arc<Shared>,
    stop: Arc<AtomicBool>,
    /// Rows queued, being read by a worker, or being read by the taker:
    /// the reads the SSD still owes the model. Direct expert fills yield
    /// while it is above zero.
    pending: Arc<AtomicUsize>,
    workers: Vec<JoinHandle<()>>,
}

fn read(job: &Job) -> std::io::Result<Vec<u8>> {
    let mut buf = vec![0; job.len];
    job.file.read_exact_at(&mut buf, job.offset)?;
    Ok(buf)
}

#[cfg(windows)]
#[repr(C, align(4096))]
struct AlignedPageBuf([u8; 8192]);

#[cfg(windows)]
fn read_worker(
    job: &Job,
    handles: &mut HashMap<PathBuf, File>,
    aligned: &mut AlignedPageBuf,
) -> std::io::Result<Vec<u8>> {
    use std::os::windows::fs::{FileExt, OpenOptionsExt};
    // Bypass Windows File System Cache on random 128-byte PLE / 2.5 KB embedding
    // row reads: when 85+ GB is pinned by CUDA on a 128 GB host, buffered reads
    // starve the Windows System Cache (~8K IOPS), whereas 4 KB sector-aligned
    // FILE_FLAG_NO_BUFFERING hits full NVMe hardware IOPS (~250K IOPS, 18× faster).
    const FILE_FLAG_RANDOM_ACCESS: u32 = 0x1000_0000;
    const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
    const SECTOR: usize = 4096;

    if job.len <= SECTOR
        && let Some(path) = job.path.as_deref()
    {
        if !handles.contains_key(path.as_path())
            && let Ok(f) = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS)
                .open(path.as_path())
        {
            handles.insert(path.to_path_buf(), f);
        }
        if let Some(f) = handles.get(path.as_path()) {
            let aligned_off = job.offset & !((SECTOR as u64) - 1);
            let head = (job.offset - aligned_off) as usize;
            let need_end = head + job.len;
            let aligned_len = need_end.next_multiple_of(SECTOR);
            if aligned_len <= aligned.0.len()
                && let Ok(n) = f.seek_read(&mut aligned.0[..aligned_len], aligned_off)
                && n >= need_end
            {
                return Ok(aligned.0[head..need_end].to_vec());
            }
        }
    }
    let mut buf = vec![0; job.len];
    job.file.read_exact_at(&mut buf, job.offset)?;
    Ok(buf)
}

impl Fetcher {
    pub(super) fn new(workers: usize, pending: Arc<AtomicUsize>) -> std::io::Result<Self> {
        Self::start(workers, pending, |builder, body| builder.spawn(body))
    }

    /// `Self` exists before the first spawn: a failed spawn drops it and
    /// `Drop` stops and joins the workers already running. Handles held in a
    /// bare `Vec` would be detached by the `?`, parked on the condvar with
    /// `stop` false for the life of the process.
    fn start(
        workers: usize,
        pending: Arc<AtomicUsize>,
        mut spawn: impl FnMut(
            std::thread::Builder,
            Box<dyn FnOnce() + Send>,
        ) -> std::io::Result<JoinHandle<()>>,
    ) -> std::io::Result<Self> {
        let mut fetcher = Self {
            shared: Arc::new(Shared {
                state: Mutex::new(State {
                    queue: VecDeque::new(),
                    queued: HashMap::new(),
                    running: HashSet::new(),
                    done: HashMap::new(),
                    done_waiters: 0,
                    batch: None,
                }),
                work_cv: Condvar::new(),
                done_cv: Condvar::new(),
            }),
            stop: Arc::new(AtomicBool::new(false)),
            pending,
            workers: Vec::with_capacity(workers),
        };
        for _ in 0..workers {
            let shared = fetcher.shared.clone();
            let stop = fetcher.stop.clone();
            let pending = fetcher.pending.clone();
            let handle = spawn(
                std::thread::Builder::new().name("mlx-qwen4-rows".into()),
                Box::new(move || run(&shared, &stop, &pending)),
            )?;
            fetcher.workers.push(handle);
        }
        Ok(fetcher)
    }

    /// Queue `jobs` behind whatever is still pending, in order; a key that is
    /// already queued, running or ready is not queued again.
    pub(super) fn request(&self, jobs: Vec<Job>) {
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if state.batch.is_none() && state.queued.len() + state.done.len() > PENDING_LIMIT {
            self.pending.fetch_sub(state.queued.len(), Ordering::AcqRel);
            state.queue.clear();
            state.queued.clear();
            state.done.clear();
        }
        let mut added = 0;
        for job in jobs {
            if state.running.contains(&job.key)
                || state.done.contains_key(&job.key)
                || state.queued.contains_key(&job.key)
            {
                continue;
            }
            let key = job.key.clone();
            state.queued.insert(key.clone(), job);
            state.queue.push_back(key);
            added += 1;
        }
        if added > 0 {
            self.pending.fetch_add(added, Ordering::AcqRel);
            self.shared.work_cv.notify_all();
        }
    }

    /// The bytes of a requested row and whether the model thread had to
    /// read or wait for them; `None` when the row was never requested.
    pub(super) fn take(&self, key: &Key) -> Option<(std::io::Result<Vec<u8>>, bool)> {
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(bytes) = state.done.remove(key) {
            return Some((bytes, false));
        }
        if let Some(job) = state.queued.remove(key) {
            drop(state);
            let bytes = read(&job);
            self.pending.fetch_sub(1, Ordering::AcqRel);
            return Some((bytes, true));
        }
        if !state.running.contains(key) {
            return None;
        }
        state.done_waiters += 1;
        loop {
            if let Some(bytes) = state.done.remove(key) {
                state.done_waiters -= 1;
                return Some((bytes, true));
            }
            state = self
                .shared
                .done_cv
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// Batch-take a slice of requested keys in O(N) total work with ONE wakeup:
    /// ready rows are drained in one lock; when workers exist, remaining keys
    /// are registered in `state.batch` so workers deposit completions directly
    /// in O(1) and signal `done_cv` once when the entire batch lands.
    pub(super) fn take_many(&self, keys: &[Key]) -> Vec<Option<(std::io::Result<Vec<u8>>, bool)>> {
        let mut out: Vec<Option<(std::io::Result<Vec<u8>>, bool)>> =
            (0..keys.len()).map(|_| None).collect();
        let mut stolen: Vec<(usize, Job)> = Vec::new();
        let has_workers = !self.workers.is_empty();
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let mut wanted: HashMap<Key, usize> = HashMap::new();
        for (idx, key) in keys.iter().enumerate() {
            if let Some(bytes) = state.done.remove(key) {
                out[idx] = Some((bytes, false));
            } else if state.running.contains(key) || (has_workers && state.queued.contains_key(key))
            {
                // Keep first occurrence only (matches single-take semantics).
                wanted.entry(key.clone()).or_insert(idx);
            } else if let Some(job) = state.queued.remove(key) {
                stolen.push((idx, job));
            }
        }
        if !wanted.is_empty() && stolen.is_empty() {
            if state.batch.is_none() {
                let cap = wanted.len();
                state.batch = Some(BatchWait {
                    wanted,
                    ready: Vec::with_capacity(cap),
                });
                while state
                    .batch
                    .as_ref()
                    .is_some_and(|batch| !batch.wanted.is_empty())
                {
                    state = self
                        .shared
                        .done_cv
                        .wait(state)
                        .unwrap_or_else(PoisonError::into_inner);
                }
                let ready = state
                    .batch
                    .take()
                    .map(|batch| batch.ready)
                    .unwrap_or_default();
                drop(state);
                for (idx, bytes) in ready {
                    out[idx] = Some((bytes, true));
                }
                return out;
            }
            // Fallback if another thread already holds `state.batch`.
            state.done_waiters += 1;
            let mut remaining: Vec<usize> = wanted.into_values().collect();
            while !remaining.is_empty() {
                let mut next = Vec::new();
                for idx in remaining {
                    let key = &keys[idx];
                    if let Some(bytes) = state.done.remove(key) {
                        out[idx] = Some((bytes, true));
                    } else if state.running.contains(key) || state.queued.contains_key(key) {
                        next.push(idx);
                    }
                }
                remaining = next;
                if remaining.is_empty() {
                    break;
                }
                state = self
                    .shared
                    .done_cv
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
            }
            state.done_waiters -= 1;
            return out;
        }
        drop(state);
        if !stolen.is_empty() {
            let stolen_count = stolen.len();
            for (idx, job) in stolen {
                out[idx] = Some((read(&job), true));
            }
            self.pending.fetch_sub(stolen_count, Ordering::AcqRel);
        }
        out
    }

    /// Block until no job is queued or running (needs at least one worker).
    #[cfg(test)]
    pub(super) fn wait_idle(&self) {
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        state.done_waiters += 1;
        while !state.queued.is_empty() || !state.running.is_empty() {
            state = self
                .shared
                .done_cv
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        state.done_waiters -= 1;
    }
}

impl Fetcher {
    /// Stop the reader threads and keep every queued and ready row: a row
    /// requested after this is read on the model thread, late.
    #[cfg(test)]
    pub(super) fn pause(&mut self) {
        self.stop_workers();
    }

    /// Under the state lock: a worker between its `stop` check and its wait
    /// holds that lock, so a store and notify outside it could fall between
    /// the two and the join would wait forever.
    fn stop_workers(&mut self) {
        let state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        self.stop.store(true, Ordering::Release);
        self.shared.work_cv.notify_all();
        self.shared.done_cv.notify_all();
        drop(state);
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

impl Drop for Fetcher {
    fn drop(&mut self) {
        self.stop_workers();
    }
}

fn run(shared: &Shared, stop: &AtomicBool, pending: &AtomicUsize) {
    #[cfg(windows)]
    let mut handles: HashMap<PathBuf, File> = HashMap::new();
    #[cfg(windows)]
    let mut aligned = Box::new(AlignedPageBuf([0u8; 8192]));
    let mut jobs: Vec<Job> = Vec::with_capacity(8);
    let mut completed: Vec<(Key, std::io::Result<Vec<u8>>)> = Vec::with_capacity(8);
    loop {
        jobs.clear();
        {
            let mut state = shared.state.lock().unwrap_or_else(PoisonError::into_inner);
            loop {
                if stop.load(Ordering::Acquire) {
                    return;
                }
                // Pop up to 8 adjacent (offset-sorted) jobs on large prefill queues
                // to cut mutex traffic 8×, tapering cleanly to 1 on small decode
                // batches so all workers read in parallel without stragglers.
                let batch_cap = (state.queued.len() / 64).clamp(1, 8);
                while jobs.len() < batch_cap {
                    let Some(key) = state.queue.pop_front() else {
                        break;
                    };
                    if let Some(job) = state.queued.remove(&key) {
                        state.running.insert(key);
                        jobs.push(job);
                    }
                }
                if !jobs.is_empty() {
                    break;
                }
                state = shared
                    .work_cv
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
            }
        }
        completed.clear();
        for job in jobs.drain(..) {
            #[cfg(windows)]
            let bytes = read_worker(&job, &mut handles, &mut aligned);
            #[cfg(not(windows))]
            let bytes = read(&job);
            completed.push((job.key, bytes));
        }
        let n = completed.len();
        let mut state = shared.state.lock().unwrap_or_else(PoisonError::into_inner);
        let mut wake_batch = false;
        for (key, bytes) in completed.drain(..) {
            state.running.remove(&key);
            if let Some(batch) = state.batch.as_mut()
                && let Some(idx) = batch.wanted.remove(&key)
            {
                batch.ready.push((idx, bytes));
                if batch.wanted.is_empty() {
                    wake_batch = true;
                }
            } else {
                state.done.insert(key, bytes);
            }
        }
        pending.fetch_sub(n, Ordering::AcqRel);
        if wake_batch || state.done_waiters > 0 {
            shared.done_cv.notify_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::atomic::AtomicUsize;

    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!("qwen4-rows-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn table(dir: &Temp) -> (Arc<File>, Vec<u8>) {
        let bytes: Vec<u8> = (0..4096u32).map(|i| (i * 7 % 251) as u8).collect();
        let path = dir.0.join("table.bin");
        File::create(&path).unwrap().write_all(&bytes).unwrap();
        (Arc::new(File::open(&path).unwrap()), bytes)
    }

    fn job(file: &Arc<File>, row: usize) -> Job {
        Job {
            key: ("table".into(), row),
            file: file.clone(),
            path: None,
            offset: (row * 90) as u64,
            len: 90,
        }
    }

    #[test]
    fn requested_rows_arrive_bit_exact_ready_or_read_by_the_taker() {
        let dir = Temp::new();
        let (file, bytes) = table(&dir);
        // Readers: every row ends up ready or awaited, never re-read wrongly.
        let fetcher = Fetcher::new(2, Arc::default()).unwrap();
        fetcher.request((0..17).map(|row| job(&file, row)).collect());
        for row in 0..17 {
            let (got, _) = fetcher.take(&("table".into(), row)).unwrap();
            assert_eq!(got.unwrap(), &bytes[row * 90..row * 90 + 90], "row {row}");
        }
        assert!(fetcher.take(&("table".into(), 3)).is_none(), "taken once");
        assert!(
            fetcher.take(&("other".into(), 0)).is_none(),
            "never requested"
        );
        // No readers: the taker finds its row queued and reads it itself.
        let alone = Fetcher::new(0, Arc::default()).unwrap();
        alone.request(vec![job(&file, 5), job(&file, 40)]);
        let (got, late) = alone.take(&("table".into(), 40)).unwrap();
        assert!(late);
        assert_eq!(got.unwrap(), &bytes[40 * 90..41 * 90]);
        let (got, late) = alone.take(&("table".into(), 5)).unwrap();
        assert!(late);
        assert_eq!(got.unwrap(), &bytes[5 * 90..6 * 90]);
        // A row past the end of the file fails at take, not silently.
        alone.request(vec![Job {
            key: ("table".into(), 99),
            file: file.clone(),
            path: None,
            offset: 4096 - 10,
            len: 90,
        }]);
        assert!(alone.take(&("table".into(), 99)).unwrap().0.is_err());
    }

    /// `pending` is what the SSD still owes the model: a queued row, a row a
    /// worker reads, a row the taker reads itself; a ready row is not owed.
    #[test]
    fn pending_counts_the_rows_queued_running_or_read_by_the_taker() {
        let temp = Temp::new();
        let (file, _) = table(&temp);
        let pending = Arc::new(AtomicUsize::new(0));
        let alone = Fetcher::new(0, Arc::clone(&pending)).unwrap();
        alone.request((0..3).map(|row| job(&file, row)).collect());
        assert_eq!(pending.load(Ordering::Acquire), 3);
        alone.request(vec![job(&file, 1)]);
        assert_eq!(
            pending.load(Ordering::Acquire),
            3,
            "a queued key is counted once"
        );
        let (_, late) = alone.take(&("table".into(), 1)).unwrap();
        assert!(late);
        assert_eq!(
            pending.load(Ordering::Acquire),
            2,
            "the taker's own read is paid"
        );
        drop(alone);
        let fetcher = Fetcher::new(2, Arc::clone(&pending)).unwrap();
        fetcher.request((0..40).map(|row| job(&file, row)).collect());
        fetcher.wait_idle();
        assert_eq!(
            pending.load(Ordering::Acquire),
            2,
            "ready rows are not owed"
        );
        for row in 0..40 {
            let (_, late) = fetcher.take(&("table".into(), row)).unwrap();
            assert!(!late);
        }
        assert_eq!(pending.load(Ordering::Acquire), 2);
    }

    #[test]
    fn a_key_already_pending_is_requested_once() {
        let dir = Temp::new();
        let (file, bytes) = table(&dir);
        let key: Key = ("table".into(), 9);
        // Ready in `done`: the second request is skipped, so the one take
        // drains the key and no second completion appears afterwards.
        let fetcher = Fetcher::new(2, Arc::default()).unwrap();
        fetcher.request(vec![job(&file, 9)]);
        fetcher.wait_idle();
        fetcher.request(vec![job(&file, 9)]);
        let (got, late) = fetcher.take(&key).unwrap();
        assert!(!late);
        assert_eq!(got.unwrap(), &bytes[9 * 90..10 * 90]);
        fetcher.wait_idle();
        assert!(
            fetcher.take(&key).is_none(),
            "a second job for a ready key must not have run"
        );
        // Still queued (no readers): the key repeated inside one batch and in
        // a later batch is one queued job, read once by the taker.
        let alone = Fetcher::new(0, Arc::default()).unwrap();
        alone.request(vec![job(&file, 9), job(&file, 9)]);
        alone.request(vec![job(&file, 9)]);
        let (got, late) = alone.take(&key).unwrap();
        assert!(late);
        assert_eq!(got.unwrap(), &bytes[9 * 90..10 * 90]);
        assert!(
            alone.take(&key).is_none(),
            "a second job for a queued key must not exist"
        );
    }

    #[test]
    fn a_failed_spawn_stops_and_joins_the_workers_already_started() {
        let exited = Arc::new(AtomicUsize::new(0));
        let mut spawned = 0;
        let result = Fetcher::start(4, Arc::default(), |builder, body| {
            if spawned == 2 {
                return Err(std::io::Error::from(std::io::ErrorKind::WouldBlock));
            }
            spawned += 1;
            let exited = exited.clone();
            builder.spawn(move || {
                body();
                exited.fetch_add(1, Ordering::Release);
            })
        });
        assert_eq!(
            result.err().map(|e| e.kind()),
            Some(std::io::ErrorKind::WouldBlock)
        );
        assert_eq!(spawned, 2);
        assert_eq!(
            exited.load(Ordering::Acquire),
            2,
            "both started workers must have returned before the error came back"
        );
    }

    #[test]
    fn take_many_delivers_large_batch_across_workers_without_loss() {
        let dir = Temp::new();
        let path = dir.0.join("big.bin");
        let row_len = 64usize;
        let nrows = 512usize;
        let bytes: Vec<u8> = (0..(nrows * row_len))
            .map(|i| ((i * 13 + 5) % 251) as u8)
            .collect();
        File::create(&path).unwrap().write_all(&bytes).unwrap();
        let file = Arc::new(File::open(&path).unwrap());
        let path_arc = Arc::new(path);
        let pending = Arc::new(AtomicUsize::new(0));
        let fetcher = Fetcher::new(8, Arc::clone(&pending)).unwrap();

        let jobs: Vec<Job> = (0..nrows)
            .map(|row| Job {
                key: ("big".into(), row),
                file: Arc::clone(&file),
                path: Some(Arc::clone(&path_arc)),
                offset: (row * row_len) as u64,
                len: row_len,
            })
            .collect();
        fetcher.request(jobs);

        // Query all 512 rows + 1 missing key + 1 duplicate key.
        let mut keys: Vec<Key> = (0..nrows).map(|row| ("big".into(), row)).collect();
        keys.push(("big".into(), 9999));
        keys.push(("big".into(), 42));
        let got = fetcher.take_many(&keys);
        assert_eq!(got.len(), nrows + 2);
        for (row, item) in got.iter().take(nrows).enumerate() {
            let (res, _) = item.as_ref().expect("requested row must arrive");
            let data = res.as_ref().expect("row read must succeed");
            assert_eq!(
                data,
                &bytes[row * row_len..(row + 1) * row_len],
                "row {row}"
            );
        }
        assert!(got[nrows].is_none(), "unrequested key must return None");
        assert!(got[nrows + 1].is_none(), "duplicate key must return None");
        fetcher.wait_idle();
        assert_eq!(pending.load(Ordering::Acquire), 0);
    }
}
