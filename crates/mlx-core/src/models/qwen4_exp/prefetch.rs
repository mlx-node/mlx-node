//! The mapped expert pack's stream: a fixed set of worker threads reads the
//! whole pack into the page cache in layer order, so a cold turn finds its
//! banks resident ahead of the layer computing instead of faulting them in
//! through the Metal driver, and the model thread never reads pack bytes.
//! The stream starts after the load and again at any turn that opens over a
//! pack that is not fully resident; a turn that opens while it runs changes
//! nothing. Its order equals every request's order, so nothing a request
//! needs is behind anything it does not. The workers read at the default
//! I/O tier: a throttled or utility-tier read is paced behind other readers
//! but its pages are speculative, and 8 % of the pack survived such a walk
//! against a full page cache, so a lower tier would only read the
//! pack for nothing. They yield to the model's row reads instead: while
//! embedding or PLE rows are queued for the reader threads or read on the
//! model thread, a worker waits between two chunks. One small read behind
//! eight 4 MiB reads in the SSD queue costs milliseconds, and a cold
//! window's 2,000 inline row reads then held layer 0 back until the stream
//! was over (layer 0 began 6 s into a 7.1 s window). A chunk
//! that is already resident is skipped (one mincore per file, before its
//! reads), so a partly evicted pack costs its holes, and every byte the
//! stream asks for goes through a worker's own read with kernel read-ahead
//! off, behind its stop and yield checks: an advisory (F_RDADVISE) read or
//! the read-ahead would run on after the pool is dropped and ahead of the
//! model's rows.
use std::ffi::c_void;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::array::{Mmap, page_size};

/// Reader threads per mapped store. Buffered 4 MiB reads reach the SSD's
/// rate at queue depth 8 (12.3 of 12.6 GiB/s measured); more
/// adds nothing.
pub(super) const WORKERS: usize = 8;
/// Bytes per `pread`, with kernel read-ahead off: a worker checks `stop`
/// and the pending rows before every chunk, resident or not, and no byte
/// past the chunk it reads is in flight.
const CHUNK: usize = 4 << 20;
/// How often a yielding worker looks at the pending rows again.
const ROW_YIELD: Duration = Duration::from_micros(100);
/// Whether the pack needs a stream is sampled, never swept: mincore costs
/// 0.22 us per page, 1.2 s for the whole pack, against ~5 ms for eight
/// 256 KiB probes per file. A worker then sweeps each file it streams.
const PROBES: usize = 8;
const PROBE_PAGES: usize = 16;

/// How the stream reaches the kernel; tests record instead.
pub(super) trait Issue: Send + Sync + 'static {
    /// Read the chunks of the mapping that are not resident, so every page
    /// is resident and referenced. Returns between two chunks once `stop`
    /// is set; waits between two while `rows` (the model's pending row
    /// reads) is above zero.
    fn fetch(&self, map: &Mmap, stop: &AtomicBool, rows: &AtomicUsize);
}

/// Buffered `pread`s through the mapping's descriptor. The pages land in the
/// page cache referenced, which the advisory alone did not give: advisory
/// pages are speculative and were reclaimed before the driver bound them (a
/// cold window read the pack twice, 142 GiB for 73). Nothing goes through
/// the mapping the banks address: pages touched there stay in this
/// process's page tables for the mapping's life (75 GiB of RSS after a cold
/// turn).
pub(super) struct Pread;

#[cfg(test)]
thread_local! {
    /// Chunks `Pread::fetch` on this thread found resident and skipped.
    pub(super) static SKIPPED_CHUNKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Chunks `Pread::fetch` on this thread read.
    pub(super) static READ_CHUNKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// When set, `Pread::fetch` on this thread raises `rows` after every
    /// chunk it reads, so a test releases the fetch one chunk at a time.
    pub(super) static HOLD_AFTER_READ: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

impl Issue for Pread {
    fn fetch(&self, map: &Mmap, stop: &AtomicBool, rows: &AtomicUsize) {
        // Kernel read-ahead would issue I/O past the last chunk a stop or a
        // pending row allows, and reports pages in flight to mincore as in
        // core; eight workers reach the SSD's rate without it (12.31 against
        // 12.33 GiB/s).
        // SAFETY: the descriptor stays open for the mapping's lifetime and
        // F_RDAHEAD only sets its read-ahead policy.
        if unsafe { libc::fcntl(map.file().as_raw_fd(), libc::F_RDAHEAD, 0) } != 0 {
            tracing::debug!(
                "qwen4 mapped pack stream: read-ahead stays on: {}",
                std::io::Error::last_os_error()
            );
        }
        let cold = cold_chunks(map);
        let mut buf = vec![0u8; CHUNK.min(map.len())];
        for (chunk, offset) in (0..map.len()).step_by(CHUNK).enumerate() {
            while rows.load(Ordering::Acquire) > 0 {
                if stop.load(Ordering::Acquire) {
                    return;
                }
                std::thread::sleep(ROW_YIELD);
            }
            if stop.load(Ordering::Acquire) {
                return;
            }
            let len = CHUNK.min(map.len() - offset);
            if !cold[chunk] {
                #[cfg(test)]
                SKIPPED_CHUNKS.with(|skipped| skipped.set(skipped.get() + 1));
                continue;
            }
            if let Err(e) = map
                .file()
                .read_exact_at(&mut buf[..len], map.offset() + offset as u64)
            {
                tracing::warn!(
                    "qwen4 mapped pack stream: read failed, the driver faults the file: {e}"
                );
                return;
            }
            #[cfg(test)]
            READ_CHUNKS.with(|read| read.set(read.get() + 1));
            #[cfg(test)]
            if HOLD_AFTER_READ.with(|hold| hold.get()) {
                rows.fetch_add(1, Ordering::AcqRel);
            }
        }
    }
}

/// Which chunks of `map` have a page that is not in the page cache: one
/// mincore over the file before its reads. A mincore between two reads of
/// the same file reports the pages the kernel's read-ahead has in flight as
/// in core: a mincore per chunk skipped a third of an evicted pack unread
/// (6,191 of 18,809 chunks) and left it 54 of 73 GiB resident; one sweep
/// per file before its reads skipped nothing and left every page resident.
fn cold_chunks(map: &Mmap) -> Vec<bool> {
    let page = page_size();
    let chunks = map.len().div_ceil(CHUNK);
    let mut vec = vec![0u8; map.len().div_ceil(page)];
    // SAFETY: the range is the live mapping and `vec` holds one byte per
    // page of it; mincore only reads.
    let rc = unsafe {
        libc::mincore(
            map.as_ptr().cast::<c_void>(),
            map.len(),
            vec.as_mut_ptr().cast::<libc::c_char>(),
        )
    };
    if rc != 0 {
        return vec![true; chunks];
    }
    (0..chunks)
        .map(|chunk| {
            let first = chunk * CHUNK / page;
            let last = ((chunk + 1) * CHUNK).min(map.len()).div_ceil(page);
            vec[first..last].iter().any(|flags| flags & 1 == 0)
        })
        .collect()
}

struct Stream {
    /// Next file to hand out; files before it are issued.
    next: usize,
    /// Workers inside `fetch`.
    active: usize,
    started: Option<Instant>,
}

struct Shared {
    files: Vec<Arc<Mmap>>,
    stream: Mutex<Stream>,
    changed: Condvar,
    stop: AtomicBool,
    /// The model's pending row reads (`row_fetch::Fetcher`'s count).
    rows: Arc<AtomicUsize>,
}

pub(super) struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl Pool {
    /// `layers[i]` holds layer `i`'s gate, up and down mappings; the stream
    /// order is the layer order. Idle until `start`.
    pub(super) fn new(
        layers: Vec<Option<[Arc<Mmap>; 3]>>,
        issue: impl Issue,
        workers: usize,
        rows: Arc<AtomicUsize>,
    ) -> std::io::Result<Self> {
        let files: Vec<Arc<Mmap>> = layers.into_iter().flatten().flatten().collect();
        let shared = Arc::new(Shared {
            stream: Mutex::new(Stream {
                next: files.len(),
                active: 0,
                started: None,
            }),
            files,
            changed: Condvar::new(),
            stop: AtomicBool::new(false),
            rows,
        });
        let issue = Arc::new(issue);
        // `Self` exists before the first spawn: a failed spawn drops it and
        // `Drop` stops and joins the workers already running.
        let mut pool = Self {
            shared,
            workers: Vec::with_capacity(workers),
        };
        for _ in 0..workers {
            let shared = Arc::clone(&pool.shared);
            let issue = Arc::clone(&issue);
            pool.workers.push(
                std::thread::Builder::new()
                    .name("mlx-qwen4-prefetch".into())
                    .spawn(move || run(&shared, &*issue))?,
            );
        }
        Ok(pool)
    }

    /// Stream the whole pack from file 0. A stream already running keeps
    /// its cursor.
    pub(super) fn start(&self) {
        let mut stream = self.lock();
        if stream.next >= self.shared.files.len() && stream.active == 0 {
            stream.next = 0;
            stream.started = Some(Instant::now());
            tracing::debug!(
                "qwen4 mapped pack stream: {} files",
                self.shared.files.len()
            );
        }
        self.shared.changed.notify_all();
    }

    /// Whether files are still queued or being read.
    pub(super) fn streaming(&self) -> bool {
        let stream = self.lock();
        stream.next < self.shared.files.len() || stream.active > 0
    }

    /// Whether every probe of every file is in the page cache. A miss means
    /// the pack is at least partly cold; a hit may still hide absent pages
    /// between probes, which the driver faults in at its own rate.
    pub(super) fn sampled_resident(&self) -> bool {
        self.shared.files.iter().all(|map| sampled_resident(map))
    }

    #[cfg(test)]
    pub(super) fn wait_idle(&self) {
        let mut stream = self.lock();
        while stream.next < self.shared.files.len() || stream.active > 0 {
            stream = self
                .shared
                .changed
                .wait(stream)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Stream> {
        self.shared
            .stream
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        // Under the stream lock: a worker between its `stop` check and its
        // wait holds that lock, so a store and notify outside it could fall
        // between the two and the join would wait forever.
        let stream = self.lock();
        self.shared.stop.store(true, Ordering::Release);
        self.shared.changed.notify_all();
        drop(stream);
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn run(shared: &Shared, issue: &dyn Issue) {
    loop {
        let map = {
            let mut stream = shared.stream.lock().unwrap_or_else(PoisonError::into_inner);
            loop {
                if shared.stop.load(Ordering::Acquire) {
                    return;
                }
                if let Some(map) = shared.files.get(stream.next) {
                    stream.next += 1;
                    stream.active += 1;
                    break Arc::clone(map);
                }
                stream = shared
                    .changed
                    .wait(stream)
                    .unwrap_or_else(PoisonError::into_inner);
            }
        };
        issue.fetch(&map, &shared.stop, &shared.rows);
        let mut stream = shared.stream.lock().unwrap_or_else(PoisonError::into_inner);
        stream.active -= 1;
        if stream.active == 0 && stream.next >= shared.files.len() {
            if let Some(started) = stream.started.take() {
                tracing::debug!(
                    "qwen4 mapped pack stream: done in {:.1?}",
                    started.elapsed()
                );
            }
            shared.changed.notify_all();
        }
    }
}

fn sampled_resident(map: &Mmap) -> bool {
    let page = page_size();
    let pages = map.len().div_ceil(page);
    let mut vec = [0u8; PROBE_PAGES];
    for probe in 0..PROBES {
        let first = pages * probe / PROBES;
        let count = PROBE_PAGES.min(pages - first);
        let bytes = (count * page).min(map.len() - first * page);
        // SAFETY: the range lies inside the live mapping and `vec` holds one
        // byte per page of it; mincore only reads.
        let rc = unsafe {
            libc::mincore(
                map.as_ptr().add(first * page).cast::<c_void>(),
                bytes,
                vec.as_mut_ptr().cast::<libc::c_char>(),
            )
        };
        if rc != 0 || vec[..count].iter().any(|flags| flags & 1 == 0) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::os::fd::AsRawFd;
    use std::path::PathBuf;

    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("qwen4-prefetch-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Written through F_NOCACHE, so no page of it is in the page cache.
    fn cold_file(path: &PathBuf, pages: usize) -> File {
        let file = File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
        let page = page_size();
        let chunk: Vec<u8> = (0..page * 4).map(|i| (i * 7 % 251) as u8).collect();
        let mut offset = 0;
        while offset < pages * page {
            let n = chunk.len().min(pages * page - offset);
            file.write_all_at(&chunk[..n], offset as u64).unwrap();
            offset += n;
        }
        file.sync_all().unwrap();
        File::open(path).unwrap()
    }

    fn layer_maps(dir: &Temp, layers: usize, pages: usize) -> Vec<Option<[Arc<Mmap>; 3]>> {
        (0..layers)
            .map(|layer| {
                Some(["gate", "up", "down"].map(|part| {
                    let path = dir.0.join(format!("blk.{layer}.{part}"));
                    Mmap::map(cold_file(&path, pages), 0, pages * page_size()).unwrap()
                }))
            })
            .collect()
    }

    fn resident_fraction(map: &Mmap) -> f64 {
        let page = page_size();
        let pages = map.len().div_ceil(page);
        let mut vec = vec![0u8; pages];
        let rc = unsafe {
            libc::mincore(
                map.as_ptr().cast::<c_void>(),
                map.len(),
                vec.as_mut_ptr().cast::<libc::c_char>(),
            )
        };
        assert_eq!(rc, 0);
        vec.iter().filter(|flags| *flags & 1 == 1).count() as f64 / pages as f64
    }

    fn pointers(layers: &[Option<[Arc<Mmap>; 3]>], which: &[usize]) -> Vec<usize> {
        which
            .iter()
            .flat_map(|&layer| layers[layer].as_ref().unwrap().iter())
            .map(|map| map.as_ptr() as usize)
            .collect()
    }

    /// Records what the pool asks for; `gate` holds every fetch until
    /// released and `pace` makes each fetch `pace` chunks of 20 ms that
    /// honour `stop` only when `obeys_stop`.
    #[derive(Default)]
    struct Recorder {
        /// Files in the order their fetch finished.
        fetched: Mutex<Vec<usize>>,
        open: AtomicUsize,
        max_open: AtomicUsize,
        gate: Option<(Mutex<bool>, Condvar)>,
        pace: usize,
        obeys_stop: bool,
        stopped_mid_file: AtomicBool,
    }
    impl Recorder {
        fn gated() -> Self {
            Self {
                gate: Some((Mutex::new(false), Condvar::new())),
                ..Default::default()
            }
        }
        fn release(&self) {
            let (lock, ready) = self.gate.as_ref().unwrap();
            *lock.lock().unwrap() = true;
            ready.notify_all();
        }
        fn fetched(&self) -> Vec<usize> {
            self.fetched.lock().unwrap().clone()
        }
    }
    impl Issue for Arc<Recorder> {
        fn fetch(&self, map: &Mmap, stop: &AtomicBool, _rows: &AtomicUsize) {
            let open = self.open.fetch_add(1, Ordering::AcqRel) + 1;
            self.max_open.fetch_max(open, Ordering::AcqRel);
            if let Some((lock, ready)) = &self.gate {
                let mut released = lock.lock().unwrap();
                // Polled: a pool dropped by a failing test stops the worker
                // without a release, and nothing signals `ready` then.
                while !*released && !stop.load(Ordering::Acquire) {
                    released = ready
                        .wait_timeout(released, Duration::from_millis(10))
                        .unwrap()
                        .0;
                }
            }
            for _ in 0..self.pace {
                if self.obeys_stop && stop.load(Ordering::Acquire) {
                    self.stopped_mid_file.store(true, Ordering::Release);
                    break;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            self.fetched.lock().unwrap().push(map.as_ptr() as usize);
            self.open.fetch_sub(1, Ordering::AcqRel);
        }
    }

    /// One stream = every file of the pack once, in layer order (the order
    /// is observable with one worker; several workers complete out of
    /// order); a start while it runs adds nothing; a later stream visits
    /// every file again (the read itself skips the resident chunks).
    /// MUTATION: a worker takes `files.len() - 1 - i` -> "one
    /// worker: every file once, in layer order" fails.
    #[test]
    fn a_stream_fetches_the_files_in_layer_order_once() {
        let dir = Temp::new();
        let mut layers = layer_maps(&dir, 9, 1);
        layers[6] = None;
        let all = pointers(&layers, &[0, 1, 2, 3, 4, 5, 7, 8]);
        let recorder = Arc::new(Recorder::gated());
        let pool = Pool::new(layers.clone(), Arc::clone(&recorder), 1, Arc::default()).unwrap();
        assert!(!pool.streaming(), "idle until started");
        pool.start();
        assert!(pool.streaming());
        pool.start();
        recorder.release();
        pool.wait_idle();
        assert!(!pool.streaming());
        assert_eq!(
            recorder.fetched(),
            all,
            "one worker: every file once, in layer order"
        );

        let recorder = Arc::new(Recorder::gated());
        let pool = Pool::new(layers.clone(), Arc::clone(&recorder), 3, Arc::default()).unwrap();
        pool.start();
        pool.start();
        recorder.release();
        pool.wait_idle();
        let mut fetched = recorder.fetched();
        fetched.sort_unstable();
        let mut every = all.clone();
        every.sort_unstable();
        assert_eq!(fetched, every, "three workers: every file once");

        for (part, map) in ["gate", "up", "down"]
            .into_iter()
            .zip(layers[0].as_ref().unwrap())
        {
            std::fs::read(dir.0.join(format!("blk.0.{part}"))).unwrap();
            assert_eq!(resident_fraction(map), 1.0);
        }
        let recorder = Arc::new(Recorder::default());
        let pool = Pool::new(layers.clone(), Arc::clone(&recorder), 1, Arc::default()).unwrap();
        pool.start();
        pool.wait_idle();
        assert_eq!(
            recorder.fetched(),
            all,
            "a later stream visits every file, resident or not"
        );
    }

    /// Only the chunks that are not resident are read: a file with chunks 1
    /// and 3 in the page cache has exactly those two skipped, costs disk
    /// reads for the two cold ones and ends fully resident.
    /// MUTATION: drop the resident-chunk skip in `Pread::fetch`
    /// -> no chunk is skipped and "the resident chunks were read again"
    /// fails.
    #[test]
    fn resident_chunks_are_skipped_and_the_holes_between_them_read() {
        let dir = Temp::new();
        let pages = (4 * CHUNK) / page_size();
        let path = dir.0.join("holes");
        let map = Mmap::map(cold_file(&path, pages), 0, pages * page_size()).unwrap();
        let cached = File::open(&path).unwrap();
        let mut buf = vec![0u8; CHUNK];
        for chunk in [1usize, 3] {
            cached
                .read_exact_at(&mut buf, (chunk * CHUNK) as u64)
                .unwrap();
        }
        assert_eq!(resident_fraction(&map), 0.5, "chunks 1 and 3 are cached");
        let before = super::super::memory::process_disk_reads();
        SKIPPED_CHUNKS.with(|skipped| skipped.set(0));
        Pread.fetch(&map, &AtomicBool::new(false), &AtomicUsize::new(0));
        let read = super::super::memory::process_disk_reads() - before;
        assert_eq!(resident_fraction(&map), 1.0, "the holes were read");
        assert_eq!(
            SKIPPED_CHUNKS.with(|skipped| skipped.get()),
            2,
            "the resident chunks were read again"
        );
        assert!(
            read >= CHUNK as u64,
            "{read} B of disk reads for two cold chunks"
        );
    }

    /// A resident file is scanned under the same contract as a cold one: a
    /// fetch with `stop` already set skips nothing, and one with rows
    /// pending waits before its first chunk.
    /// MUTATION: check `stop` and the rows after the
    /// resident-chunk skip instead of before it -> "a stopped fetch scanned
    /// the resident chunks" fails.
    #[test]
    fn resident_chunks_wait_for_stop_and_pending_rows() {
        let dir = Temp::new();
        let pages = (4 * CHUNK) / page_size();
        let path = dir.0.join("warm");
        let map = Mmap::map(cold_file(&path, pages), 0, pages * page_size()).unwrap();
        let mut buf = vec![0u8; 4 * CHUNK];
        File::open(&path)
            .unwrap()
            .read_exact_at(&mut buf, 0)
            .unwrap();
        assert_eq!(resident_fraction(&map), 1.0, "the file is cached");
        SKIPPED_CHUNKS.with(|skipped| skipped.set(0));
        Pread.fetch(&map, &AtomicBool::new(true), &AtomicUsize::new(0));
        assert_eq!(
            SKIPPED_CHUNKS.with(|skipped| skipped.get()),
            0,
            "a stopped fetch scanned the resident chunks"
        );

        let rows = Arc::new(AtomicUsize::new(1));
        let release = {
            let rows = Arc::clone(&rows);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(100));
                rows.store(0, Ordering::Release);
            })
        };
        let started = Instant::now();
        Pread.fetch(&map, &AtomicBool::new(false), &rows);
        assert!(
            started.elapsed() >= Duration::from_millis(100),
            "the scan ran on while rows were pending"
        );
        assert_eq!(SKIPPED_CHUNKS.with(|skipped| skipped.get()), 4);
        release.join().unwrap();
    }

    /// `stop` set while a file is being read ends the read at the next
    /// chunk, not at the end of the file, and no page past that chunk is
    /// read: the pages resident afterwards are exactly the chunks read. The
    /// fetch holds itself on the row gate after every chunk (a test hook),
    /// the pacer releases it twice, one chunk at a time, and sets the stop
    /// while it is held, so the stop lands after exactly two chunks
    /// whatever the scheduler does to the threads.
    /// MUTATION: drop the F_RDAHEAD fcntl in `Pread::fetch`
    /// -> the kernel reads ahead of the last chunk and "the kernel read
    /// ahead past the stop" fails.
    #[test]
    fn a_fetch_stops_between_two_chunks() {
        READ_CHUNKS.with(|read| read.set(0));
        HOLD_AFTER_READ.with(|hold| hold.set(true));
        let dir = Temp::new();
        let pages = (32 * CHUNK) / page_size();
        let map = Arc::new(
            Mmap::map(
                cold_file(&dir.0.join("long"), pages),
                0,
                pages * page_size(),
            )
            .unwrap(),
        );
        let stop = Arc::new(AtomicBool::new(false));
        let rows = Arc::new(AtomicUsize::new(1));
        let pacer = {
            let (stop, rows) = (Arc::clone(&stop), Arc::clone(&rows));
            std::thread::spawn(move || {
                let deadline = Instant::now() + Duration::from_secs(10);
                for _ in 0..2 {
                    rows.store(0, Ordering::Release);
                    while rows.load(Ordering::Acquire) == 0 && Instant::now() < deadline {
                        std::thread::sleep(Duration::from_micros(100));
                    }
                }
                stop.store(true, Ordering::Release);
            })
        };
        Pread.fetch(&map, &stop, &rows);
        pacer.join().unwrap();
        HOLD_AFTER_READ.with(|hold| hold.set(false));
        assert_eq!(
            READ_CHUNKS.with(|read| read.get()),
            2,
            "the fetch read on after the stop"
        );
        assert_eq!(
            resident_fraction(&map),
            2.0 / 32.0,
            "the kernel read ahead past the stop"
        );
    }

    /// At most `workers` fetches are open at once: the only queue is the
    /// cursor over the file array.
    #[test]
    fn in_flight_is_bounded_by_the_worker_count() {
        let dir = Temp::new();
        let layers = layer_maps(&dir, 6, 1);
        let recorder = Arc::new(Recorder {
            pace: 2,
            ..Default::default()
        });
        let pool = Pool::new(layers, Arc::clone(&recorder), 4, Arc::default()).unwrap();
        pool.start();
        pool.wait_idle();
        assert_eq!(recorder.fetched().len(), 18);
        let max = recorder.max_open.load(Ordering::Acquire);
        assert!((2..=4).contains(&max), "{max} fetches were open at once");
    }

    /// Dropping the pool (the mapped-to-slots switch, unload) returns within
    /// one chunk: the worker sees `stop` between two chunks of a file, and
    /// the real read loop reads nothing once `stop` is set.
    /// MUTATION: `Pread::fetch` ignores `stop` -> the stopped
    /// fetch reads the whole file and "a stopped fetch reads nothing" fails.
    #[test]
    fn drop_stops_the_stream_within_one_chunk() {
        let dir = Temp::new();
        let layers = layer_maps(&dir, 4, 1);
        let recorder = Arc::new(Recorder {
            pace: 50,
            obeys_stop: true,
            ..Default::default()
        });
        let pool = Pool::new(layers, Arc::clone(&recorder), 2, Arc::default()).unwrap();
        pool.start();
        std::thread::sleep(Duration::from_millis(50));
        let started = Instant::now();
        drop(pool);
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_millis(200),
            "drop took {elapsed:?}; a worker must stop between two chunks"
        );
        assert!(recorder.stopped_mid_file.load(Ordering::Acquire));
        assert!(recorder.fetched().len() < 12, "the stream did not finish");

        let pages = (3 * CHUNK) / page_size();
        let map = Mmap::map(
            cold_file(&dir.0.join("stopped"), pages),
            0,
            pages * page_size(),
        )
        .unwrap();
        Pread.fetch(&map, &AtomicBool::new(true), &AtomicUsize::new(0));
        assert_eq!(
            resident_fraction(&map),
            0.0,
            "a stopped fetch reads nothing"
        );
        Pread.fetch(&map, &AtomicBool::new(false), &AtomicUsize::new(0));
        assert_eq!(resident_fraction(&map), 1.0);
    }

    /// The stream yields to the model's row reads: while rows are pending
    /// no chunk is read, the reads resume when the count drops to zero,
    /// and a pool dropped while it yields still returns at once.
    /// MUTATION: drop the wait on `rows` in `Pread::fetch` ->
    /// "the stream read on while rows were pending" fails.
    #[test]
    fn the_stream_yields_to_pending_row_reads() {
        let dir = Temp::new();
        let pages = (3 * CHUNK) / page_size();
        let layers = layer_maps(&dir, 1, pages);
        let rows = Arc::new(AtomicUsize::new(1));
        let pool = Pool::new(layers.clone(), Pread, 1, Arc::clone(&rows)).unwrap();
        pool.start();
        std::thread::sleep(Duration::from_millis(300));
        for map in layers.iter().flatten().flatten() {
            assert_eq!(
                resident_fraction(map),
                0.0,
                "the stream read on while rows were pending"
            );
        }
        rows.store(0, Ordering::Release);
        pool.wait_idle();
        for map in layers.iter().flatten().flatten() {
            assert_eq!(resident_fraction(map), 1.0);
        }

        let layers = layer_maps(&Temp::new(), 1, pages);
        let rows = Arc::new(AtomicUsize::new(1));
        let pool = Pool::new(layers, Pread, 2, rows).unwrap();
        pool.start();
        std::thread::sleep(Duration::from_millis(50));
        let started = Instant::now();
        drop(pool);
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_millis(200),
            "drop took {elapsed:?} while the stream yielded"
        );
    }

    fn resident_bytes() -> u64 {
        let mut info: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
        let size = std::mem::size_of::<libc::proc_taskinfo>() as libc::c_int;
        let rc = unsafe {
            libc::proc_pidinfo(
                std::process::id() as libc::c_int,
                libc::PROC_PIDTASKINFO,
                0,
                (&mut info as *mut libc::proc_taskinfo).cast::<c_void>(),
                size,
            )
        };
        assert_eq!(rc, size);
        info.pti_resident_size
    }

    /// Against the page cache itself: a stream over cold files leaves every
    /// page of every file resident, the sampled check tells the cold pack
    /// from the warm one, and nothing of the files is mapped into this
    /// process by the read (the banks' own mapping must stay untouched).
    #[test]
    fn the_stream_makes_the_pack_resident_against_the_page_cache() {
        let dir = Temp::new();
        let layers = layer_maps(&dir, 3, (64 << 20) / page_size());
        for map in layers.iter().flatten().flatten() {
            assert_eq!(
                resident_fraction(map),
                0.0,
                "F_NOCACHE write left pages cached"
            );
        }
        let pool = Pool::new(layers.clone(), Pread, WORKERS, Arc::default()).unwrap();
        assert!(!pool.sampled_resident());
        let before = resident_bytes();
        pool.start();
        pool.wait_idle();
        let grown = resident_bytes().saturating_sub(before);
        let total: usize = layers.iter().flatten().flatten().map(|m| m.len()).sum();
        for (layer, maps) in layers.iter().enumerate() {
            for (part, map) in maps.as_ref().unwrap().iter().enumerate() {
                assert_eq!(resident_fraction(map), 1.0, "layer {layer} part {part}");
            }
        }
        assert!(pool.sampled_resident());
        assert!(
            grown < (total / 2) as u64,
            "the stream mapped {grown} B of the {total} B pack into this process"
        );
        pool.start();
        pool.wait_idle();
        assert!(
            pool.sampled_resident(),
            "a stream over a warm pack is a no-op"
        );
    }
}
