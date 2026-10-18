//! Read-ahead for the mapped expert pack: one worker asks the kernel to page
//! whole tensor files in ahead of the layer the model computes, so a cold
//! turn streams at SSD rate instead of the Metal driver's fault rate and the
//! model thread never reads. Issuing threads do not add rate (10.5-10.9
//! GiB/s cold from 1 to 6 threads), and F_RDADVISE itself skips resident
//! pages (0.6-1.6 ms per 0.5 GiB file when warm), so no per-file residency
//! check precedes a request: a mincore sweep of the same file costs 7-9 ms.
use std::collections::VecDeque;
use std::ffi::c_void;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;

use crate::array::{Mmap, page_size};

/// Layers read ahead of the one computing, ~1.5 GiB each on Flash Next.
const DEFAULT_DEPTH: usize = 3;
/// Residency is sampled, never swept: mincore costs 0.22 us per page, 1.2 s
/// for the whole pack, against ~5 ms for eight 256 KiB probes per file.
const PROBES: usize = 8;
const PROBE_PAGES: usize = 16;

pub(super) fn depth_from_env() -> usize {
    std::env::var_os("MLX_QWEN4_PREFETCH_LAYERS")
        .and_then(|value| value.to_str()?.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_DEPTH)
}

pub(super) fn at_load() -> bool {
    std::env::var_os("MLX_QWEN4_PREFETCH_AT_LOAD").is_some_and(|value| value == "1")
}

/// How a request reaches the kernel; tests record instead.
pub(super) trait Issue: Send + 'static {
    fn read_ahead(&self, map: &Mmap);
}

/// F_RDADVISE over the whole mapping, then one read of every page through
/// it. The advisory is asynchronous, pages in only what is absent and
/// allocates nothing in this process, but the pages it brings in are
/// speculative: with the page cache full, the pageout scan reclaims them
/// before the driver binds the bank, and a cold 6k window read the pack
/// twice (142 GiB for 73 GiB, 33 s instead of 27). A referenced page is
/// reclaimed after the inactive ones instead; the reads also pace the
/// worker behind the kernel's queue.
pub(super) struct Advise;

impl Issue for Advise {
    fn read_ahead(&self, map: &Mmap) {
        let mut offset = 0usize;
        while offset < map.len() {
            let count = (map.len() - offset).min(1 << 30);
            let advisory = libc::radvisory {
                ra_offset: (map.offset() + offset as u64) as libc::off_t,
                ra_count: count as libc::c_int,
            };
            // SAFETY: the descriptor stays open for the mapping's lifetime and
            // F_RDADVISE only reads the file.
            unsafe { libc::fcntl(map.fd().as_raw_fd(), libc::F_RDADVISE, &advisory) };
            offset += count;
        }
        let page = page_size();
        let mut touched = 0u8;
        let mut offset = 0usize;
        while offset < map.len() {
            // SAFETY: `offset` lies inside the live read-only mapping.
            touched = touched.wrapping_add(unsafe { map.as_ptr().add(offset).read_volatile() });
            offset += page;
        }
        std::hint::black_box(touched);
    }
}

struct Queue {
    files: VecDeque<Arc<Mmap>>,
    /// Queued plus in flight.
    pending: usize,
}

pub(super) struct Pool {
    layers: Vec<Option<[Arc<Mmap>; 3]>>,
    queue: Arc<(Mutex<Queue>, Condvar)>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl Pool {
    /// `layers[i]` holds layer `i`'s gate, up and down mappings.
    pub(super) fn new(
        layers: Vec<Option<[Arc<Mmap>; 3]>>,
        issue: impl Issue,
    ) -> std::io::Result<Self> {
        let queue = Arc::new((
            Mutex::new(Queue {
                files: VecDeque::new(),
                pending: 0,
            }),
            Condvar::new(),
        ));
        let stop = Arc::new(AtomicBool::new(false));
        let worker = std::thread::Builder::new()
            .name("mlx-qwen4-prefetch".into())
            .spawn({
                let queue = queue.clone();
                let stop = stop.clone();
                move || run(&queue, &stop, &issue)
            })?;
        Ok(Self {
            layers,
            queue,
            stop,
            worker: Some(worker),
        })
    }

    /// Called as layer `layer` of a prefetch pass starts: the first layer
    /// requests itself and the `depth` layers after it, every later layer
    /// the one `depth` ahead, so each layer is requested once per pass.
    pub(super) fn layer_started(&self, layer: usize, depth: usize) {
        if depth == 0 {
            return;
        }
        if layer == 0 {
            self.request(0..=depth);
        } else {
            self.request(layer + depth..=layer + depth);
        }
    }

    pub(super) fn request_all(&self) {
        self.request(0..self.layers.len());
    }

    fn request(&self, layers: impl IntoIterator<Item = usize>) {
        let files: Vec<Arc<Mmap>> = layers
            .into_iter()
            .filter_map(|layer| self.layers.get(layer)?.as_ref())
            .flat_map(|maps| maps.iter().cloned())
            .collect();
        if files.is_empty() {
            return;
        }
        let (lock, ready) = &*self.queue;
        let mut queue = lock.lock().unwrap_or_else(PoisonError::into_inner);
        queue.pending += files.len();
        queue.files.extend(files);
        ready.notify_all();
    }

    /// Whether every probe of every file is in the page cache. A miss means
    /// the pack is at least partly cold; a hit may still hide absent pages
    /// between probes, which the driver faults in at its own rate.
    pub(super) fn sampled_resident(&self) -> bool {
        self.layers
            .iter()
            .flatten()
            .flatten()
            .all(|map| sampled_resident(map))
    }

    #[cfg(test)]
    pub(super) fn wait_idle(&self) {
        let (lock, ready) = &*self.queue;
        let mut queue = lock.lock().unwrap_or_else(PoisonError::into_inner);
        while queue.pending > 0 {
            queue = ready.wait(queue).unwrap_or_else(PoisonError::into_inner);
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.queue.1.notify_all();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn run(queue: &(Mutex<Queue>, Condvar), stop: &AtomicBool, issue: &impl Issue) {
    let (lock, ready) = queue;
    loop {
        let map = {
            let mut queue = lock.lock().unwrap_or_else(PoisonError::into_inner);
            loop {
                if stop.load(Ordering::Acquire) {
                    return;
                }
                if let Some(map) = queue.files.pop_front() {
                    break map;
                }
                queue = ready.wait(queue).unwrap_or_else(PoisonError::into_inner);
            }
        };
        issue.read_ahead(&map);
        let mut queue = lock.lock().unwrap_or_else(PoisonError::into_inner);
        queue.pending -= 1;
        if queue.pending == 0 {
            ready.notify_all();
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
    use std::os::unix::fs::FileExt;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

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

    /// Read-ahead completes shortly after the request returns.
    fn wait_resident(map: &Mmap) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while resident_fraction(map) < 1.0 {
            assert!(Instant::now() < deadline, "read-ahead did not land in 10 s");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn assert_layers(layers: &[Option<[Arc<Mmap>; 3]>], resident: &[usize], cold: &[usize]) {
        for &layer in resident {
            for map in layers[layer].as_ref().unwrap() {
                wait_resident(map);
            }
        }
        for &layer in cold {
            for (part, map) in layers[layer].as_ref().unwrap().iter().enumerate() {
                assert_eq!(
                    resident_fraction(map),
                    0.0,
                    "layer {layer} part {part} was read ahead"
                );
            }
        }
    }

    struct Recorder(Arc<Mutex<Vec<usize>>>);
    impl Issue for Recorder {
        fn read_ahead(&self, map: &Mmap) {
            self.0.lock().unwrap().push(map.as_ptr() as usize);
        }
    }

    fn pointers(layers: &[Option<[Arc<Mmap>; 3]>], which: &[usize]) -> Vec<usize> {
        which
            .iter()
            .flat_map(|&layer| layers[layer].as_ref().unwrap().iter())
            .map(|map| map.as_ptr() as usize)
            .collect()
    }

    #[test]
    fn requests_are_exactly_the_files_of_the_layers_ahead() {
        let dir = Temp::new();
        let mut layers = layer_maps(&dir, 9, 1);
        layers[6] = None;
        let issued = Arc::new(Mutex::new(Vec::new()));
        let pool = Pool::new(layers.clone(), Recorder(issued.clone())).unwrap();
        let take = || std::mem::take(&mut *issued.lock().unwrap());

        pool.layer_started(0, 3);
        pool.wait_idle();
        assert_eq!(
            take(),
            pointers(&layers, &[0, 1, 2, 3]),
            "pass start: layers 0..=depth"
        );
        pool.layer_started(1, 3);
        pool.wait_idle();
        assert_eq!(
            take(),
            pointers(&layers, &[4]),
            "layer 1: the one depth ahead"
        );
        pool.layer_started(2, 3);
        pool.layer_started(3, 3);
        pool.wait_idle();
        assert_eq!(
            take(),
            pointers(&layers, &[5]),
            "layer 3 + 3 is absent from the pack"
        );
        pool.layer_started(4, 3);
        pool.layer_started(5, 3);
        pool.layer_started(6, 3);
        pool.layer_started(7, 3);
        pool.wait_idle();
        assert_eq!(
            take(),
            pointers(&layers, &[7, 8]),
            "nothing past the last layer"
        );
        pool.layer_started(0, 0);
        pool.layer_started(2, 0);
        pool.wait_idle();
        assert!(take().is_empty(), "depth 0 requests nothing");

        pool.request_all();
        pool.wait_idle();
        assert_eq!(take(), pointers(&layers, &[0, 1, 2, 3, 4, 5, 7, 8]));
    }

    /// Against the page cache itself: the layers a pass requested are
    /// resident before the model reaches them, the ones it did not are not,
    /// and the sampled check tells a partly cold pack from a warm one.
    #[test]
    fn layers_ahead_become_resident_before_their_dispatch() {
        let dir = Temp::new();
        let layers = layer_maps(&dir, 6, 64);
        for maps in layers.iter().flatten() {
            for map in maps {
                assert_eq!(
                    resident_fraction(map),
                    0.0,
                    "F_NOCACHE write left pages cached"
                );
            }
        }
        let pool = Pool::new(layers.clone(), Advise).unwrap();
        assert!(!pool.sampled_resident());

        pool.layer_started(0, 2);
        pool.wait_idle();
        assert_layers(&layers, &[0, 1, 2], &[3, 4, 5]);
        std::thread::sleep(Duration::from_millis(20));
        assert_layers(&layers, &[], &[3, 4, 5]);
        assert!(!pool.sampled_resident());

        pool.layer_started(1, 2);
        pool.wait_idle();
        assert_layers(&layers, &[3], &[4, 5]);

        pool.layer_started(3, 2);
        pool.wait_idle();
        assert_layers(&layers, &[5], &[4]);
        assert!(!pool.sampled_resident(), "layer 4 is still cold");

        pool.request_all();
        pool.wait_idle();
        assert_layers(&layers, &[0, 1, 2, 3, 4, 5], &[]);
        assert!(pool.sampled_resident());
    }
}
