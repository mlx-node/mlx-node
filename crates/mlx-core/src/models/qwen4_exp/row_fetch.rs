//! Off-thread reads of single table rows (token and PLE embeddings). The
//! decoder knows a token's 17 rows as soon as the token is known; a few
//! reader threads `pread` them from descriptors kept open while the model
//! thread encodes the layers before the rows are consumed. The model thread
//! takes a row's bytes when it imports the row: ready, or read by itself
//! when still queued, or awaited when a reader holds it.
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;

/// `(tensor name, row)`.
pub(super) type Key = (String, usize);

pub(super) struct Job {
    pub key: Key,
    pub file: Arc<File>,
    pub offset: u64,
    pub len: usize,
}

/// Rows a request may leave behind (never taken: a cancelled turn, a
/// replayed token) before the next request drops everything pending.
const PENDING_LIMIT: usize = 1 << 16;

struct State {
    queue: VecDeque<Job>,
    running: HashSet<Key>,
    done: HashMap<Key, std::io::Result<Vec<u8>>>,
}

pub(super) struct Fetcher {
    shared: Arc<(Mutex<State>, Condvar)>,
    stop: Arc<AtomicBool>,
    workers: Vec<JoinHandle<()>>,
}

fn read(job: &Job) -> std::io::Result<Vec<u8>> {
    let mut buf = vec![0; job.len];
    job.file.read_exact_at(&mut buf, job.offset)?;
    Ok(buf)
}

impl Fetcher {
    pub(super) fn new(workers: usize) -> std::io::Result<Self> {
        let shared = Arc::new((
            Mutex::new(State {
                queue: VecDeque::new(),
                running: HashSet::new(),
                done: HashMap::new(),
            }),
            Condvar::new(),
        ));
        let stop = Arc::new(AtomicBool::new(false));
        let workers = (0..workers)
            .map(|_| {
                std::thread::Builder::new()
                    .name("mlx-qwen4-rows".into())
                    .spawn({
                        let shared = shared.clone();
                        let stop = stop.clone();
                        move || run(&shared, &stop)
                    })
            })
            .collect::<std::io::Result<Vec<_>>>()?;
        Ok(Self {
            shared,
            stop,
            workers,
        })
    }

    /// Queue `jobs` behind whatever is still pending, in order.
    pub(super) fn request(&self, jobs: Vec<Job>) {
        let (lock, ready) = &*self.shared;
        let mut state = lock.lock().unwrap_or_else(PoisonError::into_inner);
        if state.queue.len() + state.done.len() > PENDING_LIMIT {
            state.queue.clear();
            state.done.clear();
        }
        state.queue.extend(jobs);
        ready.notify_all();
    }

    /// The bytes of a requested row and whether the model thread had to
    /// read or wait for them; `None` when the row was never requested.
    pub(super) fn take(&self, key: &Key) -> Option<(std::io::Result<Vec<u8>>, bool)> {
        let (lock, ready) = &*self.shared;
        let mut state = lock.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(bytes) = state.done.remove(key) {
            return Some((bytes, false));
        }
        if let Some(position) = state.queue.iter().position(|job| job.key == *key) {
            let job = state.queue.remove(position)?;
            drop(state);
            return Some((read(&job), true));
        }
        if !state.running.contains(key) {
            return None;
        }
        loop {
            if let Some(bytes) = state.done.remove(key) {
                return Some((bytes, true));
            }
            state = ready.wait(state).unwrap_or_else(PoisonError::into_inner);
        }
    }
}

impl Drop for Fetcher {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.shared.1.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn run(shared: &(Mutex<State>, Condvar), stop: &AtomicBool) {
    let (lock, ready) = shared;
    loop {
        let job = {
            let mut state = lock.lock().unwrap_or_else(PoisonError::into_inner);
            loop {
                if stop.load(Ordering::Acquire) {
                    return;
                }
                if let Some(job) = state.queue.pop_front() {
                    state.running.insert(job.key.clone());
                    break job;
                }
                state = ready.wait(state).unwrap_or_else(PoisonError::into_inner);
            }
        };
        let bytes = read(&job);
        let mut state = lock.lock().unwrap_or_else(PoisonError::into_inner);
        state.running.remove(&job.key);
        state.done.insert(job.key, bytes);
        ready.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

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
            offset: (row * 90) as u64,
            len: 90,
        }
    }

    #[test]
    fn requested_rows_arrive_bit_exact_ready_or_read_by_the_taker() {
        let dir = Temp::new();
        let (file, bytes) = table(&dir);
        // Readers: every row ends up ready or awaited, never re-read wrongly.
        let fetcher = Fetcher::new(2).unwrap();
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
        let alone = Fetcher::new(0).unwrap();
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
            offset: 4096 - 10,
            len: 90,
        }]);
        assert!(alone.take(&("table".into(), 99)).unwrap().0.is_err());
    }
}
