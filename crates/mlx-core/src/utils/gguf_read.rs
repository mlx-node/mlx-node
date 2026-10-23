#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

use napi::{Error, Result};

use crate::array::alloc::SendPtr;
use crate::utils::gguf::GgufFile;

pub(crate) const PAGE_CACHE_ENV: &str = "MLX_GGUF_PAGE_CACHE";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PageCacheMode {
    Auto,
    On,
    Off,
}

impl PageCacheMode {
    pub(crate) fn from_env() -> Result<Self> {
        Self::parse(std::env::var(PAGE_CACHE_ENV).ok().as_deref())
    }

    fn parse(value: Option<&str>) -> Result<Self> {
        let Some(raw) = value else {
            return Ok(Self::Auto);
        };
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "on" | "1" | "true" => Ok(Self::On),
            "off" | "0" | "false" => Ok(Self::Off),
            _ => Err(Error::from_reason(format!(
                "{PAGE_CACHE_ENV}={raw:?} is not valid; use auto, on or off"
            ))),
        }
    }

    pub(crate) fn nocache(self, load_bytes: u64, ram: Option<u64>) -> bool {
        match self {
            Self::On => false,
            Self::Off => true,
            Self::Auto => ram.is_some_and(|ram| load_bytes > ram / 2),
        }
    }
}

pub(crate) const READ_THREADS: usize = 8;
pub(crate) const CHUNK_BYTES: usize = 16 << 20;
pub(crate) const ALIGN: u64 = 16 << 10;
pub(crate) const SLOT_BYTES: usize = 64 << 20;
pub(crate) const SLOTS: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Band {
    pub tensor: usize,
    pub row0: usize,
    pub rows: usize,
    pub src_off: usize,
    pub bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Window {
    pub file_off: u64,
    pub len: usize,
    pub bands: Vec<Band>,
}

fn floor_align(v: u64) -> u64 {
    v - v % ALIGN
}

fn ceil_align(v: u64) -> Option<u64> {
    v.checked_next_multiple_of(ALIGN)
}

/// Rows of a tensor as the repack sees them (all dims but the innermost) and
/// the bytes of one row. A tensor whose bytes do not split evenly into rows is
/// one unit.
fn row_geometry(h: &GgufFile, index: usize) -> Result<(usize, u64, u64)> {
    let t = &h.tensors[index];
    let bytes = t.data_size()?;
    let rows = t
        .dims
        .iter()
        .skip(1)
        .try_fold(1u64, |acc, &d| acc.checked_mul(d))
        .unwrap_or(0);
    if rows == 0 || !bytes.is_multiple_of(rows) {
        return Ok((1, bytes, bytes));
    }
    let rows = usize::try_from(rows)
        .map_err(|_| Error::from_reason(format!("GGUF tensor '{}' has too many rows", t.name)))?;
    Ok((rows, bytes / rows as u64, bytes))
}

struct OpenWindow {
    file_off: u64,
    end: u64,
    bands: Vec<Band>,
}

impl OpenWindow {
    fn close(self, file_len: u64) -> Result<Window> {
        let end = ceil_align(self.end)
            .ok_or_else(|| Error::from_reason("GGUF read window end overflows u64"))?
            .min(file_len);
        let len = usize::try_from(end - self.file_off)
            .map_err(|_| Error::from_reason("GGUF read window does not fit in memory"))?;
        Ok(Window {
            file_off: self.file_off,
            len,
            bands: self.bands,
        })
    }
}

/// Windows over `h`'s tensors in offset order. Each window starts on an
/// `ALIGN` boundary, ends on one (or at EOF) and is at most `slot_bytes` long
/// after rounding. A tensor is one band or several row bands; a band's row
/// count is a multiple of `row_multiple(tensor)` except the tensor's last.
pub(crate) fn plan_reads(
    h: &GgufFile,
    file_len: u64,
    slot_bytes: usize,
    row_multiple: &dyn Fn(usize) -> usize,
) -> Result<Vec<Window>> {
    let slot = slot_bytes as u64;
    let mut order: Vec<usize> = (0..h.tensors.len()).collect();
    order.sort_by_key(|&i| h.tensors[i].offset);

    let mut windows = Vec::new();
    let mut open: Option<OpenWindow> = None;
    let mut prev_end = 0u64;
    for index in order {
        let t = &h.tensors[index];
        let (rows, row_bytes, bytes) = row_geometry(h, index)?;
        if bytes == 0 {
            continue;
        }
        let start = h.data_offset.checked_add(t.offset).ok_or_else(|| {
            Error::from_reason(format!("GGUF tensor '{}' offset overflows", t.name))
        })?;
        let end = start
            .checked_add(bytes)
            .filter(|&end| end <= file_len)
            .ok_or_else(|| {
                Error::from_reason(format!(
                    "GGUF tensor '{}' ends past the end of the file ({file_len} bytes)",
                    t.name
                ))
            })?;
        if start < prev_end {
            return Err(Error::from_reason(format!(
                "GGUF tensor '{}' overlaps the tensor before it",
                t.name
            )));
        }
        prev_end = end;
        let unit = row_multiple(index).max(1);

        let mut row0 = 0usize;
        while row0 < rows {
            let band_start = start + row0 as u64 * row_bytes;
            let window = open.get_or_insert_with(|| OpenWindow {
                file_off: floor_align(band_start),
                end: band_start,
                bands: Vec::new(),
            });
            let limit = window.file_off.saturating_add(slot);
            let max_end = if file_len <= limit {
                file_len
            } else {
                floor_align(limit)
            };
            let fit = max_end.saturating_sub(band_start) / row_bytes;
            let left = rows - row0;
            let take = if fit >= left as u64 {
                left
            } else {
                (fit as usize / unit) * unit
            };
            if take == 0 {
                if window.bands.is_empty() {
                    return Err(Error::from_reason(format!(
                        "GGUF tensor '{}': {unit} row(s) of {row_bytes} bytes do not fit in a \
                         {slot_bytes}-byte read slot",
                        t.name
                    )));
                }
                if let Some(full) = open.take() {
                    windows.push(full.close(file_len)?);
                }
                continue;
            }
            let band_bytes = take as u64 * row_bytes;
            window.bands.push(Band {
                tensor: index,
                row0,
                rows: take,
                src_off: (band_start - window.file_off) as usize,
                bytes: band_bytes as usize,
            });
            window.end = band_start + band_bytes;
            row0 += take;
        }
    }
    if let Some(last) = open.take() {
        windows.push(last.close(file_len)?);
    }
    Ok(windows)
}

pub(crate) trait ReadAt: Send + Sync {
    fn read_exact_at(&self, buf: &mut [u8], off: u64) -> std::io::Result<()>;
}

#[cfg(unix)]
impl ReadAt for std::fs::File {
    fn read_exact_at(&self, buf: &mut [u8], off: u64) -> std::io::Result<()> {
        std::os::unix::fs::FileExt::read_exact_at(self, buf, off)
    }
}

#[derive(Debug)]
pub(crate) struct GgufReadError {
    pub path: PathBuf,
    pub offset: u64,
    pub len: usize,
    pub source: std::io::Error,
}

impl fmt::Display for GgufReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "failed to read GGUF '{}' at offset {} ({} bytes): {}",
            self.path.display(),
            self.offset,
            self.len,
            self.source
        )
    }
}

impl std::error::Error for GgufReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

impl From<GgufReadError> for Error {
    fn from(e: GgufReadError) -> Self {
        Error::from_reason(e.to_string())
    }
}

struct Job {
    ticket: u64,
    dst: SendPtr<u8>,
    len: usize,
    off: u64,
}

#[derive(Default)]
struct TicketState {
    pending: usize,
    skipped: bool,
    error: Option<GgufReadError>,
}

struct FirstError {
    offset: u64,
    len: usize,
    raw: Option<i32>,
    kind: std::io::ErrorKind,
    message: String,
}

#[derive(Default)]
struct State {
    queue: VecDeque<Job>,
    tickets: HashMap<u64, TicketState>,
    in_flight: usize,
    closed: bool,
    first_error: Option<FirstError>,
}

impl State {
    /// Drop every queued chunk; their tickets end with the first error.
    fn drain(&mut self) {
        while let Some(job) = self.queue.pop_front() {
            self.in_flight -= 1;
            if let Some(t) = self.tickets.get_mut(&job.ticket) {
                t.pending -= 1;
                t.skipped = true;
            }
        }
    }
}

struct Shared {
    state: Mutex<State>,
    changed: Condvar,
    stop: AtomicBool,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Parallel positional reads into caller-owned buffers. Workers live in the
/// caller's thread scope, so a buffer handed to `submit` must outlive that
/// scope or be waited on before it is freed.
pub(crate) struct Reader<'s> {
    shared: Arc<Shared>,
    path: &'s Path,
    chunk: usize,
}

impl<'s> Reader<'s> {
    pub(crate) fn start(
        scope: &'s std::thread::Scope<'s, '_>,
        src: &'s dyn ReadAt,
        path: &'s Path,
        threads: usize,
        chunk: usize,
    ) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            changed: Condvar::new(),
            stop: AtomicBool::new(false),
        });
        for _ in 0..threads.max(1) {
            let shared = Arc::clone(&shared);
            scope.spawn(move || worker(&shared, src, path));
        }
        Self {
            shared,
            path,
            chunk: chunk.max(1),
        }
    }

    pub(crate) fn submit(&self, ticket: u64, dst: SendPtr<u8>, dst_len: usize, file_off: u64) {
        let mut state = self.shared.lock();
        let entry = state.tickets.entry(ticket).or_default();
        if self.shared.stop.load(Ordering::Acquire) {
            entry.skipped = true;
            return;
        }
        let mut chunks = 0usize;
        let mut done = 0usize;
        while done < dst_len {
            let len = self.chunk.min(dst_len - done);
            state.queue.push_back(Job {
                ticket,
                dst: SendPtr(dst.0.wrapping_add(done)),
                len,
                off: file_off + done as u64,
            });
            done += len;
            chunks += 1;
        }
        state.in_flight += chunks;
        if let Some(t) = state.tickets.get_mut(&ticket) {
            t.pending += chunks;
        }
        drop(state);
        self.shared.changed.notify_all();
    }

    pub(crate) fn wait(&self, ticket: u64) -> std::result::Result<(), GgufReadError> {
        let mut state = self.shared.lock();
        loop {
            match state.tickets.get(&ticket) {
                None => {
                    return Err(GgufReadError {
                        path: self.path.to_path_buf(),
                        offset: 0,
                        len: 0,
                        source: std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            format!("GGUF read ticket {ticket} was never submitted"),
                        ),
                    });
                }
                Some(t) if t.pending == 0 => {
                    let t = state.tickets.remove(&ticket).unwrap_or_default();
                    if let Some(e) = t.error {
                        return Err(e);
                    }
                    if t.skipped {
                        return Err(self.stopped_error(&state));
                    }
                    return Ok(());
                }
                Some(_) => {
                    state = self
                        .shared
                        .changed
                        .wait(state)
                        .unwrap_or_else(PoisonError::into_inner);
                }
            }
        }
    }

    fn stopped_error(&self, state: &State) -> GgufReadError {
        match &state.first_error {
            Some(first) => GgufReadError {
                path: self.path.to_path_buf(),
                offset: first.offset,
                len: first.len,
                source: match first.raw {
                    Some(code) => std::io::Error::from_raw_os_error(code),
                    None => std::io::Error::new(first.kind, first.message.clone()),
                },
            },
            None => GgufReadError {
                path: self.path.to_path_buf(),
                offset: 0,
                len: 0,
                source: std::io::Error::new(std::io::ErrorKind::Interrupted, "GGUF read stopped"),
            },
        }
    }

    pub(crate) fn wait_all_issued(&self) {
        let mut state = self.shared.lock();
        while state.in_flight > 0 {
            state = self
                .shared
                .changed
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    pub(crate) fn stop(&self) {
        self.shared.stop.store(true, Ordering::Release);
        self.shared.lock().drain();
        self.shared.changed.notify_all();
    }
}

impl Drop for Reader<'_> {
    fn drop(&mut self) {
        self.stop();
        self.shared.lock().closed = true;
        self.shared.changed.notify_all();
    }
}

fn worker(shared: &Shared, src: &dyn ReadAt, path: &Path) {
    loop {
        let job = {
            let mut state = shared.lock();
            loop {
                if let Some(job) = state.queue.pop_front() {
                    break job;
                }
                if state.closed {
                    return;
                }
                state = shared
                    .changed
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
            }
        };
        let result = if shared.stop.load(Ordering::Acquire) {
            None
        } else {
            // SAFETY: `submit`'s caller keeps `dst..dst+len` alive and unaliased
            // until the ticket is waited on or the scope ends.
            let buf = unsafe { std::slice::from_raw_parts_mut(job.dst.0, job.len) };
            Some(src.read_exact_at(buf, job.off))
        };
        let mut state = shared.lock();
        state.in_flight -= 1;
        match result {
            Some(Ok(())) => {}
            None => {
                if let Some(t) = state.tickets.get_mut(&job.ticket) {
                    t.skipped = true;
                }
            }
            Some(Err(source)) => {
                shared.stop.store(true, Ordering::Release);
                if state.first_error.is_none() {
                    state.first_error = Some(FirstError {
                        offset: job.off,
                        len: job.len,
                        raw: source.raw_os_error(),
                        kind: source.kind(),
                        message: source.to_string(),
                    });
                }
                if let Some(t) = state.tickets.get_mut(&job.ticket)
                    && t.error.is_none()
                {
                    t.error = Some(GgufReadError {
                        path: path.to_path_buf(),
                        offset: job.off,
                        len: job.len,
                        source,
                    });
                }
                state.drain();
            }
        }
        if let Some(t) = state.tickets.get_mut(&job.ticket) {
            t.pending -= 1;
        }
        drop(state);
        shared.changed.notify_all();
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn set_nocache(file: &std::fs::File) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;
    // SAFETY: `file` owns an open descriptor; F_NOCACHE only changes caching.
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn set_nocache(_file: &std::fs::File) -> std::io::Result<()> {
    Ok(())
}

const RESIDENT_PROBES: usize = 8;
const RESIDENT_PROBE_PAGES: usize = 16;

/// Fraction of sampled pages of `file[..len]` in the page cache. Maps the file
/// and asks mincore; never touches a page.
#[cfg(unix)]
pub(crate) fn sampled_resident_fraction(file: &std::fs::File, len: u64) -> Option<f64> {
    use std::os::fd::AsRawFd;
    let len = usize::try_from(len).ok().filter(|&l| l > 0)?;
    // SAFETY: sysconf only reads system configuration.
    let page = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
        .ok()
        .filter(|&p| p > 0)?;
    // SAFETY: a read-only shared mapping of an open descriptor; unmapped below.
    let base = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            0,
        )
    };
    if base == libc::MAP_FAILED {
        return None;
    }
    let pages = len.div_ceil(page);
    let mut vec = [0u8; RESIDENT_PROBE_PAGES];
    let mut sampled = 0usize;
    let mut resident = 0usize;
    let mut failed = false;
    for probe in 0..RESIDENT_PROBES {
        let first = pages * probe / RESIDENT_PROBES;
        let count = RESIDENT_PROBE_PAGES.min(pages - first);
        if count == 0 {
            continue;
        }
        let bytes = (count * page).min(len - first * page);
        // SAFETY: the range lies inside the live mapping and `vec` holds one
        // byte per page of it; mincore only reads.
        let rc = unsafe {
            libc::mincore(
                base.cast::<u8>()
                    .add(first * page)
                    .cast::<std::ffi::c_void>(),
                bytes,
                vec.as_mut_ptr().cast(),
            )
        };
        if rc != 0 {
            failed = true;
            break;
        }
        sampled += count;
        resident += vec[..count].iter().filter(|&&flags| flags & 1 != 0).count();
    }
    // SAFETY: `base`/`len` is the mapping created above.
    unsafe { libc::munmap(base, len) };
    if failed || sampled == 0 {
        return None;
    }
    Some(resident as f64 / sampled as f64)
}

#[cfg(not(unix))]
pub(crate) fn sampled_resident_fraction(_file: &std::fs::File, _len: u64) -> Option<f64> {
    None
}

pub(crate) fn log_load_start(path: &Path, load_bytes: u64, nocache: bool, resident: Option<f64>) {
    let resident = resident.map_or_else(|| "unknown".to_string(), |r| format!("{:.0}%", r * 100.0));
    tracing::info!(
        "GGUF load '{}': {:.2} GB, page cache {}, resident {}",
        path.display(),
        load_bytes as f64 / 1e9,
        if nocache { "bypassed" } else { "used" },
        resident
    );
}

const SLOW_DISK_MIN_BYTES: u64 = 4 << 30;
const SLOW_DISK_GBPS: f64 = 2.0;

fn slow_disk_rate(load_bytes: u64, first_gib_secs: f64) -> Option<f64> {
    if load_bytes < SLOW_DISK_MIN_BYTES || first_gib_secs <= 0.0 || !first_gib_secs.is_finite() {
        return None;
    }
    let gbps = (1u64 << 30) as f64 / first_gib_secs / 1e9;
    (gbps < SLOW_DISK_GBPS).then_some(gbps)
}

pub(crate) fn maybe_log_slow_disk(path: &Path, load_bytes: u64, first_gib_secs: f64) {
    if let Some(gbps) = slow_disk_rate(load_bytes, first_gib_secs) {
        tracing::warn!(
            "GGUF '{}' reads at {:.2} GB/s; loading {:.1} GB from this disk will be slow. \
             An internal SSD loads it several times faster.",
            path.display(),
            gbps,
            load_bytes as f64 / 1e9
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};
    use std::sync::atomic::AtomicUsize;

    fn header(
        alignment: u64,
        data_offset: u64,
        tensors: Vec<(Vec<u64>, GgufTensorType)>,
    ) -> (GgufFile, u64) {
        let mut offset = 0u64;
        let mut infos = Vec::new();
        for (i, (dims, tensor_type)) in tensors.into_iter().enumerate() {
            offset = offset.next_multiple_of(alignment);
            let info = GgufTensorInfo {
                name: format!("t{i}"),
                n_dims: dims.len() as u32,
                dims,
                tensor_type,
                offset,
            };
            offset += info.data_size().unwrap();
            infos.push(info);
        }
        let file_len = data_offset + offset;
        (
            GgufFile {
                version: 3,
                tensor_count: infos.len() as u64,
                metadata: HashMap::new(),
                tensors: infos,
                alignment,
                data_offset,
            },
            file_len,
        )
    }

    fn check_plan(h: &GgufFile, file_len: u64, slot: usize, multiple: &dyn Fn(usize) -> usize) {
        let windows = plan_reads(h, file_len, slot, multiple).unwrap();
        let mut covered: Vec<Vec<(usize, usize)>> = vec![Vec::new(); h.tensors.len()];
        for w in &windows {
            assert!(w.len <= slot, "window {w:?} longer than slot {slot}");
            assert!(w.file_off + w.len as u64 <= file_len);
            for b in &w.bands {
                assert!(
                    b.src_off + b.bytes <= w.len,
                    "band {b:?} outside window {w:?}"
                );
                let t = &h.tensors[b.tensor];
                let (rows, row_bytes, _) = row_geometry(h, b.tensor).unwrap();
                let abs = h.data_offset + t.offset + b.row0 as u64 * row_bytes;
                assert_eq!(w.file_off + b.src_off as u64, abs);
                assert_eq!(b.bytes as u64, b.rows as u64 * row_bytes);
                let unit = multiple(b.tensor).max(1);
                if b.row0 + b.rows < rows {
                    assert!(
                        b.rows.is_multiple_of(unit),
                        "band {b:?} not a multiple of {unit}"
                    );
                }
                covered[b.tensor].push((b.row0, b.rows));
            }
        }
        for (i, bands) in covered.iter_mut().enumerate() {
            let (rows, _, bytes) = row_geometry(h, i).unwrap();
            if bytes == 0 {
                continue;
            }
            bands.sort();
            let mut next = 0usize;
            for &(row0, n) in bands.iter() {
                assert_eq!(row0, next, "tensor {i}: gap or overlap at row {row0}");
                next += n;
            }
            assert_eq!(next, rows, "tensor {i}: covered {next} of {rows} rows");
        }
    }

    #[test]
    fn read_plan_covers_every_byte_once() {
        const MIB: usize = 1 << 20;
        for alignment in [32u64, 2, 1] {
            let (h, len) = header(
                alignment,
                4096 + 7,
                vec![
                    (vec![1], GgufTensorType::F32),
                    (vec![3], GgufTensorType::F16),
                    (vec![256, 64], GgufTensorType::Q4K),
                    (vec![5120, 640], GgufTensorType::Q4K),
                    (vec![4096, 6400], GgufTensorType::F32),
                    (vec![7], GgufTensorType::BF16),
                    (vec![2048, 3000], GgufTensorType::Q6K),
                    (vec![17, 5], GgufTensorType::F32),
                ],
            );
            assert_eq!(h.tensors[4].data_size().unwrap(), 100 * MIB as u64);
            check_plan(&h, len, MIB, &|_| 1);
            check_plan(&h, len, MIB, &|t| if t == 3 || t == 6 { 64 } else { 1 });
            check_plan(&h, len, 3 * MIB + 5, &|t| if t == 3 { 64 } else { 1 });
        }
    }

    #[test]
    fn row_unit_larger_than_slot_is_an_error() {
        let (h, len) = header(32, 64, vec![(vec![1 << 20], GgufTensorType::F32)]);
        assert!(plan_reads(&h, len, 1 << 20, &|_| 1).is_err());
        let (h, len) = header(32, 64, vec![(vec![4096, 128], GgufTensorType::F32)]);
        assert!(plan_reads(&h, len, 1 << 20, &|_| 128).is_err());
        plan_reads(&h, len, 1 << 20, &|_| 32).unwrap();
    }

    #[test]
    fn windows_are_16k_aligned() {
        let (h, len) = header(
            32,
            1000,
            vec![
                (vec![4096, 300], GgufTensorType::F32),
                (vec![9], GgufTensorType::F32),
                (vec![256, 999], GgufTensorType::Q5K),
            ],
        );
        let windows = plan_reads(&h, len, 1 << 20, &|_| 1).unwrap();
        assert!(windows.len() > 3);
        for w in &windows {
            assert_eq!(w.file_off % ALIGN, 0, "{w:?}");
            let end = w.file_off + w.len as u64;
            assert!(
                end == len || w.len as u64 % ALIGN == 0,
                "{w:?} file_len {len}"
            );
        }
        assert_eq!(windows.last().map(|w| w.file_off + w.len as u64), Some(len));
    }

    struct Bytes(Vec<u8>);
    impl ReadAt for Bytes {
        fn read_exact_at(&self, buf: &mut [u8], off: u64) -> std::io::Result<()> {
            let off = off as usize;
            let src = self.0.get(off..off + buf.len()).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "short read")
            })?;
            buf.copy_from_slice(src);
            Ok(())
        }
    }

    #[test]
    fn reader_matches_std_read() {
        let mut data = vec![0u8; 200 << 20];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for chunk in data.chunks_mut(8) {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bytes = state.to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
        let path = std::env::temp_dir().join(format!("gguf-read-{}.bin", uuid::Uuid::new_v4()));
        std::fs::write(&path, &data).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let mut out = vec![0u8; data.len() - 12345];
        let result = std::thread::scope(|scope| {
            let reader = Reader::start(scope, &file, &path, 8, CHUNK_BYTES);
            let half = out.len() / 2;
            let base = out.as_mut_ptr();
            reader.submit(1, SendPtr(base), half, 12345);
            reader.submit(
                2,
                SendPtr(base.wrapping_add(half)),
                out.len() - half,
                12345 + half as u64,
            );
            reader.wait(2).and_then(|()| reader.wait(1))
        });
        let _ = std::fs::remove_file(&path);
        result.unwrap();
        assert!(out == data[12345..], "reader bytes differ from the file");
    }

    struct FailFirst {
        reads: AtomicUsize,
        after_fail: AtomicUsize,
        failed: AtomicBool,
        fail_off: u64,
    }
    impl ReadAt for FailFirst {
        fn read_exact_at(&self, buf: &mut [u8], off: u64) -> std::io::Result<()> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            if self.failed.load(Ordering::SeqCst) {
                self.after_fail.fetch_add(1, Ordering::SeqCst);
            }
            if off == self.fail_off {
                self.failed.store(true, Ordering::SeqCst);
                return Err(std::io::Error::from_raw_os_error(libc::EIO));
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
            buf.fill(1);
            Ok(())
        }
    }

    #[test]
    fn reader_error_stops_workers_and_names_path_offset() {
        let threads = 4usize;
        let chunk = 4096usize;
        let chunks = 16 * threads;
        let src = FailFirst {
            reads: AtomicUsize::new(0),
            after_fail: AtomicUsize::new(0),
            failed: AtomicBool::new(false),
            fail_off: 1 << 20,
        };
        let path = PathBuf::from("/models/broken.gguf");
        let mut buf = vec![0u8; chunks * chunk];
        let (first, second) = std::thread::scope(|scope| {
            let reader = Reader::start(scope, &src, &path, threads, chunk);
            reader.submit(7, SendPtr(buf.as_mut_ptr()), buf.len(), 1 << 20);
            let first = reader.wait(7);
            reader.submit(8, SendPtr(buf.as_mut_ptr()), chunk, 0);
            let second = reader.wait(8);
            reader.wait_all_issued();
            (first, second)
        });
        let err = first.unwrap_err();
        assert_eq!(err.path, path);
        assert_eq!(err.offset, 1 << 20);
        assert_eq!(err.len, chunk);
        let text = err.to_string();
        assert!(text.contains("Input/output error"), "{text}");
        assert!(text.contains("/models/broken.gguf"), "{text}");
        let err = second.unwrap_err();
        assert_eq!(err.offset, 1 << 20);
        let after = src.after_fail.load(Ordering::SeqCst);
        assert!(after < threads, "{after} reads issued after the failure");
        assert!(src.reads.load(Ordering::SeqCst) <= threads);
    }

    #[test]
    fn reader_stop_ends_pending_tickets() {
        let src = Bytes(vec![3u8; 1 << 20]);
        let path = PathBuf::from("/m.gguf");
        let mut buf = vec![0u8; 1 << 20];
        let result = std::thread::scope(|scope| {
            let reader = Reader::start(scope, &src, &path, 2, 4096);
            reader.stop();
            reader.submit(1, SendPtr(buf.as_mut_ptr()), buf.len(), 0);
            reader.wait_all_issued();
            reader.wait(1)
        });
        assert_eq!(
            result.unwrap_err().source.kind(),
            std::io::ErrorKind::Interrupted
        );
    }

    #[test]
    fn page_cache_auto_threshold() {
        let ram = 64u64 << 30;
        assert!(!PageCacheMode::Auto.nocache(ram / 2, Some(ram)));
        assert!(PageCacheMode::Auto.nocache(ram / 2 + 1, Some(ram)));
        assert!(!PageCacheMode::Auto.nocache(u64::MAX, None));
        assert!(!PageCacheMode::On.nocache(u64::MAX, Some(ram)));
        assert!(PageCacheMode::Off.nocache(0, Some(ram)));
        assert_eq!(PageCacheMode::parse(None).unwrap(), PageCacheMode::Auto);
        assert_eq!(
            PageCacheMode::parse(Some("auto")).unwrap(),
            PageCacheMode::Auto
        );
        assert_eq!(PageCacheMode::parse(Some("ON")).unwrap(), PageCacheMode::On);
        assert_eq!(
            PageCacheMode::parse(Some("off")).unwrap(),
            PageCacheMode::Off
        );
        assert!(PageCacheMode::parse(Some("sometimes")).is_err());
    }

    #[test]
    fn resident_fraction_in_unit_range() {
        let path = std::env::temp_dir().join(format!("gguf-resident-{}.bin", uuid::Uuid::new_v4()));
        std::fs::write(&path, vec![5u8; 8 << 20]).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let fraction = sampled_resident_fraction(&file, 8 << 20);
        let empty = sampled_resident_fraction(&file, 0);
        let _ = std::fs::remove_file(&path);
        let fraction = fraction.unwrap();
        assert!((0.0..=1.0).contains(&fraction), "{fraction}");
        assert!(empty.is_none());
    }

    #[test]
    fn slow_disk_hint_threshold() {
        assert!(slow_disk_rate(3 << 30, 10.0).is_none());
        assert!(slow_disk_rate(8 << 30, 0.1).is_none());
        assert!(slow_disk_rate(8 << 30, 1.0).is_some());
        assert!(slow_disk_rate(8 << 30, 0.0).is_none());
    }
}
