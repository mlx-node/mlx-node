//! Read-only file mappings that back MLX arrays without a copy.
//!
//! [`Mmap`] owns a `PROT_READ, MAP_SHARED` mapping of a file range. An array
//! built over it ([`MxArray::from_mapped`]) keeps the mapping alive through
//! MLX's buffer deleter, so the `Arc<Mmap>` may be dropped by the caller as
//! soon as the arrays exist. The pages are clean page cache: the OS may
//! evict them, and the Metal driver wires them while a command buffer that
//! binds the array is in flight.

use super::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use std::ffi::c_void;
use std::fs::File;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd};
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Bytes of mapping storage held by live arrays. MLX counts every no-copy
/// buffer in `active_memory` (`allocator.cpp` `make_buffer`), so memory
/// budgets subtract this number to see the anonymous footprint.
pub(super) static MAPPED_ARRAY_BYTES: AtomicU64 = AtomicU64::new(0);

pub fn mapped_array_bytes() -> u64 {
    MAPPED_ARRAY_BYTES.load(Ordering::Acquire)
}

pub fn page_size() -> usize {
    // SAFETY: sysconf has no preconditions.
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    usize::try_from(size).unwrap_or(16384)
}

/// A read-only shared mapping of `len` bytes of `file` starting at `offset`.
pub struct Mmap {
    ptr: NonNull<u8>,
    len: usize,
    file: File,
    offset: u64,
}

// SAFETY: the mapping is read-only and the struct only hands out raw
// pointers; the file descriptor is only used for metadata and advice calls.
unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Mmap {
    /// Map `[offset, offset + len)` of `file`. `offset` must be page-aligned
    /// and the range must lie inside the file; the mapping takes ownership of
    /// `file` so prefetch calls can keep using its descriptor.
    pub fn map(file: File, offset: u64, len: usize) -> Result<Arc<Self>> {
        let page = page_size() as u64;
        if !offset.is_multiple_of(page) {
            return Err(Error::from_reason(format!(
                "Mmap::map: offset {offset} is not page-aligned ({page} B pages)"
            )));
        }
        if len == 0 {
            return Err(Error::from_reason("Mmap::map: empty range"));
        }
        let file_len = file.metadata()?.len();
        let end = offset
            .checked_add(len as u64)
            .ok_or_else(|| Error::from_reason("Mmap::map: range overflows u64"))?;
        if end > file_len {
            return Err(Error::from_reason(format!(
                "Mmap::map: range {offset}..{end} exceeds the file length {file_len}"
            )));
        }
        let off = libc::off_t::try_from(offset)
            .map_err(|_| Error::from_reason("Mmap::map: offset exceeds off_t"))?;
        // SAFETY: fd is open, offset is page-aligned, the range is inside the
        // file, and PROT_READ | MAP_SHARED never writes through the mapping.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                off,
            )
        };
        if ptr == libc::MAP_FAILED {
            let err = std::io::Error::last_os_error();
            return Err(Error::from_reason(format!(
                "Mmap::map: mmap of {len} bytes at offset {offset} failed: {err}"
            )));
        }
        let ptr = NonNull::new(ptr.cast::<u8>())
            .ok_or_else(|| Error::from_reason("Mmap::map: mmap returned null"))?;
        Ok(Arc::new(Self {
            ptr,
            len,
            file,
            offset,
        }))
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// File offset of the first mapped byte.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn fd(&self) -> BorrowedFd<'_> {
        self.file.as_fd()
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        // SAFETY: ptr/len came from a successful mmap and no array references
        // the mapping any more (each one holds an Arc until its buffer dies).
        unsafe {
            libc::munmap(self.ptr.as_ptr().cast::<c_void>(), self.len);
        }
    }
}

/// Handed to MLX as the buffer deleter context: keeps the mapping alive for
/// as long as any array shares the buffer.
pub(super) struct MappedLease {
    pub(super) _map: Arc<Mmap>,
    pub(super) bytes: usize,
}

pub(super) unsafe extern "C" fn release_mapped(ctx: *mut c_void) {
    if ctx.is_null() {
        return;
    }
    // SAFETY: ctx is the Box<MappedLease> leaked by from_mapped; MLX calls the
    // deleter exactly once, on whichever thread releases the buffer.
    let lease = unsafe { Box::from_raw(ctx.cast::<MappedLease>()) };
    MAPPED_ARRAY_BYTES.fetch_sub(lease.bytes as u64, Ordering::AcqRel);
    drop(lease);
}

impl MxArray {
    /// Storage pointer of the array; null unless the array is available (a
    /// mapped array, or an evaluated array that was already waited on). An
    /// async-evaluated array stays null until [`MxArray::eval`] waits on it.
    pub fn storage_ptr(&self) -> *const c_void {
        // SAFETY: the handle is a live array.
        unsafe { sys::mlx_array_data_ptr(self.handle.0) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::DType;
    use std::io::Write;
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    // The byte counter is process-wide, so the tests that build mapped arrays
    // run one at a time.
    static SERIAL: Mutex<()> = Mutex::new(());

    struct TempFile(std::path::PathBuf);

    impl TempFile {
        fn with_words(name: &str, words: &[u32]) -> Self {
            let path =
                std::env::temp_dir().join(format!("mlx-mapped-{}-{name}.bin", std::process::id()));
            let mut file = File::create(&path).unwrap();
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            file.write_all(&bytes).unwrap();
            file.sync_all().unwrap();
            Self(path)
        }

        fn open(&self) -> File {
            File::open(&self.0).unwrap()
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn pattern(words: usize) -> Vec<u32> {
        (0..words as u32)
            .map(|i| i.wrapping_mul(2_654_435_761) % 100_000)
            .collect()
    }

    fn cpu_sum(words: &[u32]) -> u32 {
        words.iter().fold(0u32, |acc, &w| acc.wrapping_add(w))
    }

    // A GPU op's inputs stay retained by its command buffer's completion
    // handler for a moment after eval returns.
    fn wait_for_leases(map: &Arc<Mmap>) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while Arc::strong_count(map) > 1 {
            assert!(
                Instant::now() < deadline,
                "a mapped buffer outlived its arrays by 10 s"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn reason<T>(result: Result<T>) -> String {
        match result {
            Ok(_) => panic!("expected an error"),
            Err(err) => err.reason,
        }
    }

    #[test]
    fn mapped_array_storage_is_the_mapping() {
        let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let page = page_size();
        let words = pattern(2 * page / 4);
        let file = TempFile::with_words("storage", &words);
        let map = Mmap::map(file.open(), 0, 2 * page).unwrap();
        let before = mapped_array_bytes();

        let arr = MxArray::from_mapped(&map, 0, &[words.len() as i64], DType::Uint32).unwrap();
        assert_eq!(arr.storage_ptr(), map.as_ptr().cast::<c_void>());
        assert_eq!(mapped_array_bytes(), before + 2 * page as u64);

        let view = arr.clone();
        drop(arr);
        assert_eq!(mapped_array_bytes(), before + 2 * page as u64);
        drop(view);
        assert_eq!(mapped_array_bytes(), before);

        let second_page =
            MxArray::from_mapped(&map, page, &[(page / 4) as i64], DType::Uint32).unwrap();
        assert_eq!(second_page.storage_ptr(), unsafe {
            map.as_ptr().add(page).cast::<c_void>()
        });
        drop(map);
        assert_eq!(
            second_page.to_uint32().unwrap().as_ref(),
            &words[page / 4..]
        );
    }

    #[test]
    fn mapped_array_gpu_sum_matches_cpu_sum() {
        let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let page = page_size();
        let words = pattern(4 * page / 4);
        let file = TempFile::with_words("sum", &words);
        let map = Mmap::map(file.open(), 0, 4 * page).unwrap();
        let before = mapped_array_bytes();
        let on_disk = std::fs::read(&file.0).unwrap();
        let disk_words: Vec<u32> = on_disk
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        assert_eq!(disk_words, words);

        let whole = MxArray::from_mapped(&map, 0, &[words.len() as i64], DType::Uint32).unwrap();
        let gpu = whole.sum(None, None).unwrap();
        assert_eq!(gpu.to_uint32().unwrap().as_ref(), &[cpu_sum(&disk_words)]);

        let tail_words = 2 * page / 4;
        let tail =
            MxArray::from_mapped(&map, 2 * page, &[2, (tail_words / 2) as i64], DType::Uint32)
                .unwrap();
        let tail_sum = tail.sum(None, None).unwrap();
        assert_eq!(
            tail_sum.to_uint32().unwrap().as_ref(),
            &[cpu_sum(&disk_words[2 * page / 4..])]
        );

        drop((whole, gpu, tail, tail_sum));
        wait_for_leases(&map);
        assert_eq!(mapped_array_bytes(), before);
    }

    #[test]
    fn storage_ptr_never_consumes_a_pending_event() {
        let words = pattern(1024);
        let total = [cpu_sum(&words)];
        let x = MxArray::from_uint32(&words, &[words.len() as i64]).unwrap();

        let lazy = x.sum(None, None).unwrap();
        assert!(lazy.storage_ptr().is_null(), "unscheduled array");

        let synced = x.sum(None, None).unwrap();
        synced.eval();
        assert!(!synced.storage_ptr().is_null(), "sync eval waited on it");

        let pending = x.sum(None, None).unwrap();
        MxArray::async_eval_arrays(&[&pending]);
        crate::array::memory::synchronize();
        assert!(
            pending.storage_ptr().is_null(),
            "async-evaluated array whose signaled event nobody waited on"
        );
        pending.eval();
        assert!(!pending.storage_ptr().is_null(), "eval waited on it");
        assert_eq!(pending.to_uint32().unwrap().as_ref(), &total);
        assert_eq!(synced.to_uint32().unwrap().as_ref(), &total);
    }

    #[test]
    fn mapped_array_rejects_unaligned_offsets_without_copying() {
        let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let page = page_size();
        let words = pattern(2 * page / 4);
        let file = TempFile::with_words("align", &words);
        let map = Mmap::map(file.open(), 0, 2 * page).unwrap();
        let before = mapped_array_bytes();

        for offset in [1usize, 4, page + 4, page - 4] {
            let text = reason(MxArray::from_mapped(&map, offset, &[1024], DType::Uint32));
            assert!(text.contains("page-aligned"), "offset {offset}: {text}");
        }
        assert_eq!(mapped_array_bytes(), before);

        let past_end = MxArray::from_mapped(&map, page, &[(page / 4 + 1) as i64], DType::Uint32);
        assert!(reason(past_end).contains("exceeds the mapping"));
        let empty = MxArray::from_mapped(&map, 0, &[0], DType::Uint32);
        assert!(reason(empty).contains("empty"));
        assert_eq!(mapped_array_bytes(), before);

        let unaligned_map = Mmap::map(file.open(), 4, page);
        assert!(reason(unaligned_map).contains("page-aligned"));
        let oversize_map = Mmap::map(file.open(), page as u64, 2 * page);
        assert!(reason(oversize_map).contains("exceeds the file"));
    }
}
