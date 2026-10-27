//! CPU-visible native Metal banks used without a copy. Volatile and empty
//! banks remain in MLX's active-allocation count until their arrays die;
//! `reclaimable_purgeable_bytes` separates their reclaimable storage for memory
//! admission.

// Expert-slot ownership is added separately from this allocation primitive.
#![allow(dead_code)]

use super::{DType, MxArray};
use crate::array::alloc::SendPtr;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use std::ffi::c_void;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

static RECLAIMABLE_PURGEABLE_BYTES: AtomicU64 = AtomicU64::new(0);

/// Metal's three explicit purgeability levels. `KeepCurrent` is a query, not a
/// level a bank can transition to.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PurgeableState {
    NonVolatile = 2,
    Volatile = 3,
    Empty = 4,
}

impl PurgeableState {
    fn from_code(code: i32) -> Result<Self> {
        match code {
            2 => Ok(Self::NonVolatile),
            3 => Ok(Self::Volatile),
            4 => Ok(Self::Empty),
            _ => Err(Error::from_reason(format!(
                "Metal returned unknown purgeable state {code}"
            ))),
        }
    }

    fn is_reclaimable(self) -> bool {
        self != Self::NonVolatile
    }
}

/// Bytes held by live arrays whose banks are currently volatile or empty.
pub(crate) fn reclaimable_purgeable_bytes() -> u64 {
    RECLAIMABLE_PURGEABLE_BYTES.load(Ordering::Acquire)
}

struct BankShared {
    nbytes: usize,
    state: AtomicI32,
    transition: Mutex<()>,
}

impl Drop for BankShared {
    fn drop(&mut self) {
        if self.state.load(Ordering::Acquire) != PurgeableState::NonVolatile as i32 {
            RECLAIMABLE_PURGEABLE_BYTES.fetch_sub(self.nbytes as u64, Ordering::AcqRel);
        }
    }
}

unsafe extern "C" fn release_bank(ctx: *mut c_void) {
    if ctx.is_null() {
        return;
    }
    // SAFETY: ctx is one `Arc<BankShared>` reference leaked by
    // `PurgeableBank::new`; MLX calls this once after releasing its buffer.
    unsafe { Arc::decrement_strong_count(ctx.cast::<BankShared>()) };
}

/// Whether any array still shares a dropped bank's buffer.
pub(crate) struct BankWatch(std::sync::Weak<BankShared>);

impl BankWatch {
    pub(crate) fn alive(&self) -> bool {
        self.0.strong_count() > 0
    }
}

/// One anonymous no-copy Metal allocation and its MLX array.
///
/// Complete GPU reads before making a bank volatile. Before any subsequent CPU
/// or GPU access, transition it to `NonVolatile` and inspect the returned prior
/// state: `Empty` means the contents were discarded and every byte must be
/// refilled. `Volatile` means the contents remain valid.
pub(crate) struct PurgeableBank {
    array: MxArray,
    data: SendPtr<u8>,
    shared: Arc<BankShared>,
}

impl PurgeableBank {
    /// Allocate an uninitialized native Metal buffer and expose its shared
    /// CPU contents directly in the non-volatile state.
    pub(crate) fn new(shape: &[i64], dtype: DType) -> Result<Self> {
        let mut numel = 1usize;
        for (axis, &dim) in shape.iter().enumerate() {
            let dim = usize::try_from(dim).map_err(|_| {
                Error::from_reason(format!(
                    "purgeable bank: negative dimension {dim} at axis {axis}"
                ))
            })?;
            if dim == 0 {
                return Err(Error::from_reason(
                    "purgeable bank: an empty array cannot be allocated",
                ));
            }
            numel = numel.checked_mul(dim).ok_or_else(|| {
                Error::from_reason(format!(
                    "purgeable bank: shape {shape:?} overflows usize at axis {axis}"
                ))
            })?;
        }
        let nbytes = numel
            .checked_mul(dtype.byte_size())
            .ok_or_else(|| Error::from_reason("purgeable bank: byte size overflows usize"))?;
        let shared = Arc::new(BankShared {
            nbytes,
            state: AtomicI32::new(PurgeableState::NonVolatile as i32),
            transition: Mutex::new(()),
        });
        let ctx = Arc::into_raw(Arc::clone(&shared));
        let mut data = std::ptr::null_mut();
        // SAFETY: `shape` and `data` are valid for the call; one Arc reference
        // stays in the MLX deleter until the last array sharing the buffer dies.
        let handle = unsafe {
            sys::mlx_array_alloc_purgeable(
                shape.as_ptr(),
                shape.len(),
                dtype.code(),
                &mut data,
                Some(release_bank),
                ctx.cast_mut().cast::<c_void>(),
            )
        };
        if handle.is_null() || data.is_null() {
            // SAFETY: the bridge did not adopt this reference on failure.
            unsafe { Arc::decrement_strong_count(ctx) };
            return Err(Error::from_reason(format!(
                "purgeable bank: Metal allocation of {nbytes} bytes failed"
            )));
        }
        let bank = Self {
            array: MxArray::from_handle(handle, "purgeable_bank")?,
            data: SendPtr(data.cast::<u8>()),
            shared,
        };
        bank.set_purgeable(PurgeableState::NonVolatile)?;
        Ok(bank)
    }

    pub(crate) fn array(&self) -> &MxArray {
        &self.array
    }

    /// CPU storage for direct reads or repack output. The caller keeps the
    /// bank non-volatile and excludes GPU access while writing through it.
    pub(crate) fn data_ptr(&self) -> SendPtr<u8> {
        self.data
    }

    pub(crate) fn nbytes(&self) -> usize {
        self.shared.nbytes
    }

    /// Bytes of the bank's pages resident in RAM now (`mincore`; every byte
    /// where `mincore` is not available).
    pub(crate) fn resident_bytes(&self) -> Result<u64> {
        #[cfg(target_os = "macos")]
        {
            let page = super::page_size();
            let mut status = vec![0u8; self.shared.nbytes.div_ceil(page)];
            // SAFETY: the pointer is the page-aligned start of the live bank
            // and the length its size; mincore writes one byte per page.
            let rc = unsafe {
                libc::mincore(
                    self.data.0.cast::<c_void>(),
                    self.shared.nbytes,
                    status.as_mut_ptr().cast(),
                )
            };
            if rc != 0 {
                return Err(Error::from_reason(format!(
                    "purgeable bank: mincore failed: {}",
                    std::io::Error::last_os_error()
                )));
            }
            let pages = status.iter().filter(|&&v| v & 1 != 0).count();
            Ok((pages * page).min(self.shared.nbytes) as u64)
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(self.shared.nbytes as u64)
        }
    }

    /// Outlives the bank: alive until the last array on its buffer dies.
    pub(crate) fn watch(&self) -> BankWatch {
        BankWatch(Arc::downgrade(&self.shared))
    }

    /// Synchronously change Metal's purgeability and return the previous (or,
    /// after an asynchronous reclaim, empty) state.
    pub(crate) fn set_purgeable(&self, state: PurgeableState) -> Result<PurgeableState> {
        let _transition = self
            .shared
            .transition
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tracked = PurgeableState::from_code(self.shared.state.load(Ordering::Acquire))?;
        // SAFETY: this is the live available array returned by the no-copy
        // allocation above; the caller has drained GPU work before reclaiming.
        let previous_code =
            unsafe { sys::mlx_array_set_purgeable_state(self.array.handle.0, state as i32) };
        if previous_code < 0 {
            return Err(Error::from_reason(format!(
                "purgeable bank: failed to set purgeable state {state:?}"
            )));
        }
        let previous = if previous_code == 1 {
            tracked
        } else {
            PurgeableState::from_code(previous_code)?
        };
        if tracked.is_reclaimable() != state.is_reclaimable() {
            if state.is_reclaimable() {
                RECLAIMABLE_PURGEABLE_BYTES.fetch_add(self.shared.nbytes as u64, Ordering::AcqRel);
            } else {
                RECLAIMABLE_PURGEABLE_BYTES.fetch_sub(self.shared.nbytes as u64, Ordering::AcqRel);
            }
        }
        self.shared.state.store(state as i32, Ordering::Release);
        Ok(previous)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_os = "macos")]
    use crate::array::memory::{anonymous_active_memory, get_active_memory};
    #[cfg(target_os = "macos")]
    use std::io::Write;
    #[cfg(target_os = "macos")]
    use std::sync::Mutex as TestMutex;

    const BANK_BYTES: usize = 32 << 10;
    #[cfg(target_os = "macos")]
    static SERIAL: TestMutex<()> = TestMutex::new(());

    #[cfg(target_os = "macos")]
    fn require_metal() {
        assert!(
            unsafe { sys::mlx_metal_is_available() },
            "purgeable-bank Metal tests require a Metal device on macOS"
        );
    }

    #[cfg(target_os = "macos")]
    struct TempFile(std::path::PathBuf);

    #[cfg(target_os = "macos")]
    impl TempFile {
        fn new(name: &str) -> (Self, Vec<u8>) {
            let path = std::env::temp_dir()
                .join(format!("mlx-purgeable-{}-{name}.bin", std::process::id()));
            let source: Vec<u8> = (0..BANK_BYTES)
                .map(|i| {
                    let half = if i < BANK_BYTES / 2 { 0x31 } else { 0xa7 };
                    (i as u8).wrapping_mul(29) ^ half
                })
                .collect();
            let mut file = std::fs::File::create(&path).unwrap();
            file.write_all(&source).unwrap();
            file.sync_all().unwrap();
            (Self(path), source)
        }
    }

    #[cfg(target_os = "macos")]
    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[cfg(target_os = "macos")]
    fn read_two_halves(bank: &PurgeableBank, file: &std::fs::File) {
        use std::os::unix::fs::FileExt;
        // SAFETY: the bank is non-volatile, uniquely owned by this test, and
        // its 32 KiB allocation is alive until both positional reads return.
        let dst = unsafe { std::slice::from_raw_parts_mut(bank.data_ptr().0, BANK_BYTES) };
        let (first, second) = dst.split_at_mut(BANK_BYTES / 2);
        file.read_exact_at(first, 0).unwrap();
        file.read_exact_at(second, (BANK_BYTES / 2) as u64).unwrap();
    }

    #[cfg(target_os = "macos")]
    unsafe fn bytes(bank: &PurgeableBank) -> &[u8] {
        // SAFETY: callers keep the bank alive and non-volatile for the read.
        unsafe { std::slice::from_raw_parts(bank.data_ptr().0, bank.nbytes()) }
    }

    #[cfg(target_os = "macos")]
    fn resident_pages(bank: &PurgeableBank) -> Vec<u8> {
        let page = super::super::page_size();
        let mut resident = vec![0u8; bank.nbytes().div_ceil(page)];
        // SAFETY: the pointer is page-aligned and the range is inside the live
        // bank; mincore writes one status byte per page.
        let rc = unsafe {
            libc::mincore(
                bank.data_ptr().0.cast::<c_void>(),
                bank.nbytes(),
                resident.as_mut_ptr().cast(),
            )
        };
        assert_eq!(rc, 0, "mincore failed: {}", std::io::Error::last_os_error());
        resident
    }

    #[cfg(target_os = "macos")]
    fn filled_bank(temp: &TempFile) -> PurgeableBank {
        let bank = PurgeableBank::new(&[BANK_BYTES as i64], DType::Uint8).unwrap();
        assert_eq!((bank.data_ptr().0 as usize) % super::super::page_size(), 0);
        read_two_halves(&bank, &std::fs::File::open(&temp.0).unwrap());
        bank
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn two_aligned_reads_share_one_no_copy_bank() {
        require_metal();
        let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (temp, source) = TempFile::new("aligned");
        let active_before = get_active_memory();
        let reclaimable_before = reclaimable_purgeable_bytes();
        let bank = filled_bank(&temp);
        assert_eq!(bank.nbytes(), BANK_BYTES);
        assert_eq!(
            bank.array().storage_ptr(),
            bank.data_ptr().0.cast::<c_void>()
        );
        assert_eq!(unsafe { bytes(&bank) }, source);
        assert_eq!(bank.array().to_uint8().unwrap(), source);
        assert_eq!(get_active_memory() - active_before, BANK_BYTES as f64);
        assert_eq!(reclaimable_purgeable_bytes(), reclaimable_before);
        drop(bank);
        assert_eq!(get_active_memory(), active_before);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn purgeable_state_round_trip_preserves_unpurged_bytes() {
        require_metal();
        let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (temp, source) = TempFile::new("round-trip");
        let bank = filled_bank(&temp);
        let reclaimable_before = reclaimable_purgeable_bytes();
        let active_before = anonymous_active_memory();

        assert_eq!(
            bank.set_purgeable(PurgeableState::Volatile).unwrap(),
            PurgeableState::NonVolatile
        );
        assert_eq!(
            reclaimable_purgeable_bytes(),
            reclaimable_before + BANK_BYTES as u64
        );
        assert_eq!(active_before - anonymous_active_memory(), BANK_BYTES as f64);
        assert_eq!(
            bank.set_purgeable(PurgeableState::NonVolatile).unwrap(),
            PurgeableState::Volatile
        );
        assert_eq!(reclaimable_purgeable_bytes(), reclaimable_before);
        assert_eq!(anonymous_active_memory(), active_before);
        assert_eq!(unsafe { bytes(&bank) }, source);
        let sum = bank
            .array()
            .astype(DType::Float32)
            .unwrap()
            .sum(None, None)
            .unwrap();
        let expected = source.iter().map(|&v| f32::from(v)).sum::<f32>();
        assert_eq!(sum.to_float32().unwrap()[0], expected);

        assert_eq!(
            bank.set_purgeable(PurgeableState::Volatile).unwrap(),
            PurgeableState::NonVolatile
        );
        assert_eq!(
            bank.set_purgeable(PurgeableState::Empty).unwrap(),
            PurgeableState::Volatile
        );
        assert_eq!(
            bank.set_purgeable(PurgeableState::NonVolatile).unwrap(),
            PurgeableState::Empty
        );
        read_two_halves(&bank, &std::fs::File::open(&temp.0).unwrap());
        assert_eq!(unsafe { bytes(&bank) }, source);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn empty_state_evicts_pages_and_valid_reads_refill_them() {
        require_metal();
        let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (temp, source) = TempFile::new("empty");
        let bank = filled_bank(&temp);
        assert!(resident_pages(&bank).iter().all(|&v| v & 1 != 0));
        bank.set_purgeable(PurgeableState::Volatile).unwrap();
        bank.set_purgeable(PurgeableState::Empty).unwrap();
        let started = std::time::Instant::now();
        while resident_pages(&bank).iter().any(|&v| v & 1 != 0) {
            assert!(
                started.elapsed() < std::time::Duration::from_secs(3),
                "empty bank stayed resident: {:?}",
                resident_pages(&bank)
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(
            bank.set_purgeable(PurgeableState::NonVolatile).unwrap(),
            PurgeableState::Empty
        );
        read_two_halves(&bank, &std::fs::File::open(&temp.0).unwrap());
        assert_eq!(unsafe { bytes(&bank) }, source);
        assert!(resident_pages(&bank).iter().all(|&v| v & 1 != 0));
    }

    #[cfg(target_os = "macos")]
    struct Pressure {
        ptr: *mut c_void,
        len: usize,
    }

    #[cfg(target_os = "macos")]
    impl Pressure {
        fn new(gib: usize) -> Self {
            let len = gib << 30;
            // SAFETY: anonymous private mapping with no file offset.
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANON,
                    -1,
                    0,
                )
            };
            assert_ne!(ptr, libc::MAP_FAILED, "pressure mmap failed");
            Self { ptr, len }
        }

        fn touch_gib(&self, chunk: usize) {
            let begin = (chunk - 1) << 30;
            let end = chunk << 30;
            let page = super::super::page_size();
            for offset in (begin..end).step_by(page) {
                // SAFETY: offset is inside the live mapping; one byte per page
                // makes the whole touched chunk physically resident.
                unsafe { self.ptr.cast::<u8>().add(offset).write(0x6b) };
            }
        }
    }

    #[cfg(target_os = "macos")]
    impl Drop for Pressure {
        fn drop(&mut self) {
            // SAFETY: this is the intact mapping returned by `Pressure::new`.
            unsafe { libc::munmap(self.ptr, self.len) };
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "touches a 64 GiB bank and up to 40 GiB of pressure; run alone"]
    fn volatile_pressure_probe_refills_only_when_empty() {
        require_metal();
        let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        const LARGE_BYTES: usize = 64 << 30;
        let (temp, source) = TempFile::new("pressure");
        let bank = PurgeableBank::new(&[64, 1 << 30], DType::Uint8).unwrap();
        let page = super::super::page_size();
        for offset in (0..LARGE_BYTES).step_by(page) {
            // SAFETY: every offset addresses the last byte of one bank page.
            unsafe { bank.data_ptr().0.add(offset + page - 1).write(0x5a) };
        }
        read_two_halves(&bank, &std::fs::File::open(&temp.0).unwrap());
        let initial_resident = resident_pages(&bank)
            .iter()
            .filter(|&&v| v & 1 != 0)
            .count();
        assert!(initial_resident > 0);
        bank.set_purgeable(PurgeableState::Volatile).unwrap();
        let pressure = Pressure::new(40);
        let started = std::time::Instant::now();
        let mut touched_gib = 0;
        for chunk in 1..=40 {
            pressure.touch_gib(chunk);
            touched_gib = chunk;
            std::thread::sleep(std::time::Duration::from_millis(100));
            if resident_pages(&bank).iter().all(|&v| v & 1 == 0) {
                break;
            }
        }
        let elapsed = started.elapsed();
        let final_resident = resident_pages(&bank);
        let resident_count = final_resident.iter().filter(|&&v| v & 1 != 0).count();
        drop(pressure);
        let previous = bank.set_purgeable(PurgeableState::NonVolatile).unwrap();
        match previous {
            PurgeableState::Empty => {
                assert!(
                    final_resident.iter().all(|&v| v & 1 == 0),
                    "Metal reported Empty while mincore still reports resident pages"
                );
                read_two_halves(&bank, &std::fs::File::open(&temp.0).unwrap());
            }
            PurgeableState::Volatile => {
                assert_eq!(&unsafe { bytes(&bank) }[..BANK_BYTES], source);
            }
            PurgeableState::NonVolatile => {
                panic!("volatile transition was not adopted before pressure")
            }
        }
        eprintln!(
            "volatile pressure probe: initial_pages={initial_resident}, final_pages={resident_count}, previous={previous:?}, hog_gib={touched_gib}, elapsed={elapsed:?}"
        );
        assert_eq!(&unsafe { bytes(&bank) }[..BANK_BYTES], source);
        drop(bank);
        crate::array::memory::synchronize_and_clear_cache();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn reclaimable_accounting_lives_until_the_last_array_drops() {
        require_metal();
        let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (temp, _) = TempFile::new("lease");
        let bank = filled_bank(&temp);
        let before = reclaimable_purgeable_bytes();
        bank.set_purgeable(PurgeableState::Volatile).unwrap();
        let view = bank.array().clone();
        drop(bank);
        assert_eq!(reclaimable_purgeable_bytes(), before + BANK_BYTES as u64);
        drop(view);
        assert_eq!(reclaimable_purgeable_bytes(), before);
    }

    #[test]
    fn invalid_shapes_are_rejected_before_allocation() {
        let before = reclaimable_purgeable_bytes();
        assert!(PurgeableBank::new(&[0], DType::Uint8).is_err());
        assert!(PurgeableBank::new(&[-1], DType::Uint8).is_err());
        assert!(PurgeableBank::new(&[i64::MAX, 2], DType::Uint8).is_err());
        assert_eq!(reclaimable_purgeable_bytes(), before);
    }

    #[test]
    fn creation_matches_backend_availability_without_a_cpu_copy() {
        let before = reclaimable_purgeable_bytes();
        if unsafe { sys::mlx_metal_is_available() } {
            let bank = PurgeableBank::new(&[BANK_BYTES as i64], DType::Uint8).unwrap();
            assert!(!bank.data_ptr().0.is_null());
        } else {
            assert!(PurgeableBank::new(&[BANK_BYTES as i64], DType::Uint8).is_err());
        }
        assert_eq!(reclaimable_purgeable_bytes(), before);
    }
}
