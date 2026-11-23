#[cfg(target_os = "macos")]
pub(crate) fn physical_memory() -> Option<u64> {
    let mut bytes = 0u64;
    let mut size = std::mem::size_of_val(&bytes);
    // SAFETY: `hw.memsize` is a u64 and `size` holds its byte length.
    let rc = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&mut bytes as *mut u64).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && bytes > 0).then_some(bytes)
}

#[cfg(target_os = "linux")]
pub(crate) fn physical_memory() -> Option<u64> {
    // SAFETY: sysconf only reads system configuration.
    let (pages, page) = unsafe {
        (
            libc::sysconf(libc::_SC_PHYS_PAGES),
            libc::sysconf(libc::_SC_PAGESIZE),
        )
    };
    if pages <= 0 || page <= 0 {
        return None;
    }
    (pages as u64).checked_mul(page as u64)
}

#[cfg(target_os = "windows")]
pub(crate) fn physical_memory() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status = MEMORYSTATUSEX {
        dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
        ..unsafe { std::mem::zeroed() }
    };
    // SAFETY: `dwLength` is initialized to `size_of::<MEMORYSTATUSEX>()`.
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    (ok != 0 && status.ullTotalPhys > 0).then_some(status.ullTotalPhys)
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
pub(crate) fn physical_memory() -> Option<u64> {
    None
}

/// One host memory snapshot, in bytes (family-neutral; callers pass the
/// value on, so tests build it by hand).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostMemory {
    pub physical: u64,
    pub free: u64,
    /// Pageable file-backed pages the kernel can drop on demand.
    pub file: u64,
    /// Pageable anonymous pages of every process.
    pub anonymous: u64,
    pub purgeable: u64,
    pub wired: u64,
    /// Physical pages the compressor pool occupies.
    pub compressor: u64,
}

impl HostMemory {
    /// Free plus clean file cache: what the kernel hands out without
    /// touching another process's data.
    pub fn available(&self) -> u64 {
        self.free.saturating_add(self.file)
    }

    /// `host_used` of the guard: wired + compressor + anonymous - purgeable.
    pub fn non_reclaimable(&self) -> u64 {
        self.wired
            .saturating_add(self.compressor)
            .saturating_add(self.anonymous.saturating_sub(self.purgeable))
    }
}

/// Raw Mach VM page counts and page size from one `host_statistics64` read.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MacosVmStats {
    pub page_size: u64,
    pub free: u64,
    pub speculative: u64,
    pub file_backed: u64,
    pub purgeable: u64,
    pub anonymous: u64,
    pub wired: u64,
    pub compressor: u64,
    pub compressed: u64,
}

/// Sample raw Mach VM page counts and host page size via `host_statistics64`.
#[cfg(target_os = "macos")]
pub(crate) fn sample_macos_vm_stats() -> Option<MacosVmStats> {
    unsafe extern "C" {
        fn mach_host_self() -> libc::mach_port_t;
        static mach_task_self_: libc::mach_port_t;
        fn mach_port_deallocate(
            task: libc::mach_port_t,
            name: libc::mach_port_t,
        ) -> libc::kern_return_t;
        fn host_page_size(
            host: libc::mach_port_t,
            size: *mut libc::vm_size_t,
        ) -> libc::kern_return_t;
    }
    let mut stats = std::mem::MaybeUninit::<libc::vm_statistics64>::zeroed();
    let mut count = libc::HOST_VM_INFO64_COUNT;
    let mut page_size: libc::vm_size_t = 0;
    // SAFETY: plain Mach calls on this task's host port; `stats` holds
    // `count` integers and the port is released after the read.
    let (ok, stats) = unsafe {
        let host = mach_host_self();
        let page_ok = host_page_size(host, &mut page_size);
        let ok = if page_ok == 0 {
            libc::host_statistics64(
                host,
                libc::HOST_VM_INFO64,
                stats.as_mut_ptr().cast(),
                &mut count,
            )
        } else {
            page_ok
        };
        mach_port_deallocate(mach_task_self_, host);
        (ok, stats.assume_init())
    };
    if ok != 0 || page_size == 0 {
        return None;
    }
    Some(MacosVmStats {
        page_size: page_size as u64,
        free: u64::from(stats.free_count),
        speculative: u64::from(stats.speculative_count),
        file_backed: u64::from(stats.external_page_count),
        purgeable: u64::from(stats.purgeable_count),
        anonymous: u64::from(stats.internal_page_count),
        wired: u64::from(stats.wire_count),
        compressor: u64::from(stats.compressor_page_count),
        compressed: stats.total_uncompressed_pages_in_compressor,
    })
}

#[cfg(target_os = "macos")]
pub fn host_memory() -> Option<HostMemory> {
    let stats = sample_macos_vm_stats()?;
    let page = stats.page_size;
    Some(HostMemory {
        physical: physical_memory()?,
        free: stats.free * page,
        file: stats.file_backed * page,
        anonymous: stats.anonymous * page,
        purgeable: stats.purgeable * page,
        wired: stats.wired * page,
        compressor: stats.compressor * page,
    })
}

#[cfg(target_os = "windows")]
pub fn host_memory() -> Option<HostMemory> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: `MEMORYSTATUSEX` is a plain Win32 POD struct; zeroed with
    // `dwLength` initialized before calling `GlobalMemoryStatusEx`.
    let mut status = MEMORYSTATUSEX {
        dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
        ..unsafe { std::mem::zeroed() }
    };
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    if ok == 0 || status.ullTotalPhys == 0 {
        return None;
    }
    let physical = status.ullTotalPhys;
    // Windows `ullAvailPhys` includes free and standby (reclaimable) pages,
    // matching macOS's `free + file`.
    let free = status.ullAvailPhys.min(physical);
    let non_reclaimable = physical.saturating_sub(free);
    Some(HostMemory {
        physical,
        free,
        file: 0,
        anonymous: non_reclaimable,
        purgeable: 0,
        wired: 0,
        compressor: 0,
    })
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn host_memory() -> Option<HostMemory> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_memory_is_known_on_supported_hosts() {
        let bytes = physical_memory();
        if cfg!(any(
            target_os = "macos",
            target_os = "linux",
            target_os = "windows"
        )) {
            assert!(bytes.is_some_and(|b| b >= 1 << 30), "{bytes:?}");
        }
    }

    #[test]
    fn host_memory_fields_sane() {
        let Some(h) = host_memory() else {
            if cfg!(any(target_os = "macos", target_os = "windows")) {
                panic!("host_memory failed on supported host");
            }
            return;
        };
        println!("{h:?} non_reclaimable {}", h.non_reclaimable());
        assert_eq!(Some(h.physical), physical_memory());
        assert!(
            (!cfg!(target_os = "macos") || h.wired > 0) && h.anonymous > 0 && h.available() > 0,
            "{h:?}"
        );
        let parts = h.free + h.file + h.anonymous + h.wired + h.compressor;
        assert!(parts <= h.physical + (h.physical >> 3), "{h:?}");
        assert!(h.non_reclaimable() < h.physical, "{h:?}");
    }

    #[test]
    fn non_reclaimable_formula() {
        let h = HostMemory {
            physical: 128 << 30,
            free: 1,
            file: 2,
            anonymous: 40 << 30,
            purgeable: 3 << 30,
            wired: 7 << 30,
            compressor: 5 << 30,
        };
        assert_eq!(h.available(), 3);
        assert_eq!(h.non_reclaimable(), (7 + 5 + 40 - 3) << 30);
        let under = HostMemory {
            purgeable: 50 << 30,
            ..h
        };
        assert_eq!(under.non_reclaimable(), 12 << 30);
    }
}
