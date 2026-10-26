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

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub(crate) fn physical_memory() -> Option<u64> {
    None
}

/// One `host_statistics64` read, in bytes (family-neutral; callers pass the
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
    /// `host_used` of the guard: wired + compressor + anonymous - purgeable.
    pub fn non_reclaimable(&self) -> u64 {
        self.wired
            .saturating_add(self.compressor)
            .saturating_add(self.anonymous.saturating_sub(self.purgeable))
    }
}

#[cfg(target_os = "macos")]
pub fn host_memory() -> Option<HostMemory> {
    unsafe extern "C" {
        fn mach_host_self() -> libc::mach_port_t;
        static mach_task_self_: libc::mach_port_t;
        fn mach_port_deallocate(
            task: libc::mach_port_t,
            name: libc::mach_port_t,
        ) -> libc::kern_return_t;
    }
    let mut stats = std::mem::MaybeUninit::<libc::vm_statistics64>::zeroed();
    let mut count = libc::HOST_VM_INFO64_COUNT;
    // SAFETY: plain Mach calls on this task's host port; `stats` holds
    // `count` integers and the port is released after the read.
    let (ok, stats) = unsafe {
        let host = mach_host_self();
        let ok = libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            stats.as_mut_ptr().cast(),
            &mut count,
        );
        mach_port_deallocate(mach_task_self_, host);
        (ok, stats.assume_init())
    };
    if ok != 0 {
        return None;
    }
    let page = crate::array::page_size() as u64;
    Some(HostMemory {
        physical: physical_memory()?,
        free: u64::from(stats.free_count) * page,
        file: u64::from(stats.external_page_count) * page,
        anonymous: u64::from(stats.internal_page_count) * page,
        purgeable: u64::from(stats.purgeable_count) * page,
        wired: u64::from(stats.wire_count) * page,
        compressor: u64::from(stats.compressor_page_count) * page,
    })
}

#[cfg(not(target_os = "macos"))]
pub fn host_memory() -> Option<HostMemory> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_memory_is_known_on_supported_hosts() {
        let bytes = physical_memory();
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            assert!(bytes.is_some_and(|b| b >= 1 << 30), "{bytes:?}");
        }
    }

    #[test]
    fn host_memory_fields_sane() {
        let Some(h) = host_memory() else {
            if cfg!(target_os = "macos") {
                panic!("host_statistics64 failed");
            }
            return;
        };
        println!("{h:?} non_reclaimable {}", h.non_reclaimable());
        assert_eq!(Some(h.physical), physical_memory());
        assert!(
            h.wired > 0 && h.anonymous > 0 && h.free + h.file > 0,
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
        assert_eq!(h.non_reclaimable(), (7 + 5 + 40 - 3) << 30);
        let under = HostMemory {
            purgeable: 50 << 30,
            ..h
        };
        assert_eq!(under.non_reclaimable(), 12 << 30);
    }
}
