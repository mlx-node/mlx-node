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
}
