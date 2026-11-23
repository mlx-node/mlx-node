//! Windows Win32 `GlobalMemoryStatusEx` host memory snapshot.

use super::HostMemory;
use napi::{Error, Result};
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

pub(super) fn host_memory() -> Result<HostMemory> {
    // SAFETY: `MEMORYSTATUSEX` is a plain Win32 POD struct; zeroed with
    // `dwLength` initialized before calling `GlobalMemoryStatusEx`.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    if ok == 0 || status.ullTotalPhys == 0 {
        return Err(Error::from_reason(
            "Cannot establish Qwen4 larger-cache memory headroom",
        ));
    }
    let physical = status.ullTotalPhys;
    // Windows `ullAvailPhys` includes free and standby (reclaimable) pages,
    // matching macOS's `free + file`.
    let free = status.ullAvailPhys.min(physical);
    let non_reclaimable = physical.saturating_sub(free);
    Ok(HostMemory {
        physical,
        free,
        file: 0,
        anonymous: non_reclaimable,
        purgeable: 0,
        wired: 0,
        compressor: 0,
    })
}
