//! macOS Mach `host_statistics64` host memory snapshot.

use super::HostMemory;
use napi::{Error, Result};

pub(super) fn host_memory() -> Result<HostMemory> {
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
    let host = unsafe { mach_host_self() };
    let ok = unsafe {
        libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            stats.as_mut_ptr().cast(),
            &mut count,
        )
    };
    unsafe {
        mach_port_deallocate(mach_task_self_, host);
    }
    let page = crate::array::page_size() as u64;
    if ok != 0 {
        return Err(Error::from_reason(
            "Cannot establish Qwen4 larger-cache memory headroom",
        ));
    }
    let stats = unsafe { stats.assume_init() };
    Ok(HostMemory {
        physical: super::physical_memory()?,
        free: u64::from(stats.free_count) * page,
        file: u64::from(stats.external_page_count) * page,
        anonymous: u64::from(stats.internal_page_count) * page,
        purgeable: u64::from(stats.purgeable_count) * page,
        wired: u64::from(stats.wire_count) * page,
        compressor: u64::from(stats.compressor_page_count) * page,
    })
}
