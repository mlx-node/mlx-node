//! The host's memory and macOS's pressure, after the host part of Splash
//! `MemoryGovernor`: `estimateHostMemory`, `queryHostMemory`,
//! `querySystemMemoryPressure` and `HostRule`.

use super::ledger::ClaimClass;

/// Splash `kHostMarginBytes`: the one host margin M.
pub(crate) const HOST_MARGIN_BYTES: u64 = 1 << 30;
/// Splash `kHostSettleMilliseconds`: host passes are this far apart.
pub(crate) const HOST_SETTLE_SECONDS: f64 = 1.0;

const GIB: u64 = 1 << 30;

/// Mach's page counts (Splash `HostMemoryPages`). `free` includes the
/// speculative pages; `compressor` is the compressor's own pages and
/// `compressed` the pages it holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct HostPages {
    pub free: u64,
    pub speculative: u64,
    pub file_backed: u64,
    pub purgeable: u64,
    pub anonymous: u64,
    pub compressor: u64,
    pub compressed: u64,
}

/// What macOS can hand out without swapping (Splash `HostMemory`):
/// `available` for memory a request needs, `spare` for memory none needs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct HostMemory {
    pub available: u64,
    pub spare: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Pressure {
    Normal,
    Warning,
    Critical,
}

impl Pressure {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Warning => "warning",
            Self::Critical => "critical",
        }
    }
}

/// Splash `compressionSavings`: what compressing the anonymous pages frees
/// at the compressor's ratio, at most half of them.
pub(crate) fn compression_savings(pages: &HostPages) -> u64 {
    if pages.compressor == 0 || pages.compressed / 2 >= pages.compressor {
        return pages.anonymous / 2;
    }
    if pages.compressed <= pages.compressor {
        return 0;
    }
    let kept =
        u128::from(pages.anonymous) * u128::from(pages.compressor) / u128::from(pages.compressed);
    pages.anonymous - kept as u64
}

/// Splash `estimateHostMemory`; zeros when the counts do not add up.
pub(crate) fn estimate_host_memory(
    pages: &HostPages,
    page_size: u64,
    compression: bool,
) -> HostMemory {
    if page_size == 0 || pages.speculative > pages.free {
        return HostMemory::default();
    }
    let Some(base) = (pages.free - pages.speculative)
        .checked_add(pages.file_backed)
        .and_then(|sum| sum.checked_add(pages.purgeable))
    else {
        return HostMemory::default();
    };
    let credit = if compression {
        compression_savings(pages)
    } else {
        0
    };
    let Some(available) = base
        .checked_add(credit)
        .and_then(|sum| sum.checked_mul(page_size))
    else {
        return HostMemory::default();
    };
    HostMemory {
        available,
        spare: base.saturating_add(credit / 2).saturating_mul(page_size),
    }
}

/// Splash `queryHostMemory`'s rule: no compression credit under critical
/// pressure or when the pressure cannot be read.
pub(crate) fn host_memory_at(
    pages: &HostPages,
    page_size: u64,
    pressure: Option<Pressure>,
) -> HostMemory {
    estimate_host_memory(
        pages,
        page_size,
        pressure.is_some_and(|level| level != Pressure::Critical),
    )
}

/// Splash `EngineMemoryPolicy::hostAvailableReserveBytes`: what macOS
/// keeps; a start needs more than this available.
pub(crate) fn available_reserve_bytes(physical: u64) -> u64 {
    (physical / 10).min(2 * GIB)
}

/// Splash `EngineMemoryPolicy::hostSpareReserveBytes`: what elective work
/// leaves other applications, 7% of the Mac's memory from 2 to 7 GiB.
pub(crate) fn spare_reserve_bytes(physical: u64) -> u64 {
    (physical / 100 * 7).clamp(2 * GIB, 7 * GIB)
}

/// The host's spare memory above its spare reserve, beside `reserved`
/// being allocated (Splash `spareHeadroom`).
pub(crate) fn spare_headroom(host: &HostMemory, spare_reserve: u64, reserved: u64) -> u64 {
    host.spare
        .saturating_sub(spare_reserve)
        .saturating_sub(reserved)
}

/// Splash `querySystemMemoryPressure`: sysctl
/// `kern.memorystatus_vm_pressure_level` (1 normal, 2 warning, 4 critical).
#[cfg(target_os = "macos")]
pub(crate) fn query_pressure() -> Option<Pressure> {
    let mut level = 0u32;
    let mut size = std::mem::size_of_val(&level);
    // SAFETY: the level is a u32 and `size` holds its byte length.
    let rc = unsafe {
        libc::sysctlbyname(
            c"kern.memorystatus_vm_pressure_level".as_ptr(),
            (&mut level as *mut u32).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 || size != std::mem::size_of_val(&level) {
        return None;
    }
    match level {
        1 => Some(Pressure::Normal),
        2 => Some(Pressure::Warning),
        4 => Some(Pressure::Critical),
        _ => None,
    }
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn query_pressure() -> Option<Pressure> {
    None
}

/// Splash `sampleHostPages`: the host's page counts and page size.
#[cfg(target_os = "macos")]
fn sample_pages() -> Option<(HostPages, u64)> {
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
    Some((
        HostPages {
            free: u64::from(stats.free_count),
            speculative: u64::from(stats.speculative_count),
            file_backed: u64::from(stats.external_page_count),
            purgeable: u64::from(stats.purgeable_count),
            anonymous: u64::from(stats.internal_page_count),
            compressor: u64::from(stats.compressor_page_count),
            compressed: stats.total_uncompressed_pages_in_compressor,
        },
        page_size as u64,
    ))
}

#[cfg(not(target_os = "macos"))]
fn sample_pages() -> Option<(HostPages, u64)> {
    None
}

/// Splash `queryHostMemory`, at the pressure read beside it.
pub(crate) fn query_host_memory(pressure: Option<Pressure>) -> Option<HostMemory> {
    let (pages, page_size) = sample_pages()?;
    Some(host_memory_at(&pages, page_size, pressure))
}

/// Splash `MemoryGovernor::HostRule`: the host's one rule at its latest
/// sample. Headroom h is the spare memory above the spare reserve, less
/// what is being allocated. The band begins when h falls below the margin
/// M or cannot be measured, and ends once h clears M by `reach`, or once a
/// host pass finds nothing to take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HostRule {
    measured: bool,
    headroom: u64,
    system: Pressure,
    constrained: bool,
}

impl Default for HostRule {
    fn default() -> Self {
        Self {
            measured: false,
            headroom: 0,
            system: Pressure::Normal,
            constrained: true,
        }
    }
}

impl HostRule {
    /// `hysteresis`: the largest unit a pool grows by (0: mlx-node has no
    /// growth pools); `reclaim_short`: the band's waiver.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sample(
        &mut self,
        host: Option<&HostMemory>,
        spare_reserve: u64,
        in_flight: u64,
        system: Pressure,
        hysteresis: u64,
        refused_waiting: u64,
        reclaim_short: bool,
    ) {
        self.measured = host.is_some();
        self.headroom = host.map_or(0, |host| spare_headroom(host, spare_reserve, in_flight));
        self.system = system;
        let reach = if refused_waiting > 0 {
            hysteresis.min(refused_waiting)
        } else {
            hysteresis
        };
        if !self.measured || self.headroom < HOST_MARGIN_BYTES {
            self.constrained = true;
        } else if reclaim_short || self.headroom - HOST_MARGIN_BYTES >= reach {
            self.constrained = false;
        }
    }

    /// Splash `HostRule::admits`.
    pub(crate) fn admits(&self, cls: ClaimClass, bytes: u64, exempt: bool) -> bool {
        if self.system == Pressure::Critical {
            return false;
        }
        if cls == ClaimClass::Floor
            || ((cls == ClaimClass::InService || cls == ClaimClass::Growth) && exempt)
        {
            return true;
        }
        let keeps_margin =
            self.measured && self.headroom >= bytes && self.headroom - bytes >= HOST_MARGIN_BYTES;
        if cls == ClaimClass::InService {
            return keeps_margin && self.system == Pressure::Normal;
        }
        keeps_margin && self.system == Pressure::Normal && !self.constrained
    }

    /// Splash `HostRule::deficitBytes`: what the host lacks of M and of
    /// what the claims it refused wait for, beyond what is landing; nothing
    /// measured, nothing lacking.
    pub(crate) fn deficit_bytes(&self, refused_waiting: u64, returning: u64) -> u64 {
        if !self.measured {
            return 0;
        }
        HOST_MARGIN_BYTES
            .saturating_add(refused_waiting)
            .saturating_sub(self.headroom)
            .saturating_sub(returning)
    }

    pub(crate) fn constrained(&self) -> bool {
        self.constrained
    }

    pub(crate) fn level(&self) -> Pressure {
        self.system
    }

    pub(crate) fn measured(&self) -> bool {
        self.measured
    }

    pub(crate) fn headroom(&self) -> u64 {
        self.headroom
    }
}

#[cfg(test)]
mod tests;
