//! Admit hot weights against live system headroom. Preserve room for other
//! applications, recurrent state, page pools and transient work.
use crate::models::qwen4_exp::runtime_flags;
use napi::{Error, Result};

pub(super) const FREE_CACHE_BYTES: u64 = 512 << 20;
pub(super) const WORKING_BYTES: u64 = 4 << 30;

#[derive(Clone, Debug)]
pub(super) struct Plan {
    pub budget: u64,
    pub hot_bytes: u64,
    pub resident: bool,
    pub policy: String,
    pub available_bytes: Option<u64>,
    pub physical_bytes: Option<u64>,
    pub resident_bank_bytes: u64,
    pub resident_expert_layers: usize,
    pub partial_expert_slots: usize,
    /// Routed expert tensors served from the mapped pack (0 = chunk reader).
    pub mapped_pack_files: usize,
    /// Every routed expert is addressed in place through the mapped pack:
    /// no slot banks, no mid-turn staging, no wired-limit context.
    pub mapped: bool,
    /// Bytes of pack sections backing live arrays; MLX counts them in
    /// `active_memory`, so budgets subtract them.
    pub mapped_bytes: u64,
    /// Bytes of mapped banks the host could hold beside everything that
    /// stays anonymous when the residency was chosen (`mapped_gate`).
    pub mapped_bank_budget: u64,
    /// `HostMemory::non_reclaimable` when the residency was chosen.
    pub live_non_reclaimable: u64,
}

/// Bytes of auxiliary (MTP/vision) inventory the mapped budget keeps room for.
const MAPPED_AUXILIARY_CAP: u64 = 8 << 30;
/// LRU rows, paired banks and import staging under the mapped budget.
const MAPPED_LRU_BYTES: u64 = 2 << 30;
/// Room the mapped decision keeps between its need and physical RAM for
/// what other processes allocate during a pass.
pub(super) const MAPPED_MARGIN: u64 = 6 << 30;
/// Free plus file cache a mapped turn must leave before it stops growing
/// anonymous memory. The wired pack sits in neither, so the slot path's
/// reserve (an eighth of RAM) cannot hold during a mapped turn: at 88 GiB
/// available a 6k turn wired 89.5 GiB and left 8.8 GiB, where llama.cpp runs
/// the same model. The load gate and the tripwire own the posture; this
/// floor only refuses at the kernel's critical level.
const MAPPED_TURN_FLOOR: u64 = 2 << 30;

/// One `host_statistics64` read, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct HostMemory {
    pub physical: u64,
    pub free: u64,
    /// Pageable file-backed pages: clean cache the kernel drops on demand
    /// (a resident mapped pack is in here between turns, wired pages are
    /// not).
    pub file: u64,
    /// Pageable anonymous pages of every process, active or not.
    pub anonymous: u64,
    pub purgeable: u64,
    pub wired: u64,
    pub compressor: u64,
    /// Page faults since boot. `host_statistics64` answers only a few
    /// fresh reads per second per host and hands every other caller its
    /// previous answer; a sample equal to the last one in every field is
    /// that copy, not a quiet host (faults never stand still for 250 ms).
    pub faults: u64,
}

impl HostMemory {
    /// Free plus clean file cache: what the kernel hands out without
    /// touching another process's data.
    pub fn available(&self) -> u64 {
        self.free.saturating_add(self.file)
    }

    /// Memory a turn cannot take from the host: wired, the compressor pool
    /// and other processes' anonymous pages (purgeable ones excepted). The
    /// kernel evicts clean file pages before it compresses those, a mapped
    /// pack between two bank bindings included: beside 20-40 GiB of them a
    /// 6k turn re-read 1.06-2.3 TB and decoded at 0.06-0.4 tokens/s.
    pub fn non_reclaimable(&self) -> u64 {
        self.wired
            .saturating_add(self.compressor)
            .saturating_add(self.anonymous.saturating_sub(self.purgeable))
    }
}

/// The residency decision as a byte budget for mapped banks. Today every
/// routed bank maps or none does; a per-layer split can read the same
/// budget.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct MappedGate {
    pub bank_budget: u64,
    pub non_reclaimable: u64,
    pub anonymous_need: u64,
}

/// Mapped banks fit when physical RAM minus the margin, the host's live
/// non-reclaimable memory and what this process keeps anonymous beside the
/// pack (dense banks, bounded auxiliary inventory, LRU room, the KV pool)
/// covers them.
pub(super) fn mapped_gate(host: &HostMemory, anonymous_need: u64, margin: u64) -> MappedGate {
    let non_reclaimable = host.non_reclaimable();
    MappedGate {
        bank_budget: host
            .physical
            .saturating_sub(margin)
            .saturating_sub(non_reclaimable)
            .saturating_sub(anonymous_need),
        non_reclaimable,
        anonymous_need,
    }
}

pub(super) fn plan(hot_bytes: u64) -> Result<Plan> {
    let policy = match std::env::var_os("MLX_QWEN4_RESIDENCY") {
        None => "auto".into(),
        Some(value) => value
            .to_str()
            .map(str::to_owned)
            .filter(|s| ["auto", "stream", "full"].contains(&s.as_str()))
            .ok_or_else(|| {
                Error::from_reason("MLX_QWEN4_RESIDENCY must be auto, stream or full")
            })?,
    };
    let explicit = std::env::var_os("MLX_QWEN4_WEIGHT_CACHE_GIB")
        .map(|value| {
            value
                .to_str()
                .and_then(|s| s.parse::<u64>().ok())
                .filter(|n| *n >= 2)
                .and_then(|g| g.checked_mul(1 << 30))
                .ok_or_else(|| {
                    Error::from_reason("MLX_QWEN4_WEIGHT_CACHE_GIB must be an integer of at least 2 that fits in bytes; live device admission still applies")
                })
        })
        .transpose()?;
    let desired = hot_bytes
        .checked_add((2 << 30) - 1)
        .map(|n| (n >> 30) << 30)
        .ok_or_else(|| Error::from_reason("Qwen4 hot-weight size overflow"))?;
    // Read the live machine at bootstrap. The streaming setting limits desired
    // residency; it must not force an 8 GiB allocation on a busy desktop.
    let snapshot = if cfg!(target_os = "macos") {
        Some((available_memory()?, physical_memory()?))
    } else {
        None
    };
    let budget = if let Some(bytes) = explicit {
        bytes
    } else if let Some((available, physical)) = snapshot {
        let desired = if policy == "stream" {
            desired.min(super::weights::CACHE_BYTES)
        } else {
            desired
        };
        automatic_budget(desired, available, physical)
    } else {
        super::weights::CACHE_BYTES
    };
    if budget == 0 {
        return Err(Error::from_reason(
            "Qwen4 bootstrap has no whole GiB for weights after system/working reserves; release memory before loading",
        ));
    }
    if policy == "full" && desired > budget {
        return Err(Error::from_reason(format!(
            "Qwen4 full hot-path residency needs {:.2} GiB plus system/working headroom; bootstrap admitted {:.2} GiB for weights",
            desired as f64 / (1u64 << 30) as f64,
            budget as f64 / (1u64 << 30) as f64
        )));
    }
    admit(budget)?;
    Ok(Plan {
        budget,
        hot_bytes,
        resident: policy != "stream" && desired <= budget,
        policy,
        available_bytes: snapshot.map(|s| s.0),
        physical_bytes: snapshot.map(|s| s.1),
        resident_bank_bytes: 0,
        resident_expert_layers: 0,
        partial_expert_slots: 0,
        mapped_pack_files: 0,
        mapped: false,
        mapped_bytes: 0,
        mapped_bank_budget: 0,
        live_non_reclaimable: 0,
    })
}

/// `auto` considers the pack when every routed tensor is in it; an explicit
/// weight budget, `stream`, `full` and `MLX_QWEN4_MAPPED_PACK=0` keep the
/// slot path. Whether the banks fit is `mapped_plan`'s live decision.
pub(super) fn mapped_residency_selected(
    policy: &str,
    explicit_budget: bool,
    pack_complete: bool,
    disabled: bool,
) -> bool {
    policy == "auto" && !explicit_budget && !disabled && pack_complete
}

pub(super) fn mapped_pack_disabled() -> bool {
    std::env::var_os("MLX_QWEN4_MAPPED_PACK").is_some_and(|v| v == "0")
}

pub(super) fn explicit_weight_budget() -> bool {
    std::env::var_os("MLX_QWEN4_WEIGHT_CACHE_GIB").is_some()
}

/// Whole GiB the mapped budget asks for: dense banks, bounded auxiliary
/// inventory and LRU room. Routed experts are not in it.
pub(super) fn mapped_desired(dense_bytes: u64, auxiliary_bytes: u64) -> u64 {
    dense_bytes
        .saturating_add(auxiliary_bytes.min(MAPPED_AUXILIARY_CAP))
        .saturating_add(MAPPED_LRU_BYTES)
        .saturating_add((1 << 30) - 1)
        >> 30
        << 30
}

/// Replace the slot budget by one that admits only what stays anonymous in
/// mapped mode, when the live host holds the pack beside it. The routed
/// experts are clean page cache the OS may evict between turns; during a
/// pass the driver wires every bank, so the whole pack must fit at once.
/// `pool_bytes` is the paged KV pool, allocated before this decision.
pub(super) fn mapped_plan(
    plan: &mut Plan,
    dense_bytes: u64,
    auxiliary_bytes: u64,
    pool_bytes: u64,
    pack_bytes: u64,
) -> Result<()> {
    let desired = mapped_desired(dense_bytes, auxiliary_bytes);
    let anonymous_need = desired.saturating_add(pool_bytes);
    // A bootstrap snapshot exists only on the live macOS host; fixtures and
    // other platforms take the desired budget without a gate.
    let gate = match plan.physical_bytes {
        Some(_) => {
            let host = host_memory()?;
            plan.available_bytes = Some(host.available());
            plan.physical_bytes = Some(host.physical);
            mapped_gate(&host, anonymous_need, MAPPED_MARGIN)
        }
        None => MappedGate {
            bank_budget: pack_bytes,
            non_reclaimable: 0,
            anonymous_need,
        },
    };
    plan.mapped_bank_budget = gate.bank_budget;
    plan.live_non_reclaimable = gate.non_reclaimable;
    if gate.bank_budget < pack_bytes {
        return Err(Error::from_reason(format!(
            "mapped banks need {:.1} GiB, the host leaves {:.1} GiB beside {:.1} GiB live memory, {:.1} GiB anonymous need and a {:.0} GiB margin",
            pack_bytes as f64 / (1u64 << 30) as f64,
            gate.bank_budget as f64 / (1u64 << 30) as f64,
            gate.non_reclaimable as f64 / (1u64 << 30) as f64,
            anonymous_need as f64 / (1u64 << 30) as f64,
            MAPPED_MARGIN as f64 / (1u64 << 30) as f64,
        )));
    }
    plan.budget = desired;
    plan.resident = false;
    plan.mapped = true;
    Ok(())
}

/// The driver unwires a released bank 1.2-3.2 s after the turn that bound
/// it. A fallback budget taken before that counts the pack against the
/// slots for the life of the process (88 slots where ~27 GiB was there).
const UNWIRE_SETTLE: std::time::Duration = std::time::Duration::from_secs(3);
const UNWIRE_POLL: std::time::Duration = std::time::Duration::from_millis(250);
const UNWIRE_STEP: u64 = 256 << 20;
/// Fresh samples without a fall, one second of polls, before the release
/// counts as finished: the driver lets the banks go in steps, and one flat
/// sample between two steps would make the budget of that plateau permanent.
const UNWIRE_FLAT_SAMPLES: u32 = 4;

/// The slot budget for a process that leaves mapped mode after a tripwire:
/// the bootstrap rule over the live host, partial residency only. Called
/// with the banks already released; the budget has to cover what the store
/// still holds. A host that cannot be read leaves the held bytes as the
/// budget, so the store is never left between the two paths.
pub(super) fn slot_fallback_plan_with(
    plan: &mut Plan,
    held_bytes: u64,
    host: &mut dyn FnMut() -> Result<HostMemory>,
    sleep: &mut dyn FnMut(std::time::Duration),
) {
    let desired = plan.hot_bytes.saturating_add((2 << 30) - 1) >> 30 << 30;
    let budget = match plan.physical_bytes {
        Some(_) => match settled_host(host, sleep) {
            Ok(host) => {
                plan.available_bytes = Some(host.available());
                automatic_budget(desired, host.available(), host.physical)
            }
            Err(error) => {
                tracing::warn!(
                    "qwen4 residency: host memory unreadable after leaving mapped mode ({error}); slots sized from the held bytes"
                );
                0
            }
        },
        None => desired.min(super::weights::CACHE_BYTES),
    };
    plan.budget = budget.max(
        held_bytes
            .saturating_add(super::weights::MAX_READ_BYTES * 3)
            .saturating_add((1 << 30) - 1)
            >> 30
            << 30,
    );
    plan.resident = false;
    plan.mapped = false;
    plan.mapped_bytes = 0;
    plan.mapped_bank_budget = 0;
    plan.partial_expert_slots = 0;
}

/// Sample the host until wired memory has fallen and then held for
/// `UNWIRE_FLAT_SAMPLES` fresh samples, or `UNWIRE_SETTLE` has passed; the
/// last fresh sample is the one to size from. A sample equal to the previous
/// one is the kernel's cached answer (`HostMemory::faults`): it neither
/// counts as flat nor ends the loop.
fn settled_host(
    host: &mut dyn FnMut() -> Result<HostMemory>,
    sleep: &mut dyn FnMut(std::time::Duration),
) -> Result<HostMemory> {
    let mut last = host()?;
    let mut fell = false;
    let mut flat = 0;
    let mut waited = std::time::Duration::ZERO;
    while waited < UNWIRE_SETTLE {
        sleep(UNWIRE_POLL);
        waited += UNWIRE_POLL;
        let next = host()?;
        if next == last {
            continue;
        }
        if last.wired.saturating_sub(next.wired) > UNWIRE_STEP {
            fell = true;
            flat = 0;
        } else if fell {
            flat += 1;
        }
        last = next;
        if flat >= UNWIRE_FLAT_SAMPLES {
            break;
        }
    }
    Ok(last)
}

fn automatic_budget(desired: u64, available: u64, physical: u64) -> u64 {
    desired
        // Keep the system reserve after the decoder uses its separately
        // bounded 4 GiB working allowance, rather than filling that allowance
        // with more cached weights during automatic sizing.
        .min(available.saturating_sub(system_reserve(physical) + WORKING_BYTES))
        .min(physical.saturating_mul(3) / 4)
        >> 30
        << 30
}

/// Reconcile the bootstrap ceiling after tokenizer and private page pools have
/// been created, or after releasing an abandoned full-residency load. This is
/// called with no owned weight arrays, so live headroom needs no inferred credit
/// from MLX's process-global allocation counters.
pub(super) fn refresh_plan(plan: &mut Plan) -> Result<()> {
    if !cfg!(target_os = "macos") {
        return Ok(());
    }
    let available = available_memory()?;
    let physical = physical_memory()?;
    refresh_from_snapshot(plan, available, physical)
}

fn refresh_from_snapshot(plan: &mut Plan, available: u64, physical: u64) -> Result<()> {
    let budget = automatic_budget(plan.budget, available, physical);
    let desired = plan.hot_bytes.saturating_add((2 << 30) - 1) >> 30 << 30;
    if budget == 0 || (plan.policy == "full" && desired > budget) {
        return Err(Error::from_reason(
            "Qwen4 load has insufficient headroom for weights and first-request working memory",
        ));
    }
    plan.budget = budget;
    plan.resident = plan.policy != "stream" && desired <= budget;
    plan.available_bytes = Some(available);
    plan.physical_bytes = Some(physical);
    Ok(())
}

pub(super) fn admit(bytes: u64) -> Result<()> {
    if !cfg!(target_os = "macos") && bytes <= super::weights::CACHE_BYTES {
        return Ok(());
    }
    let mut available = available_memory()?;
    let physical = physical_memory()?;
    let ceiling = physical.saturating_mul(3) / 4;
    let reserve = system_reserve(physical);
    if bytes <= ceiling
        && !fits_with_headroom(bytes, available, reserve)
        && crate::array::memory::get_cache_memory() > 0.0
    {
        // Unused allocator buffers must not turn an otherwise admissible SSD
        // read into a failed turn. Reclaim only the freelist, then obtain fresh
        // pressure/headroom evidence and apply exactly the same safety limits.
        crate::array::memory::clear_cache();
        available = available_memory()?;
    }
    if bytes > ceiling || !fits_with_headroom(bytes, available, reserve) {
        return Err(Error::from_reason(format!(
            "Qwen4 cache admission refused: requested {:.1} GiB, free/file-backed estimate {:.1} GiB; {:.1} GiB system headroom is required",
            bytes as f64 / (1u64 << 30) as f64,
            available as f64 / (1u64 << 30) as f64,
            system_reserve(physical) as f64 / (1u64 << 30) as f64
        )));
    }
    Ok(())
}

/// Reuse a small bounded allocator freelist instead of destroying every
/// temporary Metal buffer at every layer. Also called before cold reads.
fn workspace_cache_bytes(physical: Option<u64>) -> u64 {
    // Spend at most half the separately reserved working allowance on reusable
    // buffers. This is derived from the bootstrap device snapshot, not a
    // 128-GiB-machine assumption. Live cold-read admission can reclaim it.
    physical.map_or(FREE_CACHE_BYTES, |bytes| {
        (bytes / 64).min(WORKING_BYTES / 2)
    })
}

pub(super) fn maintain_freelist(physical: Option<u64>) {
    let limit = if !runtime_flags::is_zero(c"MLX_QWEN4_REUSE_WORKSPACE") {
        workspace_cache_bytes(physical)
    } else {
        FREE_CACHE_BYTES
    };
    if runtime_flags::is_one(c"MLX_QWEN4_FLUSH_EVERY_LAYER")
        || crate::array::memory::get_cache_memory() > limit as f64
    {
        crate::array::memory::clear_cache();
    }
}

pub(super) fn check_growth_headroom(mapped: bool) -> Result<()> {
    let host = host_memory()?;
    let floor = if mapped {
        MAPPED_TURN_FLOOR
    } else {
        system_reserve(host.physical)
    };
    if host.available() < floor {
        return Err(Error::from_reason(
            "Qwen4 stopped weight growth to preserve system headroom",
        ));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn physical_memory() -> Result<u64> {
    let mut bytes = 0u64;
    let mut size = std::mem::size_of_val(&bytes);
    if unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&mut bytes as *mut u64).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    } != 0
    {
        return Err(Error::from_reason(
            "Cannot read physical memory for Qwen4 residency",
        ));
    }
    Ok(bytes)
}
#[cfg(not(target_os = "macos"))]
fn physical_memory() -> Result<u64> {
    Ok(0)
}

fn system_reserve(physical: u64) -> u64 {
    (physical / 8).clamp(2 << 30, 16 << 30)
}

fn fits_with_headroom(bytes: u64, available: u64, reserve: u64) -> bool {
    bytes
        .checked_add(reserve)
        .is_some_and(|need| need <= available)
}

/// Re-check the SSD read safety limit after freeing the allocator freelist.
/// A freelist spike is transient; real pressure still refuses after a
/// bounded re-poll.
pub(super) fn check_active_memory_budget(
    cache_limit: u64,
    mut active: impl FnMut() -> f64,
    mut reclaim: impl FnMut(),
    mut sleep: impl FnMut(std::time::Duration),
) -> Result<()> {
    if active() <= (cache_limit + (4 << 30)) as f64 {
        return Ok(());
    }
    reclaim();
    for _ in 0..3 {
        sleep(std::time::Duration::from_millis(100));
        if active() <= (cache_limit + (4 << 30)) as f64 {
            return Ok(());
        }
    }
    Err(Error::from_reason(
        "qwen4_exp SSD safety limit: active MLX memory exceeds weight budget plus 4 GiB",
    ))
}

fn available_memory() -> Result<u64> {
    Ok(host_memory()?.available())
}

#[cfg(target_os = "macos")]
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
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if ok != 0 || page_size <= 0 {
        return Err(Error::from_reason(
            "Cannot establish Qwen4 larger-cache memory headroom",
        ));
    }
    let stats = unsafe { stats.assume_init() };
    let page = page_size as u64;
    Ok(HostMemory {
        physical: physical_memory()?,
        free: u64::from(stats.free_count) * page,
        file: u64::from(stats.external_page_count) * page,
        anonymous: u64::from(stats.internal_page_count) * page,
        purgeable: u64::from(stats.purgeable_count) * page,
        wired: u64::from(stats.wire_count) * page,
        compressor: u64::from(stats.compressor_page_count) * page,
        faults: stats.faults,
    })
}

#[cfg(not(target_os = "macos"))]
pub(super) fn host_memory() -> Result<HostMemory> {
    Err(Error::from_reason(
        "Qwen4 larger-cache admission is currently available only on macOS",
    ))
}

/// Bytes this process has read from disk since it started
/// (`proc_pid_rusage`, the counter the mapped pack's page-ins land in).
#[cfg(target_os = "macos")]
pub(super) fn process_disk_reads() -> u64 {
    let mut info = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    let rc = unsafe {
        libc::proc_pid_rusage(
            std::process::id() as libc::c_int,
            libc::RUSAGE_INFO_V4,
            info.as_mut_ptr().cast(),
        )
    };
    if rc != 0 {
        return 0;
    }
    unsafe { info.assume_init() }.ri_diskio_bytesread
}

#[cfg(not(target_os = "macos"))]
pub(super) fn process_disk_reads() -> u64 {
    0
}

#[cfg(test)]
mod tests {
    #[test]
    fn workspace_reuse_scales_down_and_stays_within_working_reserve() {
        assert_eq!(super::workspace_cache_bytes(Some(16 << 30)), 256 << 20);
        assert_eq!(super::workspace_cache_bytes(Some(32 << 30)), 512 << 20);
        assert_eq!(super::workspace_cache_bytes(Some(128 << 30)), 2 << 30);
        assert_eq!(super::workspace_cache_bytes(Some(512 << 30)), 2 << 30);
        assert_eq!(super::workspace_cache_bytes(None), super::FREE_CACHE_BYTES);
    }

    use super::*;
    #[test]
    fn admission_reserves_headroom_and_rejects_overflow() {
        assert_eq!(automatic_budget(80 << 30, 66 << 30, 128 << 30), 46 << 30);
        assert_eq!(automatic_budget(80 << 30, 112 << 30, 128 << 30), 80 << 30);
        assert_eq!(automatic_budget(300 << 30, 200 << 30, 128 << 30), 96 << 30);
        assert_eq!(automatic_budget(80 << 30, 26 << 30, 128 << 30), 6 << 30);
        assert_eq!(automatic_budget(80 << 30, 19 << 30, 128 << 30), 0);
        assert_eq!(automatic_budget(80 << 30, 21 << 30, 128 << 30), 1 << 30);
        assert_eq!(automatic_budget(1 << 30, 64 << 30, 128 << 30), 1 << 30);
        assert_eq!(automatic_budget(8 << 30, 23 << 30, 128 << 30), 3 << 30);
        assert_eq!(automatic_budget(80 << 30, 28 << 30, 32 << 30), 20 << 30);
        assert_eq!(automatic_budget(2 << 30, 10 << 30, 16 << 30), 2 << 30);
        // A larger device is not capped by the 96 GiB limit of a 128 GiB Mac.
        assert_eq!(automatic_budget(180 << 30, 220 << 30, 256 << 30), 180 << 30);
        assert_eq!(automatic_budget(300 << 30, 230 << 30, 256 << 30), 192 << 30);
        // The same policy leaves system and execution room on smaller devices.
        for gib in [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512] {
            let physical = gib << 30;
            let available = physical * 7 / 8;
            let budget = automatic_budget(300 << 30, available, physical);
            assert!(budget <= physical * 3 / 4);
            if budget > 0 {
                assert!(budget + WORKING_BYTES + system_reserve(physical) <= available);
            }
        }
        assert!(fits_with_headroom(16 << 30, 32 << 30, 16 << 30));
        assert!(!fits_with_headroom(16 << 30, (32 << 30) - 1, 16 << 30));
        assert!(!fits_with_headroom(32 << 30, 40 << 30, 16 << 30));
        assert!(!fits_with_headroom(u64::MAX, u64::MAX, 16 << 30));
    }

    #[test]
    fn mapped_residency_needs_auto_and_a_complete_pack() {
        assert!(mapped_residency_selected("auto", false, true, false));
        // MLX_QWEN4_WEIGHT_CACHE_GIB pins an anonymous budget: slots.
        assert!(!mapped_residency_selected("auto", true, true, false));
        assert!(!mapped_residency_selected("stream", false, true, false));
        assert!(!mapped_residency_selected("full", false, true, false));
        // MLX_QWEN4_MAPPED_PACK=0 is the slot A/B arm.
        assert!(!mapped_residency_selected("auto", false, true, true));
        assert!(!mapped_residency_selected("auto", false, false, false));
    }

    fn gib(n: f64) -> u64 {
        (n * (1u64 << 30) as f64) as u64
    }

    /// The desktop of 2026-10-19 15:47 (vm_stat): the same free memory with
    /// a 40 GiB anonymous hog beside it must send the pack to slots, and
    /// free memory alone must not admit it.
    #[test]
    fn mapped_gate_reads_live_memory_not_free() {
        let pack = gib(73.15);
        let need = gib(13.0);
        let quiet = HostMemory {
            physical: 128 << 30,
            free: gib(77.3),
            file: gib(10.0),
            anonymous: gib(20.1),
            purgeable: gib(0.7),
            wired: gib(13.2),
            compressor: gib(7.5),
            faults: 0,
        };
        let gate = mapped_gate(&quiet, need, 0);
        assert_eq!(
            gate.non_reclaimable,
            gib(13.2) + gib(7.5) + gib(20.1) - gib(0.7)
        );
        assert_eq!(
            gate.bank_budget,
            (128u64 << 30) - gate.non_reclaimable - need
        );
        assert!(gate.bank_budget >= pack, "the quiet desktop holds the pack");
        assert!(
            mapped_gate(&quiet, need, MAPPED_MARGIN).bank_budget < pack,
            "with the margin this desktop is 2 GiB short"
        );
        // A touching 40 GiB hog: the kernel compressed and swapped idle apps
        // for it, so free barely moved while anonymous grew.
        let hog = HostMemory {
            free: gib(70.0),
            anonymous: gib(20.1) + gib(40.0),
            compressor: gib(9.0),
            ..quiet
        };
        assert!(
            mapped_gate(&hog, need, 0).bank_budget < pack,
            "live anonymous memory decides, not free"
        );
        assert!(mapped_gate(&hog, need, 0).bank_budget < gate.bank_budget - gib(39.0));
        // M3's 20 GiB hog (det-hog-1: 1.06 TB of re-reads) is refused too.
        let hog20 = HostMemory {
            anonymous: gib(20.1) + gib(20.0),
            ..quiet
        };
        assert!(mapped_gate(&hog20, need, 0).bank_budget < pack);
        // Purgeable pages are reclaimable; wired and compressed are not.
        let purgeable = HostMemory {
            purgeable: gib(30.0),
            ..hog20
        };
        assert!(mapped_gate(&purgeable, need, 0).bank_budget >= pack);
        let wired = HostMemory {
            wired: gib(13.2) + gib(20.0),
            ..quiet
        };
        assert!(mapped_gate(&wired, need, 0).bank_budget < pack);
        assert_eq!(mapped_gate(&quiet, u64::MAX, 0).bank_budget, 0);
    }

    fn plan_without_snapshot() -> Plan {
        Plan {
            budget: 73 << 30,
            hot_bytes: 83_631_869_440,
            resident: true,
            policy: "auto".into(),
            available_bytes: None,
            physical_bytes: None,
            resident_bank_bytes: 0,
            resident_expert_layers: 0,
            partial_expert_slots: 442,
            mapped_pack_files: 144,
            mapped: false,
            mapped_bytes: 0,
            mapped_bank_budget: 0,
            live_non_reclaimable: 0,
        }
    }

    #[test]
    fn mapped_budget_holds_dense_banks_bounded_auxiliary_and_lru_room() {
        assert_eq!(mapped_desired(gib(3.8), 0), 6 << 30);
        assert_eq!(mapped_desired(gib(3.8), gib(4.7)), 11 << 30);
        assert_eq!(mapped_desired(gib(3.8), 20 << 30), 14 << 30);
        assert_eq!(mapped_desired(0, 0), 2 << 30);
        let mut plan = plan_without_snapshot();
        // No live snapshot (fixtures, other platforms): the desired budget is
        // taken as is and the bank budget is the pack.
        mapped_plan(&mut plan, gib(3.8), 0, 1 << 30, gib(73.15)).unwrap();
        assert!(plan.mapped && !plan.resident);
        assert_eq!(plan.budget, 6 << 30);
        assert_eq!(plan.mapped_bank_budget, gib(73.15));
    }

    #[test]
    fn slot_fallback_plan_covers_what_the_store_holds() {
        let mut plan = plan_without_snapshot();
        plan.mapped = true;
        plan.mapped_bytes = gib(73.15);
        plan.mapped_bank_budget = gib(74.0);
        let mut unread =
            || -> Result<HostMemory> { panic!("a plan without a snapshot reads no host") };
        slot_fallback_plan_with(&mut plan, gib(11.4), &mut unread, &mut |_| {});
        assert!(!plan.mapped && !plan.resident);
        assert_eq!(plan.mapped_bytes, 0);
        assert_eq!(plan.mapped_bank_budget, 0);
        assert_eq!(
            plan.budget,
            super::super::weights::CACHE_BYTES.max(12 << 30)
        );
        plan.budget = 0;
        slot_fallback_plan_with(&mut plan, 0, &mut unread, &mut |_| {});
        assert_eq!(plan.budget, super::super::weights::CACHE_BYTES);
    }

    /// Request 1 of M5-hog40-during-1 started 13 ms after request 0 ended,
    /// while 23 GiB of the pack were still wired: the budget read 20 GiB
    /// where the unwired host gave 27-29. The plan sizes from the sample
    /// after wired has fallen and settled, not from the first one.
    #[test]
    fn slot_fallback_sizes_from_the_host_after_the_banks_unwire() {
        let still_wired = HostMemory {
            physical: 128 << 30,
            free: gib(0.13),
            file: gib(43.69),
            anonymous: gib(30.0),
            purgeable: 0,
            wired: gib(23.31),
            compressor: gib(10.0),
            faults: 1,
        };
        let unwiring = HostMemory {
            free: gib(0.2),
            file: gib(47.2),
            wired: gib(19.0),
            faults: 2,
            ..still_wired
        };
        let settled = HostMemory {
            free: gib(0.3),
            file: gib(50.6),
            wired: gib(16.7),
            faults: 3,
            ..still_wired
        };
        let flat = |n: u64| HostMemory {
            faults: settled.faults + n,
            ..settled
        };
        let mut samples = std::collections::VecDeque::from([
            still_wired,
            unwiring,
            settled,
            flat(1),
            flat(2),
            flat(3),
            flat(4),
        ]);
        let mut slept = Vec::new();
        let mut plan = plan_without_snapshot();
        plan.physical_bytes = Some(128 << 30);
        plan.hot_bytes = gib(83.0);
        plan.mapped = true;
        slot_fallback_plan_with(
            &mut plan,
            gib(5.0),
            &mut || Ok(samples.pop_front().expect("more samples than polls")),
            &mut |d| slept.push(d),
        );
        assert_eq!(plan.available_bytes, Some(settled.available()));
        assert_eq!(
            plan.budget,
            automatic_budget(gib(84.0), settled.available(), 128 << 30)
        );
        assert_eq!(plan.budget, gib(30.0));
        assert!(
            automatic_budget(gib(84.0), still_wired.available(), 128 << 30) < plan.budget,
            "the wired sample would have undersized the slots"
        );
        assert_eq!(
            slept,
            vec![UNWIRE_POLL; 2 + UNWIRE_FLAT_SAMPLES as usize],
            "one second of flat fresh samples past the last fall"
        );
        assert_eq!(samples.len(), 0);
        assert!(!plan.mapped);

        // A host that never falls is polled for the settle window, no longer.
        let mut polls = 0usize;
        let mut slept = std::time::Duration::ZERO;
        let mut plan = plan_without_snapshot();
        plan.physical_bytes = Some(128 << 30);
        plan.hot_bytes = gib(83.0);
        slot_fallback_plan_with(
            &mut plan,
            0,
            &mut || {
                polls += 1;
                Ok(settled)
            },
            &mut |d| slept += d,
        );
        assert_eq!(slept, UNWIRE_SETTLE);
        assert_eq!(
            polls,
            1 + (UNWIRE_SETTLE.as_millis() / UNWIRE_POLL.as_millis()) as usize
        );
        assert_eq!(plan.budget, gib(30.0));

        // An unreadable host sizes from the held bytes and still leaves mapped mode.
        let mut plan = plan_without_snapshot();
        plan.physical_bytes = Some(128 << 30);
        plan.mapped = true;
        slot_fallback_plan_with(
            &mut plan,
            gib(11.4),
            &mut || Err(Error::from_reason("no host statistics")),
            &mut |_| panic!("no poll without a first sample"),
        );
        assert!(!plan.mapped);
        assert_eq!(plan.budget, 12 << 30);
    }

    fn release(wired: f64, file: f64, faults: u64) -> HostMemory {
        HostMemory {
            physical: 128 << 30,
            free: 1 << 30,
            file: gib(file),
            anonymous: 30 << 30,
            purgeable: 0,
            wired: gib(wired),
            compressor: 10 << 30,
            faults,
        }
    }

    /// The driver lets the banks go in steps. One flat sample between two
    /// steps (50 -> 45 -> 45 -> 16 GiB wired) is not the end of the release:
    /// the budget comes from the 16 GiB host, not the 45 GiB plateau.
    #[test]
    fn slot_fallback_waits_out_a_release_plateau() {
        let plateau = release(45.0, 25.0, 3);
        let done = release(16.0, 54.0, 8);
        let mut samples = std::collections::VecDeque::from([
            release(50.0, 20.0, 1),
            release(45.0, 25.0, 2),
            plateau,
            release(16.0, 54.0, 4),
            release(16.0, 54.0, 5),
            release(16.0, 54.0, 6),
            release(16.0, 54.0, 7),
            done,
        ]);
        let mut polls = 0usize;
        let mut plan = plan_without_snapshot();
        plan.physical_bytes = Some(128 << 30);
        plan.hot_bytes = gib(83.0);
        plan.mapped = true;
        slot_fallback_plan_with(
            &mut plan,
            0,
            &mut || Ok(samples.pop_front().expect("more samples than polls")),
            &mut |_| polls += 1,
        );
        assert_eq!(samples.len(), 0);
        assert_eq!(polls, 7);
        assert_eq!(plan.available_bytes, Some(done.available()));
        assert_eq!(plan.budget, gib(35.0));
        assert_eq!(
            automatic_budget(gib(84.0), plateau.available(), 128 << 30),
            gib(6.0),
            "stopping at the plateau would have left 6 GiB of slots for good"
        );
    }

    /// `host_statistics64` answers a few fresh reads per second and repeats
    /// its last answer otherwise; a run of those copies right after the first
    /// fall is skipped, not taken as a settled host.
    #[test]
    fn slot_fallback_skips_the_kernels_cached_samples() {
        let first_fall = release(45.0, 25.0, 2);
        let done = release(16.0, 54.0, 7);
        let mut samples = std::collections::VecDeque::from([
            release(50.0, 20.0, 1),
            first_fall,
            first_fall,
            first_fall,
            first_fall,
            first_fall,
            release(16.0, 54.0, 3),
            release(16.0, 54.0, 4),
            release(16.0, 54.0, 5),
            release(16.0, 54.0, 6),
            done,
        ]);
        let mut polls = 0usize;
        let mut plan = plan_without_snapshot();
        plan.physical_bytes = Some(128 << 30);
        plan.hot_bytes = gib(83.0);
        plan.mapped = true;
        slot_fallback_plan_with(
            &mut plan,
            0,
            &mut || Ok(samples.pop_front().expect("more samples than polls")),
            &mut |_| polls += 1,
        );
        assert_eq!(samples.len(), 0);
        assert_eq!(polls, 10);
        assert_eq!(plan.available_bytes, Some(done.available()));
        assert_eq!(plan.budget, gib(35.0));
    }

    #[test]
    fn load_reconciliation_preserves_working_room_and_explicit_full_policy() {
        let original = Plan {
            budget: 79 << 30,
            hot_bytes: 83_631_869_440,
            resident: true,
            policy: "auto".into(),
            available_bytes: Some(99 << 30),
            physical_bytes: Some(128 << 30),
            resident_bank_bytes: 0,
            resident_expert_layers: 0,
            partial_expert_slots: 0,
            mapped_pack_files: 0,
            mapped: false,
            mapped_bytes: 0,
            mapped_bank_budget: 0,
            live_non_reclaimable: 0,
        };
        let mut plan = original.clone();
        refresh_from_snapshot(&mut plan, 94 << 30, 128 << 30).unwrap();
        assert_eq!(plan.budget, 74 << 30);
        assert!(!plan.resident);
        assert!(plan.budget + WORKING_BYTES + system_reserve(128 << 30) <= 94 << 30);
        // A later free-memory increase cannot exceed the admitted ceiling.
        refresh_from_snapshot(&mut plan, 112 << 30, 128 << 30).unwrap();
        assert_eq!(plan.budget, 74 << 30);
        let mut full = original.clone();
        full.policy = "full".into();
        assert!(refresh_from_snapshot(&mut full, 94 << 30, 128 << 30).is_err());
        assert_eq!(full.budget, original.budget);
        refresh_from_snapshot(&mut full, 112 << 30, 128 << 30).unwrap();
        assert!(full.resident);
        assert!(refresh_from_snapshot(&mut plan, 19 << 30, 128 << 30).is_err());
    }

    #[test]
    fn active_budget_reclaims_and_repolls_before_refusing() {
        use std::sync::{Arc, Mutex};
        let readings = Arc::new(Mutex::new(
            vec![40.0 * (1 << 30) as f64, 10.0 * (1 << 30) as f64].into_iter(),
        ));
        let mut reclaims = 0;
        let mut sleeps = 0;
        super::check_active_memory_budget(
            32 << 30,
            {
                let readings = Arc::clone(&readings);
                move || readings.lock().unwrap().next().unwrap_or(0.0)
            },
            || reclaims += 1,
            |_| sleeps += 1,
        )
        .unwrap();
        assert_eq!(reclaims, 1);
        assert_eq!(sleeps, 1);
    }

    #[test]
    fn active_budget_refuses_after_bounded_repolls() {
        let mut reclaims = 0;
        let mut sleeps = 0;
        let err = super::check_active_memory_budget(
            8 << 30,
            || 40.0 * (1 << 30) as f64,
            || reclaims += 1,
            |_| sleeps += 1,
        )
        .unwrap_err();
        assert_eq!(reclaims, 1);
        assert_eq!(sleeps, 3);
        assert!(err.reason.contains("SSD safety limit"));
    }
}

/// Bound a layer's prompt scratch from checkpoint dimensions and the same live
/// bootstrap snapshot that admitted weights. The rest of the 4 GiB working
/// allowance covers pages, owner snapshots, the allocator and import staging.
pub(super) fn prefill_scratch(c: &super::config::Config, tokens: usize) -> u64 {
    let h = c.hidden_size as u64;
    let hc = c.hc_count as u64;
    let qkv = (2 * c.linear_num_key_heads * c.linear_key_head_dim
        + c.linear_num_value_heads * c.linear_value_head_dim) as u64;
    let experts = c.num_experts_per_tok as u64 * (4 * h + 8 * c.moe_intermediate_size as u64);
    // Window PLE owns decoded rows, key/query/value intermediates and the
    // dilated F32 convolution taps. The router completion precedes expert
    // execution, so its temporaries and PLE's do not coexist. Reserve the
    // larger stage plus common hidden storage, rather than adding both peaks.
    let ple = if c.ple_layer_ids.is_empty() {
        0
    } else {
        c.ple_embed_dim as u64 + (8 + 2 * c.ple_conv_kernel_size as u64) * hc * h
    };
    let per_token = 4
        * (8 * hc * h
            + ple.max(8 * qkv + experts + 6 * (c.num_attention_heads * c.head_dim) as u64));
    let t = tokens as u64;
    let window = 4
        * c.num_attention_heads as u64
        * t
        * c.indexer_budget.min(super::MAX_PREFILL_CHUNK * 2) as u64;
    // Batched QSA past the budget: F32 per-head and summed block scores for
    // every row against every block, plus the contiguous BF16 K/V prefix.
    let context = c.effective_context_limit() as u64;
    let blocks = context / c.indexer_compress_ratio.max(1) as u64;
    let qsa = (t * c.indexer_n_heads as u64 * blocks * 4 * 2)
        .saturating_add(t * blocks * 4 * 6)
        .saturating_add(2 * c.num_key_value_heads as u64 * context * c.head_dim as u64 * 2);
    per_token.saturating_mul(t).saturating_add(window.max(qsa))
}

/// The prefill window the current plan's scratch allowance admits; computed
/// again whenever the plan changes (a mapped process that moved to slots
/// has a smaller allowance than the one its load admitted).
pub(super) fn prefill_window(c: &super::config::Config, plan: &Plan) -> usize {
    let live = match (plan.available_bytes, plan.physical_bytes) {
        (Some(a), Some(p)) => a.saturating_sub(plan.budget + system_reserve(p) + (2 << 30)),
        _ => plan.budget / 16,
    };
    // The mapped budget holds no experts, so it does not scale the scratch.
    let allowance = if plan.mapped {
        (2 << 30).min(live)
    } else {
        (plan.budget / 16).min(2 << 30).min(live)
    };
    let mut window = super::MAX_PREFILL_CHUNK;
    while window > 1 && prefill_scratch(c, window) > allowance {
        window /= 2;
    }
    window
}

/// Runtime tuning may shorten the bootstrap-admitted window, never grow it.
/// Invalid values preserve admission. This does not allocate another page pool
/// or alter the verified/MTP frontier width.
pub(super) fn prefill_slice(admitted: usize, requested: Option<&str>) -> usize {
    requested
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| n.is_power_of_two())
        .map_or(admitted, |n| n.min(admitted))
}

#[test]
fn runtime_slices_cannot_exceed_bootstrap_admission() {
    for admitted in [1, 2, 32, 64, 256, 512, 1024] {
        for request in [
            None,
            Some("0"),
            Some("7"),
            Some("256"),
            Some("1024"),
            Some("2048"),
            Some("invalid"),
        ] {
            let actual = prefill_slice(admitted, request);
            assert!(actual > 0 && actual <= admitted && actual.is_power_of_two());
        }
        assert_eq!(prefill_slice(admitted, Some("256")), admitted.min(256));
    }
}
