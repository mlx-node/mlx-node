//! Admit hot weights against live system headroom. Preserve room for other
//! applications, recurrent state, page pools and transient work.
use napi::{Error, Result};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

pub(super) const FREE_CACHE_BYTES: u64 = 512 << 20;
pub(super) const WORKING_BYTES: u64 = 4 << 30;

#[derive(Clone, Debug)]
pub(super) struct Plan {
    pub budget: u64,
    pub hot_bytes: u64,
    pub available_bytes: Option<u64>,
    pub physical_bytes: Option<u64>,
    /// Shared with the model handle's clone: banks installed on the model
    /// thread after load (the MTP expert banks) publish their bytes here so
    /// `residencyInfo` reports what is actually pinned.
    pub resident_bank_bytes: Arc<AtomicU64>,
    /// Shared partial-slot capacity, or the expert count at full direct
    /// identity residency.
    pub partial_expert_slots: usize,
    /// Every routed expert has an anonymous direct-GGUF bank at identity
    /// capacity.
    pub direct: bool,
    /// A full-bank refusal selected partial slots under the direct gate.
    pub partial_direct_gate: bool,
    /// Anonymous bytes allocated for direct expert banks.
    pub direct_bytes: u64,
    /// Routed layers in the direct fill schedule.
    pub direct_layers: usize,
    /// Bytes of direct banks the host could hold beside everything that
    /// stays anonymous when the residency was chosen (`bank_gate`).
    pub bank_budget: u64,
    /// `HostMemory::non_reclaimable` when the residency was chosen.
    pub live_non_reclaimable: u64,
}

impl Plan {
    pub(super) fn use_direct_turn_floor(&self) -> bool {
        self.direct || self.partial_direct_gate
    }
}

/// LRU rows (token, PLE and streamed auxiliary MTP/vision rows), paired banks
/// and import staging beside full direct banks. A resident draft bank must be
/// included in `direct_desired` before the gate is evaluated.
const DIRECT_LRU_BYTES: u64 = 2 << 30;
/// Room the direct-bank decision keeps between its need and physical RAM for
/// other processes that allocate during a pass. A refusal selects bounded
/// partial slots instead of allocating the full identity bank set.
pub(super) const DIRECT_MARGIN: u64 = 4 << 30;
/// Minimum free plus file-cache memory during a direct-gated turn. The load
/// gate owns admission; this floor only refuses growth near the kernel's
/// critical level.
pub(super) const DIRECT_TURN_FLOOR: u64 = 2 << 30;

/// One `host_statistics64` read, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct HostMemory {
    pub physical: u64,
    pub free: u64,
    /// Pageable file-backed pages the kernel can drop on demand.
    pub file: u64,
    /// Pageable anonymous pages of every process, active or not.
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

    /// Memory a turn cannot take from the host: wired, the compressor pool
    /// and other processes' anonymous pages (purgeable ones excepted).
    pub fn non_reclaimable(&self) -> u64 {
        self.wired
            .saturating_add(self.compressor)
            .saturating_add(self.anonymous.saturating_sub(self.purgeable))
    }
}

/// The residency decision as a byte budget for anonymous direct banks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BankGate {
    pub bank_budget: u64,
    pub non_reclaimable: u64,
    pub anonymous_need: u64,
}

/// Direct banks fit when physical RAM minus the margin, the host's live
/// non-reclaimable memory and the other anonymous need (dense banks, bounded
/// auxiliary inventory, LRU room and the KV pool) covers the whole bank set.
pub(super) fn bank_gate(host: &HostMemory, anonymous_need: u64, margin: u64) -> BankGate {
    let non_reclaimable = host.non_reclaimable();
    BankGate {
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
    let desired = hot_bytes
        .checked_add((2 << 30) - 1)
        .map(|n| (n >> 30) << 30)
        .ok_or_else(|| Error::from_reason("Qwen4 hot-weight size overflow"))?;
    // Read the live machine at bootstrap; the budget must not force an
    // allocation on a busy desktop.
    let snapshot = if cfg!(target_os = "macos") {
        Some((available_memory()?, physical_memory()?))
    } else {
        None
    };
    let budget = match snapshot {
        Some((available, physical)) => automatic_budget(desired, available, physical),
        None => super::weights::CACHE_BYTES,
    };
    if budget == 0 {
        return Err(Error::from_reason(
            "Qwen4 bootstrap has no whole GiB for weights after system/working reserves; release memory before loading",
        ));
    }
    admit(budget)?;
    Ok(Plan {
        budget,
        hot_bytes,
        available_bytes: snapshot.map(|s| s.0),
        physical_bytes: snapshot.map(|s| s.1),
        resident_bank_bytes: Arc::default(),
        partial_expert_slots: 0,
        direct: false,
        partial_direct_gate: false,
        direct_bytes: 0,
        direct_layers: 0,
        bank_budget: 0,
        live_non_reclaimable: 0,
    })
}

/// Whole GiB of anonymous need beside direct experts: dense banks (plus Q8
/// draft banks while the MTP head is attached) and LRU room. Routed experts
/// are gated separately; auxiliary rows stream through the LRU allowance.
pub(super) fn direct_desired(dense_bytes: u64) -> u64 {
    dense_bytes
        .saturating_add(DIRECT_LRU_BYTES)
        .saturating_add((1 << 30) - 1)
        >> 30
        << 30
}

/// Size the anonymous budget and decide whether the live host can hold the
/// complete direct bank set beside it. `pool_bytes` is the paged KV pool,
/// allocated before this decision. A refusal leaves the partial bank budget
/// in the plan for the slot selector.
pub(super) fn direct_plan(
    plan: &mut Plan,
    dense_bytes: u64,
    pool_bytes: u64,
    bank_bytes: u64,
) -> Result<()> {
    let desired = direct_desired(dense_bytes);
    let anonymous_need = desired.saturating_add(pool_bytes);
    plan.budget = desired;
    // A bootstrap snapshot exists only on the live macOS host; fixtures and
    // other platforms take the desired budget without a gate.
    let gate = match plan.physical_bytes {
        Some(_) => {
            let host = host_memory()?;
            plan.available_bytes = Some(host.available());
            plan.physical_bytes = Some(host.physical);
            bank_gate(&host, anonymous_need, DIRECT_MARGIN)
        }
        None => BankGate {
            bank_budget: bank_bytes,
            non_reclaimable: 0,
            anonymous_need,
        },
    };
    plan.bank_budget = gate.bank_budget;
    plan.live_non_reclaimable = gate.non_reclaimable;
    if gate.bank_budget < bank_bytes {
        return Err(Error::from_reason(format!(
            "direct expert banks need {:.1} GiB, the host leaves {:.1} GiB beside {:.1} GiB live memory, {:.1} GiB anonymous need and a {:.0} GiB margin",
            bank_bytes as f64 / (1u64 << 30) as f64,
            gate.bank_budget as f64 / (1u64 << 30) as f64,
            gate.non_reclaimable as f64 / (1u64 << 30) as f64,
            anonymous_need as f64 / (1u64 << 30) as f64,
            DIRECT_MARGIN as f64 / (1u64 << 30) as f64,
        )));
    }
    plan.direct = true;
    plan.direct_bytes = 0;
    plan.direct_layers = 0;
    Ok(())
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
/// been created, or after releasing an abandoned bank load. This is called
/// with no owned weight arrays, so live headroom needs no inferred credit from
/// MLX's process-global allocation counters.
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
    if budget == 0 {
        return Err(Error::from_reason(
            "Qwen4 load has insufficient headroom for weights and first-request working memory",
        ));
    }
    plan.budget = budget;
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

/// Admit one actual direct-residency allocation against the same live-memory
/// budget as the full-bank gate. Unlike slot admission, this does not add the
/// bootstrap system reserve a second time or count banks already allocated by
/// this process as anonymous need.
pub(super) fn admit_direct(bytes: u64) -> Result<()> {
    let host = host_memory()?;
    let budget = bank_gate(&host, 0, DIRECT_MARGIN).bank_budget;
    if bytes > budget {
        return Err(Error::from_reason(format!(
            "Qwen4 direct residency admission refused: requested {:.1} GiB, the host leaves {:.1} GiB beside {:.1} GiB live memory and a {:.0} GiB margin",
            bytes as f64 / (1u64 << 30) as f64,
            budget as f64 / (1u64 << 30) as f64,
            host.non_reclaimable() as f64 / (1u64 << 30) as f64,
            DIRECT_MARGIN as f64 / (1u64 << 30) as f64,
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
    let limit = workspace_cache_bytes(physical);
    if crate::array::memory::get_cache_memory() > limit as f64 {
        crate::array::memory::clear_cache();
    }
}

pub(super) fn check_growth_headroom(direct_gate: bool) -> Result<()> {
    check_growth_headroom_with(&host_memory()?, direct_gate)
}

pub(super) fn check_growth_headroom_with(host: &HostMemory, direct_gate: bool) -> Result<()> {
    let floor = if direct_gate {
        DIRECT_TURN_FLOOR
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
    crate::utils::host::physical_memory()
        .ok_or_else(|| Error::from_reason("Cannot read physical memory for Qwen4 residency"))
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
    let page = crate::array::page_size() as u64;
    if ok != 0 {
        return Err(Error::from_reason(
            "Cannot establish Qwen4 larger-cache memory headroom",
        ));
    }
    let stats = unsafe { stats.assume_init() };
    Ok(HostMemory {
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
pub(super) fn host_memory() -> Result<HostMemory> {
    Err(Error::from_reason(
        "Qwen4 larger-cache admission is currently available only on macOS",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_reuse_scales_down_and_stays_within_working_reserve() {
        assert_eq!(super::workspace_cache_bytes(Some(16 << 30)), 256 << 20);
        assert_eq!(super::workspace_cache_bytes(Some(32 << 30)), 512 << 20);
        assert_eq!(super::workspace_cache_bytes(Some(128 << 30)), 2 << 30);
        assert_eq!(super::workspace_cache_bytes(Some(512 << 30)), 2 << 30);
        assert_eq!(super::workspace_cache_bytes(None), super::FREE_CACHE_BYTES);
    }

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
        assert_eq!(automatic_budget(180 << 30, 220 << 30, 256 << 30), 180 << 30);
        assert_eq!(automatic_budget(300 << 30, 230 << 30, 256 << 30), 192 << 30);
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

    fn gib(n: f64) -> u64 {
        (n * (1u64 << 30) as f64) as u64
    }

    fn coherent(host: HostMemory) -> HostMemory {
        let sum = host.free + host.file + host.anonymous + host.wired + host.compressor;
        assert!(sum <= host.physical + (1 << 30), "incoherent host fixture");
        assert!(host.purgeable <= host.anonymous);
        host
    }

    #[test]
    fn bank_gate_reads_live_memory_not_free() {
        let banks = gib(73.15);
        let need = gib(13.0);
        let quiet = coherent(HostMemory {
            physical: 128 << 30,
            free: gib(77.3),
            file: gib(10.0),
            anonymous: gib(20.1),
            purgeable: gib(0.7),
            wired: gib(13.2),
            compressor: gib(7.5),
        });
        let gate = bank_gate(&quiet, need, 0);
        assert_eq!(
            gate.non_reclaimable,
            gib(13.2) + gib(7.5) + gib(20.1) - gib(0.7)
        );
        assert_eq!(
            gate.bank_budget,
            (128u64 << 30) - gate.non_reclaimable - need
        );
        assert!(gate.bank_budget >= banks);
        assert!(bank_gate(&quiet, need, DIRECT_MARGIN).bank_budget < banks);
        let hog = coherent(HostMemory {
            free: gib(37.3),
            anonymous: gib(20.1) + gib(40.0),
            ..quiet
        });
        assert!(bank_gate(&hog, need, 0).bank_budget < banks);
        assert!(bank_gate(&hog, need, 0).bank_budget < gate.bank_budget - gib(39.0));
        let hog20 = coherent(HostMemory {
            free: gib(57.3),
            anonymous: gib(20.1) + gib(20.0),
            ..quiet
        });
        assert!(bank_gate(&hog20, need, 0).bank_budget < banks);
        let purgeable = coherent(HostMemory {
            purgeable: gib(30.0),
            ..hog20
        });
        assert!(bank_gate(&purgeable, need, 0).bank_budget >= banks);
        let wired = coherent(HostMemory {
            free: gib(57.3),
            wired: gib(13.2) + gib(20.0),
            ..quiet
        });
        assert!(bank_gate(&wired, need, 0).bank_budget < banks);
        assert_eq!(bank_gate(&quiet, u64::MAX, 0).bank_budget, 0);
    }

    pub(super) fn plan_without_snapshot() -> Plan {
        Plan {
            budget: 73 << 30,
            hot_bytes: 83_631_869_440,
            available_bytes: None,
            physical_bytes: None,
            resident_bank_bytes: Arc::default(),
            partial_expert_slots: 442,
            direct: false,
            partial_direct_gate: false,
            direct_bytes: 0,
            direct_layers: 0,
            bank_budget: 0,
            live_non_reclaimable: 0,
        }
    }

    #[test]
    fn direct_budget_holds_dense_banks_lru_room_and_full_banks() {
        assert_eq!(direct_desired(gib(3.8)), 6 << 30);
        assert_eq!(direct_desired(gib(1.9)), 4 << 30);
        assert_eq!(direct_desired(0), 2 << 30);
        let mut plan = plan_without_snapshot();
        direct_plan(&mut plan, gib(3.8), 1 << 30, gib(73.15)).unwrap();
        assert!(plan.direct);
        assert_eq!(plan.budget, 6 << 30);
        assert_eq!(plan.bank_budget, gib(73.15));
        let need = direct_desired(gib(4.9)) + (1 << 30);
        assert_eq!(need, 8 << 30);
        let desktop = |live: f64| {
            coherent(HostMemory {
                physical: 128 << 30,
                free: gib(60.0),
                file: gib(128.0 - 60.0) - gib(live),
                anonymous: gib(live) - gib(20.0),
                purgeable: 0,
                wired: gib(13.0),
                compressor: gib(7.0),
            })
        };
        assert!(bank_gate(&desktop(42.8), need, DIRECT_MARGIN).bank_budget >= gib(73.15));
        assert!(bank_gate(&desktop(42.0), need, DIRECT_MARGIN).bank_budget >= gib(73.15));
        assert!(bank_gate(&desktop(42.9), need, DIRECT_MARGIN).bank_budget < gib(73.15));
    }

    #[test]
    fn load_reconciliation_preserves_working_room() {
        let mut plan = Plan {
            budget: 79 << 30,
            hot_bytes: 83_631_869_440,
            available_bytes: Some(99 << 30),
            physical_bytes: Some(128 << 30),
            resident_bank_bytes: Arc::default(),
            partial_expert_slots: 0,
            direct: false,
            partial_direct_gate: false,
            direct_bytes: 0,
            direct_layers: 0,
            bank_budget: 0,
            live_non_reclaimable: 0,
        };
        refresh_from_snapshot(&mut plan, 94 << 30, 128 << 30).unwrap();
        assert_eq!(plan.budget, 74 << 30);
        assert!(plan.budget + WORKING_BYTES + system_reserve(128 << 30) <= 94 << 30);
        refresh_from_snapshot(&mut plan, 112 << 30, 128 << 30).unwrap();
        assert_eq!(plan.budget, 74 << 30);
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

/// Recurrent matrices, convolution windows, the compressed indexer history
/// and PLE windows one sequence keeps as F32 arrays outside the page pool.
pub(super) fn row_state_bytes(c: &super::config::Config) -> u64 {
    let linear = (0..c.num_hidden_layers).filter(|&i| c.linear(i)).count();
    let full = c.num_hidden_layers - linear;
    (linear
        * (c.linear_num_value_heads * c.linear_key_head_dim * c.linear_value_head_dim * 4
            + (2 * c.linear_num_key_heads * c.linear_key_head_dim
                + c.linear_num_value_heads * c.linear_value_head_dim)
                * c.linear_conv_kernel_dim
                * 4)
        + full * c.indexer_head_dim * c.effective_context_limit() / c.indexer_compress_ratio * 4
        + c.ple_layer_ids.len()
            * c.hc_count
            * c.hidden_size
            * c.ple_conv_kernel_size
            * c.ngram_size
            * 4) as u64
}

/// F32 draft-layer K/V per token.
pub(super) fn draft_kv_bytes_per_token(c: &super::config::Config) -> u64 {
    (c.num_key_value_heads * c.head_dim * 8) as u64
}

/// Verify snapshots of up to four accepted frontiers plus the draft K/V of
/// `tokens`: what an owner holds once every token is in its draft cache.
pub(super) fn draft_state_bytes(c: &super::config::Config, tokens: u64) -> u64 {
    4 * row_state_bytes(c) + tokens * draft_kv_bytes_per_token(c)
}

/// One BF16 target hidden row (`hc_count x hidden_size`) an MTP owner keeps
/// until the next first draft pass writes it into the draft cache.
pub(super) fn draft_hidden_row_bytes(c: &super::config::Config) -> u64 {
    (c.hc_count * c.hidden_size * 2) as u64
}

/// What an MTP owner holds now: draft K/V for the rows in its cache, the
/// hidden rows still waiting (trusted rows and the seed), and the verify
/// snapshots while it is inside an MTP turn. A waiting row has no K/V yet.
pub(super) fn draft_owner_bytes(
    c: &super::config::Config,
    cached_rows: u64,
    hidden_rows: u64,
    tracking: bool,
) -> u64 {
    let snapshots = if tracking { 4 * row_state_bytes(c) } else { 0 };
    snapshots + cached_rows * draft_kv_bytes_per_token(c) + hidden_rows * draft_hidden_row_bytes(c)
}

/// What the working allowance leaves for scheduled rows and draft state once
/// the page pool and one prefill window's scratch have their share.
pub(super) fn scheduled_state_budget(pool_bytes: u64, prefill_scratch: u64) -> u64 {
    WORKING_BYTES
        .saturating_sub(pool_bytes)
        .saturating_sub(prefill_scratch)
}

/// The prefill window the current plan's scratch allowance admits.
pub(super) fn prefill_window(c: &super::config::Config, plan: &Plan) -> usize {
    let anonymous_budget = plan.budget.saturating_sub(plan.direct_bytes);
    let live = match (plan.available_bytes, plan.physical_bytes) {
        (Some(a), Some(p)) => {
            let reserve = if plan.partial_direct_gate {
                DIRECT_TURN_FLOOR
            } else {
                system_reserve(p) + (2 << 30)
            };
            a.saturating_sub(anonymous_budget + reserve)
        }
        _ => anonymous_budget / 16,
    };
    // Full direct budgets hold experts separately, so their total bank bytes
    // do not scale the scratch allowance.
    let allowance = if plan.direct {
        (2 << 30).min(live)
    } else {
        (anonymous_budget / 16).min(2 << 30).min(live)
    };
    let mut window = super::MAX_PREFILL_CHUNK;
    while window > 1 && prefill_scratch(c, window) > allowance {
        window /= 2;
    }
    window
}

/// Flash Next geometry (48 layers, 12 full attention, K/V 2 x 256, 32,768
/// context): one 32k MTP owner fits the working budget beside the 1 GiB page
/// pool and the 1024-token prefill window, two do not, four AR rows do.
/// MUTATION: stop subtracting `prefill_scratch` in `scheduled_state_budget`;
/// two 32k MTP owners then fit and the `2 * mtp_32k > budget` assertion fails.
/// MUTATION: drop the four verify snapshots from `draft_state_bytes`; the 16k
/// pair then fits and the `2 * row + 2 * draft_16k > budget` assertion fails.
/// The released Flash Next geometry (Qwen3.8 Flash Next).
#[cfg(test)]
pub(super) fn flash_next_config() -> super::config::Config {
    super::config::Config::parse(&serde_json::json!({
        "model_type": "qwen4_exp_text",
        "hidden_size": 2560,
        "num_hidden_layers": 48,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "vocab_size": 248320,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "hc_count": 4,
        "hc_lowrank": 320,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 262144,
        "full_attention_interval": 4,
        "indexer_n_heads": 4,
        "indexer_head_dim": 128,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
        "ple_layer_ids": [2],
        "ple_embed_dim": 2560,
        "ple_conv_kernel_size": 4,
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "ngram_vocab_size_base": 20000000u64,
        "split_ngram_parts": 128,
        "eos_token_id": 248044
    }))
    .unwrap()
}

#[test]
fn direct_prefill_window_excludes_the_full_bank_bytes_from_anonymous_scratch() {
    let c = flash_next_config();
    let mut direct = tests::plan_without_snapshot();
    direct.available_bytes = Some(112 << 30);
    direct.physical_bytes = Some(128 << 30);
    direct.budget = 80 << 30;
    direct.direct = true;
    direct.direct_bytes = 73 << 30;
    direct.direct_layers = 48;
    let mut without_bank_charge = direct.clone();
    without_bank_charge.direct_bytes = 0;
    without_bank_charge.direct_layers = 0;
    without_bank_charge.budget = 7 << 30;
    assert_eq!(
        prefill_window(&c, &direct),
        prefill_window(&c, &without_bank_charge)
    );
    assert_eq!(prefill_window(&c, &direct), super::MAX_PREFILL_CHUNK);
}

#[test]
fn flash_next_scheduled_state_budget_holds_one_long_mtp_owner() {
    let c = flash_next_config();
    assert_eq!(c.effective_context_limit(), 32_768);
    assert_eq!(row_state_bytes(&c), 169_967_616);
    assert_eq!(draft_kv_bytes_per_token(&c), 4096);
    let bytes_per_block = mlx_paged_attn::profile::bytes_per_block(
        12,
        2,
        256,
        32,
        mlx_paged_attn::metal::MetalDtype::BFloat16,
    )
    .unwrap();
    assert_eq!(bytes_per_block, 786_432);
    let blocks = (1024u64 << 20) / bytes_per_block;
    assert_eq!(blocks, 1365);
    let budget = scheduled_state_budget(blocks * bytes_per_block, prefill_scratch(&c, 1024));
    let row = row_state_bytes(&c);
    let mtp_16k = row + draft_state_bytes(&c, 18_754 + 255);
    let mtp_32k = row + draft_state_bytes(&c, 32_488 + 255);
    assert!(
        mtp_16k <= mtp_32k && mtp_32k <= budget,
        "one 32k MTP owner ({mtp_32k} B) must fit the {budget} B working budget"
    );
    assert!(
        2 * mtp_32k > budget,
        "two 32k MTP owners ({} B) must wait for a row to be reclaimed from the {budget} B budget",
        2 * mtp_32k
    );
    assert!(4 * row <= budget, "four AR rows must fit");
    // Two 16k MTP owners queued in one drain pass: the page pool admits both,
    // and the second owner sees two rows plus its own draft, which fits. Only
    // the first owner's draft reservation, not materialized before its first
    // prefill chunk, makes the second defer.
    let draft_16k = draft_state_bytes(&c, 18_754 + 255);
    let blocks_16k = (18_754u64 + 255).div_ceil(32);
    assert_eq!(blocks_16k, 595);
    assert!(2 * blocks_16k <= blocks);
    assert!(2 * row + draft_16k <= budget);
    assert!(
        2 * row + 2 * draft_16k > budget,
        "two 16k MTP owners ({} B) must not both pass admission against the {budget} B budget",
        2 * row + 2 * draft_16k
    );
}

/// Two Flash Next owners parked right after a 1024-row prefill slice, with
/// 1023 trusted rows and the seed waiting and nothing drafted: each keeps
/// 1024 BF16 hidden rows of 20,480 B beside its recurrent row and no draft
/// K/V. A budget sized to the draft-K/V figure for those rows admits a third
/// owner only after reclaim.
/// MUTATION: charge waiting rows as draft K/V in
/// `draft_owner_bytes`; `held` falls to the K/V figure, the third owner fits
/// and the AfterReclaim assertion fails.
#[test]
fn flash_next_parked_owners_charge_their_waiting_hidden_rows() {
    use crate::engine::hybrid_scheduler::{
        ScheduledStateCapacity, ScheduledStateFit, scheduled_state_fit,
    };
    let c = flash_next_config();
    assert_eq!(draft_hidden_row_bytes(&c), 20_480);
    let row = row_state_bytes(&c);
    let parked = draft_owner_bytes(&c, 0, 1024, false);
    assert_eq!(parked, 1024 * 20_480);
    let kv_figure = 1024 * draft_kv_bytes_per_token(&c);
    assert!(parked > kv_figure);
    assert_eq!(
        draft_owner_bytes(&c, 7, 1, true),
        4 * row + 7 * 4096 + 20_480,
        "cached rows pay K/V, the seed pays its hidden row, a tracked owner its snapshots"
    );
    let candidate = row + draft_state_bytes(&c, 256);
    let budget = 2 * (row + kv_figure) + candidate;
    let held = 2 * (row + parked);
    assert_eq!(
        scheduled_state_fit(Some(ScheduledStateCapacity { budget, held }), 0, candidate),
        ScheduledStateFit::AfterReclaim,
        "the retained hidden rows overcommit a budget sized to their K/V figure"
    );
    assert_eq!(
        scheduled_state_fit(
            Some(ScheduledStateCapacity {
                budget,
                held: 2 * (row + kv_figure),
            }),
            0,
            candidate
        ),
        ScheduledStateFit::Fits
    );
}
