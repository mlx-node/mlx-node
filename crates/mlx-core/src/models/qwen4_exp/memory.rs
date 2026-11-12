//! Gate each weight allocation against live system headroom; the memory plan
//! sizes the store. Preserve room for other applications and transient work.
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

#[cfg(test)]
pub(super) const FIXTURE_HOST_ENV: &str = "MLX_QWEN4_FIXTURE_HOST";

#[cfg(test)]
thread_local! {
    static TEST_HOST: std::cell::Cell<Option<HostMemory>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(super) struct TestHostGuard(Option<HostMemory>);

#[cfg(test)]
impl Drop for TestHostGuard {
    fn drop(&mut self) {
        TEST_HOST.with(|host| host.set(self.0));
    }
}

/// A constrained host whose tiny direct fixtures fit the unchanged reserves.
#[cfg(test)]
fn fixture_host() -> HostMemory {
    HostMemory {
        physical: 16 << 30,
        free: 8 << 30,
        file: 4 << 30,
        anonymous: 1 << 30,
        purgeable: 0,
        wired: 2 << 30,
        compressor: 1 << 30,
    }
}

#[cfg(test)]
pub(super) fn scoped_test_host(host: HostMemory) -> TestHostGuard {
    TestHostGuard(TEST_HOST.with(|current| current.replace(Some(host))))
}

#[cfg(test)]
pub(super) fn scoped_fixture_host() -> TestHostGuard {
    scoped_test_host(fixture_host())
}

#[cfg(test)]
fn test_host_override() -> Option<HostMemory> {
    TEST_HOST
        .with(|host| host.get())
        .or_else(|| std::env::var_os(FIXTURE_HOST_ENV).map(|_| fixture_host()))
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

/// Record the live host in `plan`: the gates that follow read it, never the
/// plan's sizes. Fixtures and other platforms have no snapshot.
pub(super) fn host_snapshot(plan: &mut Plan) -> Result<()> {
    if !cfg!(target_os = "macos") {
        return Ok(());
    }
    let host = host_memory()?;
    plan.available_bytes = Some(host.available());
    plan.physical_bytes = Some(host.physical);
    Ok(())
}

/// The host gate of a whole direct bank set the store plan chose: the host
/// must hold `bank_bytes` beside the plan's `anonymous_need` (the fixed
/// weights, the draft banks, the row cache and the KV allowance). A
/// refusal leaves what the host admits in `plan.bank_budget`. Without a host
/// snapshot the gate admits the banks.
pub(super) fn direct_gate(plan: &mut Plan, anonymous_need: u64, bank_bytes: u64) -> Result<()> {
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
    #[cfg(test)]
    if let Some(host) = test_host_override() {
        return Ok(host.physical);
    }
    crate::utils::host::physical_memory()
        .ok_or_else(|| Error::from_reason("Cannot read physical memory for Qwen4 residency"))
}
#[cfg(not(target_os = "macos"))]
fn physical_memory() -> Result<u64> {
    #[cfg(test)]
    if let Some(host) = test_host_override() {
        return Ok(host.physical);
    }
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

/// Re-check the SSD read safety limit after freeing the allocator freelist:
/// MLX's active bytes within `ceiling`, the memory plan's ceiling less the
/// registered KV pools (MLX's active counter does not see them). A freelist
/// spike is transient; real pressure still refuses after a bounded re-poll.
pub(super) fn check_active_memory_budget(
    ceiling: u64,
    mut active: impl FnMut() -> f64,
    mut reclaim: impl FnMut(),
    mut sleep: impl FnMut(std::time::Duration),
) -> Result<()> {
    if active() <= ceiling as f64 {
        return Ok(());
    }
    reclaim();
    for _ in 0..3 {
        sleep(std::time::Duration::from_millis(100));
        if active() <= ceiling as f64 {
            return Ok(());
        }
    }
    Err(Error::from_reason(
        "qwen4_exp SSD safety limit: active MLX memory exceeds the memory plan's ceiling less the KV pools",
    ))
}

fn available_memory() -> Result<u64> {
    Ok(host_memory()?.available())
}

#[cfg(target_os = "macos")]
pub(super) fn host_memory() -> Result<HostMemory> {
    #[cfg(test)]
    if let Some(host) = test_host_override() {
        return Ok(host);
    }
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
    #[cfg(test)]
    if let Some(host) = test_host_override() {
        return Ok(host);
    }
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
    fn admission_keeps_headroom_and_rejects_overflow() {
        assert!(fits_with_headroom(16 << 30, 32 << 30, 16 << 30));
        assert!(!fits_with_headroom(16 << 30, (32 << 30) - 1, 16 << 30));
        assert!(!fits_with_headroom(32 << 30, 40 << 30, 16 << 30));
        assert!(!fits_with_headroom(u64::MAX, u64::MAX, 16 << 30));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn constrained_fixture_host_preserves_real_memory_gates() {
        let busy = coherent(HostMemory {
            physical: 8 << 30,
            free: 4 << 30,
            file: 0,
            anonymous: 1 << 30,
            purgeable: 0,
            wired: 2 << 30,
            compressor: 1 << 30,
        });
        let fixture = coherent(fixture_host());
        let tiny = 256 << 20;
        let _fixture = scoped_test_host(fixture);
        let mut planned = plan_without_snapshot();
        host_snapshot(&mut planned).unwrap();
        assert_eq!(planned.available_bytes, Some(12 << 30));
        assert_eq!(planned.physical_bytes, Some(16 << 30));
        let mut direct = planned.clone();
        direct_gate(&mut direct, 1 << 30, tiny).unwrap();
        assert_eq!(direct.bank_budget, 7 << 30);
        admit_direct(tiny).unwrap();

        {
            let _busy = scoped_test_host(busy);
            let mut refused = plan_without_snapshot();
            refused.physical_bytes = Some(busy.physical);
            let error = direct_gate(&mut refused, 1 << 30, tiny).unwrap_err();
            assert!(error.reason.contains("host leaves 0.0 GiB"));
            let error = admit_direct(tiny).unwrap_err();
            assert!(error.reason.contains("host leaves 0.0 GiB"));
        }
        admit_direct(tiny).unwrap();
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

/// A layer's prompt scratch for a chunk of `tokens` rows of a sequence of up
/// to `context` tokens, from checkpoint dimensions: the memory plan prices it
/// as the lane's transient.
pub(crate) fn prefill_scratch(c: &super::config::Config, tokens: usize, context: usize) -> u64 {
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
    let context = context as u64;
    let blocks = context / c.indexer_compress_ratio.max(1) as u64;
    let qsa = (t * c.indexer_n_heads as u64 * blocks * 4 * 2)
        .saturating_add(t * blocks * 4 * 6)
        .saturating_add(2 * c.num_key_value_heads as u64 * context * c.head_dim as u64 * 2);
    per_token.saturating_mul(t).saturating_add(window.max(qsa))
}

/// The byte size of the activation dtype the recurrent rows are kept in: the
/// compiled GDN path requires bf16 projections and decoded rows are bf16.
const STATE_ACTIVATION_BYTES: usize = 2;

/// The parts of one sequence's state outside the page pool (Splash
/// `cellBytes` without its 16 KiB alignment): per linear layer the
/// convolution history of `k - 1` qkv rows and the F32 recurrent matrices,
/// the PLE history of `(k - 1) * ngram` rows, and the indexer keys that have
/// not filled a block yet (Splash `indexerPendingBytes`). Pooled index keys
/// live in the KV page.
pub(crate) fn row_state_parts(c: &super::config::Config) -> [u64; 4] {
    let linear = (0..c.num_hidden_layers).filter(|&i| c.linear(i)).count();
    let full = c.num_hidden_layers - linear;
    let qkv = 2 * c.linear_num_key_heads * c.linear_key_head_dim
        + c.linear_num_value_heads * c.linear_value_head_dim;
    let conv = linear * c.linear_conv_kernel_dim.saturating_sub(1) * qkv * STATE_ACTIVATION_BYTES;
    let recurrent =
        linear * c.linear_num_value_heads * c.linear_key_head_dim * c.linear_value_head_dim * 4;
    let ple = c.ple_layer_ids.len()
        * c.ple_conv_kernel_size.saturating_sub(1)
        * c.ngram_size
        * c.hc_count
        * c.hidden_size
        * STATE_ACTIVATION_BYTES;
    let pending_keys = full
        * c.indexer_compress_ratio.saturating_sub(1)
        * c.indexer_head_dim
        * STATE_ACTIVATION_BYTES;
    [conv, recurrent, ple, pending_keys].map(|bytes| bytes as u64)
}

pub(crate) fn row_state_bytes(c: &super::config::Config) -> u64 {
    row_state_parts(c).iter().sum()
}

/// F32 draft-layer K/V per token.
pub(super) fn draft_kv_bytes_per_token(c: &super::config::Config) -> u64 {
    (c.num_key_value_heads * c.head_dim * 8) as u64
}

/// Verify snapshots of up to four accepted frontiers plus the draft K/V of
/// `tokens`: what an owner holds once every token is in its draft cache.
pub(crate) fn draft_state_bytes(c: &super::config::Config, tokens: u64) -> u64 {
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

/// The pool's share of the dynamic budget the scheduled state competes
/// with: every physical block the pool holds (it never shrinks; a block is
/// the whole priced page, index keys included), and, while the pool may
/// still grow, the one-layer copy a growth makes of it.
pub(super) struct PoolCharge {
    pub total_blocks: u64,
    pub max_blocks: u64,
    pub physical_block_bytes: u64,
    pub growth_copy_bytes: u64,
}

impl PoolCharge {
    pub(super) fn bytes(&self) -> u64 {
        let growth = if self.total_blocks < self.max_blocks {
            self.growth_copy_bytes
        } else {
            0
        };
        self.total_blocks
            .saturating_mul(self.physical_block_bytes)
            .saturating_add(growth)
    }
}

/// What the plan's dynamic budget leaves for scheduled rows and draft state
/// once one prefill chunk's transient and the pool have their share (Splash
/// lanes share one dynamic budget).
pub(super) fn scheduled_state_budget(dynamic: u64, transient: u64, pool: &PoolCharge) -> u64 {
    dynamic
        .saturating_sub(transient)
        .saturating_sub(pool.bytes())
}

/// The prefill chunk the live host admits for a sequence of up to `context`
/// tokens: the plan prices `MAX_PREFILL_CHUNK`, and only a host that cannot
/// hold its scratch beside the store halves it.
pub(super) fn prefill_window(c: &super::config::Config, plan: &Plan, context: usize) -> usize {
    let anonymous_budget = plan.budget.saturating_sub(plan.direct_bytes);
    let allowance = match (plan.available_bytes, plan.physical_bytes) {
        (Some(a), Some(p)) => {
            let reserve = if plan.partial_direct_gate {
                DIRECT_TURN_FLOOR
            } else {
                system_reserve(p) + (2 << 30)
            };
            a.saturating_sub(anonymous_budget + reserve)
        }
        _ => u64::MAX,
    };
    let mut window = super::MAX_PREFILL_CHUNK;
    while window > 1 && prefill_scratch(c, window, context) > allowance {
        window /= 2;
    }
    window
}

/// The released Flash Next geometry (Qwen3.8 Flash Next).
#[cfg(test)]
pub(crate) fn flash_next_config() -> super::config::Config {
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
    let context = 65_536;
    assert_eq!(
        prefill_window(&c, &direct, context),
        prefill_window(&c, &without_bank_charge, context)
    );
    assert_eq!(
        prefill_window(&c, &direct, context),
        super::MAX_PREFILL_CHUNK
    );
}

/// The Flash Next cell on its storage: 36 linear layers of 3 bf16 conv
/// rows of 10,240 (2,211,840) and F32 recurrent matrices (113,246,208), 9
/// bf16 PLE history rows of 10,240 (184,320), and 3 pending bf16 index keys
/// of 128 in 12 attention layers (9,216). Splash `cellBytes` aligns each part
/// to 16 KiB: 115,802,112. Mutation: the F32 formula with 32,768 tokens of
/// pooled index keys in `row_state_bytes` → red (169,967,616).
#[test]
fn flash_next_cell_matches_the_storage() {
    let c = flash_next_config();
    assert_eq!(
        row_state_parts(&c),
        [2_211_840, 113_246_208, 184_320, 9_216]
    );
    assert_eq!(row_state_bytes(&c), 115_651_584);
    const SPLASH_FLASH_NEXT_CELL: u64 = 115_802_112;
    assert_eq!(SPLASH_FLASH_NEXT_CELL - row_state_bytes(&c), 150_528);
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
