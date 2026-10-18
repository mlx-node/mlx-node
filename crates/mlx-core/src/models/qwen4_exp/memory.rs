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
    })
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

pub(super) fn check_growth_headroom() -> Result<()> {
    if available_memory()? < system_reserve(physical_memory()?) {
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

/// Read `kern.memorystatus_vm_pressure_level`: levels below 2 are normal.
#[cfg(target_os = "macos")]
fn read_pressure_level() -> Result<i32> {
    let mut pressure = 0i32;
    let mut size = std::mem::size_of_val(&pressure);
    if unsafe {
        libc::sysctlbyname(
            c"kern.memorystatus_vm_pressure_level".as_ptr(),
            (&mut pressure as *mut i32).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    } != 0
    {
        return Err(Error::from_reason(
            "Cannot read macOS memory pressure for Qwen4 bootstrap",
        ));
    }
    Ok(pressure)
}

/// Wait out a transient pressure spike before failing a load. The ignore
/// opt-out is checked before the first read so it bypasses the gate entirely,
/// including a sysctl failure. Injection points keep the timing testable.
#[cfg(any(target_os = "macos", test))]
fn wait_for_pressure(
    mut read: impl FnMut() -> Result<i32>,
    ignore_flag: impl Fn() -> bool,
    mut sleep: impl FnMut(std::time::Duration),
) -> Result<()> {
    if ignore_flag() {
        return Ok(());
    }
    let mut retries = 0;
    loop {
        match read() {
            Ok(level) if level < 2 => return Ok(()),
            _ if retries < 12 => {
                retries += 1;
                sleep(std::time::Duration::from_millis(250));
            }
            _ => {
                return Err(Error::from_reason(
                    "Qwen4 bootstrap requires normal, readable macOS memory pressure",
                ));
            }
        }
    }
}

/// Re-check the SSD read safety limit after freeing the allocator freelist.
/// A freelist spike is transient; real pressure still refuses after the same
/// bounded re-poll the bootstrap pressure gate uses.
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

#[cfg(target_os = "macos")]
fn available_memory() -> Result<u64> {
    unsafe extern "C" {
        fn mach_host_self() -> libc::mach_port_t;
        static mach_task_self_: libc::mach_port_t;
        fn mach_port_deallocate(
            task: libc::mach_port_t,
            name: libc::mach_port_t,
        ) -> libc::kern_return_t;
    }
    wait_for_pressure(
        read_pressure_level,
        || runtime_flags::is_one(c"MLX_QWEN4_IGNORE_MEMORY_PRESSURE"),
        std::thread::sleep,
    )?;
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
    // Exclude anonymous inactive memory and the compressor: growing the model
    // cache must not assume that the OS can compress/swap the user's apps.
    Ok((u64::from(stats.free_count) + u64::from(stats.external_page_count)) * page_size as u64)
}

#[cfg(not(target_os = "macos"))]
fn available_memory() -> Result<u64> {
    Err(Error::from_reason(
        "Qwen4 larger-cache admission is currently available only on macOS",
    ))
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
    fn pressure_gate_waits_out_a_transient_spike() {
        use std::sync::{Arc, Mutex};
        let levels = Arc::new(Mutex::new(vec![2, 1].into_iter()));
        let sleeps = Arc::new(Mutex::new(Vec::new()));
        let read = {
            let levels = Arc::clone(&levels);
            move || {
                levels
                    .lock()
                    .unwrap()
                    .next()
                    .ok_or_else(|| Error::from_reason("unexpected extra read"))
            }
        };
        let sleep = {
            let sleeps = Arc::clone(&sleeps);
            move |d| sleeps.lock().unwrap().push(d)
        };
        super::wait_for_pressure(read, || false, sleep).unwrap();
        assert_eq!(
            *sleeps.lock().unwrap(),
            vec![std::time::Duration::from_millis(250)]
        );
    }

    #[test]
    fn pressure_gate_gives_up_after_bounded_retries() {
        let mut sleeps = 0;
        let err = super::wait_for_pressure(|| Ok(3), || false, |_| sleeps += 1).unwrap_err();
        assert_eq!(sleeps, 12);
        assert!(
            err.reason
                .contains("requires normal, readable macOS memory pressure")
        );
    }

    #[test]
    fn pressure_gate_ignore_bypasses_unreadable_sysctl() {
        let mut reads = 0;
        super::wait_for_pressure(
            || {
                reads += 1;
                Err(Error::from_reason("sysctl unavailable"))
            },
            || true,
            |_| panic!("ignore must not wait"),
        )
        .unwrap();
        assert_eq!(reads, 0);
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
/// `tokens`.
pub(super) fn draft_state_bytes(c: &super::config::Config, tokens: u64) -> u64 {
    4 * row_state_bytes(c) + tokens * draft_kv_bytes_per_token(c)
}

/// What the working allowance leaves for scheduled rows and draft state once
/// the page pool and one prefill window's scratch have their share.
pub(super) fn scheduled_state_budget(pool_bytes: u64, prefill_scratch: u64) -> u64 {
    WORKING_BYTES
        .saturating_sub(pool_bytes)
        .saturating_sub(prefill_scratch)
}

pub(super) fn prefill_window(c: &super::config::Config, plan: &Plan) -> Result<usize> {
    let live = match (plan.available_bytes, plan.physical_bytes) {
        (Some(a), Some(p)) => a.saturating_sub(plan.budget + system_reserve(p) + (2 << 30)),
        _ => plan.budget / 16,
    };
    let allowance = (plan.budget / 16).min(2 << 30).min(live);
    let mut window = super::MAX_PREFILL_CHUNK;
    while window > 1 && prefill_scratch(c, window) > allowance {
        window /= 2;
    }
    if let Some(value) = std::env::var_os("MLX_QWEN4_PREFILL_CHUNK") {
        let requested = value
            .to_str()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|n| n.is_power_of_two() && *n <= super::MAX_PREFILL_CHUNK)
            .ok_or_else(|| {
                Error::from_reason("MLX_QWEN4_PREFILL_CHUNK must be a power of two from 1 to 1024")
            })?;
        if requested > window {
            return Err(Error::from_reason(format!(
                "Qwen4 prefill window {requested} exceeds bootstrap scratch allowance; admitted {window}"
            )));
        }
        window = requested;
    }
    Ok(window)
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

/// Flash Next geometry (48 layers, 12 full attention, K/V 2 x 256, 32,768
/// context): one 32k MTP owner fits the working budget beside the 1 GiB page
/// pool and the 1024-token prefill window, two do not, four AR rows do.
/// MUTATION: stop subtracting `prefill_scratch` in `scheduled_state_budget`;
/// two 32k MTP owners then fit and the `2 * mtp_32k > budget` assertion fails.
/// MUTATION: drop the four verify snapshots from `draft_state_bytes`; the 16k
/// pair then fits and the `2 * row + 2 * draft_16k > budget` assertion fails.
#[test]
fn flash_next_scheduled_state_budget_holds_one_long_mtp_owner() {
    let c = super::config::Config::parse(&serde_json::json!({
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
    .unwrap();
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
