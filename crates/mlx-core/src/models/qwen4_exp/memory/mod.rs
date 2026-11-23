//! Gate each weight allocation against live system headroom; the memory plan
//! sizes the store. Preserve room for other applications and transient work.

#[cfg(test)]
mod tests;

pub(super) use crate::utils::host::HostMemory;
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
    /// The host read behind `live_non_reclaimable` and `bank_budget`.
    pub host: Option<HostMemory>,
    /// What the plan holds beside the experts when the residency was chosen.
    pub anonymous_need: u64,
    /// This process's MLX active plus cache bytes when the residency was
    /// chosen.
    pub self_mlx_bytes: u64,
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

#[cfg(test)]
pub(super) const FIXTURE_HOST_ENV: &str = "MLX_QWEN4_FIXTURE_HOST";

#[cfg(test)]
thread_local! {
    static TEST_HOST: std::cell::Cell<Option<HostMemory>> =
        const { std::cell::Cell::new(None) };
    /// Anonymous bytes every read of the scoped host adds for the next read.
    static TEST_HOST_DRIFT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(super) struct TestHostGuard(Option<HostMemory>, u64);

#[cfg(test)]
impl Drop for TestHostGuard {
    fn drop(&mut self) {
        TEST_HOST.with(|host| host.set(self.0));
        TEST_HOST_DRIFT.with(|drift| drift.set(self.1));
    }
}

/// A constrained host whose tiny direct fixtures fit the unchanged reserves.
#[cfg(test)]
pub(super) fn fixture_host() -> HostMemory {
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
    scoped_drifting_test_host(host, 0)
}

/// A scoped host whose anonymous pages grow by `drift` after every read, so
/// two reads never agree.
#[cfg(test)]
pub(super) fn scoped_drifting_test_host(host: HostMemory, drift: u64) -> TestHostGuard {
    TestHostGuard(
        TEST_HOST.with(|current| current.replace(Some(host))),
        TEST_HOST_DRIFT.with(|current| current.replace(drift)),
    )
}

#[cfg(test)]
pub(super) fn scoped_fixture_host() -> TestHostGuard {
    scoped_test_host(fixture_host())
}

#[cfg(test)]
fn test_host_override() -> Option<HostMemory> {
    TEST_HOST
        .with(|host| {
            let current = host.get()?;
            let drift = TEST_HOST_DRIFT.with(std::cell::Cell::get);
            if drift > 0 {
                host.set(Some(HostMemory {
                    anonymous: current.anonymous.saturating_add(drift),
                    ..current
                }));
            }
            Some(current)
        })
        .or_else(|| std::env::var_os(FIXTURE_HOST_ENV).map(|_| fixture_host()))
}

/// The device a load plans against; a fixture-host child gets the working set
/// of [`fixture_host`], not the runner's.
pub(super) fn live_memory() -> Option<crate::engine::memory_plan::LiveMemory> {
    let live = if let Some(live) = crate::engine::memory_plan::live_memory() {
        live
    } else if !crate::accelerator::host_has_metal() && crate::accelerator::host_has_cuda() {
        #[cfg(test)]
        if crate::engine::memory_plan::LIVE_MEMORY_UNREADABLE.with(std::cell::Cell::get) {
            return None;
        }
        let working_set_bytes = crate::accelerator::caps().total_device_bytes;
        if working_set_bytes == 0 {
            return None;
        }
        let mut active_bytes = 0u64;
        // SAFETY: the probe writes one u64 through a valid pointer and reports
        // failure through its return code.
        if unsafe { mlx_sys::mlx_get_active_memory(&mut active_bytes) } != 0 {
            return None;
        }
        crate::engine::memory_plan::LiveMemory {
            working_set_bytes,
            active_bytes,
            registered_pool_bytes: crate::cache_limit::coordinator().registered_pool_bytes(),
        }
    } else {
        return None;
    };
    #[cfg(test)]
    if std::env::var_os(FIXTURE_HOST_ENV).is_some() {
        return Some(crate::engine::memory_plan::LiveMemory {
            working_set_bytes: fixture_host().physical / 4 * 3,
            ..live
        });
    }
    Some(live)
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
    bank_gate_beside(host, 0, anonymous_need, margin)
}

/// [`bank_gate`] on a host that already holds `own` bytes of this model:
/// they leave the host's non-reclaimable number first, so the gate reads
/// the host as a load that held none of them did.
pub(super) fn bank_gate_beside(
    host: &HostMemory,
    own: u64,
    anonymous_need: u64,
    margin: u64,
) -> BankGate {
    let non_reclaimable = host.non_reclaimable().saturating_sub(own);
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
/// plan's sizes. Fixtures and unsupported platforms have no snapshot.
pub(super) fn host_snapshot(plan: &mut Plan) -> Result<()> {
    if !cfg!(any(target_os = "macos", target_os = "windows")) {
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
    direct_gate_beside(plan, 0, anonymous_need, bank_bytes)
}

/// [`direct_gate`] read while this model holds `own` bytes of the host
/// (see [`bank_gate_beside`]).
pub(super) fn direct_gate_beside(
    plan: &mut Plan,
    own: u64,
    anonymous_need: u64,
    bank_bytes: u64,
) -> Result<()> {
    let gate = match plan.physical_bytes {
        Some(_) => {
            let host = host_memory()?;
            plan.available_bytes = Some(host.available());
            plan.physical_bytes = Some(host.physical);
            plan.host = Some(host);
            bank_gate_beside(&host, own, anonymous_need, DIRECT_MARGIN)
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
    // On CUDA, `clear_cache()` calls `cudaStreamSynchronize + cudaMemPoolTrimTo(0)`
    // (~70 ms stall per hit), while MLX's CudaAllocator already bounds its pool
    // automatically on allocation pressure. Keep Metal's wired freelist trim untouched.
    if crate::accelerator::cuda_ops_available() {
        return;
    }
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

fn physical_memory() -> Result<u64> {
    #[cfg(test)]
    if let Some(host) = test_host_override() {
        return Ok(host.physical);
    }
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    {
        crate::utils::host::physical_memory()
            .ok_or_else(|| Error::from_reason("Cannot read physical memory for Qwen4 residency"))
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        Ok(crate::utils::host::physical_memory().unwrap_or(0))
    }
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

pub(super) fn host_memory() -> Result<HostMemory> {
    #[cfg(test)]
    if let Some(host) = test_host_override() {
        return Ok(host);
    }
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    {
        crate::utils::host::host_memory().ok_or_else(|| {
            Error::from_reason("Cannot establish Qwen4 larger-cache memory headroom")
        })
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        Err(Error::from_reason(
            "Qwen4 larger-cache admission is currently available only on macOS and Windows",
        ))
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

/// What a layer-major group of `windows` windows of `rows` rows holds beyond
/// one pass's scratch: every window's residual stream across layers and,
/// inside a union layer from the first window's attention to the last
/// window's injection, each window's attention output, MLP input, injection
/// gate, routes, scores, routed rows and their sum.
pub(crate) fn group_scratch(c: &super::config::Config, windows: usize, rows: usize) -> u64 {
    let h = c.hidden_size as u64;
    let hc = c.hc_count as u64;
    let top = c.num_experts_per_tok as u64;
    let per_row = 2 * hc * h * 2 + hc * 4 + 2 * h * 2 + top * (4 + 4 + 4) + top * h * 2;
    per_row
        .saturating_mul(windows as u64)
        .saturating_mul(rows as u64)
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

/// The draft layer's pooled BF16 indexer keys per token.
pub(super) fn draft_index_bytes_per_token(c: &super::config::Config) -> u64 {
    (c.indexer_head_dim * 2 / c.indexer_compress_ratio.max(1)) as u64
}

/// The rows a cold prefill of `rows` tokens has in its draft cache: draft
/// passes drain whole `chunk`-row windows of the trusted rows, which end one
/// row before the frontier.
pub(super) fn cached_draft_rows(rows: u64, chunk: u64) -> u64 {
    let trusted = rows.saturating_sub(1);
    trusted - trusted % chunk.max(1)
}

/// The draft part of a state at `row`: the draft K/V and indexer keys of its
/// cached rows and the hidden rows waiting beside them (the seed included).
pub(super) fn draft_cell_bytes(c: &super::config::Config, row: u64, chunk: u64) -> u64 {
    let cached = cached_draft_rows(row, chunk);
    cached * (draft_kv_bytes_per_token(c) + draft_index_bytes_per_token(c))
        + (row - cached) * draft_hidden_row_bytes(c)
}

/// One stored state at `row` of a load whose prefill window is `chunk`, its
/// draft part with the MTP head.
pub(super) fn state_bytes(c: &super::config::Config, row: u64, chunk: u64, has_mtp: bool) -> u64 {
    state_cell_bytes(c)
        + if has_mtp {
            draft_cell_bytes(c, row, chunk)
        } else {
            0
        }
}

/// What an MTP owner holds now: draft K/V and indexer keys for the rows in
/// its cache, the hidden rows still waiting (trusted rows and the seed), and
/// the verify snapshots while it is inside an MTP turn. A waiting row has no
/// K/V yet.
pub(super) fn draft_owner_bytes(
    c: &super::config::Config,
    cached_rows: u64,
    hidden_rows: u64,
    speculative: bool,
) -> u64 {
    let snapshots = if speculative {
        4 * row_state_bytes(c)
    } else {
        0
    };
    snapshots
        + cached_rows * (draft_kv_bytes_per_token(c) + draft_index_bytes_per_token(c))
        + hidden_rows * draft_hidden_row_bytes(c)
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

/// One stored state at a grid line: the cell without indexer keys waiting
/// for a block (none wait at a multiple of the compress ratio).
pub(super) fn state_cell_bytes(c: &super::config::Config) -> u64 {
    let [conv, recurrent, ple, _] = row_state_parts(c);
    conv + recurrent + ple
}

/// An encoded state's header (shapes, dtypes, index blocks) at most.
const STATE_HEADER_BYTES: u64 = 1 << 16;

/// One state as the disk tier encodes it: the cell, the draft's waiting rows
/// and seed with the MTP head (its segments are objects of their own).
pub(super) fn encoded_state_bytes(c: &super::config::Config, has_mtp: bool) -> u64 {
    let waiting = if has_mtp {
        (super::MAX_PREFILL_CHUNK as u64 + 1) * draft_hidden_row_bytes(c)
    } else {
        0
    };
    state_cell_bytes(c) + waiting + STATE_HEADER_BYTES
}

/// Host bytes the disk tier holds on a load with it on: the one staging
/// buffer every state record is written and read through; one array's host
/// copy while the model thread encodes a state; the parker's queue of block
/// and segment copies (one copy past its bound), the writer's queue behind
/// it and the copy the writer encodes.
pub(super) fn disk_staging_bytes(c: &super::config::Config, has_mtp: bool) -> u64 {
    let window = super::MAX_PREFILL_CHUNK as u64;
    let linear = (0..c.num_hidden_layers).filter(|&i| c.linear(i)).count() as u64;
    let [conv, recurrent, ple, _] = row_state_parts(c);
    let mut largest = (conv / linear.max(1))
        .max(recurrent / linear.max(1))
        .max(ple);
    let mut item = super::load_plan::flash_next_layout(c).page_bytes();
    if has_mtp {
        largest = largest.max(window * draft_hidden_row_bytes(c));
        item = item.max(window * (draft_kv_bytes_per_token(c) + draft_index_bytes_per_token(c)));
    }
    encoded_state_bytes(c, has_mtp)
        + largest
        + super::prefix_disk::QUEUE_BYTES
        + (mlx_paged_attn::DEFAULT_QUEUE_DEPTH as u64 + 3) * item
}

/// What stored states may hold: the dynamic budget less one pass's transient
/// and the pool, the pool charged as it is and at its largest, so a state
/// never takes room the pool can still grow into, less what the lanes'
/// rows and draft state hold (`held`).
pub(super) fn state_room(dynamic: u64, transient: u64, pool: &PoolCharge, held: u64) -> u64 {
    let largest = PoolCharge {
        total_blocks: pool.max_blocks.max(pool.total_blocks),
        max_blocks: pool.max_blocks.max(pool.total_blocks),
        physical_block_bytes: pool.physical_block_bytes,
        growth_copy_bytes: 0,
    };
    scheduled_state_budget(dynamic, transient, pool)
        .min(scheduled_state_budget(dynamic, transient, &largest))
        .saturating_sub(held)
}

/// Whether a state of `bytes` leaves the host the headroom a pass checks
/// for and keeps MLX's active memory within the SSD read limit: the plan's
/// room is a ledger, these are the live readings the passes refuse on.
pub(super) fn state_fits_live(
    host: &HostMemory,
    active: f64,
    limit: Option<u64>,
    bytes: u64,
    direct_gate: bool,
) -> bool {
    let floor = if direct_gate {
        DIRECT_TURN_FLOOR
    } else {
        system_reserve(host.physical)
    };
    host.available() >= floor.saturating_add(bytes)
        && limit.is_none_or(|limit| active + bytes as f64 <= limit as f64)
}

/// The rows of a prefill pass that ends at up to `context` tokens. Every
/// store prefills the `MAX_PREFILL_CHUNK` rows the memory plan prices (Splash
/// sizes its arenas for one fixed chunk; live free memory never cuts it). A
/// store whose host gate refused the whole bank set holds its slots beside the
/// `reserved` scratch that gate priced at the allowance context, so a longer
/// sequence halves the pass until its scratch fits there.
pub(super) fn prefill_window(
    c: &super::config::Config,
    reserved: Option<u64>,
    context: usize,
) -> usize {
    let Some(reserved) = reserved else {
        return super::MAX_PREFILL_CHUNK;
    };
    let mut window = super::MAX_PREFILL_CHUNK;
    while window > 1 && prefill_scratch(c, window, context) > reserved {
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
