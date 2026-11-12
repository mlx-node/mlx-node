//! One memory plan per load, in the shape of Splash `EngineMemoryPlan`
//! (`evaluateEngineMemoryPlan`): a hard budget below the Metal working set,
//! the fixed bytes the load measured, and per lane the longest window whose
//! state, transients and KV in whole extents fit the dynamic rest.

use std::fmt::Write as _;

use crate::array::DType;
use crate::transformer::KvFormat;

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

/// Splash `kPipelineReserveBytes`.
pub(crate) const PIPELINE_RESERVE_BYTES: u64 = 256 * MIB;
/// Splash `kRuntimeOverheadReserveBytes`.
pub(crate) const RUNTIME_OVERHEAD_RESERVE_BYTES: u64 = 512 * MIB;
/// Splash `ExecutionLimits::speculativeScratchTokens`.
pub(crate) const SPECULATIVE_SCRATCH_TOKENS: u64 = 7;
/// Tokens of one KV page (Splash `kv::kPageTokens`).
pub(crate) const KV_PAGE_TOKENS: u64 = 32;
/// Splash `ExecutionLimits::warmupKvPages`.
const WARMUP_KV_PAGES: u64 = 64;
/// Splash `kExtentRegionAlignmentBytes`.
const EXTENT_REGION_ALIGNMENT_BYTES: u64 = 64 * 1024;
/// Splash `kAllocationExtentTargetBytes`.
pub(crate) const EXTENT_TARGET_BYTES: u64 = 128 * MIB;
/// Splash `SPLASH_ATTENTION_BLOCK_TOKENS`: the tokens one block-index row
/// covers.
const ATTENTION_BLOCK_TOKENS: u64 = 4;
/// Splash `kResidentAllowanceTokens`: the context a target keeps resident
/// before its streamed stores take the rest of the budget.
pub(crate) const RESIDENT_ALLOWANCE_TOKENS: u32 = 65_536;
/// Splash `kTableShareTokens`: the prompt whose table rows an automatic row
/// cache holds.
pub(crate) const TABLE_SHARE_TOKENS: u64 = 128 * 1024;

/// The share of the working set an unguarded budget keeps, less
/// [`WORKING_SET_RESERVE_BYTES`].
const WORKING_SET_PERCENT: u64 = 85;
const WORKING_SET_RESERVE_BYTES: u64 = GIB;
/// Splash `EngineMemoryPolicy::workingSetMarginBytes`: `max(1 GiB, 2%)`.
const LEDGER_MARGIN_PERCENT: u64 = 2;
const LEDGER_MARGIN_FLOOR_BYTES: u64 = GIB;

/// Which margin a plan keeps out of the working set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Margin {
    /// Splash's: the memory governor keeps the model inside its budget.
    Ledger,
    /// 15% + 1 GiB, for a model no governor guards.
    Unguarded,
}

/// The bytes of the working set kept out of the hard budget.
pub(crate) fn working_set_margin_bytes(working_set: u64, margin: Margin) -> u64 {
    if margin == Margin::Ledger {
        let share = working_set / 100 * LEDGER_MARGIN_PERCENT
            + working_set % 100 * LEDGER_MARGIN_PERCENT / 100;
        return share.max(LEDGER_MARGIN_FLOOR_BYTES).min(working_set);
    }
    let kept = (working_set / 100 * WORKING_SET_PERCENT
        + working_set % 100 * WORKING_SET_PERCENT / 100)
        .saturating_sub(WORKING_SET_RESERVE_BYTES);
    working_set - kept
}

/// Splash `EngineMemoryPolicy::hardBudgetBytes`: `--max-memory` only lowers
/// the automatic budget.
pub(crate) fn hard_budget_bytes(working_set: u64, max_memory: Option<u64>, margin: Margin) -> u64 {
    let automatic = working_set.saturating_sub(working_set_margin_bytes(working_set, margin));
    match max_memory {
        Some(max) if max > 0 && max < automatic => max,
        _ => automatic,
    }
}

/// Metal's recommended working set, `None` when unreadable.
pub(crate) fn recommended_working_set() -> Option<u64> {
    match crate::stream::WiredLimitContext::get_max_working_set_size() as u64 {
        0 => None,
        bytes => Some(bytes),
    }
}

/// What a load reads from the device to plan and audit against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LiveMemory {
    pub working_set_bytes: u64,
    pub active_bytes: u64,
    /// Every KV pool registered with the cache-limit coordinator.
    pub registered_pool_bytes: u64,
}

#[cfg(test)]
thread_local! {
    /// Stands in for the device on this thread.
    pub(crate) static LIVE_MEMORY_OVERRIDE: std::cell::Cell<Option<LiveMemory>> =
        const { std::cell::Cell::new(None) };
    /// Makes the device unreadable on this thread.
    #[cfg(test)]
    pub(crate) static LIVE_MEMORY_UNREADABLE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// The device now; `None` when the working set or MLX's active bytes are
/// unreadable.
pub(crate) fn live_memory() -> Option<LiveMemory> {
    #[cfg(test)]
    if LIVE_MEMORY_UNREADABLE.with(std::cell::Cell::get) {
        return None;
    }
    #[cfg(test)]
    if let Some(live) = LIVE_MEMORY_OVERRIDE.with(std::cell::Cell::get) {
        return Some(live);
    }
    let working_set_bytes = recommended_working_set()?;
    let mut active_bytes = 0u64;
    // SAFETY: the probe writes one u64 through a valid pointer and reports
    // failure through its return code.
    if unsafe { mlx_sys::mlx_get_active_memory(&mut active_bytes) } != 0 {
        return None;
    }
    Some(LiveMemory {
        working_set_bytes,
        active_bytes,
        registered_pool_bytes: crate::cache_limit::coordinator().registered_pool_bytes(),
    })
}

/// The hard budget with no `--max-memory`, for a load without a plan.
pub(crate) fn automatic_hard_budget() -> Result<u64, String> {
    recommended_working_set()
        .map(|working_set| hard_budget_bytes(working_set, None, Margin::Unguarded))
        .ok_or_else(|| "Metal recommended working set unavailable".to_string())
}

/// What set the hard budget (Splash `memoryBound`): a `--max-memory` equal to
/// the budget counts as the flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MemoryBound {
    MaxMemory,
    DeviceMemory,
}

impl MemoryBound {
    pub(crate) fn of(max_memory: Option<u64>, hard_budget: u64) -> Self {
        match max_memory {
            Some(max) if max > 0 && max == hard_budget => Self::MaxMemory,
            _ => Self::DeviceMemory,
        }
    }

    fn flag_text(self) -> &'static str {
        match self {
            Self::MaxMemory => "--max-memory",
            Self::DeviceMemory => "this Mac's memory",
        }
    }
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

fn lcm(left: u64, right: u64) -> u64 {
    if left == 0 || right == 0 {
        0
    } else {
        left / gcd(left, right) * right
    }
}

fn pages_for_aligned_region(bytes_per_page: u64) -> u64 {
    if bytes_per_page == 0 {
        0
    } else {
        EXTENT_REGION_ALIGNMENT_BYTES / gcd(EXTENT_REGION_ALIGNMENT_BYTES, bytes_per_page)
    }
}

/// The target KV geometry Splash sizes its KV extents from (`kv::Layout`):
/// `layers` full-attention layers of `heads` K/V heads and, with an
/// `index_dimension`, a bf16 block index beside them in every page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvLayout {
    pub layers: u64,
    pub heads: u64,
    pub head_dim: u64,
    pub format: KvFormat,
    pub activation_bytes: u64,
    pub index_dimension: u64,
}

impl KvLayout {
    pub(crate) fn of(
        full_attention_layers: usize,
        kv_heads: i32,
        head_dim: i32,
        dtype: DType,
        format: KvFormat,
    ) -> Self {
        Self {
            layers: full_attention_layers as u64,
            heads: kv_heads.max(0) as u64,
            head_dim: head_dim.max(0) as u64,
            format,
            activation_bytes: dtype.byte_size() as u64,
            index_dimension: 0,
        }
    }

    /// The layout with a block index of `dimension` per page
    /// (`Layout::indexDimension`).
    pub(crate) fn with_index(self, dimension: u64) -> Self {
        Self {
            index_dimension: dimension,
            ..self
        }
    }

    /// `indexBytesPerLayerPage`.
    fn index_bytes_per_layer_page(&self) -> u64 {
        KV_PAGE_TOKENS / ATTENTION_BLOCK_TOKENS * self.index_dimension * 2
    }

    fn data_bytes_per_layer_page(&self) -> u64 {
        let elements = KV_PAGE_TOKENS * self.heads * self.head_dim;
        match self.format {
            KvFormat::Int8 => elements,
            KvFormat::Bf16 => elements * self.activation_bytes,
        }
    }

    fn scale_bytes_per_layer_page(&self) -> u64 {
        match self.format {
            KvFormat::Int8 => KV_PAGE_TOKENS * self.heads * 4,
            KvFormat::Bf16 => 0,
        }
    }

    /// K/V and index bytes of one page across every layer
    /// (`bytesPerModelPage`).
    pub(crate) fn page_bytes(&self) -> u64 {
        self.layers
            * (2 * (self.data_bytes_per_layer_page() + self.scale_bytes_per_layer_page())
                + self.index_bytes_per_layer_page())
    }

    /// K/V and index bytes of one token row across every layer.
    #[cfg(test)]
    pub(crate) fn row_bytes(&self) -> u64 {
        self.page_bytes() / KV_PAGE_TOKENS
    }

    /// `extentAlignmentPages`.
    pub(crate) fn extent_alignment_pages(&self) -> u64 {
        let pages = match self.format {
            KvFormat::Bf16 => pages_for_aligned_region(self.data_bytes_per_layer_page()),
            KvFormat::Int8 => lcm(
                pages_for_aligned_region(self.data_bytes_per_layer_page()),
                pages_for_aligned_region(self.scale_bytes_per_layer_page()),
            ),
        };
        if self.index_dimension == 0 {
            pages
        } else {
            lcm(
                pages,
                pages_for_aligned_region(self.index_bytes_per_layer_page()),
            )
        }
    }

    /// `minimumExtentPages`.
    pub(crate) fn minimum_extent_pages(&self) -> u64 {
        let unit = self.extent_alignment_pages() * self.page_bytes();
        if unit == 0 {
            return 0;
        }
        (EXTENT_TARGET_BYTES / 2).div_ceil(unit).max(1) * self.extent_alignment_pages()
    }

    /// `maximumExtentPages`.
    fn maximum_extent_pages(&self) -> u64 {
        let unit = self.extent_alignment_pages() * self.page_bytes();
        if unit == 0 {
            return 0;
        }
        (EXTENT_TARGET_BYTES * 3 / 2 / unit * self.extent_alignment_pages())
            .max(self.minimum_extent_pages())
    }

    /// `extentPagesFor`: of the extent sizes that fit `pages`, the one that
    /// leaves the fewest pages over, the one nearest the target on a tie; 0
    /// when `pages` is below the smallest.
    pub(crate) fn extent_pages_for(&self, pages: u64) -> u64 {
        let step = self.extent_alignment_pages();
        let smallest = self.minimum_extent_pages();
        if smallest == 0 || pages < smallest {
            return 0;
        }
        let largest = self.maximum_extent_pages().min(pages);
        let (mut best, mut best_left, mut best_distance) = (0u64, 0u64, 0u64);
        let mut extent = smallest;
        while extent <= largest {
            let distance = (extent * self.page_bytes()).abs_diff(EXTENT_TARGET_BYTES);
            let left = pages % extent;
            if best == 0 || left < best_left || (left == best_left && distance < best_distance) {
                (best, best_left, best_distance) = (extent, left, distance);
            }
            extent += step;
        }
        best
    }

    /// The extent and the whole-extent capacity, in pages, of `bytes`
    /// (`kvCapacityPagesFor` of `kvPagesWithin(bytes, 1)`).
    pub(crate) fn capacity_of(&self, bytes: u64) -> (u64, u64) {
        let page = self.page_bytes();
        if page == 0 {
            return (0, 0);
        }
        let pages = bytes / page;
        match self.extent_pages_for(pages) {
            0 => (0, 0),
            extent => (extent, pages - pages % extent),
        }
    }

    /// `kvRunwayPages(minimumExtentPages)`: the whole smallest extents that
    /// hold the warmup pages.
    pub(crate) fn runway_pages(&self) -> u64 {
        let extent = self.minimum_extent_pages();
        if extent == 0 {
            return 0;
        }
        WARMUP_KV_PAGES.div_ceil(extent) * extent
    }
}

/// Splash `CompositeStateLayout::laneBytes` beyond the pool lane's cell: the
/// second GDN cell (the blob spare a verify writes) and the draft KV rings,
/// constant in the window. mlx-node holds no 4-bit draft context window
/// (`windowBytes`).
pub(crate) fn dflash2_lane_state_bytes(recurrent_state_bytes: u64, draft_ring_bytes: u64) -> u64 {
    recurrent_state_bytes.saturating_add(draft_ring_bytes)
}

/// What the load measured (Splash `ModelMemoryFootprint`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MemoryFootprint {
    /// Metal `recommendedMaxWorkingSetSize`.
    pub working_set_bytes: u64,
    /// `--max-memory`.
    pub max_memory_bytes: Option<u64>,
    /// MLX active bytes once every loaded weight is resident.
    pub active_bytes: u64,
    /// Other models' private KV pools, which MLX's counter does not see.
    pub other_pool_bytes: u64,
    /// This pool's restore staging (Splash `stateStagingBytes`).
    pub staging_bytes: u64,
    /// Reported only: it is inside `active_bytes`.
    pub draft_weights_bytes: u64,
    /// The streamed stores' planned bytes above what is live (Splash
    /// `hostHeldBackBytes`): the plan counts every store at its cap.
    pub held_back_bytes: u64,
    pub margin: Margin,
}

/// One lane kind of one request: its state besides KV, the layout its KV
/// is priced in, the rows a verify writes past the kept ones, and the
/// bytes a turn of `W` tokens allocates besides KV at its peak.
pub(crate) struct LaneShape<'a> {
    pub trained: u32,
    pub lane_bytes: u64,
    pub layout: KvLayout,
    pub scratch_rows: u64,
    pub transient: &'a dyn Fn(u32) -> u64,
}

/// The paged pool lane: KV in blocks of `bytes_per_block` across `layers`
/// layers. The pool grows one layer at a time, so its maximum keeps one
/// layer's bytes free for the last grow's copy.
pub(crate) struct PoolInput<'a> {
    pub shape: LaneShape<'a>,
    pub bytes_per_block: u64,
    pub layers: u64,
    pub runway_blocks: u32,
    /// `paged_cache_memory_mb`: a ceiling at or below the plan's.
    pub explicit_mib: Option<u32>,
    /// The lane's state takes pool blocks as well as its own bytes; `false`
    /// holds it beside the pool only.
    pub lane_in_pool: bool,
    /// Lane state that grows with the window `W` (a speculative turn's
    /// draft state), charged like `lane_bytes`: Splash `laneStateBytes`
    /// carries every speculative buffer of the lane beside its KV.
    pub state: &'a dyn Fn(u32) -> u64,
    /// `state` takes pool blocks as well (the scheduler books it there).
    pub state_in_pool: bool,
}

impl PoolInput<'_> {
    fn kv(&self, cap_blocks: Option<u64>) -> LaneKv<'_> {
        LaneKv::Pool {
            bytes_per_block: self.bytes_per_block,
            layers: self.layers,
            cap_blocks,
            lane_in_pool: self.lane_in_pool,
            state: self.state,
            state_in_pool: self.state_in_pool,
        }
    }

    /// The pool this lane plans from a dynamic budget of `dynamic`.
    pub(crate) fn plan_at(&self, dynamic: u64) -> PoolPlan {
        let (lane, blocks) = plan_lane(dynamic, &self.shape, self.kv(None));
        let max_blocks = blocks.min(u64::from(u32::MAX)) as u32;
        PoolPlan {
            max_blocks,
            runway_blocks: self.runway_blocks.min(max_blocks),
            bytes_per_block: self.bytes_per_block,
            lane,
        }
    }
}

/// Splash `ContextBounds` without spilling: the trained window and the
/// longest one resident KV holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ContextBounds {
    pub window: u32,
    pub resident: u32,
}

impl ContextBounds {
    pub(crate) fn maximum(&self) -> u32 {
        self.window.min(self.resident)
    }

    pub(crate) fn bound(&self) -> &'static str {
        if self.maximum() == self.window {
            "model"
        } else {
            "memory"
        }
    }
}

/// Splash `ServingRuntime::start` over the default turn's `lane`: an explicit
/// `max_context` it does not hold fails the load.
pub(crate) fn check_max_context(
    lane: ContextBounds,
    bound: MemoryBound,
    max_context: Option<u32>,
) -> Result<(), PlanError> {
    match max_context {
        Some(requested) if requested > lane.maximum() => Err(PlanError::ContextAbovePlan {
            requested,
            plan: lane.maximum(),
            bound,
        }),
        _ => Ok(()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LanePlan {
    pub window: ContextBounds,
    pub lane_bytes: u64,
    /// Splash `kvExtentPages` of the budget left at the window.
    pub extent_pages: u64,
    /// Splash `kvCapacityPages` of that budget.
    pub capacity_pages: u64,
    /// KV of the window and its scratch rows, in whole extents.
    pub kv_bytes: u64,
    pub transient_bytes: u64,
    /// Splash `minimumDynamicBytes`: the lane, the KV runway and the
    /// transient of a turn as long as the runway.
    pub minimum_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PoolPlan {
    pub max_blocks: u32,
    pub runway_blocks: u32,
    pub bytes_per_block: u64,
    pub lane: LanePlan,
}

impl PoolPlan {
    /// The pool's ceiling in whole MiB, as `paged_cache_memory_mb` reads it.
    pub(crate) fn max_mib(&self) -> u32 {
        (u64::from(self.max_blocks) * self.bytes_per_block / MIB).min(u64::from(u32::MAX)) as u32
    }

    /// The pool config's `gpu_memory_mb`: `validate` refuses one under
    /// [`mlx_paged_attn::MIN_GPU_MEMORY_MB`] though the pool allocates by
    /// blocks, so a planned pool below it must still construct.
    pub(crate) fn config_mib(&self) -> u32 {
        self.max_mib().max(mlx_paged_attn::MIN_GPU_MEMORY_MB)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MemoryPlan {
    pub footprint: MemoryFootprint,
    pub margin_bytes: u64,
    pub hard_budget_bytes: u64,
    pub bound: MemoryBound,
    pub fixed_bytes: u64,
    pub dynamic_budget_bytes: u64,
    pub pool: Option<PoolPlan>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PlanError {
    WorkingSetTooSmall {
        working_set: u64,
        margin: u64,
    },
    KvPoolDoesNotFit {
        required: u64,
        budget: u64,
        bound: MemoryBound,
    },
    PoolAbovePlan {
        requested_mib: u32,
        plan_mib: u32,
        bound: MemoryBound,
    },
    PoolBelowLane {
        requested_mib: u32,
        minimum_mib: u32,
    },
    /// Splash `planParts`: the floors exceed the hard budget before any
    /// store is read.
    FloorsDoNotFit {
        required: u64,
        budget: u64,
        bound: MemoryBound,
    },
    /// Splash `ServingRuntime::start`: an explicit `--max-context` above the
    /// context the default turn's lane holds fails the start.
    ContextAbovePlan {
        requested: u32,
        plan: u32,
        bound: MemoryBound,
    },
    /// A `--max-context` past the native window on a load with no memory
    /// plan, which Splash never starts without.
    ContextWithoutPlan {
        requested: u32,
        native: u32,
    },
}

fn bound_advice(bound: MemoryBound) -> &'static str {
    match bound {
        MemoryBound::MaxMemory => "raise --max-memory",
        MemoryBound::DeviceMemory => "choose a smaller model or close other applications",
    }
}

fn gibibytes(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / GIB as f64)
}

fn bytes_and_mib(bytes: u64) -> String {
    format!("{bytes} bytes ({:.2} MiB)", bytes as f64 / MIB as f64)
}

impl PlanError {
    pub(crate) fn message(&self, label: &str) -> String {
        match *self {
            Self::WorkingSetTooSmall {
                working_set,
                margin,
            } => format!(
                "{label} memory plan: recommended working set {working_set} bytes does not \
                 exceed the working-set margin {margin} bytes"
            ),
            Self::KvPoolDoesNotFit {
                required,
                budget,
                bound,
            } => {
                let limit = match bound {
                    MemoryBound::MaxMemory => "the memory limit is",
                    MemoryBound::DeviceMemory => "this Mac can give it",
                };
                let advice = bound_advice(bound);
                format!(
                    "{label} memory plan: the model weights with the runtime reserves, one \
                     lane's state and the KV runway need {} of memory but {limit} {}; {advice}",
                    bytes_and_mib(required),
                    bytes_and_mib(budget),
                )
            }
            Self::PoolAbovePlan {
                requested_mib,
                plan_mib,
                bound,
            } => format!(
                "{label}: paged_cache_memory_mb {requested_mib} exceeds the {plan_mib} MiB of KV \
                 pool the model and {} allow; omit it or set at most {plan_mib}",
                bound.flag_text()
            ),
            Self::PoolBelowLane {
                requested_mib,
                minimum_mib,
            } => format!(
                "{label}: paged_cache_memory_mb {requested_mib} is below the {minimum_mib} MiB one \
                 lane's state and the KV runway need; omit it or set at least {minimum_mib}"
            ),
            Self::FloorsDoNotFit {
                required,
                budget,
                bound,
            } => format!(
                "{label} memory plan: the floors of the memory plan require {required} bytes but \
                 the Metal memory budget is {budget} bytes; {}",
                bound_advice(bound)
            ),
            Self::ContextAbovePlan {
                requested,
                plan,
                bound,
            } => format!(
                "{label}: --max-context {requested} exceeds the {plan} tokens the model and {} \
                 allow; omit it or pass at most {plan}",
                bound.flag_text()
            ),
            Self::ContextWithoutPlan { requested, native } => format!(
                "{label}: --max-context {requested} is past the native window of {native}, and \
                 this load has no memory plan to hold it"
            ),
        }
    }
}

/// Splash `resident`: the rows `kv_bytes` hold in whole extents, less the
/// scratch rows.
fn resident_tokens(layout: &KvLayout, kv_bytes: u64, scratch_rows: u64) -> u64 {
    let (_, capacity) = layout.capacity_of(kv_bytes);
    (capacity * KV_PAGE_TOKENS).saturating_sub(scratch_rows)
}

/// The largest `W` in `[0, most]` that `fits`, 0 when none does; `fits`
/// holds for every `W` below one it holds for.
fn largest_window(most: u32, fits: impl Fn(u32) -> bool) -> u32 {
    if !fits(0) {
        return 0;
    }
    let (mut low, mut high) = (0u32, most);
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if fits(mid) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    low
}

/// How a lane's KV budget at a window follows from the dynamic budget.
#[derive(Clone, Copy)]
enum LaneKv<'a> {
    Pool {
        bytes_per_block: u64,
        layers: u64,
        cap_blocks: Option<u64>,
        lane_in_pool: bool,
        state: &'a dyn Fn(u32) -> u64,
        state_in_pool: bool,
    },
    /// Splash `KvExtent` arithmetic with no grow copy and no state: the
    /// dynamic budget less the lane and the transient.
    #[cfg(test)]
    Extents,
}

impl LaneKv<'_> {
    /// The lane's window-dependent state at `window`.
    fn state_bytes(&self, window: u32) -> u64 {
        match *self {
            Self::Pool { state, .. } => state(window),
            #[cfg(test)]
            Self::Extents => 0,
        }
    }

    /// The pool blocks and the KV bytes left for one request at `window`.
    fn budget(&self, dynamic: u64, shape: &LaneShape<'_>, window: u32) -> (u64, u64) {
        let transient = (shape.transient)(window);
        match *self {
            Self::Pool {
                bytes_per_block,
                layers,
                cap_blocks,
                lane_in_pool,
                state,
                state_in_pool,
            } => {
                // The lane state lives outside the pool (Splash
                // availableForOneRequestKv = dynamic - laneStateBytes).
                let state = state(window);
                let pool = u128::from(
                    dynamic
                        .saturating_sub(transient)
                        .saturating_sub(shape.lane_bytes)
                        .saturating_sub(state),
                ) * u128::from(layers)
                    / u128::from(layers + 1);
                let blocks = (pool / u128::from(bytes_per_block.max(1))) as u64;
                let blocks = cap_blocks.map_or(blocks, |cap| blocks.min(cap));
                let in_pool = if lane_in_pool { shape.lane_bytes } else { 0 }
                    .saturating_add(if state_in_pool { state } else { 0 });
                let kv = blocks
                    .saturating_mul(bytes_per_block)
                    .saturating_sub(in_pool);
                (blocks, kv)
            }
            #[cfg(test)]
            Self::Extents => (
                0,
                dynamic
                    .saturating_sub(shape.lane_bytes)
                    .saturating_sub(transient),
            ),
        }
    }
}

/// One lane's plan and the pool blocks at its window.
fn plan_lane(dynamic: u64, shape: &LaneShape<'_>, kv: LaneKv<'_>) -> (LanePlan, u64) {
    let fits = |window: u32| {
        let (_, bytes) = kv.budget(dynamic, shape, window);
        resident_tokens(&shape.layout, bytes, shape.scratch_rows) >= u64::from(window)
    };
    let window = largest_window(shape.trained, fits);
    let (blocks, bytes) = kv.budget(dynamic, shape, window);
    let (extent_pages, capacity_pages) = shape.layout.capacity_of(bytes);
    let page = shape.layout.page_bytes();
    let kv_bytes = match extent_pages {
        0 => 0,
        extent => (u64::from(window) + shape.scratch_rows)
            .div_ceil(KV_PAGE_TOKENS)
            .div_ceil(extent)
            .saturating_mul(extent)
            .saturating_mul(page),
    };
    let runway_pages = shape.layout.runway_pages();
    let runway_tokens = (runway_pages * KV_PAGE_TOKENS).min(u64::from(u32::MAX)) as u32;
    let minimum_bytes = shape
        .lane_bytes
        .saturating_add(kv.state_bytes(runway_tokens))
        .saturating_add(runway_pages.saturating_mul(page))
        .saturating_add((shape.transient)(runway_tokens));
    let plan = LanePlan {
        window: ContextBounds {
            window: shape.trained,
            resident: window,
        },
        lane_bytes: shape.lane_bytes.saturating_add(kv.state_bytes(window)),
        extent_pages,
        capacity_pages,
        kv_bytes,
        transient_bytes: (shape.transient)(window),
        minimum_bytes,
    };
    (plan, blocks)
}

impl MemoryPlan {
    /// Splash `evaluateEngineMemoryPlan` over the paged pool lane (every
    /// turn's, a DFlash2 turn's included); a load without a pool plans its
    /// footprint only.
    pub(crate) fn evaluate(
        footprint: MemoryFootprint,
        pool: Option<PoolInput<'_>>,
    ) -> Result<Self, PlanError> {
        let working_set = footprint.working_set_bytes;
        let margin_bytes = working_set_margin_bytes(working_set, footprint.margin);
        if working_set <= margin_bytes {
            return Err(PlanError::WorkingSetTooSmall {
                working_set,
                margin: margin_bytes,
            });
        }
        let hard_budget_bytes =
            hard_budget_bytes(working_set, footprint.max_memory_bytes, footprint.margin);
        let bound = MemoryBound::of(footprint.max_memory_bytes, hard_budget_bytes);
        let fixed_bytes = footprint
            .active_bytes
            .saturating_add(footprint.other_pool_bytes)
            .saturating_add(footprint.staging_bytes)
            .saturating_add(footprint.held_back_bytes)
            .saturating_add(PIPELINE_RESERVE_BYTES)
            .saturating_add(RUNTIME_OVERHEAD_RESERVE_BYTES);
        let dynamic = hard_budget_bytes.saturating_sub(fixed_bytes);

        let pool_plan = pool.as_ref().map(|input| input.plan_at(dynamic));
        if let Some(lane) = pool_plan.map(|pool| pool.lane)
            && (dynamic < lane.minimum_bytes || lane.extent_pages == 0)
        {
            let minimum = lane.minimum_bytes;
            return Err(PlanError::KvPoolDoesNotFit {
                required: fixed_bytes.saturating_add(minimum),
                budget: hard_budget_bytes,
                bound,
            });
        }

        let pool_plan = match (pool_plan, pool.as_ref()) {
            (Some(plan), Some(input)) => Some(Self::explicit_pool(dynamic, plan, input, bound)?),
            _ => None,
        };
        Ok(Self {
            footprint,
            margin_bytes,
            hard_budget_bytes,
            bound,
            fixed_bytes,
            dynamic_budget_bytes: dynamic,
            pool: pool_plan,
        })
    }

    /// An explicit pool size at or below the plan's becomes the ceiling, and
    /// the pool window is the one it holds.
    fn explicit_pool(
        dynamic: u64,
        plan: PoolPlan,
        input: &PoolInput<'_>,
        bound: MemoryBound,
    ) -> Result<PoolPlan, PlanError> {
        let Some(requested_mib) = input.explicit_mib else {
            return Ok(plan);
        };
        let bytes_per_block = input.bytes_per_block.max(1);
        let requested_blocks = u64::from(requested_mib) * MIB / bytes_per_block;
        if requested_blocks > u64::from(plan.max_blocks) {
            return Err(PlanError::PoolAbovePlan {
                requested_mib,
                plan_mib: plan.max_mib(),
                bound,
            });
        }
        let layout = input.shape.layout;
        let runway_tokens =
            (layout.runway_pages() * KV_PAGE_TOKENS).min(u64::from(u32::MAX)) as u32;
        let in_pool = if input.lane_in_pool {
            input.shape.lane_bytes
        } else {
            0
        }
        .saturating_add(if input.state_in_pool {
            (input.state)(runway_tokens)
        } else {
            0
        });
        let minimum_blocks = in_pool
            .saturating_add(layout.runway_pages() * layout.page_bytes())
            .div_ceil(bytes_per_block);
        if requested_blocks < minimum_blocks {
            return Err(PlanError::PoolBelowLane {
                requested_mib,
                minimum_mib: (minimum_blocks * bytes_per_block)
                    .div_ceil(MIB)
                    .min(u64::from(u32::MAX)) as u32,
            });
        }
        let capped = LaneShape {
            trained: plan.lane.window.maximum(),
            ..input.shape
        };
        let (mut lane, _) = plan_lane(dynamic, &capped, input.kv(Some(requested_blocks)));
        lane.window.window = plan.lane.window.window;
        let max_blocks = requested_blocks as u32;
        Ok(PoolPlan {
            max_blocks,
            runway_blocks: plan.runway_blocks.min(max_blocks),
            lane,
            ..plan
        })
    }

    /// A plan of nothing but the lanes a test installs.
    #[cfg(test)]
    fn bare() -> Self {
        Self {
            footprint: MemoryFootprint {
                working_set_bytes: 0,
                max_memory_bytes: None,
                active_bytes: 0,
                other_pool_bytes: 0,
                staging_bytes: 0,
                draft_weights_bytes: 0,
                held_back_bytes: 0,
                margin: Margin::Unguarded,
            },
            margin_bytes: 0,
            hard_budget_bytes: 0,
            bound: MemoryBound::DeviceMemory,
            fixed_bytes: 0,
            dynamic_budget_bytes: 0,
            pool: None,
        }
    }

    #[cfg(test)]
    fn bare_lane(resident: u32, lane_bytes: u64, extent_pages: u64) -> LanePlan {
        LanePlan {
            window: ContextBounds {
                window: u32::MAX,
                resident,
            },
            lane_bytes,
            extent_pages,
            capacity_pages: 0,
            kv_bytes: 0,
            transient_bytes: 0,
            minimum_bytes: 0,
        }
    }

    /// A plan of `dynamic` bytes for scheduled state and no lanes, for tests
    /// of a family whose fixtures have no plan.
    #[cfg(test)]
    pub(crate) fn with_dynamic(dynamic: u64) -> Self {
        Self {
            hard_budget_bytes: u64::MAX,
            dynamic_budget_bytes: dynamic,
            ..Self::bare()
        }
    }

    /// A plan whose pool of `max_blocks` blocks gives its lane of
    /// `lane_bytes` a `resident`-token window, for scheduler tests.
    #[cfg(test)]
    pub(crate) fn with_pool_window(
        max_blocks: u32,
        bytes_per_block: u64,
        lane_bytes: u64,
        resident: u32,
    ) -> Self {
        Self {
            pool: Some(PoolPlan {
                max_blocks,
                runway_blocks: max_blocks,
                bytes_per_block,
                lane: Self::bare_lane(resident, lane_bytes, 0),
            }),
            ..Self::bare()
        }
    }

    /// What the memory governor charges beside the allocations it sees:
    /// the pipeline and runtime reserves and the lane's transient, MLX
    /// allocations no claim admits.
    pub(crate) fn untracked_reserve_bytes(&self) -> u64 {
        PIPELINE_RESERVE_BYTES
            .saturating_add(RUNTIME_OVERHEAD_RESERVE_BYTES)
            .saturating_add(
                self.pool
                    .as_ref()
                    .map_or(0, |pool| pool.lane.transient_bytes),
            )
    }

    /// The ceiling every live check uses: the hard budget less the reserves.
    pub(crate) fn ceiling_bytes(&self) -> u64 {
        self.hard_budget_bytes
            .saturating_sub(PIPELINE_RESERVE_BYTES)
            .saturating_sub(RUNTIME_OVERHEAD_RESERVE_BYTES)
    }

    /// Splash's log line for a `--max-memory` above the automatic budget.
    pub(crate) fn max_memory_ignored(&self) -> Option<String> {
        let max = self.footprint.max_memory_bytes?;
        (max > self.hard_budget_bytes).then(|| {
            format!(
                "--max-memory ignored: its {} exceed this Mac's Metal budget of {}, which it \
                 can only lower.",
                gibibytes(max),
                gibibytes(self.hard_budget_bytes)
            )
        })
    }

    /// Splash `EngineMemoryBreakdown::describe`.
    pub(crate) fn describe(&self) -> String {
        let footprint = &self.footprint;
        let mut out = String::new();
        let _ = writeln!(
            out,
            "recommended working set: {}",
            bytes_and_mib(footprint.working_set_bytes)
        );
        let _ = writeln!(
            out,
            "configured memory limit: {}",
            footprint
                .max_memory_bytes
                .map_or_else(|| "automatic".to_string(), bytes_and_mib)
        );
        let _ = writeln!(
            out,
            "working-set margin: {}",
            bytes_and_mib(self.margin_bytes)
        );
        let _ = writeln!(
            out,
            "hard budget: {} ({})",
            bytes_and_mib(self.hard_budget_bytes),
            self.bound.flag_text()
        );
        let _ = writeln!(
            out,
            "MLX active (weights): {}",
            bytes_and_mib(footprint.active_bytes)
        );
        let _ = writeln!(
            out,
            "draft weights: {}",
            bytes_and_mib(footprint.draft_weights_bytes)
        );
        let _ = writeln!(
            out,
            "other KV pools: {}",
            bytes_and_mib(footprint.other_pool_bytes)
        );
        let _ = writeln!(
            out,
            "SSD cache state staging: {}",
            bytes_and_mib(footprint.staging_bytes)
        );
        let _ = writeln!(
            out,
            "expert store and row cache above what is live: {}",
            bytes_and_mib(footprint.held_back_bytes)
        );
        let _ = writeln!(
            out,
            "pipeline reserve: {}",
            bytes_and_mib(PIPELINE_RESERVE_BYTES)
        );
        let _ = writeln!(
            out,
            "allocator/runtime reserve: {}",
            bytes_and_mib(RUNTIME_OVERHEAD_RESERVE_BYTES)
        );
        let _ = writeln!(out, "fixed runtime: {}", bytes_and_mib(self.fixed_bytes));
        let _ = write!(
            out,
            "elastic state/KV budget: {}",
            bytes_and_mib(self.dynamic_budget_bytes)
        );
        let lane = |out: &mut String, name: &str, lane: &LanePlan| {
            let _ = write!(
                out,
                "\n{name}: context {} tokens ({}; window {}, resident {}); lane state {}, \
                 transient {}, KV {} in extents of {} pages ({} pages of capacity), minimum \
                 dynamic {}",
                lane.window.maximum(),
                lane.window.bound(),
                lane.window.window,
                lane.window.resident,
                bytes_and_mib(lane.lane_bytes),
                bytes_and_mib(lane.transient_bytes),
                bytes_and_mib(lane.kv_bytes),
                lane.extent_pages,
                lane.capacity_pages,
                bytes_and_mib(lane.minimum_bytes),
            );
        };
        if let Some(pool) = &self.pool {
            let _ = write!(
                out,
                "\nKV pool: {} blocks of {} at most, {} at the start",
                pool.max_blocks,
                bytes_and_mib(pool.bytes_per_block),
                pool.runway_blocks
            );
            lane(&mut out, "pool lane", &pool.lane);
        }
        out
    }
}

/// What the end of a load measured, for [`audit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MeasuredMemory {
    pub active_bytes: u64,
    /// This pool's allocated blocks, without its staging.
    pub pool_bytes: u64,
    pub other_pool_bytes: u64,
}

/// Splash `auditActualMemory` on the current footprint: the KV pool within
/// its plan, and everything with the reserves within the hard budget.
pub(crate) fn audit(plan: &MemoryPlan, measured: MeasuredMemory) -> Result<(), String> {
    if let Some(pool) = &plan.pool {
        let planned = u64::from(pool.max_blocks).saturating_mul(pool.bytes_per_block);
        if measured.pool_bytes > planned {
            return Err(format!(
                "KV actual bytes {} exceed planned bytes {planned}",
                measured.pool_bytes
            ));
        }
    }
    let footprint = measured
        .active_bytes
        .saturating_add(measured.pool_bytes)
        .saturating_add(measured.other_pool_bytes)
        .saturating_add(plan.footprint.staging_bytes)
        .saturating_add(PIPELINE_RESERVE_BYTES)
        .saturating_add(RUNTIME_OVERHEAD_RESERVE_BYTES);
    if footprint > plan.hard_budget_bytes {
        return Err(format!(
            "actual Metal footprint {footprint} bytes with the reserves exceeds hard budget {} \
             bytes",
            plan.hard_budget_bytes
        ));
    }
    Ok(())
}

/// The least dynamic budget at which every lane's window, as `lanes` give
/// it from a dynamic budget, holds `tokens` (Splash's KV allowance of
/// `planParts`, as the budget a lane plans from); `u64::MAX` when none does.
/// Each window grows with the budget.
pub(crate) fn allowance_dynamic_bytes(lanes: &[&dyn Fn(u64) -> u32], tokens: u32) -> u64 {
    let holds = |dynamic: u64| lanes.iter().all(|window| window(dynamic) >= tokens);
    let mut high = GIB;
    while !holds(high) {
        if high == u64::MAX {
            return u64::MAX;
        }
        high = high.saturating_mul(2);
    }
    let mut low = 0u64;
    while low < high {
        let mid = low + (high - low) / 2;
        if holds(mid) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    low
}

/// The rows of a table one id per token of a `TABLE_SHARE_TOKENS`-token
/// prompt per stage reads, at most the table's `rows` (Splash
/// `tableShareBytes` without its working bytes).
pub(crate) fn table_share_bytes(ids_per_token: u64, rows: u64, row_bytes: u64) -> u64 {
    TABLE_SHARE_TOKENS
        .saturating_mul(ids_per_token)
        .min(rows)
        .saturating_mul(row_bytes)
}

/// A store the plan may stream (Splash `StreamedPart`): all of it, and what
/// the floors charge it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StreamedPart {
    pub all: u64,
    pub minimum: u64,
}

impl StreamedPart {
    fn above(&self) -> u64 {
        self.all.saturating_sub(self.minimum)
    }
}

/// Splash `StreamedMemoryInputs` of a target that cannot spill, before any
/// store is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StreamedInputs {
    pub hard: u64,
    pub bound: MemoryBound,
    /// What loads besides the stores, with the other models' pools and the
    /// restore staging; the reserves are added here.
    pub fixed: u64,
    /// [`allowance_dynamic_bytes`] of the lanes the load serves.
    pub allowance: u64,
    /// The largest of those lanes' minimum dynamic bytes.
    pub minimum_dynamic: u64,
    pub experts: StreamedPart,
    /// The row cache: `all` is its share, `minimum` its read staging.
    pub rows: StreamedPart,
}

/// Splash `StreamedMemoryPlan` of the stores: the experts whole, or the
/// store's cap, and the row cache's bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StorePlan {
    pub whole: bool,
    pub expert_cap_bytes: u64,
    pub row_cache_bytes: u64,
}

/// Splash `planParts` without spilling: the floors first, then the KV
/// allowance; the experts load whole only where they leave it and the row
/// share, and otherwise stream through what the allowance and the share
/// leave; the row cache takes what the allowance leaves of its share.
pub(crate) fn plan_streamed_store(inputs: &StreamedInputs) -> Result<StorePlan, PlanError> {
    let loaded = inputs
        .fixed
        .saturating_add(PIPELINE_RESERVE_BYTES)
        .saturating_add(RUNTIME_OVERHEAD_RESERVE_BYTES)
        .saturating_add(inputs.experts.minimum)
        .saturating_add(inputs.rows.minimum);
    let floors = loaded.saturating_add(inputs.minimum_dynamic);
    if floors > inputs.hard {
        return Err(PlanError::FloorsDoNotFit {
            required: floors,
            budget: inputs.hard,
            bound: inputs.bound,
        });
    }
    let rest = i128::from(inputs.hard - loaded);
    let allowance = i128::from(inputs.allowance);
    let experts_above = i128::from(inputs.experts.above());
    let rows_above = i128::from(inputs.rows.above());
    let left = rest - allowance - rows_above;
    let whole = experts_above == 0 || left >= experts_above;
    let experts_extra = if whole {
        experts_above
    } else {
        left.clamp(0, experts_above)
    };
    let rows_extra = (rest - experts_extra - allowance).clamp(0, rows_above);
    Ok(StorePlan {
        whole,
        expert_cap_bytes: inputs.experts.minimum + experts_extra as u64,
        row_cache_bytes: inputs.rows.minimum + rows_extra as u64,
    })
}

#[cfg(test)]
mod tests;
