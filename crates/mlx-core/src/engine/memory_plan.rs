//! One memory plan per load, in the shape of Splash `EngineMemoryPlan`
//! (`evaluateEngineMemoryPlan`): a hard budget below the Metal working set,
//! the fixed bytes the load measured, and per lane the longest window whose
//! state, transients and KV in whole extents fit the dynamic rest.

use std::fmt::Write as _;

use crate::array::DType;
use crate::transformer::KvFormat;
use crate::transformer::paged_kv_cache_adapter::PagedPrefillMemorySnapshot;

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

/// The share of the working set the automatic budget keeps, less
/// [`WORKING_SET_RESERVE_BYTES`]; Splash keeps `max(1 GiB, 2%)` instead
/// (`EngineMemoryPolicy::workingSetMarginBytes`), which arrives with the
/// memory governor.
const WORKING_SET_PERCENT: u64 = 85;
const WORKING_SET_RESERVE_BYTES: u64 = GIB;

/// The bytes of the working set kept out of the hard budget.
pub(crate) fn working_set_margin_bytes(working_set: u64) -> u64 {
    let kept = (working_set / 100 * WORKING_SET_PERCENT
        + working_set % 100 * WORKING_SET_PERCENT / 100)
        .saturating_sub(WORKING_SET_RESERVE_BYTES);
    working_set - kept
}

/// Splash `EngineMemoryPolicy::hardBudgetBytes`: `--max-memory` only lowers
/// the automatic budget.
pub(crate) fn hard_budget_bytes(working_set: u64, max_memory: Option<u64>) -> u64 {
    let automatic = working_set.saturating_sub(working_set_margin_bytes(working_set));
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
}

/// The device now; `None` when the working set or MLX's active bytes are
/// unreadable.
pub(crate) fn live_memory() -> Option<LiveMemory> {
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
        .map(|working_set| hard_budget_bytes(working_set, None))
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

/// The target KV geometry Splash sizes its KV extents from (`kv::Layout`,
/// no block index): `layers` full-attention layers of `heads` K/V heads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvLayout {
    pub layers: u64,
    pub heads: u64,
    pub head_dim: u64,
    pub format: KvFormat,
    pub activation_bytes: u64,
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
        }
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

    /// K/V bytes of one page across every layer (`bytesPerModelPage`).
    pub(crate) fn page_bytes(&self) -> u64 {
        self.layers * 2 * (self.data_bytes_per_layer_page() + self.scale_bytes_per_layer_page())
    }

    /// K/V bytes of one token row across every layer.
    pub(crate) fn row_bytes(&self) -> u64 {
        self.page_bytes() / KV_PAGE_TOKENS
    }

    /// `extentAlignmentPages`.
    pub(crate) fn extent_alignment_pages(&self) -> u64 {
        match self.format {
            KvFormat::Bf16 => pages_for_aligned_region(self.data_bytes_per_layer_page()),
            KvFormat::Int8 => lcm(
                pages_for_aligned_region(self.data_bytes_per_layer_page()),
                pages_for_aligned_region(self.scale_bytes_per_layer_page()),
            ),
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

    /// Rows of the KV extent a budget of `budget_bytes` sizes, the smallest
    /// extent when the budget holds fewer pages (Splash's runway extent).
    pub(crate) fn extent_rows_for(&self, budget_bytes: u64) -> u64 {
        let page = self.page_bytes();
        if page == 0 {
            return 0;
        }
        let extent = match self.extent_pages_for(budget_bytes / page) {
            0 => self.minimum_extent_pages(),
            pages => pages,
        };
        extent * KV_PAGE_TOKENS
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

/// Splash `laneBytes` of a speculative lane in mlx-node: two GDN cells (the
/// live state and the pre-verify copy) and the draft's KV window buffers.
pub(crate) fn speculative_lane_bytes(recurrent_state_bytes: u64, draft_ring_bytes: u64) -> u64 {
    recurrent_state_bytes
        .saturating_mul(2)
        .saturating_add(draft_ring_bytes)
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
}

/// A flat KV lane beside the pool: KV grows by a copy of one of
/// `copy_layers` layers (`None`: no copy); the pool's allocated runway is
/// held from it.
pub(crate) struct FlatInput<'a> {
    pub shape: LaneShape<'a>,
    pub copy_layers: Option<u64>,
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
    pub copy_bytes: u64,
    pub held_bytes: u64,
    /// Splash `minimumDynamicBytes`: the lane, the held bytes, the KV runway
    /// and the transient of a turn as long as the runway.
    pub minimum_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PoolPlan {
    pub max_blocks: u32,
    pub runway_blocks: u32,
    pub bytes_per_block: u64,
    pub lane: LanePlan,
    layout: KvLayout,
}

impl PoolPlan {
    /// The pool's ceiling in whole MiB, as `paged_cache_memory_mb` reads it.
    pub(crate) fn max_mib(&self) -> u32 {
        (u64::from(self.max_blocks) * self.bytes_per_block / MIB).min(u64::from(u32::MAX)) as u32
    }

    /// The longest pool window, at most the lane's, of a lane that also
    /// holds `extra(W)` bytes of the pool for a `W`-token turn.
    pub(crate) fn window_with(&self, extra: &dyn Fn(u32) -> u64) -> u32 {
        let pool = u64::from(self.max_blocks) * self.bytes_per_block;
        let fits = |window: u32| {
            let kv = pool
                .saturating_sub(self.lane.lane_bytes)
                .saturating_sub(extra(window));
            resident_tokens(&self.layout, kv, SPECULATIVE_SCRATCH_TOKENS) >= u64::from(window)
        };
        largest_window(self.lane.window.maximum(), fits)
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
    pub speculative: Option<LanePlan>,
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
                let (limit, advice) = match bound {
                    MemoryBound::MaxMemory => ("the memory limit is", "raise --max-memory"),
                    MemoryBound::DeviceMemory => (
                        "this Mac can give it",
                        "choose a smaller model or close other applications",
                    ),
                };
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
enum LaneKv {
    Pool {
        bytes_per_block: u64,
        layers: u64,
        cap_blocks: Option<u64>,
    },
    Flat {
        held_bytes: u64,
        copy_layers: Option<u64>,
    },
}

impl LaneKv {
    fn copy_bytes(&self, shape: &LaneShape<'_>, window: u32) -> u64 {
        match *self {
            Self::Pool { .. }
            | Self::Flat {
                copy_layers: None, ..
            } => 0,
            Self::Flat {
                copy_layers: Some(layers),
                ..
            } => (u64::from(window) + shape.scratch_rows)
                .saturating_mul(shape.layout.row_bytes())
                .div_ceil(layers.max(1)),
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
            } => {
                // The lane state lives outside the pool (Splash
                // availableForOneRequestKv = dynamic - laneStateBytes).
                let pool = u128::from(
                    dynamic
                        .saturating_sub(transient)
                        .saturating_sub(shape.lane_bytes),
                ) * u128::from(layers)
                    / u128::from(layers + 1);
                let blocks = (pool / u128::from(bytes_per_block.max(1))) as u64;
                let blocks = cap_blocks.map_or(blocks, |cap| blocks.min(cap));
                let kv = blocks
                    .saturating_mul(bytes_per_block)
                    .saturating_sub(shape.lane_bytes);
                (blocks, kv)
            }
            Self::Flat { held_bytes, .. } => {
                let kv = dynamic
                    .saturating_sub(held_bytes)
                    .saturating_sub(shape.lane_bytes)
                    .saturating_sub(transient)
                    .saturating_sub(self.copy_bytes(shape, window));
                (0, kv)
            }
        }
    }

    fn held_bytes(&self) -> u64 {
        match *self {
            Self::Pool { .. } => 0,
            Self::Flat { held_bytes, .. } => held_bytes,
        }
    }
}

/// One lane's plan and the pool blocks at its window (0 for a flat lane).
fn plan_lane(dynamic: u64, shape: &LaneShape<'_>, kv: LaneKv) -> (LanePlan, u64) {
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
        .saturating_add(kv.held_bytes())
        .saturating_add(runway_pages.saturating_mul(page))
        .saturating_add((shape.transient)(runway_tokens));
    let plan = LanePlan {
        window: ContextBounds {
            window: shape.trained,
            resident: window,
        },
        lane_bytes: shape.lane_bytes,
        extent_pages,
        capacity_pages,
        kv_bytes,
        transient_bytes: (shape.transient)(window),
        copy_bytes: kv.copy_bytes(shape, window),
        held_bytes: kv.held_bytes(),
        minimum_bytes,
    };
    (plan, blocks)
}

impl MemoryPlan {
    /// Splash `evaluateEngineMemoryPlan` over the lanes this load serves:
    /// the paged pool lane and, with a draft, the speculative flat lane,
    /// which cannot use the pool's allocated runway.
    pub(crate) fn evaluate(
        footprint: MemoryFootprint,
        pool: Option<PoolInput<'_>>,
        speculative: Option<FlatInput<'_>>,
    ) -> Result<Self, PlanError> {
        let working_set = footprint.working_set_bytes;
        let margin_bytes = working_set_margin_bytes(working_set);
        if working_set <= margin_bytes {
            return Err(PlanError::WorkingSetTooSmall {
                working_set,
                margin: margin_bytes,
            });
        }
        let hard_budget_bytes = hard_budget_bytes(working_set, footprint.max_memory_bytes);
        let bound = MemoryBound::of(footprint.max_memory_bytes, hard_budget_bytes);
        let fixed_bytes = footprint
            .active_bytes
            .saturating_add(footprint.other_pool_bytes)
            .saturating_add(footprint.staging_bytes)
            .saturating_add(PIPELINE_RESERVE_BYTES)
            .saturating_add(RUNTIME_OVERHEAD_RESERVE_BYTES);
        let dynamic = hard_budget_bytes.saturating_sub(fixed_bytes);

        let pool_kv = |input: &PoolInput<'_>, cap_blocks| LaneKv::Pool {
            bytes_per_block: input.bytes_per_block,
            layers: input.layers,
            cap_blocks,
        };
        let pool_plan = pool.as_ref().map(|input| {
            let (lane, blocks) = plan_lane(dynamic, &input.shape, pool_kv(input, None));
            let max_blocks = blocks.min(u64::from(u32::MAX)) as u32;
            PoolPlan {
                max_blocks,
                runway_blocks: input.runway_blocks.min(max_blocks),
                bytes_per_block: input.bytes_per_block,
                lane,
                layout: input.shape.layout,
            }
        });
        let held_bytes = pool_plan.map_or(0, |pool| {
            u64::from(pool.runway_blocks).saturating_mul(pool.bytes_per_block)
        });
        let speculative_plan = speculative.as_ref().map(|input| {
            plan_lane(
                dynamic,
                &input.shape,
                LaneKv::Flat {
                    held_bytes,
                    copy_layers: input.copy_layers,
                },
            )
            .0
        });

        let lanes = pool_plan
            .iter()
            .map(|pool| pool.lane)
            .chain(speculative_plan);
        let minimum = lanes.clone().map(|lane| lane.minimum_bytes).max();
        if let Some(minimum) = minimum
            && (dynamic < minimum || lanes.clone().any(|lane| lane.extent_pages == 0))
        {
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
            speculative: speculative_plan,
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
        let minimum_blocks = input
            .shape
            .lane_bytes
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
        let kv = LaneKv::Pool {
            bytes_per_block,
            layers: input.layers,
            cap_blocks: Some(requested_blocks),
        };
        let (mut lane, _) = plan_lane(dynamic, &capped, kv);
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
            },
            margin_bytes: 0,
            hard_budget_bytes: 0,
            bound: MemoryBound::DeviceMemory,
            fixed_bytes: 0,
            dynamic_budget_bytes: 0,
            pool: None,
            speculative: None,
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
            copy_bytes: 0,
            held_bytes: 0,
            minimum_bytes: 0,
        }
    }

    /// A plan whose speculative lane holds `resident` tokens in extents of
    /// `extent_pages` and nothing else, for turn tests on tiny models. Its
    /// ceiling is unbounded, so the live headroom is the device's.
    #[cfg(test)]
    pub(crate) fn with_speculative_window(resident: u32, extent_pages: u64) -> Self {
        Self {
            hard_budget_bytes: u64::MAX,
            speculative: Some(Self::bare_lane(resident, 0, extent_pages)),
            ..Self::bare()
        }
    }

    /// A plan whose pool of `max_blocks` blocks gives its lane of
    /// `lane_bytes` a `resident`-token window in `layout`, for scheduler
    /// tests.
    #[cfg(test)]
    pub(crate) fn with_pool_window(
        max_blocks: u32,
        bytes_per_block: u64,
        layout: KvLayout,
        lane_bytes: u64,
        resident: u32,
    ) -> Self {
        Self {
            pool: Some(PoolPlan {
                max_blocks,
                runway_blocks: max_blocks,
                bytes_per_block,
                lane: Self::bare_lane(resident, lane_bytes, 0),
                layout,
            }),
            ..Self::bare()
        }
    }

    /// Whether the speculative lane's window is the plan's, so an omitted
    /// max_tokens means the rest of it, whatever bounds it.
    pub(crate) fn prices_speculative_lane(&self) -> bool {
        self.speculative.is_some()
    }

    /// The ceiling every live check uses: the hard budget less the reserves.
    pub(crate) fn ceiling_bytes(&self) -> u64 {
        self.hard_budget_bytes
            .saturating_sub(PIPELINE_RESERVE_BYTES)
            .saturating_sub(RUNTIME_OVERHEAD_RESERVE_BYTES)
    }

    /// The ceiling less what is live now; `None` when nothing is readable.
    /// MLX's cache is reclaimable. Registered pool bytes hold every pool's
    /// staging, and MLX's active counter sees neither.
    pub(crate) fn live_headroom(&self, snapshot: PagedPrefillMemorySnapshot) -> Option<u64> {
        crate::transformer::paged_policy::live_headroom(snapshot, Some(self.ceiling_bytes()))
            .selected_bytes
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
                 transient {}, growth copy {}, held {}, KV {} in extents of {} pages ({} pages \
                 of capacity), minimum dynamic {}",
                lane.window.maximum(),
                lane.window.bound(),
                lane.window.window,
                lane.window.resident,
                bytes_and_mib(lane.lane_bytes),
                bytes_and_mib(lane.transient_bytes),
                bytes_and_mib(lane.copy_bytes),
                bytes_and_mib(lane.held_bytes),
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
        if let Some(speculative) = &self.speculative {
            lane(&mut out, "speculative lane", speculative);
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

#[cfg(test)]
mod tests;
