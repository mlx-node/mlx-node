//! Shared Qwen image-feature cache, live memory budget, and image encoding.

use super::encoder::QwenVisionEncoder;
use super::prompt::IMAGE_TOKEN_ID;
use crate::array::MxArray;
use crate::engine;
use crate::engine::vision::VisionMerge;
use crate::inference_trace::elapsed_ms;
use crate::models::paddleocr_vl::processing::ProcessedImages;
use crate::nn::Embedding;
use crate::stream::{Stream, StreamContext};
use crate::vision::qwen::prompt::{get_rope_index, merge_input_ids_with_image_features};
use napi::{Error, Result, Status};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

/// Hard cap on inactive per-image feature entries retained by the vision LRU.
/// The active request is protected from eviction even when it is larger than
/// this cap; this prevents scan thrash while one large prompt is being built.
pub(crate) const VISION_CACHE_MAX_ENTRIES: usize = 128;

const VISION_GIB: u64 = 1024 * 1024 * 1024;

/// Used only when neither MLX nor Metal can report a usable cap. Do not derive
/// this from physical RAM: unified-memory pressure and the configured MLX limit
/// are the constraints that matter to inference.
const VISION_FALLBACK_EFFECTIVE_CAP_BYTES: u64 = 8 * VISION_GIB;
const VISION_SAFETY_RESERVE_MIN_BYTES: u64 = 2 * VISION_GIB;
const VISION_SAFETY_RESERVE_MAX_BYTES: u64 = 16 * VISION_GIB;
const VISION_CACHE_MAX_BYTES: u64 = VISION_GIB;
const VISION_MISS_BATCH_MIN_PATCHES: u64 = 1;
const VISION_MISS_BATCH_MAX_PATCHES: u64 = 32 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VisionMemoryCapSource {
    TotalUnifiedMemory,
    MlxMemoryLimit,
    MetalWorkingSet,
    ConservativeFallback,
}

impl VisionMemoryCapSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::TotalUnifiedMemory => "total_unified_memory_85pct",
            Self::MlxMemoryLimit => "mlx_memory_limit",
            Self::MetalWorkingSet => "metal_recommended_working_set",
            Self::ConservativeFallback => "conservative_fallback",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct VisionMemorySnapshot {
    total_system_memory_bytes: u64,
    mlx_memory_limit_bytes: u64,
    metal_working_set_bytes: u64,
    allocator_active_bytes: u64,
    allocator_active_probe_ok: bool,
    /// Process-wide Metal allocation snapshot. This includes MLX allocations
    /// plus external LayerKVPool buffers that MLX's active counter omits.
    metal_current_allocated_bytes: u64,
    metal_current_probe_ok: bool,
    /// Informational only. The caller drains MLX's reclaimable allocator cache
    /// immediately before probing, and this value is deliberately not charged
    /// against headroom. Any residue is logged for diagnosis.
    allocator_cache_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
struct VisionMemoryBudget {
    cap_source: VisionMemoryCapSource,
    effective_cap_bytes: u64,
    safety_reserve_bytes: u64,
    usage_probe_available: bool,
    metal_nonreclaimable_bytes: u64,
    used_memory_bytes: u64,
    output_headroom_bytes: u64,
    projected_output_fits: bool,
    headroom_bytes: u64,
    projected_output_bytes: u64,
    transient_budget_bytes: u64,
    cache_budget_bytes: u64,
    peak_bytes_per_patch: u64,
    miss_batch_patch_budget: i64,
}

fn percentage_of(value: u64, percent: u64) -> u64 {
    ((u128::from(value) * u128::from(percent)) / 100)
        .try_into()
        .unwrap_or(u64::MAX)
}

fn lower_nonzero_cap(snapshot: VisionMemorySnapshot) -> (u64, VisionMemoryCapSource) {
    let candidates = [
        (
            percentage_of(snapshot.total_system_memory_bytes, 85),
            VisionMemoryCapSource::TotalUnifiedMemory,
        ),
        (
            snapshot.mlx_memory_limit_bytes,
            VisionMemoryCapSource::MlxMemoryLimit,
        ),
        (
            percentage_of(snapshot.metal_working_set_bytes, 95),
            VisionMemoryCapSource::MetalWorkingSet,
        ),
    ];
    candidates
        .into_iter()
        .filter(|(bytes, _)| *bytes > 0)
        .min_by_key(|(bytes, _)| *bytes)
        .unwrap_or((
            VISION_FALLBACK_EFFECTIVE_CAP_BYTES,
            VisionMemoryCapSource::ConservativeFallback,
        ))
}

fn probe_u64(probe: unsafe extern "C-unwind" fn(*mut u64) -> i32) -> u64 {
    probe_u64_with_status(probe).0
}

fn probe_u64_with_status(probe: unsafe extern "C-unwind" fn(*mut u64) -> i32) -> (u64, bool) {
    let mut value = 0u64;
    // SAFETY: every accepted MLX probe writes at most one u64 to a valid
    // pointer and reports failure through its return code.
    if unsafe { probe(&mut value) } == 0 {
        (value, true)
    } else {
        (0, false)
    }
}

fn resolve_vision_memory_budget(
    snapshot: VisionMemorySnapshot,
    current_vision_cache_bytes: u64,
    protected_feature_bytes: u64,
    hidden_size: u64,
    intermediate_size: u64,
    activation_dtype_bytes: u64,
    raw_pixel_bytes_per_patch: u64,
    projected_output_bytes: u64,
) -> VisionMemoryBudget {
    let (effective_cap_bytes, cap_source) = lower_nonzero_cap(snapshot);

    // Keep 12.5% of the effective cap free for text prefill, paged KV growth,
    // command buffers, and process allocations outside MLX. Clamps keep the
    // reserve useful on both small and workstation-class systems.
    let reserve_ceiling = VISION_SAFETY_RESERVE_MAX_BYTES.min(effective_cap_bytes / 2);
    let reserve_floor = VISION_SAFETY_RESERVE_MIN_BYTES.min(reserve_ceiling);
    let safety_reserve_bytes = (effective_cap_bytes / 8).clamp(reserve_floor, reserve_ceiling);
    // Metal's current allocation includes both active MLX buffers, MLX's
    // reclaimable free pool, and external paged-KV buffers. Remove the
    // allocator cache, then take the larger non-reclaimable counter rather
    // than summing and double charging MLX-owned arrays.
    let usage_probe_available = (snapshot.allocator_active_probe_ok
        && snapshot.allocator_active_bytes > 0)
        || (snapshot.metal_current_probe_ok && snapshot.metal_current_allocated_bytes > 0);
    let metal_nonreclaimable_bytes = snapshot
        .metal_current_allocated_bytes
        .saturating_sub(snapshot.allocator_cache_bytes);
    let measured_used_memory_bytes = snapshot
        .allocator_active_bytes
        .max(metal_nonreclaimable_bytes);
    // A loaded vision model cannot genuinely consume zero bytes. If both
    // probes fail/return zero, fail closed by exposing no headroom; the caller
    // can still satisfy an all-hit request but must not start new encoding.
    let used_memory_bytes = if usage_probe_available {
        measured_used_memory_bytes
    } else {
        effective_cap_bytes.saturating_sub(safety_reserve_bytes)
    };
    let output_headroom_bytes = effective_cap_bytes
        .saturating_sub(safety_reserve_bytes)
        .saturating_sub(used_memory_bytes);
    let projected_output_fits = projected_output_bytes <= output_headroom_bytes;
    let headroom_bytes = output_headroom_bytes
        // Miss outputs become protected active cache entries and remain live
        // while later batches encode. Reserve their full request total before
        // choosing any batch size so the initial snapshot cannot overcommit.
        .saturating_sub(projected_output_bytes);

    // Per-image feature arrays are active MLX allocations, so subtract them
    // from active usage before resolving the total cache allowance. This
    // avoids charging the current cache twice. Allocate at most one quarter of
    // the non-model capacity to retained vision features.
    let active_without_vision_cache =
        used_memory_bytes.saturating_sub(current_vision_cache_bytes.min(used_memory_bytes));
    let cache_capacity = effective_cap_bytes
        .saturating_sub(safety_reserve_bytes)
        .saturating_sub(active_without_vision_cache);
    // The active request is the floor; only inactive retention is capped at
    // one GiB. Spend at most one quarter of remaining live capacity beyond the
    // protected request. When usage probes are unavailable, advertise zero so
    // no optional retention or new encoding is authorized.
    let cache_budget_bytes = if usage_probe_available {
        protected_feature_bytes.max(
            protected_feature_bytes
                .saturating_add(cache_capacity / 4)
                .min(VISION_CACHE_MAX_BYTES),
        )
    } else {
        0
    };

    // One barrier-delimited encoder block can retain normalized residuals,
    // QKV/rotary projections, attention/output residuals, and FC1/GELU/FC2.
    // 14H + 3I is intentionally conservative for those simultaneously-live
    // linear tensors. The saturating calculation makes invalid/extreme model
    // metadata resolve to the minimum patch batch instead of wrapping large.
    let live_elements_per_patch = hidden_size
        .saturating_mul(14)
        .saturating_add(intermediate_size.saturating_mul(3));
    let peak_bytes_per_patch = live_elements_per_patch
        .saturating_mul(activation_dtype_bytes.max(1))
        .saturating_add(raw_pixel_bytes_per_patch)
        .max(1);
    // Spend no more than one quarter of currently free headroom on one vision
    // layer; the remainder stays available to SDPA workspaces, the input
    // aggregate, text prefill, and non-MLX allocations.
    let transient_budget_bytes = headroom_bytes / 4;
    let raw_patch_budget = transient_budget_bytes / peak_bytes_per_patch;
    let miss_batch_patch_budget = if raw_patch_budget == 0 {
        0
    } else {
        raw_patch_budget
            .clamp(VISION_MISS_BATCH_MIN_PATCHES, VISION_MISS_BATCH_MAX_PATCHES)
            .try_into()
            .unwrap_or(i64::MAX)
    };

    VisionMemoryBudget {
        cap_source,
        effective_cap_bytes,
        safety_reserve_bytes,
        usage_probe_available,
        metal_nonreclaimable_bytes,
        used_memory_bytes,
        output_headroom_bytes,
        projected_output_fits,
        headroom_bytes,
        projected_output_bytes,
        transient_budget_bytes,
        cache_budget_bytes,
        peak_bytes_per_patch,
        miss_batch_patch_budget,
    }
}

fn probe_vision_memory() -> VisionMemorySnapshot {
    let (allocator_active_bytes, allocator_active_probe_ok) =
        probe_u64_with_status(mlx_sys::mlx_get_active_memory);
    let allocator_cache_bytes = probe_u64(mlx_sys::mlx_get_cache_memory);
    #[cfg(target_os = "macos")]
    let (metal_current_allocated_bytes, metal_current_probe_ok) =
        match mlx_paged_attn::metal::MetalState::get() {
            Ok(state) => (state.device.current_allocated_size(), true),
            Err(_) => (0, false),
        };
    #[cfg(not(target_os = "macos"))]
    let (metal_current_allocated_bytes, metal_current_probe_ok) = (0, false);

    VisionMemorySnapshot {
        total_system_memory_bytes: probe_u64(mlx_sys::mlx_total_system_memory),
        mlx_memory_limit_bytes: probe_u64(mlx_sys::mlx_get_memory_limit),
        metal_working_set_bytes: probe_u64(mlx_sys::mlx_max_recommended_working_set_size),
        allocator_active_bytes,
        allocator_active_probe_ok,
        metal_current_allocated_bytes,
        metal_current_probe_ok,
        allocator_cache_bytes,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct VisionFeatureCacheKey {
    image_hash: engine::ImageCacheDigest,
    grid_thw: [i32; 3],
}

struct VisionFeatureCacheEntry {
    features: MxArray,
    bytes: usize,
    lru_generation: u64,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct VisionCacheEviction {
    entries: usize,
    bytes: usize,
}

impl VisionCacheEviction {
    fn merge(&mut self, other: Self) {
        self.entries = self.entries.saturating_add(other.entries);
        self.bytes = self.bytes.saturating_add(other.bytes);
    }
}

/// LRU cache for individually materialized vision encoder embeddings. The
/// content hash makes features reusable when a later cumulative turn appends
/// new images; including the processed grid prevents reuse across a different
/// resize/processor geometry.
pub(crate) struct VisionCacheInner {
    entries: HashMap<VisionFeatureCacheKey, VisionFeatureCacheEntry>,
    /// Monotonically increasing counter for LRU generation tracking.
    generation: u64,
    retained_bytes: usize,
}

impl VisionCacheInner {
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            generation: 0,
            retained_bytes: 0,
        }
    }

    fn get(&mut self, key: &VisionFeatureCacheKey) -> Option<MxArray> {
        self.generation = self.generation.wrapping_add(1);
        let generation = self.generation;
        self.entries.get_mut(key).map(|entry| {
            entry.lru_generation = generation;
            entry.features.clone()
        })
    }

    fn insert(
        &mut self,
        key: VisionFeatureCacheKey,
        features: MxArray,
        bytes: usize,
        protected: &HashSet<VisionFeatureCacheKey>,
        max_bytes: usize,
    ) -> VisionCacheEviction {
        if let Some(previous) = self.entries.remove(&key) {
            self.retained_bytes = self.retained_bytes.saturating_sub(previous.bytes);
        }
        self.generation = self.generation.wrapping_add(1);
        self.entries.insert(
            key,
            VisionFeatureCacheEntry {
                features,
                bytes,
                lru_generation: self.generation,
            },
        );
        self.retained_bytes = self.retained_bytes.saturating_add(bytes);
        self.evict_to_limits(protected, VISION_CACHE_MAX_ENTRIES, max_bytes)
    }

    fn evict_to_limits(
        &mut self,
        protected: &HashSet<VisionFeatureCacheKey>,
        max_entries: usize,
        max_bytes: usize,
    ) -> VisionCacheEviction {
        let mut eviction = VisionCacheEviction::default();
        while self.entries.len() > max_entries || self.retained_bytes > max_bytes {
            let Some(oldest_key) = self
                .entries
                .iter()
                .filter(|(key, _)| !protected.contains(key))
                .min_by_key(|(_, entry)| entry.lru_generation)
                .map(|(key, _)| *key)
            else {
                // The active request itself is the cache floor. Keeping it for
                // the duration of this request avoids evict/re-encode scanning;
                // a later request can evict these entries once unprotected.
                break;
            };
            if let Some(evicted) = self.entries.remove(&oldest_key) {
                self.retained_bytes = self.retained_bytes.saturating_sub(evicted.bytes);
                eviction.entries = eviction.entries.saturating_add(1);
                eviction.bytes = eviction.bytes.saturating_add(evicted.bytes);
            }
        }
        eviction
    }
}

pub(crate) type VisionCache = Arc<Mutex<VisionCacheInner>>;

/// Image token ID used by Qwen3.5-VL

#[derive(Debug, Clone, Copy)]
struct VisionImageRequest {
    key: VisionFeatureCacheKey,
    patch_start: i64,
    patch_count: i64,
    feature_count: i64,
}

#[derive(Debug)]
struct VisionCacheMiss {
    request: VisionImageRequest,
    request_indices: Vec<usize>,
}

fn plan_vision_image_requests(
    grid_data: &[i32],
    per_image_hashes: &[engine::ImageCacheDigest],
    total_patches: i64,
    spatial_merge_size: i32,
) -> Result<Vec<VisionImageRequest>> {
    if spatial_merge_size <= 0 {
        return Err(Error::new(
            Status::InvalidArg,
            format!("spatial_merge_size must be positive, got {spatial_merge_size}"),
        ));
    }
    if grid_data.len() != per_image_hashes.len().saturating_mul(3) {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "vision grid/digest cardinality mismatch: {} grid values for {} image digests",
                grid_data.len(),
                per_image_hashes.len()
            ),
        ));
    }

    let merge = i64::from(spatial_merge_size);
    let merge_area = merge
        .checked_mul(merge)
        .ok_or_else(|| Error::from_reason("vision spatial merge area overflow"))?;
    let mut patch_start = 0i64;
    let mut requests = Vec::with_capacity(per_image_hashes.len());
    for (image_index, image_hash) in per_image_hashes.iter().copied().enumerate() {
        let grid = [
            grid_data[image_index * 3],
            grid_data[image_index * 3 + 1],
            grid_data[image_index * 3 + 2],
        ];
        let [t, h, w] = grid.map(i64::from);
        if t <= 0 || h <= 0 || w <= 0 {
            return Err(Error::new(
                Status::InvalidArg,
                format!("vision grid {image_index} must be positive, got {grid:?}"),
            ));
        }
        if h % merge != 0 || w % merge != 0 {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "vision grid {image_index} {grid:?} is not divisible by spatial merge size {spatial_merge_size}"
                ),
            ));
        }
        let patch_count = t
            .checked_mul(h)
            .and_then(|value| value.checked_mul(w))
            .ok_or_else(|| Error::from_reason("vision patch count overflow"))?;
        let feature_count = patch_count / merge_area;
        requests.push(VisionImageRequest {
            key: VisionFeatureCacheKey {
                image_hash,
                grid_thw: grid,
            },
            patch_start,
            patch_count,
            feature_count,
        });
        patch_start = patch_start
            .checked_add(patch_count)
            .ok_or_else(|| Error::from_reason("vision cumulative patch count overflow"))?;
    }
    if patch_start != total_patches {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "vision grids cover {patch_start} patches but processed pixels contain {total_patches}"
            ),
        ));
    }
    Ok(requests)
}

fn partition_vision_cache_misses(
    misses: &[VisionCacheMiss],
    max_batch_patches: i64,
) -> Vec<std::ops::Range<usize>> {
    let mut batches = Vec::new();
    let mut batch_start = 0usize;
    let mut batch_patches = 0i64;
    for (index, miss) in misses.iter().enumerate() {
        if index > batch_start
            && batch_patches.saturating_add(miss.request.patch_count) > max_batch_patches
        {
            batches.push(batch_start..index);
            batch_start = index;
            batch_patches = 0;
        }
        batch_patches = batch_patches.saturating_add(miss.request.patch_count);
    }
    if batch_start < misses.len() {
        batches.push(batch_start..misses.len());
    }
    batches
}

fn lookup_vision_feature_cache(
    cache: &mut VisionCacheInner,
    requests: &[VisionImageRequest],
) -> (Vec<Option<MxArray>>, Vec<VisionCacheMiss>, usize) {
    let mut ordered_features = vec![None; requests.len()];
    let mut misses: Vec<VisionCacheMiss> = Vec::new();
    let mut miss_by_key: HashMap<VisionFeatureCacheKey, usize> = HashMap::new();
    let mut cache_hits = 0usize;
    for (request_index, request) in requests.iter().copied().enumerate() {
        if let Some(features) = cache.get(&request.key) {
            ordered_features[request_index] = Some(features);
            cache_hits += 1;
        } else if let Some(&miss_index) = miss_by_key.get(&request.key) {
            misses[miss_index].request_indices.push(request_index);
        } else {
            miss_by_key.insert(request.key, misses.len());
            misses.push(VisionCacheMiss {
                request,
                request_indices: vec![request_index],
            });
        }
    }
    (ordered_features, misses, cache_hits)
}

fn vision_array_bytes(array: &MxArray) -> Result<usize> {
    let elements = array.shape()?.iter().try_fold(1usize, |product, &dim| {
        let dim = usize::try_from(dim).map_err(|_| {
            Error::from_reason("vision feature shape contains a negative dimension")
        })?;
        product
            .checked_mul(dim)
            .ok_or_else(|| Error::from_reason("vision feature element count overflow"))
    })?;
    elements
        .checked_mul(array.dtype()?.byte_size())
        .ok_or_else(|| Error::from_reason("vision feature byte count overflow"))
}

fn projected_vision_feature_bytes(
    requests: &[VisionImageRequest],
    output_size: u64,
    dtype_bytes: u64,
) -> (u64, u64) {
    let bytes_for = |request: &VisionImageRequest| {
        u64::try_from(request.feature_count)
            .unwrap_or(u64::MAX)
            .saturating_mul(output_size)
            .saturating_mul(dtype_bytes)
    };
    let request_bytes = requests.iter().fold(0u64, |bytes, request| {
        bytes.saturating_add(bytes_for(request))
    });
    let mut accounted_keys = HashSet::with_capacity(requests.len());
    let protected_bytes = requests
        .iter()
        .filter(|request| accounted_keys.insert(request.key))
        .fold(0u64, |bytes, request| {
            bytes.saturating_add(bytes_for(request))
        });
    (request_bytes, protected_bytes)
}

/// Shared VLM prefill steps 1-3: per-image vision cache lookup, bounded vision
/// encoder miss batches, embedding merge, and M-RoPE position computation.
///
/// Returns (inputs_embeds, position_ids, rope_deltas) ready for the
/// language model forward pass. Used by both dense and MoE VLM prefill.
#[allow(clippy::too_many_arguments)]
pub(crate) fn vlm_prepare_vision_features(
    input_ids: &MxArray,
    per_image_hashes: &[engine::ImageCacheDigest],
    pre_processed: &ProcessedImages,
    vision_encoder: &QwenVisionEncoder,
    spatial_merge_size: i32,
    text_model_embedding: &Embedding,
    generation_stream: Stream,
    vision_cache: &VisionCache,
) -> Result<VisionMerge> {
    // Build the text-embedding graph once up front. The packed backend gathers
    // and dequantizes only the referenced rows; keeping this handle also gives
    // the vision cache planner the exact merge dtype without materializing the
    // full vocabulary table.
    let text_embeds = {
        let _stream_ctx = StreamContext::new(generation_stream);
        text_model_embedding.forward(input_ids)?
    };

    let grid = pre_processed.grid_thw();
    let grid_data = grid.to_int32()?;
    let pv = pre_processed.pixel_values();
    let pv_shape = pv.shape()?;
    if pv_shape.len() != 4 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "processed vision pixels must have rank 4 [patches,C,H,W], got {:?}",
                pv_shape.as_ref()
            ),
        ));
    }
    // MLX's allocator cache is reclaimable, unlike active arrays. Drain it
    // before taking the live snapshot and do not double-count any reported
    // residue against headroom.
    crate::array::clear_cache();
    let memory_snapshot = probe_vision_memory();
    let (vision_hidden_size, vision_intermediate_size, vision_output_size) =
        vision_encoder.memory_widths();
    let activation_dtype_bytes = u64::try_from(pv.dtype()?.byte_size()).unwrap_or(u64::MAX);
    let cache_feature_dtype = text_embeds.dtype()?;
    let cache_feature_dtype_bytes =
        u64::try_from(cache_feature_dtype.byte_size()).unwrap_or(u64::MAX);
    let raw_pixel_bytes_per_patch = pv_shape[1..]
        .iter()
        .fold(activation_dtype_bytes, |bytes, &dimension| {
            bytes.saturating_mul(u64::try_from(dimension).unwrap_or(u64::MAX))
        });
    let processed_pixel_bytes = vision_array_bytes(&pv)?;
    let requests = plan_vision_image_requests(
        &grid_data,
        per_image_hashes,
        pv_shape[0],
        spatial_merge_size,
    )?;
    let protected_keys: HashSet<VisionFeatureCacheKey> =
        requests.iter().map(|request| request.key).collect();
    let (
        mut ordered_features,
        misses,
        cache_hits,
        cache_entries_before,
        cache_bytes_before,
        cache_entries_after_prune,
        cache_bytes_after_prune,
        initial_eviction,
        memory_budget,
    ) = {
        let mut cache = vision_cache
            .lock()
            .map_err(|_| Error::from_reason("Vision cache mutex poisoned"))?;
        let (ordered_features, misses, cache_hits) =
            lookup_vision_feature_cache(&mut cache, &requests);
        let cache_entries_before = cache.entries.len();
        let cache_bytes_before = cache.retained_bytes;
        let (request_feature_bytes, protected_feature_bytes) = projected_vision_feature_bytes(
            &requests,
            vision_output_size,
            cache_feature_dtype_bytes,
        );
        let projected_miss_output_bytes = misses.iter().fold(0u64, |bytes, miss| {
            let feature_count = u64::try_from(miss.request.feature_count).unwrap_or(u64::MAX);
            bytes.saturating_add(
                feature_count
                    .saturating_mul(vision_output_size)
                    .saturating_mul(cache_feature_dtype_bytes),
            )
        });
        let projected_concat_output_bytes = if requests.len() > 1 {
            request_feature_bytes
        } else {
            0
        };
        let projected_output_bytes =
            projected_miss_output_bytes.saturating_add(projected_concat_output_bytes);
        let memory_budget = resolve_vision_memory_budget(
            memory_snapshot,
            u64::try_from(cache.retained_bytes).unwrap_or(u64::MAX),
            protected_feature_bytes,
            vision_hidden_size,
            vision_intermediate_size,
            activation_dtype_bytes,
            raw_pixel_bytes_per_patch,
            projected_output_bytes,
        );
        // Prune after touches even when every request image is a cache hit. A
        // previous oversized protected request is allowed to exceed the floor;
        // once entries are no longer active they must become evictable.
        let initial_eviction = cache.evict_to_limits(
            &protected_keys,
            VISION_CACHE_MAX_ENTRIES,
            usize::try_from(memory_budget.cache_budget_bytes).unwrap_or(usize::MAX),
        );
        (
            ordered_features,
            misses,
            cache_hits,
            cache_entries_before,
            cache_bytes_before,
            cache.entries.len(),
            cache.retained_bytes,
            initial_eviction,
            memory_budget,
        )
    };
    if initial_eviction.entries > 0 {
        // Dropping MLX arrays moves their buffers into the allocator cache.
        // Release those buffers only after the mutex guard has been dropped.
        crate::array::clear_cache();
    }

    // This applies even when every image is a cache hit: joining multiple
    // cached feature arrays still allocates a request-sized output. Without an
    // explicit fit check, saturating the transient headroom to zero would only
    // reject cache misses and an all-hit request could overcommit unified
    // memory during the final concatenate.
    if !memory_budget.projected_output_fits {
        tracing::error!(
            target: "mlx_core::inference",
            event = "vision_output_headroom_insufficient",
            projected_output_bytes = memory_budget.projected_output_bytes,
            output_headroom_bytes = memory_budget.output_headroom_bytes,
            effective_cap_bytes = memory_budget.effective_cap_bytes,
            allocator_active_bytes = memory_snapshot.allocator_active_bytes,
            metal_current_allocated_bytes = memory_snapshot.metal_current_allocated_bytes,
            metal_nonreclaimable_bytes = memory_budget.metal_nonreclaimable_bytes,
            usage_probe_available = memory_budget.usage_probe_available,
            used_memory_bytes = memory_budget.used_memory_bytes,
            safety_reserve_bytes = memory_budget.safety_reserve_bytes,
            "Qwen3.5 vision request outputs cannot fit in the available memory budget"
        );
        return Err(Error::from_reason(format!(
            "insufficient MLX memory headroom for Qwen3.5 vision outputs: the request needs {} bytes for new/collated features, but only {} bytes remain after live allocations and the safety reserve",
            memory_budget.projected_output_bytes, memory_budget.output_headroom_bytes
        )));
    }

    let largest_miss_patches = misses
        .iter()
        .map(|miss| miss.request.patch_count)
        .max()
        .unwrap_or(0);
    let largest_miss_peak_bytes = u64::try_from(largest_miss_patches)
        .unwrap_or(u64::MAX)
        .saturating_mul(memory_budget.peak_bytes_per_patch);
    let largest_miss_exceeds_patch_budget =
        largest_miss_patches > memory_budget.miss_batch_patch_budget;
    if !misses.is_empty()
        && (largest_miss_peak_bytes > memory_budget.transient_budget_bytes
            || largest_miss_exceeds_patch_budget)
    {
        tracing::error!(
            target: "mlx_core::inference",
            event = "vision_memory_headroom_insufficient",
            largest_image_patches = largest_miss_patches,
            dynamic_patch_budget = memory_budget.miss_batch_patch_budget,
            exceeds_dynamic_patch_budget = largest_miss_exceeds_patch_budget,
            estimated_peak_bytes = largest_miss_peak_bytes,
            transient_budget_bytes = memory_budget.transient_budget_bytes,
            effective_cap_bytes = memory_budget.effective_cap_bytes,
            allocator_active_bytes = memory_snapshot.allocator_active_bytes,
            metal_current_allocated_bytes = memory_snapshot.metal_current_allocated_bytes,
            metal_nonreclaimable_bytes = memory_budget.metal_nonreclaimable_bytes,
            usage_probe_available = memory_budget.usage_probe_available,
            used_memory_bytes = memory_budget.used_memory_bytes,
            safety_reserve_bytes = memory_budget.safety_reserve_bytes,
            "Qwen3.5 vision image cannot fit in the available transient memory budget"
        );
        return Err(Error::from_reason(format!(
            "insufficient MLX memory headroom for Qwen3.5 vision image: largest miss has {largest_miss_patches} patches with an estimated {}-byte layer peak, but the safe batch limit is {} patches / {} bytes after the safety reserve",
            largest_miss_peak_bytes,
            memory_budget.miss_batch_patch_budget,
            memory_budget.transient_budget_bytes
        )));
    }
    // Self-attention cannot split one image across batches, so the check above
    // rejects any image above the dynamic/hard limit rather than silently
    // raising it and overcommitting memory.
    let resolved_miss_batch_patch_budget = memory_budget.miss_batch_patch_budget.max(1);
    let miss_batches = partition_vision_cache_misses(&misses, resolved_miss_batch_patch_budget);
    tracing::info!(
        target: "mlx_core::inference",
        event = "vision_feature_cache_plan",
        image_count = requests.len(),
        cache_hits,
        cache_misses = requests.len().saturating_sub(cache_hits),
        unique_cache_misses = misses.len(),
        miss_batches = miss_batches.len(),
        patch_count = pv_shape[0],
        processed_pixel_bytes,
        cache_entries_before,
        cache_bytes_before,
        cache_entries_after_prune,
        cache_bytes_after_prune,
        evicted_entries = initial_eviction.entries,
        evicted_bytes = initial_eviction.bytes,
        total_system_memory_bytes = memory_snapshot.total_system_memory_bytes,
        mlx_memory_limit_bytes = memory_snapshot.mlx_memory_limit_bytes,
        metal_working_set_bytes = memory_snapshot.metal_working_set_bytes,
        effective_cap_bytes = memory_budget.effective_cap_bytes,
        cap_source = memory_budget.cap_source.as_str(),
        allocator_active_bytes = memory_snapshot.allocator_active_bytes,
        allocator_active_probe_ok = memory_snapshot.allocator_active_probe_ok,
        metal_current_allocated_bytes = memory_snapshot.metal_current_allocated_bytes,
        metal_current_probe_ok = memory_snapshot.metal_current_probe_ok,
        metal_nonreclaimable_bytes = memory_budget.metal_nonreclaimable_bytes,
        usage_probe_available = memory_budget.usage_probe_available,
        used_memory_bytes = memory_budget.used_memory_bytes,
        allocator_cache_bytes = memory_snapshot.allocator_cache_bytes,
        allocator_cache_counted = false,
        safety_reserve_bytes = memory_budget.safety_reserve_bytes,
        output_headroom_bytes = memory_budget.output_headroom_bytes,
        projected_output_fits = memory_budget.projected_output_fits,
        headroom_bytes = memory_budget.headroom_bytes,
        projected_output_bytes = memory_budget.projected_output_bytes,
        transient_budget_bytes = memory_budget.transient_budget_bytes,
        cache_budget_bytes = memory_budget.cache_budget_bytes,
        peak_bytes_per_patch = memory_budget.peak_bytes_per_patch,
        dynamic_miss_batch_patch_budget = memory_budget.miss_batch_patch_budget,
        resolved_miss_batch_patch_budget,
        largest_miss_patches,
        vision_hidden_size,
        vision_intermediate_size,
        vision_output_size,
        activation_dtype_bytes,
        cache_feature_dtype_bytes,
        raw_pixel_bytes_per_patch,
        "Qwen3.5 per-image vision feature cache planned"
    );

    for (batch_index, miss_range) in miss_batches.into_iter().enumerate() {
        let batch_misses = &misses[miss_range];
        let batch_patch_count = batch_misses
            .iter()
            .try_fold(0i64, |sum, miss| sum.checked_add(miss.request.patch_count))
            .ok_or_else(|| Error::from_reason("vision miss-batch patch count overflow"))?;
        let batch_feature_count = batch_misses
            .iter()
            .try_fold(0i64, |sum, miss| {
                sum.checked_add(miss.request.feature_count)
            })
            .ok_or_else(|| Error::from_reason("vision miss-batch feature count overflow"))?;

        let materialize_start = std::time::Instant::now();
        tracing::info!(
            target: "mlx_core::inference",
            event = "vision_feature_miss_batch_start",
            batch_index,
            image_count = batch_misses.len(),
            patch_count = batch_patch_count,
            "Qwen3.5 vision cache-miss batch started"
        );
        let independent_features = {
            let mut pixel_parts = Vec::with_capacity(batch_misses.len());
            let mut batch_grid_data = Vec::with_capacity(batch_misses.len() * 3);
            for miss in batch_misses {
                pixel_parts.push(pv.slice_axis(
                    0,
                    miss.request.patch_start,
                    miss.request.patch_start + miss.request.patch_count,
                )?);
                batch_grid_data.extend_from_slice(&miss.request.key.grid_thw);
            }
            let batch_pixels = if pixel_parts.len() == 1 {
                pixel_parts.remove(0)
            } else {
                let refs: Vec<&MxArray> = pixel_parts.iter().collect();
                MxArray::concatenate_many(refs, Some(0))?
            };
            let batch_grid =
                MxArray::from_int32(&batch_grid_data, &[batch_misses.len() as i64, 3])?;
            let batch_pixels = batch_pixels.reshape(&[
                1,
                batch_patch_count,
                pv_shape[1],
                pv_shape[2],
                pv_shape[3],
            ])?;
            let batch_features = {
                let _stream_ctx = StreamContext::new(generation_stream);
                vision_encoder.forward(&batch_pixels, &batch_grid)?
            };
            let materialize_context = format!("qwen3_5_vision_miss_batch_{batch_index}");
            if let Err(error) =
                MxArray::eval_arrays_with_context(&[&batch_features], &materialize_context)
            {
                tracing::error!(
                    target: "mlx_core::inference",
                    event = "vision_feature_miss_batch_failed",
                    batch_index,
                    image_count = batch_misses.len(),
                    patch_count = batch_patch_count,
                    elapsed_ms = elapsed_ms(materialize_start),
                    error = %error,
                    "Qwen3.5 vision cache-miss batch failed"
                );
                return Err(Error::from_reason(format!(
                    "Qwen3.5 vision miss batch {batch_index} materialization failed: {error}"
                )));
            }
            if batch_features.shape_at(0)? != batch_feature_count {
                return Err(Error::from_reason(format!(
                    "Qwen3.5 vision miss batch {batch_index} produced {} features, expected {batch_feature_count}",
                    batch_features.shape_at(0)?
                )));
            }
            // The merge path casts vision output to the text embedding dtype
            // on every turn. Do that elementwise cast once before splitting so
            // retained cache entries use the same values at half the storage
            // when f32 vision activations feed bf16 text embeddings.
            let cacheable_batch_features = if batch_features.dtype()? != cache_feature_dtype {
                let cast = batch_features.astype(cache_feature_dtype)?;
                let cast_context = format!("qwen3_5_vision_miss_batch_{batch_index}_cache_cast");
                MxArray::eval_arrays_with_context(&[&cast], &cast_context).map_err(|error| {
                    Error::from_reason(format!(
                        "Qwen3.5 vision miss batch {batch_index} cache cast failed: {error}"
                    ))
                })?;
                cast
            } else {
                batch_features.clone()
            };

            // Deep-copy each image slice before caching it. MLX's ordinary
            // `copy()` shares storage; retaining that view would pin the whole
            // microbatch allocation and invalidate byte accounting.
            let mut independent_features = Vec::with_capacity(batch_misses.len());
            let mut feature_start = 0i64;
            for miss in batch_misses {
                let feature_end = feature_start + miss.request.feature_count;
                independent_features.push(
                    cacheable_batch_features
                        .slice_axis(0, feature_start, feature_end)?
                        .deep_copy()?,
                );
                feature_start = feature_end;
            }
            let independent_refs: Vec<&MxArray> = independent_features.iter().collect();
            let split_context = format!("qwen3_5_vision_miss_batch_{batch_index}_split");
            MxArray::eval_arrays_with_context(&independent_refs, &split_context).map_err(
                |error| {
                    Error::from_reason(format!(
                        "Qwen3.5 vision miss batch {batch_index} split materialization failed: {error}"
                    ))
                },
            )?;
            independent_features
            // `batch_features`, `batch_pixels`, grids, and slice views all drop
            // here, before the allocator cache is drained below.
        };
        crate::array::clear_cache();

        let (cache_entries, cache_bytes, batch_eviction) = {
            let mut cache = vision_cache
                .lock()
                .map_err(|_| Error::from_reason("Vision cache mutex poisoned"))?;
            let mut batch_eviction = VisionCacheEviction::default();
            for (miss, features) in batch_misses.iter().zip(independent_features) {
                let bytes = vision_array_bytes(&features)?;
                batch_eviction.merge(cache.insert(
                    miss.request.key,
                    features.clone(),
                    bytes,
                    &protected_keys,
                    usize::try_from(memory_budget.cache_budget_bytes).unwrap_or(usize::MAX),
                ));
                for &request_index in &miss.request_indices {
                    ordered_features[request_index] = Some(features.clone());
                }
            }
            (cache.entries.len(), cache.retained_bytes, batch_eviction)
        };
        // Insertion may evict inactive arrays. Their buffers enter MLX's free
        // pool when the mutex scope drops, so drain once more outside the lock.
        crate::array::clear_cache();
        tracing::info!(
            target: "mlx_core::inference",
            event = "vision_feature_miss_batch_done",
            batch_index,
            image_count = batch_misses.len(),
            patch_count = batch_patch_count,
            feature_tokens = batch_feature_count,
            cache_entries,
            cache_bytes,
            evicted_entries = batch_eviction.entries,
            evicted_bytes = batch_eviction.bytes,
            elapsed_ms = elapsed_ms(materialize_start),
            "Qwen3.5 vision cache-miss batch completed"
        );
    }

    let mut ordered_features: Vec<MxArray> = ordered_features
        .into_iter()
        .enumerate()
        .map(|(image_index, features)| {
            features.ok_or_else(|| {
                Error::from_reason(format!(
                    "Qwen3.5 vision feature missing for request image {image_index}"
                ))
            })
        })
        .collect::<Result<_>>()?;
    let vision_features = if ordered_features.len() == 1 {
        ordered_features.remove(0)
    } else {
        let refs: Vec<&MxArray> = ordered_features.iter().collect();
        MxArray::concatenate_many(refs, Some(0))?
    };

    let inputs_embeds = {
        let _stream_ctx = StreamContext::new(generation_stream);
        let embed_dtype = text_embeds.dtype()?;
        let vf_cast = if vision_features.dtype()? != embed_dtype {
            vision_features.astype(embed_dtype)?
        } else {
            vision_features
        };
        merge_input_ids_with_image_features(IMAGE_TOKEN_ID, &vf_cast, &text_embeds, input_ids)?
    };

    let (position_ids, rope_deltas) =
        get_rope_index(input_ids, Some(&grid), spatial_merge_size, IMAGE_TOKEN_ID)?;

    tracing::debug!(
        "VLM prefill: seq_len={}, rope_deltas={}",
        inputs_embeds.shape_at(1)?,
        rope_deltas
    );

    Ok(VisionMerge {
        inputs_embeds,
        position_ids,
        rope_deltas,
    })
}

#[cfg(test)]
#[path = "cache_tests.rs"]
mod tests;
