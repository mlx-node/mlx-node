//! Vision feature cache + memory budget, and the VLM prompt/feature helpers.

use super::*;

/// Hard cap on inactive per-image feature entries retained by the vision LRU.
/// The active request is protected from eviction even when it is larger than
/// this cap; this prevents scan thrash while one large prompt is being built.
pub(crate) const VISION_CACHE_MAX_ENTRIES: usize = 128;

pub(super) const VISION_GIB: u64 = 1024 * 1024 * 1024;

/// Used only when neither MLX nor Metal can report a usable cap. Do not derive
/// this from physical RAM: unified-memory pressure and the configured MLX limit
/// are the constraints that matter to inference.
const VISION_FALLBACK_EFFECTIVE_CAP_BYTES: u64 = 8 * VISION_GIB;
const VISION_SAFETY_RESERVE_MIN_BYTES: u64 = 2 * VISION_GIB;
const VISION_SAFETY_RESERVE_MAX_BYTES: u64 = 16 * VISION_GIB;
pub(super) const VISION_CACHE_MAX_BYTES: u64 = VISION_GIB;
const VISION_MISS_BATCH_MIN_PATCHES: u64 = 1;
const VISION_MISS_BATCH_MAX_PATCHES: u64 = 32 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum VisionMemoryCapSource {
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
pub(super) struct VisionMemorySnapshot {
    pub(super) total_system_memory_bytes: u64,
    pub(super) mlx_memory_limit_bytes: u64,
    pub(super) metal_working_set_bytes: u64,
    pub(super) allocator_active_bytes: u64,
    pub(super) allocator_active_probe_ok: bool,
    /// Process-wide Metal allocation snapshot. This includes MLX allocations
    /// plus external LayerKVPool buffers that MLX's active counter omits.
    pub(super) metal_current_allocated_bytes: u64,
    pub(super) metal_current_probe_ok: bool,
    /// Informational only. The caller drains MLX's reclaimable allocator cache
    /// immediately before probing, and this value is deliberately not charged
    /// against headroom. Any residue is logged for diagnosis.
    pub(super) allocator_cache_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct VisionMemoryBudget {
    pub(super) cap_source: VisionMemoryCapSource,
    pub(super) effective_cap_bytes: u64,
    safety_reserve_bytes: u64,
    pub(super) usage_probe_available: bool,
    pub(super) metal_nonreclaimable_bytes: u64,
    pub(super) used_memory_bytes: u64,
    pub(super) output_headroom_bytes: u64,
    pub(super) projected_output_fits: bool,
    pub(super) headroom_bytes: u64,
    projected_output_bytes: u64,
    transient_budget_bytes: u64,
    pub(super) cache_budget_bytes: u64,
    pub(super) peak_bytes_per_patch: u64,
    pub(super) miss_batch_patch_budget: i64,
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

pub(super) fn resolve_vision_memory_budget(
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
    pub(super) image_hash: engine::ImageCacheDigest,
    pub(super) grid_thw: [i32; 3],
}

pub(super) struct VisionFeatureCacheEntry {
    features: MxArray,
    bytes: usize,
    lru_generation: u64,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct VisionCacheEviction {
    pub(super) entries: usize,
    pub(super) bytes: usize,
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
    pub(super) entries: HashMap<VisionFeatureCacheKey, VisionFeatureCacheEntry>,
    /// Monotonically increasing counter for LRU generation tracking.
    generation: u64,
    pub(super) retained_bytes: usize,
}

impl VisionCacheInner {
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            generation: 0,
            retained_bytes: 0,
        }
    }

    pub(super) fn get(&mut self, key: &VisionFeatureCacheKey) -> Option<MxArray> {
        self.generation = self.generation.wrapping_add(1);
        let generation = self.generation;
        self.entries.get_mut(key).map(|entry| {
            entry.lru_generation = generation;
            entry.features.clone()
        })
    }

    pub(super) fn insert(
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

    pub(super) fn evict_to_limits(
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
pub(crate) const IMAGE_TOKEN_ID: i32 = 248056;

/// Extract all raw image bytes from chat messages.
pub(crate) fn extract_images_from_messages(messages: &[ChatMessage]) -> Vec<Vec<u8>> {
    let mut all_images: Vec<Vec<u8>> = Vec::new();
    for msg in messages {
        if let Some(ref images) = msg.images {
            for img in images {
                all_images.push(img.to_vec());
            }
        }
    }
    all_images
}

/// Compute the per-image merged-token count from a processed grid_thw
/// array. Each entry is the number of `IMAGE_TOKEN_ID` slots that image
/// must occupy in the prompt so the vision embeddings align 1:1 with the
/// corresponding token positions.
pub(crate) fn compute_image_token_counts_per_image(
    grid: &MxArray,
    spatial_merge_size: i32,
) -> Result<Vec<usize>> {
    grid.eval();
    let grid_data = grid.to_int32()?;
    let mut counts = Vec::with_capacity(grid_data.len() / 3);
    for i in 0..(grid_data.len() / 3) {
        let t = grid_data[i * 3];
        let h = grid_data[i * 3 + 1];
        let w = grid_data[i * 3 + 2];
        counts.push(merged_image_token_count(t, h, w, spatial_merge_size)?);
    }
    Ok(counts)
}

/// Return the exact length produced by [`inject_image_placeholders`] without
/// allocating the expanded token vector.
pub(crate) fn expanded_image_prompt_len(
    tokens: &[u32],
    per_image_token_counts: &[usize],
) -> Result<usize> {
    let total = per_image_token_counts
        .iter()
        .try_fold(0usize, |sum, count| {
            sum.checked_add(*count)
                .ok_or_else(|| Error::from_reason("expanded image prompt length overflow"))
        })?;
    if total == 0 {
        return Ok(tokens.len());
    }

    let existing = tokens
        .iter()
        .filter(|&&token| token == IMAGE_TOKEN_ID as u32)
        .count();
    if existing == per_image_token_counts.len() {
        return tokens
            .len()
            .checked_add(total)
            .and_then(|len| len.checked_sub(existing))
            .ok_or_else(|| Error::from_reason("expanded image prompt length overflow"));
    }
    if existing == total {
        return Ok(tokens.len());
    }

    Err(image_placeholder_shape_error(
        existing,
        per_image_token_counts.len(),
        total,
    ))
}

/// CPU-only prompt planner shared by the dense and MoE NAPI wrappers.
///
/// It reads encoded image dimensions and applies the loaded Qwen processor's
/// smart-resize geometry, but never creates normalized pixel tensors, MLX
/// arrays, vision features, or KV state.
pub(crate) fn plan_expanded_image_prompt_len(
    image_processor: &Qwen35VLImageProcessor,
    spatial_merge_size: i32,
    tokens: &[u32],
    images: &[Vec<u8>],
) -> Result<usize> {
    let image_refs: Vec<&[u8]> = images.iter().map(Vec::as_slice).collect();
    let counts = image_processor.plan_merged_token_counts(&image_refs, spatial_merge_size)?;
    expanded_image_prompt_len(tokens, &counts)
}

/// Ensure the tokenized prompt contains the right number of
/// `IMAGE_TOKEN_ID` placeholders — one per vision patch, in the order
/// produced by the chat template.
///
/// Two input shapes are accepted:
///
/// 1. **Template emitted one `<|image_pad|>` per image** (the proper
///    Qwen VLM shape, produced by
///    `tokenizer::serialize_message_for_jinja` when the user turn
///    carries images). Each placeholder is expanded in-place to its
///    image's grid count. This keeps the vision tokens inside the user
///    turn — `get_rope_index` builds correct M-RoPE positions and the
///    model attends to the image in-context.
///
/// 2. **Template already emitted the fully expanded count** (non-Qwen
///    templates that inline the full patch run). Pass through unchanged.
///
///
/// Missing or mismatched markers are rejected. The checkpoint's chat
/// template owns marker placement; inserting a fallback run after BOS would
/// move vision tokens outside the user turn and produce invalid M-RoPE
/// positions.
pub(crate) fn inject_image_placeholders(
    tokens: &[u32],
    per_image_token_counts: &[usize],
) -> Result<Vec<u32>> {
    let total = per_image_token_counts
        .iter()
        .try_fold(0usize, |sum, count| {
            sum.checked_add(*count)
                .ok_or_else(|| Error::from_reason("expanded image prompt length overflow"))
        })?;
    if total == 0 {
        return Ok(tokens.to_vec());
    }
    let existing = tokens
        .iter()
        .filter(|&&t| t == IMAGE_TOKEN_ID as u32)
        .count();

    if existing == per_image_token_counts.len() {
        // Case 1 — one placeholder per image; expand each in place to
        // its grid count. Capacity pre-sized to the final length so no
        // reallocations.
        let mut new_tokens: Vec<u32> = Vec::with_capacity(tokens.len() + total - existing);
        let mut img_iter = per_image_token_counts.iter().copied();
        for &t in tokens {
            if t == IMAGE_TOKEN_ID as u32 {
                match img_iter.next() {
                    Some(count) => {
                        new_tokens.extend(std::iter::repeat_n(IMAGE_TOKEN_ID as u32, count));
                    }
                    None => {
                        return Err(Error::from_reason(
                            "image placeholder expansion exhausted its validated image counts",
                        ));
                    }
                }
            } else {
                new_tokens.push(t);
            }
        }
        return Ok(new_tokens);
    }

    if existing == total {
        // Case 2 — the checkpoint template already emitted one marker per
        // vision patch.
        return Ok(tokens.to_vec());
    }

    Err(image_placeholder_shape_error(
        existing,
        per_image_token_counts.len(),
        total,
    ))
}

fn image_placeholder_shape_error(
    existing: usize,
    image_count: usize,
    expanded_count: usize,
) -> Error {
    if existing == 0 {
        Error::from_reason(format!(
            "model chat template emitted no image placeholder tokens for {image_count} image(s); \
expected {image_count} unexpanded marker(s) or {expanded_count} already-expanded marker(s)"
        ))
    } else {
        Error::from_reason(format!(
            "model chat template emitted {existing} image placeholder token(s) for {image_count} \
image(s); expected {image_count} unexpanded marker(s) or {expanded_count} already-expanded marker(s)"
        ))
    }
}

/// Compute M-RoPE position IDs for VLM
///
/// Text tokens get sequential positions [0, 1, 2, ...].
/// Image tokens get 2D spatial positions based on grid_thw.
///
/// Returns (position_ids [3, B, T], rope_deltas)
pub(crate) fn get_rope_index(
    input_ids: &MxArray,
    image_grid_thw: Option<&MxArray>,
    spatial_merge_size: i32,
    image_token_id: i32,
) -> Result<(MxArray, i64)> {
    let shape = input_ids.shape()?;
    let batch_size = shape[0];
    let seq_len = shape[1];

    // If no images, use simple sequential positions
    let Some(grid_thw) = image_grid_thw else {
        let pos = MxArray::arange(0.0, seq_len as f64, Some(1.0), None)?;
        let pos = pos.reshape(&[1, 1, seq_len])?;
        let position_ids = MxArray::tile(&pos, &[3, batch_size as i32, 1])?;
        return Ok((position_ids, 0));
    };
    let input_ids_data = input_ids.to_int32()?;
    grid_thw.eval();
    let grid_data = grid_thw.to_int32()?;

    let mut all_position_ids: Vec<Vec<i64>> = vec![Vec::new(); 3];

    for batch_idx in 0..batch_size as usize {
        let start = batch_idx * seq_len as usize;
        let end = start + seq_len as usize;
        let batch_tokens: Vec<i32> = input_ids_data[start..end].to_vec();

        // Scan `batch_tokens` for maximal contiguous runs of
        // `image_token_id`. After the tokenizer fix that serialises
        // one `<|image_pad|>` per image inline in the user turn and
        // `inject_image_placeholders` expands each marker in place,
        // the prompt can carry MULTIPLE separated image runs when
        // history is replayed (e.g. two image-bearing user turns
        // joined by an assistant reply). Flattening
        // `positions[0]`..`positions[last]` into one span would skip
        // every interior text token and blow up the reshape below.
        let mut image_runs: Vec<(usize, usize)> = Vec::new();
        {
            let mut i = 0;
            while i < batch_tokens.len() {
                if batch_tokens[i] == image_token_id {
                    let start = i;
                    while i < batch_tokens.len() && batch_tokens[i] == image_token_id {
                        i += 1;
                    }
                    image_runs.push((start, i));
                } else {
                    i += 1;
                }
            }
        }

        if image_runs.is_empty() {
            for i in 0..seq_len {
                all_position_ids[0].push(i);
                all_position_ids[1].push(i);
                all_position_ids[2].push(i);
            }
            continue;
        }

        let num_images = grid_data.len() / 3;
        if num_images == 0 || grid_data.len() % 3 != 0 {
            return Err(Error::new(
                Status::InvalidArg,
                format!("grid_data must have 3N elements, got {}", grid_data.len()),
            ));
        }

        // Calculate token info for each image
        let mut image_token_info: Vec<(i64, i64, i64, usize)> = Vec::new();
        let mut total_expected_tokens = 0usize;

        for img_idx in 0..num_images {
            let t = grid_data[img_idx * 3] as i64;
            let h = grid_data[img_idx * 3 + 1] as i64;
            let w = grid_data[img_idx * 3 + 2] as i64;

            let llm_grid_t = t;
            let llm_grid_h = h / spatial_merge_size as i64;
            let llm_grid_w = w / spatial_merge_size as i64;
            let num_tokens = (llm_grid_t * llm_grid_h * llm_grid_w) as usize;

            image_token_info.push((llm_grid_t, llm_grid_h, llm_grid_w, num_tokens));
            total_expected_tokens += num_tokens;
        }

        let total_image_tokens: usize = image_runs.iter().map(|(s, e)| e - s).sum();
        if total_expected_tokens != total_image_tokens {
            return Err(Error::new(
                Status::GenericFailure,
                format!(
                    "Image token count mismatch: expected {} from grid, found {} in prompt",
                    total_expected_tokens, total_image_tokens,
                ),
            ));
        }

        // Two token layouts are valid here:
        //
        //  (a) N runs, one per image — the proper Qwen VLM shape after
        //      the tokenizer serialiser emits a `{type:"image"}` part
        //      per image and `inject_image_placeholders` expands each
        //      marker in place. Per-run length must match its grid.
        //
        //  (b) 1 big run whose length equals the grids' total — a
        //      checkpoint template may emit the fully expanded markers
        //      as one contiguous span. No text gap sits between images
        //      in this layout, so the position walk collapses consecutive
        //      sub-runs into one span without emitting interior text.
        //
        // We canonicalise both into a `per_image_offsets: Vec<(start,
        // grid_info)>` list of length `num_images` and feed it to the
        // position walk below. Any other shape is ambiguous (we'd have
        // to guess which grid goes with which run) — reject it.
        let per_image_offsets: Vec<(usize, (i64, i64, i64, usize))> = if image_runs.len()
            == num_images
        {
            // Case (a): validate per-run length, then pair by ordinal.
            for (run_idx, (run_start, run_end)) in image_runs.iter().enumerate() {
                let expected = image_token_info[run_idx].3;
                let actual = run_end - run_start;
                if expected != actual {
                    return Err(Error::new(
                        Status::GenericFailure,
                        format!(
                            "Image run {run_idx} has {actual} placeholder tokens but its grid expects {expected}",
                        ),
                    ));
                }
            }
            image_runs
                .iter()
                .zip(image_token_info.iter().copied())
                .map(|((start, _), info)| (*start, info))
                .collect()
        } else if image_runs.len() == 1 {
            // Case (b): already-expanded contiguous span — synthesise
            // per-image start offsets by walking `image_token_info`
            // lengths from the single run's start. Total was already
            // validated above.
            let big_start = image_runs[0].0;
            let mut offsets = Vec::with_capacity(num_images);
            let mut cursor = big_start;
            for info in image_token_info.iter().copied() {
                offsets.push((cursor, info));
                cursor += info.3;
            }
            offsets
        } else {
            return Err(Error::new(
                Status::GenericFailure,
                format!(
                    "Image run layout mismatch: prompt carries {} contiguous image-token runs but {} images \
                     were processed; expected either one run per image or a single contiguous fallback run \
                     containing every image's tokens.",
                    image_runs.len(),
                    num_images,
                ),
            ));
        };

        // End of the last image token in the token stream — everything
        // beyond is trailing text. For case (a) this is the last run's
        // end; for case (b) it's the shared run's end. In both cases
        // it equals the end of the validated non-empty `image_runs` list.
        let last_image_end = image_runs
            .last()
            .ok_or_else(|| Error::from_reason("image run validation lost its non-empty run"))?
            .1;

        // Emit positions by walking the sequence: text gap, image,
        // text gap, image, … final text gap. `current_pos` carries the
        // M-RoPE counter forward across both text and image segments so
        // every token gets a monotonically non-decreasing position id
        // in each axis. Synthesised case-(b) sub-runs sit back-to-back
        // so their text-gap loops iterate zero times between them —
        // the walk collapses naturally.
        let mut cursor: usize = 0;
        let mut current_pos: i64 = 0;

        for (run_start, info) in per_image_offsets.iter().copied() {
            // Text gap before this image run (zero-length for adjacent
            // case-(b) sub-runs after the first).
            for _ in cursor..run_start {
                all_position_ids[0].push(current_pos);
                all_position_ids[1].push(current_pos);
                all_position_ids[2].push(current_pos);
                current_pos += 1;
            }

            // Spatial positions for the image at this run
            let (llm_grid_t, llm_grid_h, llm_grid_w, count) = info;
            let image_base = current_pos;
            for t_idx in 0..llm_grid_t {
                for h_idx in 0..llm_grid_h {
                    for w_idx in 0..llm_grid_w {
                        all_position_ids[0].push(image_base + t_idx);
                        all_position_ids[1].push(image_base + h_idx);
                        all_position_ids[2].push(image_base + w_idx);
                    }
                }
            }
            let max_axis = std::cmp::max(
                llm_grid_t - 1,
                std::cmp::max(llm_grid_h - 1, llm_grid_w - 1),
            );
            current_pos = image_base + max_axis + 1;
            cursor = run_start + count;
        }

        // Trailing text after the last image (run in case (a), sub-run
        // end in case (b) — both resolve to `last_image_end`).
        debug_assert_eq!(cursor, last_image_end);
        let _ = last_image_end;
        for _ in cursor..seq_len as usize {
            all_position_ids[0].push(current_pos);
            all_position_ids[1].push(current_pos);
            all_position_ids[2].push(current_pos);
            current_pos += 1;
        }
    }

    // Convert to MxArray [3, batch, seq_len]
    let t_positions: Vec<i32> = all_position_ids[0].iter().map(|&x| x as i32).collect();
    let h_positions: Vec<i32> = all_position_ids[1].iter().map(|&x| x as i32).collect();
    let w_positions: Vec<i32> = all_position_ids[2].iter().map(|&x| x as i32).collect();

    let t_arr = MxArray::from_int32(&t_positions, &[batch_size, seq_len])?;
    let h_arr = MxArray::from_int32(&h_positions, &[batch_size, seq_len])?;
    let w_arr = MxArray::from_int32(&w_positions, &[batch_size, seq_len])?;

    let position_ids = MxArray::stack(vec![&t_arr, &h_arr, &w_arr], Some(0))?;

    // Decode offset must reference the GLOBAL max M-RoPE position, i.e. the max
    // over all three (t, h, w) axes — matching mlx-vlm's `llm_positions.max()`.
    // For an image the spatial (h, w) axes exceed the temporal one, so an
    // image-final prompt (no trailing text) would get a too-small delta if only
    // axis 0 were considered.
    let max_position = all_position_ids
        .iter()
        .flat_map(|axis| axis.iter().copied())
        .max()
        .unwrap_or(0);
    let rope_deltas = max_position + 1 - seq_len;

    Ok((position_ids, rope_deltas))
}

/// Merge image features into input embeddings at image token positions
pub(crate) fn merge_input_ids_with_image_features(
    image_token_id: i32,
    image_features: &MxArray,
    inputs_embeds: &MxArray,
    input_ids: &MxArray,
) -> Result<MxArray> {
    let input_shape = input_ids.shape()?;
    let batch_size = input_shape[0];

    let image_token = MxArray::scalar_int(image_token_id)?;
    let image_positions = input_ids.equal(&image_token)?;
    let inputs_embeds_shape = inputs_embeds.shape()?;
    let hidden_dim = inputs_embeds_shape[2];

    let mut batch_outputs: Vec<MxArray> = Vec::new();
    let mut feature_start_idx = 0i64;

    for batch_idx in 0..batch_size {
        let batch_mask = image_positions.slice_axis(0, batch_idx, batch_idx + 1)?;
        let batch_mask = batch_mask.squeeze(Some(&[0]))?;

        let mask_sum = batch_mask.sum(None, None)?;
        let num_positions = mask_sum.to_int32()?[0] as i64;

        if num_positions > 0 {
            let batch_features = image_features.slice_axis(
                0,
                feature_start_idx,
                feature_start_idx + num_positions,
            )?;

            let batch_embeds = inputs_embeds.slice_axis(0, batch_idx, batch_idx + 1)?;
            let batch_embeds = batch_embeds.squeeze(Some(&[0]))?;

            let mask_int = batch_mask.astype(crate::array::DType::Int32)?;
            let cumsum = mask_int.cumsum(0)?;

            let ones = MxArray::scalar_int(1)?;
            let feature_indices = cumsum.sub(&ones)?;
            let zeros =
                MxArray::zeros(&feature_indices.shape()?, Some(crate::array::DType::Int32))?;
            let feature_indices = batch_mask.where_(&feature_indices, &zeros)?;

            let gathered_features = batch_features.take(&feature_indices, 0)?;

            let mask_expanded = batch_mask.reshape(&[-1, 1])?;
            let mask_expanded =
                MxArray::broadcast_to(&mask_expanded, &[batch_mask.shape()?[0], hidden_dim])?;

            let batch_output = mask_expanded.where_(&gathered_features, &batch_embeds)?;
            batch_outputs.push(batch_output);
            feature_start_idx += num_positions;
        } else {
            let batch_embeds = inputs_embeds.slice_axis(0, batch_idx, batch_idx + 1)?;
            batch_outputs.push(batch_embeds.squeeze(Some(&[0]))?);
        }
    }

    let refs: Vec<&MxArray> = batch_outputs.iter().collect();
    MxArray::stack(refs, Some(0))
}

#[derive(Debug, Clone, Copy)]
pub(super) struct VisionImageRequest {
    pub(super) key: VisionFeatureCacheKey,
    pub(super) patch_start: i64,
    pub(super) patch_count: i64,
    pub(super) feature_count: i64,
}

#[derive(Debug)]
pub(super) struct VisionCacheMiss {
    pub(super) request: VisionImageRequest,
    pub(super) request_indices: Vec<usize>,
}

pub(super) fn plan_vision_image_requests(
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

pub(super) fn partition_vision_cache_misses(
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

pub(super) fn lookup_vision_feature_cache(
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

pub(super) fn projected_vision_feature_bytes(
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
    vision_encoder: &Qwen3_5VisionEncoder,
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
