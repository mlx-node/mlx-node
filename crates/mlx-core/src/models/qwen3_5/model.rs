use std::cell::Cell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunctionCallMode;
use napi_derive::napi;
use tracing::{debug, info, warn};

use super::quantized_linear::LinearProj;
use crate::array::MxArray;
use crate::engine::backend::{
    ChatBackend, ChunkSink, DecodeStep, MtpBackend, MtpStepper, MtpTurnSetup, PagedBackend,
    PagedPrefix, PagedTurnSetup, ResetScope, SaveStateArgs, ThinkingSetup, TrainBackend,
    TurnOutput, TurnSetup, WholeTurnArgs,
};
use crate::engine::cmd::{
    ChatCmd, FromChatCmd, FromTrainCmd, TrainCmd, handle_chat_cmd, handle_train_cmd,
};
use crate::engine::plan::{
    DecoderPlan, ExecutionPlan, MediaCapabilities, MediaPlan, PagedAttentionPlan, SpeculativeKind,
    SpeculativePlan,
};
use crate::inference_trace::{
    elapsed_ms, enabled as inference_trace_enabled, write as write_inference_trace,
};
use crate::model_thread::ResponseTx;
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::sampling::{SamplingConfig, sample};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer, ToolDefinition};
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

use super::config::Qwen3_5Config;
use super::decoder_layer::DecoderLayer;
#[cfg(test)]
use super::gdn_checkpoint_store::{
    GDN_PREFIX_CHECKPOINTS_PER_OWNER, GDN_PREFIX_CHECKPOINTS_PER_OWNER_NO_LADDER,
    compute_paged_prefix_block_hash,
};
use super::gdn_checkpoint_store::{
    GdnCheckpointLineage, GdnSidecarBoundary, compute_paged_prefix_block_hashes,
    find_longest_valid_gdn_checkpoint_index, gdn_retention_caps, prune_gdn_checkpoints,
    replay_gdn_cache_and_commit, select_gdn_sidecar_boundary,
};
use super::layer_cache::Qwen3_5LayerCache;
use super::mtp::Qwen3_5MTPModule;
use super::mtp_decode;
use super::persistence;
use super::processing::{Qwen35VLImageProcessor, merged_image_token_count};
use super::vision::Qwen3_5VisionEncoder;
use crate::engine;
use crate::engine::vision::VisionMerge;
use crate::engine::{
    apply_all_penalties, compute_performance_metrics, extract_chat_params, finalize_chat_result,
    save_cache_state_direct, verify_cache_prefix_direct,
};
use crate::models::paddleocr_vl::processing::ProcessedImages;

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

// The shared model-id counter lives in `crate::engine::compiled_lock` so the
// dense + MoE families draw from one id space (per-instance ids never
// overlap). Re-exported here for unqualified use by this module's
// `Qwen35Inner::new`; MoE imports it from `crate::engine::compiled_lock`
// directly.
pub(crate) use crate::engine::compiled_lock::QWEN35_MODEL_ID_COUNTER;

fn fresh_dense_layer_caches(config: &Qwen3_5Config) -> Vec<Qwen3_5LayerCache> {
    (0..config.num_layers as usize)
        .map(|i| {
            if config.is_linear_layer(i) {
                Qwen3_5LayerCache::new_linear()
            } else {
                Qwen3_5LayerCache::new_full_attention()
            }
        })
        .collect()
}

/// Create the profiler owned by a paged-MTP turn before its prefill starts.
///
/// Paged AR intentionally keeps its existing terminal-only metrics path and
/// therefore does not allocate a decode profiler here. This helper starts the
/// prefill timer immediately before the actual paged prefill; the caller ends
/// it immediately afterward, then carries this same instance into
/// `run_mtp_turn` for decode and acceptance accounting.
fn configure_paged_mtp_profiler(
    mut profiler: crate::decode_profiler::DecodeProfiler,
    fresh_suffix_tokens: u32,
) -> crate::decode_profiler::DecodeProfiler {
    profiler.set_prompt_tokens(fresh_suffix_tokens);
    profiler.snapshot_memory_before();
    profiler.begin_prefill();
    profiler
}

fn begin_paged_mtp_profiler(
    eager_mtp_paged: bool,
    fresh_suffix_tokens: u32,
) -> Option<crate::decode_profiler::DecodeProfiler> {
    eager_mtp_paged.then(|| {
        configure_paged_mtp_profiler(
            crate::decode_profiler::DecodeProfiler::new("chat_paged_mtp_eager", "qwen3_5"),
            fresh_suffix_tokens,
        )
    })
}

#[cfg(test)]
mod paged_mtp_profiler_tests {
    use super::*;

    #[test]
    fn paged_mtp_profiler_is_gated_away_from_ar() {
        assert!(begin_paged_mtp_profiler(false, 37).is_none());
        assert!(begin_paged_mtp_profiler(true, 37).is_some());
    }

    #[test]
    fn paged_mtp_profiler_starts_with_fresh_suffix_metadata() {
        let mut profiler = crate::decode_profiler::DecodeProfiler::new("paged_mtp_test", "qwen3_5");
        profiler.enable_for_test();

        let mut profiler = configure_paged_mtp_profiler(profiler, 37);
        let (prompt_tokens, prefill_active, memory_before, prefill_ms) =
            profiler.turn_start_state_for_test();
        assert_eq!(prompt_tokens, 37);
        assert!(prefill_active);
        assert!(memory_before);
        assert_eq!(prefill_ms, 0.0);

        profiler.end_prefill();
        let (_, prefill_active, _, prefill_ms) = profiler.turn_start_state_for_test();
        assert!(!prefill_active);
        assert!(prefill_ms.is_finite());
    }
}

/// Enforce one fixed paged-pool token ceiling against already-resolved chat
/// parameters. Shared by dense and MoE so every native paged executor uses the
/// same prompt rejection marker and the same last-sampled-token accounting.
pub(crate) fn constrain_paged_context_params(
    family: &str,
    prompt_tokens: usize,
    capacity: u32,
    params: &mut engine::ChatParams,
) -> Result<()> {
    let prompt = u32::try_from(prompt_tokens).unwrap_or(u32::MAX);
    if prompt > capacity {
        return Err(Error::from_reason(format!(
            "context_length_exceeded: rendered prompt has {prompt} tokens, effective active \
             context is {capacity} tokens"
        )));
    }
    // The final sampled token is returned without another model forward, so N
    // generated tokens consume only N-1 additional KV positions.
    let max_output = capacity.saturating_sub(prompt).saturating_add(1);
    let requested = params.max_new_tokens.max(0) as u32;
    if requested > max_output {
        warn!(
            "{family} clamping max_new_tokens from {} to {} for effective context {} \
             (prompt_tokens={})",
            requested, max_output, capacity, prompt
        );
        params.max_new_tokens = max_output as i32;
    }
    Ok(())
}

#[cfg(test)]
mod paged_context_capacity_tests {
    use super::*;
    use crate::engine::types::ChatConfig;

    fn make_params(max_new_tokens: i32) -> engine::ChatParams {
        extract_chat_params(&ChatConfig {
            cache_owner_id: None,
            cache_root_owner_id: None,
            max_new_tokens: Some(max_new_tokens),
            ..ChatConfig::default()
        })
    }

    #[test]
    fn rejects_a_prompt_larger_than_the_physical_pool() {
        let mut params = make_params(32);
        let err = constrain_paged_context_params("test", 65, 64, &mut params).unwrap_err();
        assert!(
            err.reason
                .starts_with("context_length_exceeded: rendered prompt has 65 tokens")
        );
    }

    #[test]
    fn clamps_output_with_last_sampled_token_accounting() {
        let mut params = make_params(100);
        constrain_paged_context_params("test", 48, 64, &mut params).unwrap();
        assert_eq!(params.max_new_tokens, 17);

        let mut full_prompt = make_params(100);
        constrain_paged_context_params("test", 64, 64, &mut full_prompt).unwrap();
        assert_eq!(full_prompt.max_new_tokens, 1);
    }

    #[test]
    fn preserves_an_already_safe_output_budget() {
        let mut params = make_params(10);
        constrain_paged_context_params("test", 48, 64, &mut params).unwrap();
        assert_eq!(params.max_new_tokens, 10);
    }
}

struct DenseGdnPrefixCheckpoint {
    owner_id: String,
    prefix_len: u32,
    block_size: u32,
    final_block_hash: u64,
    block_hashes: Vec<u64>,
    tokens: Vec<u32>,
    caches: Vec<Qwen3_5LayerCache>,
}

impl GdnCheckpointLineage for DenseGdnPrefixCheckpoint {
    fn owner_id(&self) -> &str {
        &self.owner_id
    }

    fn prefix_len(&self) -> u32 {
        self.prefix_len
    }

    fn block_size(&self) -> u32 {
        self.block_size
    }

    fn final_block_hash(&self) -> u64 {
        self.final_block_hash
    }

    fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    fn block_hashes(&self) -> &[u64] {
        &self.block_hashes
    }
}

struct DenseGdnHistoryCheckpoint {
    owner_id: String,
    image_key: Option<u64>,
    tokens: Vec<u32>,
    caches: Vec<Qwen3_5LayerCache>,
}

struct DenseGdnPrefixPreparation {
    state: &'static str,
    already_primed: bool,
    restored_prefix_tokens: u32,
    replayed_prefix_tokens: u32,
}

#[derive(Default)]
struct DenseGdnCheckpointStoreTrace {
    stored: bool,
    eval_ms: f64,
    clone_ms: f64,
    token_clone_ms: f64,
    update_ms: f64,
    total_ms: f64,
}

impl DenseGdnCheckpointStoreTrace {
    fn finish(mut self, start: Option<std::time::Instant>) -> Self {
        self.total_ms = start.map(elapsed_ms).unwrap_or(0.0);
        self
    }
}

#[derive(Clone, Copy)]
struct TokenPrefixMismatchTrace {
    index: i64,
    prompt_token: i64,
    cached_token: i64,
}

impl Default for TokenPrefixMismatchTrace {
    fn default() -> Self {
        Self {
            index: -1,
            prompt_token: -1,
            cached_token: -1,
        }
    }
}

fn token_prefix_mismatch_trace(prompt: &[u32], cached: &[u32]) -> TokenPrefixMismatchTrace {
    let common_len = prompt.len().min(cached.len());
    for i in 0..common_len {
        if prompt[i] != cached[i] {
            return TokenPrefixMismatchTrace {
                index: i as i64,
                prompt_token: prompt[i] as i64,
                cached_token: cached[i] as i64,
            };
        }
    }

    TokenPrefixMismatchTrace {
        index: common_len as i64,
        prompt_token: prompt.get(common_len).map_or(-1, |token| *token as i64),
        cached_token: cached.get(common_len).map_or(-1, |token| *token as i64),
    }
}

fn dense_paged_linear_caches_ready(
    config: &Qwen3_5Config,
    caches: Option<&[Qwen3_5LayerCache]>,
) -> bool {
    let Some(caches) = caches else {
        return false;
    };
    if caches.len() != config.num_layers as usize {
        return false;
    }
    for (i, cache) in caches.iter().enumerate() {
        if !config.is_linear_layer(i) {
            continue;
        }
        let Qwen3_5LayerCache::Linear(arrays) = cache else {
            return false;
        };
        if arrays.get(0).is_none() || arrays.get(1).is_none() {
            return false;
        }
    }
    true
}

fn clone_dense_linear_layer_caches(
    config: &Qwen3_5Config,
    caches: &[Qwen3_5LayerCache],
) -> Option<Vec<Qwen3_5LayerCache>> {
    if !dense_paged_linear_caches_ready(config, Some(caches)) {
        return None;
    }

    let mut cloned = fresh_dense_layer_caches(config);
    for i in 0..config.num_layers as usize {
        if !config.is_linear_layer(i) {
            continue;
        }
        let Qwen3_5LayerCache::Linear(arrays) = &caches[i] else {
            return None;
        };
        cloned[i] = Qwen3_5LayerCache::Linear(arrays.clone());
    }
    Some(cloned)
}

/// Internal model state owned exclusively by the dedicated model thread.
///
/// No `Arc<RwLock<>>` — the model thread has sole ownership of all inference
/// and training state. Training commands are routed via `TrainingDispatch`.
pub(crate) struct Qwen35Inner {
    pub(crate) config: Qwen3_5Config,
    pub(crate) embedding: Embedding,
    pub(crate) layers: Vec<DecoderLayer>,
    pub(crate) final_norm: RMSNorm,
    pub(crate) lm_head: Option<LinearProj>,
    pub(crate) caches: Option<Vec<Qwen3_5LayerCache>>,
    pub(crate) tokenizer: Option<Arc<Qwen3Tokenizer>>,
    pub(crate) vision_encoder: Option<Arc<Qwen3_5VisionEncoder>>,
    pub(crate) image_processor: Option<Arc<Qwen35VLImageProcessor>>,
    pub(crate) spatial_merge_size: Option<i32>,
    pub(crate) vision_cache: VisionCache,
    pub(crate) cached_token_history: Vec<u32>,
    pub(crate) cached_image_key: Option<u64>,
    /// Absolute expanded-token positions paired with all four per-image digest
    /// words for the live paged request. These keys must remain attached to
    /// the image-conditioned prefix when a later text turn extends it and
    /// re-finalizes the request in the shared prefix cache.
    pub(crate) cached_paged_image_token_positions: Vec<(u32, u64)>,
    pub(crate) cached_rope_deltas: Option<i32>,
    pub(crate) model_id: u64,
    active_cache_owner_id: String,
    gdn_root_cache_owner_id: Option<String>,
    gdn_root_cache_owner_is_explicit: bool,
    gdn_prefix_checkpoints: VecDeque<DenseGdnPrefixCheckpoint>,
    gdn_last_history_checkpoint: Option<DenseGdnHistoryCheckpoint>,
    /// Set when the infallible [`PagedBackend::finalize_paged_turn`] hook could
    /// not register or release the adapter request. The engine calls
    /// `save_paged_history` immediately afterwards, so that save must consume
    /// this latch and refuse to republish placeholder-token history as a live
    /// image-derived session.
    paged_finalize_failed: bool,
    /// Block-paged KV adapter (vLLM-style refcounted prefix cache) for
    /// full-attention layers.
    ///
    /// **Opt-in via `Qwen3_5Config::use_block_paged_cache`** — see the
    /// flag's rustdoc for the full architectural rationale. When
    /// `Some(...)`, full-attention layers route through this adapter
    /// while linear-attention (GDN) layers stay on
    /// `Qwen3_5LayerCache::Linear` with no cross-request prefix reuse.
    /// Paged turns run the pure-Rust eager paged forward
    /// (`paged_forward::run_paged_prefill_chunk` / `run_paged_decode_step`).
    pub(crate) paged_adapter: Option<PagedKVCacheAdapter>,
    /// True when a paged-core turn has populated
    /// the paged adapter's `LayerKVPool` since the last flat full-attention
    /// prefill, so the flat `self.caches` full-attention slots do NOT
    /// reflect the conversation history. The streaming dense-MTP fallback
    /// (`chat_stream_tokens_delta_sync_inner`) consults this to decide
    /// whether it must rebuild the flat caches from the full history before
    /// decoding. Set by the paged cores; cleared after a flat prefill. This
    /// keeps the rebuild a ONE-TIME cost on the paged→dense transition
    /// instead of re-prefilling the whole history on every MTP turn.
    pub(crate) paged_full_attn_caches_dirty: bool,
    /// Set when a flat eager-MTP turn stopped mid-cycle leaving `self.caches`
    /// advanced past the emitted token history (GDN state cannot be rewound).
    /// Forces the next turn to discard `self.caches` and re-prefill the full
    /// history. Pure-flat sessions only; the paged path rolls back its adapter
    /// directly.
    pub(crate) flat_mtp_caches_desynced: bool,
    /// Count of full-history flat re-prefills taken by the streaming delta path
    /// because the caches were desynced (the discard+re-prefill heal at
    /// `chat_stream_tokens_delta_sync_inner`). Monotonic; lets a test confirm a
    /// continue turn actually took the heal path (the streaming chunk's
    /// `prompt_tokens`/`cached_tokens` are reported identically for heal and warm,
    /// so they can't distinguish the two).
    pub(crate) flat_full_reprefill_count: u64,
    /// Engine-observed accepted-token tail passed to the most recent flat MTP
    /// turn's `rollback_unemitted` hook. Unlike
    /// `flat_mtp_caches_desynced`, this value is computed before the family
    /// stepper handles the rollback, so tests can verify that a positive tail
    /// actually arms the family latch.
    pub(crate) flat_mtp_last_rollback_unemitted: usize,
    /// Training state owned by the model thread.
    /// Created when `InitTraining` command is received, destroyed when training ends.
    pub(crate) training_state: Option<crate::training_state::ModelThreadTrainingState>,
    /// Optional Multi-Token Prediction head.
    ///
    /// Constructed when `config.n_mtp_layers > 0`. The speculative decode loop
    /// is the only intended caller; the single-token decode path ignores this
    /// field. Weight loading is performed by `persistence::apply_weights_inner`
    /// after the main per-layer weights are loaded.
    pub(crate) mtp: Option<Qwen3_5MTPModule>,
    /// True only after persistence has seen a complete MTP tensor set and
    /// applied it to `mtp`. The module may exist from config alone; this
    /// flag prevents random-init MTP modules from advertising capability.
    pub(crate) mtp_weights_loaded: bool,
    /// Whether the CURRENT generic-flow turn is streaming. Set by the
    /// [`ChatBackend::profiler_label`] hook (the session core calls it
    /// exactly once per generic-flow turn, before `begin_decode`);
    /// consumed by [`ChatBackend::begin_decode`]'s
    /// profiler relabel, which must pick the `chat_*` vs `chat_stream_*`
    /// label family (`TurnSetup` does not carry streaming-ness).
    /// Whole-turn override paths (vision/paged/MTP) never consult it.
    turn_is_streaming: Cell<bool>,
    /// Sampling + stop-token defaults parsed from the checkpoint's
    /// `generation_config.json` at load time. Empty for checkpoints that
    /// ship no such file. Consumed by the [`ChatBackend`] sampling/EOS
    /// hooks and the raw `generate` loop.
    gen_defaults: crate::engine::ModelGenerationDefaults,
}

/// Build the dense media admission contract from the components wired into
/// one loaded Qwen3.5 instance.
///
/// Image execution needs all three pieces. Missing combinations remain
/// backend-validated so the family core preserves its established diagnostic
/// instead of the engine replacing it with a generic unsupported-media error.
pub(crate) const fn qwen35_dense_vision_active(
    has_vision_encoder: bool,
    has_image_processor: bool,
    has_paged_adapter: bool,
) -> bool {
    has_vision_encoder && has_image_processor && has_paged_adapter
}

const fn qwen35_dense_media_plan(
    has_vision_encoder: bool,
    has_image_processor: bool,
    has_paged_adapter: bool,
) -> MediaPlan {
    let images_available =
        qwen35_dense_vision_active(has_vision_encoder, has_image_processor, has_paged_adapter);
    MediaPlan::with_backend_validation(
        MediaCapabilities {
            images: images_available,
            audio: false,
        },
        MediaCapabilities::IMAGES,
    )
}

/// Media represented by the live dense session state.
///
/// The content hash identifies a freshly saved image turn. Generic paged
/// continuations intentionally clear that key, but retain the M-RoPE delta
/// while they keep extending the same live image-derived KV prefix. Treat
/// either signal as image context so request planning stays truthful across
/// every successive text continuation.
const fn qwen35_dense_session_media(
    has_cached_image_key: bool,
    has_cached_rope_delta: bool,
) -> MediaCapabilities {
    if has_cached_image_key || has_cached_rope_delta {
        MediaCapabilities::IMAGES
    } else {
        MediaCapabilities::NONE
    }
}

fn qwen35_dense_session_media_matches_payloads(
    cached_image_key: Option<u64>,
    images: &[Vec<u8>],
    audio: &[Vec<u8>],
) -> bool {
    audio.is_empty()
        && !images.is_empty()
        && cached_image_key == Some(engine::compute_image_cache_key(images))
}

/// Make the resolved decoder authoritative for legacy Qwen3.5 whole-turn
/// cores, which still derive their local `ChatParams` from a `ChatConfig`.
fn apply_qwen35_dense_planned_decoder(config: &mut ChatConfig, decoder: DecoderPlan) -> bool {
    let planned_mtp = matches!(
        decoder,
        DecoderPlan::Speculative(SpeculativeKind::NativeMtp)
    );
    config.enable_mtp = Some(planned_mtp);
    planned_mtp
}

/// Commands dispatched from NAPI methods to the dedicated model thread.
pub(crate) enum Qwen35Cmd {
    /// All chat-session traffic (sync + streaming starts/continues/tool
    /// turns + cache reset), routed through the model-neutral engine
    /// dispatcher ([`crate::engine::cmd::handle_chat_cmd`]) against the
    /// [`ChatBackend`] impl on [`Qwen35Inner`]. The per-variant
    /// behavioural contracts live on [`crate::engine::cmd::ChatCmd`].
    Chat(ChatCmd),
    Generate {
        prompt_tokens: MxArray,
        config: Qwen3_5GenerationConfig,
        reply: ResponseTx<Qwen3_5GenerationResult>,
    },
    /// Static FP8 activation-amax calibration prefill (NVIDIA modelopt
    /// `MaxCalibrator`). For each raw text: tokenize WITHOUT the chat template,
    /// truncate to `calib_seq` tokens, then run PREFILL ONLY (no generation, no
    /// generated token) so every mxfp8 attn/GDN projection's activation tap
    /// fires once, resetting caches between rows. Runs on the model thread where
    /// the tokenizer lives. The command body SELF-ARMS this model thread's
    /// thread-local `ActivationAmaxCollector` flag (via `CalibrationArmGuard`)
    /// for the prefill's duration; the NAPI caller drains+persists the collected
    /// amax afterwards — this command never touches `config.json`. Replies with
    /// the number of rows actually prefilled (rows that were empty after
    /// tokenize+truncate are skipped).
    CalibratePrefillRaw {
        texts: Vec<String>,
        calib_seq: u32,
        reply: ResponseTx<u32>,
    },
    SaveModel {
        save_path: String,
        reply: ResponseTx<()>,
    },
    /// Test-only: snapshot the flat-MTP cache state between turns —
    /// `(cached_token_history.len(), flat_mtp_caches_desynced,
    /// flat_full_reprefill_count, flat_mtp_last_rollback_unemitted)`. The length
    /// is the committed prompt+generation history (how many tokens a turn
    /// actually committed, independent of the warm/heal path a later turn
    /// takes); the flag is whether a mid-cycle stop stranded tokens and armed
    /// the heal; the count is the monotonic number of full-history re-prefill
    /// heals taken so far; and the final value is the independently
    /// engine-computed tail passed to the family rollback hook.
    #[doc(hidden)]
    MtpFlatStateForTest {
        reply: ResponseTx<(usize, bool, u64, usize)>,
    },
    /// Test-only: arm the flat-MTP desync heal (`flat_mtp_caches_desynced =
    /// true`) so the NEXT delta turn takes the discard+re-prefill path
    /// deterministically. The heal re-prefills from `cached_token_history` and
    /// ignores the (discarded) cache contents, so arming the flag on a clean
    /// session faithfully exercises the heal without a host-timing-dependent
    /// mid-cycle cancel.
    #[doc(hidden)]
    ForceFlatMtpDesyncForTest { reply: ResponseTx<()> },
    /// Training-session commands shared with the model-neutral engine. The
    /// thread loop routes these to
    /// [`crate::engine::cmd::handle_train_cmd`], which drives the
    /// [`TrainBackend`] impl on [`Qwen35Inner`].
    Train(TrainCmd),
}

impl FromChatCmd for Qwen35Cmd {
    #[inline]
    fn from_chat(cmd: ChatCmd) -> Self {
        Qwen35Cmd::Chat(cmd)
    }
}

impl FromTrainCmd for Qwen35Cmd {
    #[inline]
    fn from_train(cmd: TrainCmd) -> Self {
        Qwen35Cmd::Train(cmd)
    }
}

/// Training backend the model-neutral [`handle_train_cmd`] drives. Each
/// method forwards to the inherent `*_sync_impl` body on [`Qwen35Inner`].
impl TrainBackend for Qwen35Inner {
    fn training_state_mut(
        &mut self,
    ) -> &mut Option<crate::training_state::ModelThreadTrainingState> {
        &mut self.training_state
    }

    fn init_training_sync(
        &mut self,
        config: Box<crate::grpo::engine::GRPOEngineConfig>,
        model_type: crate::training_model::ModelType,
    ) -> Result<()> {
        self.init_training_sync_impl(*config, model_type)
    }

    fn generate_for_training_thread_sync(
        &mut self,
        prompts: Vec<Vec<ChatMessage>>,
        group_size: usize,
        gen_config: crate::models::qwen3::GenerationConfig,
        enable_thinking: Option<bool>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Result<crate::training_model::GenerationPlainData> {
        self.generate_for_training_thread_sync_impl(
            prompts,
            group_size,
            gen_config,
            enable_thinking,
            tools,
        )
    }

    fn train_step_grpo_sync(
        &mut self,
        rewards: Vec<f64>,
        group_size: i32,
        loss_config: crate::grpo::loss::GRPOLossConfig,
        valid_indices: Option<Vec<usize>>,
    ) -> Result<crate::training_model::TrainStepPlainMetrics> {
        self.train_step_grpo_sync_impl(rewards, group_size, loss_config, valid_indices)
    }

    fn train_step_sft_sync(
        &mut self,
        input_ids: Vec<i32>,
        input_shape: Vec<i64>,
        labels: Vec<i32>,
        labels_shape: Vec<i64>,
        config: crate::sft::engine::SftEngineConfig,
    ) -> Result<crate::training_model::TrainStepPlainMetrics> {
        self.train_step_sft_sync_impl(input_ids, input_shape, labels, labels_shape, config)
    }

    fn save_optimizer_state_sync(&self, path: String) -> Result<()> {
        self.save_optimizer_state_sync_impl(path)
    }

    fn load_optimizer_state_sync(&mut self, path: String) -> Result<()> {
        self.load_optimizer_state_sync_impl(path)
    }
}

/// Command handler for the dedicated model thread.
pub(crate) fn handle_qwen35_cmd(inner: &mut Qwen35Inner, cmd: Qwen35Cmd) {
    match cmd {
        // All chat-session traffic routes through the model-neutral
        // engine dispatcher against `Qwen35Inner`'s `ChatBackend` impl.
        // (The engine dispatcher carries the historical NOTE forward: no
        // per-request cache drain here — the TS idle sweeper in
        // `@mlx-node/server` handles between-turn drains.)
        Qwen35Cmd::Chat(chat_cmd) => {
            handle_chat_cmd(inner, chat_cmd);
        }
        Qwen35Cmd::Generate {
            prompt_tokens,
            config,
            reply,
        } => {
            let _ = reply.send(inner.generate_sync(prompt_tokens, config));
        }
        Qwen35Cmd::CalibratePrefillRaw {
            texts,
            calib_seq,
            reply,
        } => {
            let _ = reply.send(inner.calibrate_prefill_raw_sync(texts, calib_seq));
        }
        Qwen35Cmd::SaveModel { save_path, reply } => {
            let _ = reply.send(inner.save_model_sync(&save_path));
        }
        Qwen35Cmd::MtpFlatStateForTest { reply } => {
            let _ = reply.send(Ok((
                inner.cached_token_history.len(),
                inner.flat_mtp_caches_desynced,
                inner.flat_full_reprefill_count,
                inner.flat_mtp_last_rollback_unemitted,
            )));
        }
        Qwen35Cmd::ForceFlatMtpDesyncForTest { reply } => {
            inner.flat_mtp_caches_desynced = true;
            let _ = reply.send(Ok(()));
        }
        // --- Training commands ---
        Qwen35Cmd::Train(train_cmd) => {
            handle_train_cmd(inner, train_cmd);
        }
    }
}

/// Input bundle for [`Qwen35Inner::chat_with_caches_inner`].
///
/// Packs every value the shared post-prefill pipeline needs into a single
/// named struct so callers don't have to thread 20+ positional arguments.
/// Constructed by the prefill-side of [`Qwen35Inner::vision_mtp_whole_turn_core`] and
/// [`Qwen35Inner::chat_tokens_delta_sync`].
///
/// The caller is responsible for:
///   - constructing a `WiredLimitContext` tied to `generation_stream` for
///     the lifetime of the call,
///   - running prefill and packaging the resulting `last_logits` and
///     `seq_len`.
pub(crate) struct ChatDecodeInputs {
    // --- Prefill outputs -------------------------------------------------
    /// Logits for the last position of the prefill chunk. Penalties and
    /// sampling run against this to produce the first decoded token.
    pub last_logits: MxArray,
    /// Total context length after prefill (cached + newly-prefilled).
    pub seq_len: i64,
    /// `true` when this invocation is a session DELTA continuation
    /// (text-only append on top of the live KV cache). Drives the
    /// post-decode save pathway: deltas keep `cached_image_key` sticky
    /// so image attention state baked into the KV caches by a prior
    /// prefill stays addressable; prefills (re)set the key based on
    /// the fresh turn's `has_images`.
    pub is_delta: bool,

    /// `true` when the current turn carries images.
    pub has_images: bool,

    // --- Token bookkeeping ----------------------------------------------
    /// Full pre-decode token sequence. Seeds the decode loop's running
    /// history (mutated in place) and the penalty context.
    pub token_history_init: Vec<u32>,
    /// Token snapshot handed to `save_cache_state_direct`. For text-only
    /// this equals `token_history_init`; for VLM it's the pre-expansion
    /// tokens.
    pub save_tokens: Vec<u32>,
    /// Expanded token sequence (with image placeholders expanded) used by
    /// the VLM save path. `None` for text-only.
    pub save_expanded_tokens: Option<Vec<u32>>,
    /// Image cache key for the current turn. 0 for text-only.
    pub save_image_cache_key: u64,

    // --- Tokenizer / reasoning state ------------------------------------
    pub tokenizer: Arc<Qwen3Tokenizer>,
    pub think_end_id: Option<u32>,
    pub think_end_str: Option<String>,
    /// Resolved thinking-mode state for the turn — the single source of
    /// truth, threaded from `WholeTurnArgs::thinking` so the cores share
    /// one `resolve_enable_thinking` result.
    pub thinking: ThinkingSetup,
    /// End-of-sequence token id for the decode loop. For `vision_mtp_whole_turn_core` this
    /// is `config.eos_token_id`; for the session delta path it's
    /// `<|im_end|>` so cache boundaries stay clean.
    pub eos_id: u32,

    // --- Profiler / perf metrics ----------------------------------------
    pub profiler: crate::decode_profiler::DecodeProfiler,
    pub generation_start: Option<std::time::Instant>,
    pub first_token_instant: Option<std::time::Instant>,
    /// Number of tokens actually prefilled this turn (for throughput math).
    pub prefill_tokens_len: usize,
    /// Prompt token count reported on the `ChatResult`.
    pub prompt_tokens_for_result: u32,
    /// Length of the reused cached prefix to report on `ChatResult.cached_tokens`.
    ///
    /// For fresh prefills this is `cached_prefix_len` (0 on a miss, full
    /// cached length on an exact-append hit). For the session delta path
    /// this is the full prior-history length because the delta is
    /// appended on top of the existing caches — we skip the `cached_prefix_len`
    /// driver (which only gates the VLM rope-delta replay branch) while
    /// still reporting the reused prefix accurately for observability.
    pub cached_tokens_for_result: u32,

    // --- MLX state ------------------------------------------------------
    pub embedding_weight: MxArray,
    pub embedding_weight_t: MxArray,
    pub generation_stream: Stream,
    pub params: crate::engine::ChatParams,

    // --- prompt-prefix MTP prefill --------------------------------------
    /// Post-final-norm hidden state for every prefilled prompt token,
    /// `[1, prefill_len, hidden]`. `Some` only when MTP is active for this
    /// turn (`params.enable_mtp && has_mtp_weights`) and the prefill ran
    /// the hidden-emitting `chunked_prefill_with_hidden`. Consumed once,
    /// by `begin_mtp_decode`'s prompt-prefix seed, to commit the prompt
    /// prefix into the MTP committed-history cache.
    /// `None` for non-MTP turns and for the streaming/delta paths.
    pub prompt_hidden: Option<MxArray>,
    /// The exact prompt token ids whose hiddens `prompt_hidden` holds —
    /// i.e. the `prefill_tokens` slice the hidden-emitting prefill
    /// forwarded. `prompt_hidden.shape(1) == prompt_hidden_ids.len()`.
    /// `Some` iff `prompt_hidden` is `Some`.
    pub prompt_hidden_ids: Option<Vec<u32>>,
    /// Absolute committed-history position of `prompt_hidden_ids[0]`'s hidden
    /// row. Zero for full committed history; non-zero for last-window prompt
    /// seeding.
    pub prompt_hidden_position_base: usize,
}

// ========== Qwen35Inner implementation ==========
// All these methods run on the dedicated model thread (synchronous, no locks).

impl Qwen35Inner {
    /// Create a new Qwen35Inner with the given configuration.
    pub(crate) fn new(config: Qwen3_5Config) -> Result<Self> {
        let embedding = Embedding::new(config.vocab_size as u32, config.hidden_size as u32)?;

        let layers = (0..config.num_layers as usize)
            .map(|i| DecoderLayer::new(&config, i))
            .collect::<Result<Vec<_>>>()?;

        let final_norm = RMSNorm::new(config.hidden_size as u32, Some(config.rms_norm_eps))?;

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(LinearProj::Standard(Linear::new(
                config.hidden_size as u32,
                config.vocab_size as u32,
                Some(false),
            )?))
        };

        let model_id = QWEN35_MODEL_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

        // The physical paged pool is intentionally created only after weight
        // loading/materialization. At this point the MLX allocator does not yet
        // know the resident model footprint, so sizing here can overcommit
        // unified memory. Persistence calls `initialize_paged_adapter()` at the
        // post-materialization seam.
        let paged_adapter = None;

        // MTP head — constructed only when the checkpoint config
        // declares MTP layers. Weight load happens later, inside
        // `persistence::apply_weights_inner`, so the module starts
        // with random init here.
        let mtp = if config.n_mtp_layers > 0 {
            Some(Qwen3_5MTPModule::new(&config)?)
        } else {
            None
        };

        Ok(Self {
            config,
            embedding,
            layers,
            final_norm,
            lm_head,
            caches: None,
            tokenizer: None,
            vision_encoder: None,
            image_processor: None,
            spatial_merge_size: None,
            vision_cache: Arc::new(Mutex::new(VisionCacheInner::new())),
            cached_token_history: Vec::new(),
            cached_image_key: None,
            cached_paged_image_token_positions: Vec::new(),
            cached_rope_deltas: None,
            model_id,
            active_cache_owner_id: String::new(),
            gdn_root_cache_owner_id: None,
            gdn_root_cache_owner_is_explicit: false,
            gdn_prefix_checkpoints: VecDeque::new(),
            gdn_last_history_checkpoint: None,
            paged_finalize_failed: false,
            paged_adapter,
            paged_full_attn_caches_dirty: false,
            flat_mtp_caches_desynced: false,
            flat_full_reprefill_count: 0,
            flat_mtp_last_rollback_unemitted: 0,
            training_state: None,
            mtp,
            mtp_weights_loaded: false,
            turn_is_streaming: Cell::new(false),
            gen_defaults: crate::engine::ModelGenerationDefaults::default(),
        })
    }

    /// Construct the physical paged KV pool after checkpoint weights have
    /// been installed and materialized. The configured memory value is a
    /// requested maximum; live unified-memory/Metal probes may only reduce it.
    pub(crate) fn initialize_paged_adapter(&mut self) -> Result<()> {
        if !self.config.use_block_paged_cache.unwrap_or(false)
            || !crate::engine::persistence::compiled_forward_backend_available()
        {
            return Ok(());
        }
        if self.paged_adapter.is_some() {
            return Ok(());
        }

        let attn_layer_count = self.config.full_attention_layer_count() as u32;
        if attn_layer_count == 0 {
            return Err(Error::from_reason(
                "Qwen3.5 block-paged adapter: config has no full_attention layers; paged KV \
                 cache requires at least one attention layer",
            ));
        }
        let block_size = self.config.paged_block_size.unwrap_or(16);
        let head_size = self.config.head_dim as u32;
        let num_kv_heads = self.config.num_kv_heads as u32;
        let max_seq_len = self.config.max_position_embeddings as u32;
        let default_memory_mb = super::config::qwen35_default_paged_cache_memory_mb(
            max_seq_len,
            block_size,
            head_size,
            num_kv_heads,
            attn_layer_count,
        );
        let (requested_memory_mb, requested_source) =
            super::config::qwen35_resolve_paged_cache_memory_mb(
                self.config.paged_cache_memory_mb,
                default_memory_mb,
            );
        let pa_config = mlx_paged_attn::PagedAttentionConfig {
            block_size,
            gpu_memory_mb: requested_memory_mb,
            head_size,
            num_kv_heads,
            num_layers: attn_layer_count,
            use_fp8_cache: Some(false),
            max_seq_len: Some(max_seq_len),
            max_batch_size: Some(32),
        };
        let requested_blocks = pa_config.calculate_num_blocks();
        if requested_blocks == 0 {
            return Err(Error::from_reason(format!(
                "Qwen3.5 block-paged adapter: requested paged cache {requested_memory_mb} MiB \
                 cannot hold one block"
            )));
        }
        let cache_dtype = mlx_paged_attn::metal::MetalDtype::BFloat16;
        let (num_blocks, sizing_source) = match mlx_paged_attn::profile::load_time_pool_sizing(
            requested_blocks,
            attn_layer_count,
            num_kv_heads,
            head_size,
            block_size,
            cache_dtype,
        ) {
            Ok(sizing) => (
                sizing.selected_blocks,
                format!(
                    "adaptive(requested_blocks={}, active_mib={}, working_set_mib={})",
                    sizing.requested_blocks,
                    sizing.metal_active_bytes / (1024 * 1024),
                    sizing
                        .metal_working_set_bytes
                        .map(|v| (v / (1024 * 1024)).to_string())
                        .unwrap_or_else(|| "n/a".to_string())
                ),
            ),
            Err(e) => {
                return Err(Error::from_reason(format!(
                    "Qwen3.5 adaptive paged cache sizing failed safely; refusing an uncapped \
                     pool request: {e}"
                )));
            }
        };

        let allocator = Arc::new(std::sync::Mutex::new(mlx_paged_attn::BlockAllocator::new(
            num_blocks, block_size,
        )));
        let pool = mlx_paged_attn::LayerKVPool::new(pa_config, num_blocks, cache_dtype)
            .map_err(|e| Error::from_reason(format!("Failed to construct Qwen3.5 KV pool: {e}")))?;
        self.paged_adapter = Some(
            PagedKVCacheAdapter::new(allocator, Arc::new(pool), block_size).map_err(|e| {
                Error::from_reason(format!("Failed to construct Qwen3.5 paged adapter: {e}"))
            })?,
        );
        info!(
            "Qwen3.5 paged adapter enabled after weight materialization: num_blocks={}, \
             block_size={}, effective_window_tokens={}, trained_window_tokens={}, \
             requested_memory_mib={}, requested_source={}, sizing_source={}",
            num_blocks,
            block_size,
            num_blocks.saturating_mul(block_size).min(max_seq_len),
            max_seq_len,
            requested_memory_mb,
            requested_source,
            sizing_source,
        );
        Ok(())
    }

    /// Build the process-global SSD cold-tier context (manager + COMPLETE
    /// content fingerprint) for `model_path` WITHOUT attaching it, mirroring
    /// `Qwen3Inner::build_cold_tier_context` — see its doc for how the weight
    /// identity is established and why the caller brackets the load around it.
    ///
    /// The qwen3_5 difference is the [`mlx_paged_attn::ColdSidecarPolicy`]:
    /// qwen3_5's pool covers the FULL-ATTENTION layers only, so a K/V-only
    /// restore would resume from GDN recurrent state the pool never held. The
    /// policy turns the restore walk into vLLM's reconcile-down — the candidate
    /// prefix is reduced to the deepest boundary a validated GDN sidecar backs,
    /// and a boundary nothing backs restores nothing.
    ///
    /// The GDN geometry (and the pool's cache dtype, which the sidecar is
    /// written and read in) is folded into the fingerprint explicitly:
    /// [`crate::cold_tier::ColdTierGeometry`] describes the POOL, which here
    /// covers only the full-attention layers, so two configs differing ONLY in
    /// GDN geometry would otherwise share a pool geometry.
    ///
    /// Returns `None` (fail-open) when the paged adapter is absent, the tier
    /// cannot be opened, this checkpoint has no GDN state to persist, or a
    /// complete content fingerprint cannot be established.
    ///
    /// The `weights` witness pins this call after the loader's
    /// `materialize_weights` pass: MLX preads shard bytes lazily, so an identity
    /// read before that pass can describe bytes the model never runs.
    pub(crate) fn build_cold_tier_context(
        &self,
        model_path: &str,
        weights: &crate::array::memory::WeightsResident,
    ) -> Option<crate::transformer::paged_kv_cache_adapter::ColdTierContext> {
        let adapter = self.paged_adapter.as_ref()?;
        let manager = crate::cold_tier::global_cold_cache()?;
        let pool = adapter.layer_kv_pool();
        let cache_dtype = format!("{:?}", pool.cache_dtype());
        // No GDN layers means no out-of-pool state — but it also means this is
        // not the hybrid qwen3_5 the sidecar work validated, so stay off rather
        // than silently behaving like a dense family.
        let geometry = super::gdn_sidecar::geometry(&self.config, &cache_dtype)?;
        let sidecar_policy = super::gdn_sidecar::policy(&self.config, &cache_dtype)?;
        let mut config_json = serde_json::to_vec(&self.config).ok()?;
        config_json.extend_from_slice(&geometry.fingerprint_component());
        let pool_geometry = crate::cold_tier::ColdTierGeometry {
            block_size: pool.block_size() as u64,
            num_layers: pool.num_layers() as u64,
            num_kv_heads: pool.config().num_kv_heads as u64,
            head_size: pool.config().head_size as u64,
            cache_dtype,
        };
        match crate::cold_tier::build_model_fingerprint(
            "qwen3_5",
            model_path,
            Some(&config_json),
            &pool_geometry,
            weights,
        ) {
            Some(fingerprint) => Some(
                crate::transformer::paged_kv_cache_adapter::ColdTierContext {
                    manager,
                    fingerprint,
                    sidecar_policy: Some(sidecar_policy),
                },
            ),
            None => {
                tracing::warn!(
                    "cold-tier persistence disabled for {model_path}: could not establish a \
                     content fingerprint (unreadable or missing weight shard)"
                );
                None
            }
        }
    }

    /// Attach a previously-built cold-tier context to the paged adapter. A
    /// no-op (fail-open) when the paged adapter is absent. Split from
    /// [`Self::build_cold_tier_context`] so the caller can verify shard
    /// identity is still stable AFTER the fingerprint read and BEFORE the cold
    /// tier is committed.
    ///
    /// Takes the same `materialize_weights` witness as the build step so the
    /// COMMIT point, not just the identity read, is compiler-pinned below
    /// materialization.
    pub(crate) fn attach_cold_tier(
        &mut self,
        ctx: crate::transformer::paged_kv_cache_adapter::ColdTierContext,
        _weights: &crate::array::memory::WeightsResident,
    ) {
        if let Some(adapter) = self.paged_adapter.as_mut() {
            adapter.set_cold_tier(ctx);
        }
    }

    pub(crate) fn paged_context_limits(&self) -> (u32, u32, u32, u32) {
        let trained = self.config.max_position_embeddings.max(0) as u32;
        let Some(adapter) = self.paged_adapter.as_ref() else {
            return (trained, trained, 0, 0);
        };
        let blocks = adapter.block_capacity();
        let block_size = adapter.block_size();
        (
            trained,
            trained.min(adapter.max_capacity_tokens()),
            blocks,
            block_size,
        )
    }

    fn preflight_paged_context(
        &self,
        prompt_tokens: usize,
        params: &mut engine::ChatParams,
    ) -> Result<()> {
        let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
            Error::from_reason("context_length_exceeded: paged cache is not initialized")
        })?;
        let capacity = adapter
            .max_capacity_tokens()
            .min(self.config.max_position_embeddings.max(0) as u32);
        constrain_paged_context_params("Qwen3.5", prompt_tokens, capacity, params)
    }

    /// Store the checkpoint's parsed `generation_config.json` defaults.
    /// Called once at load time after construction.
    pub(crate) fn set_gen_defaults(&mut self, defaults: crate::engine::ModelGenerationDefaults) {
        self.gen_defaults = defaults;
    }

    /// Initialize KV caches.
    pub(crate) fn init_caches_sync(&mut self) -> Result<()> {
        self.caches = Some(fresh_dense_layer_caches(&self.config));
        self.clear_reuse_state();
        Ok(())
    }

    /// Reset all caches.
    pub(crate) fn reset_caches_sync(&mut self) -> Result<()> {
        if let Some(ref mut caches) = self.caches {
            for cache in caches.iter_mut() {
                cache.reset();
            }
        }
        self.caches = None;
        self.clear_reuse_state();
        Ok(())
    }

    /// Clear cached token history, image key, and rope deltas.
    fn clear_reuse_state(&mut self) {
        self.cached_token_history.clear();
        self.cached_image_key = None;
        self.cached_paged_image_token_positions.clear();
        self.cached_rope_deltas = None;
        self.gdn_prefix_checkpoints.clear();
        self.gdn_last_history_checkpoint = None;
        self.gdn_root_cache_owner_id = None;
        self.gdn_root_cache_owner_is_explicit = false;
        self.paged_finalize_failed = false;
    }

    /// Tear down a partially prepared or partially executed paged turn.
    ///
    /// Releasing only the adapter is insufficient for this hybrid model: the
    /// GDN caches may already have advanced while `cached_token_history`, the
    /// image identity, and the M-RoPE delta still describe the previous turn.
    /// A subsequent delta would then observe `caches.is_some()` and attempt to
    /// continue a session whose K/V and recurrent state no longer agree. Keep
    /// content-addressed full blocks in the allocator, but invalidate every
    /// model-local live-session signal and recurrent checkpoint.
    fn discard_dense_paged_session(&mut self) {
        if let Some(adapter) = self.paged_adapter.as_mut()
            && let Err(release_error) = adapter.release_request()
        {
            tracing::warn!(
                target: "mlx_core::qwen3_5::paged",
                "failed to release dense paged request during invalidation: {release_error}",
            );
        }
        if let Some(caches) = self.caches.as_mut() {
            for cache in caches {
                cache.reset();
            }
        }
        self.caches = None;
        self.clear_reuse_state();
        self.paged_full_attn_caches_dirty = false;
        self.flat_mtp_caches_desynced = false;
    }

    fn invalidate_dense_paged_session(&mut self, context: &str) {
        tracing::warn!(
            target: "mlx_core::qwen3_5::paged",
            "invalidating dense paged session after {context}",
        );
        self.discard_dense_paged_session();
    }

    /// Fallible terminal lifecycle for the hand-written paged cores.
    ///
    /// Unlike [`PagedBackend::finalize_paged_turn`], these cores can propagate
    /// an error directly. Never publish token history or a GDN sidecar after a
    /// failed registration: release the request and make the session cold.
    ///
    /// On success the GDN sidecar capture runs here, mirroring the engine hook —
    /// these cores own every planned-MTP turn, and without this an MTP session
    /// persists K/V blocks whose recurrent half no restore can ever reconstruct.
    /// Unlike the engine hook there is no release to order against: these cores
    /// only ever keep the request live, so the adapter's cold-chain frontier is
    /// still set when the capture reads it.
    fn finalize_dense_manual_paged_turn(
        &mut self,
        image_token_positions: &[(u32, u64)],
    ) -> Result<()> {
        let finalize_result = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| "dense manual paged finalization: paged_adapter is None".to_owned())
            .and_then(|adapter| {
                let finalize_extra_keys = engine::build_paged_extra_keys(
                    adapter.request_tokens().len(),
                    adapter.block_size(),
                    image_token_positions,
                );
                adapter.finalize_turn_keep_live_per_block(&finalize_extra_keys, 0)
            });
        match finalize_result {
            Ok(_) => {
                self.capture_dense_gdn_cold_sidecar(image_token_positions);
                Ok(())
            }
            Err(error) => {
                self.invalidate_dense_paged_session("manual finalization failure");
                Err(Error::from_reason(format!(
                    "dense paged finalization failed: {error}"
                )))
            }
        }
    }

    /// Downgrade a paged turn whose terminal adapter lifecycle failed.
    ///
    /// `PagedBackend::finalize_paged_turn` is intentionally infallible, and the
    /// engine calls `save_paged_history` immediately after it. Merely releasing
    /// the adapter is therefore insufficient: the save would otherwise publish
    /// the expanded image-placeholder ids as a continuable session even though
    /// their image-conditioned K/V was not kept live or registered. Drop every
    /// live-session signal here and leave a latch for `save_paged_history` to
    /// consume without publishing.
    fn downgrade_failed_paged_finalize(&mut self, error: &str) {
        tracing::warn!(
            target: "mlx_core::qwen3_5::paged",
            "paged adapter finalization failed; invalidating the dense session: {error}",
        );
        self.discard_dense_paged_session();
        self.paged_finalize_failed = true;
    }

    fn find_dense_gdn_history_checkpoint(
        &self,
        tokens: &[u32],
        prefix_len: u32,
        expected_image_key: Option<u64>,
    ) -> Option<Vec<Qwen3_5LayerCache>> {
        let prefix_tokens = tokens.get(..prefix_len as usize)?;
        let checkpoint = self.gdn_last_history_checkpoint.as_ref()?;
        if checkpoint.owner_id != self.active_cache_owner_id
            || checkpoint.image_key != expected_image_key
            || checkpoint.tokens.as_slice() != prefix_tokens
        {
            return None;
        }
        clone_dense_linear_layer_caches(&self.config, &checkpoint.caches)
    }

    fn remember_dense_gdn_history_checkpoint(&mut self) -> Result<DenseGdnCheckpointStoreTrace> {
        let trace_enabled = inference_trace_enabled();
        let total_start = trace_enabled.then(std::time::Instant::now);
        let mut trace = DenseGdnCheckpointStoreTrace::default();
        if self.cached_token_history.is_empty() {
            self.gdn_last_history_checkpoint = None;
            return Ok(trace.finish(total_start));
        }

        let eval_start = trace_enabled.then(std::time::Instant::now);
        eval_layer_caches(&self.caches)?;
        trace.eval_ms = eval_start.map(elapsed_ms).unwrap_or(0.0);
        let clone_start = trace_enabled.then(std::time::Instant::now);
        let Some(caches) = self
            .caches
            .as_ref()
            .and_then(|caches| clone_dense_linear_layer_caches(&self.config, caches))
        else {
            self.gdn_last_history_checkpoint = None;
            trace.clone_ms = clone_start.map(elapsed_ms).unwrap_or(0.0);
            return Ok(trace.finish(total_start));
        };
        trace.clone_ms = clone_start.map(elapsed_ms).unwrap_or(0.0);
        let token_clone_start = trace_enabled.then(std::time::Instant::now);
        let tokens = self.cached_token_history.clone();
        trace.token_clone_ms = token_clone_start.map(elapsed_ms).unwrap_or(0.0);

        let update_start = trace_enabled.then(std::time::Instant::now);
        self.gdn_last_history_checkpoint = Some(DenseGdnHistoryCheckpoint {
            owner_id: self.active_cache_owner_id.clone(),
            image_key: self.cached_image_key,
            tokens,
            caches,
        });
        trace.update_ms = update_start.map(elapsed_ms).unwrap_or(0.0);
        trace.stored = true;
        Ok(trace.finish(total_start))
    }

    fn find_dense_gdn_prefix_checkpoint(
        &mut self,
        tokens: &[u32],
        requested_prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
    ) -> Option<(u32, Vec<Qwen3_5LayerCache>)> {
        let checkpoint_idx = find_longest_valid_gdn_checkpoint_index(
            &self.gdn_prefix_checkpoints,
            &self.active_cache_owner_id,
            tokens,
            requested_prefix_len,
            block_size,
            extra_keys_per_block,
            cache_salt,
            |checkpoint| dense_paged_linear_caches_ready(&self.config, Some(&checkpoint.caches)),
        )?;
        let checkpoint = &self.gdn_prefix_checkpoints[checkpoint_idx];
        let prefix_len = checkpoint.prefix_len;
        let caches = clone_dense_linear_layer_caches(&self.config, &checkpoint.caches)?;
        // Successful lookup is an LRU touch. The model thread serializes all
        // turns, so moving the owner entry cannot race another request.
        let checkpoint = self.gdn_prefix_checkpoints.remove(checkpoint_idx)?;
        self.gdn_prefix_checkpoints.push_back(checkpoint);
        Some((prefix_len, caches))
    }

    /// Install GDN recurrent state the SSD cold tier restored alongside this
    /// turn's paged K/V prefix.
    ///
    /// This is the cold-tier twin of the in-memory checkpoint lookups above:
    /// same destination (`self.caches`), different source (an on-disk
    /// [`mlx_paged_attn::ColdSidecar`] instead of a live materialized
    /// checkpoint). Consulted ONLY after every in-memory source missed, because
    /// those are already materialized and cost no decode.
    ///
    /// `ColdTierWalk::restore_extend` guarantees the sidecar backs EXACTLY the
    /// prefix the adapter reported, so no boundary re-derivation is needed — but
    /// every structural precondition is re-checked here (group, layout equality
    /// against this config's geometry, boundary == the reported prefix). A
    /// contract slip degrades to a MISS (`Ok(false)`) that falls through to the
    /// caller's replay, never state installed at the wrong offset.
    ///
    /// Taking the sidecar is unconditional so a rejected one cannot be
    /// reconsidered later in the same turn. On success the decoded state is also
    /// fed into the in-memory prefix store so later same-process turns hit RAM.
    fn install_dense_gdn_cold_sidecar(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
    ) -> Result<bool> {
        let sidecar = {
            let Some(adapter) = self.paged_adapter.as_mut() else {
                return Ok(false);
            };
            match adapter.take_restored_sidecar() {
                Some(sidecar) => sidecar,
                None => return Ok(false),
            }
        };
        if sidecar.layout.group != mlx_paged_attn::ColdGroup::GdnState {
            return Ok(false);
        }
        let boundary = sidecar.layout.boundary_tokens;
        // The walk reconciles the prefix and the state together, so a sidecar
        // that does not describe EXACTLY the reported prefix is a broken
        // contract, not a deeper opportunity: refuse it.
        if boundary == 0 || boundary != cached_prefix_len {
            return Ok(false);
        }
        let cache_dtype = {
            let Some(adapter) = self.paged_adapter.as_ref() else {
                return Ok(false);
            };
            format!("{:?}", adapter.layer_kv_pool().cache_dtype())
        };
        let Some(geometry) = super::gdn_sidecar::geometry(&self.config, &cache_dtype) else {
            return Ok(false);
        };
        // `load_sidecar` already compared the layout to the policy's template;
        // comparing again against the geometry derived from the LOADED config
        // makes the install independent of that earlier check.
        if super::gdn_sidecar::layout_at(&geometry, boundary) != sidecar.layout {
            return Ok(false);
        }
        let Some(caches) =
            super::gdn_sidecar::decode_caches(&self.config, &geometry, &sidecar.tensors, boundary)?
        else {
            return Ok(false);
        };
        self.caches = Some(caches);
        // The one observable that separates "the tier restored the recurrent
        // half" from "the tier read it and every arm above declined". Both
        // produce identical text, identical `num_tokens` and identical
        // `cached_tokens`; only the second re-forwards the whole prefix.
        crate::cold_tier::cold_sidecar_counters().record_install();
        // Feed the in-memory store so later turns in this process hit RAM
        // instead of decoding the sidecar again. Best-effort: a failure to
        // clone/store never invalidates the freshly installed live caches.
        if let Some(snapshot) = self
            .caches
            .as_ref()
            .and_then(|caches| clone_dense_linear_layer_caches(&self.config, caches))
        {
            let checkpoint = super::paged_forward::MaterializedGdnPrefixCheckpoint {
                prefix_len: boundary,
                caches: snapshot,
            };
            self.remember_dense_gdn_materialized_prefix_checkpoint(
                tokens,
                block_size,
                extra_keys_per_block,
                cache_salt,
                checkpoint,
            );
        }
        Ok(true)
    }

    fn remember_dense_gdn_materialized_prefix_checkpoint(
        &mut self,
        tokens: &[u32],
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
        checkpoint: super::paged_forward::MaterializedGdnPrefixCheckpoint,
    ) -> bool {
        let prefix_len = checkpoint.prefix_len;
        if !dense_paged_linear_caches_ready(&self.config, Some(&checkpoint.caches)) {
            return false;
        }
        let Some(block_hashes) = compute_paged_prefix_block_hashes(
            tokens,
            prefix_len,
            block_size,
            extra_keys_per_block,
            cache_salt,
        ) else {
            return false;
        };
        let Some(final_block_hash) = block_hashes.last().copied() else {
            return false;
        };
        let Some(prefix_tokens) = tokens.get(..prefix_len as usize) else {
            return false;
        };

        self.gdn_prefix_checkpoints.retain(|existing| {
            !(existing.owner_id == self.active_cache_owner_id
                && existing.prefix_len == prefix_len
                && existing.block_size == block_size
                && existing.final_block_hash == final_block_hash
                && existing.tokens.as_slice() == prefix_tokens)
        });
        self.gdn_prefix_checkpoints
            .push_back(DenseGdnPrefixCheckpoint {
                owner_id: self.active_cache_owner_id.clone(),
                prefix_len,
                block_size,
                final_block_hash,
                block_hashes,
                tokens: prefix_tokens.to_vec(),
                caches: checkpoint.caches,
            });
        self.prune_dense_gdn_prefix_checkpoints();
        true
    }

    fn prune_dense_gdn_prefix_checkpoints(&mut self) {
        // Probe the ladder predicate BEFORE the `&mut self` borrows below:
        // `get_or_insert_with` takes one and `wants_gdn_checkpoint_ladder`
        // needs `&self`, so an inline call in the argument list does not
        // borrow-check.
        let caps = gdn_retention_caps(
            self.wants_gdn_checkpoint_ladder(),
            self.gdn_root_cache_owner_is_explicit,
        );
        let active_owner_id = self.active_cache_owner_id.clone();
        let root_owner_id = self
            .gdn_root_cache_owner_id
            .get_or_insert_with(|| active_owner_id.clone())
            .clone();
        prune_gdn_checkpoints(
            &mut self.gdn_prefix_checkpoints,
            caps,
            &root_owner_id,
            &active_owner_id,
        );
    }

    fn publish_dense_gdn_materialized_prefix_checkpoint(
        &mut self,
        tokens: &[u32],
        checkpoints: Vec<super::paged_forward::MaterializedGdnPrefixCheckpoint>,
    ) {
        let Some(block_size) = self
            .paged_adapter
            .as_ref()
            .map(|adapter| adapter.block_size())
        else {
            return;
        };
        let extra_keys = engine::build_paged_extra_keys(
            tokens.len(),
            block_size,
            &self.cached_paged_image_token_positions,
        );
        self.publish_dense_gdn_materialized_prefix_checkpoint_with_keys(
            tokens,
            &extra_keys,
            0,
            checkpoints,
        );
    }

    fn publish_dense_gdn_materialized_prefix_checkpoint_with_keys(
        &mut self,
        tokens: &[u32],
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
        checkpoints: Vec<super::paged_forward::MaterializedGdnPrefixCheckpoint>,
    ) {
        if checkpoints.is_empty() {
            return;
        }
        let Some(block_size) = self
            .paged_adapter
            .as_ref()
            .map(|adapter| adapter.block_size())
        else {
            return;
        };
        // Ascending, so the deepest boundary is remembered last and is the
        // most recent entry the lookup prefers on a tie.
        for checkpoint in checkpoints {
            let prefix_len = checkpoint.prefix_len;
            let stored = self.remember_dense_gdn_materialized_prefix_checkpoint(
                tokens,
                block_size,
                extra_keys_per_block,
                cache_salt,
                checkpoint,
            );
            if tracing::enabled!(target: "mlx_core::inference", tracing::Level::INFO) {
                tracing::info!(
                    target: "mlx_core::inference",
                    event = "gdn_prefix_checkpoint_store",
                    prefix_tokens = prefix_len,
                    block_size,
                    cache_owner_id = %self.active_cache_owner_id,
                    cache_root_owner_id = %self.gdn_root_cache_owner_id.as_deref().unwrap_or(""),
                    stored,
                    retained_checkpoints = self.gdn_prefix_checkpoints.len(),
                    "dense GDN prefix checkpoint stored"
                );
            }
        }
    }

    /// Effective paged-prefill chunk size for EVERY paged prefill this model
    /// runs — the engine's `paged_prefill` and the hand-written sync/stream
    /// cores alike.
    ///
    /// The GDN cold sidecar can only persist recurrent state at a boundary the
    /// prefill actually materialized state at, and those are exactly the
    /// `gdn_prefill_checkpoint_boundaries` ladder rungs
    /// (`paged_prefill_ranges` forces a break at each rung's offset).
    /// Single-shot prefill (the default when `MLX_PAGED_PREFILL_CHUNK_SIZE` is
    /// unset) crosses no rung, so it produces none of the in-memory checkpoints
    /// the capture reads from.
    ///
    /// So when the SSD cold tier carries a GDN sidecar policy (persistence on),
    /// return a chunk size large enough that the ONLY breaks are the ladder's.
    /// An explicit `MLX_PAGED_PREFILL_CHUNK_SIZE` still wins.
    ///
    /// With no cold GDN policy and no env override this is the unchanged
    /// single-shot default (0), which crosses no boundary at all. With no cold
    /// GDN policy but the env var SET, chunking is on and the break set is
    /// `prefill_checkpoint_boundaries(.., want_ladder = false)` — the single
    /// deep `gdn_checkpoint_target`, exactly what a persist-off turn split at
    /// before the ladder existed. Both arms read
    /// `paged_forward::gdn_cold_sidecar_ladder_wanted` so the chunk size and the
    /// break set cannot disagree about whether this is a persist turn.
    ///
    /// Splitting is algebraically transparent — every attention query still
    /// attends over the whole cumulative range, and a break only marks where
    /// the recurrent state is snapshotted — but it is NOT numerically
    /// bit-identical: the chunk length is the GEMM's M, which selects a
    /// different kernel class and accumulation order. The measured size of that
    /// drift is documented on
    /// `test_chunked_prefill_qwen3_5_moe_matches_single_shot_logits`, which
    /// needs a RELAXED logit tolerance rather than bit-equality. That test
    /// skips on `test_support::half_gemm_untrustworthy` — an 8x64 bf16 GEMM
    /// correctness canary, NOT a GPU-generation gate, so on a host whose
    /// half-precision GEMM is sound it runs and the tolerance is the real
    /// claim. Either way it measures ONE logit vector at 96 tokens, which
    /// bounds nothing about an argmax flip 30 greedy steps into a 1400-token
    /// prompt. So turning persistence on can change the sampled tokens of an
    /// otherwise identical request. That trade was already taken for autoregressive turns, which
    /// have read this since the sidecar landed; the hand-written cores now take
    /// it too rather than silently persisting K/V whose recurrent half no
    /// restore can reconstruct.
    ///
    /// A caller must read this BEFORE it takes the `&mut self` borrows the
    /// prefill needs (`&mut self.layers` + `&mut self.paged_adapter`); an inline
    /// call in the argument list does not borrow-check.
    fn cold_gdn_prefill_chunk_size(&self) -> i32 {
        let env_chunk = crate::array::paged_prefill_chunk_size();
        if env_chunk > 0 {
            return env_chunk;
        }
        if self.wants_gdn_checkpoint_ladder() {
            i32::MAX
        } else {
            0
        }
    }

    /// Whether this turn's prefill should publish the whole checkpoint ladder.
    ///
    /// Thin wrapper over [`super::paged_forward::gdn_cold_sidecar_ladder_wanted`]
    /// so the model layer and the prefill body probe the SAME predicate.
    fn wants_gdn_checkpoint_ladder(&self) -> bool {
        self.paged_adapter
            .as_ref()
            .is_some_and(super::paged_forward::gdn_cold_sidecar_ladder_wanted)
    }

    /// Prefill for the hand-written paged cores (sync + stream, text).
    ///
    /// The engine's `PagedBackend::paged_prefill` serves only AR turns; a
    /// planned-MTP turn returns from `paged_whole_turn` before the engine runs
    /// and prefills here instead. Both cores share this one body so the chunk
    /// size — and with it the whole checkpoint ladder a cold sidecar is anchored
    /// on — cannot be right in one core and wrong in the other.
    ///
    /// `keep_prompt_hidden_tokens` is `Some` exactly when the caller wants the
    /// per-token prompt hidden for the MTP prompt-prefix seed; the returned
    /// hidden is `Some` on that arm and `None` otherwise.
    #[allow(clippy::too_many_arguments)]
    fn run_dense_core_paged_prefill(
        &mut self,
        tokens: &[u32],
        suffix: &[u32],
        cached_prefix_len: u32,
        gdn_prefix_already_primed: bool,
        keep_prompt_hidden_tokens: Option<usize>,
        layer_kinds: &[super::decoder_layer::Qwen3_5LayerKind],
        context: &'static str,
    ) -> Result<(
        MxArray,
        Option<MxArray>,
        Vec<super::paged_forward::MaterializedGdnPrefixCheckpoint>,
    )> {
        let embed = self.embedding.clone();
        let embedding_weight = embed.get_weight();
        // Cross-turn M-RoPE delta (0 unless this text turn warm-continues an
        // image prefill); feeds the scalar-offset RoPE for the suffix.
        let rope_deltas = self.cached_rope_deltas.unwrap_or(0);
        let chunk_size = self.cold_gdn_prefill_chunk_size();
        let caches_ref = self
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason(format!("{context}: caches not initialized")))?;
        let adapter = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason(format!("{context}: paged_adapter dropped")))?;
        if let Some(keep_tokens) = keep_prompt_hidden_tokens {
            let (logits, hidden, checkpoints) =
                super::paged_forward::run_paged_prefill_chunk_with_hidden_with_size(
                    tokens,
                    suffix,
                    cached_prefix_len,
                    gdn_prefix_already_primed,
                    &embed,
                    &mut self.layers,
                    caches_ref,
                    &self.final_norm,
                    &self.lm_head,
                    &embedding_weight,
                    layer_kinds,
                    adapter,
                    chunk_size,
                    Some(keep_tokens),
                    rope_deltas,
                )?;
            Ok((logits, Some(hidden), checkpoints))
        } else {
            let (logits, checkpoints) = super::paged_forward::run_paged_prefill_chunk_with_size(
                tokens,
                suffix,
                cached_prefix_len,
                gdn_prefix_already_primed,
                &embed,
                &mut self.layers,
                caches_ref,
                &self.final_norm,
                &self.lm_head,
                &embedding_weight,
                layer_kinds,
                adapter,
                chunk_size,
                rope_deltas,
            )?;
            Ok((logits, None, checkpoints))
        }
    }

    /// Persist this turn's GDN recurrent state to the SSD cold tier, so a later
    /// process can resume from the restored paged K/V prefix WITHOUT replaying
    /// GDN over it (`run_gdn_only_prefill_materialized`).
    ///
    /// Best-effort and infallible by construction — every failure path is a
    /// silent skip. A missing sidecar is never a correctness problem: the
    /// restore walk reconciles the candidate prefix down past that boundary and
    /// the state is recomputed exactly as it is today.
    ///
    /// A GDN recurrent state is valid ONLY at the exact prefix length it was
    /// produced at (vLLM `MambaSpec`), so the boundary is not chosen freely: it
    /// is the DEEPEST already-materialized in-memory GDN checkpoint (a rung of
    /// the `gdn_prefill_checkpoint_boundaries` ladder a prior prefill published)
    /// whose block-aligned prefix the persisted K/V chain ALSO reaches
    /// (`cold_captured_blocks`). A sidecar past the chain's frontier could never
    /// be selected on restore, so writing one would only burn quota.
    ///
    /// At most ONE sidecar per turn (~75 MiB on the 27B); the writer queue is
    /// bounded. Media turns are skipped in v1: their image-aware keys already
    /// isolate them, but the GDN VLM exactness gate is not modeled here, so
    /// refusing is the fail-closed answer.
    ///
    /// `image_token_positions` is the turn's OWN media map, passed in rather
    /// than read off `self`: the VLM cores clear
    /// `cached_paged_image_token_positions` before their prefill and reassign it
    /// only after finalization returns, so a `self`-read guard would see an
    /// empty vec on exactly the media turns it exists to refuse.
    fn capture_dense_gdn_cold_sidecar(&self, image_token_positions: &[(u32, u64)]) {
        crate::cold_tier::cold_sidecar_counters().record_capture_reached();
        let Some(adapter) = self.paged_adapter.as_ref() else {
            return;
        };
        let Some(cold) = adapter.cold_tier() else {
            return;
        };
        let Some(policy) = cold.sidecar_policy.as_ref() else {
            return;
        };
        if policy.group() != mlx_paged_attn::ColdGroup::GdnState {
            return;
        }
        // v1: text-only. See the doc comment.
        if !image_token_positions.is_empty() {
            return;
        }
        let cache_dtype = format!("{:?}", adapter.layer_kv_pool().cache_dtype());
        let Some(geometry) = super::gdn_sidecar::geometry(&self.config, &cache_dtype) else {
            return;
        };
        let block_size = adapter.block_size();
        // The K/V capture walk that just ran spends its budget waiting for
        // writer-queue slots, so it hands this sidecar a queue it may well have
        // filled microseconds ago. A non-blocking offer here loses that race,
        // and a dropped sidecar is strictly worse than a dropped block: the
        // restore reconciles down to the deepest boundary a VALIDATED sidecar
        // backs, so losing it makes the turn's entire persisted K/V chain
        // unusable. Wait out the same budget the walk did.
        let sidecar_wait = adapter.cold_capture_budget().max_walk;
        if block_size == 0 {
            return;
        }
        let request_tokens = adapter.request_tokens();
        // Whole blocks of this request the persisted K/V chain actually covers.
        let full_blocks = request_tokens.len() / block_size as usize;
        let chain_blocks = (adapter.cold_captured_blocks() as usize).min(full_blocks);
        if chain_blocks == 0 {
            // A different diagnosis from the boundary miss below: the persisted
            // chain covers no whole block of this request, so no checkpoint at
            // any depth could have been used. This is the arm a genuinely cold
            // process hits on its first turns, while the bounded writer queue is
            // still ratcheting the chain forward.
            crate::cold_tier::cold_sidecar_counters().record_chain_empty();
            if inference_trace_enabled() {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] qwen3_5 gdn_cold_sidecar_capture_skipped reason=persisted_chain_covers_no_whole_block cold_captured_blocks={} full_blocks={} block_size={} prompt_len={}",
                    adapter.cold_captured_blocks(),
                    full_blocks,
                    block_size,
                    request_tokens.len()
                ));
            }
            return;
        }
        let ceiling_tokens = u32::try_from((chain_blocks as u64).saturating_mul(block_size as u64))
            .unwrap_or(u32::MAX);
        let extra_keys_per_block =
            engine::build_paged_extra_keys(request_tokens.len(), block_size, image_token_positions);

        // Deepest same-owner in-memory GDN checkpoint whose block-hash chain
        // matches this request and whose prefix the KV chain reaches. `cache_salt`
        // is 0 here to match the KV-chain finalize
        // (`finalize_turn_keep_live_per_block(.., 0)`) and the in-memory publish,
        // so the restore walk — which chains the sidecar under the SAME salt as
        // the KV blocks — can select it.
        let (idx, boundary) = match select_gdn_sidecar_boundary(
            &self.gdn_prefix_checkpoints,
            &self.active_cache_owner_id,
            request_tokens,
            ceiling_tokens,
            block_size,
            &extra_keys_per_block,
            0,
            |checkpoint| dense_paged_linear_caches_ready(&self.config, Some(&checkpoint.caches)),
        ) {
            GdnSidecarBoundary::Selected { index, boundary } => (index, boundary),
            GdnSidecarBoundary::Missed(miss) => {
                // Nothing is written and the restore walk will later reconcile
                // the whole candidate prefix away. Count and trace it, otherwise
                // a session that never persists recurrent state is
                // indistinguishable from one that needs none.
                crate::cold_tier::cold_sidecar_counters().record_boundary_skip();
                if inference_trace_enabled() {
                    let reason = match miss.deepest_retained {
                        Some(deepest) if deepest > ceiling_tokens => {
                            "deepest_checkpoint_past_chain_reach"
                        }
                        _ => "no_checkpoint_on_this_lineage_at_or_below_chain_reach",
                    };
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] qwen3_5 gdn_cold_sidecar_capture_skipped reason={} ceiling_tokens={} chain_blocks={} cold_captured_blocks={} block_size={} prompt_len={} deepest_retained_boundary={} retained_for_owner={} retained_total={}",
                        reason,
                        ceiling_tokens,
                        chain_blocks,
                        adapter.cold_captured_blocks(),
                        block_size,
                        request_tokens.len(),
                        miss.deepest_retained.unwrap_or(0),
                        miss.retained_for_owner,
                        miss.retained_total
                    ));
                }
                return;
            }
        };
        let checkpoint = &self.gdn_prefix_checkpoints[idx];

        // Derive the GdnState chain key for blocks `0..boundary/block_size` — the
        // KV chain recomputed under the `GdnState` domain tag (vLLM's
        // `BlockHashWithGroupId`). `ColdTierWalk::deepest_backed_boundary`
        // derives the identical chain on restore. Built BEFORE the payload so the
        // `contains_in` dedup below can skip the whole encode (a device->host
        // readback of tens of MiB) when the chain is already on disk.
        let blocks = boundary as usize / block_size as usize;
        let mut parent: Option<mlx_paged_attn::ColdCacheKey> = None;
        for index in 0..blocks {
            let (Some(extra_keys), Some(tokens)) = (
                extra_keys_per_block.get(index),
                request_tokens.get(index * block_size as usize..(index + 1) * block_size as usize),
            ) else {
                return;
            };
            parent = Some(mlx_paged_attn::ColdCacheKey::chain(
                mlx_paged_attn::ColdGroup::GdnState,
                cold.fingerprint,
                parent,
                tokens,
                extra_keys,
                0,
                index,
            ));
        }
        let Some(key) = parent else {
            return;
        };
        // Already persisted for this exact chain: nothing to do. `contains_in` is
        // explicitly side-effect free (no hit/miss accounting), so this arm must
        // do its own — an unrecorded exit here reads downstream as `enqueued=0`,
        // which is also what a collapsed checkpoint ladder produces.
        if cold
            .manager
            .contains_in(&key, mlx_paged_attn::ColdGroup::GdnState)
        {
            crate::cold_tier::cold_sidecar_counters().record_already_persisted();
            return;
        }

        let Ok(Some(tensors)) = super::gdn_sidecar::encode_tensors(
            &self.config,
            &geometry,
            &checkpoint.caches,
            boundary,
        ) else {
            return;
        };
        let sidecar = mlx_paged_attn::ColdSidecar {
            key,
            fingerprint: cold.fingerprint,
            layout: super::gdn_sidecar::layout_at(&geometry, boundary),
            tensors,
        };
        match cold
            .manager
            .enqueue_sidecar_before(sidecar, std::time::Instant::now() + sidecar_wait)
        {
            Ok(true) => crate::cold_tier::cold_sidecar_counters().record_enqueued(),
            // The bounded writer queue stayed full for the whole capture
            // budget. Nothing is written and nothing failed, so this turn is
            // otherwise indistinguishable from a successful capture.
            Ok(false) => {
                crate::cold_tier::cold_sidecar_counters().record_queue_drop();
                tracing::debug!(
                    target: "mlx_core::qwen3_5::paged",
                    "qwen3.5 GDN sidecar dropped at boundary {boundary}: cold-cache writer queue full"
                );
            }
            Err(error) => tracing::debug!(
                target: "mlx_core::qwen3_5::paged",
                "qwen3.5 GDN sidecar enqueue failed at boundary {boundary}: {error}"
            ),
        }
    }

    fn prepare_dense_gdn_prefix_state(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
        continued_live_prefix: bool,
    ) -> Result<DenseGdnPrefixPreparation> {
        let image_aware_prefix = extra_keys_per_block.iter().any(|keys| !keys.is_empty());
        let trace_enabled = inference_trace_enabled();
        let inference_info_enabled =
            tracing::enabled!(target: "mlx_core::inference", tracing::Level::INFO);
        let prepare_start = (trace_enabled || inference_info_enabled).then(std::time::Instant::now);
        let cache_owner_id = self.active_cache_owner_id.clone();
        let cache_root_owner_id = self.gdn_root_cache_owner_id.clone().unwrap_or_default();
        let finish = |state: &'static str,
                      restored_prefix_tokens: u32,
                      replayed_prefix_tokens: u32|
         -> DenseGdnPrefixPreparation {
            if inference_info_enabled {
                tracing::info!(
                    target: "mlx_core::inference",
                    event = "gdn_prefix_prepare",
                    cache_owner_id = %cache_owner_id,
                    cache_root_owner_id = %cache_root_owner_id,
                    state,
                    cached_prefix_tokens = cached_prefix_len,
                    restored_prefix_tokens,
                    replayed_prefix_tokens,
                    elapsed_ms = prepare_start.map(elapsed_ms).unwrap_or(0.0),
                    "dense GDN prefix state prepared"
                );
            }
            let preparation = DenseGdnPrefixPreparation {
                state,
                already_primed: cached_prefix_len > 0,
                restored_prefix_tokens,
                replayed_prefix_tokens,
            };
            debug_assert_eq!(
                preparation.already_primed,
                preparation.restored_prefix_tokens > 0 || preparation.replayed_prefix_tokens > 0,
                "dense GDN prefix preparation must account for every primed prefix"
            );
            preparation
        };
        let gdn_caches_ready =
            dense_paged_linear_caches_ready(&self.config, self.caches.as_deref());
        if gdn_caches_ready && continued_live_prefix {
            return Ok(finish("live", cached_prefix_len, 0));
        }

        let gdn_prefix_from_history = cached_prefix_len > 0
            && self.cached_token_history.len() == cached_prefix_len as usize
            && tokens.starts_with(&self.cached_token_history);
        if gdn_caches_ready && gdn_prefix_from_history {
            return Ok(finish("last_history", cached_prefix_len, 0));
        }

        if cached_prefix_len > 0 {
            let history_lookup_start = trace_enabled.then(std::time::Instant::now);
            let history_checkpoint =
                self.find_dense_gdn_history_checkpoint(tokens, cached_prefix_len, None);
            let history_lookup_ms = history_lookup_start.map(elapsed_ms);
            if let Some(checkpoint) = history_checkpoint {
                self.caches = Some(checkpoint);
                return Ok(finish("last_history_checkpoint", cached_prefix_len, 0));
            } else if trace_enabled {
                let history_checkpoint_len = self
                    .gdn_last_history_checkpoint
                    .as_ref()
                    .map_or(0, |checkpoint| checkpoint.tokens.len());
                let history_mismatch =
                    token_prefix_mismatch_trace(tokens, &self.cached_token_history);
                write_inference_trace(format_args!(
                    "[MLX_TRACE] qwen3.5-dense gdn_history_checkpoint_miss \
                     cached_prefix_tokens={} history_len={} checkpoint_len={} \
                     history_match={} history_mismatch_at={} prompt_token={} \
                     history_token={} history_lookup_ms={:.1}",
                    cached_prefix_len,
                    self.cached_token_history.len(),
                    history_checkpoint_len,
                    gdn_prefix_from_history,
                    history_mismatch.index,
                    history_mismatch.prompt_token,
                    history_mismatch.cached_token,
                    history_lookup_ms.unwrap_or(0.0)
                ));
            }
        }

        let prefix_checkpoint = self.find_dense_gdn_prefix_checkpoint(
            tokens,
            cached_prefix_len,
            block_size,
            extra_keys_per_block,
            cache_salt,
        );
        if let Some((restored_prefix_len, checkpoint)) = prefix_checkpoint {
            let replayed_prefix_len = cached_prefix_len
                .checked_sub(restored_prefix_len)
                .ok_or_else(|| {
                    Error::from_reason(
                        "dense GDN checkpoint is longer than the cached paged prefix",
                    )
                })?;
            if replayed_prefix_len == 0 {
                self.caches = Some(checkpoint);
                return Ok(finish("checkpoint", restored_prefix_len, 0));
            }
            if image_aware_prefix {
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    let _ = adapter.release_request();
                }
                self.caches = Some(fresh_dense_layer_caches(&self.config));
                return Err(Error::from_reason(
                    "image-conditioned GDN prefix requires an exact checkpoint or the original image embeddings",
                ));
            }

            let replay_suffix = tokens
                .get(restored_prefix_len as usize..cached_prefix_len as usize)
                .ok_or_else(|| {
                    Error::from_reason(
                        "dense paged GDN checkpoint replay range exceeds prompt length",
                    )
                })?;
            let embed = self.embedding.clone();
            let layers = &mut self.layers;
            replay_gdn_cache_and_commit(&mut self.caches, checkpoint, |staged| {
                super::paged_forward::run_gdn_only_prefill_materialized(
                    replay_suffix,
                    &embed,
                    layers,
                    staged,
                )
            })?;
            return Ok(finish(
                "checkpoint_replay_materialized",
                restored_prefix_len,
                replayed_prefix_len,
            ));
        }

        // Cold-tier GDN sidecar: on-SSD recurrent state the restore walk brought
        // back for EXACTLY this cached prefix (reconcile-down guarantees the
        // boundary). Consulted only after every in-memory source above missed —
        // i.e. the common process-restart case. v1 is text-only: an image-aware
        // prefix still goes through the exactness gates below.
        if cached_prefix_len > 0
            && !image_aware_prefix
            && self.install_dense_gdn_cold_sidecar(
                tokens,
                cached_prefix_len,
                block_size,
                extra_keys_per_block,
                cache_salt,
            )?
        {
            return Ok(finish("cold_sidecar", cached_prefix_len, 0));
        }

        let fresh_caches = fresh_dense_layer_caches(&self.config);
        if cached_prefix_len == 0 {
            self.caches = Some(fresh_caches);
            return Ok(finish("cold", 0, 0));
        }
        if image_aware_prefix {
            if let Some(adapter) = self.paged_adapter.as_mut() {
                let _ = adapter.release_request();
            }
            self.caches = Some(fresh_caches);
            return Err(Error::from_reason(
                "image-conditioned GDN prefix cannot be reconstructed from placeholder token ids",
            ));
        }

        let prefix = tokens.get(..cached_prefix_len as usize).ok_or_else(|| {
            Error::from_reason("dense paged GDN prefix replay length exceeds prompt length")
        })?;
        let embed = self.embedding.clone();
        let layers = &mut self.layers;
        replay_gdn_cache_and_commit(&mut self.caches, fresh_caches, |staged| {
            super::paged_forward::run_gdn_only_prefill_materialized(prefix, &embed, layers, staged)
        })?;
        Ok(finish("replay_materialized", 0, cached_prefix_len))
    }

    /// Restore only an identity-matched GDN sidecar for an image-bearing
    /// paged prefix. If no exact sidecar exists, install fresh recurrent caches;
    /// the caller must discard the K/V candidate and rerun the full image merge
    /// from position zero.
    fn prepare_dense_gdn_vlm_prefix_state(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
        continued_live_prefix: bool,
        image_key: u64,
    ) -> Result<bool> {
        if cached_prefix_len == 0 {
            self.caches = Some(fresh_dense_layer_caches(&self.config));
            return Ok(false);
        }

        let caches_ready = dense_paged_linear_caches_ready(&self.config, self.caches.as_deref());
        if caches_ready && continued_live_prefix && self.cached_image_key == Some(image_key) {
            return Ok(true);
        }
        let active_history_matches = caches_ready
            && self.cached_image_key == Some(image_key)
            && self.cached_token_history.len() == cached_prefix_len as usize
            && tokens.starts_with(&self.cached_token_history);
        if active_history_matches {
            return Ok(true);
        }
        if let Some(checkpoint) =
            self.find_dense_gdn_history_checkpoint(tokens, cached_prefix_len, Some(image_key))
        {
            self.caches = Some(checkpoint);
            return Ok(true);
        }
        if let Some((restored_prefix_len, checkpoint)) = self.find_dense_gdn_prefix_checkpoint(
            tokens,
            cached_prefix_len,
            block_size,
            extra_keys_per_block,
            cache_salt,
        ) && restored_prefix_len == cached_prefix_len
        {
            self.caches = Some(checkpoint);
            return Ok(true);
        }

        self.caches = Some(fresh_dense_layer_caches(&self.config));
        Ok(false)
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_dense_vlm_paged_prefix(
        &mut self,
        tokens: &[u32],
        total_budget: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        reuse_cache: bool,
        allow_live_continue: bool,
        image_key: u64,
    ) -> Result<engine::VlmPagedPrefixResolution> {
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
        let candidate_plan_result = match self.paged_adapter.as_mut() {
            Some(adapter) => adapter
                .prepare_turn_per_block_with_max_cache_hit_tokens(
                    0,
                    tokens,
                    total_budget,
                    allow_live_continue,
                    extra_keys_per_block,
                    0,
                    !reuse_cache,
                    max_cache_hit_tokens,
                )
                .map_err(Error::from_reason),
            None => Err(Error::from_reason(
                "prepare_dense_vlm_paged_prefix: paged_adapter is None",
            )),
        };
        let candidate_plan = match candidate_plan_result {
            Ok(plan) => plan,
            Err(error) => {
                self.invalidate_dense_paged_session("VLM paged-prefix preparation failure");
                return Err(error);
            }
        };

        let gdn_prefix_already_primed = match self.prepare_dense_gdn_vlm_prefix_state(
            tokens,
            candidate_plan.cached_prefix_len,
            block_size,
            extra_keys_per_block,
            0,
            candidate_plan.continued_live_prefix,
            image_key,
        ) {
            Ok(primed) => primed,
            Err(error) => {
                self.invalidate_dense_paged_session("VLM GDN-prefix preparation failure");
                return Err(error);
            }
        };

        let resolution =
            engine::resolve_vlm_paged_prefix(candidate_plan, gdn_prefix_already_primed, || {
                self.paged_adapter
                    .as_mut()
                    .ok_or_else(|| {
                        "prepare_dense_vlm_paged_prefix: paged_adapter dropped before cold restart"
                            .to_string()
                    })?
                    .restart_prepared_turn_cold_per_block(
                        0,
                        tokens,
                        total_budget,
                        extra_keys_per_block,
                        0,
                    )
            });
        match resolution {
            Ok(resolution) => Ok(resolution),
            Err(error) => {
                self.invalidate_dense_paged_session("VLM cold-restart failure");
                Err(Error::from_reason(error))
            }
        }
    }

    fn first_quantized_save_component(&self) -> Option<String> {
        if self.embedding.is_quantized() {
            return Some("embedding".to_string());
        }
        for (index, layer) in self.layers.iter().enumerate() {
            if layer.is_quantized() {
                return Some(format!("layers.{index}"));
            }
        }
        if self
            .lm_head
            .as_ref()
            .is_some_and(|head| head.is_quantized())
        {
            return Some("lm_head".to_string());
        }
        if self.mtp_weights_loaded
            && self
                .mtp
                .as_ref()
                .is_some_and(|mtp| mtp.has_quantized_weights())
        {
            return Some("mtp".to_string());
        }
        None
    }

    /// Save model weights and configuration to a directory (synchronous).
    pub(crate) fn save_model_sync(&self, save_path: &str) -> Result<()> {
        use super::decoder_layer::AttentionType;

        if let Some(component) = self.first_quantized_save_component() {
            return Err(Error::from_reason(format!(
                "Cannot save model: save_model is dense/BF16-only, but main-model component \
                 '{component}' contains quantized projections. Refusing before creating the \
                 destination because packed weights require sidecars and quantization metadata \
                 that this save format cannot preserve losslessly."
            )));
        }

        let mut params = HashMap::new();

        // Embedding
        params.insert("embedding.weight".to_string(), self.embedding.get_weight());

        // Layers
        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{}", i);
            match &layer.attn {
                AttentionType::Linear(gdn) => {
                    params.insert(
                        format!("{}.linear_attn.in_proj_qkvz.weight", prefix),
                        gdn.get_in_proj_qkvz_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.in_proj_ba.weight", prefix),
                        gdn.get_in_proj_ba_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.conv1d.weight", prefix),
                        gdn.get_conv1d_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.norm.weight", prefix),
                        gdn.get_norm_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.out_proj.weight", prefix),
                        gdn.get_out_proj_weight(),
                    );
                    params.insert(format!("{}.linear_attn.dt_bias", prefix), gdn.get_dt_bias());
                    params.insert(format!("{}.linear_attn.a_log", prefix), gdn.get_a_log());
                }
                AttentionType::Full(attn) => {
                    params.insert(
                        format!("{}.self_attn.q_proj.weight", prefix),
                        attn.get_q_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.k_proj.weight", prefix),
                        attn.get_k_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.v_proj.weight", prefix),
                        attn.get_v_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.o_proj.weight", prefix),
                        attn.get_o_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.q_norm.weight", prefix),
                        attn.get_q_norm_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.k_norm.weight", prefix),
                        attn.get_k_norm_weight(),
                    );
                }
            }
            params.insert(
                format!("{}.mlp.gate_proj.weight", prefix),
                layer.mlp.get_gate_proj_weight(),
            );
            params.insert(
                format!("{}.mlp.up_proj.weight", prefix),
                layer.mlp.get_up_proj_weight(),
            );
            params.insert(
                format!("{}.mlp.down_proj.weight", prefix),
                layer.mlp.get_down_proj_weight(),
            );
            params.insert(
                format!("{}.input_layernorm.weight", prefix),
                layer.get_input_layernorm_weight(),
            );
            params.insert(
                format!("{}.post_attention_layernorm.weight", prefix),
                layer.get_post_attention_layernorm_weight(),
            );
        }

        // Final norm
        params.insert(
            "final_norm.weight".to_string(),
            self.final_norm.get_weight(),
        );

        // LM head
        if !self.config.tie_word_embeddings
            && let Some(ref lm_head) = self.lm_head
        {
            params.insert("lm_head.weight".to_string(), lm_head.get_weight());
        }

        // Include vision encoder weights
        if let Some(ref vision_enc) = self.vision_encoder {
            let vision_params = vision_enc.get_parameters();
            params.extend(vision_params);
        }

        // Multi-Token Prediction head. `config.n_mtp_layers` round-trips
        // through config.json, so a reloaded checkpoint will reconstruct the
        // MTP module from config and expect the `mtp.*` tensors present —
        // without this block the loader would find them absent, set
        // `mtp_weights_loaded = false`, and silently disable speculative
        // decode (only a `warn!`). The `mtp_weights_loaded` guard is
        // essential: `mtp.is_some()` alone would serialize a random-init
        // module (constructed from config even when no weights were loaded).
        if self.mtp_weights_loaded
            && let Some(ref mtp) = self.mtp
        {
            // The early fail-closed quantized-state gate above guarantees this
            // MTP head is dense; partial omission would make the saved model
            // silently lose speculative decoding.
            params.extend(mtp.get_parameters());
        }

        // Validate for NaN/Inf
        for (name, param) in params.iter() {
            let data = param.to_float32()?;
            let invalid_count = data
                .iter()
                .filter(|v| v.is_nan() || v.is_infinite())
                .count();
            if invalid_count > 0 {
                return Err(napi::Error::new(
                    napi::Status::GenericFailure,
                    format!(
                        "Cannot save model: parameter '{}' contains {} NaN/Inf values.",
                        name, invalid_count
                    ),
                ));
            }
        }

        let mut params_clone: HashMap<String, MxArray> =
            params.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

        // Weights metadata
        let mut weights_metadata = serde_json::Map::new();
        for (key, array) in params.iter() {
            let shape_data = array.shape()?;
            let shape: Vec<i64> = shape_data.as_ref().to_vec();
            let dtype = array.dtype()?;
            let mut param_info = serde_json::Map::new();
            param_info.insert("shape".to_string(), serde_json::json!(shape));
            param_info.insert("dtype".to_string(), serde_json::json!(dtype as i32));
            weights_metadata.insert(key.clone(), serde_json::Value::Object(param_info));
        }

        // Config JSON
        let mut config_value = serde_json::to_value(&self.config).map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Failed to serialize config: {e}"),
            )
        })?;
        if let serde_json::Value::Object(ref mut map) = config_value {
            map.insert("model_type".to_string(), serde_json::json!("qwen3_5"));
            // `parse_config` reads the MTP layer count ONLY from the
            // HF-convention keys `mtp_num_hidden_layers` /
            // `num_nextn_predict_layers`; the serde field name
            // `n_mtp_layers` is ignored on load. Without this, a saved MTP
            // checkpoint reloads with `n_mtp_layers = 0` and its head is
            // silently dropped. Mirrors the MoE saver
            // (`qwen3_5_moe::model::Qwen35MoeInner::save_model_sync`).
            map.insert(
                "mtp_num_hidden_layers".to_string(),
                serde_json::json!(self.config.n_mtp_layers),
            );
        }

        let weights_json = serde_json::json!({
            "version": "1.0",
            "config": config_value,
            "weights": weights_metadata,
            "note": "Full weights are in weights.safetensors"
        });

        let path = std::path::Path::new(save_path);
        std::fs::create_dir_all(path)?;

        info!("Saving model to {}", save_path);

        let config_path = path.join("config.json");
        let config_json = serde_json::to_string_pretty(&config_value)?;
        std::fs::write(&config_path, config_json)?;
        info!("Saved config.json");

        let safetensors_path = path.join("weights.safetensors");
        let metadata = Some(serde_json::json!({
            "format": "mlx-node",
            "version": "1.0"
        }));
        crate::utils::safetensors::save_safetensors(
            &safetensors_path,
            &mut params_clone,
            metadata,
        )?;
        info!("Saved weights.safetensors");

        let weights_str = serde_json::to_string_pretty(&weights_json)?;
        let weights_path = path.join("weights.mlx");
        std::fs::write(&weights_path, weights_str)?;
        info!("Saved weights.mlx metadata");

        Ok(())
    }

    /// Set the tokenizer.
    pub(crate) fn set_tokenizer(&mut self, tokenizer: Arc<Qwen3Tokenizer>) {
        self.tokenizer = Some(tokenizer);
    }

    /// Set the vision encoder.
    ///
    /// Paged VLM checkpoints wire this alongside the image processor and
    /// adapter. Image-bearing turns then route through the dedicated paged
    /// vision prefill/decode cores; incomplete stacks stay backend-validated
    /// so those cores can report the precise missing component.
    ///
    /// For text-only inputs M-RoPE collapses to standard scalar-offset
    /// RoPE — `Qwen3_5Attention::forward` uses `self.rope` whenever
    /// `position_ids` is `None`, which is the case for every text-only
    /// flat call. The paged forward (`Qwen3_5Attention::forward_paged`)
    /// also goes through `self.rope` unconditionally. Both paths share
    /// the same RoPE on text-only inputs, so byte-equal parity holds
    /// on VLM checkpoints provided no images are passed.
    pub(crate) fn set_vision_encoder(&mut self, enc: Qwen3_5VisionEncoder) -> Result<()> {
        self.vision_encoder = Some(Arc::new(enc));
        self.vision_cache = Arc::new(Mutex::new(VisionCacheInner::new()));
        Ok(())
    }

    /// Set the image processor.
    pub(crate) fn set_image_processor(&mut self, proc: Qwen35VLImageProcessor) {
        self.image_processor = Some(Arc::new(proc));
        self.vision_cache = Arc::new(Mutex::new(VisionCacheInner::new()));
    }

    /// Set spatial merge size.
    pub(crate) fn set_spatial_merge_size(&mut self, size: i32) {
        self.spatial_merge_size = Some(size);
        self.vision_cache = Arc::new(Mutex::new(VisionCacheInner::new()));
    }

    /// Initialize M-RoPE on all full attention layers (VLM mode).
    pub(crate) fn init_mrope_layers(
        &mut self,
        mrope_section: Vec<i32>,
        rope_theta: f64,
        max_position_embeddings: i32,
    ) -> Result<()> {
        let rope_dims = self.config.rope_dims();
        for layer in self.layers.iter_mut() {
            if let super::decoder_layer::AttentionType::Full(ref mut attn) = layer.attn {
                attn.init_mrope(
                    mrope_section.clone(),
                    rope_theta,
                    max_position_embeddings,
                    rope_dims,
                )?;
            }
        }
        Ok(())
    }

    /// Core synchronous chat implementation (runs on the model thread).
    ///
    /// Whole-turn core for fresh SYNC turns reached through the engine's
    /// `vision_turn` (image-bearing) and `mtp_turn` (MTP-enabled) probes.
    /// The engine already rendered the prompt (`tokens`) and extracted the
    /// raw image payloads (`images`); everything from the paged dispatch
    /// onward runs the whole-turn pipeline. `eos_token_id` is the
    /// caller-supplied stop-on token id (`<|im_end|>` for ChatML
    /// boundaries) so the cached history ends on a clean delimiter that
    /// subsequent session-delta turns can append to.
    fn vision_mtp_whole_turn_core(
        &mut self,
        tokens: Vec<u32>,
        images: &[Vec<u8>],
        config: ChatConfig,
        eos_token_id: u32,
        thinking: ThinkingSetup,
    ) -> Result<ChatResult> {
        let reuse_cache = config.reuse_cache.unwrap_or(true);
        let report_perf = config.report_performance.unwrap_or(false);

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();

        let has_images = !images.is_empty();

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());

        let mut p = extract_chat_params(&config);
        p.extra_eos_ids = self.gen_defaults.eos_token_ids.clone();

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let first_token_instant: Option<std::time::Instant> = None;

        // Paged dispatch with native MTP support; the paged path self-handles
        // MTP via the gate inside `paged_turn_sync_core_inner`.
        if self.paged_adapter.is_some() {
            if has_images {
                // All image turns prefill through the paged-vision core, which
                // runs plain autoregressive decode. MTP weights are ignored
                // here (the core has no draft/verify), so an MTP-enabled
                // session decodes cleanly as AR.
                return self.vision_paged_turn_sync_core(
                    tokens,
                    images,
                    tokenizer,
                    eos_token_id,
                    p,
                    report_perf,
                    thinking,
                );
            }
            return self.paged_turn_sync_core(
                tokens,
                tokenizer,
                eos_token_id,
                p,
                report_perf,
                thinking,
            );
        }

        // The flat fallback below is text-only. A dense image turn requires the
        // block-paged backend; reaching here with images means the model was
        // loaded without a paged adapter (use_block_paged_cache=false, non-Metal
        // build, or a sym8 checkpoint).
        if has_images {
            return Err(Error::from_reason(
                "qwen3.5 dense image turns require the block-paged KV backend; the model was \
                 loaded without a paged adapter (use_block_paged_cache=false, non-Metal build, \
                 or sym8 checkpoint)",
            ));
        }

        let embedding_weight = self.embedding.get_weight();

        // Text-only from here: the `has_images` early-return above is the only
        // image path. These bindings preserve the shared cache-reuse / decode
        // plumbing (`has_images` is always false on this branch).
        let (expanded_tokens, current_image_cache_key) = (tokens.clone(), 0u64);

        // === Cache reuse: prefix verification ===
        let cached_prefix_len = if self.flat_mtp_caches_desynced {
            0
        } else {
            verify_cache_prefix_direct(
                reuse_cache,
                has_images,
                &tokens,
                &expanded_tokens,
                current_image_cache_key,
                &self.cached_token_history,
                &self.cached_image_key,
                self.caches.is_some(),
            )
        };

        let prefill_tokens = if cached_prefix_len > 0 {
            if has_images {
                info!(
                    "VLM cache reuse: {} cached tokens, {} new tokens to prefill",
                    cached_prefix_len,
                    expanded_tokens.len() - cached_prefix_len
                );
                expanded_tokens[cached_prefix_len..].to_vec()
            } else {
                info!(
                    "Cache reuse: {} cached tokens, {} new tokens to prefill",
                    cached_prefix_len,
                    tokens.len() - cached_prefix_len
                );
                tokens[cached_prefix_len..].to_vec()
            }
        } else {
            // Full reset
            if let Some(ref mut caches) = self.caches {
                for cache in caches.iter_mut() {
                    cache.reset();
                }
            }
            let new_caches = (0..self.config.num_layers as usize)
                .map(|i| {
                    if self.config.is_linear_layer(i) {
                        Qwen3_5LayerCache::new_linear()
                    } else {
                        Qwen3_5LayerCache::new_full_attention()
                    }
                })
                .collect();
            self.caches = Some(new_caches);
            tokens.clone()
        };

        // Zero-delta guard.
        //
        // Triggers when `cached_prefix_len == (expanded_)tokens.len()`, i.e.
        // the new prompt is byte-for-byte identical to the cached history
        // and there is literally no delta to prefill. The decode loop still
        // needs a `last_logits` to sample from, so we must run *some*
        // forward pass. Two options were considered:
        //   1. Trim every layer cache by one token and reprefill the final
        //      token. **Infeasible** for Qwen3.5 — the GDN linear-attention
        //      layers store a recurrent state (`conv_state`,
        //      `recurrent_state`) that cannot be rewound mid-sequence
        //      without corrupting the hidden representation. Only the
        //      full-attention layers support `KVCache::trim`, and applying
        //      trim to the hybrid stack would produce silent miscompiles.
        //   2. Full reset + full re-prefill. Wasteful when it triggers but
        //      always correct, and this branch is a cold edge case —
        //      real-world turns always append at least a user message, so
        //      the cached prefix is strictly shorter than the new prompt.
        //
        // We take option 2. The full-reset is intentional, not a "wrong
        // force-reset" to be patched out. See the invariant doc on
        // `verify_cache_prefix_direct` for the underlying linear-attention
        // rewind constraint.
        let (prefill_tokens, cached_prefix_len) = if prefill_tokens.is_empty() {
            info!("Zero-delta cache hit: resetting caches for full re-prefill");
            if let Some(ref mut caches) = self.caches {
                for cache in caches.iter_mut() {
                    cache.reset();
                }
            }
            let new_caches = (0..self.config.num_layers as usize)
                .map(|i| {
                    if self.config.is_linear_layer(i) {
                        Qwen3_5LayerCache::new_linear()
                    } else {
                        Qwen3_5LayerCache::new_full_attention()
                    }
                })
                .collect();
            self.caches = Some(new_caches);
            let tokens = if has_images {
                expanded_tokens.clone()
            } else {
                tokens.clone()
            };
            (tokens, 0)
        } else {
            (prefill_tokens, cached_prefix_len)
        };

        let eos_id = eos_token_id;

        let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        let mut profiler = crate::decode_profiler::DecodeProfiler::new("chat", "qwen3_5");
        profiler.set_prompt_tokens(prefill_tokens.len() as u32);
        profiler.snapshot_memory_before();

        // Prompt-prefix MTP prefill. When MTP is active for this turn the
        // prefill runs the hidden-emitting `chunked_prefill_with_hidden` so the
        // per-prompt-token hiddens can be committed into the MTP
        // committed-history cache; this raises draft acceptance (especially for
        // long prompts). The VLM / cached-prefix branches keep the cheaper
        // logits-only prefill — they do not feed the dense MTP
        // committed-history path.
        //
        // `MLX_MTP_NO_PROMPT_PREFILL=1` opts OUT — the prefill stays
        // logits-only and the MTP committed-history starts empty.
        //
        // Cache-reuse turns: when `cached_prefix_len > 0` the prefill
        // only processes the uncached SUFFIX, so the captured hidden
        // tensor would cover the suffix — not the full prompt. The
        // prompt-prefill seed REQUIRES the full prompt's hiddens, so it
        // is skipped on cache-reuse turns; committed-history still runs
        // (it starts empty and builds from decode tokens — correct).
        let want_prompt_hidden = p.enable_mtp
            && self.has_mtp_weights()
            && !mtp_decode::mtp_no_prompt_prefill()
            && cached_prefix_len == 0;
        let mtp_prompt_history = mtp_decode::mtp_prompt_history_selection(prefill_tokens.len());

        // === Text prefill ===
        profiler.begin_prefill();
        let mut prompt_hidden: Option<MxArray> = None;
        let (last_logits, seq_len) = {
            let prompt = MxArray::from_uint32(&prefill_tokens, &[1, prefill_tokens.len() as i64])?;
            let last_logits = if want_prompt_hidden {
                let (logits, ph) = chunked_prefill_with_hidden(
                    &prompt,
                    &embedding_weight,
                    &mut self.layers,
                    &mut self.caches,
                    &self.final_norm,
                    &self.lm_head,
                    Some(&embedding_weight_t),
                    generation_stream,
                    Some(mtp_prompt_history.keep_tokens),
                )?;
                prompt_hidden = Some(ph);
                logits
            } else {
                chunked_prefill(
                    &prompt,
                    &embedding_weight,
                    &mut self.layers,
                    &mut self.caches,
                    &self.final_norm,
                    &self.lm_head,
                    Some(&embedding_weight_t),
                    generation_stream,
                )?
            };

            (last_logits, tokens.len() as i64)
        };
        profiler.end_prefill();
        // caches now reflect the prefilled history
        self.flat_mtp_caches_desynced = false;

        let prompt_tokens_for_result = if has_images {
            expanded_tokens.len() as u32
        } else {
            tokens.len() as u32
        };

        let save_expanded_tokens = if has_images {
            Some(expanded_tokens.clone())
        } else {
            None
        };

        self.chat_with_caches_inner(ChatDecodeInputs {
            last_logits,
            seq_len,
            is_delta: false,
            has_images,
            token_history_init: tokens.clone(),
            save_tokens: tokens,
            save_expanded_tokens,
            save_image_cache_key: current_image_cache_key,
            tokenizer,
            think_end_id,
            think_end_str,
            thinking,
            eos_id,
            profiler,
            generation_start,
            first_token_instant,
            prefill_tokens_len: prefill_tokens.len(),
            prompt_tokens_for_result,
            // Fresh prefill: report the matched prefix length.
            cached_tokens_for_result: cached_prefix_len as u32,
            embedding_weight,
            embedding_weight_t,
            generation_stream,
            params: p,
            // `prompt_hidden` is `Some` iff the hidden-emitting prefill ran;
            // pair it with the exact prompt tail whose hiddens it holds.
            prompt_hidden_ids: prompt_hidden.as_ref().map(|_| {
                let start = mtp_prompt_history
                    .hidden_start_token_index()
                    .min(prefill_tokens.len());
                prefill_tokens[start..].to_vec()
            }),
            prompt_hidden_position_base: prompt_hidden
                .as_ref()
                .map(|_| mtp_prompt_history.position_base)
                .unwrap_or(0),
            prompt_hidden,
        })
    }

    /// Session-based chat continuation via a pre-tokenized delta.
    ///
    /// Runs a text-only prefill of `delta_tokens` on top of the existing KV
    /// caches and decodes the next reply. This path:
    /// - skips the jinja chat template entirely (caller produces the delta),
    /// - skips prefix verification (caller owns cache coherence by construction),
    /// - uses `<|im_end|>` (from the tokenizer vocab) as its stop token instead
    ///   of `config.eos_token_id`, yielding clean cache boundaries for the next
    ///   turn's delta,
    /// - resolves `enable_thinking` from `config.reasoning_effort` via
    ///   `engine::resolve_enable_thinking`,
    /// - is text-only: errors if the session has images.
    ///
    /// Requires a live session: `self.caches` must have been initialized by a
    /// prior session-start turn. Errors otherwise. (The engine's delta
    /// guards already enforce this; the checks here are defense-in-depth
    /// for the `mtp_turn` caller.)
    pub(crate) fn chat_tokens_delta_sync(
        &mut self,
        delta_tokens: Vec<u32>,
        config: ChatConfig,
        thinking: ThinkingSetup,
    ) -> Result<ChatResult> {
        // The delta path is a session-reuse operation by construction: it
        // prefills on top of the existing caches. `reuse_cache = Some(false)`
        // would make the post-decode `save_cache_state_direct` wipe those
        // caches + `cached_token_history`, making the delta turn both depend
        // on and then destroy the session — confusing and wrong. Reject early
        // so no state is mutated.
        if config.reuse_cache == Some(false) {
            return Err(Error::from_reason(
                "chat_tokens_delta_sync requires reuse_cache to be enabled; \
                 the delta path operates on session state by construction",
            ));
        }
        if self.caches.is_none() {
            return Err(Error::from_reason(
                "chat_tokens_delta_sync requires an initialized session (call chatSessionStart first)",
            ));
        }
        if delta_tokens.is_empty() {
            return Err(Error::from_reason(
                "chat_tokens_delta_sync requires a non-empty delta",
            ));
        }
        // A populated `cached_image_key` means the live KV cache carries
        // attention state for images seen on the preceding prefill. The
        // delta path appends a text delta on top of that — the image
        // context stays intact and the model can keep reasoning about
        // it. We do NOT reject here; the outer `chat_session_continue_*`
        // gate already rejects IMAGE-SET CHANGES (non-empty new images
        // that don't match the cached key) with a prefixed error the TS
        // `ChatSession` can catch and route through `chatSessionStart`.

        let report_perf = config.report_performance.unwrap_or(false);

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();

        // Session path: use <|im_end|> as eos, NOT config.eos_token_id.
        // This yields clean cache boundaries.
        let eos_id = tokenizer
            .im_end_id()
            .ok_or_else(|| Error::from_reason("Tokenizer missing <|im_end|> special token"))?;

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());

        // Build full token history = cached_history + delta. Used for
        // penalty context AND as the running token history in the decode loop.
        // Also used as the snapshot we hand to `save_cache_state_direct` so
        // the saved `cached_token_history` correctly reflects the appended
        // delta plus the generated tokens.
        let mut full_token_history = self.cached_token_history.clone();
        full_token_history.extend(delta_tokens.iter().copied());

        let mut p = extract_chat_params(&config);
        p.extra_eos_ids = self.gen_defaults.eos_token_ids.clone();

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let first_token_instant: Option<std::time::Instant> = None;

        // Paged dispatch with native MTP support inside
        // `paged_turn_sync_core_inner`.
        if self.paged_adapter.is_some() {
            return self.paged_turn_sync_core(
                full_token_history.clone(),
                tokenizer.clone(),
                eos_id,
                p,
                report_perf,
                thinking,
            );
        }

        let embedding_weight = self.embedding.get_weight();
        let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        let mut profiler = crate::decode_profiler::DecodeProfiler::new("chat_delta", "qwen3_5");
        profiler.set_prompt_tokens(delta_tokens.len() as u32);
        profiler.snapshot_memory_before();

        // Text-only prefill of the delta on top of the existing caches.
        profiler.begin_prefill();
        let last_logits = if self.flat_mtp_caches_desynced {
            // A prior eager-MTP turn stopped mid-cycle, leaving self.caches
            // advanced past the emitted history; GDN state cannot be rewound,
            // so discard and re-prefill the full conversation into fresh caches.
            self.caches = Some(fresh_dense_layer_caches(&self.config));
            profiler.set_prompt_tokens(full_token_history.len() as u32);
            let prompt =
                MxArray::from_uint32(&full_token_history, &[1, full_token_history.len() as i64])?;
            let logits = chunked_prefill(
                &prompt,
                &embedding_weight,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                Some(&embedding_weight_t),
                generation_stream,
            )?;
            self.flat_mtp_caches_desynced = false;
            logits
        } else {
            let prompt = MxArray::from_uint32(&delta_tokens, &[1, delta_tokens.len() as i64])?;
            chunked_prefill(
                &prompt,
                &embedding_weight,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                Some(&embedding_weight_t),
                generation_stream,
            )?
        };
        // Total context length post-prefill = full history length.
        let total_seq_len = full_token_history.len() as i64;
        profiler.end_prefill();

        let prompt_tokens_for_result = full_token_history.len() as u32;

        // For the delta path the caches already reflect the entire prior
        // history. For ChatResult observability we still report the
        // cached-prefix length so clients can see the session delta reused
        // the full history: `prior_cached_len` feeds the reported
        // `cached_tokens`.
        let prior_cached_len = full_token_history.len().saturating_sub(delta_tokens.len());

        // For cache save, pass the full token history (cached + delta) as
        // `save_tokens`; the helper / `save_cache_state_direct` will append
        // the generated tokens.
        let save_tokens = full_token_history.clone();

        self.chat_with_caches_inner(ChatDecodeInputs {
            last_logits,
            seq_len: total_seq_len,
            is_delta: true,
            has_images: false,
            token_history_init: full_token_history,
            save_tokens,
            save_expanded_tokens: None,
            save_image_cache_key: 0,
            tokenizer,
            think_end_id,
            think_end_str,
            thinking,
            eos_id,
            profiler,
            generation_start,
            first_token_instant,
            prefill_tokens_len: delta_tokens.len(),
            prompt_tokens_for_result,
            // Delta path reuses the full prior history by construction.
            cached_tokens_for_result: prior_cached_len as u32,
            embedding_weight,
            embedding_weight_t,
            generation_stream,
            params: p,
            // Delta path: the prefill runs on top of the live KV caches,
            // so there is no fresh full-prompt hidden to commit into the
            // MTP committed-history cache.
            prompt_hidden: None,
            prompt_hidden_ids: None,
            prompt_hidden_position_base: 0,
        })
    }

    /// Shared post-prefill pipeline: penalty → sample → decode loop (eager
    /// MTP or AR) → save cache state → finalize result.
    ///
    /// Extracted from `vision_mtp_whole_turn_core` so it can also be driven by the text-only
    /// session path (`chat_tokens_delta_sync`). Preserves the exact semantics
    /// of `vision_mtp_whole_turn_core` for the existing caller — `token_history_init` is the
    /// full pre-decode token sequence (used for penalty context and the decode
    /// loop's running history), and the decode loop mutates it in place.
    ///
    /// The caller is responsible for:
    /// - Creating a `WiredLimitContext` tied to `inputs.generation_stream` for
    ///   the lifetime of this call.
    /// - Running prefill and populating the resulting `last_logits` and
    ///   `seq_len` fields of `ChatDecodeInputs`.
    /// - Pre-starting the profiler (`set_prompt_tokens`, `snapshot_memory_before`,
    ///   `begin_prefill`, `end_prefill`).
    fn chat_with_caches_inner(&mut self, inputs: ChatDecodeInputs) -> Result<ChatResult> {
        let ChatDecodeInputs {
            last_logits,
            seq_len,
            is_delta,
            has_images,
            token_history_init,
            save_tokens,
            save_expanded_tokens,
            save_image_cache_key,
            tokenizer,
            think_end_id,
            think_end_str,
            thinking,
            eos_id,
            mut profiler,
            generation_start,
            mut first_token_instant,
            prefill_tokens_len,
            prompt_tokens_for_result,
            cached_tokens_for_result,
            embedding_weight,
            embedding_weight_t,
            generation_stream,
            params: p,
            prompt_hidden,
            prompt_hidden_ids,
            prompt_hidden_position_base,
        } = inputs;

        // Pure-Rust ("eager") dense MTP. Gated on the same per-request /
        // per-checkpoint preconditions (`enable_mtp`, MTP weights present),
        // restricted to the dense FLAT path (no live paged adapter, text-only
        // — the paged adapter has its own MTP gate and VLM routes through the
        // text decode path).
        let eager_mtp =
            p.enable_mtp && self.has_mtp_weights() && self.paged_adapter.is_none() && !has_images;

        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");
        let max_new_tokens = p.max_new_tokens;

        // Decode-entry trace. Snapshot all the inputs that decide the
        // AR-vs-MTP branch so MLX_NODE_LOG=info captures everything
        // needed to reconstruct a turn's control flow.
        {
            let prefill_len = seq_len as i32;
            let max_kv_len_estimate =
                engine::kv_capacity_round_up_saturating(prefill_len, max_new_tokens);
            let has_mtp = self.has_mtp_weights();
            let branch = if eager_mtp {
                "MTP (eager)"
            } else if !p.enable_mtp {
                "AR (enable_mtp=false)"
            } else if !has_mtp {
                "AR (no MTP weights on model)"
            } else {
                "AR"
            };
            info!(
                "Qwen3.5 chat_decode entry: prompt_len={} max_new_tokens={} enable_mtp={} \
                 mtp_depth={} prefill_seq_len={} max_kv_len={} has_mtp_weights={} \
                 is_delta={} has_images={} branch=\"{}\"",
                token_history_init.len(),
                max_new_tokens,
                p.enable_mtp,
                p.mtp_depth,
                prefill_len,
                max_kv_len_estimate,
                has_mtp,
                is_delta,
                has_images,
                branch,
            );
        }

        let last_logits = apply_all_penalties(last_logits, &token_history_init, &p)?;
        let mut y = sample(&last_logits, p.sampling_config)?;
        MxArray::async_eval_arrays(&[&y]);

        let mut token_history: Vec<u32> = token_history_init;

        let mut reasoning_tracker = engine::ReasoningTracker::from_setup(&thinking, think_end_id);

        // Whether the final committed token reached the physical KV/GDN cache.
        // The decode macros write `false` when they stop on an unforwarded
        // token so the save below can drop it from `cached_token_history`.
        let mut last_in_cache = true;

        if eager_mtp {
            // Pure-Rust eager dense MTP — the propose/verify whole-turn loop is
            // engine-owned (`engine::run_mtp_turn`) and drives the
            // `DenseMtpStepper` (`MtpBackend::begin_mtp_decode`). The stepper
            // captures the embedding table + config and runs the prompt-prefix
            // seed before the loop; the 11 former `MtpOps` closures are its
            // `MtpStepper` methods. The `profiler.set_label("mtp_eager")` relabel
            // moved into `DenseMtpStepper::profiler_relabel` (applied once at
            // turn entry by the engine).
            let mut rng = rand::rng();

            // Preserve the eager block's initial `async_eval_arrays(&[&y])`
            // (scheduling hint for the first sampled token) right before the
            // engine takes over.
            MxArray::async_eval_arrays(&[&y]);

            let outcome = crate::engine::mtp_turn::run_mtp_turn(
                self,
                &mut rng,
                crate::engine::mtp_turn::MtpTurnArgs {
                    // Cheap refcounted clone: `run_mtp_turn` consumes `y`, but the
                    // post-block `let _final_sampled_token = y;` discard (shared
                    // with the AR `decode_loop!` arm, which reassigns `y`) still
                    // reads it. The clone is the same lazy handle — byte-identical.
                    y: y.clone(),
                    depth: p.mtp_depth,
                    params: &p,
                    reasoning_tracker: &mut reasoning_tracker,
                    profiler: &mut profiler,
                    max_new_tokens,
                    eos_id,
                    generated_tokens: &mut generated_tokens,
                    token_history: &mut token_history,
                    finish_reason: &mut finish_reason,
                    first_token_instant: &mut first_token_instant,
                    report_perf: p.report_performance,
                    generation_stream,
                    prompt_hidden,
                    prompt_hidden_ids,
                    prompt_hidden_position_base,
                },
                // SYNC site: no streaming sink (the streaming flat site wires
                // its own `StreamingCtx` and shares this one loop).
                None,
            )?;

            last_in_cache = outcome.last_in_cache;
            self.flat_mtp_last_rollback_unemitted = outcome.rollback_unemitted;
            // Propagate a mid-cycle stop: self.caches advanced past the emitted
            // history, so force a full re-prefill next turn.
            if outcome.desynced {
                self.flat_mtp_caches_desynced = true;
            }
        } else {
            profiler.set_label("chat_rust");

            MxArray::async_eval_arrays(&[&y]);

            let mut ops = mtp_decode::DecodeOps {
                forward: |ids: &MxArray, emb: &MxArray| -> Result<(MxArray, bool)> {
                    let logits = forward_inner(
                        ids,
                        emb,
                        &mut self.layers,
                        &mut self.caches,
                        &self.final_norm,
                        &self.lm_head,
                        Some(&embedding_weight_t),
                    )?;
                    Ok((logits, true))
                },
                eval_step: |token: &MxArray, logits: &MxArray, _budget_forced: bool| {
                    MxArray::async_eval_arrays(&[token, logits]);
                },
            };
            mtp_decode::decode_loop!(
                ops: ops,
                y: y,
                embedding_weight: embedding_weight,
                params: p,
                reasoning_tracker: reasoning_tracker,
                profiler: profiler,
                max_new_tokens: max_new_tokens,
                eos_id: eos_id,
                generated_tokens: generated_tokens,
                token_history: token_history,
                finish_reason: finish_reason,
                last_in_cache: last_in_cache,
                first_token_instant: first_token_instant,
                report_perf: p.report_performance,
                generation_stream: generation_stream
            );
        }

        // Save cache state. Delta continuations preserve
        // `cached_image_key` — the KV cache still holds the prior
        // prefill's image attention state even though this turn was
        // text-only. Prefill paths (re)set the key based on the fresh
        // turn's `has_images`.
        if is_delta {
            engine::save_cache_state_after_delta(
                p.reuse_cache,
                &generated_tokens,
                &finish_reason,
                /* drop_last_always */ !last_in_cache,
                &save_tokens,
                &mut self.cached_token_history,
                &mut self.cached_image_key,
                &mut self.cached_rope_deltas,
                &mut self.caches,
            );
        } else {
            save_cache_state_direct(
                p.reuse_cache,
                has_images,
                &generated_tokens,
                &finish_reason,
                /* drop_last_always */ !last_in_cache,
                &save_tokens,
                save_expanded_tokens.as_deref(),
                save_image_cache_key,
                &mut self.cached_token_history,
                &mut self.cached_image_key,
                &mut self.cached_rope_deltas,
                &mut self.caches,
            );
        }
        // This turn committed flat caches, not the paged adapter. Never carry
        // per-block image identities into a later unrelated paged lifecycle.
        self.cached_paged_image_token_positions.clear();

        let performance = compute_performance_metrics(
            generation_start,
            first_token_instant,
            prefill_tokens_len,
            generated_tokens.len(),
        )
        .map(|mut m| {
            profiler.fill_mtp_acceptance(&mut m);
            m
        });

        // `y` is the last sampled token from the decode loop. The
        // `decode_loop!` macro assigns to `y` each iteration and the final
        // assignment in the last iteration is never observed, which without
        // this explicit discard trips `clippy::unused_assignments` (the
        // macro repetition hides the usage pattern from the lint). Binding
        // here is cleaner than spraying `#[allow]` inside the macro body.
        let _final_sampled_token = y;

        let mut result = finalize_chat_result(
            &tokenizer,
            &generated_tokens,
            finish_reason,
            think_end_id,
            think_end_str.as_deref(),
            performance,
            p.include_reasoning,
            thinking.enabled,
            prompt_tokens_for_result,
            reasoning_tracker.reasoning_token_count(),
        )?;
        // Report the length of the reused cached prefix for observability.
        // Driven by the caller-supplied `cached_tokens_for_result`:
        //   - fresh prefill: equals `cached_prefix_len` (0 miss / full hit)
        //   - session delta: equals the prior-history length (full reuse)
        // See the invariant doc on `verify_cache_prefix_direct`.
        result.cached_tokens = cached_tokens_for_result;
        Ok(result)
    }

    /// Shared pure-Rust eager-MTP decode loop for the dense FLAT STREAMING
    /// cores (`chat_stream_sync_inner` / `chat_stream_tokens_delta_sync_inner`).
    ///
    /// This is the streaming analogue of the `eager_mtp` arm of
    /// [`Self::chat_with_caches_inner`]: the propose/verify whole-turn loop is
    /// engine-owned ([`crate::engine::mtp_turn::run_mtp_turn`]) and drives the
    /// [`DenseMtpStepper`] ([`MtpBackend::begin_mtp_decode`]) — the SAME stepper
    /// and prompt-prefix committed-history seed the SYNC site uses. The only
    /// difference is the streaming sink: this site wires a
    /// [`crate::engine::decode::StreamingCtx`] (incremental detokenization plus
    /// the default [`crate::engine::backend::DefaultStreamEmitter`]) so accepted
    /// tokens stream out the `cb` sink incrementally, sharing ONE loop with the
    /// sync site. Caller owns prefill, sampling of the first `y`, the
    /// `WiredLimitContext`, and the post-loop save-cache / final-chunk tail.
    ///
    /// Preconditions (enforced by the callers' gate): `enable_mtp`,
    /// `has_mtp_weights()`, `paged_adapter.is_none()`, text-only. The body is
    /// byte-identical to the non-streaming eager MTP decode (same accept/rewind
    /// math, same GDN tape replay) — only the streamed deltas differ.
    #[allow(clippy::too_many_arguments)]
    fn run_flat_stream_eager_mtp<'a>(
        &mut self,
        y: MxArray,
        token_history: &mut Vec<u32>,
        generated_tokens: &mut Vec<u32>,
        finish_reason: &mut String,
        reasoning_tracker: &mut engine::ReasoningTracker,
        profiler: &mut crate::decode_profiler::DecodeProfiler,
        first_token_instant: &mut Option<std::time::Instant>,
        streamed_text_len: &mut usize,
        last_is_reasoning: &mut bool,
        decode_stream: &mut tokenizers::DecodeStream<
            'a,
            tokenizers::ModelWrapper,
            tokenizers::NormalizerWrapper,
            tokenizers::PreTokenizerWrapper,
            tokenizers::PostProcessorWrapper,
            tokenizers::DecoderWrapper,
        >,
        tokenizer: &'a Arc<Qwen3Tokenizer>,
        cb: &StreamSender<'_>,
        cancelled: &AtomicBool,
        // The embedding table + its transpose are owned by the model and
        // pulled inside `begin_mtp_decode` (`self.embedding.get_weight()`),
        // so the engine-owned loop does not read these; kept in the signature
        // for call-site parity with the AR streaming arm.
        _embedding_weight: MxArray,
        _embedding_weight_t: MxArray,
        p: &engine::ChatParams,
        eos_id: u32,
        max_new_tokens: i32,
        generation_stream: Stream,
        prompt_hidden: Option<MxArray>,
        prompt_hidden_ids: Option<Vec<u32>>,
        prompt_hidden_position_base: usize,
        last_in_cache: &mut bool,
    ) -> Result<()> {
        // The turn profiler relabel ("mtp_eager") now moves into
        // `DenseMtpStepper::profiler_relabel`, applied once at turn entry by
        // the engine — mirroring the SYNC site.
        MxArray::async_eval_arrays(&[&y]);

        let mut rng = rand::rng();

        // Wire the streaming sink: incremental detokenization through the
        // shared `step_decode_stream` + the default ChatML emitter (qwen3_5
        // does not override `stream_emitter`, so the macro's inline emit is
        // byte-identical to `DefaultStreamEmitter::on_token_text`). The
        // engine's `run_mtp_turn` routes the SAME three emit sites + the
        // pre-loop cancel break through this `StreamingCtx`.
        let mut emitter = crate::engine::backend::DefaultStreamEmitter;
        let streaming = crate::engine::decode::StreamingCtx {
            callback: cb.0,
            cancelled,
            decode_stream,
            tokenizer: tokenizer.inner(),
            streamed_text_len,
            last_is_reasoning,
            emitter: &mut emitter,
        };

        let outcome = crate::engine::mtp_turn::run_mtp_turn(
            self,
            &mut rng,
            crate::engine::mtp_turn::MtpTurnArgs {
                y,
                depth: p.mtp_depth,
                params: p,
                reasoning_tracker,
                profiler,
                max_new_tokens,
                eos_id,
                generated_tokens,
                token_history,
                finish_reason,
                first_token_instant,
                report_perf: p.report_performance,
                generation_stream,
                prompt_hidden,
                prompt_hidden_ids,
                prompt_hidden_position_base,
            },
            Some(streaming),
        )?;

        *last_in_cache = outcome.last_in_cache;
        self.flat_mtp_last_rollback_unemitted = outcome.rollback_unemitted;
        // Propagate a mid-cycle stop: self.caches advanced past the emitted
        // history, so force a full re-prefill next turn.
        if outcome.desynced {
            self.flat_mtp_caches_desynced = true;
        }

        Ok(())
    }

    /// Block-paged variant of [`Self::vision_mtp_whole_turn_core`].
    ///
    /// Mirrors the flat path's control flow (penalty stack, decode
    /// loop, EOS / repetition cutoff, performance timing, output
    /// post-processing) but routes full-attention layers through
    /// `forward_paged_or_flat` against the paged KV adapter. GDN
    /// (linear-attention) layers continue to use their existing
    /// `Qwen3_5LayerCache::Linear(ArraysCache)` storage and are
    /// reset+re-prefilled every turn (no cross-request prefix reuse —
    /// vLLM's `MambaManager` stance).
    ///
    /// Per-turn lifecycle:
    /// 1. Adapter lifecycle: warm-continue when the prior turn ended
    ///    via `finalize_turn_keep_live`; cold-start (reset →
    ///    find_cached_prefix → allocate_suffix) otherwise.
    /// 2. Prepare GDN prefix state from live/session checkpoints when
    ///    available; otherwise replay the cached prefix through GDN.
    /// 3. Prefill via `paged_forward::run_paged_prefill_chunk`.
    /// 4. Decode loop via `paged_forward::run_paged_decode_step`.
    /// 5. End-of-turn: `finalize_turn_keep_live` keeps the partial
    ///    trailing block live for the next turn's warm
    ///    `continue_turn` (mirrors LFM2 / Qwen3).
    ///
    /// Limitations:
    /// * VLM is rejected upstream — paged dispatch is text-only.
    /// * Cross-turn GDN prefix reuse is limited to live/history/prefix
    ///   checkpoints whose identity matches the paged KV prefix. Misses
    ///   fall back to GDN replay from token 0.
    /// * Pure-cache prompt (every prompt token already in the paged
    ///   pool) is rejected — same caveat as LFM2 / Qwen3 paged paths.
    /// * Paged turns run the pure-Rust
    ///   `DecoderLayer::forward_paged_or_flat`.
    fn paged_turn_sync_core(
        &mut self,
        tokens: Vec<u32>,
        tokenizer: Arc<Qwen3Tokenizer>,
        eos_token_id: u32,
        mut p: engine::ChatParams,
        report_perf: bool,
        thinking: ThinkingSetup,
    ) -> Result<ChatResult> {
        if tokens.is_empty() {
            return Err(Error::from_reason("Empty prompt"));
        }
        self.preflight_paged_context(tokens.len(), &mut p)?;

        // This paged turn writes full-attention K/V into the paged adapter
        // pool, NOT the flat `self.caches`, so the flat full-attention slots no
        // longer reflect the conversation. A later streaming dense-MTP fallback
        // must rebuild the flat caches before decoding. See
        // `paged_full_attn_caches_dirty`.
        self.paged_full_attn_caches_dirty = true;

        let prompt_token_count = tokens.len() as u32;
        let sampling_config = p.sampling_config;

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());
        // Thinking is resolved ONCE at turn entry and honors
        // `enable_thinking=false`.
        let thinking_enabled = thinking.enabled;
        let mut reasoning_tracker = engine::ReasoningTracker::from_setup(&thinking, think_end_id);

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;
        let trace_enabled = inference_trace_enabled();

        // === Adapter lifecycle: warm continuation OR cold start ===
        let seq_id: u32 = 0;
        // Lazy decode allocation: pass the prompt length only.
        let total_budget = tokens.len() as u32;
        // Per-block extra_keys for prefix-cache lookup. Text-only dispatch
        // (image-bearing turns route to the flat path) yields all-empty
        // per-block vecs; the resulting hashes are bit-equal to passing `&[]`
        // to the uniform API. VLM-paged forward integration would swap in real
        // image-position pairs here to enable image-aware cache isolation.
        let block_size = {
            let adapter = self
                .paged_adapter
                .as_ref()
                .ok_or_else(|| Error::from_reason("paged_turn_sync_core: paged_adapter is None"))?;
            adapter.block_size()
        };
        let carries_image_lineage = self.cached_image_key.is_some()
            && !self.cached_paged_image_token_positions.is_empty()
            && !self.cached_token_history.is_empty()
            && tokens.starts_with(&self.cached_token_history);
        let image_positions = if carries_image_lineage {
            self.cached_paged_image_token_positions.as_slice()
        } else {
            &[]
        };
        let lookup_extra_keys =
            engine::build_paged_extra_keys(tokens.len(), block_size, image_positions);
        let cache_salt = 0;
        // vLLM exact-prefix cap — see qwen3/model.rs:paged_turn_sync_core.
        // Ensures every paged turn has at least one suffix token to prefill,
        // even when the live cache (or a prior request's residue) already
        // covers the entire new prompt.
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
        let live_ready;
        let live_prefix_match;
        let live_tokens_len;
        let mut live_mismatch = TokenPrefixMismatchTrace::default();
        // Adapter-owned warm/cold lifecycle. The [MLX_TRACE] line below
        // reads the PRE-turn live state, so probe the adapter immutably FIRST
        // (prepare_turn mutates request_tokens via continue_turn/reset). The
        // adapter re-reads the same state internally, so live_* matches what
        // prepare_turn decides. extra_keys=&[] (uniform API) is bit-equal to
        // `&lookup_extra_keys` for text-only dispatch (all-empty per-block vec
        // → identical hashes; see the block_size comment above).
        // Suffix blocks are allocated inside prepare_turn.
        {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason(
                    "paged_turn_sync_core: paged_adapter is None — caller must check \
                     use_block_paged_cache before dispatch",
                )
            })?;
            live_ready = adapter.is_live_for_continue();
            let live_tokens = adapter.request_tokens();
            live_tokens_len = live_tokens.len();
            live_prefix_match = tokens.starts_with(live_tokens);
            if trace_enabled && live_ready && !live_prefix_match {
                live_mismatch = token_prefix_mismatch_trace(&tokens, live_tokens);
            }
        }
        let plan_result = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| {
                Error::from_reason(
                    "paged_turn_sync_core: paged_adapter is None — caller must check \
                     use_block_paged_cache before dispatch",
                )
            })?
            .prepare_turn_per_block_with_max_cache_hit_tokens(
                seq_id,
                &tokens,
                total_budget,
                true,
                &lookup_extra_keys,
                cache_salt,
                false,
                max_cache_hit_tokens,
            )
            .map_err(Error::from_reason);
        let plan = match plan_result {
            Ok(plan) => plan,
            Err(error) => {
                self.invalidate_dense_paged_session("planned-MTP adapter preparation failure");
                return Err(error);
            }
        };
        let cached_prefix_len = plan.cached_prefix_len;
        let continued_live_prefix = plan.continued_live_prefix;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-dense paged_prefix_lookup prompt_tokens={} \
                 cached_prefix_tokens={} continued_live_prefix={} live_ready={} \
                 live_match={} live_tokens={} live_mismatch_at={} prompt_token={} live_token={}",
                tokens.len(),
                cached_prefix_len,
                continued_live_prefix,
                live_ready,
                live_prefix_match,
                live_tokens_len,
                live_mismatch.index,
                live_mismatch.prompt_token,
                live_mismatch.cached_token
            ));
        }

        let gdn_prefix_preparation = match self.prepare_dense_gdn_prefix_state(
            &tokens,
            cached_prefix_len,
            block_size,
            &lookup_extra_keys,
            cache_salt,
            continued_live_prefix,
        ) {
            Ok(preparation) => preparation,
            Err(error) => {
                self.invalidate_dense_paged_session("planned-MTP GDN-prefix preparation failure");
                return Err(error);
            }
        };
        let gdn_prefix_already_primed = gdn_prefix_preparation.already_primed;
        // Discharge the adapter's auxiliary (GDN) prefix obligation before the
        // turn's first `record_tokens`. See `prime_prefix_state` for the full
        // rationale; a no-op unless a HOT K/V hit handed back a prefix the cold
        // tier's `ColdSidecarPolicy` was not gated against.
        let confirm_result = self
            .paged_adapter
            .as_mut()
            .map(|adapter| adapter.confirm_aux_prefix_primed(cached_prefix_len))
            .unwrap_or(Ok(()));
        if let Err(error) = confirm_result {
            self.invalidate_dense_paged_session("planned-MTP GDN auxiliary-prefix confirmation");
            return Err(Error::from_reason(error));
        }
        let preserves_image_lineage = carries_image_lineage && cached_prefix_len > 0;
        self.cached_token_history.clear();
        if !preserves_image_lineage {
            self.cached_image_key = None;
            self.cached_paged_image_token_positions.clear();
        }
        // Carry the cross-turn M-RoPE delta only when this turn extends the live
        // image sequence (continued_live_prefix). A cold start OR a non-live
        // prefix-cache hit (cached_prefix_len > 0 but not a live continuation)
        // can only restore pure-text prefix blocks, so a stale image delta is
        // dropped and the text suffix rotates at the raw physical slot.
        self.cached_rope_deltas = super::paged_forward::rope_delta_for_paged_turn(
            self.cached_rope_deltas,
            preserves_image_lineage,
        );

        let suffix_len = match prompt_token_count.checked_sub(cached_prefix_len) {
            Some(suffix_len) => suffix_len,
            None => {
                self.invalidate_dense_paged_session("planned-MTP prefix length mismatch");
                return Err(Error::from_reason(
                    "paged_turn_sync_core: cached_prefix_len > total_prompt_tokens",
                ));
            }
        };

        let forward_result = self.paged_turn_sync_core_inner(
            &tokens,
            cached_prefix_len,
            suffix_len,
            &p,
            eos_token_id,
            &sampling_config,
            &mut reasoning_tracker,
            report_perf,
            &mut first_token_instant,
            gdn_prefix_already_primed,
        );

        let (generated_tokens, finish_reason, mtp_profiler) = match forward_result {
            Ok(t) => {
                let image_token_positions = self.cached_paged_image_token_positions.clone();
                self.finalize_dense_manual_paged_turn(&image_token_positions)?;
                t
            }
            Err(e) => {
                self.invalidate_dense_paged_session("planned-MTP sync prefill/decode failure");
                return Err(e);
            }
        };

        // Persist the full token history so subsequent
        // `chat_session_continue` /
        // `chat_tokens_delta_sync` calls find an initialized session
        // to extend. The paged decode loop never feeds the LAST
        // sampled token through the model, so drop it from the
        // saved history (mirrors LFM2 / Qwen3 paged path).
        let last_token_in_cache = false;
        let mut full_history = tokens.clone();
        if !generated_tokens.is_empty() {
            let upto = if last_token_in_cache {
                generated_tokens.len()
            } else {
                generated_tokens.len().saturating_sub(1)
            };
            full_history.extend_from_slice(&generated_tokens[..upto]);
        }
        self.cached_token_history = full_history;
        let gdn_history_checkpoint_store = match self.remember_dense_gdn_history_checkpoint() {
            Ok(store) => store,
            Err(error) => {
                self.invalidate_dense_paged_session(
                    "planned-MTP sync GDN history checkpoint failure",
                );
                return Err(error);
            }
        };
        if inference_trace_enabled() {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-dense gdn_history_checkpoint stored={} tokens={} \
                 eval_ms={:.1} clone_ms={:.1} token_clone_ms={:.1} update_ms={:.1} total_ms={:.1}",
                gdn_history_checkpoint_store.stored,
                self.cached_token_history.len(),
                gdn_history_checkpoint_store.eval_ms,
                gdn_history_checkpoint_store.clone_ms,
                gdn_history_checkpoint_store.token_clone_ms,
                gdn_history_checkpoint_store.update_ms,
                gdn_history_checkpoint_store.total_ms
            ));
        }

        let performance = if report_perf {
            compute_performance_metrics(
                generation_start,
                first_token_instant,
                tokens.len() - cached_prefix_len as usize,
                generated_tokens.len(),
            )
            .map(|mut m| {
                if let Some(prof) = mtp_profiler.as_ref() {
                    prof.fill_mtp_acceptance(&mut m);
                }
                m
            })
        } else {
            None
        };

        let mut result = finalize_chat_result(
            &tokenizer,
            &generated_tokens,
            finish_reason,
            think_end_id,
            think_end_str.as_deref(),
            performance,
            p.include_reasoning,
            thinking_enabled,
            prompt_token_count,
            reasoning_tracker.reasoning_token_count(),
        )?;
        result.cached_tokens = cached_prefix_len;
        Ok(result)
    }

    /// Inner forward + decode loop for `paged_turn_sync_core`. Split
    /// out so the caller can wrap it with `release_request` on either
    /// path.
    ///
    /// The pure-Rust paged prefill populates the GDN linear caches and
    /// writes K/V into the adapter pool; decode steps then run through
    /// `paged_forward::run_paged_decode_step`.
    #[allow(clippy::too_many_arguments)]
    fn paged_turn_sync_core_inner(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        suffix_len: u32,
        p: &engine::ChatParams,
        eos_token_id: u32,
        sampling_config: &Option<crate::sampling::SamplingConfig>,
        reasoning_tracker: &mut engine::ReasoningTracker,
        report_perf: bool,
        first_token_instant: &mut Option<std::time::Instant>,
        gdn_prefix_already_primed: bool,
    ) -> Result<(
        Vec<u32>,
        String,
        Option<crate::decode_profiler::DecodeProfiler>,
    )> {
        // Invariant: caller-applied vLLM cap guarantees suffix_len > 0.
        debug_assert!(
            suffix_len > 0,
            "paged_turn_sync_core_inner: caller must cap max_cache_hit_tokens at prompt.len() - 1"
        );

        let suffix = &tokens[(cached_prefix_len as usize)..];
        let layer_kinds =
            super::decoder_layer::compute_layer_kinds(self.config.num_layers as usize, |i| {
                self.config.is_linear_layer(i)
            });

        // Paged prompt-prefix MTP prefill. Mirrors the dense gate's
        // `want_prompt_hidden` predicate. Capturing the post-`final_norm`
        // hidden for every prompt token lets `begin_mtp_decode`'s
        // prompt-prefix seed commit the full prompt (advancing the stepper's
        // `committed_len` to N) before cycle 1, so
        // MTP drafts attend over the prompt (matches the dense MTP path). The
        // `cached_prefix_len == 0` clause matches dense: on a cache-reuse turn
        // the suffix-only prefill cannot produce the full prompt's hidden
        // tensor.
        let eager_mtp_paged = p.enable_mtp && self.has_mtp_weights();
        let want_prompt_hidden =
            eager_mtp_paged && !mtp_decode::mtp_no_prompt_prefill() && cached_prefix_len == 0;
        let mtp_prompt_history = mtp_decode::mtp_prompt_history_selection(tokens.len());
        let mut mtp_profiler = begin_paged_mtp_profiler(eager_mtp_paged, suffix_len);

        // === PREFILL ===
        let (last_logits, prompt_hidden, gdn_checkpoint) = self.run_dense_core_paged_prefill(
            tokens,
            suffix,
            cached_prefix_len,
            gdn_prefix_already_primed,
            want_prompt_hidden.then_some(mtp_prompt_history.keep_tokens),
            &layer_kinds,
            "paged_turn_sync_core_inner",
        )?;
        if let Some(profiler) = mtp_profiler.as_mut() {
            profiler.end_prefill();
        }
        self.publish_dense_gdn_materialized_prefix_checkpoint(tokens, gdn_checkpoint);

        // First-token sample.
        let mut token_history: Vec<u32> = tokens.to_vec();
        let last_logits = apply_all_penalties(last_logits, &token_history, p)?;
        let mut y = sample(&last_logits, *sampling_config)?;
        y.eval();

        // Smooth memory peak: drop transient prefill buffers before decode
        // starts allocating. Prefill builds a massive MLX subgraph; once
        // we have the last logits, those intermediates are dead but
        // MLX's caching allocator holds them.
        crate::array::synchronize_and_clear_cache();

        if let Some(profiler) = mtp_profiler.as_mut() {
            profiler.mark_first_token();
        }
        if report_perf {
            *first_token_instant = Some(std::time::Instant::now());
        }

        // === DECODE LOOP ===
        let max_new_tokens = p.max_new_tokens;
        let mut generated_tokens: Vec<u32> =
            Vec::with_capacity(engine::generated_capacity_hint(max_new_tokens));
        let mut finish_reason = String::from("length");

        // Pure-Rust ("eager") paged MTP gate. The paged adapter IS present
        // here (this is the paged core), so — unlike the flat eager gate —
        // the gate does NOT require `paged_adapter.is_none()`.
        info!(
            "Qwen3.5 MTP gate (paged): enable_mtp={} has_mtp_weights={} -> eager_mtp_paged={}",
            p.enable_mtp,
            self.has_mtp_weights(),
            eager_mtp_paged
        );

        if eager_mtp_paged {
            // Pure-Rust ("eager") paged MTP.
            // The main Step-A / verify forwards route through the paged adapter
            // (`run_paged_step_with_hidden` / `run_paged_verify_step`); the GDN
            // recurrent state stays FLAT in `self.caches` Linear slots, so the
            // GDN tape replay (the rollback keystone) is IDENTICAL to the flat
            // eager arm. Full-attention K/V lives in the paged pool, so the
            // rollback rewinds it via `adapter.rollback_last_tokens(rejected)`,
            // NOT a `self.caches` KV trim.
            MxArray::async_eval_arrays(&[&y]);

            let mut profiler = mtp_profiler
                .take()
                .expect("paged MTP profiler must be created before prefill");

            let eos_id = eos_token_id;
            let generation_stream = crate::stream::Stream::new(crate::stream::DeviceType::Gpu);
            let model_size_bytes = self.config.estimate_memory_bytes() as usize;
            let _wired_ctx =
                crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

            // Prompt-tail ids + position base for the committed-history seed.
            // `prompt_hidden` is only `Some` when `want_prompt_hidden` held,
            // which already requires `cached_prefix_len == 0` (=> position_base
            // is 0 on those turns), so committed-history v2 is correct.
            let prompt_hidden_position_base = mtp_prompt_history.position_base;
            let prompt_hidden_ids: Vec<u32> = {
                let start = mtp_prompt_history
                    .hidden_start_token_index()
                    .min(tokens.len());
                tokens[start..].to_vec()
            };

            let mut rng = rand::rng();

            // The propose/verify whole-turn loop is engine-owned
            // (`run_mtp_turn`) and drives the `DenseMtpStepper` in its PAGED
            // mode: `begin_mtp_decode` moves `self.paged_adapter` into the
            // stepper for the turn, runs the committed-history prompt seed, and
            // routes the Step-A / verify forwards through the adapter; the
            // stepper's `Drop` restores the adapter into `self.paged_adapter`
            // before this call returns, so the paged-history save below finds
            // it. The paged path commits cache state through its own
            // paged-history save, so `outcome.last_in_cache` is unused here.
            let outcome = crate::engine::mtp_turn::run_mtp_turn(
                self,
                &mut rng,
                crate::engine::mtp_turn::MtpTurnArgs {
                    y: y.clone(),
                    depth: p.mtp_depth,
                    params: p,
                    reasoning_tracker,
                    profiler: &mut profiler,
                    max_new_tokens,
                    eos_id,
                    generated_tokens: &mut generated_tokens,
                    token_history: &mut token_history,
                    finish_reason: &mut finish_reason,
                    first_token_instant,
                    report_perf: p.report_performance,
                    generation_stream,
                    prompt_hidden,
                    prompt_hidden_ids: Some(prompt_hidden_ids),
                    prompt_hidden_position_base,
                },
                None,
            )?;
            let _ = outcome.last_in_cache;

            // `self.caches` already holds the live GDN state (the eager paged
            // forwards wrote it directly) — nothing to export.
            return Ok((generated_tokens, finish_reason, Some(profiler)));
        }

        for step in 0..max_new_tokens {
            let token_id = y.item_at_int32(0)? as u32;
            generated_tokens.push(token_id);
            token_history.push(token_id);
            reasoning_tracker.observe_token(token_id);

            if token_id == eos_token_id || p.extra_eos_ids.contains(&token_id) {
                finish_reason = String::from("stop");
                break;
            }
            if let Some(reason) = crate::sampling::check_repetition_cutoff(
                &generated_tokens,
                p.max_consecutive_tokens,
                p.max_ngram_repeats,
                p.ngram_size,
            ) {
                finish_reason = reason.to_string();
                break;
            }
            if step + 1 >= max_new_tokens {
                break;
            }

            // Decode forward (pure-Rust paged step).
            let next_logits = {
                let embed = self.embedding.clone();
                let embedding_weight = embed.get_weight();
                let caches_ref = self.caches.as_mut().ok_or_else(|| {
                    Error::from_reason("paged_turn_sync_core_inner: caches dropped mid-decode")
                })?;
                let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                    Error::from_reason(
                        "paged_turn_sync_core_inner: paged_adapter dropped mid-decode",
                    )
                })?;
                let logits = super::paged_forward::run_paged_decode_step(
                    token_id,
                    &embed,
                    &mut self.layers,
                    caches_ref,
                    &self.final_norm,
                    &self.lm_head,
                    &embedding_weight,
                    &layer_kinds,
                    adapter,
                    self.cached_rope_deltas.unwrap_or(0),
                )?;
                logits.squeeze(Some(&[1]))?
            };

            let next_logits = if reasoning_tracker.should_force_think_end() {
                let forced_id = reasoning_tracker.forced_token_id()? as i32;
                y = MxArray::from_int32(&[forced_id], &[1])?;
                y.eval();
                continue;
            } else {
                apply_all_penalties(next_logits, &token_history, p)?
            };

            y = sample(&next_logits, *sampling_config)?;
            y.eval();

            crate::array::maybe_clear_cache_for_paged_step(step);
        }

        Ok((generated_tokens, finish_reason, None))
    }

    /// Single-turn image-bearing block-paged dispatch (non-streaming).
    ///
    /// The paged sibling of the flat VLM prefill: it processes the images,
    /// merges the vision features into the token embeddings, computes M-RoPE
    /// positions, then prefills through the paged adapter via
    /// [`super::paged_forward::run_paged_vlm_prefill`] and runs the plain
    /// autoregressive decode loop.
    ///
    /// Same-image live histories continue in place; fresh histories look up
    /// full blocks using per-image content keys and prefill only the uncached
    /// suffix. Decode rotates at the physical token count plus the cached
    /// M-RoPE delta (`cached_rope_deltas`). This core runs plain autoregressive
    /// decode with no draft/verify; MTP weights are ignored, so an MTP-enabled
    /// session's image turns route here and decode as AR.
    #[allow(clippy::too_many_arguments)]
    fn vision_paged_turn_sync_core(
        &mut self,
        tokens: Vec<u32>,
        images: &[Vec<u8>],
        tokenizer: Arc<Qwen3Tokenizer>,
        eos_token_id: u32,
        mut p: engine::ChatParams,
        report_perf: bool,
        thinking: ThinkingSetup,
    ) -> Result<ChatResult> {
        if tokens.is_empty() {
            return Err(Error::from_reason("Empty prompt"));
        }
        self.preflight_paged_context(tokens.len(), &mut p)?;

        let (vision_encoder, img_proc) =
            match (self.vision_encoder.clone(), self.image_processor.as_ref()) {
                (Some(enc), Some(proc)) => (enc, proc),
                _ => {
                    return Err(Error::from_reason(
                        "VLM prefill requested but vision encoder/processor not loaded",
                    ));
                }
            };

        // This paged turn writes full-attention K/V into the paged adapter pool,
        // leaving the flat `self.caches` full-attention slots stale. A later
        // dense-MTP fallback must rebuild them first.
        self.paged_full_attn_caches_dirty = true;

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());
        let thinking_enabled = thinking.enabled;
        let mut reasoning_tracker = engine::ReasoningTracker::from_setup(&thinking, think_end_id);

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;
        let sampling_config = p.sampling_config;

        // === VLM image processing: expand placeholders + merge features ===
        let sms = self.spatial_merge_size.unwrap_or(2);
        let image_refs: Vec<&[u8]> = images.iter().map(|v| v.as_slice()).collect();
        let processed = img_proc.process_many(&image_refs)?;
        let per_image_token_counts =
            compute_image_token_counts_per_image(&processed.grid_thw(), sms)?;
        let expanded_tokens = inject_image_placeholders(&tokens, &per_image_token_counts)?;
        self.preflight_paged_context(expanded_tokens.len(), &mut p)?;
        let (image_cache_key, per_image_hashes) = engine::compute_image_cache_keys(images);
        let image_token_positions = engine::map_expanded_image_token_positions(
            &expanded_tokens,
            IMAGE_TOKEN_ID as u32,
            &per_image_token_counts,
            &per_image_hashes,
        )
        .map_err(Error::from_reason)?;
        let prompt_token_count = expanded_tokens.len() as u32;

        let embed = self.embedding.clone();
        let embedding_weight = embed.get_weight();
        let input_ids = MxArray::from_uint32(&expanded_tokens, &[1, expanded_tokens.len() as i64])?;

        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        let merge = vlm_prepare_vision_features(
            &input_ids,
            &per_image_hashes,
            &processed,
            &vision_encoder,
            sms,
            &embedding_weight,
            generation_stream,
            &self.vision_cache,
        )?;
        // `merge` only retains evaluated per-image features and newly-built
        // text/position arrays. Release the aggregate f32 pixel tensor before
        // language prefill instead of keeping every historical image alive for
        // the rest of the turn.
        drop(processed);
        crate::array::clear_cache();

        // === Image-aware paged-prefix lifecycle ===
        let total_budget = expanded_tokens.len() as u32;
        let block_size = self
            .paged_adapter
            .as_ref()
            .ok_or_else(|| {
                Error::from_reason("vision_paged_turn_sync_core: paged_adapter is None")
            })?
            .block_size();
        let lookup_extra_keys = engine::build_paged_extra_keys(
            expanded_tokens.len(),
            block_size,
            &image_token_positions,
        );
        let same_live_image = self.cached_image_key == Some(image_cache_key);
        let prefix_resolution = self.prepare_dense_vlm_paged_prefix(
            &expanded_tokens,
            total_budget,
            block_size,
            &lookup_extra_keys,
            p.reuse_cache,
            p.reuse_cache && same_live_image,
            image_cache_key,
        )?;
        let plan = prefix_resolution.effective_plan;
        let cached_prefix_len = plan.cached_prefix_len;
        tracing::info!(
            target: "mlx_core::inference",
            event = "vlm_prefix_plan",
            model = "qwen3_5_dense",
            prompt_tokens = expanded_tokens.len(),
            image_count = images.len(),
            image_tokens = image_token_positions.len(),
            candidate_cached_prefix_tokens = prefix_resolution.candidate_cached_prefix_len,
            effective_cached_prefix_tokens = cached_prefix_len,
            cached_prefix_tokens = cached_prefix_len,
            suffix_tokens = expanded_tokens.len() - cached_prefix_len as usize,
            continued_live_prefix = plan.continued_live_prefix,
            same_live_image,
            gdn_prefix_already_primed = prefix_resolution.gdn_prefix_already_primed,
            downgraded_to_cold = prefix_resolution.downgraded_to_cold,
            "image-aware paged prefix planned"
        );
        let gdn_prefix_already_primed = prefix_resolution.gdn_prefix_already_primed;
        self.cached_token_history.clear();
        self.cached_image_key = None;
        self.cached_paged_image_token_positions.clear();
        // Store the image prefill's compressed-position delta so a later text
        // warm-continuation rotates its queries at the same compressed M-RoPE
        // positions the image keys were written with.
        self.cached_rope_deltas = Some(merge.rope_deltas as i32);

        let layer_kinds =
            super::decoder_layer::compute_layer_kinds(self.config.num_layers as usize, |i| {
                self.config.is_linear_layer(i)
            });

        let forward_result = (|| -> Result<(Vec<u32>, String)> {
            // === PREFILL ===
            let last_logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                let caches_ref = self.caches.as_mut().ok_or_else(|| {
                    Error::from_reason("vision_paged_turn_sync_core: caches not initialized")
                })?;
                let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                    Error::from_reason("vision_paged_turn_sync_core: paged_adapter dropped")
                })?;
                super::paged_forward::run_paged_vlm_prefill(
                    &expanded_tokens,
                    &merge,
                    cached_prefix_len,
                    gdn_prefix_already_primed,
                    &embed,
                    &mut self.layers,
                    caches_ref,
                    &self.final_norm,
                    &self.lm_head,
                    &embedding_weight,
                    &layer_kinds,
                    adapter,
                )?
            };
            let (last_logits, gdn_checkpoint) = last_logits;
            self.publish_dense_gdn_materialized_prefix_checkpoint_with_keys(
                &expanded_tokens,
                &lookup_extra_keys,
                0,
                gdn_checkpoint,
            );

            let mut token_history: Vec<u32> = expanded_tokens.clone();
            let last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
            let mut y = sample(&last_logits, sampling_config)?;
            y.eval();

            crate::array::synchronize_and_clear_cache();
            if report_perf {
                first_token_instant = Some(std::time::Instant::now());
            }

            // === DECODE LOOP (autoregressive, scalar-offset RoPE) ===
            let max_new_tokens = p.max_new_tokens;
            let mut generated_tokens: Vec<u32> =
                Vec::with_capacity(engine::generated_capacity_hint(max_new_tokens));
            let mut finish_reason = String::from("length");

            for step in 0..max_new_tokens {
                let token_id = y.item_at_int32(0)? as u32;
                generated_tokens.push(token_id);
                token_history.push(token_id);
                reasoning_tracker.observe_token(token_id);

                if token_id == eos_token_id || p.extra_eos_ids.contains(&token_id) {
                    finish_reason = String::from("stop");
                    break;
                }
                if let Some(reason) = crate::sampling::check_repetition_cutoff(
                    &generated_tokens,
                    p.max_consecutive_tokens,
                    p.max_ngram_repeats,
                    p.ngram_size,
                ) {
                    finish_reason = reason.to_string();
                    break;
                }
                if step + 1 >= max_new_tokens {
                    break;
                }

                let next_logits = {
                    let _stream_ctx = StreamContext::new(generation_stream);
                    let caches_ref = self.caches.as_mut().ok_or_else(|| {
                        Error::from_reason("vision_paged_turn_sync_core: caches dropped mid-decode")
                    })?;
                    let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "vision_paged_turn_sync_core: paged_adapter dropped mid-decode",
                        )
                    })?;
                    let logits = super::paged_forward::run_paged_decode_step(
                        token_id,
                        &embed,
                        &mut self.layers,
                        caches_ref,
                        &self.final_norm,
                        &self.lm_head,
                        &embedding_weight,
                        &layer_kinds,
                        adapter,
                        self.cached_rope_deltas.unwrap_or(0),
                    )?;
                    logits.squeeze(Some(&[1]))?
                };

                if reasoning_tracker.should_force_think_end() {
                    let forced_id = reasoning_tracker.forced_token_id()? as i32;
                    y = MxArray::from_int32(&[forced_id], &[1])?;
                    y.eval();
                    continue;
                }
                let next_logits = apply_all_penalties(next_logits, &token_history, &p)?;

                y = sample(&next_logits, sampling_config)?;
                y.eval();

                crate::array::maybe_clear_cache_for_paged_step(step);
            }

            Ok((generated_tokens, finish_reason))
        })();

        // Terminal lifecycle, mirroring the text paged core
        // (`paged_turn_sync_core` / `finalize_paged_turn`). The error path
        // always releases the request and returns. The success path is
        // resolved below so the session ends in exactly one of two states,
        // never a partial one: FULLY continuable (keep-live registered AND GDN
        // checkpoint stored AND history + image key published) or
        // NON-continuable (request released AND history cleared AND image key
        // None) so a follow-up text continue is safely rejected instead of
        // cold-prefilling image-placeholder ids as ordinary tokens.
        let (generated_tokens, finish_reason) = match forward_result {
            Ok(t) => t,
            Err(e) => {
                self.invalidate_dense_paged_session("VLM sync prefill/decode failure");
                return Err(e);
            }
        };

        // Build the saved history: the EXPANDED (image-placeholder) prompt plus
        // all generated tokens except the last — the paged decode loop never
        // forwards the final sampled token into the cache, so it would not match
        // the live `request_tokens` (drop-last rule shared with the text paged
        // core).
        let mut full_history = expanded_tokens.clone();
        if !generated_tokens.is_empty() {
            full_history.extend_from_slice(&generated_tokens[..generated_tokens.len() - 1]);
        }

        // Keep-live registration must run before the GDN checkpoint, which
        // snapshots the live recurrent state; short-circuit `&&` preserves that
        // order. `remember_dense_gdn_history_checkpoint` snapshots from
        // `cached_token_history`, so publish the history first, then checkpoint.
        // Any failure downgrades to NON-continuable rather than discarding the
        // already-successful generation output.
        let keep_live_ok = p.reuse_cache
            && self
                .finalize_dense_manual_paged_turn(&image_token_positions)
                .is_ok();
        let continuable = if keep_live_ok {
            self.cached_token_history = full_history;
            self.cached_image_key = Some(image_cache_key);
            self.cached_paged_image_token_positions = image_token_positions.clone();
            match self.remember_dense_gdn_history_checkpoint() {
                Ok(_) => true,
                Err(error) => {
                    tracing::warn!(
                        target: "mlx_core::qwen3_5::paged",
                        "dense VLM GDN history checkpoint failed: {error}",
                    );
                    self.invalidate_dense_paged_session("VLM sync GDN history checkpoint failure");
                    false
                }
            }
        } else {
            false
        };

        if continuable {
            // FULLY continuable: live KV + GDN recurrent state encode the image
            // context; `cached_image_key` records it (flat vision save contract).
        } else if self.caches.is_some() {
            // No-reuse, keep-live failure, or checkpoint failure: release the
            // request and reset to a pristine non-live state. `reset_caches_sync`
            // nulls `self.caches` (so `has_live_session()` reports false) and
            // clears token history, image key, rope deltas, and GDN checkpoints,
            // so a follow-up continue is rejected ("requires an initialized
            // session") instead of cold-prefilling image-placeholder ids.
            if p.reuse_cache {
                self.invalidate_dense_paged_session("non-continuable VLM sync completion");
            } else {
                self.discard_dense_paged_session();
            }
        }

        let performance = if report_perf {
            compute_performance_metrics(
                generation_start,
                first_token_instant,
                expanded_tokens.len() - cached_prefix_len as usize,
                generated_tokens.len(),
            )
        } else {
            None
        };

        let mut result = finalize_chat_result(
            &tokenizer,
            &generated_tokens,
            finish_reason,
            think_end_id,
            think_end_str.as_deref(),
            performance,
            p.include_reasoning,
            thinking_enabled,
            prompt_token_count,
            reasoning_tracker.reasoning_token_count(),
        )?;
        result.cached_tokens = cached_prefix_len;
        Ok(result)
    }

    /// Streaming twin of [`Self::vision_paged_turn_sync_core`].
    ///
    /// Single-turn image-bearing block-paged dispatch that emits each
    /// generated token through the streaming callback. Same prefill + decode
    /// spine; plain AR decode, MTP weights ignored.
    #[allow(clippy::too_many_arguments)]
    fn vision_paged_turn_stream_core(
        &mut self,
        tokens: Vec<u32>,
        images: &[Vec<u8>],
        tokenizer: Arc<Qwen3Tokenizer>,
        eos_token_id: u32,
        mut p: engine::ChatParams,
        report_perf: bool,
        cb: &StreamSender<'_>,
        cancelled: &AtomicBool,
        thinking: ThinkingSetup,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Err(Error::from_reason("Empty prompt"));
        }
        self.preflight_paged_context(tokens.len(), &mut p)?;

        let (vision_encoder, img_proc) =
            match (self.vision_encoder.clone(), self.image_processor.as_ref()) {
                (Some(enc), Some(proc)) => (enc, proc),
                _ => {
                    return Err(Error::from_reason(
                        "VLM prefill requested but vision encoder/processor not loaded",
                    ));
                }
            };

        self.paged_full_attn_caches_dirty = true;

        let include_reasoning = p.include_reasoning;
        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());
        let thinking_enabled = thinking.enabled;
        let mut reasoning_tracker = engine::ReasoningTracker::from_setup(&thinking, think_end_id);

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;
        let sampling_config = p.sampling_config;

        let mut decode_stream = tokenizer.inner().decode_stream(true);
        let mut streamed_text_len = 0usize;
        let mut last_is_reasoning = thinking_enabled;

        // === VLM image processing: expand placeholders + merge features ===
        let sms = self.spatial_merge_size.unwrap_or(2);
        let image_refs: Vec<&[u8]> = images.iter().map(|v| v.as_slice()).collect();
        let processed = img_proc.process_many(&image_refs)?;
        let per_image_token_counts =
            compute_image_token_counts_per_image(&processed.grid_thw(), sms)?;
        let expanded_tokens = inject_image_placeholders(&tokens, &per_image_token_counts)?;
        self.preflight_paged_context(expanded_tokens.len(), &mut p)?;
        let (image_cache_key, per_image_hashes) = engine::compute_image_cache_keys(images);
        let image_token_positions = engine::map_expanded_image_token_positions(
            &expanded_tokens,
            IMAGE_TOKEN_ID as u32,
            &per_image_token_counts,
            &per_image_hashes,
        )
        .map_err(Error::from_reason)?;
        let prompt_token_count = expanded_tokens.len() as u32;

        let embed = self.embedding.clone();
        let embedding_weight = embed.get_weight();
        let input_ids = MxArray::from_uint32(&expanded_tokens, &[1, expanded_tokens.len() as i64])?;

        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        let merge = vlm_prepare_vision_features(
            &input_ids,
            &per_image_hashes,
            &processed,
            &vision_encoder,
            sms,
            &embedding_weight,
            generation_stream,
            &self.vision_cache,
        )?;
        drop(processed);
        crate::array::clear_cache();

        // === Image-aware paged-prefix lifecycle ===
        let total_budget = expanded_tokens.len() as u32;
        let block_size = self
            .paged_adapter
            .as_ref()
            .ok_or_else(|| {
                Error::from_reason("vision_paged_turn_stream_core: paged_adapter is None")
            })?
            .block_size();
        let lookup_extra_keys = engine::build_paged_extra_keys(
            expanded_tokens.len(),
            block_size,
            &image_token_positions,
        );
        let same_live_image = self.cached_image_key == Some(image_cache_key);
        let prefix_resolution = self.prepare_dense_vlm_paged_prefix(
            &expanded_tokens,
            total_budget,
            block_size,
            &lookup_extra_keys,
            p.reuse_cache,
            p.reuse_cache && same_live_image,
            image_cache_key,
        )?;
        let plan = prefix_resolution.effective_plan;
        let cached_prefix_len = plan.cached_prefix_len;
        tracing::info!(
            target: "mlx_core::inference",
            event = "vlm_prefix_plan",
            model = "qwen3_5_dense",
            prompt_tokens = expanded_tokens.len(),
            image_count = images.len(),
            image_tokens = image_token_positions.len(),
            candidate_cached_prefix_tokens = prefix_resolution.candidate_cached_prefix_len,
            effective_cached_prefix_tokens = cached_prefix_len,
            cached_prefix_tokens = cached_prefix_len,
            suffix_tokens = expanded_tokens.len() - cached_prefix_len as usize,
            continued_live_prefix = plan.continued_live_prefix,
            same_live_image,
            gdn_prefix_already_primed = prefix_resolution.gdn_prefix_already_primed,
            downgraded_to_cold = prefix_resolution.downgraded_to_cold,
            "image-aware paged prefix planned"
        );
        let gdn_prefix_already_primed = prefix_resolution.gdn_prefix_already_primed;
        self.cached_token_history.clear();
        self.cached_image_key = None;
        self.cached_paged_image_token_positions.clear();
        // Store the image prefill's compressed-position delta so a later text
        // warm-continuation rotates its queries at the same compressed M-RoPE
        // positions the image keys were written with.
        self.cached_rope_deltas = Some(merge.rope_deltas as i32);

        let layer_kinds =
            super::decoder_layer::compute_layer_kinds(self.config.num_layers as usize, |i| {
                self.config.is_linear_layer(i)
            });

        let forward_result = (|| -> Result<(Vec<u32>, String)> {
            // === PREFILL ===
            let last_logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                let caches_ref = self.caches.as_mut().ok_or_else(|| {
                    Error::from_reason("vision_paged_turn_stream_core: caches not initialized")
                })?;
                let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                    Error::from_reason("vision_paged_turn_stream_core: paged_adapter dropped")
                })?;
                super::paged_forward::run_paged_vlm_prefill(
                    &expanded_tokens,
                    &merge,
                    cached_prefix_len,
                    gdn_prefix_already_primed,
                    &embed,
                    &mut self.layers,
                    caches_ref,
                    &self.final_norm,
                    &self.lm_head,
                    &embedding_weight,
                    &layer_kinds,
                    adapter,
                )?
            };
            let (last_logits, gdn_checkpoint) = last_logits;
            self.publish_dense_gdn_materialized_prefix_checkpoint_with_keys(
                &expanded_tokens,
                &lookup_extra_keys,
                0,
                gdn_checkpoint,
            );

            let mut token_history: Vec<u32> = expanded_tokens.clone();
            let last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
            let mut y = sample(&last_logits, sampling_config)?;
            y.eval();

            crate::array::synchronize_and_clear_cache();
            if report_perf {
                first_token_instant = Some(std::time::Instant::now());
            }

            let max_new_tokens = p.max_new_tokens;
            let mut generated_tokens: Vec<u32> =
                Vec::with_capacity(engine::generated_capacity_hint(max_new_tokens));
            let mut finish_reason = String::from("length");

            for step in 0..max_new_tokens {
                let token_id = y.item_at_int32(0)? as u32;
                generated_tokens.push(token_id);
                token_history.push(token_id);
                let is_reasoning = reasoning_tracker.observe_token(token_id);
                last_is_reasoning = is_reasoning;

                if token_id == eos_token_id || p.extra_eos_ids.contains(&token_id) {
                    finish_reason = String::from("stop");
                    break;
                }
                if cancelled.load(Ordering::Relaxed) {
                    finish_reason = String::from("cancelled");
                    break;
                }

                let token_text = Qwen3Tokenizer::step_decode_stream(
                    &mut decode_stream,
                    tokenizer.inner(),
                    token_id,
                    &generated_tokens,
                    streamed_text_len,
                );
                streamed_text_len += token_text.len();
                if include_reasoning || !is_reasoning {
                    cb.call(
                        Ok(ChatStreamChunk {
                            text: token_text,
                            done: false,
                            finish_reason: None,
                            tool_calls: None,
                            thinking: None,
                            thinking_enabled: None,
                            num_tokens: None,
                            prompt_tokens: None,
                            reasoning_tokens: None,
                            raw_text: None,
                            public_raw_text: None,
                            text_authoritative: None,
                            cached_tokens: None,
                            performance: None,
                            is_reasoning: Some(is_reasoning),
                        }),
                        ThreadsafeFunctionCallMode::NonBlocking,
                    );
                }

                if let Some(reason) = crate::sampling::check_repetition_cutoff(
                    &generated_tokens,
                    p.max_consecutive_tokens,
                    p.max_ngram_repeats,
                    p.ngram_size,
                ) {
                    finish_reason = reason.to_string();
                    break;
                }
                if step + 1 >= max_new_tokens {
                    break;
                }

                let next_logits = {
                    let _stream_ctx = StreamContext::new(generation_stream);
                    let caches_ref = self.caches.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "vision_paged_turn_stream_core: caches dropped mid-decode",
                        )
                    })?;
                    let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "vision_paged_turn_stream_core: paged_adapter dropped mid-decode",
                        )
                    })?;
                    let logits = super::paged_forward::run_paged_decode_step(
                        token_id,
                        &embed,
                        &mut self.layers,
                        caches_ref,
                        &self.final_norm,
                        &self.lm_head,
                        &embedding_weight,
                        &layer_kinds,
                        adapter,
                        self.cached_rope_deltas.unwrap_or(0),
                    )?;
                    logits.squeeze(Some(&[1]))?
                };

                if reasoning_tracker.should_force_think_end() {
                    let forced_id = reasoning_tracker.forced_token_id()? as i32;
                    y = MxArray::from_int32(&[forced_id], &[1])?;
                    y.eval();
                    continue;
                }
                let next_logits = apply_all_penalties(next_logits, &token_history, &p)?;

                y = sample(&next_logits, sampling_config)?;
                y.eval();

                crate::array::maybe_clear_cache_for_paged_step(step);
            }

            Ok((generated_tokens, finish_reason))
        })();

        // Terminal lifecycle, mirroring the text paged core. The error path
        // always releases and returns. The success path is resolved below so
        // the session ends FULLY continuable (keep-live + GDN checkpoint +
        // history + image key) or NON-continuable (released + history cleared +
        // image key None), never partial — a follow-up text continue must never
        // cold-prefill image-placeholder ids as ordinary tokens.
        let (generated_tokens, finish_reason) = match forward_result {
            Ok(t) => t,
            Err(e) => {
                self.invalidate_dense_paged_session("VLM stream prefill/decode failure");
                return Err(e);
            }
        };

        // Saved history: expanded prompt + generated[..len-1] (drop-last rule
        // shared with the sync sibling / text paged core).
        let mut full_history = expanded_tokens.clone();
        if !generated_tokens.is_empty() {
            full_history.extend_from_slice(&generated_tokens[..generated_tokens.len() - 1]);
        }

        // Keep-live before the GDN checkpoint (which snapshots the live state);
        // checkpoint reads `cached_token_history`, so publish it first. Any
        // failure downgrades to NON-continuable rather than discarding output.
        let keep_live_ok = p.reuse_cache
            && self
                .finalize_dense_manual_paged_turn(&image_token_positions)
                .is_ok();
        let continuable = if keep_live_ok {
            self.cached_token_history = full_history;
            self.cached_image_key = Some(image_cache_key);
            self.cached_paged_image_token_positions = image_token_positions.clone();
            match self.remember_dense_gdn_history_checkpoint() {
                Ok(_) => true,
                Err(error) => {
                    tracing::warn!(
                        target: "mlx_core::qwen3_5::paged",
                        "dense streaming VLM GDN history checkpoint failed: {error}",
                    );
                    self.invalidate_dense_paged_session(
                        "VLM stream GDN history checkpoint failure",
                    );
                    false
                }
            }
        } else {
            false
        };

        if continuable {
        } else if self.caches.is_some() {
            // Non-continuable: release the request and reset to a pristine
            // non-live state so a follow-up continue is rejected instead of
            // cold-prefilling image-placeholder ids. `reset_caches_sync` nulls
            // `self.caches` (so `has_live_session()` is false) and clears token
            // history, image key, rope deltas, and GDN checkpoints.
            if p.reuse_cache {
                self.invalidate_dense_paged_session("non-continuable VLM stream completion");
            } else {
                self.discard_dense_paged_session();
            }
        }

        // Flush residual buffered bytes (mirrors flat / text paged streaming).
        let full_text = tokenizer
            .decode_sync(&generated_tokens, true)
            .unwrap_or_else(|e| {
                tracing::warn!("Failed to decode generated tokens: {}", e);
                String::new()
            });
        if full_text.len() > streamed_text_len {
            let residual = full_text[streamed_text_len..].to_string();
            if include_reasoning || !last_is_reasoning {
                cb.call(
                    Ok(ChatStreamChunk {
                        text: residual,
                        done: false,
                        finish_reason: None,
                        tool_calls: None,
                        thinking: None,
                        thinking_enabled: None,
                        num_tokens: None,
                        prompt_tokens: None,
                        reasoning_tokens: None,
                        raw_text: None,
                        public_raw_text: None,
                        text_authoritative: None,
                        cached_tokens: None,
                        performance: None,
                        is_reasoning: Some(last_is_reasoning),
                    }),
                    ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
        }

        let performance = if report_perf {
            compute_performance_metrics(
                generation_start,
                first_token_instant,
                expanded_tokens.len() - cached_prefix_len as usize,
                generated_tokens.len(),
            )
        } else {
            None
        };

        let reasoning_tokens = reasoning_tracker.reasoning_token_count();
        let result = finalize_chat_result(
            &tokenizer,
            &generated_tokens,
            finish_reason,
            think_end_id,
            think_end_str.as_deref(),
            performance,
            include_reasoning,
            thinking_enabled,
            prompt_token_count,
            reasoning_tokens,
        )?;

        cb.call(
            Ok(ChatStreamChunk {
                text: result.text.clone(),
                done: true,
                finish_reason: Some(result.finish_reason.clone()),
                tool_calls: Some(result.tool_calls.clone()),
                thinking: result.thinking.clone(),
                thinking_enabled: Some(result.thinking_enabled),
                num_tokens: Some(result.num_tokens),
                prompt_tokens: Some(result.prompt_tokens),
                reasoning_tokens: Some(result.reasoning_tokens),
                raw_text: Some(result.raw_text.clone()),
                public_raw_text: result.public_raw_text.clone(),
                text_authoritative: Some(true),
                cached_tokens: Some(cached_prefix_len),
                performance: result.performance.clone(),
                is_reasoning: None,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );

        Ok(())
    }

    /// Block-paged streaming variant of [`Self::chat_stream_sync_inner`].
    ///
    /// Mirrors `paged_turn_sync_core`'s adapter lifecycle and
    /// per-layer dispatch but emits each generated token through the
    /// streaming callback as it is produced.
    #[allow(clippy::too_many_arguments)]
    fn paged_turn_stream_core(
        &mut self,
        tokens: Vec<u32>,
        tokenizer: Arc<Qwen3Tokenizer>,
        eos_token_id: u32,
        mut p: engine::ChatParams,
        report_perf: bool,
        cb: &StreamSender<'_>,
        cancelled: &AtomicBool,
        thinking: ThinkingSetup,
    ) -> Result<()> {
        static NEXT_TRACE_TURN_ID: AtomicU64 = AtomicU64::new(1);
        if tokens.is_empty() {
            return Err(Error::from_reason("Empty prompt"));
        }
        let inference_info_enabled =
            tracing::enabled!(target: "mlx_core::inference", tracing::Level::INFO);
        let trace_turn_id = if inference_info_enabled {
            NEXT_TRACE_TURN_ID.fetch_add(1, Ordering::Relaxed)
        } else {
            0
        };
        let turn_trace_start = inference_info_enabled.then(std::time::Instant::now);
        if inference_info_enabled {
            tracing::info!(
                target: "mlx_core::inference",
                event = "turn_start",
                turn_id = trace_turn_id,
                path = "qwen3_5_dense_paged_stream",
                prompt_tokens = tokens.len(),
                mtp_requested = p.enable_mtp,
                mtp_depth = p.mtp_depth,
                report_performance = report_perf,
                "inference turn started"
            );
        }
        self.preflight_paged_context(tokens.len(), &mut p)?;

        // This paged turn writes full-attention K/V into the paged adapter
        // pool, NOT the flat `self.caches`, so a later streaming dense-MTP
        // fallback must rebuild the flat caches before decoding. See
        // `paged_full_attn_caches_dirty`.
        self.paged_full_attn_caches_dirty = true;

        let prompt_token_count = tokens.len() as u32;
        let sampling_config = p.sampling_config;
        let include_reasoning = p.include_reasoning;

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());
        // Thinking is resolved ONCE at turn entry and honors
        // `enable_thinking=false`.
        let thinking_enabled = thinking.enabled;
        let mut reasoning_tracker = engine::ReasoningTracker::from_setup(&thinking, think_end_id);

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;
        let trace_enabled = inference_trace_enabled();

        // Streaming decode state.
        let mut decode_stream = tokenizer.inner().decode_stream(true);
        let mut streamed_text_len = 0usize;
        let mut last_is_reasoning = thinking_enabled;
        let prefix_plan_start = inference_info_enabled.then(std::time::Instant::now);

        // === Adapter lifecycle: warm continue OR cold start ===
        let seq_id: u32 = 0;
        // Lazy decode allocation: pass the prompt length only.
        let total_budget = tokens.len() as u32;
        // Per-block extra_keys for prefix-cache lookup. See the matching
        // comment in `paged_turn_sync_core`.
        let block_size = {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason("paged_turn_stream_core: paged_adapter is None")
            })?;
            adapter.block_size()
        };
        let carries_image_lineage = self.cached_image_key.is_some()
            && !self.cached_paged_image_token_positions.is_empty()
            && !self.cached_token_history.is_empty()
            && tokens.starts_with(&self.cached_token_history);
        let image_positions = if carries_image_lineage {
            self.cached_paged_image_token_positions.as_slice()
        } else {
            &[]
        };
        let lookup_extra_keys =
            engine::build_paged_extra_keys(tokens.len(), block_size, image_positions);
        let cache_salt = 0;
        // See `paged_turn_sync_core` for the vLLM exact-prefix cap rationale.
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
        let live_ready;
        let live_prefix_match;
        let live_tokens_len;
        let mut live_mismatch = TokenPrefixMismatchTrace::default();
        // Adapter-owned warm/cold lifecycle (see paged_turn_sync_core for
        // the full bit-identity rationale: pre-turn immutable probe for the
        // trace, extra_keys=&[] bit-equal to per-block for text-only,
        // reuse_cache=true, suffix blocks allocated internally).
        {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason(
                    "paged_turn_stream_core: paged_adapter is None — caller must check \
                     use_block_paged_cache before dispatch",
                )
            })?;
            live_ready = adapter.is_live_for_continue();
            let live_tokens = adapter.request_tokens();
            live_tokens_len = live_tokens.len();
            live_prefix_match = tokens.starts_with(live_tokens);
            if trace_enabled && live_ready && !live_prefix_match {
                live_mismatch = token_prefix_mismatch_trace(&tokens, live_tokens);
            }
        }
        let plan_result = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| {
                Error::from_reason(
                    "paged_turn_stream_core: paged_adapter is None — caller must check \
                     use_block_paged_cache before dispatch",
                )
            })?
            .prepare_turn_per_block_with_max_cache_hit_tokens(
                seq_id,
                &tokens,
                total_budget,
                true,
                &lookup_extra_keys,
                cache_salt,
                false,
                max_cache_hit_tokens,
            )
            .map_err(Error::from_reason);
        let plan = match plan_result {
            Ok(plan) => plan,
            Err(error) => {
                self.invalidate_dense_paged_session(
                    "planned-MTP stream adapter preparation failure",
                );
                return Err(error);
            }
        };
        let cached_prefix_len = plan.cached_prefix_len;
        let continued_live_prefix = plan.continued_live_prefix;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-dense paged_prefix_lookup prompt_tokens={} \
                 cached_prefix_tokens={} continued_live_prefix={} live_ready={} \
                 live_match={} live_tokens={} live_mismatch_at={} prompt_token={} live_token={}",
                tokens.len(),
                cached_prefix_len,
                continued_live_prefix,
                live_ready,
                live_prefix_match,
                live_tokens_len,
                live_mismatch.index,
                live_mismatch.prompt_token,
                live_mismatch.cached_token
            ));
        }

        let gdn_prefix_preparation = match self.prepare_dense_gdn_prefix_state(
            &tokens,
            cached_prefix_len,
            block_size,
            &lookup_extra_keys,
            cache_salt,
            continued_live_prefix,
        ) {
            Ok(preparation) => preparation,
            Err(error) => {
                self.invalidate_dense_paged_session(
                    "planned-MTP stream GDN-prefix preparation failure",
                );
                return Err(error);
            }
        };
        let gdn_prefix_already_primed = gdn_prefix_preparation.already_primed;
        // Discharge the adapter's auxiliary (GDN) prefix obligation before the
        // turn's first `record_tokens`. See `prime_prefix_state` for the full
        // rationale; a no-op unless a HOT K/V hit handed back a prefix the cold
        // tier's `ColdSidecarPolicy` was not gated against.
        let confirm_result = self
            .paged_adapter
            .as_mut()
            .map(|adapter| adapter.confirm_aux_prefix_primed(cached_prefix_len))
            .unwrap_or(Ok(()));
        if let Err(error) = confirm_result {
            self.invalidate_dense_paged_session(
                "planned-MTP stream GDN auxiliary-prefix confirmation",
            );
            return Err(Error::from_reason(error));
        }
        let gdn_prefix_state = gdn_prefix_preparation.state;
        let gdn_restored_prefix_tokens = gdn_prefix_preparation.restored_prefix_tokens;
        let gdn_replayed_prefix_tokens = gdn_prefix_preparation.replayed_prefix_tokens;
        let preserves_image_lineage = carries_image_lineage && cached_prefix_len > 0;
        self.cached_token_history.clear();
        if !preserves_image_lineage {
            self.cached_image_key = None;
            self.cached_paged_image_token_positions.clear();
        }
        // Carry the cross-turn M-RoPE delta only when this turn extends the live
        // image sequence (continued_live_prefix); a cold start or a non-live
        // prefix-cache hit (text-only prefix) drops a stale image delta so the
        // text suffix prefill + decode rotate at the raw physical slot.
        self.cached_rope_deltas = super::paged_forward::rope_delta_for_paged_turn(
            self.cached_rope_deltas,
            preserves_image_lineage,
        );

        let suffix_len = match prompt_token_count.checked_sub(cached_prefix_len) {
            Some(suffix_len) => suffix_len,
            None => {
                self.invalidate_dense_paged_session("planned-MTP stream prefix length mismatch");
                return Err(Error::from_reason(
                    "paged_turn_stream_core: cached_prefix_len > total_prompt_tokens",
                ));
            }
        };

        if inference_info_enabled {
            tracing::info!(
                target: "mlx_core::inference",
                event = "prefix_plan",
                turn_id = trace_turn_id,
                prompt_tokens = prompt_token_count,
                cached_prefix_tokens = cached_prefix_len,
                suffix_tokens = suffix_len,
                continued_live_prefix,
                live_ready,
                live_prefix_match,
                live_tokens = live_tokens_len,
                block_size,
                gdn_prefix_state,
                gdn_already_primed = gdn_prefix_already_primed,
                gdn_restored_prefix_tokens,
                gdn_replayed_prefix_tokens,
                elapsed_ms = prefix_plan_start.map(elapsed_ms).unwrap_or(0.0),
                "paged prefix plan resolved"
            );
        }

        if trace_enabled {
            // The size this turn's prefill actually splits on, not the raw
            // `MLX_PAGED_PREFILL_CHUNK_SIZE`: under a GDN cold policy the two
            // differ, and reading 0 here while the same turn reports
            // `gdn_checkpoint_materialized=true` points at the wrong diagnosis.
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-dense stream_paged_start prompt_tokens={} \
                 cached_prefix_tokens={} suffix_tokens={} block_size={} \
                 prefill_chunk_size={} prefill_eval_interval={} decode_clear_interval={} \
                 gdn_prefix_state={}",
                prompt_token_count,
                cached_prefix_len,
                suffix_len,
                block_size,
                self.cold_gdn_prefill_chunk_size(),
                crate::array::paged_prefill_eval_interval(),
                crate::array::paged_decode_cache_clear_interval(),
                gdn_prefix_state
            ));
        }

        let result = self.paged_turn_stream_core_inner(
            &tokens,
            cached_prefix_len,
            suffix_len,
            &p,
            sampling_config,
            eos_token_id,
            &mut reasoning_tracker,
            report_perf,
            &mut first_token_instant,
            &tokenizer,
            &mut decode_stream,
            &mut streamed_text_len,
            &mut last_is_reasoning,
            cb,
            cancelled,
            gdn_prefix_already_primed,
            trace_turn_id,
            turn_trace_start,
        );

        let (generated_tokens, finish_reason, decode_profiler) = match result {
            Ok(t) => {
                let image_token_positions = self.cached_paged_image_token_positions.clone();
                self.finalize_dense_manual_paged_turn(&image_token_positions)?;
                t
            }
            Err(e) => {
                self.invalidate_dense_paged_session("planned-MTP stream prefill/decode failure");
                if inference_info_enabled {
                    tracing::info!(
                        target: "mlx_core::inference",
                        event = "turn_done",
                        turn_id = trace_turn_id,
                        status = "error",
                        stage = "prefill_or_decode",
                        elapsed_ms = turn_trace_start.map(elapsed_ms).unwrap_or(0.0),
                        "inference turn completed"
                    );
                }
                return Err(e);
            }
        };

        // Persist token history for subsequent session-continue calls.
        let last_token_in_cache = false;
        let mut full_history = tokens.clone();
        if !generated_tokens.is_empty() {
            let upto = if last_token_in_cache {
                generated_tokens.len()
            } else {
                generated_tokens.len().saturating_sub(1)
            };
            full_history.extend_from_slice(&generated_tokens[..upto]);
        }
        self.cached_token_history = full_history;
        let gdn_history_checkpoint_store = match self.remember_dense_gdn_history_checkpoint() {
            Ok(store) => store,
            Err(error) => {
                self.invalidate_dense_paged_session(
                    "planned-MTP stream GDN history checkpoint failure",
                );
                return Err(error);
            }
        };
        if inference_info_enabled {
            tracing::info!(
                target: "mlx_core::inference",
                event = "cache_commit",
                turn_id = trace_turn_id,
                stored = gdn_history_checkpoint_store.stored,
                history_tokens = self.cached_token_history.len(),
                eval_ms = gdn_history_checkpoint_store.eval_ms,
                clone_ms = gdn_history_checkpoint_store.clone_ms,
                update_ms = gdn_history_checkpoint_store.update_ms,
                elapsed_ms = gdn_history_checkpoint_store.total_ms,
                "inference cache committed"
            );
        }
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-dense gdn_history_checkpoint stored={} tokens={} \
                 eval_ms={:.1} clone_ms={:.1} token_clone_ms={:.1} update_ms={:.1} total_ms={:.1}",
                gdn_history_checkpoint_store.stored,
                self.cached_token_history.len(),
                gdn_history_checkpoint_store.eval_ms,
                gdn_history_checkpoint_store.clone_ms,
                gdn_history_checkpoint_store.token_clone_ms,
                gdn_history_checkpoint_store.update_ms,
                gdn_history_checkpoint_store.total_ms
            ));
        }

        // Flush residual buffered bytes (mirrors flat streaming).
        let full_text = tokenizer
            .decode_sync(&generated_tokens, true)
            .unwrap_or_else(|e| {
                tracing::warn!("Failed to decode generated tokens: {}", e);
                String::new()
            });
        if full_text.len() > streamed_text_len {
            let residual = full_text[streamed_text_len..].to_string();
            // Suppress residual when it is reasoning text and
            // include_reasoning == false.
            if include_reasoning || !last_is_reasoning {
                cb.call(
                    Ok(ChatStreamChunk {
                        text: residual,
                        done: false,
                        finish_reason: None,
                        tool_calls: None,
                        thinking: None,
                        thinking_enabled: None,
                        num_tokens: None,
                        prompt_tokens: None,
                        reasoning_tokens: None,
                        raw_text: None,
                        public_raw_text: None,
                        text_authoritative: None,
                        cached_tokens: None,
                        performance: None,
                        is_reasoning: Some(last_is_reasoning),
                    }),
                    ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
        }

        let performance = if report_perf {
            compute_performance_metrics(
                generation_start,
                first_token_instant,
                tokens.len() - cached_prefix_len as usize,
                generated_tokens.len(),
            )
            .map(|mut metrics| {
                if let Some(profiler) = decode_profiler.as_ref() {
                    profiler.fill_mtp_acceptance(&mut metrics);
                }
                metrics
            })
        } else {
            None
        };

        let reasoning_tokens = reasoning_tracker.reasoning_token_count();

        let mut result = finalize_chat_result(
            &tokenizer,
            &generated_tokens,
            finish_reason,
            think_end_id,
            think_end_str.as_deref(),
            performance,
            include_reasoning,
            thinking_enabled,
            prompt_token_count,
            reasoning_tokens,
        )?;
        result.cached_tokens = cached_prefix_len;

        if inference_info_enabled {
            let (ttft_ms, prefill_tok_s, decode_tok_s, mtp_cycles, mtp_mean_total) = result
                .performance
                .as_ref()
                .map(|metrics| {
                    (
                        metrics.ttft_ms,
                        metrics.prefill_tokens_per_second,
                        metrics.decode_tokens_per_second,
                        metrics.mtp_cycles.unwrap_or(0),
                        metrics.mtp_mean_accepted_tokens_total.unwrap_or(0.0),
                    )
                })
                .unwrap_or((0.0, 0.0, 0.0, 0, 0.0));
            tracing::info!(
                target: "mlx_core::inference",
                event = "turn_done",
                turn_id = trace_turn_id,
                status = "ok",
                prompt_tokens = prompt_token_count,
                cached_prefix_tokens = cached_prefix_len,
                fresh_prefill_tokens = tokens.len().saturating_sub(cached_prefix_len as usize),
                generated_tokens = generated_tokens.len(),
                finish_reason = result.finish_reason.as_str(),
                elapsed_ms = turn_trace_start.map(elapsed_ms).unwrap_or(0.0),
                ttft_ms,
                prefill_tok_s,
                decode_tok_s,
                mtp_cycles,
                mtp_mean_total,
                "inference turn completed"
            );
        }

        // Terminal chunk.
        cb.call(
            Ok(ChatStreamChunk {
                text: result.text.clone(),
                done: true,
                finish_reason: Some(result.finish_reason.clone()),
                tool_calls: Some(result.tool_calls.clone()),
                thinking: result.thinking.clone(),
                thinking_enabled: Some(result.thinking_enabled),
                num_tokens: Some(result.num_tokens),
                prompt_tokens: Some(result.prompt_tokens),
                reasoning_tokens: Some(result.reasoning_tokens),
                raw_text: Some(result.raw_text.clone()),
                public_raw_text: result.public_raw_text.clone(),
                text_authoritative: Some(true),
                cached_tokens: Some(cached_prefix_len),
                performance: result.performance.clone(),
                is_reasoning: None,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );

        Ok(())
    }

    /// Inner forward + streaming decode loop for
    /// [`Self::paged_turn_stream_core`]. Mirrors LFM2's
    /// `paged_turn_stream_core_inner`.
    ///
    /// Runs the pure-Rust paged path — same dispatch as the sync sibling
    /// `paged_turn_sync_core_inner`: prefill populates the GDN linear
    /// caches and writes K/V into the adapter pool, then decode steps run
    /// through `paged_forward::run_paged_decode_step` (or the eager paged
    /// MTP arm when the MTP gate holds).
    #[allow(clippy::too_many_arguments)]
    fn paged_turn_stream_core_inner<'a>(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        suffix_len: u32,
        p: &engine::ChatParams,
        sampling_config: Option<crate::sampling::SamplingConfig>,
        eos_token_id: u32,
        reasoning_tracker: &mut engine::ReasoningTracker,
        report_perf: bool,
        first_token_instant: &mut Option<std::time::Instant>,
        tokenizer: &'a Arc<Qwen3Tokenizer>,
        decode_stream: &mut tokenizers::DecodeStream<
            'a,
            tokenizers::ModelWrapper,
            tokenizers::NormalizerWrapper,
            tokenizers::PreTokenizerWrapper,
            tokenizers::PostProcessorWrapper,
            tokenizers::DecoderWrapper,
        >,
        streamed_text_len: &mut usize,
        last_is_reasoning: &mut bool,
        cb: &StreamSender<'_>,
        cancelled: &AtomicBool,
        gdn_prefix_already_primed: bool,
        trace_turn_id: u64,
        turn_trace_start: Option<std::time::Instant>,
    ) -> Result<(
        Vec<u32>,
        String,
        Option<crate::decode_profiler::DecodeProfiler>,
    )> {
        // Invariant: caller-applied vLLM cap guarantees suffix_len > 0.
        debug_assert!(
            suffix_len > 0,
            "paged_turn_stream_core_inner: caller must cap max_cache_hit_tokens at prompt.len() - 1"
        );

        let suffix = &tokens[(cached_prefix_len as usize)..];
        let layer_kinds =
            super::decoder_layer::compute_layer_kinds(self.config.num_layers as usize, |i| {
                self.config.is_linear_layer(i)
            });

        // Eager paged MTP needs the prompt-tail hidden for the
        // committed-history v2 seed, same as the sync core. Only capture it
        // when the eager paged MTP arm will actually run.
        let eager_mtp_paged = p.enable_mtp && self.has_mtp_weights();
        let want_prompt_hidden =
            eager_mtp_paged && !mtp_decode::mtp_no_prompt_prefill() && cached_prefix_len == 0;
        let mtp_prompt_history = mtp_decode::mtp_prompt_history_selection(tokens.len());
        let inference_info_enabled =
            tracing::enabled!(target: "mlx_core::inference", tracing::Level::INFO);
        let prefill_trace_start = inference_info_enabled.then(std::time::Instant::now);
        if inference_info_enabled {
            tracing::info!(
                target: "mlx_core::inference",
                event = "prefill_start",
                turn_id = trace_turn_id,
                suffix_tokens = suffix_len,
                cached_prefix_tokens = cached_prefix_len,
                mtp = eager_mtp_paged,
                keep_prompt_hidden_tokens = if want_prompt_hidden {
                    mtp_prompt_history.keep_tokens
                } else {
                    0
                },
                "prefill started"
            );
        }

        let mut mtp_profiler = begin_paged_mtp_profiler(eager_mtp_paged, suffix_len);
        let (last_logits, prompt_hidden, gdn_checkpoint) = self.run_dense_core_paged_prefill(
            tokens,
            suffix,
            cached_prefix_len,
            gdn_prefix_already_primed,
            want_prompt_hidden.then_some(mtp_prompt_history.keep_tokens),
            &layer_kinds,
            "paged_turn_stream_core_inner",
        )?;
        if let Some(profiler) = mtp_profiler.as_mut() {
            profiler.end_prefill();
        }
        self.publish_dense_gdn_materialized_prefix_checkpoint(tokens, gdn_checkpoint);

        let mut token_history: Vec<u32> = tokens.to_vec();
        let last_logits = apply_all_penalties(last_logits, &token_history, p)?;
        let mut y = sample(&last_logits, sampling_config)?;
        y.eval();

        if inference_info_enabled {
            let prefill_elapsed = prefill_trace_start.map(elapsed_ms).unwrap_or(0.0);
            tracing::info!(
                target: "mlx_core::inference",
                event = "first_token_sampled",
                turn_id = trace_turn_id,
                suffix_tokens = suffix_len,
                elapsed_ms = prefill_elapsed,
                materialized_prefill_tok_s = if prefill_elapsed > 0.0 {
                    suffix_len as f64 / (prefill_elapsed / 1000.0)
                } else {
                    0.0
                },
                "first token sampled"
            );
        }

        // Smooth memory peak: drop transient prefill buffers before decode
        // starts allocating (see paged_turn_sync_core_inner for rationale).
        crate::array::synchronize_and_clear_cache();

        if let Some(profiler) = mtp_profiler.as_mut() {
            profiler.mark_first_token();
        }
        if report_perf {
            *first_token_instant = Some(std::time::Instant::now());
        }

        let max_new_tokens = p.max_new_tokens;
        let mut generated_tokens: Vec<u32> =
            Vec::with_capacity(engine::generated_capacity_hint(max_new_tokens));
        let mut finish_reason = String::from("length");
        let decode_progress_start = std::time::Instant::now();
        let mut decode_progress_window_start = decode_progress_start;
        let mut decode_progress_last_generated = 0usize;
        let mut decode_progress_next_generated = 32usize;

        if eager_mtp_paged {
            // Pure-Rust ("eager") paged MTP — streaming twin of the sync core's
            // `eager_mtp_paged` arm. Same stepper spine; the engine's
            // `run_mtp_turn` streaming path emits decoded text per token via `cb`.
            MxArray::async_eval_arrays(&[&y]);

            let mut profiler = mtp_profiler
                .take()
                .expect("paged MTP profiler must be created before prefill");

            let eos_id = eos_token_id;
            let generation_stream = crate::stream::Stream::new(crate::stream::DeviceType::Gpu);
            let model_size_bytes = self.config.estimate_memory_bytes() as usize;
            let _wired_ctx =
                crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

            let prompt_hidden_position_base = mtp_prompt_history.position_base;
            let prompt_hidden_ids: Vec<u32> = {
                let start = mtp_prompt_history
                    .hidden_start_token_index()
                    .min(tokens.len());
                tokens[start..].to_vec()
            };

            let mut rng = rand::rng();

            if inference_info_enabled {
                tracing::info!(
                    target: "mlx_core::inference",
                    event = "mtp_decode_start",
                    turn_id = trace_turn_id,
                    depth = p.mtp_depth,
                    prompt_hidden_tokens = prompt_hidden_ids.len(),
                    position_base = prompt_hidden_position_base,
                    elapsed_ms = turn_trace_start.map(elapsed_ms).unwrap_or(0.0),
                    "MTP decode started"
                );
            }

            // Streaming twin of the sync paged arm: the propose/verify whole-turn
            // loop is engine-owned (`run_mtp_turn`) and drives the
            // `DenseMtpStepper` in its PAGED mode. The streaming sink wires the
            // shared incremental detokenizer + the default ChatML emitter so
            // accepted tokens stream out `cb` per token; the stepper's `Drop`
            // restores `self.paged_adapter` before this call returns (the
            // paged-history save below relies on it). `outcome.last_in_cache` is
            // unused (the paged-history save owns cache state).
            let mut emitter = crate::engine::backend::DefaultStreamEmitter;
            let streaming = crate::engine::decode::StreamingCtx {
                callback: cb.0,
                cancelled,
                decode_stream,
                tokenizer: tokenizer.inner(),
                streamed_text_len,
                last_is_reasoning,
                emitter: &mut emitter,
            };

            let outcome = crate::engine::mtp_turn::run_mtp_turn(
                self,
                &mut rng,
                crate::engine::mtp_turn::MtpTurnArgs {
                    y: y.clone(),
                    depth: p.mtp_depth,
                    params: p,
                    reasoning_tracker,
                    profiler: &mut profiler,
                    max_new_tokens,
                    eos_id,
                    generated_tokens: &mut generated_tokens,
                    token_history: &mut token_history,
                    finish_reason: &mut finish_reason,
                    first_token_instant,
                    report_perf: p.report_performance,
                    generation_stream,
                    prompt_hidden,
                    prompt_hidden_ids: Some(prompt_hidden_ids),
                    prompt_hidden_position_base,
                },
                Some(streaming),
            )?;
            let _ = outcome.last_in_cache;

            return Ok((generated_tokens, finish_reason, Some(profiler)));
        }

        for step in 0..max_new_tokens {
            let token_id = y.item_at_int32(0)? as u32;
            generated_tokens.push(token_id);
            token_history.push(token_id);
            let is_reasoning = reasoning_tracker.observe_token(token_id);
            *last_is_reasoning = is_reasoning;

            if inference_info_enabled && generated_tokens.len() >= decode_progress_next_generated {
                let now = std::time::Instant::now();
                let window_tokens = generated_tokens
                    .len()
                    .saturating_sub(decode_progress_last_generated);
                let window_elapsed_ms = now
                    .saturating_duration_since(decode_progress_window_start)
                    .as_secs_f64()
                    * 1000.0;
                tracing::info!(
                    target: "mlx_core::inference",
                    event = "decode_progress",
                    mode = "ar",
                    generated_tokens = generated_tokens.len(),
                    window_tokens,
                    window_elapsed_ms,
                    window_tok_s = if window_elapsed_ms > 0.0 {
                        window_tokens as f64 / (window_elapsed_ms / 1000.0)
                    } else {
                        0.0
                    },
                    elapsed_ms = now
                        .saturating_duration_since(decode_progress_start)
                        .as_secs_f64()
                        * 1000.0,
                    "decode progress"
                );
                decode_progress_last_generated = generated_tokens.len();
                decode_progress_window_start = now;
                decode_progress_next_generated = generated_tokens
                    .len()
                    .saturating_div(32)
                    .saturating_add(1)
                    .saturating_mul(32);
            }

            if token_id == eos_token_id || p.extra_eos_ids.contains(&token_id) {
                finish_reason = String::from("stop");
                break;
            }
            if cancelled.load(Ordering::Relaxed) {
                finish_reason = String::from("cancelled");
                break;
            }

            // Stream delta chunk.
            let token_text = Qwen3Tokenizer::step_decode_stream(
                decode_stream,
                tokenizer.inner(),
                token_id,
                &generated_tokens,
                *streamed_text_len,
            );
            *streamed_text_len += token_text.len();
            // Suppress reasoning deltas when include_reasoning == false.
            // Detokenize + length-advance above stay OUTSIDE this gate.
            if p.include_reasoning || !is_reasoning {
                cb.call(
                    Ok(ChatStreamChunk {
                        text: token_text,
                        done: false,
                        finish_reason: None,
                        tool_calls: None,
                        thinking: None,
                        thinking_enabled: None,
                        num_tokens: None,
                        prompt_tokens: None,
                        reasoning_tokens: None,
                        raw_text: None,
                        public_raw_text: None,
                        text_authoritative: None,
                        cached_tokens: None,
                        performance: None,
                        is_reasoning: Some(is_reasoning),
                    }),
                    ThreadsafeFunctionCallMode::NonBlocking,
                );
            }

            if let Some(reason) = crate::sampling::check_repetition_cutoff(
                &generated_tokens,
                p.max_consecutive_tokens,
                p.max_ngram_repeats,
                p.ngram_size,
            ) {
                finish_reason = reason.to_string();
                break;
            }
            if step + 1 >= max_new_tokens {
                break;
            }

            // Decode forward (pure-Rust paged step).
            let next_logits = {
                let embed = self.embedding.clone();
                let embedding_weight = embed.get_weight();
                let caches_ref = self.caches.as_mut().ok_or_else(|| {
                    Error::from_reason("paged_turn_stream_core_inner: caches dropped mid-decode")
                })?;
                let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                    Error::from_reason(
                        "paged_turn_stream_core_inner: paged_adapter dropped mid-decode",
                    )
                })?;
                let logits = super::paged_forward::run_paged_decode_step(
                    token_id,
                    &embed,
                    &mut self.layers,
                    caches_ref,
                    &self.final_norm,
                    &self.lm_head,
                    &embedding_weight,
                    &layer_kinds,
                    adapter,
                    self.cached_rope_deltas.unwrap_or(0),
                )?;
                logits.squeeze(Some(&[1]))?
            };

            let next_logits = if reasoning_tracker.should_force_think_end() {
                let forced_id = reasoning_tracker.forced_token_id()? as i32;
                y = MxArray::from_int32(&[forced_id], &[1])?;
                y.eval();
                continue;
            } else {
                apply_all_penalties(next_logits, &token_history, p)?
            };

            y = sample(&next_logits, sampling_config)?;
            y.eval();

            crate::array::maybe_clear_cache_for_paged_step(step);
        }

        Ok((generated_tokens, finish_reason, None))
    }

    /// Prefill the delta tokens and run the streaming decode loop.
    ///
    /// Whole-turn core for STREAMING delta turns reached through the
    /// engine's `mtp_turn` probe (MTP-enabled sessions; non-MTP
    /// streaming deltas run the engine's generic flow or the paged
    /// probe). Mirrors [`Self::chat_stream_sync_inner`] but skips the
    /// message rendering + prefix verification stages — the caller owns
    /// cache coherence by construction. Uses `<|im_end|>` as eos so the
    /// cached history continues to end on a clean ChatML boundary after
    /// the reply is saved.
    fn chat_stream_tokens_delta_sync_inner(
        &mut self,
        delta_tokens: Vec<u32>,
        config: ChatConfig,
        cb: &StreamSender<'_>,
        cancelled: &AtomicBool,
        thinking: ThinkingSetup,
    ) -> Result<()> {
        let _reuse_cache = config.reuse_cache.unwrap_or(true);
        let report_perf = config.report_performance.unwrap_or(false);

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();

        // Session path: use <|im_end|> as eos, NOT config.eos_token_id.
        let eos_id = tokenizer
            .im_end_id()
            .ok_or_else(|| Error::from_reason("Tokenizer missing <|im_end|> special token"))?;

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());
        let tokenizer_for_decode = tokenizer.clone();

        // Build full token history = cached_history + delta.
        // Capture `prior_cached_len` BEFORE the extend — this is the
        // reused-prefix length reported on the terminal ChatStreamChunk's
        // `cached_tokens` field (mirrors the non-streaming delta path's
        // `cached_tokens_for_result`).
        let prior_cached_len = self.cached_token_history.len() as u32;
        let mut full_token_history = self.cached_token_history.clone();
        full_token_history.extend(delta_tokens.iter().copied());

        let mut p = extract_chat_params(&config);
        p.extra_eos_ids = self.gen_defaults.eos_token_ids.clone();

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;

        // MTP-on-paged delta streams fall through to the dense (flat)
        // streaming path; non-MTP paged streams take the paged
        // streaming core.
        let mtp_takes_dense_path =
            p.enable_mtp && self.has_mtp_weights() && self.paged_adapter.is_some();
        if mtp_takes_dense_path
            && let Some(ref mut adapter) = self.paged_adapter
            && let Err(e) = adapter.release_request()
        {
            tracing::warn!(
                target: "mlx_core::qwen3_5::paged",
                "MTP-on-paged dispatch (stream-delta): release_request failed (ignored): {e}",
            );
        }
        if self.paged_adapter.is_some() && !mtp_takes_dense_path {
            return self.paged_turn_stream_core(
                full_token_history.clone(),
                tokenizer_for_decode,
                eos_id,
                p,
                report_perf,
                cb,
                cancelled,
                thinking,
            );
        }

        let embedding_weight = self.embedding.get_weight();
        let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        let mut profiler =
            crate::decode_profiler::DecodeProfiler::new("chat_stream_delta", "qwen3_5");
        // Prefill token count is reported below per-branch: the paged→dense
        // fallback re-prefills the FULL history (see `rebuild_full_flat_prefill`),
        // every other delta turn prefills only the delta.
        profiler.snapshot_memory_before();

        // Paged→dense-MTP cache-source transition.
        //
        // When `mtp_takes_dense_path` is true we arrived here off a PAGED
        // session (`self.paged_adapter.is_some()`) that fell through the
        // paged streaming core because it carries no MTP gate yet. On the
        // paged path the authoritative FULL-ATTENTION K/V lives in the
        // paged adapter's `LayerKVPool`, NOT in the flat `self.caches`
        // (which only ever received GDN linear conv/recurrent state). A
        // prior NON-streaming paged turn (`send()` → `chat_tokens_delta_sync`
        // → `paged_turn_sync_core`) therefore leaves `self.caches`'
        // full-attention slots EMPTY/STALE for the prior turn's tokens.
        // Delta-prefilling only `delta_tokens` on top of that and running
        // the eager MTP decode against `self.caches` would decode from an
        // incomplete flat prefix (missing the prior turn's attention KV).
        //
        // Fix: when taking this dense fallback AND the flat caches are dirty
        // (a paged-core turn ran since the last flat prefill —
        // `paged_full_attn_caches_dirty`), rebuild the flat caches from
        // scratch over the FULL token history: reset to fresh caches and
        // prefill `full_token_history` instead of just the delta. This
        // mirrors how the dense session-start path recovers from a
        // cache-prefix miss (full reset + full re-prefill). The GDN recurrent
        // state cannot be rewound mid-sequence, so a full reset + full
        // prefill is the only coherent way to seed both the GDN linear and
        // full-attention flat caches for the eager MTP decode.
        //
        // The dirty gate keeps this a ONE-TIME cost on the paged→dense
        // transition: the flag is cleared once at end-of-turn success (atomic
        // with the `cached_token_history` commit), so subsequent streaming MTP
        // turns delta-prefill on the now-authoritative flat caches (no O(n²)
        // full re-prefill every turn). It re-arms only if a later paged-core
        // turn runs again. The common non-paged dense
        // session (`paged_adapter.is_none()`, `mtp_takes_dense_path == false`,
        // flag never set) is untouched: it keeps the delta-on-existing-caches
        // prefill below, byte-identical.
        let rebuild_full_flat_prefill = (mtp_takes_dense_path && self.paged_full_attn_caches_dirty)
            || self.flat_mtp_caches_desynced;
        profiler.set_prompt_tokens(if rebuild_full_flat_prefill {
            full_token_history.len() as u32
        } else {
            delta_tokens.len() as u32
        });
        profiler.begin_prefill();
        let mut last_logits = if rebuild_full_flat_prefill {
            // Discard the paged-session flat caches (full-attn slots are
            // stale, GDN state belongs to the released paged request) and
            // re-prefill the entire conversation into fresh flat caches.
            self.flat_full_reprefill_count += 1;
            self.caches = Some(fresh_dense_layer_caches(&self.config));
            let prompt =
                MxArray::from_uint32(&full_token_history, &[1, full_token_history.len() as i64])?;
            chunked_prefill(
                &prompt,
                &embedding_weight,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                Some(&embedding_weight_t),
                generation_stream,
            )?
        } else {
            // Text-only prefill of the delta on top of the existing caches.
            let prompt = MxArray::from_uint32(&delta_tokens, &[1, delta_tokens.len() as i64])?;
            chunked_prefill(
                &prompt,
                &embedding_weight,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                Some(&embedding_weight_t),
                generation_stream,
            )?
        };
        // caches now reflect the prefilled history
        self.flat_mtp_caches_desynced = false;
        // The flat full-attention caches now cover the full history (rebuild
        // branch) or were already authoritative (delta branch). We do NOT
        // clear `paged_full_attn_caches_dirty` here: the clear is co-located
        // with the `cached_token_history` commit at the end-of-turn success
        // boundary (`save_cache_state_after_delta` below), so that ANY
        // mid-turn error — prefill OR decode — leaves the flag dirty. That
        // way the next paged→dense turn still performs the protective one-time
        // full rebuild over the authoritative history rather than trusting
        // half-advanced flat caches paired with a stale `cached_token_history`.
        let _seq_len = full_token_history.len() as i64;
        profiler.end_prefill();

        // Save snapshot for save_cache_state_direct (prior history + delta).
        let save_tokens = full_token_history.clone();

        // Decode setup
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");
        let mut decode_stream = tokenizer_for_decode.inner().decode_stream(true);
        let mut streamed_text_len: usize = 0;

        let mut token_history: Vec<u32> = full_token_history;
        last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
        let mut y = sample(&last_logits, p.sampling_config)?;
        MxArray::async_eval_arrays(&[&y]);

        let starts_in_thinking = thinking.enabled;
        let mut last_is_reasoning = starts_in_thinking;
        let mut reasoning_tracker = engine::ReasoningTracker::from_setup(&thinking, think_end_id);

        // Pure-Rust ("eager") dense MTP gate for the FLAT streaming delta path.
        // Delta is text-only (no `has_images`) by construction; paged sessions
        // returned earlier. Continuations have a live cache prefix so the
        // committed-history builds from decode tokens with NO prompt seed
        // (mirrors the non-stream delta path's `None, None, 0`).
        let eager_mtp = p.enable_mtp && self.has_mtp_weights() && self.paged_adapter.is_none();

        // Whether the final committed token reached the physical KV/GDN cache;
        // written by the decode driver so the save below drops it when it was
        // never forwarded (unforwarded stop token).
        let mut last_in_cache = true;

        if eager_mtp {
            self.run_flat_stream_eager_mtp(
                y,
                &mut token_history,
                &mut generated_tokens,
                &mut finish_reason,
                &mut reasoning_tracker,
                &mut profiler,
                &mut first_token_instant,
                &mut streamed_text_len,
                &mut last_is_reasoning,
                &mut decode_stream,
                &tokenizer_for_decode,
                cb,
                cancelled,
                embedding_weight,
                embedding_weight_t,
                &p,
                eos_id,
                p.max_new_tokens,
                generation_stream,
                None,
                None,
                0,
                &mut last_in_cache,
            )?;
        } else {
            profiler.set_label("chat_stream_delta_rust");

            let mut ops = mtp_decode::DecodeOps {
                forward: |ids: &MxArray, emb: &MxArray| -> Result<(MxArray, bool)> {
                    let logits = forward_inner(
                        ids,
                        emb,
                        &mut self.layers,
                        &mut self.caches,
                        &self.final_norm,
                        &self.lm_head,
                        Some(&embedding_weight_t),
                    )?;
                    Ok((logits, true))
                },
                eval_step: |token: &MxArray, logits: &MxArray, _budget_forced: bool| {
                    MxArray::async_eval_arrays(&[token, logits]);
                },
            };
            mtp_decode::decode_loop!(
                ops: ops,
                y: y,
                embedding_weight: embedding_weight,
                params: p,
                reasoning_tracker: reasoning_tracker,
                profiler: profiler,
                max_new_tokens: p.max_new_tokens,
                eos_id: eos_id,
                generated_tokens: generated_tokens,
                token_history: token_history,
                finish_reason: finish_reason,
                last_in_cache: last_in_cache,
                first_token_instant: first_token_instant,
                report_perf: p.report_performance,
                generation_stream: generation_stream,
                streaming: {
                    callback: cb,
                    cancelled: cancelled,
                    decode_stream: decode_stream,
                    tokenizer: tokenizer_for_decode,
                    streamed_text_len: streamed_text_len,
                    last_is_reasoning: last_is_reasoning
                }
            );
        }

        // Save cache state unconditionally — even on cancellation, the
        // partial generated_tokens must be appended so the session stays
        // consistent for the next turn. Delta continuations preserve
        // `cached_image_key` so the next turn's cache-prefix verify
        // still sees the prior prefill's image state.
        engine::save_cache_state_after_delta(
            p.reuse_cache,
            &generated_tokens,
            &finish_reason,
            /* drop_last_always */ !last_in_cache,
            &save_tokens,
            &mut self.cached_token_history,
            &mut self.cached_image_key,
            &mut self.cached_rope_deltas,
            &mut self.caches,
        );
        // Clear the paged-dirty flag ATOMICALLY with the
        // `cached_token_history` commit above: a successful turn updates the
        // committed history and the flat caches together, so the one-time
        // protective rebuild is no longer needed until a later paged-core turn
        // re-dirties it. Placing the clear here (not after prefill) guarantees
        // any mid-turn `?`-error leaves the flag dirty.
        self.paged_full_attn_caches_dirty = false;

        // Decode the full reply text and emit the final done chunk.
        let text = tokenizer_for_decode
            .decode_sync(&generated_tokens, true)
            .unwrap_or_else(|e| {
                warn!("Failed to decode generated tokens: {}", e);
                String::new()
            });

        if text.len() > streamed_text_len {
            let residual = text[streamed_text_len..].to_string();
            // Suppress residual when it is reasoning text and
            // include_reasoning == false.
            if p.include_reasoning || !last_is_reasoning {
                cb.call(
                    Ok(ChatStreamChunk {
                        text: residual,
                        done: false,
                        finish_reason: None,
                        tool_calls: None,
                        thinking: None,
                        thinking_enabled: None,
                        num_tokens: None,
                        prompt_tokens: None,
                        reasoning_tokens: None,
                        raw_text: None,
                        public_raw_text: None,
                        text_authoritative: None,
                        cached_tokens: None,
                        performance: None,
                        is_reasoning: Some(last_is_reasoning),
                    }),
                    ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
        }

        let num_tokens = generated_tokens.len() as u32;
        let prompt_token_count = delta_tokens.len() as u32;

        let (clean_text, tool_calls, thinking) = engine::parse_thinking_and_tools(
            &text,
            &generated_tokens,
            starts_in_thinking,
            think_end_id,
            think_end_str.as_deref(),
            p.include_reasoning,
        );

        let finish_reason = if tool_calls.iter().any(|tc| tc.status == "ok") {
            "tool_calls".to_string()
        } else {
            finish_reason
        };

        let perf_metrics = compute_performance_metrics(
            generation_start,
            first_token_instant,
            delta_tokens.len(),
            generated_tokens.len(),
        )
        .map(|mut m| {
            profiler.fill_mtp_acceptance(&mut m);
            m
        });

        cb.call(
            Ok(ChatStreamChunk {
                text: clean_text,
                done: true,
                finish_reason: Some(finish_reason),
                tool_calls: Some(tool_calls),
                thinking,
                thinking_enabled: Some(
                    crate::engine::resolve_enable_thinking(&config).unwrap_or(true),
                ),
                num_tokens: Some(num_tokens),
                prompt_tokens: Some(prompt_token_count),
                reasoning_tokens: Some(reasoning_tracker.reasoning_token_count()),
                raw_text: Some(engine::raw_text_with_reasoning_suppressed(
                    &text,
                    &generated_tokens,
                    starts_in_thinking,
                    think_end_id,
                    think_end_str.as_deref(),
                    p.include_reasoning,
                )),
                public_raw_text: Some(engine::raw_text_with_reasoning_suppressed(
                    &text,
                    &generated_tokens,
                    starts_in_thinking,
                    think_end_id,
                    think_end_str.as_deref(),
                    false,
                )),
                text_authoritative: Some(true),
                // Delta path reuses the full prior history by construction
                // — report `prior_cached_len` (captured before the
                // `self.cached_token_history` extend above) as the
                // authoritative cached-prefix length.
                cached_tokens: Some(prior_cached_len),
                performance: perf_metrics,
                is_reasoning: None,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );

        Ok(())
    }

    /// Whole-turn core for fresh STREAMING turns reached through the
    /// engine's `vision_turn` (image-bearing) and `mtp_turn`
    /// (MTP-enabled) probes. The engine already rendered the prompt
    /// (`tokens`) and extracted the raw image payloads (`images`);
    /// everything from the MTP-on-paged dispatch onward runs the
    /// whole-turn pipeline.
    fn chat_stream_sync_inner(
        &mut self,
        tokens: Vec<u32>,
        images: &[Vec<u8>],
        config: ChatConfig,
        eos_token_id: u32,
        cb: &StreamSender<'_>,
        cancelled: &AtomicBool,
        thinking: ThinkingSetup,
    ) -> Result<()> {
        let reuse_cache = config.reuse_cache.unwrap_or(true);
        let report_perf = config.report_performance.unwrap_or(false);

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?
            .clone();

        let has_images = !images.is_empty();

        let think_end_id = tokenizer.think_end_id();
        let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());
        let tokenizer_for_decode = tokenizer.clone();

        let mut p = engine::extract_chat_params(&config);
        p.extra_eos_ids = self.gen_defaults.eos_token_ids.clone();

        let generation_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_instant: Option<std::time::Instant> = None;

        // All image turns route to the paged-vision streaming core, which runs
        // plain autoregressive decode regardless of MTP (the core has no
        // draft/verify; MTP weights are ignored). This precedes the text-only
        // MTP-on-paged gate below so an image+MTP stream still reaches the
        // paged-vision core rather than the text dense fallback.
        if has_images && self.paged_adapter.is_some() {
            return self.vision_paged_turn_stream_core(
                tokens,
                images,
                tokenizer_for_decode,
                eos_token_id,
                p,
                report_perf,
                cb,
                cancelled,
                thinking,
            );
        }

        // Text-only paged dispatch. MTP-on-paged streams fall through to the
        // dense (flat) streaming path; non-MTP paged streams take the paged
        // streaming core.
        let mtp_takes_dense_path =
            p.enable_mtp && self.has_mtp_weights() && self.paged_adapter.is_some();
        if mtp_takes_dense_path
            && let Some(ref mut adapter) = self.paged_adapter
            && let Err(e) = adapter.release_request()
        {
            tracing::warn!(
                target: "mlx_core::qwen3_5::paged",
                "MTP-on-paged dispatch (stream-start): release_request failed (ignored): {e}",
            );
        }
        if self.paged_adapter.is_some() && !mtp_takes_dense_path {
            return self.paged_turn_stream_core(
                tokens,
                tokenizer_for_decode,
                eos_token_id,
                p,
                report_perf,
                cb,
                cancelled,
                thinking,
            );
        }

        // The dense (flat) streaming fallback is text-only. A dense image turn
        // requires the block-paged backend; reaching here with images means the
        // model was loaded without a paged adapter (use_block_paged_cache=false,
        // non-Metal build, or a sym8 checkpoint).
        if has_images {
            return Err(Error::from_reason(
                "qwen3.5 dense image turns require the block-paged KV backend; the model was \
                 loaded without a paged adapter (use_block_paged_cache=false, non-Metal build, \
                 or sym8 checkpoint)",
            ));
        }

        let embedding_weight = self.embedding.get_weight();

        // Text-only from here: the `has_images` early-return above is the only
        // image path. These bindings preserve the shared cache-reuse / decode
        // plumbing (`has_images` is always false on this branch).
        let (expanded_tokens, current_image_cache_key) = (tokens.clone(), 0u64);

        // Cache reuse
        let cached_prefix_len = verify_cache_prefix_direct(
            reuse_cache,
            has_images,
            &tokens,
            &expanded_tokens,
            current_image_cache_key,
            &self.cached_token_history,
            &self.cached_image_key,
            self.caches.is_some(),
        );

        // Same paged→dense-MTP stale-flat-cache hazard
        // as the streaming delta path, but for the stream-START dense
        // fallback. A prior paged-core turn wrote full-attention K/V into the
        // paged adapter pool, leaving the flat `self.caches` full-attention
        // slots empty/stale (only GDN linear state was imported). A prefix
        // hit from `verify_cache_prefix_direct` (matched against
        // `cached_token_history`) would then decode from an incomplete flat
        // prefix. When the flat caches are dirty, drop any prefix reuse so the
        // branch below does a full reset + re-prefill (cached_prefix_len = 0),
        // rebuilding the flat full-attention caches over the whole prompt.
        //
        // Reachability note: the flag is set ONLY by the two paged cores, and
        // a non-MTP paged turn returns earlier at the
        // `self.paged_adapter.is_some() && !mtp_takes_dense_path` branch above.
        // So whenever control reaches here the flag can be true only on the
        // paged+MTP (`mtp_takes_dense_path`) fallback; a non-paged dense start
        // (`paged_adapter.is_none()`) never sets it → this is a no-op and the
        // common path stays byte-identical. The ungated read is therefore
        // equivalent to gating on `mtp_takes_dense_path`.
        //
        // The flag is cleared at the END-OF-TURN success boundary, co-located
        // with the `cached_token_history` commit (`save_cache_state_direct`
        // below), NOT here and NOT right after prefill. This makes the clear
        // atomic with the history commit: ANY mid-turn `?`-error — prefill OR
        // decode — aborts the turn with the flat caches still un-rebuilt and
        // `cached_token_history` still holding the prior paged turn's tokens,
        // so the flag stays dirty and the NEXT paged→dense turn performs the
        // protective one-time full rebuild instead of decoding from an
        // incomplete flat prefix. Mirrors the reviewed delta path.
        let cached_prefix_len =
            if self.paged_full_attn_caches_dirty || self.flat_mtp_caches_desynced {
                0
            } else {
                cached_prefix_len
            };

        let prefill_tokens = if cached_prefix_len > 0 {
            if has_images {
                info!(
                    "VLM cache reuse: {} cached tokens, {} new tokens to prefill",
                    cached_prefix_len,
                    expanded_tokens.len() - cached_prefix_len
                );
                expanded_tokens[cached_prefix_len..].to_vec()
            } else {
                info!(
                    "Cache reuse: {} cached tokens, {} new tokens to prefill",
                    cached_prefix_len,
                    tokens.len() - cached_prefix_len
                );
                tokens[cached_prefix_len..].to_vec()
            }
        } else {
            if let Some(ref mut caches) = self.caches {
                for cache in caches.iter_mut() {
                    cache.reset();
                }
            }
            let new_caches = (0..self.config.num_layers as usize)
                .map(|i| {
                    if self.config.is_linear_layer(i) {
                        Qwen3_5LayerCache::new_linear()
                    } else {
                        Qwen3_5LayerCache::new_full_attention()
                    }
                })
                .collect();
            self.caches = Some(new_caches);
            tokens.clone()
        };

        // Zero-delta guard. See the matching `vision_mtp_whole_turn_core` comment for
        // the design rationale — rewinding a GDN recurrent cache by one
        // token is not possible, so the only safe response to an exact-
        // match prompt is a full reset + re-prefill.
        let (prefill_tokens, cached_prefix_len) = if prefill_tokens.is_empty() {
            info!("Zero-delta cache hit: resetting caches for full re-prefill");
            if let Some(ref mut caches) = self.caches {
                for cache in caches.iter_mut() {
                    cache.reset();
                }
            }
            let new_caches = (0..self.config.num_layers as usize)
                .map(|i| {
                    if self.config.is_linear_layer(i) {
                        Qwen3_5LayerCache::new_linear()
                    } else {
                        Qwen3_5LayerCache::new_full_attention()
                    }
                })
                .collect();
            self.caches = Some(new_caches);
            let tokens = if has_images {
                expanded_tokens.clone()
            } else {
                tokens.clone()
            };
            (tokens, 0)
        } else {
            (prefill_tokens, cached_prefix_len)
        };

        let eos_id = eos_token_id;
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");
        let mut decode_stream = tokenizer_for_decode.inner().decode_stream(true);
        let mut streamed_text_len: usize = 0;

        let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
        let generation_stream = Stream::new(DeviceType::Gpu);
        let model_size_bytes = self.config.estimate_memory_bytes() as usize;
        let _wired_ctx =
            crate::stream::WiredLimitContext::new(model_size_bytes, vec![generation_stream]);

        let mut profiler = crate::decode_profiler::DecodeProfiler::new("chat_stream", "qwen3_5");
        profiler.set_prompt_tokens(prefill_tokens.len() as u32);
        profiler.snapshot_memory_before();

        // Pure-Rust ("eager") dense MTP gate for the FLAT streaming path.
        // Same preconditions as `chat_with_caches_inner`: per-request /
        // per-checkpoint enablement, no live paged adapter (paged streams
        // returned earlier), text-only.
        let eager_mtp =
            p.enable_mtp && self.has_mtp_weights() && self.paged_adapter.is_none() && !has_images;
        // The committed-history v2 seed needs the prompt tail's hiddens, which
        // only the hidden-emitting prefill produces. Skip on cache-reuse turns
        // (the captured hidden would cover the SUFFIX, not the full prompt) —
        // committed-history still runs (it builds from decode tokens).
        let want_prompt_hidden =
            eager_mtp && !mtp_decode::mtp_no_prompt_prefill() && cached_prefix_len == 0;
        let mtp_prompt_history = mtp_decode::mtp_prompt_history_selection(prefill_tokens.len());
        let mut prompt_hidden: Option<MxArray> = None;

        // Text prefill
        profiler.begin_prefill();
        let (mut last_logits, _seq_len) = {
            let prompt = MxArray::from_uint32(&prefill_tokens, &[1, prefill_tokens.len() as i64])?;
            let last_logits = if want_prompt_hidden {
                let (logits, ph) = chunked_prefill_with_hidden(
                    &prompt,
                    &embedding_weight,
                    &mut self.layers,
                    &mut self.caches,
                    &self.final_norm,
                    &self.lm_head,
                    Some(&embedding_weight_t),
                    generation_stream,
                    Some(mtp_prompt_history.keep_tokens),
                )?;
                prompt_hidden = Some(ph);
                logits
            } else {
                chunked_prefill(
                    &prompt,
                    &embedding_weight,
                    &mut self.layers,
                    &mut self.caches,
                    &self.final_norm,
                    &self.lm_head,
                    Some(&embedding_weight_t),
                    generation_stream,
                )?
            };

            (last_logits, tokens.len() as i64)
        };
        profiler.end_prefill();
        // caches now reflect the prefilled history
        self.flat_mtp_caches_desynced = false;

        // On a paged→dense-MTP transition the dirty gate
        // forced `cached_prefix_len = 0`, so the prefill above was a full reset
        // + full re-prefill and the flat full-attention caches now cover the
        // entire prompt. We do NOT clear `paged_full_attn_caches_dirty` here:
        // the clear is co-located with the `cached_token_history` commit at the
        // end-of-turn success boundary (`save_cache_state_direct` below), so it
        // is atomic with the history write. That way ANY mid-turn `?`-error —
        // prefill OR decode — leaves the flag dirty and the next paged→dense
        // turn still performs the protective one-time full rebuild rather than
        // trusting half-advanced flat caches against a stale committed history.
        // (No-op on the common non-paged path, where the flag is never set.)

        let mut token_history: Vec<u32> = tokens.clone();
        last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
        let mut y = sample(&last_logits, p.sampling_config)?;
        MxArray::async_eval_arrays(&[&y]);

        let starts_in_thinking = thinking.enabled;
        let mut last_is_reasoning = starts_in_thinking;
        let mut reasoning_tracker = engine::ReasoningTracker::from_setup(&thinking, think_end_id);

        // Pair the captured prompt hidden with the exact prompt tail whose
        // hiddens it holds (mirrors the non-stream caller at 1708-1717).
        let prompt_hidden_ids: Option<Vec<u32>> = prompt_hidden.as_ref().map(|_| {
            let start = mtp_prompt_history
                .hidden_start_token_index()
                .min(prefill_tokens.len());
            prefill_tokens[start..].to_vec()
        });
        let prompt_hidden_position_base = prompt_hidden
            .as_ref()
            .map(|_| mtp_prompt_history.position_base)
            .unwrap_or(0);

        // Whether the final committed token reached the physical KV/GDN cache;
        // written by the decode driver so the save below drops it when it was
        // never forwarded (unforwarded stop token).
        let mut last_in_cache = true;

        if eager_mtp {
            self.run_flat_stream_eager_mtp(
                y,
                &mut token_history,
                &mut generated_tokens,
                &mut finish_reason,
                &mut reasoning_tracker,
                &mut profiler,
                &mut first_token_instant,
                &mut streamed_text_len,
                &mut last_is_reasoning,
                &mut decode_stream,
                &tokenizer_for_decode,
                cb,
                cancelled,
                embedding_weight,
                embedding_weight_t,
                &p,
                eos_id,
                p.max_new_tokens,
                generation_stream,
                prompt_hidden,
                prompt_hidden_ids,
                prompt_hidden_position_base,
                &mut last_in_cache,
            )?;
        } else {
            profiler.set_label("chat_stream_rust");

            let mut ops = mtp_decode::DecodeOps {
                forward: |ids: &MxArray, emb: &MxArray| -> Result<(MxArray, bool)> {
                    let logits = forward_inner(
                        ids,
                        emb,
                        &mut self.layers,
                        &mut self.caches,
                        &self.final_norm,
                        &self.lm_head,
                        Some(&embedding_weight_t),
                    )?;
                    Ok((logits, true))
                },
                eval_step: |token: &MxArray, logits: &MxArray, _budget_forced: bool| {
                    MxArray::async_eval_arrays(&[token, logits]);
                },
            };
            mtp_decode::decode_loop!(
                ops: ops,
                y: y,
                embedding_weight: embedding_weight,
                params: p,
                reasoning_tracker: reasoning_tracker,
                profiler: profiler,
                max_new_tokens: p.max_new_tokens,
                eos_id: eos_id,
                generated_tokens: generated_tokens,
                token_history: token_history,
                finish_reason: finish_reason,
                last_in_cache: last_in_cache,
                first_token_instant: first_token_instant,
                report_perf: p.report_performance,
                generation_stream: generation_stream,
                streaming: {
                    callback: cb,
                    cancelled: cancelled,
                    decode_stream: decode_stream,
                    tokenizer: tokenizer_for_decode,
                    streamed_text_len: streamed_text_len,
                    last_is_reasoning: last_is_reasoning
                }
            );
        }

        // Save cache state
        save_cache_state_direct(
            p.reuse_cache,
            has_images,
            &generated_tokens,
            &finish_reason,
            /* drop_last_always */ !last_in_cache,
            &tokens,
            Some(&expanded_tokens),
            current_image_cache_key,
            &mut self.cached_token_history,
            &mut self.cached_image_key,
            &mut self.cached_rope_deltas,
            &mut self.caches,
        );
        self.cached_paged_image_token_positions.clear();
        // Clear the paged-dirty flag ATOMICALLY with the
        // `cached_token_history` commit above: a successful turn updates the
        // committed history and the rebuilt flat caches together, so the
        // one-time protective rebuild is no longer needed until a later
        // paged-core turn re-dirties it. Placing the clear here (not after
        // prefill) guarantees any mid-turn `?`-error — prefill OR decode —
        // leaves the flag dirty so the next paged→dense turn rebuilds.
        self.paged_full_attn_caches_dirty = false;

        let text = tokenizer_for_decode
            .decode_sync(&generated_tokens, true)
            .unwrap_or_else(|e| {
                warn!("Failed to decode generated tokens: {}", e);
                String::new()
            });

        // Flush residual bytes
        if text.len() > streamed_text_len {
            let residual = text[streamed_text_len..].to_string();
            // Suppress residual when it is reasoning text and
            // include_reasoning == false.
            if p.include_reasoning || !last_is_reasoning {
                cb.call(
                    Ok(ChatStreamChunk {
                        text: residual,
                        done: false,
                        finish_reason: None,
                        tool_calls: None,
                        thinking: None,
                        thinking_enabled: None,
                        num_tokens: None,
                        prompt_tokens: None,
                        reasoning_tokens: None,
                        raw_text: None,
                        public_raw_text: None,
                        text_authoritative: None,
                        cached_tokens: None,
                        performance: None,
                        is_reasoning: Some(last_is_reasoning),
                    }),
                    ThreadsafeFunctionCallMode::NonBlocking,
                );
            }
        }

        let num_tokens = generated_tokens.len() as u32;
        let prompt_token_count = if has_images {
            expanded_tokens.len() as u32
        } else {
            tokens.len() as u32
        };

        let (clean_text, tool_calls, thinking) = engine::parse_thinking_and_tools(
            &text,
            &generated_tokens,
            starts_in_thinking,
            think_end_id,
            think_end_str.as_deref(),
            p.include_reasoning,
        );

        let finish_reason = if tool_calls.iter().any(|tc| tc.status == "ok") {
            "tool_calls".to_string()
        } else {
            finish_reason
        };

        let perf_metrics = compute_performance_metrics(
            generation_start,
            first_token_instant,
            prefill_tokens.len(),
            generated_tokens.len(),
        )
        .map(|mut m| {
            profiler.fill_mtp_acceptance(&mut m);
            m
        });

        // Send final done chunk
        cb.call(
            Ok(ChatStreamChunk {
                text: clean_text,
                done: true,
                finish_reason: Some(finish_reason),
                tool_calls: Some(tool_calls),
                thinking,
                thinking_enabled: Some(
                    crate::engine::resolve_enable_thinking(&config).unwrap_or(true),
                ),
                num_tokens: Some(num_tokens),
                prompt_tokens: Some(prompt_token_count),
                reasoning_tokens: Some(reasoning_tracker.reasoning_token_count()),
                raw_text: Some(engine::raw_text_with_reasoning_suppressed(
                    &text,
                    &generated_tokens,
                    starts_in_thinking,
                    think_end_id,
                    think_end_str.as_deref(),
                    p.include_reasoning,
                )),
                public_raw_text: Some(engine::raw_text_with_reasoning_suppressed(
                    &text,
                    &generated_tokens,
                    starts_in_thinking,
                    think_end_id,
                    think_end_str.as_deref(),
                    false,
                )),
                text_authoritative: Some(true),
                // Start path: report the matched prefix length from
                // `verify_cache_prefix_direct`. Zero on a miss, full
                // cached length on an exact-append hit.
                cached_tokens: Some(cached_prefix_len as u32),
                performance: perf_metrics,
                is_reasoning: None,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );

        Ok(())
    }

    /// Generate text from prompt tokens (synchronous, runs on model thread).
    pub(crate) fn generate_sync(
        &mut self,
        prompt_tokens: MxArray,
        config: Qwen3_5GenerationConfig,
    ) -> Result<Qwen3_5GenerationResult> {
        let tokenizer = self.tokenizer.clone();

        // Init caches
        self.init_caches_sync()?;

        let embedding_weight = self.embedding.get_weight();
        let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
        let generation_stream = Stream::new(DeviceType::Gpu);

        // Prefill
        let prompt = prompt_tokens.reshape(&[1, -1])?;
        let logits = {
            let _stream_ctx = StreamContext::new(generation_stream);
            forward_inner(
                &prompt,
                &embedding_weight,
                &mut self.layers,
                &mut self.caches,
                &self.final_norm,
                &self.lm_head,
                Some(&embedding_weight_t),
            )?
        };

        let seq_len = logits.shape_at(1)?;
        let last_logits = logits.slice_axis(1, seq_len - 1, seq_len)?;
        let last_logits = last_logits.squeeze(Some(&[1]))?;

        // Request value wins; otherwise fall back to the checkpoint's
        // generation_config.json default; otherwise the sampler's builtin.
        // When the request omits temperature, a `do_sample:false` in
        // generation_config.json forces greedy decoding (temperature 0),
        // overriding any gen-config temperature (HuggingFace transformers
        // semantics) — `effective_temperature()` folds that rule in.
        // This raw `generate` surface exposes only the four SamplingConfig
        // fields (no repetition/presence/frequency penalty), so a
        // generation_config repetition_penalty is honored on the ChatSession
        // path but intentionally not here. ChatSession is the full-parity surface.
        let sampling_config = Some(SamplingConfig {
            temperature: config
                .temperature
                .or(self.gen_defaults.effective_temperature()),
            top_k: config.top_k.or(self.gen_defaults.top_k),
            top_p: config.top_p.or(self.gen_defaults.top_p),
            min_p: config.min_p.or(self.gen_defaults.min_p),
        });

        let eos_id = self.config.eos_token_id as u32;
        // Extra stop ids from generation_config.json (e.g. a second EOS).
        let extra_eos_ids = self.gen_defaults.eos_token_ids.clone();
        let is_eos = |t: u32| t == eos_id || extra_eos_ids.contains(&t);
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut y = sample(&last_logits, sampling_config)?;

        for _step in 0..config.max_new_tokens {
            y.eval();
            let token_id = y.item_at_int32(0)? as u32;
            generated_tokens.push(token_id);

            if is_eos(token_id) {
                break;
            }

            let next_ids = y.reshape(&[1, 1])?;
            let logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                forward_inner(
                    &next_ids,
                    &embedding_weight,
                    &mut self.layers,
                    &mut self.caches,
                    &self.final_norm,
                    &self.lm_head,
                    Some(&embedding_weight_t),
                )?
            };

            let logits = logits.squeeze(Some(&[1]))?;
            y = sample(&logits, sampling_config)?;
            MxArray::async_eval_arrays(&[&y]);

            if (_step + 1) % 256 == 0 {
                crate::array::synchronize_and_clear_cache();
            }
        }

        let stopped_on_eos = generated_tokens.last().is_some_and(|&t| is_eos(t));
        self.reset_caches_sync()?;

        let finish_reason = if stopped_on_eos { "stop" } else { "length" };

        let text = if let Some(ref tok) = tokenizer {
            tok.decode_sync(&generated_tokens, true).unwrap_or_default()
        } else {
            String::new()
        };

        Ok(Qwen3_5GenerationResult {
            tokens: generated_tokens.clone(),
            text,
            num_tokens: generated_tokens.len() as u32,
            finish_reason: finish_reason.to_string(),
        })
    }

    /// Static FP8 activation-amax calibration prefill (runs on the model
    /// thread). Inherent body of [`Qwen35Cmd::CalibratePrefillRaw`].
    ///
    /// For each raw text: tokenize WITHOUT the chat template (`encode_sync` with
    /// `add_special_tokens = false`, so no `<|im_start|>`/`<|im_end|>` control
    /// tokens and no BOS), truncate to `calib_seq` tokens, then run PREFILL ONLY
    /// (no generation) so every mxfp8 attention/GDN projection's activation tap
    /// fires exactly once over realistic raw-text activations — the NVIDIA
    /// modelopt `MaxCalibrator` is defined over raw-text prefill, NOT
    /// chat-templated prompts plus a decode step. Caches are re-initialized per
    /// row so each is an independent turn-0 position-0 prefill.
    ///
    /// This method SELF-ARMS the model thread's thread-local
    /// [`ActivationAmaxCollector`] flag for the prefill's duration (RAII
    /// `CalibrationArmGuard`, disarmed on every exit path), so the tap records
    /// each projection's running `max|activation|` during the forward. The NAPI
    /// caller drains+persists afterwards; this method never touches
    /// `config.json`. Returns the number of rows actually prefilled (rows that
    /// tokenized to nothing after truncation are skipped).
    pub(crate) fn calibrate_prefill_raw_sync(
        &mut self,
        texts: Vec<String>,
        calib_seq: u32,
    ) -> Result<u32> {
        let tokenizer = self.tokenizer.clone().ok_or_else(|| {
            Error::from_reason("calibration prefill requires a tokenizer, but none is loaded")
        })?;
        let cap = calib_seq.max(1) as usize;
        let mut rows_prefilled: u32 = 0;

        // Arm THIS (model) thread's calibration flag for the whole prefill loop.
        // The tap in `QuantizedLinear::forward` runs synchronously on this same
        // thread, so it observes the armed flag and records raw `max|x|`; any
        // other loaded model on its own thread stays unaffected. The RAII guard
        // disarms on EVERY exit path — normal return, a `?` error, or a panic —
        // so a later inference command on this thread never sees a stray armed
        // flag. The NAPI caller drains + persists the collected amax afterwards.
        let _calib_guard = crate::calibration::activation_amax::CalibrationArmGuard::arm();

        for text in &texts {
            // RAW tokenize — no chat template, no special/control tokens. This
            // is the crux of the modelopt-parity fix: repeated chat control
            // tokens must not dominate a tensor's activation amax.
            let mut tokens = tokenizer.encode_sync(text, Some(false))?;
            tokens.truncate(cap);
            if tokens.is_empty() {
                continue;
            }

            // Fresh turn-0 caches per row (prefill asserts `self.caches` is set,
            // and a stale cache would append rows into one growing context).
            self.init_caches_sync()?;
            let generation_stream = Stream::new(DeviceType::Gpu);
            // PREFILL ONLY — trips every mxfp8 attn/GDN tap once. No generated
            // token: a decode step would fold a synthetic argmax token's
            // activations into the amax.
            let logits = self.prefill(&tokens, generation_stream)?;
            // Force the row's forward to complete (the taps already `.item()`
            // each projection input, but eval bounds the lazy graph and makes
            // per-row memory reclamation deterministic).
            logits.eval();
            self.reset_caches_sync()?;
            rows_prefilled += 1;

            // Bound resident scratch across many rows (large models × 1024 rows).
            crate::array::synchronize_and_clear_cache();
        }

        Ok(rows_prefilled)
    }

    // ========== Training methods (run on model thread) ==========

    /// Initialize training state with optimizer and configuration.
    /// Inherent body of [`TrainBackend::init_training_sync`].
    fn init_training_sync_impl(
        &mut self,
        config: crate::grpo::engine::GRPOEngineConfig,
        _model_type: crate::training_model::ModelType,
    ) -> Result<()> {
        if self.training_state.is_some() {
            return Err(napi::Error::from_reason(
                "Training state already initialized. A single model thread can host only one active training run.",
            ));
        }
        let optimizer = if config.optimizer_type.as_deref().unwrap_or("adamw") == "adamw" {
            Some(crate::optimizers::AdamW::new(
                config.learning_rate,
                config.adamw_beta1,
                config.adamw_beta2,
                config.adamw_eps,
                config.weight_decay,
                Some(true), // bias correction
            ))
        } else {
            None
        };

        self.training_state = Some(crate::training_state::ModelThreadTrainingState::new(
            config.learning_rate.unwrap_or(1e-6),
            config.gradient_accumulation_steps.unwrap_or(1),
            config.gradient_clip_norm,
            config.gradient_clip_value,
            config.max_nan_gradients.unwrap_or(100),
            config.emergency_save_threshold.unwrap_or(5),
            config.verbose_nan_detection.unwrap_or(false),
            config.gradient_checkpointing.unwrap_or(true),
            optimizer,
        ));
        info!("Training state initialized on model thread (Qwen3.5 Dense)");
        Ok(())
    }

    /// Inherent body of [`TrainBackend::save_optimizer_state_sync`].
    fn save_optimizer_state_sync_impl(&self, path: String) -> Result<()> {
        let ts = self.training_state.as_ref().ok_or_else(|| {
            napi::Error::from_reason("Training state not initialized. Call InitTraining first.")
        })?;
        ts.save_optimizer_state_sync(&path)
    }

    /// Inherent body of [`TrainBackend::load_optimizer_state_sync`].
    fn load_optimizer_state_sync_impl(&mut self, path: String) -> Result<()> {
        let ts = self.training_state.as_mut().ok_or_else(|| {
            napi::Error::from_reason("Training state not initialized. Call InitTraining first.")
        })?;
        ts.load_optimizer_state_sync(&path)
    }

    /// Generate completions for training.
    ///
    /// Tokenizes prompts using Jinja2 chat template, generates completions,
    /// caches MxArray results in training_state for the subsequent training step,
    /// and returns plain data across the thread boundary.
    /// Inherent body of [`TrainBackend::generate_for_training_thread_sync`].
    fn generate_for_training_thread_sync_impl(
        &mut self,
        prompts: Vec<Vec<ChatMessage>>,
        group_size: usize,
        gen_config: crate::models::qwen3::GenerationConfig,
        enable_thinking: Option<bool>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Result<crate::training_model::GenerationPlainData> {
        use crate::array::heavy_cleanup;

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("Tokenizer not available."))?
            .clone();

        let num_prompts = prompts.len();
        let total_completions = num_prompts * group_size;

        let mut completion_texts = Vec::with_capacity(total_completions);
        let mut prompt_texts = Vec::with_capacity(total_completions);
        let mut completion_tokens_plain = Vec::with_capacity(total_completions);
        let mut completion_logprobs_plain = Vec::with_capacity(total_completions);
        let mut token_counts = Vec::with_capacity(total_completions);
        let mut finish_reasons = Vec::with_capacity(total_completions);

        // Cache MxArrays for the training step (prompt-major layout)
        let mut cached_prompt_tokens: Vec<MxArray> = Vec::with_capacity(num_prompts);
        let mut cached_completion_tokens: Vec<MxArray> = Vec::with_capacity(total_completions);
        let mut cached_completion_logprobs: Vec<MxArray> = Vec::with_capacity(total_completions);

        for prompt_messages in prompts.iter() {
            // Tokenize the prompt using Jinja2 chat template (supports tools + thinking)
            let prompt_token_ids = tokenizer.apply_chat_template_sync(
                prompt_messages,
                Some(true),
                tools.as_deref(),
                enable_thinking,
            )?;

            let prompt_array =
                MxArray::from_uint32(&prompt_token_ids, &[1, prompt_token_ids.len() as i64])?;
            let prompt_array_1d = prompt_array.squeeze(Some(&[0]))?;
            let prompt_text = tokenizer.decode_sync(&prompt_token_ids, true)?;

            // Generate group_size completions for this prompt
            for _g in 0..group_size {
                let result = self
                    .generate_single_for_training_sync(&prompt_array, Some(gen_config.clone()))?;

                // Extract plain data for crossing thread boundary
                let tok_ids: Vec<i32> = result
                    .tokens
                    .to_uint32()?
                    .iter()
                    .map(|&t| t as i32)
                    .collect();
                let lp_data: Vec<f32> = result.logprobs.to_float32()?.to_vec();
                let decoded = tokenizer
                    .decode_sync(&tok_ids.iter().map(|&t| t as u32).collect::<Vec<_>>(), true)?;

                completion_texts.push(decoded);
                prompt_texts.push(prompt_text.clone());
                completion_tokens_plain.push(tok_ids);
                completion_logprobs_plain.push(lp_data);
                token_counts.push(result.num_tokens as u32);
                finish_reasons.push(result.finish_reason.clone());

                // Cache MxArrays (these stay on the model thread)
                cached_completion_tokens.push(result.tokens);
                cached_completion_logprobs.push(result.logprobs);

                // Clean up between completions to prevent Metal context accumulation
                heavy_cleanup();
            }

            cached_prompt_tokens.push(prompt_array_1d);
        }

        // Store cached MxArrays in training_state (prompt-major layout)
        if let Some(ref mut ts) = self.training_state {
            ts.cached_prompt_tokens = Some(cached_prompt_tokens);
            ts.cached_completion_tokens = Some(cached_completion_tokens);
            ts.cached_completion_logprobs = Some(cached_completion_logprobs);
        }

        Ok(crate::training_model::GenerationPlainData {
            completion_texts,
            prompt_texts,
            completion_tokens: completion_tokens_plain,
            completion_logprobs: completion_logprobs_plain,
            token_counts,
            finish_reasons,
        })
    }

    /// Generate a single completion for training purposes.
    ///
    /// Uses fresh local KV caches (not the shared inference caches).
    /// Returns GenerationResult with MxArray tokens and logprobs.
    fn generate_single_for_training_sync(
        &mut self,
        input_ids: &MxArray,
        config: Option<crate::models::qwen3::GenerationConfig>,
    ) -> Result<crate::models::qwen3::GenerationResult> {
        use crate::array::synchronize_and_clear_cache;
        use crate::sampling::{
            apply_frequency_penalty, apply_presence_penalty, apply_repetition_penalty,
            check_repetition_cutoff, sample_and_logprobs,
        };

        let config = config.unwrap_or_default();
        let max_new_tokens = config.max_new_tokens.unwrap_or(2048);
        let temperature = config.temperature.unwrap_or(1.0);
        let top_k = config.top_k.unwrap_or(0);
        let top_p = config.top_p.unwrap_or(1.0);
        let min_p = config.min_p.unwrap_or(0.0);
        let repetition_penalty = config.repetition_penalty.unwrap_or(1.0);
        let repetition_context_size = config.repetition_context_size.unwrap_or(256);
        let presence_penalty = config.presence_penalty.unwrap_or(0.0);
        let presence_context_size = config.presence_context_size.unwrap_or(20);
        let frequency_penalty = config.frequency_penalty.unwrap_or(0.0);
        let frequency_context_size = config.frequency_context_size.unwrap_or(20);
        let max_consecutive_tokens = config
            .max_consecutive_tokens
            .unwrap_or(crate::sampling::DEFAULT_MAX_CONSECUTIVE_TOKENS);
        let max_ngram_repeats = config
            .max_ngram_repeats
            .unwrap_or(crate::sampling::DEFAULT_MAX_NGRAM_REPEATS);
        let ngram_size = config
            .ngram_size
            .unwrap_or(crate::sampling::DEFAULT_NGRAM_SIZE);
        let eos_token_id = config.eos_token_id.or(Some(self.config.eos_token_id));
        let return_logprobs = config.return_logprobs.unwrap_or(true);

        let embedding_weight = self.embedding.get_weight();
        let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
        let generation_stream = Stream::new(DeviceType::Gpu);

        // Use fresh caches for training (not shared inference caches)
        let mut training_caches: Option<Vec<Qwen3_5LayerCache>> = Some(
            (0..self.config.num_layers as usize)
                .map(|i| {
                    if self.config.is_linear_layer(i) {
                        Qwen3_5LayerCache::new_linear()
                    } else {
                        Qwen3_5LayerCache::new_full_attention()
                    }
                })
                .collect(),
        );

        let input_tokens = input_ids.to_uint32()?;
        let current_ids = input_ids.clone();
        // Bounded, floored capacity hint (see `generated_capacity_hint`): this
        // training-only path takes `max_new_tokens` from training config
        // (SFT/GRPO) where panics are banned. The helper prevents both the
        // negative-budget `.. as usize` wrap to `usize::MAX` (which would abort)
        // and a multi-GiB eager reservation for an absurd budget, without
        // changing behavior for valid budgets — the buffer still grows to hold
        // every generated token.
        let mut generated_tokens: Vec<u32> =
            Vec::with_capacity(engine::generated_capacity_hint(max_new_tokens));
        let mut generated_logprobs: Vec<f32> = if return_logprobs {
            Vec::with_capacity(engine::generated_capacity_hint(max_new_tokens))
        } else {
            Vec::new()
        };
        let mut finish_reason = "length";

        let sampling_config = SamplingConfig {
            temperature: Some(temperature),
            top_k: Some(top_k),
            top_p: Some(top_p),
            min_p: Some(min_p),
        };

        // PREFILL
        let mut last_logits = {
            let logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                forward_inner(
                    &current_ids,
                    &embedding_weight,
                    &mut self.layers,
                    &mut training_caches,
                    &self.final_norm,
                    &self.lm_head,
                    Some(&embedding_weight_t),
                )?
            };
            let seq_len = logits.shape_at(1)?;
            logits
                .slice_axis(1, seq_len - 1, seq_len)?
                .squeeze(Some(&[0, 1]))?
        };

        if repetition_penalty != 1.0 && !input_tokens.is_empty() {
            last_logits = apply_repetition_penalty(
                &last_logits,
                &input_tokens,
                repetition_penalty,
                Some(repetition_context_size),
            )?;
        }
        if presence_penalty != 0.0 {
            last_logits = apply_presence_penalty(
                &last_logits,
                &input_tokens,
                presence_penalty,
                Some(presence_context_size),
            )?;
        }
        if frequency_penalty != 0.0 {
            last_logits = apply_frequency_penalty(
                &last_logits,
                &input_tokens,
                frequency_penalty,
                Some(frequency_context_size),
            )?;
        }

        let (mut token, mut logprobs) = if return_logprobs {
            let (tok, lp) = sample_and_logprobs(&last_logits, Some(sampling_config))?;
            (tok, Some(lp))
        } else {
            (sample(&last_logits, Some(sampling_config))?, None)
        };

        // DECODE
        const DECODE_CLEANUP_INTERVAL: i32 = 256;
        for step in 0..max_new_tokens {
            let _stream_ctx = StreamContext::new(generation_stream);
            token.eval();
            if step > 0 && step % DECODE_CLEANUP_INTERVAL == 0 {
                synchronize_and_clear_cache();
            }
            let token_value = token.item_at_int32(0)? as u32;
            generated_tokens.push(token_value);
            if return_logprobs && let Some(ref lp) = logprobs {
                lp.eval();
                let lp_value = lp.item_at_float32(0)?;
                generated_logprobs.push(lp_value);
            }
            if let Some(eos) = eos_token_id
                && token_value == eos as u32
            {
                finish_reason = "stop";
                break;
            }
            if let Some(reason) = check_repetition_cutoff(
                &generated_tokens,
                max_consecutive_tokens,
                max_ngram_repeats,
                ngram_size,
            ) {
                finish_reason = reason;
                break;
            }
            let next_ids = MxArray::from_uint32(&[token_value], &[1, 1])?;
            let next_logits = forward_inner(
                &next_ids,
                &embedding_weight,
                &mut self.layers,
                &mut training_caches,
                &self.final_norm,
                &self.lm_head,
                Some(&embedding_weight_t),
            )?;
            let next_last_logits = next_logits.slice_axis(1, 0, 1)?.squeeze(Some(&[0, 1]))?;
            last_logits = next_last_logits;
            if repetition_penalty != 1.0 || presence_penalty != 0.0 || frequency_penalty != 0.0 {
                let context_tokens: Vec<u32> = input_tokens
                    .iter()
                    .copied()
                    .chain(generated_tokens.iter().copied())
                    .collect();
                if repetition_penalty != 1.0 {
                    last_logits = apply_repetition_penalty(
                        &last_logits,
                        &context_tokens,
                        repetition_penalty,
                        Some(repetition_context_size),
                    )?;
                }
                if presence_penalty != 0.0 {
                    last_logits = apply_presence_penalty(
                        &last_logits,
                        &context_tokens,
                        presence_penalty,
                        Some(presence_context_size),
                    )?;
                }
                if frequency_penalty != 0.0 {
                    last_logits = apply_frequency_penalty(
                        &last_logits,
                        &context_tokens,
                        frequency_penalty,
                        Some(frequency_context_size),
                    )?;
                }
            }
            let (next_tok, next_lp) = if return_logprobs {
                let (tok, lp) = sample_and_logprobs(&last_logits, Some(sampling_config))?;
                (tok, Some(lp))
            } else {
                (sample(&last_logits, Some(sampling_config))?, None)
            };
            token = next_tok;
            logprobs = next_lp;
        }

        let tokens_array =
            MxArray::from_uint32(&generated_tokens, &[generated_tokens.len() as i64])?;
        let logprobs_array = if return_logprobs {
            MxArray::from_float32(&generated_logprobs, &[generated_logprobs.len() as i64])?
        } else {
            MxArray::from_float32(&[], &[0])?
        };

        Ok(crate::models::qwen3::GenerationResult {
            text: String::new(), // Text decoding done by caller
            tokens: tokens_array,
            logprobs: logprobs_array,
            finish_reason: finish_reason.to_string(),
            num_tokens: generated_tokens.len(),
        })
    }

    /// GRPO training step: compute loss, gradients, and apply optimizer.
    ///
    /// Consumes cached MxArrays from the generation phase, computes loss and
    /// gradients via autograd, validates and clips gradients, accumulates them,
    /// and applies the optimizer step when accumulation is complete.
    /// Inherent body of [`TrainBackend::train_step_grpo_sync`].
    fn train_step_grpo_sync_impl(
        &mut self,
        rewards: Vec<f64>,
        group_size: i32,
        loss_config: crate::grpo::loss::GRPOLossConfig,
        valid_indices: Option<Vec<usize>>,
    ) -> Result<crate::training_model::TrainStepPlainMetrics> {
        use crate::array::memory::{get_active_memory, get_peak_memory, reset_peak_memory};
        use crate::array::{heavy_cleanup, synchronize_and_clear_cache};
        use crate::grpo::advantages::compute_advantages;
        use crate::grpo::autograd::compute_loss_and_gradients_autograd;
        use crate::optimizers::GradientUtils;
        use crate::training_model::ModelType;

        reset_peak_memory();

        // Get cached generation results from training_state
        let ts = self.training_state.as_ref().ok_or_else(|| {
            napi::Error::from_reason("Training state not initialized. Call InitTraining first.")
        })?;

        let prompt_tokens = ts.cached_prompt_tokens.as_ref().ok_or_else(|| {
            napi::Error::from_reason("No cached prompt tokens. Call GenerateForTraining first.")
        })?;
        let completion_tokens = ts.cached_completion_tokens.as_ref().ok_or_else(|| {
            napi::Error::from_reason("No cached completion tokens. Call GenerateForTraining first.")
        })?;
        let completion_logprobs = ts.cached_completion_logprobs.as_ref().ok_or_else(|| {
            napi::Error::from_reason(
                "No cached completion logprobs. Call GenerateForTraining first.",
            )
        })?;

        let use_checkpointing = ts.gradient_checkpointing;
        let gradient_clip_value = ts.gradient_clip_value;
        let gradient_clip_norm = ts.gradient_clip_norm;
        let verbose_nan = ts.verbose_nan_detection;
        let learning_rate = ts.learning_rate;
        let max_nan_gradients = ts.max_nan_gradients;
        let emergency_save_threshold = ts.emergency_save_threshold;

        // Get model parameters
        let params = self.get_parameters_sync()?;
        let model_type = ModelType::Qwen35Dense(self.config.clone());

        // Build completion/logprob refs, optionally filtering by valid_indices from
        // the engine's degenerate-completion filter. prompt_refs always has one
        // entry per prompt — the autograd function expands them to one per
        // completion via repeat_n(group_size), so the `group_size` passed here
        // must be the effective group size after filtering (the engine computes
        // effective_group_size = valid_indices.len() / num_prompts).
        let prompt_refs: Vec<&MxArray> = prompt_tokens.iter().collect();
        let (completion_refs, logprob_refs): (Vec<&MxArray>, Vec<&MxArray>) =
            if let Some(ref indices) = valid_indices {
                let n = completion_tokens.len();
                for &i in indices {
                    if i >= n {
                        return Err(napi::Error::from_reason(format!(
                            "valid_indices contains out-of-range index {} (completion count = {})",
                            i, n
                        )));
                    }
                }
                let c: Vec<&MxArray> = indices.iter().map(|&i| &completion_tokens[i]).collect();
                let l: Vec<&MxArray> = indices.iter().map(|&i| &completion_logprobs[i]).collect();
                (c, l)
            } else {
                (
                    completion_tokens.iter().collect(),
                    completion_logprobs.iter().collect(),
                )
            };

        let (loss_value, gradients) = compute_loss_and_gradients_autograd(
            &model_type,
            &params,
            &prompt_refs,
            &completion_refs,
            &logprob_refs,
            &rewards,
            group_size,
            loss_config,
            use_checkpointing,
        )?;

        // Check for NaN/Inf loss
        if loss_value.is_nan() || loss_value.is_infinite() {
            warn!("Skipping step due to invalid loss: {}", loss_value);
            synchronize_and_clear_cache();
            // Skipped steps must still advance the authoritative step counter
            // (H1) and drop the cached generation so the next cycle starts
            // clean.
            let ts = self.training_state.as_mut().unwrap();
            ts.clear_generation_cache();
            ts.step += 1;
            let new_step = ts.step;
            let nan_count = ts.nan_gradient_count;
            return Ok(crate::training_model::TrainStepPlainMetrics {
                loss: loss_value,
                gradients_applied: false,
                mean_advantage: 0.0,
                std_advantage: 0.0,
                nan_gradient_count: nan_count,
                peak_memory_mb: get_peak_memory() / 1e6,
                active_memory_mb: get_active_memory() / 1e6,
                total_tokens: 0,
                step: new_step,
            });
        }

        // Validate ALL gradients — skip entire step if ANY has NaN/Inf
        for (name, grad) in gradients.iter() {
            grad.eval();
            let has_invalid = grad.has_nan_or_inf()?;
            if has_invalid {
                if verbose_nan {
                    let data = grad.to_float32()?;
                    let invalid_count = data
                        .iter()
                        .filter(|v| v.is_nan() || v.is_infinite())
                        .count();
                    warn!(
                        "Gradient '{}' contains {} invalid values - SKIPPING STEP",
                        name, invalid_count
                    );
                } else {
                    warn!("Gradient '{}' contains NaN/Inf - SKIPPING STEP", name);
                }

                let ts = self.training_state.as_mut().unwrap();
                ts.nan_gradient_count += 1;
                ts.consecutive_nan_count += 1;

                if ts.nan_gradient_count >= max_nan_gradients as u64 {
                    return Err(napi::Error::from_reason(format!(
                        "Training stopped: exceeded max NaN gradient count ({}/{})",
                        ts.nan_gradient_count, max_nan_gradients
                    )));
                }

                if ts.consecutive_nan_count >= emergency_save_threshold as u32 {
                    warn!(
                        "Emergency save triggered: {} consecutive NaN gradients",
                        ts.consecutive_nan_count
                    );
                }

                // Advance the authoritative step counter (H1) and clear the
                // cached generation data so the next cycle starts clean.
                ts.clear_generation_cache();
                ts.step += 1;
                let new_step = ts.step;
                let nan_count = ts.nan_gradient_count;
                synchronize_and_clear_cache();
                return Ok(crate::training_model::TrainStepPlainMetrics {
                    loss: loss_value,
                    gradients_applied: false,
                    mean_advantage: 0.0,
                    std_advantage: 0.0,
                    nan_gradient_count: nan_count,
                    peak_memory_mb: get_peak_memory() / 1e6,
                    active_memory_mb: get_active_memory() / 1e6,
                    total_tokens: 0,
                    step: new_step,
                });
            }
        }

        // Element-wise gradient clipping
        let grad_clip_val = gradient_clip_value.unwrap_or(1.0);
        let mut clamped_gradients: HashMap<String, MxArray> = HashMap::new();
        for (name, grad) in gradients.iter() {
            let clamped = grad.clip(Some(-grad_clip_val), Some(grad_clip_val))?;
            clamped.eval();
            clamped_gradients.insert(name.clone(), clamped);
        }

        // Gradient norm clipping
        let clipped_gradients = if let Some(max_norm) = gradient_clip_norm {
            let grad_refs: HashMap<String, &MxArray> = clamped_gradients
                .iter()
                .map(|(k, v)| (k.clone(), v))
                .collect();
            GradientUtils::clip_grad_norm(grad_refs, max_norm)?
        } else {
            clamped_gradients
        };

        // Accumulate gradients
        let ts = self.training_state.as_mut().unwrap();
        // Reset consecutive NaN count on successful gradient computation
        ts.consecutive_nan_count = 0;

        Self::accumulate_gradients_inner(ts, clipped_gradients)?;
        ts.micro_step += 1;

        let grad_acc_steps = ts.grad_accumulation_steps;
        let gradients_applied = if ts.micro_step >= grad_acc_steps {
            let grads = ts
                .accumulated_gradients
                .take()
                .ok_or_else(|| napi::Error::from_reason("No accumulated gradients"))?;

            // Apply optimizer step
            if let Some(ref mut optimizer) = ts.optimizer {
                // AdamW path
                let mut param_names_vec: Vec<String> = Vec::new();
                let mut param_refs: Vec<&MxArray> = Vec::new();
                let mut grad_refs: Vec<&MxArray> = Vec::new();

                // Scale gradients if using accumulation
                let scaled_grads: HashMap<String, MxArray>;
                let grads_to_use = if grad_acc_steps > 1 {
                    let scale = 1.0 / grad_acc_steps as f32;
                    let scale_arr = MxArray::from_float32(&[scale], &[])?;
                    scaled_grads = grads
                        .iter()
                        .map(|(name, grad)| Ok((name.clone(), grad.mul(&scale_arr)?)))
                        .collect::<Result<HashMap<_, _>>>()?;
                    &scaled_grads
                } else {
                    &grads
                };

                for (name, grad) in grads_to_use {
                    if let Some(param) = params.get(name) {
                        param_names_vec.push(name.clone());
                        param_refs.push(param);
                        grad_refs.push(grad);
                    }
                }

                let updated = optimizer.update_batch(
                    param_names_vec.clone(),
                    param_refs.clone(),
                    grad_refs,
                )?;

                // `update_batch` above has already committed the optimizer's step
                // counter and moment tensors. If this fallible delta build (or the
                // atomic apply below) errors, the optimizer is left one step ahead
                // of the still-unchanged model params (all deltas are built before
                // any are applied). Accepted over panicking; such `?` failures are
                // fatal device errors that abort the run.
                // Create deltas: delta = param - updated (so param - 1.0 * delta = updated)
                let delta_map: HashMap<String, MxArray> = param_names_vec
                    .iter()
                    .enumerate()
                    .map(|(i, name)| {
                        let delta = param_refs[i].sub(&updated[i])?;
                        Ok((name.clone(), delta))
                    })
                    .collect::<Result<HashMap<_, _>>>()?;

                let delta_refs: HashMap<String, &MxArray> =
                    delta_map.iter().map(|(k, v)| (k.clone(), v)).collect();
                self.apply_gradients_inner(delta_refs, 1.0, &params)?;

                tracing::debug!(
                    "Applied AdamW update (step={})",
                    self.training_state.as_ref().unwrap().step
                );
            } else {
                // SGD path
                let lr = learning_rate / grad_acc_steps as f64;
                let grads_refs: HashMap<String, &MxArray> =
                    grads.iter().map(|(k, v)| (k.clone(), v)).collect();
                self.apply_gradients_inner(grads_refs, lr, &params)?;
                tracing::debug!("Applied SGD gradients with lr: {}", lr);
            }

            let ts = self.training_state.as_mut().unwrap();
            ts.accumulated_gradients = None;
            ts.micro_step = 0;
            ts.step += 1;
            true
        } else {
            ts.step += 1;
            false
        };

        // Compute advantage statistics
        let rewards_f32: Vec<f32> = rewards.iter().map(|&r| r as f32).collect();
        let rewards_array = MxArray::from_float32(&rewards_f32, &[rewards.len() as i64])?;
        let advantages = compute_advantages(&rewards_array, group_size, "group".to_string())?;
        let adv_data = advantages.to_float32()?;
        let mean_advantage =
            adv_data.iter().map(|&a| a as f64).sum::<f64>() / adv_data.len().max(1) as f64;
        let std_advantage = {
            let variance = adv_data
                .iter()
                .map(|&a| {
                    let diff = a as f64 - mean_advantage;
                    diff * diff
                })
                .sum::<f64>()
                / adv_data.len().max(1) as f64;
            variance.sqrt()
        };

        // Count tokens BEFORE clearing the cache — otherwise total_tokens is
        // always zero on the success path.
        let ts = self.training_state.as_ref().unwrap();
        let total_tokens: i32 = if let Some(ref ct) = ts.cached_completion_tokens {
            ct.iter()
                .filter_map(|t| t.shape_at(0).ok())
                .map(|n| n as i32)
                .sum()
        } else {
            0
        };

        // Clear cached generation data
        if let Some(ref mut ts) = self.training_state {
            ts.clear_generation_cache();
        }

        // CRITICAL: heavy_cleanup after autograd to clear compiled graph cache
        heavy_cleanup();

        let ts = self.training_state.as_ref().unwrap();
        Ok(crate::training_model::TrainStepPlainMetrics {
            loss: loss_value,
            gradients_applied,
            mean_advantage,
            std_advantage,
            nan_gradient_count: ts.nan_gradient_count,
            peak_memory_mb: get_peak_memory() / 1e6,
            active_memory_mb: get_active_memory() / 1e6,
            total_tokens,
            step: ts.step,
        })
    }

    /// SFT training step: compute loss, gradients, and apply optimizer.
    ///
    /// Receives plain data (Vec<i32> + shape) from the SFT engine, reconstructs
    /// MxArrays on the model thread, computes SFT loss + gradients, validates,
    /// clips, accumulates, and applies optimizer step when accumulation is complete.
    /// Inherent body of [`TrainBackend::train_step_sft_sync`].
    fn train_step_sft_sync_impl(
        &mut self,
        input_ids: Vec<i32>,
        input_shape: Vec<i64>,
        labels: Vec<i32>,
        labels_shape: Vec<i64>,
        config: crate::sft::engine::SftEngineConfig,
    ) -> Result<crate::training_model::TrainStepPlainMetrics> {
        use crate::array::memory::{get_active_memory, get_peak_memory, reset_peak_memory};
        use crate::array::{heavy_cleanup, synchronize_and_clear_cache};
        use crate::optimizers::GradientUtils;

        reset_peak_memory();

        // Ensure training state is initialized
        let ts = self.training_state.as_ref().ok_or_else(|| {
            napi::Error::from_reason("Training state not initialized. Call InitTraining first.")
        })?;
        let _ = ts;

        // Reconstruct MxArrays from plain data
        let input_ids_arr = MxArray::from_int32(&input_ids, &input_shape)?;
        let labels_arr = MxArray::from_int32(&labels, &labels_shape)?;

        // Get model parameters
        let params = self.get_parameters_sync()?;
        let model_type = crate::training_model::ModelType::Qwen35Dense(self.config.clone());

        // Build loss config from SftEngineConfig
        let loss_config = crate::sft::SftLossConfig {
            ignore_index: Some(-100),
            label_smoothing: config.label_smoothing,
        };

        let use_checkpointing = config.gradient_checkpointing.unwrap_or(true);
        let verbose_nan = config.verbose_nan_detection.unwrap_or(false);
        let max_nan_gradients = config.max_nan_gradients.unwrap_or(100);
        let emergency_save_threshold = config.emergency_save_threshold.unwrap_or(5);

        // Compute loss and gradients
        let (loss_value, gradients) = crate::sft::autograd::compute_sft_loss_and_gradients(
            &model_type,
            &params,
            &input_ids_arr,
            &labels_arr,
            loss_config,
            use_checkpointing,
        )?;

        // Check for NaN/Inf loss
        if loss_value.is_nan() || loss_value.is_infinite() {
            warn!("SFT: Skipping step due to invalid loss: {}", loss_value);
            synchronize_and_clear_cache();
            let ts = self.training_state.as_mut().unwrap();
            ts.nan_gradient_count += 1;
            ts.consecutive_nan_count += 1;

            if ts.nan_gradient_count >= max_nan_gradients as u64 {
                return Err(napi::Error::from_reason(format!(
                    "Training stopped: exceeded max NaN gradient count ({}/{})",
                    ts.nan_gradient_count, max_nan_gradients
                )));
            }

            if ts.consecutive_nan_count >= emergency_save_threshold as u32 {
                warn!(
                    "Emergency save triggered: {} consecutive NaN losses",
                    ts.consecutive_nan_count
                );
            }

            return Ok(crate::training_model::TrainStepPlainMetrics {
                loss: 0.0,
                gradients_applied: false,
                mean_advantage: 0.0,
                std_advantage: 0.0,
                nan_gradient_count: ts.nan_gradient_count,
                peak_memory_mb: get_peak_memory() / 1e6,
                active_memory_mb: get_active_memory() / 1e6,
                total_tokens: 0,
                step: ts.step,
            });
        }

        // Validate ALL gradients — skip entire step if ANY has NaN/Inf
        for (name, grad) in gradients.iter() {
            grad.eval();
            let has_invalid = grad.has_nan_or_inf()?;
            if has_invalid {
                if verbose_nan {
                    let data = grad.to_float32()?;
                    let invalid_count = data
                        .iter()
                        .filter(|v| v.is_nan() || v.is_infinite())
                        .count();
                    warn!(
                        "SFT: Gradient '{}' contains {} invalid values - SKIPPING STEP",
                        name, invalid_count
                    );
                } else {
                    warn!("SFT: Gradient '{}' contains NaN/Inf - SKIPPING STEP", name);
                }

                let ts = self.training_state.as_mut().unwrap();
                ts.nan_gradient_count += 1;
                ts.consecutive_nan_count += 1;

                if ts.nan_gradient_count >= max_nan_gradients as u64 {
                    return Err(napi::Error::from_reason(format!(
                        "Training stopped: exceeded max NaN gradient count ({}/{})",
                        ts.nan_gradient_count, max_nan_gradients
                    )));
                }

                if ts.consecutive_nan_count >= emergency_save_threshold as u32 {
                    warn!(
                        "Emergency save triggered: {} consecutive NaN gradients",
                        ts.consecutive_nan_count
                    );
                }

                synchronize_and_clear_cache();
                return Ok(crate::training_model::TrainStepPlainMetrics {
                    loss: loss_value,
                    gradients_applied: false,
                    mean_advantage: 0.0,
                    std_advantage: 0.0,
                    nan_gradient_count: ts.nan_gradient_count,
                    peak_memory_mb: get_peak_memory() / 1e6,
                    active_memory_mb: get_active_memory() / 1e6,
                    total_tokens: 0,
                    step: ts.step,
                });
            }
        }

        // Element-wise gradient clipping (if configured)
        let clipped_gradients = if let Some(clip_val) = config.gradient_clip_value {
            let mut clamped: HashMap<String, MxArray> = HashMap::new();
            for (name, grad) in gradients.iter() {
                let c = grad.clip(Some(-clip_val), Some(clip_val))?;
                c.eval();
                clamped.insert(name.clone(), c);
            }
            clamped
        } else {
            gradients.clone()
        };

        // Gradient norm clipping (if configured)
        let final_gradients = if let Some(clip_norm) = config.gradient_clip_norm {
            let grad_refs: HashMap<String, &MxArray> = clipped_gradients
                .iter()
                .map(|(k, v)| (k.clone(), v))
                .collect();
            GradientUtils::clip_grad_norm(grad_refs, clip_norm)?
        } else {
            clipped_gradients
        };

        // Accumulate gradients
        let ts = self.training_state.as_mut().unwrap();
        ts.consecutive_nan_count = 0;

        Self::accumulate_gradients_inner(ts, final_gradients)?;
        ts.micro_step += 1;

        let grad_acc_steps = config.gradient_accumulation_steps.unwrap_or(1);
        let learning_rate = config.learning_rate.unwrap_or(2e-5);
        let weight_decay = config.weight_decay.unwrap_or(0.01);

        let gradients_applied = if ts.micro_step >= grad_acc_steps {
            let grads = ts
                .accumulated_gradients
                .take()
                .ok_or_else(|| napi::Error::from_reason("No accumulated gradients"))?;

            if let Some(ref mut optimizer) = ts.optimizer {
                let mut param_names_vec: Vec<String> = Vec::new();
                let mut param_refs: Vec<&MxArray> = Vec::new();
                let mut grad_refs: Vec<&MxArray> = Vec::new();

                let scaled_grads: HashMap<String, MxArray>;
                let grads_to_use = if grad_acc_steps > 1 {
                    let scale = 1.0 / grad_acc_steps as f32;
                    let scale_arr = MxArray::from_float32(&[scale], &[])?;
                    scaled_grads = grads
                        .iter()
                        .map(|(name, grad)| Ok((name.clone(), grad.mul(&scale_arr)?)))
                        .collect::<Result<HashMap<_, _>>>()?;
                    &scaled_grads
                } else {
                    &grads
                };

                for (name, grad) in grads_to_use {
                    if let Some(param) = params.get(name) {
                        param_names_vec.push(name.clone());
                        param_refs.push(param);
                        grad_refs.push(grad);
                    }
                }

                let updated = optimizer.update_batch(
                    param_names_vec.clone(),
                    param_refs.clone(),
                    grad_refs,
                )?;

                // `update_batch` above has already committed the optimizer's step
                // counter and moment tensors. If this fallible delta build (or the
                // atomic apply below) errors, the optimizer is left one step ahead
                // of the still-unchanged model params (all deltas are built before
                // any are applied). Accepted over panicking; such `?` failures are
                // fatal device errors that abort the run.
                let delta_map: HashMap<String, MxArray> = param_names_vec
                    .iter()
                    .enumerate()
                    .map(|(i, name)| {
                        let delta = param_refs[i].sub(&updated[i])?;
                        Ok((name.clone(), delta))
                    })
                    .collect::<Result<HashMap<_, _>>>()?;

                let delta_refs: HashMap<String, &MxArray> =
                    delta_map.iter().map(|(k, v)| (k.clone(), v)).collect();
                self.apply_gradients_inner(delta_refs, 1.0, &params)?;

                tracing::debug!(
                    "SFT: Applied AdamW update (step={})",
                    self.training_state.as_ref().unwrap().step
                );
            } else {
                let lr = learning_rate / grad_acc_steps as f64;

                let grads_with_decay = if weight_decay > 0.0 {
                    grads
                        .into_iter()
                        .map(|(name, grad)| {
                            if let Some(param) = params.get(&name) {
                                if let Ok(decay_term) = param.mul_scalar(weight_decay)
                                    && let Ok(new_grad) = grad.add(&decay_term)
                                {
                                    return (name, new_grad);
                                }
                                (name, grad)
                            } else {
                                (name, grad)
                            }
                        })
                        .collect::<HashMap<_, _>>()
                } else {
                    grads
                };

                let grads_refs: HashMap<String, &MxArray> = grads_with_decay
                    .iter()
                    .map(|(k, v)| (k.clone(), v))
                    .collect();
                self.apply_gradients_inner(grads_refs, lr, &params)?;
                tracing::debug!("SFT: Applied SGD gradients with lr: {}", lr);
            }

            let ts = self.training_state.as_mut().unwrap();
            ts.accumulated_gradients = None;
            ts.micro_step = 0;
            ts.step += 1;
            true
        } else {
            ts.step += 1;
            false
        };

        // Count valid tokens from the labels
        let total_tokens = {
            let ignore_val = MxArray::scalar_int(-100)?;
            let valid_mask = labels_arr.not_equal(&ignore_val)?;
            let count = valid_mask.sum(None, Some(false))?;
            count.eval();
            count.item_at_int32(0).unwrap_or(0)
        };

        // CRITICAL: heavy_cleanup after autograd to clear compiled graph cache
        heavy_cleanup();

        let ts = self.training_state.as_ref().unwrap();
        Ok(crate::training_model::TrainStepPlainMetrics {
            loss: loss_value,
            gradients_applied,
            mean_advantage: 0.0,
            std_advantage: 0.0,
            nan_gradient_count: ts.nan_gradient_count,
            peak_memory_mb: get_peak_memory() / 1e6,
            active_memory_mb: get_active_memory() / 1e6,
            total_tokens,
            step: ts.step,
        })
    }

    /// Accumulate gradients into training state.
    fn accumulate_gradients_inner(
        ts: &mut crate::training_state::ModelThreadTrainingState,
        new_grads: HashMap<String, MxArray>,
    ) -> Result<()> {
        match &mut ts.accumulated_gradients {
            Some(acc) => {
                for (name, grad) in new_grads {
                    grad.eval();
                    if grad.has_nan_or_inf()? {
                        warn!(
                            "Skipping gradient accumulation for '{}' due to NaN/Inf",
                            name
                        );
                        continue;
                    }
                    if let Some(existing) = acc.get_mut(&name) {
                        let summed = existing.add(&grad)?;
                        summed.eval();
                        *existing = summed;
                    } else {
                        acc.insert(name, grad);
                    }
                }
            }
            None => {
                let mut evaluated_grads = HashMap::with_capacity(new_grads.len());
                for (name, grad) in new_grads {
                    grad.eval();
                    if grad.has_nan_or_inf()? {
                        warn!("Skipping initial gradient for '{}' due to NaN/Inf", name);
                        continue;
                    }
                    evaluated_grads.insert(name, grad);
                }
                ts.accumulated_gradients = Some(evaluated_grads);
            }
        }
        Ok(())
    }

    /// Apply gradients to model weights (SGD or AdamW delta application).
    ///
    /// Direct field access on Qwen35Inner — no locks needed.
    fn apply_gradients_inner(
        &mut self,
        gradients: HashMap<String, &MxArray>,
        learning_rate: f64,
        current_params: &HashMap<String, MxArray>,
    ) -> Result<()> {
        use super::decoder_layer::AttentionType;

        let updated_params =
            crate::training_model::compute_sgd_updates(&gradients, learning_rate, current_params)?;

        // Apply updated parameters directly to model fields
        for (name, updated_param) in updated_params.iter() {
            if name == "lm_head.weight" {
                if let Some(ref mut lm) = self.lm_head {
                    lm.set_weight(updated_param, "lm_head")?;
                }
            } else if name == "final_norm.weight" {
                self.final_norm.set_weight(updated_param)?;
            } else if name == "embedding.weight" {
                self.embedding.set_weight(updated_param)?;
            } else if name.starts_with("layers.") {
                let parts: Vec<&str> = name.split('.').collect();
                if parts.len() >= 3
                    && let Ok(layer_idx) = parts[1].parse::<usize>()
                    && layer_idx < self.layers.len()
                {
                    let layer = &mut self.layers[layer_idx];
                    if name.contains(".linear_attn.") {
                        if let AttentionType::Linear(ref mut gdn) = layer.attn {
                            if name.ends_with(".in_proj_qkvz.weight") {
                                gdn.set_in_proj_qkvz_weight(updated_param)?;
                            } else if name.ends_with(".in_proj_ba.weight") {
                                gdn.set_in_proj_ba_weight(updated_param)?;
                            } else if name.ends_with(".conv1d.weight") {
                                gdn.set_conv1d_weight(updated_param)?;
                            } else if name.ends_with(".norm.weight") {
                                gdn.set_norm_weight(updated_param)?;
                            } else if name.ends_with(".out_proj.weight") {
                                gdn.set_out_proj_weight(updated_param)?;
                            } else if name.ends_with(".dt_bias") {
                                gdn.set_dt_bias(updated_param);
                            } else if name.ends_with(".a_log") {
                                gdn.set_a_log(updated_param)?;
                            }
                        }
                    } else if name.contains(".self_attn.") {
                        if let AttentionType::Full(ref mut attn) = layer.attn {
                            if name.ends_with(".q_proj.weight") {
                                attn.set_q_proj_weight(updated_param)?;
                            } else if name.ends_with(".k_proj.weight") {
                                attn.set_k_proj_weight(updated_param)?;
                            } else if name.ends_with(".v_proj.weight") {
                                attn.set_v_proj_weight(updated_param)?;
                            } else if name.ends_with(".o_proj.weight") {
                                attn.set_o_proj_weight(updated_param)?;
                            } else if name.ends_with(".q_norm.weight") {
                                attn.set_q_norm_weight(updated_param)?;
                            } else if name.ends_with(".k_norm.weight") {
                                attn.set_k_norm_weight(updated_param)?;
                            }
                        }
                    } else if name.contains(".mlp.") {
                        if name.ends_with(".gate_proj.weight") {
                            layer.mlp.set_gate_proj_weight(updated_param)?;
                        } else if name.ends_with(".up_proj.weight") {
                            layer.mlp.set_up_proj_weight(updated_param)?;
                        } else if name.ends_with(".down_proj.weight") {
                            layer.mlp.set_down_proj_weight(updated_param)?;
                        }
                    } else if name.ends_with(".input_layernorm.weight") {
                        layer.set_input_layernorm_weight(updated_param)?;
                    } else if name.ends_with(".post_attention_layernorm.weight") {
                        layer.set_post_attention_layernorm_weight(updated_param)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Extract all trainable parameters from the model.
    /// Direct field access — no locks needed on model thread.
    fn get_parameters_sync(&self) -> Result<HashMap<String, MxArray>> {
        use super::decoder_layer::AttentionType;

        let mut params = HashMap::new();

        // Embedding
        params.insert("embedding.weight".to_string(), self.embedding.get_weight());

        // Transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{}", i);

            match &layer.attn {
                AttentionType::Linear(gdn) => {
                    params.insert(
                        format!("{}.linear_attn.in_proj_qkvz.weight", prefix),
                        gdn.get_in_proj_qkvz_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.in_proj_ba.weight", prefix),
                        gdn.get_in_proj_ba_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.conv1d.weight", prefix),
                        gdn.get_conv1d_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.norm.weight", prefix),
                        gdn.get_norm_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.out_proj.weight", prefix),
                        gdn.get_out_proj_weight(),
                    );
                    params.insert(format!("{}.linear_attn.dt_bias", prefix), gdn.get_dt_bias());
                    params.insert(format!("{}.linear_attn.a_log", prefix), gdn.get_a_log());
                }
                AttentionType::Full(attn) => {
                    params.insert(
                        format!("{}.self_attn.q_proj.weight", prefix),
                        attn.get_q_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.k_proj.weight", prefix),
                        attn.get_k_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.v_proj.weight", prefix),
                        attn.get_v_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.o_proj.weight", prefix),
                        attn.get_o_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.q_norm.weight", prefix),
                        attn.get_q_norm_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.k_norm.weight", prefix),
                        attn.get_k_norm_weight(),
                    );
                }
            }

            // MLP (all layers have dense MLP)
            params.insert(
                format!("{}.mlp.gate_proj.weight", prefix),
                layer.mlp.get_gate_proj_weight(),
            );
            params.insert(
                format!("{}.mlp.up_proj.weight", prefix),
                layer.mlp.get_up_proj_weight(),
            );
            params.insert(
                format!("{}.mlp.down_proj.weight", prefix),
                layer.mlp.get_down_proj_weight(),
            );

            // Layer norms
            params.insert(
                format!("{}.input_layernorm.weight", prefix),
                layer.get_input_layernorm_weight(),
            );
            params.insert(
                format!("{}.post_attention_layernorm.weight", prefix),
                layer.get_post_attention_layernorm_weight(),
            );
        }

        // Final norm
        params.insert(
            "final_norm.weight".to_string(),
            self.final_norm.get_weight(),
        );

        // LM head (only if not tied to embeddings)
        if !self.config.tie_word_embeddings
            && let Some(ref lm_head) = self.lm_head
        {
            params.insert("lm_head.weight".to_string(), lm_head.get_weight());
        }

        Ok(params)
    }

    /// True when this model checkpoint includes an MTP head (module loaded by
    /// `persistence::apply_weights_inner`). The speculative decode loop gates
    /// on this together with the per-request `enable_mtp` flag — both must be
    /// true for the MTP-accelerated path to take over.
    pub(crate) fn has_mtp_weights(&self) -> bool {
        self.mtp.is_some() && self.mtp_weights_loaded
    }
}

/// Adapter giving the engine's [`ChunkSink`] the `.call()` shape the
/// `decode_loop!` macro and the engine's `run_mtp_turn` loop (and the
/// streaming cores behind the whole-turn probes) expect from a
/// `ThreadsafeFunction`-like callback.
///
/// The engine owns the channel and hands the probes a `&dyn ChunkSink`,
/// so the wrapper forwards `.call()` to [`ChunkSink::send`]; the call
/// mode is meaningless on the mpsc path and is dropped.
struct StreamSender<'a>(&'a dyn ChunkSink);

impl StreamSender<'_> {
    fn call(&self, result: napi::Result<ChatStreamChunk>, _mode: ThreadsafeFunctionCallMode) {
        self.0.send(result);
    }
}

/// Paged decode stepper for qwen3_5 dense (the paged analog of the FLAT
/// [`Qwen35Decode`]). Drives
/// [`crate::engine::decode::run_decode_loop`] through the generic
/// [`crate::engine::paged_turn::run_paged_turn`]: each `forward` runs the
/// pure-Rust eager paged step against the live post-prefill adapter pools +
/// GDN caches. Created by `<Qwen35Inner as PagedBackend>::begin_paged_decode`,
/// consumed across the whole decode loop.
pub(crate) struct Qwen35PagedDecode<'a> {
    inner: &'a mut Qwen35Inner,
}

impl DecodeStep for Qwen35PagedDecode<'_> {
    fn forward(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)> {
        // NOT on the hot path — the engine drives decode via
        // `forward_with_token` (which hands the scalar the loop already read).
        // Kept only to satisfy the trait; extract then delegate.
        let token_id = input_ids.item_at_int32(0)? as u32;
        self.forward_with_token(input_ids, token_id)
    }

    fn forward_with_token(
        &mut self,
        _input_ids: &MxArray,
        token_id: u32,
    ) -> Result<(MxArray, bool)> {
        // Pure-Rust eager paged decode step.
        //
        // PERF: `token_id` is HANDED by the engine (already read once at the
        // loop top via `y.item_at_int32`), so we do NOT re-`item_at_int32` the
        // fresh `_input_ids` reshape — that redundant second per-step eval/sync
        // measurably regressed decode. `_input_ids` is unused (kept for
        // signature parity).
        let logits = {
            let embed = self.inner.embedding.clone();
            let embedding_weight = embed.get_weight();
            let layer_kinds = super::decoder_layer::compute_layer_kinds(
                self.inner.config.num_layers as usize,
                |i| self.inner.config.is_linear_layer(i),
            );
            let caches_ref = self.inner.caches.as_mut().ok_or_else(|| {
                Error::from_reason("Qwen35PagedDecode::forward: caches dropped mid-decode")
            })?;
            let adapter = self.inner.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("Qwen35PagedDecode::forward: paged_adapter dropped mid-decode")
            })?;
            super::paged_forward::run_paged_decode_step(
                token_id,
                &embed,
                &mut self.inner.layers,
                caches_ref,
                &self.inner.final_norm,
                &self.inner.lm_head,
                &embedding_weight,
                &layer_kinds,
                adapter,
                self.inner.cached_rope_deltas.unwrap_or(0),
            )?
            .squeeze(Some(&[1]))?
        };

        // `run_paged_decode_step` returns [1, 1, vocab]; the `squeeze([1])`
        // above already collapses to [1, vocab], so `needs_squeeze = FALSE`.
        Ok((logits, false))
    }

    fn eval_step(&mut self, next_token: &MxArray, _logits: &MxArray, _budget_forced: bool) {
        // Single SYNCHRONOUS eval of `next_token`: the paged forward
        // is bandwidth-bound, so an async two-wait (bottom `async_eval` +
        // loop-top `y.eval`) would buy ZERO overlap. One `y.eval()` per sample
        // is the cheapest correct cadence.
        next_token.eval();
    }

    fn maintain_cache(&mut self, step: i32) {
        // Per-step paged cache-clear cadence.
        crate::array::maybe_clear_cache_for_paged_step(step);
    }

    // `materialize_final` — DO NOT override (default no-op). CRITICAL: dense
    // paged drops the last token UNCONDITIONALLY (see `save_paged_history`).
    // The adapter only advanced for the tokens the loop actually forwarded;
    // re-running a decode step here for the final length-exit token would
    // record a token the GDN/adapter state never advanced → recurrent-state
    // desync vs the saved drop-last history.
}

/// qwen3_5 dense paged prefix state — the effective prefix/suffix split from
/// `prepare_turn_with_max_cache_hit_tokens`, PLUS the full prompt tokens and
/// the GDN-prime flag.
///
/// `full_tokens` is needed because the engine hands `paged_prefill` ONLY the
/// suffix (`tokens[effective_cached_prefix_len..]`), but
/// `run_paged_prefill_chunk` needs the FULL prompt for the GDN pre-pass over
/// the cached prefix. `gdn_prefix_already_primed` is the dense-specific bit the
/// prime resolves (the GDN recurrent state was already populated live / from a
/// checkpoint / via replay) and `paged_prefill` threads into
/// `run_paged_prefill_chunk` so the prefill skips re-priming the GDN prefix.
pub(crate) struct Qwen35PrefixState {
    effective_cached_prefix_len: usize,
    suffix_len: usize,
    full_tokens: Vec<u32>,
    gdn_prefix_already_primed: bool,
}

impl PagedPrefix for Qwen35PrefixState {
    fn effective_cached_prefix_len(&self) -> usize {
        self.effective_cached_prefix_len
    }
    fn suffix_len(&self) -> usize {
        self.suffix_len
    }
}

impl PagedBackend for Qwen35Inner {
    type PagedDecode<'a>
        = Qwen35PagedDecode<'a>
    where
        Self: 'a;
    type PrefixState = Qwen35PrefixState;

    fn prime_prefix_state(
        &mut self,
        plan: &[u32],
        _reuse_cache: bool,
        _block_size: usize,
        _extra_keys: &[u64],
        cache_salt: u64,
    ) -> Result<Self::PrefixState> {
        // The `prepare_turn_…` + `prepare_dense_gdn_prefix_state` block that
        // opens a dense paged turn.
        self.paged_finalize_failed = false;
        let trace_enabled = inference_trace_enabled();
        let total_budget = plan.len() as u32;
        // vLLM exact-prefix cap: leave at least one prompt token to prefill so
        // the decoder always has something to consume.
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
        let seq_id: u32 = 0;
        let block_size = {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason(
                    "prime_prefix_state: paged_adapter is None — caller must check \
                     use_block_paged_cache before dispatch",
                )
            })?;
            adapter.block_size()
        };
        let carries_image_lineage = self.cached_image_key.is_some()
            && !self.cached_paged_image_token_positions.is_empty()
            && !self.cached_token_history.is_empty()
            && plan.starts_with(&self.cached_token_history);
        let image_positions = if carries_image_lineage {
            self.cached_paged_image_token_positions.as_slice()
        } else {
            &[]
        };
        let lookup_extra_keys =
            engine::build_paged_extra_keys(plan.len(), block_size, image_positions);

        // Adapter-owned warm/cold lifecycle. The [MLX_TRACE] line below
        // reads the PRE-turn live state, so probe the adapter immutably FIRST
        // (prepare_turn mutates request_tokens via continue_turn/reset). The
        // adapter re-reads the same state internally, so live_* is identical to
        // what prepare_turn decides on. reuse_cache=true literal: continuation
        // eligibility carries no reuse term (the engine's reuse_cache drives
        // finalize/save instead). Suffix blocks are allocated inside
        // prepare_turn.
        let live_ready;
        let live_prefix_match;
        let live_tokens_len;
        let mut live_mismatch = TokenPrefixMismatchTrace::default();
        {
            let adapter = self
                .paged_adapter
                .as_ref()
                .ok_or_else(|| Error::from_reason("prime_prefix_state: paged_adapter is None"))?;
            live_ready = adapter.is_live_for_continue();
            let live_tokens = adapter.request_tokens();
            live_tokens_len = live_tokens.len();
            live_prefix_match = plan.starts_with(live_tokens);
            if trace_enabled && live_ready && !live_prefix_match {
                live_mismatch = token_prefix_mismatch_trace(plan, live_tokens);
            }
        }
        let turn_plan = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason("prime_prefix_state: paged_adapter is None"))?
            .prepare_turn_per_block_with_max_cache_hit_tokens(
                seq_id,
                plan,
                total_budget,
                true,
                &lookup_extra_keys,
                cache_salt,
                false,
                max_cache_hit_tokens,
            )
            .map_err(Error::from_reason)?;
        let cached_prefix_len = turn_plan.cached_prefix_len;
        let continued_live_prefix = turn_plan.continued_live_prefix;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-dense paged_prefix_lookup prompt_tokens={} \
                 cached_prefix_tokens={} continued_live_prefix={} live_ready={} \
                 live_match={} live_tokens={} live_mismatch_at={} prompt_token={} live_token={}",
                plan.len(),
                cached_prefix_len,
                continued_live_prefix,
                live_ready,
                live_prefix_match,
                live_tokens_len,
                live_mismatch.index,
                live_mismatch.prompt_token,
                live_mismatch.cached_token
            ));
        }

        // GDN recurrent-state prime (live / checkpoint / replay). No qwen3/lfm2
        // analog — dense carries GDN recurrent state across turns.
        let gdn_prefix_preparation = self.prepare_dense_gdn_prefix_state(
            plan,
            cached_prefix_len,
            block_size,
            &lookup_extra_keys,
            cache_salt,
            continued_live_prefix,
        )?;
        let gdn_prefix_already_primed = gdn_prefix_preparation.already_primed;
        // GDN recurrent state now covers exactly the cached prefix (installed
        // live, restored from a cold sidecar, or replayed just above). Discharge
        // the adapter's auxiliary-state obligation — set when the cold tier
        // carries a `ColdSidecarPolicy` and a HOT K/V hit handed back a prefix
        // no sidecar was gated against — before the turn's first `record_tokens`.
        // A no-op for a live continuation or a sidecar-backed restore.
        if let Some(adapter) = self.paged_adapter.as_mut() {
            adapter
                .confirm_aux_prefix_primed(cached_prefix_len)
                .map_err(Error::from_reason)?;
        }
        let preserves_image_lineage = carries_image_lineage && cached_prefix_len > 0;
        // Clear the per-turn session state here (history is re-set in
        // `save_paged_history`; image key is reset because the paged path does
        // not carry it across turns). The cross-turn M-RoPE delta is carried
        // only when this turn extends the live image sequence
        // (continued_live_prefix); a cold start or a non-live prefix-cache hit
        // (text-only prefix) drops a stale image delta so the text suffix
        // rotates at the raw physical slot.
        self.cached_token_history.clear();
        if !preserves_image_lineage {
            self.cached_image_key = None;
            self.cached_paged_image_token_positions.clear();
        }
        self.cached_rope_deltas = super::paged_forward::rope_delta_for_paged_turn(
            self.cached_rope_deltas,
            preserves_image_lineage,
        );

        let suffix_len = total_budget.checked_sub(cached_prefix_len).ok_or_else(|| {
            Error::from_reason("prime_prefix_state: cached_prefix_len > total_prompt_tokens")
        })? as usize;

        Ok(Qwen35PrefixState {
            effective_cached_prefix_len: cached_prefix_len as usize,
            suffix_len,
            full_tokens: plan.to_vec(),
            gdn_prefix_already_primed,
        })
    }

    fn paged_prefill(
        &mut self,
        suffix_tokens: &[u32],
        prefix: &Self::PrefixState,
        _stream: Stream,
    ) -> Result<MxArray> {
        // The NON-hidden paged prefill. `run_paged_prefill_chunk` writes K/V
        // into the adapter pool, populates the GDN linear caches, runs the GDN
        // pre-pass over the cached prefix from `full_tokens` (skipped when
        // `gdn_prefix_already_primed`), then the full forward over the suffix,
        // folding in the last-token slice (returns `[vocab]`). The engine fires
        // the post-prefill `synchronize_and_clear_cache` AFTER this returns
        // (NOT here). The MTP `_with_hidden` variant is NOT used here — MTP
        // turns route through `paged_turn_sync_core`, not the engine.
        let layer_kinds =
            super::decoder_layer::compute_layer_kinds(self.config.num_layers as usize, |i| {
                self.config.is_linear_layer(i)
            });
        let embed = self.embedding.clone();
        let embedding_weight = embed.get_weight();
        // Cross-turn M-RoPE delta (0 unless this engine-driven text turn warm-
        // continues an image prefill); aligns the suffix keys with the
        // compressed-position image keys.
        let rope_deltas = self.cached_rope_deltas.unwrap_or(0);
        let chunk_size = self.cold_gdn_prefill_chunk_size();
        let (logits, checkpoint) = {
            let caches_ref = self
                .caches
                .as_mut()
                .ok_or_else(|| Error::from_reason("paged_prefill: caches not initialized"))?;
            let adapter = self
                .paged_adapter
                .as_mut()
                .ok_or_else(|| Error::from_reason("paged_prefill: paged_adapter dropped"))?;
            super::paged_forward::run_paged_prefill_chunk_with_size(
                &prefix.full_tokens,
                suffix_tokens,
                prefix.effective_cached_prefix_len as u32,
                prefix.gdn_prefix_already_primed,
                &embed,
                &mut self.layers,
                caches_ref,
                &self.final_norm,
                &self.lm_head,
                &embedding_weight,
                &layer_kinds,
                adapter,
                chunk_size,
                rope_deltas,
            )?
        };
        self.publish_dense_gdn_materialized_prefix_checkpoint(&prefix.full_tokens, checkpoint);
        Ok(logits)
    }

    fn begin_paged_decode(&mut self, _setup: &PagedTurnSetup<'_>) -> Result<Self::PagedDecode<'_>> {
        // Pure-Rust eager paged decode: the stepper drives
        // `run_paged_decode_step` against the live post-prefill adapter pools +
        // GDN caches. No compiled-graph seeding / lifecycle locks needed.
        Ok(Qwen35PagedDecode { inner: self })
    }

    fn finalize_paged_turn(&mut self, reuse_cache: bool) {
        // Terminal lifecycle block of a paged turn. Success: keep the request
        // live across turns when reuse is on, using PER-BLOCK extra keys (NOT
        // qwen3's empty `&[]`), so the next turn's continue builds on the
        // partial trailing block's live K/V; otherwise register full blocks
        // for reuse + release. The trait hook remains infallible, but an adapter
        // error must downgrade the session before the engine's unconditional
        // `save_paged_history` call can publish image-placeholder history.
        self.paged_finalize_failed = false;
        let mut release_pending = false;
        let mut finalize_error = match self.paged_adapter.as_mut() {
            Some(adapter) if reuse_cache => {
                let total_for_finalize = adapter.request_tokens().len();
                let block_size = adapter.block_size();
                let finalize_extra_keys = engine::build_paged_extra_keys(
                    total_for_finalize,
                    block_size,
                    &self.cached_paged_image_token_positions,
                );
                adapter
                    .finalize_turn_keep_live_per_block(&finalize_extra_keys, 0)
                    .err()
            }
            Some(adapter) => {
                let total_for_finalize = adapter.request_tokens().len();
                let block_size = adapter.block_size();
                let finalize_extra_keys = engine::build_paged_extra_keys(
                    total_for_finalize,
                    block_size,
                    &self.cached_paged_image_token_positions,
                );
                // The release is attempted below (even when registration fails),
                // but only AFTER the GDN sidecar capture — releasing resets the
                // adapter's cold-chain frontier to 0.
                release_pending = true;
                adapter
                    .register_full_blocks_for_reuse_per_block(&finalize_extra_keys, 0)
                    .err()
            }
            None => Some("paged_adapter is None during dense finalization".to_owned()),
        };
        // Persist the out-of-pool GDN recurrent state for the SAME chain the
        // adapter just captured (its `cold_captured_blocks` is now set), so a
        // later process can resume from the restored K/V prefix instead of
        // replaying GDN over it. Runs BEFORE the release below, which resets the
        // captured-chain frontier to 0. Skipped when the finalize failed: the
        // K/V chain the sidecar would anchor on was not published, so nothing
        // could ever select it.
        if finalize_error.is_none() {
            self.capture_dense_gdn_cold_sidecar(&self.cached_paged_image_token_positions);
        }
        // Always attempt the release for the non-reuse path, even when
        // registration failed (mirrors the prior `register_error.or(release)`).
        if release_pending && let Some(adapter) = self.paged_adapter.as_mut() {
            finalize_error = finalize_error.or(adapter.release_request().err());
        }
        if let Some(error) = finalize_error {
            self.downgrade_failed_paged_finalize(&error);
        }
    }

    fn abort_paged_turn(&mut self) {
        // Hybrid error-path teardown: K/V and GDN may have advanced by
        // different amounts, so a bare adapter release would leave a false-live
        // recurrent session behind. Never register / keep live; invalidate all
        // model-local continuation state without masking the original error.
        self.invalidate_dense_paged_session("generic paged turn abort");
    }

    fn paged_decode_stream(&self, _generation_stream: Stream) -> Stream {
        // Run the paged DECODE on the canonical DEFAULT stream, NOT the
        // per-turn `generation_stream`. dense's paged forward + every
        // `y.eval()` run on the MLX DEFAULT stream; running the forward on a
        // queue separate from the shared loop's top-of-iteration `y.eval()`
        // (always on the default stream) would force a cross-queue
        // completion-wait every token (~5% on bandwidth-bound decode).
        // `paged_prefill` still runs on `generation_stream`. See the
        // `PagedBackend::paged_decode_stream` doc for the full mechanism.
        Stream::default(crate::stream::DeviceType::Gpu)
    }

    fn save_paged_history(
        &mut self,
        save_tokens: &[u32],
        generated: &[u32],
        _keep_all: bool,
        reuse_cache: bool,
    ) -> Result<()> {
        // `finalize_paged_turn` is an infallible trait hook. When its adapter
        // operation failed it already invalidated the native caches and left
        // this one-turn latch so the engine's unconditional save cannot revive
        // the session from expanded image-placeholder ids.
        if std::mem::take(&mut self.paged_finalize_failed) {
            self.cached_token_history.clear();
            self.cached_image_key = None;
            self.cached_paged_image_token_positions.clear();
            self.cached_rope_deltas = None;
            self.gdn_last_history_checkpoint = None;
            self.caches = None;
            return Ok(());
        }
        // dense paged ALWAYS drops the last token, regardless of the engine's
        // `keep_all` (length-exit) signal — the paged decode loop NEVER forwards
        // the LAST sampled token (the engine's forward gate skips it AND
        // `materialize_final` is a no-op for dense), so the last `generated`
        // entry is NOT in the adapter / GDN caches and must be dropped to keep
        // the saved history aligned with the live cache state. Ordering:
        // finalize → set history (drop-last, last_token_in_cache=false) → GDN
        // checkpoint → clear image key. PRESERVE THIS EXACT ORDER — it is the
        // most delicate part for T=0 byte-equality.
        if !reuse_cache {
            self.cached_token_history.clear();
            self.cached_image_key = None;
            self.cached_paged_image_token_positions.clear();
            self.cached_rope_deltas = None;
            return Ok(());
        }
        let mut full_history = save_tokens.to_vec();
        if !generated.is_empty() {
            // last_token_in_cache == false → drop-last UNCONDITIONAL.
            let upto = generated.len().saturating_sub(1);
            full_history.extend_from_slice(&generated[..upto]);
        }
        self.cached_token_history = full_history;
        // GDN history checkpoint — must run AFTER the history is set (it
        // snapshots the live recurrent caches keyed on `cached_token_history`),
        // BEFORE clearing the image key. A checkpoint/eval failure here
        // PROPAGATES (`?`) to abort the turn: a half-snapshotted or
        // failed-eval GDN state must NOT be published as a reusable
        // warm-continue checkpoint, or the next turn reads corrupt
        // recurrent state.
        let store = self.remember_dense_gdn_history_checkpoint()?;
        if inference_trace_enabled() {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-dense gdn_history_checkpoint stored={} tokens={} \
                 eval_ms={:.1} clone_ms={:.1} token_clone_ms={:.1} update_ms={:.1} \
                 total_ms={:.1}",
                store.stored,
                self.cached_token_history.len(),
                store.eval_ms,
                store.clone_ms,
                store.token_clone_ms,
                store.update_ms,
                store.total_ms
            ));
        }
        Ok(())
    }

    fn reconcile_paged_request_tokens(
        &mut self,
        prompt_len: usize,
        generated: &[u32],
        _keep_all: bool,
    ) -> bool {
        // dense ALWAYS drops the last token (see `save_paged_history`), so the
        // to-be-saved history length is `prompt_len + (generated.len() - 1)` (or
        // `prompt_len` when nothing was generated). Roll the adapter back to that
        // length so the next turn's warm-continue gate
        // (`prompt.starts_with(request_tokens())`) is not defeated by a trailing
        // token the pipelined loop recorded at the loop top before the
        // stop-check. `_keep_all` is intentionally ignored (qwen3 signal).
        //
        // Token accounting: on BOTH length and early-stop exits the to-be-saved
        // history equals the adapter cursor (the final/terminal forward was
        // skipped), so `surplus` is 0 and this is a true no-op for dense — but
        // the rollback is kept as the defensive contract the trait mandates.
        let Some(adapter) = self.paged_adapter.as_mut() else {
            return true;
        };
        let history_len = if generated.is_empty() {
            0
        } else {
            generated.len() - 1
        };
        let target_len = prompt_len + history_len;
        let surplus = adapter.request_tokens().len().saturating_sub(target_len);
        if surplus > 0
            && let Err(e) = adapter.rollback_last_tokens(surplus as u32)
        {
            tracing::warn!(
                target: "mlx_core::qwen3_5::paged",
                "reconcile_paged_request_tokens: rollback_last_tokens({surplus}) failed \
                 (finalize releases the request; next turn cold-prefills): {e}",
            );
            return false;
        }
        true
    }
}

impl Qwen35Inner {
    /// Whole-turn dense dispatch behind the engine's multimodal and
    /// speculative handlers.
    ///
    /// Routes the four turn shapes onto the whole-turn cores:
    /// fresh sync → [`Self::vision_mtp_whole_turn_core`], delta sync →
    /// [`Self::chat_tokens_delta_sync`], fresh streaming →
    /// [`Self::chat_stream_sync_inner`], delta streaming →
    /// [`Self::chat_stream_tokens_delta_sync_inner`]. These cores own
    /// every dense-path subtlety the generic flow does not model: VLM
    /// prefill + M-RoPE deltas, the MTP gate (eager MTP, falling back to
    /// AR when ineligible), the legacy defensive paged-to-flat cache rebuild,
    /// and requires-paged image routing (including the missing-adapter
    /// diagnostic).
    ///
    /// Delta turns recover the raw delta from the engine-composed
    /// `args.tokens` (`cached_history + delta` by construction — the
    /// probes run before any state mutation).
    fn dense_whole_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        // Fold generation_config.json defaults into the config the whole-turn
        // cores re-extract params from, so VLM/MTP turns honor the same
        // sampling defaults as the generic AR path (whose `args.params`
        // already had them applied via `resolve_params`). No-op when the
        // checkpoint ships no defaults (`gen_defaults` all-None).
        let mut config = args.config.clone();
        crate::engine::apply_generation_defaults(&mut config, &self.gen_defaults);
        // The engine resolved MTP admission once, before any cache mutation.
        // These legacy whole-turn cores still extract their own `ChatParams`,
        // so project the selected decoder back into their config instead of
        // letting the raw request independently re-open the MTP gate.
        let planned_mtp = apply_qwen35_dense_planned_decoder(&mut config, args.plan.decoder);
        debug_assert!(!planned_mtp || self.has_mtp_weights());
        let thinking = args.thinking;
        match (args.sink, args.cancelled) {
            (Some(sink), Some(cancelled)) => {
                let cb = StreamSender(sink);
                if args.plan.is_delta {
                    let delta_start = self.cached_token_history.len().min(args.tokens.len());
                    let delta_tokens = args.tokens[delta_start..].to_vec();
                    self.chat_stream_tokens_delta_sync_inner(
                        delta_tokens,
                        config,
                        &cb,
                        cancelled,
                        thinking,
                    )?;
                } else {
                    self.chat_stream_sync_inner(
                        args.tokens.to_vec(),
                        args.media.images,
                        config,
                        args.eos_id,
                        &cb,
                        cancelled,
                        thinking,
                    )?;
                }
                Ok(TurnOutput::Streamed)
            }
            _ => {
                let result = if args.plan.is_delta {
                    let delta_start = self.cached_token_history.len().min(args.tokens.len());
                    let delta_tokens = args.tokens[delta_start..].to_vec();
                    self.chat_tokens_delta_sync(delta_tokens, config, thinking)?
                } else {
                    self.vision_mtp_whole_turn_core(
                        args.tokens.to_vec(),
                        args.media.images,
                        config,
                        args.eos_id,
                        thinking,
                    )?
                };
                Ok(TurnOutput::Complete(Box::new(result)))
            }
        }
    }

    /// Whole-turn block-paged dispatch behind [`ChatBackend::run_paged_turn`].
    ///
    /// Conditional router (dense differs from MoE here — `run_decode_loop` has
    /// no MTP gate, so planned MTP must use the native paged-MTP cores):
    ///   * planned MTP turns (sync or stream) use the matching eager paged-MTP
    ///     core, including text deltas over an image-derived live session;
    ///   * autoregressive turns (sync or stream) use the generic paged path via
    ///     `engine::paged_turn::run_paged_turn`, which drives the adapter
    ///     lifecycle through [`PagedBackend`] and reuses `run_decode_loop`.
    fn paged_whole_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        // The MTP cores re-derive `p` from config (`extract_chat_params`). To
        // match the engine's default `resolve_params`, fold
        // generation_config.json defaults in first so the paged-MTP path
        // honors them too (no-op when the checkpoint ships none).
        let mut config = args.config.clone();
        crate::engine::apply_generation_defaults(&mut config, &self.gen_defaults);
        let planned_mtp = apply_qwen35_dense_planned_decoder(&mut config, args.plan.decoder);
        debug_assert!(!planned_mtp || self.has_mtp_weights());
        let mut p = extract_chat_params(&config);
        p.extra_eos_ids = self.gen_defaults.eos_token_ids.clone();
        if planned_mtp {
            let report_perf = args.config.report_performance.unwrap_or(false);
            let tokenizer = args.tokenizer.clone();
            let thinking = args.thinking;
            return match (args.sink, args.cancelled) {
                (Some(sink), Some(cancelled)) => {
                    let cb = StreamSender(sink);
                    self.paged_turn_stream_core(
                        args.tokens.to_vec(),
                        tokenizer,
                        args.eos_id,
                        p,
                        report_perf,
                        &cb,
                        cancelled,
                        thinking,
                    )?;
                    Ok(TurnOutput::Streamed)
                }
                _ => {
                    let result = self.paged_turn_sync_core(
                        args.tokens.to_vec(),
                        tokenizer,
                        args.eos_id,
                        p,
                        report_perf,
                        thinking,
                    )?;
                    Ok(TurnOutput::Complete(Box::new(result)))
                }
            };
        }

        // NON-MTP (sync or stream) → the generic AR+paged engine path.
        //
        // This paged turn writes full-attention K/V into the paged adapter
        // pool, NOT the flat `self.caches`, so the flat full-attention slots no
        // longer reflect the conversation. A later streaming dense-MTP fallback
        // must rebuild the flat caches before decoding. The MTP paged cores
        // set this at their core entry; this is the set-site for the
        // generic path. See `paged_full_attn_caches_dirty`.
        // The specialized MTP/VLM paged cores perform their own hard capacity
        // preflight. The ordinary AR path delegates directly to the generic
        // paged engine, so constrain a copy of the already-resolved params here
        // and make that exact copy authoritative for prefill + decode. This is
        // the native backstop for callers that bypass the TypeScript
        // `ChatSession` preflight.
        let mut constrained_params = args.params.clone();
        self.preflight_paged_context(args.tokens.len(), &mut constrained_params)?;
        // Only dirty the flat-attention mirror after deterministic preflight
        // succeeds. A rejected request has not touched the paged adapter.
        self.paged_full_attn_caches_dirty = true;
        let mut constrained_args = WholeTurnArgs {
            tokens: args.tokens,
            tokenizer: args.tokenizer,
            eos_id: args.eos_id,
            config: args.config,
            params: &constrained_params,
            thinking: args.thinking,
            plan: args.plan,
            sink: args.sink,
            cancelled: args.cancelled,
            media: args.media,
        };
        crate::engine::paged_turn::run_paged_turn(self, &mut constrained_args)
    }
}

/// Per-turn decode stepper for the engine's generic (text-only,
/// non-paged, non-MTP) flow on Qwen3.5 dense
/// ([`ChatBackend::begin_decode`]).
///
/// Drives the pure-Rust eager `forward_inner` over the flat caches.
pub(crate) struct Qwen35Decode<'a> {
    inner: &'a mut Qwen35Inner,
    embedding_weight: MxArray,
    embedding_weight_t: MxArray,
    /// Decode-path profiler relabel (`chat_rust` and its
    /// `chat_stream[_delta]_*` streaming variants), resolved in
    /// `begin_decode` from the turn's streaming-ness.
    relabel: &'static str,
}

impl DecodeStep for Qwen35Decode<'_> {
    fn forward(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)> {
        let inner = &mut *self.inner;
        let logits = forward_inner(
            input_ids,
            &self.embedding_weight,
            &mut inner.layers,
            &mut inner.caches,
            &inner.final_norm,
            &inner.lm_head,
            Some(&self.embedding_weight_t),
        )?;
        // `true` == the eager Rust forward returns `[1, 1, vocab]`;
        // the loop squeezes axis 1.
        Ok((logits, true))
    }

    fn eval_step(&mut self, next_token: &MxArray, logits: &MxArray, _budget_forced: bool) {
        MxArray::async_eval_arrays(&[next_token, logits]);
    }

    fn profiler_relabel(&self) -> Option<&'static str> {
        Some(self.relabel)
    }
}

/// Per-turn pure-Rust ("eager") dense MTP stepper the engine-owned
/// [`crate::engine::mtp_turn::run_mtp_turn`] drives.
///
/// The 11 `MtpOps` closures of the old `chat_with_caches_inner` eager-MTP
/// block become [`MtpStepper`] methods; the per-cycle scratch the closures
/// captured in `RefCell`/`Cell` (the GDN tape, the pre-verify snapshot, the
/// stashed replay error, the desync latch, the committed-history bookkeeping)
/// becomes PLAIN struct fields — the engine calls the methods strictly
/// sequentially and non-nested, so the interior mutability the closures needed
/// to share `&mut self` is gone.
///
/// Drives BOTH the FLAT and the block-PAGED dense MTP turn, selected by
/// [`MtpStepMode`]: the flat mode runs the eager pre-norm forwards against
/// `inner.caches`; the paged mode routes the main Step-A / verify forwards
/// through the captured [`PagedKVCacheAdapter`]
/// ([`super::paged_forward::run_paged_step_with_hidden`] /
/// [`super::paged_forward::run_paged_verify_step`]) while the GDN recurrent
/// state stays FLAT in `inner.caches` Linear slots. Only four methods
/// (`forward_with_hidden`, `verify_step`, `rollback`, `rollback_unemitted`)
/// branch on the mode; every other method is mode-identical (the drafter and
/// the committed-history commit are paged-agnostic). The paged adapter is moved
/// into the stepper at [`MtpBackend::begin_mtp_decode`] and restored into
/// `inner.paged_adapter` by [`Drop`] so the post-turn paged-history save finds
/// it, on EVERY exit path of `run_mtp_turn`.
pub(crate) struct DenseMtpStepper<'a> {
    /// The model — owns layers / caches / mtp / final_norm / lm_head and the
    /// `flat_mtp_caches_desynced` latch. == the closures'
    /// `inner_cell.borrow_mut()`.
    inner: &'a mut Qwen35Inner,
    /// Drafter K/V caches. == the closures' `mtp_caches_cell`. v2
    /// committed-history mode holds the persistent committed prefix; v1
    /// cycle-history mode is reset fresh by `begin_cycle`.
    mtp_caches: Vec<Qwen3_5LayerCache>,
    /// Eager analogue of `g_mtp_committed_len`: committed tokens whose exact
    /// K/V live in `mtp_caches`. == the closures' `committed_len` cell.
    committed_len: i32,
    /// Committed-history active iff the prompt tail's hiddens start at
    /// absolute position 0. == the closures' `use_committed`.
    use_committed: bool,
    /// Chained verify-hidden reuse is only numerically stable when this turn
    /// seeded the drafter with the full prompt. Warm suffix-only prefills have
    /// no such seed and must retain Step A between cycles.
    chained_cycles_supported: bool,
    /// Pre-verify snapshot of the main caches, taken in
    /// `snapshot_main_linear`, consumed by `rollback`. == the closures'
    /// `snap_cell`.
    snap: Option<Result<Vec<super::layer_cache::Qwen3_5LayerSnapshot>>>,
    /// GDN tape recorded by `verify_step`, consumed by `rollback`. == the
    /// closures' `tape_cell`.
    tape: Vec<Option<super::gated_delta_net::GdnLayerTape>>,
    /// Error stashed by the infallible `rollback` replay, surfaced by
    /// `take_replay_error`. == the closures' `replay_err_cell`.
    replay_err: Option<Error>,
    /// Mid-cycle-stop desync latch (set by `rollback_unemitted`), reported by
    /// `into_desynced`. == the closures' `mtp_desynced` cell.
    mtp_desynced: bool,
    /// The model's embedding table. == the closures' `embedding_weight` /
    /// `emb_capture`.
    embedding_weight: MxArray,
    /// Transposed embedding for the tied-LM-head projection. ==
    /// `Some(&embedding_weight_t)` (`emb_t_ref`) in the closures.
    embedding_weight_t: MxArray,
    /// Config clone for the per-cycle drafter cache reset/fresh build. == the
    /// closures' captured `config`.
    config: Qwen3_5Config,
    /// Flat vs. paged main-forward routing. `Paged` owns the turn's
    /// [`PagedKVCacheAdapter`] (moved in at `begin_mtp_decode`, restored into
    /// `inner.paged_adapter` by [`Drop`]).
    mode: MtpStepMode,
    /// Per-layer attention/linear classification consumed by the paged
    /// forwards. Empty on the flat path (unused there).
    layer_kinds: Vec<super::decoder_layer::Qwen3_5LayerKind>,
}

/// Main-forward routing for [`DenseMtpStepper`]: `Flat` runs the eager
/// pre-norm forwards against `inner.caches`; `Paged` routes full-attention
/// K/V through the owned adapter while GDN state stays flat. The adapter is
/// boxed so the unit `Flat` variant does not pad out to the adapter's size.
enum MtpStepMode {
    Flat,
    Paged(Box<PagedKVCacheAdapter>),
}

impl Drop for DenseMtpStepper<'_> {
    fn drop(&mut self) {
        // Restore the paged adapter into the model so the post-turn
        // paged-history save (which runs AFTER `run_mtp_turn` returns) finds
        // `inner.paged_adapter == Some`. Firing in `Drop` covers EVERY exit
        // path of `run_mtp_turn` — the `Ok` tail, the `take_replay_error`
        // early return, and any mid-loop `?` propagation. Idempotent: the
        // adapter is taken out of `mode` here, so a second drop is a no-op.
        if let MtpStepMode::Paged(adapter) = std::mem::replace(&mut self.mode, MtpStepMode::Flat) {
            self.inner.paged_adapter = Some(*adapter);
        }
    }
}

impl MtpStepper for DenseMtpStepper<'_> {
    fn embedding_weight(&self) -> &MxArray {
        &self.embedding_weight
    }

    fn committed_history_active(&self) -> bool {
        self.use_committed
    }

    fn chained_cycles_supported(&self) -> bool {
        self.chained_cycles_supported
    }

    fn profiler_relabel(&self) -> Option<&'static str> {
        // The eager dense MTP path set the turn label via
        // `profiler.set_label("mtp_eager")` at the migration site; the engine
        // applies this relabel once at turn entry instead.
        Some("mtp_eager")
    }

    // Step A main forward: eager pre-norm + final-norm + project. Returns
    // `hidden` shaped `[1, hidden]` (squeeze the time axis) to match the
    // [`MtpStepper::forward_with_hidden`] contract; `logits` stays
    // `[1, 1, vocab]` with `needs_squeeze = true`.
    fn forward_with_hidden(
        &mut self,
        ids: &MxArray,
        emb: &MxArray,
    ) -> Result<(MxArray, MxArray, bool)> {
        match &mut self.mode {
            MtpStepMode::Flat => {
                let inner = &mut *self.inner;
                let pre = forward_pre_norm_inner(ids, emb, &mut inner.layers, &mut inner.caches)?;
                let h3 = inner.final_norm.forward(&pre)?;
                let logits = project_logits_from_hidden(
                    &h3,
                    &inner.lm_head,
                    emb,
                    Some(&self.embedding_weight_t),
                )?;
                let hidden = h3.squeeze(Some(&[1]))?;
                Ok((logits, hidden, true))
            }
            MtpStepMode::Paged(adapter) => {
                ids.eval();
                let token_id = ids.item_at_int32(0)? as u32;
                let inner = &mut *self.inner;
                // Cross-turn M-RoPE delta carried by a text turn that warm-
                // continues an image prefill; 0 for pure-text sessions.
                let rope_deltas = inner.cached_rope_deltas.unwrap_or(0);
                let caches = inner.caches.as_mut().ok_or_else(|| {
                    Error::from_reason("eager paged MTP forward_with_hidden: caches is None")
                })?;
                let (logits, hidden) = super::paged_forward::run_paged_step_with_hidden(
                    token_id,
                    &inner.embedding,
                    &mut inner.layers,
                    caches,
                    &inner.final_norm,
                    &inner.lm_head,
                    emb,
                    Some(&self.embedding_weight_t),
                    &self.layer_kinds,
                    adapter,
                    rope_deltas,
                )?;
                Ok((logits, hidden, true))
            }
        }
    }

    // One MTP draft step on the eager drafter. `h_next` is `[1, 1, hidden]`;
    // project to `draft_logits` `[1, 1, vocab]` then squeeze the time axis to
    // `[1, vocab]`.
    fn draft_step(
        &mut self,
        prev_hidden: &MxArray,
        prev_emb: &MxArray,
    ) -> Result<(MxArray, MxArray)> {
        let inner = &mut *self.inner;
        let mtp_caches = &mut self.mtp_caches;
        let mtp = inner.mtp.as_mut().ok_or_else(|| {
            Error::from_reason(
                "eager MTP draft_step: inner.mtp is None despite \
                 has_mtp_weights() gate",
            )
        })?;
        let h_next = mtp.forward(prev_hidden, prev_emb, Some(mtp_caches))?;
        let dl3 = project_logits_from_hidden(
            &h_next,
            &inner.lm_head,
            &self.embedding_weight,
            Some(&self.embedding_weight_t),
        )?;
        let draft_logits = dl3.squeeze(Some(&[1]))?;
        Ok((h_next, draft_logits))
    }

    // Batched verify: run the K+1 verify ids through the main stack,
    // advancing `inner.caches` by K+1, recording the GDN tape.
    fn verify_step(
        &mut self,
        ids: &MxArray,
        emb: &MxArray,
        depth: usize,
    ) -> Result<mtp_decode::MtpVerifyOutput> {
        match &mut self.mode {
            MtpStepMode::Flat => {
                let _ = depth;
                let inner = &mut *self.inner;
                let tape = &mut self.tape;
                eager_verify_step(
                    &mut inner.layers,
                    &mut inner.caches,
                    &inner.final_norm,
                    &inner.lm_head,
                    ids,
                    emb,
                    Some(&self.embedding_weight_t),
                    Some(tape),
                )
            }
            MtpStepMode::Paged(adapter) => {
                // Slice `ids` to exactly `depth+1` defensively so the adapter
                // records exactly K+1 tokens — the rollback count depends on it.
                let id_window = ids.to_int32().map_err(|e| {
                    Error::from_reason(format!(
                        "eager paged MTP verify_step: ids to_int32: {}",
                        e.reason
                    ))
                })?;
                if id_window.len() < depth + 1 {
                    return Err(Error::from_reason(format!(
                        "eager paged MTP verify_step: ids has {} elements, need {}",
                        id_window.len(),
                        depth + 1
                    )));
                }
                let id_slice: Vec<i32> = id_window.iter().take(depth + 1).copied().collect();
                let verify_in = MxArray::from_int32(&id_slice, &[1, (depth + 1) as i64])?;
                let inner = &mut *self.inner;
                // Cross-turn M-RoPE delta carried by a text turn that warm-
                // continues an image prefill; 0 for pure-text sessions.
                let rope_deltas = inner.cached_rope_deltas.unwrap_or(0);
                let caches = inner.caches.as_mut().ok_or_else(|| {
                    Error::from_reason("eager paged MTP verify_step: caches is None")
                })?;
                let tape = &mut self.tape;
                super::paged_forward::run_paged_verify_step(
                    &verify_in,
                    &inner.embedding,
                    &mut inner.layers,
                    caches,
                    &inner.final_norm,
                    &inner.lm_head,
                    emb,
                    Some(&self.embedding_weight_t),
                    &self.layer_kinds,
                    adapter,
                    tape,
                    rope_deltas,
                )
            }
        }
    }

    // No native argmax-only / sparse verify on the eager path — the accept
    // loop falls back to dense-logits accept. (Defaults `None`.)

    // Snapshot the main caches before verify mutates them. Stash the fallible
    // result; surfaced in `rollback` / `restore_and_replay_main`.
    fn snapshot_main_linear(&mut self) {
        // On the paged backend the FullAttention K/V lives in the paged pool,
        // not `inner.caches`, so its flat slot is an empty shell. Snapshot
        // paged-aware so we capture only the GDN (Linear) state and skip the
        // shells — `rollback` rewinds those via the adapter and never reads
        // their snapshot.
        let paged = matches!(self.mode, MtpStepMode::Paged(_));
        let inner = &*self.inner;
        let snap = match inner.caches.as_ref() {
            Some(caches) => super::layer_cache::snapshot_all_mtp(caches, paged),
            None => Err(Error::from_reason(
                "eager MTP snapshot_main_linear: inner.caches is None",
            )),
        };
        self.snap = Some(snap);
    }

    // Pure-Rust GDN tape replay — the correctness keystone. Fires on BOTH
    // full and partial accept. Infallible signature: any error is stashed in
    // `self.replay_err` and surfaced later.
    fn rollback(&mut self, accepted_drafts: usize, depth: usize) {
        if self.replay_err.is_some() {
            return;
        }
        // Paged path rewinds the full-attention K/V (which lives in the paged
        // pool, not `inner.caches`) by `rejected` tokens before the shared GDN
        // tape replay. On full accept `rejected == 0` (no-op). Flat layers keep
        // their full-attention K/V in `inner.caches` and rewind it via `kv.trim`
        // inside the replay loop instead.
        let paged = matches!(self.mode, MtpStepMode::Paged(_));
        if let MtpStepMode::Paged(adapter) = &mut self.mode {
            let rejected = depth.saturating_sub(accepted_drafts);
            if rejected > 0
                && let Err(e) = adapter.rollback_last_tokens(rejected as u32)
            {
                tracing::warn!(
                    target: "mlx_core::qwen3_5::paged",
                    "eager MTP-paged rollback_last_tokens({rejected}) failed \
                     (ignored): {e}",
                );
            }
        }
        let accepted_steps = accepted_drafts + 1;
        let result: Result<()> = (|| {
            let snap = match self.snap.as_ref() {
                Some(Ok(s)) => s,
                Some(Err(e)) => {
                    return Err(Error::from_reason(format!(
                        "eager MTP rollback: snapshot failed: {}",
                        e.reason
                    )));
                }
                None => {
                    return Err(Error::from_reason(
                        "eager MTP rollback: snapshot missing (snapshot_main_linear \
                         did not run)",
                    ));
                }
            };
            let tape = &self.tape;
            let inner = &mut *self.inner;
            let caches = inner
                .caches
                .as_mut()
                .ok_or_else(|| Error::from_reason("eager MTP rollback: inner.caches is None"))?;
            if caches.len() != snap.len() || caches.len() != tape.len() {
                return Err(Error::from_reason(format!(
                    "eager MTP rollback: length mismatch (caches {}, snapshot {}, \
                     tape {})",
                    caches.len(),
                    snap.len(),
                    tape.len(),
                )));
            }
            for (idx, cache) in caches.iter_mut().enumerate() {
                let Some(layer_tape) = tape[idx].as_ref() else {
                    if paged {
                        // Full-attention layer on the paged path: K/V lives in
                        // the paged pool and was already rewound by
                        // `adapter.rollback_last_tokens` above. The
                        // `inner.caches` FullAttention slot is unused on the
                        // paged path, so skip it.
                        continue;
                    }
                    // Full-attention layer: rewind the offset to
                    // `snapshot_offset + accepted_steps` so the next forward
                    // overwrites the rejected drafts. No-op on full accept.
                    match &snap[idx] {
                        super::layer_cache::Qwen3_5LayerSnapshot::FullAttention { offset } => {
                            let kv = cache.as_kv_cache_mut().ok_or_else(|| {
                                Error::from_reason(format!(
                                    "eager MTP rollback: layer {idx} has a \
                                     FullAttention snapshot but its cache slot is \
                                     not FullAttention",
                                ))
                            })?;
                            let target = *offset + accepted_steps as i32;
                            kv.trim(target);
                        }
                        super::layer_cache::Qwen3_5LayerSnapshot::Linear { .. } => {
                            return Err(Error::from_reason(format!(
                                "eager MTP rollback: layer {idx} has no GDN tape \
                                 but a Linear snapshot",
                            )));
                        }
                    }
                    continue;
                };
                let arrays = cache.as_arrays_cache_mut().ok_or_else(|| {
                    Error::from_reason(format!(
                        "eager MTP rollback: layer {idx} has a GDN tape but its \
                         cache slot is not Linear",
                    ))
                })?;
                let (snap_conv, snap_rec) = match &snap[idx] {
                    super::layer_cache::Qwen3_5LayerSnapshot::Linear {
                        conv_state,
                        recurrent_state,
                    } => (conv_state.as_ref(), recurrent_state.as_ref()),
                    super::layer_cache::Qwen3_5LayerSnapshot::FullAttention { .. } => {
                        return Err(Error::from_reason(format!(
                            "eager MTP rollback: layer {idx} GDN tape but \
                             FullAttention snapshot",
                        )));
                    }
                };
                let window = layer_tape.kernel.window_len()? as usize;
                if accepted_steps > window {
                    return Err(Error::from_reason(format!(
                        "eager MTP rollback: accepted_steps {accepted_steps} \
                         exceeds recorded window {window} at layer {idx}",
                    )));
                }
                layer_tape.replay_into(arrays, snap_conv, snap_rec, accepted_steps)?;
            }
            Ok(())
        })();
        if let Err(e) = result {
            self.replay_err = Some(e);
        }
    }

    // On rejection (partial accept): the GDN tape replay in `rollback`
    // already reconstructed the AR-exact main cache state, so no re-forward
    // loop is needed. This only surfaces a stashed replay error and clears
    // the per-cycle snapshot + tape.
    fn restore_and_replay_main(&mut self, _accepted: &[u32], _emb: &MxArray) -> Result<()> {
        self.snap = None;
        self.tape.clear();
        if let Some(e) = self.replay_err.take() {
            return Err(e);
        }
        Ok(())
    }

    // Committed-history commit.
    //
    // v1 (`!use_committed`): no-op.
    //
    // v2 (`use_committed`): append the M newly committed tokens' EXACT K/V to
    // the persistent MTP cache via one multi-token drafter forward.
    fn commit_mtp(
        &mut self,
        anchor: mtp_decode::MtpCommitAnchor,
        seed_hidden: &MxArray,
        verify_hiddens: &MxArray,
        committed_ids: &[u32],
        _k_accepted: usize,
        emb: &MxArray,
    ) -> Result<()> {
        if !self.use_committed {
            return Ok(());
        }
        let m = committed_ids.len();
        if m == 0 {
            return Ok(());
        }
        let hidden_dim = verify_hiddens.shape_at(2)?;

        // Assemble hidden_seq [1, M, hidden] per anchor.
        let hidden_seq = match anchor {
            mtp_decode::MtpCommitAnchor::IncludeAnchor => {
                // seed_hidden ++ verify_hiddens[:, 0..M-1, :].
                let vh_prefix =
                    verify_hiddens.slice(&[0, 0, 0], &[1, (m - 1) as i64, hidden_dim])?;
                MxArray::concatenate(seed_hidden, &vh_prefix, 1)?
            }
            mtp_decode::MtpCommitAnchor::SkipAlreadyCommittedAnchor => {
                // verify_hiddens[:, 0..M, :].
                verify_hiddens.slice(&[0, 0, 0], &[1, m as i64, hidden_dim])?
            }
        };

        // Gather the M committed-token input embeddings → [1, M, hidden].
        let ids_i32: Vec<i32> = committed_ids.iter().map(|&v| v as i32).collect();
        let ids_arr = MxArray::from_int32(&ids_i32, &[m as i64])?;
        let gathered = emb.take(&ids_arr, 0)?;
        let emb_seq = gathered.reshape(&[1, m as i64, hidden_dim])?;

        // Drop this cycle's draft K/V (written past committed_len by the draft
        // steps), then write the exact committed K/V via one multi-token
        // forward.
        let inner = &mut *self.inner;
        let mtp = inner.mtp.as_mut().ok_or_else(|| {
            Error::from_reason(
                "eager MTP commit_mtp: inner.mtp is None despite \
                 has_mtp_weights() gate",
            )
        })?;
        let caches = &mut self.mtp_caches;
        for c in caches.iter_mut() {
            if let Some(kv) = c.as_kv_cache_mut() {
                kv.trim(self.committed_len);
            }
        }
        let _ = mtp.forward(&hidden_seq, &emb_seq, Some(caches))?;
        self.committed_len += m as i32;
        Ok(())
    }

    // Re-anchor the drafter cache at the start of each cycle.
    //
    // v1 (`!use_committed`): reset to a fresh cache.
    //
    // v2 (`use_committed`): the cache is PERSISTENT; truncate the prior
    // cycle's draft tail back to the re-anchor target. `chained_anchor`
    // cycles anchor one slot earlier (`committed_len - 1`); Step-A cycles at
    // `committed_len`.
    fn begin_cycle(&mut self, chained_anchor: bool) {
        if !self.use_committed {
            self.mtp_caches = Qwen3_5MTPModule::fresh_caches(&self.config);
            return;
        }
        let target = if chained_anchor {
            (self.committed_len - 1).max(0)
        } else {
            self.committed_len
        };
        for c in self.mtp_caches.iter_mut() {
            if let Some(kv) = c.as_kv_cache_mut() {
                kv.trim(target);
            }
        }
    }

    // Bound the lazy graph: materialize the token plus the main GDN/full-attn
    // caches; on a budget-forced step also the logits.
    fn eval_step(&self, token: &MxArray, logits: &MxArray, budget_forced: bool) {
        async_eval_layer_caches(&self.inner.caches);
        token.eval();
        if budget_forced {
            logits.eval();
        }
    }

    // Chained end-of-iteration eval: keep the chained `verify_hidden[K]` slice
    // materialized alongside the token and the main caches so the next cycle's
    // draft graph does not force a separate Metal roundtrip.
    fn eval_step_with_chained_hidden(&self, token: &MxArray, chained_hidden: &MxArray) {
        async_eval_layer_caches(&self.inner.caches);
        MxArray::async_eval_arrays(&[token, chained_hidden]);
    }

    fn rollback_unemitted(&mut self, unemitted: usize) {
        match &mut self.mode {
            MtpStepMode::Flat => {
                if unemitted > 0 {
                    self.mtp_desynced = true;
                }
            }
            MtpStepMode::Paged(adapter) => {
                // Truncate the live paged adapter by the accepted-but-unemitted
                // tokens; the paged path never sets the FLAT desync latch.
                if let Err(e) = adapter.rollback_last_tokens(unemitted as u32) {
                    tracing::warn!(
                        target: "mlx_core::qwen3_5::paged",
                        "eager MTP-paged rollback_unemitted({unemitted}) failed \
                         (ignored): {e}",
                    );
                }
            }
        }
    }

    fn take_replay_error(&mut self) -> Option<Error> {
        self.replay_err.take()
    }

    fn into_desynced(self) -> bool {
        // Paged truncates its adapter in `rollback_unemitted` and never touches
        // the FLAT desync latch, so it always reports `false`. The adapter is
        // restored into `inner.paged_adapter` by the `Drop` impl that runs as
        // `self` falls out of scope here. (`self` is consumed by value rather
        // than destructured because the `Drop` impl forbids moving fields out.)
        match self.mode {
            MtpStepMode::Flat => self.mtp_desynced,
            MtpStepMode::Paged(_) => false,
        }
    }
}

impl MtpBackend for Qwen35Inner {
    type MtpDecode<'a>
        = DenseMtpStepper<'a>
    where
        Self: 'a;

    fn begin_mtp_decode(&mut self, setup: &MtpTurnSetup<'_>) -> Result<Self::MtpDecode<'_>> {
        let inference_info_enabled =
            tracing::enabled!(target: "mlx_core::inference", tracing::Level::INFO);
        let seed_trace_start = inference_info_enabled.then(std::time::Instant::now);
        let mut seed_tokens = 0usize;
        let mut seed_chunks = 0usize;
        // Turn-constant captures the eager-MTP block built before the loop:
        // the embedding table (+ its transpose for the tied projection) and a
        // config clone for the per-cycle drafter cache reset.
        let embedding_weight = self.embedding.get_weight();
        let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
        let config = self.config.clone();

        // Committed-history is only correct when the prompt tail's hiddens
        // start at absolute position 0 (the eager drafter derives RoPE purely
        // from the local cache offset) AND the matching prompt seed is
        // actually present. Cache-reuse continuations do not capture
        // full-prompt hiddens; their default position base is still zero, so
        // checking the base alone would falsely advertise an empty drafter
        // cache as prompt-committed. Chained cycles would then skip the
        // main-model anchor against that nonexistent history and could
        // diverge from the full-reprefill heal. Seedless turns fall back to
        // v1 cycle-history.
        let has_prompt_seed = setup.prompt_hidden_position_base == 0
            && setup.prompt_hidden.is_some()
            && setup.prompt_hidden_ids.is_some_and(|ids| !ids.is_empty());
        let use_committed = has_prompt_seed;

        // Auto-select the main-forward routing: the paged cores leave a paged
        // adapter on `self`, so `take()` moves it into the stepper for the turn
        // (restored by `Drop`); the flat cores have none and run flat. The
        // paged forwards need the per-layer kind classification (unused flat).
        let (mode, layer_kinds) = match self.paged_adapter.take() {
            Some(adapter) => {
                let layer_kinds = super::decoder_layer::compute_layer_kinds(
                    self.config.num_layers as usize,
                    |i| self.config.is_linear_layer(i),
                );
                (MtpStepMode::Paged(Box::new(adapter)), layer_kinds)
            }
            None => (MtpStepMode::Flat, Vec::new()),
        };

        let mut stepper = DenseMtpStepper {
            inner: self,
            mtp_caches: Qwen3_5MTPModule::fresh_caches(&config),
            committed_len: 0,
            use_committed,
            chained_cycles_supported: has_prompt_seed,
            snap: None,
            tape: Vec::new(),
            replay_err: None,
            mtp_desynced: false,
            embedding_weight,
            embedding_weight_t,
            config,
            mode,
            layer_kinds,
        };

        // Prompt-prefix seed (v2 committed-history only): commit the
        // contiguous run
        // `[prompt_hidden_ids[1..], y]` (length P, token 0 skipped, the first
        // sampled token `y` appended) into the persistent MTP cache so the
        // drafter attends the prompt from cycle 1. Each committed token `x` is
        // paired with `prompt_hidden[:, idx, :]` = h(token before `x`). Chunk
        // into pieces of size <= 7 and run the SAME multi-token eager KV-writer
        // as `commit_mtp` per chunk. `position_base == 0` is guaranteed by
        // `use_committed`, so RoPE (= local cache offset) aligns with absolute
        // position.
        if use_committed
            && let (Some(ph), Some(ph_ids)) = (setup.prompt_hidden, setup.prompt_hidden_ids)
            && !ph_ids.is_empty()
        {
            let prompt_len = ph_ids.len();
            let hidden_dim = ph.shape_at(2)?;
            let hidden_len = ph.shape_at(1)? as usize;
            if hidden_len != prompt_len {
                return Err(Error::from_reason(format!(
                    "eager MTP prompt-seed: prompt_hidden length {hidden_len} \
                     does not match prompt_hidden_ids length {prompt_len}"
                )));
            }
            // The first sampled token `y` is supplied by the engine via the
            // setup so the prompt seed can commit `[prompt_ids[1..], y]`.
            let y_id = setup.first_sampled_token;

            // Committed run = [prompt_ids[1..prompt_len], y] (length P).
            let mut committed_ids: Vec<i32> = Vec::with_capacity(prompt_len);
            committed_ids.extend(ph_ids[1..prompt_len].iter().map(|&v| v as i32));
            committed_ids.push(y_id as i32);

            let chunk_sizes = partition_prefill_chunks(prompt_len);
            seed_tokens = prompt_len;
            seed_chunks = chunk_sizes.len();
            let mut cursor: usize = 0;
            for &chunk in &chunk_sizes {
                let chunk_i64 = chunk as i64;
                let start = cursor as i64;
                // hidden_seq = prompt_hidden[:, cursor..cursor+chunk, :].
                let hidden_seq = ph.slice(&[0, start, 0], &[1, start + chunk_i64, hidden_dim])?;
                // emb_seq = gather embedding rows for the chunk's ids.
                let ids_arr =
                    MxArray::from_int32(&committed_ids[cursor..cursor + chunk], &[chunk_i64])?;
                let gathered = stepper.embedding_weight.take(&ids_arr, 0)?;
                let emb_seq = gathered.reshape(&[1, chunk_i64, hidden_dim])?;

                let inner = &mut *stepper.inner;
                let mtp = inner.mtp.as_mut().ok_or_else(|| {
                    Error::from_reason(
                        "eager MTP prompt-seed: inner.mtp is None despite \
                         has_mtp_weights() gate",
                    )
                })?;
                let caches = &mut stepper.mtp_caches;
                let _ = mtp.forward(&hidden_seq, &emb_seq, Some(caches))?;
                stepper.committed_len += chunk as i32;
                cursor += chunk;
            }
        }

        if inference_info_enabled {
            tracing::info!(
                target: "mlx_core::inference",
                event = "mtp_prompt_seed_done",
                mode = if use_committed {
                    "committed_history"
                } else {
                    "cycle_history"
                },
                seed_tokens,
                chunks = seed_chunks,
                max_chunk_tokens = 7,
                position_base = setup.prompt_hidden_position_base,
                setup_elapsed_ms = seed_trace_start.map(elapsed_ms).unwrap_or(0.0),
                "MTP prompt seed completed"
            );
        }

        Ok(stepper)
    }
}

impl ChatBackend for Qwen35Inner {
    fn tokenizer(&self) -> Result<Arc<Qwen3Tokenizer>> {
        self.tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))
    }

    fn family_name(&self) -> &'static str {
        "qwen3_5"
    }

    fn set_cache_owner_id(&mut self, owner_id: &str, root_owner_id: Option<&str>) {
        self.active_cache_owner_id.clear();
        self.active_cache_owner_id.push_str(owner_id);
        if let Some(root_owner_id) = root_owner_id {
            self.gdn_root_cache_owner_id = Some(root_owner_id.to_owned());
            self.gdn_root_cache_owner_is_explicit = true;
        } else {
            self.gdn_root_cache_owner_id = Some(owner_id.to_owned());
            self.gdn_root_cache_owner_is_explicit = false;
        }
    }

    fn session_eos_id(&self, tok: &Qwen3Tokenizer) -> Result<u32> {
        tok.im_end_id()
            .ok_or_else(|| Error::from_reason("Tokenizer missing <|im_end|> special token"))
    }

    fn generation_defaults(&self) -> Option<&crate::engine::ModelGenerationDefaults> {
        Some(&self.gen_defaults)
    }

    fn extra_eos_ids(&self) -> Vec<u32> {
        self.gen_defaults.eos_token_ids.clone()
    }

    // thinking: engine default `policy()` == `ThinkingPolicy::TemplateHonoring`
    // → `thinking_setup` resolves to
    // `{enabled: resolve_enable_thinking(config).unwrap_or(true),
    //   budget: config.thinking_token_budget}`.

    fn cached_token_history(&self) -> &[u32] {
        &self.cached_token_history
    }

    fn reset_caches(&mut self, scope: ResetScope) -> Result<()> {
        match scope {
            // Prefix-miss reset: reset each live layer cache, then install a
            // fresh hybrid cache vec. PRESERVES `cached_token_history` /
            // `cached_image_key` / `cached_rope_deltas` (the end-of-turn
            // save overwrites them) and the GDN checkpoints (paged-path
            // state the flat reset never touches).
            ResetScope::PrefixMiss => {
                if let Some(ref mut caches) = self.caches {
                    for cache in caches.iter_mut() {
                        cache.reset();
                    }
                }
                self.caches = Some(fresh_dense_layer_caches(&self.config));
                Ok(())
            }
            // Full clear including history, image key, rope deltas, GDN
            // checkpoints, via `reset_caches_sync`.
            ResetScope::Command => {
                self.reset_caches_sync()?;
                // The EXPLICIT command reset must restore a fully cold
                // state. `reset_caches_sync` clears the flat caches +
                // reuse/GDN state but leaves the paged request's FULL
                // blocks content-addressed in the per-instance
                // BlockAllocator's prefix cache, so a reset-then-rerun of
                // the same prompt would take the prefix-hit suffix-prefill
                // path (`verify_cache_prefix_direct` > 0) — a different
                // bf16 reduction order than the cold full prefill, enough
                // to flip a greedy near-tie (observed on the lfm2 sibling:
                // "says," vs "said" at token ~6; qwen3.5 shares the
                // identical adapter lifecycle).
                // Releasing the live request AND purging the prefix cache
                // makes the next turn replay the cold prefill byte-for-byte.
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    adapter
                        .release_request_and_purge_prefix_cache()
                        .map_err(|e| {
                            Error::from_reason(format!(
                                "qwen3_5 reset_caches: paged prefix-cache purge failed: {e}"
                            ))
                        })?;
                }
                Ok(())
            }
        }
    }

    /// All-or-nothing prefix match (NO exact-match rewind — the GDN
    /// recurrent state cannot rewind one slot; the engine's
    /// exact-match-as-miss handling performs a full reset + re-prefill on a
    /// zero-delta hit). Text-only by construction: the
    /// generic flow never carries images (the vision probe owns those
    /// turns), so the expanded-token / image-key inputs collapse to the
    /// plain prompt.
    fn verify_cache_prefix(&self, tokens: &[u32], reuse_cache: bool) -> usize {
        verify_cache_prefix_direct(
            reuse_cache,
            false,
            tokens,
            tokens,
            0,
            &self.cached_token_history,
            &self.cached_image_key,
            self.caches.is_some(),
        )
    }

    fn flat_caches_desynced(&self) -> bool {
        self.flat_mtp_caches_desynced
    }

    fn clear_flat_caches_desynced(&mut self) {
        self.flat_mtp_caches_desynced = false;
    }

    fn save_cache_state(&mut self, args: SaveStateArgs<'_>) {
        // Delta continuations preserve `cached_image_key` — the KV cache
        // still holds the prior prefill's image attention state even
        // though this turn was text-only. Fresh turns (re)set the key
        // from the turn's (always-false here) `has_images`.
        //
        // `drop_last_always = true`: this is the generic `run_decode_loop`
        // flow (flat, non-MTP, non-image dense turns), which never forwards
        // the final committed token into the physical cache on ANY exit
        // kind. The GDN recurrent state is non-invertible, so we drop that
        // token (rather than materialize it) to keep
        // `cached_token_history.len() == physical_cache_len`.
        if args.is_delta {
            engine::save_cache_state_after_delta(
                args.reuse_cache,
                args.generated_tokens,
                args.finish_reason,
                /* drop_last_always */ true,
                args.save_tokens,
                &mut self.cached_token_history,
                &mut self.cached_image_key,
                &mut self.cached_rope_deltas,
                &mut self.caches,
            );
        } else {
            save_cache_state_direct(
                args.reuse_cache,
                args.has_images,
                args.generated_tokens,
                args.finish_reason,
                /* drop_last_always */ true,
                args.save_tokens,
                args.save_expanded_tokens,
                args.image_cache_key,
                &mut self.cached_token_history,
                &mut self.cached_image_key,
                &mut self.cached_rope_deltas,
                &mut self.caches,
            );
        }
        self.cached_paged_image_token_positions.clear();
    }

    fn eval_caches(&self) -> Result<()> {
        // No post-prefill cache sync on qwen3.5's reference paths:
        // `chunked_prefill` evals internally per chunk and the decode
        // loop schedules async evals. A blocking sync here would
        // introduce an unnecessary stall.
        Ok(())
    }

    fn prefill(&mut self, prompt_tokens: &[u32], stream: Stream) -> Result<MxArray> {
        // Text-only prefill block (the engine's reset-or-delta split already
        // ran; `self.caches` holds either fresh caches or the live session
        // state).
        let embedding_weight = self.embedding.get_weight();
        let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
        let prompt = MxArray::from_uint32(prompt_tokens, &[1, prompt_tokens.len() as i64])?;
        chunked_prefill(
            &prompt,
            &embedding_weight,
            &mut self.layers,
            &mut self.caches,
            &self.final_norm,
            &self.lm_head,
            Some(&embedding_weight_t),
            stream,
        )
    }

    type Decode<'a>
        = Qwen35Decode<'a>
    where
        Self: 'a;

    fn begin_decode(&mut self, turn: &TurnSetup<'_>) -> Result<Self::Decode<'_>> {
        let p = turn.params;

        let is_streaming = self.turn_is_streaming.get();

        // Decode-entry trace (sync paths only — the streaming cores never
        // logged it). `enable_mtp && has_mtp_weights` turns route through
        // `mtp_turn`, so the MTP branch string is unreachable here.
        if !is_streaming {
            let prefill_len = turn.total_seq_len as i32;
            let max_kv_len_estimate =
                engine::kv_capacity_round_up_saturating(prefill_len, p.max_new_tokens);
            let has_mtp = self.has_mtp_weights();
            let branch = if !p.enable_mtp {
                "AR (enable_mtp=false)"
            } else if !has_mtp {
                "AR (no MTP weights on model)"
            } else {
                "AR"
            };
            info!(
                "Qwen3.5 chat_decode entry: prompt_len={} max_new_tokens={} enable_mtp={} \
                 mtp_depth={} prefill_seq_len={} max_kv_len={} has_mtp_weights={} \
                 is_delta={} has_images={} branch=\"{}\"",
                turn.total_seq_len,
                p.max_new_tokens,
                p.enable_mtp,
                p.mtp_depth,
                prefill_len,
                max_kv_len_estimate,
                has_mtp,
                turn.is_delta,
                turn.has_images,
                branch,
            );
        }

        let embedding_weight = self.embedding.get_weight();
        let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;

        let relabel = match (is_streaming, turn.is_delta) {
            (false, _) => "chat_rust",
            (true, false) => "chat_stream_rust",
            (true, true) => "chat_stream_delta_rust",
        };

        Ok(Qwen35Decode {
            inner: self,
            embedding_weight,
            embedding_weight_t,
            relabel,
        })
    }

    fn execution_plan(&self) -> ExecutionPlan {
        ExecutionPlan {
            media: qwen35_dense_media_plan(
                self.vision_encoder.is_some(),
                self.image_processor.is_some(),
                self.paged_adapter.is_some(),
            ),
            paged_attention: self.paged_adapter.as_ref().map(|_| PagedAttentionPlan {
                supports_delta: true,
            }),
            speculative: self.has_mtp_weights().then_some(SpeculativePlan {
                kind: SpeculativeKind::NativeMtp,
                supported_input_media: MediaCapabilities::NONE,
                // A text delta can continue an image-derived live session
                // with native MTP. The current turn carries no image bytes;
                // the eager paged-MTP core verifies against the target's
                // existing paged KV context. This is the same route the old
                // session dispatcher selected: its paged probe ran before the
                // MTP probe and dense `paged_turn` handled MTP directly.
                supported_context_media: MediaCapabilities::IMAGES,
                supports_paged_attention: true,
            }),
        }
    }

    fn wired_limit_bytes(&self) -> Option<usize> {
        // Per-turn wired-memory limit = the model's estimated footprint.
        Some(self.config.estimate_memory_bytes() as usize)
    }

    fn profiler_label(&self, is_delta: bool, is_streaming: bool) -> &'static str {
        // Record the turn's streaming-ness for `begin_decode`'s relabel
        // (`TurnSetup` does not carry it). The session core calls this
        // hook exactly once per generic-flow turn, before
        // `begin_decode`; specialized whole-turn paths return from their
        // planned executor earlier and never consult either hook. The labels
        // are the engine defaults.
        self.turn_is_streaming.set(is_streaming);
        match (is_streaming, is_delta) {
            (false, false) => "chat",
            (false, true) => "chat_delta",
            (true, false) => "chat_stream",
            (true, true) => "chat_stream_delta",
        }
    }

    fn has_live_session(&self) -> bool {
        // Delta guard: `self.caches.is_none()` means there is no
        // initialized session to continue.
        self.caches.is_some()
    }

    fn session_media(&self) -> MediaCapabilities {
        qwen35_dense_session_media(
            self.cached_image_key.is_some(),
            self.cached_rope_deltas.is_some(),
        )
    }

    fn session_media_matches_payloads(&self, images: &[Vec<u8>], audio: &[Vec<u8>]) -> bool {
        qwen35_dense_session_media_matches_payloads(self.cached_image_key, images, audio)
    }

    fn template_history_comparison_tokens<'a>(
        &self,
        tokens: &'a [u32],
    ) -> std::borrow::Cow<'a, [u32]> {
        engine::collapse_cached_media_placeholder_runs(
            tokens,
            IMAGE_TOKEN_ID as u32,
            &self.cached_paged_image_token_positions,
        )
    }

    fn run_paged_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        // Both SYNC and STREAMING turns take the paged core. The paged
        // cores self-handle MTP via the `eager_mtp_paged` arm
        // (`paged_turn_sync_core_inner` / `paged_turn_stream_core_inner`),
        // the streaming eager-MTP path.
        debug_assert!(args.plan.use_paged_attention);
        debug_assert!(self.paged_adapter.is_some());
        self.paged_whole_turn(args)
    }

    fn run_speculative_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        // The execution plan has already established request opt-in, loaded
        // MTP weights, flat-cache admission, and text-only input. The dense
        // core retains the algorithm-specific MTP gate and AR fallback.
        debug_assert!(args.media.is_empty());
        debug_assert!(matches!(
            args.plan.decoder,
            DecoderPlan::Speculative(SpeculativeKind::NativeMtp)
        ));
        self.dense_whole_turn(args)
    }

    fn run_multimodal_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        // The dense cores own the full image pipeline (VLM prefill, M-RoPE
        // deltas, paged-backend validation, missing-encoder error).
        self.dense_whole_turn(args)
    }
}

/// Generation configuration for Qwen3.5
#[napi(object)]
#[derive(Debug, Clone)]
pub struct Qwen3_5GenerationConfig {
    pub max_new_tokens: i32,
    #[napi(ts_type = "number | undefined")]
    pub temperature: Option<f64>,
    #[napi(ts_type = "number | undefined")]
    pub top_k: Option<i32>,
    #[napi(ts_type = "number | undefined")]
    pub top_p: Option<f64>,
    #[napi(ts_type = "number | undefined")]
    pub min_p: Option<f64>,
}

/// Generation result
#[napi(object)]
#[derive(Debug, Clone)]
pub struct Qwen3_5GenerationResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub num_tokens: u32,
    pub finish_reason: String,
}

/// Trained and physically available active-context limits for one loaded
/// Qwen3.5 model. Values are snapshots because the physical pool is fixed for
/// the lifetime of the resident model.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct Qwen3_5ContextLimits {
    pub trained_window_tokens: u32,
    pub effective_window_tokens: u32,
    pub paged_block_capacity: u32,
    pub paged_block_size: u32,
}

impl Qwen3_5ContextLimits {
    pub(crate) fn from_tuple(value: (u32, u32, u32, u32)) -> Self {
        Self {
            trained_window_tokens: value.0,
            effective_window_tokens: value.1,
            paged_block_capacity: value.2,
            paged_block_size: value.3,
        }
    }
}

// Shared chat types live in the model-neutral engine module; import them for
// internal use (no re-export — consumers import from `crate::engine::types`).
use crate::engine::types::{ChatConfig, ChatResult, ChatStreamChunk, ChatStreamHandle};

/// Qwen3.5 Model -- hybrid linear/full attention with optional MoE.
///
/// All inference and training state lives on a dedicated OS thread. NAPI methods
/// dispatch commands via channels and await responses. Training commands are
/// routed through `TrainingDispatch` to the model thread.
#[napi]
pub struct Qwen3_5Model {
    /// Dedicated model thread for inference and training.
    pub(crate) thread: crate::model_thread::ModelThread<Qwen35Cmd>,
    /// Cloned from inner for pure-getter NAPI methods (no command dispatch needed).
    pub(crate) config: Qwen3_5Config,
    /// Snapshot of `Qwen35Inner::paged_adapter.is_some()` captured at
    /// construction time. Text-only checkpoints default-OFF on Qwen3.5
    /// (parity-pending — see CLAUDE.md and
    /// `Qwen3_5Config::use_block_paged_cache`). VLM checkpoints default the
    /// adapter ON: dense image turns ONLY run on the paged-vision core, and a
    /// vision turn that reaches a None adapter errors at dispatch. Surfaced
    /// through the `hasBlockPagedCache()` NAPI method.
    pub(crate) paged_active: bool,
    /// Snapshot of `Qwen35Inner::has_mtp_weights()` captured
    /// at construction time, mirroring `paged_active`. Surfaced through
    /// the `hasMtpWeights()` NAPI method so the TS ChatSession can
    /// auto-default `enableMtp = true` for checkpoints that ship an MTP
    /// head without round-tripping through the model thread.
    pub(crate) mtp_active: bool,
    /// Snapshot of the fully loaded image execution stack. `true` only when
    /// the vision encoder and image processor were both installed and the
    /// block-paged adapter required by the Qwen3.5 vision core is active.
    /// Config metadata alone is insufficient: sym8 checkpoints can retain a
    /// `vision_config` while deliberately dropping the incompatible vision
    /// weights at load time.
    pub(crate) vision_active: bool,
    /// Loaded image processor retained by the public wrapper for CPU-only
    /// expanded-token planning before a streaming response commits headers.
    /// This is the same `Arc` used by the model thread's real vision prefill.
    pub(crate) image_processor: Option<Arc<Qwen35VLImageProcessor>>,
    /// Actual merge size installed on the loaded inner model, paired with
    /// `image_processor` for exact preflight geometry.
    pub(crate) spatial_merge_size: i32,
    pub(crate) context_limits: Qwen3_5ContextLimits,
    /// RAII: unregisters this model's baseline from the cache-limit
    /// coordinator on drop, so the global cap can shrink once JS GCs
    /// the wrapper.
    pub(crate) _cache_limit_guard: crate::cache_limit::CacheLimitGuard,
}

#[napi]
impl Qwen3_5Model {
    /// Whether the block-paged KV cache adapter is active on this model
    /// instance.
    ///
    /// `true` iff `Qwen35Inner::paged_adapter` was successfully
    /// constructed at load time (driven by
    /// `Qwen3_5Config::use_block_paged_cache`, default-OFF for text-only
    /// checkpoints because parity is pending real-weights validation, and
    /// default-ON for VLM checkpoints). On VLM checkpoints dense image turns
    /// ONLY run on the paged-vision core; a vision turn that reaches a None
    /// adapter errors at dispatch. Surfaced through this NAPI method so
    /// server endpoints can branch on it without round-tripping through
    /// the model thread.
    #[napi]
    pub fn has_block_paged_cache(&self) -> bool {
        self.paged_active
    }

    /// Whether this checkpoint shipped an MTP head (module loaded by
    /// `persistence::apply_weights_inner`). Snapshotted at load time from
    /// `Qwen35Inner::has_mtp_weights()` so the TS `ChatSession` can
    /// auto-default `enableMtp = true` for MTP-capable checkpoints without
    /// dispatching a command into the model thread.
    ///
    /// Note: this only reports weight availability. Whether the
    /// speculative-decode path actually runs on a given call also requires the
    /// per-request `enableMtp` flag.
    #[napi]
    pub fn has_mtp_weights(&self) -> bool {
        self.mtp_active
    }

    /// Whether this loaded model instance can execute image-bearing turns.
    ///
    /// This is an authoritative load-time snapshot, not a `config.json`
    /// family guess: it requires a loaded vision encoder, image processor,
    /// and the block-paged KV adapter used by the dense vision path.
    #[napi]
    pub fn supports_images(&self) -> bool {
        self.vision_active
    }

    /// Synchronous snapshot used by higher layers to preflight rendered
    /// prompts and clamp output before native cache allocation.
    #[napi]
    pub fn context_limits(&self) -> Qwen3_5ContextLimits {
        self.context_limits.clone()
    }

    /// Compute the exact prompt length after Qwen image-placeholder expansion
    /// without running the vision encoder or touching inference caches.
    ///
    /// `prompt_tokens` is the already-rendered chat-template output. `messages`
    /// supplies the complete image history so both fresh and leased-session
    /// preflights account for every image in template order.
    #[napi]
    pub async fn expanded_prompt_token_count(
        &self,
        prompt_tokens: Uint32Array,
        messages: Vec<ChatMessage>,
    ) -> Result<u32> {
        qwen35_expanded_prompt_token_count(
            self.image_processor.clone(),
            self.spatial_merge_size,
            prompt_tokens,
            messages,
        )
        .await
    }

    /// Load a pretrained model from a directory.
    ///
    /// Expects the directory to contain:
    /// - config.json
    /// - model.safetensors (or model-*.safetensors)
    /// - tokenizer.json + tokenizer_config.json
    #[napi]
    pub async fn load(path: String) -> Result<Qwen3_5Model> {
        persistence::load_with_thread(&path).await
    }

    /// Generate text from a prompt token sequence.
    #[napi]
    pub async fn generate(
        &self,
        prompt_tokens: &MxArray,
        mut config: Qwen3_5GenerationConfig,
    ) -> Result<Qwen3_5GenerationResult> {
        if config.max_new_tokens <= 0 {
            return Err(Error::from_reason(format!(
                "max_new_tokens must be > 0, got {}",
                config.max_new_tokens
            )));
        }
        let batch_size = prompt_tokens.shape_at(0)?;
        if batch_size != 1 {
            return Err(Error::from_reason(format!(
                "generate() only supports batch_size=1, got batch_size={}",
                batch_size
            )));
        }
        let prompt_len = prompt_tokens.shape_at(1)? as u32;
        let capacity = self.context_limits.effective_window_tokens;
        if prompt_len > capacity {
            return Err(Error::from_reason(format!(
                "context_length_exceeded: prompt has {prompt_len} tokens, effective active \
                 context is {capacity} tokens"
            )));
        }
        let max_output = capacity.saturating_sub(prompt_len).saturating_add(1);
        config.max_new_tokens = config.max_new_tokens.min(max_output as i32);
        crate::model_thread::send_and_await(&self.thread, |reply| Qwen35Cmd::Generate {
            prompt_tokens: prompt_tokens.clone(),
            config,
            reply,
        })
        .await
    }

    // ---------------------------------------------------------------
    // Test-only helpers: streaming session entry points that bypass
    // ThreadsafeFunction and expose the mpsc receiver directly. Used
    // by `crates/mlx-core/tests/qwen3_5_delta_chat.rs` to exercise the
    // streaming path from a pure-Rust integration test without a NAPI
    // host. Marked `#[doc(hidden)]` because they're not part of the
    // public API surface.
    // ---------------------------------------------------------------

    /// Test-only entry point that dispatches `ChatStreamSessionStart`
    /// and returns the raw mpsc receiver the model thread writes into.
    /// Callers can iterate the receiver directly rather than going
    /// through a NAPI callback.
    #[doc(hidden)]
    pub fn chat_stream_session_start_for_test(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
    ) -> Result<(
        ChatStreamHandle,
        tokio::sync::mpsc::UnboundedReceiver<Result<ChatStreamChunk>>,
    )> {
        let config = config.unwrap_or_default();
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<ChatStreamChunk>>();
        self.thread
            .send(Qwen35Cmd::Chat(ChatCmd::StreamSessionStart {
                messages,
                config,
                stream_tx,
                cancelled: cancelled_inner,
            }))?;
        Ok((ChatStreamHandle { cancelled }, stream_rx))
    }

    /// Test-only entry point that dispatches `ChatStreamSessionContinue`
    /// and returns the raw mpsc receiver the model thread writes into.
    #[doc(hidden)]
    pub fn chat_stream_session_continue_for_test(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
    ) -> Result<(
        ChatStreamHandle,
        tokio::sync::mpsc::UnboundedReceiver<Result<ChatStreamChunk>>,
    )> {
        let config = config.unwrap_or_default();
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<ChatStreamChunk>>();
        self.thread
            .send(Qwen35Cmd::Chat(ChatCmd::StreamSessionContinue {
                messages,
                config,
                stream_tx,
                cancelled: cancelled_inner,
            }))?;
        Ok((ChatStreamHandle { cancelled }, stream_rx))
    }

    /// Test-only snapshot of the flat-MTP cache state, read *between* turns:
    /// `(committed_history_len, flat_mtp_caches_desynced,
    /// full_reprefill_count, rollback_unemitted)`.
    ///
    /// `committed_history_len` is `cached_token_history.len()` — the prompt plus
    /// the committed generation of every completed turn — i.e. exactly how many
    /// tokens a turn committed. Unlike `ChatStreamChunk.prompt_tokens` (hardcoded
    /// to the delta length on the streaming delta path, heal or warm), it is
    /// path-independent and comparable across MTP and AR turns.
    /// `flat_mtp_caches_desynced` reports whether the preceding turn stranded
    /// tokens mid-cycle and armed the heal. `full_reprefill_count` is the
    /// monotonic number of discard+re-prefill heals the streaming delta path has
    /// taken. `rollback_unemitted` is the independently engine-computed accepted
    /// tail sent to the family rollback hook, so a positive value must agree with
    /// the desync flag on flat MTP. Serialized behind the model thread, so it
    /// observes the fully-finalized preceding turn.
    #[doc(hidden)]
    pub async fn mtp_flat_state_for_test(&self) -> (usize, bool, u64, usize) {
        crate::model_thread::send_and_await(&self.thread, |reply| Qwen35Cmd::MtpFlatStateForTest {
            reply,
        })
        .await
        .expect("mtp_flat_state_for_test: model thread reply failed")
    }

    /// Test-only: arm the flat-MTP desync heal so the NEXT delta turn takes the
    /// discard+re-prefill path. Lets a test exercise the heal deterministically
    /// (the mid-cycle cancel that naturally arms it is host-timing-dependent).
    #[doc(hidden)]
    pub async fn force_flat_mtp_desync_for_test(&self) {
        crate::model_thread::send_and_await(&self.thread, |reply| {
            Qwen35Cmd::ForceFlatMtpDesyncForTest { reply }
        })
        .await
        .expect("force_flat_mtp_desync_for_test: model thread reply failed")
    }

    /// Get the number of parameters in the model.
    ///
    /// Pure config computation — no model-thread dispatch needed.
    #[napi]
    pub fn num_parameters(&self) -> i64 {
        let h = self.config.hidden_size as i64;
        let v = self.config.vocab_size as i64;
        let n = self.config.num_layers as usize;
        let dense_i = self.config.intermediate_size as i64;

        let mut total = v * h;
        if !self.config.tie_word_embeddings {
            total += v * h;
        }

        let kd = self.config.linear_key_dim() as i64;
        let vd = self.config.linear_value_dim() as i64;

        for layer_idx in 0..n {
            let is_linear = self.config.is_linear_layer(layer_idx);
            if is_linear {
                let num_vh = self.config.linear_num_value_heads as i64;
                let vhd = self.config.linear_value_head_dim as i64;
                total += h * (kd * 2 + vd * 2)
                    + h * (num_vh * 2)
                    + (kd * 2 + vd) * self.config.linear_conv_kernel_dim as i64
                    + vd * h
                    + num_vh
                    + num_vh
                    + vhd;
            } else {
                let d = self.config.head_dim as i64;
                total += h * h * 2 + h * (self.config.num_kv_heads as i64 * d) * 2 + h * h + d * 2;
            }
            total += 3 * h * dense_i;
            total += h * 2;
        }
        total += h;
        total
    }

    /// Save the model weights and configuration to a directory.
    ///
    /// Dispatches to model thread.
    #[napi]
    pub fn save_model<'env>(
        &self,
        env: &'env Env,
        save_path: String,
    ) -> Result<PromiseRaw<'env, ()>> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.thread.send(Qwen35Cmd::SaveModel {
            save_path,
            reply: tx,
        })?;
        let promise = env.spawn_future(async move {
            rx.await
                .map_err(|_| napi::Error::from_reason("Model thread exited unexpectedly"))?
        })?;
        Ok(promise)
    }
}

/// Shared dense/MoE NAPI implementation for exact, non-mutating image prompt
/// planning. Inputs are copied before entering the blocking worker so no JS
/// backing-store references cross threads.
pub(crate) async fn qwen35_expanded_prompt_token_count(
    image_processor: Option<Arc<Qwen35VLImageProcessor>>,
    spatial_merge_size: i32,
    prompt_tokens: Uint32Array,
    messages: Vec<ChatMessage>,
) -> Result<u32> {
    let tokens = prompt_tokens.to_vec();
    let images = extract_images_from_messages(&messages);
    if images.is_empty() {
        return u32::try_from(tokens.len())
            .map_err(|_| Error::from_reason("rendered prompt token count exceeds u32"));
    }
    let image_processor = image_processor.ok_or_else(|| {
        Error::from_reason(
            "cannot plan expanded image tokens: Qwen3.5 image processor is not loaded",
        )
    })?;

    napi::bindgen_prelude::spawn_blocking(move || {
        let prompt_len =
            plan_expanded_image_prompt_len(&image_processor, spatial_merge_size, &tokens, &images)?;
        u32::try_from(prompt_len)
            .map_err(|_| Error::from_reason("expanded prompt token count exceeds u32"))
    })
    .await
    .map_err(|join_error| {
        Error::new(
            Status::GenericFailure,
            format!("Expanded prompt planning worker failed: {join_error}"),
        )
    })?
}

impl Qwen3_5Model {
    /// Test-only deterministic teardown for memory-constrained real-weight
    /// integration tests. Requires exclusive command-sender ownership and
    /// leaves production NAPI drops non-blocking.
    #[doc(hidden)]
    pub fn shutdown_for_test(mut self) -> std::result::Result<(), String> {
        self.thread.shutdown_and_join()
    }
}

crate::models::chat_napi::chat_napi_surface! {
    class: Qwen3_5Model,
    thread_cmd: Qwen35Cmd,
    thread: direct,
    image_guard: none,
    ts_stream_start: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue_tool: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
}

/// Default prefill chunk size (tokens per chunk).
/// Matches Python mlx-lm's `prefill_step_size` default of 2048.
///
/// E55: bumped 1024 → 2048 after benching against mlx-lm at 20k prompt:
/// chunk=1024 incurred 20 chunk boundaries vs mlx-lm's 10 (mlx-lm uses
/// 2048 by default); the doubled per-chunk overhead cost ~14% at 20k.
/// At 1024-prompt single-chunk the value is irrelevant — the loop is
/// guarded by `total_len - offset > PREFILL_STEP_SIZE` so any T < step
/// goes through the single `remaining` branch unchanged.
const PREFILL_STEP_SIZE: i64 = 2048;

/// Evaluate all cache arrays across all layers to materialize them on GPU.
/// Must be called between prefill chunks to break lazy dependency chains.
pub(crate) fn eval_layer_caches(caches: &Option<Vec<Qwen3_5LayerCache>>) -> Result<()> {
    if let Some(caches) = caches {
        let mut arrays: Vec<&MxArray> = Vec::new();
        for cache in caches.iter() {
            cache.collect_arrays(&mut arrays);
        }
        MxArray::eval_arrays(&arrays)?;
    }
    Ok(())
}

/// Async variant of `eval_layer_caches`: kicks GPU on cache materialization
/// but does NOT block the CPU. Used between prefill chunks so the CPU can
/// start building the next chunk's graph while the previous chunk's cache
/// writes are still in flight.
pub(crate) fn async_eval_layer_caches(caches: &Option<Vec<Qwen3_5LayerCache>>) {
    if let Some(caches) = caches {
        let mut arrays: Vec<&MxArray> = Vec::new();
        for cache in caches.iter() {
            cache.collect_arrays(&mut arrays);
        }
        MxArray::async_eval_arrays(&arrays);
    }
}

/// Chunked prefill: process prompt in chunks of `PREFILL_STEP_SIZE`, evaluating
/// caches and clearing compute cache between chunks to bound peak memory.
///
/// Accepts `&MxArray` shaped `[1, seq_len]`. Slices on GPU — no data roundtrip.
/// For `&[u32]` inputs (from tokenizer), callers convert with `MxArray::from_uint32` first.
fn chunked_prefill(
    prompt: &MxArray,
    embedding_weight: &MxArray,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    embedding_weight_t: Option<&MxArray>,
    generation_stream: crate::stream::Stream,
) -> Result<MxArray> {
    chunked_prefill_with_size(
        prompt,
        embedding_weight,
        layers,
        caches,
        final_norm,
        lm_head,
        embedding_weight_t,
        generation_stream,
        PREFILL_STEP_SIZE,
    )
}

fn chunked_prefill_with_size(
    prompt: &MxArray,
    embedding_weight: &MxArray,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    embedding_weight_t: Option<&MxArray>,
    generation_stream: crate::stream::Stream,
    chunk_size: i64,
) -> Result<MxArray> {
    let total_len = prompt.shape_at(1)?;
    if total_len <= 0 {
        return Err(Error::from_reason("chunked_prefill: empty prompt"));
    }
    let chunk_size = if chunk_size <= 0 {
        total_len
    } else {
        chunk_size
    };
    let mut offset: i64 = 0;

    // E28: env-var toggle for A/B. Default: async between chunks. When set,
    // falls back to synchronous eval_layer_caches (the prior behavior).
    let chunk_async = std::env::var("MLX_PREFILL_SYNC_BETWEEN_CHUNKS").is_err();
    while total_len - offset > chunk_size {
        let chunk = prompt.slice_axis(1, offset, offset + chunk_size)?;
        {
            let _stream_ctx = StreamContext::new(generation_stream);
            let _hidden = forward_pre_norm_inner(&chunk, embedding_weight, layers, caches)?;
        }
        if chunk_async {
            async_eval_layer_caches(caches);
        } else {
            eval_layer_caches(caches)?;
        }
        crate::array::clear_cache();
        offset += chunk_size;
    }

    let remaining = prompt.slice_axis(1, offset, total_len)?;
    let last_logits = {
        let _stream_ctx = StreamContext::new(generation_stream);
        let hidden = forward_pre_norm_inner(&remaining, embedding_weight, layers, caches)?;
        project_last_logits_from_pre_norm_hidden(
            &hidden,
            final_norm,
            lm_head,
            embedding_weight,
            embedding_weight_t,
        )?
    };
    Ok(last_logits)
}

/// `chunked_prefill` variant that ALSO returns the post-final-norm hidden
/// state for the prompt tail needed by MTP, concatenated along the time axis
/// -> `[1, kept_len, hidden]`.
///
/// Used only when MTP is active for the turn: the prompt hiddens flow
/// through `ChatDecodeInputs::prompt_hidden` into `begin_mtp_decode`'s
/// prompt-prefix seed, which commits the prompt prefix into the MTP
/// committed-history caches. Logits-only callers keep the cheaper
/// `chunked_prefill`. The per-chunk forward op sequence is identical for
/// chunks whose hidden is kept; chunks before the requested tail use the
/// logits-only path and discard hidden to avoid materializing prompt history
/// MTPLX would not seed.
fn chunked_prefill_with_hidden(
    prompt: &MxArray,
    embedding_weight: &MxArray,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    embedding_weight_t: Option<&MxArray>,
    generation_stream: crate::stream::Stream,
    keep_last_hidden: Option<usize>,
) -> Result<(MxArray, MxArray)> {
    chunked_prefill_with_hidden_with_size(
        prompt,
        embedding_weight,
        layers,
        caches,
        final_norm,
        lm_head,
        embedding_weight_t,
        generation_stream,
        keep_last_hidden,
        PREFILL_STEP_SIZE,
    )
}

fn chunked_prefill_with_hidden_with_size(
    prompt: &MxArray,
    embedding_weight: &MxArray,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    embedding_weight_t: Option<&MxArray>,
    generation_stream: crate::stream::Stream,
    keep_last_hidden: Option<usize>,
    chunk_size: i64,
) -> Result<(MxArray, MxArray)> {
    let total_len = prompt.shape_at(1)?;
    if total_len <= 0 {
        return Err(Error::from_reason(
            "chunked_prefill_with_hidden: empty prompt",
        ));
    }
    let chunk_size = if chunk_size <= 0 {
        total_len
    } else {
        chunk_size
    };
    let mut offset: i64 = 0;
    let mut hidden_chunks: Vec<MxArray> = Vec::new();
    let keep_start = keep_last_hidden
        .map(|keep| total_len.saturating_sub(keep.max(1) as i64))
        .unwrap_or(0);

    while total_len - offset > chunk_size {
        let end = offset + chunk_size;
        let chunk = prompt.slice_axis(1, offset, end)?;
        let overlaps_kept_tail = end > keep_start;
        let kept_hidden = if overlaps_kept_tail {
            let _stream_ctx = StreamContext::new(generation_stream);
            let hidden = forward_pre_norm_inner(&chunk, embedding_weight, layers, caches)?;
            let keep_from = keep_start.max(offset);
            let hidden = if keep_from > offset {
                hidden.slice_axis(1, keep_from - offset, end - offset)?
            } else {
                hidden
            };
            Some(final_norm.forward(&hidden)?)
        } else {
            let _stream_ctx = StreamContext::new(generation_stream);
            let _hidden = forward_pre_norm_inner(&chunk, embedding_weight, layers, caches)?;
            None
        };
        eval_layer_caches(caches)?;
        if let Some(kept_hidden) = kept_hidden {
            // Materialize the kept hidden before clearing the MLX cache — it
            // is a lazy handle referencing graph nodes that `clear_cache`
            // would otherwise free.
            kept_hidden.eval();
            hidden_chunks.push(kept_hidden);
        }
        crate::array::clear_cache();
        offset = end;
    }

    let remaining = prompt.slice_axis(1, offset, total_len)?;
    let (last_logits, last_hidden) = {
        let _stream_ctx = StreamContext::new(generation_stream);
        let hidden = forward_pre_norm_inner(&remaining, embedding_weight, layers, caches)?;
        let logits = project_last_logits_from_pre_norm_hidden(
            &hidden,
            final_norm,
            lm_head,
            embedding_weight,
            embedding_weight_t,
        )?;
        let keep_from = keep_start.max(offset);
        let hidden = if keep_from > offset {
            hidden.slice_axis(1, keep_from - offset, total_len - offset)?
        } else {
            hidden
        };
        (logits, final_norm.forward(&hidden)?)
    };
    hidden_chunks.push(last_hidden);

    // Concatenate every kept `[1, chunk, hidden]` along axis 1 →
    // `[1, kept_len, hidden]`.
    let prompt_hidden = if hidden_chunks.len() == 1 {
        hidden_chunks
            .into_iter()
            .next()
            .ok_or_else(|| Error::from_reason("chunked_prefill_with_hidden: empty hidden chunks"))?
    } else {
        let mut acc = hidden_chunks[0].clone();
        for chunk in &hidden_chunks[1..] {
            acc = MxArray::concatenate(&acc, chunk, 1)?;
        }
        acc
    };
    Ok((last_logits, prompt_hidden))
}

/// Lock-free forward pass through all layers.
/// Attention layer handles causal masking internally via "causal" SDPA mode.
/// Format an `MxArray`'s shape for logging. Returns `[d0, d1, ...]`
/// or `"<unavailable>"` if `ndim()` fails.
fn shape_dbg(arr: &MxArray) -> String {
    let ndim = match arr.ndim() {
        Ok(n) => n,
        Err(_) => return "<unavailable>".to_string(),
    };
    let mut dims: Vec<i64> = Vec::with_capacity(ndim as usize);
    for axis in 0..ndim {
        match arr.shape_at(axis) {
            Ok(d) => dims.push(d),
            Err(_) => return "<unavailable>".to_string(),
        }
    }
    format!("{:?}", dims)
}

fn forward_inner(
    input_ids: &MxArray,
    embedding_weight: &MxArray,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    embedding_weight_t: Option<&MxArray>,
) -> Result<MxArray> {
    let hidden = forward_pre_norm_inner(input_ids, embedding_weight, layers, caches)?;
    let hidden = final_norm.forward(&hidden)?;
    project_logits_from_hidden(&hidden, lm_head, embedding_weight, embedding_weight_t)
}

fn forward_pre_norm_inner(
    input_ids: &MxArray,
    embedding_weight: &MxArray,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
) -> Result<MxArray> {
    let embedding = Embedding::from_weight(embedding_weight)?;
    let hidden_states = embedding.forward(input_ids)?;
    let mut h = hidden_states.clone();

    debug!(
        "Qwen3.5 forward_inner: input_ids_shape={} post_embed_shape={}",
        shape_dbg(input_ids),
        shape_dbg(&h),
    );

    let num_layers = layers.len();
    // Plain layer loop. In-loop async_eval was tested (every 8 layers,
    // including h + all cache arrays) and found neutral-to-negative at
    // single-chunk prefill on M3 (back-to-back A/B in the same binary
    // showed deltas inside the run-to-run noise band). The CPU/GPU
    // overlap benefit only materializes across the inter-chunk barrier
    // in chunked_prefill, which now uses async_eval_layer_caches.
    //
    // This is the SHARED pre-norm primitive: it MUST return the full
    // per-position hidden. The MTP prompt-hidden path
    // (`chunked_prefill_with_hidden_with_size`) keeps the result and
    // re-slices it by chunk length, so a last-token slice here would
    // corrupt it. The logits-only callers get the equivalent of the
    // upstream E37 last-token optimization from
    // `project_last_logits_from_pre_norm_hidden` (which slices before
    // `final_norm` + `lm_head`), so the slice deliberately does NOT
    // live in this loop.
    for i in 0..num_layers {
        let cache = caches.as_mut().map(|c| &mut c[i]);
        h = layers[i].forward(&h, None, cache, None, true)?;
        if i == 0 || i + 1 == num_layers {
            debug!(
                "Qwen3.5 forward_inner: post_layer[{}/{}] shape={}",
                i,
                num_layers,
                shape_dbg(&h),
            );
        }
    }

    Ok(h)
}

/// Tape-recording variant of [`forward_pre_norm_inner`] for the eager MTP
/// verify forward.
///
/// Identical to `forward_pre_norm_inner` except it records a per-layer
/// [`GdnLayerTape`] for every GDN (`Linear`) layer into `tape`, indexed by
/// ABSOLUTE layer index (`tape[i]` is `Some` for GDN layers, stays `None` for
/// full-attention layers). `tape` is pre-sized to `layers.len()` by the caller.
/// Recording is by lazy `.clone()` (no eval), so it stays inside the fused MLX
/// graph that `eval_step`/`async_eval_layer_caches` materializes.
fn forward_pre_norm_inner_with_tape(
    input_ids: &MxArray,
    embedding_weight: &MxArray,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    tape: &mut [Option<super::gated_delta_net::GdnLayerTape>],
) -> Result<MxArray> {
    let embedding = Embedding::from_weight(embedding_weight)?;
    let hidden_states = embedding.forward(input_ids)?;
    let mut h = hidden_states.clone();

    let num_layers = layers.len();
    debug_assert_eq!(
        tape.len(),
        num_layers,
        "forward_pre_norm_inner_with_tape: tape length must equal layer count"
    );
    for i in 0..num_layers {
        let cache = caches.as_mut().map(|c| &mut c[i]);
        let mut slot: Option<super::gated_delta_net::GdnLayerTape> = None;
        h = layers[i].forward_with_tape(&h, None, cache, None, true, Some(&mut slot))?;
        tape[i] = slot;
    }

    Ok(h)
}

fn project_logits_from_hidden(
    hidden: &MxArray,
    lm_head: &Option<LinearProj>,
    embedding_weight: &MxArray,
    embedding_weight_t: Option<&MxArray>,
) -> Result<MxArray> {
    match lm_head {
        Some(head) => head.forward(hidden),
        None => match embedding_weight_t {
            Some(wt) => hidden.matmul(wt),
            None => {
                let wt = embedding_weight.transpose(Some(&[1, 0]))?;
                hidden.matmul(&wt)
            }
        },
    }
}

/// Eager (pure-Rust) MTP verify step.
///
/// Translation of the deleted compiled `forward_mtp_verify_compiled_with_hidden`
/// FFI: runs the `verify_ids` (`[1, K+1]` int32) through the SAME main-model
/// stack the AR path uses (`forward_pre_norm_inner` + `final_norm` +
/// `project_logits_from_hidden`), advancing `inner.caches` by `K+1` positions.
///
/// Returns `MtpVerifyOutput::logits_only(logits, hiddens)` where:
///   * `logits` is `[1, K+1, vocab]` (the verifier target distribution at
///     every verify position),
///   * `hiddens` is `[1, K+1, hidden]` — the post-final-norm hidden at every
///     verify position (the chained-seed and commit context).
///
/// `emb` is the embedding table; `emb_t` is its precomputed transpose for the
/// tied-embedding projection (passed straight through to
/// `project_logits_from_hidden`).
#[allow(clippy::too_many_arguments)]
fn eager_verify_step(
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    verify_ids: &MxArray,
    emb: &MxArray,
    emb_t: Option<&MxArray>,
    tape: Option<&mut Vec<Option<super::gated_delta_net::GdnLayerTape>>>,
) -> Result<mtp_decode::MtpVerifyOutput> {
    let pre = match tape {
        Some(tape) => {
            // Record a per-layer GDN tape during the verify forward so the
            // rollback replay can reconstruct the AR-exact carried state.
            tape.clear();
            tape.resize(layers.len(), None);
            forward_pre_norm_inner_with_tape(verify_ids, emb, layers, caches, tape)?
        }
        None => forward_pre_norm_inner(verify_ids, emb, layers, caches)?,
    };
    let hiddens = final_norm.forward(&pre)?;
    let logits = project_logits_from_hidden(&hiddens, lm_head, emb, emb_t)?;
    Ok(mtp_decode::MtpVerifyOutput::logits_only(logits, hiddens))
}

fn project_last_logits_from_pre_norm_hidden(
    hidden: &MxArray,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    embedding_weight: &MxArray,
    embedding_weight_t: Option<&MxArray>,
) -> Result<MxArray> {
    let seq_len = hidden.shape_at(1)?;
    let last_hidden = hidden.slice_axis(1, seq_len - 1, seq_len)?;
    let last_hidden = final_norm.forward(&last_hidden)?;
    let logits =
        project_logits_from_hidden(&last_hidden, lm_head, embedding_weight, embedding_weight_t)?;
    logits.squeeze(Some(&[1]))
}

/// Partition `total` committed tokens into chunk sizes all within the
/// commit graph's `M in [1, 7]` window.
///
/// Strategy: greedily take size-6 chunks. The final remainder `r` is
/// `total % 6`:
///   - `r == 0`           → all chunks are size 6.
///   - `r >= 2`           → append one chunk of size `r`.
///   - `r == 1`           -> append one chunk of size 1.
///
/// Precondition: `total >= 1`. For `total in {1..7}` the single chunk is
/// `total` itself.
///
/// `pub(crate)`: also used by `MoeMtpStepper::begin_mtp_decode`'s
/// committed-history v2 prompt-prefix seed
/// (`crate::models::qwen3_5_moe::model`), which mirrors this dense chunking.
pub(crate) fn partition_prefill_chunks(total: usize) -> Vec<usize> {
    debug_assert!(total >= 1, "partition_prefill_chunks: total must be >= 1");
    const CHUNK: usize = 6;
    if total == 1 {
        return vec![1];
    }
    if total <= 7 {
        // A single chunk in [1, 7] covers it directly.
        return vec![total];
    }
    let mut chunks: Vec<usize> = Vec::new();
    let mut remaining = total;
    while remaining > 7 {
        chunks.push(CHUNK);
        remaining -= CHUNK;
    }
    // `remaining` is now in [1, 7]. Push it directly.
    debug_assert!(
        (1..=7).contains(&remaining),
        "partition_prefill_chunks: remainder {remaining} out of [1, 7]"
    );
    chunks.push(remaining);
    chunks
}

// ============================================================================
// VLM helper functions (moved from vl_model.rs for unification)
// ============================================================================

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
                        unreachable!("placeholder count was validated before expansion");
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
    if image_grid_thw.is_none() {
        let pos = MxArray::arange(0.0, seq_len as f64, Some(1.0), None)?;
        let pos = pos.reshape(&[1, 1, seq_len])?;
        let position_ids = MxArray::tile(&pos, &[3, batch_size as i32, 1])?;
        return Ok((position_ids, 0));
    }

    let grid_thw = image_grid_thw.unwrap();
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
        // joined by an assistant reply). Flattening the span from
        // `positions[0]` to `positions[last]` the way the old
        // single-span code did would skip every interior text token
        // between runs and blow up the downstream shape check.
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
        // it equals `image_runs.last().unwrap().1`.
        let last_image_end = image_runs.last().expect("at least one run").1;

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
    vision_encoder: &Qwen3_5VisionEncoder,
    spatial_merge_size: i32,
    text_model_embedding: &MxArray,
    generation_stream: Stream,
    vision_cache: &VisionCache,
) -> Result<VisionMerge> {
    // === STEP 1: Compute vision features with per-image reuse ===
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
    let cache_feature_dtype = text_model_embedding.dtype()?;
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

    // === STEP 2: Get text embeddings and merge with vision features ===
    let text_embeds = {
        let _stream_ctx = StreamContext::new(generation_stream);
        let embedding = Embedding::from_weight(text_model_embedding)?;
        embedding.forward(input_ids)?
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

    // === STEP 3: Compute M-RoPE position IDs ===
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
mod vision_feature_cache_tests {
    use super::*;

    fn digest(word: u64) -> engine::ImageCacheDigest {
        engine::ImageCacheDigest::from_test_word(word)
    }

    fn key(image_hash: u64, grid_thw: [i32; 3]) -> VisionFeatureCacheKey {
        VisionFeatureCacheKey {
            image_hash: digest(image_hash),
            grid_thw,
        }
    }

    fn request(
        key: VisionFeatureCacheKey,
        patch_start: i64,
        patch_count: i64,
    ) -> VisionImageRequest {
        VisionImageRequest {
            key,
            patch_start,
            patch_count,
            feature_count: patch_count,
        }
    }

    fn feature(value: f32) -> MxArray {
        let array = MxArray::from_float32(&[value], &[1]).unwrap();
        array.eval();
        array
    }

    fn snapshot(
        total: u64,
        mlx_limit: u64,
        metal_limit: u64,
        active: u64,
        metal_current: u64,
        allocator_cache: u64,
    ) -> VisionMemorySnapshot {
        VisionMemorySnapshot {
            total_system_memory_bytes: total,
            mlx_memory_limit_bytes: mlx_limit,
            metal_working_set_bytes: metal_limit,
            allocator_active_bytes: active,
            allocator_active_probe_ok: active > 0,
            metal_current_allocated_bytes: metal_current,
            metal_current_probe_ok: metal_current > 0,
            allocator_cache_bytes: allocator_cache,
        }
    }

    fn miss(key: VisionFeatureCacheKey, patch_count: i64) -> VisionCacheMiss {
        VisionCacheMiss {
            request: request(key, 0, patch_count),
            request_indices: vec![0],
        }
    }

    #[test]
    fn plans_per_image_ranges_and_rejects_invalid_geometry() {
        let planned =
            plan_vision_image_requests(&[1, 4, 4, 2, 2, 4], &[digest(10), digest(20)], 32, 2)
                .expect("valid image geometry");
        assert_eq!(planned[0].patch_start, 0);
        assert_eq!(planned[0].patch_count, 16);
        assert_eq!(planned[0].feature_count, 4);
        assert_eq!(planned[1].patch_start, 16);
        assert_eq!(planned[1].patch_count, 16);
        assert_eq!(planned[1].feature_count, 4);

        assert!(plan_vision_image_requests(&[1, 4, 4], &[digest(1), digest(2)], 16, 2).is_err());
        assert!(plan_vision_image_requests(&[1, 3, 4], &[digest(1)], 12, 2).is_err());
        assert!(plan_vision_image_requests(&[1, 4, 4], &[digest(1)], 15, 2).is_err());
        assert!(plan_vision_image_requests(&[0, 4, 4], &[digest(1)], 0, 2).is_err());
    }

    #[test]
    fn partitions_at_boundary_over_boundary_and_single_oversize_image() {
        let a = key(1, [1, 1, 1]);
        let b = key(2, [1, 1, 1]);
        let exact = vec![miss(a, 16_000), miss(b, 16_768)];
        assert_eq!(partition_vision_cache_misses(&exact, 32_768), vec![0..2]);

        let over = vec![miss(a, 16_000), miss(b, 16_769)];
        assert_eq!(
            partition_vision_cache_misses(&over, 32_768),
            vec![0..1, 1..2]
        );

        let oversize = vec![miss(a, 40_000)];
        assert_eq!(partition_vision_cache_misses(&oversize, 32_768), vec![0..1]);
    }

    #[test]
    fn cache_reuses_appended_images_preserves_order_and_deduplicates_misses() {
        let a = key(10, [1, 2, 2]);
        let b = key(20, [1, 2, 2]);
        let c = key(30, [1, 2, 2]);
        let mut cache = VisionCacheInner::new();
        let no_protection = HashSet::new();
        cache.insert(a, feature(1.0), 4, &no_protection, usize::MAX);
        cache.insert(b, feature(2.0), 4, &no_protection, usize::MAX);

        let appended = [request(a, 0, 4), request(b, 4, 4), request(c, 8, 4)];
        let (_, misses, hits) = lookup_vision_feature_cache(&mut cache, &appended);
        assert_eq!(hits, 2);
        assert_eq!(misses.len(), 1);
        assert_eq!(misses[0].request.key, c);

        let reordered = [request(b, 0, 4), request(a, 4, 4)];
        let (ordered, misses, hits) = lookup_vision_feature_cache(&mut cache, &reordered);
        assert_eq!(hits, 2);
        assert!(misses.is_empty());
        assert_eq!(
            ordered[0].as_ref().unwrap().to_float32().unwrap().to_vec(),
            [2.0]
        );
        assert_eq!(
            ordered[1].as_ref().unwrap().to_float32().unwrap().to_vec(),
            [1.0]
        );

        let mut duplicate_cache = VisionCacheInner::new();
        duplicate_cache.insert(b, feature(2.0), 4, &no_protection, usize::MAX);
        let duplicated = [request(a, 0, 4), request(a, 4, 4), request(b, 8, 4)];
        let (_, misses, hits) = lookup_vision_feature_cache(&mut duplicate_cache, &duplicated);
        assert_eq!(hits, 1);
        assert_eq!(misses.len(), 1, "duplicate A should be encoded once");
        assert_eq!(misses[0].request_indices, vec![0, 1]);
    }

    #[test]
    fn cache_key_includes_grid_for_identical_content_hash() {
        let requests = [
            request(key(42, [1, 2, 2]), 0, 4),
            request(key(42, [1, 4, 4]), 4, 16),
        ];
        let mut cache = VisionCacheInner::new();
        let (_, misses, hits) = lookup_vision_feature_cache(&mut cache, &requests);
        assert_eq!(hits, 0);
        assert_eq!(misses.len(), 2);
        assert_ne!(misses[0].request.key, misses[1].request.key);
    }

    #[test]
    fn mixed_hits_and_duplicate_miss_fill_exact_request_order() {
        let a = key(10, [1, 1, 1]);
        let b = key(20, [1, 1, 1]);
        let c = key(30, [1, 1, 1]);
        let mut cache = VisionCacheInner::new();
        let no_protection = HashSet::new();
        cache.insert(b, feature(2.0), 4, &no_protection, usize::MAX);
        cache.insert(c, feature(3.0), 4, &no_protection, usize::MAX);

        let requests = [
            request(b, 0, 1),
            request(a, 1, 1),
            request(a, 2, 1),
            request(c, 3, 1),
        ];
        let (mut ordered, misses, hits) = lookup_vision_feature_cache(&mut cache, &requests);
        assert_eq!(hits, 2);
        assert_eq!(misses.len(), 1);
        assert_eq!(misses[0].request_indices, vec![1, 2]);
        let encoded_a = feature(1.0);
        for &request_index in &misses[0].request_indices {
            ordered[request_index] = Some(encoded_a.clone());
        }
        let ordered: Vec<MxArray> = ordered.into_iter().map(Option::unwrap).collect();
        let refs: Vec<&MxArray> = ordered.iter().collect();
        let concatenated = MxArray::concatenate_many(refs, Some(0)).unwrap();
        assert_eq!(
            concatenated.to_float32().unwrap().to_vec(),
            [2.0, 1.0, 1.0, 3.0]
        );
    }

    #[test]
    fn duplicate_images_count_once_for_cache_floor_but_once_per_request_for_concat() {
        let a = key(10, [1, 1, 1]);
        let b = key(20, [1, 1, 1]);
        let requests = [request(a, 0, 1), request(a, 1, 1), request(b, 2, 1)];
        let (request_bytes, protected_bytes) = projected_vision_feature_bytes(&requests, 8, 2);
        assert_eq!(request_bytes, 48, "concat contains A, A, and B");
        assert_eq!(protected_bytes, 32, "cache stores only unique A and B");
    }

    #[test]
    fn replacing_cache_key_updates_retained_byte_accounting() {
        let a = key(1, [1, 1, 1]);
        let no_protection = HashSet::new();
        let mut cache = VisionCacheInner::new();
        cache.insert(a, feature(1.0), 4, &no_protection, usize::MAX);
        cache.insert(a, feature(2.0), 12, &no_protection, usize::MAX);
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.retained_bytes, 12);
        assert_eq!(cache.get(&a).unwrap().to_float32().unwrap().to_vec(), [2.0]);
    }

    #[test]
    fn later_all_hit_subset_prunes_inactive_oversized_floor() {
        let a = key(1, [1, 1, 1]);
        let b = key(2, [1, 1, 1]);
        let c = key(3, [1, 1, 1]);
        let all_protected: HashSet<_> = [a, b, c].into_iter().collect();
        let mut cache = VisionCacheInner::new();
        cache.insert(a, feature(1.0), 4, &all_protected, 4);
        cache.insert(b, feature(2.0), 4, &all_protected, 4);
        cache.insert(c, feature(3.0), 4, &all_protected, 4);
        assert_eq!(cache.entries.len(), 3, "active request is the floor");
        assert_eq!(cache.retained_bytes, 12);

        let current = [request(a, 0, 1)];
        let (_, misses, hits) = lookup_vision_feature_cache(&mut cache, &current);
        assert_eq!(hits, 1);
        assert!(misses.is_empty());
        let current_protected: HashSet<_> = [a].into_iter().collect();
        let eviction = cache.evict_to_limits(&current_protected, 1, 4);
        assert_eq!(eviction.entries, 2);
        assert_eq!(eviction.bytes, 8);
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.retained_bytes, 4);
        assert!(cache.entries.contains_key(&a));
    }

    #[test]
    fn memory_budget_scales_across_16_32_and_128_gib_caps() {
        let b16 = resolve_vision_memory_budget(
            snapshot(16 * VISION_GIB, 0, 0, 8 * VISION_GIB, 8 * VISION_GIB, 0),
            0,
            0,
            1152,
            4304,
            4,
            3_072,
            0,
        );
        let b32 = resolve_vision_memory_budget(
            snapshot(32 * VISION_GIB, 0, 0, 10 * VISION_GIB, 10 * VISION_GIB, 0),
            0,
            0,
            1152,
            4304,
            4,
            3_072,
            0,
        );
        let b128 = resolve_vision_memory_budget(
            snapshot(128 * VISION_GIB, 0, 0, 40 * VISION_GIB, 40 * VISION_GIB, 0),
            0,
            0,
            1152,
            4304,
            4,
            3_072,
            0,
        );
        assert!(b16.effective_cap_bytes < b32.effective_cap_bytes);
        assert!(b32.effective_cap_bytes < b128.effective_cap_bytes);
        assert!(b16.miss_batch_patch_budget < b32.miss_batch_patch_budget);
        assert!(b32.miss_batch_patch_budget <= b128.miss_batch_patch_budget);
        assert_eq!(b128.miss_batch_patch_budget, 32_768);
        assert!(b16.cache_budget_bytes <= b32.cache_budget_bytes);
        assert!(b32.cache_budget_bytes <= b128.cache_budget_bytes);
        assert_eq!(b128.cache_budget_bytes, VISION_CACHE_MAX_BYTES);
    }

    #[test]
    fn memory_budget_uses_lower_haircut_cap_and_external_metal_usage() {
        let budget = resolve_vision_memory_budget(
            snapshot(
                128 * VISION_GIB,
                64 * VISION_GIB,
                96 * VISION_GIB,
                10 * VISION_GIB,
                18 * VISION_GIB,
                VISION_GIB,
            ),
            0,
            0,
            1152,
            4304,
            2,
            1_536,
            0,
        );
        assert_eq!(budget.effective_cap_bytes, 64 * VISION_GIB);
        assert_eq!(budget.cap_source, VisionMemoryCapSource::MlxMemoryLimit);
        assert_eq!(budget.metal_nonreclaimable_bytes, 17 * VISION_GIB);
        assert_eq!(budget.used_memory_bytes, 17 * VISION_GIB);
    }

    #[test]
    fn memory_budget_fails_closed_for_missing_usage_and_near_cap() {
        let missing =
            resolve_vision_memory_budget(snapshot(0, 0, 0, 0, 0, 0), 0, 0, 1152, 4304, 4, 3_072, 0);
        assert_eq!(
            missing.cap_source,
            VisionMemoryCapSource::ConservativeFallback
        );
        assert!(!missing.usage_probe_available);
        assert_eq!(missing.headroom_bytes, 0);
        assert_eq!(missing.cache_budget_bytes, 0);
        assert_eq!(missing.miss_batch_patch_budget, 0);

        let near_cap = resolve_vision_memory_budget(
            snapshot(0, 16 * VISION_GIB, 0, 15 * VISION_GIB, 15 * VISION_GIB, 0),
            0,
            0,
            1152,
            4304,
            4,
            3_072,
            0,
        );
        assert!(near_cap.usage_probe_available);
        assert_eq!(near_cap.headroom_bytes, 0);
        assert_eq!(near_cap.miss_batch_patch_budget, 0);
    }

    #[test]
    fn memory_budget_rejects_outputs_larger_than_post_reserve_headroom() {
        let budget = resolve_vision_memory_budget(
            snapshot(0, 16 * VISION_GIB, 0, 12 * VISION_GIB, 12 * VISION_GIB, 0),
            0,
            0,
            1152,
            4304,
            4,
            3_072,
            3 * VISION_GIB,
        );
        assert_eq!(budget.output_headroom_bytes, 2 * VISION_GIB);
        assert!(!budget.projected_output_fits);
        assert_eq!(budget.headroom_bytes, 0);
        assert_eq!(budget.miss_batch_patch_budget, 0);
    }

    #[test]
    fn memory_budget_saturates_extreme_geometry_without_overflow() {
        let budget = resolve_vision_memory_budget(
            snapshot(u64::MAX, u64::MAX, u64::MAX, 1, 1, 0),
            0,
            0,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
        );
        assert_eq!(budget.peak_bytes_per_patch, u64::MAX);
        assert_eq!(budget.miss_batch_patch_budget, 0);
        assert!(budget.cache_budget_bytes <= VISION_CACHE_MAX_BYTES);
    }
}

#[cfg(test)]
mod image_placeholder_tests {
    use super::*;

    const BOS: u32 = 1;
    const USER: u32 = 100;
    const TEXT: u32 = 200;
    const IMG: u32 = IMAGE_TOKEN_ID as u32;

    #[test]
    fn expands_single_placeholder_per_image_inline() {
        // Template emitted: BOS, USER, <|image_pad|>, TEXT
        // Expected: BOS, USER, <|image_pad|>×5, TEXT  (vision wrapper stays
        // INSIDE the user turn instead of getting spliced after BOS).
        let tokens = vec![BOS, USER, IMG, TEXT];
        let out = inject_image_placeholders(&tokens, &[5]).unwrap();
        assert_eq!(out, vec![BOS, USER, IMG, IMG, IMG, IMG, IMG, TEXT]);
    }

    #[test]
    fn expands_distinct_grid_counts_for_multiple_images_in_order() {
        // Two images with different grid sizes — each placeholder must be
        // replaced by its own image's count, not the other way around.
        let tokens = vec![BOS, IMG, TEXT, IMG];
        let out = inject_image_placeholders(&tokens, &[2, 3]).unwrap();
        assert_eq!(out, vec![BOS, IMG, IMG, TEXT, IMG, IMG, IMG]);
    }

    #[test]
    fn empty_counts_is_passthrough() {
        let tokens = vec![BOS, USER, TEXT];
        let out = inject_image_placeholders(&tokens, &[]).unwrap();
        assert_eq!(out, tokens);
    }

    #[test]
    fn rejects_template_that_emits_no_image_markers() {
        let tokens = vec![BOS, USER, TEXT];
        let error = inject_image_placeholders(&tokens, &[3]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("model chat template emitted no image placeholder tokens"),
            "unexpected error: {error}",
        );
        assert!(expanded_image_prompt_len(&tokens, &[3]).is_err());
    }

    #[test]
    fn fully_expanded_input_passes_through_unchanged() {
        // Case 2: template already emitted the full 5-token run. `existing`
        // (5) != `per_image.len()` (1) so the "one-per-image" branch
        // doesn't fire; total (5) matches, so the input is preserved.
        let tokens = vec![BOS, USER, IMG, IMG, IMG, IMG, IMG, TEXT];
        let out = inject_image_placeholders(&tokens, &[5]).unwrap();
        assert_eq!(out, tokens);
    }

    #[test]
    fn preserves_relative_position_of_surrounding_tokens() {
        // Regression guard: every non-IMG token must survive in its
        // original relative order.
        let tokens = vec![BOS, USER, 10, 11, IMG, 12, 13];
        let out = inject_image_placeholders(&tokens, &[4]).unwrap();
        assert_eq!(out, vec![BOS, USER, 10, 11, IMG, IMG, IMG, IMG, 12, 13]);
    }

    #[test]
    fn rejects_mismatched_image_marker_count() {
        let tokens = vec![BOS, IMG, IMG, TEXT];
        let error = inject_image_placeholders(&tokens, &[3]).unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("emitted 2 image placeholder token(s)"),
            "unexpected error: {message}",
        );
        assert!(
            message.contains("expected 1 unexpanded marker(s) or 3 already-expanded marker(s)"),
            "unexpected error: {message}",
        );
        assert!(expanded_image_prompt_len(&tokens, &[3]).is_err());
    }

    #[test]
    fn non_allocating_length_plan_matches_placeholder_injection() {
        let cases = [
            (vec![BOS, USER, IMG, TEXT], vec![5]),
            (vec![BOS, IMG, TEXT, IMG], vec![2, 3]),
            (vec![BOS, USER, TEXT], Vec::new()),
            (vec![BOS, USER, IMG, IMG, IMG, IMG, IMG, TEXT], vec![5]),
        ];

        for (tokens, counts) in cases {
            let planned = expanded_image_prompt_len(&tokens, &counts).expect("plan prompt length");
            let expanded =
                inject_image_placeholders(&tokens, &counts).expect("expand image placeholders");
            assert_eq!(
                planned,
                expanded.len(),
                "tokens={tokens:?}, counts={counts:?}"
            );
        }
    }
}

#[cfg(test)]
mod rope_index_tests {
    //! `get_rope_index` builds M-RoPE position IDs for VLM prefill. The
    //! pre-fix implementation collapsed every `IMAGE_TOKEN_ID` in the
    //! prompt to a single contiguous span from `positions[0]` to
    //! `positions[last]`; a multi-turn history with two image-bearing
    //! user turns joined by an assistant reply silently skipped every
    //! interior text token, leaving `all_position_ids` shorter than
    //! `seq_len` and crashing the downstream reshape with a cryptic
    //! "length mismatch". These tests pin per-run indexing against that
    //! regression.
    use super::*;
    use crate::array::MxArray;
    use std::sync::Mutex;
    use std::sync::OnceLock;
    const IMG: i32 = IMAGE_TOKEN_ID;
    const TEXT_A: i32 = 100;
    const TEXT_B: i32 = 200;

    /// MLX's MPS backend is not re-entrant — every test that touches an
    /// `MxArray` must hold this mutex so only one such test runs at a
    /// time across the test binary.
    fn mlx_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    /// Encode a flat `Vec<i32>` token stream as a `[1, seq_len]`
    /// `MxArray` and a `[num_images, 3]` grid array (or `None`) to
    /// feed `get_rope_index`.
    fn mk_inputs(tokens: &[i32], grids: &[(i64, i64, i64)]) -> (MxArray, Option<MxArray>) {
        let seq_len = tokens.len() as i64;
        let input_ids = MxArray::from_int32(tokens, &[1, seq_len]).unwrap();
        let grid = if grids.is_empty() {
            None
        } else {
            let flat: Vec<i32> = grids
                .iter()
                .flat_map(|(t, h, w)| [*t as i32, *h as i32, *w as i32])
                .collect();
            Some(MxArray::from_int32(&flat, &[grids.len() as i64, 3]).unwrap())
        };
        (input_ids, grid)
    }

    fn extract_positions(pos: &MxArray) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        // pos shape [3, 1, seq_len] — flatten to Vec<i32> and split.
        pos.eval();
        let flat = pos.to_int32().unwrap();
        let n = flat.len() / 3;
        (
            flat[0..n].to_vec(),
            flat[n..2 * n].to_vec(),
            flat[2 * n..3 * n].to_vec(),
        )
    }

    #[test]
    fn pure_text_prompt_gets_sequential_positions() {
        let _g = mlx_lock().lock().unwrap();
        let tokens = vec![TEXT_A, TEXT_B, TEXT_A, TEXT_B];
        let (ids, grid) = mk_inputs(&tokens, &[]);
        let (pos, rope_deltas) = get_rope_index(&ids, grid.as_ref(), 2, IMG).unwrap();
        let (t, h, w) = extract_positions(&pos);
        assert_eq!(t, vec![0, 1, 2, 3]);
        assert_eq!(h, vec![0, 1, 2, 3]);
        assert_eq!(w, vec![0, 1, 2, 3]);
        assert_eq!(rope_deltas, 0);
    }

    #[test]
    fn single_image_run_preserves_baseline_shape() {
        let _g = mlx_lock().lock().unwrap();
        // 2 text + (grid 2x2x2=8 tokens after spatial_merge=2, so t=2,h=4,w=4, split=2
        //   → llm grid 2×2×2 = 8 image tokens) + 2 text
        let tokens: Vec<i32> = [TEXT_A, TEXT_B]
            .iter()
            .chain(std::iter::repeat_n(&IMG, 8))
            .chain([TEXT_A, TEXT_B].iter())
            .copied()
            .collect();
        let (ids, grid) = mk_inputs(&tokens, &[(2, 4, 4)]);
        let (pos, rope_deltas) = get_rope_index(&ids, grid.as_ref(), 2, IMG).unwrap();
        let (t, h, w) = extract_positions(&pos);
        // Leading text
        assert_eq!(&t[..2], &[0, 1]);
        assert_eq!(&h[..2], &[0, 1]);
        // Image span starts at 2; with llm_grid_t=2 h=2 w=2 → max_axis=1
        // so current_pos advances to 2 + 1 + 1 = 4 after the image.
        // Trailing text: 4, 5
        assert_eq!(&t[10..], &[4, 5]);
        assert_eq!(&h[10..], &[4, 5]);
        assert_eq!(&w[10..], &[4, 5]);

        // The image run compresses 8 placeholder tokens into 4 distinct
        // temporal positions, so the running M-RoPE counter lags the physical
        // sequence length: `rope_deltas = max_position + 1 - seq_len` MUST be
        // negative. This is the per-session delta the paged decode/warm-
        // continuation path adds to the physical KV slot to recover the
        // compressed rotation position; previously it was dropped, leaving
        // image-turn decode rotating ~|delta| positions too far ahead.
        let max_position = *t.iter().max().unwrap() as i64; // temporal axis (axis 0)
        let seq_len = tokens.len() as i64;
        assert_eq!(rope_deltas, max_position + 1 - seq_len);
        assert!(
            rope_deltas < 0,
            "image prefill must compress positions (rope_deltas={rope_deltas})"
        );
        // 2 text + 8 image (compressed to positions 2..=3) + 2 text → max
        // temporal position 5 over 12 tokens → delta = 5 + 1 - 12 = -6.
        assert_eq!(rope_deltas, -6);
    }

    #[test]
    fn image_final_prompt_delta_uses_global_max_axis() {
        // An image-FINAL prompt (no trailing text) exposes which axis feeds the
        // decode delta: the spatial (h, w) axes outrun the temporal one, so the
        // global max M-RoPE position lives on a spatial axis. The delta must use
        // that global max (mlx-vlm `llm_positions.max()`), NOT the temporal axis
        // alone — otherwise the first generated token rotates at a position
        // INSIDE the image's spatial range instead of at global_max + 1.
        let _g = mlx_lock().lock().unwrap();
        // 1 text + (grid 1x4x4, spatial_merge=2 → llm grid t=1,h=2,w=2 = 4 image
        // tokens) and NOTHING after the image.
        let tokens: Vec<i32> = std::iter::once(TEXT_A)
            .chain(std::iter::repeat_n(IMG, 4))
            .collect();
        let (ids, grid) = mk_inputs(&tokens, &[(1, 4, 4)]);
        let (pos, rope_deltas) = get_rope_index(&ids, grid.as_ref(), 2, IMG).unwrap();
        let (t, h, w) = extract_positions(&pos);

        let t_max = *t.iter().max().unwrap() as i64;
        let global_max = *t.iter().chain(&h).chain(&w).max().unwrap() as i64;
        let seq_len = tokens.len() as i64;

        // The spatial axes must outrun the temporal one here, else the test
        // would not distinguish the global-max fix from the axis-0 regression.
        assert!(
            global_max > t_max,
            "test grid is not asymmetric: global_max={global_max} t_max={t_max}"
        );
        // The delta references the GLOBAL max, not the temporal-axis max.
        assert_eq!(rope_deltas, global_max + 1 - seq_len);
        assert_ne!(
            rope_deltas,
            t_max + 1 - seq_len,
            "delta must not use the temporal axis alone (axis-0 regression)"
        );
    }

    #[test]
    fn two_image_runs_separated_by_text_emits_every_position() {
        // Two image runs separated by interior text must emit a position for
        // EVERY token; a dropped interior-text position makes the downstream
        // reshape in get_rope_index fail with a length mismatch.
        let _g = mlx_lock().lock().unwrap();
        let mut tokens: Vec<i32> = Vec::new();
        tokens.push(TEXT_A); // position 0
        tokens.extend(std::iter::repeat_n(IMG, 8)); // 1 image → llm 2×2×2=8
        tokens.push(TEXT_A); // interior text between images
        tokens.push(TEXT_B);
        tokens.extend(std::iter::repeat_n(IMG, 8)); // 2nd image → same grid
        tokens.push(TEXT_A); // trailing text

        let (ids, grid) = mk_inputs(&tokens, &[(2, 4, 4), (2, 4, 4)]);
        let (pos, _) = get_rope_index(&ids, grid.as_ref(), 2, IMG).unwrap();
        let (t, _h, _w) = extract_positions(&pos);

        // seq_len == tokens.len() — every token must have a position;
        // dropping the interior text entries fails the reshape at the end
        // of get_rope_index.
        assert_eq!(
            t.len(),
            tokens.len(),
            "position count must equal token count"
        );

        // Leading text at pos 0
        assert_eq!(t[0], 0);
        // Image 1 at base=1, max_axis=1 → current_pos after = 3
        // Interior text: 3, 4
        assert_eq!(t[9], 3);
        assert_eq!(t[10], 4);
        // Image 2 at base=5, max_axis=1 → current_pos after = 7
        // Trailing text: 7
        assert_eq!(*t.last().unwrap(), 7);
    }

    #[test]
    fn leading_image_run_no_text_prefix() {
        let _g = mlx_lock().lock().unwrap();
        let tokens: Vec<i32> = std::iter::repeat_n(IMG, 8)
            .chain([TEXT_A, TEXT_B].iter().copied())
            .collect();
        let (ids, grid) = mk_inputs(&tokens, &[(2, 4, 4)]);
        let (pos, _) = get_rope_index(&ids, grid.as_ref(), 2, IMG).unwrap();
        let (t, _, _) = extract_positions(&pos);
        assert_eq!(t.len(), tokens.len());
        // Image at base=0, max_axis=1 → current_pos=2 after, trailing text 2, 3
        assert_eq!(&t[8..], &[2, 3]);
    }

    #[test]
    fn trailing_image_run_no_text_suffix() {
        let _g = mlx_lock().lock().unwrap();
        let tokens: Vec<i32> = [TEXT_A, TEXT_B]
            .iter()
            .copied()
            .chain(std::iter::repeat_n(IMG, 8))
            .collect();
        let (ids, grid) = mk_inputs(&tokens, &[(2, 4, 4)]);
        let (pos, _) = get_rope_index(&ids, grid.as_ref(), 2, IMG).unwrap();
        let (t, _, _) = extract_positions(&pos);
        assert_eq!(t.len(), tokens.len());
        assert_eq!(&t[..2], &[0, 1]);
    }

    #[test]
    fn run_count_must_match_image_count() {
        // 2 image runs in the prompt but only 1 grid supplied — ambiguous
        // pairing; reject.
        let _g = mlx_lock().lock().unwrap();
        let mut tokens = vec![TEXT_A];
        tokens.extend(std::iter::repeat_n(IMG, 4));
        tokens.push(TEXT_A);
        tokens.extend(std::iter::repeat_n(IMG, 4));
        let (ids, grid) = mk_inputs(&tokens, &[(2, 4, 4)]);
        let err = match get_rope_index(&ids, grid.as_ref(), 2, IMG) {
            Ok(_) => panic!("expected get_rope_index to error"),
            Err(e) => e,
        };
        assert!(
            err.reason.contains("Image run layout mismatch"),
            "got: {}",
            err.reason
        );
    }

    #[test]
    fn per_run_length_must_match_its_grid_count() {
        // 2 runs, 2 grids — but run 0 has too few tokens for grid 0.
        let _g = mlx_lock().lock().unwrap();
        let mut tokens = vec![TEXT_A];
        tokens.extend(std::iter::repeat_n(IMG, 4)); // should be 8 for (2,4,4)
        tokens.push(TEXT_A);
        tokens.extend(std::iter::repeat_n(IMG, 12)); // compensates the total, but per-run wrong
        let (ids, grid) = mk_inputs(&tokens, &[(2, 4, 4), (2, 4, 4)]);
        let err = match get_rope_index(&ids, grid.as_ref(), 2, IMG) {
            Ok(_) => panic!("expected get_rope_index to error"),
            Err(e) => e,
        };
        assert!(err.reason.contains("Image run 0"), "got: {}", err.reason);
    }

    #[test]
    fn multi_image_already_expanded_single_contiguous_run_is_accepted() {
        // A checkpoint template may emit one fully expanded contiguous
        // run whose length is `sum(per_image_counts)`. The path
        // synthesises per-image sub-run offsets from the shared span and
        // emits correct M-RoPE positions for each image.
        let _g = mlx_lock().lock().unwrap();
        // Two 1×2×2 grids → 4 image tokens each, 8 total.
        let mut tokens = vec![TEXT_A];
        tokens.extend(std::iter::repeat_n(IMG, 8));
        tokens.push(TEXT_B);
        let (ids, grid) = mk_inputs(&tokens, &[(1, 4, 4), (1, 4, 4)]);
        let (pos, _) = get_rope_index(&ids, grid.as_ref(), 2, IMG)
            .expect("already-expanded single-run layout for two images must be accepted");
        let (t, _, _) = extract_positions(&pos);
        assert_eq!(t.len(), tokens.len(), "every token must have a position");
        // Leading text at 0.
        assert_eq!(t[0], 0);
        // First image base = 1, llm grid 1×2×2, max_axis=1 → current_pos
        // after = 3. Next image base = 3, max_axis=1 → current_pos
        // after = 5. Trailing TEXT_B at 5.
        assert_eq!(*t.last().unwrap(), 5);
    }

    #[test]
    fn multi_image_already_expanded_distinct_grids_preserve_per_image_offsets() {
        // Same already-expanded shape but with different grid sizes per
        // image. The synthesised sub-run offsets must distribute the
        // shared span correctly.
        let _g = mlx_lock().lock().unwrap();
        // image 0: 1×2×2 → 4 tokens. image 1: 1×4×4 → 16 tokens. Total 20.
        let mut tokens = vec![TEXT_A];
        tokens.extend(std::iter::repeat_n(IMG, 20));
        tokens.push(TEXT_B);
        let (ids, grid) = mk_inputs(&tokens, &[(1, 4, 4), (1, 8, 8)]);
        let (pos, _) = get_rope_index(&ids, grid.as_ref(), 2, IMG)
            .expect("already-expanded layout with distinct per-image grids must succeed");
        let (t, _, _) = extract_positions(&pos);
        assert_eq!(t.len(), tokens.len());
        assert_eq!(t[0], 0);
        // image 0 base=1, max_axis = max(0,1,1) = 1 → current_pos = 3
        // image 1 base=3, max_axis = max(0,3,3) = 3 → current_pos = 7
        // Trailing TEXT_B at 7.
        assert_eq!(*t.last().unwrap(), 7);
    }
}

#[cfg(test)]
mod prefix_cache_reuse_integration_tests {
    //! End-to-end tests for the prefix KV cache reuse refactor on
    //! Qwen3.5 Dense. These verify that `chat_session_start_sync` no
    //! longer unconditionally wipes the cache — stateless agent clients
    //! (Aider, Codex CLI, pi-mono, etc.) that resend the full transcript
    //! on every turn should hit the `verify_cache_prefix_direct`
    //! exact-append path and pay only the delta prefill cost.
    //!
    //! These tests are `#[ignore]`-marked because they require loading a
    //! real Qwen3.5 model file and a tokenizer. Run them with:
    //!
    //!     cargo test -p mlx-core --test '*' -- --ignored prefix_cache_reuse_integration
    //!
    //! with `MLX_NODE_QWEN35_MODEL_DIR` set to a local Qwen3.5 dir
    //! (e.g. `~/models/Qwen3.5-0.8B`).
    //!
    //! The test bodies are intentionally skeletal — they document what
    //! needs to hold rather than wiring up the full model-loading
    //! boilerplate. Flesh them out alongside the end-to-end harness in
    //! `serve.ts`.

    /// Append hit: two back-to-back session-start calls where the second's
    /// token sequence is `first_tokens + extra_tokens`. The result of the
    /// second call must report `cached_tokens > 0` and a prefill that
    /// counts only the delta tokens.
    #[ignore = "requires a real Qwen3.5 Dense model directory; run with --ignored"]
    #[test]
    fn append_hit_reuses_cached_prefix() {
        // Pseudocode for the real test body:
        //
        //   let mut model = Qwen35Model::load(env!("MLX_NODE_QWEN35_MODEL_DIR"))?;
        //   let turn1 = vec![ChatMessage::user("What is 2+2?")];
        //   let r1 = model.chat_session_start_sync(turn1, cfg())?;
        //   assert_eq!(r1.cached_tokens, 0);
        //
        //   let turn2 = vec![
        //       ChatMessage::user("What is 2+2?"),
        //       ChatMessage::assistant(&r1.text),
        //       ChatMessage::user("And 3+3?"),
        //   ];
        //   let r2 = model.chat_session_start_sync(turn2, cfg())?;
        //   assert!(r2.cached_tokens > 0);
        //   assert!(r2.performance.as_ref().unwrap().prefill_tokens
        //             < turn2_token_count);  // only the delta was prefilled
    }

    /// Divergence miss: two calls with totally different histories. The
    /// result of the second call must report `cached_tokens == 0` and a
    /// full-history prefill.
    #[ignore = "requires a real Qwen3.5 Dense model directory; run with --ignored"]
    #[test]
    fn divergence_miss_resets_and_full_prefills() {
        // Pseudocode:
        //
        //   let r1 = model.chat_session_start_sync(
        //       vec![ChatMessage::user("Hello")],
        //       cfg(),
        //   )?;
        //   let r2 = model.chat_session_start_sync(
        //       vec![ChatMessage::user("Goodbye")],
        //       cfg(),
        //   )?;
        //   assert_eq!(r2.cached_tokens, 0);
        //   // And the second call must have done a full prefill, not
        //   // attempted to decode from stale caches.
    }
}

#[cfg(test)]
mod paged_construction_tests {
    //! Smoke tests for the block-paged adapter construction on Qwen3.5
    //! dense. The forward dispatch lives in `paged_turn_sync_core`
    //! / `paged_turn_stream_core`; these tests cover the
    //! Inner-construction surface in isolation.
    //!
    //! Tests that allocate a `LayerKVPool` require Metal. Construction-only
    //! cases are `#[ignore]`-marked behind `MLX_TEST_PAGED=1`; forward-path
    //! checks are also ignored because no-Metal hosts can abort inside MLX
    //! before Rust receives an `Err`.

    use super::*;
    use crate::array::DType;
    use crate::models::qwen3_5::config::Qwen3_5Config;
    use crate::models::qwen3_5::decoder_layer::{self, AttentionType};
    use crate::models::qwen3_5::quantized_linear::{
        MXFP8_BITS, MXFP8_GROUP_SIZE, MXFP8_MODE, QuantizedLinear,
    };

    fn tiny_cfg(use_block_paged: bool) -> Qwen3_5Config {
        Qwen3_5Config {
            vocab_size: 1024,
            hidden_size: 64,
            num_layers: 8,
            num_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            head_dim: 16,
            tie_word_embeddings: true,
            attention_bias: false,
            max_position_embeddings: 1024,
            pad_token_id: 0,
            eos_token_id: 0,
            bos_token_id: 0,
            linear_num_value_heads: 4,
            linear_num_key_heads: 2,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            partial_rotary_factor: 0.25,
            rope_theta: 100_000.0,
            paged_cache_memory_mb: Some(64),
            paged_block_size: Some(16),
            use_block_paged_cache: if use_block_paged { Some(true) } else { None },
            persist_paged_cache: None,
            n_mtp_layers: 0,
        }
    }

    fn tiny_paged_forward_cfg() -> Qwen3_5Config {
        let mut cfg = tiny_cfg(true);
        // Paged attention's Metal kernels require head_dim=32+; keep the
        // production-forward tests on a separate shape so construction tests
        // preserve their smaller historical config.
        cfg.hidden_size = 128;
        cfg.intermediate_size = 256;
        cfg.head_dim = 32;
        cfg.linear_key_head_dim = 32;
        cfg.linear_value_head_dim = 32;
        cfg.paged_cache_memory_mb = Some(256);
        cfg
    }

    #[test]
    fn save_model_rejects_quantized_dense_projection_before_creating_destination() {
        let mut inner = Qwen35Inner::new(tiny_cfg(false)).expect("construct tiny dense model");
        let weight = MxArray::zeros(&[128, 16], Some(DType::Uint32)).unwrap();
        let scales = MxArray::zeros(&[128, 2], Some(DType::Uint8)).unwrap();
        let quantized = QuantizedLinear::new(
            weight,
            scales,
            None,
            None,
            MXFP8_GROUP_SIZE,
            MXFP8_BITS,
            MXFP8_MODE.to_string(),
        );
        match &mut inner.layers[3].attn {
            AttentionType::Full(attn) => attn.set_quantized_q_proj(quantized),
            AttentionType::Linear(_) => panic!("layer 3 must be full attention"),
        }

        let destination = std::env::temp_dir().join(format!(
            "mlx_node_dense_quant_save_reject_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        assert!(!destination.exists());
        let err = inner
            .save_model_sync(destination.to_str().unwrap())
            .expect_err("quantized main-model projection must reject dense-only save");
        let message = err.reason.to_string();
        assert!(message.contains("dense/BF16-only"), "{message}");
        assert!(message.contains("layers.3"), "{message}");
        assert!(message.contains("before creating"), "{message}");
        assert!(
            !destination.exists(),
            "rejected save must not create its destination directory"
        );
    }

    fn paged_inner_or_skip(test_name: &str) -> Option<(Qwen35Inner, Qwen3_5Config)> {
        // Only a device-less host licenses a skip. A `LayerKVPool::new: ...
        // must be > 0` means the test config stopped producing a usable pool,
        // and skipping on that is a silent green — see `metal_device_absent`.
        let unavailable = crate::test_support::metal_device_absent;
        let cfg = tiny_paged_forward_cfg();
        match Qwen35Inner::new(cfg.clone()) {
            Ok(mut inner) => match inner.initialize_paged_adapter() {
                Ok(()) => Some((inner, cfg)),
                Err(err) => {
                    let msg = err.reason.to_string();
                    if unavailable(&msg) {
                        eprintln!("skipping {test_name} (paged adapter unavailable): {msg}");
                        None
                    } else {
                        panic!("unexpected paged init failure in {test_name}: {msg}");
                    }
                }
            },
            Err(err) => {
                let msg = err.reason.to_string();
                if unavailable(&msg) {
                    eprintln!("skipping {test_name} (paged adapter unavailable): {msg}");
                    None
                } else {
                    panic!("unexpected Qwen35Inner::new failure in {test_name}: {msg}");
                }
            }
        }
    }

    fn cast_qwen35_inner_weights_bf16(inner: &mut Qwen35Inner) {
        let cast = |a: &MxArray| -> MxArray { a.astype(DType::BFloat16).expect("astype bf16") };

        let w = inner.embedding.get_weight();
        inner.embedding.set_weight(&cast(&w)).expect("set embed");

        let w = inner.final_norm.get_weight();
        inner
            .final_norm
            .set_weight(&cast(&w))
            .expect("set final_norm");

        if let Some(head) = inner.lm_head.as_mut() {
            let w = head.get_weight();
            head.set_weight(&cast(&w), "lm_head").expect("set lm_head");
        }

        for layer in inner.layers.iter_mut() {
            let w = layer.get_input_layernorm_weight();
            layer
                .set_input_layernorm_weight(&cast(&w))
                .expect("set input_layernorm");
            let w = layer.get_post_attention_layernorm_weight();
            layer
                .set_post_attention_layernorm_weight(&cast(&w))
                .expect("set post_attention_layernorm");

            match &mut layer.attn {
                AttentionType::Linear(gdn) => {
                    let w = gdn.get_dt_bias();
                    gdn.set_dt_bias(&cast(&w));
                    let w = gdn.get_a_log();
                    gdn.set_a_log(&cast(&w)).expect("set a_log");
                    let w = gdn.get_in_proj_qkvz_weight();
                    gdn.set_in_proj_qkvz_weight(&cast(&w))
                        .expect("set in_proj_qkvz");
                    let w = gdn.get_in_proj_ba_weight();
                    gdn.set_in_proj_ba_weight(&cast(&w))
                        .expect("set in_proj_ba");
                    let w = gdn.get_conv1d_weight();
                    gdn.set_conv1d_weight(&cast(&w)).expect("set conv1d");
                    let w = gdn.get_norm_weight();
                    gdn.set_norm_weight(&cast(&w)).expect("set gdn norm");
                    let w = gdn.get_out_proj_weight();
                    gdn.set_out_proj_weight(&cast(&w)).expect("set out_proj");
                }
                AttentionType::Full(attn) => {
                    let w = attn.get_q_proj_weight();
                    attn.set_q_proj_weight(&cast(&w)).expect("set q_proj");
                    let w = attn.get_k_proj_weight();
                    attn.set_k_proj_weight(&cast(&w)).expect("set k_proj");
                    let w = attn.get_v_proj_weight();
                    attn.set_v_proj_weight(&cast(&w)).expect("set v_proj");
                    let w = attn.get_o_proj_weight();
                    attn.set_o_proj_weight(&cast(&w)).expect("set o_proj");
                    let w = attn.get_q_norm_weight();
                    attn.set_q_norm_weight(&cast(&w)).expect("set q_norm");
                    let w = attn.get_k_norm_weight();
                    attn.set_k_norm_weight(&cast(&w)).expect("set k_norm");
                }
            }

            let w = layer.mlp.get_gate_proj_weight();
            layer
                .mlp
                .set_gate_proj_weight(&cast(&w))
                .expect("set gate_proj");
            let w = layer.mlp.get_up_proj_weight();
            layer
                .mlp
                .set_up_proj_weight(&cast(&w))
                .expect("set up_proj");
            let w = layer.mlp.get_down_proj_weight();
            layer
                .mlp
                .set_down_proj_weight(&cast(&w))
                .expect("set down_proj");
        }
    }

    fn reset_paged_request(inner: &mut Qwen35Inner, prompt: &[u32]) {
        inner.caches = Some(
            (0..inner.config.num_layers as usize)
                .map(|i| {
                    if inner.config.is_linear_layer(i) {
                        Qwen3_5LayerCache::new_linear()
                    } else {
                        Qwen3_5LayerCache::new_full_attention()
                    }
                })
                .collect(),
        );

        let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
        if adapter.block_table().is_some() {
            adapter.release_request().expect("release_request");
        }
        adapter.reset_for_new_request(0).expect("reset request");
        let prefix = adapter
            .find_cached_prefix(prompt, &[], 0, false)
            .expect("find_cached_prefix");
        assert_eq!(
            prefix.cached_token_count, 0,
            "dense chunking tests must start from a cold adapter prefix"
        );
        adapter
            .allocate_suffix_blocks(prompt.len() as u32)
            .expect("allocate suffix blocks");
    }

    fn run_dense_paged_prefill_with_size(
        inner: &mut Qwen35Inner,
        full_tokens: &[u32],
        suffix_tokens: &[u32],
        cached_prefix_len: u32,
        chunk_size: i32,
    ) -> Result<MxArray> {
        run_dense_paged_prefill_with_size_and_checkpoint(
            inner,
            full_tokens,
            suffix_tokens,
            cached_prefix_len,
            chunk_size,
        )
        .map(|(logits, _checkpoint)| logits)
    }

    fn run_dense_paged_prefill_with_size_and_checkpoint(
        inner: &mut Qwen35Inner,
        full_tokens: &[u32],
        suffix_tokens: &[u32],
        cached_prefix_len: u32,
        chunk_size: i32,
    ) -> Result<(
        MxArray,
        Vec<super::super::paged_forward::MaterializedGdnPrefixCheckpoint>,
    )> {
        let layer_kinds =
            decoder_layer::compute_layer_kinds(inner.config.num_layers as usize, |i| {
                inner.config.is_linear_layer(i)
            });
        let embed = inner.embedding.clone();
        let embedding_weight = embed.get_weight();
        let caches = inner.caches.as_mut().expect("qwen35 caches initialized");
        let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");

        super::super::paged_forward::run_paged_prefill_chunk_with_size(
            full_tokens,
            suffix_tokens,
            cached_prefix_len,
            false,
            &embed,
            &mut inner.layers,
            caches,
            &inner.final_norm,
            &inner.lm_head,
            &embedding_weight,
            &layer_kinds,
            adapter,
            chunk_size,
            /* cached_rope_deltas */ 0,
        )
    }

    /// The planned-MTP twin of [`run_dense_paged_prefill_with_size_and_checkpoint`].
    ///
    /// `run_dense_core_paged_prefill` forks on `keep_prompt_hidden_tokens`: an
    /// AR turn lands in `run_paged_prefill_chunk_with_size`, a planned-MTP turn
    /// in `run_paged_prefill_chunk_with_hidden_with_size`. Those are two
    /// separate bodies that each decide their own checkpoint break set, so a
    /// helper that only reaches the first cannot see the second regress.
    fn run_dense_paged_prefill_with_hidden_and_checkpoint(
        inner: &mut Qwen35Inner,
        full_tokens: &[u32],
        suffix_tokens: &[u32],
        cached_prefix_len: u32,
        chunk_size: i32,
    ) -> Result<(
        MxArray,
        MxArray,
        Vec<super::super::paged_forward::MaterializedGdnPrefixCheckpoint>,
    )> {
        let layer_kinds =
            decoder_layer::compute_layer_kinds(inner.config.num_layers as usize, |i| {
                inner.config.is_linear_layer(i)
            });
        let embed = inner.embedding.clone();
        let embedding_weight = embed.get_weight();
        let keep_tokens = full_tokens.len();
        let caches = inner.caches.as_mut().expect("qwen35 caches initialized");
        let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");

        super::super::paged_forward::run_paged_prefill_chunk_with_hidden_with_size(
            full_tokens,
            suffix_tokens,
            cached_prefix_len,
            false,
            &embed,
            &mut inner.layers,
            caches,
            &inner.final_norm,
            &inner.lm_head,
            &embedding_weight,
            &layer_kinds,
            adapter,
            chunk_size,
            Some(keep_tokens),
            /* cached_rope_deltas */ 0,
        )
    }

    fn logits_to_f32_vec(logits: &MxArray) -> Vec<f32> {
        let f32_arr = logits.astype(DType::Float32).expect("astype f32");
        f32_arr.eval();
        let n = f32_arr.shape_at(0).expect("shape_at(0)") as usize;
        (0..n)
            .map(|i| f32_arr.item_at_float32(i).expect("item_at_float32"))
            .collect()
    }

    fn batch_vocab_logits_to_f32_vec(logits: &MxArray) -> Vec<f32> {
        assert_eq!(logits.ndim().expect("ndim"), 2, "batch logits ndim");
        assert_eq!(logits.shape_at(0).expect("shape_at(0)"), 1);
        let squeezed = logits.squeeze(Some(&[0])).expect("squeeze batch");
        logits_to_f32_vec(&squeezed)
    }

    fn assert_finite_vocab_logits(logits: &MxArray, vocab_size: i32, context: &str) {
        assert_eq!(logits.ndim().expect("ndim"), 1, "{context}: logits ndim");
        assert_eq!(
            logits.shape_at(0).expect("shape_at(0)"),
            vocab_size as i64,
            "{context}: logits shape"
        );
        let values = logits_to_f32_vec(logits);
        for (i, v) in values.iter().enumerate() {
            assert!(v.is_finite(), "{context}: logits[{i}] is not finite: {v}");
        }
    }

    fn assert_finite_batch_vocab_logits(logits: &MxArray, vocab_size: i32, context: &str) {
        assert_eq!(logits.ndim().expect("ndim"), 2, "{context}: logits ndim");
        assert_eq!(
            logits.shape_at(0).expect("shape_at(0)"),
            1,
            "{context}: logits batch"
        );
        assert_eq!(
            logits.shape_at(1).expect("shape_at(1)"),
            vocab_size as i64,
            "{context}: logits vocab"
        );
        let values = batch_vocab_logits_to_f32_vec(logits);
        for (i, v) in values.iter().enumerate() {
            assert!(v.is_finite(), "{context}: logits[{i}] is not finite: {v}");
        }
    }

    fn assert_close_batch_vocab_logits(
        left: &MxArray,
        right: &MxArray,
        abs_tol: f32,
        context: &str,
    ) {
        let left = batch_vocab_logits_to_f32_vec(left);
        let right = batch_vocab_logits_to_f32_vec(right);
        assert_eq!(left.len(), right.len(), "{context}: logits len");
        for (i, (a, b)) in left.iter().zip(right.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff <= abs_tol,
                "{context}: logits[{i}] differ: left={a}, right={b}, abs_diff={diff}, tol={abs_tol}"
            );
        }
    }

    fn assert_finite_hidden(hidden: &MxArray, context: &str) {
        let f32_arr = hidden.astype(DType::Float32).expect("astype hidden f32");
        f32_arr.eval();
        let total = (0..f32_arr.ndim().expect("hidden ndim"))
            .map(|axis| f32_arr.shape_at(axis).expect("hidden shape") as usize)
            .product::<usize>();
        for i in 0..total {
            let value = f32_arr.item_at_float32(i).expect("hidden item");
            assert!(
                value.is_finite(),
                "{context}: hidden[{i}] is not finite: {value}"
            );
        }
    }

    fn reset_dense_caches(inner: &mut Qwen35Inner) {
        inner.caches = Some(fresh_dense_layer_caches(&inner.config));
    }

    fn run_dense_final_logits_legacy_chunked_projection(
        inner: &mut Qwen35Inner,
        prompt: &MxArray,
        embedding_weight: &MxArray,
        embedding_weight_t: &MxArray,
        chunk_size: i64,
    ) -> Result<MxArray> {
        reset_dense_caches(inner);
        let total_len = prompt.shape_at(1)?;
        let chunk_size = if chunk_size <= 0 {
            total_len
        } else {
            chunk_size
        };
        let generation_stream = Stream::new(DeviceType::Gpu);
        let mut offset = 0;
        while total_len - offset > chunk_size {
            let chunk = prompt.slice_axis(1, offset, offset + chunk_size)?;
            {
                let _stream_ctx = StreamContext::new(generation_stream);
                let _logits = forward_inner(
                    &chunk,
                    embedding_weight,
                    &mut inner.layers,
                    &mut inner.caches,
                    &inner.final_norm,
                    &inner.lm_head,
                    Some(embedding_weight_t),
                )?;
            }
            eval_layer_caches(&inner.caches)?;
            crate::array::clear_cache();
            offset += chunk_size;
        }

        let remaining = prompt.slice_axis(1, offset, total_len)?;
        let logits = {
            let _stream_ctx = StreamContext::new(generation_stream);
            forward_inner(
                &remaining,
                embedding_weight,
                &mut inner.layers,
                &mut inner.caches,
                &inner.final_norm,
                &inner.lm_head,
                Some(embedding_weight_t),
            )?
        };
        let seq_len = logits.shape_at(1)?;
        logits
            .slice_axis(1, seq_len - 1, seq_len)?
            .squeeze(Some(&[1]))
    }

    fn run_dense_final_logits_chunked(
        inner: &mut Qwen35Inner,
        prompt: &MxArray,
        embedding_weight: &MxArray,
        embedding_weight_t: &MxArray,
        chunk_size: i64,
    ) -> Result<MxArray> {
        reset_dense_caches(inner);
        chunked_prefill_with_size(
            prompt,
            embedding_weight,
            &mut inner.layers,
            &mut inner.caches,
            &inner.final_norm,
            &inner.lm_head,
            Some(embedding_weight_t),
            Stream::new(DeviceType::Gpu),
            chunk_size,
        )
    }

    /// `use_block_paged_cache` defaults to `None` and round-trips
    /// through serde.
    #[test]
    fn test_use_block_paged_cache_serde_default_none() {
        let json = serde_json::json!({
            "vocab_size": 1024,
            "hidden_size": 64,
            "num_layers": 8,
            "num_heads": 4,
            "num_kv_heads": 2,
            "intermediate_size": 128,
            "rms_norm_eps": 1e-6,
            "head_dim": 16,
            "tie_word_embeddings": true,
            "max_position_embeddings": 1024,
            "pad_token_id": 0,
            "eos_token_id": 0,
            "bos_token_id": 0,
        });
        let cfg: Qwen3_5Config = serde_json::from_value(json).unwrap();
        assert_eq!(
            cfg.use_block_paged_cache, None,
            "use_block_paged_cache must default to None on JSON without the key"
        );
        assert_eq!(cfg.paged_block_size, None);
        assert_eq!(cfg.paged_cache_memory_mb, None);
    }

    #[test]
    fn test_use_block_paged_cache_serde_true_round_trip() {
        let json = serde_json::json!({
            "vocab_size": 1024,
            "hidden_size": 64,
            "num_layers": 8,
            "num_heads": 4,
            "num_kv_heads": 2,
            "intermediate_size": 128,
            "rms_norm_eps": 1e-6,
            "head_dim": 16,
            "tie_word_embeddings": true,
            "max_position_embeddings": 1024,
            "pad_token_id": 0,
            "eos_token_id": 0,
            "bos_token_id": 0,
            "use_block_paged_cache": true,
            "paged_block_size": 16,
            "paged_cache_memory_mb": 256,
        });
        let cfg: Qwen3_5Config = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.use_block_paged_cache, Some(true));
        assert_eq!(cfg.paged_block_size, Some(16));
        assert_eq!(cfg.paged_cache_memory_mb, Some(256));
    }

    #[test]
    fn test_full_attention_layer_count() {
        let cfg = tiny_cfg(false);
        // 8 layers, full_attention_interval=4 → layers 3 and 7 are
        // full-attention (2 layers).
        assert_eq!(cfg.full_attention_layer_count(), 2);
    }

    #[test]
    fn test_dense_gdn_root_rotation_and_legacy_retention_policy() {
        fn push_checkpoint(
            inner: &mut Qwen35Inner,
            owner_id: &str,
            root_owner_id: Option<&str>,
            marker: u32,
        ) {
            crate::engine::backend::ChatBackend::set_cache_owner_id(inner, owner_id, root_owner_id);
            let tokens: Vec<u32> = (0..16).map(|offset| marker * 100 + offset).collect();
            inner
                .gdn_prefix_checkpoints
                .push_back(DenseGdnPrefixCheckpoint {
                    owner_id: inner.active_cache_owner_id.clone(),
                    prefix_len: 16,
                    block_size: 16,
                    final_block_hash: marker as u64,
                    block_hashes: vec![marker as u64],
                    tokens,
                    caches: Vec::new(),
                });
            inner.prune_dense_gdn_prefix_checkpoints();
        }

        let mut inner = Qwen35Inner::new(tiny_cfg(false)).expect("construct tiny dense model");
        push_checkpoint(&mut inner, "root-0", Some("root-0"), 1);
        push_checkpoint(&mut inner, "root-1", Some("root-1"), 2);
        for (index, owner) in ["child-0", "child-1", "child-2", "child-3"]
            .into_iter()
            .enumerate()
        {
            push_checkpoint(&mut inner, owner, Some("root-1"), 10 + index as u32);
        }
        assert_eq!(inner.gdn_root_cache_owner_id.as_deref(), Some("root-1"));
        assert!(inner.gdn_root_cache_owner_is_explicit);
        assert_eq!(inner.gdn_prefix_checkpoints.len(), 5);
        assert!(
            !inner
                .gdn_prefix_checkpoints
                .iter()
                .any(|checkpoint| checkpoint.owner_id == "root-0")
        );
        assert!(
            inner
                .gdn_prefix_checkpoints
                .iter()
                .any(|checkpoint| checkpoint.owner_id == "root-1")
        );

        // Without an explicit root there is no separate global budget, so the
        // owner cap is also the global cap. This model has no paged adapter and
        // therefore no cold GDN policy, so the owner cap is the PRE-LADDER one:
        // nothing here could ever read a ladder, and widening the store on that
        // arm changes which checkpoint a later persistence-OFF turn lands on.
        let mut legacy = Qwen35Inner::new(tiny_cfg(false)).expect("construct legacy dense model");
        assert!(
            !legacy.wants_gdn_checkpoint_ladder(),
            "a model with no paged adapter cannot have a cold GDN sidecar policy"
        );
        for marker in 1..=(GDN_PREFIX_CHECKPOINTS_PER_OWNER as u32 + 1) {
            push_checkpoint(&mut legacy, "", None, marker);
        }
        assert!(!legacy.gdn_root_cache_owner_is_explicit);
        assert_eq!(
            legacy.gdn_prefix_checkpoints.len(),
            GDN_PREFIX_CHECKPOINTS_PER_OWNER_NO_LADDER
        );
    }

    /// The retention cap is chosen at this call site, not inside the store, and
    /// the two arms are reachable only from here: `wants_gdn_checkpoint_ladder`
    /// reads the adapter's installed cold-tier policy, which no store-level test
    /// can see.
    ///
    /// Push the SAME four-rung ladder both ways. With a GDN sidecar policy
    /// installed the whole ladder must survive — that is what the ladder is for.
    /// With the policy removed the same push must collapse to the pre-ladder
    /// number, because a turn that publishes no ladder must retain what it
    /// retained before the ladder existed.
    ///
    /// Catches: hardcoding either arm at the call site. `true` regresses every
    /// persistence-OFF request's emitted tokens; `false` silently reverts
    /// persist-ON to single-endpoint retention.
    #[test]
    fn dense_gdn_retention_follows_the_installed_cold_sidecar_policy() {
        let Some(mut inner) = dense_inner_with_test_adapter_or_skip(
            "dense_gdn_retention_follows_the_installed_cold_sidecar_policy",
        ) else {
            return;
        };
        let tokens: Vec<u32> = (0..64u32).collect();
        let extra_keys = vec![Vec::new(); 4];

        let push_ladder = |inner: &mut Qwen35Inner| {
            inner.gdn_prefix_checkpoints.clear();
            crate::engine::backend::ChatBackend::set_cache_owner_id(inner, "", None);
            for rung in 1..=4u32 {
                let prefix_len = rung * 16;
                let block_hashes =
                    compute_paged_prefix_block_hashes(&tokens, prefix_len, 16, &extra_keys, 0)
                        .expect("block-aligned rung must hash");
                inner
                    .gdn_prefix_checkpoints
                    .push_back(DenseGdnPrefixCheckpoint {
                        owner_id: inner.active_cache_owner_id.clone(),
                        prefix_len,
                        block_size: 16,
                        final_block_hash: block_hashes.last().copied().unwrap_or_default(),
                        block_hashes,
                        tokens: tokens[..prefix_len as usize].to_vec(),
                        caches: Vec::new(),
                    });
                inner.prune_dense_gdn_prefix_checkpoints();
            }
            inner.gdn_prefix_checkpoints.len()
        };

        assert!(!inner.wants_gdn_checkpoint_ladder());
        assert_eq!(
            push_ladder(&mut inner),
            GDN_PREFIX_CHECKPOINTS_PER_OWNER_NO_LADDER,
            "with no cold GDN policy the store must stay at the pre-ladder cap"
        );

        let root = temp_cold_root("dense-gdn-retention-policy");
        install_gdn_cold_tier(&mut inner, &root);
        assert!(inner.wants_gdn_checkpoint_ladder());
        assert_eq!(
            push_ladder(&mut inner),
            GDN_PREFIX_CHECKPOINTS_PER_OWNER,
            "a cold GDN sidecar policy must keep the whole published ladder"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The retention policy lives in `gdn_checkpoint_store`, but WHICH owner it
    /// protects is decided here, by what this call site passes. A twin that
    /// hands `prune_gdn_checkpoints` the root where the active owner belongs
    /// collapses a subagent's ladder to its endpoint rung while every
    /// store-level test stays green, so drive it through the model.
    ///
    /// The root is seeded with a spare rung on purpose: that is the redundancy
    /// the first arm spends, and without it there is nothing for the arm to
    /// prefer and the mutation is invisible.
    ///
    /// The cold tier is installed for the same reason. Publisher deferral is a
    /// `GdnRetentionPolicy::Ladder` behaviour and `active_owner_id` is read on
    /// no other arm, so a turn with no GDN sidecar policy cannot express the
    /// mutation at all — this test used to run on that arm and asserted the
    /// ladder answer anyway, which only agreed while the policy was ungated.
    /// The persistence-OFF push below pins the other arm's shape rather than
    /// leaving it unmeasured.
    #[test]
    fn test_dense_ladder_survives_sibling_owners_through_the_model_call_site() {
        fn push(inner: &mut Qwen35Inner, owner_id: &str, root_owner_id: &str, blocks: u32) {
            crate::engine::backend::ChatBackend::set_cache_owner_id(
                inner,
                owner_id,
                Some(root_owner_id),
            );
            inner
                .gdn_prefix_checkpoints
                .push_back(DenseGdnPrefixCheckpoint {
                    owner_id: inner.active_cache_owner_id.clone(),
                    prefix_len: blocks * 16,
                    block_size: 16,
                    final_block_hash: u64::from(blocks),
                    block_hashes: (1..=u64::from(blocks)).collect(),
                    tokens: (0..blocks * 16).collect(),
                    caches: Vec::new(),
                });
            inner.prune_dense_gdn_prefix_checkpoints();
        }

        /// Root seeded with a spare rung, two siblings holding one each, then a
        /// third subagent publishing a whole quartered ladder rung by rung the
        /// way `publish_dense_gdn_materialized_prefix_checkpoint_with_keys`
        /// does. Every slot is taken before the ladder starts.
        fn run_fleet(inner: &mut Qwen35Inner) -> Vec<(String, u32)> {
            inner.gdn_prefix_checkpoints.clear();
            push(inner, "root", "root", 1);
            push(inner, "root", "root", 64);
            for sibling in ["child-0", "child-1"] {
                push(inner, sibling, "root", 64);
            }
            for blocks in [1, 4, 16, 64] {
                push(inner, "child-3", "root", blocks);
            }
            inner
                .gdn_prefix_checkpoints
                .iter()
                .map(|checkpoint| (checkpoint.owner_id.clone(), checkpoint.prefix_len))
                .collect()
        }

        let Some(mut inner) = dense_inner_with_test_adapter_or_skip(
            "test_dense_ladder_survives_sibling_owners_through_the_model_call_site",
        ) else {
            return;
        };

        // Persistence OFF: no deferral, and the pre-ladder per-owner cap of 2.
        assert!(!inner.wants_gdn_checkpoint_ladder());
        assert_eq!(
            run_fleet(&mut inner),
            vec![
                ("root".to_string(), 1024),
                ("child-0".to_string(), 1024),
                ("child-1".to_string(), 1024),
                ("child-3".to_string(), 256),
                ("child-3".to_string(), 1024),
            ],
            "a persistence-OFF turn takes the pre-ladder victim order, which \
             keeps the publisher's DEEPEST rungs and defers nobody"
        );

        // Persistence ON: the publisher's shallow rung is what this turn's cold
        // capture anchors on, so it outlives the root's spare rung.
        let root = temp_cold_root("dense-ladder-sibling-call-site");
        install_gdn_cold_tier(&mut inner, &root);
        assert!(inner.wants_gdn_checkpoint_ladder());
        assert_eq!(
            run_fleet(&mut inner),
            vec![
                ("root".to_string(), 1024),
                ("child-0".to_string(), 1024),
                ("child-1".to_string(), 1024),
                ("child-3".to_string(), 16),
                ("child-3".to_string(), 1024),
            ],
            "the publishing subagent must keep the shallow rung this turn's \
             capture anchors on, paid for by the root's spare rung, and no \
             sibling may be pushed to zero"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_qwen35_media_plan_requires_complete_paged_vision_stack() {
        for has_encoder in [false, true] {
            for has_processor in [false, true] {
                for has_paged in [false, true] {
                    let plan = qwen35_dense_media_plan(has_encoder, has_processor, has_paged);
                    let available = has_encoder && has_processor && has_paged;
                    assert_eq!(plan.available.images, available);
                    assert!(!plan.available.audio);
                    assert_eq!(plan.backend_validated.images, !available);
                    assert!(!plan.backend_validated.audio);
                    assert!(plan.admitted().images);
                }
            }
        }
    }

    #[test]
    fn test_qwen35_session_media_survives_paged_image_key_clear() {
        assert_eq!(
            qwen35_dense_session_media(true, false),
            MediaCapabilities::IMAGES
        );
        assert_eq!(
            qwen35_dense_session_media(false, true),
            MediaCapabilities::IMAGES,
            "a retained M-RoPE delta proves successive paged text turns still extend image KV"
        );
        assert_eq!(
            qwen35_dense_session_media(false, false),
            MediaCapabilities::NONE
        );
    }

    #[test]
    fn test_qwen35_session_media_payload_identity() {
        let images = vec![vec![1, 2, 3]];
        let cached_key = Some(engine::compute_image_cache_key(&images));

        assert!(qwen35_dense_session_media_matches_payloads(
            cached_key,
            &images,
            &[]
        ));
        assert!(!qwen35_dense_session_media_matches_payloads(
            cached_key,
            &[vec![1, 2, 4]],
            &[]
        ));
        assert!(!qwen35_dense_session_media_matches_payloads(
            cached_key,
            &images,
            &[vec![9]]
        ));
        assert!(!qwen35_dense_session_media_matches_payloads(
            None,
            &images,
            &[]
        ));
    }

    #[test]
    fn test_dense_paged_finalize_failure_cannot_republish_image_session() {
        let mut inner = Qwen35Inner::new(tiny_cfg(false)).expect("construct tiny dense model");
        inner.caches = Some(fresh_dense_layer_caches(&inner.config));
        inner.cached_token_history = vec![7, 248_056, 248_056, 8];
        inner.cached_image_key = Some(0xD35E);
        inner.cached_paged_image_token_positions = vec![(1, 0xA11C), (2, 0xA11C)];
        inner.cached_rope_deltas = Some(-2);

        // A missing adapter exercises the same infallible-hook downgrade used
        // for registration/evaluation/release failures without allocating a
        // Metal LayerKVPool in this unit test.
        <Qwen35Inner as crate::engine::backend::PagedBackend>::finalize_paged_turn(
            &mut inner, true,
        );
        assert!(inner.paged_finalize_failed);
        assert!(inner.caches.is_none());
        assert!(inner.cached_token_history.is_empty());
        assert!(inner.cached_image_key.is_none());
        assert!(inner.cached_paged_image_token_positions.is_empty());
        assert!(inner.cached_rope_deltas.is_none());

        // The engine always calls save after the infallible finalize hook. It
        // must consume the failure latch rather than reviving the session from
        // expanded image-placeholder ids.
        <Qwen35Inner as crate::engine::backend::PagedBackend>::save_paged_history(
            &mut inner,
            &[7, 248_056, 248_056, 8],
            &[9],
            false,
            true,
        )
        .expect("failed finalization must downgrade history save");
        assert!(!inner.paged_finalize_failed);
        assert!(inner.cached_token_history.is_empty());
        assert!(!crate::engine::backend::ChatBackend::has_live_session(
            &inner
        ));
        assert_eq!(
            crate::engine::backend::ChatBackend::session_media(&inner),
            MediaCapabilities::NONE
        );
    }

    fn seed_dense_paged_image_session(inner: &mut Qwen35Inner) {
        inner.caches = Some(fresh_dense_layer_caches(&inner.config));
        inner.cached_token_history = vec![7, 248_056, 248_056, 8];
        inner.cached_image_key = Some(0xD35E);
        inner.cached_paged_image_token_positions = vec![(1, 0xA11C), (2, 0xA11C)];
        inner.cached_rope_deltas = Some(-2);
        inner.paged_full_attn_caches_dirty = true;
        inner.flat_mtp_caches_desynced = true;
    }

    #[test]
    fn test_dense_manual_paged_finalize_failure_invalidates_session() {
        let mut inner = Qwen35Inner::new(tiny_cfg(false)).expect("construct tiny dense model");
        seed_dense_paged_image_session(&mut inner);

        let error = inner
            .finalize_dense_manual_paged_turn(&[(1, 0xA11C), (2, 0xA11C)])
            .expect_err("missing adapter must fail manual finalization");
        assert!(error.to_string().contains("paged finalization failed"));
        assert!(inner.caches.is_none());
        assert!(inner.cached_token_history.is_empty());
        assert!(inner.cached_image_key.is_none());
        assert!(inner.cached_paged_image_token_positions.is_empty());
        assert!(inner.cached_rope_deltas.is_none());
        assert!(!inner.paged_full_attn_caches_dirty);
        assert!(!inner.flat_mtp_caches_desynced);
    }

    #[test]
    fn test_dense_vlm_prefix_prepare_failure_invalidates_session() {
        let mut inner = Qwen35Inner::new(tiny_cfg(false)).expect("construct tiny dense model");
        seed_dense_paged_image_session(&mut inner);

        inner
            .prepare_dense_vlm_paged_prefix(
                &[7, 248_056, 248_056, 8],
                4,
                16,
                &[vec![0xA11C]],
                true,
                true,
                0xD35E,
            )
            .expect_err("missing adapter must fail VLM prefix preparation");
        assert!(inner.caches.is_none());
        assert!(inner.cached_token_history.is_empty());
        assert!(inner.cached_image_key.is_none());
        assert!(inner.cached_paged_image_token_positions.is_empty());
        assert!(inner.cached_rope_deltas.is_none());
        assert!(!inner.paged_full_attn_caches_dirty);
        assert!(!crate::engine::backend::ChatBackend::has_live_session(
            &inner
        ));
    }

    #[test]
    fn test_dense_generic_paged_abort_invalidates_session() {
        let mut inner = Qwen35Inner::new(tiny_cfg(false)).expect("construct tiny dense model");
        seed_dense_paged_image_session(&mut inner);

        <Qwen35Inner as crate::engine::backend::PagedBackend>::abort_paged_turn(&mut inner);
        assert!(inner.caches.is_none());
        assert!(inner.cached_token_history.is_empty());
        assert!(inner.cached_image_key.is_none());
        assert!(inner.cached_paged_image_token_positions.is_empty());
        assert!(inner.cached_rope_deltas.is_none());
        assert!(!inner.paged_full_attn_caches_dirty);
        assert!(!inner.flat_mtp_caches_desynced);
    }

    #[test]
    fn test_qwen35_planned_decoder_overrides_raw_mtp_flag() {
        let mut config = ChatConfig {
            cache_owner_id: None,
            cache_root_owner_id: None,
            enable_mtp: Some(true),
            ..ChatConfig::default()
        };
        assert!(!apply_qwen35_dense_planned_decoder(
            &mut config,
            DecoderPlan::Autoregressive
        ));
        assert_eq!(config.enable_mtp, Some(false));

        assert!(apply_qwen35_dense_planned_decoder(
            &mut config,
            DecoderPlan::Speculative(SpeculativeKind::NativeMtp)
        ));
        assert_eq!(config.enable_mtp, Some(true));

        assert!(!apply_qwen35_dense_planned_decoder(
            &mut config,
            DecoderPlan::Speculative(SpeculativeKind::DraftModel)
        ));
        assert_eq!(config.enable_mtp, Some(false));
    }

    /// When `use_block_paged_cache` is `None`, `paged_adapter` is None.
    #[test]
    fn test_inner_no_paged_adapter_when_flag_is_none() {
        let cfg = tiny_cfg(false);
        let inner =
            Qwen35Inner::new(cfg).expect("Qwen35Inner::new must succeed without paged adapter");
        assert!(
            inner.paged_adapter.is_none(),
            "paged_adapter must be None when use_block_paged_cache is None"
        );
    }

    #[test]
    fn test_fresh_dense_layer_caches_are_not_gdn_reuse_ready() {
        let cfg = tiny_cfg(true);
        let caches = fresh_dense_layer_caches(&cfg);
        assert_eq!(caches.len(), cfg.num_layers as usize);
        assert!(
            !dense_paged_linear_caches_ready(&cfg, Some(&caches)),
            "fresh linear caches have empty conv/recurrent slots, so a live continuation must replay GDN"
        );
        assert!(matches!(caches[0], Qwen3_5LayerCache::Linear(_)));
        assert!(matches!(caches[3], Qwen3_5LayerCache::FullAttention(_)));
    }

    fn run_dense_chunked_prefill_final_logits_only_matches_legacy_chunking() -> Result<()> {
        let mut inner = Qwen35Inner::new(tiny_cfg(false))?;
        cast_qwen35_inner_weights_bf16(&mut inner);

        let prompt_tokens: Vec<u32> = (0u32..33).map(|i| (i * 17 + 5) % 997).collect();
        let prompt = MxArray::from_uint32(&prompt_tokens, &[1, prompt_tokens.len() as i64])?;
        let embedding_weight = inner.embedding.get_weight();
        let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;

        let expected = run_dense_final_logits_legacy_chunked_projection(
            &mut inner,
            &prompt,
            &embedding_weight,
            &embedding_weight_t,
            16,
        )?;
        assert_finite_batch_vocab_logits(
            &expected,
            inner.config.vocab_size,
            "legacy chunked final logits",
        );

        let chunked = run_dense_final_logits_chunked(
            &mut inner,
            &prompt,
            &embedding_weight,
            &embedding_weight_t,
            16,
        )?;
        assert_finite_batch_vocab_logits(&chunked, inner.config.vocab_size, "chunked final logits");
        assert_close_batch_vocab_logits(
            &expected,
            &chunked,
            1e-6,
            "chunked final logits vs legacy chunking",
        );

        Ok(())
    }

    #[test]
    fn test_dense_chunked_prefill_final_logits_only_matches_legacy_chunking() {
        if let Err(err) = run_dense_chunked_prefill_final_logits_only_matches_legacy_chunking() {
            let msg = err.reason.to_string();
            if msg.contains("Metal") || msg.contains("device") {
                eprintln!(
                    "skipping test_dense_chunked_prefill_final_logits_only_matches_legacy_chunking: {msg}"
                );
                return;
            }
            panic!("unexpected dense chunked prefill failure: {msg}");
        }
    }

    fn run_dense_chunked_prefill_with_hidden_keeps_tail_contract() -> Result<()> {
        let mut inner = Qwen35Inner::new(tiny_cfg(false))?;
        cast_qwen35_inner_weights_bf16(&mut inner);

        let prompt_tokens: Vec<u32> = (0u32..35).map(|i| (i * 23 + 3) % 997).collect();
        let prompt = MxArray::from_uint32(&prompt_tokens, &[1, prompt_tokens.len() as i64])?;
        let embedding_weight = inner.embedding.get_weight();
        let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;

        reset_dense_caches(&mut inner);
        let (logits, hidden) = chunked_prefill_with_hidden_with_size(
            &prompt,
            &embedding_weight,
            &mut inner.layers,
            &mut inner.caches,
            &inner.final_norm,
            &inner.lm_head,
            Some(&embedding_weight_t),
            Stream::new(DeviceType::Gpu),
            Some(5),
            16,
        )?;
        assert_finite_batch_vocab_logits(
            &logits,
            inner.config.vocab_size,
            "chunked hidden final logits",
        );
        assert_eq!(hidden.ndim()?, 3, "prompt hidden ndim");
        assert_eq!(hidden.shape_at(0)?, 1, "prompt hidden batch");
        assert_eq!(hidden.shape_at(1)?, 5, "prompt hidden tail len");
        assert_eq!(
            hidden.shape_at(2)?,
            inner.config.hidden_size as i64,
            "prompt hidden width"
        );
        assert_finite_hidden(&hidden, "chunked hidden tail");

        let logits_without_hidden = run_dense_final_logits_chunked(
            &mut inner,
            &prompt,
            &embedding_weight,
            &embedding_weight_t,
            16,
        )?;
        assert_close_batch_vocab_logits(
            &logits,
            &logits_without_hidden,
            1e-6,
            "hidden and logits-only chunked final logits",
        );

        reset_dense_caches(&mut inner);
        let (_logits, full_tail_hidden) = chunked_prefill_with_hidden_with_size(
            &prompt,
            &embedding_weight,
            &mut inner.layers,
            &mut inner.caches,
            &inner.final_norm,
            &inner.lm_head,
            Some(&embedding_weight_t),
            Stream::new(DeviceType::Gpu),
            Some(100),
            16,
        )?;
        assert_eq!(
            full_tail_hidden.shape_at(1)?,
            prompt_tokens.len() as i64,
            "oversized keep window keeps the whole prompt"
        );
        assert_finite_hidden(&full_tail_hidden, "chunked full prompt hidden");

        Ok(())
    }

    #[test]
    fn test_dense_chunked_prefill_with_hidden_keeps_tail_contract() {
        if let Err(err) = run_dense_chunked_prefill_with_hidden_keeps_tail_contract() {
            let msg = err.reason.to_string();
            if msg.contains("Metal") || msg.contains("device") {
                eprintln!(
                    "skipping test_dense_chunked_prefill_with_hidden_keeps_tail_contract: {msg}"
                );
                return;
            }
            panic!("unexpected dense chunked prefill hidden failure: {msg}");
        }
    }

    #[test]
    fn test_dense_paged_prefix_block_hash_matches_allocator_chain() {
        let tokens: Vec<u32> = (1..=12).collect();
        let per_block = vec![vec![11], vec![], vec![33, 44]];

        let h0 = mlx_paged_attn::hash_tokens(&tokens[0..4], 0, &per_block[0]);
        let h1 = mlx_paged_attn::hash_tokens(&tokens[4..8], h0, &per_block[1]);
        let h2 = mlx_paged_attn::hash_tokens(&tokens[8..12], h1, &per_block[2]);

        assert_eq!(
            compute_paged_prefix_block_hash(&tokens, 12, 4, &per_block, 0),
            Some(h2)
        );
    }

    #[test]
    fn test_dense_paged_prefix_block_hash_applies_salt_to_first_block_only() {
        let tokens: Vec<u32> = (1..=8).collect();
        let per_block = vec![vec![11], vec![22]];
        let salt = 99;

        let mut first_block_keys = per_block[0].clone();
        first_block_keys.push(salt);
        let h0 = mlx_paged_attn::hash_tokens(&tokens[0..4], 0, &first_block_keys);
        let h1 = mlx_paged_attn::hash_tokens(&tokens[4..8], h0, &per_block[1]);

        assert_eq!(
            compute_paged_prefix_block_hash(&tokens, 8, 4, &per_block, salt),
            Some(h1)
        );
    }

    #[test]
    fn test_dense_paged_prefix_block_hash_rejects_non_full_or_unkeyed_prefix() {
        let tokens: Vec<u32> = (1..=8).collect();
        let per_block = vec![vec![]];

        assert_eq!(
            compute_paged_prefix_block_hash(&tokens, 6, 4, &per_block, 0),
            None
        );
        assert_eq!(
            compute_paged_prefix_block_hash(&tokens, 8, 4, &per_block, 0),
            None
        );
    }

    /// Allocates a `LayerKVPool`. Requires Metal; gate on
    /// `MLX_TEST_PAGED=1`.
    #[test]
    #[ignore = "Allocates Metal LayerKVPool; gate on MLX_TEST_PAGED=1"]
    fn test_inner_constructs_paged_adapter_when_flag_is_true() {
        if std::env::var_os("MLX_TEST_PAGED").is_none() {
            return;
        }
        let cfg = tiny_cfg(true);
        let mut inner = Qwen35Inner::new(cfg).expect(
            "Qwen35Inner::new with use_block_paged_cache=true must succeed on Metal-capable host",
        );
        inner
            .initialize_paged_adapter()
            .expect("post-load paged adapter initialization must succeed");
        assert!(
            inner.paged_adapter.is_some(),
            "paged_adapter must be Some when use_block_paged_cache = Some(true)"
        );
    }

    /// VLM checkpoints are accepted under paged dispatch. The media plan is
    /// not executable with only an encoder; it becomes available only after
    /// the processor completes the paged vision stack.
    #[test]
    #[ignore = "Allocates Metal LayerKVPool; gate on MLX_TEST_PAGED=1"]
    fn test_vlm_loads_when_paged_enabled() {
        if std::env::var_os("MLX_TEST_PAGED").is_none() {
            return;
        }
        use crate::models::qwen3_5::vision::Qwen3_5VisionConfig;
        use crate::models::qwen3_5::vision::Qwen3_5VisionEncoder;

        let cfg = tiny_cfg(true);
        let mut inner = Qwen35Inner::new(cfg).unwrap();
        inner
            .initialize_paged_adapter()
            .expect("post-load paged adapter initialization");
        let vision_cfg = Qwen3_5VisionConfig {
            hidden_size: 64,
            intermediate_size: 256,
            num_heads: 4,
            num_layers: 2,
            patch_size: 16,
            spatial_merge_size: 2,
            image_size: 256,
            out_hidden_size: 64,
        };
        let vision_enc =
            Qwen3_5VisionEncoder::new(vision_cfg).expect("vision encoder construction");
        let result = inner.set_vision_encoder(vision_enc);
        assert!(
            result.is_ok(),
            "set_vision_encoder must succeed when paged_adapter is Some so VLM \
             checkpoints can complete their paged media stack; got {result:?}"
        );
        assert!(
            inner.vision_encoder.is_some(),
            "vision_encoder field must be populated after a successful set"
        );
        let incomplete = inner.execution_plan().media;
        assert_eq!(incomplete.available, MediaCapabilities::NONE);
        assert_eq!(incomplete.backend_validated, MediaCapabilities::IMAGES);

        inner.set_image_processor(Qwen35VLImageProcessor::new(None));
        let complete = inner.execution_plan().media;
        assert_eq!(complete.available, MediaCapabilities::IMAGES);
        assert_eq!(complete.backend_validated, MediaCapabilities::NONE);
    }

    /// Dense Qwen3.5 paged-prefill chunking state test. This drives the
    /// production chunk-size worker once and asserts the adapter cursor,
    /// request token log, and block table cover the whole prompt after all
    /// chunks have been recorded.
    #[test]
    #[ignore = "requires Metal GPU; run with --ignored"]
    fn test_dense_paged_prefill_chunks_advance_adapter_state() {
        let Some((mut inner, cfg)) =
            paged_inner_or_skip("test_dense_paged_prefill_chunks_advance_adapter_state")
        else {
            return;
        };
        cast_qwen35_inner_weights_bf16(&mut inner);

        let prompt: Vec<u32> = (0u32..64).map(|i| (i * 5 + 7) % 257).collect();
        reset_paged_request(&mut inner, &prompt);

        let logits = match run_dense_paged_prefill_with_size(
            &mut inner, &prompt, &prompt, 0, /* chunk_size */ 16,
        ) {
            Ok(logits) => logits,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("Metal GPU not available") || msg.contains("No Metal device") {
                    eprintln!(
                        "skipping test_dense_paged_prefill_chunks_advance_adapter_state: {msg}"
                    );
                    return;
                }
                panic!("unexpected dense paged chunk failure: {msg}");
            }
        };

        let adapter = inner.paged_adapter.as_ref().expect("paged_adapter");
        assert_eq!(
            adapter.current_token_count() as usize,
            prompt.len(),
            "adapter cursor after chunked prefill"
        );
        assert_eq!(
            adapter.request_tokens(),
            prompt.as_slice(),
            "request token log after chunked prefill"
        );
        let block_table = adapter.block_table().expect("block_table");
        let expected_min_blocks = prompt.len().div_ceil(adapter.block_size() as usize);
        assert!(
            block_table.num_blocks() >= expected_min_blocks,
            "block table has {} blocks, expected at least {expected_min_blocks}",
            block_table.num_blocks()
        );
        assert_finite_vocab_logits(&logits, cfg.vocab_size, "final dense paged chunk prefill");

        let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
        let _ = adapter.register_full_blocks_for_reuse(&[], 0);
        adapter.release_request().expect("release_request");
    }

    /// Uneven-tail coverage for dense Qwen3.5 paged prefill: a 33-token
    /// prompt with chunk_size=16 must record two full chunks plus a
    /// one-token tail and return valid logits for the tail chunk.
    #[test]
    #[ignore = "requires Metal GPU; run with --ignored"]
    fn test_dense_paged_prefill_chunks_handle_uneven_tail() {
        let Some((mut inner, cfg)) =
            paged_inner_or_skip("test_dense_paged_prefill_chunks_handle_uneven_tail")
        else {
            return;
        };
        cast_qwen35_inner_weights_bf16(&mut inner);

        let prompt: Vec<u32> = (0u32..33).map(|i| (i * 11 + 3) % 257).collect();
        reset_paged_request(&mut inner, &prompt);

        let final_logits = run_dense_paged_prefill_with_size(
            &mut inner, &prompt, &prompt, 0, /* chunk_size */ 16,
        )
        .expect("dense paged uneven-tail chunked prefill");

        assert_eq!(
            prompt.len(),
            33,
            "test setup must exercise a one-token tail"
        );
        let adapter = inner.paged_adapter.as_ref().expect("paged_adapter");
        assert_eq!(
            adapter.current_token_count(),
            33,
            "adapter cursor must include the uneven tail"
        );
        assert_eq!(
            adapter.request_tokens(),
            prompt.as_slice(),
            "request token log must include the uneven tail"
        );
        assert_finite_vocab_logits(
            &final_logits,
            cfg.vocab_size,
            "uneven-tail dense paged chunk prefill",
        );

        let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
        let _ = adapter.register_full_blocks_for_reuse(&[], 0);
        adapter.release_request().expect("release_request");
    }

    /// Regression coverage for Pi-style agent turns: a block-aligned prefill
    /// must retain GDN state at the largest reusable paged-block boundary (16
    /// of 32 tokens here). A one-block rollback restores exactly, while a
    /// full-boundary hit replays only one GDN block.
    #[test]
    #[ignore = "requires Metal GPU; run with --ignored"]
    fn test_dense_paged_prefill_captures_exact_gdn_block_checkpoint() {
        let Some((mut inner, cfg)) =
            paged_inner_or_skip("test_dense_paged_prefill_captures_exact_gdn_block_checkpoint")
        else {
            return;
        };
        cast_qwen35_inner_weights_bf16(&mut inner);

        let prompt: Vec<u32> = (0u32..32).map(|i| (i * 13 + 9) % 257).collect();
        reset_paged_request(&mut inner, &prompt);

        let (logits, checkpoint) = run_dense_paged_prefill_with_size_and_checkpoint(
            &mut inner, &prompt, &prompt, 0, /* chunk_size */ 16,
        )
        .expect("dense paged checkpoint prefill");
        assert_finite_vocab_logits(&logits, cfg.vocab_size, "checkpoint prefill logits");

        let mut checkpoint = checkpoint;
        let checkpoint = checkpoint
            .pop()
            .expect("stable reusable-block GDN checkpoint");
        assert_eq!(checkpoint.prefix_len, 16);
        assert!(dense_paged_linear_caches_ready(
            &inner.config,
            Some(&checkpoint.caches)
        ));

        let block_size = inner
            .paged_adapter
            .as_ref()
            .expect("paged_adapter")
            .block_size();
        let extra_keys = engine::build_paged_extra_keys(prompt.len(), block_size, &[]);
        assert!(inner.remember_dense_gdn_materialized_prefix_checkpoint(
            &prompt,
            block_size,
            &extra_keys,
            0,
            checkpoint,
        ));
        inner.active_cache_owner_id = "child-session".to_owned();
        assert!(
            inner
                .find_dense_gdn_prefix_checkpoint(&prompt, 16, block_size, &extra_keys, 0)
                .is_none(),
            "an exact token/hash checkpoint owned by another session must not restore"
        );
        inner.active_cache_owner_id.clear();
        let restored = inner
            .find_dense_gdn_prefix_checkpoint(&prompt, 16, block_size, &extra_keys, 0)
            .expect("exact checkpoint restore");
        assert_eq!(restored.0, 16);
        assert!(dense_paged_linear_caches_ready(
            &inner.config,
            Some(&restored.1)
        ));
        let prepared = inner
            .prepare_dense_gdn_prefix_state(&prompt, 16, block_size, &extra_keys, 0, false)
            .expect("prepare one-block rollback checkpoint");
        assert_eq!(prepared.state, "checkpoint");
        assert!(prepared.already_primed);
        assert_eq!(prepared.restored_prefix_tokens, 16);
        assert_eq!(prepared.replayed_prefix_tokens, 0);

        let prepared = inner
            .prepare_dense_gdn_prefix_state(&prompt, 32, block_size, &extra_keys, 0, false)
            .expect("prepare full-boundary hit from stable checkpoint");
        assert_eq!(prepared.state, "checkpoint_replay_materialized");
        assert!(prepared.already_primed);
        assert_eq!(prepared.restored_prefix_tokens, 16);
        assert_eq!(prepared.replayed_prefix_tokens, 16);

        let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
        let _ = adapter.register_full_blocks_for_reuse(&[], 0);
        adapter.release_request().expect("release_request");
    }

    /// Compatibility guard for the current/default dense Qwen3.5 paged
    /// prefill behavior: a full suffix passed in one call remains a valid
    /// single-shot prefill and is stable across a fresh adapter reset.
    #[test]
    #[ignore = "requires Metal GPU; run with --ignored"]
    fn test_dense_paged_prefill_single_shot_default_still_works() {
        let Some((mut inner, cfg)) =
            paged_inner_or_skip("test_dense_paged_prefill_single_shot_default_still_works")
        else {
            return;
        };
        cast_qwen35_inner_weights_bf16(&mut inner);

        // Longer than two 16-token blocks: if chunk_size=0 accidentally starts
        // splitting at checkpoint boundaries, this test observes a sidecar.
        let prompt: Vec<u32> = (0u32..33).map(|i| (i * 17 + 5) % 257).collect();

        reset_paged_request(&mut inner, &prompt);
        let (logits_a, checkpoint_a) = run_dense_paged_prefill_with_size_and_checkpoint(
            &mut inner, &prompt, &prompt, 0, /* chunk_size */ 0,
        )
        .expect("single-shot A");
        assert!(
            checkpoint_a.is_empty(),
            "chunk_size=0 must not split to capture a GDN checkpoint"
        );
        assert_finite_vocab_logits(&logits_a, cfg.vocab_size, "single-shot A");
        {
            let adapter = inner.paged_adapter.as_ref().expect("paged_adapter");
            assert_eq!(
                adapter.current_token_count() as usize,
                prompt.len(),
                "single-shot cursor"
            );
            assert_eq!(adapter.request_tokens(), prompt.as_slice());
        }

        reset_paged_request(&mut inner, &prompt);
        let (logits_b, checkpoint_b) = run_dense_paged_prefill_with_size_and_checkpoint(
            &mut inner, &prompt, &prompt, 0, /* chunk_size */ 0,
        )
        .expect("single-shot B");
        assert!(
            checkpoint_b.is_empty(),
            "chunk_size=0 must stay single-shot after adapter reset"
        );
        let a = logits_to_f32_vec(&logits_a);
        let b = logits_to_f32_vec(&logits_b);
        assert_eq!(a.len(), cfg.vocab_size as usize);
        assert_eq!(b.len(), cfg.vocab_size as usize);
        for (i, (left, right)) in a.iter().zip(b.iter()).enumerate() {
            let abs_diff = (left - right).abs();
            assert!(
                abs_diff <= 1e-6,
                "single-shot dense paged prefill changed after fresh reset at index {i}: \
                 first={left}, second={right}, abs_diff={abs_diff}"
            );
        }

        let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
        let _ = adapter.register_full_blocks_for_reuse(&[], 0);
        adapter.release_request().expect("release_request");
    }

    /// Dense inner carrying a real adapter over a test-sized `LayerKVPool`.
    ///
    /// `initialize_paged_adapter` sizes a pool for a real checkpoint and needs a
    /// production block geometry; the lifecycle below dispatches no kernel, so a
    /// two-layer placeholder pool is enough and runs wherever the adapter's own
    /// lifecycle tests run.
    fn dense_inner_with_test_adapter_or_skip(test_name: &str) -> Option<Qwen35Inner> {
        const BLOCK_SIZE: u32 = 16;
        const NUM_BLOCKS: u32 = 8;
        let pool_config = mlx_paged_attn::PagedAttentionConfig {
            block_size: BLOCK_SIZE,
            num_kv_heads: 1,
            head_size: 32,
            num_layers: 2,
            ..mlx_paged_attn::PagedAttentionConfig::default()
        };
        let pool = match mlx_paged_attn::LayerKVPool::new_for_test(
            pool_config,
            NUM_BLOCKS,
            2,
            mlx_paged_attn::metal::MetalDtype::Float16,
        ) {
            Ok(pool) => Arc::new(pool),
            // Routed through `test_support::metal_device_absent` rather than
            // matching the string inline, so `MLX_TEST_REQUIRE_METAL=1` can
            // turn a self-skip into a failure here too. libtest counts a
            // skipped-and-returned test as passed, and these two are the only
            // gates that can see a hardcoded arm at the prune call site.
            Err(err) if crate::test_support::metal_device_absent(&err) => {
                eprintln!("skipping {test_name} (no Metal device): {err}");
                return None;
            }
            Err(err) => {
                panic!("unexpected LayerKVPool::new_for_test failure in {test_name}: {err}")
            }
        };
        let allocator = Arc::new(Mutex::new(mlx_paged_attn::BlockAllocator::new(
            NUM_BLOCKS, BLOCK_SIZE,
        )));
        let adapter = PagedKVCacheAdapter::new(allocator, pool, BLOCK_SIZE)
            .expect("test paged adapter must construct");
        let mut inner = Qwen35Inner::new(tiny_cfg(true)).expect("construct tiny dense model");
        inner.paged_adapter = Some(adapter);
        Some(inner)
    }

    /// Attach a cold tier carrying the GDN sidecar policy — the exact shape
    /// `build_cold_tier_context` installs when persistence is on.
    fn install_gdn_cold_tier(inner: &mut Qwen35Inner, root: &std::path::Path) {
        let manager = mlx_paged_attn::ColdCacheManager::open_default_at(root.to_path_buf())
            .expect("temp-dir cold cache must open");
        let cache_dtype = {
            let adapter = inner.paged_adapter.as_ref().expect("paged_adapter");
            format!("{:?}", adapter.layer_kv_pool().cache_dtype())
        };
        let policy = crate::models::qwen3_5::gdn_sidecar::policy(&inner.config, &cache_dtype)
            .expect("a hybrid config must yield a GDN sidecar policy");
        let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
        adapter.set_cold_tier(
            crate::transformer::paged_kv_cache_adapter::ColdTierContext {
                manager: Arc::new(manager),
                fingerprint: mlx_paged_attn::ColdCacheFingerprint::from_components([
                    b"qwen3-5-dense-sidecar-test".as_slice(),
                ]),
                sidecar_policy: Some(policy),
            },
        );
    }

    fn temp_cold_root(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "mlx-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ))
    }

    fn record_whole_paged_request(inner: &mut Qwen35Inner, prompt: &[u32]) {
        let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
        adapter.reset_for_new_request(0).expect("reset request");
        adapter
            .allocate_suffix_blocks(prompt.len() as u32)
            .expect("allocate suffix blocks");
        adapter.record_tokens(prompt).expect("record tokens");
    }

    /// The hand-written cores finalize through `finalize_dense_manual_paged_turn`
    /// — every planned-MTP turn ends there and never reaches the engine hook. It
    /// must run the GDN sidecar capture, otherwise an MTP session persists K/V
    /// blocks whose recurrent half no restore can reconstruct.
    ///
    /// Catches: deleting `capture_dense_gdn_cold_sidecar` from the `Ok` arm of
    /// `finalize_dense_manual_paged_turn` — both deltas drop to zero.
    #[test]
    fn dense_manual_paged_finalize_reaches_the_gdn_sidecar_capture() {
        let Some(mut inner) = dense_inner_with_test_adapter_or_skip(
            "dense_manual_paged_finalize_reaches_the_gdn_sidecar_capture",
        ) else {
            return;
        };
        let root = temp_cold_root("dense-manual-finalize-capture");
        install_gdn_cold_tier(&mut inner, &root);
        let prompt: Vec<u32> = (0u32..32).collect();
        record_whole_paged_request(&mut inner, &prompt);

        let _serialized = crate::cold_tier::sidecar_counter_test_lock();
        let before = crate::cold_tier::cold_sidecar_telemetry();
        inner
            .finalize_dense_manual_paged_turn(&[])
            .expect("manual paged finalization must succeed");
        let after = crate::cold_tier::cold_sidecar_telemetry();

        assert_eq!(
            after.capture_reached,
            before.capture_reached + 1,
            "the manual finalize must enter the GDN sidecar capture exactly once"
        );
        // The chain really carried both blocks off the GPU. Asserted here
        // because everything below depends on it and nothing below can see it:
        // a pool whose buffers are too small makes `read_block_all_layers`
        // return `Err`, the chain capture stops at its first block, and the
        // boundary assertion after this one becomes unreachable rather than
        // false. Without this line that failure reads as "the capture did not
        // reach its boundary selection", three layers from the real cause.
        assert_eq!(
            inner
                .paged_adapter
                .as_ref()
                .expect("paged_adapter")
                .cold_captured_blocks(),
            2,
            "both whole blocks must have been captured off the pool"
        );
        // The finalize published two whole blocks to the cold chain, so the
        // capture ran all the way to the boundary selection and found no
        // in-memory checkpoint to anchor on. Reaching THAT arm proves it got
        // past the policy and media guards rather than bailing at its first line.
        assert_eq!(
            after.boundary_skips,
            before.boundary_skips + 1,
            "the capture must reach its boundary selection under a GdnState policy"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    /// The v1 text-only guard reads the turn's OWN media map. The VLM cores clear
    /// `cached_paged_image_token_positions` before prefill and reassign it only
    /// after finalization returns, so a `self`-read guard sees an empty vec on
    /// exactly the media turns it exists to refuse.
    ///
    /// Catches: reverting the guard to `self.cached_paged_image_token_positions`
    /// — that field is empty here, so the capture would run on past it and bump
    /// `chain_empty`.
    #[test]
    fn dense_gdn_sidecar_capture_refuses_the_turns_own_media_map() {
        let Some(mut inner) = dense_inner_with_test_adapter_or_skip(
            "dense_gdn_sidecar_capture_refuses_the_turns_own_media_map",
        ) else {
            return;
        };
        let root = temp_cold_root("dense-manual-finalize-media");
        install_gdn_cold_tier(&mut inner, &root);
        let prompt: Vec<u32> = (0u32..32).collect();
        record_whole_paged_request(&mut inner, &prompt);
        assert!(
            inner.cached_paged_image_token_positions.is_empty(),
            "the model-level media map is empty mid-turn; only the argument carries the truth"
        );

        let _serialized = crate::cold_tier::sidecar_counter_test_lock();
        let before = crate::cold_tier::cold_sidecar_telemetry();
        inner
            .finalize_dense_manual_paged_turn(&[(4, 99)])
            .expect("manual paged finalization must succeed");
        let after = crate::cold_tier::cold_sidecar_telemetry();

        assert_eq!(
            after.capture_reached,
            before.capture_reached + 1,
            "a media turn still enters the capture"
        );
        assert_eq!(
            after.boundary_skips, before.boundary_skips,
            "a media turn must be refused before the capture inspects the cold chain"
        );
        assert_eq!(
            after.chain_empty, before.chain_empty,
            "a media turn must be refused before the capture inspects the cold chain"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    /// The chunk size every paged prefill reads. Single-shot materializes no GDN
    /// checkpoint, so a sidecar has nothing to anchor on; an installed GdnState
    /// policy has to turn the ladder's rungs into real splits.
    ///
    /// Catches: dropping the policy probe from `cold_gdn_prefill_chunk_size`
    /// (returns 0 with a policy installed) — the defect that leaves a persist-cold
    /// turn publishing no rung at all.
    #[test]
    fn dense_cold_gdn_prefill_chunk_size_follows_the_installed_sidecar_policy() {
        let Some(mut inner) = dense_inner_with_test_adapter_or_skip(
            "dense_cold_gdn_prefill_chunk_size_follows_the_installed_sidecar_policy",
        ) else {
            return;
        };
        assert_eq!(
            inner.cold_gdn_prefill_chunk_size(),
            0,
            "without a cold GDN policy the prefill stays single-shot"
        );

        let root = temp_cold_root("dense-chunk-size-policy");
        install_gdn_cold_tier(&mut inner, &root);
        assert!(
            inner.cold_gdn_prefill_chunk_size() > 0,
            "an installed GdnState policy must make the prefill split at the ladder"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    /// Both hand-written cores prefill through `run_dense_core_paged_prefill`, so
    /// this drives that shared body directly: with a GdnState policy installed it
    /// must publish checkpoint ladder rungs, and with no cold tier it must stay
    /// single-shot.
    ///
    /// Catches: passing a literal `0` or `crate::array::paged_prefill_chunk_size()`
    /// instead of `self.cold_gdn_prefill_chunk_size()` inside
    /// `run_dense_core_paged_prefill` — the cold arm then returns no rungs, which
    /// is exactly the MTP-turn defect.
    ///
    /// The last two arms use an EXPLICIT positive chunk size, which is the only
    /// way to reach the break-set decision with no policy installed — the shape
    /// a persist-off `mlx agent` turn has, since `run-agent.ts` seeds
    /// `MLX_PAGED_PREFILL_CHUNK_SIZE=2048` before any persistence decision. They
    /// catch hardcoding `gdn_cold_sidecar_ladder_wanted`'s result at the prefill
    /// body in either direction, which the arms above cannot: at 64 tokens the
    /// ladder is already a single rung, so both arms agree there.
    #[test]
    #[ignore = "requires Metal GPU; run with --ignored"]
    fn dense_core_paged_prefill_publishes_ladder_rungs_under_a_cold_policy() {
        let Some((mut inner, _cfg)) = paged_inner_or_skip(
            "dense_core_paged_prefill_publishes_ladder_rungs_under_a_cold_policy",
        ) else {
            return;
        };
        cast_qwen35_inner_weights_bf16(&mut inner);
        let layer_kinds =
            decoder_layer::compute_layer_kinds(inner.config.num_layers as usize, |i| {
                inner.config.is_linear_layer(i)
            });
        let prompt: Vec<u32> = (0u32..64).map(|i| (i * 5 + 7) % 257).collect();

        reset_paged_request(&mut inner, &prompt);
        let (_, hidden, cold_checkpoints) = inner
            .run_dense_core_paged_prefill(
                &prompt,
                &prompt,
                0,
                false,
                None,
                &layer_kinds,
                "cold-policy prefill",
            )
            .expect("prefill with no cold tier");
        assert!(hidden.is_none());
        assert!(
            cold_checkpoints.is_empty(),
            "with no cold GDN policy the cores must stay single-shot"
        );

        // A prompt long enough that the two break sets DIFFER: ladder
        // [48, 192] against the single pre-ladder boundary [192]. Run BEFORE
        // the cold tier is installed — the adapter has no way to drop one.
        let long_prompt: Vec<u32> = (0u32..200).map(|i| (i * 5 + 7) % 257).collect();
        reset_paged_request(&mut inner, &long_prompt);
        let (_, no_policy_rungs) = run_dense_paged_prefill_with_size_and_checkpoint(
            &mut inner,
            &long_prompt,
            &long_prompt,
            0,
            2048,
        )
        .expect("explicit-chunk prefill with no cold tier");
        assert_eq!(
            no_policy_rungs
                .iter()
                .map(|c| c.prefix_len)
                .collect::<Vec<_>>(),
            vec![192],
            "with no cold GDN policy an explicit chunk size must take the single pre-ladder \
             boundary, not the ladder — the extra breaks change the prefill GEMM's M and so the \
             sampled tokens of a persistence-off request"
        );
        inner
            .paged_adapter
            .as_mut()
            .expect("paged_adapter")
            .release_request()
            .expect("release_request");

        // The SAME claim on the planned-MTP body. `run_dense_core_paged_prefill`
        // forks on `keep_prompt_hidden_tokens` into two separate prefill
        // functions that each pick their own break set, and every arm above
        // takes the AR fork only. Measured: hardcoding `want_ladder = true` at
        // the MTP call site alone (`run_paged_prefill_chunk_with_hidden_with_size`)
        // left this whole gate and its MoE twin GREEN before this arm existed.
        // `ChatSession::mergeConfig` auto-defaults `enableMtp` to true on any
        // checkpoint carrying an MTP head, so on those checkpoints the MTP fork
        // is the one an `mlx agent` turn actually takes.
        reset_paged_request(&mut inner, &long_prompt);
        let (_, _, no_policy_mtp_rungs) = run_dense_paged_prefill_with_hidden_and_checkpoint(
            &mut inner,
            &long_prompt,
            &long_prompt,
            0,
            2048,
        )
        .expect("explicit-chunk MTP prefill with no cold tier");
        assert_eq!(
            no_policy_mtp_rungs
                .iter()
                .map(|c| c.prefix_len)
                .collect::<Vec<_>>(),
            vec![192],
            "the planned-MTP prefill body must take the same single pre-ladder boundary with no \
             cold GDN policy as the AR body does"
        );
        // Release WITHOUT registering: publishing this prompt's blocks would
        // give the next arm's shared 64-token prefix a cache hit, and every arm
        // here has to start cold.
        inner
            .paged_adapter
            .as_mut()
            .expect("paged_adapter")
            .release_request()
            .expect("release_request");

        let root = temp_cold_root("dense-core-ladder");
        install_gdn_cold_tier(&mut inner, &root);
        reset_paged_request(&mut inner, &prompt);
        let (_, _, ladder) = inner
            .run_dense_core_paged_prefill(
                &prompt,
                &prompt,
                0,
                false,
                None,
                &layer_kinds,
                "cold-policy prefill",
            )
            .expect("prefill under a cold GDN policy");
        assert!(
            !ladder.is_empty(),
            "a persist-cold turn must materialize the GDN checkpoint ladder the sidecar anchors on"
        );
        for checkpoint in &ladder {
            assert!(
                checkpoint.prefix_len > 0 && (checkpoint.prefix_len as usize) < prompt.len(),
                "a rung must sit strictly inside the prompt, got {}",
                checkpoint.prefix_len
            );
        }

        // Same long prompt, same explicit chunk size, policy now installed:
        // every rung of the ladder is a real break.
        reset_paged_request(&mut inner, &long_prompt);
        let (_, cold_rungs) = run_dense_paged_prefill_with_size_and_checkpoint(
            &mut inner,
            &long_prompt,
            &long_prompt,
            0,
            2048,
        )
        .expect("explicit-chunk prefill under a cold GDN policy");
        assert_eq!(
            cold_rungs.iter().map(|c| c.prefix_len).collect::<Vec<_>>(),
            vec![48, 192],
            "an explicit chunk size under a GDN cold policy still publishes the whole ladder"
        );
        inner
            .paged_adapter
            .as_mut()
            .expect("paged_adapter")
            .release_request()
            .expect("release_request");

        // And the MTP fork under the policy: the ladder is what the sidecar
        // anchors on, so an MTP turn that publishes only the deep rung is the
        // silent-zero-reuse defect `29be89e0` set out to fix, on the fork it
        // did not cover.
        reset_paged_request(&mut inner, &long_prompt);
        let (_, _, cold_mtp_rungs) = run_dense_paged_prefill_with_hidden_and_checkpoint(
            &mut inner,
            &long_prompt,
            &long_prompt,
            0,
            2048,
        )
        .expect("explicit-chunk MTP prefill under a cold GDN policy");
        assert_eq!(
            cold_mtp_rungs
                .iter()
                .map(|c| c.prefix_len)
                .collect::<Vec<_>>(),
            vec![48, 192],
            "the planned-MTP prefill body must publish the whole ladder under a cold GDN policy"
        );

        let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
        let _ = adapter.register_full_blocks_for_reuse(&[], 0);
        adapter.release_request().expect("release_request");
        let _ = std::fs::remove_dir_all(&root);
    }
}
