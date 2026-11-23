//! Qwen 3.8-style **StateDisk SSD Cache** (`8 Global KV chains + 1 State Blob`) for Inkling-2-Mini.
//!
//! Architecture:
//! - **Group 0 (8 Global KV visits)** persist as content-addressed chained `ColdGroup::Kv` blocks
//!   via `ColdContextBacking::StateOnly` on `full_adapter`.
//! - **Group 1 (40 SWA-1024 trailing KV windows, ~82.5 MiB INT8) + 48 ShortConv(k=4) ring states
//!   (~1.9 MiB BF16)** pack into **1 content-keyed state blob** (~84.3 MiB constant regardless of
//!   context length) stored via `ColdCacheManager`'s prioritized state lane (`ColdGroup::DFlash2State`
//!   isolated by Inkling2's persistent namespace & geometry string).
//! - Captured at turn end (`StateClass::Ordinary`) AND at rolling prefill checkpoints every
//!   4,096 tokens (`StateClass::Checkpoint`, disposable first under governor pressure).
//! - **Async restore (`TurnStatus::WaitingForSsd`)**: restoring a cold prompt from SSD runs on
//!   `ColdCacheManager`'s reader thread via `ChainPrefetch` without stalling active decode lanes.
//! - **Restart survival**: persistent `DiskTier` (`persistPagedCache` / `MLX_PERSIST_PAGED_CACHE=1`)
//!   adopts complete `(state blob + 8-layer Global KV chain)` entries via `adopt_states` on startup.

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
#[cfg(test)]
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use mlx_paged_attn::{
    CacheRecency, ColdCacheFingerprint, ColdCacheKey, ColdGroup, ColdStateLabel, MakeRoom,
    block_object_bytes,
};
use napi::bindgen_prelude::{Error, Result};

use crate::array::memory::WeightsResident;
use crate::array::{DType, MxArray};
use crate::cold_tier::{
    CheckpointFiles, CheckpointLoadGuard, ColdTierGeometry, build_model_fingerprint,
};
use crate::disk_tier::state_disk::{
    KvPoolHandle, Reclaimed, Staged, StateDiskCodec, StateDiskEngine, Tick, state_context,
};
use crate::disk_tier::store::{
    Entry, EntryId, HybridStateStore, Lineage, StateBytes, StateClass, StoreHit,
};
use crate::disk_tier::{DiskTier, DiskTierOptions, PersistentIdentity, open_disk_tier};
use crate::transformer::KvFormat;
use crate::transformer::paged_kv_cache_adapter::{
    ChainPrefetch, ChainPrefetchStart, ColdTierContext, PagedKVCacheAdapter, RestoreUploadBudget,
    SeqId,
};

pub(crate) use crate::disk_tier::state_disk::ReadWait;

use super::config::Inkling2Config;
use super::kv_cache::Inkling2KVCacheCoordinator;
use super::memory::{
    global_visit_count, short_conv_state_bytes, swa_visit_count, swa_window_blocks,
};
use super::short_conv::{Inkling2ShortConvState, VisitShortConvState};

/// Default rolling prefill checkpoint spacing (tokens).
pub(crate) const INKLING2_CHECKPOINT_INTERVAL: u32 = 4096;

/// Default minimum prefix depth eligible for background write-behind.
pub(crate) const INKLING2_WRITE_BEHIND_MIN_TOKENS: usize = 2048;

/// Default write-behind delay (seconds) after ordinary publication before spilling to SSD.
pub(crate) const INKLING2_WRITE_BEHIND_DELAY_SECS: f64 = 10.0;

#[cfg(test)]
thread_local! {
    pub(crate) static CHECKPOINT_INTERVAL_OVERRIDE: std::cell::Cell<Option<u32>> =
        const { std::cell::Cell::new(None) };
    pub(crate) static WRITE_BEHIND_MIN_TOKENS_OVERRIDE: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
    pub(crate) static WRITE_BEHIND_DELAY_OVERRIDE: std::cell::Cell<Option<f64>> =
        const { std::cell::Cell::new(None) };
}

pub(crate) fn checkpoint_interval_tokens() -> u32 {
    #[cfg(test)]
    if let Some(v) = CHECKPOINT_INTERVAL_OVERRIDE.with(std::cell::Cell::get) {
        return v;
    }
    INKLING2_CHECKPOINT_INTERVAL
}

fn write_behind_min_tokens() -> usize {
    #[cfg(test)]
    if let Some(v) = WRITE_BEHIND_MIN_TOKENS_OVERRIDE.with(std::cell::Cell::get) {
        return v;
    }
    INKLING2_WRITE_BEHIND_MIN_TOKENS
}

fn write_behind_delay_secs() -> f64 {
    #[cfg(test)]
    if let Some(v) = WRITE_BEHIND_DELAY_OVERRIDE.with(std::cell::Cell::get) {
        return v;
    }
    INKLING2_WRITE_BEHIND_DELAY_SECS
}

/// Exact geometry of Inkling-2-Mini's 1 content-keyed state blob:
/// - Span 0: `max_swa_blocks` raw physical blocks across all `swa_layers` (`40`) SWA visits
///   (INT8 KTM codes + fp32 scales, or BF16; zero-padded when `boundary < sliding_window`).
/// - Span 1: all `num_visits` (`48`) ShortConv(`k=4`) rolling states (`k, v, attn_post, mlp_post`)
///   in BF16.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Inkling2StateGeometry {
    pub swa_layers: u32,
    pub max_swa_blocks: u32,
    pub swa_key_blob_bytes: u64,
    pub swa_value_blob_bytes: u64,
    pub swa_span_bytes: u64,
    pub num_visits: u32,
    pub conv_history: u32,
    pub kv_dim: u32,
    pub hidden_size: u32,
    pub conv_span_bytes: u64,
    pub block_size: u32,
    pub sliding_window: u32,
    pub kv_format: KvFormat,
}

impl Inkling2StateGeometry {
    pub(crate) fn from_coordinator(
        config: &Inkling2Config,
        coordinator: &Inkling2KVCacheCoordinator,
    ) -> Result<Self> {
        let block_size = coordinator.block_size().max(1);
        let sliding_window = u32::try_from(config.sliding_window)
            .map_err(|_| Error::from_reason("Inkling2 sliding_window overflows u32"))?;
        let swa_layers = u32::try_from(swa_visit_count(config))
            .map_err(|_| Error::from_reason("Inkling2 swa_layers overflows u32"))?;
        let max_swa_blocks = if swa_layers > 0 {
            swa_window_blocks(config, block_size)
        } else {
            0
        };
        let (swa_key_blob_bytes, swa_value_blob_bytes) = if swa_layers > 0 {
            coordinator
                .sliding_block_blob_bytes()
                .map_err(Error::from_reason)?
                .ok_or_else(|| {
                    Error::from_reason("Inkling2 StateDisk: missing sliding adapter blob geometry")
                })?
        } else {
            (0, 0)
        };
        let per_swa_block = u64::from(swa_layers)
            .checked_mul(swa_key_blob_bytes.saturating_add(swa_value_blob_bytes))
            .ok_or_else(|| Error::from_reason("Inkling2 StateDisk: SWA block bytes overflow"))?;
        let swa_span_bytes = u64::from(max_swa_blocks)
            .checked_mul(per_swa_block)
            .ok_or_else(|| Error::from_reason("Inkling2 StateDisk: SWA span bytes overflow"))?;

        let num_visits = u32::try_from(config.num_layer_visits())
            .map_err(|_| Error::from_reason("Inkling2 num_visits overflows u32"))?;
        let conv_history = u32::try_from(config.sconv_kernel_size.saturating_sub(1))
            .map_err(|_| Error::from_reason("Inkling2 conv_history overflows u32"))?;
        let kv_dim = u32::try_from(config.num_key_value_heads.saturating_mul(config.head_dim))
            .map_err(|_| Error::from_reason("Inkling2 kv_dim overflows u32"))?;
        let hidden_size = u32::try_from(config.hidden_size)
            .map_err(|_| Error::from_reason("Inkling2 hidden_size overflows u32"))?;
        let conv_span_bytes = short_conv_state_bytes(config);

        Ok(Self {
            swa_layers,
            max_swa_blocks,
            swa_key_blob_bytes,
            swa_value_blob_bytes,
            swa_span_bytes,
            num_visits,
            conv_history,
            kv_dim,
            hidden_size,
            conv_span_bytes,
            block_size,
            sliding_window,
            kv_format: coordinator.kv_format(),
        })
    }

    pub(crate) fn spans(&self) -> Vec<u64> {
        let mut s = Vec::with_capacity(2);
        if self.swa_span_bytes > 0 {
            s.push(self.swa_span_bytes);
        }
        if self.conv_span_bytes > 0 {
            s.push(self.conv_span_bytes);
        }
        s
    }

    pub(crate) fn payload_bytes(&self) -> u64 {
        self.swa_span_bytes.saturating_add(self.conv_span_bytes)
    }

    pub(crate) fn geometry_string(&self) -> String {
        format!(
            "inkling2-state:v1 swa={}:{}:{}:{}:{} conv={}:{}:{}:{} bs={} fmt={}",
            self.swa_layers,
            self.max_swa_blocks,
            self.swa_key_blob_bytes,
            self.swa_value_blob_bytes,
            self.sliding_window,
            self.num_visits,
            self.conv_history,
            self.kv_dim,
            self.hidden_size,
            self.block_size,
            self.kv_format.as_str(),
        )
    }

    /// Number of live SWA physical blocks at block-aligned `boundary_tokens`.
    pub(crate) fn live_swa_blocks(&self, boundary_tokens: u32) -> u32 {
        if self.swa_layers == 0 || boundary_tokens == 0 {
            return 0;
        }
        let first_live = boundary_tokens.saturating_sub(self.sliding_window) / self.block_size;
        let total = boundary_tokens / self.block_size;
        total.saturating_sub(first_live)
    }

    /// Encode an [`Inkling2StateAnchor`] into the fixed-size `into` staging buffer.
    pub(crate) fn encode_into(
        &self,
        boundary_tokens: u32,
        anchor: &Inkling2StateAnchor,
        into: &mut [u8],
    ) -> std::result::Result<(), String> {
        let total_len = usize::try_from(self.payload_bytes())
            .map_err(|_| "Inkling2 state payload overflows usize".to_string())?;
        if into.len() != total_len {
            return Err(format!(
                "Inkling2 state staging size mismatch: {} != {total_len}",
                into.len()
            ));
        }
        let swa_len = usize::try_from(self.swa_span_bytes)
            .map_err(|_| "Inkling2 SWA span overflows usize".to_string())?;
        let (swa_buf, conv_buf) = into.split_at_mut(swa_len);

        // Span 0: SWA trailing window raw blocks (zero-padded to fixed span length).
        if swa_len > 0 {
            let expected_live = self.live_swa_blocks(boundary_tokens) as usize;
            if anchor.swa_blocks.len() != expected_live
                || expected_live > self.max_swa_blocks as usize
            {
                return Err(format!(
                    "Inkling2 SWA block count {} != expected {expected_live} (max {})",
                    anchor.swa_blocks.len(),
                    self.max_swa_blocks
                ));
            }
            let k_len = self.swa_key_blob_bytes as usize;
            let v_len = self.swa_value_blob_bytes as usize;
            let mut cursor = 0usize;
            for block in &anchor.swa_blocks {
                if block.len() != self.swa_layers as usize {
                    return Err(format!(
                        "Inkling2 SWA block layer count {} != {}",
                        block.len(),
                        self.swa_layers
                    ));
                }
                for (k, v) in block {
                    if k.len() != k_len || v.len() != v_len {
                        return Err(format!(
                            "Inkling2 SWA blob length mismatch: ({}, {}) != ({k_len}, {v_len})",
                            k.len(),
                            v.len()
                        ));
                    }
                    swa_buf[cursor..cursor + k_len].copy_from_slice(k);
                    cursor += k_len;
                    swa_buf[cursor..cursor + v_len].copy_from_slice(v);
                    cursor += v_len;
                }
            }
            if cursor < swa_len {
                swa_buf[cursor..].fill(0);
            }
        }

        // Span 1: 48-visit ShortConv(k=4) ring states in BF16 little-endian u16.
        if anchor.conv_state.visits.len() != self.num_visits as usize {
            return Err(format!(
                "Inkling2 ShortConv visit count {} != {}",
                anchor.conv_state.visits.len(),
                self.num_visits
            ));
        }
        let hist = self.conv_history as usize;
        let kv_elems = hist * self.kv_dim as usize;
        let hidden_elems = hist * self.hidden_size as usize;
        let mut cursor = 0usize;
        let mut write_bf16 =
            |arr: &MxArray, expected_elems: usize| -> std::result::Result<(), String> {
                let dt = arr
                    .dtype()
                    .map_err(|e| format!("Inkling2 ShortConv dtype failed: {e}"))?;
                let bf16 = if dt == DType::BFloat16 {
                    arr.clone()
                } else {
                    arr.astype(DType::BFloat16)
                        .map_err(|e| format!("Inkling2 ShortConv cast to bf16 failed: {e}"))?
                };
                let raw = bf16
                    .to_uint16_native()
                    .map_err(|e| format!("Inkling2 ShortConv readback failed: {e}"))?;
                if raw.len() != expected_elems {
                    return Err(format!(
                        "Inkling2 ShortConv element count {} != {expected_elems}",
                        raw.len()
                    ));
                }
                let byte_len = expected_elems * 2;
                let dst = conv_buf
                    .get_mut(cursor..cursor + byte_len)
                    .ok_or_else(|| "Inkling2 ShortConv buffer overflow".to_string())?;
                for (i, val) in raw.into_iter().enumerate() {
                    dst[i * 2..i * 2 + 2].copy_from_slice(&val.to_le_bytes());
                }
                cursor += byte_len;
                Ok(())
            };
        for visit in &anchor.conv_state.visits {
            write_bf16(&visit.k, kv_elems)?;
            write_bf16(&visit.v, kv_elems)?;
            write_bf16(&visit.attn_post, hidden_elems)?;
            write_bf16(&visit.mlp_post, hidden_elems)?;
        }
        if cursor != conv_buf.len() {
            return Err(format!(
                "Inkling2 ShortConv encoded bytes {cursor} != expected {}",
                conv_buf.len()
            ));
        }
        Ok(())
    }

    /// Decode an [`Inkling2StateAnchor`] from a validated state blob payload.
    pub(crate) fn decode_from(
        &self,
        boundary_tokens: u32,
        payload: &[u8],
    ) -> std::result::Result<Inkling2StateAnchor, String> {
        let total_len = usize::try_from(self.payload_bytes())
            .map_err(|_| "Inkling2 state payload overflows usize".to_string())?;
        if payload.len() != total_len {
            return Err(format!(
                "Inkling2 state payload length {} != {total_len}",
                payload.len()
            ));
        }
        let swa_len = usize::try_from(self.swa_span_bytes)
            .map_err(|_| "Inkling2 SWA span overflows usize".to_string())?;
        let (swa_buf, conv_buf) = payload.split_at(swa_len);

        // Decode Span 0: live SWA blocks.
        let mut swa_blocks = Vec::new();
        let mut swa_ram_bytes = 0u64;
        if swa_len > 0 {
            let live_blocks = self.live_swa_blocks(boundary_tokens) as usize;
            if live_blocks > self.max_swa_blocks as usize {
                return Err(format!(
                    "Inkling2 SWA live_blocks {live_blocks} exceeds max {}",
                    self.max_swa_blocks
                ));
            }
            let k_len = self.swa_key_blob_bytes as usize;
            let v_len = self.swa_value_blob_bytes as usize;
            let mut cursor = 0usize;
            swa_blocks.reserve(live_blocks);
            for _ in 0..live_blocks {
                let mut layers = Vec::with_capacity(self.swa_layers as usize);
                for _ in 0..self.swa_layers {
                    let k = swa_buf
                        .get(cursor..cursor + k_len)
                        .ok_or_else(|| "Inkling2 SWA payload truncated on keys".to_string())?
                        .to_vec();
                    cursor += k_len;
                    let v = swa_buf
                        .get(cursor..cursor + v_len)
                        .ok_or_else(|| "Inkling2 SWA payload truncated on values".to_string())?
                        .to_vec();
                    cursor += v_len;
                    swa_ram_bytes = swa_ram_bytes.saturating_add((k_len + v_len) as u64);
                    layers.push((k, v));
                }
                swa_blocks.push(layers);
            }
        }

        // Decode Span 1: 48-visit ShortConv states.
        let hist = self.conv_history as i64;
        let kv_cols = self.kv_dim as i64;
        let hidden_cols = self.hidden_size as i64;
        let kv_elems = (self.conv_history as usize) * (self.kv_dim as usize);
        let hidden_elems = (self.conv_history as usize) * (self.hidden_size as usize);
        let mut cursor = 0usize;
        let mut read_bf16 = |elems: usize, shape: &[i64]| -> std::result::Result<MxArray, String> {
            let byte_len = elems * 2;
            let src = conv_buf
                .get(cursor..cursor + byte_len)
                .ok_or_else(|| "Inkling2 ShortConv payload truncated".to_string())?;
            cursor += byte_len;
            let mut raw = Vec::with_capacity(elems);
            for pair in src.as_chunks::<2>().0 {
                raw.push(u16::from_le_bytes(*pair));
            }
            MxArray::from_bfloat16(&raw, shape)
                .map_err(|e| format!("Inkling2 ShortConv decode array failed: {e}"))
        };

        let mut visits = Vec::with_capacity(self.num_visits as usize);
        for _ in 0..self.num_visits {
            let k = read_bf16(kv_elems, &[hist, kv_cols])?;
            let v = read_bf16(kv_elems, &[hist, kv_cols])?;
            let attn_post = read_bf16(hidden_elems, &[hist, hidden_cols])?;
            let mlp_post = read_bf16(hidden_elems, &[hist, hidden_cols])?;
            visits.push(VisitShortConvState {
                k,
                v,
                attn_post,
                mlp_post,
            });
        }
        let conv_state = Inkling2ShortConvState {
            visits,
            kernel_size: (self.conv_history as usize) + 1,
        };
        Ok(Inkling2StateAnchor {
            swa_blocks,
            conv_state,
            cached_bytes: swa_ram_bytes.saturating_add(self.conv_span_bytes),
        })
    }
}

/// In-memory hybrid state snapshot at one block-aligned prefix boundary:
/// - `swa_blocks`: raw physical blocks for Group 1 (40 SWA visits, `<= 32` blocks).
/// - `conv_state`: 48-visit ShortConv(`k=4`) rolling states in BF16.
///
/// Holds NO Global (Group 0) KV copy: Group 0 blocks stay content-addressed in the Global
/// `PagedKVCacheAdapter` and are kept resident via `set_state_marks`.
#[derive(Clone)]
pub(crate) struct Inkling2StateAnchor {
    pub swa_blocks: Vec<Vec<(Vec<u8>, Vec<u8>)>>,
    pub conv_state: Inkling2ShortConvState,
    pub cached_bytes: u64,
}

impl StateBytes for Inkling2StateAnchor {
    fn state_bytes(&self) -> u64 {
        self.cached_bytes
    }
}

/// Counters tracking Inkling2's StateDisk lifecycle (useful for telemetry and parity gates).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Inkling2DiskCounters {
    pub ram_hits: u64,
    pub ssd_hits: u64,
    pub async_ssd_restores: u64,
    pub offloads: u64,
    pub offload_failures: u64,
    pub checkpoints_published: u64,
    pub ordinary_published: u64,
    pub adopted: u64,
}

/// Pending write-behind entry awaiting delay expiry.
struct WriteBehindEntry {
    id: EntryId,
    published_at: Instant,
}

/// Inkling2 codec for the shared governor-driven [`StateDiskEngine`].
pub(crate) struct Inkling2StateCodec {
    pub(crate) geometry: Inkling2StateGeometry,
}

impl StateDiskCodec for Inkling2StateCodec {
    type State = Inkling2StateAnchor;
    type EncodeCtx<'a> = ();

    const STOP_CHAIN_AT_DEADLINE: bool = true;

    fn encode_into(
        &self,
        _ctx: (),
        _spans: &[u64],
        entry: &Entry<Inkling2StateAnchor>,
        state: &Inkling2StateAnchor,
        into: &mut [u8],
    ) -> Option<u32> {
        let depth = u32::try_from(entry.depth()).ok()?;
        if let Err(err) = self.geometry.encode_into(depth, state, into) {
            tracing::warn!(
                "Inkling2 StateDisk encode failed for entry {}: {err}",
                entry.id()
            );
            return None;
        }
        Some(depth.saturating_sub(self.geometry.sliding_window))
    }

    fn adopt_copy(&self, _spans: &[u64], label: &ColdStateLabel) -> Option<(u64, u32)> {
        let row = u32::try_from(label.tokens.len()).ok()?;
        Some((
            self.geometry.payload_bytes(),
            row.saturating_sub(self.geometry.sliding_window),
        ))
    }
}

/// Model-owned StateDisk tier for Inkling-2-Mini backed by the shared
/// [`StateDiskEngine`] (single reusable host staging buffer, dual-lane
/// `ColdCacheManager` writer, 8-layer Global KV chain cursor, write-behind).
pub(crate) struct Inkling2StateDisk {
    pub(crate) engine: StateDiskEngine<Inkling2StateCodec>,
    write_behind: Vec<WriteBehindEntry>,
    pub(crate) counters: Inkling2DiskCounters,
}

impl Deref for Inkling2StateDisk {
    type Target = StateDiskEngine<Inkling2StateCodec>;

    fn deref(&self) -> &Self::Target {
        &self.engine
    }
}

impl DerefMut for Inkling2StateDisk {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.engine
    }
}

impl Inkling2StateDisk {
    pub(crate) fn new(
        tier: Arc<DiskTier>,
        geometry: Inkling2StateGeometry,
        global_block_bytes: u64,
    ) -> Result<Self> {
        let spans = geometry.spans();
        let geometry_str = geometry.geometry_string();
        let engine = StateDiskEngine::new(
            tier,
            spans,
            geometry_str,
            global_block_bytes,
            Inkling2StateCodec { geometry },
        )
        .ok_or_else(|| Error::from_reason("Inkling2 StateDisk staging size overflows usize"))?;
        Ok(Self {
            engine,
            write_behind: Vec::new(),
            counters: Inkling2DiskCounters::default(),
        })
    }

    pub(crate) fn entry_disk_restorable(&self, entry: &Entry<Inkling2StateAnchor>) -> bool {
        entry.disk().is_some_and(|copy| {
            copy.landed
                && self
                    .tier
                    .manager
                    .contains_in(&copy.key, ColdGroup::DFlash2State)
                && !entry.chain().is_empty()
                && self.engine.chain_on_disk(entry)
        })
    }
}

/// Complete Inkling-2-Mini hybrid prefix state store (in-memory RAM store + optional StateDisk).
pub(crate) struct Inkling2StateStore {
    pub(crate) store: HybridStateStore<Inkling2StateAnchor>,
    pub(crate) disk: Option<Inkling2StateDisk>,
    /// Most recent rolling prefill checkpoint published per active sequence (retired on next
    /// checkpoint or ordinary publication).
    seq_checkpoints: HashMap<SeqId, EntryId>,
    /// Counters tracked even when `disk` is disabled.
    pub(crate) ram_hits: u64,
    pub(crate) checkpoints_published: u64,
    pub(crate) ordinary_published: u64,
}

impl Inkling2StateStore {
    pub(crate) fn new(block_size: u32, recency: Arc<CacheRecency>) -> Self {
        let mut store = HybridStateStore::new(block_size);
        store.share_recency(recency);
        Self {
            store,
            disk: None,
            seq_checkpoints: HashMap::new(),
            ram_hits: 0,
            checkpoints_published: 0,
            ordinary_published: 0,
        }
    }

    pub(crate) fn clear_ram_and_checkpoints(&mut self) {
        self.seq_checkpoints.clear();
        let ids: Vec<EntryId> = self.store.entries().map(Entry::id).collect();
        for id in ids {
            self.store.remove(id);
        }
    }

    /// Clear only RAM copies of entries that have a durable disk copy (and remove entries without one).
    #[cfg(test)]
    pub(crate) fn drop_all_ram_keeping_disk(&mut self) {
        let ids: Vec<(EntryId, bool)> = self
            .store
            .entries()
            .map(|e| (e.id(), e.disk().is_some_and(|d| d.landed && !d.chain_owed)))
            .collect();
        for (id, has_durable_disk) in ids {
            if has_durable_disk {
                self.store.drop_ram(id);
            } else {
                self.store.remove(id);
            }
        }
    }

    pub(crate) fn total_ram_bytes(&self) -> u64 {
        self.store.bytes().state_bytes
    }

    pub(crate) fn generation(&self) -> u64 {
        self.store.generation()
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn disk_counters(&self) -> Inkling2DiskCounters {
        let mut c = self
            .disk
            .as_ref()
            .map_or_else(Inkling2DiskCounters::default, |d| d.counters);
        c.ram_hits = self.ram_hits;
        c.checkpoints_published = self.checkpoints_published;
        c.ordinary_published = self.ordinary_published;
        c
    }

    /// Compute the active state marks (Global KV block hashes that must stay resident).
    pub(crate) fn sync_marks(&self, coordinator: &Inkling2KVCacheCoordinator) {
        coordinator.set_full_state_marks(self.store.marks());
    }

    /// Check whether `id` can currently restore (either RAM + hot Global KV, or landed SSD blob +
    /// complete 8-layer Global KV chain on SSD).
    pub(crate) fn is_restorable(
        &self,
        coordinator: &Inkling2KVCacheCoordinator,
        id: EntryId,
    ) -> bool {
        let Some(entry) = self.store.get(id) else {
            return false;
        };
        let full_adapter = coordinator.full_adapter();
        let chain_hot = !entry.chain().is_empty()
            && entry
                .chain()
                .iter()
                .all(|&hash| full_adapter.prefix_cached(hash));
        if entry.ram().is_some() && chain_hot {
            return true;
        }
        if let Some(disk) = self.disk.as_ref()
            && disk.entry_disk_restorable(entry)
        {
            return true;
        }
        false
    }

    /// Deepest restorable state strictly shorter than `tokens.len()`.
    pub(crate) fn lookup_deepest_restorable(
        &self,
        coordinator: &Inkling2KVCacheCoordinator,
        tokens: &[u32],
        cache_salt: u64,
    ) -> Option<StoreHit> {
        if tokens.len() <= 1 {
            return None;
        }
        self.store
            .deepest_accepted_within(
                Lineage::text(tokens, cache_salt),
                tokens.len().saturating_sub(1),
                |id| self.is_restorable(coordinator, id),
            )
            .filter(|hit| hit.depth > 0)
    }

    /// Whether `id`'s Global KV chain is already hot in Group 0's prefix cache.
    pub(crate) fn chain_is_hot(
        &self,
        coordinator: &Inkling2KVCacheCoordinator,
        id: EntryId,
    ) -> bool {
        let Some(entry) = self.store.get(id) else {
            return false;
        };
        let full = coordinator.full_adapter();
        !entry.chain().is_empty() && entry.chain().iter().all(|&h| full.prefix_cached(h))
    }

    /// Capture `(40 SWA trailing blocks + 48 ShortConv states)` at block-aligned `boundary`
    /// and publish into `HybridStateStore` as `class` (`Ordinary` or `Checkpoint`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn capture_boundary(
        &mut self,
        coordinator: &mut Inkling2KVCacheCoordinator,
        conv_state: &Inkling2ShortConvState,
        seq_id: SeqId,
        tokens: &[u32],
        boundary: u32,
        cache_salt: u64,
        class: StateClass,
    ) -> Result<Option<EntryId>> {
        let bs = coordinator.block_size().max(1);
        if boundary == 0 || !boundary.is_multiple_of(bs) || boundary as usize > tokens.len() {
            return Ok(None);
        }
        // Ensure all Metal pool writes & ShortConv arrays are materialized before readback.
        coordinator
            .eval_pending_pool_writes_all()
            .map_err(Error::from_reason)?;
        eval_conv_state_arrays(conv_state)?;

        // Register Group 0 full blocks so their chained hashes are live in the prefix cache.
        coordinator
            .register_full_committed_blocks(seq_id, boundary, cache_salt)
            .map_err(Error::from_reason)?;

        // Snapshot raw SWA blocks covering `[max(0, boundary - sliding_window) .. boundary)`.
        let swa_blocks = if coordinator.sliding_adapter().is_some() {
            coordinator
                .read_sliding_raw_blocks_for(seq_id, boundary)
                .map_err(Error::from_reason)?
        } else {
            Vec::new()
        };
        let swa_bytes: u64 = swa_blocks
            .iter()
            .flat_map(|b| b.iter())
            .map(|(k, v)| (k.len() + v.len()) as u64)
            .sum();
        let conv_bytes = conv_state
            .visits
            .iter()
            .map(|v| {
                (v.k.nbytes() + v.v.nbytes() + v.attn_post.nbytes() + v.mlp_post.nbytes()) as u64
            })
            .sum::<u64>();
        let anchor = Inkling2StateAnchor {
            swa_blocks,
            conv_state: conv_state.clone(),
            cached_bytes: swa_bytes.saturating_add(conv_bytes),
        };

        // Retire any previous rolling checkpoint for this sequence when publishing a newer point.
        if let Some(prev_cp) = self.seq_checkpoints.remove(&seq_id) {
            self.retire_entry(prev_cp);
        }

        let lineage = Lineage::text(&tokens[..boundary as usize], cache_salt);
        let inserted = self.store.insert(lineage, anchor, class)?;

        match class {
            StateClass::Checkpoint => {
                self.checkpoints_published += 1;
                self.seq_checkpoints.insert(seq_id, inserted.id);
            }
            StateClass::Ordinary => {
                self.ordinary_published += 1;
                if let Some(disk) = self.disk.as_mut()
                    && boundary as usize >= write_behind_min_tokens()
                    && !disk.write_behind.iter().any(|w| w.id == inserted.id)
                {
                    disk.write_behind.push(WriteBehindEntry {
                        id: inserted.id,
                        published_at: Instant::now(),
                    });
                }
            }
        }
        self.sync_marks(coordinator);
        Ok(Some(inserted.id))
    }

    pub(crate) fn clear_seq_checkpoint(&mut self, seq_id: SeqId) {
        self.seq_checkpoints.remove(&seq_id);
    }

    /// Remove a disposable checkpoint from RAM and SSD (if not currently in flight).
    fn retire_entry(&mut self, id: EntryId) {
        let Some(entry) = self.store.get(id) else {
            return;
        };
        if !entry.checkpoint() {
            return;
        }
        if let Some(disk) = self.disk.as_mut() {
            if disk.pending.as_ref().is_some_and(|p| p.id == id) {
                return;
            }
            if disk.chain.as_ref().is_some_and(|c| c.id == id) {
                disk.chain = None;
            }
            if let Some(copy) = entry.disk().copied() {
                let key = copy.key;
                let tier = Arc::clone(&disk.tier);
                let _ = tier.while_held(|| {
                    tier.manager.remove(&key);
                });
            }
        }
        self.store.remove(id);
    }

    /// Stage `id`'s RAM anchor into the reusable host buffer and enqueue its SSD write.
    pub(crate) fn stage_entry(
        &mut self,
        coordinator: &Inkling2KVCacheCoordinator,
        id: EntryId,
        room: MakeRoom,
        keep_ram: bool,
    ) -> bool {
        let Some(disk) = self.disk.as_mut() else {
            return false;
        };
        if !disk.writable() || disk.writing() {
            return false;
        }
        let Some(entry) = self.store.get(id) else {
            return false;
        };
        if entry.disk().is_some() || !disk.point_fits(entry, room) {
            return false;
        }
        match disk.engine.stage(&mut self.store, id, room, keep_ram, ()) {
            Staged::Started => {
                self.sync_marks(coordinator);
                true
            }
            Staged::Busy => false,
            Staged::Refused => {
                disk.counters.offload_failures += 1;
                false
            }
        }
    }

    /// Reclaim RAM for `id` under governor pressure:
    /// - If it already has a landed disk copy, drop RAM immediately.
    /// - If SSD is writable and not busy, spill it to SSD (`MakeRoom::Evict`) and drop RAM.
    /// - If a write is already in flight, return `Pending` so the ledger tries another victim.
    /// - If no SSD tier or point doesn't fit, remove the entry.
    pub(crate) fn reclaim_entry(
        &mut self,
        coordinator: &Inkling2KVCacheCoordinator,
        id: EntryId,
    ) -> ReclaimOutcome {
        let Some(disk) = self.disk.as_mut() else {
            if let Some(entry) = self.store.get(id)
                && entry.ram().is_some()
            {
                self.store.remove(id);
                self.sync_marks(coordinator);
            }
            return ReclaimOutcome::Erased;
        };
        let step = disk.engine.reclaim(&mut self.store, id, ());
        if step.encode_or_enqueue_failed {
            disk.counters.offload_failures += 1;
        }
        if step.resync {
            self.sync_marks(coordinator);
        }
        match step.outcome {
            Reclaimed::Written => ReclaimOutcome::Spilled,
            Reclaimed::RamDropped => ReclaimOutcome::RamDropped,
            Reclaimed::Pending => ReclaimOutcome::Pending,
            Reclaimed::Erased | Reclaimed::Nothing => ReclaimOutcome::Erased,
        }
    }

    /// Poll completed SSD writes & LRU evictions from `ColdCacheManager`.
    pub(crate) fn poll_offload(&mut self, coordinator: &Inkling2KVCacheCoordinator) -> bool {
        let Some(disk) = self.disk.as_mut() else {
            return false;
        };
        let out = disk.engine.poll_offload(&mut self.store);
        disk.counters.offloads += out.state_landed_ok;
        disk.counters.offload_failures += out.offload_failures;
        let progressed = out.landed || out.resync;
        if progressed {
            self.sync_marks(coordinator);
        }
        progressed
    }

    /// Advance persistence of 8-layer Global KV blocks owed by disk-backed states.
    pub(crate) fn chain_step(
        &mut self,
        coordinator: &Inkling2KVCacheCoordinator,
        deadline: Instant,
    ) -> bool {
        let Some(disk) = self.disk.as_mut() else {
            return false;
        };
        if !disk.writable() {
            return false;
        }
        let kv = KvPoolHandle::from_adapter(coordinator.full_adapter());
        let out = disk.engine.chain_step(&mut self.store, &kv, deadline);
        disk.counters.offload_failures += out.offload_failures;
        if out.step.failed || out.step.landed_all {
            self.sync_marks(coordinator);
        }
        out.step.enqueued > 0 || out.step.landed_all || out.step.failed
    }

    /// Drive write-behind for eligible ordinary states whose delay has elapsed.
    fn drive_write_behind(
        &mut self,
        coordinator: &Inkling2KVCacheCoordinator,
        force: bool,
    ) -> bool {
        let (writable, writing, persistent, has_waiters) = match self.disk.as_ref() {
            Some(d) => (
                d.writable(),
                d.writing(),
                d.tier.persistent,
                d.has_read_waiters(),
            ),
            None => return false,
        };
        if !writable || writing || (!force && has_waiters) {
            return false;
        }
        if !persistent && !force {
            return false;
        }
        let now = Instant::now();
        let delay = Duration::from_secs_f64(write_behind_delay_secs());
        let candidate = {
            let Some(disk) = self.disk.as_mut() else {
                return false;
            };
            disk.write_behind
                .retain(|w| self.store.get(w.id).is_some_and(|e| e.disk().is_none()));
            disk.write_behind
                .iter()
                .position(|w| force || now.duration_since(w.published_at) >= delay)
                .map(|idx| disk.write_behind.remove(idx).id)
        };
        let Some(id) = candidate else {
            return false;
        };
        let room = if persistent {
            MakeRoom::Evict
        } else {
            MakeRoom::Refuse
        };
        self.stage_entry(coordinator, id, room, true)
    }

    /// One tick of the StateDisk tier (called on scheduler drive passes and allocations).
    pub(crate) fn tick(&mut self, coordinator: &Inkling2KVCacheCoordinator, tick: Tick) -> bool {
        let kv = KvPoolHandle::from_adapter(coordinator.full_adapter());
        let out = {
            let Some(disk) = self.disk.as_mut() else {
                return false;
            };
            let kv_ref = disk.writable().then_some(&kv);
            let out = disk.engine.tick(&mut self.store, kv_ref, tick);
            disk.counters.offloads += out.state_landed_ok;
            disk.counters.offload_failures += out.offload_failures;
            out
        };
        let mut progressed = out.progressed || out.resync;
        if out.resync {
            self.sync_marks(coordinator);
        }
        if matches!(tick, Tick::Full { .. }) {
            progressed |= self.drive_write_behind(coordinator, false);
            progressed |= self.poll_offload(coordinator);
        }
        progressed
    }

    /// Synchronously flush all pending write-behind ordinary states & owed Global KV chains
    /// to SSD until `deadline` (used by `close_cache`, `Drop`, and restart tests).
    pub(crate) fn flush_to_disk(
        &mut self,
        coordinator: &Inkling2KVCacheCoordinator,
        deadline: Instant,
    ) -> bool {
        let has_work = |this: &Self| {
            let has_pending_write = this.disk.as_ref().is_some_and(|d| d.writing());
            let has_owed_chain = this
                .store
                .entries()
                .any(|e| e.disk().is_some_and(|c| c.chain_owed));
            let has_waiting_wb = this.disk.as_ref().is_some_and(|d| {
                d.write_behind
                    .iter()
                    .any(|w| this.store.get(w.id).is_some_and(|e| e.disk().is_none()))
            });
            has_pending_write || has_owed_chain || has_waiting_wb
        };
        loop {
            self.poll_offload(coordinator);
            if !has_work(self) {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            let wb_before = self.disk.as_ref().map_or(0, |d| d.write_behind.len());
            let staged = self.drive_write_behind(coordinator, true);
            let chained = self.chain_step(coordinator, deadline);
            let polled = self.poll_offload(coordinator);
            let wb_progressed =
                staged || self.disk.as_ref().map_or(0, |d| d.write_behind.len()) < wb_before;

            if !has_work(self) {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            if polled || chained || wb_progressed {
                continue;
            }
            let Some(disk) = self.disk.as_ref() else {
                return false;
            };
            if !disk.engine.await_landing(deadline) {
                return false;
            }
        }
    }

    /// Flush waiting ordinary states (if persistent) and close the underlying `DiskTier`.
    pub(crate) fn close_cache(
        &mut self,
        coordinator: &Inkling2KVCacheCoordinator,
        deadline: Instant,
    ) -> bool {
        let persistent = self.disk.as_ref().is_some_and(|d| d.tier.persistent);
        let mut all_flushed = true;
        if persistent {
            all_flushed = self.flush_to_disk(coordinator, deadline);
        }
        if let Some(disk) = self.disk.as_ref() {
            all_flushed &= disk.tier.close(deadline);
        }
        if !persistent && let Some(disk) = self.disk.as_mut() {
            disk.engine.strip_disk_copies(&mut self.store);
        }
        all_flushed
    }

    /// Load an entry's [`Inkling2StateAnchor`] — either from RAM (instant clone) or by reading
    /// its 1 content-keyed state blob from SSD into the single reusable host staging buffer.
    pub(crate) fn load_anchor(
        &mut self,
        coordinator: &Inkling2KVCacheCoordinator,
        id: EntryId,
    ) -> Option<(Inkling2StateAnchor, bool)> {
        if let Some(ram) = self.store.get(id).and_then(Entry::ram).cloned() {
            self.ram_hits += 1;
            return Some((ram, false));
        }
        // Wait for an in-flight state write to hand back the staging buffer.
        let deadline = Instant::now()
            + Duration::from_secs_f64(crate::engine::governor::wait::RESOURCE_WAIT_SECONDS);
        loop {
            let disk = self.disk.as_ref()?;
            if disk.staging_free() {
                break;
            }
            if !disk.writing() || crate::engine::governor::governor::cancelled() {
                return None;
            }
            if !disk.engine.await_landing(deadline) {
                return None;
            }
            self.poll_offload(coordinator);
        }
        let disk = self.disk.as_mut()?;
        let decoded = disk.engine.read_into_staging(
            &self.store,
            id,
            |codec, _spans, entry, _copy, staging| {
                let depth = u32::try_from(entry.depth()).ok()?;
                codec.geometry.decode_from(depth, staging).ok()
            },
        );
        if let Some(anchor) = decoded {
            disk.counters.ssd_hits += 1;
            Some((anchor, true))
        } else {
            None
        }
    }

    /// Adopt complete `(state blob + 8-layer Global KV chain)` entries from a persistent tier
    /// on startup using the shared [`StateDiskEngine::adopt_all`].
    pub(crate) fn adopt_all(&mut self) -> u64 {
        let Some(disk) = self.disk.as_mut() else {
            return 0;
        };
        let adopted = disk.engine.adopt_all(&mut self.store, false);
        disk.counters.adopted += adopted;
        adopted
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReclaimOutcome {
    RamDropped,
    Spilled,
    Pending,
    Erased,
}

/// Async SSD restore ticket for `HybridSchedulerBackend::RestoreTicket` (`TurnStatus::WaitingForSsd`).
pub(crate) struct Inkling2Restore {
    pub(crate) phase: Inkling2RestorePhase,
    pub(crate) cache_salt: u64,
    #[allow(dead_code)]
    pub(crate) retried: bool,
    pub(crate) parked_at: Instant,
    pub(crate) read_wait: Option<ReadWait>,
    pub(crate) read_wait_since: Option<f64>,
}

pub(crate) enum Inkling2RestorePhase {
    Reading(ChainPrefetch),
    Landed(ChainPrefetch),
    Lost,
    Cold,
}

impl Inkling2Restore {
    pub(crate) fn lands(&self) -> bool {
        !matches!(self.phase, Inkling2RestorePhase::Reading(_))
    }

    pub(crate) fn held_blocks(&self) -> u32 {
        match &self.phase {
            Inkling2RestorePhase::Reading(p) | Inkling2RestorePhase::Landed(p) => p.held_blocks(),
            Inkling2RestorePhase::Lost | Inkling2RestorePhase::Cold => 0,
        }
    }

    pub(crate) fn reading(prefetch: ChainPrefetch, cache_salt: u64) -> Self {
        Self {
            phase: Inkling2RestorePhase::Reading(prefetch),
            cache_salt,
            retried: false,
            parked_at: Instant::now(),
            read_wait: None,
            read_wait_since: None,
        }
    }
}

/// Forward arithmetic epoch for Inkling-2-Mini SSD prefix fingerprints.
///
/// Bump whenever fused Metal kernels or projection/norm rounding order change so cached SSD
/// prefixes produced by an older build are never adopted and extended under different arithmetic.
const INKLING2_FORWARD_ARITH_VERSION: &str = "v4";

fn query_precision_tag(int8_query_from: Option<u32>) -> String {
    match int8_query_from {
        Some(from) => format!("int8@{from}"),
        None => "bf16".to_string(),
    }
}

/// Snapshot of runtime Metal/precision toggles resolved once and shared between forward dispatch
/// and SSD prefix fingerprinting so changing any toggle isolates the SSD namespace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ForwardArithmeticSettings {
    nax: u8,
    loop_bf16: &'static str,
    lctrl_min: i64,
    lctrl_max: i64,
    rtr_min: i64,
    rtr_max: i64,
    moe_fused: u8,
    attn_res_vec4_max_t: u32,
    moe_mode: u32,
    moe_max_rows: u32,
    moe_col_cutoff_override: i32,
}

impl ForwardArithmeticSettings {
    fn current() -> Self {
        let nax = {
            #[cfg(target_os = "macos")]
            {
                u8::from(unsafe { mlx_sys::mlx_metal_is_nax_available() })
            }
            #[cfg(not(target_os = "macos"))]
            {
                0u8
            }
        };
        let (lctrl_min, lctrl_max) = super::attn_res::fused_loop_ctrl_range();
        let (rtr_min, rtr_max) = super::moe::moe_router_proj_range();
        let moe_fused = u8::from(super::moe::moe_fused_enabled());
        // SAFETY: pure read-only process-cached FFI getters returning scalar integers.
        let (attn_res_vec4_max_t, moe_mode, moe_max_rows, moe_col_cutoff_override) = unsafe {
            (
                mlx_sys::mlx_inkling2_attn_res_vec4_max_t(),
                mlx_sys::mlx_inkling2_moe_mode(),
                mlx_sys::mlx_inkling2_moe_max_rows(),
                mlx_sys::mlx_inkling2_moe_col_cutoff_override(),
            )
        };
        Self {
            nax,
            loop_bf16: super::attn_res::loop_bf16_override_tag(),
            lctrl_min,
            lctrl_max,
            rtr_min,
            rtr_max,
            moe_fused,
            attn_res_vec4_max_t,
            moe_mode,
            moe_max_rows,
            moe_col_cutoff_override,
        }
    }

    fn tag(&self) -> String {
        format!(
            "{INKLING2_FORWARD_ARITH_VERSION}:nax={}:lbf16={}:lctrl={}-{}:rtr={}-{}:moef={}:arvec4={}:moem={}:moer={}:moec={}",
            self.nax,
            self.loop_bf16,
            self.lctrl_min,
            self.lctrl_max,
            self.rtr_min,
            self.rtr_max,
            self.moe_fused,
            self.attn_res_vec4_max_t,
            self.moe_mode,
            self.moe_max_rows,
            self.moe_col_cutoff_override,
        )
    }
}

/// Deterministic tag capturing the forward arithmetic epoch + runtime Metal/precision toggles that
/// can alter cached KV or ShortConv values.
fn forward_arithmetic_tag() -> String {
    ForwardArithmeticSettings::current().tag()
}

/// Compute deterministic `PersistentIdentity` for an Inkling-2-Mini checkpoint + KV/state geometry
/// + resolved prefill query precision policy (`activations`) + forward arithmetic epoch.
pub(crate) fn compute_persistent_identity(
    config: &Inkling2Config,
    coordinator: &Inkling2KVCacheCoordinator,
    geometry: &Inkling2StateGeometry,
    activations: crate::transformer::Activations,
    files: CheckpointFiles<'_>,
    weights: &WeightsResident,
    guard: &CheckpointLoadGuard<'_>,
) -> Option<PersistentIdentity> {
    let config_json = serde_json::to_vec(config).ok()?;
    let global_layers = u64::try_from(global_visit_count(config)).ok()?;
    let kv_heads = u64::try_from(config.num_key_value_heads).ok()?;
    let head_dim = u64::try_from(config.head_dim).ok()?;
    let block_size = u64::from(coordinator.block_size());
    let layout_tag = if coordinator
        .full_adapter()
        .layer_kv_pool()
        .layout()
        .keys_token_major
    {
        "ktm"
    } else {
        "vllm"
    };
    let q_tag = query_precision_tag(super::paged_forward::resolved_int8_query_from(
        coordinator.kv_format(),
        activations,
    ));
    let arith_tag = forward_arithmetic_tag();
    let tier_geo = ColdTierGeometry {
        block_size,
        num_layers: global_layers,
        num_kv_heads: kv_heads,
        head_size: head_dim,
        cache_dtype: format!(
            "{}:{layout_tag}:q={q_tag}:arith={arith_tag}",
            coordinator.kv_format().as_str()
        ),
    };
    let kv_fp = build_model_fingerprint("inkling2", files, Some(&config_json), &tier_geo, weights)?;
    if !guard.stable_after_fingerprint() {
        return None;
    }
    let state_desc = geometry.geometry_string();
    let state_fp = ColdCacheFingerprint::from_components([
        b"inkling2-state-fp:v2".as_slice(),
        kv_fp.as_bytes().as_slice(),
        state_desc.as_bytes(),
    ]);
    let namespace = state_fp.to_hex().get(..32)?.to_string();
    Some(PersistentIdentity {
        namespace,
        state: state_fp,
        kv: kv_fp,
    })
}

/// Compute a deterministic test/path-based `PersistentIdentity` (for direct `load_from_dir` or
/// unit tests without a `CheckpointLoadGuard`).
#[cfg(test)]
pub(crate) fn compute_path_identity(
    model_dir: &Path,
    config: &Inkling2Config,
    coordinator: &Inkling2KVCacheCoordinator,
    geometry: &Inkling2StateGeometry,
    int8_query_from: Option<u32>,
) -> Option<PersistentIdentity> {
    compute_path_identity_with_arith(
        model_dir,
        config,
        coordinator,
        geometry,
        int8_query_from,
        ForwardArithmeticSettings::current(),
    )
}

#[cfg(test)]
fn compute_path_identity_with_arith(
    model_dir: &Path,
    config: &Inkling2Config,
    coordinator: &Inkling2KVCacheCoordinator,
    geometry: &Inkling2StateGeometry,
    int8_query_from: Option<u32>,
    arith: ForwardArithmeticSettings,
) -> Option<PersistentIdentity> {
    let config_json = serde_json::to_vec(config).ok()?;
    let canon = model_dir
        .canonicalize()
        .unwrap_or_else(|_| model_dir.to_path_buf());
    let q_tag = query_precision_tag(int8_query_from);
    let arith_tag = arith.tag();
    let kv_fp = ColdCacheFingerprint::from_components([
        b"inkling2-path-kv:v2".as_slice(),
        canon.to_string_lossy().as_bytes(),
        config_json.as_slice(),
        coordinator.kv_format().as_str().as_bytes(),
        q_tag.as_bytes(),
        arith_tag.as_bytes(),
        &coordinator.block_size().to_le_bytes(),
    ]);
    let state_desc = geometry.geometry_string();
    let state_fp = ColdCacheFingerprint::from_components([
        b"inkling2-path-state:v2".as_slice(),
        kv_fp.as_bytes().as_slice(),
        state_desc.as_bytes(),
    ]);
    let namespace = state_fp.to_hex().get(..32)?.to_string();
    Some(PersistentIdentity {
        namespace,
        state: state_fp,
        kv: kv_fp,
    })
}

/// Open Inkling-2-Mini's StateDisk tier, attach `ColdContextBacking::StateOnly` to Group 0,
/// and return `(ColdTierContext, Inkling2StateDisk)`.
pub(crate) fn open_inkling2_state_disk(
    config: &Inkling2Config,
    coordinator: &Inkling2KVCacheCoordinator,
    options: &DiskTierOptions,
    identity: impl FnOnce() -> Option<PersistentIdentity>,
    model_id: u64,
) -> Option<(ColdTierContext, Inkling2StateDisk)> {
    let geometry = Inkling2StateGeometry::from_coordinator(config, coordinator).ok()?;
    let state_bytes = geometry.payload_bytes();
    if state_bytes == 0 {
        return None;
    }
    let tier = open_disk_tier(options, identity, state_bytes)?;
    let global_block_bytes = block_object_bytes(coordinator.full_adapter().layer_kv_pool());
    let disk = Inkling2StateDisk::new(Arc::clone(&tier), geometry, global_block_bytes).ok()?;
    disk.wake_model_on_landing(model_id);
    let context = state_context(&tier);
    Some((context, disk))
}

/// Compute rolling prefill checkpoint boundaries in `(cached_prefix, prompt_tokens)` aligned to
/// `block_size`.
pub(crate) fn planned_rolling_checkpoints(
    prompt_tokens: u32,
    cached_prefix: u32,
    block_size: u32,
) -> Vec<u32> {
    let interval = checkpoint_interval_tokens();
    let bs = block_size.max(1);
    if interval == 0 || prompt_tokens <= cached_prefix {
        return Vec::new();
    }
    let step = interval.next_multiple_of(bs);
    let mut out = Vec::new();
    let mut boundary = (cached_prefix / step)
        .saturating_add(1)
        .saturating_mul(step);
    while boundary < prompt_tokens {
        out.push(boundary);
        match boundary.checked_add(step) {
            Some(next) => boundary = next,
            None => break,
        }
    }
    out
}

fn eval_conv_state_arrays(state: &Inkling2ShortConvState) -> Result<()> {
    let mut arrays: Vec<&MxArray> = Vec::with_capacity(state.visits.len() * 4);
    for v in &state.visits {
        arrays.push(&v.k);
        arrays.push(&v.v);
        arrays.push(&v.attn_post);
        arrays.push(&v.mlp_post);
    }
    MxArray::eval_arrays(&arrays)
}

/// Helper: restore an [`Inkling2StateAnchor`] onto `seq_id` across Group 1 (SWA raw blocks)
/// and `conv_states` (48-visit ShortConv state). Caller has already restored Group 0 (Global KV)
/// at `boundary`.
pub(crate) fn install_anchor_swa_and_conv(
    coordinator: &mut Inkling2KVCacheCoordinator,
    conv_states: &mut HashMap<SeqId, Inkling2ShortConvState>,
    seq_id: SeqId,
    prompt_tokens: &[u32],
    boundary: u32,
    anchor: Inkling2StateAnchor,
) -> Result<()> {
    if coordinator.sliding_adapter().is_some() {
        coordinator
            .restore_sliding_raw_blocks_for(seq_id, prompt_tokens, boundary, &anchor.swa_blocks)
            .map_err(Error::from_reason)?;
    }
    conv_states.insert(seq_id, anchor.conv_state);
    Ok(())
}

/// Helper: check whether `adapter` has a hot or restorable chain for `tokens` using `begin_chain_prefetch`.
pub(crate) fn start_global_chain_prefetch(
    adapter: &mut PagedKVCacheAdapter,
    tokens: &[u32],
    depth: usize,
    cache_salt: u64,
    state_key: Option<ColdCacheKey>,
) -> Option<ChainPrefetchStart> {
    let cap = u32::try_from(tokens.len()).ok()?.saturating_sub(1);
    let bs = adapter.block_size();
    let keys = crate::engine::build_paged_extra_keys(tokens.len(), bs, &[]);
    Some(adapter.begin_chain_prefetch(
        tokens,
        cap,
        depth,
        mlx_paged_attn::PrefixKeys::PerBlock(&keys),
        cache_salt,
        state_key,
    ))
}

/// Poll a `ChainPrefetch` inside an [`Inkling2Restore`] within the pass's upload budget.
pub(crate) fn poll_restore_ticket(
    adapter: &PagedKVCacheAdapter,
    restore: &mut Inkling2Restore,
    uploads: &mut RestoreUploadBudget,
) {
    let Inkling2RestorePhase::Reading(prefetch) = &mut restore.phase else {
        return;
    };
    let polled = adapter.poll_chain_prefetch(prefetch, uploads);
    if polled == crate::transformer::paged_kv_cache_adapter::ChainPrefetchPoll::Pending {
        return;
    }
    let Inkling2RestorePhase::Reading(prefetch) =
        std::mem::replace(&mut restore.phase, Inkling2RestorePhase::Cold)
    else {
        return;
    };
    if matches!(
        polled,
        crate::transformer::paged_kv_cache_adapter::ChainPrefetchPoll::Landed { .. }
    ) {
        restore.phase = Inkling2RestorePhase::Landed(prefetch);
    } else {
        drop(prefetch);
        restore.phase = Inkling2RestorePhase::Lost;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::inkling2::kv_cache::Inkling2PagedConfig;

    fn tiny_inkling2_config() -> Inkling2Config {
        Inkling2Config::from_json_str(
            r#"{
                "model_type": "inkling2",
                "hidden_size": 64,
                "num_hidden_layers": 6,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "max_position_embeddings": 8192,
                "sliding_window": 64,
                "sliding_window_pattern": [true, true, true, true, true, false]
            }"#,
            true,
        )
        .expect("tiny inkling2 config")
    }

    #[test]
    fn rolling_checkpoint_planner_emits_block_aligned_rungs() {
        let _guard = CHECKPOINT_INTERVAL_OVERRIDE.with(|c| c.replace(Some(4096)));
        assert_eq!(planned_rolling_checkpoints(10_000, 0, 32), vec![4096, 8192]);
        assert_eq!(planned_rolling_checkpoints(10_000, 4096, 32), vec![8192]);
        assert_eq!(planned_rolling_checkpoints(4096, 0, 32), Vec::<u32>::new());
        CHECKPOINT_INTERVAL_OVERRIDE.with(|c| c.set(None));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn state_blob_encode_decode_roundtrip_int8_and_bf16() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let cfg = tiny_inkling2_config();
        for kv_format in [KvFormat::Int8, KvFormat::Bf16] {
            let paged_cfg = Inkling2PagedConfig::default_for_config(&cfg)
                .with_kv_format(kv_format)
                .with_max_model_len(512)
                .with_max_concurrent_sequences(2);
            let coord =
                Inkling2KVCacheCoordinator::new_for_config(&cfg, paged_cfg).expect("coordinator");
            let geom = Inkling2StateGeometry::from_coordinator(&cfg, &coord).expect("geometry");
            assert_eq!(geom.swa_layers, 5);
            assert_eq!(geom.max_swa_blocks, 2); // sliding_window=64, bs=32
            assert_eq!(geom.num_visits, 6);

            // Test both sub-window (boundary=32 -> 1 SWA block) and full-window (boundary=128 -> 2 SWA blocks).
            for boundary in [32u32, 128u32] {
                let live_blocks = geom.live_swa_blocks(boundary) as usize;
                let mut swa_blocks = Vec::with_capacity(live_blocks);
                for b in 0..live_blocks {
                    let mut layers = Vec::with_capacity(geom.swa_layers as usize);
                    for l in 0..geom.swa_layers as usize {
                        let k = vec![
                            ((b * 31 + l * 7 + 1) % 251) as u8;
                            geom.swa_key_blob_bytes as usize
                        ];
                        let v = vec![
                            ((b * 17 + l * 13 + 3) % 251) as u8;
                            geom.swa_value_blob_bytes as usize
                        ];
                        layers.push((k, v));
                    }
                    swa_blocks.push(layers);
                }
                let conv = Inkling2ShortConvState::new(
                    cfg.num_layer_visits(),
                    cfg.sconv_kernel_size,
                    cfg.num_key_value_heads * cfg.head_dim,
                    cfg.hidden_size,
                    DType::BFloat16,
                )
                .expect("conv state");
                let anchor = Inkling2StateAnchor {
                    swa_blocks: swa_blocks.clone(),
                    conv_state: conv,
                    cached_bytes: geom.payload_bytes(),
                };
                let mut buf = vec![0xAAu8; geom.payload_bytes() as usize];
                geom.encode_into(boundary, &anchor, &mut buf)
                    .expect("encode_into");
                let decoded = geom.decode_from(boundary, &buf).expect("decode_from");
                assert_eq!(decoded.swa_blocks, swa_blocks);
                assert_eq!(decoded.conv_state.visits.len(), cfg.num_layer_visits());
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn state_disk_flush_spill_restore_and_adopt_roundtrip() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use crate::disk_tier::{CacheDisk, DiskTierOptions};

        WRITE_BEHIND_MIN_TOKENS_OVERRIDE.with(|c| c.set(Some(32)));
        WRITE_BEHIND_DELAY_OVERRIDE.with(|c| c.set(Some(0.0)));

        let cfg = tiny_inkling2_config();
        let temp_root =
            std::env::temp_dir().join(format!("inkling2-state-disk-unit-{}", uuid::Uuid::new_v4()));
        let cache_dir = temp_root.join("cache");
        std::fs::create_dir_all(&cache_dir).expect("create cache dir");

        let disk_opts = DiskTierOptions {
            cache_disk: CacheDisk::Bytes(512u64 << 20),
            persistent: true,
            cache_dir: Some(cache_dir.clone()),
            temp_root: Some(temp_root.clone()),
        };

        let paged_cfg = Inkling2PagedConfig::default_for_config(&cfg)
            .with_kv_format(KvFormat::Int8)
            .with_max_model_len(512)
            .with_max_concurrent_sequences(2);
        let recency = Arc::new(CacheRecency::new());
        let mut coord = Inkling2KVCacheCoordinator::new_for_config_with_recency(
            &cfg,
            paged_cfg,
            Arc::clone(&recency),
        )
        .expect("coord");
        let geom = Inkling2StateGeometry::from_coordinator(&cfg, &coord).expect("geom");
        let id = compute_path_identity(&temp_root, &cfg, &coord, &geom, Some(1024));
        // Switching activation policy (INT8 query threshold vs BF16 queries) must isolate SSD cache identity.
        let id_bf16_q = compute_path_identity(&temp_root, &cfg, &coord, &geom, None)
            .expect("bf16 query identity");
        let id_int8_q = id.as_ref().expect("int8 query identity");
        assert_ne!(id_int8_q.namespace, id_bf16_q.namespace);
        assert_ne!(id_int8_q.kv, id_bf16_q.kv);
        assert_ne!(id_int8_q.state, id_bf16_q.state);
        let (ctx, disk) = open_inkling2_state_disk(&cfg, &coord, &disk_opts, move || id, 0)
            .expect("open_inkling2_state_disk");
        coord.set_full_cold_tier(ctx);
        let mut store = Inkling2StateStore::new(coord.block_size(), recency);
        store.disk = Some(disk);

        // Record 96 tokens (= 3 full blocks of 32) into seq 1.
        let tokens: Vec<u32> = (100u32..196u32).collect();
        coord.reset_scheduled_request(1).expect("reset seq 1");
        coord.record_tokens_all(1, &tokens).expect("record tokens");
        let conv = Inkling2ShortConvState::new(
            cfg.num_layer_visits(),
            cfg.sconv_kernel_size,
            cfg.num_key_value_heads * cfg.head_dim,
            cfg.hidden_size,
            DType::BFloat16,
        )
        .expect("conv");

        let entry_id = store
            .capture_boundary(&mut coord, &conv, 1, &tokens, 64, 0, StateClass::Ordinary)
            .expect("capture_boundary")
            .expect("entry_id");
        assert!(store.is_restorable(&coord, entry_id), "restorable in RAM");

        let flushed = store.flush_to_disk(&coord, Instant::now() + Duration::from_secs(10));
        let e_after_flush = store.store.get(entry_id).expect("entry exists");
        eprintln!(
            "[unit] after flush: flushed={flushed} disk={:?} offloads={} failures={}",
            e_after_flush.disk(),
            store.disk_counters().offloads,
            store.disk_counters().offload_failures,
        );
        assert!(flushed, "flush_to_disk must finish");
        assert!(
            e_after_flush
                .disk()
                .is_some_and(|d| d.landed && !d.chain_owed),
            "entry must have landed state blob + full KV chain on disk, got {:?}",
            e_after_flush.disk()
        );

        // Drop RAM + purge hot pool -> must still be restorable from SSD!
        store.drop_all_ram_keeping_disk();
        coord.release_all_and_purge().expect("purge hot pools");
        let query: Vec<u32> = (100u32..170u32).collect();
        let hit = store
            .lookup_deepest_restorable(&coord, &query, 0)
            .expect("SSD lookup hit");
        assert_eq!(hit.depth, 64);
        assert!(!store.chain_is_hot(&coord, hit.id));
        let (_, from_ssd) = store
            .load_anchor(&coord, hit.id)
            .expect("load_anchor from SSD");
        assert!(from_ssd);

        // Close and reopen persistent tier -> adopt_all must recover the entry!
        assert!(store.close_cache(&coord, Instant::now() + Duration::from_secs(10)));
        drop(store);
        drop(coord);

        let recency2 = Arc::new(CacheRecency::new());
        let mut coord2 = Inkling2KVCacheCoordinator::new_for_config_with_recency(
            &cfg,
            paged_cfg,
            Arc::clone(&recency2),
        )
        .expect("coord2");
        let geom2 = Inkling2StateGeometry::from_coordinator(&cfg, &coord2).expect("geom2");
        // Reopening with a different activation query policy (None = BF16) must miss the old cache!
        let id_mismatch = compute_path_identity(&temp_root, &cfg, &coord2, &geom2, None);
        let (_ctx_miss, disk_miss) =
            open_inkling2_state_disk(&cfg, &coord2, &disk_opts, move || id_mismatch, 0)
                .expect("open mismatched policy disk");
        let mut store_miss = Inkling2StateStore::new(coord2.block_size(), Arc::clone(&recency2));
        store_miss.disk = Some(disk_miss);
        assert_eq!(
            store_miss.adopt_all(),
            0,
            "different activation query precision must NOT adopt old SSD cache"
        );
        drop(store_miss);

        // Reopening with a different AttnRes vec4 threshold (`MLX_INKLING2_ATTN_RES_VEC4_MAX_T`)
        // must also isolate the SSD namespace and reject the previous prefix!
        let mut alt_arith = ForwardArithmeticSettings::current();
        alt_arith.attn_res_vec4_max_t = if alt_arith.attn_res_vec4_max_t == 0 {
            64
        } else {
            0
        };
        let id_alt_attn_res = compute_path_identity_with_arith(
            &temp_root,
            &cfg,
            &coord2,
            &geom2,
            Some(1024),
            alt_arith,
        );
        let (_ctx_ar_miss, disk_ar_miss) =
            open_inkling2_state_disk(&cfg, &coord2, &disk_opts, move || id_alt_attn_res, 0)
                .expect("open mismatched attn_res disk");
        let mut store_ar_miss = Inkling2StateStore::new(coord2.block_size(), Arc::clone(&recency2));
        store_ar_miss.disk = Some(disk_ar_miss);
        assert_eq!(
            store_ar_miss.adopt_all(),
            0,
            "different MLX_INKLING2_ATTN_RES_VEC4_MAX_T must NOT adopt old SSD cache"
        );
        drop(store_ar_miss);

        let id2 = compute_path_identity(&temp_root, &cfg, &coord2, &geom2, Some(1024));
        let (ctx2, disk2) = open_inkling2_state_disk(&cfg, &coord2, &disk_opts, move || id2, 0)
            .expect("reopen state disk");
        coord2.set_full_cold_tier(ctx2);
        let mut store2 = Inkling2StateStore::new(coord2.block_size(), recency2);
        store2.disk = Some(disk2);
        let adopted = store2.adopt_all();
        assert_eq!(adopted, 1, "expected 1 adopted state on restart");
        let hit2 = store2
            .lookup_deepest_restorable(&coord2, &query, 0)
            .expect("post-restart lookup hit");
        assert_eq!(hit2.depth, 64);

        let _ = std::fs::remove_dir_all(&temp_root);
        WRITE_BEHIND_MIN_TOKENS_OVERRIDE.with(|c| c.set(None));
        WRITE_BEHIND_DELAY_OVERRIDE.with(|c| c.set(None));
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "Requires real Inkling-2-Mini GGUF on disk (~18 GB)"]
    fn inkling2_cold_tier_parity() {
        use std::path::Path;
        use std::sync::atomic::AtomicBool;

        use tokio::sync::mpsc;

        use crate::disk_tier::{CacheDisk, DiskTierOptions};
        use crate::engine::backend::ResetScope;
        use crate::engine::cmd::ChatCmd;
        use crate::engine::governor::ledger::UnitKey;
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        use crate::engine::types::ChatConfig;
        use crate::models::inkling2::model::Inkling2LoadOptions;
        use crate::models::inkling2::persistence::{load_from_dir, test_real_model_path};
        use crate::models::inkling2::scheduler::{Inkling2Cmd, Inkling2SchedulerState};
        use crate::tokenizer::ChatMessage;

        let model_path = test_real_model_path();
        assert!(
            Path::new(&model_path).exists(),
            "Real model path missing: {model_path}"
        );
        let load_rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("create load runtime");

        let temp_root = std::env::temp_dir().join(format!(
            "inkling2-cold-tier-parity-{}",
            uuid::Uuid::new_v4()
        ));
        let cache_dir = temp_root.join("persistent-cache");
        std::fs::create_dir_all(&cache_dir).expect("create temp cache_dir");

        // Build a long >4,200-token prompt crossing both SWA=1024 and the 4,096-token rolling
        // prefill checkpoint boundary, embedding an early needle `OMEGA-4829` that can ONLY be
        // recalled if the 8 Global KV layers + 40 SWA layers + 48 ShortConv states restore bit-exact!
        let mut long_prefix = String::from(
            "Remember the secret verification token OMEGA-4829 from the very beginning of this document.\n\n",
        );
        for i in 0..135 {
            long_prefix.push_str(&format!(
                "Paragraph {i}: The quick brown fox jumps over the lazy dog while verifying \
                 hybrid sliding-window attention and content-addressed state caching across long contexts.\n"
            ));
        }
        let prompt_a = format!(
            "{long_prefix}\nTask: recall the exact secret verification token from this document."
        );
        let prompt_b = format!(
            "{long_prefix}\nTask: recall the exact secret verification token from this document. Reply with ONLY the token."
        );
        let max_tokens = 48;

        let make_msg = |text: &str| {
            vec![ChatMessage {
                role: "user".to_string(),
                content: text.to_string(),
                tool_calls: None,
                tool_call_id: None,
                is_error: None,
                reasoning_content: None,
                thinking_enabled: None,
                images: None,
                audio: None,
            }]
        };
        let make_cfg = |owner: &str, salt: Option<&str>| ChatConfig {
            max_new_tokens: Some(max_tokens),
            temperature: Some(0.0),
            top_p: Some(1.0),
            top_k: Some(0),
            reasoning_effort: Some("none".to_string()),
            reuse_cache: Some(true),
            cache_salt: salt.map(str::to_string),
            cache_owner_id: Some(owner.to_string()),
            ..ChatConfig::default()
        };

        // Enable low write-behind threshold/delay so StateDisk persists promptly during the test.
        WRITE_BEHIND_MIN_TOKENS_OVERRIDE.with(|c| c.set(Some(64)));
        WRITE_BEHIND_DELAY_OVERRIDE.with(|c| c.set(Some(0.0)));

        // =====================================================================
        // Phase 1: Fresh cold baseline (no reuse) on real model
        // =====================================================================
        let mut inner = load_rt
            .block_on(load_from_dir(&model_path))
            .expect("load real Inkling-2-Mini");
        let disk_opts = DiskTierOptions {
            cache_disk: CacheDisk::Bytes(8u64 << 30),
            persistent: true,
            cache_dir: Some(cache_dir.clone()),
            temp_root: Some(temp_root.clone()),
        };
        let geom = Inkling2StateGeometry::from_coordinator(
            &inner.config,
            &inner.paged.as_ref().expect("paged").coordinator,
        )
        .expect("geom");
        let id = compute_path_identity(
            Path::new(&model_path),
            &inner.config,
            &inner.paged.as_ref().expect("paged").coordinator,
            &geom,
            crate::models::inkling2::paged_forward::resolved_int8_query_from(
                inner.paged.as_ref().expect("paged").coordinator.kv_format(),
                inner.activations,
            ),
        );
        assert!(
            inner.attach_state_disk(&disk_opts, move || id).is_some(),
            "StateDisk must open"
        );

        let mut sched = Inkling2SchedulerState::new(inner).expect("scheduler");
        let (tx, mut rx) = mpsc::unbounded_channel::<Inkling2Cmd>();
        let drive_until = |sched: &mut Inkling2SchedulerState,
                           rx: &mut mpsc::UnboundedReceiver<Inkling2Cmd>,
                           mut done_rx: tokio::sync::oneshot::Receiver<
            Result<crate::engine::types::ChatResult>,
        >|
         -> crate::engine::types::ChatResult {
            loop {
                match done_rx.try_recv() {
                    Ok(res) => return res.expect("turn succeeded"),
                    Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                        let _ = sched.drive(rx);
                    }
                    Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                        panic!("reply channel closed unexpectedly");
                    }
                }
            }
        };

        // 1a. Fresh baseline for `prompt_b` (isolated salt, then reset caches so nothing is shared).
        let (rep_base, rx_base) = tokio::sync::oneshot::channel();
        tx.send(Inkling2Cmd::Chat(Box::new(ChatCmd::SessionStart {
            messages: make_msg(&prompt_b),
            config: make_cfg("fresh-baseline", Some("fresh-baseline-salt")),
            reply: rep_base,
            cancelled: Arc::new(AtomicBool::new(false)),
        })))
        .unwrap();
        let baseline_b = drive_until(&mut sched, &mut rx, rx_base);
        eprintln!(
            "[cold_tier_parity] Fresh baseline prompt_b: tokens={} raw={:?}",
            baseline_b.num_tokens, baseline_b.raw_text
        );
        assert!(
            baseline_b.raw_text.contains("OMEGA-4829"),
            "Expected needle recall OMEGA-4829 in baseline, got {:?}",
            baseline_b.raw_text
        );
        // Reset caches cleanly before testing StateDisk capture & reuse on default salt.
        let (rep_reset, rx_reset) = tokio::sync::oneshot::channel();
        tx.send(Inkling2Cmd::Chat(Box::new(ChatCmd::ResetCaches {
            reply: rep_reset,
        })))
        .unwrap();
        while rx_reset.is_empty() {
            let _ = sched.drive(&mut rx);
        }
        rx_reset.blocking_recv().unwrap().expect("reset caches");

        // =====================================================================
        // Phase 2: Prefill `prompt_a` (>4096 tokens) -> Rolling Checkpoint + Ordinary state
        // =====================================================================
        let (rep_a, rx_a) = tokio::sync::oneshot::channel();
        tx.send(Inkling2Cmd::Chat(Box::new(ChatCmd::SessionStart {
            messages: make_msg(&prompt_a),
            config: make_cfg("owner-a", None),
            reply: rep_a,
            cancelled: Arc::new(AtomicBool::new(false)),
        })))
        .unwrap();
        let res_a = drive_until(&mut sched, &mut rx, rx_a);
        let counters_after_a = sched
            .inner
            .paged
            .as_ref()
            .unwrap()
            .state_store
            .disk_counters();
        eprintln!(
            "[cold_tier_parity] After prompt_a: prompt_tokens={} checkpoints={} ordinary={} raw={:?}",
            res_a.prompt_tokens,
            counters_after_a.checkpoints_published,
            counters_after_a.ordinary_published,
            res_a.raw_text
        );
        assert!(
            res_a.prompt_tokens > 4096,
            "Prompt must exceed 4096 tokens to trigger rolling checkpoint, got {}",
            res_a.prompt_tokens
        );
        assert!(
            counters_after_a.checkpoints_published >= 1,
            "Expected >=1 rolling prefill checkpoint at 4096 tokens"
        );
        assert!(
            counters_after_a.ordinary_published >= 1,
            "Expected >=1 turn-end ordinary state"
        );

        // Verify governor victim order places disposable checkpoints BEFORE ordinary states.
        let units = sched.inner.victim_units(
            &crate::engine::governor::ledger::VictimQuery {
                rights: crate::engine::governor::ledger::TakeRights {
                    units: true,
                    waits_for_writes: true,
                    waits_for_landings: true,
                    in_use: false,
                    certain_reuse: crate::engine::governor::ledger::CertainReuse::None,
                    reserves: crate::engine::governor::ledger::Reserves::None,
                    rebuilt_pools: false,
                },
                need: crate::engine::governor::ledger::Need::default(),
            },
            &[],
        );
        let has_ordinary_in_merged = units
            .merged
            .iter()
            .any(|v| matches!(v.key, UnitKey::Store(_)));
        assert!(
            has_ordinary_in_merged,
            "Ordinary states must appear in merged LRU victim run"
        );

        // =====================================================================
        // Phase 3: RAM prefix hit on `prompt_b` (different owner sharing >4096 prefix)
        // =====================================================================
        let (rep_ram, rx_ram) = tokio::sync::oneshot::channel();
        tx.send(Inkling2Cmd::Chat(Box::new(ChatCmd::SessionStart {
            messages: make_msg(&prompt_b),
            config: make_cfg("owner-ram-hit", None),
            reply: rep_ram,
            cancelled: Arc::new(AtomicBool::new(false)),
        })))
        .unwrap();
        let ram_hit_b = drive_until(&mut sched, &mut rx, rx_ram);
        let counters_ram = sched
            .inner
            .paged
            .as_ref()
            .unwrap()
            .state_store
            .disk_counters();
        eprintln!(
            "[cold_tier_parity] RAM hit prompt_b: ram_hits={} cached_tokens={} raw={:?}",
            counters_ram.ram_hits, ram_hit_b.cached_tokens, ram_hit_b.raw_text
        );
        assert!(counters_ram.ram_hits >= 1, "Expected RAM state hit");
        assert!(
            ram_hit_b.cached_tokens > 4000,
            "Expected >4000 cached tokens"
        );
        assert_eq!(
            ram_hit_b.raw_text, baseline_b.raw_text,
            "RAM prefix hit diverged from fresh prefill!"
        );

        // =====================================================================
        // Phase 3b: Contended concurrent RAM hit baseline (`BatchedDecode M=2` beside `active-lane`)
        // Captures the exact concurrent `M=2` greedy output for `prompt_b` from RAM (4,768 tokens)
        // so Phase 4 verifies `WaitingForSsd` async SSD restore is bit-exact under the identical
        // `M=2` batch schedule (while Phase 3 & Phase 5 verify bit-exact `M=1` parity vs Phase 1).
        // =====================================================================
        let run_contended_pair = |sched: &mut Inkling2SchedulerState,
                                  tx: &mpsc::UnboundedSender<Inkling2Cmd>,
                                  rx: &mut mpsc::UnboundedReceiver<Inkling2Cmd>,
                                  active_owner: &str,
                                  active_salt: &str,
                                  target_owner: &str|
         -> crate::engine::types::ChatResult {
            let (rep_active, mut rx_active) = tokio::sync::oneshot::channel();
            let (rep_target, mut rx_target) = tokio::sync::oneshot::channel();
            tx.send(Inkling2Cmd::Chat(Box::new(ChatCmd::SessionStart {
                messages: make_msg("Count from 1 to 10."),
                config: make_cfg(active_owner, Some(active_salt)),
                reply: rep_active,
                cancelled: Arc::new(AtomicBool::new(false)),
            })))
            .unwrap();
            tx.send(Inkling2Cmd::Chat(Box::new(ChatCmd::SessionStart {
                messages: make_msg(&prompt_b),
                config: make_cfg(target_owner, None),
                reply: rep_target,
                cancelled: Arc::new(AtomicBool::new(false)),
            })))
            .unwrap();
            let mut active_done = None;
            let mut target_done = None;
            while active_done.is_none() || target_done.is_none() {
                if active_done.is_none()
                    && let Ok(res) = rx_active.try_recv()
                {
                    active_done = Some(res.expect("active lane succeeded"));
                }
                if target_done.is_none()
                    && let Ok(res) = rx_target.try_recv()
                {
                    target_done = Some(res.expect("contended target turn succeeded"));
                }
                if active_done.is_none() || target_done.is_none() {
                    let _ = sched.drive(rx);
                }
            }
            target_done.unwrap()
        };
        let contended_ram_b = run_contended_pair(
            &mut sched,
            &tx,
            &mut rx,
            "active-lane-ram",
            "active-lane-ram-salt",
            "owner-ram-contended",
        );
        assert!(
            contended_ram_b.raw_text.contains("OMEGA-4829"),
            "Expected needle recall OMEGA-4829 in contended RAM hit, got {:?}",
            contended_ram_b.raw_text
        );

        // =====================================================================
        // Phase 4: SSD spill + Async `WaitingForSsd` restore beside active decode lane
        // =====================================================================
        {
            let rt = sched.inner.paged.as_mut().unwrap();
            let flushed = rt
                .state_store
                .flush_to_disk(&rt.coordinator, Instant::now() + Duration::from_secs(30));
            assert!(flushed, "StateDisk flush_to_disk timed out");
            // Drop all RAM copies and evict hot Group 0 KV blocks so restore MUST read from SSD!
            rt.state_store.drop_all_ram_keeping_disk();
            rt.coordinator
                .evict_all_hot_blocks_without_suppress()
                .expect("evict hot pools");
            rt.conv_states.clear();
            rt.block_boundary_conv.clear();
        }

        // Submit an unrelated active turn first so `scheduler.has_live_turns()` is true, then
        // submit `prompt_b` right behind it to trigger contended `WaitingForSsd` async restore!
        let ssd_hit_b = run_contended_pair(
            &mut sched,
            &tx,
            &mut rx,
            "active-lane",
            "active-lane-salt",
            "owner-ssd-async",
        );
        let counters_ssd = sched
            .inner
            .paged
            .as_ref()
            .unwrap()
            .state_store
            .disk_counters();
        eprintln!(
            "[cold_tier_parity] Async WaitingForSsd restore prompt_b: ssd_hits={} async_ssd_restores={} cached_tokens={} raw={:?}",
            counters_ssd.ssd_hits,
            counters_ssd.async_ssd_restores,
            ssd_hit_b.cached_tokens,
            ssd_hit_b.raw_text
        );
        assert!(counters_ssd.ssd_hits >= 1, "Expected SSD state hit");
        assert!(
            counters_ssd.async_ssd_restores >= 1,
            "Expected contended WaitingForSsd async restore to engage!"
        );
        assert!(
            ssd_hit_b.cached_tokens > 4000,
            "Expected >4000 restored tokens from SSD, got {}",
            ssd_hit_b.cached_tokens
        );
        assert!(
            ssd_hit_b.raw_text.contains("OMEGA-4829"),
            "Expected needle recall OMEGA-4829 in async SSD restore, got {:?}",
            ssd_hit_b.raw_text
        );
        assert_eq!(
            ssd_hit_b.raw_text, contended_ram_b.raw_text,
            "Async WaitingForSsd restore diverged from contended RAM baseline!"
        );

        // Flush & close persistent StateDisk cleanly before simulated restart.
        let closed = sched
            .inner
            .close_state_disk(Instant::now() + Duration::from_secs(30));
        assert!(closed, "Persistent close_state_disk must flush cleanly");
        let _ = crate::engine::backend::ChatBackend::reset_caches(
            &mut sched.inner,
            ResetScope::Command,
        );
        drop(sched);

        // =====================================================================
        // Phase 5: Persistent Restart (`adopt_states`) -> exact parity vs fresh prefill!
        // =====================================================================
        let mut restarted = load_rt
            .block_on(load_from_dir(&model_path))
            .expect("reload Inkling-2-Mini for restart adoption test");
        let geom2 = Inkling2StateGeometry::from_coordinator(
            &restarted.config,
            &restarted.paged.as_ref().expect("paged").coordinator,
        )
        .expect("geom2");
        let id2 = compute_path_identity(
            Path::new(&model_path),
            &restarted.config,
            &restarted.paged.as_ref().expect("paged").coordinator,
            &geom2,
            crate::models::inkling2::paged_forward::resolved_int8_query_from(
                restarted
                    .paged
                    .as_ref()
                    .expect("paged")
                    .coordinator
                    .kv_format(),
                restarted.activations,
            ),
        );
        assert!(
            restarted
                .attach_state_disk(&disk_opts, move || id2)
                .is_some(),
            "Restarted persistent StateDisk must reopen"
        );
        let adopted_count = restarted
            .paged
            .as_ref()
            .unwrap()
            .state_store
            .disk_counters()
            .adopted;
        eprintln!("[cold_tier_parity] Restart adopted {adopted_count} state(s) via adopt_states");
        assert!(
            adopted_count >= 1,
            "Expected >=1 adopted persistent state on restart!"
        );

        let mut sched2 = Inkling2SchedulerState::new(restarted).expect("scheduler2");
        let (tx2, mut rx2) = mpsc::unbounded_channel::<Inkling2Cmd>();
        let (rep_restart, rx_restart) = tokio::sync::oneshot::channel();
        tx2.send(Inkling2Cmd::Chat(Box::new(ChatCmd::SessionStart {
            messages: make_msg(&prompt_b),
            config: make_cfg("owner-after-restart", None),
            reply: rep_restart,
            cancelled: Arc::new(AtomicBool::new(false)),
        })))
        .unwrap();
        let restart_b = drive_until(&mut sched2, &mut rx2, rx_restart);
        let counters_restart = sched2
            .inner
            .paged
            .as_ref()
            .unwrap()
            .state_store
            .disk_counters();
        eprintln!(
            "[cold_tier_parity] Post-restart SSD hit prompt_b: adopted={} ssd_hits={} cached_tokens={} raw={:?}",
            counters_restart.adopted,
            counters_restart.ssd_hits,
            restart_b.cached_tokens,
            restart_b.raw_text
        );
        assert!(
            counters_restart.ssd_hits >= 1,
            "Expected SSD hit after restart"
        );
        assert!(
            restart_b.cached_tokens > 4000,
            "Expected >4000 cached tokens after restart"
        );
        assert_eq!(
            restart_b.raw_text, baseline_b.raw_text,
            "Post-restart adopted state diverged from fresh prefill!"
        );

        let _ = Inkling2LoadOptions::default();
        drop(sched2);
        let _ = std::fs::remove_dir_all(&temp_root);
        WRITE_BEHIND_MIN_TOKENS_OVERRIDE.with(|c| c.set(None));
        WRITE_BEHIND_DELAY_OVERRIDE.with(|c| c.set(None));
    }

    #[test]
    fn persistent_fingerprint_v2_rejects_v1_and_tracks_arithmetic_mode() {
        let cfg = tiny_inkling2_config();
        let cfg_json = serde_json::to_vec(&cfg).unwrap();
        let dir = Path::new("/tmp/inkling2-fp-v2-test");
        let q_tag = query_precision_tag(Some(128));
        let block_size = 32u32;

        // Legacy v1 path KV/state fingerprints (commit 7e347dc0b format)
        let v1_kv = ColdCacheFingerprint::from_components([
            b"inkling2-path-kv:v1".as_slice(),
            dir.to_string_lossy().as_bytes(),
            cfg_json.as_slice(),
            b"int8",
            q_tag.as_bytes(),
            &block_size.to_le_bytes(),
        ]);
        let v1_state = ColdCacheFingerprint::from_components([
            b"inkling2-path-state:v1".as_slice(),
            v1_kv.as_bytes().as_slice(),
            b"geom",
        ]);

        // Current v2 fingerprints include forward arithmetic epoch + NAX/runtime tags.
        let arith_tag = forward_arithmetic_tag();
        assert!(arith_tag.starts_with(&format!("{INKLING2_FORWARD_ARITH_VERSION}:nax=")));
        let v2_kv = ColdCacheFingerprint::from_components([
            b"inkling2-path-kv:v2".as_slice(),
            dir.to_string_lossy().as_bytes(),
            cfg_json.as_slice(),
            b"int8",
            q_tag.as_bytes(),
            arith_tag.as_bytes(),
            &block_size.to_le_bytes(),
        ]);
        let v2_state = ColdCacheFingerprint::from_components([
            b"inkling2-path-state:v2".as_slice(),
            v2_kv.as_bytes().as_slice(),
            b"geom",
        ]);
        assert_ne!(
            v2_kv, v1_kv,
            "v2 KV fingerprint must reject legacy v1 prefixes"
        );
        assert_ne!(
            v2_state, v1_state,
            "v2 state fingerprint must reject legacy v1 prefixes"
        );

        // Changing arithmetic mode (e.g., nax=0 vs nax=1) must produce distinct KV & state keys.
        let alt_arith = if arith_tag.contains("nax=1") {
            arith_tag.replace("nax=1", "nax=0")
        } else {
            arith_tag.replace("nax=0", "nax=1")
        };
        let alt_kv = ColdCacheFingerprint::from_components([
            b"inkling2-path-kv:v2".as_slice(),
            dir.to_string_lossy().as_bytes(),
            cfg_json.as_slice(),
            b"int8",
            q_tag.as_bytes(),
            alt_arith.as_bytes(),
            &block_size.to_le_bytes(),
        ]);
        assert_ne!(
            v2_kv, alt_kv,
            "distinct NAX/arithmetic modes must not collide"
        );

        // Changing arithmetic epoch (v3 -> current v4 after fused Q5_K MoE) must reject v3 prefixes.
        let v3_arith = arith_tag.replacen(INKLING2_FORWARD_ARITH_VERSION, "v3", 1);
        let v3_kv = ColdCacheFingerprint::from_components([
            b"inkling2-path-kv:v2".as_slice(),
            dir.to_string_lossy().as_bytes(),
            cfg_json.as_slice(),
            b"int8",
            q_tag.as_bytes(),
            v3_arith.as_bytes(),
            &block_size.to_le_bytes(),
        ]);
        assert_ne!(
            v2_kv, v3_kv,
            "current arithmetic epoch must reject stale v3 SSD prefixes"
        );
    }
}
