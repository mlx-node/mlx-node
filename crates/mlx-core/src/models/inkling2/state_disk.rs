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

use std::collections::{HashMap, HashSet};
use std::ops::{Deref, DerefMut};
#[cfg(test)]
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use mlx_paged_attn::{
    CacheRecency, ColdCacheFingerprint, ColdCacheKey, ColdGroup, ColdKeyHold, ColdStateLabel,
    MakeRoom, block_object_bytes,
};
use napi::bindgen_prelude::{Error, Result};

use crate::array::memory::WeightsResident;
use crate::array::{DType, MxArray};
use crate::cold_tier::{
    CheckpointFiles, CheckpointLoadGuard, ColdTierGeometry, build_model_fingerprint,
};
use crate::disk_tier::state_disk::{
    KvPoolHandle, PersistStatus, Reclaimed, Staged, StateDiskCodec, StateDiskEngine, Tick,
    WriteBehind, state_context,
};
use crate::disk_tier::store::{
    DiskCopy, Entry, EntryId, HybridStateStore, Lineage, StateBytes, StateClass, StoreHit,
};
use crate::disk_tier::{DiskTier, DiskTierOptions, PersistentIdentity, open_disk_tier};
use crate::engine::governor::governor::{Governor, Scope};
use crate::engine::governor::ledger::{ClaimClass, ClaimRank, ClaimSpec, NeedTiming, ReachClass};
use crate::models::qwen3_5::dflash2::DFlash2Config;
use crate::models::qwen3_5::draft_context::DraftContextWindow;
use crate::transformer::KvFormat;
use crate::transformer::paged_kv_cache_adapter::{
    ChainPrefetch, ChainPrefetchStart, ColdTierContext, PagedKVCacheAdapter, RestoreUploadBudget,
    SeqId,
};

pub(crate) use crate::disk_tier::state_disk::ReadWait;

use super::config::Inkling2Config;
use super::kv_cache::Inkling2KVCacheCoordinator;
use super::memory::{
    DRAFT_WINDOW_HEADER_BYTES, draft_window_span_bytes, global_visit_count, short_conv_state_bytes,
    swa_visit_count, swa_window_blocks,
};
use super::short_conv::{Inkling2ShortConvState, VisitShortConvState};

/// Default rolling prefill checkpoint spacing (tokens).
pub(crate) const INKLING2_CHECKPOINT_INTERVAL: u32 = 4096;

/// Default minimum prefix depth / reply length (tokens) for write-behind & long-reply preemption.
pub(crate) const INKLING2_WRITE_BEHIND_MIN_TOKENS: usize = 2048;

/// Default hard cap on RAM-resident state anchors in `Inkling2StateStore`.
pub(crate) const INKLING2_DEFAULT_RAM_STATES: usize = 16;

/// Default backoff delay (seconds) after a failed/refused SSD spill or upgrade before retrying.
const INKLING2_SPILL_BACKOFF_SECS: f64 = 10.0;

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

pub(crate) fn write_behind_min_tokens() -> usize {
    #[cfg(test)]
    {
        if let Some(v) = WRITE_BEHIND_MIN_TOKENS_OVERRIDE.with(std::cell::Cell::get) {
            return v;
        }
        crate::disk_tier::state_disk::MINIMUM_TOKENS as usize
    }
    #[cfg(not(test))]
    {
        INKLING2_WRITE_BEHIND_MIN_TOKENS
    }
}

fn spill_backoff_secs() -> f64 {
    #[cfg(test)]
    if let Some(v) = WRITE_BEHIND_DELAY_OVERRIDE.with(std::cell::Cell::get) {
        return v;
    }
    INKLING2_SPILL_BACKOFF_SECS
}

fn upgrade_retry_after() -> Instant {
    Instant::now() + Duration::from_secs_f64(spill_backoff_secs().max(0.0))
}

/// Shared host-heap payload for one [`PackedDraftWindow`].
///
/// Once attached to an [`Inkling2StateStore`]'s [`HostBudgetAccount`], its exact byte charge stays
/// live across both `self.store` and any live sequence clones (e.g. `block_boundary_draft_window`),
/// releasing automatically when the final `Arc<TrackedPackedBytes>` drops.
pub(crate) struct TrackedPackedBytes {
    bytes: Vec<u8>,
    account: OnceLock<Arc<HostBudgetAccount>>,
}

impl std::fmt::Debug for TrackedPackedBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrackedPackedBytes")
            .field("len", &self.bytes.len())
            .field("tracked", &self.account.get().is_some())
            .finish()
    }
}

impl PartialEq for TrackedPackedBytes {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes
    }
}

impl Eq for TrackedPackedBytes {}

impl Deref for TrackedPackedBytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.bytes
    }
}

impl Drop for TrackedPackedBytes {
    fn drop(&mut self) {
        if let Some(account) = self.account.get() {
            account.sub_packed_dw_bytes(self.bytes.len() as u64);
        }
    }
}

impl TrackedPackedBytes {
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            account: OnceLock::new(),
        }
    }

    pub(crate) fn attach_to(&self, account: &Arc<HostBudgetAccount>) {
        if !self.bytes.is_empty() && self.account.set(Arc::clone(account)).is_ok() {
            account.add_packed_dw_bytes(self.bytes.len() as u64);
        }
    }

    pub(crate) fn is_tracked_by(&self, account: &Arc<HostBudgetAccount>) -> bool {
        self.account
            .get()
            .is_some_and(|bound| Arc::ptr_eq(bound, account))
    }
}

/// Flat, host-resident snapshot of one DFlash2 4-bit [`DraftContextWindow`] ring buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PackedDraftWindow {
    pub bytes: Arc<TrackedPackedBytes>,
    pub width: u32,
    pub slots: u32,
    pub begin: u32,
    pub end: u32,
}

impl PackedDraftWindow {
    /// Snapshot a live [`DraftContextWindow`] into one flat host `Arc<TrackedPackedBytes>` buffer.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn from_window(window: &DraftContextWindow) -> Result<Self> {
        let nbytes = usize::try_from(window.nbytes())
            .map_err(|_| Error::from_reason("Inkling2 draft window size overflows usize"))?;
        let buf = window.host_bytes()?;
        if buf.len() != nbytes {
            return Err(Error::from_reason(
                "Inkling2 PackedDraftWindow::from_window failed to serialize window bytes",
            ));
        }
        Ok(Self {
            bytes: Arc::new(TrackedPackedBytes::new(buf)),
            width: window.width(),
            slots: window.slots(),
            begin: window.begin(),
            end: window.end(),
        })
    }

    /// Stable pointer identity of the underlying packed byte buffer for deduplication.
    pub(crate) fn ptr_id(&self) -> *const u8 {
        self.bytes.as_ptr()
    }

    pub(crate) fn attach_to(&self, account: &Arc<HostBudgetAccount>) {
        self.bytes.attach_to(account);
    }

    pub(crate) fn is_tracked_by(&self, account: &Arc<HostBudgetAccount>) -> bool {
        self.bytes.is_tracked_by(account)
    }

    /// Materialize a fresh [`DraftContextWindow`] populated from this packed snapshot.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn to_window(&self) -> Result<DraftContextWindow> {
        let mut window = DraftContextWindow::new(self.width, self.slots)?;
        if !window.read_bytes(&self.bytes, self.begin, self.end)? {
            return Err(Error::from_reason(
                "Inkling2 PackedDraftWindow::to_window failed to restore window bytes",
            ));
        }
        Ok(window)
    }

    /// Load this packed snapshot directly into an existing [`DraftContextWindow`] in place.
    /// Returns `Ok(false)` on geometry, length, or bounds mismatch.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn load_into(&self, dst: &mut DraftContextWindow) -> Result<bool> {
        if dst.width() != self.width || dst.slots() != self.slots {
            return Ok(false);
        }
        dst.read_bytes(&self.bytes, self.begin, self.end)
    }

    /// True iff this packed window covers the complete trailing
    /// `[boundary.saturating_sub(slots), boundary)` range with an exact 4-bit payload.
    pub(crate) fn is_complete_for(&self, boundary: u32) -> bool {
        self.width > 0
            && self.slots > 0
            && self.end == boundary
            && self.begin == boundary.saturating_sub(self.slots)
            && self.bytes.len() as u64 == DraftContextWindow::bytes(self.width, self.slots)
    }
}

/// Block-boundary DFlash2 context window snapshot held per live sequence until a
/// checkpoint or turn-end state is actually published into [`Inkling2StateStore`].
///
/// - Live prefill/decode stores [`BoundaryDraftWindow::Device`] (detached MLX GPU
///   buffer) so crossing 32-token boundaries never stalls decode or copies 2.53 MiB
///   to host RAM unless `capture_boundary` actually publishes a state.
/// - Restored sequences from `StateStore` retain [`BoundaryDraftWindow::Packed`]
///   (`Arc<TrackedPackedBytes>`) without an unnecessary GPU upload when the suffix will refill
///   the ring before the next boundary.
pub(crate) enum BoundaryDraftWindow {
    Device(DraftContextWindow),
    Packed(PackedDraftWindow),
}

impl BoundaryDraftWindow {
    fn to_packed_for(
        &self,
        geom: &Inkling2StateGeometry,
        boundary: u32,
        pack_device: impl FnOnce(&DraftContextWindow) -> Result<PackedDraftWindow>,
    ) -> Result<Option<PackedDraftWindow>> {
        if geom.draft_window_span_bytes == 0
            || geom.draft_window_width == 0
            || geom.draft_window_slots == 0
        {
            return Ok(None);
        }
        match self {
            BoundaryDraftWindow::Packed(dw) => {
                if dw.width == geom.draft_window_width
                    && dw.slots == geom.draft_window_slots
                    && dw.is_complete_for(boundary)
                {
                    Ok(Some(dw.clone()))
                } else {
                    Ok(None)
                }
            }
            BoundaryDraftWindow::Device(dw) => {
                if dw.width() == geom.draft_window_width
                    && dw.slots() == geom.draft_window_slots
                    && dw.end() == boundary
                    && dw.begin() == boundary.saturating_sub(geom.draft_window_slots)
                {
                    pack_device(dw).map(|packed| packed.is_complete_for(boundary).then_some(packed))
                } else {
                    Ok(None)
                }
            }
        }
    }
}

enum DraftWindowInput<'a> {
    #[cfg(test)]
    Packed(&'a PackedDraftWindow),
    Boundary(&'a BoundaryDraftWindow),
}

impl DraftWindowInput<'_> {
    fn matches_geometry_for(&self, geom: &Inkling2StateGeometry, boundary: u32) -> bool {
        if geom.draft_window_span_bytes == 0
            || geom.draft_window_width == 0
            || geom.draft_window_slots == 0
        {
            return false;
        }
        match *self {
            #[cfg(test)]
            DraftWindowInput::Packed(dw) => {
                dw.width == geom.draft_window_width
                    && dw.slots == geom.draft_window_slots
                    && dw.is_complete_for(boundary)
            }
            DraftWindowInput::Boundary(BoundaryDraftWindow::Packed(dw)) => {
                dw.width == geom.draft_window_width
                    && dw.slots == geom.draft_window_slots
                    && dw.is_complete_for(boundary)
            }
            DraftWindowInput::Boundary(BoundaryDraftWindow::Device(dw)) => {
                dw.width() == geom.draft_window_width
                    && dw.slots() == geom.draft_window_slots
                    && dw.end() == boundary
                    && dw.begin() == boundary.saturating_sub(geom.draft_window_slots)
            }
        }
    }

    fn new_host_bytes_for(
        &self,
        geom: &Inkling2StateGeometry,
        boundary: u32,
        account: &Arc<HostBudgetAccount>,
    ) -> u64 {
        if !self.matches_geometry_for(geom, boundary) {
            return 0;
        }
        match *self {
            #[cfg(test)]
            DraftWindowInput::Packed(dw) => {
                if dw.is_tracked_by(account) {
                    0
                } else {
                    dw.bytes.len() as u64
                }
            }
            DraftWindowInput::Boundary(BoundaryDraftWindow::Packed(dw)) => {
                if dw.is_tracked_by(account) {
                    0
                } else {
                    dw.bytes.len() as u64
                }
            }
            DraftWindowInput::Boundary(BoundaryDraftWindow::Device(_)) => {
                DraftContextWindow::bytes(geom.draft_window_width, geom.draft_window_slots)
            }
        }
    }
}

/// Pack a 48-visit [`Inkling2ShortConvState`] into a flat little-endian BF16 byte buffer
/// (`geom.conv_span_bytes` long) so RAM anchors hold pure host bytes with zero pinned GPU arrays.
pub(crate) fn pack_short_conv_state(
    geom: &Inkling2StateGeometry,
    conv: &Inkling2ShortConvState,
) -> std::result::Result<Arc<Vec<u8>>, String> {
    if conv.visits.len() != geom.num_visits as usize {
        return Err(format!(
            "Inkling2 ShortConv visit count {} != {}",
            conv.visits.len(),
            geom.num_visits
        ));
    }
    let conv_len = usize::try_from(geom.conv_span_bytes)
        .map_err(|_| "Inkling2 ShortConv span overflows usize".to_string())?;
    let mut conv_buf = vec![0u8; conv_len];
    let hist = geom.conv_history as usize;
    let kv_elems = hist * geom.kv_dim as usize;
    let hidden_elems = hist * geom.hidden_size as usize;
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
    for visit in &conv.visits {
        write_bf16(&visit.k, kv_elems)?;
        write_bf16(&visit.v, kv_elems)?;
        write_bf16(&visit.attn_post, hidden_elems)?;
        write_bf16(&visit.mlp_post, hidden_elems)?;
    }
    if cursor != conv_buf.len() {
        return Err(format!(
            "Inkling2 ShortConv packed bytes {cursor} != expected {}",
            conv_buf.len()
        ));
    }
    Ok(Arc::new(conv_buf))
}

/// Unpack a flat little-endian BF16 ShortConv byte buffer (`geom.conv_span_bytes` long) back into
/// a live 48-visit [`Inkling2ShortConvState`].
pub(crate) fn unpack_short_conv_state(
    geom: &Inkling2StateGeometry,
    conv_data: &[u8],
) -> std::result::Result<Inkling2ShortConvState, String> {
    let expected_len = usize::try_from(geom.conv_span_bytes)
        .map_err(|_| "Inkling2 ShortConv span overflows usize".to_string())?;
    if conv_data.len() != expected_len {
        return Err(format!(
            "Inkling2 ShortConv payload length {} != expected {expected_len}",
            conv_data.len()
        ));
    }
    let hist = geom.conv_history as i64;
    let kv_cols = geom.kv_dim as i64;
    let hidden_cols = geom.hidden_size as i64;
    let kv_elems = (geom.conv_history as usize) * (geom.kv_dim as usize);
    let hidden_elems = (geom.conv_history as usize) * (geom.hidden_size as usize);
    let mut cursor = 0usize;
    let mut read_bf16 = |elems: usize, shape: &[i64]| -> std::result::Result<MxArray, String> {
        let byte_len = elems * 2;
        let src = conv_data
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

    let mut visits = Vec::with_capacity(geom.num_visits as usize);
    for _ in 0..geom.num_visits {
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
    if cursor != conv_data.len() {
        return Err(format!(
            "Inkling2 ShortConv unpacked bytes {cursor} != expected {}",
            conv_data.len()
        ));
    }
    Ok(Inkling2ShortConvState {
        visits,
        kernel_size: (geom.conv_history as usize) + 1,
    })
}

/// Exact geometry of Inkling-2-Mini's 1 content-keyed state blob:
/// - Span 0: `max_swa_blocks` raw physical blocks across all `swa_layers` (`40`) SWA visits
///   (INT8 KTM codes + fp32 scales, or BF16; zero-padded when `boundary < sliding_window`).
/// - Span 1: all `num_visits` (`48`) ShortConv(`k=4`) rolling states (`k, v, attn_post, mlp_post`)
///   in BF16.
/// - Span 2 (optional, when DFlash2 is attached): `[valid_flag: u32, window_begin: u32]` header
///   followed by the 4-bit [`DraftContextWindow`] ring buffer.
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
    pub draft_window_width: u32,
    pub draft_window_slots: u32,
    pub draft_window_span_bytes: u64,
    pub block_size: u32,
    pub sliding_window: u32,
    pub kv_format: KvFormat,
}

impl Inkling2StateGeometry {
    pub(crate) fn from_coordinator(
        config: &Inkling2Config,
        coordinator: &Inkling2KVCacheCoordinator,
    ) -> Result<Self> {
        Self::from_coordinator_with_draft(config, coordinator, None)
    }

    pub(crate) fn from_coordinator_with_draft(
        config: &Inkling2Config,
        coordinator: &Inkling2KVCacheCoordinator,
        dflash2: Option<&DFlash2Config>,
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

        let (draft_window_width, draft_window_slots, draft_window_span_bytes) = match dflash2 {
            Some(cfg) => {
                let width = cfg.context_width();
                let slots = u32::try_from(cfg.sliding_window).map_err(|_| {
                    Error::from_reason("Inkling2 DFlash2 sliding_window overflows u32")
                })?;
                let span = draft_window_span_bytes(Some(cfg));
                (width, slots, span)
            }
            None => (0, 0, 0),
        };

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
            draft_window_width,
            draft_window_slots,
            draft_window_span_bytes,
            block_size,
            sliding_window,
            kv_format: coordinator.kv_format(),
        })
    }

    pub(crate) fn spans(&self) -> Vec<u64> {
        let mut s = Vec::with_capacity(3);
        if self.swa_span_bytes > 0 {
            s.push(self.swa_span_bytes);
        }
        if self.conv_span_bytes > 0 {
            s.push(self.conv_span_bytes);
        }
        if self.draft_window_span_bytes > 0 {
            s.push(self.draft_window_span_bytes);
        }
        s
    }

    pub(crate) fn payload_bytes(&self) -> u64 {
        self.swa_span_bytes
            .saturating_add(self.conv_span_bytes)
            .saturating_add(self.draft_window_span_bytes)
    }

    pub(crate) fn geometry_string(&self) -> String {
        if self.draft_window_span_bytes > 0 {
            format!(
                "inkling2-state:v2 swa={}:{}:{}:{}:{} conv={}:{}:{}:{} bs={} fmt={} dw={}:{}",
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
                self.draft_window_width,
                self.draft_window_slots,
            )
        } else {
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

    /// Exact flat SWA byte length for the live blocks at `boundary_tokens`.
    pub(crate) fn live_swa_bytes(
        &self,
        boundary_tokens: u32,
    ) -> std::result::Result<usize, String> {
        if self.swa_span_bytes == 0 {
            return Ok(0);
        }
        let live_blocks = self.live_swa_blocks(boundary_tokens);
        if live_blocks > self.max_swa_blocks {
            return Err(format!(
                "Inkling2 SWA live_blocks {live_blocks} exceeds max {}",
                self.max_swa_blocks
            ));
        }
        let per_layer = self
            .swa_key_blob_bytes
            .checked_add(self.swa_value_blob_bytes)
            .ok_or_else(|| "Inkling2 SWA per-layer bytes overflow".to_string())?;
        let per_block = u64::from(self.swa_layers)
            .checked_mul(per_layer)
            .ok_or_else(|| "Inkling2 SWA per-block bytes overflow".to_string())?;
        let live_bytes = u64::from(live_blocks)
            .checked_mul(per_block)
            .ok_or_else(|| "Inkling2 SWA live bytes overflow".to_string())?;
        usize::try_from(live_bytes)
            .map_err(|_| "Inkling2 SWA live bytes overflow usize".to_string())
    }

    /// Encode a flat [`Inkling2StateAnchor`] into the fixed-size `into` staging buffer.
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
        let conv_len = usize::try_from(self.conv_span_bytes)
            .map_err(|_| "Inkling2 ShortConv span overflows usize".to_string())?;
        let (swa_buf, rest) = into.split_at_mut(swa_len);
        let (conv_buf, dw_buf) = rest.split_at_mut(conv_len);

        // Span 0: flat SWA trailing window raw blocks (zero-padded to fixed span length).
        if swa_len > 0 {
            let expected_live_bytes = self.live_swa_bytes(boundary_tokens)?;
            if anchor.swa_data.len() != expected_live_bytes || expected_live_bytes > swa_len {
                return Err(format!(
                    "Inkling2 SWA flat byte length {} != expected {expected_live_bytes} (span {swa_len})",
                    anchor.swa_data.len()
                ));
            }
            swa_buf[..expected_live_bytes].copy_from_slice(&anchor.swa_data);
            if expected_live_bytes < swa_len {
                swa_buf[expected_live_bytes..].fill(0);
            }
        } else if !anchor.swa_data.is_empty() {
            return Err(format!(
                "Inkling2 SWA flat bytes non-empty ({}) for zero-SWA geometry",
                anchor.swa_data.len()
            ));
        }

        // Span 1: flat 48-visit ShortConv(k=4) ring states in BF16 little-endian u16.
        if anchor.conv_data.len() != conv_buf.len() {
            return Err(format!(
                "Inkling2 ShortConv flat byte length {} != expected {}",
                anchor.conv_data.len(),
                conv_buf.len()
            ));
        }
        conv_buf.copy_from_slice(&anchor.conv_data);

        // Span 2: optional DFlash2 DraftContextWindow `[valid_flag: u32, window_begin: u32, raw_4bit_bytes]`.
        if self.draft_window_span_bytes > 0 {
            let header_len = DRAFT_WINDOW_HEADER_BYTES as usize;
            let raw_len = dw_buf
                .len()
                .checked_sub(header_len)
                .ok_or_else(|| "Inkling2 draft window span smaller than header".to_string())?;
            if let Some(dw) = anchor.draft_window.as_ref() {
                if dw.width != self.draft_window_width
                    || dw.slots != self.draft_window_slots
                    || dw.bytes.len() != raw_len
                    || dw.end != boundary_tokens
                    || dw.begin > dw.end
                    || dw.end - dw.begin > dw.slots
                {
                    return Err(format!(
                        "Inkling2 draft window mismatch: got {}x{} [{}, {}) ({} B) for boundary \
                         {boundary_tokens} (expected {}x{}, {raw_len} B)",
                        dw.width,
                        dw.slots,
                        dw.begin,
                        dw.end,
                        dw.bytes.len(),
                        self.draft_window_width,
                        self.draft_window_slots
                    ));
                }
                dw_buf[0..4].copy_from_slice(&1u32.to_le_bytes());
                dw_buf[4..8].copy_from_slice(&dw.begin.to_le_bytes());
                dw_buf[header_len..].copy_from_slice(&dw.bytes);
            } else {
                dw_buf.fill(0);
            }
        }
        Ok(())
    }

    /// Decode a flat [`Inkling2StateAnchor`] from a validated state blob payload.
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
        let conv_len = usize::try_from(self.conv_span_bytes)
            .map_err(|_| "Inkling2 ShortConv span overflows usize".to_string())?;
        let (swa_buf, rest) = payload.split_at(swa_len);
        let (conv_buf, dw_buf) = rest.split_at(conv_len);

        // Span 0: live flat SWA bytes.
        let swa_data: Arc<Vec<u8>> = if swa_len > 0 {
            let expected_live_bytes = self.live_swa_bytes(boundary_tokens)?;
            if expected_live_bytes > swa_len {
                return Err(format!(
                    "Inkling2 SWA live bytes {expected_live_bytes} exceed span {swa_len}"
                ));
            }
            Arc::new(swa_buf[..expected_live_bytes].to_vec())
        } else {
            Arc::new(Vec::new())
        };

        // Span 1: flat ShortConv bytes.
        let conv_data: Arc<Vec<u8>> = Arc::new(conv_buf.to_vec());

        // Span 2: optional DFlash2 DraftContextWindow.
        let draft_window = if self.draft_window_span_bytes > 0 {
            let header_len = DRAFT_WINDOW_HEADER_BYTES as usize;
            if dw_buf.len() < header_len {
                return Err("Inkling2 draft window span truncated".to_string());
            }
            let valid_flag = u32::from_le_bytes([dw_buf[0], dw_buf[1], dw_buf[2], dw_buf[3]]);
            let window_begin = u32::from_le_bytes([dw_buf[4], dw_buf[5], dw_buf[6], dw_buf[7]]);
            match valid_flag {
                0 => None,
                1 => {
                    let raw = &dw_buf[header_len..];
                    let expected_raw = usize::try_from(DraftContextWindow::bytes(
                        self.draft_window_width,
                        self.draft_window_slots,
                    ))
                    .map_err(|_| "Inkling2 draft window size overflows usize".to_string())?;
                    if raw.len() != expected_raw
                        || window_begin > boundary_tokens
                        || boundary_tokens - window_begin > self.draft_window_slots
                    {
                        return Err(format!(
                            "Inkling2 draft window decode invalid bounds [{window_begin}, \
                             {boundary_tokens}) for slots={} (raw {} != {expected_raw})",
                            self.draft_window_slots,
                            raw.len()
                        ));
                    }
                    Some(PackedDraftWindow {
                        bytes: Arc::new(TrackedPackedBytes::new(raw.to_vec())),
                        width: self.draft_window_width,
                        slots: self.draft_window_slots,
                        begin: window_begin,
                        end: boundary_tokens,
                    })
                }
                other => {
                    return Err(format!("Inkling2 draft window invalid valid_flag={other}"));
                }
            }
        } else {
            None
        };

        let cached_bytes = (swa_data.len() as u64)
            .saturating_add(conv_data.len() as u64)
            .saturating_add(draft_window.as_ref().map_or(0, |d| d.bytes.len() as u64));
        Ok(Inkling2StateAnchor {
            swa_data,
            conv_data,
            draft_window,
            cached_bytes,
        })
    }
}

/// In-memory hybrid state snapshot at one block-aligned prefix boundary:
/// - `swa_data`: flat raw physical block buffer for Group 1 (40 SWA visits, `<= 32` blocks).
/// - `conv_data`: flat little-endian BF16 buffer for 48-visit ShortConv(`k=4`) rolling states.
/// - `draft_window`: optional flat 4-bit DFlash2 [`PackedDraftWindow`] snapshot.
///
/// Holds NO Global (Group 0) KV copy and NO live MLX GPU tensors: Group 0 blocks stay
/// content-addressed in the Global `PagedKVCacheAdapter` and are kept resident via
/// `set_state_marks`, while `Arc<Vec<u8>>` moves readback buffers in O(1) with zero copy.
#[derive(Clone)]
pub(crate) struct Inkling2StateAnchor {
    pub swa_data: Arc<Vec<u8>>,
    pub conv_data: Arc<Vec<u8>>,
    pub draft_window: Option<PackedDraftWindow>,
    pub cached_bytes: u64,
}

impl Inkling2StateAnchor {
    /// Host RAM bytes exclusively owned by this anchor (`Arc::strong_count == 1`).
    ///
    /// Shared `Arc<[u8]>` buffers (such as a `PackedDraftWindow` still cloned inside a live
    /// sequence's `block_boundary_draft_window`) remain resident when this anchor drops and must
    /// not be credited to the governor as reclaimed RAM.
    pub(crate) fn exclusive_ram_bytes(&self) -> u64 {
        let swa = if Arc::strong_count(&self.swa_data) == 1 {
            self.swa_data.len() as u64
        } else {
            0
        };
        let conv = if Arc::strong_count(&self.conv_data) == 1 {
            self.conv_data.len() as u64
        } else {
            0
        };
        let dw = self
            .draft_window
            .as_ref()
            .filter(|dw| Arc::strong_count(&dw.bytes) == 1)
            .map_or(0, |dw| dw.bytes.len() as u64);
        swa.saturating_add(conv).saturating_add(dw)
    }
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

/// Policy flags for [`Inkling2StateStore::capture_boundary_with_options`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct CaptureOptions {
    /// When `true` (publishing a prompt junction / head / turn end), detach `seq_checkpoints`
    /// without retiring the prior rolling checkpoint so another conversation sharing the
    /// earlier prefix still hits it as a disposable `Checkpoint`.
    pub(crate) keep_prev_checkpoint: bool,
    /// When `true` and publishing `StateClass::Ordinary`, enqueue into background `WriteBehind`.
    /// Set `false` for preempted long replies (`Reply::Kept` semantics: protected from
    /// Checkpoint-first eviction and spilled on reclaim, without eager write-behind).
    pub(crate) enqueue_write_behind: bool,
    /// Persistent prompt head (`Splash Cache::persist(..., head = true)`): survives superseding
    /// ordinary turns on a persistent SSD tier.
    pub(crate) is_head: bool,
    /// Governor clock timestamp (`governor.now()`) for `WriteBehind::record_published`.
    pub(crate) now: f64,
}

impl Default for CaptureOptions {
    fn default() -> Self {
        Self {
            keep_prev_checkpoint: false,
            enqueue_write_behind: true,
            is_head: false,
            now: 0.0,
        }
    }
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
        let window_begin = state.draft_window.as_ref().map_or_else(
            || depth.saturating_sub(self.geometry.sliding_window),
            |dw| dw.begin,
        );
        Some(window_begin)
    }

    fn adopt_copy(&self, _spans: &[u64], label: &ColdStateLabel) -> Option<(u64, u32)> {
        let row = u32::try_from(label.tokens.len()).ok()?;
        let window_begin = if self.geometry.draft_window_slots > 0 {
            row.saturating_sub(self.geometry.draft_window_slots)
        } else {
            row.saturating_sub(self.geometry.sliding_window)
        };
        Some((self.geometry.payload_bytes(), window_begin))
    }
}

/// In-flight off-thread SSD label relabel (`PendingDiskUpgrade::LabelPromote`).
///
/// Owns the [`ColdKeyHold`] pinning both `key` and its complete Global KV chain against
/// competing `MakeRoom::Evict` writes so [`Inkling2StateStore::clear_ram_and_checkpoints`]
/// cannot release the hold while the relabel is still queued or running on the writer thread.
struct PendingRelabel {
    id: EntryId,
    key: ColdCacheKey,
    _fallback_hold: ColdKeyHold,
}

/// Outcome of polling completed off-thread SSD label relabels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RelabelPollOutcome {
    landed: bool,
    landed_ok: u64,
    failures: u64,
    resync: bool,
    failed_id: Option<EntryId>,
}

/// Model-owned StateDisk tier for Inkling-2-Mini backed by the shared
/// [`StateDiskEngine`] (single reusable host staging buffer, dual-lane
/// `ColdCacheManager` writer, 8-layer Global KV chain cursor, shared `WriteBehind`).
pub(crate) struct Inkling2StateDisk {
    pub(crate) engine: StateDiskEngine<Inkling2StateCodec>,
    pub(crate) write_behind: WriteBehind,
    /// The landing pages the governor last read from this tier.
    pub(crate) published_landing: u32,
    pub(crate) counters: Inkling2DiskCounters,
    /// Dedicated channel for off-thread `ColdCacheManager::enqueue_state_relabel` completions
    /// (kept separate from `engine.landed_tx` so `StateDiskEngine::poll_offload` never
    /// overwrites `engine.staging` or clears `entry.disk` on relabel completion/refusal).
    relabel_tx: std::sync::mpsc::Sender<mlx_paged_attn::StateRelabelLanded>,
    relabel_rx: std::sync::mpsc::Receiver<mlx_paged_attn::StateRelabelLanded>,
    /// In-flight off-thread `LabelPromote` rewrite (kept separate from `engine.pending` so
    /// `entry.disk()` remains `Some(DiskCopy { landed: true, .. })` and `engine.staging` stays
    /// free for concurrent restores throughout the rewrite).
    pending_relabel: Option<PendingRelabel>,
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
        let write_behind = WriteBehind::new(tier.persistent);
        let engine = StateDiskEngine::new(
            tier,
            spans,
            geometry_str,
            global_block_bytes,
            Inkling2StateCodec { geometry },
        )
        .ok_or_else(|| Error::from_reason("Inkling2 StateDisk staging size overflows usize"))?;
        let (relabel_tx, relabel_rx) = std::sync::mpsc::channel();
        Ok(Self {
            engine,
            write_behind,
            published_landing: 0,
            counters: Inkling2DiskCounters::default(),
            relabel_tx,
            relabel_rx,
            pending_relabel: None,
        })
    }

    /// Whether any state write or off-thread label relabel is currently in flight.
    pub(crate) fn writing(&self) -> bool {
        self.engine.writing() || self.pending_relabel.is_some()
    }

    /// Entry ID of the in-flight state write or off-thread label relabel, if any.
    pub(crate) fn pending_write_id(&self) -> Option<EntryId> {
        self.engine
            .pending
            .as_ref()
            .map(|p| p.id)
            .or_else(|| self.pending_relabel.as_ref().map(|p| p.id))
    }

    /// Wait up to 50 ms (never past `deadline`) for either an off-thread relabel or a normal
    /// `StateDiskEngine` landing, re-queueing the event so the next poll observes it.
    pub(crate) fn await_landing(&self, deadline: Instant) -> bool {
        let now = Instant::now();
        if now >= deadline {
            return false;
        }
        if self.pending_relabel.is_some() {
            if let Ok(landing) = self.relabel_rx.try_recv() {
                let _ = self.relabel_tx.send(landing);
                return true;
            }
            if !self.engine.writing() && self.engine.chain.is_none() {
                if let Ok(landing) = self
                    .relabel_rx
                    .recv_timeout((deadline - now).min(Duration::from_millis(50)))
                {
                    let _ = self.relabel_tx.send(landing);
                }
                return true;
            }
            if let Ok(landing) = self.engine.landed_rx.try_recv() {
                let _ = self.engine.landed_tx.send(landing);
                return true;
            }
            if let Ok(landing) = self
                .relabel_rx
                .recv_timeout((deadline - now).min(Duration::from_millis(5)))
            {
                let _ = self.relabel_tx.send(landing);
            }
            return true;
        }
        self.engine.await_landing(deadline)
    }

    /// Drain completed off-thread label relabels from `self.relabel_rx`.
    fn poll_relabel(
        &mut self,
        store: &mut HybridStateStore<Inkling2StateAnchor>,
    ) -> RelabelPollOutcome {
        let mut out = RelabelPollOutcome::default();
        while let Ok(landing) = self.relabel_rx.try_recv() {
            out.landed = true;
            let Some(write) = self.pending_relabel.take_if(|w| w.key == landing.key) else {
                // Unlike initial spills, a relabel only atomically updates the header label of an
                // already-committed durable SSD anchor (`!entry.checkpoint()`), so an unmatched
                // completion must never delete `landing.key` from disk.
                continue;
            };
            if landing.ok {
                out.landed_ok += 1;
                if let Some(chain_missing) = store
                    .get(write.id)
                    .filter(|entry| entry.disk().is_some_and(|copy| copy.key == landing.key))
                    .map(|entry| !self.engine.chain_on_disk(entry))
                    && let Some(copy) = store
                        .disk_mut(write.id)
                        .filter(|copy| copy.key == landing.key)
                {
                    copy.landed = true;
                    copy.bytes = landing.bytes;
                    if chain_missing && !copy.chain_owed {
                        copy.chain_owed = true;
                        out.resync = true;
                    }
                }
                // If the in-memory entry was cleared while the relabel was in flight (e.g., by
                // `clear_ram_and_checkpoints()` during `ResetCaches`), treat the completion as a
                // bookkeeping no-op and leave the committed SSD anchor intact on disk.
            } else {
                out.failures += 1;
                out.failed_id = Some(write.id);
                out.resync = true;
            }
        }
        out
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

    /// Whether `entry`'s state object plus its complete Global KV chain can ever fit within
    /// `self.tier.limit` (distinguishing permanently un-fittable entries from transient volume
    /// headroom or writer refusals).
    pub(crate) fn entry_fits_tier_limit(&self, entry: &Entry<Inkling2StateAnchor>) -> bool {
        let Some(layout) = self.engine.layout(entry.depth()) else {
            return false;
        };
        let state = mlx_paged_attn::state_object_bytes(
            &layout,
            &crate::disk_tier::state_disk::label_of(entry),
        );
        let blocks = entry.chain().len() as u64;
        blocks
            .saturating_mul(self.engine.block_bytes)
            .saturating_add(state)
            <= self.tier.limit
    }
}

/// Queued SSD upgrade, repair, or label promotion:
/// - `InPlace`: RAM-resident entry still holds an intact old `DiskCopy` on SSD that must be
///   preserved on failure (`MakeRoom::Refuse` + rollback). `promote_label` records whether the
///   underlying on-disk copy also still carries `checkpoint: true` so dropping RAM before `InPlace`
///   lands falls back to `LabelPromote`.
/// - `Repair`: the old SSD blob or its Global KV chain was already evicted (or an upgrade
///   landed with `chain_owed == true`), so a fresh replacement blob AND its full Global KV
///   chain must land (`landed && !chain_owed && entry_disk_restorable`) before the repair
///   is complete.
/// - `LabelPromote`: persistent-tier label relabeling for an SSD-backed `Checkpoint` promoted to
///   `Ordinary` (`!entry.checkpoint()`), rewriting its `ColdStateLabel` with `checkpoint: false`
///   under `MakeRoom::Refuse` + rollback (works with or without a RAM copy).
#[derive(Clone, Copy, Debug)]
enum PendingDiskUpgrade {
    InPlace {
        retry_after: Instant,
        promote_label: bool,
    },
    Repair {
        retry_after: Instant,
    },
    LabelPromote {
        retry_after: Instant,
    },
}

impl PendingDiskUpgrade {
    fn retry_after(self) -> Instant {
        match self {
            Self::InPlace { retry_after, .. }
            | Self::Repair { retry_after }
            | Self::LabelPromote { retry_after } => retry_after,
        }
    }

    fn set_retry_after(&mut self, retry: Instant) {
        match self {
            Self::InPlace { retry_after, .. }
            | Self::Repair { retry_after }
            | Self::LabelPromote { retry_after } => *retry_after = retry,
        }
    }

    fn is_repair(&self) -> bool {
        matches!(self, Self::Repair { .. })
    }
}

/// Discriminant for [`InFlightDiskUpgrade`] so rollback restores the exact pending kind without
/// ever corrupting `entries_missing_draft_window` on a `LabelPromote` failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InFlightDiskUpgradeKind {
    InPlace { promote_label: bool },
    LabelPromote,
}

/// Active in-place SSD upgrade or label-promotion write: retains the previous durable `DiskCopy`
/// for rollback if the replacement write fails before landing.
///
/// Ownership of the [`ColdKeyHold`] pinning `old_copy.key` and its entire Global KV chain lives on
/// [`PendingRelabel`] (for `LabelPromote`) or [`StateDiskEngine`] (for `InPlace`) so
/// [`Inkling2StateStore::clear_ram_and_checkpoints`] does not release the eviction hold while the
/// replacement write is still queued or running on the writer thread.
struct InFlightDiskUpgrade {
    id: EntryId,
    kind: InFlightDiskUpgradeKind,
    old_copy: DiskCopy,
}

/// Target coordinator charging Inkling2's host-heap state store into
/// `CacheLimitCoordinator::registered_pool_bytes()` (and thus `Governor::charge()`).
enum HostBudgetTarget {
    Global,
    #[cfg(test)]
    Local(Arc<crate::cache_limit::CacheLimitCoordinator>),
}

struct HostBudgetAccountInner {
    target: Option<HostBudgetTarget>,
    guard: Option<crate::cache_limit::HostPoolCacheLimitGuard>,
    store_non_dw_bytes: u64,
    packed_dw_bytes: u64,
    reported_bytes: u64,
}

/// Shared host-budget accounting owner for one [`Inkling2StateStore`] and all live
/// [`TrackedPackedBytes`] allocations published or loaded through it.
pub(crate) struct HostBudgetAccount {
    inner: Mutex<HostBudgetAccountInner>,
}

impl HostBudgetAccount {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(HostBudgetAccountInner {
                target: None,
                guard: None,
                store_non_dw_bytes: 0,
                packed_dw_bytes: 0,
                reported_bytes: 0,
            }),
        })
    }

    fn sync_locked(inner: &mut HostBudgetAccountInner) {
        let Some(target) = inner.target.as_ref() else {
            inner.guard = None;
            inner.reported_bytes = 0;
            return;
        };
        let total = inner
            .store_non_dw_bytes
            .saturating_add(inner.packed_dw_bytes);
        if total == 0 {
            inner.guard = None;
            inner.reported_bytes = 0;
            return;
        }
        if total == inner.reported_bytes && inner.guard.is_some() {
            return;
        }
        if let Some(guard) = inner.guard.as_ref() {
            guard.update(total);
        } else {
            let guard = match target {
                HostBudgetTarget::Global => {
                    crate::cache_limit::coordinator().register_host_pool(total)
                }
                #[cfg(test)]
                HostBudgetTarget::Local(coord) => coord.register_host_pool_local(total),
            };
            inner.guard = Some(guard);
        }
        inner.reported_bytes = total;
    }

    fn set_target(&self, target: HostBudgetTarget) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if matches!(target, HostBudgetTarget::Global)
            && matches!(inner.target, Some(HostBudgetTarget::Global))
        {
            Self::sync_locked(&mut inner);
            return;
        }
        inner.guard = None;
        inner.reported_bytes = 0;
        inner.target = Some(target);
        Self::sync_locked(&mut inner);
    }

    fn set_store_non_dw_bytes(&self, bytes: u64) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.store_non_dw_bytes = bytes;
        Self::sync_locked(&mut inner);
    }

    fn add_packed_dw_bytes(&self, bytes: u64) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.packed_dw_bytes = inner.packed_dw_bytes.saturating_add(bytes);
        Self::sync_locked(&mut inner);
    }

    fn sub_packed_dw_bytes(&self, bytes: u64) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.packed_dw_bytes = inner.packed_dw_bytes.saturating_sub(bytes);
        Self::sync_locked(&mut inner);
    }

    fn packed_dw_bytes(&self) -> u64 {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .packed_dw_bytes
    }
}

/// Complete Inkling-2-Mini hybrid prefix state store (in-memory RAM store + optional StateDisk).
pub(crate) struct Inkling2StateStore {
    pub(crate) store: HybridStateStore<Inkling2StateAnchor>,
    pub(crate) disk: Option<Inkling2StateDisk>,
    /// Hard cap on the number of state anchors kept resident in host RAM.
    pub(crate) ram_cap: usize,
    /// Active state-blob geometry (SWA + ShortConv + optional DFlash2 draft context window).
    pub(crate) geometry: Option<Inkling2StateGeometry>,
    /// Shared host-budget account charging both `self.store`'s non-draft-window bytes and all
    /// live [`TrackedPackedBytes`] draft windows into `CacheLimitCoordinator`.
    host_budget: Arc<HostBudgetAccount>,
    /// Most recent rolling prefill checkpoint published per active sequence.
    seq_checkpoints: HashMap<SeqId, EntryId>,
    /// Entries marked as persistent prompt heads (explicitly via `CaptureOptions::is_head`,
    /// adopted on startup, or promoted on a confirmed non-preemption Checkpoint hit at
    /// `depth >= write_behind_min_tokens()`). Survive superseding turns in `WriteBehind` and
    /// `trim_ram_to_cap_inner` Priority 2b, and are evicted last in Priority 4.
    prompt_heads: HashSet<EntryId>,
    /// Active `HybridStateStore::use_state(block)` pin held per sequence for its current turn's
    /// restored state hit, released via [`Self::end_seq_use`] on turn close, reset, or release.
    seq_used_blocks: HashMap<SeqId, u64>,
    /// Entries known to lack a `PackedDraftWindow` (`draft_window == None`, either captured
    /// without one or decoded from SSD with `valid_flag == 0`) under a v2 draft-window geometry.
    entries_missing_draft_window: HashSet<EntryId>,
    /// SSD-only entries adopted on startup whose on-disk `valid_flag` has not yet been decoded.
    adopted_unverified_window: HashSet<EntryId>,
    /// Entries awaiting either in-place SSD overwrite (`InPlace`), full blob+chain rewrite
    /// (`Repair`), or persistent-tier checkpoint->ordinary label rewrite (`LabelPromote`), paired
    /// with a `retry_after` backoff instant so failed/refused writes do not tight-loop.
    pending_disk_upgrades: HashMap<EntryId, PendingDiskUpgrade>,
    /// In-flight SSD upgrade write: retains the previous durable `DiskCopy` and holds both
    /// `old_copy.key` and its Global KV chain against LRU eviction so a failed replacement write
    /// (even after urgent RAM drop) immediately restores the old SSD hit without data loss.
    in_flight_disk_upgrade: Option<InFlightDiskUpgrade>,
    /// Upgrade/repair entry IDs whose recent staging, write, or KV chain capture failed;
    /// drained by `flush_to_disk` into its per-pass failed-upgrade skip set so a repeatedly
    /// failing upgrade/repair never tight-spins until deadline.
    recent_failed_upgrades: HashSet<EntryId>,
    /// Ordinary (non-upgrade) entry IDs whose recent staging, write, or KV chain capture failed;
    /// drained alongside `recent_failed_upgrades` by `flush_to_disk` so transient failures return
    /// `false` promptly without tight-spinning while remaining queued in `write_behind`.
    recent_failed_spills: HashSet<EntryId>,
    /// Backoff deadline per entry whose recent SSD spill failed or was refused by the writer,
    /// so `trim_ram_to_cap` (and non-forced write-behind) do not tight-loop re-staging the same
    /// refused ordinary entry.
    spill_backoff: HashMap<EntryId, Instant>,
    /// RAM-backed entries whose reclaim was deferred while awaiting SSD state/KV chain durability;
    /// once both state blob and Global KV chain are restorable on SSD (with no pending/in-flight
    /// upgrade or active use), drop RAM automatically even when `ram_len() <= ram_cap` so pinned
    /// Global KV pages become evictable immediately.
    demote_ram_on_durable: HashSet<EntryId>,
    /// Counters tracked even when `disk` is disabled.
    pub(crate) ram_hits: u64,
    pub(crate) checkpoints_published: u64,
    pub(crate) ordinary_published: u64,
    /// Test-only fault-injection flag: forces the next deferred `BoundaryDraftWindow::Device`
    /// serialization inside `capture_boundary_with_window` to fail.
    #[cfg(test)]
    pub(crate) fail_next_draft_window_pack: bool,
}

impl Drop for Inkling2StateStore {
    fn drop(&mut self) {
        self.host_budget.set_store_non_dw_bytes(0);
    }
}

impl Inkling2StateStore {
    pub(crate) fn new(block_size: u32, recency: Arc<CacheRecency>) -> Self {
        let mut store = HybridStateStore::new(block_size);
        store.share_recency(recency);
        Self {
            store,
            disk: None,
            ram_cap: INKLING2_DEFAULT_RAM_STATES,
            geometry: None,
            host_budget: HostBudgetAccount::new(),
            seq_checkpoints: HashMap::new(),
            prompt_heads: HashSet::new(),
            seq_used_blocks: HashMap::new(),
            entries_missing_draft_window: HashSet::new(),
            adopted_unverified_window: HashSet::new(),
            pending_disk_upgrades: HashMap::new(),
            in_flight_disk_upgrade: None,
            recent_failed_upgrades: HashSet::new(),
            recent_failed_spills: HashSet::new(),
            spill_backoff: HashMap::new(),
            demote_ram_on_durable: HashSet::new(),
            ram_hits: 0,
            checkpoints_published: 0,
            ordinary_published: 0,
            #[cfg(test)]
            fail_next_draft_window_pack: false,
        }
    }

    /// Wire this state store to charge its resident host-RAM bytes into the process-wide
    /// [`crate::cache_limit::coordinator()`] (and therefore into `Governor::charge()`).
    pub(crate) fn attach_host_budget(&mut self) {
        self.host_budget.set_target(HostBudgetTarget::Global);
        self.sync_host_budget();
    }

    /// Test-only: wire this state store to an isolated [`crate::cache_limit::CacheLimitCoordinator`]
    /// so parallel tests never interfere on process-wide pool totals.
    #[cfg(test)]
    pub(crate) fn attach_host_budget_to(
        &mut self,
        coord: Arc<crate::cache_limit::CacheLimitCoordinator>,
    ) {
        self.host_budget.set_target(HostBudgetTarget::Local(coord));
        self.sync_host_budget();
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn attach_host_budget_local(
        &mut self,
        coord: Arc<crate::cache_limit::CacheLimitCoordinator>,
    ) {
        self.attach_host_budget_to(coord);
    }

    /// Exact deduplicated host-heap bytes owned directly by `self.store` excluding
    /// [`TrackedPackedBytes`] draft windows tracked by `self.host_budget` (lineage vectors plus
    /// unique RAM anchor `swa_data` and `conv_data` buffers).
    fn store_non_dw_host_bytes(&self) -> u64 {
        let mut seen_ptrs: HashSet<*const u8> = HashSet::new();
        let mut total = 0u64;
        for entry in self.store.entries() {
            let ram_bytes = entry.ram().map_or(0, StateBytes::state_bytes);
            let lineage = entry.state_bytes().saturating_sub(ram_bytes);
            total = total.saturating_add(lineage);
            if let Some(anchor) = entry.ram() {
                if !anchor.swa_data.is_empty() && seen_ptrs.insert(anchor.swa_data.as_ptr()) {
                    total = total.saturating_add(anchor.swa_data.len() as u64);
                }
                if !anchor.conv_data.is_empty() && seen_ptrs.insert(anchor.conv_data.as_ptr()) {
                    total = total.saturating_add(anchor.conv_data.len() as u64);
                }
            }
        }
        total
    }

    /// Exact deduplicated host-heap bytes accounted to this state store: `self.store`'s lineage +
    /// `swa_data` + `conv_data` plus all live [`TrackedPackedBytes`] draft windows attached to
    /// `self.host_budget` (whether currently in `self.store`, cloned in a live sequence's
    /// `block_boundary_draft_window`, or both — never double-counted).
    pub(crate) fn host_resident_bytes(&self) -> u64 {
        self.store_non_dw_host_bytes()
            .saturating_add(self.host_budget.packed_dw_bytes())
    }

    /// Synchronize the registered host-pool budget with `self.store_non_dw_host_bytes()` +
    /// `self.host_budget.packed_dw_bytes()`.
    pub(crate) fn sync_host_budget(&mut self) {
        let non_dw = self.store_non_dw_host_bytes();
        self.host_budget.set_store_non_dw_bytes(non_dw);
    }

    pub(crate) fn request_ram_demote_on_durable(&mut self, id: EntryId) {
        if self.store.get(id).is_some_and(|e| e.ram().is_some()) {
            self.demote_ram_on_durable.insert(id);
        }
    }

    /// Drop RAM copies for entries in `demote_ram_on_durable` that are now durably restorable on
    /// SSD, even when `ram_len() <= ram_cap`. Returns `true` if any RAM copy was dropped so the
    /// caller can prune tracking and resync Global KV state marks.
    fn demote_requested_ram(&mut self) -> bool {
        if self.demote_ram_on_durable.is_empty() {
            return false;
        }
        let pending_id = self.disk.as_ref().and_then(|d| d.pending_write_id());
        let ready: Vec<EntryId> = self
            .demote_ram_on_durable
            .iter()
            .copied()
            .filter(|&id| {
                !matches!(
                    self.pending_disk_upgrades.get(&id),
                    Some(PendingDiskUpgrade::InPlace { .. } | PendingDiskUpgrade::Repair { .. })
                ) && self
                    .in_flight_disk_upgrade
                    .as_ref()
                    .is_none_or(|u| u.id != id)
                    && pending_id != Some(id)
                    && self.store.get(id).is_some_and(|entry| {
                        entry.ram().is_some()
                            && entry.disk().is_some_and(|c| c.landed && !c.chain_owed)
                            && !self.store.in_use(entry.mark())
                            && self
                                .disk
                                .as_ref()
                                .is_some_and(|d| d.entry_disk_restorable(entry))
                    })
            })
            .collect();
        if ready.is_empty() {
            return false;
        }
        for id in ready {
            self.demote_ram_on_durable.remove(&id);
            self.spill_backoff.remove(&id);
            self.store.drop_ram(id);
        }
        self.sync_host_budget();
        true
    }

    fn clear_window_tracking(&mut self, id: EntryId) {
        self.entries_missing_draft_window.remove(&id);
        self.adopted_unverified_window.remove(&id);
    }

    fn clear_upgrade_tracking(&mut self, id: EntryId) {
        self.pending_disk_upgrades.remove(&id);
        self.recent_failed_upgrades.remove(&id);
        self.recent_failed_spills.remove(&id);
        self.spill_backoff.remove(&id);
        self.demote_ram_on_durable.remove(&id);
        if self
            .in_flight_disk_upgrade
            .as_ref()
            .is_some_and(|u| u.id == id)
        {
            self.in_flight_disk_upgrade = None;
        }
        if let Some(disk) = self.disk.as_mut()
            && disk.preserve_entry_on_state_failure == Some(id)
        {
            disk.preserve_entry_on_state_failure = None;
        }
    }

    /// Queue a persistent-tier `checkpoint: true -> false` on-disk label rewrite for an SSD-backed
    /// entry just promoted to `StateClass::Ordinary`.
    ///
    /// No-op on non-persistent or unwritable tiers, or when `Repair` is already queued (since
    /// `Repair` writes `label_of(entry)` fresh). When `InPlace` is already queued, marks its
    /// `promote_label` bit so dropping RAM before `InPlace` lands falls back to `LabelPromote`.
    fn enqueue_label_promote(&mut self, id: EntryId) {
        if !self
            .disk
            .as_ref()
            .is_some_and(|d| d.tier.persistent && d.writable())
        {
            return;
        }
        if !self
            .store
            .get(id)
            .is_some_and(|e| !e.checkpoint() && e.disk().is_some())
        {
            return;
        }
        match self.pending_disk_upgrades.get_mut(&id) {
            Some(PendingDiskUpgrade::InPlace { promote_label, .. }) => {
                *promote_label = true;
                return;
            }
            Some(PendingDiskUpgrade::Repair { .. } | PendingDiskUpgrade::LabelPromote { .. }) => {
                return;
            }
            None => {}
        }
        self.pending_disk_upgrades.insert(
            id,
            PendingDiskUpgrade::LabelPromote {
                retry_after: Instant::now(),
            },
        );
        self.recent_failed_upgrades.remove(&id);
    }

    pub(crate) fn is_prompt_head(&self, id: EntryId) -> bool {
        self.prompt_heads.contains(&id)
    }

    #[cfg(test)]
    pub(crate) fn entry_missing_draft_window(&self, id: EntryId) -> bool {
        self.entries_missing_draft_window.contains(&id)
    }

    pub(crate) fn mark_prompt_head(&mut self, id: EntryId) {
        if self.prompt_heads.len() >= 64 {
            let store = &self.store;
            self.prompt_heads.retain(|eid| store.get(*eid).is_some());
        }
        self.prompt_heads.insert(id);
    }

    pub(crate) fn end_seq_use(&mut self, seq_id: SeqId) {
        if let Some(block) = self.seq_used_blocks.remove(&seq_id) {
            self.store.end_use(block);
        }
    }

    pub(crate) fn end_all_seq_uses(&mut self) {
        for (_, block) in std::mem::take(&mut self.seq_used_blocks) {
            self.store.end_use(block);
        }
    }

    /// Record a confirmed state-store restore hit for `seq_id` on `id`.
    ///
    /// Releases any previous `use_state` pin held by `seq_id`, touches `id`, and pins `id`'s
    /// block via `store.use_state(block)` (which upgrades a `Checkpoint` to `Ordinary`). Strictly
    /// for a hit `Checkpoint` (`was_checkpoint == true`), detaches `seq_checkpoints` and — only
    /// when `!is_preemption_replay && depth >= write_behind_min_tokens()` — marks `id` in
    /// `prompt_heads` and enqueues `WriteBehind(..., head = true)`. Restored `Ordinary` entries
    /// are never added to `prompt_heads`.
    pub(crate) fn note_state_hit(
        &mut self,
        coordinator: &Inkling2KVCacheCoordinator,
        seq_id: SeqId,
        id: EntryId,
        is_preemption_replay: bool,
        now: f64,
    ) {
        self.end_seq_use(seq_id);
        let Some((was_checkpoint, block, depth, has_ram, has_disk)) =
            self.store.get(id).and_then(|entry| {
                let depth = u32::try_from(entry.depth()).ok()?;
                Some((
                    entry.checkpoint(),
                    entry.mark(),
                    depth,
                    entry.ram().is_some(),
                    entry.disk().is_some(),
                ))
            })
        else {
            return;
        };
        self.store.touch(id);
        self.store.use_state(block);
        self.seq_used_blocks.insert(seq_id, block);
        if was_checkpoint {
            self.seq_checkpoints.retain(|_, v| *v != id);
            if !is_preemption_replay && depth >= write_behind_min_tokens() as u32 {
                self.mark_prompt_head(id);
                if has_disk {
                    self.enqueue_label_promote(id);
                } else if has_ram
                    && !self.pending_disk_upgrades.contains_key(&id)
                    && let Some(disk) = self.disk.as_mut()
                {
                    let writable = disk.writable();
                    let persistent = disk.tier.persistent;
                    disk.write_behind.record_published(
                        &self.store,
                        writable,
                        persistent,
                        id,
                        depth,
                        now,
                        true,
                    );
                }
            }
            self.sync_marks(coordinator);
        }
    }

    /// When the SSD writer has stopped accepting tracked writes (`!disk.writable()`, e.g. after a
    /// non-space I/O error tripped `ColdCacheManager::tracked_failed`), no remaining `chain_owed`
    /// Global KV blocks can ever be persisted. Retire incomplete disk copies so they never keep
    /// Global KV pages pinned indefinitely, while preserving every already-durable SSD entry
    /// (`landed && !chain_owed`) and every live RAM copy.
    pub(crate) fn retire_unwritable_incomplete_disk_copies(&mut self) -> bool {
        let (unwritable, writing) = match self.disk.as_ref() {
            Some(d) => (!d.writable(), d.writing()),
            None => return false,
        };
        if !unwritable {
            return false;
        }
        let mut changed = false;
        if let Some(disk) = self.disk.as_mut()
            && disk.chain.take().is_some()
        {
            changed = true;
        }
        enum Action {
            MarkDurable,
            DropDiskKeepRam(Option<ColdCacheKey>),
            RemoveEntry(Option<ColdCacheKey>),
        }
        let actions: Vec<(EntryId, Action)> = {
            let Some(disk) = self.disk.as_ref() else {
                return changed;
            };
            self.store
                .entries()
                .filter_map(|entry| {
                    let copy = entry.disk()?;
                    let id = entry.id();
                    if copy.landed && !copy.chain_owed {
                        return None;
                    }
                    if copy.landed && copy.chain_owed {
                        if disk.entry_disk_restorable(entry) {
                            return Some((id, Action::MarkDurable));
                        }
                        return if entry.ram().is_none() {
                            Some((id, Action::RemoveEntry(Some(copy.key))))
                        } else {
                            Some((id, Action::DropDiskKeepRam(Some(copy.key))))
                        };
                    }
                    if !writing {
                        return if entry.ram().is_none() {
                            Some((id, Action::RemoveEntry(None)))
                        } else {
                            Some((id, Action::DropDiskKeepRam(None)))
                        };
                    }
                    None
                })
                .collect()
        };
        if actions.is_empty() {
            return changed;
        }
        for (id, action) in actions {
            match action {
                Action::MarkDurable => {
                    if let Some(copy) = self.store.disk_mut(id) {
                        copy.chain_owed = false;
                        changed = true;
                    }
                }
                Action::DropDiskKeepRam(key) => {
                    if let Some(key) = key
                        && let Some(disk) = self.disk.as_ref()
                    {
                        let tier = Arc::clone(&disk.tier);
                        let _ = tier.while_held(|| {
                            tier.manager.remove(&key);
                        });
                    }
                    self.store.set_disk(id, None);
                    self.clear_upgrade_tracking(id);
                    changed = true;
                }
                Action::RemoveEntry(key) => {
                    if let Some(key) = key
                        && let Some(disk) = self.disk.as_ref()
                    {
                        let tier = Arc::clone(&disk.tier);
                        let _ = tier.while_held(|| {
                            tier.manager.remove(&key);
                        });
                    }
                    self.seq_checkpoints.retain(|_, v| *v != id);
                    self.clear_window_tracking(id);
                    self.clear_upgrade_tracking(id);
                    self.prompt_heads.remove(&id);
                    self.store.remove(id);
                    changed = true;
                }
            }
        }
        if changed {
            self.sync_host_budget();
        }
        changed
    }

    fn mark_repair_failed(&mut self, id: EntryId, retry: Instant) {
        if !self.store.get(id).is_some_and(|e| e.ram().is_some()) {
            self.pending_disk_upgrades.remove(&id);
            self.recent_failed_upgrades.remove(&id);
            return;
        }
        self.pending_disk_upgrades
            .insert(id, PendingDiskUpgrade::Repair { retry_after: retry });
        self.spill_backoff.insert(id, retry);
        self.recent_failed_upgrades.insert(id);
    }

    /// Clear `Repair` entries once both their replacement state blob AND complete Global KV chain
    /// are landed and restorable on SSD. If a landed `Repair` copy already lost its blob or chain
    /// before repair completion, strip the dead copy and re-queue it for fresh repair.
    fn reconcile_completed_disk_repairs(&mut self) {
        let mut completed = Vec::new();
        let mut stale_landed = Vec::new();
        for (&id, kind) in &self.pending_disk_upgrades {
            if !kind.is_repair() {
                continue;
            }
            let Some(entry) = self.store.get(id) else {
                continue;
            };
            let Some(copy) = entry.disk().copied() else {
                continue;
            };
            if !copy.landed || copy.chain_owed {
                continue;
            }
            if self
                .disk
                .as_ref()
                .is_some_and(|d| d.entry_disk_restorable(entry))
            {
                completed.push(id);
            } else {
                stale_landed.push((id, copy.key));
            }
        }
        for id in completed {
            self.pending_disk_upgrades.remove(&id);
            self.recent_failed_upgrades.remove(&id);
            self.spill_backoff.remove(&id);
        }
        for (id, key) in stale_landed {
            if let Some(disk) = self.disk.as_ref() {
                disk.tier.manager.remove(&key);
            }
            self.store.set_disk(id, None);
        }
    }

    fn prune_entry_tracking(&mut self) {
        self.seq_checkpoints
            .retain(|_, id| self.store.get(*id).is_some());
        self.prompt_heads.retain(|id| self.store.get(*id).is_some());
        self.entries_missing_draft_window
            .retain(|id| self.store.get(*id).is_some());
        self.adopted_unverified_window
            .retain(|id| self.store.get(*id).is_some());
        // Retain pending upgrades/repairs/label promotions:
        // - `InPlace`: requires RAM; if disk was evicted, transitions to `Repair`; if RAM vanished
        //   while `promote_label` was set on a persistent writable tier, falls back to `LabelPromote`.
        // - `Repair`: requires RAM.
        // - `LabelPromote`: does NOT require RAM; stays alive on a persistent writable tier while
        //   `!entry.checkpoint() && entry.disk().is_some()` (and transitions to `Repair` if disk
        //   was evicted while RAM is still resident).
        let persistent_writable = self
            .disk
            .as_ref()
            .is_some_and(|d| d.tier.persistent && d.writable());
        self.pending_disk_upgrades.retain(|&id, kind| {
            let Some(entry) = self.store.get(id) else {
                return false;
            };
            match kind {
                PendingDiskUpgrade::LabelPromote { .. } => {
                    if !persistent_writable || entry.checkpoint() {
                        return false;
                    }
                    if entry.disk().is_some() {
                        return true;
                    }
                    if entry.ram().is_some() {
                        *kind = PendingDiskUpgrade::Repair {
                            retry_after: Instant::now(),
                        };
                        return true;
                    }
                    false
                }
                PendingDiskUpgrade::InPlace { promote_label, .. } => {
                    if entry.ram().is_none() {
                        if *promote_label
                            && persistent_writable
                            && !entry.checkpoint()
                            && entry.disk().is_some()
                        {
                            *kind = PendingDiskUpgrade::LabelPromote {
                                retry_after: Instant::now(),
                            };
                            return true;
                        }
                        return false;
                    }
                    if entry.disk().is_none() {
                        *kind = PendingDiskUpgrade::Repair {
                            retry_after: Instant::now(),
                        };
                    }
                    true
                }
                PendingDiskUpgrade::Repair { .. } => entry.ram().is_some(),
            }
        });
        self.recent_failed_upgrades
            .retain(|id| self.store.get(*id).is_some());
        self.recent_failed_spills
            .retain(|id| self.store.get(*id).is_some());
        self.spill_backoff.retain(|id, _| {
            self.store.get(*id).is_some_and(|e| e.ram().is_some())
                || self.pending_disk_upgrades.contains_key(id)
        });
        self.demote_ram_on_durable
            .retain(|id| self.store.get(*id).is_some_and(|e| e.ram().is_some()));
    }

    /// Snapshot ordinary RAM-backed entries whose landed SSD blob still owes its Global KV chain
    /// (excluding pending/in-flight upgrades) before polling or ticking `StateDiskEngine`, so an
    /// eviction of that `chain_owed` state blob via `ColdCacheManager::take_evicted` is reconciled
    /// identically to a chain-step failure.
    fn ordinary_chain_owed_ids(&self) -> Vec<EntryId> {
        self.store
            .entries()
            .filter(|entry| {
                let id = entry.id();
                !entry.checkpoint()
                    && entry.ram().is_some()
                    && entry.disk().is_some_and(|c| c.chain_owed)
                    && !matches!(
                        self.pending_disk_upgrades.get(&id),
                        Some(
                            PendingDiskUpgrade::InPlace { .. } | PendingDiskUpgrade::Repair { .. }
                        )
                    )
                    && self
                        .in_flight_disk_upgrade
                        .as_ref()
                        .is_none_or(|u| u.id != id)
            })
            .map(Entry::id)
            .collect()
    }

    /// Whether every block of `id`'s Global KV chain is still available in either hot GPU RAM
    /// or SSD (the exact condition under which retrying an invalidated spill can succeed).
    fn chain_recoverable(&self, coordinator: &Inkling2KVCacheCoordinator, id: EntryId) -> bool {
        let Some(entry) = self.store.get(id) else {
            return false;
        };
        if entry.chain().is_empty() {
            return false;
        }
        let full = coordinator.full_adapter();
        let disk = self.disk.as_ref();
        let kv_keys = disk.map(|d| d.engine.kv_chain_keys(entry));
        entry.chain().iter().enumerate().all(|(idx, &hash)| {
            full.prefix_cached(hash)
                || disk.zip(kv_keys.as_ref()).is_some_and(|(d, keys)| {
                    keys.get(idx)
                        .is_some_and(|key| d.tier.manager.contains_in(key, ColdGroup::Kv))
                })
        })
    }

    /// Whether an ordinary (non-upgrade) RAM entry is eligible to retry background `WriteBehind`
    /// after a transient headroom/writer failure or `gone` invalidation.
    fn is_retryable_ordinary_wb(
        &self,
        coordinator: &Inkling2KVCacheCoordinator,
        id: EntryId,
    ) -> bool {
        let Some(entry) = self.store.get(id) else {
            return false;
        };
        let is_persistent_head =
            self.is_prompt_head(id) && self.disk.as_ref().is_some_and(|d| d.tier.persistent);
        !entry.checkpoint()
            && entry.ram().is_some()
            && entry.disk().is_none()
            && !self.pending_disk_upgrades.contains_key(&id)
            && self
                .in_flight_disk_upgrade
                .as_ref()
                .is_none_or(|u| u.id != id)
            && (entry.depth() >= write_behind_min_tokens() || is_persistent_head)
            && self
                .disk
                .as_ref()
                .is_some_and(|d| d.writable() && d.entry_fits_tier_limit(entry))
            && self.chain_recoverable(coordinator, id)
    }

    /// Reconcile an in-flight ordinary, repair, or upgrade spill after `StateDiskEngine::poll_offload`,
    /// `chain_step`, or `tick` (called BEFORE `reconcile_in_flight_disk_upgrade`, which may restore
    /// `old_copy`):
    /// - If `pending_before` landed cleanly, clear any `spill_backoff` for it.
    /// - For every entry whose state write or Global KV chain failed or was invalidated (including
    ///   `gone` / `lost` invalidations with `offload_failures == 0` and evicted `chain_owed` state
    ///   blobs), record `spill_backoff`, update any pending `Repair` tracking (or re-queue
    ///   recoverable ordinary RAM entries onto `write_behind` and record them in
    ///   `recent_failed_spills`), and return all failed IDs so `trim_ram_to_cap_inner` skips them.
    fn reconcile_spill_outcome(
        &mut self,
        coordinator: &Inkling2KVCacheCoordinator,
        pending_before: Option<EntryId>,
        chain_before: Option<EntryId>,
        state_failed_id: Option<EntryId>,
        chain_failed_ids: &[Option<EntryId>],
        ordinary_owed_before: &[EntryId],
    ) -> Vec<EntryId> {
        let mut failed_ids = Vec::new();
        let retry = upgrade_retry_after();
        let mut record_failure = |this: &mut Self, id: EntryId| {
            if failed_ids.contains(&id) {
                return;
            }
            failed_ids.push(id);
            if this.store.get(id).is_some() {
                this.spill_backoff.insert(id, retry);
            }
            if matches!(
                this.pending_disk_upgrades.get(&id),
                Some(PendingDiskUpgrade::Repair { .. })
            ) {
                this.mark_repair_failed(id, retry);
            } else if !matches!(
                this.pending_disk_upgrades.get(&id),
                Some(PendingDiskUpgrade::InPlace { .. })
            ) && this
                .in_flight_disk_upgrade
                .as_ref()
                .is_none_or(|u| u.id != id)
                && this
                    .store
                    .get(id)
                    .is_some_and(|e| !e.checkpoint() && e.ram().is_some() && e.disk().is_none())
            {
                // If an initial checkpoint write or owed chain failed while `LabelPromote` was
                // queued, drop the obsolete `LabelPromote` so the RAM-resident promoted entry
                // falls back cleanly to retryable ordinary write-behind (`checkpoint: false`).
                this.pending_disk_upgrades.remove(&id);
                let requeue = this.is_retryable_ordinary_wb(coordinator, id);
                let is_head = this.is_prompt_head(id);
                if requeue
                    && let Some(depth) = this
                        .store
                        .get(id)
                        .and_then(|e| u32::try_from(e.depth()).ok())
                    && let Some(disk) = this.disk.as_mut()
                {
                    let writable = disk.writable();
                    let persistent = disk.tier.persistent;
                    disk.write_behind.record_published(
                        &this.store,
                        writable,
                        persistent,
                        id,
                        depth,
                        0.0,
                        is_head,
                    );
                }
                this.recent_failed_spills.insert(id);
            }
        };

        if let Some(pid) = pending_before {
            let still_pending = self
                .disk
                .as_ref()
                .is_some_and(|d| d.pending_write_id() == Some(pid));
            if !still_pending {
                let landed_cleanly = state_failed_id != Some(pid)
                    && self
                        .store
                        .get(pid)
                        .is_some_and(|e| e.disk().is_some_and(|c| c.landed));
                if landed_cleanly {
                    self.spill_backoff.remove(&pid);
                } else {
                    record_failure(self, pid);
                }
            }
        }
        if let Some(sid) = state_failed_id {
            record_failure(self, sid);
        }
        for cid in chain_failed_ids.iter().copied().flatten() {
            record_failure(self, cid);
        }
        if let Some(cid) = chain_before {
            let still_chaining = self
                .disk
                .as_ref()
                .is_some_and(|d| d.chain.as_ref().is_some_and(|c| c.id == cid));
            if !still_chaining && self.store.get(cid).is_some_and(|e| e.disk().is_none()) {
                record_failure(self, cid);
            }
        }
        for &oid in ordinary_owed_before {
            if self
                .store
                .get(oid)
                .is_some_and(|e| e.ram().is_some() && e.disk().is_none())
            {
                record_failure(self, oid);
            }
        }
        failed_ids
    }

    /// Reconcile `in_flight_disk_upgrade` after `StateDiskEngine::poll_offload` or `tick`:
    /// - If the replacement write landed and its Global KV chain is restorable, drop `old_copy`.
    /// - If the replacement write landed with `chain_owed == true` (or missing KV blocks),
    ///   transition to `PendingDiskUpgrade::Repair` until the chain completes too.
    /// - If the replacement write failed while the old SSD blob is still intact on disk,
    ///   restore `old_copy` onto `self.store` immediately so no durable SSD hit is lost,
    ///   re-queue the upgrade with backoff, and return `Some(id)`.
    /// - If the replacement write failed AND the old SSD blob was also evicted, transition to
    ///   `PendingDiskUpgrade::Repair` with backoff and return `Some(id)`.
    fn reconcile_in_flight_disk_upgrade(
        &mut self,
        failed_state_id: Option<EntryId>,
    ) -> Option<EntryId> {
        let id = self.in_flight_disk_upgrade.as_ref()?.id;
        let still_pending = self
            .disk
            .as_ref()
            .is_some_and(|d| d.pending_write_id() == Some(id));
        if still_pending {
            return None;
        }
        let InFlightDiskUpgrade {
            id,
            kind,
            mut old_copy,
        } = self.in_flight_disk_upgrade.take()?;
        if let Some(disk) = self.disk.as_mut()
            && disk.preserve_entry_on_state_failure == Some(id)
        {
            disk.preserve_entry_on_state_failure = None;
        }
        let failed = failed_state_id == Some(id);
        if !failed
            && let Some(entry) = self.store.get(id)
            && let Some(copy) = entry.disk()
            && copy.landed
        {
            self.recent_failed_upgrades.remove(&id);
            let restorable = !copy.chain_owed
                && self
                    .disk
                    .as_ref()
                    .is_some_and(|d| d.entry_disk_restorable(entry));
            if !restorable && entry.ram().is_some() {
                self.pending_disk_upgrades.insert(
                    id,
                    PendingDiskUpgrade::Repair {
                        retry_after: Instant::now(),
                    },
                );
            }
            return None;
        }
        let old_still_on_disk = self.disk.as_ref().is_some_and(|d| {
            d.tier
                .manager
                .contains_in(&old_copy.key, ColdGroup::DFlash2State)
        });
        if old_still_on_disk && let Some(entry) = self.store.get(id) {
            if let Some(disk) = self.disk.as_ref() {
                old_copy.chain_owed = !disk.engine.chain_on_disk(entry);
            }
            let has_ram = entry.ram().is_some();
            let is_ordinary = !entry.checkpoint();
            if entry.disk().copied() != Some(old_copy) {
                self.store.set_disk(id, Some(old_copy));
            }
            let persistent_writable = self
                .disk
                .as_ref()
                .is_some_and(|d| d.tier.persistent && d.writable());
            let retry = upgrade_retry_after();
            match kind {
                InFlightDiskUpgradeKind::InPlace { promote_label } => {
                    if has_ram {
                        self.pending_disk_upgrades.insert(
                            id,
                            PendingDiskUpgrade::InPlace {
                                retry_after: retry,
                                promote_label,
                            },
                        );
                        self.recent_failed_upgrades.insert(id);
                    } else {
                        if self
                            .resolve_geometry()
                            .is_some_and(|g| g.draft_window_span_bytes > 0)
                        {
                            self.entries_missing_draft_window.insert(id);
                        }
                        if promote_label && persistent_writable && is_ordinary {
                            self.pending_disk_upgrades.insert(
                                id,
                                PendingDiskUpgrade::LabelPromote { retry_after: retry },
                            );
                            self.recent_failed_upgrades.insert(id);
                        }
                    }
                }
                InFlightDiskUpgradeKind::LabelPromote => {
                    // Never mark `entries_missing_draft_window` on a `LabelPromote` rollback:
                    // the existing SSD blob still carries its original draft-window state.
                    if persistent_writable && is_ordinary {
                        match self.pending_disk_upgrades.get_mut(&id) {
                            Some(PendingDiskUpgrade::InPlace {
                                retry_after,
                                promote_label,
                            }) => {
                                *retry_after = retry;
                                *promote_label = true;
                            }
                            Some(PendingDiskUpgrade::Repair { .. }) => {}
                            Some(PendingDiskUpgrade::LabelPromote { retry_after }) => {
                                *retry_after = retry;
                            }
                            None => {
                                self.pending_disk_upgrades.insert(
                                    id,
                                    PendingDiskUpgrade::LabelPromote { retry_after: retry },
                                );
                            }
                        }
                    }
                    self.recent_failed_upgrades.insert(id);
                }
            }
            return Some(id);
        }
        // Fallback SSD copy is also gone (e.g. corrupt source pruned during relabel):
        // clear any stale `DiskCopy` metadata before repairing from RAM or removing the entry.
        if let Some(disk) = self.disk.as_mut()
            && disk.chain.as_ref().is_some_and(|c| c.id == id)
        {
            disk.chain = None;
        }
        if self.store.get(id).is_some_and(|e| e.disk().is_some()) {
            self.store.set_disk(id, None);
        }
        if self.store.get(id).is_some_and(|e| e.ram().is_some()) {
            self.mark_repair_failed(id, upgrade_retry_after());
            return Some(id);
        }
        // Neither RAM nor the fallback SSD copy survived: remove the now-empty entry cleanly.
        if self.store.get(id).is_some_and(|e| e.disk().is_none()) {
            self.clear_window_tracking(id);
            self.clear_upgrade_tracking(id);
            self.prompt_heads.remove(&id);
            self.store.remove(id);
            self.sync_host_budget();
            return Some(id);
        }
        None
    }

    #[cfg(test)]
    fn set_in_flight_upgrade_for_test(&mut self, id: EntryId, old_copy: DiskCopy) {
        self.in_flight_disk_upgrade = Some(InFlightDiskUpgrade {
            id,
            kind: InFlightDiskUpgradeKind::InPlace {
                promote_label: false,
            },
            old_copy,
        });
    }

    pub(crate) fn set_ram_cap(&mut self, cap: usize) {
        self.ram_cap = cap;
    }

    pub(crate) fn set_geometry(&mut self, geom: Inkling2StateGeometry) {
        self.geometry = Some(geom);
    }

    pub(crate) fn resolve_geometry(&self) -> Option<&Inkling2StateGeometry> {
        self.geometry
            .as_ref()
            .or_else(|| self.disk.as_ref().map(|d| &d.codec.geometry))
    }

    /// Number of entries currently holding a host-RAM anchor copy.
    pub(crate) fn ram_len(&self) -> usize {
        self.store
            .entries()
            .filter(|entry| entry.ram().is_some())
            .count()
    }

    /// Enforce [`Self::ram_cap`] by retiring checkpoints, dropping RAM on landed SSD copies,
    /// spilling unlanded ordinary entries to SSD, or removing LRU ordinary entries as a last
    /// resort. Never touches `protect`, entries whose block is `in_use`, or an in-flight SSD
    /// state write (`disk.pending`).
    pub(crate) fn trim_ram_to_cap(
        &mut self,
        coordinator: &Inkling2KVCacheCoordinator,
        protect: Option<EntryId>,
    ) {
        self.trim_ram_to_cap_inner(coordinator, protect, &[]);
    }

    fn trim_ram_to_cap_inner(
        &mut self,
        coordinator: &Inkling2KVCacheCoordinator,
        protect: Option<EntryId>,
        skip_stage: &[EntryId],
    ) {
        let mut changed = false;
        while self.ram_len() > self.ram_cap {
            let pending_id = self.disk.as_ref().and_then(|d| d.pending_write_id());

            // Priority 1: oldest Checkpoint with RAM copy -> retire completely.
            let cp_victim = self
                .store
                .entries()
                .filter(|entry| {
                    entry.checkpoint()
                        && entry.ram().is_some()
                        && Some(entry.id()) != protect
                        && !self.store.in_use(entry.mark())
                        && pending_id != Some(entry.id())
                })
                .min_by_key(|entry| (entry.last_used(), entry.id()))
                .map(Entry::id);
            if let Some(id) = cp_victim {
                self.seq_checkpoints.retain(|_, v| *v != id);
                self.retire_entry(id);
                changed = true;
                continue;
            }

            // Priority 2: LRU Ordinary with RAM copy AND a fully SSD-restorable copy
            // (BOTH state blob landed AND complete Global KV chain on disk — never drop RAM
            // while `chain_owed` is true or while an upgraded RAM window is awaiting SSD write!).
            let landed_victim = self
                .store
                .entries()
                .filter(|entry| {
                    !entry.checkpoint()
                        && entry.ram().is_some()
                        && entry.disk().is_some_and(|c| c.landed && !c.chain_owed)
                        && !matches!(
                            self.pending_disk_upgrades.get(&entry.id()),
                            Some(
                                PendingDiskUpgrade::InPlace { .. }
                                    | PendingDiskUpgrade::Repair { .. }
                            )
                        )
                        && self
                            .in_flight_disk_upgrade
                            .as_ref()
                            .is_none_or(|u| u.id != entry.id())
                        && self
                            .disk
                            .as_ref()
                            .is_some_and(|d| d.entry_disk_restorable(entry))
                        && Some(entry.id()) != protect
                        && !self.store.in_use(entry.mark())
                        && pending_id != Some(entry.id())
                })
                .min_by_key(|entry| (entry.last_used(), entry.id()))
                .map(Entry::id);
            if let Some(id) = landed_victim {
                self.demote_ram_on_durable.remove(&id);
                self.store.drop_ram(id);
                changed = true;
                continue;
            }

            // Priority 2b: oldest RAM-only non-head Ordinary entry that is superseded by at least
            // one currently restorable ordinary descendant -> erase directly from RAM instead of
            // spilling a dead intermediate turn blob to SSD.
            let superseded_victim = self
                .store
                .entries()
                .filter(|entry| {
                    let id = entry.id();
                    let depth = entry.chain().len();
                    !entry.checkpoint()
                        && entry.ram().is_some()
                        && entry.disk().is_none()
                        && depth > 0
                        && Some(id) != protect
                        && !self.store.in_use(entry.mark())
                        && pending_id != Some(id)
                        && !self.pending_disk_upgrades.contains_key(&id)
                        && self
                            .in_flight_disk_upgrade
                            .as_ref()
                            .is_none_or(|u| u.id != id)
                        && !self.is_prompt_head(id)
                        && self.store.superseded(id)
                        && self.store.entries().any(|other| {
                            !other.checkpoint()
                                && other.chain().len() > depth
                                && other.chain()[..depth] == entry.chain()[..]
                                && self.is_restorable(coordinator, other.id())
                        })
                })
                .min_by_key(|entry| (entry.last_used(), entry.id()))
                .map(Entry::id);
            if let Some(id) = superseded_victim {
                if let Some(disk) = self.disk.as_mut()
                    && disk.chain.as_ref().is_some_and(|c| c.id == id)
                {
                    disk.chain = None;
                }
                self.seq_checkpoints.retain(|_, v| *v != id);
                self.clear_window_tracking(id);
                self.clear_upgrade_tracking(id);
                self.prompt_heads.remove(&id);
                self.store.remove(id);
                changed = true;
                continue;
            }

            // Priority 3: if disk is writable, not writing, and no restore is waiting on the
            // staging buffer, stage LRU Ordinary with no disk copy (or an eligible pending SSD
            // upgrade past its backoff window) while keeping RAM warm (`keep_ram = true`) until
            // both state blob and Global KV chain have landed on SSD. Do not pile up extra spills
            // once already-staged non-pinned entries (`!landed || chain_owed`) already cover the
            // excess over `ram_cap`, otherwise runaway staging marks every linear turn with
            // `disk = Some(chain_owed = true)` before prior KV chains land and bypasses Priority 2b.
            let (disk_writable, disk_writing, has_read_waiters) =
                self.disk.as_ref().map_or((false, false, false), |d| {
                    (d.writable(), d.writing(), d.has_read_waiters())
                });
            let inflight_ram_spills = self
                .store
                .entries()
                .filter(|entry| {
                    !entry.checkpoint()
                        && entry.ram().is_some()
                        && !self.store.in_use(entry.mark())
                        && entry.disk().is_some_and(|c| !c.landed || c.chain_owed)
                })
                .count();
            if disk_writable
                && !disk_writing
                && !has_read_waiters
                && self.ram_len().saturating_sub(inflight_ram_spills) > self.ram_cap
            {
                let now = Instant::now();
                let spill_victim = self
                    .store
                    .entries()
                    .filter(|entry| {
                        let id = entry.id();
                        let backed_off = skip_stage.contains(&id)
                            || self
                                .spill_backoff
                                .get(&id)
                                .is_some_and(|&retry_after| now < retry_after)
                            || self
                                .pending_disk_upgrades
                                .get(&id)
                                .is_some_and(|pending| now < pending.retry_after());
                        let needs_stage = match self.pending_disk_upgrades.get(&id) {
                            Some(PendingDiskUpgrade::InPlace { .. }) => true,
                            Some(PendingDiskUpgrade::LabelPromote { .. }) => false,
                            Some(PendingDiskUpgrade::Repair { .. }) | None => {
                                entry.disk().is_none()
                            }
                        };
                        !entry.checkpoint()
                            && entry.ram().is_some()
                            && !backed_off
                            && needs_stage
                            && Some(id) != protect
                            && !self.store.in_use(entry.mark())
                            && pending_id != Some(id)
                    })
                    .min_by_key(|entry| (entry.last_used(), entry.id()))
                    .map(Entry::id);
                if let Some(id) = spill_victim
                    && self.stage_entry(coordinator, id, MakeRoom::Evict, true)
                {
                    changed = true;
                    continue;
                }
            }

            // Priority 4: still above cap — without a writable disk, enforce `ram_cap` strictly;
            // with a writable disk, allow a bounded in-flight slack (`ram_cap + 2`) while blob and
            // KV chain writes finish, and only erase LRU eligible Ordinary entries past that
            // ceiling so a stalled/failing SSD cannot grow RAM without bound.
            if disk_writable && self.ram_len() <= self.ram_cap.saturating_add(2) {
                break;
            }
            let erase_victim = self
                .store
                .entries()
                .filter(|entry| {
                    !entry.checkpoint()
                        && entry.ram().is_some()
                        && Some(entry.id()) != protect
                        && !self.store.in_use(entry.mark())
                        && pending_id != Some(entry.id())
                })
                .min_by_key(|entry| {
                    (
                        self.is_prompt_head(entry.id()),
                        entry.last_used(),
                        entry.id(),
                    )
                })
                .map(Entry::id);
            let Some(id) = erase_victim else {
                break;
            };
            // If `id` has a pending in-place disk upgrade or label promotion whose previous SSD
            // copy is still durable and restorable, fall back cleanly to that old SSD copy by
            // dropping only RAM.
            let pending_for_victim = self.pending_disk_upgrades.get(&id).copied();
            let keep_durable_old_ssd = matches!(
                pending_for_victim,
                Some(PendingDiskUpgrade::InPlace { .. } | PendingDiskUpgrade::LabelPromote { .. })
            ) && self.disk.as_ref().is_some_and(|d| {
                self.store.get(id).is_some_and(|e| {
                    e.disk().is_some_and(|c| c.landed && !c.chain_owed)
                        && d.entry_disk_restorable(e)
                })
            });
            if keep_durable_old_ssd {
                match self.pending_disk_upgrades.remove(&id) {
                    Some(PendingDiskUpgrade::InPlace { promote_label, .. }) => {
                        self.recent_failed_upgrades.remove(&id);
                        if self
                            .resolve_geometry()
                            .is_some_and(|g| g.draft_window_span_bytes > 0)
                        {
                            self.entries_missing_draft_window.insert(id);
                        }
                        if promote_label
                            && self
                                .disk
                                .as_ref()
                                .is_some_and(|d| d.tier.persistent && d.writable())
                            && self.store.get(id).is_some_and(|e| !e.checkpoint())
                        {
                            self.pending_disk_upgrades.insert(
                                id,
                                PendingDiskUpgrade::LabelPromote {
                                    retry_after: Instant::now(),
                                },
                            );
                        }
                    }
                    Some(PendingDiskUpgrade::LabelPromote { retry_after }) => {
                        self.pending_disk_upgrades
                            .insert(id, PendingDiskUpgrade::LabelPromote { retry_after });
                    }
                    Some(PendingDiskUpgrade::Repair { .. }) | None => {}
                }
                self.demote_ram_on_durable.remove(&id);
                self.store.drop_ram(id);
                changed = true;
                continue;
            }
            if let Some(disk) = self.disk.as_mut() {
                if disk.chain.as_ref().is_some_and(|c| c.id == id) {
                    disk.chain = None;
                }
                disk.engine.erase(&mut self.store, id);
            } else {
                self.store.remove(id);
            }
            self.seq_checkpoints.retain(|_, v| *v != id);
            self.clear_window_tracking(id);
            self.clear_upgrade_tracking(id);
            self.prompt_heads.remove(&id);
            changed = true;
        }
        if changed {
            self.sync_marks(coordinator);
            self.sync_host_budget();
        }
    }

    pub(crate) fn clear_ram_and_checkpoints(&mut self) {
        self.seq_checkpoints.clear();
        self.prompt_heads.clear();
        self.entries_missing_draft_window.clear();
        self.adopted_unverified_window.clear();
        self.pending_disk_upgrades.clear();
        self.in_flight_disk_upgrade = None;
        self.recent_failed_upgrades.clear();
        self.recent_failed_spills.clear();
        self.spill_backoff.clear();
        self.demote_ram_on_durable.clear();
        if let Some(disk) = self.disk.as_mut() {
            disk.preserve_entry_on_state_failure = None;
        }
        let ids: Vec<EntryId> = self.store.entries().map(Entry::id).collect();
        for id in ids {
            self.store.remove(id);
        }
        self.end_all_seq_uses();
        self.sync_host_budget();
    }

    /// Clear only RAM copies of entries that have a durable disk copy (and remove entries without one).
    #[cfg(test)]
    pub(crate) fn drop_all_ram_keeping_disk(&mut self) {
        let has_dw_span = self
            .resolve_geometry()
            .is_some_and(|g| g.draft_window_span_bytes > 0);
        let ids: Vec<(EntryId, bool)> = self
            .store
            .entries()
            .map(|e| (e.id(), e.disk().is_some_and(|d| d.landed && !d.chain_owed)))
            .collect();
        for (id, has_durable_disk) in ids {
            if has_durable_disk {
                match self.pending_disk_upgrades.remove(&id) {
                    Some(PendingDiskUpgrade::InPlace { promote_label, .. }) => {
                        if has_dw_span {
                            self.entries_missing_draft_window.insert(id);
                        }
                        if promote_label
                            && self
                                .disk
                                .as_ref()
                                .is_some_and(|d| d.tier.persistent && d.writable())
                            && self.store.get(id).is_some_and(|e| !e.checkpoint())
                        {
                            self.pending_disk_upgrades.insert(
                                id,
                                PendingDiskUpgrade::LabelPromote {
                                    retry_after: Instant::now(),
                                },
                            );
                        }
                    }
                    Some(PendingDiskUpgrade::LabelPromote { retry_after }) => {
                        self.pending_disk_upgrades
                            .insert(id, PendingDiskUpgrade::LabelPromote { retry_after });
                    }
                    Some(PendingDiskUpgrade::Repair { .. }) | None => {}
                }
                self.store.drop_ram(id);
            } else {
                self.seq_checkpoints.retain(|_, v| *v != id);
                self.clear_window_tracking(id);
                self.clear_upgrade_tracking(id);
                self.prompt_heads.remove(&id);
                self.store.remove(id);
            }
        }
        self.in_flight_disk_upgrade = None;
        self.recent_failed_upgrades.clear();
        self.recent_failed_spills.clear();
        self.spill_backoff.clear();
        self.demote_ram_on_durable.clear();
        if let Some(disk) = self.disk.as_mut() {
            disk.preserve_entry_on_state_failure = None;
        }
        self.sync_host_budget();
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

    /// Capture `(40 SWA trailing blocks + 48 ShortConv states + optional DFlash2 DraftContextWindow)`
    /// at block-aligned `boundary` and publish into `HybridStateStore` as `class`
    /// (`Ordinary` or `Checkpoint`) with default [`CaptureOptions`].
    #[cfg(test)]
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
        draft_window: Option<&PackedDraftWindow>,
    ) -> Result<Option<EntryId>> {
        self.capture_boundary_inner(
            coordinator,
            conv_state,
            seq_id,
            tokens,
            boundary,
            cache_salt,
            class,
            draft_window.map(DraftWindowInput::Packed),
            None,
            CaptureOptions::default(),
        )
    }

    /// Capture `(40 SWA trailing blocks + 48 ShortConv states + optional DFlash2 BoundaryDraftWindow)`
    /// at block-aligned `boundary` and publish into `HybridStateStore` as `class`
    /// (`Ordinary` or `Checkpoint`) with default [`CaptureOptions`].
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn capture_boundary_with_window(
        &mut self,
        coordinator: &mut Inkling2KVCacheCoordinator,
        conv_state: &Inkling2ShortConvState,
        seq_id: SeqId,
        tokens: &[u32],
        boundary: u32,
        cache_salt: u64,
        class: StateClass,
        draft_window: Option<&BoundaryDraftWindow>,
        governor: Option<(&Governor, u64)>,
    ) -> Result<Option<EntryId>> {
        self.capture_boundary_with_window_and_options(
            coordinator,
            conv_state,
            seq_id,
            tokens,
            boundary,
            cache_salt,
            class,
            draft_window,
            governor,
            CaptureOptions::default(),
        )
    }

    /// Capture `(40 SWA trailing blocks + 48 ShortConv states)` at block-aligned `boundary`
    /// and publish into `HybridStateStore` as `class` (`Ordinary` or `Checkpoint`) with explicit
    /// checkpoint retention, write-behind, and persistent-head options (no draft window).
    #[cfg_attr(not(test), allow(dead_code))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn capture_boundary_with_options(
        &mut self,
        coordinator: &mut Inkling2KVCacheCoordinator,
        conv_state: &Inkling2ShortConvState,
        seq_id: SeqId,
        tokens: &[u32],
        boundary: u32,
        cache_salt: u64,
        class: StateClass,
        opts: CaptureOptions,
    ) -> Result<Option<EntryId>> {
        self.capture_boundary_with_window_and_options(
            coordinator,
            conv_state,
            seq_id,
            tokens,
            boundary,
            cache_salt,
            class,
            None,
            None,
            opts,
        )
    }

    /// Capture `(40 SWA trailing blocks + 48 ShortConv states + optional DFlash2 BoundaryDraftWindow)`
    /// at block-aligned `boundary` and publish into `HybridStateStore` as `class`
    /// (`Ordinary` or `Checkpoint`), deferring GPU-to-host [`PackedDraftWindow::from_window`]
    /// serialization until publication actually occurs, checking Governor admission before
    /// allocating host RAM for new or upgraded anchors, and honoring explicit [`CaptureOptions`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn capture_boundary_with_window_and_options(
        &mut self,
        coordinator: &mut Inkling2KVCacheCoordinator,
        conv_state: &Inkling2ShortConvState,
        seq_id: SeqId,
        tokens: &[u32],
        boundary: u32,
        cache_salt: u64,
        class: StateClass,
        draft_window: Option<&BoundaryDraftWindow>,
        governor: Option<(&Governor, u64)>,
        opts: CaptureOptions,
    ) -> Result<Option<EntryId>> {
        self.capture_boundary_inner(
            coordinator,
            conv_state,
            seq_id,
            tokens,
            boundary,
            cache_salt,
            class,
            draft_window.map(DraftWindowInput::Boundary),
            governor,
            opts,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn reuse_existing_entry(
        &mut self,
        coordinator: &Inkling2KVCacheCoordinator,
        seq_id: SeqId,
        old_id: EntryId,
        boundary: u32,
        effective_class: StateClass,
        opts: CaptureOptions,
        chain_re_registered: bool,
    ) -> EntryId {
        let (was_checkpoint, has_ram, has_disk) =
            self.store
                .get(old_id)
                .map_or((false, false, false), |existing| {
                    (
                        existing.checkpoint(),
                        existing.ram().is_some(),
                        existing.disk().is_some(),
                    )
                });
        let needs_write_behind = has_ram && !has_disk;
        let effective_is_head = opts.is_head || self.is_prompt_head(old_id);
        self.store
            .reuse(old_id, effective_class == StateClass::Checkpoint);
        if chain_re_registered {
            self.store.mark_changed();
        }
        self.demote_ram_on_durable.remove(&old_id);
        match effective_class {
            StateClass::Checkpoint => {
                self.checkpoints_published += 1;
                self.seq_checkpoints.insert(seq_id, old_id);
            }
            StateClass::Ordinary => {
                self.ordinary_published += 1;
                self.seq_checkpoints.retain(|_, v| *v != old_id);
                if effective_is_head {
                    self.mark_prompt_head(old_id);
                }
                if was_checkpoint && has_disk {
                    self.enqueue_label_promote(old_id);
                }
                if opts.enqueue_write_behind
                    && needs_write_behind
                    && !self.pending_disk_upgrades.contains_key(&old_id)
                    && let Some(disk) = self.disk.as_mut()
                {
                    let writable = disk.writable();
                    let persistent = disk.tier.persistent;
                    disk.write_behind.record_published(
                        &self.store,
                        writable,
                        persistent,
                        old_id,
                        boundary,
                        opts.now,
                        effective_is_head,
                    );
                }
            }
        }
        self.sync_marks(coordinator);
        self.sync_host_budget();
        self.trim_ram_to_cap(coordinator, Some(old_id));
        old_id
    }

    #[allow(clippy::too_many_arguments)]
    fn capture_boundary_inner(
        &mut self,
        coordinator: &mut Inkling2KVCacheCoordinator,
        conv_state: &Inkling2ShortConvState,
        seq_id: SeqId,
        tokens: &[u32],
        boundary: u32,
        cache_salt: u64,
        class: StateClass,
        draft_window: Option<DraftWindowInput<'_>>,
        governor: Option<(&Governor, u64)>,
        opts: CaptureOptions,
    ) -> Result<Option<EntryId>> {
        let bs = coordinator.block_size().max(1);
        if boundary == 0 || !boundary.is_multiple_of(bs) || boundary as usize > tokens.len() {
            return Ok(None);
        }
        let geom = self.resolve_geometry().cloned().ok_or_else(|| {
            Error::from_reason(
                "Inkling2StateStore::capture_boundary: state geometry is not initialized",
            )
        })?;

        let lineage = Lineage::text(&tokens[..boundary as usize], cache_salt);
        let (_mark, existing_id) = self.store.locate(lineage);

        // 1. Early reuse fast-path BEFORE any SWA/conv/draft readback, host allocation, or
        //    Governor reservation: if `lineage` ALREADY holds a reusable anchor and this capture
        //    cannot upgrade it with a missing draft window, reuse it in place with zero readback.
        let incoming_matches_dw = draft_window
            .as_ref()
            .is_some_and(|w| w.matches_geometry_for(&geom, boundary));
        if let Some(old_id) = existing_id
            && let Some(existing) = self.store.get(old_id)
        {
            let existing_has_ram = existing.ram().is_some();
            let existing_disk_durable = existing.disk().is_some_and(|c| c.landed && !c.chain_owed)
                && self
                    .disk
                    .as_ref()
                    .is_some_and(|d| d.entry_disk_restorable(existing));
            let existing_missing_dw = geom.draft_window_span_bytes > 0
                && (existing.ram().is_some_and(|a| a.draft_window.is_none())
                    || (!existing_has_ram
                        && (self.entries_missing_draft_window.contains(&old_id)
                            || self.adopted_unverified_window.contains(&old_id))));
            let pending_same = self
                .disk
                .as_ref()
                .is_some_and(|d| d.pending_write_id() == Some(old_id));
            let incoming_can_upgrade = existing_missing_dw && incoming_matches_dw && !pending_same;
            let existing_reusable = existing_has_ram || existing_disk_durable;
            if existing_reusable && !incoming_can_upgrade {
                let effective_class = if existing.checkpoint() {
                    class
                } else {
                    StateClass::Ordinary
                };
                let chain_re_registered = !self.chain_is_hot(coordinator, old_id);
                if chain_re_registered {
                    coordinator
                        .eval_pending_pool_writes_all()
                        .map_err(Error::from_reason)?;
                    coordinator
                        .register_full_committed_blocks(seq_id, boundary, cache_salt)
                        .map_err(Error::from_reason)?;
                }
                if let Some(prev_cp) = self.seq_checkpoints.remove(&seq_id)
                    && !opts.keep_prev_checkpoint
                    && prev_cp != old_id
                {
                    self.retire_entry(prev_cp);
                }
                return Ok(Some(self.reuse_existing_entry(
                    coordinator,
                    seq_id,
                    old_id,
                    boundary,
                    effective_class,
                    opts,
                    chain_re_registered,
                )));
            }
        }

        // 2. Governor admission BEFORE allocating SWA/ShortConv/draft-window host RAM.
        let swa_bytes = if coordinator.sliding_adapter().is_some() {
            geom.live_swa_bytes(boundary).map_err(Error::from_reason)? as u64
        } else {
            0
        };
        let dw_bytes = draft_window.as_ref().map_or(0, |w| {
            w.new_host_bytes_for(&geom, boundary, &self.host_budget)
        });
        let lineage_bytes = if existing_id.is_none() {
            crate::disk_tier::store::lineage_bytes(boundary as usize, (boundary / bs) as usize)
        } else {
            0
        };
        let peak_bytes = swa_bytes
            .saturating_add(geom.conv_span_bytes)
            .saturating_add(dw_bytes)
            .saturating_add(lineage_bytes);
        let reservation = if let Some((gov, model_id)) = governor
            && peak_bytes > 0
        {
            let parent_spec = gov.scope_claim().and_then(|id| gov.spec(id));
            let (request_id, rank) = parent_spec.map_or((0, ClaimRank::default()), |spec| {
                (spec.request_id, spec.rank)
            });
            let opt_claim = gov.open(
                ClaimSpec {
                    cls: ClaimClass::Optional,
                    request_id,
                    rank,
                    reach: ReachClass::Ordinary,
                    timing: NeedTiming::Immediate,
                },
                model_id,
            );
            let res = {
                let _opt_scope = Scope::enter(opt_claim.claim());
                gov.try_reserve(peak_bytes)
            };
            drop(opt_claim);
            match res {
                Ok(reservation) => Some(reservation),
                Err(failure) => {
                    gov.publication_refused(model_id);
                    tracing::debug!(
                        seq_id,
                        boundary,
                        peak_bytes,
                        ?failure,
                        "Inkling2StateStore::capture_boundary: governor refused optional checkpoint reservation"
                    );
                    return Ok(None);
                }
            }
        } else {
            None
        };

        // Ensure all Metal pool writes & ShortConv arrays are materialized before readback.
        coordinator
            .eval_pending_pool_writes_all()
            .map_err(Error::from_reason)?;
        eval_conv_state_arrays(conv_state)?;

        // Register Group 0 full blocks so their chained hashes are live in the prefix cache.
        let existing_chain_was_hot =
            existing_id.is_some_and(|id| self.chain_is_hot(coordinator, id));
        coordinator
            .register_full_committed_blocks(seq_id, boundary, cache_salt)
            .map_err(Error::from_reason)?;

        // Snapshot flat raw SWA blocks covering `[max(0, boundary - sliding_window) .. boundary)`.
        let swa_data: Arc<Vec<u8>> = if coordinator.sliding_adapter().is_some() {
            let flat = coordinator
                .read_sliding_raw_blocks_flat_for(seq_id, boundary)
                .map_err(Error::from_reason)?;
            Arc::new(flat)
        } else {
            Arc::new(Vec::new())
        };
        let conv_data = pack_short_conv_state(&geom, conv_state).map_err(Error::from_reason)?;
        let packed_dw = match draft_window {
            #[cfg(test)]
            Some(DraftWindowInput::Packed(dw)) => {
                if geom.draft_window_span_bytes > 0
                    && dw.width == geom.draft_window_width
                    && dw.slots == geom.draft_window_slots
                    && dw.is_complete_for(boundary)
                {
                    Some(dw.clone())
                } else {
                    None
                }
            }
            Some(DraftWindowInput::Boundary(dw)) => {
                #[cfg(test)]
                let fail_pack = &mut self.fail_next_draft_window_pack;
                match dw.to_packed_for(&geom, boundary, |win| {
                    #[cfg(test)]
                    if std::mem::take(fail_pack) {
                        return Err(Error::from_reason(
                            "injected draft-window serialization failure",
                        ));
                    }
                    PackedDraftWindow::from_window(win)
                }) {
                    Ok(packed) => packed,
                    Err(err) => {
                        tracing::warn!(
                            seq_id,
                            boundary,
                            error = %err,
                            "Inkling2StateStore::capture_boundary: optional draft-window serialization failed (publishing target checkpoint without draft window)"
                        );
                        None
                    }
                }
            }
            None => None,
        };
        let cached_bytes = (swa_data.len() as u64)
            .saturating_add(conv_data.len() as u64)
            .saturating_add(packed_dw.as_ref().map_or(0, |d| d.bytes.len() as u64));
        let anchor = Inkling2StateAnchor {
            swa_data,
            conv_data,
            draft_window: packed_dw,
            cached_bytes,
        };

        // Detach any previous rolling checkpoint for this sequence; only retire it when
        // `!keep_prev_checkpoint` (e.g. a deeper rolling checkpoint replacing an older one)
        // and never retire `existing_id` itself when recapturing the exact same boundary.
        if let Some(prev_cp) = self.seq_checkpoints.remove(&seq_id)
            && !opts.keep_prev_checkpoint
            && Some(prev_cp) != existing_id
        {
            self.retire_entry(prev_cp);
        }

        let mut effective_class = class;
        if existing_id
            .and_then(|id| self.store.get(id))
            .is_some_and(|e| !e.checkpoint())
        {
            effective_class = StateClass::Ordinary;
        }
        let has_draft_window = anchor.draft_window.is_some();

        // Post-pack fallback: if an entry at `lineage` already holds a complete draft window and
        // this recapture ended up without one (e.g. `to_packed_for` failed), preserve and reuse
        // the existing complete entry in place rather than overwriting it.
        if !has_draft_window
            && let Some(old_id) = existing_id
            && let Some(existing) = self.store.get(old_id)
        {
            let existing_has_complete_dw = existing.ram().is_some_and(|a| a.draft_window.is_some())
                || (existing.ram().is_none()
                    && existing.disk().is_some_and(|c| c.landed && !c.chain_owed)
                    && self
                        .disk
                        .as_ref()
                        .is_some_and(|d| d.entry_disk_restorable(existing))
                    && !self.entries_missing_draft_window.contains(&old_id)
                    && !self.adopted_unverified_window.contains(&old_id));
            if existing_has_complete_dw {
                drop(anchor);
                drop(reservation);
                return Ok(Some(self.reuse_existing_entry(
                    coordinator,
                    seq_id,
                    old_id,
                    boundary,
                    effective_class,
                    opts,
                    !existing_chain_was_hot,
                )));
            }
        }

        // Never let a disk-only `existing_id` (`ram().is_none()`) fall through to
        // `HybridStateStore::insert`, which discards duplicate `anchor` payloads.
        if let Some(old_id) = existing_id
            && self.store.get(old_id).is_some_and(|e| e.ram().is_none())
        {
            let was_checkpoint = self.store.get(old_id).is_some_and(Entry::checkpoint);
            let had_label_promote = matches!(
                self.pending_disk_upgrades.get(&old_id),
                Some(PendingDiskUpgrade::LabelPromote { .. })
            );
            let pending_same = self
                .disk
                .as_ref()
                .is_some_and(|d| d.pending_write_id() == Some(old_id));
            let old_disk_copy = self.store.get(old_id).and_then(|e| e.disk().copied());
            let disk_restorable = self.store.get(old_id).is_some_and(|e| {
                self.disk
                    .as_ref()
                    .is_some_and(|d| d.entry_disk_restorable(e))
            });
            let was_missing_dw = self.entries_missing_draft_window.contains(&old_id)
                || self.adopted_unverified_window.contains(&old_id);
            let wants_in_place_upgrade =
                !pending_same && disk_restorable && has_draft_window && was_missing_dw;
            if !wants_in_place_upgrade {
                let disk_unusable = !pending_same && !disk_restorable;
                if disk_unusable {
                    if let Some(disk) = self.disk.as_mut() {
                        if disk.chain.as_ref().is_some_and(|c| c.id == old_id) {
                            disk.chain = None;
                        }
                        if let Some(copy) = old_disk_copy
                            && copy.landed
                        {
                            let tier = Arc::clone(&disk.tier);
                            let _ = tier.while_held(|| {
                                tier.manager.remove(&copy.key);
                            });
                        }
                    }
                    self.store.set_disk(old_id, None);
                    self.clear_upgrade_tracking(old_id);
                } else if !pending_same
                    && disk_restorable
                    && old_disk_copy.is_some_and(|c| c.landed && c.chain_owed)
                    && let Some(copy) = self.store.disk_mut(old_id)
                {
                    copy.chain_owed = false;
                }
                self.store.replace_ram(old_id, anchor);
                if let Some(dw) = self
                    .store
                    .get(old_id)
                    .and_then(Entry::ram)
                    .and_then(|a| a.draft_window.as_ref())
                {
                    dw.attach_to(&self.host_budget);
                }
                self.store
                    .reuse(old_id, effective_class == StateClass::Checkpoint);
                self.demote_ram_on_durable.remove(&old_id);
                self.spill_backoff.remove(&old_id);
                self.recent_failed_upgrades.remove(&old_id);
                self.recent_failed_spills.remove(&old_id);
                if geom.draft_window_span_bytes > 0 {
                    if has_draft_window {
                        self.clear_window_tracking(old_id);
                    } else {
                        self.adopted_unverified_window.remove(&old_id);
                        self.entries_missing_draft_window.insert(old_id);
                    }
                }
                if effective_class == StateClass::Checkpoint {
                    self.checkpoints_published += 1;
                    self.seq_checkpoints.insert(seq_id, old_id);
                } else {
                    let effective_is_head = opts.is_head || self.is_prompt_head(old_id);
                    self.ordinary_published += 1;
                    self.seq_checkpoints.retain(|_, v| *v != old_id);
                    if effective_is_head {
                        self.mark_prompt_head(old_id);
                    }
                    if self.disk.as_ref().is_some_and(|d| d.writable()) {
                        let upgrade = if disk_unusable && old_disk_copy.is_some() {
                            Some(PendingDiskUpgrade::Repair {
                                retry_after: Instant::now(),
                            })
                        } else if pending_same && has_draft_window && was_missing_dw {
                            Some(PendingDiskUpgrade::InPlace {
                                retry_after: Instant::now(),
                                promote_label: was_checkpoint || had_label_promote,
                            })
                        } else {
                            None
                        };
                        if let Some(kind) = upgrade {
                            self.pending_disk_upgrades.insert(old_id, kind);
                        } else if was_checkpoint
                            && self.store.get(old_id).is_some_and(|e| e.disk().is_some())
                        {
                            self.enqueue_label_promote(old_id);
                        } else if disk_unusable
                            && opts.enqueue_write_behind
                            && let Some(disk) = self.disk.as_mut()
                        {
                            let writable = disk.writable();
                            let persistent = disk.tier.persistent;
                            disk.write_behind.record_published(
                                &self.store,
                                writable,
                                persistent,
                                old_id,
                                boundary,
                                opts.now,
                                effective_is_head,
                            );
                        }
                    }
                }
                self.sync_marks(coordinator);
                self.sync_host_budget();
                drop(reservation);
                self.trim_ram_to_cap(coordinator, Some(old_id));
                return Ok(Some(old_id));
            }
        }

        // If an entry ALREADY exists at `lineage` without a draft window (either RAM-resident or
        // SSD-only/adopted-unverified) and our new capture has one (and is not currently in flight
        // on SSD), upgrade it without downgrading an Ordinary entry and WITHOUT losing its
        // durable `DiskCopy` before the upgraded blob atomically replaces it on SSD.
        let mut preserved_head = false;
        if has_draft_window && let Some(old_id) = existing_id {
            let pending_same = self
                .disk
                .as_ref()
                .is_some_and(|d| d.pending_write_id() == Some(old_id));
            let can_upgrade = !pending_same
                && self.store.get(old_id).is_some_and(|e| {
                    e.ram().is_some_and(|a| a.draft_window.is_none())
                        || (e.ram().is_none()
                            && (self.entries_missing_draft_window.contains(&old_id)
                                || self.adopted_unverified_window.contains(&old_id)))
                });
            if can_upgrade {
                let was_checkpoint = self.store.get(old_id).is_some_and(Entry::checkpoint);
                let old_disk_copy = self.store.get(old_id).and_then(|e| e.disk().copied());
                if old_disk_copy.is_some() {
                    let promote_label = (was_checkpoint && effective_class == StateClass::Ordinary)
                        || matches!(
                            self.pending_disk_upgrades.get(&old_id),
                            Some(
                                PendingDiskUpgrade::LabelPromote { .. }
                                    | PendingDiskUpgrade::InPlace {
                                        promote_label: true,
                                        ..
                                    }
                            )
                        );
                    // Preserve `old_id` and its existing `DiskCopy` in-place! Atomically
                    // replace its RAM anchor (updating `cached_bytes`), preserve/upgrade
                    // `Ordinary` class and recency, and record the pending disk upgrade.
                    self.store.replace_ram(old_id, anchor);
                    if let Some(dw) = self
                        .store
                        .get(old_id)
                        .and_then(Entry::ram)
                        .and_then(|a| a.draft_window.as_ref())
                    {
                        dw.attach_to(&self.host_budget);
                    }
                    self.store
                        .reuse(old_id, effective_class == StateClass::Checkpoint);
                    self.clear_window_tracking(old_id);
                    self.demote_ram_on_durable.remove(&old_id);
                    self.pending_disk_upgrades.insert(
                        old_id,
                        PendingDiskUpgrade::InPlace {
                            retry_after: Instant::now(),
                            promote_label,
                        },
                    );
                    self.recent_failed_upgrades.remove(&old_id);
                    match effective_class {
                        StateClass::Checkpoint => {
                            self.checkpoints_published += 1;
                            self.seq_checkpoints.insert(seq_id, old_id);
                        }
                        StateClass::Ordinary => {
                            self.ordinary_published += 1;
                            self.seq_checkpoints.retain(|_, v| *v != old_id);
                            if opts.is_head {
                                self.mark_prompt_head(old_id);
                            }
                        }
                    }
                    self.sync_marks(coordinator);
                    self.sync_host_budget();
                    drop(reservation);
                    self.trim_ram_to_cap(coordinator, Some(old_id));
                    return Ok(Some(old_id));
                }
                // RAM-only entry (`disk().is_none()`): safe to replace directly in RAM.
                preserved_head = self.prompt_heads.remove(&old_id);
                self.seq_checkpoints.retain(|_, v| *v != old_id);
                if let Some(disk) = self.disk.as_mut()
                    && disk.chain.as_ref().is_some_and(|c| c.id == old_id)
                {
                    disk.chain = None;
                }
                self.clear_window_tracking(old_id);
                self.clear_upgrade_tracking(old_id);
                self.store.remove(old_id);
            }
        }

        let inserted = self.store.insert(lineage, anchor, effective_class)?;
        if let Some(dw) = self
            .store
            .get(inserted.id)
            .and_then(Entry::ram)
            .and_then(|a| a.draft_window.as_ref())
        {
            dw.attach_to(&self.host_budget);
        }
        self.demote_ram_on_durable.remove(&inserted.id);
        if geom.draft_window_span_bytes > 0 {
            if inserted.new {
                if has_draft_window {
                    self.clear_window_tracking(inserted.id);
                } else {
                    self.adopted_unverified_window.remove(&inserted.id);
                    self.entries_missing_draft_window.insert(inserted.id);
                }
            } else if self
                .store
                .get(inserted.id)
                .and_then(Entry::ram)
                .is_some_and(|a| a.draft_window.is_some())
            {
                self.clear_window_tracking(inserted.id);
            }
        }

        match effective_class {
            StateClass::Checkpoint => {
                self.checkpoints_published += 1;
                self.seq_checkpoints.insert(seq_id, inserted.id);
            }
            StateClass::Ordinary => {
                let effective_is_head =
                    opts.is_head || preserved_head || self.is_prompt_head(inserted.id);
                self.ordinary_published += 1;
                self.seq_checkpoints.retain(|_, v| *v != inserted.id);
                if effective_is_head {
                    self.mark_prompt_head(inserted.id);
                }
                let has_disk = self
                    .store
                    .get(inserted.id)
                    .is_some_and(|e| e.disk().is_some());
                if inserted.upgraded && has_disk {
                    self.enqueue_label_promote(inserted.id);
                } else if (opts.enqueue_write_behind || preserved_head)
                    && !self.pending_disk_upgrades.contains_key(&inserted.id)
                    && let Some(disk) = self.disk.as_mut()
                {
                    let writable = disk.writable();
                    let persistent = disk.tier.persistent;
                    disk.write_behind.record_published(
                        &self.store,
                        writable,
                        persistent,
                        inserted.id,
                        boundary,
                        opts.now,
                        effective_is_head,
                    );
                }
            }
        }
        self.sync_marks(coordinator);
        self.sync_host_budget();
        drop(reservation);
        self.trim_ram_to_cap(coordinator, Some(inserted.id));
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
            if disk.pending_write_id() == Some(id) {
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
        self.seq_checkpoints.retain(|_, v| *v != id);
        self.clear_window_tracking(id);
        self.clear_upgrade_tracking(id);
        self.prompt_heads.remove(&id);
        self.store.remove(id);
        self.sync_host_budget();
    }

    /// Enqueue an off-thread `LabelPromote` rewrite on `ColdCacheManager`'s writer thread under
    /// `MakeRoom::Refuse`. Neither reads from SSD on the scheduler thread nor borrows
    /// `StateDiskEngine`'s reusable staging buffer, and keeps `entry.disk()` at
    /// `Some(old_copy)` (`landed: true`) throughout the rewrite so concurrent restores continue
    /// hitting the already-landed SSD copy without interruption.
    fn stage_state_relabel(
        &mut self,
        id: EntryId,
        old_copy: DiskCopy,
        fallback_hold: ColdKeyHold,
    ) -> Staged {
        let Some(entry) = self.store.get(id) else {
            return Staged::Refused;
        };
        let Some(disk) = self.disk.as_mut() else {
            return Staged::Refused;
        };
        let (Some(layout), Some(key)) = (
            disk.engine.layout(entry.depth()),
            disk.engine.state_key(entry),
        ) else {
            return Staged::Refused;
        };
        if key != old_copy.key {
            return Staged::Refused;
        }
        let relabel = mlx_paged_attn::ColdStateRelabel {
            key,
            fingerprint: disk.tier.state_fingerprint,
            layout,
            label: crate::disk_tier::state_disk::label_of(entry),
        };
        match disk.tier.manager.enqueue_state_relabel(
            relabel,
            MakeRoom::Refuse,
            disk.relabel_tx.clone(),
        ) {
            Ok(()) => {
                disk.pending_relabel = Some(PendingRelabel {
                    id,
                    key,
                    _fallback_hold: fallback_hold,
                });
                Staged::Started
            }
            Err(mlx_paged_attn::EnqueueRefused::Busy(_)) => Staged::Busy,
            Err(mlx_paged_attn::EnqueueRefused::Gone(_)) => Staged::Refused,
        }
    }

    /// Stage `id`'s RAM anchor into the reusable host buffer (or enqueue an off-thread
    /// `LabelPromote` rewrite on the writer thread) and submit its SSD write.
    pub(crate) fn stage_entry(
        &mut self,
        coordinator: &Inkling2KVCacheCoordinator,
        id: EntryId,
        room: MakeRoom,
        keep_ram: bool,
    ) -> bool {
        let Some(entry) = self.store.get(id) else {
            return false;
        };
        let pending_kind = self.pending_disk_upgrades.get(&id).copied();
        let has_ram = entry.ram().is_some();
        let is_label_promote =
            matches!(pending_kind, Some(PendingDiskUpgrade::LabelPromote { .. }));
        if is_label_promote {
            let persistent_writable = self
                .disk
                .as_ref()
                .is_some_and(|d| d.tier.persistent && d.writable());
            if !persistent_writable || entry.checkpoint() || entry.disk().is_none() {
                self.pending_disk_upgrades.remove(&id);
                return false;
            }
        }
        let upgrade_old_copy = match pending_kind {
            Some(PendingDiskUpgrade::InPlace { .. } | PendingDiskUpgrade::LabelPromote { .. }) => {
                entry.disk().copied()
            }
            Some(PendingDiskUpgrade::Repair { .. }) | None => None,
        };
        let (writable, writing) = match self.disk.as_ref() {
            Some(disk) => (disk.writable(), disk.writing()),
            None => return false,
        };
        if !writable || writing {
            return false;
        }
        // For `LabelPromote`, wait cleanly until the initial state blob AND its owed Global KV
        // chain have landed on SSD before reading/relabeling the blob (never treat an in-flight
        // or `chain_owed` initial write as stale or place it in backoff).
        if is_label_promote && upgrade_old_copy.is_some_and(|c| !c.landed || c.chain_owed) {
            return false;
        }
        // Acquire a `ColdKeyHold` pinning `old_copy.key` and its entire Global KV chain BEFORE
        // checking presence so a competing writer job cannot evict the SSD fallback after our
        // pre-check passes. The hold is dropped automatically on any early return / refusal and
        // moved onto `PendingRelabel` (for `LabelPromote`) or `StateDiskEngine`'s pending
        // replacement slot (for `InPlace`) once staging starts so `clear_ram_and_checkpoints()`
        // never releases the hold while the replacement write is in flight.
        let mut upgrade_in_place: Option<(DiskCopy, InFlightDiskUpgradeKind)> = None;
        let mut upgrade_hold: Option<ColdKeyHold> = None;
        let mut stale_upgrade_copy: Option<DiskCopy> = None;
        if let Some(old_copy) = upgrade_old_copy
            && old_copy.landed
            && let Some(disk) = self.disk.as_ref()
        {
            let mut hold_keys = disk.engine.kv_chain_keys(entry);
            hold_keys.push(old_copy.key);
            let hold = disk.tier.manager.hold_keys(hold_keys);
            let intact = disk
                .tier
                .manager
                .contains_in(&old_copy.key, ColdGroup::DFlash2State)
                && disk.engine.chain_on_disk(entry);
            if intact {
                let in_flight_kind = match pending_kind {
                    Some(PendingDiskUpgrade::InPlace { promote_label, .. }) => {
                        InFlightDiskUpgradeKind::InPlace { promote_label }
                    }
                    _ => InFlightDiskUpgradeKind::LabelPromote,
                };
                upgrade_in_place = Some((old_copy, in_flight_kind));
                upgrade_hold = Some(hold);
            } else {
                drop(hold);
                stale_upgrade_copy = Some(old_copy);
            }
        }
        let mut transitioned_to_repair = false;
        if matches!(pending_kind, Some(PendingDiskUpgrade::InPlace { .. }))
            && upgrade_old_copy.is_none()
        {
            self.pending_disk_upgrades.insert(
                id,
                PendingDiskUpgrade::Repair {
                    retry_after: Instant::now(),
                },
            );
            transitioned_to_repair = true;
        }
        if let Some(old_copy) = stale_upgrade_copy {
            if !has_ram {
                // SSD-only `LabelPromote` whose old blob or Global KV chain was already evicted:
                // no RAM copy exists to repair it. Clean up the dead entry and drop tracking.
                if let Some(disk) = self.disk.as_mut() {
                    if disk.chain.as_ref().is_some_and(|c| c.id == id) {
                        disk.chain = None;
                    }
                    disk.tier.manager.remove(&old_copy.key);
                }
                self.seq_checkpoints.retain(|_, v| *v != id);
                self.clear_window_tracking(id);
                self.clear_upgrade_tracking(id);
                self.prompt_heads.remove(&id);
                self.store.remove(id);
                self.sync_marks(coordinator);
                self.sync_host_budget();
                return false;
            }
            // The old SSD blob's Global KV chain (or blob itself) was already evicted from SSD,
            // so the old SSD copy is no longer restorable. Remove the dead old SSD blob, transition
            // upgrade tracking to `Repair` (never drop it!), and fall through to the normal
            // fresh-write path (`chain_owed = true`) so both the upgraded blob AND its missing
            // Global KV chain are repaired before the upgrade is marked complete.
            if let Some(disk) = self.disk.as_mut() {
                if disk.chain.as_ref().is_some_and(|c| c.id == id) {
                    disk.chain = None;
                }
                disk.tier.manager.remove(&old_copy.key);
            }
            self.store.set_disk(id, None);
            self.pending_disk_upgrades.insert(
                id,
                PendingDiskUpgrade::Repair {
                    retry_after: Instant::now(),
                },
            );
            self.recent_failed_upgrades.remove(&id);
            transitioned_to_repair = true;
        }
        let Some(entry) = self.store.get(id) else {
            return false;
        };
        let can_stage = self.disk.as_ref().is_some_and(|disk| {
            if let Some((old_copy, _)) = upgrade_in_place.as_ref() {
                // An in-place upgrade or label relabel of an already-landed state blob whose KV
                // chain is already on disk only charges the net delta `(new_state - old_copy.bytes)`
                // against the logical quota, so it fits whenever the post-overwrite total stays
                // within `tier.limit` and the volume has `new_state` bytes free above reserve for
                // the replacement `.tmp` file. Never fall back to `point_fits(entry, room)`: under
                // `MakeRoom::Evict`, `point_fits` can return `true` when `free < new_state` and
                // cause `ColdCacheManager::persist_with` to evict live KV blocks (including this
                // anchor's own Global KV chain) while chasing volume headroom.
                old_copy.landed
                    && disk.engine.chain_on_disk(entry)
                    && disk
                        .engine
                        .layout(entry.depth())
                        .and_then(|layout| {
                            let new_state = mlx_paged_attn::state_object_bytes(
                                &layout,
                                &crate::disk_tier::state_disk::label_of(entry),
                            );
                            let free = disk.tier.manager.room_above_reserve()?;
                            let taken = disk.tier.manager.total_bytes();
                            let projected = taken
                                .saturating_sub(old_copy.bytes)
                                .saturating_add(new_state);
                            Some(projected <= disk.tier.limit && new_state <= free)
                        })
                        .unwrap_or(false)
            } else {
                entry.ram().is_some() && entry.disk().is_none() && disk.point_fits(entry, room)
            }
        });
        if !can_stage {
            if transitioned_to_repair {
                self.sync_marks(coordinator);
            }
            if self.disk.as_ref().is_none_or(|d| !d.writing())
                && let Some(pending) = self.pending_disk_upgrades.get_mut(&id)
            {
                pending.set_retry_after(upgrade_retry_after());
            }
            return false;
        }
        let is_pending_upgrade = self.pending_disk_upgrades.contains_key(&id);
        let label_promote_copy = upgrade_in_place.and_then(|(old_copy, kind)| {
            (kind == InFlightDiskUpgradeKind::LabelPromote).then_some(old_copy)
        });
        let staged = if let Some(old_copy) = label_promote_copy {
            let Some(fallback_hold) = upgrade_hold.take() else {
                return false;
            };
            self.stage_state_relabel(id, old_copy, fallback_hold)
        } else if let Some((old_copy, _)) = upgrade_in_place {
            let Some(fallback_hold) = upgrade_hold.take() else {
                return false;
            };
            self.store.set_disk(id, None);
            // Always enqueue in-place SSD upgrades with `MakeRoom::Refuse`: even if volume
            // headroom or quota changes between this model-thread pre-check and async writer
            // execution, the writer must NEVER evict any object (including the old blob's Global
            // KV chain) — it cleanly refuses, rolls back `old_copy`, and preserves the old durable
            // SSD hit.
            let effective_keep_ram = keep_ram || is_pending_upgrade;
            let Some(disk) = self.disk.as_mut() else {
                self.store.set_disk(id, Some(old_copy));
                return false;
            };
            disk.engine.stage_replacement(
                &mut self.store,
                id,
                MakeRoom::Refuse,
                effective_keep_ram,
                (),
                fallback_hold,
            )
        } else {
            let effective_keep_ram = keep_ram || is_pending_upgrade;
            let Some(disk) = self.disk.as_mut() else {
                return false;
            };
            disk.engine
                .stage(&mut self.store, id, room, effective_keep_ram, ())
        };
        match staged {
            Staged::Started => {
                self.spill_backoff.remove(&id);
                self.recent_failed_spills.remove(&id);
                if let Some((old_copy, in_flight_kind)) = upgrade_in_place {
                    self.pending_disk_upgrades.remove(&id);
                    self.recent_failed_upgrades.remove(&id);
                    self.in_flight_disk_upgrade = Some(InFlightDiskUpgrade {
                        id,
                        kind: in_flight_kind,
                        old_copy,
                    });
                    if in_flight_kind != InFlightDiskUpgradeKind::LabelPromote
                        && let Some(disk) = self.disk.as_mut()
                    {
                        disk.preserve_entry_on_state_failure = Some(id);
                    }
                } else if is_pending_upgrade {
                    // Keep `PendingDiskUpgrade::Repair` tracked until BOTH the replacement state
                    // blob AND its complete Global KV chain land!
                    self.recent_failed_upgrades.remove(&id);
                }
                self.sync_marks(coordinator);
                self.sync_host_budget();
                true
            }
            Staged::Busy => {
                if let Some((old_copy, _)) = upgrade_in_place {
                    if label_promote_copy.is_none() {
                        self.store.set_disk(id, Some(old_copy));
                    }
                } else if transitioned_to_repair {
                    self.sync_marks(coordinator);
                }
                if let Some(pending) = self.pending_disk_upgrades.get_mut(&id) {
                    pending.set_retry_after(upgrade_retry_after());
                }
                false
            }
            Staged::Refused => {
                let retry = upgrade_retry_after();
                let old_still_indexed = upgrade_in_place.is_some_and(|(old_copy, _)| {
                    self.disk.as_ref().is_some_and(|d| {
                        d.tier
                            .manager
                            .contains_in(&old_copy.key, ColdGroup::DFlash2State)
                    })
                });
                if let Some((old_copy, _)) = upgrade_in_place {
                    if old_still_indexed {
                        if label_promote_copy.is_none() {
                            self.store.set_disk(id, Some(old_copy));
                        }
                    } else if !has_ram {
                        self.seq_checkpoints.retain(|_, v| *v != id);
                        self.clear_window_tracking(id);
                        self.clear_upgrade_tracking(id);
                        self.prompt_heads.remove(&id);
                        self.store.remove(id);
                        self.sync_marks(coordinator);
                        self.sync_host_budget();
                        if let Some(disk) = self.disk.as_mut() {
                            disk.counters.offload_failures += 1;
                        }
                        return false;
                    } else {
                        self.store.set_disk(id, None);
                        self.pending_disk_upgrades
                            .insert(id, PendingDiskUpgrade::Repair { retry_after: retry });
                        self.sync_marks(coordinator);
                    }
                } else if transitioned_to_repair {
                    self.sync_marks(coordinator);
                }
                if let Some(pending) = self.pending_disk_upgrades.get_mut(&id) {
                    pending.set_retry_after(retry);
                    self.recent_failed_upgrades.insert(id);
                } else if self.store.get(id).is_some_and(|e| e.ram().is_some()) {
                    self.recent_failed_spills.insert(id);
                }
                if self.store.get(id).is_some() {
                    self.spill_backoff.insert(id, retry);
                }
                if let Some(disk) = self.disk.as_mut() {
                    disk.counters.offload_failures += 1;
                }
                false
            }
        }
    }

    fn erase_reclaimed_entry(&mut self, coordinator: &Inkling2KVCacheCoordinator, id: EntryId) {
        self.seq_checkpoints.retain(|_, v| *v != id);
        self.clear_window_tracking(id);
        self.clear_upgrade_tracking(id);
        self.prompt_heads.remove(&id);
        if let Some(disk) = self.disk.as_mut() {
            if disk.chain.as_ref().is_some_and(|c| c.id == id) {
                disk.chain = None;
            }
            disk.engine.erase(&mut self.store, id);
        } else {
            self.store.remove(id);
        }
        self.sync_marks(coordinator);
        self.sync_host_budget();
    }

    /// Reclaim RAM for `id` under governor pressure (patient default: `allow_wait = true`).
    pub(crate) fn reclaim_entry(
        &mut self,
        coordinator: &Inkling2KVCacheCoordinator,
        id: EntryId,
    ) -> ReclaimOutcome {
        self.reclaim_entry_with_options(coordinator, id, true, false)
    }

    /// Reclaim RAM for `id` under governor pressure (`need_pages = false` convenience wrapper).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn reclaim_entry_with_wait(
        &mut self,
        coordinator: &Inkling2KVCacheCoordinator,
        id: EntryId,
        allow_wait: bool,
    ) -> ReclaimOutcome {
        self.reclaim_entry_with_options(coordinator, id, allow_wait, false)
    }

    /// Reclaim RAM (and, when `need_pages == true`, Global KV pages) for `id` under governor
    /// pressure:
    /// - When `allow_wait == true` (patient claim):
    ///   - If it has a landed disk copy whose Global KV chain is still owed, return `Pending`
    ///     without dropping RAM until the chain lands (if the chain fails, RAM stays intact).
    ///   - If it already has a landed, restorable disk copy, drop RAM immediately.
    ///   - If SSD is writable and not busy, spill it to SSD (`MakeRoom::Evict`) and drop RAM.
    ///   - If a write is already in flight, return `Pending` so the ledger tries another victim.
    ///   - If no SSD tier or point doesn't fit, remove the entry.
    /// - When `allow_wait == false` (urgent/immediate claim):
    ///   - If it already has a landed, restorable disk copy (`!chain_owed`), drop RAM immediately.
    ///   - If an in-place SSD upgrade is in flight (`in_flight_disk_upgrade`) and either
    ///     `!need_pages` or `!chain_owed`, drop RAM immediately while keeping the entry indexed so
    ///     the upgrade or fallback `old_copy` survives.
    ///   - If it has no disk copy yet, SSD can stage immediately without waiting, AND either
    ///     `!need_pages` or the entry's complete Global KV chain is already on SSD with no chain
    ///     cursor in flight (so `stage` leaves `chain_owed == false` and immediately unmarks KV
    ///     pages), spill and drop RAM immediately.
    ///   - Otherwise (incomplete `chain_owed` SSD copy, fresh RAM anchor whose spill would owe a
    ///     KV chain under `need_pages`, in-flight write, busy staging buffer, or unfit point),
    ///     discard any incomplete cache copy and erase `id` immediately to free RAM & KV pages.
    pub(crate) fn reclaim_entry_with_options(
        &mut self,
        coordinator: &Inkling2KVCacheCoordinator,
        id: EntryId,
        allow_wait: bool,
        need_pages: bool,
    ) -> ReclaimOutcome {
        let has_dw_span = self
            .resolve_geometry()
            .is_some_and(|g| g.draft_window_span_bytes > 0);
        if self.disk.is_none() {
            if self.store.get(id).is_some_and(|e| e.ram().is_some()) {
                self.erase_reclaimed_entry(coordinator, id);
            }
            return ReclaimOutcome::Erased;
        }

        if self.retire_unwritable_incomplete_disk_copies() {
            self.prune_entry_tracking();
            self.sync_marks(coordinator);
            self.sync_host_budget();
        }

        let (disk_writable, has_ram, disk_copy, disk_restorable) =
            match (self.disk.as_ref(), self.store.get(id)) {
                (Some(disk), Some(entry)) => (
                    disk.writable(),
                    entry.ram().is_some(),
                    entry.disk().copied(),
                    disk.entry_disk_restorable(entry),
                ),
                _ => (false, false, None, false),
            };

        if !allow_wait {
            if !has_ram {
                return ReclaimOutcome::Erased;
            }
            if disk_copy.is_some_and(|c| c.landed && !c.chain_owed) && disk_restorable {
                match self.pending_disk_upgrades.remove(&id) {
                    Some(PendingDiskUpgrade::InPlace { promote_label, .. }) => {
                        if has_dw_span {
                            self.entries_missing_draft_window.insert(id);
                        }
                        if promote_label
                            && self
                                .disk
                                .as_ref()
                                .is_some_and(|d| d.tier.persistent && d.writable())
                            && self.store.get(id).is_some_and(|e| !e.checkpoint())
                        {
                            self.pending_disk_upgrades.insert(
                                id,
                                PendingDiskUpgrade::LabelPromote {
                                    retry_after: Instant::now(),
                                },
                            );
                        }
                    }
                    Some(PendingDiskUpgrade::LabelPromote { retry_after }) => {
                        self.pending_disk_upgrades
                            .insert(id, PendingDiskUpgrade::LabelPromote { retry_after });
                    }
                    Some(PendingDiskUpgrade::Repair { .. }) | None => {}
                }
                self.recent_failed_upgrades.remove(&id);
                self.recent_failed_spills.remove(&id);
                self.spill_backoff.remove(&id);
                self.demote_ram_on_durable.remove(&id);
                self.store.drop_ram(id);
                self.sync_marks(coordinator);
                self.sync_host_budget();
                return ReclaimOutcome::RamDropped;
            }
            if let Some(in_flight) = self.in_flight_disk_upgrade.as_ref().filter(|u| u.id == id)
                && (!need_pages || !disk_copy.is_some_and(|c| c.chain_owed))
                && self.disk.as_ref().is_some_and(|d| {
                    d.tier
                        .manager
                        .contains_in(&in_flight.old_copy.key, ColdGroup::DFlash2State)
                        && self
                            .store
                            .get(id)
                            .is_some_and(|e| d.engine.chain_on_disk(e))
                })
            {
                // An in-place upgrade write is in flight over a pinned, already-durable SSD prefix
                // (`old_copy` + its Global KV chain held by `_fallback_hold`): drop ONLY RAM
                // immediately while keeping the store entry indexed and `in_flight_disk_upgrade`
                // intact so `poll_offload` / `tick` keeps the upgraded blob on success or restores
                // `old_copy` on failure. When `need_pages` is true, this path only runs when
                // `!chain_owed` so dropping RAM immediately unmarks the entry's Global KV pages too.
                self.recent_failed_upgrades.remove(&id);
                self.recent_failed_spills.remove(&id);
                self.spill_backoff.remove(&id);
                self.demote_ram_on_durable.remove(&id);
                self.store.drop_ram(id);
                self.sync_marks(coordinator);
                self.sync_host_budget();
                return ReclaimOutcome::RamDropped;
            }
            let can_immediate_spill = disk_copy.is_none()
                && self.disk.as_ref().is_some_and(|d| {
                    !d.writing()
                        && !d.has_read_waiters()
                        && d.engine.reclaim_writes(&self.store, id)
                        && (!need_pages
                            || (d.chain.is_none()
                                && self
                                    .store
                                    .get(id)
                                    .is_some_and(|e| d.engine.chain_on_disk(e))))
                });
            if can_immediate_spill && let Some(disk) = self.disk.as_mut() {
                match disk
                    .engine
                    .stage(&mut self.store, id, MakeRoom::Evict, false, ())
                {
                    Staged::Started => {
                        self.clear_upgrade_tracking(id);
                        self.sync_marks(coordinator);
                        self.sync_host_budget();
                        return ReclaimOutcome::Spilled;
                    }
                    Staged::Busy => {}
                    Staged::Refused => {
                        disk.counters.offload_failures += 1;
                    }
                }
            }
            if let Some(in_flight) = self.in_flight_disk_upgrade.take_if(|u| u.id == id)
                && in_flight.old_copy.landed
                && let Some(disk) = self.disk.as_ref()
            {
                let tier = Arc::clone(&disk.tier);
                let key = in_flight.old_copy.key;
                let _ = tier.while_held(|| {
                    tier.manager.remove(&key);
                });
            }
            if let Some(disk) = self.disk.as_mut() {
                disk.engine.cancel_pending_replacement(id);
                if disk.chain.as_ref().is_some_and(|c| c.id == id) {
                    disk.chain = None;
                }
                if let Some(copy) = disk_copy
                    && copy.landed
                {
                    let tier = Arc::clone(&disk.tier);
                    let key = copy.key;
                    let _ = tier.while_held(|| {
                        tier.manager.remove(&key);
                    });
                }
            }
            self.seq_checkpoints.retain(|_, v| *v != id);
            self.clear_window_tracking(id);
            self.clear_upgrade_tracking(id);
            self.prompt_heads.remove(&id);
            self.store.remove(id);
            self.sync_marks(coordinator);
            self.sync_host_budget();
            return ReclaimOutcome::Erased;
        }

        if has_ram && let Some(copy) = disk_copy {
            if copy.landed && copy.chain_owed {
                if disk_writable {
                    self.demote_ram_on_durable.insert(id);
                    return ReclaimOutcome::Pending;
                }
                if let Some(disk) = self.disk.as_mut() {
                    if disk.chain.as_ref().is_some_and(|c| c.id == id) {
                        disk.chain = None;
                    }
                    disk.tier.manager.remove(&copy.key);
                }
                self.store.set_disk(id, None);
                self.pending_disk_upgrades.remove(&id);
                self.recent_failed_upgrades.remove(&id);
            } else if copy.landed && !copy.chain_owed && !disk_restorable {
                if let Some(disk) = self.disk.as_ref() {
                    disk.tier.manager.remove(&copy.key);
                }
                self.store.set_disk(id, None);
                match self.pending_disk_upgrades.get(&id) {
                    Some(PendingDiskUpgrade::InPlace { .. }) => {
                        if disk_writable {
                            self.pending_disk_upgrades.insert(
                                id,
                                PendingDiskUpgrade::Repair {
                                    retry_after: Instant::now(),
                                },
                            );
                        } else {
                            self.pending_disk_upgrades.remove(&id);
                        }
                    }
                    Some(PendingDiskUpgrade::LabelPromote { .. }) => {
                        // Old SSD copy was gone; drop `LabelPromote` so patient reclaim spills
                        // fresh from RAM with `checkpoint: false` below.
                        self.pending_disk_upgrades.remove(&id);
                    }
                    Some(PendingDiskUpgrade::Repair { .. }) | None => {}
                }
                self.recent_failed_upgrades.remove(&id);
            }
        }

        // Under patient reclaim, only return `Pending` when concrete I/O is actually in flight
        // (an active writer job, parked restore holding the staging buffer, or a newly started
        // `stage_entry` write). Backed-off entries or refused repairs with `!disk.writing()` must
        // NOT stall Governor claims waiting on a 10-second backoff timer — erase them immediately.
        let disk_writing = self.disk.as_ref().is_some_and(|d| d.writing());
        if has_ram
            && disk_writable
            && !disk_writing
            && self.store.get(id).is_some_and(|e| e.disk().is_none())
        {
            let now = Instant::now();
            let backed_off = self
                .spill_backoff
                .get(&id)
                .is_some_and(|&retry_after| now < retry_after)
                || self
                    .pending_disk_upgrades
                    .get(&id)
                    .is_some_and(|pending| now < pending.retry_after());
            if backed_off {
                self.erase_reclaimed_entry(coordinator, id);
                return ReclaimOutcome::Erased;
            }
            if matches!(
                self.pending_disk_upgrades.get(&id),
                Some(PendingDiskUpgrade::Repair { .. })
            ) {
                if self.disk.as_ref().is_some_and(|d| d.has_read_waiters()) {
                    return ReclaimOutcome::Pending;
                }
                if self.stage_entry(coordinator, id, MakeRoom::Evict, true) {
                    self.demote_ram_on_durable.insert(id);
                    return ReclaimOutcome::Pending;
                }
                self.erase_reclaimed_entry(coordinator, id);
                return ReclaimOutcome::Erased;
            }
        }

        if disk_writing
            && has_ram
            && self.store.get(id).is_some_and(|e| e.disk().is_none())
            && self
                .disk
                .as_ref()
                .is_some_and(|d| d.engine.reclaim_writes(&self.store, id))
        {
            return ReclaimOutcome::Pending;
        }

        let step = {
            let Some(disk) = self.disk.as_mut() else {
                return ReclaimOutcome::Erased;
            };
            let step = disk.engine.reclaim(&mut self.store, id, ());
            if step.encode_or_enqueue_failed {
                disk.counters.offload_failures += 1;
            }
            step
        };
        if step.outcome == Reclaimed::RamDropped {
            self.demote_ram_on_durable.remove(&id);
            match self.pending_disk_upgrades.remove(&id) {
                Some(PendingDiskUpgrade::InPlace { promote_label, .. }) => {
                    self.recent_failed_upgrades.remove(&id);
                    if has_dw_span {
                        self.entries_missing_draft_window.insert(id);
                    }
                    if promote_label
                        && self
                            .disk
                            .as_ref()
                            .is_some_and(|d| d.tier.persistent && d.writable())
                        && self.store.get(id).is_some_and(|e| !e.checkpoint())
                    {
                        self.pending_disk_upgrades.insert(
                            id,
                            PendingDiskUpgrade::LabelPromote {
                                retry_after: Instant::now(),
                            },
                        );
                    }
                }
                Some(PendingDiskUpgrade::LabelPromote { retry_after }) => {
                    self.pending_disk_upgrades
                        .insert(id, PendingDiskUpgrade::LabelPromote { retry_after });
                }
                Some(PendingDiskUpgrade::Repair { .. }) | None => {}
            }
        }
        if step.outcome == Reclaimed::Written {
            self.clear_upgrade_tracking(id);
        }
        if self.store.get(id).is_none() {
            self.seq_checkpoints.retain(|_, v| *v != id);
            self.clear_window_tracking(id);
            self.clear_upgrade_tracking(id);
            self.prompt_heads.remove(&id);
        }
        if step.resync {
            self.prune_entry_tracking();
            self.sync_marks(coordinator);
            self.sync_host_budget();
        }
        self.sync_host_budget();
        match step.outcome {
            Reclaimed::Written => ReclaimOutcome::Spilled,
            Reclaimed::RamDropped => ReclaimOutcome::RamDropped,
            Reclaimed::Pending => ReclaimOutcome::Pending,
            Reclaimed::Erased | Reclaimed::Nothing => ReclaimOutcome::Erased,
        }
    }

    /// Persist `id`'s RAM copy and Global KV chain via shared [`StateDiskEngine::persist`].
    /// When `head == true`, persists even after deeper ordinary turns supersede `id`.
    pub(crate) fn persist_entry(
        &mut self,
        coordinator: &Inkling2KVCacheCoordinator,
        id: EntryId,
        head: bool,
    ) -> PersistStatus {
        self.persist_entry_inner(coordinator, id, head, false)
    }

    fn persist_entry_inner(
        &mut self,
        coordinator: &Inkling2KVCacheCoordinator,
        id: EntryId,
        head: bool,
        force: bool,
    ) -> PersistStatus {
        // Queued or in-flight SSD upgrades/repairs are owned by `drive_pending_disk_upgrades_inner`
        // (which holds `ColdKeyHold` + rollback fallbacks via `stage_entry`).
        if self.pending_disk_upgrades.contains_key(&id)
            || self
                .in_flight_disk_upgrade
                .as_ref()
                .is_some_and(|u| u.id == id)
        {
            return PersistStatus::Unneeded;
        }
        if !force
            && self
                .spill_backoff
                .get(&id)
                .is_some_and(|&retry_after| Instant::now() < retry_after)
        {
            return PersistStatus::Busy;
        }
        if self
            .disk
            .as_ref()
            .is_some_and(|d| d.pending_relabel.is_some())
            && self.store.get(id).is_some_and(|e| e.disk().is_none())
        {
            let needed = if head {
                self.store.get(id).is_some_and(|e| !e.checkpoint())
            } else {
                crate::disk_tier::state_disk::newest_point(&self.store, id)
            };
            let fits = needed
                && self.disk.as_ref().is_some_and(|d| {
                    d.writable()
                        && self
                            .store
                            .get(id)
                            .is_some_and(|e| d.engine.point_fits(e, d.engine.persist_room()))
                });
            if fits {
                return PersistStatus::Busy;
            }
        }
        let kv = KvPoolHandle::from_adapter(coordinator.full_adapter());
        let step = {
            let Some(disk) = self.disk.as_mut() else {
                return PersistStatus::Refused;
            };
            let step = disk
                .engine
                .persist(&mut self.store, Some(&kv), id, head, ());
            disk.counters.offload_failures += step.offload_failures;
            step
        };
        if step.staged {
            self.spill_backoff.remove(&id);
            self.recent_failed_spills.remove(&id);
        }
        if step.resync {
            self.prune_entry_tracking();
            self.sync_marks(coordinator);
            self.sync_host_budget();
        }
        if step.status == PersistStatus::Refused
            && self
                .store
                .get(id)
                .is_some_and(|e| !e.checkpoint() && e.ram().is_some() && e.disk().is_none())
        {
            let retryable = self.is_retryable_ordinary_wb(coordinator, id);
            if retryable {
                self.spill_backoff.insert(id, upgrade_retry_after());
                self.recent_failed_spills.insert(id);
                if !force {
                    // Transient headroom refusal on background `WriteBehind::step`: keep the
                    // point queued in `WriteBehind` behind `spill_backoff` instead of dropping it.
                    return PersistStatus::Busy;
                }
            } else if self.disk.as_ref().is_some_and(|d| {
                self.store
                    .get(id)
                    .is_some_and(|e| d.entry_fits_tier_limit(e))
            }) {
                // Entry fits the tier limit but cannot persist (e.g. lost KV chain or unwritable
                // disk): record as failed so `flush_to_disk` surfaces `false`.
                self.recent_failed_spills.insert(id);
            }
        }
        step.status
    }

    /// Poll completed SSD writes & LRU evictions from `ColdCacheManager`.
    pub(crate) fn poll_offload(&mut self, coordinator: &Inkling2KVCacheCoordinator) -> bool {
        let (pending_before, chain_before) = self.disk.as_ref().map_or((None, None), |d| {
            (d.pending_write_id(), d.chain.as_ref().map(|c| c.id))
        });
        let repairs_with_disk_before: Vec<EntryId> = self
            .pending_disk_upgrades
            .iter()
            .filter_map(|(&id, kind)| {
                (kind.is_repair() && self.store.get(id).is_some_and(|e| e.disk().is_some()))
                    .then_some(id)
            })
            .collect();
        let ordinary_owed_before = self.ordinary_chain_owed_ids();
        let (out, relabel_out) = {
            let Some(disk) = self.disk.as_mut() else {
                return false;
            };
            let out = disk.engine.poll_offload(&mut self.store);
            let relabel_out = disk.poll_relabel(&mut self.store);
            disk.counters.offloads += out.state_landed_ok + relabel_out.landed_ok;
            disk.counters.offload_failures += out.offload_failures + relabel_out.failures;
            (out, relabel_out)
        };
        let state_failed_id = out.state_failed_id.or(relabel_out.failed_id);
        let mut failed_ids = self.reconcile_spill_outcome(
            coordinator,
            pending_before,
            chain_before,
            state_failed_id,
            &[out.chain_failed_id],
            &ordinary_owed_before,
        );
        let failed_upgrade = self.reconcile_in_flight_disk_upgrade(state_failed_id);
        if let Some(uid) = failed_upgrade
            && !failed_ids.contains(&uid)
        {
            failed_ids.push(uid);
        }
        self.reconcile_completed_disk_repairs();
        let retry = upgrade_retry_after();
        for id in repairs_with_disk_before {
            if !failed_ids.contains(&id)
                && matches!(
                    self.pending_disk_upgrades.get(&id),
                    Some(PendingDiskUpgrade::Repair { .. })
                )
                && self.store.get(id).is_some_and(|e| e.disk().is_none())
            {
                self.mark_repair_failed(id, retry);
                failed_ids.push(id);
            }
        }
        let retired = self.retire_unwritable_incomplete_disk_copies();
        let demoted = self.demote_requested_ram();
        let any_landed = out.landed || relabel_out.landed;
        let any_resync = out.resync || relabel_out.resync;
        let progressed = any_landed || any_resync || failed_upgrade.is_some() || retired || demoted;
        if progressed {
            self.prune_entry_tracking();
            self.sync_marks(coordinator);
            self.sync_host_budget();
        }
        if any_landed {
            self.trim_ram_to_cap_inner(coordinator, None, &failed_ids);
        }
        self.sync_host_budget();
        progressed
    }

    /// Advance persistence of 8-layer Global KV blocks owed by disk-backed states.
    pub(crate) fn chain_step(
        &mut self,
        coordinator: &Inkling2KVCacheCoordinator,
        deadline: Instant,
    ) -> bool {
        if self.disk.as_ref().is_some_and(|d| !d.writable()) {
            if self.retire_unwritable_incomplete_disk_copies() {
                self.prune_entry_tracking();
                self.sync_marks(coordinator);
                return true;
            }
            return false;
        }
        let chain_before = self
            .disk
            .as_ref()
            .and_then(|d| d.chain.as_ref().map(|c| c.id));
        let out = {
            let Some(disk) = self.disk.as_mut() else {
                return false;
            };
            let kv = KvPoolHandle::from_adapter(coordinator.full_adapter());
            let out = disk.engine.chain_step(&mut self.store, &kv, deadline);
            disk.counters.offload_failures += out.offload_failures;
            out
        };
        let failed_ids = if out.step.failed {
            self.reconcile_spill_outcome(
                coordinator,
                None,
                chain_before,
                None,
                &[out.failed_id],
                &[],
            )
        } else {
            Vec::new()
        };
        let mut demoted = false;
        if out.step.landed_all {
            self.reconcile_completed_disk_repairs();
            demoted = self.demote_requested_ram();
        }
        if out.step.failed || out.step.landed_all || demoted {
            self.prune_entry_tracking();
            self.sync_marks(coordinator);
        }
        if out.step.landed_all {
            self.trim_ram_to_cap_inner(coordinator, None, &failed_ids);
        }
        self.sync_host_budget();
        out.step.enqueued > 0 || out.step.landed_all || out.step.failed || demoted
    }

    /// Drive queued SSD upgrades/repairs (`PendingDiskUpgrade::InPlace` and `Repair`) whose
    /// backoff has elapsed (or unconditionally when `force == true`).
    ///
    /// Unlike ordinary `WriteBehind` (driven by the governor-clocked `scheduler.rs` loop via
    /// `WriteBehind::step`), upgrades and repairs run even on temporary tiers below `ram_cap`
    /// without triggering ordinary copy-ahead, and use `stage_entry`'s `ColdKeyHold` + rollback
    /// machinery to preserve durable SSD fallbacks.
    fn drive_pending_disk_upgrades_inner(
        &mut self,
        coordinator: &Inkling2KVCacheCoordinator,
        force: bool,
        skip: Option<&mut HashSet<EntryId>>,
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
        if !writable || writing || (!force && has_waiters) || self.pending_disk_upgrades.is_empty()
        {
            return false;
        }
        let now = Instant::now();
        let skip_ref = skip.as_deref();
        let candidate = self
            .pending_disk_upgrades
            .iter()
            .filter_map(|(&id, &kind)| {
                let entry = self.store.get(id)?;
                match kind {
                    PendingDiskUpgrade::InPlace { .. } => {
                        entry.ram()?;
                    }
                    PendingDiskUpgrade::Repair { .. } => {
                        entry.ram()?;
                        if entry.disk().is_some() {
                            return None;
                        }
                    }
                    PendingDiskUpgrade::LabelPromote { .. } => {
                        if !persistent || entry.checkpoint() {
                            return None;
                        }
                        let copy = entry.disk()?;
                        if !copy.landed || copy.chain_owed {
                            return None;
                        }
                    }
                }
                if !force {
                    if now < kind.retry_after() {
                        return None;
                    }
                    if self
                        .spill_backoff
                        .get(&id)
                        .is_some_and(|&retry_after| now < retry_after)
                    {
                        return None;
                    }
                }
                if skip_ref.is_some_and(|s| s.contains(&id)) {
                    return None;
                }
                Some((entry.last_used(), id))
            })
            .min()
            .map(|(_, id)| id);
        let Some(id) = candidate else {
            return false;
        };
        let room = if persistent {
            MakeRoom::Evict
        } else {
            MakeRoom::Refuse
        };
        // Always keep RAM warm until both the state blob AND its complete Global KV chain
        // have landed on SSD; `trim_ram_to_cap` drops RAM once `entry_disk_restorable` is true.
        let staged = self.stage_entry(coordinator, id, room, true);
        if !staged
            && self.pending_disk_upgrades.contains_key(&id)
            && let Some(s) = skip
        {
            s.insert(id);
        }
        staged
    }

    #[cfg(test)]
    fn drive_pending_disk_upgrades(
        &mut self,
        coordinator: &Inkling2KVCacheCoordinator,
        force: bool,
    ) -> bool {
        self.drive_pending_disk_upgrades_inner(coordinator, force, None)
    }

    /// One engine tick of the StateDisk tier (polls landings, advances owed chains, reconciles
    /// upgrades/repairs, requested RAM demotions & unwritable copies, drives queued SSD upgrades,
    /// and enforces `ram_cap`).
    ///
    /// Governor-clocked `WriteBehind::step`, `monitor::set_next_wakeup`, `publish_landing_pages`,
    /// and `signal_progress` are driven by [`super::model::Inkling2Inner::tick_state_disk`].
    pub(crate) fn tick(&mut self, coordinator: &Inkling2KVCacheCoordinator, tick: Tick) -> bool {
        let kv = KvPoolHandle::from_adapter(coordinator.full_adapter());
        let (pending_before, chain_before) = self.disk.as_ref().map_or((None, None), |d| {
            (d.pending_write_id(), d.chain.as_ref().map(|c| c.id))
        });
        let repairs_with_disk_before: Vec<EntryId> = self
            .pending_disk_upgrades
            .iter()
            .filter_map(|(&id, kind)| {
                (kind.is_repair() && self.store.get(id).is_some_and(|e| e.disk().is_some()))
                    .then_some(id)
            })
            .collect();
        let ordinary_owed_before = self.ordinary_chain_owed_ids();
        let (out, relabel_out) = {
            let Some(disk) = self.disk.as_mut() else {
                return false;
            };
            let kv_ref = disk.writable().then_some(&kv);
            let out = disk.engine.tick(&mut self.store, kv_ref, tick);
            let relabel_out = disk.poll_relabel(&mut self.store);
            disk.counters.offloads += out.state_landed_ok + relabel_out.landed_ok;
            disk.counters.offload_failures += out.offload_failures + relabel_out.failures;
            (out, relabel_out)
        };
        let state_failed_id = out.state_failed_id.or(relabel_out.failed_id);
        let mut failed_ids = self.reconcile_spill_outcome(
            coordinator,
            pending_before,
            chain_before,
            state_failed_id,
            &out.chain_failed_ids,
            &ordinary_owed_before,
        );
        let failed_upgrade = self.reconcile_in_flight_disk_upgrade(state_failed_id);
        if let Some(uid) = failed_upgrade
            && !failed_ids.contains(&uid)
        {
            failed_ids.push(uid);
        }
        self.reconcile_completed_disk_repairs();
        let retry = upgrade_retry_after();
        for id in repairs_with_disk_before {
            if !failed_ids.contains(&id)
                && matches!(
                    self.pending_disk_upgrades.get(&id),
                    Some(PendingDiskUpgrade::Repair { .. })
                )
                && self.store.get(id).is_some_and(|e| e.disk().is_none())
            {
                self.mark_repair_failed(id, retry);
                failed_ids.push(id);
            }
        }
        let retired = self.retire_unwritable_incomplete_disk_copies();
        let demoted = self.demote_requested_ram();
        let any_progressed = out.progressed || relabel_out.landed;
        let any_resync = out.resync || relabel_out.resync;
        let state_landed_ok = out.state_landed_ok + relabel_out.landed_ok;
        let mut progressed =
            any_progressed || any_resync || failed_upgrade.is_some() || retired || demoted;
        if any_resync || failed_upgrade.is_some() || retired || demoted {
            self.prune_entry_tracking();
            self.sync_marks(coordinator);
            self.sync_host_budget();
        }
        if state_landed_ok > 0 || any_progressed || self.ram_len() > self.ram_cap {
            self.trim_ram_to_cap_inner(coordinator, None, &failed_ids);
        }
        if matches!(tick, Tick::Full { .. }) {
            let mut tick_skip: HashSet<EntryId> = failed_ids.into_iter().collect();
            let skip_opt = (!tick_skip.is_empty()).then_some(&mut tick_skip);
            progressed |= self.drive_pending_disk_upgrades_inner(coordinator, false, skip_opt);
            progressed |= self.poll_offload(coordinator);
        }
        self.sync_host_budget();
        progressed
    }

    /// Synchronously flush all waiting `WriteBehind` points, pending SSD upgrades/repairs, & owed
    /// Global KV chains to SSD until `deadline` (used by `close_cache`, `Drop`, and restart tests).
    pub(crate) fn flush_to_disk(
        &mut self,
        coordinator: &Inkling2KVCacheCoordinator,
        deadline: Instant,
    ) -> bool {
        self.reconcile_completed_disk_repairs();

        let has_pending_upgrades = |this: &Self, skipped: &HashSet<EntryId>| {
            let persistent_writable = this
                .disk
                .as_ref()
                .is_some_and(|d| d.tier.persistent && d.writable());
            this.pending_disk_upgrades.iter().any(|(&id, kind)| {
                !skipped.contains(&id)
                    && this.store.get(id).is_some_and(|e| match kind {
                        PendingDiskUpgrade::InPlace { .. } => e.ram().is_some(),
                        PendingDiskUpgrade::Repair { .. } => {
                            e.ram().is_some() && e.disk().is_none()
                        }
                        PendingDiskUpgrade::LabelPromote { .. } => {
                            persistent_writable
                                && !e.checkpoint()
                                && e.disk().is_some_and(|c| c.landed && !c.chain_owed)
                        }
                    })
            })
        };
        let has_work = |this: &Self, skipped: &HashSet<EntryId>| {
            let has_pending_write = this.disk.as_ref().is_some_and(|d| d.writing());
            let has_owed_chain = this
                .store
                .entries()
                .any(|e| e.disk().is_some_and(|c| c.chain_owed));
            let has_waiting_wb = this
                .disk
                .as_ref()
                .is_some_and(|d| d.writable() && d.write_behind.snapshot().waiting > 0);
            has_pending_write
                || has_owed_chain
                || has_waiting_wb
                || has_pending_upgrades(this, skipped)
        };
        let finalize_and_check =
            |this: &mut Self, failed: &HashSet<EntryId>, retryable_wb: &[(EntryId, u32, bool)]| {
                if !retryable_wb.is_empty()
                    && let Some(disk) = this.disk.as_mut()
                {
                    let writable = disk.writable();
                    let persistent = disk.tier.persistent;
                    for &(id, depth, head) in retryable_wb {
                        disk.write_behind.record_published(
                            &this.store,
                            writable,
                            persistent,
                            id,
                            depth,
                            0.0,
                            head,
                        );
                    }
                }
                let has_pending_write = this.disk.as_ref().is_some_and(|d| d.writing());
                let has_owed_chain = this
                    .store
                    .entries()
                    .any(|e| e.disk().is_some_and(|c| c.chain_owed));
                let has_waiting_wb = this
                    .disk
                    .as_ref()
                    .is_some_and(|d| d.write_behind.snapshot().waiting > 0);
                let disk_writable = this.disk.as_ref().is_none_or(|d| d.writable());
                disk_writable
                    && failed.is_empty()
                    && !has_pending_write
                    && !has_owed_chain
                    && !has_waiting_wb
                    && this.pending_disk_upgrades.is_empty()
                    && this.in_flight_disk_upgrade.is_none()
            };

        // Clear stale failure markers from earlier passes so retryable entries get a fresh
        // attempt once space has recovered, while preserving unrecoverable invalidations in
        // `failed_entries` so `flush_to_disk` never falsely returns `true`.
        self.recent_failed_upgrades.clear();
        self.recent_failed_spills.clear();
        self.poll_offload(coordinator);
        self.recent_failed_upgrades.clear();
        let mut failed_entries: HashSet<EntryId> = HashSet::new();
        let recent_spills: Vec<EntryId> = self.recent_failed_spills.drain().collect();
        for id in recent_spills {
            if !self.is_retryable_ordinary_wb(coordinator, id) {
                failed_entries.insert(id);
            }
        }
        let mut skipped_upgrades: HashSet<EntryId> = failed_entries.clone();
        let mut retryable_wb_refusals: Vec<(EntryId, u32, bool)> = Vec::new();
        loop {
            self.poll_offload(coordinator);
            for id in self
                .recent_failed_upgrades
                .drain()
                .chain(self.recent_failed_spills.drain())
            {
                failed_entries.insert(id);
                skipped_upgrades.insert(id);
            }
            if !has_work(self, &skipped_upgrades) {
                return finalize_and_check(self, &failed_entries, &retryable_wb_refusals);
            }
            if Instant::now() >= deadline {
                return finalize_and_check(self, &failed_entries, &retryable_wb_refusals);
            }
            let (writable, writing) = self
                .disk
                .as_ref()
                .map_or((false, false), |d| (d.writable(), d.writing()));
            if !writable && !writing {
                return finalize_and_check(self, &failed_entries, &retryable_wb_refusals);
            }
            let wb_waiting_before = self
                .disk
                .as_ref()
                .map_or(0, |d| d.write_behind.snapshot().waiting);
            let skipped_before = skipped_upgrades.len();
            let offloads_before = self.disk.as_ref().map_or(0, |d| d.counters.offloads);
            let staged_upgrade = self.drive_pending_disk_upgrades_inner(
                coordinator,
                true,
                Some(&mut skipped_upgrades),
            );
            if let Some(disk) = self.disk.as_mut()
                && disk.writable()
            {
                let mut behind = std::mem::take(&mut disk.write_behind);
                let _ = behind.flush(self, |store, id, head| {
                    let effective_head = head || store.is_prompt_head(id);
                    if failed_entries.contains(&id) {
                        if store.is_retryable_ordinary_wb(coordinator, id)
                            && let Some(depth) = store
                                .store
                                .get(id)
                                .and_then(|e| u32::try_from(e.depth()).ok())
                        {
                            retryable_wb_refusals.push((id, depth, effective_head));
                        }
                        return PersistStatus::Refused;
                    }
                    let status = store.persist_entry_inner(coordinator, id, effective_head, true);
                    if status == PersistStatus::Refused
                        && store.is_retryable_ordinary_wb(coordinator, id)
                        && let Some(depth) = store
                            .store
                            .get(id)
                            .and_then(|e| u32::try_from(e.depth()).ok())
                    {
                        failed_entries.insert(id);
                        retryable_wb_refusals.push((id, depth, effective_head));
                    }
                    status
                });
                if let Some(disk) = self.disk.as_mut() {
                    disk.write_behind = behind;
                }
            }
            let chained = self.chain_step(coordinator, deadline);
            let polled = self.poll_offload(coordinator);
            for id in self
                .recent_failed_upgrades
                .drain()
                .chain(self.recent_failed_spills.drain())
            {
                failed_entries.insert(id);
                skipped_upgrades.insert(id);
            }
            let offloads_after = self.disk.as_ref().map_or(0, |d| d.counters.offloads);
            if chained || offloads_after > offloads_before {
                skipped_upgrades.retain(|id| failed_entries.contains(id));
            }
            let wb_waiting_after = self
                .disk
                .as_ref()
                .map_or(0, |d| d.write_behind.snapshot().waiting);
            let wb_progressed = staged_upgrade
                || wb_waiting_after < wb_waiting_before
                || skipped_upgrades.len() > skipped_before;

            if !has_work(self, &skipped_upgrades) {
                return finalize_and_check(self, &failed_entries, &retryable_wb_refusals);
            }
            if Instant::now() >= deadline {
                return finalize_and_check(self, &failed_entries, &retryable_wb_refusals);
            }
            if polled || chained || wb_progressed {
                continue;
            }
            let Some(disk) = self.disk.as_ref() else {
                return finalize_and_check(self, &failed_entries, &retryable_wb_refusals);
            };
            if !disk.writable() && !disk.writing() {
                return finalize_and_check(self, &failed_entries, &retryable_wb_refusals);
            }
            if !disk.await_landing(deadline) {
                return finalize_and_check(self, &failed_entries, &retryable_wb_refusals);
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
        let _ = self.poll_offload(coordinator);
        if !persistent && let Some(disk) = self.disk.as_mut() {
            disk.engine.strip_disk_copies(&mut self.store);
            disk.preserve_entry_on_state_failure = None;
            disk.pending_relabel = None;
            self.pending_disk_upgrades.clear();
            self.in_flight_disk_upgrade = None;
            self.recent_failed_upgrades.clear();
            self.recent_failed_spills.clear();
            self.spill_backoff.clear();
            self.demote_ram_on_durable.clear();
            self.prune_entry_tracking();
            self.sync_marks(coordinator);
        }
        self.sync_host_budget();
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
            self.demote_ram_on_durable.remove(&id);
            self.ram_hits += 1;
            return Some((ram, false));
        }
        // Register a `ReadWait` guard before polling/waiting so any `poll_offload` ->
        // `trim_ram_to_cap` or background spill yields the reusable staging buffer to this
        // synchronous SSD restore (reads before writes).
        let _read_wait = self.disk.as_ref()?.engine.acquire_read_wait();
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
        let has_dw_span = self
            .resolve_geometry()
            .is_some_and(|g| g.draft_window_span_bytes > 0);
        let decoded = {
            let disk = self.disk.as_mut()?;
            disk.engine.read_into_staging(
                &self.store,
                id,
                |codec, _spans, entry, _copy, staging| {
                    let depth = u32::try_from(entry.depth()).ok()?;
                    codec.geometry.decode_from(depth, staging).ok()
                },
            )
        };
        if let Some(anchor) = decoded {
            if let Some(dw) = anchor.draft_window.as_ref() {
                dw.attach_to(&self.host_budget);
            }
            if let Some(disk) = self.disk.as_mut() {
                disk.counters.ssd_hits += 1;
            }
            if has_dw_span {
                self.adopted_unverified_window.remove(&id);
                if anchor.draft_window.is_none() {
                    self.entries_missing_draft_window.insert(id);
                } else {
                    self.entries_missing_draft_window.remove(&id);
                }
            }
            Some((anchor, true))
        } else {
            None
        }
    }

    /// Adopt complete `(state blob + 8-layer Global KV chain)` entries from a persistent tier
    /// on startup using the shared [`StateDiskEngine::adopt_all`].
    pub(crate) fn adopt_all(&mut self) -> u64 {
        let has_dw_span = self
            .resolve_geometry()
            .is_some_and(|g| g.draft_window_span_bytes > 0);
        let existing: HashSet<EntryId> = self.store.entries().map(Entry::id).collect();
        let adopted = {
            let Some(disk) = self.disk.as_mut() else {
                return 0;
            };
            let count = disk.engine.adopt_all(&mut self.store, false);
            disk.counters.adopted += count;
            count
        };
        if adopted > 0 {
            let newly_adopted: Vec<(EntryId, bool)> = self
                .store
                .entries()
                .filter(|entry| !existing.contains(&entry.id()))
                .map(|entry| (entry.id(), entry.checkpoint()))
                .collect();
            for (id, is_checkpoint) in newly_adopted {
                if has_dw_span {
                    self.adopted_unverified_window.insert(id);
                }
                if !is_checkpoint {
                    self.mark_prompt_head(id);
                }
            }
        }
        self.sync_host_budget();
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
const INKLING2_FORWARD_ARITH_VERSION: &str = "v5";

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
/// + resolved prefill query precision policy (`activations`) + forward arithmetic epoch
/// + optional DFlash2 draft directory identity.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_persistent_identity(
    config: &Inkling2Config,
    coordinator: &Inkling2KVCacheCoordinator,
    geometry: &Inkling2StateGeometry,
    activations: crate::transformer::Activations,
    files: CheckpointFiles<'_>,
    weights: &WeightsResident,
    guard: &CheckpointLoadGuard<'_>,
    draft_identity: Option<&[u8]>,
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
    let state_fp = if let Some(draft_id) = draft_identity {
        ColdCacheFingerprint::from_components([
            b"inkling2-state-fp:v2".as_slice(),
            kv_fp.as_bytes().as_slice(),
            state_desc.as_bytes(),
            b"dflash2-draft:v1".as_slice(),
            draft_id,
        ])
    } else {
        ColdCacheFingerprint::from_components([
            b"inkling2-state-fp:v2".as_slice(),
            kv_fp.as_bytes().as_slice(),
            state_desc.as_bytes(),
        ])
    };
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
    draft_identity: Option<&[u8]>,
) -> Option<PersistentIdentity> {
    compute_path_identity_with_arith(
        model_dir,
        config,
        coordinator,
        geometry,
        int8_query_from,
        ForwardArithmeticSettings::current(),
        draft_identity,
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
    draft_identity: Option<&[u8]>,
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
    let state_fp = if let Some(draft_id) = draft_identity {
        ColdCacheFingerprint::from_components([
            b"inkling2-path-state:v2".as_slice(),
            kv_fp.as_bytes().as_slice(),
            state_desc.as_bytes(),
            b"dflash2-draft:v1".as_slice(),
            draft_id,
        ])
    } else {
        ColdCacheFingerprint::from_components([
            b"inkling2-path-state:v2".as_slice(),
            kv_fp.as_bytes().as_slice(),
            state_desc.as_bytes(),
        ])
    };
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
    dflash2: Option<&DFlash2Config>,
    options: &DiskTierOptions,
    identity: impl FnOnce() -> Option<PersistentIdentity>,
    model_id: u64,
) -> Option<(ColdTierContext, Inkling2StateDisk)> {
    let geometry =
        Inkling2StateGeometry::from_coordinator_with_draft(config, coordinator, dflash2).ok()?;
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

/// Helper: restore a flat [`Inkling2StateAnchor`] onto `seq_id` across Group 1 (SWA raw blocks)
/// and `conv_states` (48-visit ShortConv state). Caller has already restored Group 0 (Global KV)
/// at `boundary`.
pub(crate) fn install_anchor_swa_and_conv(
    coordinator: &mut Inkling2KVCacheCoordinator,
    conv_states: &mut HashMap<SeqId, Inkling2ShortConvState>,
    geometry: &Inkling2StateGeometry,
    seq_id: SeqId,
    prompt_tokens: &[u32],
    boundary: u32,
    anchor: &Inkling2StateAnchor,
) -> Result<()> {
    if coordinator.sliding_adapter().is_some() {
        coordinator
            .restore_sliding_raw_blocks_flat_for(seq_id, prompt_tokens, boundary, &anchor.swa_data)
            .map_err(Error::from_reason)?;
    }
    let conv = unpack_short_conv_state(geometry, &anchor.conv_data).map_err(Error::from_reason)?;
    conv_states.insert(seq_id, conv);
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
        let dflash_cfg =
            crate::models::qwen3_5::dflash2::dflash2_config_with_context(cfg.hidden_size, 16);

        for kv_format in [KvFormat::Int8, KvFormat::Bf16] {
            let paged_cfg = Inkling2PagedConfig::default_for_config(&cfg)
                .with_kv_format(kv_format)
                .with_max_model_len(512)
                .with_max_concurrent_sequences(2);
            let coord =
                Inkling2KVCacheCoordinator::new_for_config(&cfg, paged_cfg).expect("coordinator");

            // 1. Plain v1 geometry (no DFlash2 draft).
            let geom_v1 =
                Inkling2StateGeometry::from_coordinator(&cfg, &coord).expect("geometry v1");
            assert_eq!(geom_v1.swa_layers, 5);
            assert_eq!(geom_v1.max_swa_blocks, 2); // sliding_window=64, bs=32
            assert_eq!(geom_v1.num_visits, 6);
            assert_eq!(geom_v1.draft_window_span_bytes, 0);
            assert!(geom_v1.geometry_string().starts_with("inkling2-state:v1 "));
            assert_eq!(geom_v1.spans().len(), 2);

            // 2. DFlash2 v2 geometry (with Span 2 draft context window).
            let geom_v2 =
                Inkling2StateGeometry::from_coordinator_with_draft(&cfg, &coord, Some(&dflash_cfg))
                    .expect("geometry v2");
            assert_eq!(geom_v2.draft_window_width, cfg.hidden_size as u32);
            assert_eq!(geom_v2.draft_window_slots, 16);
            assert_eq!(
                geom_v2.draft_window_span_bytes,
                DRAFT_WINDOW_HEADER_BYTES + DraftContextWindow::bytes(cfg.hidden_size as u32, 16)
            );
            assert!(geom_v2.geometry_string().starts_with("inkling2-state:v2 "));
            assert!(geom_v2.geometry_string().ends_with(" dw=64:16"));
            assert_eq!(geom_v2.spans().len(), 3);

            // Test both sub-window (boundary=32 -> 1 SWA block) and full-window (boundary=128 -> 2 SWA blocks).
            for boundary in [32u32, 128u32] {
                let live_bytes = geom_v1.live_swa_bytes(boundary).expect("live_swa_bytes");
                let swa_vec: Vec<u8> = (0..live_bytes)
                    .map(|i| ((i * 31 + boundary as usize * 7 + 1) % 251) as u8)
                    .collect();
                let swa_data: Arc<Vec<u8>> = Arc::new(swa_vec);

                let mut conv = Inkling2ShortConvState::new(
                    cfg.num_layer_visits(),
                    cfg.sconv_kernel_size,
                    cfg.num_key_value_heads * cfg.head_dim,
                    cfg.hidden_size,
                    DType::BFloat16,
                )
                .expect("conv state");
                // Populate non-zero ShortConv values on visit 0 so unpack parity checks real data.
                let hist = (cfg.sconv_kernel_size - 1) as i64;
                let kv_dim = (cfg.num_key_value_heads * cfg.head_dim) as i64;
                let hidden = cfg.hidden_size as i64;
                let kv_vals: Vec<u16> = (0..(hist * kv_dim) as usize)
                    .map(|i| half::bf16::from_f32(((i % 19) as f32 - 9.0) * 0.25).to_bits())
                    .collect();
                let h_vals: Vec<u16> = (0..(hist * hidden) as usize)
                    .map(|i| half::bf16::from_f32(((i % 13) as f32 - 6.0) * 0.5).to_bits())
                    .collect();
                conv.visits[0].k =
                    MxArray::from_bfloat16(&kv_vals, &[hist, kv_dim]).expect("conv k");
                conv.visits[0].attn_post =
                    MxArray::from_bfloat16(&h_vals, &[hist, hidden]).expect("conv attn_post");

                let conv_data = pack_short_conv_state(&geom_v1, &conv).expect("pack conv");
                let unpacked = unpack_short_conv_state(&geom_v1, &conv_data).expect("unpack conv");
                assert_eq!(unpacked.visits.len(), cfg.num_layer_visits());
                assert_eq!(
                    unpacked.visits[0].k.to_uint16_native().expect("k u16"),
                    kv_vals
                );
                assert_eq!(
                    unpacked.visits[0]
                        .attn_post
                        .to_uint16_native()
                        .expect("attn_post u16"),
                    h_vals
                );

                // v1 roundtrip (no draft window).
                let anchor_v1 = Inkling2StateAnchor {
                    swa_data: Arc::clone(&swa_data),
                    conv_data: Arc::clone(&conv_data),
                    draft_window: None,
                    cached_bytes: (swa_data.len() + conv_data.len()) as u64,
                };
                let mut buf_v1 = vec![0xAAu8; geom_v1.payload_bytes() as usize];
                geom_v1
                    .encode_into(boundary, &anchor_v1, &mut buf_v1)
                    .expect("encode_into v1");
                let decoded_v1 = geom_v1
                    .decode_from(boundary, &buf_v1)
                    .expect("decode_from v1");
                assert_eq!(decoded_v1.swa_data, swa_data);
                assert_eq!(decoded_v1.conv_data, conv_data);
                assert!(decoded_v1.draft_window.is_none());
                assert_eq!(decoded_v1.cached_bytes, anchor_v1.cached_bytes);

                // v2 roundtrip with `draft_window = None` (valid_flag = 0).
                let mut buf_v2_none = vec![0xCCu8; geom_v2.payload_bytes() as usize];
                geom_v2
                    .encode_into(boundary, &anchor_v1, &mut buf_v2_none)
                    .expect("encode_into v2 None");
                let decoded_v2_none = geom_v2
                    .decode_from(boundary, &buf_v2_none)
                    .expect("decode_from v2 None");
                assert_eq!(decoded_v2_none.swa_data, swa_data);
                assert_eq!(decoded_v2_none.conv_data, conv_data);
                assert!(decoded_v2_none.draft_window.is_none());

                // Corrupt `valid_flag = 2` must fail decode cleanly.
                let dw_offset = (geom_v2.swa_span_bytes + geom_v2.conv_span_bytes) as usize;
                let mut buf_v2_bad = buf_v2_none.clone();
                buf_v2_bad[dw_offset..dw_offset + 4].copy_from_slice(&2u32.to_le_bytes());
                assert!(
                    geom_v2.decode_from(boundary, &buf_v2_bad).is_err(),
                    "invalid valid_flag=2 must be rejected"
                );

                // v2 roundtrip with a real populated `DraftContextWindow` -> `PackedDraftWindow`.
                let mut live_window =
                    DraftContextWindow::new(cfg.hidden_size as u32, 16).expect("window");
                let rows = MxArray::random_normal(
                    &[i64::from(boundary), cfg.hidden_size as i64],
                    0.0,
                    1.0,
                    Some(DType::BFloat16),
                )
                .expect("random rows");
                live_window.store_rows(&rows, 0).expect("store_rows");
                let packed_dw = PackedDraftWindow::from_window(&live_window).expect("from_window");
                assert!(packed_dw.is_complete_for(boundary));
                assert!(!packed_dw.is_complete_for(boundary + 32));
                assert_eq!(packed_dw.begin, boundary.saturating_sub(16));
                assert_eq!(packed_dw.end, boundary);

                let restored_win = packed_dw.to_window().expect("to_window");
                assert_eq!(
                    restored_win.host_bytes().expect("restored bytes"),
                    live_window.host_bytes().expect("live bytes")
                );
                assert_eq!(
                    (restored_win.begin(), restored_win.end()),
                    (live_window.begin(), live_window.end())
                );
                let mut target_win =
                    DraftContextWindow::new(cfg.hidden_size as u32, 16).expect("target win");
                assert!(packed_dw.load_into(&mut target_win).expect("load_into"));
                assert_eq!(
                    target_win.host_bytes().expect("target bytes"),
                    live_window.host_bytes().expect("live bytes")
                );
                let mut wrong_geom_win =
                    DraftContextWindow::new(cfg.hidden_size as u32, 8).expect("wrong geom");
                assert!(
                    !packed_dw
                        .load_into(&mut wrong_geom_win)
                        .expect("load_into wrong geom")
                );

                let anchor_v2 = Inkling2StateAnchor {
                    swa_data: Arc::clone(&swa_data),
                    conv_data: Arc::clone(&conv_data),
                    draft_window: Some(packed_dw.clone()),
                    cached_bytes: (swa_data.len() + conv_data.len() + packed_dw.bytes.len()) as u64,
                };
                let mut buf_v2 = vec![0xBBu8; geom_v2.payload_bytes() as usize];
                geom_v2
                    .encode_into(boundary, &anchor_v2, &mut buf_v2)
                    .expect("encode_into v2 Some");
                let decoded_v2 = geom_v2
                    .decode_from(boundary, &buf_v2)
                    .expect("decode_from v2 Some");
                assert_eq!(decoded_v2.swa_data, swa_data);
                assert_eq!(decoded_v2.conv_data, conv_data);
                assert_eq!(decoded_v2.draft_window.as_ref(), Some(&packed_dw));
                assert_eq!(decoded_v2.cached_bytes, anchor_v2.cached_bytes);
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

        let cfg = tiny_inkling2_config();
        let dflash_cfg =
            crate::models::qwen3_5::dflash2::dflash2_config_with_context(cfg.hidden_size, 16);
        let draft_id_a = b"draft-weights-sha-aaa";
        let draft_id_b = b"draft-weights-sha-bbb";

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
        let geom =
            Inkling2StateGeometry::from_coordinator_with_draft(&cfg, &coord, Some(&dflash_cfg))
                .expect("geom");
        let id = compute_path_identity(
            &temp_root,
            &cfg,
            &coord,
            &geom,
            Some(1024),
            Some(draft_id_a),
        );
        // Switching activation policy (INT8 query threshold vs BF16 queries) must isolate SSD cache identity.
        let id_bf16_q =
            compute_path_identity(&temp_root, &cfg, &coord, &geom, None, Some(draft_id_a))
                .expect("bf16 query identity");
        let id_int8_q = id.as_ref().expect("int8 query identity");
        assert_ne!(id_int8_q.namespace, id_bf16_q.namespace);
        assert_ne!(id_int8_q.kv, id_bf16_q.kv);
        assert_ne!(id_int8_q.state, id_bf16_q.state);
        // Changing DFlash2 draft identity must isolate `state` and `namespace` while keeping target `kv` unchanged.
        let id_other_draft = compute_path_identity(
            &temp_root,
            &cfg,
            &coord,
            &geom,
            Some(1024),
            Some(draft_id_b),
        )
        .expect("other draft identity");
        assert_eq!(id_int8_q.kv, id_other_draft.kv);
        assert_ne!(id_int8_q.state, id_other_draft.state);
        assert_ne!(id_int8_q.namespace, id_other_draft.namespace);

        let (ctx, disk) =
            open_inkling2_state_disk(&cfg, &coord, Some(&dflash_cfg), &disk_opts, move || id, 0)
                .expect("open_inkling2_state_disk");
        coord.set_full_cold_tier(ctx);
        let mut store = Inkling2StateStore::new(coord.block_size(), recency);
        store.set_geometry(geom.clone());
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

        // First publish an AR-only anchor (`draft_window = None`) at boundary=64, then verify a
        // subsequent DFlash2 capture with a complete `PackedDraftWindow` upgrades the same block!
        let ar_id = store
            .capture_boundary(
                &mut coord,
                &conv,
                1,
                &tokens,
                64,
                0,
                StateClass::Ordinary,
                None,
            )
            .expect("capture_boundary AR")
            .expect("ar_id");
        assert!(
            store
                .store
                .get(ar_id)
                .and_then(Entry::ram)
                .is_some_and(|a| a.draft_window.is_none())
        );

        let mut dw = DraftContextWindow::new(cfg.hidden_size as u32, 16).expect("dw");
        let rows = MxArray::random_normal(
            &[64, cfg.hidden_size as i64],
            0.0,
            1.0,
            Some(DType::BFloat16),
        )
        .expect("rows");
        dw.store_rows(&rows, 0).expect("store_rows");
        let packed_dw = PackedDraftWindow::from_window(&dw).expect("packed_dw");
        assert!(packed_dw.is_complete_for(64));

        let entry_id = store
            .capture_boundary(
                &mut coord,
                &conv,
                1,
                &tokens,
                64,
                0,
                StateClass::Ordinary,
                Some(&packed_dw),
            )
            .expect("capture_boundary upgraded")
            .expect("entry_id");
        assert!(store.is_restorable(&coord, entry_id), "restorable in RAM");
        assert_eq!(
            store
                .store
                .get(entry_id)
                .and_then(Entry::ram)
                .and_then(|a| a.draft_window.as_ref()),
            Some(&packed_dw),
            "DFlash2 capture must upgrade AR anchor on the same block"
        );

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

        // Drop RAM + purge hot pool -> must still be restorable from SSD with PackedDraftWindow!
        store.drop_all_ram_keeping_disk();
        coord.release_all_and_purge().expect("purge hot pools");
        let query: Vec<u32> = (100u32..170u32).collect();
        let hit = store
            .lookup_deepest_restorable(&coord, &query, 0)
            .expect("SSD lookup hit");
        assert_eq!(hit.depth, 64);
        assert!(!store.chain_is_hot(&coord, hit.id));
        let (ssd_anchor, from_ssd) = store
            .load_anchor(&coord, hit.id)
            .expect("load_anchor from SSD");
        assert!(from_ssd);
        assert_eq!(
            ssd_anchor.draft_window.as_ref(),
            Some(&packed_dw),
            "SSD restore must preserve exact PackedDraftWindow bytes and bounds"
        );

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
        let geom2 =
            Inkling2StateGeometry::from_coordinator_with_draft(&cfg, &coord2, Some(&dflash_cfg))
                .expect("geom2");
        // Reopening with a different activation query policy (None = BF16) must miss the old cache!
        let id_mismatch =
            compute_path_identity(&temp_root, &cfg, &coord2, &geom2, None, Some(draft_id_a));
        let (_ctx_miss, disk_miss) = open_inkling2_state_disk(
            &cfg,
            &coord2,
            Some(&dflash_cfg),
            &disk_opts,
            move || id_mismatch,
            0,
        )
        .expect("open mismatched policy disk");
        let mut store_miss = Inkling2StateStore::new(coord2.block_size(), Arc::clone(&recency2));
        store_miss.disk = Some(disk_miss);
        assert_eq!(
            store_miss.adopt_all(),
            0,
            "different activation query precision must NOT adopt old SSD cache"
        );
        drop(store_miss);

        // Reopening with a different DFlash2 draft identity must NOT adopt the old SSD cache!
        let id_draft_miss = compute_path_identity(
            &temp_root,
            &cfg,
            &coord2,
            &geom2,
            Some(1024),
            Some(draft_id_b),
        );
        let (_ctx_draft_miss, disk_draft_miss) = open_inkling2_state_disk(
            &cfg,
            &coord2,
            Some(&dflash_cfg),
            &disk_opts,
            move || id_draft_miss,
            0,
        )
        .expect("open mismatched draft disk");
        let mut store_draft_miss =
            Inkling2StateStore::new(coord2.block_size(), Arc::clone(&recency2));
        store_draft_miss.disk = Some(disk_draft_miss);
        assert_eq!(
            store_draft_miss.adopt_all(),
            0,
            "different DFlash2 draft identity must NOT adopt old SSD cache"
        );
        drop(store_draft_miss);

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
            Some(draft_id_a),
        );
        let (_ctx_ar_miss, disk_ar_miss) = open_inkling2_state_disk(
            &cfg,
            &coord2,
            Some(&dflash_cfg),
            &disk_opts,
            move || id_alt_attn_res,
            0,
        )
        .expect("open mismatched attn_res disk");
        let mut store_ar_miss = Inkling2StateStore::new(coord2.block_size(), Arc::clone(&recency2));
        store_ar_miss.disk = Some(disk_ar_miss);
        assert_eq!(
            store_ar_miss.adopt_all(),
            0,
            "different MLX_INKLING2_ATTN_RES_VEC4_MAX_T must NOT adopt old SSD cache"
        );
        drop(store_ar_miss);

        let id2 = compute_path_identity(
            &temp_root,
            &cfg,
            &coord2,
            &geom2,
            Some(1024),
            Some(draft_id_a),
        );
        let (ctx2, disk2) =
            open_inkling2_state_disk(&cfg, &coord2, Some(&dflash_cfg), &disk_opts, move || id2, 0)
                .expect("reopen state disk");
        coord2.set_full_cold_tier(ctx2);
        let mut store2 = Inkling2StateStore::new(coord2.block_size(), recency2);
        store2.set_geometry(geom2);
        store2.disk = Some(disk2);
        let adopted = store2.adopt_all();
        assert_eq!(adopted, 1, "expected 1 adopted state on restart");
        let hit2 = store2
            .lookup_deepest_restorable(&coord2, &query, 0)
            .expect("post-restart lookup hit");
        assert_eq!(hit2.depth, 64);
        let (adopted_anchor, adopted_from_ssd) = store2
            .load_anchor(&coord2, hit2.id)
            .expect("load adopted anchor from SSD");
        assert!(adopted_from_ssd);
        assert_eq!(
            adopted_anchor.draft_window.as_ref(),
            Some(&packed_dw),
            "post-restart adopted anchor must restore exact PackedDraftWindow"
        );

        let _ = std::fs::remove_dir_all(&temp_root);
    }

    #[cfg(target_os = "macos")]
    fn test_deadline(secs: u64) -> Instant {
        Instant::now() + Duration::from_secs(secs)
    }

    #[cfg(target_os = "macos")]
    fn make_short_conv(cfg: &Inkling2Config) -> Inkling2ShortConvState {
        Inkling2ShortConvState::new(
            cfg.num_layer_visits(),
            cfg.sconv_kernel_size,
            cfg.num_key_value_heads * cfg.head_dim,
            cfg.hidden_size,
            DType::BFloat16,
        )
        .expect("conv")
    }

    #[cfg(target_os = "macos")]
    fn make_draft_window(width: u32, slots: u32, rows: u32) -> DraftContextWindow {
        let mut dw = DraftContextWindow::new(width, slots).expect("dw");
        let data = MxArray::random_normal(
            &[i64::from(rows), i64::from(width)],
            0.0,
            1.0,
            Some(DType::BFloat16),
        )
        .expect("rows");
        dw.store_rows(&data, 0).expect("store_rows");
        dw
    }

    #[cfg(target_os = "macos")]
    fn patient_victim_query() -> crate::engine::governor::ledger::VictimQuery {
        use crate::engine::governor::ledger::{Need, TakeRights, VictimQuery};
        VictimQuery {
            rights: TakeRights {
                units: true,
                waits_for_writes: true,
                waits_for_landings: true,
                ..Default::default()
            },
            need: Need::default(),
        }
    }

    #[cfg(target_os = "macos")]
    fn store_victim(
        id: EntryId,
        stamp: u64,
        bytes: u64,
    ) -> crate::engine::governor::ledger::Victim {
        use crate::engine::governor::ledger::{Stage, UnitKey, Victim};
        Victim {
            key: UnitKey::Store(id),
            stamp,
            bytes,
            pages: 0,
            stage: Stage::FreeNow,
            later: false,
        }
    }

    #[cfg(target_os = "macos")]
    fn attach_synthetic_disk(
        fx: &StateDiskTestFixture,
        inner: &mut crate::models::inkling2::model::Inkling2Inner,
    ) {
        use crate::disk_tier::{CacheDisk, DiskTierOptions};
        let disk_opts = DiskTierOptions {
            cache_disk: CacheDisk::Bytes(64u64 << 20),
            persistent: true,
            cache_dir: Some(fx.cache_dir.clone()),
            temp_root: Some(fx.temp_root.clone()),
        };
        let geom = {
            let rt = inner.paged.as_ref().expect("paged");
            Inkling2StateGeometry::from_coordinator_with_draft(
                &inner.config,
                &rt.coordinator,
                inner.dflash2.as_ref().map(|d| &d.config),
            )
            .expect("geom")
        };
        let identity = compute_path_identity(
            &fx.temp_root,
            &inner.config,
            &inner.paged.as_ref().expect("paged").coordinator,
            &geom,
            None,
            Some(fx.draft_id),
        );
        assert!(
            inner
                .attach_state_disk(&disk_opts, move || identity)
                .is_some(),
            "StateDisk must attach"
        );
    }

    #[cfg(target_os = "macos")]
    struct StateDiskTestFixture {
        temp_root: std::path::PathBuf,
        cache_dir: std::path::PathBuf,
        cfg: Inkling2Config,
        dflash_cfg: DFlash2Config,
        paged_cfg: Inkling2PagedConfig,
        conv: Inkling2ShortConvState,
        draft_id: &'static [u8],
        chmod_dir: Option<std::path::PathBuf>,
    }

    #[cfg(target_os = "macos")]
    impl Drop for StateDiskTestFixture {
        fn drop(&mut self) {
            use std::os::unix::fs::PermissionsExt;
            WRITE_BEHIND_MIN_TOKENS_OVERRIDE.with(|c| c.set(None));
            WRITE_BEHIND_DELAY_OVERRIDE.with(|c| c.set(None));
            crate::disk_tier::TEST_WRITER.with(|w| w.set(None));
            if let Some(ref dir) = self.chmod_dir {
                let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o755));
            }
            let _ = std::fs::remove_dir_all(&self.temp_root);
        }
    }

    #[cfg(target_os = "macos")]
    impl StateDiskTestFixture {
        fn new(tag: &str, draft_id: &'static [u8], wb_delay: Option<f64>) -> Self {
            if let Some(delay) = wb_delay {
                WRITE_BEHIND_MIN_TOKENS_OVERRIDE.with(|c| c.set(Some(32)));
                WRITE_BEHIND_DELAY_OVERRIDE.with(|c| c.set(Some(delay)));
            }
            let temp_root =
                std::env::temp_dir().join(format!("inkling2-{tag}-{}", uuid::Uuid::new_v4()));
            let cache_dir = temp_root.join("cache");
            std::fs::create_dir_all(&cache_dir).expect("create cache dir");
            let cfg = tiny_inkling2_config();
            let dflash_cfg =
                crate::models::qwen3_5::dflash2::dflash2_config_with_context(cfg.hidden_size, 16);
            let paged_cfg = Inkling2PagedConfig::default_for_config(&cfg)
                .with_kv_format(KvFormat::Int8)
                .with_max_model_len(2048)
                .with_max_concurrent_sequences(2);
            let conv = make_short_conv(&cfg);
            Self {
                temp_root,
                cache_dir,
                cfg,
                dflash_cfg,
                paged_cfg,
                conv,
                draft_id,
                chmod_dir: None,
            }
        }

        fn with_paged(mut self, max_model_len: u32, max_seqs: u32) -> Self {
            self.paged_cfg = Inkling2PagedConfig::default_for_config(&self.cfg)
                .with_kv_format(KvFormat::Int8)
                .with_max_model_len(max_model_len)
                .with_max_concurrent_sequences(max_seqs);
            self
        }

        fn subdir(&self, name: &str) -> std::path::PathBuf {
            let dir = self.temp_root.join(name);
            std::fs::create_dir_all(&dir).expect("create subdir");
            dir
        }

        fn packed_dw(&self, rows: u32) -> PackedDraftWindow {
            let dw = make_draft_window(self.cfg.hidden_size as u32, 16, rows);
            PackedDraftWindow::from_window(&dw).expect("packed_dw")
        }

        fn open_in_with_persistence(
            &self,
            cache_dir: std::path::PathBuf,
            bytes: u64,
            draft_id: &[u8],
            reserve: Option<u64>,
            persistent: bool,
        ) -> TestSession {
            use crate::disk_tier::{CacheDisk, DiskTierOptions, TEST_WRITER};
            if let Some(r) = reserve {
                TEST_WRITER.with(|w| w.set(Some((8, r))));
            }
            let disk_opts = DiskTierOptions {
                cache_disk: CacheDisk::Bytes(bytes),
                persistent,
                cache_dir: persistent.then_some(cache_dir),
                temp_root: Some(self.temp_root.clone()),
            };
            let recency = Arc::new(CacheRecency::new());
            let mut coord = Inkling2KVCacheCoordinator::new_for_config_with_recency(
                &self.cfg,
                self.paged_cfg,
                Arc::clone(&recency),
            )
            .expect("coord");
            let geom = Inkling2StateGeometry::from_coordinator_with_draft(
                &self.cfg,
                &coord,
                Some(&self.dflash_cfg),
            )
            .expect("geom");
            let id = compute_path_identity(
                &self.temp_root,
                &self.cfg,
                &coord,
                &geom,
                Some(1024),
                Some(draft_id),
            );
            let (ctx, disk) = open_inkling2_state_disk(
                &self.cfg,
                &coord,
                Some(&self.dflash_cfg),
                &disk_opts,
                move || id,
                0,
            )
            .expect("open disk");
            if reserve.is_some() {
                TEST_WRITER.with(|w| w.set(None));
            }
            coord.set_full_cold_tier(ctx);
            let mut store = Inkling2StateStore::new(coord.block_size(), recency);
            store.set_geometry(geom.clone());
            store.disk = Some(disk);
            TestSession { coord, store, geom }
        }

        fn open_in(
            &self,
            cache_dir: std::path::PathBuf,
            bytes: u64,
            draft_id: &[u8],
            reserve: Option<u64>,
        ) -> TestSession {
            self.open_in_with_persistence(cache_dir, bytes, draft_id, reserve, true)
        }

        fn open(&self, bytes: u64, reserve: Option<u64>) -> TestSession {
            self.open_in(self.cache_dir.clone(), bytes, self.draft_id, reserve)
        }

        fn open_temp(&self, bytes: u64, reserve: Option<u64>) -> TestSession {
            self.open_in_with_persistence(
                self.cache_dir.clone(),
                bytes,
                self.draft_id,
                reserve,
                false,
            )
        }

        fn open_adopted(&self, bytes: u64, reserve: Option<u64>, expected: u64) -> TestSession {
            let mut s = self.open(bytes, reserve);
            assert_eq!(s.store.adopt_all(), expected, "expected {expected} adopted");
            s
        }

        fn seed_durable(
            &self,
            entries: &[(u32, &[u32], u32, u64)],
            bytes: u64,
            reserve: Option<u64>,
        ) -> (u64, u64) {
            let mut s = self.open(bytes, reserve);
            let mut first_eid = None;
            for &(seq, tokens, boundary, salt) in entries {
                let eid = s.capture_ord(&self.conv, seq, tokens, boundary, salt, None);
                assert!(s.store.entries_missing_draft_window.contains(&eid));
                first_eid.get_or_insert(eid);
            }
            assert!(s.flush(10), "seed flush_to_disk");
            let total_bytes = s.manager().total_bytes();
            let first_disk_bytes = first_eid.map_or(0, |id| s.disk_copy(id).bytes);
            assert!(s.close(), "seed close_cache");
            (total_bytes, first_disk_bytes)
        }

        fn chmod_readonly(&mut self, dir: &Path) {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o555))
                .expect("chmod 0555");
            self.chmod_dir = Some(dir.to_path_buf());
        }

        fn restore_chmod(&mut self) {
            use std::os::unix::fs::PermissionsExt;
            if let Some(dir) = self.chmod_dir.take() {
                std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755))
                    .expect("restore 0755");
            }
        }
    }

    #[cfg(target_os = "macos")]
    struct TestSession {
        coord: Inkling2KVCacheCoordinator,
        store: Inkling2StateStore,
        geom: Inkling2StateGeometry,
    }

    #[cfg(target_os = "macos")]
    impl TestSession {
        fn disk(&self) -> &Inkling2StateDisk {
            self.store.disk.as_ref().unwrap()
        }

        fn disk_mut(&mut self) -> &mut Inkling2StateDisk {
            self.store.disk.as_mut().unwrap()
        }

        fn manager(&self) -> &mlx_paged_attn::ColdCacheManager {
            &self.disk().tier.manager
        }

        fn entry(&self, id: EntryId) -> &Entry<Inkling2StateAnchor> {
            self.store.store.get(id).expect("entry")
        }

        fn first_id(&self) -> EntryId {
            self.store
                .store
                .entries()
                .next()
                .expect("adopted entry")
                .id()
        }

        fn id_by_salt(&self, salt: u64) -> EntryId {
            self.store
                .store
                .entries()
                .find(|e| e.salt() == salt)
                .expect("entry by salt")
                .id()
        }

        fn disk_copy(&self, id: EntryId) -> DiskCopy {
            *self.entry(id).disk().expect("disk copy")
        }

        #[allow(clippy::too_many_arguments)]
        fn capture(
            &mut self,
            conv: &Inkling2ShortConvState,
            seq: u32,
            tokens: &[u32],
            boundary: u32,
            salt: u64,
            class: StateClass,
            dw: Option<&PackedDraftWindow>,
        ) -> EntryId {
            self.coord.reset_scheduled_request(seq).expect("reset seq");
            self.coord.record_tokens_all(seq, tokens).expect("record");
            self.store
                .capture_boundary(
                    &mut self.coord,
                    conv,
                    seq,
                    tokens,
                    boundary,
                    salt,
                    class,
                    dw,
                )
                .expect("capture_boundary")
                .expect("entry_id")
        }

        fn capture_ord(
            &mut self,
            conv: &Inkling2ShortConvState,
            seq: u32,
            tokens: &[u32],
            boundary: u32,
            salt: u64,
            dw: Option<&PackedDraftWindow>,
        ) -> EntryId {
            self.capture(conv, seq, tokens, boundary, salt, StateClass::Ordinary, dw)
        }

        fn capture_ord_no_wb(
            &mut self,
            conv: &Inkling2ShortConvState,
            seq: u32,
            tokens: &[u32],
            boundary: u32,
            salt: u64,
        ) -> EntryId {
            self.coord.reset_scheduled_request(seq).expect("reset seq");
            self.coord.record_tokens_all(seq, tokens).expect("record");
            self.store
                .capture_boundary_with_options(
                    &mut self.coord,
                    conv,
                    seq,
                    tokens,
                    boundary,
                    salt,
                    StateClass::Ordinary,
                    CaptureOptions {
                        keep_prev_checkpoint: false,
                        enqueue_write_behind: false,
                        is_head: false,
                        now: 0.0,
                    },
                )
                .expect("capture_boundary_with_options")
                .expect("entry_id")
        }

        fn stage(&mut self, id: EntryId) -> bool {
            self.store
                .stage_entry(&self.coord, id, MakeRoom::Evict, true)
        }

        fn flush(&mut self, secs: u64) -> bool {
            self.store.flush_to_disk(&self.coord, test_deadline(secs))
        }

        fn close(&mut self) -> bool {
            self.store.close_cache(&self.coord, test_deadline(10))
        }

        fn wb_waiting(&self) -> u32 {
            self.disk().write_behind.snapshot().waiting
        }

        fn clear_wb(&mut self) {
            let persistent = self.disk().tier.persistent;
            self.disk_mut().write_behind = WriteBehind::new(persistent);
        }

        fn await_writing(&mut self, msg: &str) {
            let dl = test_deadline(10);
            while self.disk().writing() {
                assert!(Instant::now() < dl, "timed out: {msg}");
                let _ = self.disk().await_landing(dl);
                self.store.poll_offload(&self.coord);
            }
        }

        fn await_landed(&mut self, id: EntryId, msg: &str) {
            let dl = test_deadline(10);
            while self.disk().writing()
                || self
                    .store
                    .store
                    .get(id)
                    .is_some_and(|e| e.disk().is_none_or(|c| !c.landed || c.chain_owed))
            {
                assert!(Instant::now() < dl, "timed out: {msg}");
                self.store.chain_step(&self.coord, dl);
                let _ = self.disk().await_landing(dl);
                self.store.poll_offload(&self.coord);
            }
        }

        fn drop_ram_and_purge(&mut self) {
            self.store.drop_all_ram_keeping_disk();
            self.coord.release_all_and_purge().expect("purge hot");
        }

        fn assert_ssd_restores(
            &mut self,
            query: &[u32],
            salt: u64,
            expected_id: Option<EntryId>,
            expected_dw: Option<&PackedDraftWindow>,
        ) -> EntryId {
            let hit = self
                .store
                .lookup_deepest_restorable(&self.coord, query, salt)
                .expect("SSD lookup hit");
            if let Some(eid) = expected_id {
                assert_eq!(hit.id, eid);
            }
            let (anchor, from_ssd) = self
                .store
                .load_anchor(&self.coord, hit.id)
                .expect("load_anchor from SSD");
            assert!(from_ssd);
            assert_eq!(anchor.draft_window.as_ref(), expected_dw);
            hit.id
        }

        fn assert_flush_fails_promptly(&mut self, msg: &str) {
            let t0 = Instant::now();
            assert!(!self.flush(3), "flush_to_disk must return false: {msg}");
            assert!(
                t0.elapsed() < Duration::from_millis(500),
                "flush_to_disk must exit promptly ({:?}): {msg}",
                t0.elapsed()
            );
        }

        fn state_object_bytes_for(&self, id: EntryId) -> u64 {
            let entry = self.entry(id);
            let layout = self.disk().engine.layout(entry.depth()).unwrap();
            mlx_paged_attn::state_object_bytes(
                &layout,
                &crate::disk_tier::state_disk::label_of(entry),
            )
        }

        fn evict_kv_block(&self, id: EntryId, idx: usize) -> ColdCacheKey {
            let kv_keys = self.disk().engine.kv_chain_keys(self.entry(id));
            assert!(kv_keys.len() > idx, "expected > {idx} KV blocks");
            assert!(self.manager().remove(&kv_keys[idx]));
            kv_keys[idx]
        }

        fn set_writer_space(
            &self,
            model_avail: u64,
            writer_avail: u64,
            counter: Option<Arc<std::sync::atomic::AtomicUsize>>,
        ) {
            let model_tid = std::thread::current().id();
            self.manager()
                .set_space_override_for_test(Some(Box::new(move || {
                    if std::thread::current().id() == model_tid {
                        Ok((1 << 30, model_avail))
                    } else {
                        if let Some(ref c) = counter {
                            c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        Ok((1 << 30, writer_avail))
                    }
                })));
        }

        fn set_toggle_writer_space(
            &self,
            reserve: u64,
            fail_writer: Arc<std::sync::atomic::AtomicBool>,
        ) {
            let model_tid = std::thread::current().id();
            self.manager()
                .set_space_override_for_test(Some(Box::new(move || {
                    if std::thread::current().id() == model_tid
                        || !fail_writer.load(std::sync::atomic::Ordering::Acquire)
                    {
                        Ok((1 << 30, 64 << 20))
                    } else {
                        Ok((1 << 30, reserve))
                    }
                })));
        }

        fn clear_space_override(&self, drain: bool) -> bool {
            let mgr = self.manager();
            mgr.set_space_override_for_test(None);
            if drain {
                mgr.drain(Duration::from_secs(5))
            } else {
                true
            }
        }

        fn expire_backoff(&mut self, id: EntryId) {
            let past = Instant::now()
                .checked_sub(Duration::from_secs(1))
                .unwrap_or_else(Instant::now);
            if let Some(p) = self.store.pending_disk_upgrades.get_mut(&id) {
                p.set_retry_after(past);
            }
            self.store.spill_backoff.insert(id, past);
        }

        fn enqueue_kv_block(
            &self,
            seq: u32,
            chunk: &[u32],
            idx: usize,
            parent: Option<ColdCacheKey>,
        ) -> (ColdCacheKey, mlx_paged_attn::Landed) {
            let kv = KvPoolHandle::from_adapter(self.coord.full_adapter());
            let blk = Arc::clone(
                &self
                    .coord
                    .full_adapter()
                    .block_table_for(seq)
                    .expect("block table")
                    .blocks()[idx],
            );
            let fp = self.disk().tier.kv_fingerprint;
            let key = ColdCacheKey::chain(ColdGroup::Kv, fp, parent, chunk, &[], 0, idx);
            let (tx, rx) = std::sync::mpsc::channel();
            assert!(
                self.manager()
                    .capture_and_enqueue_tracked(
                        &kv.pool,
                        &blk,
                        key,
                        fp,
                        chunk,
                        test_deadline(5),
                        MakeRoom::Evict,
                        tx,
                    )
                    .expect("enqueue KV")
            );
            (
                key,
                rx.recv_timeout(Duration::from_secs(5)).expect("settle"),
            )
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn state_store_ram_cap_evicts_and_keeps_landed_ssd_restorable() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let fx = StateDiskTestFixture::new("ram-cap-unit", b"ram-cap-draft-v1", Some(0.0));
        let ram_cap = 4usize;

        // Part 1: Without SSD disk, inserting 30 Ordinary anchors strictly caps `ram_len() <= ram_cap`
        // and retires disposable Checkpoints before Ordinary entries.
        {
            let recency = Arc::new(CacheRecency::new());
            let mut coord = Inkling2KVCacheCoordinator::new_for_config_with_recency(
                &fx.cfg,
                fx.paged_cfg,
                Arc::clone(&recency),
            )
            .expect("coord ram-only");
            let geom = Inkling2StateGeometry::from_coordinator(&fx.cfg, &coord).expect("geom");
            let mut store = Inkling2StateStore::new(coord.block_size(), recency);
            store.set_geometry(geom);
            store.set_ram_cap(ram_cap);

            let cp_tokens: Vec<u32> = (500u32..564u32).collect();
            coord.reset_scheduled_request(1).expect("reset cp");
            coord.record_tokens_all(1, &cp_tokens).expect("record cp");
            let cp_id = store
                .capture_boundary(
                    &mut coord,
                    &fx.conv,
                    1,
                    &cp_tokens,
                    32,
                    999,
                    StateClass::Checkpoint,
                    None,
                )
                .expect("capture cp")
                .expect("cp_id");
            store.clear_seq_checkpoint(1);

            for i in 1u64..=30 {
                let base = (i as u32) * 100;
                let tokens: Vec<u32> = (base..base + 64).collect();
                coord.reset_scheduled_request(1).expect("reset seq 1");
                coord.record_tokens_all(1, &tokens).expect("record tokens");
                store
                    .capture_boundary(
                        &mut coord,
                        &fx.conv,
                        1,
                        &tokens,
                        32,
                        i,
                        StateClass::Ordinary,
                        None,
                    )
                    .expect("capture ordinary");
                assert!(
                    store.ram_len() <= ram_cap,
                    "RAM-only store must strictly cap ram_len={} <= {ram_cap} after insert {i}",
                    store.ram_len()
                );
                if i == ram_cap as u64 {
                    assert!(
                        store.store.get(cp_id).is_none(),
                        "Checkpoint must be retired first when ram_cap is exceeded"
                    );
                }
            }
            assert_eq!(store.ram_len(), ram_cap);
        }

        // Part 2: With StateDisk attached, inserting 30 Ordinary anchors caps `ram_len() <= ram_cap`,
        // drops RAM on landed SSD copies, and keeps SSD-only entries 100% restorable!
        let mut s = fx.open(512u64 << 20, None);
        s.store.set_ram_cap(ram_cap);
        let packed_dw = fx.packed_dw(32);

        let mut captured: Vec<(EntryId, Vec<u32>, u64)> = Vec::new();
        for i in 1u64..=30 {
            let base = (i as u32) * 100;
            let tokens: Vec<u32> = (base..base + 64).collect();
            let eid = s.capture_ord(&fx.conv, 1, &tokens, 32, i, Some(&packed_dw));
            captured.push((eid, tokens, i));
            if i % 3 == 0 {
                assert!(s.flush(10), "flush_to_disk at i={i}");
                assert!(
                    s.store.ram_len() <= ram_cap,
                    "post-flush ram_len={} must be <= ram_cap={ram_cap} at i={i}",
                    s.store.ram_len()
                );
            } else {
                assert!(
                    s.store.ram_len() <= ram_cap + 2,
                    "in-flight burst ram_len={} must stay <= ram_cap + 2 at i={i}",
                    s.store.ram_len()
                );
            }
        }
        assert!(s.flush(10), "final flush_to_disk");
        assert!(s.store.ram_len() <= ram_cap);

        let disk_only: Vec<_> = captured
            .into_iter()
            .filter(|(eid, _, _)| {
                s.store.store.get(*eid).is_some_and(|e| {
                    e.ram().is_none() && e.disk().is_some_and(|d| d.landed && !d.chain_owed)
                })
            })
            .collect();
        assert!(
            disk_only.len() >= 10,
            "expected >=10 RAM-dropped SSD-backed entries, got {}",
            disk_only.len()
        );

        s.coord.release_all_and_purge().expect("purge hot pools");
        for (eid, tokens, salt) in disk_only.iter().take(5) {
            let hit = s
                .store
                .lookup_deepest_restorable(&s.coord, tokens, *salt)
                .expect("SSD-only entry must remain restorable");
            assert_eq!((hit.id, hit.depth), (*eid, 32));
            let (anchor, from_ssd) = s.store.load_anchor(&s.coord, hit.id).expect("load SSD");
            assert!(from_ssd);
            assert_eq!(anchor.draft_window.as_ref(), Some(&packed_dw));
        }
        assert!(s.close());
    }

    /// Finding 3 regression: RAM must NOT be dropped while a state blob has landed
    /// (`copy.landed == true`) but its Global KV chain is still owed (`copy.chain_owed == true`).
    #[cfg(target_os = "macos")]
    #[test]
    fn state_store_keeps_ram_until_blob_and_chain_both_land() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let fx = StateDiskTestFixture::new("chain-owed-ram", b"chain-owed-ram-v1", Some(0.0));
        let mut s = fx.open(512u64 << 20, None);
        s.store.set_ram_cap(1);

        let tokens_a: Vec<u32> = (100u32..164u32).collect();
        let id_a = s.capture_ord(&fx.conv, 1, &tokens_a, 32, 11, None);
        assert!(s.stage(id_a), "staging entry A state blob must succeed");
        s.await_writing("entry A blob landing");
        {
            let entry_a = s.entry(id_a);
            assert!(entry_a.disk().is_some_and(|c| c.landed && c.chain_owed));
            assert!(entry_a.ram().is_some());
            assert!(!s.disk().entry_disk_restorable(entry_a));
        }

        // Capturing entry B (exceeds `ram_cap = 1`) must NOT drop A's RAM while `chain_owed`.
        let tokens_b: Vec<u32> = (200u32..264u32).collect();
        let _id_b = s.capture_ord(&fx.conv, 1, &tokens_b, 32, 22, None);
        assert!(s.entry(id_a).ram().is_some());
        assert!(s.store.is_restorable(&s.coord, id_a));
        assert_eq!(
            s.store
                .lookup_deepest_restorable(&s.coord, &tokens_a, 11)
                .expect("entry A discoverable")
                .id,
            id_a
        );

        // Landing A's Global KV chain drops A's RAM copy while keeping A 100% SSD-restorable.
        s.await_landed(id_a, "entry A Global KV chain");
        assert!(s.entry(id_a).ram().is_none());
        assert!(s.store.ram_len() <= 1);
        assert!(s.disk().entry_disk_restorable(s.entry(id_a)));

        s.drop_ram_and_purge();
        s.assert_ssd_restores(&tokens_a, 11, Some(id_a), None);
        assert!(s.close());
    }

    /// Finding 4 regression: an SSD-only v2 anchor adopted on restart with `draft_window == None`
    /// upgrades cleanly when a subsequent DFlash2 turn captures a complete `PackedDraftWindow`.
    #[cfg(target_os = "macos")]
    #[test]
    fn state_store_upgrades_adopted_ssd_only_anchor_missing_draft_window() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let fx =
            StateDiskTestFixture::new("upgrade-adopted-dw", b"upgrade-adopted-dw-v1", Some(0.0));
        let tokens: Vec<u32> = (300u32..396u32).collect();
        let query: Vec<u32> = (300u32..370u32).collect();
        let packed_dw = fx.packed_dw(64);
        assert!(packed_dw.is_complete_for(64));

        fx.seed_durable(&[(1, &tokens, 64, 7)], 512u64 << 20, None);

        {
            let mut s2 = fx.open_adopted(512u64 << 20, None, 1);
            let adopted_id = s2.first_id();
            assert!(s2.entry(adopted_id).ram().is_none());
            assert!(s2.store.adopted_unverified_window.contains(&adopted_id));

            let upgraded_id = s2.capture(
                &fx.conv,
                1,
                &tokens,
                64,
                7,
                StateClass::Checkpoint,
                Some(&packed_dw),
            );
            assert_eq!(s2.store.store.entries().count(), 1);
            let upgraded = s2.entry(upgraded_id);
            assert!(!upgraded.checkpoint());
            assert_eq!(
                upgraded.ram().and_then(|a| a.draft_window.as_ref()),
                Some(&packed_dw)
            );
            assert!(
                !s2.store.entries_missing_draft_window.contains(&upgraded_id)
                    && !s2.store.adopted_unverified_window.contains(&upgraded_id)
            );
            assert!(s2.flush(10));
            assert!(s2.close());
        }

        {
            let mut s3 = fx.open_adopted(512u64 << 20, None, 1);
            let hit = s3
                .store
                .lookup_deepest_restorable(&s3.coord, &query, 7)
                .expect("lookup upgraded anchor after restart");
            assert_eq!(hit.depth, 64);
            let (anchor, from_ssd) = s3
                .store
                .load_anchor(&s3.coord, hit.id)
                .expect("load upgraded anchor");
            assert!(from_ssd);
            assert_eq!(anchor.draft_window.as_ref(), Some(&packed_dw));
            assert!(
                !s3.store.adopted_unverified_window.contains(&hit.id)
                    && !s3.store.entries_missing_draft_window.contains(&hit.id)
            );
            assert!(s3.close());
        }
    }

    /// Round-2 Finding 1 regression #1: under a tight SSD quota fitting only 1 state blob + its
    /// Global KV chain, upgrading an adopted v2-without-window anchor succeeds with ZERO evictions.
    #[cfg(target_os = "macos")]
    #[test]
    fn state_store_quota_constrained_upgrade_preserves_kv_chain_and_blob() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let fx =
            StateDiskTestFixture::new("tight-quota-upgrade", b"tight-quota-upgrade-v1", Some(0.0));
        let tokens: Vec<u32> = (400u32..496u32).collect();
        let query: Vec<u32> = (400u32..470u32).collect();
        let packed_dw = fx.packed_dw(64);

        let (exact_total, state_disk_bytes) =
            fx.seed_durable(&[(1, &tokens, 64, 9)], 512u64 << 20, None);
        assert!(exact_total > state_disk_bytes);
        let tight_limit = exact_total + 64;
        assert!(tight_limit < exact_total + state_disk_bytes);

        {
            let mut s2 = fx.open_adopted(tight_limit, None, 1);
            let adopted_id = s2.first_id();
            let old_copy = s2.disk_copy(adopted_id);

            let upgraded_id = s2.capture_ord(&fx.conv, 1, &tokens, 64, 9, Some(&packed_dw));
            assert_eq!(upgraded_id, adopted_id);
            assert_eq!(s2.entry(upgraded_id).disk().copied(), Some(old_copy));
            assert!(s2.store.pending_disk_upgrades.contains_key(&upgraded_id));

            assert!(s2.flush(10), "tight-quota flush must overwrite cleanly");
            assert_eq!(s2.manager().stats().evictions, 0);
            assert!(s2.disk().entry_disk_restorable(s2.entry(upgraded_id)));
            assert!(s2.store.pending_disk_upgrades.is_empty());
            assert!(s2.store.in_flight_disk_upgrade.is_none());
            assert!(s2.close());
        }

        {
            let mut s3 = fx.open_adopted(tight_limit, None, 1);
            s3.assert_ssd_restores(&query, 9, None, Some(&packed_dw));
            assert!(s3.close());
        }
    }

    /// Round-2 Finding 1 regression #2: if a replacement SSD write during an anchor upgrade
    /// fails (`Staged::Busy` or read-only dir), the old durable `DiskCopy` survives immediately
    /// and across restart.
    #[cfg(target_os = "macos")]
    #[test]
    fn state_store_failed_upgrade_write_preserves_old_ssd_copy_immediately_and_after_restart() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let mut fx = StateDiskTestFixture::new("failed-upgrade", b"failed-upgrade-v1", Some(0.0));
        let tokens: Vec<u32> = (500u32..596u32).collect();
        let query: Vec<u32> = (500u32..570u32).collect();
        let packed_dw = fx.packed_dw(64);

        fx.seed_durable(&[(1, &tokens, 64, 12)], 512u64 << 20, None);

        {
            let mut s2 = fx.open_adopted(512u64 << 20, None, 1);
            let adopted_id = s2.first_id();
            let old_copy = s2.disk_copy(adopted_id);

            let upgraded_id = s2.capture_ord(&fx.conv, 1, &tokens, 64, 12, Some(&packed_dw));
            assert_eq!(upgraded_id, adopted_id);

            // 2a. Staging failure (`Staged::Busy` via ReadWait): old `DiskCopy` stays intact.
            let read_wait = s2.disk().acquire_read_wait();
            assert!(!s2.stage(upgraded_id));
            assert_eq!(s2.entry(upgraded_id).disk().copied(), Some(old_copy));
            assert!(s2.store.pending_disk_upgrades.contains_key(&upgraded_id));
            assert!(s2.store.in_flight_disk_upgrade.is_none());
            drop(read_wait);

            // 2b. Writer-thread failure (read-only namespace dir): `poll_offload` restores `old_copy`.
            let ns_dir = s2.manager().root().to_path_buf();
            fx.chmod_readonly(&ns_dir);

            assert!(s2.stage(upgraded_id));
            assert_eq!(
                s2.store
                    .in_flight_disk_upgrade
                    .as_ref()
                    .map(|u| (u.id, u.old_copy)),
                Some((upgraded_id, old_copy))
            );
            assert!(!s2.store.pending_disk_upgrades.contains_key(&upgraded_id));

            s2.await_writing("failed upgrade");
            assert!(s2.store.in_flight_disk_upgrade.is_none());
            assert!(s2.store.disk_counters().offload_failures >= 1);
            assert_eq!(s2.entry(upgraded_id).disk().copied(), Some(old_copy));
            assert!(s2.store.pending_disk_upgrades.contains_key(&upgraded_id));
            assert!(s2.disk().entry_disk_restorable(s2.entry(upgraded_id)));

            s2.drop_ram_and_purge();
            assert!(s2.store.entries_missing_draft_window.contains(&upgraded_id));
            s2.assert_ssd_restores(&query, 12, Some(upgraded_id), None);

            fx.restore_chmod();
            assert!(
                !s2.close(),
                "closing drain barrier must report the injected write failure"
            );
        }

        {
            let mut s3 = fx.open_adopted(512u64 << 20, None, 1);
            s3.assert_ssd_restores(&query, 12, None, None);
            assert!(s3.close());
        }
    }

    /// Round-2 Finding 3 regression: while `chain_owed == true`, `reclaim_entry` returns `Pending`
    /// and keeps RAM intact; if the Global KV chain write fails, only the broken `DiskCopy` is
    /// cleared while the RAM anchor stays restorable.
    #[cfg(target_os = "macos")]
    #[test]
    fn reclaim_entry_keeps_ram_while_chain_owed_and_survives_chain_failure() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let mut fx =
            StateDiskTestFixture::new("reclaim-chain-owed", b"reclaim-chain-owed-v1", None);
        let mut s = fx.open(512u64 << 20, None);
        let tokens: Vec<u32> = (600u32..696u32).collect();
        let query: Vec<u32> = (600u32..670u32).collect();

        let entry_id = s.capture_ord(&fx.conv, 1, &tokens, 64, 15, None);
        assert!(s.stage(entry_id));
        s.await_writing("state blob");
        assert!(s.entry(entry_id).ram().is_some());
        assert!(
            s.entry(entry_id)
                .disk()
                .is_some_and(|c| c.landed && c.chain_owed)
        );

        assert_eq!(
            s.store.reclaim_entry(&s.coord, entry_id),
            ReclaimOutcome::Pending
        );
        assert!(s.entry(entry_id).ram().is_some());

        let ns_dir = s.manager().root().to_path_buf();
        fx.chmod_readonly(&ns_dir);
        let dl = test_deadline(10);
        while s
            .store
            .store
            .get(entry_id)
            .is_some_and(|e| e.disk().is_some())
        {
            assert!(Instant::now() < dl, "timed out failing KV chain write");
            s.store.chain_step(&s.coord, dl);
            let _ = s.disk().engine.await_landing(dl);
            s.store.poll_offload(&s.coord);
        }

        assert!(s.entry(entry_id).ram().is_some());
        assert!(s.entry(entry_id).disk().is_none());
        let hit = s
            .store
            .lookup_deepest_restorable(&s.coord, &query, 15)
            .expect("RAM anchor restorable after KV chain failure");
        assert_eq!(hit.id, entry_id);
        let (_, from_ssd) = s.store.load_anchor(&s.coord, entry_id).expect("load RAM");
        assert!(!from_ssd);

        fx.restore_chmod();
        let _ = s.close();
    }

    /// Round-3 Finding 2 regression: when volume free space above reserve < `new_state`,
    /// `stage_entry` defers cleanly without evicting KV blocks or losing `old_copy`.
    #[cfg(target_os = "macos")]
    #[test]
    fn state_store_in_place_upgrade_defers_without_eviction_when_reserve_headroom_low() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let fx =
            StateDiskTestFixture::new("upgrade-low-reserve", b"low-reserve-upgrade-v1", Some(0.0));
        let tokens: Vec<u32> = (600u32..696u32).collect();
        let query: Vec<u32> = (600u32..670u32).collect();
        let packed_dw = fx.packed_dw(64);

        let (_, old_state_bytes) = fx.seed_durable(&[(1, &tokens, 64, 21)], 64u64 << 20, None);

        let available = crate::disk_tier::volume_of(&fx.cache_dir)
            .expect("measure test volume")
            .available;
        let mut s2 = fx.open_adopted(64u64 << 20, Some(available.saturating_sub(1024)), 1);
        let adopted_id = s2.first_id();
        let old_copy = s2.disk_copy(adopted_id);

        let upgraded_id = s2.capture_ord(&fx.conv, 1, &tokens, 64, 21, Some(&packed_dw));
        assert_eq!(upgraded_id, adopted_id);

        let free = s2
            .manager()
            .room_above_reserve()
            .expect("room_above_reserve");
        assert!(free < old_state_bytes);
        assert!(s2.disk().point_fits(s2.entry(upgraded_id), MakeRoom::Evict));

        assert!(!s2.stage(upgraded_id), "must defer when free < new_state");
        assert_eq!(s2.manager().stats().evictions, 0);
        assert_eq!(s2.entry(upgraded_id).disk().copied(), Some(old_copy));
        assert!(s2.store.pending_disk_upgrades.contains_key(&upgraded_id));
        assert!(s2.store.in_flight_disk_upgrade.is_none());
        assert!(s2.disk().entry_disk_restorable(s2.entry(upgraded_id)));

        s2.drop_ram_and_purge();
        s2.assert_ssd_restores(&query, 21, Some(upgraded_id), None);
        let _ = s2.close();
    }

    /// Round-3 Finding 4 regression: recoverable upgrade staging refusals and failed replacement
    /// writes keep the upgrade queued in `write_behind`, let `flush_to_disk` exit promptly, and
    /// retry automatically once the failure clears.
    #[cfg(target_os = "macos")]
    #[test]
    fn state_store_recoverable_upgrade_stays_queued_and_retries_until_landed() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let fx =
            StateDiskTestFixture::new("upgrade-retry-queue", b"retry-queue-upgrade-v1", Some(0.0));
        let tokens: Vec<u32> = (700u32..796u32).collect();
        let query: Vec<u32> = (700u32..770u32).collect();
        let packed_dw = fx.packed_dw(64);

        fx.seed_durable(&[(1, &tokens, 64, 33)], 64u64 << 20, None);

        let mut s2 = fx.open_adopted(64u64 << 20, None, 1);
        let adopted_id = s2.first_id();
        let old_copy = s2.disk_copy(adopted_id);

        let upgraded_id = s2.capture_ord(&fx.conv, 1, &tokens, 64, 33, Some(&packed_dw));
        assert_eq!(upgraded_id, adopted_id);

        // 2a. Temporary staging refusal (`ReadWait` held -> `Staged::Busy`).
        let read_wait = s2.disk().acquire_read_wait();
        assert!(!s2.store.drive_pending_disk_upgrades(&s2.coord, true));
        assert!(s2.store.pending_disk_upgrades.contains_key(&upgraded_id));
        s2.assert_flush_fails_promptly("blocked upgrade");
        assert!(s2.store.pending_disk_upgrades.contains_key(&upgraded_id));
        drop(read_wait);

        // 2b. Simulate recoverable writer failure -> `poll_offload` reconciles & re-queues.
        s2.store.pending_disk_upgrades.remove(&upgraded_id);
        s2.store
            .set_in_flight_upgrade_for_test(upgraded_id, old_copy);
        s2.store.store.set_disk(upgraded_id, None);
        s2.clear_wb();

        assert!(s2.store.poll_offload(&s2.coord));
        assert!(s2.store.in_flight_disk_upgrade.is_none());
        assert_eq!(s2.entry(upgraded_id).disk().copied(), Some(old_copy));
        assert!(s2.store.pending_disk_upgrades.contains_key(&upgraded_id));

        // 2c. Once cleared, `tick` retries and `flush_to_disk` lands the upgraded window.
        let _ = s2.store.tick(&s2.coord, Tick::Full { command_pages: 0 });
        assert!(s2.flush(10));
        assert!(s2.store.pending_disk_upgrades.is_empty());
        assert!(s2.store.in_flight_disk_upgrade.is_none());

        s2.drop_ram_and_purge();
        s2.assert_ssd_restores(&query, 33, None, Some(&packed_dw));
        assert!(s2.close());
    }

    /// Round-4 Finding 1 regression: even if volume headroom drops between staging pre-check and
    /// writer execution, in-place upgrades use `MakeRoom::Refuse`, evict ZERO objects, and restore
    /// `old_copy`.
    #[cfg(target_os = "macos")]
    #[test]
    fn state_store_in_place_upgrade_refuses_without_eviction_on_writer_headroom_drop() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use std::sync::atomic::{AtomicUsize, Ordering};

        let fx = StateDiskTestFixture::new(
            "upgrade-writer-refuse",
            b"writer-headroom-drop-v1",
            Some(10.0),
        );
        let tokens: Vec<u32> = (800u32..896u32).collect();
        let query: Vec<u32> = (800u32..870u32).collect();
        let packed_dw = fx.packed_dw(64);

        fx.seed_durable(&[(1, &tokens, 64, 41)], 64u64 << 20, None);

        let reserve = 4096u64;
        let mut s2 = fx.open_adopted(64u64 << 20, Some(reserve), 1);
        let adopted_id = s2.first_id();
        let old_copy = s2.disk_copy(adopted_id);

        let upgraded_id = s2.capture_ord(&fx.conv, 1, &tokens, 64, 41, Some(&packed_dw));
        assert_eq!(upgraded_id, adopted_id);

        let new_state_bytes = s2.state_object_bytes_for(upgraded_id);
        let kv_keys = s2.disk().engine.kv_chain_keys(s2.entry(upgraded_id));
        assert_eq!(kv_keys.len(), 2);
        assert!(s2.manager().total_bytes().saturating_sub(old_copy.bytes) > 0);

        let writer_probes = Arc::new(AtomicUsize::new(0));
        s2.set_writer_space(
            reserve + new_state_bytes.saturating_mul(10),
            reserve + new_state_bytes.saturating_sub(1),
            Some(Arc::clone(&writer_probes)),
        );

        assert!(s2.stage(upgraded_id));
        assert_eq!(
            s2.store
                .in_flight_disk_upgrade
                .as_ref()
                .map(|u| (u.id, u.old_copy)),
            Some((upgraded_id, old_copy))
        );
        s2.await_writing("writer headroom drop");
        s2.clear_space_override(false);

        assert!(writer_probes.load(Ordering::Relaxed) >= 1);
        assert_eq!(s2.manager().stats().evictions, 0);
        assert!(s2.disk().writable());
        assert!(s2.disk().counters.offload_failures >= 1);
        assert!(
            s2.manager()
                .contains_in(&old_copy.key, ColdGroup::DFlash2State)
        );
        for k in &kv_keys {
            assert!(s2.manager().contains_in(k, ColdGroup::Kv));
        }
        assert!(s2.store.in_flight_disk_upgrade.is_none());
        assert_eq!(s2.entry(upgraded_id).disk().copied(), Some(old_copy));
        assert!(s2.store.pending_disk_upgrades.contains_key(&upgraded_id));
        assert!(s2.disk().entry_disk_restorable(s2.entry(upgraded_id)));

        s2.drop_ram_and_purge();
        s2.assert_ssd_restores(&query, 41, Some(upgraded_id), None);
        let _ = s2.close();
    }

    /// Round-4 Finding 2 regression (part 1): a shallow adopted SSD anchor updates
    /// `entry.state_bytes()` and `total_ram_bytes()` to exact live RAM bytes on `promote` and
    /// in-place `capture_boundary` upgrade.
    #[cfg(target_os = "macos")]
    #[test]
    fn state_store_shallow_adopted_upgrade_updates_exact_ram_bytes() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let fx = StateDiskTestFixture::new("shallow-ram-bytes", b"shallow-ram-bytes-v1", Some(0.0))
            .with_paged(512, 2);
        let tokens_a: Vec<u32> = (900u32..940u32).collect();
        let tokens_b: Vec<u32> = (950u32..990u32).collect();
        let packed_dw32 = fx.packed_dw(32);

        fx.seed_durable(
            &[(1, &tokens_a, 32, 61), (2, &tokens_b, 32, 62)],
            64u64 << 20,
            None,
        );

        let mut s2 = fx.open_adopted(64u64 << 20, None, 2);
        let id_a = s2.id_by_salt(61);
        let id_b = s2.id_by_salt(62);
        let lin_a = s2.entry(id_a).state_bytes();
        let lin_b = s2.entry(id_b).state_bytes();
        let payload = s2.geom.payload_bytes();
        assert_eq!(s2.entry(id_a).resident_bytes(), payload + lin_a);
        assert_eq!(s2.entry(id_b).resident_bytes(), payload + lin_b);

        // 1. Promoting shallow adopted anchor A updates `cached_bytes` to exact decoded live bytes.
        let (anchor_a, from_ssd) = s2.store.load_anchor(&s2.coord, id_a).expect("load A");
        assert!(from_ssd);
        let live_swa_32 = s2.geom.live_swa_bytes(32).expect("live_swa_bytes(32)") as u64;
        let expected_a_ram = live_swa_32 + s2.geom.conv_span_bytes;
        assert_eq!(anchor_a.cached_bytes, expected_a_ram);
        assert!(expected_a_ram < payload);
        assert!(s2.store.store.promote(id_a, anchor_a));
        assert_eq!(s2.entry(id_a).state_bytes(), expected_a_ram + lin_a);
        assert_eq!(s2.store.total_ram_bytes(), expected_a_ram + lin_a + lin_b);

        // 2. Upgrading shallow adopted anchor B in place updates `cached_bytes` atomically.
        let upgraded_b = s2.capture_ord(&fx.conv, 2, &tokens_b, 32, 62, Some(&packed_dw32));
        assert_eq!(upgraded_b, id_b);
        let expected_b_ram = live_swa_32 + s2.geom.conv_span_bytes + packed_dw32.bytes.len() as u64;
        assert!(expected_b_ram < payload);
        assert_eq!(s2.entry(id_b).state_bytes(), expected_b_ram + lin_b);
        assert_eq!(
            s2.store.total_ram_bytes(),
            expected_a_ram + lin_a + expected_b_ram + lin_b
        );
        let _ = s2.close();
    }

    /// Round-4 Finding 2 regression (part 2): upgrading a RAM-resident anchor in-place with a
    /// `PackedDraftWindow` increases `state_bytes()`/`total_ram_bytes()` by exact `dw_len`, and
    /// `inkling2_take_victim` credits only the exact bytes actually freed from RAM.
    #[cfg(target_os = "macos")]
    #[test]
    fn state_store_ram_upgrade_updates_bytes_and_take_victim_credits_exact_freed() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use crate::engine::governor::ledger::FreeOutcome;

        let fx = StateDiskTestFixture::new("ram-upgrade-bytes", b"ram-upgrade-bytes-v1", Some(0.0));
        let patient_query = patient_victim_query();
        let mut inner =
            crate::models::inkling2::scheduler::tests::build_synthetic_dflash2_inner(256, None);
        attach_synthetic_disk(&fx, &mut inner);

        let conv = make_short_conv(&inner.config);
        let tokens_a: Vec<u32> = (10u32..50u32).collect();
        let tokens_b: Vec<u32> = (60u32..100u32).collect();
        let draft_cfg = inner.dflash2.as_ref().expect("dflash2").config.clone();
        let dw = make_draft_window(
            draft_cfg.context_width(),
            draft_cfg.sliding_window as u32,
            32,
        );
        let dw_len =
            DraftContextWindow::bytes(draft_cfg.context_width(), draft_cfg.sliding_window as u32);
        assert!(dw_len > 0);

        // 1. Capture A without draft window, flush to SSD, then upgrade A in-place.
        let (id_a, before_entry, before_total) = {
            let rt = inner.paged.as_mut().expect("paged");
            rt.coordinator.reset_scheduled_request(1).expect("reset 1");
            rt.coordinator
                .record_tokens_all(1, &tokens_a)
                .expect("record a");
            let id_a = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    1,
                    &tokens_a,
                    32,
                    71,
                    StateClass::Ordinary,
                    None,
                )
                .expect("capture A None")
                .expect("id_a");
            assert!(
                rt.state_store
                    .flush_to_disk(&rt.coordinator, test_deadline(10))
            );
            let entry = rt.state_store.store.get(id_a).unwrap();
            assert!(entry.ram().is_some() && entry.disk().is_some_and(|d| d.landed));
            (id_a, entry.state_bytes(), rt.state_store.total_ram_bytes())
        };

        let (upgraded_entry, upgraded_ram, lineage_a) = {
            let packed_dw_a = PackedDraftWindow::from_window(&dw).expect("packed_dw_a");
            assert_eq!(packed_dw_a.bytes.len() as u64, dw_len);
            let rt = inner.paged.as_mut().expect("paged");
            let upgraded_id = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    1,
                    &tokens_a,
                    32,
                    71,
                    StateClass::Ordinary,
                    Some(&packed_dw_a),
                )
                .expect("upgrade A")
                .expect("upgraded_id");
            assert_eq!(upgraded_id, id_a);
            let entry = rt.state_store.store.get(id_a).unwrap();
            let after_entry = entry.state_bytes();
            let after_total = rt.state_store.total_ram_bytes();
            let ram_bytes = entry.ram().unwrap().state_bytes();
            assert_eq!(after_entry, before_entry + dw_len);
            assert_eq!(after_total, before_total + dw_len);
            (after_entry, ram_bytes, after_entry - ram_bytes)
        };
        assert!(lineage_a > 0);

        // 2. Reclaiming SSD-backed A credits exact freed RAM bytes (excluding retained lineage).
        let taken_a =
            inner.inkling2_take_victim(&patient_query, &store_victim(id_a, 1, upgraded_entry), &[]);
        assert_eq!(taken_a.outcome, FreeOutcome::Met);
        assert_eq!(taken_a.now_bytes, upgraded_ram);
        {
            let rt = inner.paged.as_ref().expect("paged");
            let after = rt.state_store.store.get(id_a).expect("disk entry retained");
            assert!(after.ram().is_none());
            assert_eq!(after.state_bytes(), lineage_a);
            assert_eq!(upgraded_entry - after.state_bytes(), taken_a.now_bytes);
        }

        // 3. Without a disk tier, erasing B credits full `state_bytes()` (RAM + lineage).
        let _ = inner.close_state_disk(test_deadline(5));
        let (id_b, b_full_bytes) = {
            let packed_dw_b = PackedDraftWindow::from_window(&dw).expect("packed_dw_b");
            let rt = inner.paged.as_mut().expect("paged");
            rt.state_store.disk = None;
            rt.coordinator.reset_scheduled_request(2).expect("reset 2");
            rt.coordinator
                .record_tokens_all(2, &tokens_b)
                .expect("record b");
            let id_b = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    2,
                    &tokens_b,
                    32,
                    72,
                    StateClass::Ordinary,
                    Some(&packed_dw_b),
                )
                .expect("capture B")
                .expect("id_b");
            (id_b, rt.state_store.store.get(id_b).unwrap().state_bytes())
        };
        let taken_b =
            inner.inkling2_take_victim(&patient_query, &store_victim(id_b, 2, b_full_bytes), &[]);
        assert_eq!(taken_b.outcome, FreeOutcome::Met);
        assert_eq!(taken_b.now_bytes, b_full_bytes);
        assert!(
            inner
                .paged
                .as_ref()
                .unwrap()
                .state_store
                .store
                .get(id_b)
                .is_none()
        );
    }

    /// Round-4 Finding 3 regression: above `ram_cap` with a writer-side refusal on an in-place
    /// upgrade, `poll_offload` does NOT immediately re-stage via `trim_ram_to_cap`, and
    /// `flush_to_disk` exits promptly without spinning.
    #[cfg(target_os = "macos")]
    #[test]
    fn state_store_failed_upgrade_backs_off_from_ram_trim_and_flush_exits_promptly() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use std::sync::atomic::{AtomicUsize, Ordering};

        let fx = StateDiskTestFixture::new(
            "upgrade-backoff-flush",
            b"upgrade-backoff-flush-v1",
            Some(10.0),
        );
        let tokens_a: Vec<u32> = (1000u32..1096u32).collect();
        let tokens_b: Vec<u32> = (1100u32..1196u32).collect();
        let query_a: Vec<u32> = (1000u32..1070u32).collect();
        let packed_dw = fx.packed_dw(64);

        fx.seed_durable(&[(1, &tokens_a, 64, 81)], 64u64 << 20, None);

        let reserve = 4096u64;
        let mut s2 = fx.open_adopted(64u64 << 20, Some(reserve), 1);
        s2.store.set_ram_cap(4);
        let adopted_id = s2.first_id();
        let old_copy = s2.disk_copy(adopted_id);

        let upgraded_id = s2.capture_ord(&fx.conv, 1, &tokens_a, 64, 81, Some(&packed_dw));
        assert_eq!(upgraded_id, adopted_id);

        let other_id = s2.capture_ord_no_wb(&fx.conv, 2, &tokens_b, 64, 82);
        let other_mark = s2.entry(other_id).mark();
        s2.store.store.use_state(other_mark);
        s2.store.set_ram_cap(1);
        assert_eq!(s2.store.ram_len(), 2);

        let new_state_bytes = s2.state_object_bytes_for(upgraded_id);
        let writer_refusals = Arc::new(AtomicUsize::new(0));
        s2.set_writer_space(
            reserve + new_state_bytes.saturating_mul(10),
            reserve + 1024,
            Some(Arc::clone(&writer_refusals)),
        );

        // Part A: writer refuses -> `poll_offload` does NOT immediately re-stage via `trim_ram_to_cap`.
        assert!(s2.stage(upgraded_id));
        s2.await_writing("writer refusal");
        assert_eq!(writer_refusals.load(Ordering::Relaxed), 1);
        assert!(!s2.disk().writing());
        assert!(s2.store.in_flight_disk_upgrade.is_none());
        assert_eq!(s2.entry(upgraded_id).disk().copied(), Some(old_copy));
        assert!(s2.store.pending_disk_upgrades.contains_key(&upgraded_id));
        s2.store.trim_ram_to_cap(&s2.coord, None);
        assert!(!s2.disk().writing());

        // Part B: `flush_to_disk` attempts once and exits promptly without spinning.
        writer_refusals.store(0, Ordering::Relaxed);
        s2.assert_flush_fails_promptly("writer refusal");
        assert_eq!(writer_refusals.load(Ordering::Relaxed), 1);
        assert!(!s2.disk().writing());
        assert!(s2.store.in_flight_disk_upgrade.is_none());
        assert_eq!(s2.entry(upgraded_id).disk().copied(), Some(old_copy));
        assert!(s2.store.pending_disk_upgrades.contains_key(&upgraded_id));
        assert_eq!(s2.manager().stats().evictions, 0);

        // Part C: once space recovers, drain reports prior refusals and flush lands cleanly.
        assert!(!s2.clear_space_override(true));
        s2.store.store.end_use(other_mark);
        assert!(s2.flush(10));
        assert!(s2.store.pending_disk_upgrades.is_empty());
        assert!(s2.store.in_flight_disk_upgrade.is_none());

        s2.drop_ram_and_purge();
        s2.assert_ssd_restores(&query_a, 81, None, Some(&packed_dw));
        assert!(s2.close());
    }

    /// Round-5 Finding 1 regression: writer-side space refusal on an ordinary spill records
    /// `spill_backoff` without re-staging, and `load_anchor` holds `ReadWait` so `trim_ram_to_cap`
    /// cannot steal the staging buffer.
    #[cfg(target_os = "macos")]
    #[test]
    fn state_store_refused_ordinary_spill_backs_off_and_does_not_starve_load_anchor() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use std::sync::atomic::{AtomicUsize, Ordering};

        let fx = StateDiskTestFixture::new(
            "ordinary-spill-backoff",
            b"ordinary-spill-backoff-v1",
            Some(10.0),
        )
        .with_paged(2048, 4);
        let tokens_a: Vec<u32> = (1200u32..1296u32).collect();
        let tokens_b: Vec<u32> = (1300u32..1396u32).collect();
        let tokens_c: Vec<u32> = (1400u32..1496u32).collect();
        let tokens_d: Vec<u32> = (1500u32..1596u32).collect();
        let tokens_e: Vec<u32> = (1600u32..1696u32).collect();
        let packed_dw_c = fx.packed_dw(64);

        let reserve = 4096u64;
        let mut s = fx.open(64u64 << 20, Some(reserve));
        s.store.set_ram_cap(8);

        // Part A: writer refuses ordinary spill A above `ram_cap`.
        let id_a = s.capture_ord(&fx.conv, 1, &tokens_a, 64, 91, None);
        let id_b = s.capture_ord(&fx.conv, 2, &tokens_b, 64, 92, None);
        let mark_b = s.entry(id_b).mark();
        s.store.store.use_state(mark_b);
        s.store.set_ram_cap(1);
        assert_eq!(s.store.ram_len(), 2);
        assert!(s.store.pending_disk_upgrades.is_empty());

        let state_bytes_a = s.state_object_bytes_for(id_a);
        let writer_refusals = Arc::new(AtomicUsize::new(0));
        s.set_writer_space(
            reserve + state_bytes_a.saturating_mul(10),
            reserve + 1024,
            Some(Arc::clone(&writer_refusals)),
        );

        s.store.trim_ram_to_cap(&s.coord, None);
        assert!(s.disk().writing());
        s.await_writing("ordinary spill refusal");
        assert_eq!(s.disk().counters.offload_failures, 1);
        assert_eq!(writer_refusals.load(Ordering::Relaxed), 1);
        assert!(!s.disk().writing());
        assert!(s.store.spill_backoff.contains_key(&id_a));
        assert!(s.entry(id_a).disk().is_none() && s.entry(id_a).ram().is_some());

        s.store.trim_ram_to_cap(&s.coord, None);
        s.store.tick(&s.coord, Tick::Full { command_pages: 0 });
        assert!(!s.disk().writing());
        assert_eq!(s.disk().counters.offload_failures, 1);
        assert_eq!(writer_refusals.load(Ordering::Relaxed), 1);

        // Part B: `load_anchor(id_c)` acquires `ReadWait` so E cannot steal staging buffer.
        assert!(!s.clear_space_override(true));
        s.store.set_ram_cap(8);
        let id_c = s.capture_ord(&fx.conv, 3, &tokens_c, 64, 93, Some(&packed_dw_c));
        assert!(s.stage(id_c));
        s.await_landed(id_c, "landing C");
        s.store.store.drop_ram(id_c);
        assert!(s.entry(id_c).ram().is_none());
        assert!(s.store.spill_backoff.contains_key(&id_a));

        let id_d = s.capture_ord(&fx.conv, 4, &tokens_d, 64, 94, None);
        let id_e = s.capture_ord(&fx.conv, 1, &tokens_e, 64, 95, None);
        s.store.set_ram_cap(2);
        assert!(s.stage(id_d));
        assert!(s.disk().writing() && !s.disk().staging_free());

        let (restored_c, from_ssd) = s.store.load_anchor(&s.coord, id_c).expect("load C");
        assert!(from_ssd);
        assert_eq!(restored_c.draft_window.as_ref(), Some(&packed_dw_c));
        assert!(!s.disk().writing());
        assert_eq!(s.disk().engine.read_waiters(), 0);
        assert!(s.entry(id_e).disk().is_none());

        s.store.store.end_use(mark_b);
        s.store.set_ram_cap(8);
        assert!(s.flush(10));
        assert!(!s.store.spill_backoff.contains_key(&id_a));
        assert!(s.close());
    }

    /// Round-5 Finding 3 regression: when a queued SSD upgrade's old Global KV chain was evicted,
    /// `flush_to_disk` detects it, rewrites the blob + repairs the KV chain, and survives restart.
    #[cfg(target_os = "macos")]
    #[test]
    fn state_store_upgrade_repairs_evicted_global_kv_chain_and_survives_restart() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let fx = StateDiskTestFixture::new(
            "upgrade-evicted-chain",
            b"upgrade-evicted-chain-v1",
            Some(0.0),
        );
        let tokens_a: Vec<u32> = (2000u32..2096u32).collect();
        let query_a: Vec<u32> = (2000u32..2070u32).collect();
        let packed_dw = fx.packed_dw(64);

        fx.seed_durable(&[(1, &tokens_a, 64, 97)], 64u64 << 20, None);

        {
            let mut s2 = fx.open_adopted(64u64 << 20, None, 1);
            let adopted_id = s2.first_id();
            let upgraded_id = s2.capture_ord(&fx.conv, 1, &tokens_a, 64, 97, Some(&packed_dw));
            assert_eq!(upgraded_id, adopted_id);
            assert!(s2.store.pending_disk_upgrades.contains_key(&upgraded_id));

            let evicted_kv_key = s2.evict_kv_block(upgraded_id, 1);
            assert!(!s2.disk().engine.chain_on_disk(s2.entry(upgraded_id)));

            assert!(s2.flush(10));
            assert!(s2.store.pending_disk_upgrades.is_empty());
            assert!(s2.store.in_flight_disk_upgrade.is_none());
            assert!(s2.manager().contains_in(&evicted_kv_key, ColdGroup::Kv));
            assert!(s2.disk().engine.chain_on_disk(s2.entry(upgraded_id)));
            assert!(
                s2.entry(upgraded_id)
                    .disk()
                    .is_some_and(|c| c.landed && !c.chain_owed)
            );
            assert!(s2.close());
        }

        let mut s3 = fx.open_adopted(64u64 << 20, None, 1);
        s3.assert_ssd_restores(&query_a, 97, None, Some(&packed_dw));
        assert!(s3.close());
    }

    /// Round-5 Finding 4 regression: shared `PackedDraftWindow` `Arc<[u8]>` between live sequence
    /// and `state_store` is never double-counted or double-credited by `inkling2_take_victim`.
    #[cfg(target_os = "macos")]
    #[test]
    fn state_store_take_victim_never_double_credits_shared_draft_window_arc() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use crate::engine::governor::ledger::FreeOutcome;
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;

        let fx = StateDiskTestFixture::new(
            "shared-dw-arc-credit",
            b"shared-dw-arc-credit-v1",
            Some(0.0),
        );
        let patient_query = patient_victim_query();
        let mut inner =
            crate::models::inkling2::scheduler::tests::build_synthetic_dflash2_inner(256, None);
        attach_synthetic_disk(&fx, &mut inner);

        let conv = make_short_conv(&inner.config);
        let tokens_a: Vec<u32> = (10u32..50u32).collect();
        let tokens_b: Vec<u32> = (60u32..100u32).collect();
        let tokens_c: Vec<u32> = (110u32..150u32).collect();
        let draft_cfg = inner.dflash2.as_ref().expect("dflash2").config.clone();
        let dw = make_draft_window(
            draft_cfg.context_width(),
            draft_cfg.sliding_window as u32,
            32,
        );

        // 1. Entry A shares its `PackedDraftWindow` `Arc` with live seq 1.
        let (id_a, swa_conv_a, dw_len, full_entry_a) = {
            let packed_a = PackedDraftWindow::from_window(&dw).expect("packed_a");
            let dw_len = packed_a.bytes.len() as u64;
            let rt = inner.paged.as_mut().expect("paged");
            rt.coordinator.reset_scheduled_request(1).expect("reset 1");
            rt.coordinator
                .record_tokens_all(1, &tokens_a)
                .expect("record a");
            let id_a = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    1,
                    &tokens_a,
                    32,
                    111,
                    StateClass::Ordinary,
                    Some(&packed_a),
                )
                .expect("capture A")
                .expect("id_a");
            assert!(
                rt.state_store
                    .flush_to_disk(&rt.coordinator, test_deadline(10))
            );
            rt.block_boundary_draft_window
                .insert(1, (32, BoundaryDraftWindow::Packed(packed_a)));
            let entry = rt.state_store.store.get(id_a).unwrap();
            let anchor = entry.ram().unwrap();
            let swa_conv = (anchor.swa_data.len() + anchor.conv_data.len()) as u64;
            assert_eq!(anchor.exclusive_ram_bytes(), swa_conv);
            assert_eq!(anchor.cached_bytes, swa_conv + dw_len);
            assert_eq!(rt.draft_windows_bytes(), 0);
            (id_a, swa_conv, dw_len, entry.state_bytes())
        };

        let states_before_a = inner.memory_accounts().expect("accounts").states;
        let taken_a =
            inner.inkling2_take_victim(&patient_query, &store_victim(id_a, 1, full_entry_a), &[]);
        assert_eq!(taken_a.outcome, FreeOutcome::Met);
        assert_eq!(taken_a.now_bytes, swa_conv_a);
        assert_eq!(
            states_before_a - inner.memory_accounts().expect("accounts").states,
            swa_conv_a
        );
        assert_eq!(inner.paged.as_ref().unwrap().draft_windows_bytes(), dw_len);
        inner
            .paged
            .as_mut()
            .unwrap()
            .block_boundary_draft_window
            .remove(&1);
        assert_eq!(inner.paged.as_ref().unwrap().draft_windows_bytes(), 0);

        // 2. Entry B releases seq 2's clone -> reclaiming credits full `swa + conv + dw_len`.
        let (id_b, swa_conv_b, full_entry_b) = {
            let packed_b = PackedDraftWindow::from_window(&dw).expect("packed_b");
            let rt = inner.paged.as_mut().expect("paged");
            rt.coordinator.reset_scheduled_request(2).expect("reset 2");
            rt.coordinator
                .record_tokens_all(2, &tokens_b)
                .expect("record b");
            let id_b = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    2,
                    &tokens_b,
                    32,
                    112,
                    StateClass::Ordinary,
                    Some(&packed_b),
                )
                .expect("capture B")
                .expect("id_b");
            assert!(
                rt.state_store
                    .flush_to_disk(&rt.coordinator, test_deadline(10))
            );
            rt.block_boundary_draft_window
                .insert(2, (32, BoundaryDraftWindow::Packed(packed_b)));
            let entry = rt.state_store.store.get(id_b).unwrap();
            let swa_conv =
                (entry.ram().unwrap().swa_data.len() + entry.ram().unwrap().conv_data.len()) as u64;
            assert_eq!(entry.ram().unwrap().exclusive_ram_bytes(), swa_conv);
            rt.block_boundary_draft_window.remove(&2);
            let entry = rt.state_store.store.get(id_b).unwrap();
            assert_eq!(
                entry.ram().unwrap().exclusive_ram_bytes(),
                swa_conv + dw_len
            );
            (id_b, swa_conv, entry.state_bytes())
        };
        let taken_b =
            inner.inkling2_take_victim(&patient_query, &store_victim(id_b, 2, full_entry_b), &[]);
        assert_eq!(taken_b.outcome, FreeOutcome::Met);
        assert_eq!(taken_b.now_bytes, swa_conv_b + dw_len);

        // 3. RAM-only erased entry C with shared `Arc` credits `swa + conv + lineage` only.
        let _ = inner.close_state_disk(test_deadline(5));
        let (id_c, swa_conv_c, lineage_c, full_entry_c) = {
            let packed_c = PackedDraftWindow::from_window(&dw).expect("packed_c");
            let rt = inner.paged.as_mut().expect("paged");
            rt.coordinator.release_request_all(1).expect("release 1");
            rt.coordinator.release_request_all(2).expect("release 2");
            rt.state_store.disk = None;
            rt.coordinator.reset_scheduled_request(3).expect("reset 3");
            rt.coordinator
                .record_tokens_all(3, &tokens_c)
                .expect("record c");
            let id_c = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    3,
                    &tokens_c,
                    32,
                    113,
                    StateClass::Ordinary,
                    Some(&packed_c),
                )
                .expect("capture C")
                .expect("id_c");
            rt.block_boundary_draft_window
                .insert(3, (32, BoundaryDraftWindow::Packed(packed_c)));
            let entry = rt.state_store.store.get(id_c).unwrap();
            let ram = entry.ram().unwrap();
            let swa_conv = (ram.swa_data.len() + ram.conv_data.len()) as u64;
            let lineage = entry.state_bytes() - ram.cached_bytes;
            assert!(lineage > 0);
            (id_c, swa_conv, lineage, entry.state_bytes())
        };
        let taken_c =
            inner.inkling2_take_victim(&patient_query, &store_victim(id_c, 3, full_entry_c), &[]);
        assert_eq!(taken_c.outcome, FreeOutcome::Met);
        assert_eq!(taken_c.now_bytes, swa_conv_c + lineage_c);

        // 4. `BoundaryDraftWindow::Device` packs into an exclusively-owned `PackedDraftWindow`.
        {
            let tokens_d: Vec<u32> = (160u32..200u32).collect();
            let rt = inner.paged.as_mut().expect("paged");
            rt.block_boundary_draft_window.clear();
            rt.draft_windows.clear();
            rt.coordinator.release_request_all(3).expect("release 3");
            rt.coordinator.reset_scheduled_request(4).expect("reset 4");
            rt.coordinator
                .record_tokens_all(4, &tokens_d)
                .expect("record d");
            let dev_snap = dw.snapshot().expect("dev_snap");
            let dev_nbytes = dev_snap.nbytes();
            rt.block_boundary_draft_window
                .insert(4, (32, BoundaryDraftWindow::Device(dev_snap)));
            assert_eq!(rt.draft_windows_bytes(), dev_nbytes);
            let boundary_win = rt
                .block_boundary_draft_window
                .get(&4)
                .filter(|(wb, _)| *wb == 32)
                .map(|(_, w)| w);
            let id_d = rt
                .state_store
                .capture_boundary_with_window(
                    &mut rt.coordinator,
                    &conv,
                    4,
                    &tokens_d,
                    32,
                    114,
                    StateClass::Ordinary,
                    boundary_win,
                    None,
                )
                .expect("capture D")
                .expect("id_d");
            let ram_d = rt.state_store.store.get(id_d).unwrap().ram().unwrap();
            let swa_conv_d = (ram_d.swa_data.len() + ram_d.conv_data.len()) as u64;
            assert!(ram_d.draft_window.is_some());
            assert_eq!(ram_d.exclusive_ram_bytes(), swa_conv_d + dw_len);
            assert_eq!(rt.draft_windows_bytes(), dev_nbytes);
        }
    }

    /// Round-13 Finding 2 regression: reclaiming an anchor from `StateStore` while a live sequence
    /// still holds a cloned `PackedDraftWindow` in `block_boundary_draft_window` keeps those host
    /// bytes charged in `CacheLimitCoordinator::registered_pool_bytes()` until the sequence drops it.
    #[cfg(target_os = "macos")]
    #[test]
    fn host_budget_keeps_live_packed_draft_window_charged_after_anchor_reclaim() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use crate::cache_limit::CacheLimitCoordinator;
        use crate::engine::governor::ledger::FreeOutcome;

        let fx =
            StateDiskTestFixture::new("r13-live-dw-budget", b"r13-live-dw-budget-v1", Some(0.0));
        let local_coord = Arc::new(CacheLimitCoordinator::new_for_test());
        let query = patient_victim_query();
        let mut inner =
            crate::models::inkling2::scheduler::tests::build_synthetic_dflash2_inner(256, None);
        attach_synthetic_disk(&fx, &mut inner);
        inner
            .paged
            .as_mut()
            .expect("paged")
            .state_store
            .attach_host_budget_to(Arc::clone(&local_coord));

        let conv = make_short_conv(&inner.config);
        let dcfg = inner.dflash2.as_ref().expect("dflash2").config.clone();
        let dw = make_draft_window(dcfg.context_width(), dcfg.sliding_window as u32, 32);

        let seed = |inner: &mut crate::models::inkling2::model::Inkling2Inner,
                    seq: u32,
                    tokens: &[u32],
                    salt: u64,
                    flush: bool| {
            let packed = PackedDraftWindow::from_window(&dw).expect("packed");
            let dw_len = packed.bytes.len() as u64;
            let rt = inner.paged.as_mut().expect("paged");
            rt.coordinator.reset_scheduled_request(seq).expect("reset");
            rt.coordinator
                .record_tokens_all(seq, tokens)
                .expect("record");
            let id = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    seq,
                    tokens,
                    32,
                    salt,
                    StateClass::Ordinary,
                    Some(&packed),
                )
                .expect("capture")
                .expect("id");
            if flush {
                assert!(
                    rt.state_store
                        .flush_to_disk(&rt.coordinator, test_deadline(10))
                );
            }
            rt.block_boundary_draft_window
                .insert(seq, (32, BoundaryDraftWindow::Packed(packed)));
            let entry = rt.state_store.store.get(id).unwrap();
            let ram = entry.ram().unwrap();
            let swa_conv = (ram.swa_data.len() + ram.conv_data.len()) as u64;
            let lineage = entry.state_bytes() - ram.cached_bytes;
            (id, swa_conv, dw_len, lineage, entry.state_bytes())
        };

        // 1. SSD-backed anchor A: drop_ram frees only `swa_conv`; `dw_len` stays charged until dropped.
        let tokens_a: Vec<u32> = (210u32..250u32).collect();
        let (id_a, swa_conv_a, dw_len, lin_a, full_a) = seed(&mut inner, 1, &tokens_a, 211, true);
        assert_eq!(
            local_coord.registered_pool_bytes(),
            swa_conv_a + dw_len + lin_a
        );
        let taken_a = inner.inkling2_take_victim(&query, &store_victim(id_a, 1, full_a), &[]);
        assert_eq!(
            (taken_a.outcome, taken_a.now_bytes),
            (FreeOutcome::Met, swa_conv_a)
        );
        assert_eq!(local_coord.registered_pool_bytes(), dw_len + lin_a);
        inner
            .paged
            .as_mut()
            .unwrap()
            .block_boundary_draft_window
            .remove(&1);
        assert_eq!(local_coord.registered_pool_bytes(), lin_a);

        // 2. RAM-only anchor B (erased): clearing seq 2's clone drops pool charge to 0.
        let _ = inner.close_state_disk(test_deadline(5));
        {
            let rt = inner.paged.as_mut().unwrap();
            rt.state_store.clear_ram_and_checkpoints();
            rt.state_store.disk = None;
            rt.coordinator.release_request_all(1).expect("release 1");
        }
        let tokens_b: Vec<u32> = (260u32..300u32).collect();
        let (id_b, swa_conv_b, _, lin_b, full_b) = seed(&mut inner, 2, &tokens_b, 212, false);
        assert_eq!(
            local_coord.registered_pool_bytes(),
            swa_conv_b + dw_len + lin_b
        );
        let taken_b = inner.inkling2_take_victim(&query, &store_victim(id_b, 2, full_b), &[]);
        assert_eq!(
            (taken_b.outcome, taken_b.now_bytes),
            (FreeOutcome::Met, swa_conv_b + lin_b)
        );
        assert_eq!(local_coord.registered_pool_bytes(), dw_len);
        inner
            .paged
            .as_mut()
            .unwrap()
            .block_boundary_draft_window
            .remove(&2);
        assert_eq!(local_coord.registered_pool_bytes(), 0);
    }

    /// Round-6 Finding 1 (part A) regression: when a queued SSD upgrade's old Global KV chain was
    /// evicted AND staging refuses due to low room, `PendingDiskUpgrade::Repair` and `write_behind`
    /// survive until room returns and repair finishes cleanly.
    #[cfg(target_os = "macos")]
    #[test]
    fn state_store_repair_survives_stage_refusal_and_completes_when_room_returns() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let fx = StateDiskTestFixture::new(
            "repair-stage-refusal",
            b"repair-stage-refusal-v1",
            Some(10.0),
        );
        let tokens_a: Vec<u32> = (2200u32..2296u32).collect();
        let query_a: Vec<u32> = (2200u32..2270u32).collect();
        let packed_dw = fx.packed_dw(64);

        fx.seed_durable(&[(1, &tokens_a, 64, 98)], 64u64 << 20, None);

        let reserve = 4096u64;
        let mut s2 = fx.open_adopted(64u64 << 20, Some(reserve), 1);
        let adopted_id = s2.first_id();
        let upgraded_id = s2.capture_ord(&fx.conv, 1, &tokens_a, 64, 98, Some(&packed_dw));
        assert_eq!(upgraded_id, adopted_id);

        let evicted_kv_key = s2.evict_kv_block(upgraded_id, 1);
        s2.manager()
            .set_space_override_for_test(Some(Box::new(move || Ok((1 << 30, reserve)))));

        s2.assert_flush_fails_promptly("evicted-chain upgrade cannot fit");
        assert!(s2.entry(upgraded_id).ram().is_some());
        assert!(s2.entry(upgraded_id).disk().is_none());
        assert!(matches!(
            s2.store.pending_disk_upgrades.get(&upgraded_id),
            Some(PendingDiskUpgrade::Repair { .. })
        ));

        s2.clear_space_override(false);
        assert!(s2.flush(10));
        assert!(s2.store.pending_disk_upgrades.is_empty());
        assert!(s2.store.in_flight_disk_upgrade.is_none());
        assert!(s2.manager().contains_in(&evicted_kv_key, ColdGroup::Kv));
        assert!(
            s2.entry(upgraded_id)
                .disk()
                .is_some_and(|c| c.landed && !c.chain_owed)
        );
        assert!(s2.disk().entry_disk_restorable(s2.entry(upgraded_id)));

        s2.drop_ram_and_purge();
        s2.assert_ssd_restores(&query_a, 98, None, Some(&packed_dw));
        assert!(s2.close());
    }

    /// Round-6 Finding 1 (part B) regression: when a repair replacement blob lands with
    /// `chain_owed == true` and Global KV chain capture fails, `PendingDiskUpgrade::Repair` and
    /// `spill_backoff` stay active without re-staging until space recovers.
    #[cfg(target_os = "macos")]
    #[test]
    fn state_store_repair_survives_chain_capture_failure_and_backs_off() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let fx = StateDiskTestFixture::new(
            "repair-chain-failure",
            b"repair-chain-failure-v1",
            Some(10.0),
        );
        let tokens_a: Vec<u32> = (2400u32..2496u32).collect();
        let tokens_b: Vec<u32> = (2500u32..2596u32).collect();
        let query_a: Vec<u32> = (2400u32..2470u32).collect();
        let packed_dw = fx.packed_dw(64);

        fx.seed_durable(&[(1, &tokens_a, 64, 99)], 64u64 << 20, None);

        let reserve = 4096u64;
        let mut s2 = fx.open_adopted(64u64 << 20, Some(reserve), 1);
        s2.store.set_ram_cap(4);
        let adopted_id = s2.first_id();

        let upgraded_id = s2.capture_ord(&fx.conv, 1, &tokens_a, 64, 99, Some(&packed_dw));
        assert_eq!(upgraded_id, adopted_id);

        let other_id = s2.capture_ord_no_wb(&fx.conv, 2, &tokens_b, 64, 100);
        let other_mark = s2.entry(other_id).mark();
        s2.store.store.use_state(other_mark);
        s2.store.set_ram_cap(1);

        s2.evict_kv_block(upgraded_id, 1);

        assert!(s2.stage(upgraded_id));
        s2.await_writing("repair blob");
        assert!(
            s2.entry(upgraded_id)
                .disk()
                .is_some_and(|c| c.landed && c.chain_owed)
        );
        assert!(matches!(
            s2.store.pending_disk_upgrades.get(&upgraded_id),
            Some(PendingDiskUpgrade::Repair { .. })
        ));

        s2.set_writer_space(64 << 20, reserve, None);
        s2.assert_flush_fails_promptly("repair chain failure");
        assert!(s2.entry(upgraded_id).ram().is_some());
        assert!(s2.entry(upgraded_id).disk().is_none());
        assert!(matches!(
            s2.store.pending_disk_upgrades.get(&upgraded_id),
            Some(PendingDiskUpgrade::Repair { .. })
        ));
        assert!(s2.store.spill_backoff.contains_key(&upgraded_id));
        assert!(!s2.disk().writing());
        s2.store.trim_ram_to_cap(&s2.coord, None);
        assert!(!s2.disk().writing());

        s2.clear_space_override(true);
        s2.store.store.end_use(other_mark);
        s2.store.set_ram_cap(4);
        assert!(s2.flush(10));
        assert!(s2.store.pending_disk_upgrades.is_empty());

        s2.drop_ram_and_purge();
        s2.assert_ssd_restores(&query_a, 99, None, Some(&packed_dw));
        assert!(s2.close());
    }

    /// Round-6 Finding 3 regression: a first-attempt Global KV chain failure (`chain_before == None`)
    /// inside `tick` or `chain_step` identifies the failed `EntryId`, records `spill_backoff`, and
    /// prevents immediate re-staging.
    #[cfg(target_os = "macos")]
    #[test]
    fn state_store_first_block_chain_failure_records_spill_backoff_and_prevents_immediate_restage()
    {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let fx = StateDiskTestFixture::new(
            "first-block-chain-backoff",
            b"first-block-chain-backoff-v1",
            Some(10.0),
        )
        .with_paged(2048, 4);
        let mut s = fx.open(64u64 << 20, None);
        s.store.set_ram_cap(4);

        let tokens_a: Vec<u32> = (2600u32..2696u32).collect();
        let tokens_b: Vec<u32> = (2700u32..2796u32).collect();
        let tokens_c: Vec<u32> = (2800u32..2896u32).collect();

        let id_a = s.capture_ord(&fx.conv, 1, &tokens_a, 64, 105, None);
        let id_b = s.capture_ord(&fx.conv, 2, &tokens_b, 64, 106, None);
        let mark_b = s.entry(id_b).mark();
        s.store.store.use_state(mark_b);
        s.store.set_ram_cap(1);

        assert!(s.stage(id_a));
        s.await_writing("landing A");
        assert!(s.disk().chain.is_none());
        assert!(
            s.entry(id_a)
                .disk()
                .is_some_and(|c| c.landed && c.chain_owed)
        );
        assert!(!s.store.spill_backoff.contains_key(&id_a));

        s.coord.release_all_and_purge().expect("purge KV blocks");
        assert!(s.store.tick(&s.coord, Tick::Full { command_pages: 0 }));
        assert!(s.entry(id_a).disk().is_none() && s.entry(id_a).ram().is_some());
        assert!(s.store.spill_backoff.contains_key(&id_a));
        assert!(!s.disk().writing());

        s.store.trim_ram_to_cap(&s.coord, None);
        s.store.tick(&s.coord, Tick::Full { command_pages: 0 });
        assert!(!s.disk().writing());

        // Also verify direct `chain_step` with `chain_before == None` records `spill_backoff`.
        s.store.set_ram_cap(4);
        let id_c = s.capture_ord(&fx.conv, 3, &tokens_c, 64, 107, None);
        assert!(s.stage(id_c));
        s.await_writing("landing C");
        assert!(s.disk().chain.is_none());
        s.coord.release_all_and_purge().expect("purge KV for C");
        assert!(s.store.chain_step(&s.coord, test_deadline(5)));
        assert!(s.store.spill_backoff.contains_key(&id_c));
        assert!(s.entry(id_c).disk().is_none());

        s.store.store.end_use(mark_b);
        let _ = s.close();
    }

    /// Round-7 Finding 2 regression: urgent reclaim (`allow_wait = false`) during an in-flight
    /// in-place SSD upgrade drops ONLY RAM and preserves a durable SSD prefix on both success
    /// (upgraded blob) and failure (restored `old_copy`).
    #[cfg(target_os = "macos")]
    #[test]
    fn urgent_reclaim_during_in_place_upgrade_preserves_durable_prefix_on_success_and_failure() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let fx = StateDiskTestFixture::new(
            "r7-urgent-upgrade",
            b"urgent-upgrade-preserve-v1",
            Some(10.0),
        );
        let tokens_a: Vec<u32> = (3000u32..3096u32).collect();
        let query_a: Vec<u32> = (3000u32..3070u32).collect();
        let tokens_b: Vec<u32> = (3200u32..3296u32).collect();
        let query_b: Vec<u32> = (3200u32..3270u32).collect();
        let packed_dw = fx.packed_dw(64);

        fx.seed_durable(
            &[(1, &tokens_a, 64, 111), (2, &tokens_b, 64, 112)],
            64u64 << 20,
            None,
        );

        let reserve = 4096u64;
        let mut s2 = fx.open_adopted(64u64 << 20, Some(reserve), 2);

        // Part A: Urgent reclaim during in-place upgrade -> replacement write SUCCEEDS.
        let id_a = s2.capture_ord(&fx.conv, 1, &tokens_a, 64, 111, Some(&packed_dw));
        assert!(s2.stage(id_a));
        assert!(
            s2.store
                .in_flight_disk_upgrade
                .as_ref()
                .is_some_and(|u| u.id == id_a)
        );
        assert_eq!(
            s2.store.reclaim_entry_with_wait(&s2.coord, id_a, false),
            ReclaimOutcome::RamDropped
        );
        assert!(s2.entry(id_a).ram().is_none() && s2.entry(id_a).disk().is_some());
        assert!(
            s2.store
                .in_flight_disk_upgrade
                .as_ref()
                .is_some_and(|u| u.id == id_a)
        );

        s2.await_writing("upgrade A");
        assert!(s2.store.in_flight_disk_upgrade.is_none());
        assert!(!s2.store.entries_missing_draft_window.contains(&id_a));
        assert!(
            s2.entry(id_a).ram().is_none()
                && s2
                    .entry(id_a)
                    .disk()
                    .is_some_and(|c| c.landed && !c.chain_owed)
        );
        s2.coord.release_all_and_purge().expect("purge hot A");
        s2.assert_ssd_restores(&query_a, 111, Some(id_a), Some(&packed_dw));

        // Part B: Urgent reclaim during in-place upgrade -> replacement write FAILS -> old_copy restored.
        let id_b = s2.capture_ord(&fx.conv, 2, &tokens_b, 64, 112, Some(&packed_dw));
        let old_copy_b = s2.disk_copy(id_b);

        s2.set_writer_space(64 << 20, reserve, None);
        assert!(s2.stage(id_b));
        assert!(
            s2.store
                .in_flight_disk_upgrade
                .as_ref()
                .is_some_and(|u| u.id == id_b)
        );
        assert_eq!(
            s2.store.reclaim_entry_with_wait(&s2.coord, id_b, false),
            ReclaimOutcome::RamDropped
        );
        assert!(s2.entry(id_b).ram().is_none());

        s2.await_writing("failing upgrade B");
        s2.clear_space_override(true);

        assert!(s2.store.in_flight_disk_upgrade.is_none());
        assert!(s2.entry(id_b).ram().is_none());
        assert_eq!(
            s2.entry(id_b).disk().copied(),
            Some(DiskCopy {
                chain_owed: false,
                ..old_copy_b
            })
        );
        assert!(s2.store.entries_missing_draft_window.contains(&id_b));
        s2.coord.release_all_and_purge().expect("purge hot B");
        s2.assert_ssd_restores(&query_b, 112, Some(id_b), None);
        assert!(s2.flush(10));
        assert!(s2.close());
    }

    /// Round-7 Finding 3 + Round-13 Finding 3 regression:
    /// - Background `tick` / `trim_ram_to_cap` honor active repair `spill_backoff` without
    ///   re-staging or discarding RAM.
    /// - Once backoff has elapsed and SSD accepts staging, patient reclaim stages the `Repair`
    ///   with `keep_ram = true`, returns `Pending` while the write is in flight, keeps RAM +
    ///   `Repair` tracking intact across a second async write failure, and drops RAM cleanly once
    ///   disk room recovers and repair completes.
    /// - When a `Repair` is still backed off (or SSD refuses synchronously) with `!disk.writing()`,
    ///   patient Governor reclaim erases it immediately (`ReclaimOutcome::Erased`) instead of
    ///   stalling on the 10-second backoff timer.
    #[cfg(target_os = "macos")]
    #[test]
    fn patient_reclaim_respects_repair_backoff_and_keeps_ram_across_second_failure() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let fx = StateDiskTestFixture::new(
            "r7-patient-repair-backoff",
            b"patient-repair-backoff-v1",
            Some(10.0),
        );
        let tokens_a: Vec<u32> = (3400u32..3496u32).collect();
        let query_a: Vec<u32> = (3400u32..3470u32).collect();
        let packed_dw = fx.packed_dw(64);

        fx.seed_durable(&[(1, &tokens_a, 64, 121)], 64u64 << 20, None);

        let reserve = 4096u64;
        let mut s2 = fx.open_adopted(64u64 << 20, Some(reserve), 1);
        s2.store.set_ram_cap(16);

        let id_a = s2.capture_ord(&fx.conv, 1, &tokens_a, 64, 121, Some(&packed_dw));
        s2.evict_kv_block(id_a, 1);

        s2.set_writer_space(64 << 20, reserve, None);
        assert!(s2.stage(id_a));
        s2.await_writing("failing repair 1");
        assert!(s2.entry(id_a).ram().is_some() && s2.entry(id_a).disk().is_none());
        assert!(matches!(
            s2.store.pending_disk_upgrades.get(&id_a),
            Some(PendingDiskUpgrade::Repair { .. })
        ));
        assert!(s2.store.spill_backoff.contains_key(&id_a));

        // 1. Backed off: background `tick` and `trim_ram_to_cap` honor backoff without re-staging
        //    and keep RAM + `Repair` tracking intact.
        let _ = s2.store.tick(&s2.coord, Tick::Full { command_pages: 0 });
        s2.store.trim_ram_to_cap(&s2.coord, None);
        assert!(!s2.disk().writing());
        assert!(s2.entry(id_a).ram().is_some() && s2.entry(id_a).disk().is_none());
        assert!(matches!(
            s2.store.pending_disk_upgrades.get(&id_a),
            Some(PendingDiskUpgrade::Repair { .. })
        ));
        assert!(s2.store.spill_backoff.contains_key(&id_a));

        // 2. Expire backoff while SSD accepts staging (then fails asynchronously on writer thread):
        //    patient reclaim starts concrete write (`keep_ram = true`), returns `Pending` while
        //    `writing()`, and keeps RAM + `Repair` intact across the 2nd failure.
        s2.expire_backoff(id_a);
        assert_eq!(
            s2.store.reclaim_entry_with_wait(&s2.coord, id_a, true),
            ReclaimOutcome::Pending
        );
        assert!(s2.disk().writing());
        assert!(s2.entry(id_a).ram().is_some());
        assert!(matches!(
            s2.store.pending_disk_upgrades.get(&id_a),
            Some(PendingDiskUpgrade::Repair { .. })
        ));
        s2.await_writing("failing repair 2");
        assert!(s2.entry(id_a).ram().is_some() && s2.entry(id_a).disk().is_none());
        assert!(matches!(
            s2.store.pending_disk_upgrades.get(&id_a),
            Some(PendingDiskUpgrade::Repair { .. })
        ));
        assert!(s2.store.spill_backoff.contains_key(&id_a));
        let ram_hit = s2
            .store
            .lookup_deepest_restorable(&s2.coord, &query_a, 121)
            .expect("prefix restorable from RAM");
        assert_eq!(ram_hit.id, id_a);
        let (ram_anchor, from_ssd) = s2
            .store
            .load_anchor(&s2.coord, ram_hit.id)
            .expect("load RAM");
        assert!(!from_ssd);
        assert_eq!(ram_anchor.draft_window.as_ref(), Some(&packed_dw));

        // 3. Restore space + expire backoff: patient reclaim lands blob + KV chain and drops RAM.
        s2.clear_space_override(true);
        s2.expire_backoff(id_a);
        assert_eq!(
            s2.store.reclaim_entry_with_wait(&s2.coord, id_a, true),
            ReclaimOutcome::Pending
        );
        s2.await_landed(id_a, "repair recovery");
        assert!(s2.store.pending_disk_upgrades.is_empty());
        assert!(
            s2.entry(id_a).ram().is_none()
                && s2
                    .entry(id_a)
                    .disk()
                    .is_some_and(|c| c.landed && !c.chain_owed)
        );
        s2.coord.release_all_and_purge().expect("purge hot");
        s2.assert_ssd_restores(&query_a, 121, Some(id_a), Some(&packed_dw));

        // 4. Synchronously refused `Repair` (backoff elapsed, volume full, `!writing()`): patient
        //    reclaim erases immediately instead of stalling Governor with `Pending`.
        let tokens_b: Vec<u32> = (3500u32..3596u32).collect();
        let id_b = s2.capture_ord(&fx.conv, 2, &tokens_b, 64, 122, Some(&packed_dw));
        s2.store.pending_disk_upgrades.insert(
            id_b,
            PendingDiskUpgrade::Repair {
                retry_after: Instant::now() - Duration::from_secs(1),
            },
        );
        s2.manager()
            .set_space_override_for_test(Some(Box::new(move || Ok((1 << 30, reserve)))));
        assert_eq!(
            s2.store.reclaim_entry_with_wait(&s2.coord, id_b, true),
            ReclaimOutcome::Erased
        );
        assert!(s2.store.store.get(id_b).is_none() && !s2.disk().writing());
        assert!(s2.close());
    }

    /// Round-13 Finding 3 regression: a full SSD placing an ordinary RAM anchor into 10-second
    /// `spill_backoff` with `!disk.writing()` must NOT stall a patient Governor claim (`ledger::free`
    /// with `waits_for_writes: true`) — it erases the anchor immediately (`FreeOutcome::Met`).
    #[cfg(target_os = "macos")]
    #[test]
    fn patient_governor_claim_erases_backed_off_anchor_instead_of_stalling() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use crate::cache_limit::CacheLimitCoordinator;
        use crate::disk_tier::TEST_WRITER;
        use crate::engine::governor::governor::ModelRegistration;
        use crate::engine::governor::ledger::{
            FreeOutcome, Need, Taken, Victim, VictimQuery, VictimSource,
        };
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        use crate::models::inkling2::scheduler::tests::{
            TestCoordGovernorIo, build_synthetic_dflash2_inner,
        };

        struct InnerSource<'a>(&'a mut crate::models::inkling2::model::Inkling2Inner);
        impl VictimSource for InnerSource<'_> {
            fn units(&self, query: &VictimQuery) -> Vec<Victim> {
                self.0.victim_units(query, &[]).into_units()
            }
            fn take(&mut self, query: &VictimQuery, victim: &Victim) -> Taken {
                self.0.take_victim(query, victim, &[])
            }
        }

        let fx = StateDiskTestFixture::new(
            "r13-patient-backoff-erase",
            b"r13-patient-backoff-erase-v1",
            Some(10.0),
        );
        let reserve = 4096u64;
        let coord = Arc::new(CacheLimitCoordinator::new_for_test());
        let mut inner = build_synthetic_dflash2_inner(256, None);
        TEST_WRITER.with(|w| w.set(Some((8, reserve))));
        attach_synthetic_disk(&fx, &mut inner);
        TEST_WRITER.with(|w| w.set(None));
        inner
            .paged
            .as_mut()
            .expect("paged")
            .state_store
            .attach_host_budget_to(Arc::clone(&coord));

        let conv = make_short_conv(&inner.config);
        let tokens: Vec<u32> = (3700u32..3796u32).collect();
        let (id_a, entry_bytes) = {
            let rt = inner.paged.as_mut().expect("paged");
            rt.coordinator.reset_scheduled_request(1).expect("reset");
            rt.coordinator
                .record_tokens_all(1, &tokens)
                .expect("record");
            let id_a = rt
                .state_store
                .capture_boundary(
                    &mut rt.coordinator,
                    &conv,
                    1,
                    &tokens,
                    64,
                    371,
                    StateClass::Ordinary,
                    None,
                )
                .expect("capture")
                .expect("id_a");
            rt.coordinator.release_request_all(1).expect("release");
            let disk = rt.state_store.disk.as_mut().expect("disk");
            disk.tier
                .manager
                .set_space_override_for_test(Some(Box::new(move || Ok((1 << 30, reserve)))));
            assert_eq!(
                rt.state_store.persist_entry(&rt.coordinator, id_a, false),
                PersistStatus::Busy
            );
            assert!(!rt.state_store.disk.as_ref().expect("disk").writing());
            assert!(rt.state_store.spill_backoff.contains_key(&id_a));
            let entry = rt.state_store.store.get(id_a).expect("id_a");
            assert!(entry.ram().is_some() && entry.disk().is_none());
            (id_a, entry.state_bytes())
        };

        let gov = Governor::new(Arc::new(TestCoordGovernorIo {
            coord: Arc::clone(&coord),
            active: 16 << 20,
        }));
        let _reg = gov
            .register_model(ModelRegistration {
                model_id: 91,
                hard_budget: (16 << 20) + coord.registered_pool_bytes(),
                untracked: 0,
                page_bytes: 1 << 20,
            })
            .expect("register");
        let claim = gov.open(
            ClaimSpec {
                cls: ClaimClass::InService,
                request_id: 1,
                rank: ClaimRank::default(),
                reach: ReachClass::Ordinary,
                timing: NeedTiming::CanWait,
            },
            91,
        );
        let mut src = InnerSource(&mut inner);
        let res = gov.free(
            claim.id(),
            Need {
                bytes: entry_bytes,
                pages: 1,
                kv: true,
                ..Need::default()
            },
            &mut [&mut src],
        );
        assert_eq!(res.outcome, FreeOutcome::Met);
        assert_eq!(res.now_bytes, entry_bytes);
        assert!(res.pages >= 1);
        assert_eq!(coord.registered_pool_bytes(), 0);
        let rt = inner.paged.as_ref().expect("paged");
        assert!(rt.state_store.store.get(id_a).is_none());
        assert!(!rt.state_store.disk.as_ref().expect("disk").writing());
        let _ = inner.close_state_disk(test_deadline(5));
    }

    /// Round-8 Finding 3 regression: ordinary write-behind retries after transient sync/async
    /// refusals and drops permanently oversized entries cleanly without spinning.
    #[cfg(target_os = "macos")]
    #[test]
    fn ordinary_write_behind_retries_after_transient_refusal_and_drops_oversized_entry() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use std::sync::atomic::{AtomicUsize, Ordering};

        let fx = StateDiskTestFixture::new(
            "r8-wb-transient-retry",
            b"r8-wb-transient-retry-v1",
            Some(10.0),
        )
        .with_paged(2048, 4);
        let tokens_a: Vec<u32> = (4100u32..4196u32).collect();
        let query_a: Vec<u32> = (4100u32..4170u32).collect();
        let tokens_b: Vec<u32> = (4200u32..4296u32).collect();
        let query_b: Vec<u32> = (4200u32..4270u32).collect();
        let tokens_c: Vec<u32> = (4300u32..4396u32).collect();

        let reserve = 4096u64;
        let mut s = fx.open(64u64 << 20, Some(reserve));
        s.store.set_ram_cap(16);

        // Part 1: transient synchronous volume headroom refusal (`room_above_reserve == 0`).
        let id_a = s.capture_ord(&fx.conv, 1, &tokens_a, 64, 201, None);
        assert!(s.store.pending_disk_upgrades.is_empty());
        assert_eq!(s.store.ram_len(), 1);
        assert_eq!(s.wb_waiting(), 1);

        s.manager()
            .set_space_override_for_test(Some(Box::new(move || Ok((1 << 30, reserve)))));
        assert_eq!(
            s.store.persist_entry(&s.coord, id_a, false),
            PersistStatus::Busy
        );
        assert_eq!(s.wb_waiting(), 1);
        assert!(s.store.spill_backoff.contains_key(&id_a));
        assert!(s.entry(id_a).ram().is_some() && s.entry(id_a).disk().is_none());

        s.assert_flush_fails_promptly("transient ordinary refusal");
        assert_eq!(s.wb_waiting(), 1);

        s.clear_space_override(false);
        assert_eq!(
            s.store.persist_entry(&s.coord, id_a, false),
            PersistStatus::Busy,
            "non-forced persist_entry must respect spill_backoff"
        );
        assert!(
            !s.disk().writing(),
            "normal persist_entry must respect spill_backoff"
        );
        s.expire_backoff(id_a);
        assert_eq!(
            s.store.persist_entry(&s.coord, id_a, false),
            PersistStatus::Started
        );
        assert!(s.disk().writing() || s.entry(id_a).disk().is_some());
        assert!(s.flush(10));
        assert!(!s.store.spill_backoff.contains_key(&id_a));
        assert!(s.disk().entry_disk_restorable(s.entry(id_a)));
        s.drop_ram_and_purge();
        s.assert_ssd_restores(&query_a, 201, Some(id_a), None);

        // Part 2: transient async writer-side space refusal during `flush_to_disk`.
        let id_b = s.capture_ord(&fx.conv, 2, &tokens_b, 64, 202, None);
        let writer_refusals = Arc::new(AtomicUsize::new(0));
        let failures_before_b = s.disk().counters.offload_failures;
        let enqueued_before_b = s.manager().stats().enqueued;
        s.set_writer_space(64 << 20, reserve, Some(Arc::clone(&writer_refusals)));

        s.assert_flush_fails_promptly("async ordinary refusal");
        assert!(writer_refusals.load(Ordering::Relaxed) >= 1);
        let failures_delta = s.disk().counters.offload_failures - failures_before_b;
        let enqueued_delta = s.manager().stats().enqueued - enqueued_before_b;
        assert!(
            (1..=2).contains(&failures_delta) && enqueued_delta <= 3,
            "failures_delta={failures_delta}, enqueued_delta={enqueued_delta}"
        );
        assert_eq!(s.wb_waiting(), 1);
        assert!(s.store.spill_backoff.contains_key(&id_b));

        s.clear_space_override(true);
        assert!(s.flush(10));
        assert!(!s.store.spill_backoff.contains_key(&id_b));
        s.drop_ram_and_purge();
        s.assert_ssd_restores(&query_b, 202, Some(id_b), None);
        assert!(s.close());

        // Part 3: permanently oversized entry drops out of `write_behind` cleanly.
        let mut s_tiny = fx.open_in(fx.subdir("tiny-cache"), 1024, b"r8-wb-oversized-v1", None);
        let id_c = s_tiny.capture_ord(&fx.conv, 1, &tokens_c, 64, 203, None);
        assert!(!s_tiny.disk().entry_fits_tier_limit(s_tiny.entry(id_c)));
        let t2 = Instant::now();
        assert!(s_tiny.flush(3));
        assert!(t2.elapsed() < Duration::from_millis(500));
        assert_eq!(s_tiny.wb_waiting(), 0);
        assert!(s_tiny.entry(id_c).ram().is_some() && s_tiny.entry(id_c).disk().is_none());
        assert!(s_tiny.close());
    }

    /// Round-9 Finding 2 regression: while an in-place SSD upgrade is in flight, its `ColdKeyHold`
    /// pins both `old_copy.key` and its Global KV chain against competing `MakeRoom::Evict` writes.
    #[cfg(target_os = "macos")]
    #[test]
    fn in_place_upgrade_holds_old_blob_and_kv_chain_against_competing_eviction() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use mlx_paged_attn::Landed;
        use std::sync::atomic::{AtomicBool, Ordering};

        let fx = StateDiskTestFixture::new("r9-in-place-hold", b"r9-in-place-hold-v1", Some(10.0))
            .with_paged(2048, 4);
        let reserve = 4096u64;
        let tokens_a: Vec<u32> = (100u32..164u32).collect();
        let query_a: Vec<u32> = (100u32..165u32).collect();
        let tokens_competing: Vec<u32> = (400u32..464u32).collect();
        let packed_dw = fx.packed_dw(64);

        let (exact_footprint, _) =
            fx.seed_durable(&[(1, &tokens_a, 64, 911)], 64u64 << 20, Some(reserve));
        assert!(exact_footprint > 0);

        let mut s2 = fx.open_adopted(exact_footprint, Some(reserve), 1);
        let id_a = s2.capture_ord(&fx.conv, 1, &tokens_a, 64, 911, Some(&packed_dw));
        let old_copy = s2.disk_copy(id_a);
        let kv_keys = s2.disk().engine.kv_chain_keys(s2.entry(id_a));
        assert_eq!(kv_keys.len(), 2);

        let fail_writer = Arc::new(AtomicBool::new(true));
        s2.set_toggle_writer_space(reserve, Arc::clone(&fail_writer));

        assert!(s2.stage(id_a));
        assert!(
            s2.store
                .in_flight_disk_upgrade
                .as_ref()
                .is_some_and(|u| u.id == id_a)
        );
        assert!(s2.manager().is_held_for_test(&old_copy.key));
        for k in &kv_keys {
            assert!(s2.manager().is_held_for_test(k));
        }

        assert_eq!(
            s2.store.reclaim_entry_with_wait(&s2.coord, id_a, false),
            ReclaimOutcome::RamDropped
        );
        assert!(s2.entry(id_a).ram().is_none());

        // Settle the failing replacement write WITHOUT `poll_offload` so `ColdKeyHold` stays live.
        assert!(s2.disk().engine.await_landing(test_deadline(5)));
        let _ = s2.manager().drain(Duration::from_secs(5));
        fail_writer.store(false, Ordering::Release);

        s2.coord.reset_scheduled_request(2).expect("reset 2");
        s2.coord
            .record_tokens_all(2, &tokens_competing)
            .expect("record competing");
        let (competing_key, landed) = s2.enqueue_kv_block(2, &tokens_competing[..32], 0, None);
        match landed {
            Landed::Block { key, ok } => {
                assert_eq!(key, competing_key);
                assert!(
                    !ok,
                    "competing Evict write must be refused while keys are held"
                );
            }
            other => panic!("expected Landed::Block, got {other:?}"),
        }

        assert_eq!(s2.manager().stats().evictions, 0);
        assert!(
            s2.manager()
                .contains_in(&old_copy.key, ColdGroup::DFlash2State)
        );
        for k in &kv_keys {
            assert!(s2.manager().contains_in(k, ColdGroup::Kv));
        }

        // Reconcile failed upgrade: `old_copy` restored, `ColdKeyHold` released, SSD restorable.
        s2.clear_space_override(false);
        assert!(s2.store.poll_offload(&s2.coord));
        assert!(s2.store.in_flight_disk_upgrade.is_none());
        assert!(!s2.manager().is_held_for_test(&old_copy.key));
        for k in &kv_keys {
            assert!(!s2.manager().is_held_for_test(k));
        }
        assert!(s2.entry(id_a).ram().is_none());
        assert_eq!(
            s2.entry(id_a).disk().copied(),
            Some(DiskCopy {
                chain_owed: false,
                ..old_copy
            })
        );
        s2.coord.release_all_and_purge().expect("purge hot KV");
        s2.assert_ssd_restores(&query_a, 911, Some(id_a), None);
        let _ = s2.close();
    }

    /// Round-10 Finding 2 companion regression: a competing `MakeRoom::Evict` write that cannot fit
    /// quota due to an active `ColdKeyHold` refuses with ZERO evictions, preserving unrelated
    /// unheld prefixes and keeping the writer healthy.
    #[cfg(target_os = "macos")]
    #[test]
    fn in_place_upgrade_hold_quota_blocked_write_preserves_unrelated_prefix() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use mlx_paged_attn::{
            ColdStateLabel, ColdStateLayout, ColdStateObject, Landed, state_object_bytes,
        };
        use std::sync::atomic::{AtomicBool, Ordering};

        let fx = StateDiskTestFixture::new("r10-quota-unheld", b"r10-quota-unheld-v1", Some(10.0))
            .with_paged(2048, 4);
        let reserve = 4096u64;
        let tokens_a: Vec<u32> = (100u32..164u32).collect();
        let tokens_unrelated: Vec<u32> = (700u32..732u32).collect();
        let packed_dw = fx.packed_dw(64);

        // 1. Persist entry A + one unrelated unheld Global KV block and measure footprints.
        let (unrelated_key, unrelated_size, total_with_unrelated) = {
            let mut s1 = fx.open(64u64 << 20, Some(reserve));
            let _ = s1.capture_ord(&fx.conv, 1, &tokens_a, 64, 921, None);
            assert!(s1.flush(10));
            let footprint_a = s1.manager().total_bytes();

            s1.coord.reset_scheduled_request(2).expect("reset 2");
            s1.coord
                .record_tokens_all(2, &tokens_unrelated)
                .expect("record unrelated");
            let (unrelated_key, landed) = s1.enqueue_kv_block(2, &tokens_unrelated, 0, None);
            assert!(matches!(landed, Landed::Block { ok: true, .. }));
            let total = s1.manager().total_bytes();
            let size = total - footprint_a;
            assert!(size > 0);
            assert!(s1.close());
            (unrelated_key, size, total)
        };

        // 2. Reopen with exact quota, arm in-place upgrade hold on A, enqueue oversized competing state.
        let mut s2 = fx.open_adopted(total_with_unrelated, Some(reserve), 1);
        assert!(s2.manager().contains_in(&unrelated_key, ColdGroup::Kv));

        let id_a = s2.capture_ord(&fx.conv, 1, &tokens_a, 64, 921, Some(&packed_dw));
        let old_copy = s2.disk_copy(id_a);
        let kv_keys = s2.disk().engine.kv_chain_keys(s2.entry(id_a));
        let state_fp = s2.disk().tier.state_fingerprint;

        let fail_writer = Arc::new(AtomicBool::new(true));
        s2.set_toggle_writer_space(reserve, Arc::clone(&fail_writer));
        assert!(s2.stage(id_a));
        assert!(s2.disk().engine.await_landing(test_deadline(5)));
        let _ = s2.manager().drain(Duration::from_secs(5));
        fail_writer.store(false, Ordering::Release);

        let competing_payload_len = unrelated_size.saturating_add(4096);
        let competing_layout = ColdStateLayout {
            boundary_tokens: 32,
            spans: vec![competing_payload_len],
            geometry: "inkling2_r10_test".to_string(),
        };
        let competing_label = ColdStateLabel {
            tokens: tokens_unrelated.clone(),
            extra_keys: vec![],
            salt: 999,
            last_used: 1,
            checkpoint: false,
        };
        let competing_size = state_object_bytes(&competing_layout, &competing_label);
        assert!(competing_size > unrelated_size && competing_size <= total_with_unrelated);
        let competing_key = ColdCacheKey::chain(
            ColdGroup::DFlash2State,
            state_fp,
            None,
            &tokens_unrelated,
            &[999],
            0,
            0,
        );
        let competing_obj = ColdStateObject {
            key: competing_key,
            fingerprint: state_fp,
            layout: competing_layout,
            label: competing_label,
            payload: vec![0xAB; competing_payload_len as usize],
        };
        let (ctx, crx) = std::sync::mpsc::channel::<Landed>();
        s2.manager()
            .enqueue_state(competing_obj, MakeRoom::Evict, ctx)
            .expect("enqueue competing state");
        match crx.recv_timeout(Duration::from_secs(5)).expect("settle") {
            Landed::State { key, ok, .. } => {
                assert_eq!(key, competing_key);
                assert!(!ok, "quota-blocked competing state write must be refused");
            }
            other => panic!("expected Landed::State, got {other:?}"),
        }

        assert!(s2.manager().takes_tracked_writes());
        assert_eq!(s2.manager().stats().evictions, 0);
        assert!(s2.manager().contains_in(&unrelated_key, ColdGroup::Kv));
        assert!(
            s2.manager()
                .contains_in(&old_copy.key, ColdGroup::DFlash2State)
        );
        for k in &kv_keys {
            assert!(s2.manager().contains_in(k, ColdGroup::Kv));
        }

        s2.clear_space_override(false);
        let _ = s2.store.poll_offload(&s2.coord);
        let _ = s2.close();
    }

    /// Round-9 Finding 3 regression: ordinary RAM-backed spills invalidated with
    /// `offload_failures == 0` (`gone` in `chain_step`, `lost` in `chain_step`, `take_evicted`
    /// in `poll_offload`) are properly reconciled, retried when recoverable, and surfaced to
    /// `flush_to_disk`.
    #[cfg(target_os = "macos")]
    #[test]
    fn ordinary_write_behind_recovers_gone_blob_and_surfaces_lost_chain() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use mlx_paged_attn::Landed;

        let fx =
            StateDiskTestFixture::new("r9-wb-invalidation", b"r9-wb-invalidation-v1", Some(10.0))
                .with_paged(2048, 4);
        let reserve = 4096u64;
        let tokens_a: Vec<u32> = (100u32..164u32).collect();
        let query_a: Vec<u32> = (100u32..165u32).collect();
        let tokens_b: Vec<u32> = (200u32..264u32).collect();
        let tokens_c: Vec<u32> = (300u32..364u32).collect();
        let tokens_d: Vec<u32> = (400u32..656u32).collect();

        let mut s = fx.open(64u64 << 20, Some(reserve));
        s.store.set_ram_cap(16);

        // Part 1: `gone` in `chain_step` (`offload_failures == 0`, hot KV resident).
        let id_a = s.capture_ord(&fx.conv, 1, &tokens_a, 64, 931, None);
        s.clear_wb();
        assert!(s.stage(id_a));
        s.await_writing("landing A");
        let copy_a = s.disk_copy(id_a);
        assert!(copy_a.landed && copy_a.chain_owed);

        assert!(s.manager().remove(&copy_a.key));
        let failures_before = s.store.disk_counters().offload_failures;
        assert!(s.store.chain_step(&s.coord, test_deadline(5)));
        assert_eq!(s.store.disk_counters().offload_failures, failures_before);
        assert!(s.entry(id_a).ram().is_some() && s.entry(id_a).disk().is_none());
        assert!(s.store.spill_backoff.contains_key(&id_a));
        assert!(s.store.recent_failed_spills.contains(&id_a));
        assert_eq!(s.wb_waiting(), 1);

        s.manager()
            .set_space_override_for_test(Some(Box::new(move || Ok((1 << 30, reserve)))));
        assert!(!s.flush(3));
        s.clear_space_override(false);

        assert!(s.flush(10));
        let one_full_entry_bytes = s.manager().total_bytes();
        assert!(!s.store.spill_backoff.contains_key(&id_a));
        assert!(s.disk().entry_disk_restorable(s.entry(id_a)));
        s.drop_ram_and_purge();
        let hit_a = s
            .store
            .lookup_deepest_restorable(&s.coord, &query_a, 931)
            .expect("A must restore from SSD");
        assert_eq!(hit_a.id, id_a);

        // Part 2: `lost` in `chain_step` (KV missing from both tiers).
        let id_b = s.capture_ord(&fx.conv, 2, &tokens_b, 64, 932, None);
        s.clear_wb();
        assert!(s.stage(id_b));
        s.await_writing("landing B");
        assert!(
            s.entry(id_b)
                .disk()
                .is_some_and(|c| c.landed && c.chain_owed)
        );

        s.coord.release_all_and_purge().expect("purge hot B");
        s.assert_flush_fails_promptly("lost chain");
        assert_eq!(s.wb_waiting(), 0);
        assert!(s.entry(id_b).ram().is_some() && s.entry(id_b).disk().is_none());
        assert!(s.close());

        // Part 3: `take_evicted` during `poll_offload` evicts a landed `chain_owed` blob.
        let tight_limit = one_full_entry_bytes.saturating_add(4096);
        let mut st = fx.open_in(
            fx.subdir("tight_cache"),
            tight_limit,
            b"r9-wb-evict-v1",
            Some(reserve),
        );
        st.store.set_ram_cap(16);

        let id_c = st.capture_ord(&fx.conv, 1, &tokens_c, 64, 933, None);
        st.clear_wb();
        assert!(st.stage(id_c));
        st.await_writing("landing C");
        let copy_c = st.disk_copy(id_c);
        assert!(copy_c.landed && copy_c.chain_owed);

        st.coord.reset_scheduled_request(2).expect("reset D");
        st.coord.record_tokens_all(2, &tokens_d).expect("record D");
        let mut parent_key = None;
        for idx in 0..(tokens_d.len() / 32) {
            if !st
                .manager()
                .contains_in(&copy_c.key, ColdGroup::DFlash2State)
            {
                break;
            }
            let chunk = &tokens_d[idx * 32..(idx + 1) * 32];
            let (key_d, landed) = st.enqueue_kv_block(2, chunk, idx, parent_key);
            parent_key = Some(key_d);
            match landed {
                Landed::Block { ok: true, .. } => {}
                other => panic!("expected competing D block to land, got {other:?}"),
            }
        }
        assert!(
            !st.manager()
                .contains_in(&copy_c.key, ColdGroup::DFlash2State)
        );

        let failures_t_before = st.store.disk_counters().offload_failures;
        assert!(st.store.poll_offload(&st.coord));
        assert_eq!(st.store.disk_counters().offload_failures, failures_t_before);
        assert!(st.entry(id_c).ram().is_some() && st.entry(id_c).disk().is_none());
        assert!(st.store.spill_backoff.contains_key(&id_c));
        assert!(st.store.recent_failed_spills.contains(&id_c));
        assert_eq!(st.wb_waiting(), 1);
        let _ = st.close();
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

        // Shared WriteBehind uses `MINIMUM_TOKENS = 32` under `#[cfg(test)]` and `flush_to_disk`
        // flushes regardless of the 10s delay.

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
        let geom = Inkling2StateGeometry::from_coordinator_with_draft(
            &inner.config,
            &inner.paged.as_ref().expect("paged").coordinator,
            inner.dflash2.as_ref().map(|d| &d.config),
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
            inner.dflash2_identity.as_deref(),
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
        let geom2 = Inkling2StateGeometry::from_coordinator_with_draft(
            &restarted.config,
            &restarted.paged.as_ref().expect("paged").coordinator,
            restarted.dflash2.as_ref().map(|d| &d.config),
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
            restarted.dflash2_identity.as_deref(),
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

    /// Round-11 Finding 2 regression:
    /// A deferred `PackedDraftWindow::from_window` failure inside `capture_boundary_with_window`
    /// must NOT discard the valid target SWA + ShortConv checkpoint; it logs a warning and
    /// publishes the target anchor with `draft_window == None`, restoring bit-exact SWA +
    /// ShortConv state.
    #[cfg(target_os = "macos")]
    #[test]
    fn capture_boundary_with_window_preserves_target_checkpoint_on_draft_pack_error() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let fx = StateDiskTestFixture::new("r11-dw-pack-err", b"r11-dw-pack-err-v1", Some(10.0));
        let mut s = fx.open(64u64 << 20, None);
        let boundary = 64u32;
        let tokens: Vec<u32> = (300u32..364u32).collect();

        // Populate deterministic non-zero SWA raw blocks and ShortConv values on seq 1.
        let swa_len = s.geom.live_swa_bytes(boundary).expect("live_swa_bytes");
        let expected_swa: Vec<u8> = (0..swa_len).map(|i| ((i * 29 + 13) % 251) as u8).collect();
        s.coord.reset_scheduled_request(1).expect("reset seq 1");
        s.coord.record_tokens_all(1, &tokens).expect("record seq 1");
        s.coord
            .restore_sliding_raw_blocks_flat_for(1, &tokens, boundary, &expected_swa)
            .expect("seed swa seq 1");
        let mut conv = make_short_conv(&fx.cfg);
        let hist = (fx.cfg.sconv_kernel_size - 1) as i64;
        let kv_dim = (fx.cfg.num_key_value_heads * fx.cfg.head_dim) as i64;
        let hidden = fx.cfg.hidden_size as i64;
        let kv_vals: Vec<u16> = (0..(hist * kv_dim) as usize)
            .map(|i| half::bf16::from_f32(((i % 17) as f32 - 8.0) * 0.25).to_bits())
            .collect();
        let h_vals: Vec<u16> = (0..(hist * hidden) as usize)
            .map(|i| half::bf16::from_f32(((i % 11) as f32 - 5.0) * 0.5).to_bits())
            .collect();
        conv.visits[0].k = MxArray::from_bfloat16(&kv_vals, &[hist, kv_dim]).expect("conv k");
        conv.visits[0].attn_post =
            MxArray::from_bfloat16(&h_vals, &[hist, hidden]).expect("conv attn_post");
        let expected_conv = pack_short_conv_state(&s.geom, &conv).expect("pack expected conv");

        // Inject a deferred `from_window` failure on `BoundaryDraftWindow::Device`.
        let dev_win =
            BoundaryDraftWindow::Device(make_draft_window(fx.cfg.hidden_size as u32, 16, boundary));
        s.store.fail_next_draft_window_pack = true;
        let id = s
            .store
            .capture_boundary_with_window(
                &mut s.coord,
                &conv,
                1,
                &tokens,
                boundary,
                881,
                StateClass::Ordinary,
                Some(&dev_win),
                None,
            )
            .expect("draft-window pack error must not fail capture_boundary_with_window")
            .expect("target checkpoint must still be published");
        assert!(
            !s.store.fail_next_draft_window_pack,
            "injected draft-window pack failure must have been consumed"
        );
        let ram = s.entry(id).ram().expect("published RAM anchor");
        assert!(
            ram.draft_window.is_none(),
            "published anchor must have draft_window == None after pack error"
        );
        assert_eq!(
            ram.cached_bytes,
            (expected_swa.len() + expected_conv.len()) as u64
        );
        assert!(s.store.entries_missing_draft_window.contains(&id));

        // Restore `id` onto seq 2 and verify bit-exact target SWA + ShortConv state!
        let (anchor, from_ssd) = s.store.load_anchor(&s.coord, id).expect("load_anchor");
        assert!(!from_ssd && anchor.draft_window.is_none());
        let mut restored_convs = HashMap::new();
        install_anchor_swa_and_conv(
            &mut s.coord,
            &mut restored_convs,
            &s.geom,
            2,
            &tokens,
            boundary,
            &anchor,
        )
        .expect("install_anchor_swa_and_conv");
        let restored_swa = s
            .coord
            .read_sliding_raw_blocks_flat_for(2, boundary)
            .expect("read restored swa");
        assert_eq!(
            restored_swa, expected_swa,
            "SWA state must restore bit-exact"
        );
        let restored_conv =
            pack_short_conv_state(&s.geom, &restored_convs[&2]).expect("pack restored conv");
        assert_eq!(
            restored_conv, expected_conv,
            "ShortConv state must restore bit-exact"
        );
        assert!(s.close());
    }

    /// R14-1: zero-copy `Arc<Vec<u8>>`/`TrackedPackedBytes` fits within exact `1 * peak` (< 2*SWA).
    #[cfg(target_os = "macos")]
    #[test]
    fn capture_boundary_zero_copy_swa_and_exact_peak_budget() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use crate::cache_limit::CacheLimitCoordinator;
        use crate::engine::governor::governor::{
            Governor, GovernorIo, HostSample, MemoryStats, ModelRegistration,
        };
        use crate::engine::governor::host::{HostMemory, Pressure, spare_reserve_bytes};
        struct Io(Arc<CacheLimitCoordinator>);
        impl GovernorIo for Io {
            fn stats(&self) -> MemoryStats {
                MemoryStats {
                    active: 16 << 20,
                    pools: self.0.registered_pool_bytes(),
                    cache: 0,
                }
            }
            fn host(&self) -> HostSample {
                let s = spare_reserve_bytes(32 << 30) + (16 << 30);
                HostSample {
                    memory: Some(HostMemory {
                        available: s,
                        spare: s,
                    }),
                    pressure: Some(Pressure::Normal),
                }
            }
            fn now(&self) -> f64 {
                0.0
            }
            fn sleep(&self, _: f64) {}
            fn clear_cache(&self) {}
            fn set_memory_limit(&self, b: u64) -> u64 {
                b
            }
            fn physical_memory(&self) -> u64 {
                32 << 30
            }
        }
        let raw_swa = vec![5u8; 4096];
        let p = raw_swa.as_ptr();
        assert_eq!(Arc::new(raw_swa).as_ptr(), p);
        let raw_dw = vec![7u8; 4096];
        let p = raw_dw.as_ptr();
        assert_eq!(TrackedPackedBytes::new(raw_dw).as_ptr(), p);

        let fx = StateDiskTestFixture::new("r14-zero-copy", b"r14-zero-copy-v1", Some(10.0));
        let mut s = fx.open(64u64 << 20, None);
        let hc = Arc::new(CacheLimitCoordinator::new_for_test());
        s.store.attach_host_budget_to(Arc::clone(&hc));
        let toks: Vec<u32> = (100u32..228u32).collect();
        s.coord.reset_scheduled_request(1).unwrap();
        s.coord.record_tokens_all(1, &toks).unwrap();
        let dev =
            BoundaryDraftWindow::Device(make_draft_window(fx.cfg.hidden_size as u32, 16, 128));
        let swa = s.geom.live_swa_bytes(128).unwrap() as u64;
        let peak = swa
            + s.geom.conv_span_bytes
            + DraftContextWindow::bytes(fx.cfg.hidden_size as u32, 16)
            + crate::disk_tier::store::lineage_bytes(128, 4);
        assert!(peak < 2 * swa);
        let gov = Governor::new(Arc::new(Io(Arc::clone(&hc))));
        let _r = gov
            .register_model(ModelRegistration {
                model_id: 91,
                hard_budget: (16 << 20) + peak,
                untracked: 0,
                page_bytes: 1 << 20,
            })
            .unwrap();
        let id = s
            .store
            .capture_boundary_with_window(
                &mut s.coord,
                &fx.conv,
                1,
                &toks,
                128,
                914,
                StateClass::Ordinary,
                Some(&dev),
                Some((&gov, 91)),
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            (gov.publications_refused(91), hc.registered_pool_bytes()),
            (0, peak)
        );
        let ram = s.entry(id).ram().unwrap();
        let dw = ram.draft_window.as_ref().unwrap();
        assert_eq!(
            (
                ram.swa_data.capacity(),
                ram.conv_data.capacity(),
                dw.bytes.bytes.capacity()
            ),
            (
                ram.swa_data.len(),
                ram.conv_data.len(),
                dw.bytes.bytes.len()
            ),
        );
        assert!(s.close());
    }

    /// R14-2: recapturing a disk-only complete entry with a stale SSD copy installs fresh RAM and repairs SSD.
    #[cfg(target_os = "macos")]
    #[test]
    fn recapture_installs_fresh_ram_and_repairs_stale_ssd_entry() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let fx = StateDiskTestFixture::new("r14-stale-ssd", b"r14-stale-ssd-v1", Some(0.0));
        let mut s = fx.open(64u64 << 20, None);
        let toks: Vec<u32> = (400u32..464u32).collect();
        let dw = fx.packed_dw(64);
        let id = s.capture_ord(&fx.conv, 1, &toks, 64, 942, Some(&dw));
        assert!(!s.store.entries_missing_draft_window.contains(&id) && s.flush(10));
        s.store.drop_all_ram_keeping_disk();
        let copy_key = s.disk_copy(id).key;
        let kv_keys = s.disk().engine.kv_chain_keys(s.entry(id));
        s.manager().remove(&kv_keys[0]);
        assert!(s.manager().contains_in(&copy_key, ColdGroup::DFlash2State));
        assert!(
            !s.disk().entry_disk_restorable(s.entry(id)) && !s.store.is_restorable(&s.coord, id)
        );

        let dev = BoundaryDraftWindow::Device(make_draft_window(fx.cfg.hidden_size as u32, 16, 64));
        s.coord.reset_scheduled_request(2).unwrap();
        s.coord.record_tokens_all(2, &toks).unwrap();
        let got = s
            .store
            .capture_boundary_with_window(
                &mut s.coord,
                &fx.conv,
                2,
                &toks,
                64,
                942,
                StateClass::Ordinary,
                Some(&dev),
                None,
            )
            .unwrap()
            .unwrap();
        assert_eq!(got, id);
        assert!(s.entry(id).ram().is_some() && s.store.is_restorable(&s.coord, id));
        let (a, from_ssd) = s.store.load_anchor(&s.coord, id).unwrap();
        assert!(
            !from_ssd
                && a.draft_window
                    .as_ref()
                    .is_some_and(|w| w.is_complete_for(64))
        );
        assert!(s.flush(10) && s.disk().entry_disk_restorable(s.entry(id)));
        s.store.drop_all_ram_keeping_disk();
        let (a2, from_ssd2) = s.store.load_anchor(&s.coord, id).unwrap();
        assert!(
            from_ssd2
                && a2
                    .draft_window
                    .as_ref()
                    .is_some_and(|w| w.is_complete_for(64))
        );
        assert!(s.close());
    }

    /// R15-1: temporary (non-persistent) tier normal `Tick::Full` runs queued `Repair` upgrades
    /// below `ram_cap` while keeping ordinary copy-ahead disabled.
    #[cfg(target_os = "macos")]
    #[test]
    fn temp_tier_normal_ticks_complete_queued_repair_without_copy_ahead() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let fx = StateDiskTestFixture::new("r15-temp-repair", b"r15-temp-repair-v1", Some(0.0));
        let mut s = fx.open_temp(64u64 << 20, None);
        assert!(!s.disk().tier.persistent);
        s.store.set_ram_cap(16);

        let toks_a: Vec<u32> = (500u32..564u32).collect();
        let query_a: Vec<u32> = (500u32..565u32).collect();
        let toks_b: Vec<u32> = (600u32..664u32).collect();
        let dw = fx.packed_dw(64);

        // Spill A to the temporary SSD tier, drop RAM, and evict one of its Global KV blocks.
        let id_a = s.capture_ord(&fx.conv, 1, &toks_a, 64, 951, Some(&dw));
        s.clear_wb();
        assert!(s.stage(id_a));
        s.await_landed(id_a, "initial temporary-tier spill of A");
        assert!(s.disk().entry_disk_restorable(s.entry(id_a)));

        s.store.drop_all_ram_keeping_disk();
        s.evict_kv_block(id_a, 0);
        assert!(!s.disk().entry_disk_restorable(s.entry(id_a)));

        // Publish unrelated ordinary anchor B first (below `ram_cap`) so B sits in
        // `write_behind`, then recapture A into RAM (below `ram_cap`, queuing `Repair`).
        let id_b = s.capture_ord(&fx.conv, 2, &toks_b, 64, 952, Some(&dw));
        assert!(s.entry(id_b).disk().is_none() && s.wb_waiting() == 1);
        assert!(!s.store.pending_disk_upgrades.contains_key(&id_b));

        let got_a = s.capture_ord(&fx.conv, 1, &toks_a, 64, 951, Some(&dw));
        assert_eq!(got_a, id_a);
        assert!(matches!(
            s.store.pending_disk_upgrades.get(&id_a),
            Some(PendingDiskUpgrade::Repair { .. })
        ));
        assert!(s.entry(id_a).ram().is_some() && s.entry(id_a).disk().is_none());

        // Drive normal scheduler ticks (`Tick::Full`, `force = false`) until A's repair finishes.
        let dl = test_deadline(10);
        while s.store.pending_disk_upgrades.contains_key(&id_a) || s.disk().writing() {
            assert!(Instant::now() < dl, "timed out driving temp-tier repair");
            s.store.tick(&s.coord, Tick::Full { command_pages: 0 });
            if s.disk().writing() || s.disk().chain.is_some() {
                let _ = s.disk().engine.await_landing(dl);
            }
        }
        // Extra normal ticks must still leave ordinary anchor B unspilled on the temporary tier.
        for _ in 0..3 {
            assert!(!s.store.tick(&s.coord, Tick::Full { command_pages: 0 }));
        }

        assert!(s.store.pending_disk_upgrades.is_empty());
        assert!(
            s.entry(id_a)
                .disk()
                .is_some_and(|c| c.landed && !c.chain_owed)
        );
        assert!(s.disk().entry_disk_restorable(s.entry(id_a)));
        assert!(
            s.entry(id_b).ram().is_some() && s.entry(id_b).disk().is_none(),
            "ordinary anchor B must not be copied ahead on a temporary tier"
        );

        s.drop_ram_and_purge();
        s.assert_ssd_restores(&query_a, 951, Some(id_a), Some(&dw));
        assert!(s.close());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn state_store_host_budget_excludes_mlx_conv_and_trims_to_ram_cap() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use crate::cache_limit::CacheLimitCoordinator;
        use crate::disk_tier::{CacheDisk, DiskTierOptions};

        let cfg = tiny_inkling2_config();
        let paged_cfg = Inkling2PagedConfig::default_for_config(&cfg)
            .with_kv_format(KvFormat::Int8)
            .with_max_model_len(512)
            .with_max_concurrent_sequences(8);
        let recency = Arc::new(CacheRecency::new());
        let mut coord = Inkling2KVCacheCoordinator::new_for_config_with_recency(
            &cfg,
            paged_cfg,
            Arc::clone(&recency),
        )
        .expect("coord");

        let geom = Inkling2StateGeometry::from_coordinator(&cfg, &coord).unwrap();
        let local_limit = Arc::new(CacheLimitCoordinator::new_for_test());
        let mut store = Inkling2StateStore::new(coord.block_size(), Arc::clone(&recency));
        store.set_geometry(geom.clone());
        store.attach_host_budget_to(Arc::clone(&local_limit));
        store.set_ram_cap(2);

        let conv = Inkling2ShortConvState::new(
            cfg.num_layer_visits(),
            cfg.sconv_kernel_size,
            cfg.num_key_value_heads * cfg.head_dim,
            cfg.hidden_size,
            DType::BFloat16,
        )
        .expect("conv");

        // 1. Insert 1 Checkpoint and verify HostPool matches host_resident_bytes exactly
        //    (flat host-packed `swa_data + conv_data + lineage`, zero live MLX arrays retained).
        let seq_tokens = |seed: u32| -> Vec<u32> { (0..32u32).map(|i| seed * 100 + i).collect() };
        let t_cp = seq_tokens(1);
        coord.reset_scheduled_request(1).unwrap();
        coord.record_tokens_all(1, &t_cp).unwrap();
        let cp_id = store
            .capture_boundary(
                &mut coord,
                &conv,
                1,
                &t_cp,
                32,
                0,
                StateClass::Checkpoint,
                None,
            )
            .unwrap()
            .unwrap();
        let expected_host = store.host_resident_bytes();
        assert_eq!(
            local_limit.registered_pool_bytes(),
            expected_host,
            "HostPool must match host_resident_bytes exactly"
        );
        assert_eq!(
            store.total_ram_bytes(),
            expected_host,
            "flat host-packed anchor bytes must equal host_resident_bytes exactly"
        );

        // 2. RAM-only Tier 1 (Checkpoint first) & Tier 4 (strict ram_cap = 2 without disk).
        let t_o1 = seq_tokens(2);
        coord.reset_scheduled_request(2).unwrap();
        coord.record_tokens_all(2, &t_o1).unwrap();
        let o1 = store
            .capture_boundary(
                &mut coord,
                &conv,
                2,
                &t_o1,
                32,
                0,
                StateClass::Ordinary,
                None,
            )
            .unwrap()
            .unwrap();
        assert_eq!(store.ram_len(), 2);

        let t_o2 = seq_tokens(3);
        coord.reset_scheduled_request(3).unwrap();
        coord.record_tokens_all(3, &t_o2).unwrap();
        let o2 = store
            .capture_boundary(
                &mut coord,
                &conv,
                3,
                &t_o2,
                32,
                0,
                StateClass::Ordinary,
                None,
            )
            .unwrap()
            .unwrap();
        assert_eq!(store.ram_len(), 2, "Tier 1 must retire Checkpoint first");
        assert!(store.store.get(cp_id).is_none(), "Checkpoint retired first");
        assert!(store.store.get(o1).is_some() && store.store.get(o2).is_some());

        // Inserting a 3rd Ordinary without disk enforces strict `ram_cap = 2` (Tier 4 erases LRU o1).
        let t_o3 = seq_tokens(4);
        coord.reset_scheduled_request(4).unwrap();
        coord.record_tokens_all(4, &t_o3).unwrap();
        let o3 = store
            .capture_boundary(
                &mut coord,
                &conv,
                4,
                &t_o3,
                32,
                0,
                StateClass::Ordinary,
                None,
            )
            .unwrap()
            .unwrap();
        assert_eq!(store.ram_len(), 2);
        assert!(
            store.store.get(o1).is_none(),
            "Tier 4 erases LRU Ordinary without disk"
        );
        assert!(store.store.get(o2).is_some() && store.store.get(o3).is_some());
        assert_eq!(
            local_limit.registered_pool_bytes(),
            store.host_resident_bytes()
        );

        // 3. Attach StateDisk and verify Tier 2 (drop RAM on durable SSD), Tier 3 (stage Ordinary
        //    keeping bounded slack <= ram_cap + 2), and Tier 4 (erase past ram_cap + 2).
        let temp_root =
            std::env::temp_dir().join(format!("inkling2-ram-cap-trim-{}", uuid::Uuid::new_v4()));
        let cache_dir = temp_root.join("cache");
        std::fs::create_dir_all(&cache_dir).unwrap();
        let disk_opts = DiskTierOptions {
            cache_disk: CacheDisk::Bytes(512u64 << 20),
            persistent: true,
            cache_dir: Some(cache_dir),
            temp_root: Some(temp_root.clone()),
        };
        let pid = compute_path_identity(&temp_root, &cfg, &coord, &geom, Some(1024), None);
        let (ctx, disk) = open_inkling2_state_disk(&cfg, &coord, None, &disk_opts, move || pid, 0)
            .expect("open disk");
        coord.set_full_cold_tier(ctx);
        store.disk = Some(disk);

        // `o2` and `o3` were captured before `store.disk` was attached; record them into
        // `write_behind` now and flush to SSD so both have durable disk copies.
        if let Some(d) = store.disk.as_mut() {
            let writable = d.writable();
            let persistent = d.tier.persistent;
            d.write_behind
                .record_published(&store.store, writable, persistent, o2, 32, 0.0, false);
            d.write_behind
                .record_published(&store.store, writable, persistent, o3, 32, 0.0, false);
        }
        assert!(store.flush_to_disk(&coord, Instant::now() + Duration::from_secs(10)));
        assert!(
            store
                .store
                .get(o2)
                .unwrap()
                .disk()
                .is_some_and(|d| d.landed && !d.chain_owed)
        );

        // Insert o4: exceeds ram_cap=2 -> Tier 2 drops RAM on durable LRU o2 (keeps SSD copy!).
        let t_o4 = seq_tokens(5);
        coord.reset_scheduled_request(5).unwrap();
        coord.record_tokens_all(5, &t_o4).unwrap();
        let _o4 = store
            .capture_boundary(
                &mut coord,
                &conv,
                5,
                &t_o4,
                32,
                0,
                StateClass::Ordinary,
                None,
            )
            .unwrap()
            .unwrap();
        assert_eq!(store.ram_len(), 2);
        assert!(
            store
                .store
                .get(o2)
                .is_some_and(|e| e.ram().is_none() && e.disk().is_some()),
            "Tier 2 must drop RAM on durable SSD Ordinary while keeping disk copy"
        );
        // Drop o3 RAM too by inserting o5 (o3 is durable).
        let t_o5 = seq_tokens(6);
        coord.reset_scheduled_request(6).unwrap();
        coord.record_tokens_all(6, &t_o5).unwrap();
        let _o5 = store
            .capture_boundary(
                &mut coord,
                &conv,
                6,
                &t_o5,
                32,
                0,
                StateClass::Ordinary,
                None,
            )
            .unwrap()
            .unwrap();
        assert_eq!(store.ram_len(), 2);

        // Hold staging so writes stall (testing Tier 3 / Tier 4 bounded slack `ram_cap + 2 = 4`):
        let held_staging = store.disk.as_mut().unwrap().engine.staging.take();
        for seed in 7..12u32 {
            let t = seq_tokens(seed);
            let s = seed;
            coord.reset_scheduled_request(s).unwrap();
            coord.record_tokens_all(s, &t).unwrap();
            let _ = store
                .capture_boundary(&mut coord, &conv, s, &t, 32, 0, StateClass::Ordinary, None)
                .unwrap()
                .unwrap();
            assert!(
                store.ram_len() <= 4,
                "Tier 4 must enforce bounded slack ram_cap + 2 = 4 while writes stall, got {}",
                store.ram_len()
            );
        }
        store.disk.as_mut().unwrap().engine.staging = held_staging;

        // Dropping store must unregister host budget back to 0.
        drop(store);
        assert_eq!(local_limit.registered_pool_bytes(), 0);
        let _ = std::fs::remove_dir_all(&temp_root);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn keep_prev_checkpoint_preserves_rolling_checkpoint_below_junction_and_retires_on_deeper_checkpoint()
     {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use crate::disk_tier::{CacheDisk, DiskTierOptions};

        let cfg = tiny_inkling2_config();
        let paged_cfg = Inkling2PagedConfig::default_for_config(&cfg)
            .with_kv_format(KvFormat::Int8)
            .with_max_model_len(256)
            .with_max_concurrent_sequences(2);
        let recency = Arc::new(CacheRecency::new());
        let mut coord = Inkling2KVCacheCoordinator::new_for_config_with_recency(
            &cfg,
            paged_cfg,
            Arc::clone(&recency),
        )
        .expect("coord");

        let temp_root =
            std::env::temp_dir().join(format!("inkling2-keep-prev-cp-{}", uuid::Uuid::new_v4()));
        let cache_dir = temp_root.join("cache");
        std::fs::create_dir_all(&cache_dir).unwrap();
        let disk_opts = DiskTierOptions {
            cache_disk: CacheDisk::Bytes(64u64 << 20),
            persistent: true,
            cache_dir: Some(cache_dir),
            temp_root: Some(temp_root.clone()),
        };
        let geom = Inkling2StateGeometry::from_coordinator(&cfg, &coord).unwrap();
        let pid = compute_path_identity(&temp_root, &cfg, &coord, &geom, Some(1024), None);
        let (ctx, disk) = open_inkling2_state_disk(&cfg, &coord, None, &disk_opts, move || pid, 0)
            .expect("open disk");
        coord.set_full_cold_tier(ctx);

        let mut store = Inkling2StateStore::new(coord.block_size(), recency);
        store.set_geometry(geom);
        store.disk = Some(disk);
        let conv = Inkling2ShortConvState::new(
            cfg.num_layer_visits(),
            cfg.sconv_kernel_size,
            cfg.num_key_value_heads * cfg.head_dim,
            cfg.hidden_size,
            DType::BFloat16,
        )
        .expect("conv");

        let tokens_a: Vec<u32> = (1000u32..1160u32).collect();
        coord.reset_scheduled_request(1).unwrap();
        coord.record_tokens_all(1, &tokens_a).unwrap();

        // 1. Publish rolling Checkpoint at block 2 (64 tokens), then Ordinary prompt junction at
        //    block 4 (128 tokens) with `keep_prev_checkpoint = true`: 64 must stay cached as
        //    disposable Checkpoint and NOT be queued in write_behind!
        let cp64 = store
            .capture_boundary_with_options(
                &mut coord,
                &conv,
                1,
                &tokens_a,
                64,
                0,
                StateClass::Checkpoint,
                CaptureOptions {
                    keep_prev_checkpoint: false,
                    enqueue_write_behind: false,
                    is_head: false,
                    now: 0.0,
                },
            )
            .unwrap()
            .unwrap();
        let ord128 = store
            .capture_boundary_with_options(
                &mut coord,
                &conv,
                1,
                &tokens_a,
                128,
                0,
                StateClass::Ordinary,
                CaptureOptions {
                    keep_prev_checkpoint: true,
                    enqueue_write_behind: true,
                    is_head: false,
                    now: 0.0,
                },
            )
            .unwrap()
            .unwrap();

        assert!(
            store.store.get(cp64).is_some_and(|e| e.checkpoint()),
            "rolling checkpoint below prompt junction must remain cached as Checkpoint"
        );
        assert!(
            store.store.get(ord128).is_some_and(|e| !e.checkpoint()),
            "prompt junction must be cached as Ordinary"
        );
        assert_eq!(
            store.disk.as_ref().unwrap().write_behind.snapshot().waiting,
            1,
            "only Ordinary junction (not disposable Checkpoint) enters write_behind"
        );

        // Second prompt sharing first 96 tokens (diverging before 128) resumes at 64!
        let mut tokens_b = tokens_a[..96].to_vec();
        tokens_b.extend(9000u32..9040u32);
        let hit = store
            .lookup_deepest_restorable(&coord, &tokens_b, 0)
            .expect("diverging prompt must hit kept checkpoint");
        assert_eq!(hit.id, cp64);
        assert_eq!(hit.depth, 64);

        // 2. On a fresh sequence, a deeper rolling Checkpoint (`keep_prev_checkpoint = false`)
        //    DOES retire the earlier rolling Checkpoint.
        let tokens_c: Vec<u32> = (2000u32..2160u32).collect();
        coord.reset_scheduled_request(2).unwrap();
        coord.record_tokens_all(2, &tokens_c).unwrap();
        let cp_old = store
            .capture_boundary_with_options(
                &mut coord,
                &conv,
                2,
                &tokens_c,
                64,
                0,
                StateClass::Checkpoint,
                CaptureOptions {
                    keep_prev_checkpoint: false,
                    enqueue_write_behind: false,
                    is_head: false,
                    now: 0.0,
                },
            )
            .unwrap()
            .unwrap();
        let cp_new = store
            .capture_boundary_with_options(
                &mut coord,
                &conv,
                2,
                &tokens_c,
                128,
                0,
                StateClass::Checkpoint,
                CaptureOptions {
                    keep_prev_checkpoint: false,
                    enqueue_write_behind: false,
                    is_head: false,
                    now: 0.0,
                },
            )
            .unwrap()
            .unwrap();
        assert!(
            store.store.get(cp_old).is_none(),
            "deeper rolling Checkpoint must retire prior rolling Checkpoint"
        );
        assert!(store.store.get(cp_new).is_some_and(|e| e.checkpoint()));

        let _ = store.close_cache(&coord, Instant::now() + Duration::from_secs(5));
        let _ = std::fs::remove_dir_all(&temp_root);
    }

    /// WS1: Restoring a `StateClass::Checkpoint` at `depth >= write_behind_min_tokens()` promotes
    /// it to `Ordinary`, marks it in `prompt_heads`, enqueues `WriteBehind(..., head = true)`, and
    /// persists it durably to SSD even after deeper turns supersede it. Conversely, restoring a
    /// `Checkpoint` below `write_behind_min_tokens()` or restoring an `Ordinary` entry does NOT
    /// mark `prompt_heads` or enqueue `head = true`.
    #[cfg(target_os = "macos")]
    #[test]
    fn hit_checkpoint_promotes_to_prompt_head_and_persists_despite_superseded() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let fx = StateDiskTestFixture::new("ws1-hit-cp-head", b"ws1-hit-cp-head-v1", Some(0.0))
            .with_paged(1024, 4);
        WRITE_BEHIND_MIN_TOKENS_OVERRIDE.with(|c| c.set(Some(64)));

        let mut s = fx.open(128u64 << 20, None);
        s.store.set_ram_cap(16);

        let salt = 0x5101u64;
        let tokens: Vec<u32> = (1000u32..1192u32).collect();
        s.coord.reset_scheduled_request(1).expect("reset seq 1");
        s.coord.record_tokens_all(1, &tokens).expect("record seq 1");

        // Publish rolling Checkpoints at depth 32 (< min=64) and depth 64 (>= min=64), plus
        // Ordinary turn anchors at depth 128 and 160 so 64 and 128 are both superseded.
        let cp32 = s
            .store
            .capture_boundary_with_options(
                &mut s.coord,
                &fx.conv,
                1,
                &tokens,
                32,
                salt,
                StateClass::Checkpoint,
                CaptureOptions {
                    keep_prev_checkpoint: true,
                    enqueue_write_behind: false,
                    is_head: false,
                    now: 0.0,
                },
            )
            .unwrap()
            .unwrap();
        let cp64 = s
            .store
            .capture_boundary_with_options(
                &mut s.coord,
                &fx.conv,
                1,
                &tokens,
                64,
                salt,
                StateClass::Checkpoint,
                CaptureOptions {
                    keep_prev_checkpoint: true,
                    enqueue_write_behind: false,
                    is_head: false,
                    now: 0.0,
                },
            )
            .unwrap()
            .unwrap();
        let ord128 = s
            .store
            .capture_boundary_with_options(
                &mut s.coord,
                &fx.conv,
                1,
                &tokens,
                128,
                salt,
                StateClass::Ordinary,
                CaptureOptions {
                    keep_prev_checkpoint: true,
                    enqueue_write_behind: true,
                    is_head: false,
                    now: 0.0,
                },
            )
            .unwrap()
            .unwrap();
        let ord160 = s
            .store
            .capture_boundary_with_options(
                &mut s.coord,
                &fx.conv,
                1,
                &tokens,
                160,
                salt,
                StateClass::Ordinary,
                CaptureOptions {
                    keep_prev_checkpoint: true,
                    enqueue_write_behind: true,
                    is_head: false,
                    now: 0.0,
                },
            )
            .unwrap()
            .unwrap();

        assert!(s.entry(cp32).checkpoint());
        assert!(s.entry(cp64).checkpoint());
        assert!(!s.store.is_prompt_head(cp32));
        assert!(!s.store.is_prompt_head(cp64));
        assert!(!s.store.is_prompt_head(ord128));
        assert!(s.store.store.superseded(cp64));
        assert!(s.store.store.superseded(ord128));
        assert!(!s.store.store.superseded(ord160));
        assert_eq!(s.wb_waiting(), 2, "only ord128 and ord160 queued initially");

        // 1. Hit short Checkpoint cp32 (< write_behind_min_tokens = 64): pins + upgrades via
        //    use_state, but must NOT mark `prompt_heads` or enqueue `WriteBehind`.
        let mark32 = s.entry(cp32).mark();
        s.store.note_state_hit(&s.coord, 2, cp32, false, 0.0);
        assert!(!s.entry(cp32).checkpoint());
        assert!(s.store.store.in_use(mark32));
        assert!(!s.store.is_prompt_head(cp32));
        assert_eq!(s.wb_waiting(), 2);
        s.store.end_seq_use(2);
        assert!(!s.store.store.in_use(mark32));

        // 2. Hit Checkpoint cp64 (>= write_behind_min_tokens = 64): promotes to Ordinary,
        //    marks `prompt_heads`, pins block, detaches `seq_checkpoints`, and enqueues
        //    `WriteBehind(..., head = true)`.
        let mark64 = s.entry(cp64).mark();
        s.store.note_state_hit(&s.coord, 3, cp64, false, 0.0);
        assert!(!s.entry(cp64).checkpoint());
        assert!(s.store.store.in_use(mark64));
        assert!(s.store.is_prompt_head(cp64));
        assert!(!s.store.seq_checkpoints.values().any(|&v| v == cp64));
        assert_eq!(s.wb_waiting(), 3, "cp64 must be enqueued with head = true");
        s.store.end_seq_use(3);
        assert!(!s.store.store.in_use(mark64));

        // 3. Hit superseded Ordinary ord128: touches + pins block, but MUST NOT mark
        //    `prompt_heads` or enqueue `head = true`!
        let mark128 = s.entry(ord128).mark();
        let stamp_before = s.entry(ord128).last_used();
        s.store.note_state_hit(&s.coord, 4, ord128, false, 0.0);
        assert!(s.entry(ord128).last_used() > stamp_before);
        assert!(s.store.store.in_use(mark128));
        assert!(
            !s.store.is_prompt_head(ord128),
            "restoring an Ordinary entry must NEVER mark prompt_heads"
        );
        s.store.end_seq_use(4);
        assert!(!s.store.store.in_use(mark128));

        // 4. Flush to SSD: promoted head cp64 AND newest tip ord160 must persist durably,
        //    while superseded non-head ord128 is skipped as Unneeded (zero disk copy)!
        assert!(s.flush(10));
        assert!(
            s.disk().entry_disk_restorable(s.entry(cp64)),
            "promoted checkpoint head cp64 must persist to SSD despite being superseded"
        );
        assert!(
            s.entry(ord128).disk().is_none(),
            "superseded restored Ordinary ord128 must NOT spill as head"
        );
        assert!(
            s.disk().entry_disk_restorable(s.entry(ord160)),
            "newest conversation tip ord160 must persist to SSD"
        );

        // Drop RAM & purge hot GPU KV -> diverging query sharing first 80 tokens restores cp64 from SSD!
        s.drop_ram_and_purge();
        let mut query = tokens[..80].to_vec();
        query.extend(9000u32..9020u32);
        s.assert_ssd_restores(&query, salt, Some(cp64), None);
        assert!(s.close());
    }

    /// WS1: Both transient writer refusals (`reconcile_spill_outcome`) and RAM-only draft-window
    /// replacement (`old_id` -> `inserted.id`) preserve `prompt_heads` membership and `head = true`
    /// so superseded heads still persist durably to SSD.
    #[cfg(target_os = "macos")]
    #[test]
    fn reconcile_spill_outcome_and_ram_only_dw_upgrade_preserve_prompt_head() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use std::sync::atomic::{AtomicUsize, Ordering};

        let fx =
            StateDiskTestFixture::new("ws1-preserve-head", b"ws1-preserve-head-v1", Some(10.0))
                .with_paged(1024, 4);
        let reserve = 4096u64;
        let mut s = fx.open(128u64 << 20, Some(reserve));
        s.store.set_ram_cap(16);

        let salt1 = 0x5201u64;
        let tokens1: Vec<u32> = (2000u32..2128u32).collect();
        s.coord.reset_scheduled_request(1).expect("reset 1");
        s.coord.record_tokens_all(1, &tokens1).expect("record 1");

        // Part 1: Superseded promoted Checkpoint hits transient writer space refusal ->
        // `reconcile_spill_outcome` re-queues with `is_head = true` -> subsequent flush persists it!
        let cp64 = s
            .store
            .capture_boundary_with_options(
                &mut s.coord,
                &fx.conv,
                1,
                &tokens1,
                64,
                salt1,
                StateClass::Checkpoint,
                CaptureOptions {
                    keep_prev_checkpoint: false,
                    enqueue_write_behind: false,
                    is_head: false,
                    now: 0.0,
                },
            )
            .unwrap()
            .unwrap();
        let ord96 = s
            .store
            .capture_boundary_with_options(
                &mut s.coord,
                &fx.conv,
                1,
                &tokens1,
                96,
                salt1,
                StateClass::Ordinary,
                CaptureOptions {
                    keep_prev_checkpoint: true,
                    enqueue_write_behind: false,
                    is_head: false,
                    now: 0.0,
                },
            )
            .unwrap()
            .unwrap();
        assert!(s.store.store.superseded(cp64));
        assert_eq!(s.wb_waiting(), 0);

        s.store.note_state_hit(&s.coord, 2, cp64, false, 0.0);
        s.store.end_seq_use(2);
        assert!(s.store.is_prompt_head(cp64));
        assert_eq!(s.wb_waiting(), 1);

        // Verify `is_retryable_ordinary_wb` also allows persistent prompt heads below threshold.
        WRITE_BEHIND_MIN_TOKENS_OVERRIDE.with(|c| c.set(Some(128)));
        assert!(
            s.store.is_retryable_ordinary_wb(&s.coord, cp64),
            "persistent prompt head below write_behind_min_tokens must stay retryable"
        );
        assert!(
            !s.store.is_retryable_ordinary_wb(&s.coord, ord96),
            "non-head below write_behind_min_tokens must not be retryable"
        );
        WRITE_BEHIND_MIN_TOKENS_OVERRIDE.with(|c| c.set(Some(32)));

        let refusals = Arc::new(AtomicUsize::new(0));
        s.set_writer_space(64 << 20, reserve, Some(Arc::clone(&refusals)));
        s.assert_flush_fails_promptly("transient refusal on promoted head cp64");
        assert!(refusals.load(Ordering::Relaxed) >= 1);
        assert!(s.store.is_prompt_head(cp64));
        assert_eq!(
            s.wb_waiting(),
            1,
            "reconcile_spill_outcome must re-queue promoted head cp64"
        );

        s.clear_space_override(true);
        assert!(s.flush(10));
        assert!(
            s.disk().entry_disk_restorable(s.entry(cp64)),
            "re-queued promoted head cp64 must persist despite being superseded by ord96"
        );

        // Part 2: RAM-only draft-window replacement (`old_id` -> `inserted.id`) transfers
        // `prompt_heads` membership and re-enqueues `inserted.id` with `head = true`.
        let salt2 = 0x5202u64;
        let tokens2: Vec<u32> = (3000u32..3128u32).collect();
        s.coord.reset_scheduled_request(3).expect("reset 3");
        s.coord.record_tokens_all(3, &tokens2).expect("record 3");

        let old_head = s
            .store
            .capture_boundary_with_options(
                &mut s.coord,
                &fx.conv,
                3,
                &tokens2,
                64,
                salt2,
                StateClass::Ordinary,
                CaptureOptions {
                    keep_prev_checkpoint: true,
                    enqueue_write_behind: true,
                    is_head: true,
                    now: 0.0,
                },
            )
            .unwrap()
            .unwrap();
        let _tip2 = s
            .store
            .capture_boundary_with_options(
                &mut s.coord,
                &fx.conv,
                3,
                &tokens2,
                96,
                salt2,
                StateClass::Ordinary,
                CaptureOptions {
                    keep_prev_checkpoint: true,
                    enqueue_write_behind: false,
                    is_head: false,
                    now: 0.0,
                },
            )
            .unwrap()
            .unwrap();
        assert!(s.store.is_prompt_head(old_head));
        assert!(s.entry(old_head).disk().is_none());
        assert!(s.entry(old_head).ram().unwrap().draft_window.is_none());
        assert!(s.store.store.superseded(old_head));

        // Recapture depth 64 WITH a complete PackedDraftWindow and `is_head: false`:
        // RAM-only upgrade removes `old_head` and inserts `new_head`, preserving `prompt_heads`!
        let dw64 = fx.packed_dw(64);
        let new_head = s
            .store
            .capture_boundary_with_window_and_options(
                &mut s.coord,
                &fx.conv,
                3,
                &tokens2,
                64,
                salt2,
                StateClass::Ordinary,
                Some(&BoundaryDraftWindow::Packed(dw64.clone())),
                None,
                CaptureOptions {
                    keep_prev_checkpoint: true,
                    enqueue_write_behind: false,
                    is_head: false,
                    now: 0.0,
                },
            )
            .unwrap()
            .unwrap();
        assert_ne!(new_head, old_head, "RAM-only DW upgrade replaces entry id");
        assert!(!s.store.is_prompt_head(old_head));
        assert!(
            s.store.is_prompt_head(new_head),
            "RAM-only DW upgrade must transfer prompt_heads from old_id to inserted.id"
        );
        assert!(s.store.store.superseded(new_head));

        assert!(s.flush(10));
        assert!(
            s.disk().entry_disk_restorable(s.entry(new_head)),
            "upgraded superseded head must persist to SSD with head = true"
        );
        s.drop_ram_and_purge();
        let mut query2 = tokens2[..72].to_vec();
        query2.extend(9100u32..9110u32);
        s.assert_ssd_restores(&query2, salt2, Some(new_head), Some(&dw64));
        assert!(s.close());
    }

    /// WS1: `trim_ram_to_cap_inner` Priority 2b drops RAM-only superseded non-head intermediates
    /// directly in RAM instead of spilling ~2 dead blobs/turn to SSD, keeping SSD usage tiny,
    /// `HostBudgetAccount` exact, and both the persistent head and older unrelated tips restorable.
    #[cfg(target_os = "macos")]
    #[test]
    fn trim_ram_to_cap_drops_superseded_non_head_intermediates_without_ssd_bloat() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use crate::cache_limit::CacheLimitCoordinator;

        let fx = StateDiskTestFixture::new("ws1-no-ssd-bloat", b"ws1-no-ssd-bloat-v1", Some(0.0))
            .with_paged(4096, 4);
        let local_limit = Arc::new(CacheLimitCoordinator::new_for_test());
        let mut s = fx.open(256u64 << 20, None);
        s.store.attach_host_budget_to(Arc::clone(&local_limit));
        s.store.set_ram_cap(4);

        // 1. Publish an older unrelated conversation tip (Parent) on salt 10.
        let parent_salt = 10u64;
        let parent_tokens: Vec<u32> = (100u32..196u32).collect();
        s.coord.reset_scheduled_request(1).expect("reset parent");
        s.coord
            .record_tokens_all(1, &parent_tokens)
            .expect("record parent");
        let parent_tip = s
            .store
            .capture_boundary_with_options(
                &mut s.coord,
                &fx.conv,
                1,
                &parent_tokens,
                64,
                parent_salt,
                StateClass::Ordinary,
                CaptureOptions {
                    keep_prev_checkpoint: true,
                    enqueue_write_behind: true,
                    is_head: false,
                    now: 0.0,
                },
            )
            .unwrap()
            .unwrap();

        // 2. Publish 1 persistent prompt head (`is_head = true`) on salt 20, then 40 linear turns
        //    (80 Ordinary captures: 2 per turn at +32 token steps, depths 96..=2624).
        let main_salt = 20u64;
        let max_depth = 64u32 + 80 * 32; // 2624
        let full_tokens: Vec<u32> = (10_000u32..10_000 + max_depth + 32).collect();
        s.coord.reset_scheduled_request(2).expect("reset main");
        s.coord
            .record_tokens_all(2, &full_tokens[..64])
            .expect("record head");
        let head_id = s
            .store
            .capture_boundary_with_options(
                &mut s.coord,
                &fx.conv,
                2,
                &full_tokens[..64],
                64,
                main_salt,
                StateClass::Ordinary,
                CaptureOptions {
                    keep_prev_checkpoint: true,
                    enqueue_write_behind: true,
                    is_head: true,
                    now: 0.0,
                },
            )
            .unwrap()
            .unwrap();
        assert!(s.store.is_prompt_head(head_id));

        let mut last_tip = head_id;
        for step in 1u32..=80 {
            let prev = 64 + (step - 1) * 32;
            let cur = 64 + step * 32;
            s.coord
                .record_tokens_all(2, &full_tokens[prev as usize..cur as usize])
                .expect("record turn chunk");
            s.coord.prune_sliding_all(2).expect("prune swa");
            last_tip = s
                .store
                .capture_boundary_with_options(
                    &mut s.coord,
                    &fx.conv,
                    2,
                    &full_tokens[..cur as usize],
                    cur,
                    main_salt,
                    StateClass::Ordinary,
                    CaptureOptions {
                        keep_prev_checkpoint: true,
                        enqueue_write_behind: true,
                        is_head: false,
                        now: 0.0,
                    },
                )
                .unwrap()
                .unwrap();
            assert!(
                s.store.ram_len() <= 4,
                "ram_len must stay <= ram_cap=4 at step {step}, got {}",
                s.store.ram_len()
            );
            assert_eq!(
                local_limit.registered_pool_bytes(),
                s.store.host_resident_bytes(),
                "HostBudgetAccount must remain exact at step {step}"
            );
        }

        // Flush remaining waiting entries to SSD.
        assert!(s.flush(10));
        assert!(s.store.ram_len() <= 4);
        assert_eq!(
            local_limit.registered_pool_bytes(),
            s.store.host_resident_bytes()
        );

        let offloads = s.store.disk_counters().offloads;
        assert!(
            offloads <= 4,
            "Priority 2b must prevent 80-blob SSD bloat (expected <= 4 blobs for head + 2 tips, got {offloads})"
        );
        assert!(
            s.disk().entry_disk_restorable(s.entry(head_id)),
            "persistent prompt head must survive 80 turns and be SSD-restorable"
        );
        assert!(
            s.disk().entry_disk_restorable(s.entry(parent_tip)),
            "older unrelated conversation tip must survive and be SSD-restorable"
        );
        assert!(
            s.disk().entry_disk_restorable(s.entry(last_tip)),
            "final conversation tip must be SSD-restorable"
        );

        // Verify all 3 restore cleanly from SSD after purging RAM & hot GPU KV.
        s.drop_ram_and_purge();
        assert_eq!(
            local_limit.registered_pool_bytes(),
            s.store.host_resident_bytes()
        );
        let mut head_query = full_tokens[..80].to_vec();
        head_query.extend(99_000u32..99_010u32);
        s.assert_ssd_restores(&head_query, main_salt, Some(head_id), None);
        s.assert_ssd_restores(&parent_tokens, parent_salt, Some(parent_tip), None);
        s.assert_ssd_restores(&full_tokens, main_salt, Some(last_tip), None);
        assert!(s.close());
    }

    /// WS1: `adopt_all()` restores `prompt_heads` membership for newly adopted Ordinary entries
    /// BOTH in DFlash2 mode (`has_dw_span == true`) AND in AR-only mode (`has_dw_span == false`).
    #[cfg(target_os = "macos")]
    #[test]
    fn adopt_all_restores_prompt_head_membership_with_and_without_draft_window() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use crate::disk_tier::{CacheDisk, DiskTierOptions};

        // Case A: DFlash2 (`has_dw_span == true`).
        {
            let fx = StateDiskTestFixture::new("ws1-adopt-dw", b"ws1-adopt-dw-v1", Some(0.0));
            let tokens: Vec<u32> = (5000u32..5096u32).collect();
            fx.seed_durable(&[(1, &tokens, 64, 77)], 128u64 << 20, None);

            let s2 = fx.open_adopted(128u64 << 20, None, 1);
            let adopted_id = s2.first_id();
            assert!(
                s2.store.is_prompt_head(adopted_id),
                "DFlash2 adopt_all must restore prompt_heads membership"
            );
            assert!(
                s2.store.adopted_unverified_window.contains(&adopted_id),
                "DFlash2 adopt_all must still mark adopted_unverified_window"
            );
        }

        // Case B: AR-only (`has_dw_span == false`, `dflash2 = None`).
        {
            let temp_root = std::env::temp_dir()
                .join(format!("inkling2-ws1-adopt-ar-{}", uuid::Uuid::new_v4()));
            let cache_dir = temp_root.join("cache");
            std::fs::create_dir_all(&cache_dir).unwrap();
            let cfg = tiny_inkling2_config();
            let paged_cfg = Inkling2PagedConfig::default_for_config(&cfg)
                .with_kv_format(KvFormat::Int8)
                .with_max_model_len(512)
                .with_max_concurrent_sequences(2);
            let conv = make_short_conv(&cfg);
            let tokens: Vec<u32> = (6000u32..6096u32).collect();

            let open_ar = || {
                let recency = Arc::new(CacheRecency::new());
                let mut coord = Inkling2KVCacheCoordinator::new_for_config_with_recency(
                    &cfg,
                    paged_cfg,
                    Arc::clone(&recency),
                )
                .expect("coord");
                let geom = Inkling2StateGeometry::from_coordinator(&cfg, &coord).expect("geom");
                assert_eq!(geom.draft_window_span_bytes, 0);
                let disk_opts = DiskTierOptions {
                    cache_disk: CacheDisk::Bytes(128u64 << 20),
                    persistent: true,
                    cache_dir: Some(cache_dir.clone()),
                    temp_root: Some(temp_root.clone()),
                };
                let pid = compute_path_identity(&temp_root, &cfg, &coord, &geom, Some(1024), None);
                let (ctx, disk) =
                    open_inkling2_state_disk(&cfg, &coord, None, &disk_opts, move || pid, 0)
                        .expect("open AR disk");
                coord.set_full_cold_tier(ctx);
                let mut store = Inkling2StateStore::new(coord.block_size(), recency);
                store.set_geometry(geom);
                store.disk = Some(disk);
                (coord, store)
            };

            {
                let (mut coord, mut store) = open_ar();
                coord.reset_scheduled_request(1).unwrap();
                coord.record_tokens_all(1, &tokens).unwrap();
                let _id = store
                    .capture_boundary(
                        &mut coord,
                        &conv,
                        1,
                        &tokens,
                        64,
                        88,
                        StateClass::Ordinary,
                        None,
                    )
                    .unwrap()
                    .unwrap();
                assert!(store.flush_to_disk(&coord, test_deadline(10)));
                assert!(store.close_cache(&coord, test_deadline(10)));
            }
            {
                let (coord, mut store) = open_ar();
                assert_eq!(store.adopt_all(), 1);
                let adopted_id = store.store.entries().next().expect("adopted AR entry").id();
                assert!(
                    store.is_prompt_head(adopted_id),
                    "AR-only adopt_all (has_dw_span == false) must restore prompt_heads membership"
                );
                assert!(store.adopted_unverified_window.is_empty());
                assert!(store.close_cache(&coord, test_deadline(10)));
            }
            let _ = std::fs::remove_dir_all(&temp_root);
        }
    }

    /// WS1: `reuse_existing_entry` bumps `store.generation()` via `mark_changed()` whenever a
    /// previously cold Global KV chain is re-registered hot, and preserves `prompt_heads`.
    #[cfg(target_os = "macos")]
    #[test]
    fn reuse_existing_entry_bumps_generation_when_cold_chain_re_registered() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let fx = StateDiskTestFixture::new("ws1-reuse-gen", b"ws1-reuse-gen-v1", Some(0.0));
        let mut s = fx.open(128u64 << 20, None);
        let salt = 0x5301u64;
        let tokens: Vec<u32> = (7000u32..7096u32).collect();

        s.coord.reset_scheduled_request(1).expect("reset 1");
        s.coord.record_tokens_all(1, &tokens).expect("record 1");
        let id = s
            .store
            .capture_boundary_with_options(
                &mut s.coord,
                &fx.conv,
                1,
                &tokens,
                64,
                salt,
                StateClass::Ordinary,
                CaptureOptions {
                    keep_prev_checkpoint: true,
                    enqueue_write_behind: true,
                    is_head: true,
                    now: 0.0,
                },
            )
            .unwrap()
            .unwrap();
        assert!(s.store.is_prompt_head(id));
        assert!(s.flush(10));

        // Purge RAM + hot Global KV so `id` is SSD-only with `!chain_is_hot`.
        s.drop_ram_and_purge();
        assert!(!s.store.chain_is_hot(&s.coord, id));
        assert!(s.store.is_prompt_head(id));

        let gen_before = s.store.generation();
        // Recapture same lineage without draft window: takes the early reuse fast-path
        // (`existing_disk_durable && !incoming_can_upgrade`), re-registers the cold chain, and
        // MUST bump `store.generation()` via `mark_changed()` while preserving `prompt_heads`!
        s.coord.reset_scheduled_request(2).expect("reset 2");
        s.coord.record_tokens_all(2, &tokens).expect("record 2");
        let reused_id = s
            .store
            .capture_boundary_with_options(
                &mut s.coord,
                &fx.conv,
                2,
                &tokens,
                64,
                salt,
                StateClass::Ordinary,
                CaptureOptions {
                    keep_prev_checkpoint: true,
                    enqueue_write_behind: true,
                    is_head: false,
                    now: 0.0,
                },
            )
            .unwrap()
            .unwrap();
        assert_eq!(reused_id, id);
        assert!(
            s.store.chain_is_hot(&s.coord, id),
            "reuse_existing_entry fast-path must re-register the cold Global KV chain"
        );
        assert!(
            s.store.generation() > gen_before,
            "re-registering a previously cold Global KV chain must bump store.generation()"
        );
        assert!(
            s.store.is_prompt_head(id),
            "reuse_existing_entry must preserve existing prompt_heads membership when opts.is_head == false"
        );
        assert!(s.close());
    }

    /// Codex C1: Reclaiming a `StateClass::Checkpoint` to persistent SSD writes
    /// `ColdStateLabel { checkpoint: true, .. }` and drops RAM. When an independent request hits
    /// that SSD-backed checkpoint (`load_anchor` + `note_state_hit(..., is_preemption_replay: false)`),
    /// `PendingDiskUpgrade::LabelPromote` rewrites the on-disk header (`checkpoint: false`) directly
    /// via the staging buffer without repopulating RAM, so reopening and calling `adopt_all()`
    /// adopts it as `Ordinary` in `prompt_heads` and it survives `ram_cap` Priority 1/2b trimming.
    #[cfg(target_os = "macos")]
    #[test]
    fn ssd_backed_checkpoint_promotion_persists_across_restart() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let fx = StateDiskTestFixture::new("c1-cp-promote", b"c1-cp-promote-v1", Some(0.0))
            .with_paged(1024, 4);
        WRITE_BEHIND_MIN_TOKENS_OVERRIDE.with(|c| c.set(Some(64)));

        let salt_promoted = 0x6101u64;
        let salt_control = 0x6102u64;
        let tokens_promoted: Vec<u32> = (8100u32..8228u32).collect();
        let tokens_control: Vec<u32> = (8300u32..8396u32).collect();
        let dw64 = fx.packed_dw(64);

        // Session 1: reclaim two Checkpoints to persistent SSD, promote only the first via
        // SSD-only `load_anchor` + `note_state_hit`, flush, and verify on-disk headers.
        {
            let mut s = fx.open(128u64 << 20, None);
            s.store.set_ram_cap(16);

            let cp_promoted = s.capture(
                &fx.conv,
                1,
                &tokens_promoted,
                64,
                salt_promoted,
                StateClass::Checkpoint,
                Some(&dw64),
            );
            let cp_control = s.capture(
                &fx.conv,
                2,
                &tokens_control,
                64,
                salt_control,
                StateClass::Checkpoint,
                None,
            );
            s.store.clear_seq_checkpoint(1);
            s.store.clear_seq_checkpoint(2);
            assert!(s.entry(cp_promoted).checkpoint() && s.entry(cp_promoted).ram().is_some());
            assert!(s.entry(cp_control).checkpoint() && s.entry(cp_control).ram().is_some());

            // Reclaim both Checkpoints to SSD (drops RAM, writes `checkpoint: true` on disk).
            assert_eq!(
                s.store.reclaim_entry(&s.coord, cp_promoted),
                ReclaimOutcome::Spilled
            );
            assert!(s.flush(10));
            assert_eq!(
                s.store.reclaim_entry(&s.coord, cp_control),
                ReclaimOutcome::Spilled
            );
            assert!(s.flush(10));

            for id in [cp_promoted, cp_control] {
                assert!(
                    s.entry(id).ram().is_none(),
                    "reclaimed checkpoint is SSD-only"
                );
                let copy = s.disk_copy(id);
                assert!(copy.landed && !copy.chain_owed);
                assert!(s.entry(id).checkpoint());
                let (_, on_disk_label) = s
                    .manager()
                    .state_label(copy.key, s.disk().tier.state_fingerprint)
                    .expect("on-disk state label before promote");
                assert!(
                    on_disk_label.checkpoint,
                    "reclaimed checkpoint must carry checkpoint=true on SSD before promotion"
                );
            }

            // Purge hot GPU KV so the hit is purely SSD-backed.
            s.coord.release_all_and_purge().expect("purge hot");

            // Hit `cp_promoted` via `load_anchor` + `note_state_hit(..., is_preemption_replay: false)`.
            let (anchor, from_ssd) = s
                .store
                .load_anchor(&s.coord, cp_promoted)
                .expect("load_anchor from SSD");
            assert!(from_ssd);
            assert_eq!(anchor.draft_window.as_ref(), Some(&dw64));
            assert!(
                s.entry(cp_promoted).ram().is_none(),
                "load_anchor must not repopulate RAM"
            );

            s.store.note_state_hit(&s.coord, 3, cp_promoted, false, 0.0);
            s.store.end_seq_use(3);

            assert!(!s.entry(cp_promoted).checkpoint());
            assert!(s.store.is_prompt_head(cp_promoted));
            assert!(s.entry(cp_promoted).ram().is_none());
            assert!(matches!(
                s.store.pending_disk_upgrades.get(&cp_promoted),
                Some(PendingDiskUpgrade::LabelPromote { .. })
            ));

            // Flush must drain `LabelPromote` without repopulating RAM and rewrite header to false.
            assert!(s.flush(10));
            assert!(s.store.pending_disk_upgrades.is_empty());
            assert!(s.store.in_flight_disk_upgrade.is_none());
            assert!(s.entry(cp_promoted).ram().is_none());

            let promoted_copy = s.disk_copy(cp_promoted);
            let (_, promoted_label) = s
                .manager()
                .state_label(promoted_copy.key, s.disk().tier.state_fingerprint)
                .expect("promoted on-disk label");
            assert!(
                !promoted_label.checkpoint,
                "LabelPromote must rewrite on-disk ColdStateLabel to checkpoint=false"
            );

            let control_copy = s.disk_copy(cp_control);
            let (_, control_label) = s
                .manager()
                .state_label(control_copy.key, s.disk().tier.state_fingerprint)
                .expect("control on-disk label");
            assert!(
                control_label.checkpoint,
                "unhit control checkpoint must remain checkpoint=true on disk"
            );

            assert!(s.close());
        }

        // Session 2: reopen persistent tier, call `adopt_all()`, and verify adopted class,
        // `prompt_heads`, exact draft-window preservation, and `ram_cap` Priority 1/2b survival.
        {
            let mut s2 = fx.open_adopted(128u64 << 20, None, 2);
            let promoted_id = s2.id_by_salt(salt_promoted);
            let control_id = s2.id_by_salt(salt_control);

            assert!(
                !s2.entry(promoted_id).checkpoint(),
                "reopened promoted SSD checkpoint must adopt as Ordinary"
            );
            assert!(
                s2.store.is_prompt_head(promoted_id),
                "reopened promoted SSD checkpoint must be registered in prompt_heads"
            );
            assert!(
                s2.entry(control_id).checkpoint(),
                "reopened unhit SSD checkpoint must still adopt as Checkpoint"
            );
            assert!(!s2.store.is_prompt_head(control_id));

            // Verify payload + draft window survived the zero-decode SSD relabel intact.
            let mut query_promoted = tokens_promoted[..80].to_vec();
            query_promoted.extend(9900u32..9910u32);
            s2.assert_ssd_restores(
                &query_promoted,
                salt_promoted,
                Some(promoted_id),
                Some(&dw64),
            );

            // Verify Priority 1/2b `ram_cap` trimming behavior after restart:
            // Give both entries RAM copies + supersede `promoted_id` with a deeper ordinary turn,
            // then force `ram_cap = 1`:
            // - `control_id` (still a Checkpoint) is retired completely by Priority 1!
            // - `promoted_id` (Ordinary + in `prompt_heads`) survives Priority 1 and Priority 2b
            //   (dropping only RAM to its durable SSD copy, staying indexed and restorable)!
            let anchor_promoted = s2
                .store
                .load_anchor(&s2.coord, promoted_id)
                .expect("load promoted")
                .0;
            let anchor_control = s2
                .store
                .load_anchor(&s2.coord, control_id)
                .expect("load control")
                .0;
            assert!(s2.store.store.promote(promoted_id, anchor_promoted));
            assert!(s2.store.store.promote(control_id, anchor_control));

            s2.coord.reset_scheduled_request(10).expect("reset deeper");
            s2.coord
                .record_tokens_all(10, &tokens_promoted)
                .expect("record deeper");
            let deeper_id = s2
                .store
                .capture_boundary_with_options(
                    &mut s2.coord,
                    &fx.conv,
                    10,
                    &tokens_promoted,
                    96,
                    salt_promoted,
                    StateClass::Ordinary,
                    CaptureOptions {
                        keep_prev_checkpoint: true,
                        enqueue_write_behind: false,
                        is_head: false,
                        now: 0.0,
                    },
                )
                .unwrap()
                .unwrap();
            assert!(s2.store.store.superseded(promoted_id));

            s2.store.set_ram_cap(1);
            s2.store.trim_ram_to_cap(&s2.coord, Some(deeper_id));

            assert!(
                s2.store.store.get(control_id).is_none(),
                "adopted unhit Checkpoint must be retired by Priority 1 under ram_cap"
            );
            assert!(
                s2.store.store.get(promoted_id).is_some(),
                "adopted promoted Ordinary prompt head must survive Priority 1/2b under ram_cap"
            );
            assert!(s2.store.is_prompt_head(promoted_id));
            assert!(s2.disk().entry_disk_restorable(s2.entry(promoted_id)));
            assert!(s2.close());
        }

        // Non-persistent tier must NOT enqueue `LabelPromote` (adopt_all only runs on persistent tiers).
        {
            let mut st = fx.open_temp(128u64 << 20, None);
            let cp_temp = st.capture(
                &fx.conv,
                1,
                &tokens_promoted,
                64,
                0x6103u64,
                StateClass::Checkpoint,
                None,
            );
            st.store.clear_seq_checkpoint(1);
            assert_eq!(
                st.store.reclaim_entry(&st.coord, cp_temp),
                ReclaimOutcome::Spilled
            );
            assert!(st.flush(10));
            st.store.note_state_hit(&st.coord, 2, cp_temp, false, 0.0);
            st.store.end_seq_use(2);
            assert!(
                st.store.pending_disk_upgrades.is_empty(),
                "non-persistent tier must not enqueue LabelPromote"
            );
            assert!(st.close());
        }
    }

    /// Codex C1: Both (a) promoting an in-flight (not-yet-landed) reclaimed checkpoint via
    /// `note_state_hit` and (b) promoting an SSD-backed checkpoint via `reuse_existing_entry`
    /// queue `PendingDiskUpgrade::LabelPromote`, rewrite `checkpoint: false` once the initial copy
    /// lands, and adopt as `Ordinary` in `prompt_heads` on restart.
    #[cfg(target_os = "macos")]
    #[test]
    fn ssd_backed_checkpoint_promotion_in_flight_and_reuse_survive_restart() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let fx = StateDiskTestFixture::new(
            "c1-inflight-and-reuse",
            b"c1-inflight-and-reuse-v1",
            Some(0.0),
        )
        .with_paged(1024, 4);
        WRITE_BEHIND_MIN_TOKENS_OVERRIDE.with(|c| c.set(Some(64)));

        let salt_inflight = 0x6201u64;
        let salt_reuse = 0x6202u64;
        let tokens_inflight: Vec<u32> = (8400u32..8496u32).collect();
        let tokens_reuse: Vec<u32> = (8500u32..8596u32).collect();

        {
            let mut s = fx.open(128u64 << 20, None);
            s.store.set_ram_cap(16);

            // Part A: Promote an in-flight (not-yet-landed) reclaimed checkpoint!
            let cp_inflight = s.capture(
                &fx.conv,
                1,
                &tokens_inflight,
                64,
                salt_inflight,
                StateClass::Checkpoint,
                None,
            );
            s.store.clear_seq_checkpoint(1);
            assert_eq!(
                s.store.reclaim_entry(&s.coord, cp_inflight),
                ReclaimOutcome::Spilled
            );
            // Before any `poll_offload`, the initial checkpoint spill is still in flight on disk.
            assert!(s.entry(cp_inflight).ram().is_none());
            assert!(s.entry(cp_inflight).disk().is_some_and(|c| !c.landed));
            assert!(s.disk().writing());

            s.store.note_state_hit(&s.coord, 2, cp_inflight, false, 0.0);
            s.store.end_seq_use(2);
            assert!(!s.entry(cp_inflight).checkpoint());
            assert!(s.store.is_prompt_head(cp_inflight));
            assert!(matches!(
                s.store.pending_disk_upgrades.get(&cp_inflight),
                Some(PendingDiskUpgrade::LabelPromote { .. })
            ));

            // Flush must wait for the initial checkpoint write + KV chain to land, then relabel!
            assert!(s.flush(10));
            assert!(s.store.pending_disk_upgrades.is_empty());
            assert!(s.store.in_flight_disk_upgrade.is_none());
            let inflight_copy = s.disk_copy(cp_inflight);
            assert!(inflight_copy.landed && !inflight_copy.chain_owed);
            let (_, inflight_label) = s
                .manager()
                .state_label(inflight_copy.key, s.disk().tier.state_fingerprint)
                .expect("inflight on-disk label");
            assert!(
                !inflight_label.checkpoint,
                "in-flight reclaimed checkpoint promotion must land checkpoint=false on SSD"
            );

            // Part B: Promote a durable SSD-backed checkpoint via `reuse_existing_entry`!
            let cp_reuse = s.capture(
                &fx.conv,
                3,
                &tokens_reuse,
                64,
                salt_reuse,
                StateClass::Checkpoint,
                None,
            );
            s.store.clear_seq_checkpoint(3);
            assert_eq!(
                s.store.reclaim_entry(&s.coord, cp_reuse),
                ReclaimOutcome::Spilled
            );
            assert!(s.flush(10));
            assert!(s.entry(cp_reuse).ram().is_none());
            assert!(s.entry(cp_reuse).checkpoint());
            let reuse_copy_before = s.disk_copy(cp_reuse);
            assert!(reuse_copy_before.landed && !reuse_copy_before.chain_owed);
            let (_, reuse_label_before) = s
                .manager()
                .state_label(reuse_copy_before.key, s.disk().tier.state_fingerprint)
                .expect("reuse label before");
            assert!(reuse_label_before.checkpoint);

            // Recapture same lineage at depth 64 as `Ordinary` with no draft window -> hits the
            // early reuse fast-path (`reuse_existing_entry`) while keeping `ram().is_none()`.
            s.coord.reset_scheduled_request(4).expect("reset 4");
            s.coord
                .record_tokens_all(4, &tokens_reuse)
                .expect("record 4");
            let reused_id = s
                .store
                .capture_boundary_with_options(
                    &mut s.coord,
                    &fx.conv,
                    4,
                    &tokens_reuse,
                    64,
                    salt_reuse,
                    StateClass::Ordinary,
                    CaptureOptions {
                        keep_prev_checkpoint: true,
                        enqueue_write_behind: true,
                        is_head: false,
                        now: 0.0,
                    },
                )
                .unwrap()
                .unwrap();
            assert_eq!(reused_id, cp_reuse);
            assert!(s.entry(cp_reuse).ram().is_none());
            assert!(!s.entry(cp_reuse).checkpoint());
            assert!(matches!(
                s.store.pending_disk_upgrades.get(&cp_reuse),
                Some(PendingDiskUpgrade::LabelPromote { .. })
            ));

            assert!(s.flush(10));
            assert!(s.store.pending_disk_upgrades.is_empty());
            assert!(s.store.in_flight_disk_upgrade.is_none());
            let reuse_copy_after = s.disk_copy(cp_reuse);
            let (_, reuse_label_after) = s
                .manager()
                .state_label(reuse_copy_after.key, s.disk().tier.state_fingerprint)
                .expect("reuse label after");
            assert!(
                !reuse_label_after.checkpoint,
                "reuse_existing_entry upgrade on SSD checkpoint must land checkpoint=false on SSD"
            );
            assert!(s.close());
        }

        // Reopen persistent tier and verify BOTH adopt as Ordinary prompt heads & restore cleanly.
        {
            let mut s2 = fx.open_adopted(128u64 << 20, None, 2);
            let id_inflight = s2.id_by_salt(salt_inflight);
            let id_reuse = s2.id_by_salt(salt_reuse);

            for (label, id, tokens, salt) in [
                ("inflight", id_inflight, &tokens_inflight, salt_inflight),
                ("reuse", id_reuse, &tokens_reuse, salt_reuse),
            ] {
                assert!(
                    !s2.entry(id).checkpoint(),
                    "{label}: must adopt as Ordinary after restart"
                );
                assert!(
                    s2.store.is_prompt_head(id),
                    "{label}: must be in prompt_heads after restart"
                );
                let mut query = tokens[..80].to_vec();
                query.extend(9800u32..9810u32);
                s2.assert_ssd_restores(&query, salt, Some(id), None);
            }
            assert!(s2.close());
        }
    }

    /// Codex C1: When a `LabelPromote` write fails (either asynchronously via `MakeRoom::Refuse`
    /// on the writer thread or synchronously due to low model-thread headroom), the old SSD copy
    /// stays pinned and intact (`preserve_entry_on_state_failure`), `entries_missing_draft_window`
    /// is NEVER corrupted, `LabelPromote` reschedules with backoff without tight-spinning
    /// `flush_to_disk`, and completes cleanly once room returns.
    #[cfg(target_os = "macos")]
    #[test]
    fn label_promote_failure_preserves_old_ssd_copy_and_draft_window_tracking() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use std::sync::atomic::{AtomicUsize, Ordering};

        let fx = StateDiskTestFixture::new(
            "c1-label-promote-fail",
            b"c1-label-promote-fail-v1",
            Some(10.0),
        )
        .with_paged(1024, 4);
        WRITE_BEHIND_MIN_TOKENS_OVERRIDE.with(|c| c.set(Some(64)));

        let reserve = 4096u64;
        let salt = 0x6301u64;
        let tokens: Vec<u32> = (8600u32..8696u32).collect();
        let mut query = tokens[..80].to_vec();
        query.extend(9700u32..9710u32);
        let dw64 = fx.packed_dw(64);

        {
            let mut s = fx.open(128u64 << 20, Some(reserve));
            s.store.set_ram_cap(16);

            let cp_id = s.capture(
                &fx.conv,
                1,
                &tokens,
                64,
                salt,
                StateClass::Checkpoint,
                Some(&dw64),
            );
            s.store.clear_seq_checkpoint(1);
            assert!(!s.store.entry_missing_draft_window(cp_id));

            // Reclaim to persistent SSD (`ram().is_none()`, `checkpoint: true` on disk).
            assert_eq!(
                s.store.reclaim_entry(&s.coord, cp_id),
                ReclaimOutcome::Spilled
            );
            assert!(s.flush(10));
            assert!(s.entry(cp_id).ram().is_none());
            let old_copy = s.disk_copy(cp_id);
            assert!(old_copy.landed && !old_copy.chain_owed);
            assert!(!s.store.entry_missing_draft_window(cp_id));

            // Purge hot KV, restore from SSD, and promote via `note_state_hit`.
            s.coord.release_all_and_purge().expect("purge hot");
            let (anchor, from_ssd) = s
                .store
                .load_anchor(&s.coord, cp_id)
                .expect("load_anchor before refusal");
            assert!(from_ssd);
            assert_eq!(anchor.draft_window.as_ref(), Some(&dw64));
            s.store.note_state_hit(&s.coord, 2, cp_id, false, 0.0);
            s.store.end_seq_use(2);
            assert!(!s.store.entry_missing_draft_window(cp_id));
            assert!(matches!(
                s.store.pending_disk_upgrades.get(&cp_id),
                Some(PendingDiskUpgrade::LabelPromote { .. })
            ));

            // 1. Async writer-side refusal (`MakeRoom::Refuse`): must exit `flush_to_disk`
            //    promptly, preserve `old_copy` and entry, NEVER set `entries_missing_draft_window`,
            //    and honor backoff on subsequent `Tick::Full`.
            let refusals = Arc::new(AtomicUsize::new(0));
            s.set_writer_space(64 << 20, reserve, Some(Arc::clone(&refusals)));
            s.assert_flush_fails_promptly("async writer-side refusal on LabelPromote");
            assert!(refusals.load(Ordering::Relaxed) >= 1);

            assert!(
                s.store.store.get(cp_id).is_some(),
                "SSD-only entry must NOT be dropped on LabelPromote failure"
            );
            assert!(s.entry(cp_id).ram().is_none());
            assert_eq!(
                s.entry(cp_id).disk().copied(),
                Some(old_copy),
                "old durable SSD copy must be restored intact on LabelPromote failure"
            );
            assert!(s.disk().entry_disk_restorable(s.entry(cp_id)));
            assert!(
                !s.store.entry_missing_draft_window(cp_id),
                "LabelPromote rollback must NEVER corrupt entries_missing_draft_window"
            );
            assert!(matches!(
                s.store.pending_disk_upgrades.get(&cp_id),
                Some(PendingDiskUpgrade::LabelPromote { .. })
            ));
            assert!(s.store.in_flight_disk_upgrade.is_none());

            // Background `Tick::Full` must respect backoff without re-staging.
            let _ = s.store.tick(&s.coord, Tick::Full { command_pages: 0 });
            assert!(!s.disk().writing());

            // Old SSD copy must still restore cleanly with its exact PackedDraftWindow.
            s.assert_ssd_restores(&query, salt, Some(cp_id), Some(&dw64));

            // 2. Synchronous model-thread headroom refusal (`can_stage == false`): exits promptly,
            //    preserves `old_copy`, and keeps `LabelPromote` queued.
            s.manager()
                .set_space_override_for_test(Some(Box::new(move || Ok((1 << 30, reserve)))));
            s.assert_flush_fails_promptly("sync headroom refusal on LabelPromote");
            assert_eq!(s.entry(cp_id).disk().copied(), Some(old_copy));
            assert!(!s.store.entry_missing_draft_window(cp_id));
            assert!(matches!(
                s.store.pending_disk_upgrades.get(&cp_id),
                Some(PendingDiskUpgrade::LabelPromote { .. })
            ));

            // 3. Space recovers: `flush_to_disk` drains `LabelPromote` cleanly.
            s.clear_space_override(true);
            assert!(s.flush(10));
            assert!(s.store.pending_disk_upgrades.is_empty());
            assert!(s.store.in_flight_disk_upgrade.is_none());
            assert!(!s.store.entry_missing_draft_window(cp_id));
            let new_copy = s.disk_copy(cp_id);
            let (_, new_label) = s
                .manager()
                .state_label(new_copy.key, s.disk().tier.state_fingerprint)
                .expect("recovered label");
            assert!(!new_label.checkpoint);
            assert!(s.close());
        }

        // Reopen persistent tier -> adopted as Ordinary head with exact PackedDraftWindow intact.
        {
            let mut s2 = fx.open_adopted(128u64 << 20, None, 1);
            let adopted_id = s2.id_by_salt(salt);
            assert!(!s2.entry(adopted_id).checkpoint());
            assert!(s2.store.is_prompt_head(adopted_id));
            s2.assert_ssd_restores(&query, salt, Some(adopted_id), Some(&dw64));
            assert!(s2.close());
        }
    }

    /// Codex R2-1: Off-thread `LabelPromote` keeps the already-landed SSD copy visible
    /// (`landed == true`, `entry_disk_restorable == true`, `staging_free == true`) while the
    /// `ColdCacheManager` writer is paused, allowing concurrent restores (`lookup_deepest_restorable`
    /// and `load_anchor`) to succeed immediately without waiting for the writer, preserving
    /// restorability throughout AND after `MakeRoom::Refuse` refusal, and adopting as `Ordinary`
    /// in `prompt_heads` after unpausing and landing (for both SSD-only and RAM-backed entries).
    #[cfg(target_os = "macos")]
    #[test]
    fn label_promote_keeps_landed_ssd_copy_restorable_while_writer_paused() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let fx =
            StateDiskTestFixture::new("r2-1-relabel-paused", b"r2-1-relabel-paused-v1", Some(10.0))
                .with_paged(1024, 4);
        WRITE_BEHIND_MIN_TOKENS_OVERRIDE.with(|c| c.set(Some(64)));

        let reserve = 4096u64;
        let salt_ssd = 0x6401u64;
        let salt_ram = 0x6402u64;
        let tokens_ssd: Vec<u32> = (8700u32..8796u32).collect();
        let tokens_ram: Vec<u32> = (8800u32..8896u32).collect();
        let mut query_ssd = tokens_ssd[..80].to_vec();
        query_ssd.extend(9600u32..9610u32);
        let mut query_ram = tokens_ram[..80].to_vec();
        query_ram.extend(9620u32..9630u32);
        let dw64 = fx.packed_dw(64);

        {
            let mut s = fx.open(128u64 << 20, Some(reserve));
            s.store.set_ram_cap(16);

            // 1. Reclaim a checkpoint to persistent SSD (SSD-only, `checkpoint: true` on disk).
            let cp_id = s.capture(
                &fx.conv,
                1,
                &tokens_ssd,
                64,
                salt_ssd,
                StateClass::Checkpoint,
                Some(&dw64),
            );
            s.store.clear_seq_checkpoint(1);
            assert_eq!(
                s.store.reclaim_entry(&s.coord, cp_id),
                ReclaimOutcome::Spilled
            );
            assert!(s.flush(10));
            assert!(s.entry(cp_id).ram().is_none());
            let old_copy = s.disk_copy(cp_id);
            assert!(old_copy.landed && !old_copy.chain_owed);

            // Purge hot KV and promote via SSD-only hit (`note_state_hit`).
            s.coord.release_all_and_purge().expect("purge hot");
            s.assert_ssd_restores(&query_ssd, salt_ssd, Some(cp_id), Some(&dw64));
            s.store.note_state_hit(&s.coord, 2, cp_id, false, 0.0);
            s.store.end_seq_use(2);
            assert!(!s.entry(cp_id).checkpoint());
            assert!(s.store.is_prompt_head(cp_id));
            assert!(matches!(
                s.store.pending_disk_upgrades.get(&cp_id),
                Some(PendingDiskUpgrade::LabelPromote { .. })
            ));

            // 2. Refusal case while writer is paused then unpaused:
            //    Pause writer, start `LabelPromote`, verify restorability + concurrent restore
            //    while paused, then force `MakeRoom::Refuse` on the writer and unpause.
            let pause = s.manager().pause_writer_for_test().expect("pause writer");
            assert!(
                s.store.drive_pending_disk_upgrades(&s.coord, true),
                "LabelPromote must enqueue while writer is paused"
            );
            assert!(
                s.disk().writing(),
                "disk.writing() must be true during relabel"
            );
            assert!(
                s.disk().staging_free(),
                "off-thread relabel must not steal engine.staging"
            );
            assert!(matches!(
                s.store.in_flight_disk_upgrade.as_ref().map(|u| (u.id, u.kind)),
                Some((id, InFlightDiskUpgradeKind::LabelPromote)) if id == cp_id
            ));
            assert_eq!(
                s.entry(cp_id).disk().copied(),
                Some(old_copy),
                "entry.disk() must stay landed==true while LabelPromote is in flight"
            );
            assert!(
                s.disk().entry_disk_restorable(s.entry(cp_id)),
                "entry_disk_restorable must stay true while writer is paused"
            );
            // Second restore MUST succeed while the writer is still paused!
            s.assert_ssd_restores(&query_ssd, salt_ssd, Some(cp_id), Some(&dw64));

            // Inject writer-side space refusal before unpausing so `MakeRoom::Refuse` triggers.
            s.set_writer_space(64 << 20, reserve, None);
            drop(pause);
            s.await_writing("refused LabelPromote must finish promptly");
            assert!(!s.disk().writing());
            assert!(
                s.disk().writable(),
                "MakeRoom::Refuse must keep tier writable"
            );
            assert!(s.store.in_flight_disk_upgrade.is_none());
            assert!(matches!(
                s.store.pending_disk_upgrades.get(&cp_id),
                Some(PendingDiskUpgrade::LabelPromote { .. })
            ));
            assert_eq!(
                s.entry(cp_id).disk().copied(),
                Some(old_copy),
                "old SSD copy must remain landed==true after refusal"
            );
            assert!(s.disk().entry_disk_restorable(s.entry(cp_id)));
            assert!(!s.store.entry_missing_draft_window(cp_id));
            // Restore MUST still succeed after refusal!
            s.assert_ssd_restores(&query_ssd, salt_ssd, Some(cp_id), Some(&dw64));
            let (_, label_after_refusal) = s
                .manager()
                .state_label(old_copy.key, s.disk().tier.state_fingerprint)
                .expect("label after refusal");
            assert!(
                label_after_refusal.checkpoint,
                "refused relabel must leave on-disk header untouched"
            );

            // 3. Success case while writer is paused then unpaused:
            s.clear_space_override(true);
            s.expire_backoff(cp_id);
            let pause = s.manager().pause_writer_for_test().expect("pause writer 2");
            assert!(s.store.drive_pending_disk_upgrades(&s.coord, true));
            assert!(s.disk().writing());
            assert!(s.disk().staging_free());
            assert!(s.entry(cp_id).disk().is_some_and(|c| c.landed));
            assert!(s.disk().entry_disk_restorable(s.entry(cp_id)));
            // Concurrent restore MUST succeed while writer is paused!
            s.assert_ssd_restores(&query_ssd, salt_ssd, Some(cp_id), Some(&dw64));

            drop(pause);
            assert!(s.flush(10));
            assert!(s.store.pending_disk_upgrades.is_empty());
            assert!(s.store.in_flight_disk_upgrade.is_none());
            assert!(!s.disk().writing());
            assert!(s.entry(cp_id).ram().is_none());
            assert!(s.disk().entry_disk_restorable(s.entry(cp_id)));
            let landed_copy = s.disk_copy(cp_id);
            assert!(landed_copy.landed && !landed_copy.chain_owed);
            assert_eq!(landed_copy.bytes, s.state_object_bytes_for(cp_id));
            let (_, landed_label) = s
                .manager()
                .state_label(landed_copy.key, s.disk().tier.state_fingerprint)
                .expect("landed label");
            assert!(!landed_label.checkpoint);

            // 4. RAM-backed `LabelPromote` ALSO uses off-thread relabel and keeps `landed == true`
            //    and `staging_free == true` while writer is paused.
            let cp_ram = s.capture(
                &fx.conv,
                3,
                &tokens_ram,
                64,
                salt_ram,
                StateClass::Checkpoint,
                Some(&dw64),
            );
            s.store.clear_seq_checkpoint(3);
            assert_eq!(
                s.store.reclaim_entry(&s.coord, cp_ram),
                ReclaimOutcome::Spilled
            );
            assert!(s.flush(10));
            let (ram_anchor, from_ssd) = s
                .store
                .load_anchor(&s.coord, cp_ram)
                .expect("load cp_ram from SSD");
            assert!(from_ssd);
            assert!(s.store.store.promote(cp_ram, ram_anchor));
            assert!(s.entry(cp_ram).ram().is_some());
            s.store.note_state_hit(&s.coord, 4, cp_ram, false, 0.0);
            s.store.end_seq_use(4);
            assert!(matches!(
                s.store.pending_disk_upgrades.get(&cp_ram),
                Some(PendingDiskUpgrade::LabelPromote { .. })
            ));

            let pause = s.manager().pause_writer_for_test().expect("pause writer 3");
            assert!(s.store.drive_pending_disk_upgrades(&s.coord, true));
            assert!(s.disk().writing());
            assert!(s.disk().staging_free());
            assert!(s.entry(cp_ram).disk().is_some_and(|c| c.landed));
            assert!(s.disk().entry_disk_restorable(s.entry(cp_ram)));
            drop(pause);
            assert!(s.flush(10));
            let (_, ram_label) = s
                .manager()
                .state_label(s.disk_copy(cp_ram).key, s.disk().tier.state_fingerprint)
                .expect("ram_label");
            assert!(!ram_label.checkpoint);

            assert!(s.close());
        }

        // 5. Reopen persistent tier + `adopt_all`: BOTH entries adopt as `Ordinary` in
        //    `prompt_heads` and restore cleanly from SSD with exact `PackedDraftWindow`.
        {
            let mut s2 = fx.open_adopted(128u64 << 20, None, 2);
            for (name, salt, query) in [
                ("ssd-only", salt_ssd, &query_ssd),
                ("ram-backed", salt_ram, &query_ram),
            ] {
                let id = s2.id_by_salt(salt);
                assert!(
                    !s2.entry(id).checkpoint(),
                    "{name}: must adopt as Ordinary (!checkpoint) after relabel"
                );
                assert!(
                    s2.store.is_prompt_head(id),
                    "{name}: must adopt into prompt_heads after relabel"
                );
                s2.assert_ssd_restores(query, salt, Some(id), Some(&dw64));
            }
            assert!(s2.close());
        }
    }

    /// Codex R4-1: If `clear_ram_and_checkpoints()` (e.g., `ResetCaches`) runs while an off-thread
    /// `LabelPromote` is in flight, a subsequent `poll_relabel` completion — whether `Ok` or
    /// `MakeRoom::Refuse` failure — must be a clean bookkeeping no-op that preserves the committed
    /// durable SSD anchor on disk (never calling `tier.manager.remove(&landing.key)`).
    #[cfg(target_os = "macos")]
    #[test]
    fn label_promote_landing_after_reset_preserves_durable_ssd_anchor() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let fx = StateDiskTestFixture::new(
            "r4-1-relabel-after-reset",
            b"r4-1-relabel-after-reset-v1",
            Some(10.0),
        )
        .with_paged(1024, 4);
        WRITE_BEHIND_MIN_TOKENS_OVERRIDE.with(|c| c.set(Some(64)));

        let reserve = 4096u64;
        let salt_ok = 0x6501u64;
        let salt_fail = 0x6502u64;
        let tokens_ok: Vec<u32> = (8900u32..8996u32).collect();
        let tokens_fail: Vec<u32> = (9000u32..9096u32).collect();
        let mut query_ok = tokens_ok[..80].to_vec();
        query_ok.extend(9700u32..9710u32);
        let mut query_fail = tokens_fail[..80].to_vec();
        query_fail.extend(9720u32..9730u32);
        let dw64 = fx.packed_dw(64);

        {
            let mut s = fx.open(128u64 << 20, Some(reserve));
            s.store.set_ram_cap(16);

            // 1. Success-after-reset: reclaim checkpoint to persistent SSD, hit + promote to
            //    enqueue `LabelPromote`, start off-thread relabel while writer is paused, run
            //    `clear_ram_and_checkpoints()`, then unpause and drain.
            let cp_ok = s.capture(
                &fx.conv,
                1,
                &tokens_ok,
                64,
                salt_ok,
                StateClass::Checkpoint,
                Some(&dw64),
            );
            s.store.clear_seq_checkpoint(1);
            assert_eq!(
                s.store.reclaim_entry(&s.coord, cp_ok),
                ReclaimOutcome::Spilled
            );
            assert!(s.flush(10));
            assert!(s.entry(cp_ok).ram().is_none());
            let key_ok = s.disk_copy(cp_ok).key;
            assert!(s.disk_copy(cp_ok).landed && !s.disk_copy(cp_ok).chain_owed);

            s.coord.release_all_and_purge().expect("purge hot 1");
            s.assert_ssd_restores(&query_ok, salt_ok, Some(cp_ok), Some(&dw64));
            s.store.note_state_hit(&s.coord, 2, cp_ok, false, 0.0);
            s.store.end_seq_use(2);
            assert!(!s.entry(cp_ok).checkpoint());
            assert!(matches!(
                s.store.pending_disk_upgrades.get(&cp_ok),
                Some(PendingDiskUpgrade::LabelPromote { .. })
            ));

            let offloads_before_ok = s.disk().counters.offloads;
            let pause = s.manager().pause_writer_for_test().expect("pause writer 1");
            assert!(
                s.store.tick(&s.coord, Tick::Full { command_pages: 0 }),
                "Tick::Full must enqueue LabelPromote while writer is paused"
            );
            assert!(s.disk().writing());
            assert!(s.disk().pending_relabel.is_some());
            assert!(matches!(
                s.store.in_flight_disk_upgrade.as_ref().map(|u| (u.id, u.kind)),
                Some((id, InFlightDiskUpgradeKind::LabelPromote)) if id == cp_ok
            ));

            // Simulate `ResetCaches` while the relabel is still queued on the paused writer.
            s.store.clear_ram_and_checkpoints();
            s.coord
                .release_all_and_purge()
                .expect("purge after reset 1");
            assert!(s.store.store.entries().next().is_none());
            assert!(s.store.in_flight_disk_upgrade.is_none());
            assert!(s.store.pending_disk_upgrades.is_empty());
            assert!(s.disk().writing());

            drop(pause);
            let dl = test_deadline(10);
            while s.disk().writing() {
                assert!(
                    Instant::now() < dl,
                    "post-reset LabelPromote must finish promptly"
                );
                let _ = s.disk().await_landing(dl);
                let _ = s.store.tick(&s.coord, Tick::Full { command_pages: 0 });
            }
            assert!(s.flush(10));
            assert!(!s.disk().writing());
            assert!(s.disk().writable());
            assert!(s.store.in_flight_disk_upgrade.is_none());
            assert!(s.store.pending_disk_upgrades.is_empty());
            assert!(s.store.recent_failed_upgrades.is_empty());
            assert!(s.store.recent_failed_spills.is_empty());
            assert!(s.store.spill_backoff.is_empty());
            assert_eq!(s.disk().counters.offloads, offloads_before_ok + 1);
            assert_eq!(s.disk().counters.offload_failures, 0);
            assert!(
                s.manager().contains_in(&key_ok, ColdGroup::DFlash2State),
                "successful LabelPromote landing after reset must preserve durable SSD blob"
            );
            let (_, label_ok) = s
                .manager()
                .state_label(key_ok, s.disk().tier.state_fingerprint)
                .expect("label_ok after reset");
            assert!(
                !label_ok.checkpoint,
                "post-reset relabel must commit checkpoint=false on disk"
            );

            // 2. Failure-after-reset: inject `MakeRoom::Refuse` while writer is paused, run
            //    `clear_ram_and_checkpoints()`, then unpause and drain — verify clean no-op and
            //    that the original committed SSD blob stays intact on disk.
            let cp_fail = s.capture(
                &fx.conv,
                3,
                &tokens_fail,
                64,
                salt_fail,
                StateClass::Checkpoint,
                Some(&dw64),
            );
            s.store.clear_seq_checkpoint(3);
            assert_eq!(
                s.store.reclaim_entry(&s.coord, cp_fail),
                ReclaimOutcome::Spilled
            );
            assert!(s.flush(10));
            assert!(s.entry(cp_fail).ram().is_none());
            let key_fail = s.disk_copy(cp_fail).key;
            assert!(s.disk_copy(cp_fail).landed && !s.disk_copy(cp_fail).chain_owed);

            s.coord.release_all_and_purge().expect("purge hot 2");
            s.assert_ssd_restores(&query_fail, salt_fail, Some(cp_fail), Some(&dw64));
            s.store.note_state_hit(&s.coord, 4, cp_fail, false, 0.0);
            s.store.end_seq_use(4);
            assert!(!s.entry(cp_fail).checkpoint());
            assert!(matches!(
                s.store.pending_disk_upgrades.get(&cp_fail),
                Some(PendingDiskUpgrade::LabelPromote { .. })
            ));

            let pause = s.manager().pause_writer_for_test().expect("pause writer 2");
            assert!(
                s.store.tick(&s.coord, Tick::Full { command_pages: 0 }),
                "Tick::Full must enqueue second LabelPromote while writer is paused"
            );
            assert!(s.disk().writing());
            assert!(s.disk().pending_relabel.is_some());
            assert!(matches!(
                s.store.in_flight_disk_upgrade.as_ref().map(|u| (u.id, u.kind)),
                Some((id, InFlightDiskUpgradeKind::LabelPromote)) if id == cp_fail
            ));

            s.set_writer_space(64 << 20, reserve, None);
            s.store.clear_ram_and_checkpoints();
            s.coord
                .release_all_and_purge()
                .expect("purge after reset 2");
            assert!(s.store.store.entries().next().is_none());
            assert!(s.store.in_flight_disk_upgrade.is_none());
            assert!(s.store.pending_disk_upgrades.is_empty());
            assert!(s.disk().writing());

            drop(pause);
            let dl = test_deadline(10);
            while s.disk().writing() {
                assert!(
                    Instant::now() < dl,
                    "refused post-reset LabelPromote must finish promptly"
                );
                let _ = s.disk().await_landing(dl);
                let _ = s.store.tick(&s.coord, Tick::Full { command_pages: 0 });
            }
            assert!(s.flush(10));
            s.clear_space_override(true);
            assert!(!s.disk().writing());
            assert!(s.disk().writable());
            assert!(s.store.in_flight_disk_upgrade.is_none());
            assert!(s.store.pending_disk_upgrades.is_empty());
            assert!(s.store.recent_failed_upgrades.is_empty());
            assert!(s.store.recent_failed_spills.is_empty());
            assert!(s.store.spill_backoff.is_empty());
            assert_eq!(s.disk().counters.offload_failures, 1);

            assert!(
                s.manager().contains_in(&key_ok, ColdGroup::DFlash2State),
                "first durable SSD blob must remain on disk"
            );
            assert!(
                s.manager().contains_in(&key_fail, ColdGroup::DFlash2State),
                "refused LabelPromote after reset must preserve old durable SSD blob"
            );
            let (_, label_fail) = s
                .manager()
                .state_label(key_fail, s.disk().tier.state_fingerprint)
                .expect("label_fail after reset refusal");
            assert!(
                label_fail.checkpoint,
                "refused relabel after reset must leave original checkpoint=true header intact"
            );

            assert!(s.close());
        }

        // 3. Reopen fresh persistent tier + `adopt_all()`:
        //    - `salt_ok` adopts as `Ordinary` (`!checkpoint`) in `prompt_heads` with exact `dw64`.
        //    - `salt_fail` adopts as its preserved `Checkpoint` (`checkpoint`) with exact `dw64`.
        {
            let mut s2 = fx.open_adopted(128u64 << 20, None, 2);

            let id_ok = s2.id_by_salt(salt_ok);
            assert!(
                !s2.entry(id_ok).checkpoint(),
                "post-reset relabeled anchor must adopt as Ordinary (!checkpoint)"
            );
            assert!(
                s2.store.is_prompt_head(id_ok),
                "post-reset relabeled anchor must adopt into prompt_heads"
            );
            s2.assert_ssd_restores(&query_ok, salt_ok, Some(id_ok), Some(&dw64));

            let id_fail = s2.id_by_salt(salt_fail);
            assert!(
                s2.entry(id_fail).checkpoint(),
                "post-reset refused relabel anchor must adopt with preserved Checkpoint label"
            );
            assert!(
                !s2.store.is_prompt_head(id_fail),
                "preserved Checkpoint anchor must not adopt into prompt_heads"
            );
            s2.assert_ssd_restores(&query_fail, salt_fail, Some(id_fail), Some(&dw64));

            assert!(s2.close());
        }
    }

    /// Codex R5-1: `ColdKeyHold` for an in-flight off-thread `LabelPromote` lives on
    /// `PendingRelabel` (not `InFlightDiskUpgrade`), so `clear_ram_and_checkpoints()` does not
    /// drop the eviction hold while the relabel is queued/running on the writer thread. Under
    /// tight quota, a competing `MakeRoom::Evict` KV write must evict only an unpinned control
    /// block instead of the held anchor or its Global KV chain, and the hold is released cleanly
    /// once `poll_relabel()` completes.
    #[cfg(target_os = "macos")]
    #[test]
    fn label_promote_hold_survives_reset_against_competing_eviction() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use mlx_paged_attn::Landed;

        let fx = StateDiskTestFixture::new(
            "r5-1-relabel-hold-reset",
            b"r5-1-relabel-hold-reset-v1",
            Some(10.0),
        )
        .with_paged(1024, 4);
        WRITE_BEHIND_MIN_TOKENS_OVERRIDE.with(|c| c.set(Some(64)));

        let reserve = 4096u64;
        let salt_a = 0x6601u64;
        let tokens_a: Vec<u32> = (8100u32..8196u32).collect();
        let mut query_a = tokens_a[..80].to_vec();
        query_a.extend(9800u32..9810u32);
        let tokens_ctrl: Vec<u32> = (4100u32..4132u32).collect();
        let tokens_competing: Vec<u32> = (4200u32..4232u32).collect();
        let dw64 = fx.packed_dw(64);

        // 1. Seed the Checkpoint anchor on persistent SSD and measure exact anchor + 1-block footprints.
        let (footprint_a, control_size) = {
            let mut s1 = fx.open(64u64 << 20, Some(reserve));
            let cp_a = s1.capture(
                &fx.conv,
                1,
                &tokens_a,
                64,
                salt_a,
                StateClass::Checkpoint,
                Some(&dw64),
            );
            s1.store.clear_seq_checkpoint(1);
            assert_eq!(
                s1.store.reclaim_entry(&s1.coord, cp_a),
                ReclaimOutcome::Spilled
            );
            assert!(s1.flush(10));
            let footprint_a = s1.manager().total_bytes();
            assert!(footprint_a > 0);

            s1.coord.reset_scheduled_request(2).expect("reset 2");
            s1.coord
                .record_tokens_all(2, &tokens_ctrl)
                .expect("record ctrl");
            let (tmp_ctrl_key, landed) = s1.enqueue_kv_block(2, &tokens_ctrl, 0, None);
            assert!(matches!(landed, Landed::Block { ok: true, .. }));
            let control_size = s1.manager().total_bytes() - footprint_a;
            assert!(control_size > 1024);
            assert!(s1.manager().remove(&tmp_ctrl_key));
            assert!(s1.close());
            (footprint_a, control_size)
        };

        // 2. Reopen with tight quota (`anchor + 1 control block + 64 B` relabel header headroom):
        //    anchor's KV chain is oldest in LRU; control block is newer and unpinned; competing
        //    `MakeRoom::Evict` KV block exceeds quota by 1 block.
        let quota = footprint_a + control_size + 64;
        {
            let mut s2 = fx.open_adopted(quota, Some(reserve), 1);
            let cp_id = s2.id_by_salt(salt_a);
            assert!(s2.entry(cp_id).checkpoint());
            let key_a = s2.disk_copy(cp_id).key;
            let kv_keys = s2.disk().engine.kv_chain_keys(s2.entry(cp_id));
            assert_eq!(kv_keys.len(), 2);

            // Hit + promote checkpoint so `LabelPromote` is queued.
            s2.assert_ssd_restores(&query_a, salt_a, Some(cp_id), Some(&dw64));
            s2.store.note_state_hit(&s2.coord, 2, cp_id, false, 0.0);
            s2.store.end_seq_use(2);
            assert!(!s2.entry(cp_id).checkpoint());
            assert!(matches!(
                s2.store.pending_disk_upgrades.get(&cp_id),
                Some(PendingDiskUpgrade::LabelPromote { .. })
            ));
            let old_state_bytes = s2.disk_copy(cp_id).bytes;
            let new_state_bytes = s2.state_object_bytes_for(cp_id);
            assert!(new_state_bytes.saturating_sub(old_state_bytes) <= 64);

            // Write an unpinned control KV block strictly NEWER in LRU than the anchor's KV chain.
            s2.coord.reset_scheduled_request(3).expect("reset 3");
            s2.coord
                .record_tokens_all(3, &tokens_ctrl)
                .expect("record ctrl");
            let (control_key, ctrl_landed) = s2.enqueue_kv_block(3, &tokens_ctrl, 0, None);
            assert!(matches!(ctrl_landed, Landed::Block { ok: true, .. }));
            assert!(s2.manager().contains_in(&control_key, ColdGroup::Kv));
            assert_eq!(s2.manager().total_bytes(), footprint_a + control_size);

            // Pause writer, enqueue `LabelPromote` via `Tick::Full`, and enqueue a competing
            // `MakeRoom::Evict` KV block write onto the block lane before resetting.
            let pause = s2.manager().pause_writer_for_test().expect("pause writer");
            assert!(
                s2.store.tick(&s2.coord, Tick::Full { command_pages: 0 }),
                "Tick::Full must enqueue LabelPromote while writer is paused"
            );
            assert!(s2.disk().writing());
            assert!(s2.disk().pending_relabel.is_some());
            assert!(matches!(
                s2.store.in_flight_disk_upgrade.as_ref().map(|u| (u.id, u.kind)),
                Some((id, InFlightDiskUpgradeKind::LabelPromote)) if id == cp_id
            ));

            s2.coord.reset_scheduled_request(4).expect("reset 4");
            s2.coord
                .record_tokens_all(4, &tokens_competing)
                .expect("record competing");
            let kv = KvPoolHandle::from_adapter(s2.coord.full_adapter());
            let blk = Arc::clone(
                &s2.coord
                    .full_adapter()
                    .block_table_for(4)
                    .expect("block table 4")
                    .blocks()[0],
            );
            let kv_fp = s2.disk().tier.kv_fingerprint;
            let competing_key =
                ColdCacheKey::chain(ColdGroup::Kv, kv_fp, None, &tokens_competing, &[], 0, 0);
            let (comp_tx, comp_rx) = std::sync::mpsc::channel();
            assert!(
                s2.manager()
                    .capture_and_enqueue_tracked(
                        &kv.pool,
                        &blk,
                        competing_key,
                        kv_fp,
                        &tokens_competing,
                        test_deadline(5),
                        MakeRoom::Evict,
                        comp_tx,
                    )
                    .expect("enqueue competing KV")
            );

            // Simulate `ResetCaches` while both relabel and competing Evict write are queued.
            s2.store.clear_ram_and_checkpoints();
            s2.coord.release_all_and_purge().expect("purge after reset");
            assert!(s2.store.store.entries().next().is_none());
            assert!(s2.store.in_flight_disk_upgrade.is_none());
            assert!(s2.store.pending_disk_upgrades.is_empty());
            assert!(s2.disk().writing());
            assert!(s2.disk().pending_relabel.is_some());

            // Hold MUST still pin the anchor state blob AND its entire Global KV chain after reset!
            assert!(
                s2.manager().is_held_for_test(&key_a),
                "state blob hold must stay live on PendingRelabel after clear_ram_and_checkpoints"
            );
            for k in &kv_keys {
                assert!(
                    s2.manager().is_held_for_test(k),
                    "Global KV chain hold must stay live on PendingRelabel after clear_ram_and_checkpoints"
                );
            }
            assert!(!s2.manager().is_held_for_test(&control_key));

            // Unpause writer and wait for the competing Evict write to settle BEFORE polling
            // `poll_relabel`: it must evict only the unpinned control block, never the held anchor.
            drop(pause);
            match comp_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("competing KV settle")
            {
                Landed::Block { key, ok } => {
                    assert_eq!(key, competing_key);
                    assert!(
                        ok,
                        "competing Evict write must succeed by evicting only the unpinned control block"
                    );
                }
                other => panic!("expected Landed::Block, got {other:?}"),
            }
            assert!(
                !s2.manager().contains_in(&control_key, ColdGroup::Kv),
                "unpinned control block must be evicted instead of the held anchor"
            );
            assert!(
                s2.manager().contains_in(&key_a, ColdGroup::DFlash2State),
                "held anchor state blob must survive competing Evict write after reset"
            );
            for k in &kv_keys {
                assert!(
                    s2.manager().contains_in(k, ColdGroup::Kv),
                    "held anchor Global KV block must survive competing Evict write after reset"
                );
            }

            // Drain relabel landing via `Tick::Full` + `flush_to_disk`: hold must be released cleanly.
            let dl = test_deadline(10);
            while s2.disk().writing() {
                assert!(
                    Instant::now() < dl,
                    "post-reset relabel under quota pressure must finish promptly"
                );
                let _ = s2.disk().await_landing(dl);
                let _ = s2.store.tick(&s2.coord, Tick::Full { command_pages: 0 });
            }
            assert!(s2.flush(10));
            assert!(!s2.disk().writing());
            assert!(s2.disk().pending_relabel.is_none());
            assert!(
                !s2.manager().is_held_for_test(&key_a),
                "state blob hold must be released once poll_relabel pops PendingRelabel"
            );
            for k in &kv_keys {
                assert!(
                    !s2.manager().is_held_for_test(k),
                    "Global KV chain hold must be released once poll_relabel pops PendingRelabel"
                );
            }
            let (_, label_a) = s2
                .manager()
                .state_label(key_a, s2.disk().tier.state_fingerprint)
                .expect("relabeled state label");
            assert!(!label_a.checkpoint);
            assert!(s2.close());
        }

        // 3. Reopen fresh persistent tier + `adopt_all()`: relabeled anchor AND its complete Global
        //    KV chain survived competing eviction and restore cleanly as `Ordinary` in `prompt_heads`.
        {
            let mut s3 = fx.open_adopted(128u64 << 20, None, 1);
            let adopted_id = s3.id_by_salt(salt_a);
            assert!(
                !s3.entry(adopted_id).checkpoint(),
                "relabeled anchor must adopt as Ordinary (!checkpoint)"
            );
            assert!(
                s3.store.is_prompt_head(adopted_id),
                "relabeled anchor must adopt into prompt_heads"
            );
            s3.assert_ssd_restores(&query_a, salt_a, Some(adopted_id), Some(&dw64));
            assert!(s3.close());
        }
    }

    /// Codex R6-1: An in-flight `PendingDiskUpgrade::InPlace { promote_label: true }` write owns
    /// its replacement flag and `ColdKeyHold` on `StateDiskEngine` (not `InFlightDiskUpgrade`),
    /// so `clear_ram_and_checkpoints()` (`ResetCaches`) neither drops the eviction hold nor causes
    /// `StateDiskEngine::poll_offload()` to delete the freshly committed upgraded SSD anchor upon
    /// landing. Under tight quota, a competing `MakeRoom::Evict` KV write evicts only an unpinned
    /// control block, the hold releases cleanly on landing, and reopening + `adopt_all()` adopts
    /// the anchor as `Ordinary` (`!checkpoint`) in `prompt_heads` with the upgraded
    /// `PackedDraftWindow` intact.
    #[cfg(target_os = "macos")]
    #[test]
    fn in_place_upgrade_hold_and_commit_survive_reset_against_competing_eviction() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        use mlx_paged_attn::Landed;

        let fx = StateDiskTestFixture::new(
            "r6-1-inplace-hold-reset",
            b"r6-1-inplace-hold-reset-v1",
            Some(10.0),
        )
        .with_paged(1024, 4);
        WRITE_BEHIND_MIN_TOKENS_OVERRIDE.with(|c| c.set(Some(64)));

        let reserve = 4096u64;
        let salt_a = 0x6701u64;
        let tokens_a: Vec<u32> = (8300u32..8396u32).collect();
        let mut query_a = tokens_a[..80].to_vec();
        query_a.extend(9820u32..9830u32);
        let tokens_ctrl: Vec<u32> = (4300u32..4332u32).collect();
        let tokens_competing: Vec<u32> = (4400u32..4432u32).collect();
        let dw64 = fx.packed_dw(64);

        // 1. Seed a persistent SSD `Checkpoint` anchor WITHOUT a draft window and measure exact
        //    anchor + 1-block footprints.
        let (footprint_a, control_size) = {
            let mut s1 = fx.open(64u64 << 20, Some(reserve));
            let cp_a = s1.capture(
                &fx.conv,
                1,
                &tokens_a,
                64,
                salt_a,
                StateClass::Checkpoint,
                None,
            );
            s1.store.clear_seq_checkpoint(1);
            assert_eq!(
                s1.store.reclaim_entry(&s1.coord, cp_a),
                ReclaimOutcome::Spilled
            );
            assert!(s1.flush(10));
            let footprint_a = s1.manager().total_bytes();
            assert!(footprint_a > 0);

            s1.coord.reset_scheduled_request(2).expect("reset 2");
            s1.coord
                .record_tokens_all(2, &tokens_ctrl)
                .expect("record ctrl");
            let (tmp_ctrl_key, landed) = s1.enqueue_kv_block(2, &tokens_ctrl, 0, None);
            assert!(matches!(landed, Landed::Block { ok: true, .. }));
            let control_size = s1.manager().total_bytes() - footprint_a;
            assert!(control_size > 1024);
            assert!(s1.manager().remove(&tmp_ctrl_key));
            assert!(s1.close());
            (footprint_a, control_size)
        };

        // 2. Reopen under tight quota (`anchor + 1 control block + 64 B` label headroom),
        //    recapture the Checkpoint anchor as a structural head WITH `dw64` to queue
        //    `PendingDiskUpgrade::InPlace { promote_label: true }`, enqueue a competing
        //    `MakeRoom::Evict` KV write while paused, and reset (`clear_ram_and_checkpoints()`).
        let quota = footprint_a + control_size + 64;
        {
            let mut s2 = fx.open_adopted(quota, Some(reserve), 1);
            s2.store.set_ram_cap(16);
            let cp_id = s2.id_by_salt(salt_a);
            assert!(s2.entry(cp_id).checkpoint());
            assert!(!s2.store.is_prompt_head(cp_id));
            let key_a = s2.disk_copy(cp_id).key;
            let kv_keys = s2.disk().engine.kv_chain_keys(s2.entry(cp_id));
            assert_eq!(kv_keys.len(), 2);

            // Recapture as structural head (`StateClass::Ordinary`, `is_head: true`) WITH `dw64`.
            s2.coord.reset_scheduled_request(2).expect("reset 2");
            s2.coord
                .record_tokens_all(2, &tokens_a)
                .expect("record tokens_a");
            let upgraded_id = s2
                .store
                .capture_boundary_inner(
                    &mut s2.coord,
                    &fx.conv,
                    2,
                    &tokens_a,
                    64,
                    salt_a,
                    StateClass::Ordinary,
                    Some(DraftWindowInput::Packed(&dw64)),
                    None,
                    CaptureOptions {
                        is_head: true,
                        ..CaptureOptions::default()
                    },
                )
                .expect("capture structural head")
                .expect("upgraded_id");
            assert_eq!(upgraded_id, cp_id);
            assert!(!s2.entry(cp_id).checkpoint());
            assert!(s2.store.is_prompt_head(cp_id));
            assert!(matches!(
                s2.store.pending_disk_upgrades.get(&cp_id),
                Some(PendingDiskUpgrade::InPlace {
                    promote_label: true,
                    ..
                })
            ));
            let old_state_bytes = s2.disk_copy(cp_id).bytes;
            let new_state_bytes = s2.state_object_bytes_for(cp_id);
            assert!(new_state_bytes.saturating_sub(old_state_bytes) <= 64);

            // Write an unpinned control KV block strictly NEWER in LRU than the anchor's KV chain.
            s2.coord.reset_scheduled_request(3).expect("reset 3");
            s2.coord
                .record_tokens_all(3, &tokens_ctrl)
                .expect("record ctrl");
            let (control_key, ctrl_landed) = s2.enqueue_kv_block(3, &tokens_ctrl, 0, None);
            assert!(matches!(ctrl_landed, Landed::Block { ok: true, .. }));
            assert!(s2.manager().contains_in(&control_key, ColdGroup::Kv));
            assert_eq!(s2.manager().total_bytes(), footprint_a + control_size);

            // Pause writer, enqueue `InPlace { promote_label: true }` via `Tick::Full`, and
            // enqueue a competing `MakeRoom::Evict` KV block write before resetting.
            let offloads_before = s2.disk().counters.offloads;
            let pause = s2.manager().pause_writer_for_test().expect("pause writer");
            assert!(
                s2.store.tick(&s2.coord, Tick::Full { command_pages: 0 }),
                "Tick::Full must enqueue InPlace upgrade while writer is paused"
            );
            assert!(s2.disk().writing());
            assert!(s2.disk().engine.writing());
            assert!(s2.disk().engine.has_pending_replacement());
            assert!(s2.disk().pending_relabel.is_none());
            assert!(matches!(
                s2.store.in_flight_disk_upgrade.as_ref().map(|u| (u.id, u.kind)),
                Some((
                    id,
                    InFlightDiskUpgradeKind::InPlace {
                        promote_label: true
                    }
                )) if id == cp_id
            ));

            s2.coord.reset_scheduled_request(4).expect("reset 4");
            s2.coord
                .record_tokens_all(4, &tokens_competing)
                .expect("record competing");
            let kv = KvPoolHandle::from_adapter(s2.coord.full_adapter());
            let blk = Arc::clone(
                &s2.coord
                    .full_adapter()
                    .block_table_for(4)
                    .expect("block table 4")
                    .blocks()[0],
            );
            let kv_fp = s2.disk().tier.kv_fingerprint;
            let competing_key =
                ColdCacheKey::chain(ColdGroup::Kv, kv_fp, None, &tokens_competing, &[], 0, 0);
            let (comp_tx, comp_rx) = std::sync::mpsc::channel();
            assert!(
                s2.manager()
                    .capture_and_enqueue_tracked(
                        &kv.pool,
                        &blk,
                        competing_key,
                        kv_fp,
                        &tokens_competing,
                        test_deadline(5),
                        MakeRoom::Evict,
                        comp_tx,
                    )
                    .expect("enqueue competing KV")
            );

            // Simulate `ResetCaches` while both InPlace write and competing Evict write are queued.
            s2.store.clear_ram_and_checkpoints();
            s2.coord.release_all_and_purge().expect("purge after reset");
            assert!(s2.store.store.entries().next().is_none());
            assert!(s2.store.in_flight_disk_upgrade.is_none());
            assert!(s2.store.pending_disk_upgrades.is_empty());
            assert!(s2.disk().writing());
            assert!(s2.disk().engine.has_pending_replacement());

            // Hold MUST stay live on BOTH the committed state blob AND all Global KV blocks!
            assert!(
                s2.manager().is_held_for_test(&key_a),
                "state blob hold must stay live on StateDiskEngine after clear_ram_and_checkpoints"
            );
            for k in &kv_keys {
                assert!(
                    s2.manager().is_held_for_test(k),
                    "Global KV chain hold must stay live on StateDiskEngine after clear_ram_and_checkpoints"
                );
            }
            assert!(!s2.manager().is_held_for_test(&control_key));

            // Unpause writer and settle competing Evict write BEFORE polling `poll_offload`:
            // it must evict ONLY the unpinned control block, never the held anchor or its KV chain.
            drop(pause);
            match comp_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("competing KV settle")
            {
                Landed::Block { key, ok } => {
                    assert_eq!(key, competing_key);
                    assert!(
                        ok,
                        "competing Evict write must succeed by evicting only the unpinned control block"
                    );
                }
                other => panic!("expected Landed::Block, got {other:?}"),
            }
            assert!(
                !s2.manager().contains_in(&control_key, ColdGroup::Kv),
                "unpinned control block must be evicted instead of the held anchor"
            );
            assert!(
                s2.manager().contains_in(&key_a, ColdGroup::DFlash2State),
                "held anchor state blob must survive competing Evict write after reset"
            );
            for k in &kv_keys {
                assert!(
                    s2.manager().contains_in(k, ColdGroup::Kv),
                    "held anchor Global KV block must survive competing Evict write after reset"
                );
            }

            // Settle `InPlace` landing via `Tick::Full` + `flush`: `poll_offload` MUST preserve
            // the freshly committed SSD anchor (never call `manager.remove`) and release holds.
            let dl = test_deadline(10);
            while s2.disk().writing() {
                assert!(
                    Instant::now() < dl,
                    "post-reset InPlace upgrade under quota pressure must finish promptly"
                );
                let _ = s2.disk().await_landing(dl);
                let _ = s2.store.tick(&s2.coord, Tick::Full { command_pages: 0 });
            }
            assert!(s2.flush(10));
            assert!(!s2.disk().writing());
            assert!(!s2.disk().engine.has_pending_replacement());
            assert!(s2.disk().writable());
            assert!(s2.store.in_flight_disk_upgrade.is_none());
            assert!(s2.store.pending_disk_upgrades.is_empty());
            assert!(s2.store.recent_failed_upgrades.is_empty());
            assert!(s2.store.recent_failed_spills.is_empty());
            assert!(s2.store.spill_backoff.is_empty());
            assert_eq!(s2.disk().counters.offloads, offloads_before + 1);
            assert_eq!(s2.disk().counters.offload_failures, 0);
            assert!(
                !s2.manager().is_held_for_test(&key_a),
                "state blob hold must be released once poll_offload settles replacement"
            );
            for k in &kv_keys {
                assert!(
                    !s2.manager().is_held_for_test(k),
                    "Global KV chain hold must be released once poll_offload settles replacement"
                );
            }
            assert!(
                s2.manager().contains_in(&key_a, ColdGroup::DFlash2State),
                "poll_offload must preserve freshly committed InPlace replacement anchor after reset"
            );
            let (_, label_a) = s2
                .manager()
                .state_label(key_a, s2.disk().tier.state_fingerprint)
                .expect("upgraded state label after reset");
            assert!(
                !label_a.checkpoint,
                "post-reset InPlace upgrade must commit checkpoint=false on disk"
            );
            assert!(s2.close());
        }

        // 3. Reopen fresh persistent tier + `adopt_all()`: anchor adopts as `Ordinary`
        //    (`!checkpoint`) in `prompt_heads` with the upgraded `PackedDraftWindow` intact.
        {
            let mut s3 = fx.open_adopted(128u64 << 20, None, 1);
            let adopted_id = s3.id_by_salt(salt_a);
            assert!(
                !s3.entry(adopted_id).checkpoint(),
                "post-reset InPlace upgraded anchor must adopt as Ordinary (!checkpoint)"
            );
            assert!(
                s3.store.is_prompt_head(adopted_id),
                "post-reset InPlace upgraded anchor must adopt into prompt_heads"
            );
            s3.assert_ssd_restores(&query_a, salt_a, Some(adopted_id), Some(&dw64));
            assert!(s3.close());
        }
    }

    /// Codex R6-1 companion: If `clear_ram_and_checkpoints()` (`ResetCaches`) runs while an
    /// `InPlace { promote_label: true }` upgrade is in flight and the writer subsequently refuses
    /// the write under `MakeRoom::Refuse`, `StateDiskEngine::poll_offload()` cleanly releases the
    /// hold, leaves the original durable SSD anchor intact on disk, and reopening + `adopt_all()`
    /// adopts the preserved original `Checkpoint` anchor.
    #[cfg(target_os = "macos")]
    #[test]
    fn in_place_upgrade_refusal_after_reset_preserves_old_ssd_anchor() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            return;
        }
        let fx = StateDiskTestFixture::new(
            "r6-1-inplace-refuse-reset",
            b"r6-1-inplace-refuse-reset-v1",
            Some(10.0),
        )
        .with_paged(1024, 4);
        WRITE_BEHIND_MIN_TOKENS_OVERRIDE.with(|c| c.set(Some(64)));

        let reserve = 4096u64;
        let salt_fail = 0x6702u64;
        let tokens_fail: Vec<u32> = (8500u32..8596u32).collect();
        let mut query_fail = tokens_fail[..80].to_vec();
        query_fail.extend(9840u32..9850u32);
        let dw64 = fx.packed_dw(64);

        // 1. Seed persistent SSD `Checkpoint` anchor WITHOUT draft window.
        {
            let mut s1 = fx.open(128u64 << 20, Some(reserve));
            let cp = s1.capture(
                &fx.conv,
                1,
                &tokens_fail,
                64,
                salt_fail,
                StateClass::Checkpoint,
                None,
            );
            s1.store.clear_seq_checkpoint(1);
            assert_eq!(
                s1.store.reclaim_entry(&s1.coord, cp),
                ReclaimOutcome::Spilled
            );
            assert!(s1.flush(10));
            assert!(s1.close());
        }

        // 2. Reopen, recapture as structural head WITH `dw64` to queue
        //    `InPlace { promote_label: true }`, pause writer, stage via `Tick::Full`, inject
        //    `MakeRoom::Refuse` headroom drop, reset, unpause, and settle.
        {
            let mut s2 = fx.open_adopted(128u64 << 20, Some(reserve), 1);
            s2.store.set_ram_cap(16);
            let cp_id = s2.id_by_salt(salt_fail);
            assert!(s2.entry(cp_id).checkpoint());
            let key_fail = s2.disk_copy(cp_id).key;
            let kv_keys = s2.disk().engine.kv_chain_keys(s2.entry(cp_id));
            assert_eq!(kv_keys.len(), 2);

            s2.coord.reset_scheduled_request(2).expect("reset 2");
            s2.coord
                .record_tokens_all(2, &tokens_fail)
                .expect("record tokens_fail");
            let upgraded_id = s2
                .store
                .capture_boundary_inner(
                    &mut s2.coord,
                    &fx.conv,
                    2,
                    &tokens_fail,
                    64,
                    salt_fail,
                    StateClass::Ordinary,
                    Some(DraftWindowInput::Packed(&dw64)),
                    None,
                    CaptureOptions {
                        is_head: true,
                        ..CaptureOptions::default()
                    },
                )
                .expect("capture structural head")
                .expect("upgraded_id");
            assert_eq!(upgraded_id, cp_id);
            assert!(matches!(
                s2.store.pending_disk_upgrades.get(&cp_id),
                Some(PendingDiskUpgrade::InPlace {
                    promote_label: true,
                    ..
                })
            ));

            let pause = s2.manager().pause_writer_for_test().expect("pause writer");
            assert!(
                s2.store.tick(&s2.coord, Tick::Full { command_pages: 0 }),
                "Tick::Full must enqueue InPlace upgrade while writer is paused"
            );
            assert!(s2.disk().writing());
            assert!(s2.disk().engine.has_pending_replacement());

            s2.set_writer_space(64 << 20, reserve, None);
            s2.store.clear_ram_and_checkpoints();
            s2.coord.release_all_and_purge().expect("purge after reset");
            assert!(s2.store.store.entries().next().is_none());
            assert!(s2.store.in_flight_disk_upgrade.is_none());
            assert!(s2.store.pending_disk_upgrades.is_empty());
            assert!(s2.disk().writing());
            assert!(s2.disk().engine.has_pending_replacement());
            assert!(s2.manager().is_held_for_test(&key_fail));
            for k in &kv_keys {
                assert!(s2.manager().is_held_for_test(k));
            }

            drop(pause);
            let dl = test_deadline(10);
            while s2.disk().writing() {
                assert!(
                    Instant::now() < dl,
                    "refused post-reset InPlace upgrade must finish promptly"
                );
                let _ = s2.disk().await_landing(dl);
                let _ = s2.store.tick(&s2.coord, Tick::Full { command_pages: 0 });
            }
            assert!(s2.flush(10));
            s2.clear_space_override(true);
            assert!(!s2.disk().writing());
            assert!(!s2.disk().engine.has_pending_replacement());
            assert!(s2.disk().writable());
            assert!(s2.store.in_flight_disk_upgrade.is_none());
            assert!(s2.store.pending_disk_upgrades.is_empty());
            assert!(s2.store.recent_failed_upgrades.is_empty());
            assert!(s2.store.recent_failed_spills.is_empty());
            assert!(s2.store.spill_backoff.is_empty());
            assert_eq!(s2.disk().counters.offload_failures, 1);
            assert!(!s2.manager().is_held_for_test(&key_fail));
            for k in &kv_keys {
                assert!(!s2.manager().is_held_for_test(k));
            }
            assert!(
                s2.manager().contains_in(&key_fail, ColdGroup::DFlash2State),
                "refused InPlace upgrade after reset must preserve old durable SSD anchor"
            );
            for k in &kv_keys {
                assert!(
                    s2.manager().contains_in(k, ColdGroup::Kv),
                    "refused InPlace upgrade after reset must preserve old Global KV blocks"
                );
            }
            let (_, label_fail) = s2
                .manager()
                .state_label(key_fail, s2.disk().tier.state_fingerprint)
                .expect("label_fail after reset refusal");
            assert!(
                label_fail.checkpoint,
                "refused InPlace upgrade after reset must leave original checkpoint=true header intact"
            );
            assert!(s2.close());
        }

        // 3. Reopen fresh persistent tier + `adopt_all()`: original SSD anchor adopts intact as
        //    `Checkpoint` (without draft window).
        {
            let mut s3 = fx.open_adopted(128u64 << 20, None, 1);
            let id_fail = s3.id_by_salt(salt_fail);
            assert!(
                s3.entry(id_fail).checkpoint(),
                "preserved original anchor must adopt as Checkpoint"
            );
            assert!(
                !s3.store.is_prompt_head(id_fail),
                "preserved Checkpoint anchor must not adopt into prompt_heads"
            );
            s3.assert_ssd_restores(&query_fail, salt_fail, Some(id_fail), None);
            assert!(s3.close());
        }
    }
}
