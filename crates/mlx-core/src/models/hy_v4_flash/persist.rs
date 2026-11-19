//! The disk tier of the stored states: a state and the KV blocks of its
//! chain survive an engine restart, a copy of Flash Next's
//! `qwen4_exp/prefix_disk.rs` (`ColdContextBacking::KvOnly`) over Hy4's
//! states.
//!
//! ```text
//! tier      the model's own (`disk_tier.rs`): a locked namespace directory, or a temporary one
//! objects   Kv block (12 indexed layers, the pool page's blobs) under its chain key
//!           HybridDraft segment (32 MTP rows) under its draft chain key
//!           DFlash2State record (one fixed span) on the one-slot state lane, lineage in its label
//! sources   paged: the pool's cached block │ flat: the sequence's latent and pooled rows, same bytes
//! write     only lineages of MINIMUM_ROWS or more; a replay state DELAY after publication unless
//!           its conversation went on; durable once its whole chain landed; close: newest first
//! restore   paged: the adapter's cold walk below a stored state's row (never without one)
//!           flat: the blocks read back into the latent and pooled rows
//! ```
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use mlx_paged_attn::{
    ColdCacheBlock, ColdCacheFingerprint, ColdCacheKey, ColdCacheLayout, ColdCacheManager,
    ColdGroup, ColdLayerBlock, ColdObject, ColdSidecar, ColdSidecarLayout, ColdStateLayout,
    ColdStateObject, EnqueueRefused, Landed, MakeRoom, state_object_bytes,
};
use napi::{Error, Result};
use sha2::{Digest, Sha256};

use super::cache::{LayerCache, WINDOW_SLACK};
use super::config::HyV4FlashConfig;
use super::ladder::{Ladder, Snapshot};
use super::mtp::{LagMark, Lagged};
use super::owner::draft_state_at;
use super::paged::{PAGE_DTYPE, PageGeometry};
use super::persist_codec::{
    StateShape, block_layout, decode_segment, decode_state, encode_segment, encode_state_into,
    split_block,
};
use super::persist_writer::{Closer, Parker, SLOT_WAIT, Telemetry, Wake, object_bytes, run};
use super::prefill::{BLOCK_ROWS, StateRow, turn_row};
use super::prefix_state::{
    DraftState, Held, Key, PrefixState, Segment, States, durable, prompt_keys, resumes, turn_keys,
};
use crate::array::MxArray;
use crate::disk_tier::{DiskTier, DiskTierOptions, PersistentIdentity, TierHandle};
use crate::models::qwen3_5::dflash2_state::disk::{adopt_states, label_of, lineage_keys};
use crate::models::qwen3_5::dflash2_state::store::{DiskCopy, EntryId, HybridStateStore, Lineage};
use crate::models::qwen3_5::gdn_checkpoint_store::compute_paged_prefix_block_hashes;
use crate::transformer::paged_kv_cache_adapter::{
    ColdContextBacking, ColdTierContext, PagedKVCacheAdapter, SeqId,
};

/// Splash `WriteBehind` hands a point over this long after publication.
pub const DELAY: Duration = Duration::from_secs(10);
/// Splash `WriteBehind::kMinimumTokens`: a shorter lineage prefills again
/// faster than its state is written and read back.
pub const MINIMUM_ROWS: usize = 2048;
/// A temporary tier copies while memory dropped a state this recently.
const DEMAND_WINDOW: Duration = Duration::from_secs(60);
/// A written state whose chain nothing can add to any more is dropped this
/// long after.
const GIVE_UP: Duration = Duration::from_secs(10);
/// The model thread's disk pass copies blocks out at most this often: a copy
/// waits for the GPU.
const STEP_TOP_UP: Duration = Duration::from_millis(250);
/// Host copies of blocks and segments waiting for the writer.
pub const QUEUE_BYTES: u64 = 128 << 20;
/// Blocks a flat copy reads from the GPU, or a flat restore from disk, at
/// once.
pub const FLAT_RUN: usize = 32;
/// The identity text's first line; a change of any format below bumps it.
pub const TAG: &str = "hy4-prefix-v1";

pub(crate) type Store = HybridStateStore<Held<PrefixState>>;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 disk tier: {msg}"))
}

// ---------------------------------------------------------------------------
// Identity and keys.
// ---------------------------------------------------------------------------

/// Everything that changes a stored block's or state's bits.
pub struct Identity<'a> {
    /// `source_file_identity_digest` of each GGUF shard, in order.
    pub shards: &'a [String],
    /// SHA-256 of the parsed config.
    pub config: String,
    pub page: PageGeometry,
    pub window: usize,
    pub mtp: bool,
    pub gpu: i32,
}

/// The identity as one text, compared as text and hashed into the tier's
/// fingerprint.
pub fn identity_text(id: &Identity<'_>) -> String {
    let p = &id.page;
    let mut out = format!("{TAG}\n");
    for shard in id.shards {
        out.push_str(&format!("shard {shard}\n"));
    }
    out.push_str(&format!("config {}\n", id.config));
    out.push_str(&format!(
        "page block={BLOCK_ROWS} layers={} heads=1 half={} dtype={PAGE_DTYPE:?} token-major index={}x{} index-dtype={PAGE_DTYPE:?}\n",
        p.layers, p.half, p.index_rows, p.index_dim
    ));
    out.push_str(&format!("window {} slack {WINDOW_SLACK}\n", id.window));
    out.push_str(&format!("mtp {}\n", if id.mtp { "on" } else { "off" }));
    out.push_str(&format!("gpu {}\n", id.gpu));
    out
}

/// SHA-256 hex of `cfg` as parsed.
pub fn config_digest(cfg: &HyV4FlashConfig) -> String {
    Sha256::digest(format!("{cfg:?}").as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The fingerprint of `text` and its namespace directory name.
pub fn fingerprint(text: &str) -> Option<(String, ColdCacheFingerprint)> {
    let fp = ColdCacheFingerprint::from_components([text.as_bytes()]);
    Some((fp.to_hex().get(..32)?.to_string(), fp))
}

/// The cold keys of the first `blocks` whole blocks of `tokens` in `group`,
/// block `i` keyed by `keys[i]` (none past the list).
pub fn chain(
    group: ColdGroup,
    fingerprint: ColdCacheFingerprint,
    tokens: &[u32],
    keys: &[Vec<u64>],
    salt: u64,
    blocks: usize,
) -> Vec<ColdCacheKey> {
    lineage_keys(
        group,
        fingerprint,
        Lineage {
            tokens: &tokens[..(blocks * BLOCK_ROWS).min(tokens.len())],
            extra_keys: keys,
            salt,
        },
        BLOCK_ROWS as u32,
    )
}

/// The key of MTP segment `k` (draft rows `[32k, 32k + 32)`): the
/// `HybridDraft` chain of the blocks below `32(k + 1)`, then one link over
/// the token after it, which the segment's last row was computed with.
pub fn segment_key(
    fingerprint: ColdCacheFingerprint,
    draft_chain: &[ColdCacheKey],
    tokens: &[u32],
    k: usize,
) -> Option<ColdCacheKey> {
    let end = (k + 1) * BLOCK_ROWS;
    let blocks = end / BLOCK_ROWS;
    let parent = *draft_chain.get(blocks - 1)?;
    let next = *tokens.get(end)?;
    Some(ColdCacheKey::chain(
        ColdGroup::HybridDraft,
        fingerprint,
        Some(parent),
        &[next],
        &[BLOCK_ROWS as u64],
        0,
        blocks,
    ))
}

/// Whole MTP segments of a state at `row`.
pub fn segments_at(row: usize) -> usize {
    row.saturating_sub(1) / BLOCK_ROWS
}

/// The disk tier a load plans from its options and the persistence
/// switch's environment value `env`: its options and whether it is on.
pub fn disk_plan(
    label: &str,
    cache_disk: Option<&str>,
    persist: Option<bool>,
    cache_dir: Option<String>,
    env: Option<&str>,
) -> Result<(DiskTierOptions, bool)> {
    let tier = DiskTierOptions::from_load(
        cache_disk,
        crate::disk_tier::resolve_persist(persist, env),
        cache_dir,
    )
    .map_err(|message| Error::from_reason(format!("{label} load: {message}")))?;
    let on = tier.cache_disk != crate::disk_tier::CacheDisk::Off;
    Ok((tier, on))
}

// ---------------------------------------------------------------------------
// Block sources.
// ---------------------------------------------------------------------------

/// Where a lineage's block bytes come from.
pub trait Blocks {
    /// Settles the writes before the first read of a pass; false when the
    /// source cannot be read now.
    fn begin(&mut self, lineage: &[u32]) -> bool;
    /// Block `i` (hot hash `hot`) of the lineage of the last `begin`: every
    /// pool layer's blobs; `None` when the source no longer holds it.
    fn block(&mut self, i: usize, hot: u64) -> Option<Vec<ColdLayerBlock>>;
}

/// The pool's cached blocks.
pub struct PoolBlocks<'a>(pub &'a mut PagedKVCacheAdapter);

impl Blocks for PoolBlocks<'_> {
    fn begin(&mut self, _: &[u32]) -> bool {
        if self.0.eval_pending_pool_writes().is_err() {
            return false;
        }
        crate::array::memory::synchronize();
        true
    }

    fn block(&mut self, _: usize, hot: u64) -> Option<Vec<ColdLayerBlock>> {
        let block = self.0.cached_block(hot)?;
        let layers = self
            .0
            .layer_kv_pool()
            .read_block_all_layers(block.block_id)
            .ok()?;
        Some(
            layers
                .into_iter()
                .map(|(keys, values)| ColdLayerBlock { keys, values })
                .collect(),
        )
    }
}

/// The flat sequence's latent and pooled rows, read a run of blocks at a
/// time; only blocks of rows the sequence still holds with the lineage's
/// tokens.
pub struct FlatBlocks<'a> {
    seq: &'a Lagged,
    page: PageGeometry,
    valid: usize,
    /// The first block of the run read, and per indexed layer its latent
    /// and pooled f16 bits.
    run: Option<(usize, Vec<(Vec<u16>, Vec<u16>)>)>,
}

impl<'a> FlatBlocks<'a> {
    pub fn new(seq: &'a Lagged, page: PageGeometry) -> Self {
        Self {
            seq,
            page,
            valid: 0,
            run: None,
        }
    }

    fn read_run(&self, first: usize, blocks: usize) -> Option<Vec<(Vec<u16>, Vec<u16>)>> {
        let per = self.page.index_rows as usize;
        let (lo, hi) = (first * BLOCK_ROWS, (first + blocks) * BLOCK_ROWS);
        let mut out = Vec::with_capacity(self.page.layers as usize);
        for st in self.seq.trunk().layers() {
            let (LayerCache::Latent(buf), Some(ix)) = (&st.cache, st.index.as_ref()) else {
                continue;
            };
            let latent = buf.rows().ok()?;
            let w = latent.shape_at(1).ok()?;
            let rows = latent
                .slice(&[lo as i64, 0], &[hi as i64, w])
                .ok()?
                .to_uint16_native()
                .ok()?;
            let pooled = ix.pooled_rows().ok()?;
            let d = pooled.shape_at(1).ok()?;
            let keys = pooled
                .slice(
                    &[(first * per) as i64, 0],
                    &[((first + blocks) * per) as i64, d],
                )
                .ok()?
                .to_uint16_native()
                .ok()?;
            out.push((rows, keys));
        }
        (out.len() == self.page.layers as usize).then_some(out)
    }
}

impl Blocks for FlatBlocks<'_> {
    fn begin(&mut self, lineage: &[u32]) -> bool {
        let tokens = self.seq.tokens();
        let shared = tokens
            .iter()
            .zip(lineage)
            .take_while(|(a, b)| a == b)
            .count();
        let held = self
            .seq
            .trunk()
            .layers()
            .iter()
            .filter_map(|st| match (&st.cache, st.index.as_ref()) {
                (LayerCache::Latent(buf), Some(ix)) => Some(
                    buf.len()
                        .min(pooled_rows(ix.pooled_len(), self.page.index_rows as usize)),
                ),
                _ => None,
            })
            .min()
            .unwrap_or(0);
        self.valid = shared.min(held) / BLOCK_ROWS;
        self.run = None;
        true
    }

    fn block(&mut self, i: usize, _: u64) -> Option<Vec<ColdLayerBlock>> {
        if i >= self.valid {
            return None;
        }
        let fresh = !matches!(&self.run, Some((first, _)) if i >= *first && i < first + FLAT_RUN);
        if fresh {
            let blocks = FLAT_RUN.min(self.valid - i);
            self.run = Some((i, self.read_run(i, blocks)?));
        }
        let (first, layers) = self.run.as_ref()?;
        let at = i - first;
        let half = self.page.half as usize;
        let per = self.page.index_rows as usize * self.page.index_dim as usize;
        let rows = BLOCK_ROWS * 2 * half;
        layers
            .iter()
            .map(|(latent, pooled)| {
                Some(split_block(
                    latent.get(at * rows..(at + 1) * rows)?,
                    pooled.get(at * per..(at + 1) * per)?,
                    half,
                ))
            })
            .collect()
    }
}

/// Rows whose groups the flat pooled buffers hold: `groups` groups of
/// `BLOCK_ROWS / index_rows` rows.
fn pooled_rows(groups: usize, index_rows: usize) -> usize {
    groups / index_rows.max(1) * BLOCK_ROWS
}

// ---------------------------------------------------------------------------
// The model thread's side.
// ---------------------------------------------------------------------------

/// What one turn of a sequence owes the disk: its lineage up to its last
/// point, keyed as the prefix cache keys it, and how far its blocks and
/// segments are copied out (or found on disk).
struct Cursor {
    lineage: Vec<u32>,
    /// Hot prefix-cache hashes of the lineage's blocks.
    hot: Vec<u64>,
    /// `Kv` cold keys of the lineage's blocks.
    cold: Vec<ColdCacheKey>,
    /// `HybridDraft` chain of the lineage's blocks.
    draft_chain: Vec<ColdCacheKey>,
    /// Rows whose blocks are published (the deepest publication).
    registered: usize,
    /// The MTP segments of the deepest publication (shared handles).
    pending: Vec<Arc<Segment>>,
    blocks: usize,
    segments: usize,
    /// A block left its source before its copy.
    broken: bool,
    /// The turn ended: the cursor goes once it owes nothing.
    closed: bool,
}

/// The tier's one-slot state lane: the record shape and the one staging
/// buffer every state write and read goes through.
struct Lane {
    spans: Vec<u64>,
    /// Compared as text, never parsed.
    geometry: String,
    shape: StateShape,
    /// `None` while the write in flight holds it.
    staging: Option<Vec<u8>>,
    landed_tx: Sender<Landed>,
    landed_rx: Receiver<Landed>,
    /// The state write in flight: its entry and its object.
    pending: Option<(EntryId, ColdCacheKey)>,
}

impl Lane {
    fn new(shape: StateShape) -> Option<Self> {
        let span = shape.span_bytes();
        let bytes = usize::try_from(span).ok().filter(|&bytes| bytes > 0)?;
        let (landed_tx, landed_rx) = std::sync::mpsc::channel();
        Some(Self {
            spans: vec![span],
            geometry: format!(
                "hy4-state {} {span} {}\n",
                shape.layers.len(),
                u8::from(shape.draft.is_some())
            ),
            shape,
            staging: Some(vec![0; bytes]),
            landed_tx,
            landed_rx,
            pending: None,
        })
    }

    fn layout(&self, row: usize) -> Option<ColdStateLayout> {
        Some(ColdStateLayout {
            boundary_tokens: u32::try_from(row).ok()?,
            spans: self.spans.clone(),
            geometry: self.geometry.clone(),
        })
    }
}

/// A replay state waiting for its write (Splash `WriteBehind` point).
#[derive(Clone, Copy)]
struct Point {
    id: EntryId,
    due: Instant,
}

/// A written state whose chain is not all on disk yet: it counts as durable
/// only once every object of `needs` is.
struct Owed {
    id: EntryId,
    key: ColdCacheKey,
    /// The KV blocks, then the MTP segments.
    needs: Vec<(ColdCacheKey, ColdGroup)>,
    /// Every object below it is on disk.
    landed: usize,
    bytes: u64,
    /// Since when nothing could add to the chain.
    stuck: Option<Instant>,
}

/// What [`Disk::write`] did with a point.
enum Written {
    /// On the state lane; the staging buffer is out.
    Started,
    /// Its object was on the tier already.
    OnTier,
    /// The lane's slot was taken: try again later.
    Busy,
    Refused,
}

/// What [`Disk::read_state`] gave back.
pub enum Read {
    State(PrefixState),
    /// The staging buffer did not come back in time: the object stays.
    Busy,
    /// It did not read back: the object is gone.
    Failed,
}

pub struct Disk {
    manager: Arc<ColdCacheManager>,
    closer: Arc<Closer>,
    fingerprint: ColdCacheFingerprint,
    layout: ColdCacheLayout,
    page: PageGeometry,
    pub delay: Duration,
    /// A lineage shorter than this writes nothing.
    pub minimum: usize,
    pub give_up: Duration,
    persistent: bool,
    /// A temporary tier copies until then.
    demand_until: Option<Instant>,
    /// The states memory dropped so far, as the model last said.
    seen_drops: u64,
    parker: Arc<Parker>,
    thread: Option<std::thread::JoinHandle<()>>,
    cursors: HashMap<SeqId, Cursor>,
    telemetry: Arc<Telemetry>,
    skip_once: bool,
    /// The next pass that may copy.
    step_after: Instant,
    lane: Lane,
    /// Replay states waiting for their write, in publication order.
    points: VecDeque<Point>,
    owed: Vec<Owed>,
    /// The deadline a dispose gave the waiting states; the drop closes by it.
    closing_by: Option<Instant>,
}

impl Disk {
    /// The states of `shape` over `tier`, blocks of the pool `page`
    /// describes, its parker started.
    pub(crate) fn open(
        tier: Arc<DiskTier>,
        telemetry: Arc<Telemetry>,
        shape: StateShape,
        page: PageGeometry,
    ) -> Result<Self> {
        let lane = Lane::new(shape).ok_or_else(|| bad("no state record shape"))?;
        let manager = Arc::clone(&tier.manager);
        let fingerprint = tier.state_fingerprint;
        let persistent = tier.persistent;
        let trimmed = manager.fit_quota();
        if trimmed > 0 {
            tracing::info!(
                "hy_v4_flash: took {trimmed} stored objects off the SSD cache to fit its limit"
            );
        }
        let parker = Arc::new(Parker::new());
        let wake = Arc::new(Wake::default());
        let thread = {
            let (parker, tier, telemetry, wake) = (
                Arc::clone(&parker),
                Arc::clone(&tier),
                Arc::clone(&telemetry),
                Arc::clone(&wake),
            );
            std::thread::Builder::new()
                .name("hy4-prefix-parker".to_string())
                .spawn(move || run(parker, tier, telemetry, wake))
                .map_err(|e| bad(e.to_string()))?
        };
        telemetry
            .disk_enabled
            .store(if persistent { 1 } else { 2 }, Ordering::Relaxed);
        {
            let wake = Arc::clone(&wake);
            manager.set_landing_wake(Some(Arc::new(move || wake.fire())));
        }
        let closer = Arc::new(Closer::new(
            Arc::clone(&parker),
            tier,
            Arc::clone(&telemetry),
            wake,
        ));
        Ok(Self {
            manager,
            closer,
            fingerprint,
            layout: block_layout(&page),
            page,
            delay: DELAY,
            minimum: MINIMUM_ROWS,
            give_up: GIVE_UP,
            persistent,
            demand_until: None,
            seen_drops: 0,
            parker,
            thread: Some(thread),
            cursors: HashMap::new(),
            telemetry,
            skip_once: false,
            step_after: Instant::now(),
            lane,
            points: VecDeque::new(),
            owed: Vec::new(),
            closing_by: None,
        })
    }

    /// The adapter restores the KV blocks of a state's chain from this
    /// store; its prefix lookups run only below a state's row. Refused when
    /// the pool's page is not this tier's.
    pub fn attach(&self, adapter: &mut PagedKVCacheAdapter) -> Result<()> {
        let pool = adapter.layer_kv_pool();
        let (key, value) = pool.block_blob_bytes().map_err(bad)?;
        let fits = pool.block_size() == self.layout.block_size
            && pool.num_layers() as u32 == self.layout.num_layers
            && pool.config().head_size == self.layout.head_size
            && format!("{:?}", pool.cache_dtype()) == self.layout.cache_dtype
            && key == self.layout.key_bytes_per_layer as u64
            && value == self.layout.value_bytes_per_layer as u64;
        if !fits {
            return Err(bad("the pool's page is not the stored blocks' page"));
        }
        adapter.set_cold_tier(ColdTierContext {
            manager: Arc::clone(&self.manager),
            fingerprint: self.fingerprint,
            backing: ColdContextBacking::KvOnly,
        });
        Ok(())
    }

    /// Weak: the waiting states, the queued copies and the tier go with this
    /// disk, not with the napi object.
    pub fn closer(&self) -> Weak<Closer> {
        Arc::downgrade(&self.closer)
    }

    pub(crate) fn tier_handle(&self) -> TierHandle {
        TierHandle::of(Some(&self.closer.tier))
    }

    pub fn page(&self) -> PageGeometry {
        self.page
    }

    pub fn shape(&self) -> &StateShape {
        &self.lane.shape
    }

    /// No point waits, no state write is in flight, no chain is owed.
    #[cfg(test)]
    pub fn idle(&self) -> bool {
        self.points.is_empty() && self.lane.pending.is_none() && self.owed.is_empty()
    }

    /// Whether copies go out now: always on a persistent tier, on a
    /// temporary one while memory asks for them.
    fn copying(&self) -> bool {
        self.persistent
            || self
                .demand_until
                .is_some_and(|until| Instant::now() < until)
    }

    /// Memory has dropped `total` stored states so far: a growth asks a
    /// temporary tier for copies for the next `DEMAND_WINDOW`.
    pub fn drops(&mut self, total: u64) {
        if total > self.seen_drops {
            self.seen_drops = total;
            self.demand_until = Some(Instant::now() + DEMAND_WINDOW);
        }
    }

    /// After a command reset the next admission reads nothing from disk.
    pub fn reset(&mut self) {
        self.cursors.clear();
        self.skip_once = true;
    }

    pub fn take_skip(&mut self) -> bool {
        std::mem::take(&mut self.skip_once)
    }

    /// Whether the next admission reads nothing from disk.
    pub fn skips(&self) -> bool {
        self.skip_once
    }

    fn remove_object(&self, key: &ColdCacheKey) {
        let manager = &self.manager;
        self.closer.tier.while_held(|| manager.remove(key));
    }

    fn sync_waiting(&self) {
        let waiting = self.points.len() + usize::from(self.lane.pending.is_some());
        self.telemetry
            .waiting
            .store(waiting as u64, Ordering::Relaxed);
    }

    /// `seq`'s lineage (up to its deepest point) keyed by `keys` and `hot`
    /// (the prefix cache's hashes) owes the disk its chain. A lineage that
    /// extends the cursor's keeps what it copied.
    pub fn start(
        &mut self,
        seq: SeqId,
        lineage: &[u32],
        keys: &[Vec<u64>],
        hot: &[u64],
        salt: u64,
    ) {
        if lineage.len() < self.minimum {
            self.cursors.remove(&seq);
            return;
        }
        let blocks = hot.len().min(keys.len()).min(lineage.len() / BLOCK_ROWS);
        let lineage = lineage[..blocks * BLOCK_ROWS].to_vec();
        let fp = self.fingerprint;
        let cold = chain(ColdGroup::Kv, fp, &lineage, keys, salt, blocks);
        let draft_chain = chain(ColdGroup::HybridDraft, fp, &lineage, keys, salt, blocks);
        let kept = self.cursors.remove(&seq).filter(|c| {
            !c.broken && c.cold.len() <= cold.len() && cold[..c.cold.len()] == c.cold[..]
        });
        let (registered, pending, done, segments) = kept.map_or((0, Vec::new(), 0, 0), |c| {
            (c.registered, c.pending, c.blocks, c.segments)
        });
        self.cursors.insert(
            seq,
            Cursor {
                hot: hot[..blocks].to_vec(),
                lineage,
                cold,
                draft_chain,
                registered,
                pending,
                blocks: done,
                segments,
                broken: false,
                closed: false,
            },
        );
    }

    /// `seq` published a state at `row`: its blocks below are in their
    /// source, its MTP segments are `draft`'s. A replay state (`replay`) of
    /// `minimum` rows or more waits for its write, `delay` from now.
    pub(crate) fn published(
        &mut self,
        store: &Store,
        seq: SeqId,
        row: usize,
        draft: Option<&DraftState>,
        replay: Option<EntryId>,
    ) {
        if let Some(cursor) = self.cursors.get_mut(&seq) {
            cursor.registered = cursor.registered.max(row);
            if let Some(draft) = draft
                && draft.segments.len() > cursor.pending.len()
            {
                cursor.pending = draft.segments.clone();
            }
        }
        let Some(id) = replay else {
            return;
        };
        if !self.closer.tier.is_open()
            || row < self.minimum
            || store.get(id).is_none_or(|entry| entry.disk().is_some())
            || self.points.iter().any(|point| point.id == id)
            || self.lane.pending.is_some_and(|(pending, _)| pending == id)
        {
            return;
        }
        if !self.copying() {
            // While copies wait for memory to ask, only the newest point of
            // a conversation stays (Splash `WriteBehind::published`).
            let points = std::mem::take(&mut self.points);
            for point in points {
                let gone = store
                    .get(point.id)
                    .is_none_or(|entry| entry.checkpoint() || store.superseded(point.id));
                if gone {
                    Telemetry::bump(&self.telemetry.unneeded);
                } else {
                    self.points.push_back(point);
                }
            }
        }
        self.points.push_back(Point {
            id,
            due: Instant::now() + self.delay,
        });
        self.sync_waiting();
    }

    /// `seq`'s rows past `len` were cut and may be written again: a cursor
    /// that still owes blocks from there on stops.
    pub fn cut(&mut self, seq: SeqId, len: usize) {
        if let Some(cursor) = self.cursors.get_mut(&seq)
            && cursor.registered > len
            && cursor.blocks < (cursor.registered / BLOCK_ROWS).min(cursor.cold.len())
        {
            cursor.broken = true;
        }
    }

    fn owes(&self, seq: SeqId) -> bool {
        self.cursors.get(&seq).is_some_and(|cursor| {
            !cursor.broken
                && (cursor.blocks < (cursor.registered / BLOCK_ROWS).min(cursor.cold.len())
                    || cursor.segments < cursor.pending.len())
        })
    }

    /// Copies out the blocks of `seq`'s chain its source holds and the MTP
    /// segments it published that are not on disk yet, within the host
    /// queue's bytes. Never waits for the writer.
    pub fn top_up(&mut self, seq: SeqId, source: &mut dyn Blocks) {
        if !self.closer.tier.is_open() || !self.copying() || !self.owes(seq) {
            return;
        }
        let mut bytes = self.parker.lock().item_bytes;
        if bytes >= QUEUE_BYTES {
            return;
        }
        let fp = self.fingerprint;
        let layout = self.layout.clone();
        let Some(cursor) = self.cursors.get_mut(&seq) else {
            return;
        };
        let target = (cursor.registered / BLOCK_ROWS).min(cursor.cold.len());
        let mut items = Vec::new();
        let mut begun = false;
        while cursor.blocks < target && bytes < QUEUE_BYTES {
            let i = cursor.blocks;
            if self.manager.contains(&cursor.cold[i]) {
                cursor.blocks += 1;
                continue;
            }
            if !begun {
                if !source.begin(&cursor.lineage) {
                    return;
                }
                begun = true;
            }
            let Some(layers) = source.block(i, cursor.hot[i]) else {
                cursor.broken = true;
                break;
            };
            let item = ColdObject::Block(ColdCacheBlock {
                key: cursor.cold[i],
                fingerprint: fp,
                tokens: cursor.lineage[i * BLOCK_ROWS..(i + 1) * BLOCK_ROWS].to_vec(),
                layout: layout.clone(),
                layers,
            });
            bytes += object_bytes(&item);
            items.push(item);
            cursor.blocks += 1;
        }
        while !cursor.broken && cursor.segments < cursor.pending.len() && bytes < QUEUE_BYTES {
            let k = cursor.segments;
            let Some(key) = segment_key(fp, &cursor.draft_chain, &cursor.lineage, k) else {
                cursor.broken = true;
                break;
            };
            if !self.manager.contains_in(&key, ColdGroup::HybridDraft) {
                let object =
                    encode_segment(&cursor.pending[k])
                        .ok()
                        .flatten()
                        .and_then(|payload| {
                            Some(ColdSidecar {
                                key,
                                fingerprint: fp,
                                layout: sidecar_layout(
                                    ColdGroup::HybridDraft,
                                    (k + 1) * BLOCK_ROWS,
                                    k + 1,
                                    payload.len(),
                                )?,
                                tensors: vec![payload],
                            })
                        });
                let Some(object) = object else {
                    cursor.broken = true;
                    break;
                };
                let item = ColdObject::Sidecar(object);
                bytes += object_bytes(&item);
                items.push(item);
            }
            cursor.segments += 1;
        }
        if items.is_empty() {
            return;
        }
        let mut slot = self.parker.lock();
        for item in items {
            slot.item_bytes += object_bytes(&item);
            slot.items.push_back(item);
            self.telemetry.owed.fetch_add(1, Ordering::Relaxed);
        }
        self.parker.wake.notify_all();
    }

    /// `seq`'s turn ended: its cursor goes once it owes nothing.
    pub fn close(&mut self, seq: SeqId) {
        if let Some(cursor) = self.cursors.get_mut(&seq) {
            cursor.closed = true;
        }
    }

    fn top_up_all(&mut self, source: &mut dyn Blocks) {
        let owing: Vec<SeqId> = self
            .cursors
            .keys()
            .copied()
            .filter(|&seq| self.owes(seq))
            .collect();
        for seq in owing {
            self.top_up(seq, source);
        }
        let open = self.closer.tier.is_open() && self.copying();
        let done: Vec<SeqId> = self
            .cursors
            .iter()
            .filter(|(_, cursor)| cursor.closed)
            .map(|(&seq, _)| seq)
            .filter(|&seq| !open || !self.owes(seq))
            .collect();
        for seq in done {
            self.cursors.remove(&seq);
        }
    }

    /// The model thread's disk work, from every drive pass and decode step:
    /// the state writes that landed, a close's flush when one asks, and at
    /// most every `STEP_TOP_UP` (at once after a landing) the owed chains,
    /// the due points and the block copies. True when a write landed.
    pub(crate) fn step(&mut self, store: &mut Store, source: &mut dyn Blocks) -> bool {
        let landed = self.take_landings(store);
        if let Some(deadline) = self.take_close() {
            self.flush_points(store, source, deadline);
            return landed;
        }
        let now = Instant::now();
        if landed || now >= self.step_after {
            self.pass(store, source, now);
        }
        landed
    }

    /// [`Self::step`] without the interval: a turn's end.
    pub(crate) fn step_now(&mut self, store: &mut Store, source: &mut dyn Blocks) {
        self.take_landings(store);
        if let Some(deadline) = self.take_close() {
            self.flush_points(store, source, deadline);
            return;
        }
        self.pass(store, source, Instant::now());
    }

    fn take_close(&mut self) -> Option<Instant> {
        if !self.closer.closing.swap(false, Ordering::AcqRel) {
            return None;
        }
        Some(self.parker.lock().closing.unwrap_or_else(Instant::now))
    }

    fn pass(&mut self, store: &mut Store, source: &mut dyn Blocks, now: Instant) {
        self.step_after = now + STEP_TOP_UP;
        self.take_evictions(store);
        self.judge_owed(store, now);
        self.due_points(store, now);
        self.top_up_all(source);
        self.store_plan();
    }

    /// The state objects the tier's quota took: an entry with a RAM copy
    /// loses its disk copy, one on disk only goes.
    fn take_evictions(&mut self, store: &mut Store) {
        let evicted = self.manager.take_evicted();
        if evicted.is_empty() {
            return;
        }
        let pending = self.lane.pending.map(|(_, key)| key);
        let hit: Vec<(EntryId, bool)> = store
            .entries()
            .filter(|entry| {
                entry.disk().is_some_and(|copy| {
                    copy.landed && Some(copy.key) != pending && evicted.contains(&copy.key)
                })
            })
            .map(|entry| (entry.id(), entry.ram().is_some()))
            .collect();
        for (id, in_memory) in hit {
            if in_memory {
                store.set_disk(id, None);
            } else {
                store.remove(id);
            }
        }
        self.owed.retain(|owed| !evicted.contains(&owed.key));
    }

    fn store_plan(&mut self) {
        let owing = !self.owed.is_empty()
            || (self.copying() && self.cursors.keys().any(|&seq| self.owes(seq)));
        let mut next_due = None;
        if self.lane.staging.is_some() && self.copying() {
            next_due = self.points.front().map(|point| point.due);
        }
        for owed in &self.owed {
            if let Some(stuck) = owed.stuck {
                let due = stuck + self.give_up;
                next_due = Some(next_due.map_or(due, |next: Instant| next.min(due)));
            }
        }
        let mut slot = self.parker.lock();
        slot.plan = super::persist_writer::Plan {
            epoch: slot.plan.epoch + 1,
            next_due,
            owing,
            next_step: Some(self.step_after),
        };
        self.parker.wake.notify_all();
        drop(slot);
        self.sync_waiting();
    }

    fn take_landings(&mut self, store: &mut Store) -> bool {
        let mut any = false;
        while let Ok(landing) = self.lane.landed_rx.try_recv() {
            self.settle_landing(store, landing);
            any = true;
        }
        if any {
            self.sync_waiting();
        }
        any
    }

    /// A state write ended: the buffer comes back; a landed copy is marked,
    /// one whose entry let go of it is removed, a failed one leaves no copy.
    fn settle_landing(&mut self, store: &mut Store, landing: Landed) {
        let Landed::State { key, ok, payload } = landing else {
            return;
        };
        self.lane.staging = Some(payload);
        let Some((id, pending)) = self.lane.pending else {
            return;
        };
        if pending != key {
            return;
        }
        self.lane.pending = None;
        let ours = store
            .get(id)
            .and_then(|entry| entry.disk())
            .is_some_and(|copy| copy.key == key);
        if ok {
            match store.disk_mut(id) {
                Some(copy) if ours => copy.landed = true,
                _ => self.remove_object(&key),
            }
        } else {
            if ours {
                store.set_disk(id, None);
            }
            self.owed.retain(|owed| owed.key != key);
            Telemetry::bump(&self.telemetry.refused);
        }
    }

    /// Walks every owed chain from its first object not on disk: a whole
    /// chain under a landed state makes it durable; one that nothing can add
    /// to any more for `give_up` drops its state; one whose entry let go of
    /// the copy removes the object.
    fn judge_owed(&mut self, store: &mut Store, now: Instant) {
        if self.owed.is_empty() {
            return;
        }
        let quiet = (!self.copying() || !self.cursors.keys().any(|&seq| self.owes(seq)))
            && self.parker.lock().item_bytes == 0;
        let pending = self.lane.pending.map(|(_, key)| key);
        let mut kept = Vec::with_capacity(self.owed.len());
        for mut owed in std::mem::take(&mut self.owed) {
            let copy = store
                .get(owed.id)
                .and_then(|entry| entry.disk())
                .filter(|copy| copy.key == owed.key)
                .copied();
            let Some(copy) = copy else {
                if pending != Some(owed.key) {
                    self.remove_object(&owed.key);
                }
                continue;
            };
            while let Some((key, group)) = owed.needs.get(owed.landed) {
                if !self.manager.contains_in(key, *group) {
                    break;
                }
                owed.landed += 1;
            }
            if owed.landed == owed.needs.len() {
                if copy.landed {
                    if let Some(copy) = store.disk_mut(owed.id) {
                        copy.chain_owed = false;
                    }
                    Telemetry::bump(&self.telemetry.durable);
                    self.telemetry
                        .state_bytes_written
                        .fetch_add(owed.bytes, Ordering::Relaxed);
                    continue;
                }
                owed.stuck = None;
                kept.push(owed);
                continue;
            }
            if !quiet {
                owed.stuck = None;
                kept.push(owed);
                continue;
            }
            let since = *owed.stuck.get_or_insert(now);
            if now.saturating_duration_since(since) >= self.give_up {
                self.remove_object(&owed.key);
                store.set_disk(owed.id, None);
                Telemetry::bump(&self.telemetry.refused);
                continue;
            }
            kept.push(owed);
        }
        self.owed = kept;
    }

    fn wanted(&self, store: &Store, id: EntryId) -> bool {
        let Some(entry) = store.get(id) else {
            Telemetry::bump(&self.telemetry.unneeded);
            return false;
        };
        if entry.checkpoint() || store.superseded(id) {
            Telemetry::bump(&self.telemetry.unneeded);
            return false;
        }
        entry.disk().is_none()
    }

    /// The due points in publication order, one write at a time: a point
    /// that waits for the buffer stalls the rest; a temporary tier skips a
    /// point in use and stops while memory asks for nothing.
    fn due_points(&mut self, store: &mut Store, now: Instant) {
        let mut at = 0;
        while let Some(&point) = self.points.get(at) {
            if point.due > now {
                break;
            }
            if !self.wanted(store, point.id) {
                self.points.remove(at);
                continue;
            }
            if !self.persistent {
                let in_use = store
                    .get(point.id)
                    .is_some_and(|entry| store.in_use(entry.mark()));
                if in_use {
                    at += 1;
                    continue;
                }
                if !self.copying() {
                    break;
                }
            }
            if self.lane.staging.is_none() {
                break;
            }
            match self.write(store, point.id) {
                Written::Started => {
                    self.points.remove(at);
                    break;
                }
                Written::OnTier => {
                    self.points.remove(at);
                }
                Written::Refused => {
                    Telemetry::bump(&self.telemetry.refused);
                    self.points.remove(at);
                }
                Written::Busy => break,
            }
        }
    }

    /// Encodes entry `id`'s state into the staging buffer and hands it to
    /// the state lane, its chain made the newest objects so that the room
    /// it takes comes from older ones.
    fn write(&mut self, store: &mut Store, id: EntryId) -> Written {
        if !self.closer.tier.is_open() {
            return Written::Refused;
        }
        let Some(entry) = store.get(id) else {
            return Written::Refused;
        };
        let Some(Held(state)) = entry.ram() else {
            return Written::Refused;
        };
        let row = entry.depth();
        if state.row != row
            || state.layers.len() != self.lane.shape.layers.len()
            || state.draft.is_some() != self.lane.shape.draft.is_some()
        {
            return Written::Refused;
        }
        let lineage = Lineage {
            tokens: entry.tokens(),
            extra_keys: entry.extra_keys(),
            salt: entry.salt(),
        };
        let keys = |group| lineage_keys(group, self.fingerprint, lineage, BLOCK_ROWS as u32);
        let Some(&key) = keys(ColdGroup::DFlash2State).last() else {
            return Written::Refused;
        };
        let draft_chain = keys(ColdGroup::HybridDraft);
        let mut needs: Vec<(ColdCacheKey, ColdGroup)> = keys(ColdGroup::Kv)
            .into_iter()
            .map(|key| (key, ColdGroup::Kv))
            .collect();
        let segments = state.draft.as_ref().map_or(0, |d| d.segments.len());
        for k in 0..segments {
            let Some(segment) = segment_key(self.fingerprint, &draft_chain, entry.tokens(), k)
            else {
                return Written::Refused;
            };
            needs.push((segment, ColdGroup::HybridDraft));
        }
        let label = label_of(entry);
        let Some(layout) = self.lane.layout(row) else {
            return Written::Refused;
        };
        let bytes = state_object_bytes(&layout, &label);
        let copy = |landed| DiskCopy {
            key,
            bytes,
            landed,
            window_begin: row as u32,
            chain_owed: true,
        };
        let owed = Owed {
            id,
            key,
            needs,
            landed: 0,
            bytes,
            stuck: None,
        };
        if self.manager.contains_in(&key, ColdGroup::DFlash2State) {
            store.set_disk(id, Some(copy(true)));
            self.owed.push(Owed { bytes: 0, ..owed });
            return Written::OnTier;
        }
        let Some(mut payload) = self.lane.staging.take() else {
            return Written::Busy;
        };
        if !matches!(encode_state_into(state, &mut payload), Ok(true)) {
            self.lane.staging = Some(payload);
            return Written::Refused;
        }
        let chain: Vec<ColdCacheKey> = owed.needs.iter().map(|(key, _)| *key).collect();
        self.manager.touch(&chain);
        let room = if self.persistent {
            MakeRoom::Evict
        } else {
            MakeRoom::Refuse
        };
        let object = ColdStateObject {
            key,
            fingerprint: self.fingerprint,
            layout,
            label,
            payload,
        };
        match self
            .manager
            .enqueue_state(object, room, self.lane.landed_tx.clone())
        {
            Ok(()) => {
                self.lane.pending = Some((id, key));
                store.set_disk(id, Some(copy(false)));
                self.owed.push(owed);
                self.parker.lock().measure = true;
                self.parker.wake.notify_all();
                Written::Started
            }
            Err(EnqueueRefused::Busy(object)) => {
                self.lane.staging = Some(object.payload);
                Written::Busy
            }
            Err(EnqueueRefused::Gone(object)) => {
                self.lane.staging = Some(object.payload);
                Written::Refused
            }
        }
    }

    /// A clean stop on a persistent tier: every waiting point, newest first,
    /// due or not, in use or not, is written, then every copy goes to the
    /// writer and the writer drains, until `deadline`. A temporary tier
    /// keeps nothing. False when a point still waits or a copy or write did
    /// not land by then.
    pub(crate) fn flush_points(
        &mut self,
        store: &mut Store,
        source: &mut dyn Blocks,
        deadline: Instant,
    ) -> bool {
        let mut complete = true;
        if self.persistent {
            loop {
                self.take_landings(store);
                self.judge_owed(store, Instant::now());
                self.top_up_all(source);
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    break;
                }
                if self.lane.staging.is_none() {
                    if let Ok(landing) = self
                        .lane
                        .landed_rx
                        .recv_timeout(left.min(Duration::from_millis(50)))
                    {
                        self.settle_landing(store, landing);
                    }
                    continue;
                }
                let Some(point) = self.points.back().copied() else {
                    break;
                };
                if !self.wanted(store, point.id) {
                    self.points.pop_back();
                    continue;
                }
                match self.write(store, point.id) {
                    Written::Started | Written::OnTier => {
                        self.points.pop_back();
                    }
                    Written::Refused => {
                        Telemetry::bump(&self.telemetry.refused);
                        self.points.pop_back();
                    }
                    Written::Busy => {
                        std::thread::sleep(left.min(Duration::from_millis(10)));
                    }
                }
            }
            let handed = self.parker.flush(deadline);
            let drained = self
                .manager
                .drain(deadline.saturating_duration_since(Instant::now()));
            self.take_landings(store);
            self.judge_owed(store, Instant::now());
            self.sync_waiting();
            complete = self.points.is_empty() && self.lane.pending.is_none() && handed && drained;
        }
        let mut slot = self.parker.lock();
        slot.points_flushed = Some(complete);
        self.parker.done.notify_all();
        complete
    }

    /// A dispose: the waiting states and the queued copies get `SLOT_WAIT`
    /// together, on this thread; the drop then closes the tier by the same
    /// deadline.
    pub(crate) fn dispose(&mut self, store: &mut Store, source: &mut dyn Blocks) {
        let by = Instant::now() + SLOT_WAIT;
        self.flush_points(store, source, by);
        self.closing_by = Some(by);
    }

    /// The MTP segment keys of the state of `tokens` (keyed by `keys` and
    /// `salt`) at `row`, when its KV blocks past the ones `cached` holds
    /// and, with the MTP layer, its segments are all on disk.
    pub fn chain_whole(
        &self,
        tokens: &[u32],
        keys: &[Vec<u64>],
        salt: u64,
        row: usize,
        cached: &dyn Fn(u64) -> bool,
    ) -> Option<Vec<ColdCacheKey>> {
        let n = row / BLOCK_ROWS;
        if n == 0 || !row.is_multiple_of(BLOCK_ROWS) || tokens.len() < row {
            return None;
        }
        let padded: Vec<Vec<u64>> = (0..n)
            .map(|i| keys.get(i).cloned().unwrap_or_default())
            .collect();
        let cold = chain(ColdGroup::Kv, self.fingerprint, tokens, &padded, salt, n);
        let hot = compute_paged_prefix_block_hashes(
            tokens,
            u32::try_from(row).ok()?,
            BLOCK_ROWS as u32,
            &padded,
            salt,
        )
        .unwrap_or_default();
        let hit = hot.iter().take_while(|&&h| cached(h)).count();
        let kv_whole = hot.len() == n
            && cold.len() == n
            && cold[hit.min(n)..]
                .iter()
                .all(|key| self.manager.contains(key));
        if !kv_whole {
            return None;
        }
        if self.lane.shape.draft.is_none() {
            return Some(Vec::new());
        }
        let draft_chain = chain(
            ColdGroup::HybridDraft,
            self.fingerprint,
            tokens,
            &padded,
            salt,
            n,
        );
        (0..segments_at(row))
            .map(|k| segment_key(self.fingerprint, &draft_chain, tokens, k))
            .collect::<Option<Vec<_>>>()
            .filter(|keys| {
                keys.iter()
                    .all(|key| self.manager.contains_in(key, ColdGroup::HybridDraft))
            })
    }

    /// Entry `id`, a state that did not come back, goes with its disk
    /// object.
    pub(crate) fn discard(&mut self, store: &mut Store, id: EntryId) {
        if let Some(copy) = store.get(id).and_then(|entry| entry.disk()).copied() {
            self.remove_object(&copy.key);
        }
        store.remove(id);
        Telemetry::bump(&self.telemetry.dropped_chain);
    }

    /// A persistent tier's restart: with an empty store, each state record
    /// of this load's shape whose own object, whole KV chain and, with the
    /// MTP layer, segments are on disk comes back as a state on disk only.
    /// How many came back.
    pub(crate) fn adopt(&mut self, store: &mut Store) -> u64 {
        if !self.persistent || store.entries().next().is_some() {
            return 0;
        }
        let tier = Arc::clone(&self.closer.tier);
        let (spans, geometry) = (self.lane.spans.clone(), self.lane.geometry.clone());
        let (mtp, fp, span) = (
            self.lane.shape.draft.is_some(),
            self.fingerprint,
            self.lane.spans.iter().sum::<u64>(),
        );
        let manager = Arc::clone(&self.manager);
        let adopted = adopt_states(store, &tier, &spans, &geometry, |label| {
            let row = label.tokens.len();
            if mtp {
                let lineage = Lineage {
                    tokens: &label.tokens,
                    extra_keys: &label.extra_keys,
                    salt: label.salt,
                };
                let draft_chain =
                    lineage_keys(ColdGroup::HybridDraft, fp, lineage, BLOCK_ROWS as u32);
                for k in 0..segments_at(row) {
                    let key = segment_key(fp, &draft_chain, &label.tokens, k)?;
                    if !manager.contains_in(&key, ColdGroup::HybridDraft) {
                        return None;
                    }
                }
            }
            Some((span, u32::try_from(row).ok()?))
        });
        self.telemetry.adopted.fetch_add(adopted, Ordering::Relaxed);
        if adopted > 0 {
            tracing::info!("hy_v4_flash: took back {adopted} stored states from the SSD cache");
        }
        adopted
    }

    /// The state object `key` at `row` read back through the staging buffer,
    /// its label's keys `extra_keys`, with its MTP `segment_keys`. `bound`
    /// caps every segment read. No buffer within `SLOT_WAIT` leaves the
    /// object; one that does not read back is removed.
    pub(crate) fn read_state(
        &mut self,
        store: &mut Store,
        key: ColdCacheKey,
        row: usize,
        extra_keys: &[Vec<u64>],
        segment_keys: &[ColdCacheKey],
        bound: u64,
    ) -> Read {
        let deadline = Instant::now() + SLOT_WAIT;
        while self.lane.staging.is_none() {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Read::Busy;
            }
            if let Ok(landing) = self
                .lane
                .landed_rx
                .recv_timeout(left.min(Duration::from_millis(50)))
            {
                self.settle_landing(store, landing);
                self.sync_waiting();
            }
        }
        let Some(mut staging) = self.lane.staging.take() else {
            return Read::Busy;
        };
        let read = (|| {
            let layout = self.lane.layout(row)?;
            self.manager
                .load_state(key, self.fingerprint, &layout, extra_keys, &mut staging)?;
            let shape = self.lane.shape.draft.as_ref();
            let mut segments = Vec::with_capacity(segment_keys.len());
            for (k, key) in segment_keys.iter().enumerate() {
                let object = self.manager.load_object(
                    *key,
                    self.fingerprint,
                    ColdGroup::HybridDraft,
                    bound,
                )?;
                if !sidecar_fits(&object.layout, (k + 1) * BLOCK_ROWS, k + 1) {
                    return None;
                }
                segments.push(Arc::new(decode_segment(object.tensors.first()?, shape?)?));
            }
            decode_state(&staging, &self.lane.shape, segments)
        })();
        self.lane.staging = Some(staging);
        match read {
            Some(state) if state.row == row => Read::State(state),
            _ => {
                self.remove_object(&key);
                Telemetry::bump(&self.telemetry.dropped_chain);
                Read::Failed
            }
        }
    }

    /// A state read back has landed in its sequence or cell.
    pub fn restored(&self) {
        Telemetry::bump(&self.telemetry.disk_restored);
    }

    /// The KV blocks of the first `row` rows of `tokens` (keyed by `keys`
    /// and `salt`), read back a run at a time. `None` when the chain does
    /// not cover `row`.
    pub fn block_runs<'a>(
        &'a self,
        tokens: &'a [u32],
        keys: &[Vec<u64>],
        salt: u64,
        row: usize,
    ) -> Option<BlockRuns<'a>> {
        let n = row / BLOCK_ROWS;
        let padded: Vec<Vec<u64>> = (0..n)
            .map(|i| keys.get(i).cloned().unwrap_or_default())
            .collect();
        let cold = chain(ColdGroup::Kv, self.fingerprint, tokens, &padded, salt, n);
        (cold.len() == n).then(|| BlockRuns {
            disk: self,
            tokens,
            cold,
            next: 0,
            failed: false,
            scratch: vec![(Vec::new(), Vec::new()); self.page.layers as usize],
        })
    }
}

/// A stored state's KV blocks, `FLAT_RUN` at a time: per indexed layer the
/// latent rows `[32·b, 2·half]` and the pooled keys `[b·index_rows, d]`,
/// f16. A restore holds one run, never the whole chain.
pub struct BlockRuns<'a> {
    disk: &'a Disk,
    tokens: &'a [u32],
    cold: Vec<ColdCacheKey>,
    next: usize,
    failed: bool,
    scratch: Vec<(Vec<u16>, Vec<u16>)>,
}

impl BlockRuns<'_> {
    /// A block did not read back.
    pub fn failed(&self) -> bool {
        self.failed
    }

    fn read(&mut self, first: usize, blocks: usize) -> Option<Vec<(MxArray, MxArray)>> {
        let d = self.disk;
        let layers = d.page.layers as usize;
        for (rows, keys) in &mut self.scratch {
            rows.clear();
            keys.clear();
        }
        let chain = self.cold.get(first..first + blocks)?;
        for (i, key) in chain.iter().enumerate() {
            let at = (first + i) * BLOCK_ROWS;
            let tokens = self.tokens.get(at..at + BLOCK_ROWS)?;
            let block = d.manager.load(*key, d.fingerprint)?;
            if block.layout != d.layout
                || block.tokens[..] != *tokens
                || block.layers.len() != layers
            {
                return None;
            }
            for ((rows, keys), layer) in self.scratch.iter_mut().zip(&block.layers) {
                let (r, k) = super::persist_codec::join_block(layer, &d.page)?;
                rows.extend_from_slice(&r);
                keys.extend_from_slice(&k);
            }
        }
        let lines = (blocks * BLOCK_ROWS) as i64;
        let width = 2 * i64::from(d.page.half);
        let groups = (blocks * d.page.index_rows as usize) as i64;
        let dim = i64::from(d.page.index_dim);
        self.scratch
            .iter()
            .map(|(rows, keys)| {
                Some((
                    MxArray::from_float16(rows, &[lines, width]).ok()?,
                    MxArray::from_float16(keys, &[groups, dim]).ok()?,
                ))
            })
            .collect()
    }
}

impl Iterator for BlockRuns<'_> {
    type Item = Vec<(MxArray, MxArray)>;

    fn next(&mut self) -> Option<Self::Item> {
        let left = self.cold.len().saturating_sub(self.next);
        if left == 0 || self.failed {
            return None;
        }
        let blocks = FLAT_RUN.min(left);
        let run = self.read(self.next, blocks);
        match run {
            Some(_) => self.next += blocks,
            None => self.failed = true,
        }
        run
    }
}

impl Drop for Disk {
    fn drop(&mut self) {
        self.closer.finish(
            self.closing_by
                .unwrap_or_else(|| Instant::now() + SLOT_WAIT),
        );
        {
            let mut slot = self.parker.lock();
            slot.stop = true;
            self.parker.wake.notify_all();
            self.parker.done.notify_all();
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// The sidecar layout of an MTP segment object: one U8 tensor of `bytes`
/// at `boundary`, segment index plus one.
fn sidecar_layout(
    group: ColdGroup,
    boundary: usize,
    count: usize,
    bytes: usize,
) -> Option<ColdSidecarLayout> {
    Some(ColdSidecarLayout {
        group,
        boundary_tokens: u32::try_from(boundary).ok()?,
        num_layers: 1,
        tensors_per_layer: 1,
        dtype: "u8".to_string(),
        dims: vec![
            super::persist_codec::FORMAT,
            u32::try_from(count).ok()?.max(1),
            BLOCK_ROWS as u32,
        ],
        bytes_per_tensor: bytes,
    })
}

fn sidecar_fits(found: &ColdSidecarLayout, boundary: usize, count: usize) -> bool {
    sidecar_layout(
        ColdGroup::HybridDraft,
        boundary,
        count,
        found.bytes_per_tensor,
    )
    .as_ref()
        == Some(found)
}

// ---------------------------------------------------------------------------
// Opening.
// ---------------------------------------------------------------------------

/// `source_file_identity_digest` of each GGUF shard of `paths`; `None` when
/// one cannot be read.
pub fn shard_digests(paths: &[PathBuf]) -> Option<Vec<String>> {
    paths
        .iter()
        .map(|p| {
            let path = std::fs::canonicalize(p).ok()?;
            let meta = std::fs::metadata(&path).ok()?;
            Some(crate::utils::gguf::source_file_identity_digest(
                &path, &meta,
            ))
        })
        .collect()
}

/// The disk tier of `options` for a load of `cfg` from `paths`, with the
/// MTP layer when `mtp`: persistent under the identity's namespace where
/// the switch asks for it and it opens, else temporary. `None` when it is
/// off or does not open: the states stay in memory.
pub(crate) fn open(
    options: &DiskTierOptions,
    paths: &[PathBuf],
    cfg: &HyV4FlashConfig,
    mtp: bool,
    telemetry: Arc<Telemetry>,
) -> Option<Disk> {
    let (page, shape) = match (PageGeometry::of(cfg), StateShape::of(cfg, mtp)) {
        (Ok(page), Ok(shape)) => (page, shape),
        (Err(e), _) | (_, Err(e)) => {
            tracing::warn!("hy_v4_flash: stored states stay in memory: {}", e.reason);
            return None;
        }
    };
    let tier = crate::disk_tier::open_disk_tier(
        options,
        || {
            let shards = shard_digests(paths)?;
            let text = identity_text(&Identity {
                shards: &shards,
                config: config_digest(cfg),
                page,
                window: cfg.window,
                mtp,
                gpu: unsafe { mlx_sys::mlx_gpu_architecture_gen() },
            });
            let (namespace, fp) = fingerprint(&text)?;
            Some(PersistentIdentity {
                namespace,
                state: fp,
                kv: fp,
            })
        },
        shape.span_bytes(),
    );
    let Some(tier) = tier else {
        tracing::warn!("hy_v4_flash: stored states stay in memory: the SSD cache did not open");
        return None;
    };
    match Disk::open(tier, telemetry, shape, page) {
        Ok(disk) => Some(disk),
        Err(e) => {
            tracing::warn!("hy_v4_flash: stored states stay in memory: {}", e.reason);
            None
        }
    }
}

// ---------------------------------------------------------------------------
// The flat mode.
// ---------------------------------------------------------------------------

/// RAM copies the flat mode keeps beside its ladder.
const FLAT_STATES: usize = 4;
/// The one sequence of the flat mode, as the disk names it.
const FLAT_SEQ: SeqId = 0;

/// A state on disk the next prefill of `prompt` reads back.
struct Found {
    id: EntryId,
    row: usize,
    prompt: Vec<u32>,
}

/// The flat turn's keys: how its prefill and decode write the blocks.
struct FlatTurn {
    keys: Vec<Vec<u64>>,
    nonce: u64,
    /// The row its prefill started at.
    from: usize,
}

/// The flat mode's stored states over its one sequence, and their disk:
/// the sequence's latent and pooled rows are the blocks' source; a prompt
/// deeper than the ladder reaches resumes from a stored state read back.
pub struct FlatTier {
    pub disk: Disk,
    states: States<PrefixState>,
    /// The MTP segments of the sequence's states (shared handles).
    segments: Vec<Weak<Segment>>,
    /// Keys of the sequence's whole blocks as its turns wrote them.
    line: Vec<Vec<u64>>,
    /// The salt of the turn about to verify.
    salt: Cell<u64>,
    kpool: usize,
    found: RefCell<Option<Found>>,
    turn: Option<FlatTurn>,
    /// States published since load, and read back into a sequence.
    published: u64,
    restored: u64,
}

impl FlatTier {
    /// The flat states over `disk`, a persistent tier's states taken back.
    pub fn new(disk: Disk, kpool: usize) -> Self {
        let mut tier = Self {
            disk,
            states: States::with_cap(FLAT_STATES),
            segments: Vec::new(),
            line: Vec::new(),
            salt: Cell::new(0),
            kpool,
            found: RefCell::new(None),
            turn: None,
            published: 0,
            restored: 0,
        };
        tier.disk.adopt(tier.states.store_mut());
        tier
    }

    /// The salt of the turn about to verify.
    pub fn set_salt(&self, salt: u64) {
        self.salt.set(salt);
    }

    /// No state is read back at the next prefill.
    pub fn clear_found(&self) {
        *self.found.borrow_mut() = None;
    }

    /// The row of the deepest durable state `prompt` (cold keys `pass`)
    /// resumes from past `floor`, whose blocks and segments are all on
    /// disk; the next prefill reads it back. 0 when there is none.
    pub fn find(&self, prompt: &[u32], pass: &[Vec<u64>], floor: usize) -> usize {
        *self.found.borrow_mut() = None;
        if self.disk.skips() {
            return 0;
        }
        let key = Key {
            tokens: prompt,
            keys: pass,
            salt: self.salt.get(),
        };
        let store = self.states.store();
        let mut found: Vec<(EntryId, usize, u64)> = store
            .entries()
            .filter(|e| durable(e) && e.depth() > floor && resumes(e, key))
            .map(|e| (e.id(), e.depth(), e.last_used()))
            .collect();
        found.sort_unstable_by_key(|e| std::cmp::Reverse((e.1, e.2)));
        for (id, row, _) in found {
            let Some(e) = store.get(id) else {
                continue;
            };
            let whole =
                self.disk
                    .chain_whole(prompt, e.extra_keys(), self.salt.get(), row, &|_| false);
            if whole.is_some() {
                *self.found.borrow_mut() = Some(Found {
                    id,
                    row,
                    prompt: prompt.to_vec(),
                });
                return row;
            }
        }
        0
    }

    /// The full prompt of the found state when `tail` is the prompt past its
    /// row.
    pub fn found_prompt(&self, tail: &[u32]) -> Option<(usize, Vec<u32>)> {
        let found = self.found.borrow();
        let f = found.as_ref()?;
        (f.prompt.len() == f.row + tail.len() && f.prompt[f.row..] == *tail)
            .then(|| (f.row, f.prompt.clone()))
    }

    /// Reads the found state back and `fill`s a new sequence from it and
    /// its blocks; `None` when it does not come back (a state or block that
    /// does not read back goes with its object).
    pub fn restore(
        &mut self,
        fill: impl FnOnce(usize, &PrefixState, &mut BlockRuns<'_>) -> Result<Lagged>,
    ) -> Option<(Lagged, Vec<Vec<u64>>)> {
        let Found { id, row, prompt } = self.found.borrow_mut().take()?;
        let (key, extra) = {
            let e = self.states.store().get(id)?;
            (e.disk()?.key, e.extra_keys().to_vec())
        };
        let salt = self.salt.get();
        let Some(segments) = self
            .disk
            .chain_whole(&prompt, &extra, salt, row, &|_| false)
        else {
            self.disk.discard(self.states.store_mut(), id);
            return None;
        };
        let state = match self.disk.read_state(
            self.states.store_mut(),
            key,
            row,
            &extra,
            &segments,
            SEGMENT_BOUND,
        ) {
            Read::State(s) => s,
            Read::Busy => return None,
            Read::Failed => {
                self.states.store_mut().remove(id);
                return None;
            }
        };
        let Some(mut runs) = self.disk.block_runs(&prompt, &extra, salt, row) else {
            self.disk.discard(self.states.store_mut(), id);
            return None;
        };
        let filled = fill(row, &state, &mut runs);
        let failed = runs.failed();
        let seq = match filled {
            Ok(seq) => seq,
            Err(e) => {
                tracing::warn!("hy_v4_flash stored state at {row}: {}", e.reason);
                if failed {
                    self.disk.discard(self.states.store_mut(), id);
                }
                return None;
            }
        };
        self.states.touch(id);
        self.disk.restored();
        self.segments = state.draft.as_ref().map_or_else(Vec::new, |d| {
            d.segments.iter().map(Arc::downgrade).collect()
        });
        let keys = (0..row / BLOCK_ROWS)
            .map(|i| extra.get(i).cloned().unwrap_or_default())
            .collect();
        Some((seq, keys))
    }

    /// A prefill of `prompt` from `from` starts: rows past `from` are
    /// written again (`line` the keys of the blocks below, a read back's
    /// or the sequence's own); `cold_end` when `from` is a cold chunk end.
    pub fn begin_turn(
        &mut self,
        prompt: &[u32],
        from: usize,
        line: Option<Vec<Vec<u64>>>,
        pass: &[Vec<u64>],
        cold_end: bool,
    ) {
        if let Some(line) = line {
            self.line = line;
        }
        self.line.truncate(from / BLOCK_ROWS);
        self.segments.truncate(segments_at(from));
        self.disk.cut(FLAT_SEQ, from);
        self.disk.take_skip();
        let nonce = rand::random::<u64>();
        let keys = prompt_keys(prompt, from, &self.line, pass, cold_end, nonce);
        self.turn = Some(FlatTurn { keys, nonce, from });
    }

    /// The sequence started over: nothing it held is kept.
    pub fn forget(&mut self) {
        self.line.clear();
        self.segments.clear();
        self.turn = None;
        *self.found.borrow_mut() = None;
        self.disk.cut(FLAT_SEQ, 0);
    }

    /// After a prefill: the prompt's state rows `rows` the ladder holds
    /// publish, and the disk owes their chain.
    pub fn publish_prompt(&mut self, seq: &Lagged, ladder: &Ladder<LagMark>, rows: &[StateRow]) {
        let Some(turn) = self.turn.as_ref() else {
            return;
        };
        let from = turn.from;
        let tokens = seq.tokens().to_vec();
        let keys = turn.keys.clone();
        let Some(last) = rows
            .iter()
            .rev()
            .map(|r| r.row)
            .find(|&r| r > from && ladder.snap(r).is_some())
        else {
            return;
        };
        self.start(&tokens, &keys, last);
        for r in rows.iter().filter(|r| r.row > from) {
            let Some(mark) = ladder.snap(r.row) else {
                continue;
            };
            self.publish(seq, mark, &tokens, &keys, r.kind);
        }
    }

    /// At a turn's end: the newest block row a decode step held publishes
    /// as the turn state under write-path keys, and the sequence's blocks
    /// keep those keys for its next turn.
    pub fn publish_turn(&mut self, seq: &Lagged) {
        let Some(turn) = self.turn.take() else {
            return;
        };
        let tokens = seq.tokens().to_vec();
        let keys = turn_keys(&tokens, &turn.keys, tokens.len(), turn.nonce);
        self.line = keys.clone();
        let Some(mark) = seq.held() else {
            return;
        };
        let row = mark.len();
        if row <= turn.from || row > turn_row(tokens.len()) {
            return;
        }
        self.start(&tokens, &keys, row);
        self.publish(seq, mark, &tokens, &keys, super::ladder::Kind::Turn);
    }

    fn start(&mut self, tokens: &[u32], keys: &[Vec<u64>], row: usize) {
        let blocks = row / BLOCK_ROWS;
        if blocks == 0 || keys.len() < blocks || tokens.len() < row {
            return;
        }
        let hot = compute_paged_prefix_block_hashes(
            tokens,
            row as u32,
            BLOCK_ROWS as u32,
            &keys[..blocks],
            self.salt.get(),
        )
        .unwrap_or_default();
        self.disk
            .start(FLAT_SEQ, &tokens[..row], keys, &hot, self.salt.get());
    }

    fn publish(
        &mut self,
        seq: &Lagged,
        mark: &LagMark,
        tokens: &[u32],
        keys: &[Vec<u64>],
        kind: super::ladder::Kind,
    ) {
        let row = mark.len();
        let blocks = row / BLOCK_ROWS;
        if row == 0 || !row.is_multiple_of(BLOCK_ROWS) || keys.len() < blocks {
            return;
        }
        let state = match state_of(mark, seq, self.kpool, &mut self.segments) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("hy_v4_flash state at {row}: {}", e.reason);
                return;
            }
        };
        let key = Key {
            tokens: &tokens[..row],
            keys: &keys[..blocks],
            salt: self.salt.get(),
        };
        let id = match self.states.publish(key, state, kind) {
            Ok(Some(id)) => id,
            Ok(None) => return,
            Err(e) => {
                tracing::warn!("hy_v4_flash state at {row}: {}", e.reason);
                return;
            }
        };
        self.published += 1;
        let replay = (kind != super::ladder::Kind::Checkpoint).then_some(id);
        self.disk.drops(self.states.dropped());
        let draft = self.states.get(id).and_then(|s| s.draft.as_ref());
        self.disk
            .published(self.states.store(), FLAT_SEQ, row, draft, replay);
        self.disk
            .top_up(FLAT_SEQ, &mut FlatBlocks::new(seq, self.disk.page()));
    }

    /// A stored state was read back into the sequence.
    pub fn restored(&mut self) {
        self.restored += 1;
    }

    /// `[states with a RAM copy, their bytes, published, restored]`.
    pub fn counts(&self) -> [u64; 4] {
        [
            self.states.ram_len() as u64,
            self.states.held_bytes(),
            self.published,
            self.restored,
        ]
    }

    /// The disk's pass over the sequence's rows.
    pub fn tick(&mut self, seq: &Lagged) -> bool {
        self.disk.drops(self.states.dropped());
        let page = self.disk.page();
        self.disk
            .step(self.states.store_mut(), &mut FlatBlocks::new(seq, page))
    }

    /// The turn is over: what the disk owes goes out now.
    pub fn end_turn(&mut self, seq: &Lagged) {
        self.disk.close(FLAT_SEQ);
        let page = self.disk.page();
        self.disk
            .step_now(self.states.store_mut(), &mut FlatBlocks::new(seq, page));
    }

    /// A command reset: the RAM copies go, durable states stay on disk,
    /// the next admission reads nothing back.
    pub fn reset(&mut self) {
        self.states.reset(true);
        self.disk.reset();
        self.forget();
    }

    /// A dispose: the waiting states and the copies get one bound.
    pub fn dispose(&mut self, seq: &Lagged) {
        let page = self.disk.page();
        self.disk
            .dispose(self.states.store_mut(), &mut FlatBlocks::new(seq, page));
    }
}

/// The state of `seq` at its ladder mark `mark`: the mark's rings and
/// tails, and with the MTP layer the draft's rows below it.
fn state_of(
    mark: &LagMark,
    seq: &Lagged,
    kpool: usize,
    segments: &mut Vec<Weak<Segment>>,
) -> Result<PrefixState> {
    let layers = mark.trunk().state_layers()?;
    let draft = match (mark.draft()?, seq.draft()) {
        (None, None) => None,
        (Some(dm), Some(d)) => {
            let seed = dm
                .seed()
                .ok_or_else(|| bad("a draft mark without its seed"))?
                .clone();
            Some(draft_state_at(
                d,
                dm.len(),
                kpool,
                segments,
                dm.tail().clone(),
                seed,
            )?)
        }
        _ => return Err(bad("a mark of another draft")),
    };
    Ok(PrefixState {
        row: mark.len(),
        layers,
        draft,
    })
}

/// The most bytes one MTP segment object may read.
pub const SEGMENT_BOUND: u64 = 4 << 20;
