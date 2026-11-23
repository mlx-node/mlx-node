//! The disk tier of the stored states: Hy4 adapter over the shared
//! Parker-thread [`crate::disk_tier::prefix_disk::PrefixDisk`] plus Hy4's
//! flat-mode block source (`FlatBlocks`), block reader (`BlockRuns`), and
//! flat tier (`FlatTier`).
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::cell::{Cell, RefCell};
use std::ops::{Deref, DerefMut};
use std::path::PathBuf;
use std::sync::{Arc, Weak};
#[cfg(test)]
use std::time::Instant;

use mlx_paged_attn::{
    ColdCacheFingerprint, ColdCacheKey, ColdCacheLayout, ColdGroup, ColdLayerBlock,
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
use super::persist_writer::{Closer, Telemetry};
use super::prefill::{BLOCK_ROWS, StateRow, turn_row};
use super::prefix_state::{
    DraftState, Held, Key, PrefixState, Segment, States, any_written, durable, prompt_keys,
    resumes, turn_keys,
};
use crate::array::MxArray;
#[cfg(test)]
use crate::disk_tier::binary_codec::draft_segment_key;
use crate::disk_tier::prefix_disk::{DiskRead, FlushPolicy, PrefixCodec, PrefixDisk};
use crate::disk_tier::state_disk::lineage_keys;
use crate::disk_tier::store::{EntryId, HybridStateStore, Lineage};
use crate::disk_tier::{DiskTier, DiskTierOptions, PersistentIdentity, TierHandle};
use crate::transformer::paged_kv_cache_adapter::{PagedKVCacheAdapter, SeqId};

pub use crate::disk_tier::prefix_disk::{Blocks, PoolBlocks};
#[cfg(test)]
pub use crate::disk_tier::prefix_disk::{MINIMUM_ROWS, QUEUE_BYTES};

/// Blocks a flat copy reads from the GPU, or a flat restore from disk, at
/// once.
pub const FLAT_RUN: usize = 32;
/// The identity text's first line; a change of any format below bumps it.
pub const TAG: &str = "hy4-prefix-v1";
/// The most bytes one MTP segment object may read.
pub const SEGMENT_BOUND: u64 = 4 << 20;

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
#[cfg(test)]
pub fn segment_key(
    fingerprint: ColdCacheFingerprint,
    draft_chain: &[ColdCacheKey],
    tokens: &[u32],
    k: usize,
) -> Option<ColdCacheKey> {
    draft_segment_key(fingerprint, draft_chain, tokens, BLOCK_ROWS, BLOCK_ROWS, k)
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
// Flat block source.
// ---------------------------------------------------------------------------

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
// Hy4 PrefixCodec and Disk wrapper over shared PrefixDisk.
// ---------------------------------------------------------------------------

pub(crate) struct Hy4Codec {
    shape: StateShape,
    page: PageGeometry,
    layout: ColdCacheLayout,
}

impl PrefixCodec for Hy4Codec {
    type StoreState = Held<PrefixState>;
    type DecodedState = PrefixState;
    type Segment = Arc<Segment>;

    const BLOCK_ROWS: usize = BLOCK_ROWS;
    const THREAD_NAME: &'static str = "hy4-prefix-parker";
    const LOG_PREFIX: &'static str = "hy_v4_flash";
    const EXTEND_CURSOR: bool = true;
    const FLUSH_POLICY: FlushPolicy = FlushPolicy::DrainCursorsStrict;
    const COUNT_ON_TIER_BYTES: bool = false;

    fn spans(&self) -> Vec<u64> {
        vec![self.shape.span_bytes()]
    }

    fn geometry(&self) -> String {
        format!(
            "hy4-state {} {} {}\n",
            self.shape.layers.len(),
            self.shape.span_bytes(),
            u8::from(self.shape.draft.is_some())
        )
    }

    fn block_layout(&self) -> Option<ColdCacheLayout> {
        Some(self.layout.clone())
    }

    fn has_draft(&self) -> bool {
        self.shape.draft.is_some()
    }

    fn segment_stride(&self) -> usize {
        BLOCK_ROWS
    }

    fn segment_format(&self) -> u32 {
        super::persist_codec::FORMAT
    }

    fn state_matches_lane(&self, Held(state): &Held<PrefixState>, row: usize) -> bool {
        state.row == row
            && state.layers.len() == self.shape.layers.len()
            && state.draft.is_some() == self.shape.draft.is_some()
    }

    fn state_segment_count(&self, Held(state): &Held<PrefixState>) -> usize {
        state.draft.as_ref().map_or(0, |d| d.segments.len())
    }

    fn expected_segment_count(&self, row: usize, _stride: usize) -> usize {
        segments_at(row)
    }

    fn encode_state_into(&self, Held(state): &Held<PrefixState>, into: &mut [u8]) -> Result<bool> {
        encode_state_into(state, into)
    }

    fn decode_state(
        &self,
        bytes: &[u8],
        row: usize,
        segments: Vec<Arc<Segment>>,
    ) -> Option<PrefixState> {
        decode_state(bytes, &self.shape, segments).filter(|state| state.row == row)
    }

    fn encode_segment(&self, _stride: usize, segment: &Arc<Segment>) -> Result<Option<Vec<u8>>> {
        encode_segment(segment)
    }

    fn decode_segment(&self, bytes: &[u8]) -> Option<Arc<Segment>> {
        Some(Arc::new(decode_segment(bytes, self.shape.draft.as_ref()?)?))
    }

    /// Cold prompt-end replay states (`!any_written`) are never superseded by
    /// deeper turn states: a turn state cannot resume the original prompt after
    /// a restart when decode finishes faster than `DELAY`.
    fn is_superseded(&self, store: &Store, id: EntryId) -> bool {
        let Some(entry) = store.get(id) else {
            return false;
        };
        if !any_written(entry.extra_keys()) {
            return false;
        }
        store.superseded(id)
    }
}

/// What [`Disk::read_state`] gave back.
pub enum Read {
    State(PrefixState),
    /// The staging buffer did not come back in time: the object stays.
    Busy,
    /// It did not read back: the object is gone.
    Failed,
}

pub(crate) struct Disk(PrefixDisk<Hy4Codec>);

impl Deref for Disk {
    type Target = PrefixDisk<Hy4Codec>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Disk {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
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
        let codec = Hy4Codec {
            shape,
            layout: block_layout(&page),
            page,
        };
        PrefixDisk::open(tier, telemetry, codec).map(Self)
    }

    /// The adapter restores the KV blocks of a state's chain from this
    /// store; its prefix lookups run only below a state's row. Refused when
    /// the pool's page is not this tier's.
    pub fn attach(&self, adapter: &mut PagedKVCacheAdapter) -> Result<()> {
        let pool = adapter.layer_kv_pool();
        let (key, value) = pool.block_blob_bytes().map_err(bad)?;
        let layout = &self.0.codec.layout;
        let fits = pool.block_size() == layout.block_size
            && pool.num_layers() as u32 == layout.num_layers
            && pool.config().head_size == layout.head_size
            && format!("{:?}", pool.cache_dtype()) == layout.cache_dtype
            && key == layout.key_bytes_per_layer as u64
            && value == layout.value_bytes_per_layer as u64;
        if !fits {
            return Err(bad("the pool's page is not the stored blocks' page"));
        }
        self.0.attach_kv_only(adapter);
        Ok(())
    }

    /// Weak: the waiting states, the queued copies and the tier go with this
    /// disk, not with the napi object.
    pub fn closer(&self) -> Weak<Closer> {
        self.0.closer()
    }

    pub(crate) fn tier_handle(&self) -> TierHandle {
        self.0.tier_handle()
    }

    pub fn page(&self) -> PageGeometry {
        self.0.codec.page
    }

    /// Memory has dropped `total` stored states so far: a growth asks a
    /// temporary tier for copies for the next `DEMAND_WINDOW`.
    pub fn drops(&mut self, total: u64) {
        self.0.drops(total);
    }

    /// After a command reset the next admission reads nothing from disk.
    pub fn reset(&mut self) {
        self.0.reset();
    }

    pub fn take_skip(&mut self) -> bool {
        self.0.take_skip()
    }

    /// Whether the next admission reads nothing from disk.
    pub fn skips(&self) -> bool {
        self.0.skips()
    }

    pub fn start(
        &mut self,
        seq: SeqId,
        lineage: &[u32],
        keys: &[Vec<u64>],
        hot: &[u64],
        salt: u64,
    ) {
        self.0.start(seq, lineage, keys, hot, salt, BLOCK_ROWS);
    }

    pub(crate) fn published(
        &mut self,
        store: &Store,
        seq: SeqId,
        row: usize,
        draft: Option<&DraftState>,
        replay: Option<EntryId>,
    ) {
        self.0.published(
            store,
            seq,
            row,
            draft.map(|d| d.segments.as_slice()),
            replay,
        );
    }

    pub fn cut(&mut self, seq: SeqId, len: usize) {
        self.0.cut(seq, len);
    }

    pub fn top_up(&mut self, seq: SeqId, source: &mut dyn Blocks) {
        self.0.top_up(seq, source);
    }

    pub fn close(&mut self, seq: SeqId) {
        self.0.close(seq);
    }

    pub(crate) fn step(&mut self, store: &mut Store, source: &mut dyn Blocks) -> bool {
        self.0.step(store, source)
    }

    pub(crate) fn step_now(&mut self, store: &mut Store, source: &mut dyn Blocks) {
        self.0.step_now(store, source);
    }

    #[cfg(test)]
    pub(crate) fn flush_points(
        &mut self,
        store: &mut Store,
        source: &mut dyn Blocks,
        deadline: Instant,
    ) -> bool {
        self.0.flush_points(store, source, deadline)
    }

    pub(crate) fn dispose(&mut self, store: &mut Store, source: &mut dyn Blocks) {
        self.0.dispose(store, source);
    }

    pub fn chain_whole(
        &self,
        tokens: &[u32],
        keys: &[Vec<u64>],
        salt: u64,
        row: usize,
        cached: &dyn Fn(u64) -> bool,
    ) -> Option<Vec<ColdCacheKey>> {
        let draft_window = self.0.codec.has_draft().then_some(BLOCK_ROWS);
        self.0
            .chain_whole(tokens, keys, salt, row, true, cached, draft_window)
    }

    pub(crate) fn discard(&mut self, store: &mut Store, id: EntryId) {
        self.0.discard(store, id);
    }

    pub(crate) fn adopt(&mut self, store: &mut Store) -> u64 {
        let span = self.0.codec.shape.span_bytes();
        self.0.adopt(store, |_| span)
    }

    pub(crate) fn read_state(
        &mut self,
        store: &mut Store,
        key: ColdCacheKey,
        row: usize,
        extra_keys: &[Vec<u64>],
        segment_keys: &[ColdCacheKey],
        bound: u64,
    ) -> Read {
        match self
            .0
            .read_state(store, key, row, extra_keys, segment_keys, bound)
        {
            DiskRead::State(state) => Read::State(state),
            DiskRead::Busy => Read::Busy,
            DiskRead::Failed => Read::Failed,
        }
    }

    /// A state read back has landed in its sequence or cell.
    pub fn restored(&self) {
        self.0.restored();
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
        let cold = chain(ColdGroup::Kv, self.0.fingerprint, tokens, &padded, salt, n);
        (cold.len() == n).then(|| BlockRuns {
            disk: self,
            tokens,
            cold,
            next: 0,
            failed: false,
            scratch: vec![(Vec::new(), Vec::new()); self.0.codec.page.layers as usize],
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
        let d = &self.disk.0;
        let page = d.codec.page;
        let layers = page.layers as usize;
        for (rows, keys) in &mut self.scratch {
            rows.clear();
            keys.clear();
        }
        let chain = self.cold.get(first..first + blocks)?;
        for (i, key) in chain.iter().enumerate() {
            let at = (first + i) * BLOCK_ROWS;
            let tokens = self.tokens.get(at..at + BLOCK_ROWS)?;
            let block = d.manager.load(*key, d.fingerprint)?;
            if block.layout != d.codec.layout
                || block.tokens[..] != *tokens
                || block.layers.len() != layers
            {
                return None;
            }
            for ((rows, keys), layer) in self.scratch.iter_mut().zip(&block.layers) {
                let (r, k) = super::persist_codec::join_block(layer, &page)?;
                rows.extend_from_slice(&r);
                keys.extend_from_slice(&k);
            }
        }
        let lines = (blocks * BLOCK_ROWS) as i64;
        let width = 2 * i64::from(page.half);
        let groups = (blocks * page.index_rows as usize) as i64;
        let dim = i64::from(page.index_dim);
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
        let hot = mlx_paged_attn::chain_hashes_per_block(
            &tokens[..row],
            BLOCK_ROWS as u32,
            &keys[..blocks],
            self.salt.get(),
        );
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
