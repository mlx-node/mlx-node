//! The disk tier of the warm prefix states: Qwen4 adapter over the shared
//! Parker-thread [`crate::disk_tier::prefix_disk::PrefixDisk`] and
//! [`crate::disk_tier::binary_codec`].

use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
#[cfg(test)]
use std::time::Instant;

#[cfg(test)]
use mlx_paged_attn::ColdGroup;
use mlx_paged_attn::{ColdCacheFingerprint, ColdCacheKey};
use napi::Result;

use super::prefix_state::{BLOCK_ROWS, Cell, DraftCell, LayerCell, Segment};
use crate::array::MxArray;
use crate::disk_tier::DiskTier;
use crate::disk_tier::binary_codec::{Reader, Sink, Span, Writer};
use crate::disk_tier::prefix_disk::{
    DiskRead, DiskTelemetry, FlushPolicy, PoolBlocks, PrefixCodec, PrefixDisk,
};
use crate::disk_tier::store::{EntryId, HybridStateStore};
use crate::transformer::paged_kv_cache_adapter::{PagedKVCacheAdapter, SeqId};

pub(super) use crate::disk_tier::prefix_disk::{Closer, QUEUE_BYTES};
#[cfg(test)]
pub(super) use crate::disk_tier::prefix_disk::{DELAY, MINIMUM_ROWS};

/// Payload format of a state and a segment object.
const FORMAT: u32 = 3;
const MAGIC: &[u8; 4] = b"Q4PS";

/// Counters of the stored states and of their disk tier, read by
/// `residencyInfo`.
#[derive(Default)]
pub(super) struct Telemetry {
    pub(super) states: AtomicU64,
    pub(super) state_bytes: AtomicU64,
    /// The store's counters at its last sync.
    counters: Mutex<super::prefix_state::Counters>,
    /// Shared Parker-thread disk counters.
    disk: Arc<DiskTelemetry>,
}

impl Deref for Telemetry {
    type Target = DiskTelemetry;

    fn deref(&self) -> &Self::Target {
        &self.disk
    }
}

impl Telemetry {
    pub(super) fn sync(&self, states: &super::prefix_state::States) {
        let bytes = states.store.bytes();
        let in_memory = states
            .store
            .entries()
            .filter(|entry| entry.ram().is_some())
            .count();
        self.states.store(in_memory as u64, Ordering::Relaxed);
        self.state_bytes.store(bytes.state_bytes, Ordering::Relaxed);
        *self
            .counters
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = states.counters;
    }

    /// The store's counters at its last sync.
    pub(super) fn counters(&self) -> super::prefix_state::Counters {
        *self
            .counters
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The tier as a turn's row reports it: 0 off, 1 persistent, 2
    /// temporary; off once `tier` takes no write any more.
    pub(super) fn mode(&self, tier: &crate::disk_tier::TierHandle) -> u64 {
        self.disk.mode(tier)
    }
}

/// The cold keys of the first `blocks` whole blocks of `tokens` in `group`,
/// block `i` keyed by `keys[i]` (the pass keys).
#[cfg(test)]
pub(super) fn chain(
    group: ColdGroup,
    fingerprint: ColdCacheFingerprint,
    tokens: &[u32],
    keys: &[Vec<u64>],
    salt: u64,
    blocks: usize,
) -> Vec<ColdCacheKey> {
    let mut out = Vec::with_capacity(blocks);
    let mut parent = None;
    for (i, extra) in keys.iter().enumerate().take(blocks) {
        let Some(toks) = tokens.get(i * BLOCK_ROWS..(i + 1) * BLOCK_ROWS) else {
            break;
        };
        let key = ColdCacheKey::chain(group, fingerprint, parent, toks, extra, salt, i);
        out.push(key);
        parent = Some(key);
    }
    out
}

// ---------------------------------------------------------------------------
// Codec: header (magic, format, grid, layers, draft flag, each layer's
// pooled indexer blocks) + shared binary_codec arrays (BF16/F16/F32).
// ---------------------------------------------------------------------------

fn encode_to<S: Sink>(
    out: &mut Writer<S>,
    grid: usize,
    index_blocks: &[usize],
    draft: bool,
    arrays: &[Option<&MxArray>],
) -> Result<bool> {
    out.put(MAGIC);
    out.u32(FORMAT);
    out.u64(grid as u64);
    out.u32(index_blocks.len() as u32);
    out.u8(u8::from(draft));
    for &blocks in index_blocks {
        out.u64(blocks as u64);
    }
    out.arrays(arrays, true)
}

fn encode(
    grid: usize,
    index_blocks: &[usize],
    draft: bool,
    arrays: &[Option<&MxArray>],
) -> Result<Option<Vec<u8>>> {
    let mut out = Writer(Vec::new());
    let encoded = encode_to(&mut out, grid, index_blocks, draft, arrays)?;
    Ok(encoded.then(|| out.into_inner()))
}

type Decoded = (usize, Vec<usize>, bool, Vec<Option<MxArray>>);

fn decode(bytes: &[u8], arrays: usize, padded: bool) -> Option<Decoded> {
    let mut reader = Reader::new(bytes);
    if reader.take(4)? != MAGIC || reader.u32()? != FORMAT {
        return None;
    }
    let grid = reader.usize()?;
    let layers = reader.u32()? as usize;
    let draft = match reader.u8()? {
        0 => false,
        1 => true,
        _ => return None,
    };
    if layers > bytes.len() / 8 {
        return None;
    }
    let index_blocks = (0..layers)
        .map(|_| reader.usize())
        .collect::<Option<Vec<_>>>()?;
    let out = reader.arrays(arrays, true)?;
    (padded || reader.at_end()).then_some((grid, index_blocks, draft, out))
}

fn cell_arrays(cell: &Cell) -> (Vec<Option<&MxArray>>, Vec<usize>) {
    let mut arrays: Vec<Option<&MxArray>> = Vec::new();
    for layer in &cell.layers {
        arrays.extend([
            layer.conv.as_ref(),
            layer.recurrent.as_ref(),
            layer.ple_conv.as_ref(),
        ]);
    }
    if let Some(draft) = &cell.draft {
        arrays.push(draft.waiting.as_ref());
        arrays.push(Some(&draft.seed));
    }
    let blocks = cell.layers.iter().map(|layer| layer.index_blocks).collect();
    (arrays, blocks)
}

/// A state's bytes written into `into`, zeros after them to its end, so a
/// state record is one fixed span whatever its row and a reused buffer
/// carries nothing of the last state. False when they do not fit or a dtype
/// has no code.
pub(super) fn encode_cell_into(cell: &Cell, into: &mut [u8]) -> Result<bool> {
    let (arrays, blocks) = cell_arrays(cell);
    let mut out = Writer(Span::new(into));
    if !encode_to(&mut out, cell.grid, &blocks, cell.draft.is_some(), &arrays)? {
        return Ok(false);
    }
    Ok(out.into_inner().finish_zeroed())
}

/// The bytes a state's encoding takes, before any padding.
#[cfg(test)]
pub(super) fn encoded_len(cell: &Cell) -> usize {
    let (arrays, blocks) = cell_arrays(cell);
    encode(cell.grid, &blocks, cell.draft.is_some(), &arrays)
        .unwrap()
        .unwrap()
        .len()
}

/// The state of [`encode_cell_into`] with `segments` as its draft passes;
/// `None` when the bytes are not a state of `layers` layers.
pub(super) fn decode_cell(
    bytes: &[u8],
    layers: usize,
    draft: bool,
    segments: Vec<Segment>,
) -> Option<Cell> {
    let arrays = layers * 3 + if draft { 2 } else { 0 };
    let (grid, blocks, has_draft, mut arrays) = decode(bytes, arrays, true)?;
    if blocks.len() != layers || has_draft != draft {
        return None;
    }
    let draft = if draft {
        let seed = arrays.pop()??;
        let waiting = arrays.pop()?;
        Some(DraftCell {
            segments,
            waiting,
            seed,
        })
    } else {
        None
    };
    let mut arrays = arrays.into_iter();
    let layers = blocks
        .into_iter()
        .map(|index_blocks| {
            Some(LayerCell {
                conv: arrays.next()?,
                recurrent: arrays.next()?,
                ple_conv: arrays.next()?,
                index_blocks,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    Some(Cell {
        grid,
        layers,
        draft,
    })
}

fn encode_segment(grid: usize, segment: &Segment) -> Result<Option<Vec<u8>>> {
    encode(
        grid,
        &[],
        false,
        &[
            Some(&segment.keys),
            Some(&segment.values),
            Some(&segment.index),
        ],
    )
}

fn decode_segment(bytes: &[u8]) -> Option<(usize, Segment)> {
    let (grid, blocks, draft, arrays) = decode(bytes, 3, false)?;
    if !blocks.is_empty() || draft {
        return None;
    }
    let mut arrays = arrays.into_iter();
    let segment = Segment {
        keys: arrays.next()??,
        values: arrays.next()??,
        index: arrays.next()??,
    };
    let ready = [&segment.keys, &segment.values, &segment.index];
    MxArray::eval_arrays(&ready).ok()?;
    Some((grid, segment))
}

// ---------------------------------------------------------------------------
// Qwen4 PrefixCodec and Disk wrapper over shared PrefixDisk.
// ---------------------------------------------------------------------------

type Store = HybridStateStore<Cell>;

/// The shape of a load's state records.
pub(super) struct LaneShape {
    /// One state's payload: every state record is this long.
    pub(super) span: u64,
    pub(super) layers: usize,
    pub(super) grid: usize,
    pub(super) has_mtp: bool,
}

pub(super) struct Qwen4Codec {
    span: u64,
    layers: usize,
    grid: usize,
    has_mtp: bool,
}

impl PrefixCodec for Qwen4Codec {
    type StoreState = Cell;
    type DecodedState = Cell;
    type Segment = Segment;

    const BLOCK_ROWS: usize = BLOCK_ROWS;
    const THREAD_NAME: &'static str = "qwen4-prefix-parker";
    const LOG_PREFIX: &'static str = "qwen4_exp";
    const EXTEND_CURSOR: bool = false;
    const FLUSH_POLICY: FlushPolicy = FlushPolicy::PointsOnly;
    const COUNT_ON_TIER_BYTES: bool = true;

    fn spans(&self) -> Vec<u64> {
        vec![self.span]
    }

    fn geometry(&self) -> String {
        format!(
            "qwen4-cell {} {} {} {}\n",
            self.layers,
            self.grid,
            self.span,
            u8::from(self.has_mtp)
        )
    }

    fn has_draft(&self) -> bool {
        self.has_mtp
    }

    fn segment_stride(&self) -> usize {
        self.grid
    }

    fn segment_format(&self) -> u32 {
        FORMAT
    }

    fn state_matches_lane(&self, cell: &Cell, _row: usize) -> bool {
        cell.grid == self.grid
            && cell.layers.len() == self.layers
            && cell.draft.is_some() == self.has_mtp
    }

    fn state_segment_count(&self, cell: &Cell) -> usize {
        cell.draft.as_ref().map_or(0, |draft| draft.segments.len())
    }

    fn expected_segment_count(&self, row: usize, stride: usize) -> usize {
        if stride == 0 {
            0
        } else {
            super::memory::cached_draft_rows(row as u64, stride as u64) as usize / stride
        }
    }

    fn encode_state_into(&self, cell: &Cell, into: &mut [u8]) -> Result<bool> {
        encode_cell_into(cell, into)
    }

    fn decode_state(&self, bytes: &[u8], _row: usize, segments: Vec<Segment>) -> Option<Cell> {
        let cell = decode_cell(bytes, self.layers, self.has_mtp, segments)?;
        let arrays: Vec<&MxArray> = cell.arrays().collect();
        if !arrays.is_empty() {
            MxArray::eval_arrays(&arrays).ok()?;
        }
        (cell.grid == self.grid).then_some(cell)
    }

    fn encode_segment(&self, stride: usize, segment: &Segment) -> Result<Option<Vec<u8>>> {
        encode_segment(stride, segment)
    }

    fn decode_segment(&self, bytes: &[u8]) -> Option<Segment> {
        let (window, segment) = decode_segment(bytes)?;
        (window == self.grid).then_some(segment)
    }
}

/// What [`Disk::read_state`] gave back.
pub(super) enum Read {
    Cell(Cell),
    /// The staging buffer did not come back in time: the object stays.
    Busy,
    /// It did not read back: the object is gone.
    Failed,
}

pub(super) struct Disk(PrefixDisk<Qwen4Codec>);

impl Deref for Disk {
    type Target = PrefixDisk<Qwen4Codec>;

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
    /// The states over `tier`, its parker started.
    pub(super) fn open(
        tier: Arc<DiskTier>,
        telemetry: Arc<Telemetry>,
        shape: &LaneShape,
    ) -> Result<Self> {
        let codec = Qwen4Codec {
            span: shape.span,
            layers: shape.layers,
            grid: shape.grid,
            has_mtp: shape.has_mtp,
        };
        PrefixDisk::open(tier, Arc::clone(&telemetry.disk), codec).map(Self)
    }

    /// The adapter restores the KV blocks of a state's chain from this
    /// store; its prefix lookups run only below a state's row.
    pub(super) fn attach(&self, adapter: &mut PagedKVCacheAdapter) {
        self.0.attach_kv_only(adapter);
    }

    /// Weak: the waiting states, the queued copies and the tier go with this
    /// disk, not with the napi object, whose memory V8 does not see.
    pub(super) fn closer(&self) -> Weak<Closer> {
        self.0.closer()
    }

    /// The napi object's handle on the tier.
    pub(super) fn tier_handle(&self) -> crate::disk_tier::TierHandle {
        self.0.tier_handle()
    }

    /// Memory has dropped `total` stored states so far: a growth asks a
    /// temporary tier for copies for the next `DEMAND_WINDOW`.
    pub(super) fn drops(&mut self, total: u64) {
        self.0.drops(total);
    }

    /// After a command reset the next prime reads nothing from disk: the
    /// purged prefix is prefilled again.
    pub(super) fn reset(&mut self) {
        self.0.reset();
    }

    pub(super) fn take_skip(&mut self) -> bool {
        self.0.take_skip()
    }

    pub(super) fn start(
        &mut self,
        seq: SeqId,
        lineage: &[u32],
        keys: &[Vec<u64>],
        hot: &[u64],
        salt: u64,
        window: usize,
    ) {
        self.0.start(seq, lineage, keys, hot, salt, window);
    }

    pub(super) fn published(
        &mut self,
        store: &Store,
        seq: SeqId,
        row: usize,
        draft: Option<&DraftCell>,
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

    pub(super) fn top_up(&mut self, seq: SeqId, adapter: &mut PagedKVCacheAdapter) {
        self.0.top_up(seq, &mut PoolBlocks(adapter));
    }

    pub(super) fn close(&mut self, seq: SeqId) {
        self.0.close(seq);
    }

    pub(super) fn step(&mut self, store: &mut Store, adapter: &mut PagedKVCacheAdapter) -> bool {
        self.0.step(store, &mut PoolBlocks(adapter))
    }

    pub(super) fn step_now(&mut self, store: &mut Store, adapter: &mut PagedKVCacheAdapter) {
        self.0.step_now(store, &mut PoolBlocks(adapter));
    }

    #[cfg(test)]
    pub(super) fn flush_points(
        &mut self,
        store: &mut Store,
        adapter: &mut PagedKVCacheAdapter,
        deadline: Instant,
    ) -> bool {
        self.0
            .flush_points(store, &mut PoolBlocks(adapter), deadline)
    }

    pub(super) fn dispose(&mut self, store: &mut Store, adapter: &mut PagedKVCacheAdapter) {
        self.0.dispose(store, &mut PoolBlocks(adapter));
    }

    pub(super) fn chain_whole(
        &self,
        tokens: &[u32],
        keys: &[Vec<u64>],
        salt: u64,
        row: usize,
        adapter: &PagedKVCacheAdapter,
        draft_window: Option<usize>,
    ) -> Option<Vec<ColdCacheKey>> {
        self.0.chain_whole(
            tokens,
            keys,
            salt,
            row,
            false,
            &|hash| adapter.prefix_cached(hash),
            draft_window,
        )
    }

    pub(super) fn discard(&mut self, store: &mut Store, id: EntryId) {
        self.0.discard(store, id);
    }

    pub(super) fn adopt(&mut self, store: &mut Store, config: &super::config::Config) -> u64 {
        let (grid, has_mtp) = (self.0.codec.grid, self.0.codec.has_mtp);
        self.0.adopt(store, |row| {
            super::memory::state_bytes(config, row as u64, grid as u64, has_mtp)
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn read_state(
        &mut self,
        store: &mut Store,
        key: ColdCacheKey,
        row: usize,
        extra_keys: &[Vec<u64>],
        segment_keys: &[ColdCacheKey],
        (layers, grid, draft): (usize, usize, bool),
        bound: u64,
    ) -> Read {
        if (layers, grid, draft) != (self.0.codec.layers, self.0.codec.grid, self.0.codec.has_mtp) {
            self.0.remove_object(&key);
            DiskTelemetry::bump(&self.0.telemetry.dropped_chain);
            return Read::Failed;
        }
        match self
            .0
            .read_state(store, key, row, extra_keys, segment_keys, bound)
        {
            DiskRead::State(cell) => {
                self.0.restored();
                Read::Cell(cell)
            }
            DiskRead::Busy => Read::Busy,
            DiskRead::Failed => Read::Failed,
        }
    }
}

/// Everything that changes a stored state's or block's bits: the model and
/// its checkpoint files, the config (quant and KV format included), the
/// page layout, the prefill window and its halving, the assistant opening,
/// the draft head, the GPU generation and this format.
#[allow(clippy::too_many_arguments)]
pub(super) fn fingerprint(
    checkpoint: &str,
    auxiliary: Option<&std::path::Path>,
    config: &super::config::Config,
    pool: &mlx_paged_attn::LayerKVPool,
    window: usize,
    open: &[u32],
    scratch: Option<u64>,
    has_mtp: bool,
) -> Option<ColdCacheFingerprint> {
    let mut parts: Vec<Vec<u8>> = vec![
        b"qwen4_exp".to_vec(),
        b"qwen4-prefix-v3".to_vec(),
        checkpoint.as_bytes().to_vec(),
    ];
    if let Some(dir) = auxiliary {
        parts.push(b"auxiliary".to_vec());
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .ok()?
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".safetensors") || name == "config.json")
            .collect();
        names.sort();
        for name in names {
            let meta = std::fs::metadata(dir.join(&name)).ok()?;
            let mtime = meta
                .modified()
                .ok()?
                .duration_since(std::time::UNIX_EPOCH)
                .ok()?
                .as_nanos();
            parts.push(name.into_bytes());
            parts.push(meta.len().to_le_bytes().to_vec());
            parts.push(mtime.to_le_bytes().to_vec());
        }
    }
    parts.push(serde_json::to_vec(config).ok()?);
    parts.push(config.served_context_tokens().to_le_bytes().to_vec());
    parts.push(format!("{:?}", config.rope_scaling).into_bytes());
    let layout = pool.layout();
    for value in [
        u64::from(pool.block_size()),
        pool.num_layers() as u64,
        u64::from(pool.config().num_kv_heads),
        u64::from(pool.config().head_size),
        u64::from(layout.values_token_major),
        u64::from(layout.index_rows),
        u64::from(layout.index_dimension),
        window as u64,
        scratch.unwrap_or(0),
        u64::from(scratch.is_some()),
        u64::from(has_mtp),
    ] {
        parts.push(value.to_le_bytes().to_vec());
    }
    parts.push(format!("{:?}", pool.cache_dtype()).into_bytes());
    parts.push(open.iter().flat_map(|id| id.to_le_bytes()).collect());
    parts.push(
        unsafe { mlx_sys::mlx_gpu_architecture_gen() }
            .to_le_bytes()
            .to_vec(),
    );
    Some(ColdCacheFingerprint::from_components(
        parts.iter().map(Vec::as_slice),
    ))
}
