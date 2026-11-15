//! The disk tier of the warm prefix states: a turn's replay state and the KV
//! blocks of its chain survive an engine restart (Splash d8af1054
//! `WriteBehind`, docs/server.md "state persistence").
//!
//! Everything goes through the model's own tier (`disk_tier.rs`: a locked
//! namespace directory, or a temporary one where that cannot be, its one
//! quota, and a `ColdCacheManager` for temp file, fsync, rename, payload
//! checksum, identity check and LRU). A KV block is a `Kv` object under its
//! chain key (tokens, pass keys, salt), a draft segment a `HybridDraft`
//! object, all under one fingerprint of everything that changes the math. A
//! state is a state record on the tier's one-slot state lane (group
//! `DFlash2State`, one fixed span, its lineage in its label), encoded by the
//! model thread into the one staging buffer.
//!
//! Only a turn of `minimum` rows or more writes. The model thread copies a
//! published state's blocks and draft segments to host memory from its
//! publication on, and a parker thread hands them to the writer. A replay
//! state is written `delay` after its publication, its turn running or not,
//! unless its conversation went on; it counts as durable only once every
//! object of its chain is on disk. A close writes the waiting states newest
//! first on the model thread. A temporary tier copies only while memory
//! drops states. A state whose whole chain is on disk keeps its entry when
//! memory lets go of its RAM copy, and a persistent tier's restart takes
//! back every such state, on disk only; a turn reads the deepest one of its
//! prompt back. Any miss prefills as before.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant};

use mlx_paged_attn::{
    ColdCacheBlock, ColdCacheFingerprint, ColdCacheKey, ColdCacheLayout, ColdCacheManager,
    ColdGroup, ColdLayerBlock, ColdObject, ColdSidecar, ColdSidecarLayout, ColdStateLayout,
    ColdStateObject, EnqueueRefused, Landed, MakeRoom, state_object_bytes,
};
use napi::{Error, Result};

use super::prefix_state::{BLOCK_ROWS, Cell, DraftCell, LayerCell, Segment};
use crate::array::{DType, MxArray};
use crate::disk_tier::DiskTier;
use crate::models::qwen3_5::dflash2_state::disk::{adopt_states, label_of, lineage_keys};
use crate::models::qwen3_5::dflash2_state::store::{DiskCopy, EntryId, HybridStateStore, Lineage};
use crate::transformer::paged_kv_cache_adapter::{
    ColdContextBacking, ColdTierContext, PagedKVCacheAdapter, SeqId,
};

/// Payload format of a state and a segment object.
const FORMAT: u32 = 3;
const MAGIC: &[u8; 4] = b"Q4PS";
/// Splash `WriteBehind` hands a point over this long after publication.
pub(super) const DELAY: Duration = Duration::from_secs(10);
/// Splash `WriteBehind::kMinimumTokens`: a shorter prompt prefills again
/// faster than its state is written and read back.
pub(super) const MINIMUM_ROWS: usize = 2048;
/// A temporary tier copies while memory dropped a state this recently
/// (Splash `kCopyDemandSeconds`).
const DEMAND_WINDOW: Duration = Duration::from_secs(60);
/// A written state whose chain nothing can add to any more is dropped this
/// long after.
const GIVE_UP: Duration = Duration::from_secs(10);
/// The wait for a writer slot of one object on the parker thread (an object
/// that finds none stays first in line), for the staging buffer before a
/// read, and what a dispose gives the waiting states.
const SLOT_WAIT: Duration = Duration::from_secs(2);
/// The model thread's disk pass copies blocks out at most this often: a copy
/// waits for the GPU.
const STEP_TOP_UP: Duration = Duration::from_millis(250);
/// Host copies of blocks and segments waiting for the writer.
pub(super) const QUEUE_BYTES: u64 = 128 << 20;

/// Counters of the stored states and of their disk tier, read by
/// `residencyInfo`.
#[derive(Default)]
pub(super) struct Telemetry {
    pub(super) states: AtomicU64,
    pub(super) state_bytes: AtomicU64,
    pub(super) published: AtomicU64,
    pub(super) restored: AtomicU64,
    /// 0 off, 1 persistent, 2 temporary.
    pub(super) disk_enabled: AtomicU64,
    /// States waiting for their write, the one in flight included.
    pub(super) waiting: AtomicU64,
    /// States written whose whole chain is on disk.
    pub(super) durable: AtomicU64,
    /// Waiting states not written: their conversation went on, or they are
    /// gone.
    pub(super) unneeded: AtomicU64,
    /// States not written or dropped (chain incomplete, room, encoding, a
    /// failed write, a closed tier), and copies the writer refused.
    pub(super) refused: AtomicU64,
    pub(super) disk_restored: AtomicU64,
    /// States a persistent tier took back at load, on disk only.
    pub(super) adopted: AtomicU64,
    /// States on disk removed because their chain was not.
    pub(super) dropped_chain: AtomicU64,
    /// Blocks and segments copied out and not handed to the writer yet.
    pub(super) owed: AtomicU64,
    pub(super) state_bytes_written: AtomicU64,
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
        self.published
            .store(states.counters.published, Ordering::Relaxed);
        self.restored
            .store(states.counters.restored, Ordering::Relaxed);
    }

    fn bump(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// The tier as a turn's row reports it: 0 off, 1 persistent, 2
    /// temporary; off once `tier` takes no write any more (closed, or a
    /// write failed for anything but room), as Qwen3.5 reports it.
    pub(super) fn mode(&self, tier: &crate::disk_tier::TierHandle) -> u64 {
        match self.disk_enabled.load(Ordering::Relaxed) {
            mode if tier.is_open() => mode,
            _ => 0,
        }
    }
}

/// The cold keys of the first `blocks` whole blocks of `tokens` in `group`,
/// block `i` keyed by `keys[i]` (the pass keys).
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

/// The key of draft segment `k` (draft rows `[kW, (k+1)W)`): the
/// `HybridDraft` chain of the blocks below `(k+1)W`, then one link over the
/// token after it, which the segment's last row was computed with.
fn segment_key(
    fingerprint: ColdCacheFingerprint,
    draft_chain: &[ColdCacheKey],
    tokens: &[u32],
    window: usize,
    k: usize,
) -> Option<ColdCacheKey> {
    let end = (k + 1) * window;
    let blocks = end / BLOCK_ROWS;
    if blocks == 0 || !end.is_multiple_of(BLOCK_ROWS) {
        return None;
    }
    let parent = *draft_chain.get(blocks - 1)?;
    let next = *tokens.get(end)?;
    Some(ColdCacheKey::chain(
        ColdGroup::HybridDraft,
        fingerprint,
        Some(parent),
        &[next],
        &[window as u64],
        0,
        blocks,
    ))
}

// ---------------------------------------------------------------------------
// Codec: a header (magic, format, grid, layers, draft flag, each layer's
// pooled indexer blocks, then every array's dtype and shape) and the arrays'
// bytes in order, as one U8 tensor of a sidecar object.
// ---------------------------------------------------------------------------

fn dtype_code(dtype: DType) -> Option<u8> {
    match dtype {
        DType::BFloat16 => Some(1),
        DType::Float16 => Some(2),
        DType::Float32 => Some(3),
        _ => None,
    }
}

fn element_bytes(code: u8) -> Option<usize> {
    match code {
        1 | 2 => Some(2),
        3 => Some(4),
        _ => None,
    }
}

/// Where an encoding goes: a growing buffer, or one fixed span that takes
/// nothing past its end.
trait Sink {
    fn put(&mut self, bytes: &[u8]);
    fn reserve(&mut self, _bytes: usize) {}
}

impl Sink for Vec<u8> {
    fn put(&mut self, bytes: &[u8]) {
        self.extend_from_slice(bytes);
    }
    fn reserve(&mut self, bytes: usize) {
        self.reserve_exact(bytes);
    }
}

/// A fixed span; `over` once something did not fit.
struct Span<'a> {
    into: &'a mut [u8],
    at: usize,
    over: bool,
}

impl Sink for Span<'_> {
    fn put(&mut self, bytes: &[u8]) {
        let end = self.at.saturating_add(bytes.len());
        match self.into.get_mut(self.at..end) {
            Some(target) if !self.over => {
                target.copy_from_slice(bytes);
                self.at = end;
            }
            _ => self.over = true,
        }
    }
}

struct Writer<S>(S);

impl<S: Sink> Writer<S> {
    fn u8(&mut self, value: u8) {
        self.0.put(&[value]);
    }
    fn u32(&mut self, value: u32) {
        self.0.put(&value.to_le_bytes());
    }
    fn u64(&mut self, value: u64) {
        self.0.put(&value.to_le_bytes());
    }
    /// `values` little-endian, a chunk at a time.
    fn values<const N: usize, T: Copy>(&mut self, values: &[T], bytes: impl Fn(T) -> [u8; N]) {
        let mut chunk = [0u8; 4096];
        for part in values.chunks(4096 / N) {
            for (at, &value) in part.iter().enumerate() {
                chunk[at * N..(at + 1) * N].copy_from_slice(&bytes(value));
            }
            self.0.put(&chunk[..part.len() * N]);
        }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Option<&'a [u8]> {
        let end = self.at.checked_add(len)?;
        let out = self.bytes.get(self.at..end)?;
        self.at = end;
        Some(out)
    }
    fn u8(&mut self) -> Option<u8> {
        self.take(1).map(|b| b[0])
    }
    fn u32(&mut self) -> Option<u32> {
        self.take(4)
            .and_then(|b| b.try_into().ok())
            .map(u32::from_le_bytes)
    }
    fn u64(&mut self) -> Option<u64> {
        self.take(8)
            .and_then(|b| b.try_into().ok())
            .map(u64::from_le_bytes)
    }
}

/// `arrays` (absent ones kept as absent) and the header fields, encoded
/// into `out`. False when an array's dtype has no code.
fn encode_to<S: Sink>(
    out: &mut Writer<S>,
    grid: usize,
    index_blocks: &[usize],
    draft: bool,
    arrays: &[Option<&MxArray>],
) -> Result<bool> {
    out.0.put(MAGIC);
    out.u32(FORMAT);
    out.u64(grid as u64);
    out.u32(index_blocks.len() as u32);
    out.u8(u8::from(draft));
    for &blocks in index_blocks {
        out.u64(blocks as u64);
    }
    out.u32(arrays.len() as u32);
    let mut data = 0usize;
    let mut codes = Vec::with_capacity(arrays.len());
    for array in arrays {
        let Some(array) = array else {
            out.u8(0);
            codes.push(0);
            continue;
        };
        let Some(code) = dtype_code(array.dtype()?) else {
            return Ok(false);
        };
        out.u8(1);
        out.u8(code);
        let ndim = array.ndim()?;
        out.u8(ndim as u8);
        let mut count = 1usize;
        for axis in 0..ndim {
            let dim = array.shape_at(axis)?;
            out.u64(dim as u64);
            count = count.saturating_mul(usize::try_from(dim).unwrap_or(0));
        }
        data = data.saturating_add(count.saturating_mul(element_bytes(code).unwrap_or(0)));
        codes.push(code);
    }
    out.0.reserve(data);
    for (array, code) in arrays.iter().zip(codes) {
        let Some(array) = array else {
            continue;
        };
        if code == 3 {
            out.values(&array.to_float32()?, f32::to_le_bytes);
        } else {
            out.values(&array.to_uint16_native()?, u16::to_le_bytes);
        }
    }
    Ok(true)
}

/// [`encode_to`] into a buffer of its own.
fn encode(
    grid: usize,
    index_blocks: &[usize],
    draft: bool,
    arrays: &[Option<&MxArray>],
) -> Result<Option<Vec<u8>>> {
    let mut out = Writer(Vec::new());
    let encoded = encode_to(&mut out, grid, index_blocks, draft, arrays)?;
    Ok(encoded.then_some(out.0))
}

/// The inverse of [`encode`]: (grid, index blocks, draft flag, arrays).
/// `None` for any byte that does not fit the format.
type Decoded = (usize, Vec<usize>, bool, Vec<Option<MxArray>>);

fn decode(bytes: &[u8], arrays: usize, padded: bool) -> Option<Decoded> {
    let mut reader = Reader { bytes, at: 0 };
    if reader.take(4)? != MAGIC || reader.u32()? != FORMAT {
        return None;
    }
    let grid = usize::try_from(reader.u64()?).ok()?;
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
        .map(|_| reader.u64().and_then(|n| usize::try_from(n).ok()))
        .collect::<Option<Vec<_>>>()?;
    if reader.u32()? as usize != arrays {
        return None;
    }
    let mut shapes = Vec::with_capacity(arrays);
    for _ in 0..arrays {
        if reader.u8()? == 0 {
            shapes.push(None);
            continue;
        }
        let code = reader.u8()?;
        let size = element_bytes(code)?;
        let ndim = reader.u8()? as usize;
        let mut shape = Vec::with_capacity(ndim);
        let mut count = 1usize;
        for _ in 0..ndim {
            let dim = reader.u64()?;
            count = count.checked_mul(usize::try_from(dim).ok()?)?;
            shape.push(i64::try_from(dim).ok()?);
        }
        shapes.push(Some((code, shape, count.checked_mul(size)?)));
    }
    let mut out = Vec::with_capacity(arrays);
    for shape in shapes {
        let Some((code, shape, len)) = shape else {
            out.push(None);
            continue;
        };
        let raw = reader.take(len)?;
        let array = if code == 3 {
            let values: Vec<f32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            MxArray::from_float32(&values, &shape).ok()?
        } else {
            let values: Vec<u16> = raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_le_bytes(*b))
                .collect();
            if code == 1 {
                MxArray::from_bfloat16(&values, &shape).ok()?
            } else {
                MxArray::from_float16(&values, &shape).ok()?
            }
        };
        out.push(Some(array));
    }
    (padded || reader.at == bytes.len()).then_some((grid, index_blocks, draft, out))
}

/// A state's arrays in their encoded order: each layer's conv, recurrent
/// and PLE history, then the draft's waiting rows and seed. The segments are
/// objects of their own.
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
    let mut out = Writer(Span {
        into,
        at: 0,
        over: false,
    });
    if !encode_to(&mut out, cell.grid, &blocks, cell.draft.is_some(), &arrays)? || out.0.over {
        return Ok(false);
    }
    let Span { into, at, .. } = out.0;
    into[at..].fill(0);
    Ok(true)
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

/// The sidecar layout of an object: one U8 tensor of `bytes`, at
/// `boundary`, with the format, `count` (layers of a state, segment index
/// plus one of a segment) and the prefill window.
fn layout(
    group: ColdGroup,
    boundary: usize,
    count: usize,
    grid: usize,
    bytes: usize,
) -> Option<ColdSidecarLayout> {
    Some(ColdSidecarLayout {
        group,
        boundary_tokens: u32::try_from(boundary).ok()?,
        num_layers: 1,
        tensors_per_layer: 1,
        dtype: "u8".to_string(),
        dims: vec![
            FORMAT,
            u32::try_from(count).ok()?.max(1),
            u32::try_from(grid).ok()?,
        ],
        bytes_per_tensor: bytes,
    })
}

fn layout_fits(
    found: &ColdSidecarLayout,
    group: ColdGroup,
    boundary: usize,
    count: usize,
    grid: usize,
) -> bool {
    layout(group, boundary, count, grid, found.bytes_per_tensor).as_ref() == Some(found)
}

// ---------------------------------------------------------------------------
// The parker thread: block and segment copies to the writer.
// ---------------------------------------------------------------------------

fn object_bytes(object: &ColdObject) -> u64 {
    match object {
        ColdObject::Block(block) => block
            .layers
            .iter()
            .map(|layer| (layer.keys.len() + layer.values.len()) as u64)
            .sum(),
        ColdObject::Sidecar(sidecar) => sidecar.tensors.iter().map(|t| t.len() as u64).sum(),
    }
}

/// What the model thread last left for the disk to do.
#[derive(Clone, Copy, Default)]
struct Plan {
    /// Bumped by every pass that stores a plan.
    epoch: u64,
    /// When the next waiting point or stuck chain is due.
    next_due: Option<Instant>,
    /// A chain is owed: blocks to copy, or a written state whose chain is
    /// not whole yet.
    owing: bool,
    /// The model thread's next pass that may copy.
    next_step: Option<Instant>,
}

#[derive(Default)]
struct Slot {
    /// Block and segment copies, first in line first.
    items: VecDeque<ColdObject>,
    item_bytes: u64,
    flush: u64,
    flushed: u64,
    flush_until: Option<Instant>,
    /// The last flush for which the thread waited out the copies it handed.
    drained: u64,
    stop: bool,
    plan: Plan,
    /// A close asks the model thread to write the waiting states by then.
    closing: Option<Instant>,
    /// The model thread's flush of the waiting states ended: true when no
    /// state waited any more and every copy and write it handed landed.
    points_flushed: Option<bool>,
    /// A state write went out: the parker measures the cache root.
    measure: bool,
    #[cfg(test)]
    measured: u64,
    /// The plan epoch whose wake went out.
    woke: u64,
    /// The plan epoch for which the writer was drained.
    drained_epoch: u64,
    #[cfg(test)]
    hold: bool,
    #[cfg(test)]
    slot_wait: Option<Duration>,
}

impl Slot {
    fn slot_wait(&self) -> Duration {
        #[cfg(test)]
        if let Some(wait) = self.slot_wait {
            return wait;
        }
        SLOT_WAIT
    }
}

struct Parker {
    slot: Mutex<Slot>,
    wake: Condvar,
    done: Condvar,
}

impl Parker {
    fn lock(&self) -> MutexGuard<'_, Slot> {
        self.slot
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    /// Hands every queued copy to the writer and waits until the writer is
    /// past them, until `deadline`. False when a copy was not handed over by
    /// then or the thread stopped first.
    fn flush(&self, deadline: Instant) -> bool {
        let mut slot = self.lock();
        slot.flush += 1;
        let target = slot.flush;
        slot.flush_until = Some(deadline);
        self.wake.notify_all();
        while slot.flushed < target && !slot.stop {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            slot = self
                .done
                .wait_timeout(slot, left)
                .map(|(guard, _)| guard)
                .unwrap_or_else(|poison| poison.into_inner().0);
        }
        slot.flushed >= target && slot.items.is_empty()
    }
}

/// Hands the queued copies to the writer one at a time, and measures the
/// cache root after a state write went out (that can remove another cache's
/// directory: never on the model thread). With no copy left and a chain
/// owed it drains the writer once per plan, then wakes the model thread once
/// per plan, at the earliest when the plan is due and the model thread's
/// next pass may copy.
fn run(parker: Arc<Parker>, tier: Arc<DiskTier>, telemetry: Arc<Telemetry>, wake: Arc<Wake>) {
    let manager = Arc::clone(&tier.manager);
    let mut slot = parker.lock();
    loop {
        if slot.stop {
            break;
        }
        if std::mem::take(&mut slot.measure) {
            drop(slot);
            tier.measure();
            slot = parker.lock();
            #[cfg(test)]
            {
                slot.measured += 1;
            }
            continue;
        }
        let now = Instant::now();
        let flushing = slot.flush > slot.flushed;
        let until = slot.flush_until.filter(|_| flushing);
        if until.is_some_and(|until| now >= until) {
            slot.flushed = slot.flush;
            parker.done.notify_all();
            continue;
        }
        #[cfg(test)]
        let held = slot.hold;
        #[cfg(not(test))]
        let held = false;
        if !held && let Some(object) = slot.items.pop_front() {
            let bytes = object_bytes(&object);
            let wait = now + slot.slot_wait();
            drop(slot);
            let deadline = until.map_or(wait, |until| until.min(wait));
            let sent = manager.offer(object, deadline);
            slot = parker.lock();
            match sent {
                // No writer slot freed: it stays first in line, so the chain
                // keeps no hole.
                Ok(Some(object)) => slot.items.push_front(object),
                Ok(None) => {
                    slot.item_bytes = slot.item_bytes.saturating_sub(bytes);
                    telemetry.owed.fetch_sub(1, Ordering::Relaxed);
                }
                Err(_) => {
                    slot.item_bytes = slot.item_bytes.saturating_sub(bytes);
                    telemetry.owed.fetch_sub(1, Ordering::Relaxed);
                    Telemetry::bump(&telemetry.refused);
                }
            }
            continue;
        }
        if flushing && slot.items.is_empty() {
            if slot.drained < slot.flush {
                slot.drained = slot.flush;
                drop(slot);
                if let Some(until) = until {
                    manager.drain(until.saturating_duration_since(Instant::now()));
                }
                slot = parker.lock();
                continue;
            }
            slot.flushed = slot.flush;
            parker.done.notify_all();
        }
        let plan = slot.plan;
        if slot.items.is_empty() && plan.owing && slot.drained_epoch != plan.epoch {
            slot.drained_epoch = plan.epoch;
            drop(slot);
            manager.drain(SLOT_WAIT);
            slot = parker.lock();
            continue;
        }
        let mut wait = Duration::from_secs(3600);
        let want = if plan.owing { Some(now) } else { plan.next_due };
        if let Some(want) = want
            && slot.woke != plan.epoch
        {
            let at = plan.next_step.map_or(want, |step| step.max(want));
            if now >= at {
                slot.woke = plan.epoch;
                drop(slot);
                wake.fire();
                slot = parker.lock();
                continue;
            }
            wait = wait.min(at - now);
        }
        if let Some(until) = until {
            wait = wait.min(until.saturating_duration_since(now));
        }
        slot = parker
            .wake
            .wait_timeout(slot, wait.max(Duration::from_millis(1)))
            .map(|(guard, _)| guard)
            .unwrap_or_else(|poison| poison.into_inner().0);
    }
}

// ---------------------------------------------------------------------------
// The model thread's side.
// ---------------------------------------------------------------------------

type Store = HybridStateStore<Cell>;

/// What one turn of a sequence owes the disk: its lineage up to its last
/// point, keyed as the prefix cache keys it, and how far its blocks and
/// draft segments are copied out (or found on disk).
struct Cursor {
    lineage: Vec<u32>,
    /// Hot prefix-cache hashes of the lineage's blocks.
    hot: Vec<u64>,
    /// `Kv` cold keys of the lineage's blocks.
    cold: Vec<ColdCacheKey>,
    /// `HybridDraft` chain of the lineage's blocks.
    draft_chain: Vec<ColdCacheKey>,
    /// Rows whose blocks the prefix cache holds (the deepest publication).
    registered: usize,
    /// The draft segments of the deepest publication (shared handles).
    pending: Vec<Segment>,
    window: usize,
    blocks: usize,
    segments: usize,
    /// A block left the prefix cache before its copy.
    broken: bool,
    /// The turn ended: the cursor goes once it owes nothing.
    closed: bool,
}

/// The shape of a load's state records.
pub(super) struct LaneShape {
    /// One state's payload: every state record is this long.
    pub(super) span: u64,
    pub(super) layers: usize,
    pub(super) grid: usize,
    pub(super) has_mtp: bool,
}

/// The tier's one-slot state lane: the record shape and the one staging
/// buffer every state write and read goes through.
struct Lane {
    spans: Vec<u64>,
    /// Compared as text, never parsed.
    geometry: String,
    layers: usize,
    grid: usize,
    has_mtp: bool,
    /// `None` while the write in flight holds it.
    staging: Option<Vec<u8>>,
    landed_tx: Sender<Landed>,
    landed_rx: Receiver<Landed>,
    /// The state write in flight: its entry and its object.
    pending: Option<(EntryId, ColdCacheKey)>,
}

impl Lane {
    fn new(shape: &LaneShape) -> Option<Self> {
        let bytes = usize::try_from(shape.span)
            .ok()
            .filter(|&bytes| bytes > 0)?;
        let (landed_tx, landed_rx) = std::sync::mpsc::channel();
        Some(Self {
            spans: vec![shape.span],
            geometry: format!(
                "qwen4-cell {} {} {} {}\n",
                shape.layers,
                shape.grid,
                shape.span,
                u8::from(shape.has_mtp)
            ),
            layers: shape.layers,
            grid: shape.grid,
            has_mtp: shape.has_mtp,
            staging: Some(vec![0; bytes]),
            landed_tx,
            landed_rx,
            pending: None,
        })
    }

    /// A state record's layout at `row`.
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
    /// The KV blocks, then the draft segments.
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

/// The call that wakes the model thread for its disk work. A load installs
/// a weak command sender: it holds neither the closer nor the tier.
#[derive(Default)]
pub(super) struct Wake {
    call: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
}

impl Wake {
    fn slot(&self) -> MutexGuard<'_, Option<Arc<dyn Fn() + Send + Sync>>> {
        self.call
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn is_set(&self) -> bool {
        self.slot().is_some()
    }

    pub(super) fn fire(&self) {
        let call = self.slot().clone();
        if let Some(call) = call {
            call();
        }
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

/// Closes the tier from any thread: the model thread writes the waiting
/// states, the parker hands over the queued copies, then the tier drains
/// its writer and takes no write after.
pub(super) struct Closer {
    parker: Arc<Parker>,
    tier: Arc<DiskTier>,
    telemetry: Arc<Telemetry>,
    wake: Arc<Wake>,
    /// A close waits for the model thread's flush of the waiting states.
    closing: std::sync::atomic::AtomicBool,
    /// The one close's outcome once it ended; `Some(None)` while it runs.
    outcome: Mutex<Option<Option<bool>>>,
    ended: Condvar,
}

impl Closer {
    /// The call that wakes the model thread (a weak command sender).
    pub(super) fn set_wake(&self, call: Arc<dyn Fn() + Send + Sync>) {
        *self.wake.slot() = Some(call);
    }

    /// A persistent tier: the model thread writes the waiting states newest
    /// first, given all of the time but the last second (at most half of
    /// it), then every queued copy goes to the writer and the tier closes,
    /// by `deadline`; false when the drain missed it. A temporary tier
    /// keeps nothing: its writer stops and its directory goes. One close
    /// runs: a call while it runs waits for its outcome until its own
    /// deadline (false when it has none by then), and a later call gets the
    /// same outcome.
    pub(super) fn close(&self, deadline: Instant) -> bool {
        self.once(deadline, true)
    }

    /// [`Self::close`] without the model thread's flush: the model thread
    /// calls it itself.
    fn finish(&self, deadline: Instant) -> bool {
        self.once(deadline, false)
    }

    fn once(&self, deadline: Instant, ask_model: bool) -> bool {
        {
            let mut outcome = self
                .outcome
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if outcome.is_some() {
                loop {
                    if let Some(Some(landed)) = *outcome {
                        return landed;
                    }
                    let left = deadline.saturating_duration_since(Instant::now());
                    if left.is_zero() {
                        return false;
                    }
                    outcome = self
                        .ended
                        .wait_timeout(outcome, left)
                        .map(|(guard, _)| guard)
                        .unwrap_or_else(|poison| poison.into_inner().0);
                }
            }
            *outcome = Some(None);
        }
        let mut flushed = true;
        if self.tier.persistent && self.tier.is_open() {
            if ask_model && self.wake.is_set() {
                flushed = self.flush_on_model_thread(deadline);
            } else if !ask_model {
                flushed = self.parker.lock().points_flushed != Some(false);
            }
        }
        if self.tier.persistent {
            flushed &= self.parker.flush(deadline);
        }
        let landed = self.tier.close(deadline) && flushed;
        self.closing.store(false, Ordering::Release);
        self.telemetry.disk_enabled.store(0, Ordering::Relaxed);
        *self
            .outcome
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = Some(Some(landed));
        self.ended.notify_all();
        landed
    }

    /// Wakes the model thread to write its waiting states and waits for it
    /// until the last second of `deadline` (at most half of what is left).
    /// True when its flush left no state waiting and all it handed landed.
    fn flush_on_model_thread(&self, deadline: Instant) -> bool {
        let left = deadline.saturating_duration_since(Instant::now());
        let until = deadline
            .checked_sub((left / 2).min(Duration::from_secs(1)))
            .unwrap_or(deadline);
        {
            let mut slot = self.parker.lock();
            if slot.stop {
                return false;
            }
            slot.closing = Some(until);
            slot.points_flushed = None;
        }
        self.closing.store(true, Ordering::Release);
        self.wake.fire();
        let mut slot = self.parker.lock();
        while slot.points_flushed.is_none() && !slot.stop {
            let left = until.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            slot = self
                .parker
                .done
                .wait_timeout(slot, left)
                .map(|(guard, _)| guard)
                .unwrap_or_else(|poison| poison.into_inner().0);
        }
        slot.points_flushed == Some(true)
    }
}

pub(super) struct Disk {
    manager: Arc<ColdCacheManager>,
    closer: Arc<Closer>,
    fingerprint: ColdCacheFingerprint,
    pub(super) delay: Duration,
    /// A turn whose lineage is shorter writes nothing.
    pub(super) minimum: usize,
    /// A written state whose chain can no longer grow is dropped this long
    /// after.
    pub(super) give_up: Duration,
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
    #[cfg(test)]
    pub(super) encoded: usize,
    #[cfg(test)]
    pub(super) step_runs: usize,
    /// The state writes that landed, in landing order.
    #[cfg(test)]
    pub(super) landed_states: Vec<ColdCacheKey>,
}

impl Disk {
    /// The states over `tier`, its parker started.
    pub(super) fn open(
        tier: Arc<DiskTier>,
        telemetry: Arc<Telemetry>,
        shape: &LaneShape,
    ) -> Result<Self> {
        let lane = Lane::new(shape)
            .ok_or_else(|| Error::from_reason("Qwen4 prefix disk: no state record shape"))?;
        let manager = Arc::clone(&tier.manager);
        let fingerprint = tier.state_fingerprint;
        let persistent = tier.persistent;
        let trimmed = manager.fit_quota();
        if trimmed > 0 {
            tracing::info!(
                "qwen4_exp: took {trimmed} stored objects off the SSD cache to fit its limit"
            );
        }
        let parker = Arc::new(Parker {
            slot: Mutex::new(Slot::default()),
            wake: Condvar::new(),
            done: Condvar::new(),
        });
        let wake = Arc::new(Wake::default());
        let thread = {
            let (parker, tier, telemetry, wake) = (
                Arc::clone(&parker),
                Arc::clone(&tier),
                Arc::clone(&telemetry),
                Arc::clone(&wake),
            );
            std::thread::Builder::new()
                .name("qwen4-prefix-parker".to_string())
                .spawn(move || run(parker, tier, telemetry, wake))
                .map_err(|e| Error::from_reason(format!("Qwen4 prefix disk: {e}")))?
        };
        telemetry
            .disk_enabled
            .store(if persistent { 1 } else { 2 }, Ordering::Relaxed);
        // A landed state write wakes the model thread to take it.
        {
            let wake = Arc::clone(&wake);
            manager.set_landing_wake(Some(Arc::new(move || wake.fire())));
        }
        let closer = Arc::new(Closer {
            parker: Arc::clone(&parker),
            tier,
            telemetry: Arc::clone(&telemetry),
            wake,
            closing: std::sync::atomic::AtomicBool::new(false),
            outcome: Mutex::new(None),
            ended: Condvar::new(),
        });
        Ok(Self {
            manager,
            closer,
            fingerprint,
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
            #[cfg(test)]
            encoded: 0,
            #[cfg(test)]
            step_runs: 0,
            #[cfg(test)]
            landed_states: Vec::new(),
        })
    }

    /// The adapter restores the KV blocks of a state's chain from this
    /// store; its prefix lookups run only below a state's row.
    pub(super) fn attach(&self, adapter: &mut PagedKVCacheAdapter) {
        adapter.set_cold_tier(ColdTierContext {
            manager: Arc::clone(&self.manager),
            fingerprint: self.fingerprint,
            backing: ColdContextBacking::KvOnly,
        });
    }

    /// Weak: the waiting states, the queued copies and the tier go with this
    /// disk, not with the napi object, whose memory V8 does not see.
    pub(super) fn closer(&self) -> Weak<Closer> {
        Arc::downgrade(&self.closer)
    }

    /// The napi object's handle on the tier.
    pub(super) fn tier_handle(&self) -> crate::disk_tier::TierHandle {
        crate::disk_tier::TierHandle::of(Some(&self.closer.tier))
    }

    #[cfg(test)]
    pub(super) fn tier(&self) -> &Arc<DiskTier> {
        &self.closer.tier
    }

    /// Holds the block and segment copies back from the writer (`true`)
    /// until released.
    #[cfg(test)]
    pub(super) fn hold(&self, hold: bool) {
        self.parker.lock().hold = hold;
        self.parker.wake.notify_all();
    }

    /// The wait for a writer slot, `Duration::ZERO` for one try.
    #[cfg(test)]
    pub(super) fn set_slot_wait(&self, wait: Duration) {
        self.parker.lock().slot_wait = Some(wait);
    }

    /// Drops every copy the parker holds, as an offer the writer refuses
    /// drops it.
    #[cfg(test)]
    pub(super) fn discard_queue(&self) {
        let mut slot = self.parker.lock();
        let dropped = slot.items.len() as u64;
        slot.items.clear();
        slot.item_bytes = 0;
        self.telemetry.owed.fetch_sub(dropped, Ordering::Relaxed);
    }

    /// No point waits, no state write is in flight, no chain is owed.
    #[cfg(test)]
    pub(super) fn idle(&self) -> bool {
        self.points.is_empty() && self.lane.pending.is_none() && self.owed.is_empty()
    }

    /// Written states whose chain is not whole yet.
    #[cfg(test)]
    pub(super) fn owed_states(&self) -> usize {
        self.owed.len()
    }

    /// The plan the last pass left for the parker says a chain is owed.
    #[cfg(test)]
    pub(super) fn plan_owing(&self) -> bool {
        self.parker.lock().plan.owing
    }

    /// The cache root measures the parker made.
    #[cfg(test)]
    pub(super) fn measured(&self) -> u64 {
        self.parker.lock().measured
    }

    /// The record layout every state object of this load carries.
    #[cfg(test)]
    pub(super) fn spans(&self) -> &[u64] {
        &self.lane.spans
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
    pub(super) fn drops(&mut self, total: u64) {
        if total > self.seen_drops {
            self.seen_drops = total;
            self.demand_until = Some(Instant::now() + DEMAND_WINDOW);
        }
    }

    /// After a command reset the next prime reads nothing from disk: the
    /// purged prefix is prefilled again.
    pub(super) fn reset(&mut self) {
        self.cursors.clear();
        self.skip_once = true;
    }

    pub(super) fn take_skip(&mut self) -> bool {
        std::mem::take(&mut self.skip_once)
    }

    /// Removes `key`'s object while the tier holds its directory: after a
    /// close another process may hold it.
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

    /// `seq`'s text turn of `lineage` (its prompt up to its last point),
    /// keyed by `keys` and `hot` (the prefix cache's hashes), owes the disk
    /// its chain.
    pub(super) fn start(
        &mut self,
        seq: SeqId,
        lineage: &[u32],
        keys: &[Vec<u64>],
        hot: &[u64],
        salt: u64,
        window: usize,
    ) {
        if lineage.len() < self.minimum {
            self.cursors.remove(&seq);
            return;
        }
        let blocks = hot.len().min(keys.len()).min(lineage.len() / BLOCK_ROWS);
        let lineage = lineage[..blocks * BLOCK_ROWS].to_vec();
        let fingerprint = self.fingerprint;
        let keyed = |group| chain(group, fingerprint, &lineage, keys, salt, blocks);
        let (cold, draft_chain) = (keyed(ColdGroup::Kv), keyed(ColdGroup::HybridDraft));
        self.cursors.insert(
            seq,
            Cursor {
                hot: hot[..blocks].to_vec(),
                lineage,
                cold,
                draft_chain,
                registered: 0,
                pending: Vec::new(),
                window,
                blocks: 0,
                segments: 0,
                broken: false,
                closed: false,
            },
        );
    }

    /// `seq` published a state at `row`: its blocks below are in the prefix
    /// cache, its draft segments are `draft`'s. A replay state (`replay`, its
    /// entry in `store`) of `minimum` rows or more waits for its write,
    /// `delay` from now.
    pub(super) fn published(
        &mut self,
        store: &Store,
        seq: SeqId,
        row: usize,
        draft: Option<&DraftCell>,
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

    /// Whether `seq` still owes blocks or segments it can copy now.
    fn owes(&self, seq: SeqId) -> bool {
        self.cursors.get(&seq).is_some_and(|cursor| {
            !cursor.broken
                && (cursor.blocks < (cursor.registered / BLOCK_ROWS).min(cursor.cold.len())
                    || cursor.segments < cursor.pending.len())
        })
    }

    /// Copies out the blocks of `seq`'s chain the prefix cache holds and the
    /// draft segments it published that are not on disk yet, within the host
    /// queue's bytes. Never waits for the writer.
    pub(super) fn top_up(&mut self, seq: SeqId, adapter: &mut PagedKVCacheAdapter) {
        if !self.closer.tier.is_open() || !self.copying() || !self.owes(seq) {
            return;
        }
        let mut bytes = self.parker.lock().item_bytes;
        if bytes >= QUEUE_BYTES {
            return;
        }
        let Some(cursor) = self.cursors.get_mut(&seq) else {
            return;
        };
        let target = (cursor.registered / BLOCK_ROWS).min(cursor.cold.len());
        let mut items = Vec::new();
        let mut synced = false;
        while cursor.blocks < target && bytes < QUEUE_BYTES {
            let i = cursor.blocks;
            if self.manager.contains(&cursor.cold[i]) {
                cursor.blocks += 1;
                continue;
            }
            if !synced {
                if adapter.eval_pending_pool_writes().is_err() {
                    return;
                }
                crate::array::memory::synchronize();
                synced = true;
            }
            let tokens = &cursor.lineage[i * BLOCK_ROWS..(i + 1) * BLOCK_ROWS];
            let Some(block) = capture_block(
                adapter,
                cursor.hot[i],
                cursor.cold[i],
                self.fingerprint,
                tokens,
            ) else {
                cursor.broken = true;
                break;
            };
            let item = ColdObject::Block(block);
            bytes += object_bytes(&item);
            items.push(item);
            cursor.blocks += 1;
        }
        let window = cursor.window;
        while !cursor.broken && cursor.segments < cursor.pending.len() && bytes < QUEUE_BYTES {
            let k = cursor.segments;
            let Some(key) = segment_key(
                self.fingerprint,
                &cursor.draft_chain,
                &cursor.lineage,
                window,
                k,
            ) else {
                cursor.broken = true;
                break;
            };
            if !self.manager.contains_in(&key, ColdGroup::HybridDraft) {
                let object = encode_segment(window, &cursor.pending[k])
                    .ok()
                    .flatten()
                    .and_then(|payload| {
                        Some(ColdSidecar {
                            key,
                            fingerprint: self.fingerprint,
                            layout: layout(
                                ColdGroup::HybridDraft,
                                (k + 1) * window,
                                k + 1,
                                window,
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
    pub(super) fn close(&mut self, seq: SeqId) {
        if let Some(cursor) = self.cursors.get_mut(&seq) {
            cursor.closed = true;
        }
    }

    /// [`Self::top_up`] for every sequence that still owes the disk.
    fn top_up_all(&mut self, adapter: &mut PagedKVCacheAdapter) {
        let owing: Vec<SeqId> = self
            .cursors
            .keys()
            .copied()
            .filter(|&seq| self.owes(seq))
            .collect();
        for seq in owing {
            self.top_up(seq, adapter);
        }
        // A closed tier, or a temporary one memory asks nothing of, takes no
        // copy, so a closed cursor owes it nothing.
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
    pub(super) fn step(&mut self, store: &mut Store, adapter: &mut PagedKVCacheAdapter) -> bool {
        let landed = self.take_landings(store);
        if let Some(deadline) = self.take_close() {
            self.flush_points(store, adapter, deadline);
            return landed;
        }
        let now = Instant::now();
        if landed || now >= self.step_after {
            self.pass(store, adapter, now);
        }
        landed
    }

    /// [`Self::step`] without the interval: a turn's end.
    pub(super) fn step_now(&mut self, store: &mut Store, adapter: &mut PagedKVCacheAdapter) {
        self.take_landings(store);
        if let Some(deadline) = self.take_close() {
            self.flush_points(store, adapter, deadline);
            return;
        }
        self.pass(store, adapter, Instant::now());
    }

    /// The deadline of a close that waits for this thread, once.
    fn take_close(&mut self) -> Option<Instant> {
        if !self.closer.closing.swap(false, Ordering::AcqRel) {
            return None;
        }
        Some(self.parker.lock().closing.unwrap_or_else(Instant::now))
    }

    fn pass(&mut self, store: &mut Store, adapter: &mut PagedKVCacheAdapter, now: Instant) {
        self.step_after = now + STEP_TOP_UP;
        #[cfg(test)]
        {
            self.step_runs += 1;
        }
        self.take_evictions(store);
        self.judge_owed(store, now);
        self.due_points(store, now);
        self.top_up_all(adapter);
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

    /// Tells the parker what is left: whether a chain is owed and when the
    /// next point or stuck chain is due.
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
        slot.plan = Plan {
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
            #[cfg(test)]
            self.landed_states.push(key);
            match store.disk_mut(id) {
                Some(copy) if ours => copy.landed = true,
                _ => {
                    self.remove_object(&key);
                }
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

    /// Whether point `id` still needs its write: not when its entry is gone,
    /// a checkpoint or one its conversation went on from (unneeded), nor
    /// when it has a disk copy.
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

    /// The due points in publication order, one write at a time (Splash
    /// `WriteBehind::copy`): a point that waits for the buffer stalls the
    /// rest; a temporary tier skips a point in use and stops while memory
    /// asks for nothing.
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
    /// it takes comes from older ones. Its blocks and segments are copied
    /// out by its turn's cursor; it is durable once they are on disk.
    fn write(&mut self, store: &mut Store, id: EntryId) -> Written {
        if !self.closer.tier.is_open() {
            return Written::Refused;
        }
        let Some(entry) = store.get(id) else {
            return Written::Refused;
        };
        let Some(cell) = entry.ram() else {
            return Written::Refused;
        };
        let row = entry.depth();
        if cell.grid != self.lane.grid
            || cell.layers.len() != self.lane.layers
            || cell.draft.is_some() != self.lane.has_mtp
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
        let segments = cell.draft.as_ref().map_or(0, |draft| draft.segments.len());
        for k in 0..segments {
            let Some(segment) =
                segment_key(self.fingerprint, &draft_chain, entry.tokens(), cell.grid, k)
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
            self.owed.push(owed);
            return Written::OnTier;
        }
        let Some(mut payload) = self.lane.staging.take() else {
            return Written::Busy;
        };
        if !matches!(encode_cell_into(cell, &mut payload), Ok(true)) {
            self.lane.staging = Some(payload);
            return Written::Refused;
        }
        #[cfg(test)]
        {
            self.encoded += 1;
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

    /// A clean stop on a persistent tier (Splash `Engine::flushRestorePoints`):
    /// every waiting point, newest first, due or not, in use or not, is
    /// written, then every copy goes to the writer and the writer drains,
    /// until `deadline`. A temporary tier keeps nothing. False when a point
    /// still waits or a copy or write did not land by then.
    pub(super) fn flush_points(
        &mut self,
        store: &mut Store,
        adapter: &mut PagedKVCacheAdapter,
        deadline: Instant,
    ) -> bool {
        let mut complete = true;
        if self.persistent {
            loop {
                self.take_landings(store);
                self.judge_owed(store, Instant::now());
                self.top_up_all(adapter);
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
    pub(super) fn dispose(&mut self, store: &mut Store, adapter: &mut PagedKVCacheAdapter) {
        let by = Instant::now() + SLOT_WAIT;
        self.flush_points(store, adapter, by);
        self.closing_by = Some(by);
    }

    /// The draft segment keys of the state of `tokens` (keyed by `keys` and
    /// `salt`) at `row`, when its KV blocks past the prefix cache's hit and,
    /// on a load with the MTP head (`draft_window`), its draft segments are
    /// all on disk.
    pub(super) fn chain_whole(
        &self,
        tokens: &[u32],
        keys: &[Vec<u64>],
        salt: u64,
        row: usize,
        adapter: &PagedKVCacheAdapter,
        draft_window: Option<usize>,
    ) -> Option<Vec<ColdCacheKey>> {
        let n = row / BLOCK_ROWS;
        if n == 0 || !row.is_multiple_of(BLOCK_ROWS) || keys.len() < n || tokens.len() < row {
            return None;
        }
        let cold = chain(ColdGroup::Kv, self.fingerprint, tokens, keys, salt, n);
        let hot = crate::models::qwen3_5::gdn_checkpoint_store::compute_paged_prefix_block_hashes(
            tokens,
            row as u32,
            BLOCK_ROWS as u32,
            keys,
            salt,
        )
        .unwrap_or_default();
        let cached = hot
            .iter()
            .take_while(|&&hash| adapter.prefix_cached(hash))
            .count();
        let kv_whole = hot.len() == n
            && cold.len() == n
            && cold[cached.min(n)..]
                .iter()
                .all(|key| self.manager.contains(key));
        if !kv_whole {
            return None;
        }
        match draft_window {
            Some(window) if window > 0 => {
                let count =
                    super::memory::cached_draft_rows(row as u64, window as u64) as usize / window;
                let draft_chain = chain(
                    ColdGroup::HybridDraft,
                    self.fingerprint,
                    tokens,
                    keys,
                    salt,
                    n,
                );
                (0..count)
                    .map(|k| segment_key(self.fingerprint, &draft_chain, tokens, window, k))
                    .collect::<Option<Vec<_>>>()
                    .filter(|keys| {
                        keys.iter()
                            .all(|key| self.manager.contains_in(key, ColdGroup::HybridDraft))
                    })
            }
            _ => Some(Vec::new()),
        }
    }

    /// Entry `id`, a state that did not come back, goes with its disk
    /// object.
    pub(super) fn discard(&mut self, store: &mut Store, id: EntryId) {
        if let Some(copy) = store.get(id).and_then(|entry| entry.disk()).copied() {
            self.remove_object(&copy.key);
        }
        store.remove(id);
        Telemetry::bump(&self.telemetry.dropped_chain);
    }

    /// A persistent tier's restart (Splash `Cache::adopt`): with an empty
    /// store, each state record of this load's shape whose own object, whole
    /// KV chain and, with the MTP head, draft segments are on disk comes back
    /// as a state on disk only, stamped as its earlier process left it. How
    /// many came back.
    pub(super) fn adopt(&mut self, store: &mut Store, config: &super::config::Config) -> u64 {
        if !self.persistent || store.entries().next().is_some() {
            return 0;
        }
        let tier = Arc::clone(&self.closer.tier);
        let (spans, geometry) = (self.lane.spans.clone(), self.lane.geometry.clone());
        let (grid, has_mtp, fingerprint) = (self.lane.grid, self.lane.has_mtp, self.fingerprint);
        let manager = Arc::clone(&self.manager);
        let adopted = adopt_states(store, &tier, &spans, &geometry, |label| {
            let row = label.tokens.len();
            if has_mtp {
                let lineage = Lineage {
                    tokens: &label.tokens,
                    extra_keys: &label.extra_keys,
                    salt: label.salt,
                };
                let draft_chain = lineage_keys(
                    ColdGroup::HybridDraft,
                    fingerprint,
                    lineage,
                    BLOCK_ROWS as u32,
                );
                let count = super::memory::cached_draft_rows(row as u64, grid as u64) as usize
                    / grid.max(1);
                for k in 0..count {
                    let key = segment_key(fingerprint, &draft_chain, &label.tokens, grid, k)?;
                    if !manager.contains_in(&key, ColdGroup::HybridDraft) {
                        return None;
                    }
                }
            }
            Some((
                super::memory::state_bytes(config, row as u64, grid as u64, has_mtp),
                u32::try_from(row).ok()?,
            ))
        });
        self.telemetry.adopted.fetch_add(adopted, Ordering::Relaxed);
        if adopted > 0 {
            tracing::info!("qwen4_exp: took back {adopted} stored states from the SSD cache");
        }
        adopted
    }

    /// The state object `key` at `row` read back through the staging buffer,
    /// its label's pass keys `extra_keys`, with its draft `segments`: a load
    /// of `layers` layers with the prefill window `grid`, its draft part
    /// when `draft` (the MTP head). `bound` caps every segment read. No
    /// buffer within `SLOT_WAIT` leaves the object; one that does not read
    /// back is removed.
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
            let mut segments = Vec::with_capacity(segment_keys.len());
            for (k, key) in segment_keys.iter().enumerate() {
                let segment = self.manager.load_object(
                    *key,
                    self.fingerprint,
                    ColdGroup::HybridDraft,
                    bound,
                )?;
                if !layout_fits(
                    &segment.layout,
                    ColdGroup::HybridDraft,
                    (k + 1) * grid,
                    k + 1,
                    grid,
                ) {
                    return None;
                }
                let (window, segment) = decode_segment(segment.tensors.first()?)?;
                if window != grid {
                    return None;
                }
                segments.push(segment);
            }
            let cell = decode_cell(&staging, layers, draft, segments)?;
            let arrays: Vec<&MxArray> = cell.arrays().collect();
            if !arrays.is_empty() {
                MxArray::eval_arrays(&arrays).ok()?;
            }
            (cell.grid == grid).then_some(cell)
        })();
        self.lane.staging = Some(staging);
        match read {
            Some(cell) => {
                Telemetry::bump(&self.telemetry.disk_restored);
                Read::Cell(cell)
            }
            None => {
                self.remove_object(&key);
                Telemetry::bump(&self.telemetry.dropped_chain);
                Read::Failed
            }
        }
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

/// One cached block's bytes, under its cold key. `None` when the prefix
/// cache no longer holds it or the pool cannot read it.
fn capture_block(
    adapter: &PagedKVCacheAdapter,
    hot: u64,
    key: ColdCacheKey,
    fingerprint: ColdCacheFingerprint,
    tokens: &[u32],
) -> Option<ColdCacheBlock> {
    let block = adapter.cached_block(hot)?;
    let pool = adapter.layer_kv_pool();
    let layers: Vec<ColdLayerBlock> = pool
        .read_block_all_layers(block.block_id)
        .ok()?
        .into_iter()
        .map(|(keys, values)| ColdLayerBlock { keys, values })
        .collect();
    let first = layers.first()?;
    Some(ColdCacheBlock {
        key,
        fingerprint,
        tokens: tokens.to_vec(),
        layout: ColdCacheLayout {
            block_size: pool.block_size(),
            num_layers: pool.num_layers() as u32,
            num_kv_heads: pool.config().num_kv_heads,
            head_size: pool.config().head_size,
            cache_dtype: format!("{:?}", pool.cache_dtype()),
            key_bytes_per_layer: first.keys.len(),
            value_bytes_per_layer: first.values.len(),
        },
        layers,
    })
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
