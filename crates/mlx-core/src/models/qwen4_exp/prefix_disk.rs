//! The disk tier of the warm prefix states: a turn's replay state and the KV
//! blocks of its chain survive an engine restart (Splash d8af1054
//! `WriteBehind`, docs/server.md "state persistence").
//!
//! Everything goes through the paged cold store (`ColdCacheManager`: temp
//! file, fsync, rename, payload checksum, identity check, quota, LRU). A KV
//! block is a `Kv` object under its chain key (tokens, pass keys, salt), a
//! draft segment a `HybridDraft` object, a state a `HybridCell` object, all
//! under one fingerprint of everything that changes the math.
//!
//! The model thread copies to host memory only: a published state's blocks
//! and draft segments at its publication, the replay state at the turn's
//! end. A parker thread writes them, the state last and only once every
//! object of its chain is on disk, `delay` after the turn ended unless the
//! conversation went on. A restart restores the deepest state of the prompt
//! whose whole chain is on disk; any miss prefills as before.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use mlx_paged_attn::{
    ColdCacheBlock, ColdCacheFingerprint, ColdCacheKey, ColdCacheLayout, ColdCacheManager,
    ColdGroup, ColdLayerBlock, ColdObject, ColdSidecar, ColdSidecarLayout,
};
use napi::{Error, Result};

use super::prefix_state::{BLOCK_ROWS, Cell, DraftCell, LayerCell, Segment};
use crate::array::{DType, MxArray};
use crate::transformer::paged_kv_cache_adapter::{ColdTierContext, PagedKVCacheAdapter, SeqId};

/// Payload format of a state and a segment object.
const FORMAT: u32 = 2;
const MAGIC: &[u8; 4] = b"Q4PS";
/// Splash `WriteBehind` hands a point over this long after publication.
pub(super) const DELAY: Duration = Duration::from_secs(10);
/// A parked state whose chain is still not on disk this long after its
/// delay is dropped.
const GIVE_UP: Duration = Duration::from_secs(10);
/// The wait for a writer slot of one object on the parker thread; an object
/// that finds none stays first in line.
const SLOT_WAIT: Duration = Duration::from_secs(2);
/// A decode step copies blocks out at most this often: a copy waits for the
/// GPU.
const STEP_TOP_UP: Duration = Duration::from_millis(250);
/// States on disk take at most this, or a quarter of the store's quota.
const BUDGET_CAP: u64 = 16 << 30;
/// Host copies of blocks and segments waiting for the writer.
pub(super) const QUEUE_BYTES: u64 = 128 << 20;
/// Parked states, the one being written included: one per lane.
pub(super) const PARCELS: usize = super::LANES;
const STATE_GROUPS: [ColdGroup; 2] = [ColdGroup::HybridCell, ColdGroup::HybridDraft];

/// Counters of the stored states and of their disk tier, read by
/// `residencyInfo`.
#[derive(Default)]
pub(super) struct Telemetry {
    pub(super) states: AtomicU64,
    pub(super) state_bytes: AtomicU64,
    pub(super) published: AtomicU64,
    pub(super) restored: AtomicU64,
    pub(super) disk_enabled: AtomicU64,
    /// States parked and not handed to the writer yet.
    pub(super) waiting: AtomicU64,
    /// States handed over and found in the store's index.
    pub(super) durable: AtomicU64,
    /// Parked states a newer state of their conversation replaced.
    pub(super) unneeded: AtomicU64,
    /// States not written: chain incomplete, budget, queue, encoding.
    pub(super) refused: AtomicU64,
    pub(super) disk_restored: AtomicU64,
    /// States on disk removed because their chain was not.
    pub(super) dropped_chain: AtomicU64,
    /// Blocks and segments copied out and not handed to the writer yet.
    pub(super) owed: AtomicU64,
    pub(super) state_bytes_written: AtomicU64,
}

impl Telemetry {
    pub(super) fn sync(&self, states: &super::prefix_state::States) {
        let bytes = states.store.bytes();
        self.states.store(bytes.entries as u64, Ordering::Relaxed);
        self.state_bytes.store(bytes.state_bytes, Ordering::Relaxed);
        self.published
            .store(states.counters.published, Ordering::Relaxed);
        self.restored
            .store(states.counters.restored, Ordering::Relaxed);
    }

    fn bump(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
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

struct Writer(Vec<u8>);

impl Writer {
    fn u8(&mut self, value: u8) {
        self.0.push(value);
    }
    fn u32(&mut self, value: u32) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }
    fn u64(&mut self, value: u64) {
        self.0.extend_from_slice(&value.to_le_bytes());
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
/// into one buffer: the memory plan books one copy of a state for it.
fn encode(
    grid: usize,
    index_blocks: &[usize],
    draft: bool,
    arrays: &[Option<&MxArray>],
) -> Result<Option<Vec<u8>>> {
    let mut out = Writer(Vec::new());
    out.0.extend_from_slice(MAGIC);
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
            return Ok(None);
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
    out.0.reserve_exact(data);
    for (array, code) in arrays.iter().zip(codes) {
        let Some(array) = array else {
            continue;
        };
        if code == 3 {
            for value in array.to_float32()?.iter() {
                out.0.extend_from_slice(&value.to_le_bytes());
            }
        } else {
            for value in array.to_uint16_native()? {
                out.0.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    Ok(Some(out.0))
}

/// The inverse of [`encode`]: (grid, index blocks, draft flag, arrays).
/// `None` for any byte that does not fit the format.
type Decoded = (usize, Vec<usize>, bool, Vec<Option<MxArray>>);

fn decode(bytes: &[u8], arrays: usize) -> Option<Decoded> {
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
    (reader.at == bytes.len()).then_some((grid, index_blocks, draft, out))
}

/// A state's bytes: each layer's conv, recurrent and PLE history, then the
/// draft's waiting rows and seed. The segments are objects of their own.
fn encode_cell(cell: &Cell) -> Result<Option<Vec<u8>>> {
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
    let blocks: Vec<usize> = cell.layers.iter().map(|layer| layer.index_blocks).collect();
    encode(cell.grid, &blocks, cell.draft.is_some(), &arrays)
}

/// The state of [`encode_cell`] with `segments` as its draft passes;
/// `None` when the bytes are not a state of `layers` layers.
fn decode_cell(bytes: &[u8], layers: usize, draft: bool, segments: Vec<Segment>) -> Option<Cell> {
    let arrays = layers * 3 + if draft { 2 } else { 0 };
    let (grid, blocks, has_draft, mut arrays) = decode(bytes, arrays)?;
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
    let (grid, blocks, draft, arrays) = decode(bytes, 3)?;
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
// The parker thread.
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

struct Parcel {
    seq: SeqId,
    state: ColdSidecar,
    /// Every object of the state's chain: its KV blocks and draft segments.
    needs: Vec<(ColdCacheKey, ColdGroup)>,
    due: Instant,
    bytes: u64,
}

#[derive(Default)]
struct Slot {
    /// Block and segment copies, first in line first.
    items: VecDeque<ColdObject>,
    item_bytes: u64,
    parcels: Vec<Parcel>,
    /// The thread has a parcel with the writer until its write ended.
    handing: bool,
    flush: u64,
    flushed: u64,
    flush_until: Option<Instant>,
    /// The last flush for which the thread waited out the writes it handed.
    drained: u64,
    stop: bool,
    #[cfg(test)]
    hold: bool,
    #[cfg(test)]
    slot_wait: Option<Duration>,
}

impl Slot {
    /// Parked states, the one with the writer included.
    fn states(&self) -> usize {
        self.parcels.len() + usize::from(self.handing)
    }

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

    /// Hands every queued object and every parked state whose chain landed
    /// to the writer and waits for them, until `deadline`. Returns as soon
    /// as nothing more can land: a state whose chain is not on disk once
    /// the handed writes are does not hold the caller.
    fn flush(&self, deadline: Instant) {
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
    }
}

/// Whether every object of `parcel`'s chain is in the store's index.
fn chain_landed(manager: &ColdCacheManager, parcel: &Parcel) -> bool {
    parcel
        .needs
        .iter()
        .all(|(key, group)| manager.contains_in(key, *group))
}

enum Handed {
    /// On disk, these bytes.
    Durable(u64),
    Refused,
    /// No writer slot freed: still parked.
    Busy(Box<Parcel>),
}

/// Hands a parcel whose chain landed to the writer, the oldest states going
/// first to keep the budget, and waits until the writer is past it.
fn hand_over(
    manager: &ColdCacheManager,
    parker: &Parker,
    parcel: Parcel,
    budget: u64,
    wait: Duration,
) -> Handed {
    if parcel.bytes > budget {
        return Handed::Refused;
    }
    let chain: Vec<ColdCacheKey> = parcel.needs.iter().map(|(key, _)| *key).collect();
    manager.touch(&chain);
    while manager
        .group_bytes(&STATE_GROUPS)
        .saturating_add(parcel.bytes)
        > budget
    {
        let Some((key, group)) = manager.oldest_in(&STATE_GROUPS) else {
            return Handed::Refused;
        };
        if !manager.remove_object(key, group) {
            return Handed::Refused;
        }
    }
    let Parcel {
        seq,
        state,
        needs,
        due,
        bytes,
    } = parcel;
    let key = state.key;
    match manager.offer(ColdObject::Sidecar(state), Instant::now() + wait) {
        Ok(None) => {}
        Ok(Some(ColdObject::Sidecar(state))) => {
            return Handed::Busy(Box::new(Parcel {
                seq,
                state,
                needs,
                due,
                bytes,
            }));
        }
        _ => return Handed::Refused,
    }
    let since = Instant::now();
    loop {
        if manager.contains_in(&key, ColdGroup::HybridCell) {
            return Handed::Durable(bytes);
        }
        let drained = manager.drain(Duration::from_millis(500));
        if drained || parker.lock().stop || since.elapsed() >= GIVE_UP {
            return if manager.contains_in(&key, ColdGroup::HybridCell) {
                Handed::Durable(bytes)
            } else {
                Handed::Refused
            };
        }
    }
}

fn run(
    parker: Arc<Parker>,
    manager: Arc<ColdCacheManager>,
    budget: Arc<AtomicU64>,
    telemetry: Arc<Telemetry>,
) {
    let mut slot = parker.lock();
    loop {
        if slot.stop {
            break;
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
        let mut ready = None;
        let mut expired = None;
        if !slot.handing {
            for (at, parcel) in slot.parcels.iter().enumerate() {
                if !flushing && now < parcel.due {
                    continue;
                }
                if chain_landed(&manager, parcel) {
                    ready = Some(at);
                    break;
                }
                if now >= parcel.due + GIVE_UP {
                    expired = Some(at);
                }
            }
        }
        if let Some(at) = ready {
            let parcel = slot.parcels.remove(at);
            slot.handing = true;
            let wait = slot.slot_wait();
            drop(slot);
            let handed = hand_over(
                &manager,
                &parker,
                parcel,
                budget.load(Ordering::Relaxed),
                wait,
            );
            slot = parker.lock();
            slot.handing = false;
            match handed {
                Handed::Durable(bytes) => {
                    telemetry.waiting.fetch_sub(1, Ordering::Relaxed);
                    Telemetry::bump(&telemetry.durable);
                    telemetry
                        .state_bytes_written
                        .fetch_add(bytes, Ordering::Relaxed);
                }
                Handed::Refused => {
                    telemetry.waiting.fetch_sub(1, Ordering::Relaxed);
                    Telemetry::bump(&telemetry.refused);
                }
                Handed::Busy(parcel) => {
                    let at = at.min(slot.parcels.len());
                    slot.parcels.insert(at, *parcel);
                }
            }
            continue;
        }
        if let Some(at) = expired {
            slot.parcels.remove(at);
            telemetry.waiting.fetch_sub(1, Ordering::Relaxed);
            Telemetry::bump(&telemetry.refused);
            continue;
        }
        if flushing && slot.items.is_empty() && !slot.handing {
            if !slot.parcels.is_empty() && slot.drained < slot.flush {
                // Blocks handed over may still be in the writer's queue: a
                // chain is judged once they are written.
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
        let chain_owed = slot.parcels.iter().any(|parcel| now >= parcel.due);
        let next = slot
            .parcels
            .iter()
            .map(|parcel| parcel.due)
            .filter(|&due| due > now)
            .min()
            .map(|due| due.saturating_duration_since(now));
        let wait = match (chain_owed, next) {
            (true, _) => Duration::from_millis(20),
            (false, Some(next)) => next,
            (false, None) => Duration::from_secs(3600),
        };
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
    /// `HybridCell` chain of the lineage's blocks.
    cell_chain: Vec<ColdCacheKey>,
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

/// A state of the prompt on disk whose chain is whole.
pub(super) struct Candidate {
    pub(super) row: usize,
    pub(super) key: ColdCacheKey,
    pub(super) segments: Vec<ColdCacheKey>,
}

pub(super) struct Disk {
    manager: Arc<ColdCacheManager>,
    fingerprint: ColdCacheFingerprint,
    #[cfg(test)]
    budget: Arc<AtomicU64>,
    pub(super) delay: Duration,
    parker: Arc<Parker>,
    thread: Option<std::thread::JoinHandle<()>>,
    hook: u64,
    cursors: HashMap<SeqId, Cursor>,
    telemetry: Arc<Telemetry>,
    skip_once: bool,
    /// The next decode step that may copy blocks out.
    step_after: Instant,
    #[cfg(test)]
    pub(super) encoded: usize,
    #[cfg(test)]
    pub(super) step_runs: usize,
}

impl Disk {
    /// The tier over `manager` under `fingerprint`, its parker started and
    /// its flush registered with `cold_cache_drain`.
    pub(super) fn open(
        manager: Arc<ColdCacheManager>,
        fingerprint: ColdCacheFingerprint,
        telemetry: Arc<Telemetry>,
    ) -> Result<Self> {
        let budget = Arc::new(AtomicU64::new((manager.quota_bytes() / 4).min(BUDGET_CAP)));
        let parker = Arc::new(Parker {
            slot: Mutex::new(Slot::default()),
            wake: Condvar::new(),
            done: Condvar::new(),
        });
        let thread = {
            let (parker, manager, budget, telemetry) = (
                Arc::clone(&parker),
                Arc::clone(&manager),
                Arc::clone(&budget),
                Arc::clone(&telemetry),
            );
            std::thread::Builder::new()
                .name("qwen4-prefix-parker".to_string())
                .spawn(move || run(parker, manager, budget, telemetry))
                .map_err(|e| Error::from_reason(format!("Qwen4 prefix disk: {e}")))?
        };
        let hook = {
            let parker = Arc::clone(&parker);
            crate::cold_tier::register_drain_hook(Arc::new(move |deadline| parker.flush(deadline)))
        };
        telemetry.disk_enabled.store(1, Ordering::Relaxed);
        Ok(Self {
            manager,
            fingerprint,
            #[cfg(test)]
            budget,
            delay: DELAY,
            parker,
            thread: Some(thread),
            hook,
            cursors: HashMap::new(),
            telemetry,
            skip_once: false,
            step_after: Instant::now(),
            #[cfg(test)]
            encoded: 0,
            #[cfg(test)]
            step_runs: 0,
        })
    }

    /// The adapter restores the KV blocks of a state's chain from this
    /// store; its prefix lookups run only below a state's row.
    pub(super) fn attach(&self, adapter: &mut PagedKVCacheAdapter) {
        adapter.set_cold_tier(ColdTierContext {
            manager: Arc::clone(&self.manager),
            fingerprint: self.fingerprint,
            sidecar_policy: None,
        });
    }

    #[cfg(test)]
    pub(super) fn set_budget(&self, budget: u64) {
        self.budget.store(budget, Ordering::Relaxed);
    }

    /// Holds the block and segment copies back from the writer (`true`)
    /// until released.
    #[cfg(test)]
    pub(super) fn hold(&self, hold: bool) {
        self.parker.lock().hold = hold;
        self.parker.wake.notify_all();
    }

    #[cfg(test)]
    pub(super) fn flush(&self, deadline: Instant) {
        self.parker.flush(deadline);
    }

    /// The wait for a writer slot, `Duration::ZERO` for one try.
    #[cfg(test)]
    pub(super) fn set_slot_wait(&self, wait: Duration) {
        self.parker.lock().slot_wait = Some(wait);
    }

    /// The bytes of every parked state.
    #[cfg(test)]
    pub(super) fn parked_bytes(&self) -> Vec<u64> {
        self.parker
            .lock()
            .parcels
            .iter()
            .map(|parcel| parcel.bytes)
            .collect()
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
        let blocks = hot.len().min(keys.len()).min(lineage.len() / BLOCK_ROWS);
        let lineage = lineage[..blocks * BLOCK_ROWS].to_vec();
        let fingerprint = self.fingerprint;
        let keyed = |group| chain(group, fingerprint, &lineage, keys, salt, blocks);
        let (cold, draft_chain, cell_chain) = (
            keyed(ColdGroup::Kv),
            keyed(ColdGroup::HybridDraft),
            keyed(ColdGroup::HybridCell),
        );
        self.cursors.insert(
            seq,
            Cursor {
                hot: hot[..blocks].to_vec(),
                lineage,
                cold,
                draft_chain,
                cell_chain,
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
    /// cache, its draft segments are `draft`'s.
    pub(super) fn published(&mut self, seq: SeqId, row: usize, draft: Option<&DraftCell>) {
        let Some(cursor) = self.cursors.get_mut(&seq) else {
            return;
        };
        cursor.registered = cursor.registered.max(row);
        if let Some(draft) = draft
            && draft.segments.len() > cursor.pending.len()
        {
            cursor.pending = draft.segments.clone();
        }
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
        if !self.owes(seq) {
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
    pub(super) fn top_up_all(&mut self, adapter: &mut PagedKVCacheAdapter) {
        let owing: Vec<SeqId> = self
            .cursors
            .keys()
            .copied()
            .filter(|&seq| self.owes(seq))
            .collect();
        for seq in owing {
            self.top_up(seq, adapter);
        }
        let done: Vec<SeqId> = self
            .cursors
            .iter()
            .filter(|(_, cursor)| cursor.closed)
            .map(|(&seq, _)| seq)
            .filter(|&seq| !self.owes(seq))
            .collect();
        for seq in done {
            self.cursors.remove(&seq);
        }
    }

    /// [`Self::top_up_all`] from a decode step, at most once per
    /// `STEP_TOP_UP`.
    pub(super) fn top_up_step(&mut self, adapter: &mut PagedKVCacheAdapter) {
        let now = Instant::now();
        if now < self.step_after {
            return;
        }
        self.step_after = now + STEP_TOP_UP;
        #[cfg(test)]
        {
            self.step_runs += 1;
        }
        self.top_up_all(adapter);
    }

    /// `seq`'s turn ended with its replay state `cell` at `row`: whatever
    /// of its chain is still owed is copied out, and the state is parked for
    /// the writer, replacing a parked state of the same sequence. Refused
    /// before any copy while every parcel is taken. Never waits for the
    /// writer.
    pub(super) fn park(
        &mut self,
        seq: SeqId,
        cell: &Cell,
        row: usize,
        adapter: &mut PagedKVCacheAdapter,
    ) {
        self.top_up(seq, adapter);
        let Some(cursor) = self.cursors.get_mut(&seq) else {
            return;
        };
        cursor.closed = true;
        let window = cell.grid;
        let blocks = row / BLOCK_ROWS;
        let segments = cell.draft.as_ref().map_or(0, |draft| draft.segments.len());
        if cursor.broken
            || blocks == 0
            || blocks > cursor.cell_chain.len()
            || cursor.registered < row
            || window != cursor.window
        {
            Telemetry::bump(&self.telemetry.refused);
            return;
        }
        let mut needs: Vec<(ColdCacheKey, ColdGroup)> = cursor.cold[..blocks]
            .iter()
            .map(|&key| (key, ColdGroup::Kv))
            .collect();
        for k in 0..segments {
            let Some(key) = segment_key(
                self.fingerprint,
                &cursor.draft_chain,
                &cursor.lineage,
                window,
                k,
            ) else {
                Telemetry::bump(&self.telemetry.refused);
                return;
            };
            needs.push((key, ColdGroup::HybridDraft));
        }
        let key = cursor.cell_chain[blocks - 1];
        {
            // The replaced state goes before the new one is copied out, so a
            // sequence never holds two.
            let mut slot = self.parker.lock();
            if let Some(at) = slot.parcels.iter().position(|old| old.seq == seq) {
                slot.parcels.remove(at);
                self.telemetry.waiting.fetch_sub(1, Ordering::Relaxed);
                Telemetry::bump(&self.telemetry.unneeded);
            } else if slot.states() >= PARCELS {
                Telemetry::bump(&self.telemetry.refused);
                return;
            }
        }
        let layers = cell.layers.len();
        #[cfg(test)]
        {
            self.encoded += 1;
        }
        let state = encode_cell(cell).ok().flatten().and_then(|payload| {
            Some(ColdSidecar {
                key,
                fingerprint: self.fingerprint,
                layout: layout(ColdGroup::HybridCell, row, layers, window, payload.len())?,
                tensors: vec![payload],
            })
        });
        let Some(state) = state else {
            Telemetry::bump(&self.telemetry.refused);
            return;
        };
        let bytes = state.layout.bytes_per_tensor as u64;
        let now = Instant::now();
        let mut slot = self.parker.lock();
        slot.parcels.push(Parcel {
            seq,
            state,
            needs,
            due: now + self.delay,
            bytes,
        });
        // Once every parcel is taken the oldest is due at once, so one frees
        // up for the next turn's end.
        if slot.states() >= PARCELS
            && let Some(oldest) = slot.parcels.first_mut()
        {
            oldest.due = oldest.due.min(now);
        }
        self.telemetry.waiting.fetch_add(1, Ordering::Relaxed);
        self.parker.wake.notify_all();
    }

    /// The deepest state of a `tokens` prompt above `floor` on disk, at one
    /// of its block-aligned pass ends `ends` before its last token, whose KV
    /// blocks past the prefix cache's hit and whose draft segments are all on
    /// disk. A state whose chain is not is removed. At most two states are
    /// tried.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn candidate(
        &self,
        tokens: &[u32],
        ends: &[usize],
        keys: &[Vec<u64>],
        salt: u64,
        floor: usize,
        adapter: &PagedKVCacheAdapter,
        draft_window: Option<usize>,
    ) -> Option<Candidate> {
        let blocks = keys.len().min(tokens.len() / BLOCK_ROWS);
        let rows: Vec<usize> = ends
            .iter()
            .rev()
            .copied()
            .filter(|&row| {
                row > floor
                    && row < tokens.len()
                    && row.is_multiple_of(BLOCK_ROWS)
                    && row / BLOCK_ROWS <= blocks
            })
            .collect();
        if rows.is_empty() {
            return None;
        }
        let cell_chain = chain(
            ColdGroup::HybridCell,
            self.fingerprint,
            tokens,
            keys,
            salt,
            blocks,
        );
        let mut cold: Option<Vec<ColdCacheKey>> = None;
        let mut tries = 0;
        for row in rows {
            let n = row / BLOCK_ROWS;
            let key = cell_chain[n - 1];
            if !self.manager.contains_in(&key, ColdGroup::HybridCell) {
                continue;
            }
            tries += 1;
            let cold = cold.get_or_insert_with(|| {
                chain(ColdGroup::Kv, self.fingerprint, tokens, keys, salt, blocks)
            });
            let hot =
                crate::models::qwen3_5::gdn_checkpoint_store::compute_paged_prefix_block_hashes(
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
                && cold[cached.min(n)..n]
                    .iter()
                    .all(|key| self.manager.contains(key));
            let segments = match draft_window {
                Some(window) if window > 0 => {
                    let count = super::memory::cached_draft_rows(row as u64, window as u64)
                        as usize
                        / window;
                    let draft_chain = chain(
                        ColdGroup::HybridDraft,
                        self.fingerprint,
                        tokens,
                        keys,
                        salt,
                        blocks,
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
            };
            match (kv_whole, segments) {
                (true, Some(segments)) => {
                    return Some(Candidate { row, key, segments });
                }
                _ => {
                    self.manager.remove_object(key, ColdGroup::HybridCell);
                    Telemetry::bump(&self.telemetry.dropped_chain);
                }
            }
            if tries == 2 {
                break;
            }
        }
        None
    }

    /// The state of `candidate` read back: a load of `layers` layers with
    /// the prefill window `grid`, its draft part when `draft` (the MTP head).
    /// `bound` caps every read. A state that does not fit is removed.
    pub(super) fn load(
        &self,
        candidate: &Candidate,
        layers: usize,
        grid: usize,
        draft: bool,
        bound: u64,
    ) -> Option<Cell> {
        let cell = (|| {
            let object = self.manager.load_object(
                candidate.key,
                self.fingerprint,
                ColdGroup::HybridCell,
                bound,
            )?;
            if !layout_fits(
                &object.layout,
                ColdGroup::HybridCell,
                candidate.row,
                layers,
                grid,
            ) {
                return None;
            }
            let mut segments = Vec::with_capacity(candidate.segments.len());
            for (k, key) in candidate.segments.iter().enumerate() {
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
            let cell = decode_cell(object.tensors.first()?, layers, draft, segments)?;
            let arrays: Vec<&MxArray> = cell.arrays().collect();
            if !arrays.is_empty() {
                MxArray::eval_arrays(&arrays).ok()?;
            }
            (cell.grid == grid).then_some(cell)
        })();
        match cell {
            Some(cell) => {
                Telemetry::bump(&self.telemetry.disk_restored);
                Some(cell)
            }
            None => {
                self.manager
                    .remove_object(candidate.key, ColdGroup::HybridCell);
                Telemetry::bump(&self.telemetry.dropped_chain);
                None
            }
        }
    }
}

impl Drop for Disk {
    fn drop(&mut self) {
        crate::cold_tier::unregister_drain_hook(self.hook);
        self.parker.flush(Instant::now() + SLOT_WAIT);
        {
            let mut slot = self.parker.lock();
            slot.stop = true;
            self.parker.wake.notify_all();
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
        b"qwen4-prefix-v2".to_vec(),
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
