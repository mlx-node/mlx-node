//! Flat-lane payloads of the DFlash2 state store. A parked end-of-turn
//! state owns the flat caches and draft window it was moved out of; nothing
//! else holds a handle a later turn could write through.

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::transformer::KVCache;

use super::super::arrays_cache::ArraysCache;
use super::super::dflash2::DFlash2ContextCache;
use super::super::gated_delta_net::GdnCell;
use super::super::layer_cache::Qwen3_5LayerCache;
use super::store::{Anchor, Backing, HybridStateStore, KvSource, Shrink, StateBytes};

/// Capacity slack (rows past the used ones) above which a parked KV buffer
/// is worth copying down.
pub(crate) const COMPACT_SLACK_ROWS: usize = 4096;
/// Rows a compacted buffer keeps past the used ones.
pub(crate) const COMPACT_SPARE_ROWS: usize = 256;

fn arrays_bytes<'a>(arrays: impl IntoIterator<Item = &'a MxArray>) -> u64 {
    arrays
        .into_iter()
        .fold(0, |sum, array| sum.saturating_add(array.nbytes() as u64))
}

fn kv_cache_bytes(kv: &KVCache) -> u64 {
    arrays_bytes(
        [
            kv.keys_ref(),
            kv.values_ref(),
            kv.key_scales_ref(),
            kv.value_scales_ref(),
        ]
        .into_iter()
        .flatten(),
    )
}

/// The full-attention caches of a parked state, each with its decoder-layer
/// index.
pub(crate) struct FlatKv {
    layers: Vec<(usize, KVCache)>,
}

impl FlatKv {
    pub(crate) fn layers(&self) -> &[(usize, KVCache)] {
        &self.layers
    }

    /// Drops rows past `rows` from every layer's offset; the buffers stay.
    pub(crate) fn trim(&mut self, rows: usize) {
        let rows = i32::try_from(rows).unwrap_or(i32::MAX);
        for (_, kv) in &mut self.layers {
            kv.trim(rows);
        }
    }
}

impl KvSource for FlatKv {
    fn rows(&self) -> usize {
        self.layers
            .iter()
            .map(|(_, kv)| kv.get_offset().max(0) as usize)
            .min()
            .unwrap_or(0)
    }

    fn bytes(&self) -> u64 {
        self.layers
            .iter()
            .fold(0, |sum, (_, kv)| sum.saturating_add(kv_cache_bytes(kv)))
    }

    fn shrink_plan(&self, rows: usize) -> Option<Shrink> {
        let keep = rows.min(self.rows());
        let mut plan = Shrink {
            frees: 0,
            transient: 0,
        };
        for (_, kv) in &self.layers {
            let capacity = usize::try_from(kv.capacity().ok()?).ok()?;
            if capacity.saturating_sub(keep) <= COMPACT_SLACK_ROWS {
                continue;
            }
            let bytes = kv_cache_bytes(kv);
            let kept = (bytes / capacity as u64).saturating_mul((keep + COMPACT_SPARE_ROWS) as u64);
            plan.frees = plan.frees.saturating_add(bytes.saturating_sub(kept));
            plan.transient = plan.transient.saturating_add(kept);
        }
        (plan.frees > 0).then_some(plan)
    }

    fn shrink(&mut self, rows: usize) -> Result<()> {
        let keep = rows.min(self.rows());
        for (_, kv) in &mut self.layers {
            kv.trim(keep as i32);
            let capacity = usize::try_from(kv.capacity()?).unwrap_or(0);
            if capacity.saturating_sub(keep) > COMPACT_SLACK_ROWS {
                kv.compact((keep + COMPACT_SPARE_ROWS) as i64)?;
            }
        }
        Ok(())
    }
}

/// GDN slots and the draft window at the end of a turn. The next decode
/// position is the window's logical length.
pub(crate) struct FlatEnd {
    linear: Vec<(usize, ArraysCache)>,
    context: DFlash2ContextCache,
}

impl FlatEnd {
    pub(crate) fn context(&self) -> &DFlash2ContextCache {
        &self.context
    }

    pub(crate) fn next_position(&self) -> i32 {
        self.context.logical_len()
    }
}

#[cfg(test)]
impl FlatEnd {
    /// Every GDN slot in layer order, then the draft window's arrays.
    pub(crate) fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        for (_, cache) in &self.linear {
            out.extend(cache.get(0));
            out.extend(cache.get(1));
        }
        self.context.collect_arrays(out);
    }
}

impl StateBytes for FlatEnd {
    fn state_bytes(&self) -> u64 {
        let mut arrays = Vec::new();
        for (_, cache) in &self.linear {
            arrays.extend(cache.get(0));
            arrays.extend(cache.get(1));
        }
        self.context.collect_arrays(&mut arrays);
        arrays_bytes(arrays)
    }
}

/// GDN cell and draft window snapshot at a block-aligned row.
pub(crate) struct FlatAnchor {
    pub(crate) cell: GdnCell,
    pub(crate) window: DFlash2ContextCache,
}

impl StateBytes for FlatAnchor {
    fn state_bytes(&self) -> u64 {
        let mut arrays: Vec<&MxArray> = self.cell.conv.iter().chain(&self.cell.rec).collect();
        self.window.collect_arrays(&mut arrays);
        arrays_bytes(arrays)
    }
}

pub(crate) struct FlatBacking;

impl Backing for FlatBacking {
    type Kv = FlatKv;
    type End = FlatEnd;
    type Anchor = FlatAnchor;
}

/// One loaded model's parked flat states, plus what a park of its live flat
/// state takes along: the cache salt of the turn that built it (its key) and
/// the capture C(R) on its lineage, whose KV rows are the live ones.
pub(crate) struct FlatStates {
    pub(crate) store: HybridStateStore<FlatBacking>,
    pub(crate) live_salt: u64,
    pub(crate) live_anchor: Option<Anchor<FlatAnchor>>,
}

impl FlatStates {
    pub(crate) fn new(block_size: u32) -> Self {
        Self {
            store: HybridStateStore::new(block_size),
            live_salt: 0,
            live_anchor: None,
        }
    }

    /// Drops every parked state and the live state's anchor.
    pub(crate) fn clear(&mut self) {
        self.store.clear();
        self.live_anchor = None;
    }

    /// Moves the store out, leaving an empty one with the same geometry.
    pub(crate) fn take_store(&mut self) -> HybridStateStore<FlatBacking> {
        let empty =
            HybridStateStore::with_state_budget(self.store.block_size(), self.store.state_budget());
        std::mem::replace(&mut self.store, empty)
    }
}

/// Splits moved flat caches into their full-attention and GDN halves.
pub(crate) fn split_flat_caches(
    caches: Vec<Qwen3_5LayerCache>,
    context: DFlash2ContextCache,
) -> (FlatKv, FlatEnd) {
    let mut layers = Vec::new();
    let mut linear = Vec::new();
    for (index, cache) in caches.into_iter().enumerate() {
        match cache {
            Qwen3_5LayerCache::FullAttention(kv) => layers.push((index, kv)),
            Qwen3_5LayerCache::Linear(gdn) => linear.push((index, gdn)),
        }
    }
    (FlatKv { layers }, FlatEnd { linear, context })
}

/// Reassembles the per-layer caches in decoder order, plus the draft window.
pub(crate) fn join_flat_caches(
    kv: FlatKv,
    end: FlatEnd,
) -> Result<(Vec<Qwen3_5LayerCache>, DFlash2ContextCache)> {
    let total = kv.layers.len() + end.linear.len();
    let mut slots: Vec<Option<Qwen3_5LayerCache>> = (0..total).map(|_| None).collect();
    let placed = kv
        .layers
        .into_iter()
        .map(|(index, cache)| (index, Qwen3_5LayerCache::FullAttention(cache)))
        .chain(
            end.linear
                .into_iter()
                .map(|(index, cache)| (index, Qwen3_5LayerCache::Linear(cache))),
        );
    for (index, cache) in placed {
        match slots.get_mut(index) {
            Some(slot @ None) => *slot = Some(cache),
            _ => {
                return Err(Error::from_reason(format!(
                    "parked DFlash2 state has no free slot for layer {index} of {total}"
                )));
            }
        }
    }
    let caches = slots
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| Error::from_reason("parked DFlash2 state is missing a layer"))?;
    Ok((caches, end.context))
}

/// Per-layer caches in decoder order from a parked state's full-attention
/// caches and a GDN cell: every slot the KV does not hold is a GDN layer and
/// takes the cell's next conv history and recurrent state, deep-copied so
/// the cell stays the anchor's alone.
pub(crate) fn join_flat_anchor(kv: FlatKv, cell: &GdnCell) -> Result<Vec<Qwen3_5LayerCache>> {
    if cell.conv.len() != cell.rec.len() {
        return Err(Error::from_reason(format!(
            "DFlash2 GDN cell holds {} conv histories for {} states",
            cell.conv.len(),
            cell.rec.len()
        )));
    }
    let total = kv.layers.len() + cell.conv.len();
    let mut slots: Vec<Option<Qwen3_5LayerCache>> = (0..total).map(|_| None).collect();
    for (index, cache) in kv.layers {
        match slots.get_mut(index) {
            Some(slot @ None) => *slot = Some(Qwen3_5LayerCache::FullAttention(cache)),
            _ => {
                return Err(Error::from_reason(format!(
                    "parked DFlash2 state has no free slot for layer {index} of {total}"
                )));
            }
        }
    }
    let mut states = cell.conv.iter().zip(&cell.rec);
    for slot in slots.iter_mut().filter(|slot| slot.is_none()) {
        let (conv, rec) = states
            .next()
            .ok_or_else(|| Error::from_reason("DFlash2 GDN cell is missing a layer"))?;
        let mut gdn = ArraysCache::new(2);
        gdn.set(0, conv.deep_copy()?)?;
        gdn.set(1, rec.deep_copy()?)?;
        *slot = Some(Qwen3_5LayerCache::Linear(gdn));
    }
    let caches = slots.into_iter().flatten().collect::<Vec<_>>();
    let mut arrays = Vec::new();
    for cache in &caches {
        if let Qwen3_5LayerCache::Linear(gdn) = cache {
            arrays.extend(gdn.get(0));
            arrays.extend(gdn.get(1));
        }
    }
    if !arrays.is_empty() {
        MxArray::eval_arrays(&arrays)?;
    }
    Ok(caches)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::models::qwen3_5::dflash2_decode::kv_row_bytes;
    use crate::models::qwen3_5::dflash2_decode::reserve_dflash2_target_kv;
    use crate::models::qwen3_5::dflash2_decode::tests::{
        array_fingerprints, metal_is_default_device, run_tiny_dflash_turn,
        tiny_dflash_inner_with_attention_head_dim, tiny_int8_dflash_inner,
    };
    use crate::models::qwen3_5::dflash2_state::store::{
        Anchor, HybridStateStore, Lineage, Reservation,
    };
    use crate::models::qwen3_5::dflash2_state::trace::HitKind;
    use crate::models::qwen3_5::dflash2_turn::{
        dflash2_target_kv_dtype, install_flat_dflash2_end, park_flat_dflash2_state,
    };
    use crate::models::qwen3_5::model::Qwen35Inner;
    use crate::tokenizer::Qwen3Tokenizer;

    type Fingerprints = Vec<(Vec<i64>, Vec<u32>)>;

    const FIXTURES: [&str; 2] = ["bf16_d64", "int8_d256"];

    fn fixture(name: &str) -> Qwen35Inner {
        match name {
            "bf16_d64" => tiny_dflash_inner_with_attention_head_dim(0xDFA5_F1A7, 64),
            "int8_d256" => tiny_int8_dflash_inner(0xDFA5_F1A8),
            _ => unreachable!("unknown fixture {name}"),
        }
    }

    fn prompt(seed: u32, len: u32) -> Vec<u32> {
        (0..len).map(|i| (seed + i * 7) % 15 + 1).collect()
    }

    fn turn(
        inner: &mut Qwen35Inner,
        tokenizer: &Arc<Qwen3Tokenizer>,
        tokens: &[u32],
    ) -> Result<crate::engine::types::ChatResult> {
        run_tiny_dflash_turn(inner, tokenizer, tokens, true, 12)
    }

    fn next_prompt(inner: &Qwen35Inner) -> Vec<u32> {
        inner
            .cached_token_history
            .iter()
            .copied()
            .chain([9, 10])
            .collect()
    }

    /// Every buffer of the live flat state, whole (capacity rows included),
    /// plus its history.
    #[derive(Debug, PartialEq)]
    struct LiveState {
        history: Vec<u32>,
        target: Fingerprints,
        draft: Fingerprints,
        context_tokens: Vec<u32>,
    }

    fn live_state(inner: &Qwen35Inner) -> Result<LiveState> {
        let mut target = Vec::new();
        for cache in inner.caches.iter().flatten() {
            cache.collect_arrays(&mut target);
        }
        let context = inner.dflash2_context.as_ref().expect("live draft context");
        let mut draft = Vec::new();
        context.collect_arrays(&mut draft);
        Ok(LiveState {
            history: inner.cached_token_history.clone(),
            target: array_fingerprints(target)?,
            draft: array_fingerprints(draft)?,
            context_tokens: context.token_history().to_vec(),
        })
    }

    /// Full-attention rows `[0, rows)` of `caches` (keys, values and, for
    /// int8, scales), in layer order.
    fn attention_rows<'a>(
        caches: impl IntoIterator<Item = &'a KVCache>,
        rows: usize,
    ) -> Result<Fingerprints> {
        let mut arrays = Vec::new();
        for kv in caches {
            for buffer in [
                kv.keys_ref(),
                kv.values_ref(),
                kv.key_scales_ref(),
                kv.value_scales_ref(),
            ]
            .into_iter()
            .flatten()
            {
                arrays.push(buffer.slice_axis(2, 0, rows as i64)?);
            }
        }
        array_fingerprints(arrays.iter().collect())
    }

    fn live_attention(inner: &Qwen35Inner) -> impl Iterator<Item = &KVCache> {
        inner
            .caches
            .iter()
            .flatten()
            .filter_map(|cache| match cache {
                Qwen3_5LayerCache::FullAttention(kv) => Some(kv),
                Qwen3_5LayerCache::Linear(_) => None,
            })
    }

    /// The live state with full-attention buffers cut to their used rows,
    /// so states whose buffers differ only in spare capacity compare equal.
    fn live_rows(inner: &Qwen35Inner) -> Result<(Vec<u32>, Fingerprints, Fingerprints)> {
        let used = inner.cached_token_history.len();
        let attention = attention_rows(live_attention(inner), used)?;
        let mut gdn = Vec::new();
        for cache in inner.caches.iter().flatten() {
            if let Qwen3_5LayerCache::Linear(state) = cache {
                gdn.extend(state.get(0));
                gdn.extend(state.get(1));
            }
        }
        let mut arrays = array_fingerprints(gdn)?;
        let mut draft = Vec::new();
        inner
            .dflash2_context
            .as_ref()
            .expect("live draft context")
            .collect_arrays(&mut draft);
        arrays.extend(array_fingerprints(draft)?);
        Ok((inner.cached_token_history.clone(), attention, arrays))
    }

    fn row_bytes(inner: &Qwen35Inner, kv: &FlatKv) -> Result<u64> {
        Ok(kv_row_bytes(
            kv.layers().len(),
            inner.config.num_kv_heads,
            inner.config.head_dim,
            dflash2_target_kv_dtype(inner)?,
            kv.layers()[0].1.format(),
        ))
    }

    fn capacities(kv: &FlatKv) -> Result<Vec<i64>> {
        kv.layers()
            .iter()
            .map(|(_, cache)| cache.capacity())
            .collect()
    }

    /// Reference run: the same two turns with nothing parked in between.
    fn reference_second_turn(
        name: &str,
        tokenizer: &Arc<Qwen3Tokenizer>,
        first: &[u32],
    ) -> Result<(u32, (Vec<u32>, Fingerprints, Fingerprints))> {
        let mut reference = fixture(name);
        turn(&mut reference, tokenizer, first)?;
        let second = next_prompt(&reference);
        let result = turn(&mut reference, tokenizer, &second)?;
        Ok((result.cached_tokens, live_rows(&reference)?))
    }

    /// X0: park, a cold turn of another session on the live slots, then
    /// install gives back every buffer bit for bit. X1: the next turn on
    /// the installed state equals the same turn on a state never parked.
    #[test]
    fn parked_state_round_trips_through_another_session() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let tokenizer = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
        for name in FIXTURES {
            let first = prompt(0, 40);
            let (reference_cached, reference) = reference_second_turn(name, &tokenizer, &first)?;

            let mut inner = fixture(name);
            turn(&mut inner, &tokenizer, &first)?;
            let before = live_state(&inner)?;
            let mut live_bytes = Vec::new();
            for cache in inner.caches.iter().flatten() {
                cache.collect_arrays(&mut live_bytes);
            }
            let live_bytes = arrays_bytes(live_bytes);
            let mut draft = Vec::new();
            inner
                .dflash2_context
                .as_ref()
                .expect("live draft context")
                .collect_arrays(&mut draft);
            let draft_bytes = arrays_bytes(draft);
            let dtype = dflash2_target_kv_dtype(&inner)?;

            let (tokens, parts) = park_flat_dflash2_state(&mut inner).expect("parkable");
            assert_eq!(tokens, before.history, "{name}: parked history");
            assert!(
                inner.caches.is_none()
                    && inner.dflash2_context.is_none()
                    && inner.cached_token_history.is_empty(),
                "{name}: park moves the live state out"
            );
            let kv = parts.kv.as_ref().expect("parked KV");
            let end = parts.end.as_ref().expect("parked end state");
            let capacity = capacities(kv)?;
            assert!(capacity.iter().all(|&rows| rows == capacity[0]));
            let rows = kv_row_bytes(
                kv.layers().len(),
                inner.config.num_kv_heads,
                inner.config.head_dim,
                dtype,
                kv.layers()[0].1.format(),
            );
            assert_eq!(kv.bytes(), capacity[0] as u64 * rows, "{name}: kv bytes");
            assert_eq!(kv.rows(), tokens.len());
            assert_eq!(
                end.state_bytes(),
                live_bytes - kv.bytes() + draft_bytes,
                "{name}: GDN + window bytes"
            );
            assert_eq!(end.next_position() as usize, tokens.len());

            let other = turn(&mut inner, &tokenizer, &prompt(5, 33))?;
            assert_eq!(other.cached_tokens, 0, "{name}: the other session is cold");
            let (_, other_parts) = park_flat_dflash2_state(&mut inner).expect("other parks");
            drop(other_parts);

            install_flat_dflash2_end(&mut inner, tokens, parts)?;
            assert!(live_state(&inner)? == before, "{name}: X0 round trip");

            let second = next_prompt(&inner);
            let result = turn(&mut inner, &tokenizer, &second)?;
            assert_eq!(result.cached_tokens, reference_cached, "{name}: live hit");
            assert!(result.cached_tokens > 0);
            assert!(live_rows(&inner)? == reference, "{name}: X1 next turn");
        }
        Ok(())
    }

    /// Compaction under KV pressure copies the used rows into a used + 256
    /// buffer: X0 for rows `[0, used)`, GDN and window; X1 for the next turn.
    #[test]
    fn compaction_keeps_the_used_rows_and_the_next_turn() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let tokenizer = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
        for name in FIXTURES {
            let first = prompt(3, 40);
            let (reference_cached, reference) = reference_second_turn(name, &tokenizer, &first)?;

            let mut inner = fixture(name);
            turn(&mut inner, &tokenizer, &first)?;
            let used = inner.cached_token_history.len();
            reserve_dflash2_target_kv(&mut inner.caches, (used + COMPACT_SLACK_ROWS + 512) as i64)?;
            let before = live_rows(&inner)?;
            let (tokens, parts) = park_flat_dflash2_state(&mut inner).expect("parkable");
            let kv = parts.kv.as_ref().expect("parked KV");
            let row_bytes = row_bytes(&inner, kv)?;
            let plan = kv.shrink_plan(used).expect("slack worth a copy");
            let capacity = capacities(kv)?[0] as u64;
            let kept = (used + COMPACT_SPARE_ROWS) as u64;
            assert_eq!(
                plan,
                Shrink {
                    frees: (capacity - kept) * row_bytes,
                    transient: kept * row_bytes,
                }
            );

            let mut store = HybridStateStore::<FlatBacking>::with_state_budget(16, u64::MAX);
            let id = store.insert(Lineage::text(&tokens, 0), parts)?;
            let need = Reservation {
                state_bytes: 0,
                kv_bytes: plan.transient + plan.frees,
            };
            let report = store.evict_for(need, plan.transient, None)?;
            assert!(report.satisfied, "{name}: {report:?}");
            assert_eq!(
                (report.compacted, report.demoted, report.dropped_ends),
                (1, 0, 0),
                "{name}: compaction alone covers it"
            );
            assert_eq!(report.freed_kv_bytes, plan.frees);
            assert_eq!(store.get(id).expect("entry").kv_bytes(), kept * row_bytes);

            let (tokens, parts) = store.take(id).expect("entry").into_parts();
            let kv = parts.kv.as_ref().expect("parked KV");
            assert!(capacities(kv)?.iter().all(|&rows| rows as u64 == kept));
            assert_eq!(kv.rows(), used);
            install_flat_dflash2_end(&mut inner, tokens, parts)?;
            assert!(live_rows(&inner)? == before, "{name}: X0 compacted rows");

            let second = next_prompt(&inner);
            let result = turn(&mut inner, &tokenizer, &second)?;
            assert_eq!(result.cached_tokens, reference_cached, "{name}: live hit");
            assert!(live_rows(&inner)? == reference, "{name}: X1 next turn");
        }
        Ok(())
    }

    /// Demotion under state pressure drops GDN@E and the ring, keeps the
    /// anchor, and trims and compacts the KV to R: X0 for rows `[0, R)`.
    #[test]
    fn demotion_keeps_the_rows_below_the_anchor() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let tokenizer = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
        const R: usize = 32;
        for name in FIXTURES {
            let mut inner = fixture(name);
            turn(&mut inner, &tokenizer, &prompt(1, 40))?;
            let used = inner.cached_token_history.len();
            assert!(used > R);
            reserve_dflash2_target_kv(&mut inner.caches, (used + COMPACT_SLACK_ROWS + 512) as i64)?;
            let below_anchor = attention_rows(live_attention(&inner), R)?;
            let draft_config = inner.dflash2.as_ref().expect("draft").config.clone();
            let (tokens, mut parts) = park_flat_dflash2_state(&mut inner).expect("parkable");
            let row_bytes = row_bytes(&inner, parts.kv.as_ref().expect("parked KV"))?;

            // Stand-in anchor payload: this test only follows the KV rows
            // and the bytes; the GDN arrays are copies of E's, not R's.
            let end = parts.end.as_ref().expect("parked end state");
            let (mut conv, mut rec) = (Vec::new(), Vec::new());
            for (_, state) in &end.linear {
                conv.push(state.get(0).expect("conv state").deep_copy()?);
                rec.push(state.get(1).expect("recurrent state").deep_copy()?);
            }
            let anchor = FlatAnchor {
                cell: GdnCell { row: R, conv, rec },
                window: DFlash2ContextCache::new(&draft_config),
            };
            let anchor_bytes = anchor.state_bytes();
            assert!(anchor_bytes > 0);
            parts.anchor = Some(Anchor {
                row: R,
                state: anchor,
            });

            let mut store = HybridStateStore::<FlatBacking>::with_state_budget(16, anchor_bytes);
            let id = store.insert(Lineage::text(&tokens, 0), parts)?;
            let report = store.evict_for(
                Reservation {
                    state_bytes: 0,
                    kv_bytes: 0,
                },
                u64::MAX,
                None,
            )?;
            assert_eq!(
                (report.demoted, report.dropped_captures, report.dropped_ends),
                (1, 0, 0),
                "{name}: {report:?}"
            );
            let entry = store.get(id).expect("demoted entry");
            assert_eq!(entry.kind(), HitKind::Capture);
            assert_eq!(entry.tokens(), &tokens[..R]);
            assert_eq!(entry.state_bytes(), anchor_bytes);
            let kept = (R + COMPACT_SPARE_ROWS) as u64;
            assert_eq!(entry.kv_bytes(), kept * row_bytes);
            let kv = entry.kv().expect("anchor KV");
            assert_eq!(kv.rows(), R);
            assert!(capacities(kv)?.iter().all(|&rows| rows as u64 == kept));
            assert!(
                attention_rows(kv.layers().iter().map(|(_, cache)| cache), R)? == below_anchor,
                "{name}: X0 rows below the anchor"
            );
            let fork: Vec<u32> = tokens[..R + 4].iter().copied().chain([14, 14]).collect();
            let hit = store
                .lookup(Lineage::text(&fork, 0), |_| true)
                .expect("anchor hit");
            assert_eq!((hit.kind, hit.depth), (HitKind::Capture, R));
        }
        Ok(())
    }

    /// A live state that is not an exact DFlash2 state of its history is
    /// left in place.
    #[test]
    fn park_refuses_a_live_state_that_is_not_exact() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let tokenizer = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
        let mut inner = fixture("bf16_d64");
        turn(&mut inner, &tokenizer, &prompt(2, 20))?;
        let history = inner.cached_token_history.clone();

        inner.paged_full_attn_caches_dirty = true;
        assert!(park_flat_dflash2_state(&mut inner).is_none(), "dirty lane");
        inner.paged_full_attn_caches_dirty = false;

        inner.cached_token_history.push(1);
        assert!(
            park_flat_dflash2_state(&mut inner).is_none(),
            "history ahead"
        );
        inner.cached_token_history = history.clone();

        let context = inner.dflash2_context.take();
        assert!(park_flat_dflash2_state(&mut inner).is_none(), "no window");
        inner.dflash2_context = context;

        assert!(
            inner.caches.is_some()
                && inner.dflash2_context.is_some()
                && inner.cached_token_history == history
        );
        assert!(park_flat_dflash2_state(&mut inner).is_some());
        Ok(())
    }

    /// Install refuses to replace a live window, and an entry with no end
    /// state (a demoted one).
    #[test]
    fn install_needs_an_end_state_and_no_live_window() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let tokenizer = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
        let mut inner = fixture("bf16_d64");
        turn(&mut inner, &tokenizer, &prompt(4, 20))?;
        let (tokens, parts) = park_flat_dflash2_state(&mut inner).expect("parkable");
        turn(&mut inner, &tokenizer, &prompt(6, 20))?;
        let live = live_state(&inner)?;
        assert!(install_flat_dflash2_end(&mut inner, tokens.clone(), parts).is_err());
        assert!(live_state(&inner)? == live, "the live state stays");

        let (other_tokens, mut other) = park_flat_dflash2_state(&mut inner).expect("parkable");
        other.end = None;
        assert!(install_flat_dflash2_end(&mut inner, other_tokens, other).is_err());
        assert!(inner.caches.is_none() && inner.dflash2_context.is_none());
        Ok(())
    }
}
