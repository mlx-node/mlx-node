use napi::bindgen_prelude::*;

use super::DFlash2Model;
use super::config::DFlash2Config;
use super::layers::{SlidingMaskMemo, memo_ring_mask, memo_sliding_mask};
use super::spanned_head::is_wide_vocab_token_for;
use crate::array::MxArray;
use crate::models::quantized_linear::QuantizedLinear;
use crate::models::qwen3_5::draft_context::DraftContextWindow;

pub(super) fn store_window_tail(
    window: &[MxArray],
    keys: &MxArray,
    values: &MxArray,
) -> Result<Vec<MxArray>> {
    crate::kernels::kv::store_window_tail(window, keys, values)?
        .ok_or_else(|| Error::from_reason("compiled draft KV tail-store contract failed"))
}

/// Whether draft projections of the given row widths may be row-merged (and
/// the merge sliced back into views): only Tiled64 K-quant/affine linears,
/// whose kernels decode and reduce each output row the same way at any
/// width, and whose layout concatenates and slices in whole 64-row tiles, so
/// every width must be a tile multiple. A projection still on MLX's affine
/// route (no Metal, or a shape the tiled kernels do not take) is not merged:
/// its per-row `qmv` / GEMM choice follows the width.
pub(super) fn kquant_rows_merge(packed: &QuantizedLinear, widths: &[i64]) -> bool {
    use crate::models::quant_dispatch::{KQUANT_TILE_ROWS, split_kquant_layout};
    let (base_mode, tiled) = split_kquant_layout(packed.mode());
    if tiled {
        return widths.iter().all(|w| w % KQUANT_TILE_ROWS == 0);
    }
    // On CUDA, standard row-major affine Q4/g64 draft weights concatenate
    // and slice cleanly along axis 0, collapsing 35 matmuls/cycle into 11.
    crate::accelerator::cuda_ops_available()
        && base_mode == "affine"
        && packed.bits() == 4
        && packed.group_size() == 64
}

/// One draft layer's attention context (Splash `draft_attention_split_phase`):
/// a ring of `slots` K and V rows, row p at slot p % slots, whose live rows
/// are the newest `slots - 1`, logical rows `[start, end)`. The buffers hold
/// `reserve` rows past the ring, where a proposal's block lands. Appends
/// write their slots in place (`mlx_kv_store_rows`), and the attention reads
/// the buffers in place, so neither the commit nor the propose copies the
/// window, and nothing ever grows.
pub(crate) struct DraftKvWindow {
    pub(super) keys: Option<MxArray>,
    pub(super) values: Option<MxArray>,
    pub(super) start: i64,
    pub(super) end: i64,
    slots: i64,
    reserve: i64,
}

#[cfg(test)]
thread_local! {
    /// Draft ring buffers allocated on this thread.
    pub(crate) static RING_ALLOCATIONS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

fn note_ring_allocation() {
    #[cfg(test)]
    RING_ALLOCATIONS.with(|count| count.set(count.get() + 1));
}

/// Where a block's attention keys come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockKeys {
    /// An empty window: the block alone.
    Alone,
    /// Not wrapped: buffer rows `[start, end + rows)` are the live rows and
    /// the block in logical order.
    Linear,
    /// Wrapped: the whole ring in slot order, then the block at `slots`.
    Ring,
    /// The block does not fit past the rows: the live rows copied in
    /// logical order, then the block.
    Copied,
}

impl DraftKvWindow {
    pub(super) fn new(slots: i64, reserve: i64) -> Self {
        Self {
            keys: None,
            values: None,
            start: 0,
            end: 0,
            slots: slots.max(2),
            reserve: reserve.max(0),
        }
    }

    /// Live rows.
    pub(crate) fn len(&self) -> i64 {
        self.end - self.start
    }

    fn window(&self) -> i64 {
        self.slots - 1
    }

    fn capacity(&self) -> Result<i64> {
        self.keys.as_ref().map_or(Ok(0), |keys| keys.shape_at(2))
    }

    /// Rows past `slots` have been written: live rows sit at `p % slots`.
    pub(super) fn wrapped(&self) -> bool {
        self.end > self.slots
    }

    /// Drops the rows; the buffers stay.
    fn reset(&mut self) {
        self.start = 0;
        self.end = 0;
    }

    /// Bytes of the K and V buffers allocated so far.
    fn buffer_bytes(&self) -> u64 {
        [&self.keys, &self.values]
            .into_iter()
            .flatten()
            .fold(0u64, |sum, buffer| {
                sum.saturating_add(buffer.nbytes() as u64)
            })
    }

    /// The live rows as `[B, H, len, D]` in logical order, `None` when
    /// empty; a wrapped window is copied.
    pub(crate) fn view(&self) -> Option<(MxArray, MxArray)> {
        if self.end == self.start {
            return None;
        }
        let (keys, values) = (self.keys.as_ref()?, self.values.as_ref()?);
        if !self.wrapped() {
            return Some((
                keys.slice_axis(2, self.start, self.end).ok()?,
                values.slice_axis(2, self.start, self.end).ok()?,
            ));
        }
        let first = self.start % self.slots;
        let last = (self.end - 1) % self.slots + 1;
        let read = |buffer: &MxArray| -> Option<MxArray> {
            if first < last {
                return buffer.slice_axis(2, first, last).ok();
            }
            MxArray::concatenate(
                &buffer.slice_axis(2, first, self.slots).ok()?,
                &buffer.slice_axis(2, 0, last).ok()?,
                2,
            )
            .ok()
        };
        Some((read(keys)?, read(values)?))
    }

    /// Append `[B, H, T, D]` rows (any strides) at logical row `base`; only
    /// the newest `slots - 1` rows stay live. Rows that do not continue the
    /// window replace it.
    pub(super) fn append(&mut self, base: i64, keys: &MxArray, values: &MxArray) -> Result<()> {
        let mut rows = keys.shape_at(2)?;
        let mut base = base;
        let tail;
        let (keys, values) = if rows > self.window() {
            let skip = rows - self.window();
            tail = (
                keys.slice_axis(2, skip, rows)?,
                values.slice_axis(2, skip, rows)?,
            );
            base += skip;
            rows = self.window();
            (&tail.0, &tail.1)
        } else {
            (keys, values)
        };
        if base != self.end || rows == self.window() {
            self.start = base;
            self.end = base;
        }
        if self.keys.is_none() || self.values.is_none() {
            let capacity = self.slots + self.reserve;
            let buffer = |rows: &MxArray| -> Result<MxArray> {
                MxArray::zeros(
                    &[
                        rows.shape_at(0)?,
                        rows.shape_at(1)?,
                        capacity,
                        rows.shape_at(3)?,
                    ],
                    Some(rows.dtype()?),
                )
            };
            self.keys = Some(buffer(keys)?);
            self.values = Some(buffer(values)?);
            note_ring_allocation();
        }
        let slot = base % self.slots;
        let first = rows.min(self.slots - slot);
        if first == rows {
            self.store_rows(slot, keys, values)?;
        } else {
            self.store_rows(
                slot,
                &keys.slice_axis(2, 0, first)?,
                &values.slice_axis(2, 0, first)?,
            )?;
            self.store_rows(
                0,
                &keys.slice_axis(2, first, rows)?,
                &values.slice_axis(2, first, rows)?,
            )?;
        }
        self.end = base + rows;
        self.start = self.start.max(self.end - self.window());
        Ok(())
    }

    fn block_keys(&self, rows: i64) -> Result<BlockKeys> {
        if self.end == self.start {
            return Ok(BlockKeys::Alone);
        }
        let capacity = self.capacity()?;
        Ok(if !self.wrapped() && self.end + rows <= capacity {
            BlockKeys::Linear
        } else if self.wrapped() && self.slots + rows <= capacity {
            BlockKeys::Ring
        } else {
            BlockKeys::Copied
        })
    }

    /// The buffer rows a block of `rows` attends over in place, the block
    /// last; `None` when it does not attend in place.
    pub(super) fn block_range(&self, rows: i64) -> Result<Option<(i64, i64)>> {
        Ok(match self.block_keys(rows)? {
            BlockKeys::Linear => Some((self.start, self.end + rows)),
            BlockKeys::Ring => Some((0, self.slots + rows)),
            BlockKeys::Alone | BlockKeys::Copied => None,
        })
    }

    /// The mask of a block of `rows` queries at `query_base` over
    /// [`Self::with_block`]'s keys.
    pub(super) fn block_mask(
        &self,
        memo: &mut SlidingMaskMemo,
        query_base: i32,
        rows: i64,
        window: i64,
    ) -> Result<Option<MxArray>> {
        match self.block_keys(rows)? {
            BlockKeys::Alone => memo_sliding_mask(memo, query_base, rows, query_base, rows, window),
            BlockKeys::Linear | BlockKeys::Copied => {
                let start = i32::try_from(self.start)
                    .map_err(|_| Error::from_reason("DFlash2 draft window start exceeds i32"))?;
                memo_sliding_mask(memo, query_base, rows, start, self.len() + rows, window)
            }
            BlockKeys::Ring => memo_ring_mask(
                memo,
                i64::from(query_base),
                rows,
                self.start,
                self.end,
                self.slots,
                window,
            ),
        }
    }

    /// The attention keys for a block: the window, then the block's
    /// `[B, H, T, D]` rows, which land past the rows in place (one
    /// dispatch). The written rows are dead to the window and are
    /// overwritten by a later append, whose store is ordered after this one
    /// by the shared handle.
    pub(super) fn with_block(
        &mut self,
        keys: &MxArray,
        values: &MxArray,
    ) -> Result<(MxArray, MxArray)> {
        let rows = keys.shape_at(2)?;
        let at = match self.block_keys(rows)? {
            BlockKeys::Alone => return Ok((keys.clone(), values.clone())),
            BlockKeys::Copied => {
                let (context_keys, context_values) = self
                    .view()
                    .ok_or_else(|| Error::from_reason("DFlash2 draft window lost its rows"))?;
                return Ok((
                    MxArray::concatenate(&context_keys, keys, 2)?,
                    MxArray::concatenate(&context_values, values, 2)?,
                ));
            }
            BlockKeys::Linear => self.end,
            BlockKeys::Ring => self.slots,
        };
        self.store_rows(at, keys, values)?;
        let (from, to) = self
            .block_range(rows)?
            .ok_or_else(|| Error::from_reason("DFlash2 draft block lost its range"))?;
        let (Some(buffer_keys), Some(buffer_values)) = (&self.keys, &self.values) else {
            return Err(Error::from_reason("DFlash2 draft window lost its buffers"));
        };
        Ok((
            buffer_keys.slice_axis(2, from, to)?,
            buffer_values.slice_axis(2, from, to)?,
        ))
    }

    /// Write `keys` / `values` rows at `offset` in place and adopt the store
    /// primitive's handles (same buffers); without Metal, slice_update.
    fn store_rows(&mut self, offset: i64, keys: &MxArray, values: &MxArray) -> Result<()> {
        let rows = keys.shape_at(2)?;
        let (Some(buffer_keys), Some(buffer_values)) = (&self.keys, &self.values) else {
            return Err(Error::from_reason("DFlash2 draft window has no buffers"));
        };
        let k_cap = buffer_keys.shape_at(2)?;
        let v_cap = buffer_values.shape_at(2)?;
        if offset < 0 || rows < 0 || offset + rows > k_cap || offset + rows > v_cap {
            return Err(Error::from_reason(format!(
                "DFlash2 draft window store_rows out of bounds: offset={offset}, rows={rows}, k_cap={k_cap}, v_cap={v_cap}"
            )));
        }
        if let Some((new_k, new_v)) = crate::kernels::kv::store_kv_pair(
            buffer_keys,
            buffer_values,
            keys,
            values,
            offset as i32,
        )? {
            self.keys = Some(new_k);
            self.values = Some(new_v);
            return Ok(());
        }
        // Clone the C++ `mlx::core::array` handle before `slice_assign_axis_inplace`
        // overwrites its descriptor so any saved `pre_propose_handles` clone of the
        // old `Arc<MxHandle>` stays untouched on non-Metal (CUDA/CPU) backends too.
        let mut next_k = MxArray::from_handle(
            unsafe { mlx_sys::mlx_array_clone_handle(buffer_keys.as_raw_ptr()) },
            "dflash2_store_rows_clone_k",
        )?;
        let mut next_v = MxArray::from_handle(
            unsafe { mlx_sys::mlx_array_clone_handle(buffer_values.as_raw_ptr()) },
            "dflash2_store_rows_clone_v",
        )?;
        next_k.slice_assign_axis_inplace(2, offset, offset + rows, keys)?;
        next_v.slice_assign_axis_inplace(2, offset, offset + rows, values)?;
        self.keys = Some(next_k);
        self.values = Some(next_v);
        Ok(())
    }

    fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        out.extend(self.keys.as_ref());
        out.extend(self.values.as_ref());
    }
}

/// One segment's draft K/V for every layer, projected at `base` and not yet
/// stored: `(keys, values)` per layer as `[B, HK, rows, D]`.
pub(crate) struct ProjectedContext {
    base: i32,
    rows: i64,
    kv: Vec<(MxArray, MxArray)>,
}

pub(crate) struct DFlash2ContextCache {
    pub(super) layers: Vec<DraftKvWindow>,
    /// The sliding mask depends only on the window/block geometry, which
    /// repeats once the window is full, so it is built once and reused.
    pub(super) mask_memo: SlidingMaskMemo,
    pub(super) logical_len: i32,
    /// Position one past the newest row the windows hold. `record_only`
    /// advances `logical_len` alone, so the windows do not end at
    /// `logical_len` until enough later rows are appended.
    kv_end: i32,
    /// Exact token prefix represented by every layer cache above.
    ///
    /// A length alone is not cache provenance: an unrelated paged request can
    /// end at the same position. Keep the ids beside the draft K/V so a warm
    /// DFlash2 continuation is admitted only for the prefix that produced it.
    pub(super) token_history: Vec<u32>,
    /// Target vocabulary size (`config.vocab_size`), used to select the
    /// model-appropriate adaptive spanned vocabulary range.
    vocab_size: usize,
    /// True once any prompt, accepted, or anchor token falls outside this
    /// model's spanned draft vocabulary (e.g. Qwen3.8 `[0, 98_304) U [248_032, 248_320)`
    /// or Inkling-2-Mini `[0, low_end) U [199_936, 200_058)`).
    wide_vocab: bool,
    /// Saved per-layer `(keys, values)` ring handles captured before a batched
    /// `propose_with_batched` call mutates `DraftKvWindow` in place via
    /// `with_block`. Restoring them drops any lazy `KvStoreRows` handles whose
    /// inputs depend on a failed shared batched forward graph before solo retry.
    pub(super) pre_propose_handles: Option<Vec<(Option<MxArray>, Option<MxArray>)>>,
}

impl DFlash2ContextCache {
    pub(crate) fn new(config: &DFlash2Config) -> Self {
        Self {
            layers: (0..config.num_hidden_layers)
                .map(|_| {
                    DraftKvWindow::new(config.sliding_window as i64, config.block_size as i64 + 1)
                })
                .collect(),
            mask_memo: None,
            logical_len: 0,
            kv_end: 0,
            token_history: Vec::new(),
            vocab_size: config.vocab_size,
            wide_vocab: false,
            pre_propose_handles: None,
        }
    }

    pub(crate) fn save_pre_propose_handles(&mut self) {
        self.pre_propose_handles = Some(
            self.layers
                .iter()
                .map(|window| (window.keys.clone(), window.values.clone()))
                .collect(),
        );
    }

    pub(crate) fn restore_pre_propose_handles(&mut self) {
        if let Some(saved) = self.pre_propose_handles.take() {
            for (window, (keys, values)) in self.layers.iter_mut().zip(saved) {
                window.keys = keys;
                window.values = values;
            }
        }
    }

    pub(crate) fn clear_pre_propose_handles(&mut self) {
        self.pre_propose_handles = None;
    }

    pub(crate) fn observe_token(&mut self, token: u32) {
        if !self.wide_vocab && is_wide_vocab_token_for(self.vocab_size, token) {
            self.wide_vocab = true;
        }
    }

    pub(crate) fn observe_tokens(&mut self, token_ids: &[u32]) {
        if !self.wide_vocab
            && token_ids
                .iter()
                .any(|&t| is_wide_vocab_token_for(self.vocab_size, t))
        {
            self.wide_vocab = true;
        }
    }

    pub(crate) fn wide_vocab(&self) -> bool {
        self.wide_vocab
    }

    pub(crate) fn append(
        &mut self,
        model: &DFlash2Model,
        fused_context: &MxArray,
        base: i32,
        token_ids: &[u32],
    ) -> Result<()> {
        let projected = self.project(model, fused_context, base)?;
        self.append_projected(projected, token_ids)
    }

    /// Every layer's draft K/V for `fused_context` rows at `base`, the
    /// context's next position; `append_projected` stores them.
    pub(crate) fn project(
        &self,
        model: &DFlash2Model,
        fused_context: &MxArray,
        base: i32,
    ) -> Result<ProjectedContext> {
        if base != self.logical_len {
            return Err(Error::from_reason(format!(
                "DFlash2 context append starts at {base}, expected {}",
                self.logical_len
            )));
        }
        if usize::try_from(self.logical_len).ok() != Some(self.token_history.len()) {
            return Err(Error::from_reason(format!(
                "DFlash2 context provenance length {} does not match logical length {}",
                self.token_history.len(),
                self.logical_len
            )));
        }
        Ok(ProjectedContext {
            base,
            rows: fused_context.shape_at(1)?,
            kv: model.project_context(fused_context, base)?,
        })
    }

    pub(crate) fn append_projected(
        &mut self,
        projected: ProjectedContext,
        token_ids: &[u32],
    ) -> Result<()> {
        if projected.base != self.logical_len {
            return Err(Error::from_reason(format!(
                "DFlash2 context append starts at {}, expected {}",
                projected.base, self.logical_len
            )));
        }
        let rows = projected.rows as usize;
        if rows != token_ids.len() {
            return Err(Error::from_reason(format!(
                "DFlash2 context append has {rows} hidden rows for {} token ids",
                token_ids.len()
            )));
        }
        if projected.kv.len() != self.layers.len() {
            return Err(Error::from_reason(format!(
                "DFlash2 context append projected {} layers for {} windows",
                projected.kv.len(),
                self.layers.len()
            )));
        }
        let next = self
            .logical_len
            .checked_add(i32::try_from(rows).map_err(|_| {
                Error::from_reason(format!(
                    "DFlash2 context append row count {rows} exceeds i32"
                ))
            })?)
            .ok_or_else(|| Error::from_reason("DFlash2 context length overflow"))?;
        for ((keys, values), window) in projected.kv.iter().zip(self.layers.iter_mut()) {
            window.append(i64::from(projected.base), keys, values)?;
        }
        self.logical_len = next;
        self.kv_end = next;
        self.token_history.extend_from_slice(token_ids);
        self.observe_tokens(token_ids);
        self.pre_propose_handles = None;
        Ok(())
    }

    /// Batched counterpart to [`Self::append`]: validates every lane's cursor,
    /// provenance, and layer count BEFORE mutating any ring, projects all lanes'
    /// concatenated `fused_all` (`[1, sum(token_ids_i.len()), H]`) in one batched
    /// K/V pass, and appends each lane's slice into its own sliding window.
    pub(crate) fn append_batched(
        model: &DFlash2Model,
        fused_all: &MxArray,
        lanes: &mut [(&mut DFlash2ContextCache, &[u32])],
    ) -> Result<()> {
        if lanes.is_empty() {
            return Ok(());
        }
        let mut segments = Vec::with_capacity(lanes.len());
        let mut total_rows = 0usize;
        for (cache, token_ids) in lanes.iter() {
            let keep = token_ids.len();
            if keep == 0 {
                return Err(Error::from_reason(
                    "DFlash2ContextCache::append_batched requires non-empty token_ids per lane",
                ));
            }
            if usize::try_from(cache.logical_len).ok() != Some(cache.token_history.len()) {
                return Err(Error::from_reason(format!(
                    "DFlash2 context provenance length {} does not match logical length {}",
                    cache.token_history.len(),
                    cache.logical_len
                )));
            }
            if cache.layers.len() != model.layers.len() {
                return Err(Error::from_reason(format!(
                    "DFlash2 context layer count {} != model layers {}",
                    cache.layers.len(),
                    model.layers.len()
                )));
            }
            let keep_i32 = i32::try_from(keep).map_err(|_| {
                Error::from_reason(format!(
                    "DFlash2 context append row count {keep} exceeds i32"
                ))
            })?;
            cache
                .logical_len
                .checked_add(keep_i32)
                .ok_or_else(|| Error::from_reason("DFlash2 context length overflow"))?;
            segments.push((cache.logical_len, keep));
            total_rows = total_rows
                .checked_add(keep)
                .ok_or_else(|| Error::from_reason("DFlash2 batched append total rows overflow"))?;
        }
        if fused_all.shape_at(0)? != 1 || fused_all.shape_at(1)? as usize != total_rows {
            return Err(Error::from_reason(format!(
                "DFlash2ContextCache::append_batched fused_all shape {:?} does not match [1, {total_rows}, _]",
                fused_all.shape()?.as_ref()
            )));
        }

        let per_lane_kv = model.project_context_batched(fused_all, &segments)?;
        for (((cache, token_ids), &(base, keep)), kv) in lanes
            .iter_mut()
            .zip(segments.iter())
            .zip(per_lane_kv.into_iter())
        {
            cache.append_projected(
                ProjectedContext {
                    base,
                    rows: keep as i64,
                    kv,
                },
                token_ids,
            )?;
        }
        Ok(())
    }

    /// Project a full eight-row verifier block, then store only its retained
    /// prefix. This schedule keeps FC and merged K/V on the M8 kernels;
    /// unretained rows never enter a cache or its token provenance.
    pub(crate) fn append_full_width_prefix(
        &mut self,
        model: &DFlash2Model,
        fused_context: &MxArray,
        base: i32,
        token_ids: &[u32],
    ) -> Result<()> {
        let keep = token_ids.len();
        if fused_context.shape_at(1)? != 8 || model.merged_rows() != 8 || keep == 0 || keep > 8 {
            return Err(Error::from_reason(
                "full-width DFlash2 context requires eight projected rows and 1..=8 retained rows",
            ));
        }
        if base != self.logical_len
            || usize::try_from(self.logical_len).ok() != Some(self.token_history.len())
            || self.layers.len() != model.layers.len()
        {
            return Err(Error::from_reason(
                "full-width DFlash2 context prefix has inconsistent position, token provenance, or layer count",
            ));
        }
        let next = self
            .logical_len
            .checked_add(keep as i32)
            .ok_or_else(|| Error::from_reason("DFlash2 context length overflow"))?;
        // Construct and validate every prefix before adopting any store handle.
        // Projection/shape failures therefore cannot partly advance a window.
        let prefixes = model.project_context_rows(fused_context, base, Some(keep as i64))?;
        if prefixes.len() != self.layers.len() {
            return Err(Error::from_reason(
                "full-width DFlash2 context projection lost a layer",
            ));
        }
        for (keys, values) in &prefixes {
            if keys.shape_at(2)? != keep as i64
                || values.shape()?.as_ref() != keys.shape()?.as_ref()
            {
                return Err(Error::from_reason(
                    "full-width DFlash2 context K/V geometry changed",
                ));
            }
        }
        for ((keys, values), window) in prefixes.iter().zip(self.layers.iter_mut()) {
            window.append(i64::from(base), keys, values)?;
        }
        self.logical_len = next;
        self.kv_end = next;
        self.token_history.extend_from_slice(token_ids);
        self.observe_tokens(token_ids);
        self.pre_propose_handles = None;
        Ok(())
    }

    /// Advance logical length and token provenance WITHOUT writing draft
    /// K/V. Prefill uses this for rows older than the sliding window's
    /// retained tail: those rows would be evicted by later appends unread,
    /// so their fc fusion + per-layer K/V projections are dead work.
    pub(crate) fn record_only(&mut self, token_ids: &[u32]) -> Result<()> {
        self.logical_len = self
            .logical_len
            .checked_add(i32::try_from(token_ids.len()).map_err(|_| {
                Error::from_reason(format!(
                    "DFlash2 context skip row count {} exceeds i32",
                    token_ids.len()
                ))
            })?)
            .ok_or_else(|| Error::from_reason("DFlash2 context length overflow"))?;
        self.token_history.extend_from_slice(token_ids);
        self.observe_tokens(token_ids);
        self.pre_propose_handles = None;
        Ok(())
    }

    pub(crate) fn logical_len(&self) -> i32 {
        self.logical_len
    }

    /// Bytes of every layer's ring buffers allocated so far.
    pub(crate) fn buffer_bytes(&self) -> u64 {
        self.layers.iter().fold(0u64, |sum, window| {
            sum.saturating_add(window.buffer_bytes())
        })
    }

    /// Host heap capacity (in bytes) held by the token provenance history.
    pub(crate) fn host_bytes(&self) -> u64 {
        (self.token_history.capacity() as u64).saturating_mul(std::mem::size_of::<u32>() as u64)
    }

    /// Drops every row and the provenance; the ring buffers stay for the
    /// next lane (Splash `QwenStateStorage::releaseLane`).
    pub(crate) fn reset(&mut self) {
        for window in &mut self.layers {
            window.reset();
        }
        self.logical_len = 0;
        self.kv_end = 0;
        self.token_history.clear();
        self.wide_vocab = false;
        self.pre_propose_handles = None;
    }

    #[cfg(test)]
    pub(crate) fn window_views(&self) -> Vec<Option<(MxArray, MxArray)>> {
        self.layers.iter().map(DraftKvWindow::view).collect()
    }

    pub(crate) fn token_history(&self) -> &[u32] {
        &self.token_history
    }

    pub(crate) fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        for window in &self.layers {
            window.collect_arrays(out);
        }
    }

    pub(crate) fn eval(&self) -> Result<()> {
        let mut arrays = Vec::new();
        self.collect_arrays(&mut arrays);
        if arrays.is_empty() {
            Ok(())
        } else {
            MxArray::eval_arrays(&arrays)
        }
    }

    /// Splash `QwenStateStorage::restore` with the draft side: the ring
    /// of a state at `lineage.len()`, whose newest rows (at most the ring's
    /// `sliding_window - 1`) are projected again from `window`'s 4-bit rows
    /// into `ring`'s buffers.
    pub(crate) fn restore(
        mut ring: DFlash2ContextCache,
        model: &DFlash2Model,
        window: &DraftContextWindow,
        lineage: &[u32],
    ) -> Result<DFlash2ContextCache> {
        let depth = u32::try_from(lineage.len())
            .map_err(|_| Error::from_reason("DFlash2 state lineage exceeds u32"))?;
        if window.end() != depth || window.width() as usize != model.config.hidden_size {
            return Err(Error::from_reason(format!(
                "DFlash2 context window [{}, {}) of width {} does not end the {depth}-token state",
                window.begin(),
                window.end(),
                window.width()
            )));
        }
        ring.reset();
        let ring_rows = (model.config.sliding_window as u32).saturating_sub(1);
        let first = window.begin().max(depth.saturating_sub(ring_rows));
        ring.record_only(&lineage[..first as usize])?;
        if first < depth {
            let rows = window.load_rows(first, depth - first)?;
            let fused = rows.reshape(&[1, i64::from(depth - first), i64::from(window.width())])?;
            let projected = ring.project(model, &fused, first as i32)?;
            ring.append_projected(projected, &lineage[first as usize..])?;
            ring.eval()?;
        }
        ring.clear_pre_propose_handles();
        Ok(ring)
    }
}
