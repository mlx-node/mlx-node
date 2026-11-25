use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use super::cache::DFlash2ContextCache;
use super::layers::{DraftLayerOutput, DraftResidual};
use super::selector::SelectorPath;
use super::spanned_head::{CachedDraftAf4G64, ResolvedDraftHead, SpannedDraftHeads};
use super::{DFlash2Model, EmbeddedLmHead};
use crate::array::{DType, MxArray};
use crate::models::quantized_linear::LinearProj;
use crate::nn::Embedding;

const COMPILED_DRAFT_TAG: u64 = 0xDFA2_0000_0000_0000;
static NEXT_COMPILED_DRAFT: AtomicU64 = AtomicU64::new(1);

pub(super) struct CompiledDraft {
    id: u64,
    pub(super) disabled: AtomicBool,
    pub(super) prepared: AtomicBool,
    pub(super) spanned_heads: Mutex<Option<Arc<SpannedDraftHeads>>>,
    pub(super) draft_af4g64: Mutex<Option<Arc<CachedDraftAf4G64>>>,
    #[cfg(test)]
    pub(super) traces: AtomicU64,
}

impl Default for CompiledDraft {
    fn default() -> Self {
        Self {
            id: NEXT_COMPILED_DRAFT.fetch_add(1, Ordering::Relaxed),
            disabled: AtomicBool::new(false),
            prepared: AtomicBool::new(false),
            spanned_heads: Mutex::new(None),
            draft_af4g64: Mutex::new(None),
            #[cfg(test)]
            traces: AtomicU64::new(0),
        }
    }
}

impl Drop for CompiledDraft {
    fn drop(&mut self) {
        crate::compiled_graph::erase_compiled_graphs_matching(
            0xffff_ffff_ffff_0000,
            COMPILED_DRAFT_TAG | ((self.id & 0xffff_ffff) << 16),
        );
    }
}

/// Power-of-2 bucketed KV context length (`[256, 512, 1024, ..., slots]`) for
/// CUDA compiled draft graphs so linear-window KV slices and masks keep fixed
/// shapes across decode cycles (100% CUDA Graph `graph_cache_` hits instead of
/// per-cycle re-instantiation, and seamlessly matching wrapped `Ring` mode's
/// `[0, slots + rows)` shape at the top bucket).
///
/// Leaves `slots < 256` unbucketed so small synthetic unit tests
/// (`sliding_window = 8 / 64`) remain bit-identical to eager.
fn bucketed_ctx_len_for_cuda(
    start: i64,
    end: i64,
    slots: i64,
    rows: i64,
    capacity: i64,
) -> Option<i64> {
    if slots < 256 || start < 0 || end <= start || end > slots {
        return None;
    }
    let mut bucket = 256i64;
    while bucket < end && bucket < slots {
        bucket = bucket.saturating_mul(2);
    }
    let bucket = bucket.min(slots);
    (bucket + rows <= capacity).then_some(bucket)
}

fn cuda_bucketed_ctx_len(
    start: i64,
    end: i64,
    slots: i64,
    rows: i64,
    capacity: i64,
) -> Option<i64> {
    if !crate::accelerator::cuda_ops_available() {
        return None;
    }
    bucketed_ctx_len_for_cuda(start, end, slots, rows, capacity)
}

/// Build the `[1, 1, rows, bucket_ctx + rows]` boolean attention mask for an
/// unwrapped linear draft window bucketed to `[0, bucket_ctx + rows)`:
/// - Context slots `[0, bucket_ctx)`: live rows `[start, end)` sit at physical
///   slots `start..end` (since `!wrapped() => end <= slots => p % slots == p`),
///   visible iff `query - k < window`; evicted `[0, start)` and unused gap
///   `[end, bucket_ctx)` are masked out (`false`).
/// - Tail proposal slots `[bucket_ctx, bucket_ctx + rows)`: written in place
///   by `store_window_tail`, at logical positions `end + j`, visible iff
///   `query - (end + j) < window`.
fn build_bucketed_linear_mask(
    query_base: i64,
    rows: i64,
    start: i64,
    end: i64,
    bucket_ctx: i64,
    window: i64,
) -> Result<MxArray> {
    let key_len = bucket_ctx + rows;
    let mut visible = vec![0u8; (rows * key_len) as usize];
    for r in 0..rows {
        let query = query_base + r;
        let row_off = (r * key_len) as usize;
        let lo = start.max(query - window + 1).clamp(0, end);
        if lo < end {
            visible[row_off + lo as usize..row_off + end as usize].fill(1);
        }
        for j in 0..rows {
            if query - (end + j) < window {
                visible[row_off + (bucket_ctx + j) as usize] = 1;
            }
        }
    }
    MxArray::from_uint8(&visible, &[1, 1, rows, key_len])?
        .not_equal(&MxArray::from_uint8(&[0], &[1])?)
}

impl DFlash2Model {
    /// Target weight replacement must also release draft tapes, which capture
    /// that target's embedding and output head as immutable constants.
    pub(crate) fn invalidate_compiled(&self) {
        crate::compiled_graph::erase_compiled_graphs_matching(
            0xffff_ffff_ffff_0000,
            COMPILED_DRAFT_TAG | ((self.compiled.id & 0xffff_ffff) << 16),
        );
        self.compiled.prepared.store(false, Ordering::Relaxed);
        if let Ok(mut guard) = self.compiled.spanned_heads.lock() {
            *guard = None;
        }
        if let Ok(mut guard) = self.compiled.draft_af4g64.lock() {
            *guard = None;
        }
    }

    /// Append optional `(mask, broadcast_mask)` and per-layer
    /// `[keys, values, keys_view, values_view]` inputs for a compiled draft call.
    /// Returns `Ok(Some((has_mask, windows_at)))` or `Ok(None)` when the context
    /// must fall back to the eager path.
    fn prepare_compiled_window_inputs(
        &self,
        context: &mut DFlash2ContextCache,
        rows: i64,
        inputs: &mut Vec<MxArray>,
    ) -> Result<Option<(bool, usize)>> {
        let (start, end) = (context.layers[0].start, context.layers[0].end);
        // An empty window has no full buffer owner; keep the eager path.
        if start == end
            || context
                .layers
                .iter()
                .any(|w| (w.start, w.end) != (start, end))
        {
            return Ok(None);
        }
        let slots = self.config.sliding_window as i64;
        let mut min_cap = i64::MAX;
        for window in &context.layers {
            let (Some(keys), Some(values)) = (&window.keys, &window.values) else {
                return Ok(None);
            };
            if keys.dtype()? != DType::BFloat16
                || values.dtype()? != DType::BFloat16
                || keys.shape_at(0)? != 1
            {
                return Ok(None);
            }
            let k_cap = keys.shape_at(2)?;
            if values.shape_at(2)? != k_cap {
                return Ok(None);
            }
            min_cap = min_cap.min(k_cap);
        }

        let bucket_ctx = if !context.layers[0].wrapped() {
            cuda_bucketed_ctx_len(start, end, slots, rows, min_cap)
        } else {
            None
        };

        let mask = if let Some(bctx) = bucket_ctx {
            Some(build_bucketed_linear_mask(
                i64::from(context.logical_len),
                rows,
                start,
                end,
                bctx,
                slots,
            )?)
        } else {
            context.layers[0].block_mask(
                &mut context.mask_memo,
                context.logical_len,
                rows,
                slots,
            )?
        };
        let has_mask = mask.is_some();
        if let Some(mask) = mask {
            let heads = self.layers[0].attention.num_heads;
            if self
                .layers
                .iter()
                .any(|layer| layer.attention.num_heads != heads)
            {
                return Ok(None);
            }
            let broadcast_mask = mask.broadcast_to(&[1, heads, rows, mask.shape_at(3)?])?;
            inputs.extend([mask, broadcast_mask]);
        }
        let windows_at = inputs.len();
        for window in &context.layers {
            let (Some(keys), Some(values)) = (&window.keys, &window.values) else {
                return Ok(None);
            };
            let (from, to) = if let Some(bctx) = bucket_ctx {
                (0, bctx + rows)
            } else {
                let Some(range) = window.block_range(rows)? else {
                    return Ok(None);
                };
                range
            };
            inputs.extend([
                keys.clone(),
                values.clone(),
                keys.slice_axis(2, from, to)?,
                values.slice_axis(2, from, to)?,
            ]);
        }
        Ok(Some((has_mask, windows_at)))
    }

    /// Compile the complete greedy proposal, including the target head and
    /// selector. Position, anchor, mask, window views and their full owners
    /// are inputs. The only host-dependent branches are model constants and
    /// proposal width/mask/confidence modes encoded in the function id.
    pub(super) fn propose_compiled(
        &self,
        target_embedding: &Embedding,
        target_lm_head: Option<&LinearProj>,
        draft_head: Option<&ResolvedDraftHead>,
        context: &mut DFlash2ContextCache,
        block: &MxArray,
        with_confidence: bool,
    ) -> Result<Option<(MxArray, Option<MxArray>)>> {
        if self.compiled.disabled.load(Ordering::Relaxed)
            || unsafe { sys::mlx_default_device() } != 1
            || self.selector.top_k > 64
            || self.layers.is_empty()
            || context.layers.len() != self.layers.len()
        {
            return Ok(None);
        }
        let rows = block.shape_at(1)?;
        if !(2..=256).contains(&rows) || block.shape_at(0)? != 1 {
            return Ok(None);
        }
        let mut inputs = vec![
            block.clone(),
            MxArray::from_int32(&[context.logical_len], &[1])?,
        ];
        let Some((has_mask, windows_at)) =
            self.prepare_compiled_window_inputs(context, rows, &mut inputs)?
        else {
            return Ok(None);
        };
        let spanned_layout = draft_head.and_then(ResolvedDraftHead::spanned_layout);
        let is_spanned = spanned_layout.is_some();
        let fn_id = COMPILED_DRAFT_TAG
            | ((self.compiled.id & 0xffff_ffff) << 16)
            | ((is_spanned as u64) << 11)
            | ((has_mask as u64) << 9)
            | ((with_confidence as u64) << 8)
            | (rows - 1) as u64;
        let cache_output_at = 1 + usize::from(with_confidence);
        let mut builder =
            |args: &[MxArray]| -> Result<Vec<MxArray>> {
                #[cfg(test)]
                self.compiled.traces.fetch_add(1, Ordering::Relaxed);
                let embedded = self.apply_mask_embedding(target_embedding.forward(&args[0])?)?;
                let mut pending: Option<DraftLayerOutput> = None;
                let mut buffers = Vec::with_capacity(self.layers.len() * 2);
                for (i, layer) in self.layers.iter().enumerate() {
                    let input = match &pending {
                        Some(output) => output.as_input(),
                        None => DraftResidual::Hidden(&embedded),
                    };
                    let at = windows_at + i * 4;
                    let (output, keys, values) = layer.forward_compiled(
                        input,
                        &args[at..at + 4],
                        &args[1],
                        has_mask.then(|| &args[2]),
                        has_mask.then(|| &args[3]),
                        self.merged_rows(),
                    )?;
                    pending = Some(output);
                    buffers.extend([keys, values]);
                }
                let output =
                    pending.ok_or_else(|| Error::from_reason("compiled draft has no layers"))?;
                let hidden = output.hidden_and_normalized(&self.norm)?.1;
                let draft_hidden = hidden.slice_axis(1, 1, rows)?;
                let logits = if let Some(head) = draft_head {
                    head.forward_logits(&draft_hidden)?
                } else if crate::accelerator::cuda_ops_available() {
                    // On CUDA, project only the proposal rows (skipping anchor row 0).
                    match target_lm_head {
                        Some(head) => head.forward_logits(&draft_hidden)?,
                        None => target_embedding.as_linear(&draft_hidden)?,
                    }
                } else {
                    // Match `propose_with`: run the full-vocab head on the whole
                    // block (anchor + proposals, 8 rows) so Metal keeps its
                    // K-quant M = 8 NAX route and eager/compiled stay bit-identical.
                    let full_logits = match target_lm_head {
                        Some(head) => head.forward_logits(&hidden)?,
                        None => target_embedding.as_linear(&hidden)?,
                    };
                    full_logits.slice_axis(1, 1, rows)?
                };
                let mut confidence = None;
                let anchor = args[0].slice_axis(1, 0, 1)?.reshape(&[1])?;
                let (path, _) = self.selector.select_with_anchor(
                    &draft_hidden,
                    &logits,
                    &anchor,
                    0.0,
                    true,
                    spanned_layout,
                    with_confidence.then_some(&mut confidence),
                    &mut rand::rng(),
                )?;
                let SelectorPath::Device(path) = path else {
                    return Err(Error::from_reason(
                        "compiled greedy draft returned a host path",
                    ));
                };
                let mut outputs = vec![path];
                if with_confidence {
                    outputs.push(confidence.ok_or_else(|| {
                        Error::from_reason("compiled draft confidence unavailable")
                    })?);
                }
                outputs.extend(buffers);
                Ok(outputs)
            };
        // Once invoked, a failure must reset the turn: an in-place primitive
        // may already have recorded writes. Never retry the same prefix eager.
        let outputs = crate::compiled_graph::invoke_compiled_graph_with_mode(
            fn_id,
            &inputs.iter().collect::<Vec<_>>(),
            cache_output_at + self.layers.len() * 2,
            true,
            crate::accelerator::cuda_draft_no_fuse(),
            &mut builder,
        )?
        .ok_or_else(|| Error::from_reason("compiled draft invocation failed"))?;
        for (window, pair) in context
            .layers
            .iter()
            .zip(outputs[cache_output_at..].as_chunks::<2>().0)
        {
            let (Some(keys), Some(values)) = (&window.keys, &window.values) else {
                return Err(Error::from_reason(
                    "compiled draft returned incompatible cache buffers",
                ));
            };
            if pair[0].shape()?.as_ref() != keys.shape()?.as_ref()
                || pair[1].shape()?.as_ref() != values.shape()?.as_ref()
                || pair[0].dtype()? != DType::BFloat16
                || pair[1].dtype()? != DType::BFloat16
            {
                return Err(Error::from_reason(
                    "compiled draft returned incompatible cache buffers",
                ));
            }
        }
        for (window, pair) in context
            .layers
            .iter_mut()
            .zip(outputs[cache_output_at..].as_chunks::<2>().0)
        {
            window.keys = Some(pair[0].clone());
            window.values = Some(pair[1].clone());
        }
        Ok(Some((
            outputs[0].clone(),
            with_confidence.then(|| outputs[1].clone()),
        )))
    }

    /// Compiled greedy proposal taking pre-embedded `[1, rows, hidden]` BF16
    /// input (for targets whose token table lives outside an `Embedding`
    /// struct, such as `qwen4_exp` or `inkling2`), capturing the resident
    /// target `lm_head` (`LinearProj::Quantized` or `LinearProj::Standard`, or
    /// its spanned view) as a graph constant.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn propose_compiled_embedded(
        &self,
        embedded: &MxArray,
        anchor: u32,
        target_lm_head: EmbeddedLmHead<'_>,
        draft_head: Option<&ResolvedDraftHead>,
        context: &mut DFlash2ContextCache,
        head_rows: i64,
        with_confidence: bool,
    ) -> Result<Option<(MxArray, Option<MxArray>)>> {
        if self.compiled.disabled.load(Ordering::Relaxed)
            || unsafe { sys::mlx_default_device() } != 1
            || self.selector.top_k > 64
            || self.layers.is_empty()
            || context.layers.len() != self.layers.len()
            || embedded.dtype()? != DType::BFloat16
        {
            return Ok(None);
        }
        let rows = embedded.shape_at(1)?;
        if !(2..=256).contains(&rows)
            || embedded.shape_at(0)? != 1
            || head_rows < 1
            || head_rows >= rows
            || (head_rows != rows - 1 && head_rows > 7)
        {
            return Ok(None);
        }
        let mut inputs = vec![
            embedded.clone(),
            MxArray::from_int32(&[context.logical_len], &[1])?,
            MxArray::from_int32(&[anchor as i32], &[1])?,
        ];
        let Some((has_mask, windows_at)) =
            self.prepare_compiled_window_inputs(context, rows, &mut inputs)?
        else {
            return Ok(None);
        };
        const COMPILED_EMBEDDED_BIT: u64 = 1u64 << 12;
        let head_tag = if head_rows == rows - 1 {
            0u64
        } else {
            (head_rows as u64 & 0x7) << 13
        };
        let spanned_layout = draft_head.and_then(ResolvedDraftHead::spanned_layout);
        let is_spanned = spanned_layout.is_some();
        let fn_id = COMPILED_DRAFT_TAG
            | ((self.compiled.id & 0xffff_ffff) << 16)
            | head_tag
            | COMPILED_EMBEDDED_BIT
            | ((is_spanned as u64) << 11)
            | ((has_mask as u64) << 9)
            | ((with_confidence as u64) << 8)
            | (rows - 1) as u64;
        let cache_output_at = 1 + usize::from(with_confidence);
        let mut builder =
            |args: &[MxArray]| -> Result<Vec<MxArray>> {
                #[cfg(test)]
                self.compiled.traces.fetch_add(1, Ordering::Relaxed);
                let mut pending: Option<DraftLayerOutput> = None;
                let mut buffers = Vec::with_capacity(self.layers.len() * 2);
                for (i, layer) in self.layers.iter().enumerate() {
                    let input = match &pending {
                        Some(output) => output.as_input(),
                        None => DraftResidual::Hidden(&args[0]),
                    };
                    let at = windows_at + i * 4;
                    let (output, keys, values) = layer.forward_compiled(
                        input,
                        &args[at..at + 4],
                        &args[1],
                        has_mask.then(|| &args[3]),
                        has_mask.then(|| &args[4]),
                        self.merged_rows(),
                    )?;
                    pending = Some(output);
                    buffers.extend([keys, values]);
                }
                let output =
                    pending.ok_or_else(|| Error::from_reason("compiled draft has no layers"))?;
                let hidden = output.hidden_and_normalized(&self.norm)?.1;
                let draft_hidden = hidden.slice_axis(1, 1, 1 + head_rows)?;
                let logits = if let Some(head) = draft_head {
                    head.forward_logits(&draft_hidden)?
                } else if crate::accelerator::cuda_ops_available() {
                    target_lm_head.forward_logits(&draft_hidden)?
                } else {
                    // Match `propose_with` / `propose_compiled`: run the full-vocab head
                    // on the whole block (anchor + proposals, M=8) so Metal keeps its
                    // K-quant M = 8 NAX route and eager/compiled stay bit-identical.
                    target_lm_head
                        .forward_logits(&hidden)?
                        .slice_axis(1, 1, 1 + head_rows)?
                };
                let mut confidence = None;
                let (path, _) = self.selector.select_with_anchor(
                    &draft_hidden,
                    &logits,
                    &args[2],
                    0.0,
                    true,
                    spanned_layout,
                    with_confidence.then_some(&mut confidence),
                    &mut rand::rng(),
                )?;
                let SelectorPath::Device(path) = path else {
                    return Err(Error::from_reason(
                        "compiled greedy draft returned a host path",
                    ));
                };
                let mut outputs = vec![path];
                if with_confidence {
                    outputs.push(confidence.ok_or_else(|| {
                        Error::from_reason("compiled draft confidence unavailable")
                    })?);
                }
                outputs.extend(buffers);
                Ok(outputs)
            };
        let outputs = crate::compiled_graph::invoke_compiled_graph_with_mode(
            fn_id,
            &inputs.iter().collect::<Vec<_>>(),
            cache_output_at + self.layers.len() * 2,
            true,
            crate::accelerator::cuda_draft_no_fuse(),
            &mut builder,
        )?
        .ok_or_else(|| Error::from_reason("compiled embedded draft invocation failed"))?;
        for (window, pair) in context
            .layers
            .iter()
            .zip(outputs[cache_output_at..].as_chunks::<2>().0)
        {
            let (Some(keys), Some(values)) = (&window.keys, &window.values) else {
                return Err(Error::from_reason(
                    "compiled draft returned incompatible cache buffers",
                ));
            };
            if pair[0].shape()?.as_ref() != keys.shape()?.as_ref()
                || pair[1].shape()?.as_ref() != values.shape()?.as_ref()
                || pair[0].dtype()? != DType::BFloat16
                || pair[1].dtype()? != DType::BFloat16
            {
                return Err(Error::from_reason(
                    "compiled draft returned incompatible cache buffers",
                ));
            }
        }
        for (window, pair) in context
            .layers
            .iter_mut()
            .zip(outputs[cache_output_at..].as_chunks::<2>().0)
        {
            window.keys = Some(pair[0].clone());
            window.values = Some(pair[1].clone());
        }
        Ok(Some((
            outputs[0].clone(),
            with_confidence.then(|| outputs[1].clone()),
        )))
    }

    /// Compiled tree-lattice proposal returning `(candidates [1, L, K] Int32,
    /// scores [L, K, K] Float32)`. Uses free bit 10 (`1 << 10`) in `fn_id`
    /// under `COMPILED_DRAFT_TAG` so `Drop` and `invalidate_compiled` release
    /// it automatically.
    pub(super) fn propose_tree_lattice_compiled(
        &self,
        target_embedding: &Embedding,
        target_lm_head: Option<&LinearProj>,
        draft_head: Option<&ResolvedDraftHead>,
        context: &mut DFlash2ContextCache,
        block: &MxArray,
    ) -> Result<Option<(MxArray, MxArray)>> {
        if self.compiled.disabled.load(Ordering::Relaxed)
            || unsafe { sys::mlx_default_device() } != 1
            || self.selector.top_k > 64
            || self.layers.is_empty()
            || context.layers.len() != self.layers.len()
        {
            return Ok(None);
        }
        let rows = block.shape_at(1)?;
        if !(2..=256).contains(&rows) || block.shape_at(0)? != 1 {
            return Ok(None);
        }
        let mut inputs = vec![
            block.clone(),
            MxArray::from_int32(&[context.logical_len], &[1])?,
        ];
        let Some((has_mask, windows_at)) =
            self.prepare_compiled_window_inputs(context, rows, &mut inputs)?
        else {
            return Ok(None);
        };
        const COMPILED_LATTICE_BIT: u64 = 1u64 << 10;
        let spanned_layout = draft_head.and_then(ResolvedDraftHead::spanned_layout);
        let is_spanned = spanned_layout.is_some();
        let fn_id = COMPILED_DRAFT_TAG
            | ((self.compiled.id & 0xffff_ffff) << 16)
            | ((is_spanned as u64) << 11)
            | COMPILED_LATTICE_BIT
            | ((has_mask as u64) << 9)
            | (rows - 1) as u64;
        let cache_output_at = 2;
        let mut builder = |args: &[MxArray]| -> Result<Vec<MxArray>> {
            #[cfg(test)]
            self.compiled.traces.fetch_add(1, Ordering::Relaxed);
            let embedded = self.apply_mask_embedding(target_embedding.forward(&args[0])?)?;
            let mut pending: Option<DraftLayerOutput> = None;
            let mut buffers = Vec::with_capacity(self.layers.len() * 2);
            for (i, layer) in self.layers.iter().enumerate() {
                let input = match &pending {
                    Some(output) => output.as_input(),
                    None => DraftResidual::Hidden(&embedded),
                };
                let at = windows_at + i * 4;
                let (output, keys, values) = layer.forward_compiled(
                    input,
                    &args[at..at + 4],
                    &args[1],
                    has_mask.then(|| &args[2]),
                    has_mask.then(|| &args[3]),
                    self.merged_rows(),
                )?;
                pending = Some(output);
                buffers.extend([keys, values]);
            }
            let output =
                pending.ok_or_else(|| Error::from_reason("compiled draft has no layers"))?;
            let hidden = output.hidden_and_normalized(&self.norm)?.1;
            let draft_hidden = hidden.slice_axis(1, 1, rows)?;
            let logits = if let Some(head) = draft_head {
                head.forward_logits(&draft_hidden)?
            } else if crate::accelerator::cuda_ops_available() {
                // On CUDA, project only the proposal rows (skipping anchor row 0).
                match target_lm_head {
                    Some(head) => head.forward_logits(&draft_hidden)?,
                    None => target_embedding.as_linear(&draft_hidden)?,
                }
            } else {
                // Match `propose_tree_lattice_arrays`: run the full-vocab head on
                // the whole block (anchor + proposals) so Metal keeps its K-quant
                // M = 8 NAX route and eager/compiled stay bit-identical.
                let full_logits = match target_lm_head {
                    Some(head) => head.forward_logits(&hidden)?,
                    None => target_embedding.as_linear(&hidden)?,
                };
                full_logits.slice_axis(1, 1, rows)?
            };
            let anchor = args[0].slice_axis(1, 0, 1)?.reshape(&[1])?;
            let (candidates, scores) = self.selector.select_lattice_arrays(
                &draft_hidden,
                &logits,
                &anchor,
                spanned_layout,
            )?;
            let mut outputs = vec![candidates, scores];
            outputs.extend(buffers);
            Ok(outputs)
        };
        // Once invoked, a failure must reset the turn: an in-place primitive
        // may already have recorded writes. Never retry the same prefix eager.
        let outputs = crate::compiled_graph::invoke_compiled_graph_with_mode(
            fn_id,
            &inputs.iter().collect::<Vec<_>>(),
            cache_output_at + self.layers.len() * 2,
            true,
            crate::accelerator::cuda_draft_no_fuse(),
            &mut builder,
        )?
        .ok_or_else(|| Error::from_reason("compiled draft lattice invocation failed"))?;
        for (window, pair) in context
            .layers
            .iter()
            .zip(outputs[cache_output_at..].as_chunks::<2>().0)
        {
            let (Some(keys), Some(values)) = (&window.keys, &window.values) else {
                return Err(Error::from_reason(
                    "compiled draft lattice returned incompatible cache buffers",
                ));
            };
            if pair[0].shape()?.as_ref() != keys.shape()?.as_ref()
                || pair[1].shape()?.as_ref() != values.shape()?.as_ref()
                || pair[0].dtype()? != DType::BFloat16
                || pair[1].dtype()? != DType::BFloat16
            {
                return Err(Error::from_reason(
                    "compiled draft lattice returned incompatible cache buffers",
                ));
            }
        }
        for (window, pair) in context
            .layers
            .iter_mut()
            .zip(outputs[cache_output_at..].as_chunks::<2>().0)
        {
            window.keys = Some(pair[0].clone());
            window.values = Some(pair[1].clone());
        }
        Ok(Some((outputs[0].clone(), outputs[1].clone())))
    }

    /// Compiled tree-lattice proposal for targets with external embeddings
    /// (`qwen4_exp`, `inkling2`), taking pre-embedded `[1, block_size + 1, hidden]`
    /// BF16 rows and a resident `lm_head` (`LinearProj::Quantized` or
    /// `LinearProj::Standard`, plus optional spanned head
    /// `[0, 98_304) U [248_032, 248_320)`).
    ///
    /// Uses `(1 << 12) | (1 << 10)` (`COMPILED_EMBEDDED_BIT | COMPILED_LATTICE_BIT`)
    /// under `COMPILED_DRAFT_TAG` so it never collides with linear compiled
    /// embedded graphs or dense `propose_tree_lattice_compiled`, and is erased
    /// automatically by `invalidate_compiled` and `Drop`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn propose_tree_lattice_compiled_embedded(
        &self,
        embedded: &MxArray,
        anchor: u32,
        target_lm_head: EmbeddedLmHead<'_>,
        draft_head: Option<&ResolvedDraftHead>,
        context: &mut DFlash2ContextCache,
    ) -> Result<Option<(MxArray, MxArray)>> {
        if self.compiled.disabled.load(Ordering::Relaxed)
            || unsafe { sys::mlx_default_device() } != 1
            || self.selector.top_k > 64
            || self.layers.is_empty()
            || context.layers.len() != self.layers.len()
            || embedded.dtype()? != DType::BFloat16
        {
            return Ok(None);
        }
        let rows = embedded.shape_at(1)?;
        if !(2..=256).contains(&rows) || embedded.shape_at(0)? != 1 {
            return Ok(None);
        }
        let mut inputs = vec![
            embedded.clone(),
            MxArray::from_int32(&[context.logical_len], &[1])?,
            MxArray::from_int32(&[anchor as i32], &[1])?,
        ];
        let Some((has_mask, windows_at)) =
            self.prepare_compiled_window_inputs(context, rows, &mut inputs)?
        else {
            return Ok(None);
        };
        const COMPILED_LATTICE_BIT: u64 = 1u64 << 10;
        const COMPILED_EMBEDDED_BIT: u64 = 1u64 << 12;
        let spanned_layout = draft_head.and_then(ResolvedDraftHead::spanned_layout);
        let is_spanned = spanned_layout.is_some();
        let fn_id = COMPILED_DRAFT_TAG
            | ((self.compiled.id & 0xffff_ffff) << 16)
            | COMPILED_EMBEDDED_BIT
            | ((is_spanned as u64) << 11)
            | COMPILED_LATTICE_BIT
            | ((has_mask as u64) << 9)
            | (rows - 1) as u64;
        let cache_output_at = 2;
        let mut builder = |args: &[MxArray]| -> Result<Vec<MxArray>> {
            #[cfg(test)]
            self.compiled.traces.fetch_add(1, Ordering::Relaxed);
            let mut pending: Option<DraftLayerOutput> = None;
            let mut buffers = Vec::with_capacity(self.layers.len() * 2);
            for (i, layer) in self.layers.iter().enumerate() {
                let input = match &pending {
                    Some(output) => output.as_input(),
                    None => DraftResidual::Hidden(&args[0]),
                };
                let at = windows_at + i * 4;
                let (output, keys, values) = layer.forward_compiled(
                    input,
                    &args[at..at + 4],
                    &args[1],
                    has_mask.then(|| &args[3]),
                    has_mask.then(|| &args[4]),
                    self.merged_rows(),
                )?;
                pending = Some(output);
                buffers.extend([keys, values]);
            }
            let output =
                pending.ok_or_else(|| Error::from_reason("compiled draft has no layers"))?;
            let hidden = output.hidden_and_normalized(&self.norm)?.1;
            let draft_hidden = hidden.slice_axis(1, 1, rows)?;
            let logits = if let Some(head) = draft_head {
                head.forward_logits(&draft_hidden)?
            } else if crate::accelerator::cuda_ops_available() {
                // On CUDA, project only the `block_size` proposal rows (M = 7,
                // skipping unused anchor row 0) so `lm_head` hits M_TILES=1
                // directly without computing or slicing row 0.
                target_lm_head.forward_logits(&draft_hidden)?
            } else {
                // Match `propose_compiled` / `propose_tree_lattice_compiled`: run the
                // full-vocab head on the whole block (anchor + proposals, M=8) so
                // Metal keeps its K-quant M = 8 NAX route and eager/compiled stay
                // bit-identical.
                target_lm_head
                    .forward_logits(&hidden)?
                    .slice_axis(1, 1, rows)?
            };
            let (candidates, scores) = self.selector.select_lattice_arrays(
                &draft_hidden,
                &logits,
                &args[2],
                spanned_layout,
            )?;
            let mut outputs = vec![candidates, scores];
            outputs.extend(buffers);
            Ok(outputs)
        };
        let outputs = crate::compiled_graph::invoke_compiled_graph_with_mode(
            fn_id,
            &inputs.iter().collect::<Vec<_>>(),
            cache_output_at + self.layers.len() * 2,
            true,
            crate::accelerator::cuda_draft_no_fuse(),
            &mut builder,
        )?
        .ok_or_else(|| Error::from_reason("compiled embedded draft lattice invocation failed"))?;
        for (window, pair) in context
            .layers
            .iter()
            .zip(outputs[cache_output_at..].as_chunks::<2>().0)
        {
            let (Some(keys), Some(values)) = (&window.keys, &window.values) else {
                return Err(Error::from_reason(
                    "compiled embedded draft lattice returned incompatible cache buffers",
                ));
            };
            if pair[0].shape()?.as_ref() != keys.shape()?.as_ref()
                || pair[1].shape()?.as_ref() != values.shape()?.as_ref()
                || pair[0].dtype()? != DType::BFloat16
                || pair[1].dtype()? != DType::BFloat16
            {
                return Err(Error::from_reason(
                    "compiled embedded draft lattice returned incompatible cache buffers",
                ));
            }
        }
        for (window, pair) in context
            .layers
            .iter_mut()
            .zip(outputs[cache_output_at..].as_chunks::<2>().0)
        {
            window.keys = Some(pair[0].clone());
            window.values = Some(pair[1].clone());
        }
        Ok(Some((outputs[0].clone(), outputs[1].clone())))
    }

    /// Compiled draft transformer body taking pre-embedded `[1, rows, hidden]`
    /// BF16 rows and returning final-normed `[1, rows, hidden]` BF16 rows while
    /// updating every layer's KV window tail.
    ///
    /// Used on Metal for targets whose `embed()` and `lm_head` run outside
    /// `mlx::compile` (e.g. Inkling-2-Mini with `embed_norm` + custom Metal
    /// Sym-Q8_0 / BF16 `lm_head`). Uses `(1 << 10) | (1 << 8)` under
    /// `COMPILED_DRAFT_TAG`, which is disjoint from all existing compiled
    /// draft graph IDs and automatically erased by `invalidate_compiled`/`Drop`.
    pub(super) fn forward_embedded_compiled(
        &self,
        embedded: &MxArray,
        context: &mut DFlash2ContextCache,
    ) -> Result<Option<MxArray>> {
        if self.compiled.disabled.load(Ordering::Relaxed)
            || unsafe { sys::mlx_default_device() } != 1
            || self.layers.is_empty()
            || context.layers.len() != self.layers.len()
            || embedded.dtype()? != DType::BFloat16
        {
            return Ok(None);
        }
        let rows = embedded.shape_at(1)?;
        if !(2..=256).contains(&rows) || embedded.shape_at(0)? != 1 {
            return Ok(None);
        }
        let mut inputs = vec![
            embedded.clone(),
            MxArray::from_int32(&[context.logical_len], &[1])?,
        ];
        let Some((has_mask, windows_at)) =
            self.prepare_compiled_window_inputs(context, rows, &mut inputs)?
        else {
            return Ok(None);
        };
        const COMPILED_BODY_BITS: u64 = (1u64 << 10) | (1u64 << 8);
        let fn_id = COMPILED_DRAFT_TAG
            | ((self.compiled.id & 0xffff_ffff) << 16)
            | COMPILED_BODY_BITS
            | ((has_mask as u64) << 9)
            | (rows - 1) as u64;
        let cache_output_at = 1;
        let mut builder = |args: &[MxArray]| -> Result<Vec<MxArray>> {
            #[cfg(test)]
            self.compiled.traces.fetch_add(1, Ordering::Relaxed);
            let mut pending: Option<DraftLayerOutput> = None;
            let mut buffers = Vec::with_capacity(self.layers.len() * 2);
            for (i, layer) in self.layers.iter().enumerate() {
                let input = match &pending {
                    Some(output) => output.as_input(),
                    None => DraftResidual::Hidden(&args[0]),
                };
                let at = windows_at + i * 4;
                let (output, keys, values) = layer.forward_compiled(
                    input,
                    &args[at..at + 4],
                    &args[1],
                    has_mask.then(|| &args[2]),
                    has_mask.then(|| &args[3]),
                    self.merged_rows(),
                )?;
                pending = Some(output);
                buffers.extend([keys, values]);
            }
            let output =
                pending.ok_or_else(|| Error::from_reason("compiled draft body has no layers"))?;
            let hidden = output.hidden_and_normalized(&self.norm)?.1;
            let mut outputs = vec![hidden];
            outputs.extend(buffers);
            Ok(outputs)
        };
        let outputs = crate::compiled_graph::invoke_compiled_graph_with_mode(
            fn_id,
            &inputs.iter().collect::<Vec<_>>(),
            cache_output_at + self.layers.len() * 2,
            true,
            crate::accelerator::cuda_draft_no_fuse(),
            &mut builder,
        )?
        .ok_or_else(|| Error::from_reason("compiled draft body invocation failed"))?;
        for (window, pair) in context
            .layers
            .iter()
            .zip(outputs[cache_output_at..].as_chunks::<2>().0)
        {
            let (Some(keys), Some(values)) = (&window.keys, &window.values) else {
                return Err(Error::from_reason(
                    "compiled draft body returned incompatible cache buffers",
                ));
            };
            if pair[0].shape()?.as_ref() != keys.shape()?.as_ref()
                || pair[1].shape()?.as_ref() != values.shape()?.as_ref()
                || pair[0].dtype()? != DType::BFloat16
                || pair[1].dtype()? != DType::BFloat16
            {
                return Err(Error::from_reason(
                    "compiled draft body returned incompatible cache buffers",
                ));
            }
        }
        for (window, pair) in context
            .layers
            .iter_mut()
            .zip(outputs[cache_output_at..].as_chunks::<2>().0)
        {
            window.keys = Some(pair[0].clone());
            window.values = Some(pair[1].clone());
        }
        Ok(Some(outputs[0].clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_bucketed_ctx_len_covers_pow2_buckets_and_preserves_small_windows() {
        // Small synthetic windows (< 256) stay unbucketed for exact eager parity tests.
        assert_eq!(bucketed_ctx_len_for_cuda(0, 3, 8, 3, 16), None);
        assert_eq!(bucketed_ctx_len_for_cuda(0, 16, 64, 8, 72), None);

        // Production 2048-slot ring (capacity = 2056, rows = 8): 4 fixed buckets.
        assert_eq!(bucketed_ctx_len_for_cuda(0, 1, 2048, 8, 2056), Some(256));
        assert_eq!(bucketed_ctx_len_for_cuda(0, 256, 2048, 8, 2056), Some(256));
        assert_eq!(bucketed_ctx_len_for_cuda(0, 257, 2048, 8, 2056), Some(512));
        assert_eq!(bucketed_ctx_len_for_cuda(0, 512, 2048, 8, 2056), Some(512));
        assert_eq!(bucketed_ctx_len_for_cuda(0, 513, 2048, 8, 2056), Some(1024));
        assert_eq!(
            bucketed_ctx_len_for_cuda(0, 1024, 2048, 8, 2056),
            Some(1024)
        );
        assert_eq!(
            bucketed_ctx_len_for_cuda(0, 1025, 2048, 8, 2056),
            Some(2048)
        );
        assert_eq!(
            bucketed_ctx_len_for_cuda(1, 2048, 2048, 8, 2056),
            Some(2048)
        );

        // Wrapped (end > slots), empty, or insufficient capacity returns None.
        assert_eq!(bucketed_ctx_len_for_cuda(2, 2049, 2048, 8, 2056), None);
        assert_eq!(bucketed_ctx_len_for_cuda(0, 0, 2048, 8, 2056), None);
        assert_eq!(bucketed_ctx_len_for_cuda(0, 200, 2048, 8, 260), None);
    }

    #[test]
    fn bucketed_linear_mask_matches_exact_visibility_and_masks_gap() {
        // Unwrapped window near sliding boundary: start=1, end=2048, bucket=2048, rows=8.
        let mask = build_bucketed_linear_mask(2048, 8, 1, 2048, 2048, 2048).unwrap();
        assert_eq!(mask.shape().unwrap().as_ref(), [1, 1, 8, 2056]);
        let vals = mask.astype(DType::Int32).unwrap().to_int32().unwrap();
        for r in 0..8usize {
            let query = 2048 + r as i64;
            let row = &vals[r * 2056..(r + 1) * 2056];
            for (k, &val) in row.iter().enumerate().take(2048usize) {
                let expected = i32::from(k >= 1 && (query - k as i64) < 2048);
                assert_eq!(val, expected, "r={r} k={k}");
            }
            for j in 0..8usize {
                assert_eq!(row[2048 + j], 1, "r={r} tail j={j}");
            }
        }

        // Smaller bucket with gap slots [100..256) masked out.
        let mask2 = build_bucketed_linear_mask(100, 8, 0, 100, 256, 2048).unwrap();
        assert_eq!(mask2.shape().unwrap().as_ref(), [1, 1, 8, 264]);
        let vals2 = mask2.astype(DType::Int32).unwrap().to_int32().unwrap();
        for r in 0..8usize {
            let row = &vals2[r * 264..(r + 1) * 264];
            assert!(row[..100].iter().all(|&v| v == 1));
            assert!(row[100..256].iter().all(|&v| v == 0));
            assert!(row[256..264].iter().all(|&v| v == 1));
        }
    }
}
