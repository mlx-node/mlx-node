use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use super::DFlash2Model;
use super::cache::DFlash2ContextCache;
use super::layers::{DraftLayerOutput, DraftResidual};
use super::selector::SelectorPath;
use super::spanned_head::SpannedDraftHeads;
use crate::array::{DType, MxArray};
use crate::models::quantized_linear::{LinearProj, QuantizedLinear};
use crate::nn::Embedding;

const COMPILED_DRAFT_TAG: u64 = 0xDFA2_0000_0000_0000;
static NEXT_COMPILED_DRAFT: AtomicU64 = AtomicU64::new(1);

pub(super) struct CompiledDraft {
    id: u64,
    pub(super) disabled: AtomicBool,
    pub(super) prepared: AtomicBool,
    pub(super) spanned_heads: Mutex<Option<Arc<SpannedDraftHeads>>>,
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
    }

    /// Compile the complete greedy proposal, including the target head and
    /// selector. Position, anchor, mask, window views and their full owners
    /// are inputs. The only host-dependent branches are model constants and
    /// proposal width/mask/confidence modes encoded in the function id.
    pub(super) fn propose_compiled(
        &self,
        target_embedding: &Embedding,
        target_lm_head: Option<&LinearProj>,
        spanned_heads: Option<&SpannedDraftHeads>,
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
        let mut inputs = vec![
            block.clone(),
            MxArray::from_int32(&[context.logical_len], &[1])?,
        ];
        let mask = context.layers[0].block_mask(
            &mut context.mask_memo,
            context.logical_len,
            rows,
            self.config.sliding_window as i64,
        )?;
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
            let Some((from, to)) = window.block_range(rows)? else {
                return Ok(None);
            };
            if keys.dtype()? != DType::BFloat16
                || values.dtype()? != DType::BFloat16
                || keys.shape_at(0)? != 1
            {
                return Ok(None);
            }
            inputs.extend([
                keys.clone(),
                values.clone(),
                keys.slice_axis(2, from, to)?,
                values.slice_axis(2, from, to)?,
            ]);
        }
        let is_spanned = spanned_heads.is_some();
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
                let embedded = target_embedding.forward(&args[0])?;
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
                let logits = if let Some(spanned) = spanned_heads {
                    spanned.forward_logits(&draft_hidden)?
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
                    is_spanned,
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
    /// struct, such as `qwen4_exp`), capturing the resident quantized target
    /// `lm_head` (or its spanned view) as a graph constant.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn propose_compiled_embedded(
        &self,
        embedded: &MxArray,
        anchor: u32,
        target_lm_head: &QuantizedLinear,
        spanned_heads: Option<&SpannedDraftHeads>,
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
        let (start, end) = (context.layers[0].start, context.layers[0].end);
        if start == end
            || context
                .layers
                .iter()
                .any(|w| (w.start, w.end) != (start, end))
        {
            return Ok(None);
        }
        let mut inputs = vec![
            embedded.clone(),
            MxArray::from_int32(&[context.logical_len], &[1])?,
            MxArray::from_int32(&[anchor as i32], &[1])?,
        ];
        let mask = context.layers[0].block_mask(
            &mut context.mask_memo,
            context.logical_len,
            rows,
            self.config.sliding_window as i64,
        )?;
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
            let Some((from, to)) = window.block_range(rows)? else {
                return Ok(None);
            };
            if keys.dtype()? != DType::BFloat16
                || values.dtype()? != DType::BFloat16
                || keys.shape_at(0)? != 1
            {
                return Ok(None);
            }
            inputs.extend([
                keys.clone(),
                values.clone(),
                keys.slice_axis(2, from, to)?,
                values.slice_axis(2, from, to)?,
            ]);
        }
        const COMPILED_EMBEDDED_BIT: u64 = 1u64 << 12;
        let head_tag = if head_rows == rows - 1 {
            0u64
        } else {
            (head_rows as u64 & 0x7) << 13
        };
        let is_spanned = spanned_heads.is_some();
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
                let logits = if let Some(spanned) = spanned_heads {
                    spanned.forward_logits(&draft_hidden)?
                } else if head_rows < rows - 1 && crate::accelerator::cuda_ops_available() {
                    target_lm_head.forward(&draft_hidden)?
                } else {
                    // Match `propose_with` / `propose_compiled`: run the full-vocab head
                    // on the whole block (anchor + proposals, M=8) so Metal keeps its
                    // K-quant M = 8 NAX route and eager/compiled stay bit-identical.
                    target_lm_head
                        .forward(&hidden)?
                        .slice_axis(1, 1, 1 + head_rows)?
                };
                let mut confidence = None;
                let (path, _) = self.selector.select_with_anchor(
                    &draft_hidden,
                    &logits,
                    &args[2],
                    0.0,
                    true,
                    is_spanned,
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
        spanned_heads: Option<&SpannedDraftHeads>,
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
        let mut inputs = vec![
            block.clone(),
            MxArray::from_int32(&[context.logical_len], &[1])?,
        ];
        let mask = context.layers[0].block_mask(
            &mut context.mask_memo,
            context.logical_len,
            rows,
            self.config.sliding_window as i64,
        )?;
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
            let Some((from, to)) = window.block_range(rows)? else {
                return Ok(None);
            };
            if keys.dtype()? != DType::BFloat16
                || values.dtype()? != DType::BFloat16
                || keys.shape_at(0)? != 1
            {
                return Ok(None);
            }
            inputs.extend([
                keys.clone(),
                values.clone(),
                keys.slice_axis(2, from, to)?,
                values.slice_axis(2, from, to)?,
            ]);
        }
        const COMPILED_LATTICE_BIT: u64 = 1u64 << 10;
        let is_spanned = spanned_heads.is_some();
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
            let embedded = target_embedding.forward(&args[0])?;
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
            let logits = if let Some(spanned) = spanned_heads {
                spanned.forward_logits(&draft_hidden)?
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
            let (candidates, scores) =
                self.selector
                    .select_lattice_arrays(&draft_hidden, &logits, &anchor, is_spanned)?;
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
    /// (`qwen4_exp`), taking pre-embedded `[1, block_size + 1, hidden]` BF16
    /// rows and a resident quantized `lm_head` (plus optional spanned head
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
        target_lm_head: &QuantizedLinear,
        spanned_heads: Option<&SpannedDraftHeads>,
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
        let (start, end) = (context.layers[0].start, context.layers[0].end);
        if start == end
            || context
                .layers
                .iter()
                .any(|w| (w.start, w.end) != (start, end))
        {
            return Ok(None);
        }
        let mut inputs = vec![
            embedded.clone(),
            MxArray::from_int32(&[context.logical_len], &[1])?,
            MxArray::from_int32(&[anchor as i32], &[1])?,
        ];
        let mask = context.layers[0].block_mask(
            &mut context.mask_memo,
            context.logical_len,
            rows,
            self.config.sliding_window as i64,
        )?;
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
            let Some((from, to)) = window.block_range(rows)? else {
                return Ok(None);
            };
            if keys.dtype()? != DType::BFloat16
                || values.dtype()? != DType::BFloat16
                || keys.shape_at(0)? != 1
            {
                return Ok(None);
            }
            inputs.extend([
                keys.clone(),
                values.clone(),
                keys.slice_axis(2, from, to)?,
                values.slice_axis(2, from, to)?,
            ]);
        }
        const COMPILED_LATTICE_BIT: u64 = 1u64 << 10;
        const COMPILED_EMBEDDED_BIT: u64 = 1u64 << 12;
        let is_spanned = spanned_heads.is_some();
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
            let logits = if let Some(spanned) = spanned_heads {
                spanned.forward_logits(&draft_hidden)?
            } else {
                // Match `propose_compiled` / `propose_tree_lattice_compiled`: run the
                // full-vocab head on the whole block (anchor + proposals, M=8) so
                // Metal keeps its K-quant M = 8 NAX route and eager/compiled stay
                // bit-identical.
                target_lm_head.forward(&hidden)?.slice_axis(1, 1, rows)?
            };
            let (candidates, scores) = self.selector.select_lattice_arrays(
                &draft_hidden,
                &logits,
                &args[2],
                is_spanned,
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
}
