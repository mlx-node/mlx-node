use mlx_sys as sys;
use napi::bindgen_prelude::*;
use rand::{Rng, RngExt};

use super::spanned_head::remap_spanned_candidate_ids;
use crate::array::{DType, MxArray};
use crate::models::quantized_linear::LinearProj;
use crate::nn::Embedding;
use crate::sampling::{SparseDistribution, is_greedy_temperature};

pub(super) struct CandidateSelector {
    pub(super) predecessor_codebook: Embedding,
    pub(super) successor_codebook: Embedding,
    pub(super) hidden_projection: LinearProj,
    pub(super) top_k: usize,
    pub(super) rank: usize,
    pub(super) vocab_size: usize,
}

pub(super) fn normalized_selector_probs(scores: &[f32], temperature: f64) -> Result<Vec<f64>> {
    let scale = temperature.max(f64::MIN_POSITIVE);
    let max = scores
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .max_by(f32::total_cmp)
        .ok_or_else(|| Error::from_reason("DFlash2 selector scores are all non-finite"))?;
    let mut probs = scores
        .iter()
        .map(|&score| ((f64::from(score) - f64::from(max)) / scale).exp())
        .collect::<Vec<_>>();
    let total = probs.iter().sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err(Error::from_reason(
            "DFlash2 selector has no positive probability mass",
        ));
    }
    for prob in &mut probs {
        *prob /= total;
    }
    Ok(probs)
}

fn sample_selector_index<R: Rng + ?Sized>(probs: &[f64], rng: &mut R) -> usize {
    let draw = rng.random::<f64>();
    let mut cumulative = 0.0;
    let mut last = 0usize;
    for (index, &prob) in probs.iter().enumerate() {
        if prob <= 0.0 {
            continue;
        }
        cumulative += prob;
        last = index;
        if draw < cumulative {
            return index;
        }
    }
    last
}

/// The selector's drafted token path.
pub(crate) enum SelectorPath {
    /// Host-walked ids — sampled turns need the score rows on CPU anyway, so
    /// the conditional predecessor walk pays the same readback either way.
    Host(Vec<i32>),
    /// Device-resident greedy path `[L]` i32: the conditional predecessor
    /// walk ran inside the lazy graph (gather + argmax per position), so no
    /// per-cycle GPU→CPU readback gates the verify block on the proposal.
    Device(MxArray),
    /// Device-drawn sampled path `[L]` i32 with the rows it drew from.
    Sampled {
        ids: MxArray,
        rows: crate::engine::backend::DeviceSparseRows,
    },
}

/// The sampled predecessor walk on the device: candidates `[1,L,K]` i32,
/// scores `[L,K,K]` f32, one uniform per position drawn from `rng` on the
/// host. `None` (no Metal, contract miss) keeps the host walk.
pub(crate) fn device_sampled_path<R: Rng + ?Sized>(
    candidates: &MxArray,
    scores: &MxArray,
    vocab_size: usize,
    temperature: f64,
    rng: &mut R,
) -> Option<SelectorPath> {
    if !unsafe { sys::mlx_metal_is_available() } {
        return None;
    }
    let length = candidates.shape_at(1).ok()?;
    let width = candidates.shape_at(2).ok()?;
    let draws: Vec<f32> = (0..length).map(|_| rng.random::<f32>()).collect();
    let uniforms = MxArray::from_float32(&draws, &[length]).ok()?;
    let temperature = MxArray::from_float32(&[temperature as f32], &[1]).ok()?;
    let mut path = std::ptr::null_mut();
    let mut q = std::ptr::null_mut();
    if !unsafe {
        sys::mlx_dflash2_sampled_path(
            candidates.as_raw_ptr(),
            scores.as_raw_ptr(),
            uniforms.as_raw_ptr(),
            temperature.as_raw_ptr(),
            &mut path,
            &mut q,
        )
    } {
        return None;
    }
    let ids = MxArray::from_handle(path, "dflash2_sampled_path:path").ok()?;
    let probs = MxArray::from_handle(q, "dflash2_sampled_path:q").ok()?;
    let support = candidates.reshape(&[length, width]).ok()?;
    Some(SelectorPath::Sampled {
        ids,
        rows: crate::engine::backend::DeviceSparseRows {
            ids: support,
            probs,
            vocab_size,
        },
    })
}

impl CandidateSelector {
    /// Fused top-16 over the vocab via the Metal or CUDA primitive. Returns
    /// (ids [1,L,16] i32, values [1,L,16] f32) ascending by value — the same
    /// contract the argpartition slice produces. `None` falls back to the
    /// generic path for shape, dtype, or backend capability misses.
    fn fused_topk16(&self, logits: &MxArray) -> Option<(MxArray, MxArray)> {
        if self.top_k != 16 {
            return None;
        }
        let shape = logits.shape().ok()?;
        if shape.len() != 3 || shape[0] != 1 {
            return None;
        }
        let (ids, values) = crate::kernels::dflash2::topk16(logits)?;
        // The kernel emits f32 values (widened for the merge compare), while
        // the take_along_axis fallback returns logits.dtype(). bf16→f32 is
        // lossless, so casting back restores bit-parity: `scores`' add then
        // runs in the model dtype instead of promoting to f32 before the
        // final astype — a different rounding that flips near-tie picks.
        let values = values.astype(logits.dtype().ok()?).ok()?;
        let dims = [1, shape[1], self.top_k as i64];
        Some((ids.reshape(&dims).ok()?, values.reshape(&dims).ok()?))
    }

    /// One-dispatch version of the device-resident greedy predecessor walk.
    /// The Metal/CUDA kernel scans each selected score row in reverse with strict
    /// `>`, exactly matching the lazy fallback's reversed MLX argmax: the last
    /// non-NaN maximum wins, NaNs do not win, and an all-NaN/-inf row selects
    /// the final slot. `None` keeps the existing lazy graph as the fallback.
    fn fused_greedy_path(&self, candidates: &MxArray, scores: &MxArray) -> Option<MxArray> {
        crate::kernels::dflash2::greedy_path(candidates, scores)
    }

    /// Same greedy predecessor decisions, with temperature-one confidence for
    /// the GPU verification-row policy. Confidence never changes selected ids.
    pub(super) fn fused_greedy_path_confidence(
        &self,
        candidates: &MxArray,
        scores: &MxArray,
    ) -> Option<(MxArray, MxArray)> {
        crate::kernels::dflash2::greedy_path_confidence(candidates, scores)
    }

    /// Original lazy device walk, retained both as the contract fallback and
    /// as an independent parity oracle for the fused kernel.
    pub(super) fn lazy_greedy_path(
        &self,
        candidates: &MxArray,
        scores: &MxArray,
        length: usize,
    ) -> Result<MxArray> {
        let mut predecessor = MxArray::from_int32(&[0], &[1])?;
        // Preserve the established device path's last-maximum rule by
        // reversing before argmax (which otherwise returns the first max).
        // This matches the host max_by walk for ordinary finite ties, but
        // host total_cmp distinguishes signed zero and ranks NaNs. MLX
        // argmax treats signed zeros as equal and, since upstream a124ac096,
        // returns the first NaN (here the last original NaN slot); the fused
        // kernel preserves these device-path semantics.
        let descending = MxArray::from_int32(
            &(0..self.top_k as i32).rev().collect::<Vec<_>>(),
            &[self.top_k as i64],
        )?;
        let last_index = MxArray::from_int32(&[self.top_k as i32 - 1], &[1])?;
        let mut tokens = Vec::with_capacity(length);
        for position in 0..length {
            let scores_i = scores.slice_axis(0, position as i64, position as i64 + 1)?;
            let row = scores_i.take(&predecessor, 1)?; // [1, 1, K]
            let selected = last_index
                .sub(
                    &row.take(&descending, -1)?
                        .argmax(-1, Some(false))?
                        .reshape(&[1])?,
                )?
                .astype(DType::Int32)?; // [1]
            let cand_i = candidates.slice_axis(1, position as i64, position as i64 + 1)?;
            tokens.push(cand_i.take(&selected, 2)?.reshape(&[1])?); // [1]
            predecessor = selected;
        }
        MxArray::concatenate_many(tokens.iter().collect(), Some(0))
    }

    #[cfg(test)]
    pub(super) fn select<R: Rng + ?Sized>(
        &self,
        hidden: &MxArray,
        logits: &MxArray,
        anchor: u32,
        temperature: f64,
        device_path: bool,
        confidence: Option<&mut Option<MxArray>>,
        rng: &mut R,
    ) -> Result<(SelectorPath, Vec<SparseDistribution>)> {
        self.select_spanned(
            hidden,
            logits,
            anchor,
            temperature,
            device_path,
            false,
            confidence,
            rng,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn select_spanned<R: Rng + ?Sized>(
        &self,
        hidden: &MxArray,
        logits: &MxArray,
        anchor: u32,
        temperature: f64,
        device_path: bool,
        spanned: bool,
        confidence: Option<&mut Option<MxArray>>,
        rng: &mut R,
    ) -> Result<(SelectorPath, Vec<SparseDistribution>)> {
        let anchor_ids = MxArray::from_int32(&[anchor as i32], &[1])?;
        self.select_with_anchor(
            hidden,
            logits,
            &anchor_ids,
            temperature,
            device_path,
            spanned,
            confidence,
            rng,
        )
    }

    /// Compute top-K candidate vocabulary IDs (`[1, L, K]` Int32) and
    /// predecessor-conditioned transition scores (`[L, K, K]` Float32).
    pub(super) fn select_lattice_arrays(
        &self,
        hidden: &MxArray,
        logits: &MxArray,
        anchor_ids: &MxArray,
        spanned: bool,
    ) -> Result<(MxArray, MxArray)> {
        let length = hidden.shape_at(1)? as usize;
        let (candidates, unary) = match self.fused_topk16(logits) {
            Some((ids, values)) => (ids, values),
            None => {
                let vocab = logits.shape_at(2)?;
                let candidates = logits
                    .argpartition(-(self.top_k as i32), Some(-1))?
                    .slice_axis(2, vocab - self.top_k as i64, vocab)?;
                let unary = logits.take_along_axis(&candidates, -1)?;
                (candidates, unary)
            }
        };
        let projected = self.hidden_projection.forward(hidden)?.reshape(&[
            length as i64,
            1,
            self.rank as i64,
        ])?;
        let candidates = candidates.astype(DType::Int32)?;
        let candidates = if spanned {
            remap_spanned_candidate_ids(&candidates)?
        } else {
            candidates
        };
        let scores = self.edge_scores(&candidates, &unary, &projected, anchor_ids, length)?;
        Ok((candidates, scores))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn select_with_anchor<R: Rng + ?Sized>(
        &self,
        hidden: &MxArray,
        logits: &MxArray,
        anchor_ids: &MxArray,
        temperature: f64,
        device_path: bool,
        spanned: bool,
        confidence: Option<&mut Option<MxArray>>,
        rng: &mut R,
    ) -> Result<(SelectorPath, Vec<SparseDistribution>)> {
        let length = hidden.shape_at(1)? as usize;
        let (candidates, scores) =
            self.select_lattice_arrays(hidden, logits, anchor_ids, spanned)?;

        let greedy = is_greedy_temperature(temperature);
        if greedy && device_path {
            // One custom dispatch replaces the per-position gather + reverse
            // + argmax + gather chain. Contract misses retain that lazy graph,
            // and either result stays device-resident into target verify.
            if let Some(confidence) = confidence
                && let Some((path, probabilities)) =
                    self.fused_greedy_path_confidence(&candidates, &scores)
            {
                *confidence = Some(probabilities);
                return Ok((SelectorPath::Device(path), Vec::new()));
            }
            let path = match self.fused_greedy_path(&candidates, &scores) {
                Some(path) => path,
                None => self.lazy_greedy_path(&candidates, &scores, length)?,
            };
            return Ok((SelectorPath::Device(path), Vec::new()));
        }
        if !greedy
            && device_path
            && let Some(path) =
                device_sampled_path(&candidates, &scores, self.vocab_size, temperature, rng)
        {
            return Ok((path, Vec::new()));
        }

        MxArray::eval_arrays(&[&candidates, &scores])?;
        let candidate_ids = candidates.to_int32()?;
        let edge_scores = scores.to_float32()?;

        let mut path = Vec::with_capacity(length);
        let mut rows = Vec::with_capacity(if greedy { 0 } else { length });
        let mut predecessor_index = 0usize;
        for position in 0..length {
            let start = (position * self.top_k + predecessor_index) * self.top_k;
            let row = &edge_scores[start..start + self.top_k];
            let selected = if greedy {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(index, _)| index)
                    .unwrap_or(0)
            } else {
                let probs = normalized_selector_probs(row, temperature)?;
                let support_start = position * self.top_k;
                rows.push(SparseDistribution::from_parts(
                    candidate_ids[support_start..support_start + self.top_k].to_vec(),
                    probs.clone(),
                    self.vocab_size,
                )?);
                sample_selector_index(&probs, rng)
            };
            let token = candidate_ids[position * self.top_k + selected];
            path.push(token);
            predecessor_index = selected;
        }
        Ok((SelectorPath::Host(path), rows))
    }

    pub(super) fn edge_scores(
        &self,
        candidates: &MxArray,
        unary: &MxArray,
        projected: &MxArray,
        anchor_ids: &MxArray,
        length: usize,
    ) -> Result<MxArray> {
        let use_cuda = crate::accelerator::cuda_ops_available();
        let use_metal = !use_cuda
            && std::env::var("MLX_DFLASH2_SELECTOR_EDGES_FP32").is_ok_and(|v| v == "1")
            && unsafe { sys::mlx_metal_is_available() && sys::mlx_default_device() == 1 };
        if self.top_k == 16 && self.rank == 256 && (use_cuda || use_metal) {
            let predecessor = self.predecessor_codebook.get_weight();
            let successor = self.successor_codebook.get_weight();
            let unary_arg;
            let unary_ref = if use_cuda {
                unary
            } else {
                unary_arg = unary.astype(DType::Float32)?;
                &unary_arg
            };
            let scores = unsafe {
                sys::mlx_dflash2_selector_edges(
                    candidates.as_raw_ptr(),
                    unary_ref.as_raw_ptr(),
                    projected.as_raw_ptr(),
                    predecessor.as_raw_ptr(),
                    successor.as_raw_ptr(),
                    anchor_ids.as_raw_ptr(),
                )
            };
            if !scores.is_null() {
                return MxArray::from_handle(scores, "dflash2_selector_edges");
            }
        }
        let candidate_flat = candidates.reshape(&[(length * self.top_k) as i64])?;
        let successors = self
            .successor_codebook
            .forward(&candidate_flat)?
            .reshape(&[length as i64, self.top_k as i64, self.rank as i64])?;
        let anchor_embedding = self
            .predecessor_codebook
            .forward(anchor_ids)?
            .reshape(&[1, 1, self.rank as i64])?
            .broadcast_to(&[1, self.top_k as i64, self.rank as i64])?;
        let predecessor_ids = if length > 1 {
            candidates.slice_axis(1, 0, length as i64 - 1)?
        } else {
            MxArray::from_int32(&[], &[0, self.top_k as i64])?
        };
        let predecessors = if length > 1 {
            let previous = self
                .predecessor_codebook
                .forward(&predecessor_ids.reshape(&[((length - 1) * self.top_k) as i64])?)?
                .reshape(&[length as i64 - 1, self.top_k as i64, self.rank as i64])?;
            MxArray::concatenate(&anchor_embedding, &previous, 0)?
        } else {
            anchor_embedding
        };
        let edges = predecessors
            .mul(projected)?
            .matmul(&successors.transpose(Some(&[0, 2, 1]))?)?;
        let scores = edges.add(&unary.reshape(&[length as i64, 1, self.top_k as i64])?)?;
        scores.astype(DType::Float32)
    }
}
