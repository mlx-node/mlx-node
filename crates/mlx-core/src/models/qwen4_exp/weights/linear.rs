//! Linear projection dispatchers for `Weight` and `Store` (CUDA Q8_0/BF16,
//! Metal dense/mixer_act/hyper_up, and generic fallback).

use super::{MAX_READ_BYTES, Store, Weight, err};
use crate::array::{DType, MxArray};
use crate::models::quantized_linear::QuantizedLinear;
use napi::Result;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CudaQ8PrefillMode {
    /// Per-row INT8 activation quant (`gs_shift = 0`, peak ~113 TOPS).
    Row,
    /// Grouped INT8 activation quant (`gs_shift = -1` -> auto GS=256 / GS=128).
    Grouped,
    /// Bit-exact BF16 MMA (`gs_shift = -2`, zero X quantization).
    Bf16,
    /// Legacy cuBLAS dense BF16 cache fallback.
    Cublas,
}

fn cuda_q8_prefill_mode() -> CudaQ8PrefillMode {
    static MODE: std::sync::OnceLock<CudaQ8PrefillMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        match std::env::var("MLX_QWEN4_Q8_PREFILL_MODE")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("row") => CudaQ8PrefillMode::Row,
            Some("grp" | "group" | "grouped") => CudaQ8PrefillMode::Grouped,
            Some("bf16") => CudaQ8PrefillMode::Bf16,
            Some("cublas" | "off" | "0") => CudaQ8PrefillMode::Cublas,
            _ => CudaQ8PrefillMode::Grouped,
        }
    })
}

impl Weight {
    pub(in crate::models::qwen4_exp) fn with_cached_qlinear(mut self) -> Self {
        if crate::accelerator::host_has_cuda()
            && let Some(scales) = &self.scales
        {
            self.qlinear = Some(Arc::new(QuantizedLinear::new(
                self.values.clone(),
                scales.clone(),
                self.biases.clone(),
                None,
                self.group,
                self.bits,
                self.mode.clone(),
            )));
        } else {
            self.qlinear = None;
        }
        self
    }

    pub(in crate::models::qwen4_exp) fn mixer_act(&self, x: &MxArray) -> Result<Option<MxArray>> {
        let decode = x.shape_at(1)? == 1;
        if self.mode != "affine"
            || self.bits != 8
            || self.group != 32
            || !crate::engine::persistence::compiled_forward_backend_available()
        {
            return Ok(None);
        }
        let (Some(scales), Some(biases)) = (&self.scales, &self.biases) else {
            return Ok(None);
        };
        if decode {
            crate::kernels::qwen4::metal::decode_mixer_act(x, &self.values, scales, biases)
        } else {
            crate::kernels::qwen4::metal::prefill_mixer_act(x, &self.values, scales, biases)
        }
    }

    pub(in crate::models::qwen4_exp) fn hyper_up(
        &self,
        x: &MxArray,
        normed: &MxArray,
    ) -> Result<Option<MxArray>> {
        if self.mode != "affine" || self.bits != 8 || self.group != 32 {
            return Ok(None);
        }
        let (Some(scales), Some(biases)) = (&self.scales, &self.biases) else {
            return Ok(None);
        };
        crate::kernels::qwen4::metal::hyper_up(x, &self.values, scales, biases, normed)
    }

    pub fn linear(&self, x: &MxArray) -> Result<MxArray> {
        let on_cuda = crate::accelerator::cuda_ops_available();
        self.linear_inner(x, on_cuda)
    }

    fn linear_inner(&self, x: &MxArray, on_cuda: bool) -> Result<MxArray> {
        // --- CUDA Q8_0 prefill dispatcher ---
        if on_cuda
            && super::super::cuda_qwen4_fast_enabled()
            && self.mode == "affine"
            && self.bits == 8
            && self.group == 32
            && let (Some(scales), Some(biases)) = (&self.scales, &self.biases)
            && x.dtype()? == DType::BFloat16
            && scales.dtype()? == DType::Float16
            && biases.dtype()? == DType::Float16
        {
            let mode = cuda_q8_prefill_mode();
            if mode != CudaQ8PrefillMode::Cublas {
                let k = x.shape_at(x.ndim()?.saturating_sub(1))?;
                let rows = if k > 0 { x.size()? as i64 / k } else { 0 };
                if rows >= 17 {
                    // Below the GB10 roofline ridge (M < 128), Q8_0 GEMM is
                    // memory-bandwidth bound at 1 B/weight: exact BF16 MMA
                    // avoids the `quantize_x` launch AND keeps bit-exact BF16
                    // accuracy while still allocating zero BF16 weight cache.
                    let gs_shift = if rows < 128 {
                        -2
                    } else {
                        match mode {
                            CudaQ8PrefillMode::Row => 0,
                            CudaQ8PrefillMode::Grouped => -1,
                            CudaQ8PrefillMode::Bf16 => -2,
                            CudaQ8PrefillMode::Cublas => unreachable!(),
                        }
                    };
                    if let Some(out) = crate::kernels::linear::q8_prefill_cuda(
                        x,
                        &self.values,
                        scales,
                        Some(biases),
                        gs_shift,
                        !self.q8_symmetric_128,
                    )? {
                        return Ok(out);
                    }
                }
            }
        }
        // --- Metal dense Q8 decode / prefill dispatchers ---
        if !on_cuda
            && self.mode == "affine"
            && self.bits == 8
            && self.group == 32
            && let (Some(scales), Some(biases)) = (&self.scales, &self.biases)
            && let Some(compact) =
                crate::kernels::qwen4::metal::dense_decode(x, &self.values, scales, biases)?
        {
            return Ok(compact);
        }
        let compact = if !on_cuda
            && self.mode == "affine"
            && self.bits == 8
            && self.group == 32
            && let (Some(scales), Some(biases)) = (&self.scales, &self.biases)
        {
            crate::kernels::qwen4::metal::dense_prefill(x, &self.values, scales, biases)?
        } else {
            None
        };
        // --- Common quantized / dense fallback ---
        let y = if let Some(compact) = compact {
            compact
        } else if let Some(qlinear) = &self.qlinear {
            qlinear.forward(x)?
        } else if let Some(scales) = &self.scales {
            QuantizedLinear::new(
                self.values.clone(),
                scales.clone(),
                self.biases.clone(),
                None,
                self.group,
                self.bits,
                self.mode.clone(),
            )
            .forward(x)?
        } else {
            let values = if x.dtype()? == DType::BFloat16
                && let Some(dense) = &self.dense_bf16
            {
                dense
            } else {
                &self.values
            };
            if values.ndim()? == 2
                && x.dtype()? == DType::BFloat16
                && values.dtype()? == DType::BFloat16
                && let Some(out) =
                    crate::kernels::linear::bf16_matmul_few(x, values, values.shape_at(0)?)?
            {
                return Ok(out);
            }
            let weight = values.transpose(None)?.astype(x.dtype()?)?;
            let shape = x.shape()?;
            let (&width, outer_shape) = shape
                .split_last()
                .filter(|(width, _)| **width > 0)
                .ok_or_else(|| err("Qwen4 projection input requires a positive final dimension"))?;
            if crate::accelerator::host_has_metal()
                && x.dtype()? == DType::Float32
                && x.size()? / width as u64 > 1
            {
                // MLX enables TF32 GEMMs by default on NAX hardware. Preserve
                // the F32 source's GEMV arithmetic without changing global
                // process settings or lowering the fixture's accuracy gate.
                let rows = x.reshape(&[-1, width])?;
                let mut outputs = Vec::new();
                for row in 0..rows.shape()?[0] {
                    outputs.push(rows.slice_axis(0, row, row + 1)?.matmul(&weight)?);
                }
                let mut shape = outer_shape.to_vec();
                shape.push(self.values.shape_at(0)?);
                MxArray::concatenate_many(outputs.iter().collect(), Some(0))?.reshape(&shape)?
            } else {
                x.matmul(&weight)?
            }
        };
        Ok(y)
    }
}

impl Store {
    pub fn linear(&mut self, name: &str, x: &MxArray) -> Result<MxArray> {
        self.linear_impl(name, x, false)
    }

    /// Batch independent vectors while retaining the singleton projection's
    /// accumulation: an ordinary prompt GEMM can round differently and change
    /// the subsequent greedy token stream. An admitted Q8 bank runs four
    /// vectors per call of the row-exact decode kernel, every row with the
    /// bits of its single-row projection.
    pub fn linear_ple_window(&mut self, name: &str, x: &MxArray) -> Result<MxArray> {
        self.linear_vector_window(name, x)
    }

    pub fn linear_vector_window(&mut self, name: &str, x: &MxArray) -> Result<MxArray> {
        self.linear_impl(name, x, true)
    }

    fn linear_impl(&mut self, name: &str, x: &MxArray, vectors: bool) -> Result<MxArray> {
        let d = self.descriptor(name)?;
        let rows = d.rows()?;
        let shape = x.shape()?.to_vec();
        let (&width, outer_shape) = shape
            .split_last()
            .ok_or_else(|| err("Qwen4 projection input has no final dimension"))?;
        if width != d.width()? as i64 {
            return Err(err(format!("Projection shape mismatch: {name}")));
        }
        let count = x.size()? as i64 / width;
        #[cfg(test)]
        let dense_row_calls = std::cell::Cell::new(0u64);
        let project = |w: &Weight| -> Result<MxArray> {
            if !vectors || count == 1 {
                return w.linear(x);
            }
            let mut out_shape = outer_shape.to_vec();
            out_shape.push(w.values.shape_at(0)?);
            let window = x.reshape(&[1, count, width])?;
            let step = super::super::math::rows::MAX_ROWS;
            let mut parts = Vec::with_capacity(((count + step - 1) / step) as usize);
            for start in (0..count).step_by(step as usize) {
                let rows = window.slice_axis(1, start, (start + step).min(count))?;
                match super::super::math::rows::dense_decode(w, &rows)? {
                    Some(out) => parts.push(out),
                    None if parts.is_empty() => break,
                    None => return Err(err("Qwen4 row kernel refused part of a vector window")),
                }
                #[cfg(test)]
                dense_row_calls.set(dense_row_calls.get() + 1);
            }
            if !parts.is_empty() {
                return MxArray::concatenate_many(parts.iter().collect(), Some(1))?
                    .reshape(&out_shape);
            }
            if crate::accelerator::cuda_ops_available()
                && x.dtype()? == DType::BFloat16
                && ((w.mode == "affine"
                    && w.bits == 8
                    && w.group == 32
                    && w.scales.is_some()
                    && w.biases.is_some()
                    && width >= 32
                    && width % 32 == 0)
                    || w.scales.is_none())
            {
                // Chunk in steps of 16 rows so CUDA stays on `mlx_affine_q8_few_cuda` /
                // `mlx_bf16_gemm_few_cuda` (16x fewer weight reads than M=1 while
                // matching singleton row rounding bit-for-bit).
                const CUDA_FEW_STEP: i64 = 16;
                let mut cuda_parts =
                    Vec::with_capacity(((count + CUDA_FEW_STEP - 1) / CUDA_FEW_STEP) as usize);
                for start in (0..count).step_by(CUDA_FEW_STEP as usize) {
                    let rows = window.slice_axis(1, start, (start + CUDA_FEW_STEP).min(count))?;
                    cuda_parts.push(w.linear(&rows)?);
                }
                return MxArray::concatenate_many(cuda_parts.iter().collect(), Some(1))?
                    .reshape(&out_shape);
            }
            let input = x.reshape(&[count, 1, width])?;
            let output = if w.scales.is_some() && !crate::accelerator::cuda_ops_available() {
                // One shared bank, with every RHS id zero. Gather dispatches
                // M=1 GEMV for each vector, not a wide prompt GEMM.
                w.gather_rows(
                    &input,
                    &MxArray::zeros(&[count], Some(DType::Uint32))?,
                    1,
                    false,
                )?
            } else {
                let parts = (0..count)
                    .map(|i| w.linear(&input.slice_axis(0, i, i + 1)?))
                    .collect::<Result<Vec<_>>>()?;
                MxArray::concatenate_many(parts.iter().collect(), Some(0))?
            };
            output.reshape(&out_shape)
        };
        // Staging chunks bound SSD reads. A resident matrix can use a single
        // projection, particularly the large vocabulary head, without copying
        // weights or launching one kernel for every former read chunk.
        let output = if let Some(bank) = self.banks.get(name) {
            project(bank)
        } else {
            let chunk = (MAX_READ_BYTES as usize / (d.width()? * 4)).clamp(1, 4096);
            let mut outputs = Vec::new();
            for start in (0..rows).step_by(chunk) {
                let weight = self.read(name, start, (rows - start).min(chunk))?;
                outputs.push(project(&weight)?);
            }
            if outputs.len() == 1 {
                Ok(outputs.remove(0))
            } else {
                MxArray::concatenate_many(outputs.iter().collect(), Some(-1))
            }
        };
        #[cfg(test)]
        if vectors {
            self.vector_windows.push((count, dense_row_calls.get()));
        }
        output
    }

    /// `x` is `[1, R, K]`, `R <= 4`. Row `j` of the result carries the bits of
    /// `linear` applied to row `j` alone: an admitted Q8 bank streams once
    /// through the row-batched kernel, every other weight runs its single-row
    /// projection per row.
    pub(in crate::models::qwen4_exp) fn linear_rows(
        &mut self,
        name: &str,
        x: &MxArray,
    ) -> Result<MxArray> {
        let count = x.shape_at(1)?;
        if count == 1 {
            return self.linear(name, x);
        }
        let d = self.descriptor(name)?;
        let rows = d.rows()?;
        let width = d.width()?;
        if x.shape()?.last().copied() != Some(width as i64) {
            return Err(err(format!("Projection shape mismatch: {name}")));
        }
        let chunk = (MAX_READ_BYTES as usize / (width * 4)).clamp(1, 4096);
        #[cfg(test)]
        let single_rows = std::cell::Cell::new(0u64);
        let project = |w: &Weight| -> Result<MxArray> {
            if let Some(out) = super::super::math::rows::dense_decode(w, x)? {
                return Ok(out);
            }
            if crate::accelerator::cuda_ops_available()
                && (2..=16).contains(&count)
                && (w.scales.is_some() || x.dtype()? == DType::BFloat16)
            {
                return w.linear(x);
            }
            #[cfg(test)]
            single_rows.set(single_rows.get() + count as u64);
            let parts = (0..count)
                .map(|j| w.linear(&x.slice_axis(1, j, j + 1)?))
                .collect::<Result<Vec<_>>>()?;
            MxArray::concatenate_many(parts.iter().collect(), Some(1))
        };
        let output = if let Some(bank) = self.banks.get(name) {
            project(bank)?
        } else {
            let mut outputs = Vec::new();
            for start in (0..rows).step_by(chunk) {
                let weight = self.read(name, start, (rows - start).min(chunk))?;
                outputs.push(project(&weight)?);
            }
            if outputs.len() == 1 {
                outputs.remove(0)
            } else {
                MxArray::concatenate_many(outputs.iter().collect(), Some(-1))?
            }
        };
        #[cfg(test)]
        {
            self.single_row_projections += single_rows.get();
        }
        Ok(output)
    }

    /// `linear_pair` for a `[1, R, K]` window: the paired bank prepared by
    /// single-row decoding streams once, otherwise each projection runs
    /// through `linear_rows`.
    pub(in crate::models::qwen4_exp) fn linear_pair_rows(
        &mut self,
        x: &MxArray,
        first: &str,
        second: &str,
    ) -> Result<(MxArray, MxArray)> {
        let count = x.shape_at(1)?;
        if count == 1 {
            return self.linear_pair(x, first, second);
        }
        let key = (first.to_owned(), second.to_owned());
        if let Some(bank) = self.paired_banks.get(&key) {
            let output = if let Some(output) = super::super::math::rows::dense_decode(bank, x)? {
                Some(output)
            } else if crate::accelerator::cuda_ops_available()
                && (2..=16).contains(&count)
                && (bank.scales.is_some() || x.dtype()? == DType::BFloat16)
            {
                Some(bank.linear(x)?)
            } else {
                None
            };
            if let Some(output) = output {
                let n = self.descriptor(first)?.rows()? as i64;
                let axis = output.ndim()? as usize - 1;
                return Ok((
                    output.slice_axis(axis, 0, n)?,
                    output.slice_axis(axis, n, output.shape()?[axis])?,
                ));
            }
        } else if crate::accelerator::cuda_ops_available()
            && (2..=16).contains(&count)
            && (x.dtype()? == DType::BFloat16
                || self.banks.get(first).is_some_and(|w| w.scales.is_some()))
        {
            return self.linear_pair(x, first, second);
        }
        Ok((self.linear_rows(first, x)?, self.linear_rows(second, x)?))
    }
}
