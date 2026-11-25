//! CUDA dense & quantized linear kernel wrappers.

use std::ffi::CString;

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use super::intern_kquant_mode;
use crate::array::{DType, MxArray};

#[inline]
pub(super) fn bf16_matmul_few(
    input: &MxArray,
    weight: &MxArray,
    out_features: i64,
) -> Result<Option<MxArray>> {
    if input.dtype()? != DType::BFloat16 || weight.dtype()? != DType::BFloat16 {
        return Ok(None);
    }
    let ndim = input.ndim()?;
    if ndim < 2 {
        return Ok(None);
    }
    let k = input.shape_at(ndim - 1)?;
    let rows = if k > 0 { input.size()? as i64 / k } else { 0 };
    if !(1..=16).contains(&rows) || k < 1024 || k % 8 != 0 || out_features < 1 {
        return Ok(None);
    }
    let handle = unsafe { sys::mlx_bf16_gemm_few_cuda(input.as_raw_ptr(), weight.as_raw_ptr()) };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(handle, "bf16_gemm_few_cuda")?))
    }
}

#[inline]
pub(super) fn affine_q8_matmul_few(
    x: &MxArray,
    weight: &MxArray,
    scales: &MxArray,
    biases: Option<&MxArray>,
) -> Result<Option<MxArray>> {
    let biases_ptr = biases.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let handle = unsafe {
        sys::mlx_affine_q8_few_cuda(
            x.as_raw_ptr(),
            weight.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases_ptr,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(handle, "affine_q8_few_cuda")?))
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn affine_q8_matmul_few_pair(
    x0: &MxArray,
    w0: &MxArray,
    s0: &MxArray,
    b0: Option<&MxArray>,
    x1: &MxArray,
    w1: &MxArray,
    s1: &MxArray,
    b1: Option<&MxArray>,
    group_size: i32,
) -> Result<Option<(MxArray, MxArray)>> {
    if group_size != 32 || b0.is_some() != b1.is_some() {
        return Ok(None);
    }
    let b0_ptr = b0.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let b1_ptr = b1.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let mut out0 = std::ptr::null_mut();
    let mut out1 = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_affine_q8_few_pair_cuda(
            x0.as_raw_ptr(),
            w0.as_raw_ptr(),
            s0.as_raw_ptr(),
            b0_ptr,
            x1.as_raw_ptr(),
            w1.as_raw_ptr(),
            s1.as_raw_ptr(),
            b1_ptr,
            group_size,
            &mut out0,
            &mut out1,
        )
    };
    if !ok || out0.is_null() || out1.is_null() {
        unsafe {
            if !out0.is_null() {
                sys::mlx_array_delete(out0);
            }
            if !out1.is_null() {
                sys::mlx_array_delete(out1);
            }
        }
        return Ok(None);
    }
    let arr0 = match MxArray::from_handle(out0, "affine_q8_few_pair_cuda:0") {
        Ok(arr) => arr,
        Err(err) => {
            unsafe { sys::mlx_array_delete(out1) };
            return Err(err);
        }
    };
    let arr1 = MxArray::from_handle(out1, "affine_q8_few_pair_cuda:1")?;
    Ok(Some((arr0, arr1)))
}

#[inline]
pub(super) fn kquant_matmul_few(
    x: &MxArray,
    weight: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
    mode: &str,
) -> Result<Option<MxArray>> {
    let handle = if let Some(mode_c) = intern_kquant_mode(mode) {
        unsafe {
            sys::mlx_kquant_few(
                x.as_raw_ptr(),
                weight.as_raw_ptr(),
                scales.as_raw_ptr(),
                biases.as_raw_ptr(),
                mode_c.as_ptr(),
            )
        }
    } else {
        let mode_c = CString::new(mode)
            .map_err(|e| Error::from_reason(format!("Invalid mode string: {e}")))?;
        unsafe {
            sys::mlx_kquant_few(
                x.as_raw_ptr(),
                weight.as_raw_ptr(),
                scales.as_raw_ptr(),
                biases.as_raw_ptr(),
                mode_c.as_ptr(),
            )
        }
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(handle, "kquant_few")?))
    }
}

#[inline]
pub(super) fn kquant_matmul_prefill(
    x: &MxArray,
    weight: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
    mode: &str,
) -> Result<Option<MxArray>> {
    let handle = if let Some(mode_c) = intern_kquant_mode(mode) {
        unsafe {
            sys::mlx_kquant_prefill(
                x.as_raw_ptr(),
                weight.as_raw_ptr(),
                scales.as_raw_ptr(),
                biases.as_raw_ptr(),
                mode_c.as_ptr(),
            )
        }
    } else {
        let mode_c = CString::new(mode)
            .map_err(|e| Error::from_reason(format!("Invalid mode string: {e}")))?;
        unsafe {
            sys::mlx_kquant_prefill(
                x.as_raw_ptr(),
                weight.as_raw_ptr(),
                scales.as_raw_ptr(),
                biases.as_raw_ptr(),
                mode_c.as_ptr(),
            )
        }
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(handle, "kquant_prefill")?))
    }
}

#[inline]
pub(super) fn kquant_matmul_multi(
    x: &MxArray,
    rows: i64,
    projections: &[(&MxArray, &MxArray, &MxArray, &str)],
) -> Result<Option<Vec<MxArray>>> {
    if !(2..=4).contains(&projections.len()) {
        return Ok(None);
    }
    let count = projections.len();
    let mut weights = [std::ptr::null_mut(); 4];
    let mut scales = [std::ptr::null_mut(); 4];
    let mut biases = [std::ptr::null_mut(); 4];
    let mut mode_ptrs = [std::ptr::null(); 4];
    let mut owned_modes: Vec<CString> = Vec::new();
    for (i, &(w, sc, bi, mode)) in projections.iter().enumerate() {
        weights[i] = w.as_raw_ptr();
        scales[i] = sc.as_raw_ptr();
        biases[i] = bi.as_raw_ptr();
        if let Some(interned) = intern_kquant_mode(mode) {
            mode_ptrs[i] = interned.as_ptr();
        } else {
            let cs = CString::new(mode)
                .map_err(|e| Error::from_reason(format!("Invalid mode string: {e}")))?;
            owned_modes.push(cs);
            mode_ptrs[i] = owned_modes[owned_modes.len() - 1].as_ptr();
        }
    }
    let mut out_handles = [std::ptr::null_mut(); 4];
    let ok = if (1..=16).contains(&rows) {
        unsafe {
            sys::mlx_kquant_few_multi(
                x.as_raw_ptr(),
                count as u32,
                weights.as_ptr(),
                scales.as_ptr(),
                biases.as_ptr(),
                mode_ptrs.as_ptr(),
                out_handles.as_mut_ptr(),
            )
        }
    } else if rows >= 17 {
        unsafe {
            sys::mlx_kquant_prefill_multi(
                x.as_raw_ptr(),
                count as u32,
                weights.as_ptr(),
                scales.as_ptr(),
                biases.as_ptr(),
                mode_ptrs.as_ptr(),
                out_handles.as_mut_ptr(),
            )
        }
    } else {
        false
    };
    // Keep any dynamically allocated CStrings alive across the FFI call.
    drop(owned_modes);
    if !ok {
        return Ok(None);
    }
    let mut result = Vec::with_capacity(count);
    for handle in out_handles.iter().take(count) {
        if handle.is_null() {
            return Ok(None);
        }
        result.push(MxArray::from_handle(*handle, "kquant_multi")?);
    }
    Ok(Some(result))
}

#[inline]
pub(super) fn affine_q4_g64_matmul_few(
    x: &MxArray,
    weight: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
) -> Result<Option<MxArray>> {
    let handle = unsafe {
        sys::mlx_affine_q4_g64_few_cuda(
            x.as_raw_ptr(),
            weight.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(handle, "affine_q4_g64_few")?))
    }
}

#[inline]
pub(super) fn dequant_q8_bf16_cuda(
    w: &MxArray,
    scales: &MxArray,
    biases: Option<&MxArray>,
) -> Result<Option<MxArray>> {
    let biases_ptr = biases.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let handle = unsafe {
        sys::mlx_qwen4_dequant_q8_bf16_cuda(w.as_raw_ptr(), scales.as_raw_ptr(), biases_ptr)
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(
            handle,
            "qwen4_dequant_q8_bf16_cuda",
        )?))
    }
}

#[inline]
pub(super) fn q8_prefill_cuda(
    x: &MxArray,
    w: &MxArray,
    scales: &MxArray,
    biases: Option<&MxArray>,
    gs_shift: i32,
    general_bias: bool,
) -> Result<Option<MxArray>> {
    let biases_ptr = biases.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
    let handle = unsafe {
        sys::mlx_qwen4_q8_prefill_cuda(
            x.as_raw_ptr(),
            w.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases_ptr,
            gs_shift,
            general_bias,
        )
    };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(MxArray::from_handle(handle, "qwen4_q8_prefill_cuda")?))
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        affine_bf16_matmul_few, affine_bf16_matmul_few_pair_cuda, dequant_q8_bf16_cuda,
        q8_prefill_cuda,
    };
    use crate::accelerator::{ActiveBackend, active_backend};
    use crate::array::{DType, MxArray};
    use half::bf16;

    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }
        fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * ((self.next_u32() & 0x00FF_FFFF) as f32 / 16_777_216.0)
        }
    }

    fn read_bf16_f32(arr: &MxArray) -> Vec<f32> {
        arr.eval();
        arr.to_uint16_native()
            .unwrap()
            .into_iter()
            .map(|bits| bf16::from_bits(bits).to_f32())
            .collect()
    }

    fn make_symmetric_q8_bank(
        rng: &mut Lcg,
        n: usize,
        k: usize,
        scale_dtype: DType,
    ) -> (MxArray, MxArray, MxArray) {
        let w_words: Vec<u32> = (0..n * (k / 4)).map(|_| rng.next_u32()).collect();
        let w = MxArray::from_uint32(&w_words, &[n as i64, (k / 4) as i64]).unwrap();
        let groups = k / 32;
        let sc_vals: Vec<f32> = (0..n * groups).map(|_| rng.uniform(0.002, 0.05)).collect();
        let sc = MxArray::from_float32(&sc_vals, &[n as i64, groups as i64])
            .unwrap()
            .astype(scale_dtype)
            .unwrap();
        let bi = sc.mul_scalar(-128.0).unwrap().astype(scale_dtype).unwrap();
        (w, sc, bi)
    }

    fn make_bf16_input(rng: &mut Lcg, m: usize, k: usize) -> MxArray {
        let vals: Vec<f32> = (0..m * k).map(|_| rng.uniform(-1.0, 1.0)).collect();
        MxArray::from_float32(&vals, &[1, m as i64, k as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap()
    }

    fn nrmse(got: &[f32], want: &[f32]) -> f64 {
        assert_eq!(got.len(), want.len());
        let mut se = 0.0f64;
        let mut sr = 0.0f64;
        for (&g, &w) in got.iter().zip(want.iter()) {
            assert!(g.is_finite(), "non-finite output: {g}");
            let d = (g - w) as f64;
            se += d * d;
            sr += (w as f64) * (w as f64);
        }
        (se / sr.max(1e-12)).sqrt()
    }

    #[test]
    fn cuda_symmetric_q8_affine_few_parity() {
        if active_backend() != ActiveBackend::CudaGpu {
            return;
        }
        let mut rng = Lcg::new(0x51A8_0001);

        // Inkling2-like shapes across M=1, 4, 8 (u4_1row) and M=16 (Tensor Core MMA)
        for &(k, n) in &[(2304usize, 6528usize), (3072, 2304)] {
            let (w, sc, bi) = make_symmetric_q8_bank(&mut rng, n, k, DType::Float16);
            for &m in &[1usize, 4, 8, 16] {
                let x = make_bf16_input(&mut rng, m, k);
                let sym = affine_bf16_matmul_few(&x, &w, &sc, None, 32, 8)
                    .unwrap()
                    .expect("symmetric affine_bf16_matmul_few must admit");
                let biased = affine_bf16_matmul_few(&x, &w, &sc, Some(&bi), 32, 8)
                    .unwrap()
                    .expect("biased affine_bf16_matmul_few must admit");
                let err = nrmse(&read_bf16_f32(&sym), &read_bf16_f32(&biased));
                assert!(
                    err <= 5e-3,
                    "symmetric affine_q8_few mismatch (m={m}, k={k}, n={n}): nrmse={err:.3e}"
                );
            }
        }

        // Cover all 3 sidecar dtypes (F16, BF16, F32), 8-warp small-N path, and
        // non-256-multiple K tail groups in MMA (K=384, odd N=67, M=12).
        for scale_dtype in [DType::Float16, DType::BFloat16, DType::Float32] {
            let (w, sc, bi) = make_symmetric_q8_bank(&mut rng, 67, 384, scale_dtype);
            for &m in &[1usize, 4, 12] {
                let x = make_bf16_input(&mut rng, m, 384);
                let sym = affine_bf16_matmul_few(&x, &w, &sc, None, 32, 8)
                    .unwrap()
                    .expect("symmetric affine_bf16_matmul_few must admit");
                let biased = affine_bf16_matmul_few(&x, &w, &sc, Some(&bi), 32, 8)
                    .unwrap()
                    .expect("biased affine_bf16_matmul_few must admit");
                let err = nrmse(&read_bf16_f32(&sym), &read_bf16_f32(&biased));
                assert!(
                    err <= 5e-3,
                    "symmetric affine_q8_few tail mismatch (dtype={scale_dtype:?}, m={m}): nrmse={err:.3e}"
                );
            }
        }
    }

    #[test]
    fn cuda_symmetric_q8_prefill_parity() {
        if active_backend() != ActiveBackend::CudaGpu {
            return;
        }
        let mut rng = Lcg::new(0x51A8_0002);
        let (k, n) = (2304usize, 512usize);
        let (w, sc, bi) = make_symmetric_q8_bank(&mut rng, n, k, DType::Float16);

        for &m in &[32usize, 128] {
            let x = make_bf16_input(&mut rng, m, k);
            for &gs_shift in &[-2i32, -1] {
                let sym_none = q8_prefill_cuda(&x, &w, &sc, None, gs_shift, false)
                    .unwrap()
                    .expect("symmetric q8_prefill_cuda (biases=None) must admit");
                let sym_with_bi = q8_prefill_cuda(&x, &w, &sc, Some(&bi), gs_shift, false)
                    .unwrap()
                    .expect("symmetric q8_prefill_cuda (biases=Some) must admit");
                let gen_with_bi = q8_prefill_cuda(&x, &w, &sc, Some(&bi), gs_shift, true)
                    .unwrap()
                    .expect("general_bias q8_prefill_cuda must admit");

                let got = read_bf16_f32(&sym_none);
                assert_eq!(
                    got,
                    read_bf16_f32(&sym_with_bi),
                    "q8_prefill_cuda biases=None must be bit-identical to general_bias=false (m={m}, gs={gs_shift})"
                );
                let err = nrmse(&got, &read_bf16_f32(&gen_with_bi));
                let lim = if gs_shift == -2 { 3.5e-3 } else { 1.2e-2 };
                assert!(
                    err <= lim,
                    "q8_prefill_cuda sym vs biased mismatch (m={m}, gs={gs_shift}): nrmse={err:.3e} > {lim:.1e}"
                );
            }
            // Reject general_bias=true when biases=None.
            assert!(
                q8_prefill_cuda(&x, &w, &sc, None, -2, true)
                    .unwrap()
                    .is_none(),
                "general_bias=true with biases=None must decline cleanly"
            );
        }
    }

    #[test]
    fn cuda_symmetric_q8_dequant_and_large_n_grid() {
        if active_backend() != ActiveBackend::CudaGpu {
            return;
        }
        let mut rng = Lcg::new(0x51A8_0003);

        // Parity across all 3 sidecar dtypes.
        for scale_dtype in [DType::Float16, DType::BFloat16, DType::Float32] {
            let (w, sc, bi) = make_symmetric_q8_bank(&mut rng, 128, 256, scale_dtype);
            let sym = dequant_q8_bf16_cuda(&w, &sc, None)
                .unwrap()
                .expect("symmetric dequant_q8_bf16_cuda must admit");
            let biased = dequant_q8_bf16_cuda(&w, &sc, Some(&bi))
                .unwrap()
                .expect("biased dequant_q8_bf16_cuda must admit");
            assert_eq!(
                read_bf16_f32(&sym),
                read_bf16_f32(&biased),
                "dequant_q8_bf16_cuda symmetric vs biased mismatch for {scale_dtype:?}"
            );
        }

        // N > 65535 (lm_head scale, e.g. 70000 > 65535) must succeed for both
        // symmetric (biases=None) and biased paths via grid = dim3(N, grid_k, 1).
        let (n, k) = (70_000usize, 256usize);
        let w_words: Vec<u32> = (0..n * (k / 4)).map(|_| rng.next_u32()).collect();
        let w = MxArray::from_uint32(&w_words, &[n as i64, (k / 4) as i64]).unwrap();
        let groups = k / 32;
        let sc_vals: Vec<f32> = (0..n * groups).map(|_| rng.uniform(0.002, 0.05)).collect();
        let sc = MxArray::from_float32(&sc_vals, &[n as i64, groups as i64])
            .unwrap()
            .astype(DType::Float16)
            .unwrap();
        let bi = sc
            .mul_scalar(-128.0)
            .unwrap()
            .astype(DType::Float16)
            .unwrap();

        let sym = dequant_q8_bf16_cuda(&w, &sc, None)
            .unwrap()
            .expect("symmetric dequant_q8_bf16_cuda must admit N=70000 > 65535");
        let biased = dequant_q8_bf16_cuda(&w, &sc, Some(&bi))
            .unwrap()
            .expect("biased dequant_q8_bf16_cuda must admit N=70000 > 65535");
        assert_eq!(sym.shape().unwrap().to_vec(), vec![n as i64, k as i64]);
        assert_eq!(biased.shape().unwrap().to_vec(), vec![n as i64, k as i64]);

        // Verify boundary slices at row 0, around 65535/65536, and the last row
        // against both biased GPU output and exact CPU ground truth.
        let sc_f32 = sc.astype(DType::Float32).unwrap().to_float32().unwrap();
        for &(start, end) in &[(0usize, 4usize), (65534, 65538), (69996, 70000)] {
            let s_slice = sym.slice_axis(0, start as i64, end as i64).unwrap();
            let b_slice = biased.slice_axis(0, start as i64, end as i64).unwrap();
            let got = read_bf16_f32(&s_slice);
            assert_eq!(
                got,
                read_bf16_f32(&b_slice),
                "dequant_q8_bf16_cuda N=70000 mismatch on rows {start}..{end}"
            );
            let mut want = Vec::with_capacity((end - start) * k);
            for row in start..end {
                for col in 0..k {
                    let word = w_words[row * (k / 4) + (col / 4)];
                    let byte = ((word >> ((col & 3) * 8)) & 0xFF) as u8;
                    let q_s8 = (byte ^ 0x80) as i8 as f32;
                    let s = sc_f32[row * groups + (col / 32)];
                    want.push(bf16::from_f32(s * q_s8).to_f32());
                }
            }
            assert_eq!(
                got, want,
                "dequant_q8_bf16_cuda N=70000 CPU oracle mismatch on rows {start}..{end}"
            );
        }
    }

    #[test]
    fn cuda_quantized_linear_symmetric_q8_forward_parity() {
        use crate::models::quantized_linear::QuantizedLinear;
        if active_backend() != ActiveBackend::CudaGpu {
            return;
        }
        let mut rng = Lcg::new(0x51A8_0004);
        // Even N (exercises affine_q8_few + q8_prefill_cuda) and odd N (exercises
        // affine_q8_few + dequant_q8_bf16_cuda fallback).
        for &(n, k) in &[(256usize, 256usize), (131, 256)] {
            let (w, sc, bi) = make_symmetric_q8_bank(&mut rng, n, k, DType::Float16);
            let biased_lin = QuantizedLinear::new(
                w.clone(),
                sc.clone(),
                Some(bi.clone()),
                None,
                32,
                8,
                "affine".to_string(),
            );
            let mut sym_lin =
                QuantizedLinear::new(w, sc, Some(bi), None, 32, 8, "affine".to_string());
            assert!(sym_lin.try_compact_symmetric_q8_f16().unwrap());
            assert!(sym_lin.is_symmetric_q8());
            assert!(sym_lin.get_biases().is_none());

            for &m in &[1usize, 4, 12, 32, 128] {
                let x = make_bf16_input(&mut rng, m, k);
                let got = sym_lin.forward(&x).unwrap();
                let want = biased_lin.forward(&x).unwrap();
                let err = nrmse(&read_bf16_f32(&got), &read_bf16_f32(&want));
                let lim = if m >= 128 && n % 2 == 0 { 1.2e-2 } else { 5e-3 };
                assert!(
                    err <= lim,
                    "QuantizedLinear symmetric_q8 forward mismatch (n={n}, k={k}, m={m}): nrmse={err:.3e}"
                );
            }
        }
    }

    #[test]
    fn cuda_affine_q8_few_pair_matches_single_sym_and_affine() {
        if active_backend() != ActiveBackend::CudaGpu {
            return;
        }
        let mut rng = Lcg::new(0x51A8_0005);
        let (n, k) = (1024usize, 2304usize);

        // 1. Exact Inkling-2-Mini KV projection geometry `[1024, 2304]` across M in {1, 2, 4, 8, 16}
        //    covering both SymQ8 (biases = None, F16 scales) and asymmetric AffineQ8 (F16 scales/biases).
        let (w0, s0_f16, _) = make_symmetric_q8_bank(&mut rng, n, k, DType::Float16);
        let (w1, s1_f16, _) = make_symmetric_q8_bank(&mut rng, n, k, DType::Float16);
        let groups = k / 32;
        let b0_vals: Vec<f32> = (0..n * groups).map(|_| rng.uniform(-6.0, 6.0)).collect();
        let b1_vals: Vec<f32> = (0..n * groups).map(|_| rng.uniform(-6.0, 6.0)).collect();
        let b0_f16 = MxArray::from_float32(&b0_vals, &[n as i64, groups as i64])
            .unwrap()
            .astype(DType::Float16)
            .unwrap();
        let b1_f16 = MxArray::from_float32(&b1_vals, &[n as i64, groups as i64])
            .unwrap()
            .astype(DType::Float16)
            .unwrap();

        for &m in &[1usize, 2, 4, 8, 16] {
            let x0 = make_bf16_input(&mut rng, m, k);
            let x1 = make_bf16_input(&mut rng, m, k);

            // SymQ8
            let (p0, p1) = affine_bf16_matmul_few_pair_cuda(
                &x0, &w0, &s0_f16, None, &x1, &w1, &s1_f16, None, 32, 8,
            )
            .unwrap()
            .expect("SymQ8 affine_bf16_matmul_few_pair_cuda must admit [1024, 2304]");
            let ref0 = affine_bf16_matmul_few(&x0, &w0, &s0_f16, None, 32, 8)
                .unwrap()
                .expect("SymQ8 single ref0");
            let ref1 = affine_bf16_matmul_few(&x1, &w1, &s1_f16, None, 32, 8)
                .unwrap()
                .expect("SymQ8 single ref1");
            assert_eq!(p0.shape().unwrap().as_ref(), ref0.shape().unwrap().as_ref());
            assert_eq!(p1.shape().unwrap().as_ref(), ref1.shape().unwrap().as_ref());
            let g0 = read_bf16_f32(&p0);
            let g1 = read_bf16_f32(&p1);
            let r0 = read_bf16_f32(&ref0);
            let r1 = read_bf16_f32(&ref1);
            assert_ne!(g0, g1, "pair outputs must reflect distinct inputs/weights");
            let err0 = nrmse(&g0, &r0);
            let err1 = nrmse(&g1, &r1);
            assert!(
                err0 <= 5e-3 && err1 <= 5e-3,
                "SymQ8 pair mismatch (m={m}): err0={err0:.3e}, err1={err1:.3e}"
            );

            // AffineQ8 (non-symmetric biases)
            let (ap0, ap1) = affine_bf16_matmul_few_pair_cuda(
                &x0,
                &w0,
                &s0_f16,
                Some(&b0_f16),
                &x1,
                &w1,
                &s1_f16,
                Some(&b1_f16),
                32,
                8,
            )
            .unwrap()
            .expect("AffineQ8 affine_bf16_matmul_few_pair_cuda must admit [1024, 2304]");
            let aref0 = affine_bf16_matmul_few(&x0, &w0, &s0_f16, Some(&b0_f16), 32, 8)
                .unwrap()
                .expect("AffineQ8 single ref0");
            let aref1 = affine_bf16_matmul_few(&x1, &w1, &s1_f16, Some(&b1_f16), 32, 8)
                .unwrap()
                .expect("AffineQ8 single ref1");
            let ag0 = read_bf16_f32(&ap0);
            let ag1 = read_bf16_f32(&ap1);
            let ar0 = read_bf16_f32(&aref0);
            let ar1 = read_bf16_f32(&aref1);
            let aerr0 = nrmse(&ag0, &ar0);
            let aerr1 = nrmse(&ag1, &ar1);
            assert!(
                aerr0 <= 5e-3 && aerr1 <= 5e-3,
                "AffineQ8 pair mismatch (m={m}): err0={aerr0:.3e}, err1={aerr1:.3e}"
            );
        }

        // 2. Cover BF16 and F32 sidecars + 2D `[M, K]` inputs (Inkling2 attention layout).
        for scale_dtype in [DType::BFloat16, DType::Float32] {
            let (sw0, ss0, sb0) = make_symmetric_q8_bank(&mut rng, 256, 512, scale_dtype);
            let (sw1, ss1, sb1) = make_symmetric_q8_bank(&mut rng, 256, 512, scale_dtype);
            for &m in &[1usize, 8, 16] {
                let x0 = make_bf16_input(&mut rng, m, 512)
                    .reshape(&[m as i64, 512])
                    .unwrap();
                let x1 = make_bf16_input(&mut rng, m, 512)
                    .reshape(&[m as i64, 512])
                    .unwrap();
                let (p0, p1) = affine_bf16_matmul_few_pair_cuda(
                    &x0,
                    &sw0,
                    &ss0,
                    Some(&sb0),
                    &x1,
                    &sw1,
                    &ss1,
                    Some(&sb1),
                    32,
                    8,
                )
                .unwrap()
                .expect("pair must admit BF16/F32 sidecars on 2D inputs");
                assert_eq!(p0.shape().unwrap().as_ref(), &[m as i64, 256]);
                assert_eq!(p1.shape().unwrap().as_ref(), &[m as i64, 256]);
                let ref0 = affine_bf16_matmul_few(&x0, &sw0, &ss0, Some(&sb0), 32, 8)
                    .unwrap()
                    .unwrap();
                let ref1 = affine_bf16_matmul_few(&x1, &sw1, &ss1, Some(&sb1), 32, 8)
                    .unwrap()
                    .unwrap();
                let err0 = nrmse(&read_bf16_f32(&p0), &read_bf16_f32(&ref0));
                let err1 = nrmse(&read_bf16_f32(&p1), &read_bf16_f32(&ref1));
                assert!(
                    err0 <= 5e-3 && err1 <= 5e-3,
                    "pair sidecar={scale_dtype:?} m={m} mismatch: err0={err0:.3e}, err1={err1:.3e}"
                );
            }
        }

        // 3. Guard checks: mismatched symmetric/biased, wrong group_size/bits, or M > 16 decline cleanly.
        let x_ok = make_bf16_input(&mut rng, 4, k);
        assert!(
            affine_bf16_matmul_few_pair_cuda(
                &x_ok,
                &w0,
                &s0_f16,
                None,
                &x_ok,
                &w1,
                &s1_f16,
                Some(&b1_f16),
                32,
                8,
            )
            .unwrap()
            .is_none()
        );
        assert!(
            affine_bf16_matmul_few_pair_cuda(
                &x_ok, &w0, &s0_f16, None, &x_ok, &w1, &s1_f16, None, 64, 8,
            )
            .unwrap()
            .is_none()
        );
        let x_big = make_bf16_input(&mut rng, 17, k);
        assert!(
            affine_bf16_matmul_few_pair_cuda(
                &x_big, &w0, &s0_f16, None, &x_big, &w1, &s1_f16, None, 32, 8,
            )
            .unwrap()
            .is_none()
        );
    }
}
