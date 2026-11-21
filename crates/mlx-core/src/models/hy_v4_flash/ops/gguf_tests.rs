use super::*;
use crate::array::alloc::{DstPtrs, OutLayout};
use crate::models::hy_v4_flash::bank::ProjArrays;
use crate::models::hy_v4_flash::kq_slot;
use crate::models::hy_v4_flash::moe::build_prefill_tiles;
use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_u8(&mut self) -> u8 {
        (self.next_u64() >> 24) as u8
    }

    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        let u = ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32);
        lo + (hi - lo) * u
    }
}

/// Encode a finite normal `f32` in typical weight-scale range to IEEE-754 f16 bits.
fn f32_to_f16_bits(v: f32) -> u16 {
    if v == 0.0 {
        return 0;
    }
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits & 0x007F_FFFF) >> 13;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7BFF;
    }
    sign | ((exp as u16) << 10) | (mant as u16)
}

fn gen_raw_q4k(rng: &mut Rng, n_experts: usize, rows: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let nb = n_experts * rows * (k / 256);
    let mut out = vec![0u8; nb * 144];
    for b in 0..nb {
        let blk = &mut out[b * 144..(b + 1) * 144];
        let d = f32_to_f16_bits(rng.next_f32(0.0008, 0.004));
        let dmin = f32_to_f16_bits(rng.next_f32(0.0004, 0.0025));
        blk[0..2].copy_from_slice(&d.to_le_bytes());
        blk[2..4].copy_from_slice(&dmin.to_le_bytes());
        for byte in &mut blk[4..144] {
            *byte = rng.next_u8();
        }
    }
    out
}

fn gen_raw_q6k(rng: &mut Rng, n_experts: usize, rows: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let nb = n_experts * rows * (k / 256);
    let mut out = vec![0u8; nb * 210];
    for b in 0..nb {
        let blk = &mut out[b * 210..(b + 1) * 210];
        for byte in &mut blk[0..208] {
            *byte = rng.next_u8();
        }
        let sign = if (rng.next_u64() & 1) == 0 { 1.0 } else { -1.0 };
        let d = f32_to_f16_bits(sign * rng.next_f32(0.0005, 0.003));
        blk[208..210].copy_from_slice(&d.to_le_bytes());
    }
    out
}

fn make_padded_u8_buf(raw: &[u8], base_off: u64, as_2d_pages: bool) -> MxArray {
    let off = base_off as usize;
    let min_len = off + raw.len() + 64;
    let total = if as_2d_pages {
        let page = 16384usize;
        min_len.div_ceil(page) * page
    } else {
        min_len
    };
    let mut buf = vec![0xA5u8; total];
    buf[off..off + raw.len()].copy_from_slice(raw);
    if as_2d_pages {
        let page = 16384i64;
        let n_pages = (total as i64) / page;
        MxArray::from_uint8(&buf, &[n_pages, page]).unwrap()
    } else {
        MxArray::from_uint8(&buf, &[total as i64]).unwrap()
    }
}

fn repack_to_proj_arrays(
    raw: &[u8],
    ty: GgufTensorType,
    n_experts: usize,
    rows: usize,
    k: usize,
) -> ProjArrays {
    let format = ty.k_quant_format().unwrap();
    let total_rows = n_experts * rows;
    let w_cols = format.weight_cols(k) as i64;
    let s_cols = format.scales_cols(k) as i64;
    let b_cols = format.biases_cols(k) as i64;
    let s_dtype = if format.scales_are_signed() {
        DType::Int8
    } else {
        DType::Uint8
    };
    let (w, wp, _) =
        MxArray::alloc_uninit(&[n_experts as i64, rows as i64, w_cols], DType::Uint32).unwrap();
    let (s, sp, _) =
        MxArray::alloc_uninit(&[n_experts as i64, rows as i64, s_cols], s_dtype).unwrap();
    let (b, bp, _) =
        MxArray::alloc_uninit(&[n_experts as i64, rows as i64, b_cols], DType::Float16).unwrap();

    let info = GgufTensorInfo {
        name: "test.weight".to_string(),
        n_dims: 2,
        dims: vec![k as u64, total_rows as u64],
        tensor_type: ty,
        offset: 0,
    };
    let dst = DstPtrs {
        weight: wp,
        scales: Some(sp),
        biases: Some(bp),
        rows: total_rows,
        k,
        layout: OutLayout::RowMajor,
    };
    kq_slot::rows_into(&info, raw, 0, total_rows, &dst).unwrap();
    ProjArrays { w, s, b, format }
}

fn require_metal() -> bool {
    std::env::var("MLX_TEST_REQUIRE_METAL").as_deref() == Ok("1")
}

fn eval_f32(a: &MxArray) -> Vec<f32> {
    a.astype(DType::Float32)
        .unwrap()
        .to_float32()
        .unwrap()
        .as_ref()
        .to_vec()
}

fn error_stats(got: &[f32], want: &[f32]) -> (f32, f64) {
    assert_eq!(got.len(), want.len());
    let mut max_abs = 0.0f32;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&g, &w) in got.iter().zip(want) {
        assert!(g.is_finite(), "non-finite got value: {g}");
        assert!(w.is_finite(), "non-finite want value: {w}");
        let d = (g - w).abs();
        if d > max_abs {
            max_abs = d;
        }
        let df = f64::from(g) - f64::from(w);
        num += df * df;
        den += f64::from(w) * f64::from(w);
    }
    let rel_l2 = if den > 0.0 {
        (num / den).sqrt()
    } else {
        num.sqrt()
    };
    (max_abs, rel_l2)
}

fn call_kq_routed_decode(
    x: &MxArray,
    slots: &MxArray,
    weights: &MxArray,
    g: &ProjArrays,
    u: &ProjArrays,
    d: &ProjArrays,
    down_bits: i32,
    limit: f32,
) -> Option<MxArray> {
    let h = unsafe {
        mlx_sys::mlx_hy4_kq_routed_decode(
            x.as_raw_ptr(),
            slots.as_raw_ptr(),
            weights.as_raw_ptr(),
            g.w.as_raw_ptr(),
            g.s.as_raw_ptr(),
            g.b.as_raw_ptr(),
            u.w.as_raw_ptr(),
            u.s.as_raw_ptr(),
            u.b.as_raw_ptr(),
            d.w.as_raw_ptr(),
            d.s.as_raw_ptr(),
            d.b.as_raw_ptr(),
            down_bits,
            limit,
        )
    };
    if h.is_null() {
        None
    } else {
        Some(MxArray::from_handle(h, "test kq_routed_decode").unwrap())
    }
}

fn call_kq_prefill_indirect_gu(
    x: &MxArray,
    token_rows: &MxArray,
    tiles: &MxArray,
    g: &ProjArrays,
    u: &ProjArrays,
    limit: f32,
) -> Option<MxArray> {
    let h = unsafe {
        mlx_sys::mlx_hy4_kq_prefill_indirect_gu(
            x.as_raw_ptr(),
            token_rows.as_raw_ptr(),
            tiles.as_raw_ptr(),
            g.w.as_raw_ptr(),
            g.s.as_raw_ptr(),
            g.b.as_raw_ptr(),
            u.w.as_raw_ptr(),
            u.s.as_raw_ptr(),
            u.b.as_raw_ptr(),
            limit,
        )
    };
    if h.is_null() {
        None
    } else {
        Some(MxArray::from_handle(h, "test kq_prefill_indirect_gu").unwrap())
    }
}

fn call_kq_sorted_combine(down: &MxArray, weights: &MxArray, inverse: &MxArray) -> MxArray {
    let h = unsafe {
        mlx_sys::mlx_hy4_kq_sorted_combine(
            down.as_raw_ptr(),
            weights.as_raw_ptr(),
            inverse.as_raw_ptr(),
        )
    };
    assert!(!h.is_null(), "kq_sorted_combine returned null");
    MxArray::from_handle(h, "test kq_sorted_combine").unwrap()
}

fn run_decode_parity_case(
    seed: u64,
    down_bits: i32,
    n_experts: usize,
    k: usize,
    m: usize,
    h: usize,
    rows: usize,
    top: usize,
    slot_ids: &[u32],
    act_scale: f32,
    limit: f32,
    as_2d_pages: bool,
) {
    let mut rng = Rng::new(seed);
    let raw_g = gen_raw_q4k(&mut rng, n_experts, m, k);
    let raw_u = gen_raw_q4k(&mut rng, n_experts, m, k);
    let (down_ty, raw_d, down_off) = if down_bits == 6 {
        // Odd 2-byte offset (66 = 2 mod 4) stresses 2-byte-only Q6_K alignment.
        (
            GgufTensorType::Q6K,
            gen_raw_q6k(&mut rng, n_experts, h, m),
            66u64,
        )
    } else {
        (
            GgufTensorType::Q4K,
            gen_raw_q4k(&mut rng, n_experts, h, m),
            64u64,
        )
    };
    let gate_off = 32u64;
    let up_off = 96u64;

    let g_buf = make_padded_u8_buf(&raw_g, gate_off, as_2d_pages);
    let u_buf = make_padded_u8_buf(&raw_u, up_off, as_2d_pages);
    let d_buf = make_padded_u8_buf(&raw_d, down_off, as_2d_pages);

    let g_kq = repack_to_proj_arrays(&raw_g, GgufTensorType::Q4K, n_experts, m, k);
    let u_kq = repack_to_proj_arrays(&raw_u, GgufTensorType::Q4K, n_experts, m, k);
    let d_kq = repack_to_proj_arrays(&raw_d, down_ty, n_experts, h, m);

    let x_vals: Vec<f32> = (0..rows * k)
        .map(|_| rng.next_f32(-act_scale, act_scale))
        .collect();
    let w_vals: Vec<f32> = (0..rows * top).map(|_| rng.next_f32(0.05, 0.35)).collect();

    let x = MxArray::from_float32(&x_vals, &[rows as i64, k as i64]).unwrap();
    let slots = MxArray::from_uint32(slot_ids, &[rows as i64, top as i64]).unwrap();
    let weights = MxArray::from_float32(&w_vals, &[rows as i64, top as i64]).unwrap();

    let Some(ref_out) =
        call_kq_routed_decode(&x, &slots, &weights, &g_kq, &u_kq, &d_kq, down_bits, limit)
    else {
        assert!(
            !require_metal(),
            "Metal required but kq_routed_decode returned None"
        );
        return;
    };
    let got_out = gguf_routed_decode(
        &x, &slots, &weights, &g_buf, gate_off, &u_buf, up_off, &d_buf, down_off, n_experts, m, h,
        down_bits, limit,
    )
    .unwrap()
    .expect("gguf_routed_decode returned None when kq_routed_decode succeeded");

    let ref_v = eval_f32(&ref_out);
    let got_v = eval_f32(&got_out);
    let (max_abs, rel_l2) = error_stats(&got_v, &ref_v);
    eprintln!(
        "gguf_decode bits={down_bits} K={k} M={m} H={h} ROWS={rows} TOP={top}: max_abs={max_abs:e} rel_l2={rel_l2:e}"
    );
    assert!(
        rel_l2 <= 1e-5,
        "decode parity failed (bits={down_bits} K={k} M={m} H={h} ROWS={rows} TOP={top}): max_abs={max_abs:e} rel_l2={rel_l2:e}"
    );
}

#[test]
fn gguf_routed_decode_matches_kq_decode_q4k_and_q6k() {
    for down_bits in [4, 6] {
        // ROWS=1, odd K super-block count (K=768 -> nb=3), non-multiple-of-4 H tail,
        // and large activations that trigger asymmetric SwiGLU clamping (gate > lim, gate < -lim).
        run_decode_parity_case(
            100 + down_bits as u64,
            down_bits,
            6,
            768,
            256,
            259,
            1,
            4,
            &[0, 2, 3, 5],
            6.0,
            7.0,
            false,
        );

        // ROWS=2 with duplicate experts to exercise DEDUP path + 2-D page buffer layout
        // + out-of-range slot (slot == n_experts) safely treated as zero.
        run_decode_parity_case(
            200 + down_bits as u64,
            down_bits,
            6,
            512,
            256,
            256,
            2,
            4,
            &[1, 3, 4, 6, 1, 2, 3, 5],
            8.0,
            7.0,
            true,
        );

        // ROWS=4 with >1 duplicate experts and non-duplicate mix (DEDUP).
        run_decode_parity_case(
            300 + down_bits as u64,
            down_bits,
            8,
            512,
            256,
            256,
            4,
            4,
            &[0, 1, 2, 3, 1, 2, 4, 5, 0, 2, 6, 7, 1, 2, 3, 7],
            6.0,
            7.0,
            true,
        );

        // Realistic Hy4 dims (K=1536, M=1536, H=1536, TOP=16), ROWS=1 and ROWS=2 (DEDUP).
        let slots_r1: Vec<u32> = (0..16).map(|i| i % 4).collect();
        run_decode_parity_case(
            400 + down_bits as u64,
            down_bits,
            4,
            1536,
            1536,
            1536,
            1,
            16,
            &slots_r1,
            4.0,
            7.0,
            true,
        );
        let slots_r2: Vec<u32> = (0..32).map(|i| (i * 3) % 4).collect();
        run_decode_parity_case(
            500 + down_bits as u64,
            down_bits,
            4,
            1536,
            1536,
            1536,
            2,
            16,
            &slots_r2,
            4.0,
            7.0,
            true,
        );
        // Realistic MTP-1 mix: ROWS=2, TOP=16 with unique + shared experts + OOB slot.
        let mut slots_r2_mix: Vec<u32> = (0..16).chain(10..26).map(|i| i % 24).collect();
        slots_r2_mix[15] = 24;
        run_decode_parity_case(
            550 + down_bits as u64,
            down_bits,
            24,
            1536,
            1536,
            1536,
            2,
            16,
            &slots_r2_mix,
            4.0,
            7.0,
            true,
        );
        // Realistic Hy4 dims, ROWS=4, TOP=16 with shared + unique experts + OOB slot.
        let mut slots_r4: Vec<u32> = (0..64)
            .map(|i| ((i / 16) * 5 + (i % 16) * 2) % 12)
            .collect();
        slots_r4[7] = 12;
        slots_r4[39] = 12;
        run_decode_parity_case(
            600 + down_bits as u64,
            down_bits,
            12,
            1536,
            1536,
            1536,
            4,
            16,
            &slots_r4,
            4.0,
            7.0,
            true,
        );
    }
}

#[test]
fn gguf_routed_decode_multi_row_dedup_and_row_invariance() {
    for down_bits in [4, 6] {
        // Test ROWS=1,2,3,4,8 on both compact non-multiple-of-4 H (odd K blocks) and Hy4 dims.
        let configs: &[(usize, usize, usize, usize, usize, u64)] = &[
            (24, 768, 256, 259, 8, 7000),
            (32, 1536, 1536, 1536, 16, 8000),
        ];
        for &(n_experts, k, m, h, top, base_seed) in configs {
            let mut rng = Rng::new(base_seed + down_bits as u64);
            let raw_g = gen_raw_q4k(&mut rng, n_experts, m, k);
            let raw_u = gen_raw_q4k(&mut rng, n_experts, m, k);
            let (down_ty, raw_d, down_off) = if down_bits == 6 {
                // Odd 2-byte offset stresses 2-byte-only Q6_K alignment.
                (
                    GgufTensorType::Q6K,
                    gen_raw_q6k(&mut rng, n_experts, h, m),
                    66u64,
                )
            } else {
                (
                    GgufTensorType::Q4K,
                    gen_raw_q4k(&mut rng, n_experts, h, m),
                    64u64,
                )
            };
            let gate_off = 32u64;
            let up_off = 96u64;
            let g_buf = make_padded_u8_buf(&raw_g, gate_off, true);
            let u_buf = make_padded_u8_buf(&raw_u, up_off, true);
            let d_buf = make_padded_u8_buf(&raw_d, down_off, true);
            let g_kq = repack_to_proj_arrays(&raw_g, GgufTensorType::Q4K, n_experts, m, k);
            let u_kq = repack_to_proj_arrays(&raw_u, GgufTensorType::Q4K, n_experts, m, k);
            let d_kq = repack_to_proj_arrays(&raw_d, down_ty, n_experts, h, m);

            let max_rows = 8usize;
            let x_vals: Vec<f32> = (0..max_rows * k).map(|_| rng.next_f32(-5.0, 5.0)).collect();
            let w_vals: Vec<f32> = (0..max_rows * top)
                .map(|_| rng.next_f32(0.05, 0.35))
                .collect();

            // Craft routing across 8 rows with counts of 1, 2, 3, 4, 5, 6, and 8 copies
            // of the same expert (forcing >4 chunk splitting), plus multiple OOB slots.
            let mut slot_ids = vec![0u32; max_rows * top];
            for r in 0..max_rows {
                let row = &mut slot_ids[r * top..(r + 1) * top];
                row[0] = 0; // 8 copies across rows 0..8 -> splits into 4 + 4
                row[1] = if r < 6 { 1 } else { (10 + r) as u32 }; // 6 copies -> splits into 4 + 2
                row[2] = if r < 5 { 2 } else { (12 + r) as u32 }; // 5 copies -> splits into 4 + 1
                row[3] = if r < 4 { 3 } else { (14 + r) as u32 }; // 4 copies
                row[4] = if r < 3 { 4 } else { (15 + r) as u32 }; // 3 copies
                row[5] = if r < 2 { 5 } else { (16 + r) as u32 }; // 2 copies
                for (j, slot) in row.iter_mut().enumerate().skip(6) {
                    *slot = ((6 + r * 3 + j) % n_experts) as u32;
                }
                if r == 1 || r == 5 {
                    row[top - 1] = n_experts as u32; // OOB slot
                }
            }

            // Single-row baselines for strict bit-exact row invariance.
            let mut single_bits = Vec::with_capacity(max_rows);
            for r in 0..max_rows {
                let xr =
                    MxArray::from_float32(&x_vals[r * k..(r + 1) * k], &[1, k as i64]).unwrap();
                let slr = MxArray::from_uint32(&slot_ids[r * top..(r + 1) * top], &[1, top as i64])
                    .unwrap();
                let wr = MxArray::from_float32(&w_vals[r * top..(r + 1) * top], &[1, top as i64])
                    .unwrap();
                let Some(got) = gguf_routed_decode(
                    &xr, &slr, &wr, &g_buf, gate_off, &u_buf, up_off, &d_buf, down_off, n_experts,
                    m, h, down_bits, 7.0,
                )
                .unwrap() else {
                    assert!(!require_metal(), "Metal required");
                    return;
                };
                let gv = eval_f32(&got);
                single_bits.push(gv.iter().map(|v| v.to_bits()).collect::<Vec<u32>>());
            }

            for rows in [1usize, 2, 3, 4, 8] {
                let xm =
                    MxArray::from_float32(&x_vals[..rows * k], &[rows as i64, k as i64]).unwrap();
                let slm = MxArray::from_uint32(&slot_ids[..rows * top], &[rows as i64, top as i64])
                    .unwrap();
                let wm = MxArray::from_float32(&w_vals[..rows * top], &[rows as i64, top as i64])
                    .unwrap();

                let ref_out =
                    call_kq_routed_decode(&xm, &slm, &wm, &g_kq, &u_kq, &d_kq, down_bits, 7.0)
                        .unwrap();
                let got_out = gguf_routed_decode(
                    &xm, &slm, &wm, &g_buf, gate_off, &u_buf, up_off, &d_buf, down_off, n_experts,
                    m, h, down_bits, 7.0,
                )
                .unwrap()
                .unwrap();

                let ref_v = eval_f32(&ref_out);
                let got_v = eval_f32(&got_out);
                let (max_abs, rel_l2) = error_stats(&got_v, &ref_v);
                assert!(
                    rel_l2 <= 1e-5,
                    "dedup parity bits={down_bits} K={k} M={m} H={h} ROWS={rows}: max_abs={max_abs:e} rel_l2={rel_l2:e}"
                );

                // Bit-exact row invariance against ROWS=1.
                for r in 0..rows {
                    let row_bits: Vec<u32> = got_v[r * h..(r + 1) * h]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect();
                    assert_eq!(
                        row_bits, single_bits[r],
                        "gguf_routed_decode row invariance bits={down_bits} H={h} ROWS={rows} row={r}"
                    );
                }
            }
        }
    }
}

/// Explicitly verify asymmetric clamp: gate is upper-clamped at +lim, NEVER lower-clamped at -lim.
#[test]
fn gguf_decode_preserves_asymmetric_gate_clamp() {
    let n_experts = 1usize;
    let (k, m, h) = (256usize, 256usize, 256usize);
    let mut rng = Rng::new(777);
    let raw_g = gen_raw_q4k(&mut rng, n_experts, m, k);
    let raw_u = gen_raw_q4k(&mut rng, n_experts, m, k);
    let raw_d = gen_raw_q4k(&mut rng, n_experts, h, m);

    let g_buf = make_padded_u8_buf(&raw_g, 32, false);
    let u_buf = make_padded_u8_buf(&raw_u, 64, false);
    let d_buf = make_padded_u8_buf(&raw_d, 96, false);

    let g_kq = repack_to_proj_arrays(&raw_g, GgufTensorType::Q4K, n_experts, m, k);
    let u_kq = repack_to_proj_arrays(&raw_u, GgufTensorType::Q4K, n_experts, m, k);
    let d_kq = repack_to_proj_arrays(&raw_d, GgufTensorType::Q4K, n_experts, h, m);

    // Strongly negative activations drive gate << -lim (e.g. -50) while limit = 1.0.
    let x_vals: Vec<f32> = (0..k).map(|i| -12.0 - (i % 7) as f32 * 1.5).collect();
    let x = MxArray::from_float32(&x_vals, &[1, k as i64]).unwrap();
    let slots = MxArray::from_uint32(&[0u32], &[1, 1]).unwrap();
    let weights = MxArray::from_float32(&[1.0f32], &[1, 1]).unwrap();

    let Some(ref_out) = call_kq_routed_decode(&x, &slots, &weights, &g_kq, &u_kq, &d_kq, 4, 1.0)
    else {
        assert!(!require_metal(), "Metal required");
        return;
    };
    let got_out = gguf_routed_decode(
        &x, &slots, &weights, &g_buf, 32, &u_buf, 64, &d_buf, 96, n_experts, m, h, 4, 1.0,
    )
    .unwrap()
    .unwrap();
    let (max_abs, rel_l2) = error_stats(&eval_f32(&got_out), &eval_f32(&ref_out));
    eprintln!("gguf_asymmetric_clamp: max_abs={max_abs:e} rel_l2={rel_l2:e}");
    assert!(
        max_abs <= 1e-4 && rel_l2 <= 1e-5,
        "asymmetric clamp parity failed: max_abs={max_abs:e} rel_l2={rel_l2:e}"
    );
}

fn run_prefill_parity_case(
    seed: u64,
    down_bits: i32,
    n_experts: usize,
    k: usize,
    m: usize,
    h: usize,
    n_tok: usize,
    top: usize,
    act_scale: f32,
    limit: f32,
) {
    let mut rng = Rng::new(seed);
    let raw_g = gen_raw_q4k(&mut rng, n_experts, m, k);
    let raw_u = gen_raw_q4k(&mut rng, n_experts, m, k);
    let (down_ty, raw_d, down_off) = if down_bits == 6 {
        (
            GgufTensorType::Q6K,
            gen_raw_q6k(&mut rng, n_experts, h, m),
            66u64,
        )
    } else {
        (
            GgufTensorType::Q4K,
            gen_raw_q4k(&mut rng, n_experts, h, m),
            64u64,
        )
    };
    let gate_off = 32u64;
    let up_off = 96u64;

    let g_buf = make_padded_u8_buf(&raw_g, gate_off, true);
    let u_buf = make_padded_u8_buf(&raw_u, up_off, true);
    let d_buf = make_padded_u8_buf(&raw_d, down_off, true);

    let g_kq = repack_to_proj_arrays(&raw_g, GgufTensorType::Q4K, n_experts, m, k);
    let u_kq = repack_to_proj_arrays(&raw_u, GgufTensorType::Q4K, n_experts, m, k);
    let d_kq = repack_to_proj_arrays(&raw_d, down_ty, n_experts, h, m);

    let x_vals: Vec<f32> = (0..n_tok * k)
        .map(|_| rng.next_f32(-act_scale, act_scale))
        .collect();
    let pairs = n_tok * top;
    // Skew routing across experts so we get full 32-row tiles AND partial tail tiles (< 16 and 16..31).
    let mut ids = Vec::with_capacity(pairs);
    let mut w_vals = Vec::with_capacity(pairs);
    for t in 0..n_tok {
        for j in 0..top {
            let e = if (t + j) % 5 == 0 {
                0
            } else {
                ((t * 7 + j * 3) % n_experts) as u32
            };
            ids.push(e);
            w_vals.push(rng.next_f32(0.05, 0.3));
        }
    }

    // Counting sort by expert id (same as direct_prefill_sorted).
    let mut counts = vec![0usize; n_experts];
    for &e in &ids {
        counts[e as usize] += 1;
    }
    let mut offsets = vec![0usize; n_experts];
    let mut total = 0usize;
    for (e, &c) in counts.iter().enumerate() {
        offsets[e] = total;
        total += c;
    }
    let mut inverse = vec![0u32; pairs];
    let mut token_rows = vec![0u32; pairs];
    let mut sorted_slots = vec![0u32; pairs];
    for (pair_idx, &e) in ids.iter().enumerate() {
        let pos = offsets[e as usize];
        offsets[e as usize] = pos + 1;
        inverse[pair_idx] = pos as u32;
        token_rows[pos] = (pair_idx / top) as u32;
        sorted_slots[pos] = e;
    }

    let tiles = build_prefill_tiles(&sorted_slots);
    let n_tiles = tiles.len() / 3;

    let x_arr = MxArray::from_float32(&x_vals, &[n_tok as i64, k as i64]).unwrap();
    let tr_arr = MxArray::from_uint32(&token_rows, &[pairs as i64]).unwrap();
    let tiles_arr = MxArray::from_uint32(&tiles, &[n_tiles as i64, 3]).unwrap();
    let slots_arr = MxArray::from_uint32(&sorted_slots, &[pairs as i64]).unwrap();
    let weights_arr = MxArray::from_float32(&w_vals, &[n_tok as i64, top as i64]).unwrap();
    let inv_arr = MxArray::from_uint32(&inverse, &[n_tok as i64, top as i64]).unwrap();

    let Some(ref_hidden) =
        call_kq_prefill_indirect_gu(&x_arr, &tr_arr, &tiles_arr, &g_kq, &u_kq, limit)
    else {
        assert!(
            !require_metal(),
            "Metal/NAX required but kq_prefill_indirect_gu returned None"
        );
        return;
    };
    let got_hidden = gguf_prefill_indirect_gu(
        &x_arr, &tr_arr, &tiles_arr, &g_buf, gate_off, &u_buf, up_off, n_experts, m, limit,
    )
    .unwrap()
    .expect("gguf_prefill_indirect_gu returned None when kq_prefill_indirect_gu succeeded");

    let x_sorted = x_arr
        .take(&tr_arr, 0)
        .unwrap()
        .reshape(&[pairs as i64, 1, 1, k as i64])
        .unwrap();
    let slots_2d = slots_arr.reshape(&[pairs as i64, 1]).unwrap();
    let fp32_g = g_kq
        .gather(&x_sorted, &slots_2d)
        .unwrap()
        .astype(DType::Float32)
        .unwrap()
        .reshape(&[pairs as i64, m as i64])
        .unwrap();
    let fp32_u = u_kq
        .gather(&x_sorted, &slots_2d)
        .unwrap()
        .astype(DType::Float32)
        .unwrap()
        .reshape(&[pairs as i64, m as i64])
        .unwrap();
    let fp32_hidden =
        crate::nn::Activations::swiglu_clamped_compiled(&fp32_g, &fp32_u, limit).unwrap();

    let gu_hilo = std::env::var("MLX_HY4_GU_HILO").as_deref() == Ok("1");
    let down_f32 = std::env::var("MLX_HY4_DOWN_F32").as_deref() == Ok("1");
    let gu_f32_tol = if gu_hilo { 5e-4 } else { 2e-3 };
    let (gu_f32_max, gu_f32_rel) = error_stats(&eval_f32(&got_hidden), &eval_f32(&fp32_hidden));
    eprintln!(
        "gguf_prefill GU vs FP32 bits={down_bits} K={k} M={m} H={h} R={pairs}: max_abs={gu_f32_max:e} rel_l2={gu_f32_rel:e}"
    );
    assert!(
        gu_f32_rel <= gu_f32_tol,
        "prefill GU vs FP32 parity failed (bits={down_bits} K={k} M={m} H={h} R={pairs}): max_abs={gu_f32_max:e} rel_l2={gu_f32_rel:e}"
    );

    let (gu_max, gu_rel) = error_stats(&eval_f32(&got_hidden), &eval_f32(&ref_hidden));
    eprintln!(
        "gguf_prefill GU vs TF32-kq bits={down_bits} K={k} M={m} H={h} R={pairs}: max_abs={gu_max:e} rel_l2={gu_rel:e}"
    );
    assert!(
        gu_rel <= 2.5e-3,
        "prefill GU vs TF32-kq parity failed (bits={down_bits} K={k} M={m} H={h} R={pairs}): max_abs={gu_max:e} rel_l2={gu_rel:e}"
    );

    let max_group = sorted_slots.last().map_or(0, |&s| (s as i32) + 1);
    let fp32_down = d_kq
        .gather(
            &fp32_hidden
                .reshape(&[pairs as i64, 1, 1, m as i64])
                .unwrap(),
            &slots_2d,
        )
        .unwrap()
        .astype(DType::Float32)
        .unwrap()
        .reshape(&[pairs as i64, h as i64])
        .unwrap();
    let ref_down = d_kq
        .gather_sorted_nax_bounded(
            &ref_hidden.reshape(&[pairs as i64, 1, m as i64]).unwrap(),
            &slots_arr,
            n_tiles as i32,
            max_group,
        )
        .unwrap()
        .astype(DType::Float32)
        .unwrap()
        .reshape(&[pairs as i64, h as i64])
        .unwrap();
    let got_down = gguf_prefill_indirect_down(
        &got_hidden,
        &tiles_arr,
        &d_buf,
        down_off,
        n_experts,
        h,
        down_bits,
    )
    .unwrap()
    .expect("gguf_prefill_indirect_down returned None when NAX is available");
    assert_eq!(
        got_down.dtype().unwrap(),
        if down_f32 {
            DType::Float32
        } else {
            DType::Float16
        },
        "gguf_prefill_indirect_down output dtype mismatch"
    );

    let d_f32_tol = if gu_hilo && down_f32 { 5e-4 } else { 2.5e-3 };
    let (d_f32_max, d_f32_rel) = error_stats(&eval_f32(&got_down), &eval_f32(&fp32_down));
    eprintln!(
        "gguf_prefill DOWN vs FP32 bits={down_bits} K={k} M={m} H={h} R={pairs}: max_abs={d_f32_max:e} rel_l2={d_f32_rel:e}"
    );
    assert!(
        d_f32_rel <= d_f32_tol,
        "prefill DOWN vs FP32 parity failed (bits={down_bits} K={k} M={m} H={h} R={pairs}): max_abs={d_f32_max:e} rel_l2={d_f32_rel:e}"
    );

    let (d_max, d_rel) = error_stats(&eval_f32(&got_down), &eval_f32(&ref_down));
    eprintln!(
        "gguf_prefill DOWN vs TF32-kq bits={down_bits} K={k} M={m} H={h} R={pairs}: max_abs={d_max:e} rel_l2={d_rel:e}"
    );
    assert!(
        d_rel <= 3e-3,
        "prefill DOWN vs TF32-kq parity failed (bits={down_bits} K={k} M={m} H={h} R={pairs}): max_abs={d_max:e} rel_l2={d_rel:e}"
    );

    let fp32_comb = call_kq_sorted_combine(&fp32_down, &weights_arr, &inv_arr);
    let ref_comb = call_kq_sorted_combine(&ref_down, &weights_arr, &inv_arr);
    let got_comb = call_kq_sorted_combine(&got_down, &weights_arr, &inv_arr);
    let (c_f32_max, c_f32_rel) = error_stats(&eval_f32(&got_comb), &eval_f32(&fp32_comb));
    eprintln!(
        "gguf_prefill COMBINE vs FP32 bits={down_bits} K={k} M={m} H={h} R={pairs}: max_abs={c_f32_max:e} rel_l2={c_f32_rel:e}"
    );
    assert!(
        c_f32_rel <= d_f32_tol,
        "prefill COMBINE vs FP32 parity failed (bits={down_bits} K={k} M={m} H={h} R={pairs}): max_abs={c_f32_max:e} rel_l2={c_f32_rel:e}"
    );
    let (c_max, c_rel) = error_stats(&eval_f32(&got_comb), &eval_f32(&ref_comb));
    eprintln!(
        "gguf_prefill COMBINE vs TF32-kq bits={down_bits} K={k} M={m} H={h} R={pairs}: max_abs={c_max:e} rel_l2={c_rel:e}"
    );
    assert!(
        c_rel <= 3e-3,
        "prefill COMBINE vs TF32-kq parity failed (bits={down_bits} K={k} M={m} H={h} R={pairs}): max_abs={c_max:e} rel_l2={c_rel:e}"
    );
}

#[test]
fn gguf_prefill_nax_matches_kq_prefill_q4k_and_q6k() {
    for down_bits in [4, 6] {
        // Compact dimensions with partial tail tiles and clamp-hitting activations.
        run_prefill_parity_case(
            1000 + down_bits as u64,
            down_bits,
            5,
            512,
            256,
            256,
            20,
            4,
            6.0,
            7.0,
        );

        // Realistic Hy4 dimensions (K=1536, M=1536, H=1536, R=128).
        run_prefill_parity_case(
            2000 + down_bits as u64,
            down_bits,
            4,
            1536,
            1536,
            1536,
            8,
            16,
            4.0,
            7.0,
        );
    }
}

#[test]
fn gguf_prefill_oob_slot_writes_zeros() {
    let n_experts = 2usize;
    let (k, m, h) = (256usize, 256usize, 256usize);
    let mut rng = Rng::new(909);
    let g_buf = make_padded_u8_buf(&gen_raw_q4k(&mut rng, n_experts, m, k), 32, false);
    let u_buf = make_padded_u8_buf(&gen_raw_q4k(&mut rng, n_experts, m, k), 32, false);
    let d_buf = make_padded_u8_buf(&gen_raw_q6k(&mut rng, n_experts, h, m), 66, false);

    let x = MxArray::from_float32(&vec![1.0f32; 4 * k], &[4, k as i64]).unwrap();
    let tr = MxArray::from_uint32(&[0, 1, 2, 3], &[4]).unwrap();
    // Slot 99 >= n_experts must produce exact zeros rather than leaving garbage.
    let tiles = MxArray::from_uint32(&[0, 4, 99], &[1, 3]).unwrap();

    let Some(gu) =
        gguf_prefill_indirect_gu(&x, &tr, &tiles, &g_buf, 32, &u_buf, 32, n_experts, m, 7.0)
            .unwrap()
    else {
        assert!(!require_metal(), "Metal/NAX required");
        return;
    };
    let gu_v = eval_f32(&gu);
    assert!(gu_v.iter().all(|&v| v == 0.0), "OOB slot GU not zero");

    let ones = MxArray::from_float32(&vec![1.0f32; 4 * m], &[4, m as i64]).unwrap();
    let dn = gguf_prefill_indirect_down(&ones, &tiles, &d_buf, 66, n_experts, h, 6)
        .unwrap()
        .unwrap();
    let dn_v = eval_f32(&dn);
    assert!(dn_v.iter().all(|&v| v == 0.0), "OOB slot DOWN not zero");
}

#[test]
fn gguf_prefill_non_multiple_of_64_tail_matches_decode() {
    for down_bits in [4, 6] {
        let (n_experts, k, m, h, n_tok, top) =
            (3usize, 512usize, 256usize, 260usize, 5usize, 2usize);
        let mut rng = Rng::new(3100 + down_bits as u64);
        let raw_g = gen_raw_q4k(&mut rng, n_experts, m, k);
        let raw_u = gen_raw_q4k(&mut rng, n_experts, m, k);
        let (down_ty, raw_d, down_off) = if down_bits == 6 {
            (
                GgufTensorType::Q6K,
                gen_raw_q6k(&mut rng, n_experts, h, m),
                66u64,
            )
        } else {
            (
                GgufTensorType::Q4K,
                gen_raw_q4k(&mut rng, n_experts, h, m),
                64u64,
            )
        };
        let g_buf = make_padded_u8_buf(&raw_g, 32, false);
        let u_buf = make_padded_u8_buf(&raw_u, 96, false);
        let d_buf = make_padded_u8_buf(&raw_d, down_off, false);
        let d_kq = repack_to_proj_arrays(&raw_d, down_ty, n_experts, h, m);

        let x_vals: Vec<f32> = (0..n_tok * k).map(|_| rng.next_f32(-4.0, 4.0)).collect();
        let ids: Vec<u32> = vec![0, 1, 0, 2, 1, 2, 0, 2, 1, 2];
        let w_vals: Vec<f32> = (0..n_tok * top).map(|_| rng.next_f32(0.1, 0.4)).collect();

        let x = MxArray::from_float32(&x_vals, &[n_tok as i64, k as i64]).unwrap();
        let slots = MxArray::from_uint32(&ids, &[n_tok as i64, top as i64]).unwrap();
        let weights = MxArray::from_float32(&w_vals, &[n_tok as i64, top as i64]).unwrap();

        let Some(dec) = gguf_routed_decode(
            &x, &slots, &weights, &g_buf, 32, &u_buf, 96, &d_buf, down_off, n_experts, m, h,
            down_bits, 7.0,
        )
        .unwrap() else {
            assert!(!require_metal(), "Metal required");
            return;
        };

        let pairs = n_tok * top;
        let mut order: Vec<usize> = (0..pairs).collect();
        order.sort_by_key(|&i| ids[i]);
        let mut inverse = vec![0u32; pairs];
        for (pos, &orig) in order.iter().enumerate() {
            inverse[orig] = pos as u32;
        }
        let token_rows: Vec<u32> = order.iter().map(|&i| (i / top) as u32).collect();
        let sorted_slots: Vec<u32> = order.iter().map(|&i| ids[i]).collect();
        let tiles = build_prefill_tiles(&sorted_slots);
        let n_tiles = tiles.len() / 3;

        let tr = MxArray::from_uint32(&token_rows, &[pairs as i64]).unwrap();
        let tl = MxArray::from_uint32(&tiles, &[n_tiles as i64, 3]).unwrap();
        let slots_arr = MxArray::from_uint32(&sorted_slots, &[pairs as i64]).unwrap();
        let inv = MxArray::from_uint32(&inverse, &[n_tok as i64, top as i64]).unwrap();

        let Some(gu) =
            gguf_prefill_indirect_gu(&x, &tr, &tl, &g_buf, 32, &u_buf, 96, n_experts, m, 7.0)
                .unwrap()
        else {
            assert!(!require_metal(), "Metal/NAX required");
            return;
        };
        let dn = gguf_prefill_indirect_down(&gu, &tl, &d_buf, down_off, n_experts, h, down_bits)
            .unwrap()
            .unwrap();
        let max_group = sorted_slots.last().map_or(0, |&s| (s as i32) + 1);
        let ref_dn = d_kq
            .gather_sorted_nax_bounded(
                &gu.reshape(&[pairs as i64, 1, m as i64]).unwrap(),
                &slots_arr,
                n_tiles as i32,
                max_group,
            )
            .unwrap()
            .astype(DType::Float32)
            .unwrap()
            .reshape(&[pairs as i64, h as i64])
            .unwrap();
        let (dn_max, dn_rel) = error_stats(&eval_f32(&dn), &eval_f32(&ref_dn));
        eprintln!(
            "gguf_prefill_tail DOWN vs NAX bits={down_bits} H={h}: max_abs={dn_max:e} rel_l2={dn_rel:e}"
        );
        // `gather_sorted_nax_bounded` falls back to FP32 when H % 64 != 0,
        // so this compares one TF32 NAX matmul against FP32 (~2^-10 ~= 1e-3).
        assert!(
            dn_rel <= 1.5e-3,
            "prefill tail DOWN parity failed (bits={down_bits} H={h}): max_abs={dn_max:e} rel_l2={dn_rel:e}"
        );

        let comb = call_kq_sorted_combine(&dn, &weights, &inv);
        let (max_abs, rel_l2) = error_stats(&eval_f32(&comb), &eval_f32(&dec));
        eprintln!(
            "gguf_prefill_tail vs FP32 decode bits={down_bits} H={h}: max_abs={max_abs:e} rel_l2={rel_l2:e}"
        );
        // Two chained TF32 matmuls vs FP32 accumulate ~1-2 TF32 ULPs (2^-10 ~= 1e-3).
        assert!(
            rel_l2 <= 2e-3,
            "prefill tail vs FP32 parity failed (bits={down_bits} H={h}): max_abs={max_abs:e} rel_l2={rel_l2:e}"
        );
    }
}

#[test]
fn gguf_invalid_args_return_none() {
    let buf = MxArray::from_uint8(&vec![0u8; 65536], &[65536]).unwrap();
    let x = MxArray::from_float32(&vec![0.0f32; 256], &[1, 256]).unwrap();
    let slots = MxArray::from_uint32(&[0], &[1, 1]).unwrap();
    let weights = MxArray::from_float32(&[1.0], &[1, 1]).unwrap();
    let tr = MxArray::from_uint32(&[0], &[1]).unwrap();
    let tiles = MxArray::from_uint32(&[0, 1, 0], &[1, 3]).unwrap();

    // Misaligned Q4_K offset (not multiple of 4).
    assert!(
        gguf_routed_decode(
            &x, &slots, &weights, &buf, 2, &buf, 0, &buf, 0, 1, 256, 256, 4, 7.0
        )
        .unwrap()
        .is_none()
    );
    // Misaligned Q6_K offset (odd).
    assert!(
        gguf_routed_decode(
            &x, &slots, &weights, &buf, 0, &buf, 0, &buf, 1, 1, 256, 256, 6, 7.0
        )
        .unwrap()
        .is_none()
    );
    // Invalid down_bits.
    assert!(
        gguf_routed_decode(
            &x, &slots, &weights, &buf, 0, &buf, 0, &buf, 0, 1, 256, 256, 5, 7.0
        )
        .unwrap()
        .is_none()
    );
    // Non-finite limit.
    assert!(
        gguf_routed_decode(
            &x,
            &slots,
            &weights,
            &buf,
            0,
            &buf,
            0,
            &buf,
            0,
            1,
            256,
            256,
            4,
            f32::NAN
        )
        .unwrap()
        .is_none()
    );
    // Truncated buffer (requires 256*144 = 36864 bytes; offset 40000 leaves < 36864).
    assert!(
        gguf_prefill_indirect_gu(&x, &tr, &tiles, &buf, 40000, &buf, 0, 1, 256, 7.0)
            .unwrap()
            .is_none()
    );
    // Invalid down_bits for prefill down.
    assert!(
        gguf_prefill_indirect_down(&x, &tiles, &buf, 0, 1, 256, 8)
            .unwrap()
            .is_none()
    );
}

#[test]
fn gguf_prefill_strided_x_and_kq_sorted_combine_f16_tail() {
    let (n_experts, k, m, n_tok, top) = (3usize, 512usize, 256usize, 8usize, 4usize);
    let mut rng = Rng::new(4101);
    let raw_g = gen_raw_q4k(&mut rng, n_experts, m, k);
    let raw_u = gen_raw_q4k(&mut rng, n_experts, m, k);
    let g_buf = make_padded_u8_buf(&raw_g, 32, false);
    let u_buf = make_padded_u8_buf(&raw_u, 96, false);

    // Strided slice matching `z = down_shared_gu.slice([0, 0], [n, k])`.
    let x_big_vals: Vec<f32> = (0..n_tok * 3 * k)
        .map(|_| rng.next_f32(-4.0, 4.0))
        .collect();
    let x_big = MxArray::from_float32(&x_big_vals, &[n_tok as i64, (3 * k) as i64]).unwrap();
    let x_strided = x_big.slice(&[0, 0], &[n_tok as i64, k as i64]).unwrap();
    let x_contig = MxArray::from_float32(&eval_f32(&x_strided), &[n_tok as i64, k as i64]).unwrap();

    let pairs = n_tok * top;
    let ids: Vec<u32> = (0..pairs)
        .map(|i| ((i * 5 + 1) % n_experts) as u32)
        .collect();
    let mut order: Vec<usize> = (0..pairs).collect();
    order.sort_by_key(|&i| ids[i]);
    let mut inverse = vec![0u32; pairs];
    for (pos, &orig) in order.iter().enumerate() {
        inverse[orig] = pos as u32;
    }
    let token_rows: Vec<u32> = order.iter().map(|&i| (i / top) as u32).collect();
    let sorted_slots: Vec<u32> = order.iter().map(|&i| ids[i]).collect();
    let tiles = build_prefill_tiles(&sorted_slots);
    let tr = MxArray::from_uint32(&token_rows, &[pairs as i64]).unwrap();
    let tl = MxArray::from_uint32(&tiles, &[(tiles.len() / 3) as i64, 3]).unwrap();

    let Some(gu_strided) = gguf_prefill_indirect_gu(
        &x_strided, &tr, &tl, &g_buf, 32, &u_buf, 96, n_experts, m, 7.0,
    )
    .unwrap() else {
        assert!(!require_metal(), "Metal/NAX required");
        return;
    };
    let gu_contig = gguf_prefill_indirect_gu(
        &x_contig, &tr, &tl, &g_buf, 32, &u_buf, 96, n_experts, m, 7.0,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        eval_f32(&gu_strided)
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        eval_f32(&gu_contig)
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        "gguf_prefill_indirect_gu strided vs contiguous x mismatch"
    );

    // kq_sorted_combine with F16 input matches F32 input on both VEC4 (H=256) and tail (H=259).
    let weights_vals: Vec<f32> = (0..pairs).map(|_| rng.next_f32(0.05, 0.4)).collect();
    let weights = MxArray::from_float32(&weights_vals, &[n_tok as i64, top as i64]).unwrap();
    let inv = MxArray::from_uint32(&inverse, &[n_tok as i64, top as i64]).unwrap();
    for h in [256usize, 259usize] {
        let down_f32_src = MxArray::from_float32(
            &(0..pairs * h)
                .map(|_| rng.next_f32(-3.0, 3.0))
                .collect::<Vec<_>>(),
            &[pairs as i64, h as i64],
        )
        .unwrap();
        let down_f16 = down_f32_src.astype(DType::Float16).unwrap();
        let down_f32 = down_f16.astype(DType::Float32).unwrap();
        let out_f16 = call_kq_sorted_combine(&down_f16, &weights, &inv);
        let out_f32 = call_kq_sorted_combine(&down_f32, &weights, &inv);
        assert_eq!(out_f16.dtype().unwrap(), DType::Float32);
        let (max_abs, rel_l2) = error_stats(&eval_f32(&out_f16), &eval_f32(&out_f32));
        assert!(
            max_abs <= 1e-6 && rel_l2 <= 1e-6,
            "kq_sorted_combine F16 vs F32 H={h}: max_abs={max_abs:e} rel_l2={rel_l2:e}"
        );
    }
}
