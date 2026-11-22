use super::*;
use crate::array::alloc::{DstPtrs, OutLayout};
use crate::models::hy_v4_flash::bank::ProjArrays;
use crate::models::hy_v4_flash::kq_slot;
use crate::models::hy_v4_flash::moe::{build_prefill_tiles, build_prefill_tiles_bm};
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

fn require_nax() -> bool {
    std::env::var("MLX_TEST_REQUIRE_NAX").as_deref() == Ok("1")
        || (require_metal() && unsafe { mlx_sys::mlx_metal_is_nax_available() })
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

/// Verify unchunked M=32 (n_assign=512, TG_SIZE=512, PROD_CAP=512, 10-bit packing for assign >= 256)
/// matches kq_routed_decode and is bit-exact row-invariant against M=1 for both Q4_K and Q6_K.
#[test]
fn gguf_routed_decode_m32_512_assign_dedup_and_row_invariance() {
    for down_bits in [4, 6] {
        let (n_experts, k, m, h, rows, top) =
            (48usize, 512usize, 256usize, 259usize, 32usize, 16usize);
        let mut rng = Rng::new(9100 + down_bits as u64);
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
        let g_buf = make_padded_u8_buf(&raw_g, 32, true);
        let u_buf = make_padded_u8_buf(&raw_u, 96, true);
        let d_buf = make_padded_u8_buf(&raw_d, down_off, true);
        let g_kq = repack_to_proj_arrays(&raw_g, GgufTensorType::Q4K, n_experts, m, k);
        let u_kq = repack_to_proj_arrays(&raw_u, GgufTensorType::Q4K, n_experts, m, k);
        let d_kq = repack_to_proj_arrays(&raw_d, down_ty, n_experts, h, m);

        let x_vals: Vec<f32> = (0..rows * k).map(|_| rng.next_f32(-4.0, 4.0)).collect();
        let w_vals: Vec<f32> = (0..rows * top).map(|_| rng.next_f32(0.05, 0.35)).collect();
        let mut slot_ids = Vec::with_capacity(rows * top);
        for r in 0..rows {
            for t in 0..top {
                let e = if r == 25 && t == 15 {
                    n_experts as u32 // OOB slot at assign > 256
                } else if t < 3 {
                    t as u32 // high-frequency shared (>4 chunk splitting across assign 0..511)
                } else if t < 6 {
                    (3 + (r / 4) * 3 + (t - 3)) as u32
                } else {
                    ((12 + r * 5 + t * 3) % n_experts) as u32
                };
                slot_ids.push(e);
            }
        }

        let xm = MxArray::from_float32(&x_vals, &[rows as i64, k as i64]).unwrap();
        let slm = MxArray::from_uint32(&slot_ids, &[rows as i64, top as i64]).unwrap();
        let wm = MxArray::from_float32(&w_vals, &[rows as i64, top as i64]).unwrap();
        let Some(ref_out) =
            call_kq_routed_decode(&xm, &slm, &wm, &g_kq, &u_kq, &d_kq, down_bits, 7.0)
        else {
            assert!(!require_metal(), "Metal required");
            return;
        };
        let got_out = gguf_routed_decode(
            &xm, &slm, &wm, &g_buf, 32, &u_buf, 96, &d_buf, down_off, n_experts, m, h, down_bits,
            7.0,
        )
        .unwrap()
        .unwrap();
        let ref_v = eval_f32(&ref_out);
        let got_v = eval_f32(&got_out);
        let (max_abs, rel_l2) = error_stats(&got_v, &ref_v);
        assert!(
            rel_l2 <= 1e-5,
            "m32 512-assign dedup parity bits={down_bits}: max_abs={max_abs:e} rel_l2={rel_l2:e}"
        );

        // Bit-exact row invariance on sampled rows across assign < 256 and assign >= 256.
        for r in [0usize, 15, 16, 25, 31] {
            let xr = MxArray::from_float32(&x_vals[r * k..(r + 1) * k], &[1, k as i64]).unwrap();
            let slr =
                MxArray::from_uint32(&slot_ids[r * top..(r + 1) * top], &[1, top as i64]).unwrap();
            let wr =
                MxArray::from_float32(&w_vals[r * top..(r + 1) * top], &[1, top as i64]).unwrap();
            let one = gguf_routed_decode(
                &xr, &slr, &wr, &g_buf, 32, &u_buf, 96, &d_buf, down_off, n_experts, m, h,
                down_bits, 7.0,
            )
            .unwrap()
            .unwrap();
            let one_bits: Vec<u32> = eval_f32(&one).into_iter().map(f32::to_bits).collect();
            let batch_bits: Vec<u32> = got_v[r * h..(r + 1) * h]
                .iter()
                .map(|v| v.to_bits())
                .collect();
            assert_eq!(
                batch_bits, one_bits,
                "m32 512-assign row invariance bits={down_bits} row={r}"
            );
        }
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
            !require_nax(),
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
        assert!(!require_nax(), "Metal/NAX required");
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
            assert!(!require_nax(), "Metal/NAX required");
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

    if let Some(gu_strided) = gguf_prefill_indirect_gu(
        &x_strided, &tr, &tl, &g_buf, 32, &u_buf, 96, n_experts, m, 7.0,
    )
    .unwrap()
    {
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
    } else {
        assert!(!require_nax(), "Metal/NAX required");
    }

    // kq_sorted_combine with F16 input matches F32 input on both VEC4 (H=256) and tail (H=259).
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        assert!(!require_metal(), "Metal required");
        return;
    }
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

#[test]
fn moe_prefill_sort_tiles_single_and_multi_tg_bm32_and_bm64() {
    let cases: &[(usize, usize, usize, bool)] = &[
        (16, 4, 17, false),     // R=64 (single-TG)
        (128, 16, 64, true),    // R=2048 (single-TG boundary + OOB ids)
        (129, 16, 64, false),   // R=2064 (multi-TG boundary)
        (256, 16, 128, true),   // R=4096 (multi-TG + OOB ids)
        (4096, 16, 512, false), // R=65536 (32-TG full 4K prefill scale)
    ];

    for &(rows, k, n_experts, with_oob) in cases {
        let mut rng = Rng::new(7700 + (rows * 31 + n_experts) as u64);
        let r = rows * k;
        let mut ids = Vec::with_capacity(r);
        for i in 0..r {
            if with_oob && i % 97 == 13 {
                ids.push(n_experts as u32 + (i as u32 % 5));
            } else if i % 11 == 0 {
                // Force hot experts spanning >64 and >128 rows.
                ids.push((i % 3) as u32);
            } else {
                ids.push((rng.next_u64() as usize % n_experts) as u32);
            }
        }
        let ids_arr = MxArray::from_uint32(&ids, &[rows as i64, k as i64]).unwrap();

        for &tile_m in &[32usize, 64usize] {
            let Some((tr_arr, tl_arr, inv_arr, sl_arr)) =
                moe_prefill_sort_tiles(&ids_arr, n_experts, tile_m).unwrap()
            else {
                assert!(!require_metal(), "Metal required");
                return;
            };
            let tr = tr_arr.to_uint32().unwrap().to_vec();
            let tl = tl_arr.to_uint32().unwrap().to_vec();
            let inv = inv_arr.to_uint32().unwrap().to_vec();
            let sl = sl_arr.to_uint32().unwrap().to_vec();

            assert_eq!(tr.len(), r);
            assert_eq!(inv.len(), r);
            assert_eq!(sl.len(), r);
            let max_tiles = r.min(r.div_ceil(tile_m) + n_experts);
            assert_eq!(tl.len(), max_tiles * 3);

            let valid_count = ids.iter().filter(|&&e| (e as usize) < n_experts).count();
            let mut seen = vec![false; valid_count];
            for (i, &e) in ids.iter().enumerate() {
                let pos = inv[i] as usize;
                if (e as usize) < n_experts {
                    assert!(
                        pos < valid_count,
                        "rows={rows} tile_m={tile_m}: pos {pos} >= {valid_count}"
                    );
                    assert!(
                        !seen[pos],
                        "rows={rows} tile_m={tile_m}: duplicate pos {pos}"
                    );
                    seen[pos] = true;
                    assert_eq!(tr[pos], (i / k) as u32);
                    assert_eq!(sl[pos], e);
                } else {
                    assert_eq!(pos, r, "OOB id must map to R");
                }
            }
            assert!(seen.iter().all(|&s| s));
            assert!(
                sl[..valid_count].windows(2).all(|w| w[0] <= w[1]),
                "sorted_slots not non-decreasing"
            );
            for pos in valid_count..r {
                assert_eq!(tr[pos], 0);
                assert_eq!(sl[pos], 0);
            }

            let expected_tiles = build_prefill_tiles_bm(&sl[..valid_count], tile_m);
            assert!(expected_tiles.len() <= tl.len());
            assert_eq!(
                &tl[..expected_tiles.len()],
                expected_tiles.as_slice(),
                "tiles mismatch rows={rows} tile_m={tile_m}"
            );
            assert!(
                tl[expected_tiles.len()..].iter().all(|&v| v == 0),
                "trailing tiles not zeroed rows={rows} tile_m={tile_m}"
            );
        }
    }
}

#[test]
fn gguf_prefill_bm64_multi_tile_and_m32_unchunked_dedup_parity() {
    // Exercise BM=64 runs spanning 1, 17, 32, 33, 63, 64, 65, and 130 rows per expert.
    let run_lens = [1usize, 17, 32, 33, 63, 64, 65, 130];
    let n_experts = run_lens.len();
    let pairs: usize = run_lens.iter().sum();
    let (k, m, h, n_tok) = (512usize, 256usize, 256usize, 64usize);

    for down_bits in [4, 6] {
        let mut rng = Rng::new(8200 + down_bits as u64);
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
        let g_buf = make_padded_u8_buf(&raw_g, 32, true);
        let u_buf = make_padded_u8_buf(&raw_u, 96, true);
        let d_buf = make_padded_u8_buf(&raw_d, down_off, true);
        let g_kq = repack_to_proj_arrays(&raw_g, GgufTensorType::Q4K, n_experts, m, k);
        let u_kq = repack_to_proj_arrays(&raw_u, GgufTensorType::Q4K, n_experts, m, k);
        let d_kq = repack_to_proj_arrays(&raw_d, down_ty, n_experts, h, m);

        let x_vals: Vec<f32> = (0..n_tok * k).map(|_| rng.next_f32(-5.0, 5.0)).collect();
        let x_arr = MxArray::from_float32(&x_vals, &[n_tok as i64, k as i64]).unwrap();

        let mut sorted_slots = Vec::with_capacity(pairs);
        let mut token_rows = Vec::with_capacity(pairs);
        for (e, &len) in run_lens.iter().enumerate() {
            for j in 0..len {
                sorted_slots.push(e as u32);
                token_rows.push(((e * 13 + j * 5) % n_tok) as u32);
            }
        }
        let tiles64 = build_prefill_tiles_bm(&sorted_slots, 64);
        let tr_arr = MxArray::from_uint32(&token_rows, &[pairs as i64]).unwrap();
        let tl_arr = MxArray::from_uint32(&tiles64, &[(tiles64.len() / 3) as i64, 3]).unwrap();
        let slots_2d = MxArray::from_uint32(&sorted_slots, &[pairs as i64, 1]).unwrap();

        if let Some(got_gu) = gguf_prefill_indirect_gu(
            &x_arr, &tr_arr, &tl_arr, &g_buf, 32, &u_buf, 96, n_experts, m, 7.0,
        )
        .unwrap()
        {
            let got_dn = gguf_prefill_indirect_down(
                &got_gu, &tl_arr, &d_buf, down_off, n_experts, h, down_bits,
            )
            .unwrap()
            .unwrap();

            let x_sorted = x_arr
                .take(&tr_arr, 0)
                .unwrap()
                .reshape(&[pairs as i64, 1, 1, k as i64])
                .unwrap();
            let ref_g = g_kq
                .gather(&x_sorted, &slots_2d)
                .unwrap()
                .astype(DType::Float32)
                .unwrap()
                .reshape(&[pairs as i64, m as i64])
                .unwrap();
            let ref_u = u_kq
                .gather(&x_sorted, &slots_2d)
                .unwrap()
                .astype(DType::Float32)
                .unwrap()
                .reshape(&[pairs as i64, m as i64])
                .unwrap();
            let ref_gu =
                crate::nn::Activations::swiglu_clamped_compiled(&ref_g, &ref_u, 7.0).unwrap();
            let ref_dn = d_kq
                .gather(
                    &ref_gu.reshape(&[pairs as i64, 1, 1, m as i64]).unwrap(),
                    &slots_2d,
                )
                .unwrap()
                .astype(DType::Float32)
                .unwrap()
                .reshape(&[pairs as i64, h as i64])
                .unwrap();

            let (gu_max, gu_rel) = error_stats(&eval_f32(&got_gu), &eval_f32(&ref_gu));
            let (dn_max, dn_rel) = error_stats(&eval_f32(&got_dn), &eval_f32(&ref_dn));
            assert!(
                gu_rel <= 2e-3,
                "BM=64 GU bits={down_bits}: max_abs={gu_max:e} rel_l2={gu_rel:e}"
            );
            assert!(
                dn_rel <= 2.5e-3,
                "BM=64 DOWN bits={down_bits}: max_abs={dn_max:e} rel_l2={dn_rel:e}"
            );
        } else {
            assert!(!require_nax(), "Metal/NAX required");
        }

        // Single-pass unchunked M=32, top=16 (n_assign=512) dedup decode parity vs M=1 rows.
        let (rows, top) = (32usize, 16usize);
        let mut sids = Vec::with_capacity(rows * top);
        let mut ws = Vec::with_capacity(rows * top);
        for r in 0..rows {
            for t in 0..top {
                sids.push(((r * 3 + t) % n_experts) as u32);
                ws.push(rng.next_f32(0.02, 0.12));
            }
        }
        let x32 = x_arr.slice(&[0, 0], &[rows as i64, k as i64]).unwrap();
        let s32 = MxArray::from_uint32(&sids, &[rows as i64, top as i64]).unwrap();
        let w32 = MxArray::from_float32(&ws, &[rows as i64, top as i64]).unwrap();
        let Some(got_dec) = gguf_routed_decode(
            &x32, &s32, &w32, &g_buf, 32, &u_buf, 96, &d_buf, down_off, n_experts, m, h, down_bits,
            7.0,
        )
        .unwrap() else {
            assert!(!require_metal(), "Metal required");
            return;
        };
        let got_dec_v = eval_f32(&got_dec);
        for r in [0usize, 15, 31] {
            let xr = x32
                .slice(&[r as i64, 0], &[(r + 1) as i64, k as i64])
                .unwrap();
            let sr = s32
                .slice(&[r as i64, 0], &[(r + 1) as i64, top as i64])
                .unwrap();
            let wr = w32
                .slice(&[r as i64, 0], &[(r + 1) as i64, top as i64])
                .unwrap();
            let one = eval_f32(
                &gguf_routed_decode(
                    &xr, &sr, &wr, &g_buf, 32, &u_buf, 96, &d_buf, down_off, n_experts, m, h,
                    down_bits, 7.0,
                )
                .unwrap()
                .unwrap(),
            );
            let (mabs, rl2) = error_stats(&got_dec_v[r * h..(r + 1) * h], &one);
            assert!(
                mabs <= 1e-4 && rl2 <= 1e-4,
                "M=32 dedup(512) row={r} bits={down_bits}: max_abs={mabs:e} rel_l2={rl2:e}"
            );
        }
    }
}

/// Exact CPU f32 dequantization of a single raw GGUF `block_q4_K` (144 bytes -> 256 f32).
fn cpu_dequant_q4k_block(blk: &[u8]) -> [f32; 256] {
    let d = half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
    let dmin = half::f16::from_bits(u16::from_le_bytes([blk[2], blk[3]])).to_f32();
    let sc_b = &blk[4..16];
    let qs = &blk[16..144];
    let mut out = [0.0f32; 256];
    for c in 0..4 {
        let g0 = 2 * c;
        let g1 = g0 + 1;
        let group_sm = |g: usize| -> (f32, f32) {
            let (sc, m) = if g < 4 {
                (sc_b[g] & 63, sc_b[g + 4] & 63)
            } else {
                let q4 = sc_b[g + 4];
                (
                    (q4 & 0x0F) | ((sc_b[g - 4] >> 6) << 4),
                    (q4 >> 4) | ((sc_b[g] >> 6) << 4),
                )
            };
            (d * f32::from(sc), -dmin * f32::from(m))
        };
        let (d0, m0) = group_sm(g0);
        let (d1, m1) = group_sm(g1);
        for l in 0..32 {
            let byte = qs[c * 32 + l];
            out[c * 64 + l] = f32::from(byte & 0x0F).mul_add(d0, m0);
            out[c * 64 + 32 + l] = f32::from(byte >> 4).mul_add(d1, m1);
        }
    }
    out
}

/// Exact CPU f32 dequantization of a single raw GGUF `block_q6_K` (210 bytes -> 256 f32).
fn cpu_dequant_q6k_block(blk: &[u8]) -> [f32; 256] {
    let d = half::f16::from_bits(u16::from_le_bytes([blk[208], blk[209]])).to_f32();
    let mut out = [0.0f32; 256];
    for ip in 0..2 {
        let ql = &blk[ip * 64..ip * 64 + 64];
        let qh = &blk[128 + ip * 32..128 + ip * 32 + 32];
        let sc = &blk[192 + ip * 8..192 + ip * 8 + 8];
        for l in 0..32 {
            let is = l / 16;
            let q1 = i32::from((ql[l] & 0x0F) | ((qh[l] & 0x03) << 4)) - 32;
            let q2 = i32::from((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4)) - 32;
            let q3 = i32::from((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) - 32;
            let q4 = i32::from((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) - 32;
            let s0 = d * f32::from(sc[is] as i8);
            let s1 = d * f32::from(sc[is + 2] as i8);
            let s2 = d * f32::from(sc[is + 4] as i8);
            let s3 = d * f32::from(sc[is + 6] as i8);
            out[ip * 128 + l] = (q1 as f32) * s0;
            out[ip * 128 + 32 + l] = (q2 as f32) * s1;
            out[ip * 128 + 64 + l] = (q3 as f32) * s2;
            out[ip * 128 + 96 + l] = (q4 as f32) * s3;
        }
    }
    out
}

/// Regression test for FP16 early-narrowing NaNs in `gguf_prefill_indirect_gu`
/// and `gguf_prefill_indirect_down`:
/// - Q4_K finite cancellation (`d = dmin = 2048.0, sc = 63, m = 63, q = 1 -> 0.0`)
///   must stay finite (`0.0`, never `+inf - inf = NaN`) in both `gu` and `down`.
/// - Q6_K zero-code block with large scale (`d = 2048.0, sc = 64, q = 0 -> 0.0`)
///   must stay finite (`0.0`, never `0 * +inf = NaN`) in `down`.
#[test]
fn gguf_prefill_indirect_no_fp16_early_narrowing_nan() {
    crate::models::hy_v4_flash::exact_f32();
    let mut rng = Rng::new(20261123);

    let d_2048 = half::f16::from_f32(2048.0).to_bits().to_le_bytes();
    let mut q4k_cancel_blk = [0u8; 144];
    q4k_cancel_blk[0..2].copy_from_slice(&d_2048);
    q4k_cancel_blk[2..4].copy_from_slice(&d_2048);
    // All 12 scale bytes = 0xFF => sc = 63 and m = 63 for every group g=0..8.
    q4k_cancel_blk[4..16].fill(0xFF);
    // Every nibble = 1 => 1 * (2048*63) - (2048*63) = 0.0 in f32.
    q4k_cancel_blk[16..144].fill(0x11);
    assert!(
        cpu_dequant_q4k_block(&q4k_cancel_blk)
            .iter()
            .all(|&v| v == 0.0)
    );

    let mut q6k_zero_blk = [0u8; 210];
    // ql = 0, qh = 0xAA (every 2-bit field = 0b10) => raw 6-bit code = 32 => signed q = 0.
    q6k_zero_blk[0..128].fill(0x00);
    q6k_zero_blk[128..192].fill(0xAA);
    q6k_zero_blk[192..208].fill(64u8);
    q6k_zero_blk[208..210].copy_from_slice(&d_2048);
    assert!(
        cpu_dequant_q6k_block(&q6k_zero_blk)
            .iter()
            .all(|&v| v == 0.0)
    );

    let (n_experts, m, h, n_tok) = (2usize, 64usize, 64usize, 16usize);
    let pairs = n_tok * 2;
    let mut sorted_slots = Vec::with_capacity(pairs);
    let mut token_rows_v = Vec::with_capacity(pairs);
    for e in 0..n_experts {
        for j in 0..n_tok {
            sorted_slots.push(e as u32);
            token_rows_v.push(((e * 7 + j) % n_tok) as u32);
        }
    }
    let tiles_v = build_prefill_tiles_bm(&sorted_slots, 64);
    let tr_arr = MxArray::from_uint32(&token_rows_v, &[pairs as i64]).unwrap();
    let tl_arr = MxArray::from_uint32(&tiles_v, &[(tiles_v.len() / 3) as i64, 3]).unwrap();

    // 1. Q4_K finite cancellation (both pure-cancellation K=256 and mixed K=512).
    for &k in &[256usize, 512usize] {
        let nb_per_row = k / 256;
        let build_q4k = |rng: &mut Rng, rows: usize| -> (Vec<u8>, Vec<f32>) {
            let mut raw = vec![0u8; n_experts * rows * nb_per_row * 144];
            let mut ref_w = vec![0.0f32; n_experts * rows * k];
            let normal = gen_raw_q4k(rng, n_experts, rows, 256);
            for er in 0..n_experts * rows {
                for b in 0..nb_per_row {
                    let dst =
                        &mut raw[(er * nb_per_row + b) * 144..(er * nb_per_row + b + 1) * 144];
                    if b == 0 {
                        dst.copy_from_slice(&q4k_cancel_blk);
                    } else {
                        dst.copy_from_slice(&normal[er * 144..(er + 1) * 144]);
                    }
                    let dec = cpu_dequant_q4k_block(dst);
                    ref_w[er * k + b * 256..er * k + (b + 1) * 256].copy_from_slice(&dec);
                }
            }
            (raw, ref_w)
        };

        let (raw_g, ref_g) = build_q4k(&mut rng, m);
        let (raw_u, ref_u) = build_q4k(&mut rng, m);
        let g_buf = make_padded_u8_buf(&raw_g, 32, false);
        let u_buf = make_padded_u8_buf(&raw_u, 64, false);
        let x_vec: Vec<f32> = (0..n_tok * k).map(|_| rng.next_f32(-1.5, 1.5)).collect();
        let x_arr = MxArray::from_float32(&x_vec, &[n_tok as i64, k as i64]).unwrap();

        let Some(got_gu_arr) = gguf_prefill_indirect_gu(
            &x_arr, &tr_arr, &tl_arr, &g_buf, 32, &u_buf, 64, n_experts, m, 7.0,
        )
        .unwrap() else {
            assert!(!require_nax(), "Metal/NAX required");
            return;
        };
        let got_gu = eval_f32(&got_gu_arr);
        assert!(
            got_gu.iter().all(|v| v.is_finite()),
            "Q4_K gu K={k}: produced non-finite value"
        );

        let mut want_gu = vec![0.0f32; pairs * m];
        for tile in tiles_v.as_chunks::<3>().0 {
            let (begin, end, slot) = (tile[0] as usize, tile[1] as usize, tile[2] as usize);
            for p in begin..end {
                let tok = token_rows_v[p] as usize;
                let xr = &x_vec[tok * k..(tok + 1) * k];
                for o in 0..m {
                    let gw = &ref_g[(slot * m + o) * k..(slot * m + o + 1) * k];
                    let uw = &ref_u[(slot * m + o) * k..(slot * m + o + 1) * k];
                    let mut gv: f32 = xr.iter().zip(gw).map(|(a, b)| a * b).sum();
                    let mut uv: f32 = xr.iter().zip(uw).map(|(a, b)| a * b).sum();
                    gv = gv.min(7.0);
                    uv = uv.clamp(-7.0, 7.0);
                    want_gu[p * m + o] = (gv / (1.0 + (-gv).exp())) * uv;
                }
            }
        }
        let (max_abs, rel_l2) = error_stats(&got_gu, &want_gu);
        if k == 256 {
            assert_eq!(max_abs, 0.0, "Q4_K pure cancellation gu must be exact 0.0");
        } else {
            assert!(
                rel_l2 <= 5e-3,
                "Q4_K mixed cancellation gu K={k}: max_abs={max_abs:e} rel_l2={rel_l2:e}"
            );
        }

        // Now test Q4_K down with the same finite cancellation pattern.
        let (raw_d4, ref_d4) = build_q4k(&mut rng, h);
        let d4_buf = make_padded_u8_buf(&raw_d4, 48, false);
        let hid_vec: Vec<f32> = (0..pairs * k).map(|_| rng.next_f32(-1.5, 1.5)).collect();
        let hid_arr = MxArray::from_float32(&hid_vec, &[pairs as i64, k as i64]).unwrap();
        let got_d4_arr =
            gguf_prefill_indirect_down(&hid_arr, &tl_arr, &d4_buf, 48, n_experts, h, 4)
                .unwrap()
                .expect("gguf_prefill_indirect_down Q4_K must run when gu ran");
        let got_d4 = eval_f32(&got_d4_arr);
        assert!(
            got_d4.iter().all(|v| v.is_finite()),
            "Q4_K down K={k}: produced non-finite value"
        );
        let mut want_d4 = vec![0.0f32; pairs * h];
        for tile in tiles_v.as_chunks::<3>().0 {
            let (begin, end, slot) = (tile[0] as usize, tile[1] as usize, tile[2] as usize);
            for p in begin..end {
                let hr = &hid_vec[p * k..(p + 1) * k];
                for o in 0..h {
                    let dw = &ref_d4[(slot * h + o) * k..(slot * h + o + 1) * k];
                    want_d4[p * h + o] = hr.iter().zip(dw).map(|(a, b)| a * b).sum();
                }
            }
        }
        let (max_abs, rel_l2) = error_stats(&got_d4, &want_d4);
        if k == 256 {
            assert_eq!(
                max_abs, 0.0,
                "Q4_K pure cancellation down must be exact 0.0"
            );
        } else {
            assert!(
                rel_l2 <= 5e-3,
                "Q4_K mixed cancellation down K={k}: max_abs={max_abs:e} rel_l2={rel_l2:e}"
            );
        }

        // 2. Q6_K zero-code block with large scale (d=2048, sc=64, q=0 -> 0.0).
        let mut raw_d6 = vec![0u8; n_experts * h * nb_per_row * 210];
        let mut ref_d6 = vec![0.0f32; n_experts * h * k];
        let normal_q6 = gen_raw_q6k(&mut rng, n_experts, h, 256);
        for er in 0..n_experts * h {
            for b in 0..nb_per_row {
                let dst = &mut raw_d6[(er * nb_per_row + b) * 210..(er * nb_per_row + b + 1) * 210];
                if b == 0 {
                    dst.copy_from_slice(&q6k_zero_blk);
                } else {
                    dst.copy_from_slice(&normal_q6[er * 210..(er + 1) * 210]);
                }
                let dec = cpu_dequant_q6k_block(dst);
                ref_d6[er * k + b * 256..er * k + (b + 1) * 256].copy_from_slice(&dec);
            }
        }
        // Strictly 2-byte aligned offset (66 % 4 == 2) to also exercise alignment safety.
        let d6_buf = make_padded_u8_buf(&raw_d6, 66, false);
        let got_d6_arr =
            gguf_prefill_indirect_down(&hid_arr, &tl_arr, &d6_buf, 66, n_experts, h, 6)
                .unwrap()
                .expect("gguf_prefill_indirect_down Q6_K must run when gu ran");
        let got_d6 = eval_f32(&got_d6_arr);
        assert!(
            got_d6.iter().all(|v| v.is_finite()),
            "Q6_K down K={k}: produced non-finite value"
        );
        let mut want_d6 = vec![0.0f32; pairs * h];
        for tile in tiles_v.as_chunks::<3>().0 {
            let (begin, end, slot) = (tile[0] as usize, tile[1] as usize, tile[2] as usize);
            for p in begin..end {
                let hr = &hid_vec[p * k..(p + 1) * k];
                for o in 0..h {
                    let dw = &ref_d6[(slot * h + o) * k..(slot * h + o + 1) * k];
                    want_d6[p * h + o] = hr.iter().zip(dw).map(|(a, b)| a * b).sum();
                }
            }
        }
        let (max_abs, rel_l2) = error_stats(&got_d6, &want_d6);
        if k == 256 {
            assert_eq!(
                max_abs, 0.0,
                "Q6_K zero-code large-scale down must be exact 0.0"
            );
        } else {
            assert!(
                rel_l2 <= 5e-3,
                "Q6_K mixed zero-code down K={k}: max_abs={max_abs:e} rel_l2={rel_l2:e}"
            );
        }
    }
}

#[test]
fn kq_prefill_indirect_down_m3_q4k_q6k_tail_and_oob_parity() {
    crate::models::hy_v4_flash::exact_f32();
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        assert!(!require_metal(), "Metal required");
        return;
    }

    for (down_bits, n_experts, m, h, r) in [
        (4, 4usize, 512usize, 256usize, 53usize),
        (6, 4, 512, 256, 53),
        (4, 4, 512, 260, 53),
        (6, 4, 512, 260, 53),
        (6, 1, 256, 1, 5), // bd.size() == 1 < 8 exercises constant address space
    ] {
        let mut rng = Rng::new(9300 + (down_bits as u64) * 97 + h as u64);
        let (ty, raw_d) = if down_bits == 6 {
            (GgufTensorType::Q6K, gen_raw_q6k(&mut rng, n_experts, h, m))
        } else {
            (GgufTensorType::Q4K, gen_raw_q4k(&mut rng, n_experts, h, m))
        };
        let d_kq = repack_to_proj_arrays(&raw_d, ty, n_experts, h, m);

        let hid_vals: Vec<f32> = (0..r * m).map(|_| rng.next_f32(-3.0, 3.0)).collect();
        let hid_f32 = MxArray::from_float32(&hid_vals, &[r as i64, m as i64]).unwrap();
        let hid_f16 = hid_f32.astype(DType::Float16).unwrap();

        let mut sorted_slots = Vec::with_capacity(r);
        for i in 0..r {
            sorted_slots.push(((i * n_experts) / r) as u32);
        }
        let tiles_v = build_prefill_tiles(&sorted_slots);
        let tl = MxArray::from_uint32(&tiles_v, &[(tiles_v.len() / 3) as i64, 3]).unwrap();
        let slots_2d = MxArray::from_uint32(&sorted_slots, &[r as i64, 1]).unwrap();

        let ref_dn = d_kq
            .gather(
                &hid_f32.reshape(&[r as i64, 1, 1, m as i64]).unwrap(),
                &slots_2d,
            )
            .unwrap()
            .astype(DType::Float32)
            .unwrap()
            .reshape(&[r as i64, h as i64])
            .unwrap();
        let ref_v = eval_f32(&ref_dn);

        for (label, hid) in [("f32", &hid_f32), ("f16", &hid_f16)] {
            let got = kq_prefill_indirect_down(hid, &tl, &d_kq.w, &d_kq.s, &d_kq.b, down_bits)
                .unwrap()
                .expect("kq_prefill_indirect_down must run on Metal");
            assert_eq!(got.dtype().unwrap(), DType::Float32);
            let (max_abs, rel_l2) = error_stats(&eval_f32(&got), &ref_v);
            assert!(
                rel_l2 <= 2e-3,
                "kq_prefill_indirect_down bits={down_bits} H={h} in={label}: max_abs={max_abs:e} rel_l2={rel_l2:e}"
            );
        }

        // OOB slot must write exact zeros.
        let oob_tl = MxArray::from_uint32(&[0, 4, 99], &[1, 3]).unwrap();
        let h4 = hid_f32.slice(&[0, 0], &[4, m as i64]).unwrap();
        let oob = kq_prefill_indirect_down(&h4, &oob_tl, &d_kq.w, &d_kq.s, &d_kq.b, down_bits)
            .unwrap()
            .unwrap();
        assert!(
            eval_f32(&oob).iter().all(|&v| v == 0.0),
            "kq_prefill_indirect_down OOB slot not zero (bits={down_bits} H={h})"
        );
    }
}

#[test]
#[ignore = "GPU microbenchmark for Hy4 decode BW kernels (M=1,2,4)"]
fn bench_decode_bw_kernels() {
    use crate::models::hy_v4_flash::attention::grouped_mm_t_mgh;
    use crate::models::hy_v4_flash::moe::qlinear;
    use crate::models::hy_v4_flash::weights::quantize_sym_q8_array;
    use std::time::Instant;

    crate::models::hy_v4_flash::exact_f32();
    let mut rng = Rng::new(20261122);

    fn time_median_us(iters: usize, warmup_ms: f64, mut f: impl FnMut()) -> f64 {
        let w0 = Instant::now();
        let mut w_cnt = 0usize;
        while w_cnt < 6 || w0.elapsed().as_secs_f64() * 1000.0 < warmup_ms {
            f();
            w_cnt += 1;
        }
        let mut ts = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            f();
            ts.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ts[ts.len() / 2]
    }

    let zero_f32 = MxArray::from_float32(&[0.0f32], &[1, 1]).unwrap();
    MxArray::eval_arrays_with_context(&[&zero_f32], "bench zero").unwrap();

    // 0. Pure DRAM read BW ceiling check (742 MB sum, also ramps GPU DVFS to peak clock).
    {
        let n_elems = 185_597_952usize; // 742.4 MB in f32
        let buf = MxArray::ones(&[n_elems as i64], Some(DType::Float32)).unwrap();
        MxArray::eval_arrays_with_context(&[&buf], "bench dram init").unwrap();
        let med_us = time_median_us(15, 250.0, || {
            let s = buf.sum(None, None).unwrap();
            MxArray::eval_arrays_with_context(&[&s], "bench dram sum").unwrap();
        });
        let gbs = ((n_elems * 4) as f64 / 1e9) / (med_us / 1e6);
        eprintln!("BENCH dram_sum_ceiling[742MB]: {med_us:.1} us ({gbs:.1} GB/s)");
    }

    // 1a. q8_dense_decode: output [3072, 16384] x 4 cold copies (214 MB > SLC) and down_shared_gu [4608, 3072] x 14 copies
    for &(n, k, n_copies, tag) in &[
        (3072usize, 16384usize, 4usize, "o_proj[3072x16384]"),
        (4608usize, 3072usize, 14usize, "down_sgu[4608x3072]"),
    ] {
        let bytes_per = (n * k + (n * k) / 16) as f64;
        let base_vals: Vec<f32> = (0..n * k)
            .map(|i| (((i * 17) % 97) as f32 - 48.0) * 0.002)
            .collect();
        let base_w = MxArray::from_float32(&base_vals, &[n as i64, k as i64]).unwrap();
        let base_sym = quantize_sym_q8_array(&base_w).unwrap();
        let ws: Vec<Dense> = (0..n_copies)
            .map(|_| {
                let Dense::SymQ8 { w, scales } = &base_sym else {
                    unreachable!();
                };
                Dense::SymQ8 {
                    w: w.deep_copy().unwrap(),
                    scales: scales.deep_copy().unwrap(),
                }
            })
            .collect();
        for w in &ws {
            if let Dense::SymQ8 { w, scales } = w {
                MxArray::eval_arrays_with_context(&[w, scales], "bench q8_dense init").unwrap();
            }
        }
        for &m in &[1usize, 2, 4] {
            let xv: Vec<f32> = (0..m * k).map(|_| rng.next_f32(-1.0, 1.0)).collect();
            let x = MxArray::from_float32(&xv, &[m as i64, k as i64]).unwrap();
            MxArray::eval_arrays_with_context(&[&x], "bench x").unwrap();
            // Serial chained (1 dispatch active at a time, matching real layer-by-layer decode)
            let dummy = MxArray::from_float32(&[1.0f32], &[1, 1]).unwrap();
            MxArray::eval_arrays_with_context(&[&dummy], "bench dummy").unwrap();
            let dep_us = time_median_us(15, 100.0, || {
                let mut cur = x.clone();
                for _ in 0..n_copies {
                    let z = dummy.mul(&zero_f32).unwrap();
                    cur = cur.add(&z).unwrap();
                }
                MxArray::eval_arrays_with_context(&[&cur], "bench dep").unwrap();
            });
            let ser_us = time_median_us(15, 150.0, || {
                let mut cur = x.clone();
                let mut last = x.clone();
                for w in &ws {
                    last = qlinear(&cur, w).unwrap();
                    let dep = last
                        .slice(&[0, 0], &[1, 1])
                        .unwrap()
                        .mul(&zero_f32)
                        .unwrap();
                    cur = cur.add(&dep).unwrap();
                }
                MxArray::eval_arrays_with_context(&[&last, &cur], "bench q8_dense ser").unwrap();
            });
            let per_us = (ser_us - dep_us).max(1.0) / n_copies as f64;
            let gbs = (bytes_per / 1e9) / (per_us / 1e6);
            eprintln!("BENCH q8_dense {tag} M={m}: {per_us:.1} us/call ({gbs:.1} GB/s)");
        }
    }

    // 1b. q8_grouped_short k512 (v_b): G=64, N=256, K=512 x 20 cold copies (178 MB > SLC)
    {
        let (g, n, k, n_copies) = (64usize, 256usize, 512usize, 20usize);
        let bytes_per = (g * n * k + (g * n * k) / 16) as f64;
        let base_vals: Vec<f32> = (0..g * n * k)
            .map(|i| (((i * 13) % 89) as f32 - 44.0) * 0.005)
            .collect();
        let base_w = MxArray::from_float32(&base_vals, &[(g * n) as i64, k as i64]).unwrap();
        let base_sym = quantize_sym_q8_array(&base_w).unwrap();
        let ws: Vec<Dense> = (0..n_copies)
            .map(|_| {
                let Dense::SymQ8 { w, scales } = &base_sym else {
                    unreachable!();
                };
                Dense::SymQ8 {
                    w: w.deep_copy().unwrap(),
                    scales: scales.deep_copy().unwrap(),
                }
            })
            .collect();
        for w in &ws {
            if let Dense::SymQ8 { w, scales } = w {
                MxArray::eval_arrays_with_context(&[w, scales], "bench k512 init").unwrap();
            }
        }
        for &m in &[1usize, 2, 4] {
            let xv: Vec<f32> = (0..m * g * k).map(|_| rng.next_f32(-1.0, 1.0)).collect();
            let x = MxArray::from_float32(&xv, &[m as i64, g as i64, k as i64]).unwrap();
            MxArray::eval_arrays_with_context(&[&x], "bench x").unwrap();
            let dummy = MxArray::from_float32(&[1.0f32], &[1, 1, 1]).unwrap();
            let zero3 = MxArray::from_float32(&[0.0f32], &[1, 1, 1]).unwrap();
            MxArray::eval_arrays_with_context(&[&dummy, &zero3], "bench dummy3").unwrap();
            let dep_us = time_median_us(15, 100.0, || {
                let mut cur = x.clone();
                for _ in 0..n_copies {
                    let z = dummy.mul(&zero3).unwrap();
                    cur = cur.add(&z).unwrap();
                }
                MxArray::eval_arrays_with_context(&[&cur], "bench dep3").unwrap();
            });
            let ser_us = time_median_us(15, 150.0, || {
                let mut cur = x.clone();
                let mut last = x.clone();
                for w in &ws {
                    last = grouped_mm_t_mgh(w, &cur, g as i64).unwrap();
                    let dep = last
                        .slice(&[0, 0, 0], &[1, 1, 1])
                        .unwrap()
                        .mul(&zero3)
                        .unwrap();
                    cur = cur.add(&dep).unwrap();
                }
                MxArray::eval_arrays_with_context(&[&last, &cur], "bench k512 ser").unwrap();
            });
            let per_us = (ser_us - dep_us).max(1.0) / n_copies as f64;
            let gbs = (bytes_per / 1e9) / (per_us / 1e6);
            eprintln!(
                "BENCH q8_grouped_k512[64x256x512] M={m}: {per_us:.1} us/call ({gbs:.1} GB/s)"
            );
        }
    }

    // 2. gguf_expert_decode: K=1536, M=1536, H=1536, TOP=16, Q6_K down, 128 experts (589 MB > SLC),
    // serially chained `x = out` across rotating pools to match real layer-by-layer execution.
    {
        let (n_experts, k, m_ff, h, top) = (128usize, 1536usize, 1536usize, 1536usize, 16usize);
        let raw_g = gen_raw_q4k(&mut rng, n_experts, m_ff, k);
        let raw_u = gen_raw_q4k(&mut rng, n_experts, m_ff, k);
        let raw_d = gen_raw_q6k(&mut rng, n_experts, h, m_ff);
        let g_buf = make_padded_u8_buf(&raw_g, 32, false);
        let u_buf = make_padded_u8_buf(&raw_u, 96, false);
        let d_buf = make_padded_u8_buf(&raw_d, 66, false);
        MxArray::eval_arrays_with_context(&[&g_buf, &u_buf, &d_buf], "bench gguf init").unwrap();
        let bytes_per_exp = ((m_ff * (k / 256) * 144) * 2 + h * (m_ff / 256) * 210) as f64;

        for &rows in &[1usize, 2, 4, 32] {
            let xv: Vec<f32> = (0..rows * k).map(|_| rng.next_f32(-0.5, 0.5)).collect();
            let wv: Vec<f32> = (0..rows * top).map(|_| 1.0 / top as f32).collect();
            let n_rot = if rows <= 2 {
                4usize
            } else if rows == 4 {
                2usize
            } else {
                1usize
            };
            let mut slot_arrs = Vec::with_capacity(n_rot);
            let mut uniq_total = 0usize;
            for rot in 0..n_rot {
                let shift = rot * 32;
                let mut sids = Vec::with_capacity(rows * top);
                let mut uniq_set = std::collections::BTreeSet::new();
                for r in 0..rows {
                    let mut row_ids = Vec::with_capacity(top);
                    for t in 0..top {
                        let base_e = if t < 5 {
                            t * 2
                        } else if t < 9 {
                            10 + ((r / 2) * 4 + (t - 5)) % 20
                        } else {
                            30 + (r * 7 + t) % 66
                        };
                        let e = ((base_e + shift) % n_experts) as u32;
                        row_ids.push(e);
                        uniq_set.insert(e);
                    }
                    row_ids.sort_unstable();
                    sids.extend_from_slice(&row_ids);
                }
                uniq_total += uniq_set.len();
                slot_arrs.push(MxArray::from_uint32(&sids, &[rows as i64, top as i64]).unwrap());
            }
            let x = MxArray::from_float32(&xv, &[rows as i64, k as i64]).unwrap();
            let weights = MxArray::from_float32(&wv, &[rows as i64, top as i64]).unwrap();
            let mut eval_in: Vec<&MxArray> = vec![&x, &weights];
            eval_in.extend(slot_arrs.iter());
            MxArray::eval_arrays_with_context(&eval_in, "bench gguf in").unwrap();

            let med_us = time_median_us(12, 150.0, || {
                let mut cur = x.clone();
                for slots in &slot_arrs {
                    cur = gguf_routed_decode(
                        &cur, slots, &weights, &g_buf, 32, &u_buf, 96, &d_buf, 66, n_experts, m_ff,
                        h, 6, 10.0,
                    )
                    .unwrap()
                    .unwrap();
                }
                MxArray::eval_arrays_with_context(&[&cur], "bench gguf").unwrap();
            });
            let per_us = med_us / n_rot as f64;
            let avg_uniq = uniq_total as f64 / n_rot as f64;
            let gbs = (avg_uniq * bytes_per_exp / 1e9) / (per_us / 1e6);
            eprintln!(
                "BENCH gguf_routed_decode[Q6K] M={rows} (uniq={avg_uniq:.0}): {per_us:.1} us/call ({gbs:.1} GB/s, 41x={:.2} ms)",
                per_us * 41.0 / 1000.0
            );
        }
    }

    // 3. router_mv: [M, 3072] x [512, 3072]T F32 x 32 cold copies (201 MB > SLC), serial chained
    {
        let (e, k, n_copies) = (512usize, 3072usize, 32usize);
        let bytes_per = (e * k * 4) as f64;
        let base_v: Vec<f32> = (0..e * k)
            .map(|i| (((i * 23) % 97) as f32 - 48.0) * 0.005)
            .collect();
        let base_w = MxArray::from_float32(&base_v, &[e as i64, k as i64]).unwrap();
        let ws: Vec<Dense> = (0..n_copies)
            .map(|_| Dense::Array(base_w.deep_copy().unwrap()))
            .collect();
        for w in &ws {
            if let Dense::Array(a) = w {
                MxArray::eval_arrays_with_context(&[a], "bench router init").unwrap();
            }
        }
        let dummy = MxArray::from_float32(&[1.0f32], &[1, 1]).unwrap();
        MxArray::eval_arrays_with_context(&[&dummy], "bench dummy").unwrap();
        for &m in &[1usize, 2, 4] {
            let xv: Vec<f32> = (0..m * k).map(|_| rng.next_f32(-1.0, 1.0)).collect();
            let x = MxArray::from_float32(&xv, &[m as i64, k as i64]).unwrap();
            MxArray::eval_arrays_with_context(&[&x], "bench x").unwrap();
            let dep_us = time_median_us(15, 100.0, || {
                let mut cur = x.clone();
                for _ in 0..n_copies {
                    let z = dummy.mul(&zero_f32).unwrap();
                    cur = cur.add(&z).unwrap();
                }
                MxArray::eval_arrays_with_context(&[&cur], "bench router dep").unwrap();
            });
            let ser_us = time_median_us(15, 150.0, || {
                let mut cur = x.clone();
                let mut last = x.clone();
                for w in &ws {
                    last = try_router_mv(&cur, w).unwrap().unwrap();
                    let dep = last
                        .slice(&[0, 0], &[1, 1])
                        .unwrap()
                        .mul(&zero_f32)
                        .unwrap();
                    cur = cur.add(&dep).unwrap();
                }
                MxArray::eval_arrays_with_context(&[&last, &cur], "bench router ser").unwrap();
            });
            let per_us = (ser_us - dep_us).max(1.0) / n_copies as f64;
            let gbs = (bytes_per / 1e9) / (per_us / 1e6);
            eprintln!(
                "BENCH router_mv[512x3072 F32] M={m}: {per_us:.1} us/call ({gbs:.1} GB/s, 41x={:.2} ms)",
                per_us * 41.0 / 1000.0
            );
        }
    }

    // 4. q8_dual_down: [M, 1536] x 2x[3072, 1536] SymQ8 x 20 cold copies (200 MB > SLC),
    // zero-overhead serial chain via strided slice `cur[:, :1536]` + `cur`.
    {
        let (n, k, n_copies) = (3072usize, 1536usize, 20usize);
        let bytes_per = (2 * (n * k + (n * k) / 16)) as f64;
        let base_v: Vec<f32> = (0..n * k)
            .map(|i| (((i * 19) % 89) as f32 - 44.0) * 0.0005)
            .collect();
        let base_w = MxArray::from_float32(&base_v, &[n as i64, k as i64]).unwrap();
        let base_sym = quantize_sym_q8_array(&base_w).unwrap();
        let pairs: Vec<(Dense, Dense)> = (0..n_copies)
            .map(|_| {
                let Dense::SymQ8 { w, scales } = &base_sym else {
                    unreachable!();
                };
                (
                    Dense::SymQ8 {
                        w: w.deep_copy().unwrap(),
                        scales: scales.deep_copy().unwrap(),
                    },
                    Dense::SymQ8 {
                        w: w.deep_copy().unwrap(),
                        scales: scales.deep_copy().unwrap(),
                    },
                )
            })
            .collect();
        for (a, b) in &pairs {
            if let (Dense::SymQ8 { w: wa, scales: sa }, Dense::SymQ8 { w: wb, scales: sb }) = (a, b)
            {
                MxArray::eval_arrays_with_context(&[wa, sa, wb, sb], "bench dual init").unwrap();
            }
        }
        for &m in &[1usize, 2, 4] {
            let sguv: Vec<f32> = (0..m * 2 * k).map(|_| rng.next_f32(-0.5, 0.5)).collect();
            let sgu = MxArray::from_float32(&sguv, &[m as i64, (2 * k) as i64]).unwrap();
            MxArray::eval_arrays_with_context(&[&sgu], "bench dual in").unwrap();
            let med_us = time_median_us(15, 150.0, || {
                let mut cur = sgu.clone();
                for (wu, ws) in &pairs {
                    let rl = cur.slice(&[0, 0], &[m as i64, k as i64]).unwrap();
                    cur = try_q8_dual_down_add(&rl, &cur, wu, ws, 0.0)
                        .unwrap()
                        .unwrap();
                }
                MxArray::eval_arrays_with_context(&[&cur], "bench dual ser").unwrap();
            });
            let per_us = med_us / n_copies as f64;
            let gbs = (bytes_per / 1e9) / (per_us / 1e6);
            eprintln!(
                "BENCH q8_dual_down[3072x1536] M={m}: {per_us:.1} us/call ({gbs:.1} GB/s, 41x={:.2} ms)",
                per_us * 41.0 / 1000.0
            );
        }
    }

    // 5. bf16_lm_head: [M, 3072] x [120832, 3072]T BF16 (742.4 MB > SLC)
    {
        let (v, k) = (120832usize, 3072usize);
        let bytes_per = (v * k * 2) as f64;
        let w_bits = vec![0x3C00u16; v * k];
        let w_arr = MxArray::from_bfloat16(&w_bits, &[v as i64, k as i64]).unwrap();
        MxArray::eval_arrays_with_context(&[&w_arr], "bench lm_head init").unwrap();
        let head = Dense::Array(w_arr);
        for &m in &[1usize, 2, 4] {
            let xv: Vec<f32> = (0..m * k).map(|_| rng.next_f32(-1.0, 1.0)).collect();
            let x = MxArray::from_float32(&xv, &[m as i64, k as i64]).unwrap();
            MxArray::eval_arrays_with_context(&[&x], "bench x").unwrap();
            let med_us = time_median_us(15, 250.0, || {
                let out = try_bf16_lm_head(&x, &head).unwrap().unwrap();
                MxArray::eval_arrays_with_context(&[&out], "bench lm_head").unwrap();
            });
            let gbs = (bytes_per / 1e9) / (med_us / 1e6);
            eprintln!("BENCH bf16_lm_head[120832x3072] M={m}: {med_us:.1} us/call ({gbs:.1} GB/s)");
        }
    }
}

#[test]
#[ignore = "GPU microbenchmark for Metal 3 Hy4 prefill kernels"]
fn bench_m3_prefill_kernels() {
    use crate::models::hy_v4_flash::attention::grouped_mm_t_mgh;
    use crate::models::hy_v4_flash::moe::qlinear;
    use crate::models::hy_v4_flash::weights::quantize_sym_q8_array;
    use std::time::Instant;

    crate::models::hy_v4_flash::exact_f32();
    let mut rng = Rng::new(20261124);

    fn time_median_ms(iters: usize, warmup: usize, mut f: impl FnMut()) -> f64 {
        for _ in 0..warmup {
            f();
        }
        let mut ts = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            f();
            ts.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ts[ts.len() / 2]
    }

    // 1. q8_dense_prefill (dense)
    for &(m, k, n) in &[
        (4096usize, 2560usize, 2560usize),
        (4096, 2560, 3072),
        (4096, 1536, 4096),
    ] {
        let wv: Vec<f32> = (0..n * k)
            .map(|i| (((i * 19) % 89) as f32 - 44.0) * 0.001)
            .collect();
        let w = MxArray::from_float32(&wv, &[n as i64, k as i64]).unwrap();
        let sym = quantize_sym_q8_array(&w).unwrap();
        let xv: Vec<f32> = (0..m * k)
            .map(|i| (((i * 13) % 71) as f32 - 35.0) * 0.02)
            .collect();
        let x = MxArray::from_float32(&xv, &[m as i64, k as i64]).unwrap();
        MxArray::eval_arrays_with_context(&[&x], "bench q8 dense in").unwrap();
        let ms = time_median_ms(7, 3, || {
            let y = qlinear(&x, &sym).unwrap();
            MxArray::eval_arrays_with_context(&[&y], "bench q8 dense").unwrap();
        });
        let tflops = (2.0 * m as f64 * k as f64 * n as f64) / (ms * 1e9);
        eprintln!("BENCH q8_dense[M={m},K={k},N={n}]: {ms:.3} ms ({tflops:.2} TFLOPS)");
    }

    // 2. q8_grouped_prefill
    for &(m, g, k, n) in &[
        (4096usize, 64usize, 128usize, 512usize),
        (4096, 64, 512, 128),
    ] {
        let wv: Vec<f32> = (0..g * n * k)
            .map(|i| (((i * 23) % 97) as f32 - 48.0) * 0.001)
            .collect();
        let w = MxArray::from_float32(&wv, &[(g * n) as i64, k as i64]).unwrap();
        let sym = quantize_sym_q8_array(&w).unwrap();
        let xv: Vec<f32> = (0..m * g * k)
            .map(|i| (((i * 17) % 83) as f32 - 41.0) * 0.02)
            .collect();
        let x = MxArray::from_float32(&xv, &[m as i64, g as i64, k as i64]).unwrap();
        MxArray::eval_arrays_with_context(&[&x], "bench q8 grouped in").unwrap();
        let ms = time_median_ms(7, 3, || {
            let y = grouped_mm_t_mgh(&sym, &x, g as i64).unwrap();
            MxArray::eval_arrays_with_context(&[&y], "bench q8 grouped").unwrap();
        });
        let tflops = (2.0 * m as f64 * g as f64 * k as f64 * n as f64) / (ms * 1e9);
        eprintln!("BENCH q8_grouped[M={m},G={g},K={k},N={n}]: {ms:.3} ms ({tflops:.2} TFLOPS)");
    }

    // 3. q8_dual_down_prefill
    {
        let (m, k, n) = (4096usize, 1024usize, 2560usize);
        let wv: Vec<f32> = (0..n * k)
            .map(|i| (((i * 29) % 97) as f32 - 48.0) * 0.001)
            .collect();
        let w_up =
            quantize_sym_q8_array(&MxArray::from_float32(&wv, &[n as i64, k as i64]).unwrap())
                .unwrap();
        let w_sd =
            quantize_sym_q8_array(&MxArray::from_float32(&wv, &[n as i64, k as i64]).unwrap())
                .unwrap();
        let rlv: Vec<f32> = (0..m * k)
            .map(|i| (((i * 11) % 61) as f32 - 30.0) * 0.02)
            .collect();
        let sguv: Vec<f32> = (0..m * 2 * k)
            .map(|i| (((i * 13) % 67) as f32 - 33.0) * 0.02)
            .collect();
        let rl = MxArray::from_float32(&rlv, &[m as i64, k as i64]).unwrap();
        let sgu = MxArray::from_float32(&sguv, &[m as i64, (2 * k) as i64]).unwrap();
        MxArray::eval_arrays_with_context(&[&rl, &sgu], "bench dual in").unwrap();
        let ms = time_median_ms(7, 3, || {
            let y = try_q8_dual_down_add(&rl, &sgu, &w_up, &w_sd, 0.0)
                .unwrap()
                .unwrap();
            MxArray::eval_arrays_with_context(&[&y], "bench dual").unwrap();
        });
        let tflops = (4.0 * m as f64 * k as f64 * n as f64) / (ms * 1e9);
        eprintln!("BENCH q8_dual_down[M={m},K={k},N={n}]: {ms:.3} ms ({tflops:.2} TFLOPS)");
    }

    // 4. gguf_prefill_indirect_gu & gguf_prefill_indirect_down (Q4_K & Q6_K)
    // R=24576, 512 experts x 48 rows/tile, K=2560, M=1024, H=2560
    {
        let (n_experts, rows_per_exp, k, m_ff, h) =
            (512usize, 48usize, 2560usize, 1024usize, 2560usize);
        let r = n_experts * rows_per_exp;
        let n_tok = r / 16;
        let mut sorted_slots = Vec::with_capacity(r);
        let mut token_rows_v = Vec::with_capacity(r);
        for e in 0..n_experts {
            for j in 0..rows_per_exp {
                sorted_slots.push(e as u32);
                token_rows_v.push(((e * 7 + j * 13) % n_tok) as u32);
            }
        }
        let tiles64 = build_prefill_tiles_bm(&sorted_slots, 64);
        let tr = MxArray::from_uint32(&token_rows_v, &[r as i64]).unwrap();
        let tl64 = MxArray::from_uint32(&tiles64, &[(tiles64.len() / 3) as i64, 3]).unwrap();
        let xv: Vec<f32> = (0..n_tok * k)
            .map(|i| (((i * 17) % 73) as f32 - 36.0) * 0.02)
            .collect();
        let x = MxArray::from_float32(&xv, &[n_tok as i64, k as i64]).unwrap();

        let one_q4_gu = gen_raw_q4k(&mut rng, 1, m_ff, k);
        let mut raw_g = Vec::with_capacity(n_experts * one_q4_gu.len());
        for _ in 0..n_experts {
            raw_g.extend_from_slice(&one_q4_gu);
        }
        let g_buf = make_padded_u8_buf(&raw_g, 64, false);
        let u_buf = make_padded_u8_buf(&raw_g, 64, false);
        MxArray::eval_arrays_with_context(&[&x, &tr, &tl64, &g_buf, &u_buf], "bench gguf gu in")
            .unwrap();

        let gu_ms = time_median_ms(5, 2, || {
            let y = gguf_prefill_indirect_gu(
                &x, &tr, &tl64, &g_buf, 64, &u_buf, 64, n_experts, m_ff, 7.0,
            )
            .unwrap()
            .unwrap();
            MxArray::eval_arrays_with_context(&[&y], "bench gguf gu").unwrap();
        });
        let gu_tflops = (4.0 * r as f64 * k as f64 * m_ff as f64) / (gu_ms * 1e9);
        eprintln!(
            "BENCH gguf_indirect_gu[R={r},E={n_experts},K={k},M={m_ff}]: {gu_ms:.3} ms ({gu_tflops:.2} TFLOPS)"
        );

        let hid =
            gguf_prefill_indirect_gu(&x, &tr, &tl64, &g_buf, 64, &u_buf, 64, n_experts, m_ff, 7.0)
                .unwrap()
                .unwrap();
        MxArray::eval_arrays_with_context(&[&hid], "bench gguf hid").unwrap();

        for down_bits in [4, 6] {
            let one_d = if down_bits == 6 {
                gen_raw_q6k(&mut rng, 1, h, m_ff)
            } else {
                gen_raw_q4k(&mut rng, 1, h, m_ff)
            };
            let mut raw_d = Vec::with_capacity(n_experts * one_d.len());
            for _ in 0..n_experts {
                raw_d.extend_from_slice(&one_d);
            }
            let off = if down_bits == 6 { 66u64 } else { 64u64 };
            let d_buf = make_padded_u8_buf(&raw_d, off, false);
            MxArray::eval_arrays_with_context(&[&d_buf], "bench gguf dn in").unwrap();
            let dn_ms = time_median_ms(5, 2, || {
                let y =
                    gguf_prefill_indirect_down(&hid, &tl64, &d_buf, off, n_experts, h, down_bits)
                        .unwrap()
                        .unwrap();
                MxArray::eval_arrays_with_context(&[&y], "bench gguf dn").unwrap();
            });
            let dn_tflops = (2.0 * r as f64 * m_ff as f64 * h as f64) / (dn_ms * 1e9);
            eprintln!(
                "BENCH gguf_indirect_down[Q{down_bits}K,R={r},E={n_experts},M={m_ff},H={h}]: {dn_ms:.3} ms ({dn_tflops:.2} TFLOPS)"
            );
        }
    }

    // 5. kq_prefill_indirect_gu & kq_prefill_indirect_down (Q4_K & Q6_K)
    {
        let (n_experts, rows_per_exp, k, m_ff, h) =
            (128usize, 192usize, 1536usize, 1536usize, 1536usize);
        let r = n_experts * rows_per_exp;
        let n_tok = r / 16;
        let mut sorted_slots = Vec::with_capacity(r);
        let mut token_rows_v = Vec::with_capacity(r);
        for e in 0..n_experts {
            for j in 0..rows_per_exp {
                sorted_slots.push(e as u32);
                token_rows_v.push(((e * 7 + j * 13) % n_tok) as u32);
            }
        }
        let tiles32 = build_prefill_tiles(&sorted_slots);
        let tr = MxArray::from_uint32(&token_rows_v, &[r as i64]).unwrap();
        let tl32 = MxArray::from_uint32(&tiles32, &[(tiles32.len() / 3) as i64, 3]).unwrap();
        let xv: Vec<f32> = (0..n_tok * k)
            .map(|i| (((i * 19) % 79) as f32 - 39.0) * 0.02)
            .collect();
        let x = MxArray::from_float32(&xv, &[n_tok as i64, k as i64]).unwrap();

        let one_q4 = gen_raw_q4k(&mut rng, 1, m_ff, k);
        let mut raw_q4 = Vec::with_capacity(n_experts * one_q4.len());
        for _ in 0..n_experts {
            raw_q4.extend_from_slice(&one_q4);
        }
        let g_kq = repack_to_proj_arrays(&raw_q4, GgufTensorType::Q4K, n_experts, m_ff, k);
        let u_kq = repack_to_proj_arrays(&raw_q4, GgufTensorType::Q4K, n_experts, m_ff, k);
        MxArray::eval_arrays_with_context(&[&x, &tr, &tl32, &g_kq.w, &u_kq.w], "bench kq gu in")
            .unwrap();

        let gu_ms = time_median_ms(5, 2, || {
            let y = call_kq_prefill_indirect_gu(&x, &tr, &tl32, &g_kq, &u_kq, 7.0).unwrap();
            MxArray::eval_arrays_with_context(&[&y], "bench kq gu").unwrap();
        });
        let gu_tflops = (4.0 * r as f64 * k as f64 * m_ff as f64) / (gu_ms * 1e9);
        eprintln!(
            "BENCH kq_indirect_gu[R={r},E={n_experts},K={k},M={m_ff}]: {gu_ms:.3} ms ({gu_tflops:.2} TFLOPS)"
        );

        let hid = call_kq_prefill_indirect_gu(&x, &tr, &tl32, &g_kq, &u_kq, 7.0).unwrap();
        MxArray::eval_arrays_with_context(&[&hid], "bench kq hid").unwrap();

        for down_bits in [4, 6] {
            let (ty, one_d) = if down_bits == 6 {
                (GgufTensorType::Q6K, gen_raw_q6k(&mut rng, 1, h, m_ff))
            } else {
                (GgufTensorType::Q4K, gen_raw_q4k(&mut rng, 1, h, m_ff))
            };
            let mut raw_d = Vec::with_capacity(n_experts * one_d.len());
            for _ in 0..n_experts {
                raw_d.extend_from_slice(&one_d);
            }
            let d_kq = repack_to_proj_arrays(&raw_d, ty, n_experts, h, m_ff);
            MxArray::eval_arrays_with_context(&[&d_kq.w, &d_kq.s, &d_kq.b], "bench kq dn in")
                .unwrap();
            let dn_ms = time_median_ms(5, 2, || {
                let y = kq_prefill_indirect_down(&hid, &tl32, &d_kq.w, &d_kq.s, &d_kq.b, down_bits)
                    .unwrap()
                    .unwrap();
                MxArray::eval_arrays_with_context(&[&y], "bench kq dn").unwrap();
            });
            let dn_tflops = (2.0 * r as f64 * m_ff as f64 * h as f64) / (dn_ms * 1e9);
            eprintln!(
                "BENCH kq_indirect_down[Q{down_bits}K,R={r},E={n_experts},M={m_ff},H={h}]: {dn_ms:.3} ms ({dn_tflops:.2} TFLOPS)"
            );
        }
    }
}
