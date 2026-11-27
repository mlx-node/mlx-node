use super::*;
use crate::array::alloc::{DstPtrs, OutLayout};
use crate::models::hy_v4_flash::bank::ProjArrays;
use crate::models::hy_v4_flash::kq_slot;
use crate::models::hy_v4_flash::moe::{build_prefill_tiles, build_prefill_tiles_bm};
use crate::models::hy_v4_flash::test_support::{
    Rng, arr, error_stats, kquant_bytes, read_bits, read_f32, read_u32,
};
use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};
use crate::utils::gguf_kquant::KQuantFormat;

const G_OFF: u64 = 32;
const U_OFF: u64 = 96;

fn u32_arr(v: &[u32], s: &[usize]) -> MxArray {
    let sh: Vec<i64> = s.iter().map(|&d| d as i64).collect();
    MxArray::from_uint32(v, &sh).unwrap()
}

fn fmt_ty(bits: i32) -> (GgufTensorType, KQuantFormat) {
    if bits == 6 {
        (GgufTensorType::Q6K, KQuantFormat::Q6K)
    } else {
        (GgufTensorType::Q4K, KQuantFormat::Q4K)
    }
}

fn gen_raw(rng: &mut Rng, ne: usize, rows: usize, k: usize, fmt: KQuantFormat) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let mut out = kquant_bytes(rng, ne * rows * (k / 256), fmt);
    for blk in out.chunks_exact_mut(fmt.block_bytes()) {
        match fmt {
            KQuantFormat::Q4K => {
                let d = half::f16::from_f32(rng.f32_range(0.0008, 0.004)).to_bits();
                let dmin = half::f16::from_f32(rng.f32_range(0.0004, 0.0025)).to_bits();
                blk[0..2].copy_from_slice(&d.to_le_bytes());
                blk[2..4].copy_from_slice(&dmin.to_le_bytes());
            }
            KQuantFormat::Q6K => {
                let s = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
                let d = half::f16::from_f32(s * rng.f32_range(0.0005, 0.003)).to_bits();
                blk[208..210].copy_from_slice(&d.to_le_bytes());
            }
            f => panic!("unsupported {f:?}"),
        }
    }
    out
}

fn make_padded_u8_buf(raw: &[u8], base_off: u64, pages_2d: bool) -> MxArray {
    let off = base_off as usize;
    let min_len = off + raw.len() + 64;
    let total = if pages_2d {
        min_len.div_ceil(16384) * 16384
    } else {
        min_len
    };
    let mut buf = vec![0xA5u8; total];
    buf[off..off + raw.len()].copy_from_slice(raw);
    let shape: &[i64] = if pages_2d {
        &[(total as i64) / 16384, 16384]
    } else {
        &[total as i64]
    };
    MxArray::from_uint8(&buf, shape).unwrap()
}

fn repack_to_proj_arrays(
    raw: &[u8],
    ty: GgufTensorType,
    ne: usize,
    rows: usize,
    k: usize,
) -> ProjArrays {
    let format = ty.k_quant_format().unwrap();
    let total_rows = ne * rows;
    let s_dt = if format.scales_are_signed() {
        DType::Int8
    } else {
        DType::Uint8
    };
    let (ne_i, r_i) = (ne as i64, rows as i64);
    let (w, wp, _) =
        MxArray::alloc_uninit(&[ne_i, r_i, format.weight_cols(k) as i64], DType::Uint32).unwrap();
    let (s, sp, _) =
        MxArray::alloc_uninit(&[ne_i, r_i, format.scales_cols(k) as i64], s_dt).unwrap();
    let (b, bp, _) =
        MxArray::alloc_uninit(&[ne_i, r_i, format.biases_cols(k) as i64], DType::Float16).unwrap();
    let info = GgufTensorInfo {
        name: "test.weight".into(),
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

fn assert_rel(got: &MxArray, want: &MxArray, tol: f32, tag: &str) {
    let (mabs, rel) = error_stats(&read_f32(got), &read_f32(want));
    assert!(
        rel <= tol,
        "{tag}: max_abs={mabs:e} rel_l2={rel:e} > {tol:e}"
    );
}

fn kq_sorted_combine(down: &MxArray, weights: &MxArray, inv: &MxArray) -> MxArray {
    let h = unsafe {
        mlx_sys::mlx_hy4_kq_sorted_combine(
            down.as_raw_ptr(),
            weights.as_raw_ptr(),
            inv.as_raw_ptr(),
        )
    };
    assert!(!h.is_null(), "kq_sorted_combine returned null");
    MxArray::from_handle(h, "test kq_sorted_combine").unwrap()
}

struct GgufExpertsRig {
    ne: usize,
    k: usize,
    m: usize,
    h: usize,
    bits: i32,
    d_off: u64,
    g_buf: MxArray,
    u_buf: MxArray,
    d_buf: MxArray,
    g_kq: ProjArrays,
    u_kq: ProjArrays,
    d_kq: ProjArrays,
}

impl GgufExpertsRig {
    fn new(rng: &mut Rng, bits: i32, ne: usize, k: usize, m: usize, h: usize, pages: bool) -> Self {
        let (raw_g, raw_u) = (
            gen_raw(rng, ne, m, k, KQuantFormat::Q4K),
            gen_raw(rng, ne, m, k, KQuantFormat::Q4K),
        );
        // Odd 2-byte offset (66 = 2 mod 4) stresses 2-byte-only Q6_K alignment.
        let (d_ty, d_fmt) = fmt_ty(bits);
        let d_off = if bits == 6 { 66u64 } else { 64u64 };
        let raw_d = gen_raw(rng, ne, h, m, d_fmt);
        Self {
            ne,
            k,
            m,
            h,
            bits,
            d_off,
            g_buf: make_padded_u8_buf(&raw_g, G_OFF, pages),
            u_buf: make_padded_u8_buf(&raw_u, U_OFF, pages),
            d_buf: make_padded_u8_buf(&raw_d, d_off, pages),
            g_kq: repack_to_proj_arrays(&raw_g, GgufTensorType::Q4K, ne, m, k),
            u_kq: repack_to_proj_arrays(&raw_u, GgufTensorType::Q4K, ne, m, k),
            d_kq: repack_to_proj_arrays(&raw_d, d_ty, ne, h, m),
        }
    }

    fn decode_gguf(&self, x: &MxArray, sl: &MxArray, w: &MxArray, lim: f32) -> Option<MxArray> {
        gguf_routed_decode(
            x,
            sl,
            w,
            &self.g_buf,
            G_OFF,
            &self.u_buf,
            U_OFF,
            &self.d_buf,
            self.d_off,
            self.ne,
            self.m,
            self.h,
            self.bits,
            lim,
        )
        .unwrap()
    }

    fn decode_kq(&self, x: &MxArray, sl: &MxArray, w: &MxArray, lim: f32) -> Option<MxArray> {
        let (g, u, d) = (&self.g_kq, &self.u_kq, &self.d_kq);
        let h = unsafe {
            mlx_sys::mlx_hy4_kq_routed_decode(
                x.as_raw_ptr(),
                sl.as_raw_ptr(),
                w.as_raw_ptr(),
                g.w.as_raw_ptr(),
                g.s.as_raw_ptr(),
                g.b.as_raw_ptr(),
                u.w.as_raw_ptr(),
                u.s.as_raw_ptr(),
                u.b.as_raw_ptr(),
                d.w.as_raw_ptr(),
                d.s.as_raw_ptr(),
                d.b.as_raw_ptr(),
                self.bits,
                lim,
            )
        };
        (!h.is_null()).then(|| MxArray::from_handle(h, "test kq_routed_decode").unwrap())
    }

    fn prefill_gu_gguf(
        &self,
        x: &MxArray,
        tr: &MxArray,
        tl: &MxArray,
        lim: f32,
    ) -> Option<MxArray> {
        gguf_prefill_indirect_gu(
            x,
            tr,
            tl,
            &self.g_buf,
            G_OFF,
            &self.u_buf,
            U_OFF,
            self.ne,
            self.m,
            lim,
        )
        .unwrap()
    }

    fn prefill_gu_kq(&self, x: &MxArray, tr: &MxArray, tl: &MxArray, lim: f32) -> Option<MxArray> {
        let (g, u) = (&self.g_kq, &self.u_kq);
        let h = unsafe {
            mlx_sys::mlx_hy4_kq_prefill_indirect_gu(
                x.as_raw_ptr(),
                tr.as_raw_ptr(),
                tl.as_raw_ptr(),
                g.w.as_raw_ptr(),
                g.s.as_raw_ptr(),
                g.b.as_raw_ptr(),
                u.w.as_raw_ptr(),
                u.s.as_raw_ptr(),
                u.b.as_raw_ptr(),
                lim,
            )
        };
        (!h.is_null()).then(|| MxArray::from_handle(h, "test kq_prefill_indirect_gu").unwrap())
    }

    fn prefill_down_gguf(&self, hid: &MxArray, tl: &MxArray) -> Option<MxArray> {
        gguf_prefill_indirect_down(hid, tl, &self.d_buf, self.d_off, self.ne, self.h, self.bits)
            .unwrap()
    }

    fn fp32_prefill_ref(
        &self,
        x: &MxArray,
        tr: &MxArray,
        slots: &MxArray,
        pairs: usize,
        lim: f32,
    ) -> (MxArray, MxArray) {
        let pi = pairs as i64;
        let x_s = x
            .take(tr, 0)
            .unwrap()
            .reshape(&[pi, 1, 1, self.k as i64])
            .unwrap();
        let s2 = slots.reshape(&[pi, 1]).unwrap();
        let gather = |p: &ProjArrays, inp: &MxArray, od: usize| {
            p.gather(inp, &s2)
                .unwrap()
                .astype(DType::Float32)
                .unwrap()
                .reshape(&[pi, od as i64])
                .unwrap()
        };
        let hid = crate::nn::Activations::swiglu_clamped_compiled(
            &gather(&self.g_kq, &x_s, self.m),
            &gather(&self.u_kq, &x_s, self.m),
            lim,
        )
        .unwrap();
        let dn = gather(
            &self.d_kq,
            &hid.reshape(&[pi, 1, 1, self.m as i64]).unwrap(),
            self.h,
        );
        (hid, dn)
    }

    fn nax_down_ref(&self, hid: &MxArray, st: &SortedTiles, pairs: usize) -> MxArray {
        let max_g = st.sorted_slots.last().map_or(0, |&s| s as i32 + 1);
        let h3 = hid.reshape(&[pairs as i64, 1, self.m as i64]).unwrap();
        self.d_kq
            .gather_sorted_nax_bounded(&h3, &st.slots_arr, st.n_tiles as i32, max_g)
            .unwrap()
            .astype(DType::Float32)
            .unwrap()
            .reshape(&[pairs as i64, self.h as i64])
            .unwrap()
    }

    fn check_decode(
        &self,
        xv: &[f32],
        sv: &[u32],
        wv: &[f32],
        (rows, top, lim, max_tol): (usize, usize, f32, Option<f32>),
        inv_rows: &[usize],
        tag: &str,
    ) {
        let (x, sl, w) = (
            arr(xv, &[rows, self.k]),
            u32_arr(sv, &[rows, top]),
            arr(wv, &[rows, top]),
        );
        let Some(ref_out) = self.decode_kq(&x, &sl, &w, lim) else {
            assert!(!require_metal(), "Metal required ({tag})");
            return;
        };
        let got_out = self
            .decode_gguf(&x, &sl, &w, lim)
            .expect("gguf_routed_decode None");
        let got_v = read_f32(&got_out);
        let (mabs, rel) = error_stats(&got_v, &read_f32(&ref_out));
        assert!(
            rel <= 1e-5 && max_tol.is_none_or(|t| mabs <= t),
            "{tag}: {mabs:e} / {rel:e}"
        );
        for &r in inv_rows {
            let one = self
                .decode_gguf(
                    &arr(&xv[r * self.k..(r + 1) * self.k], &[1, self.k]),
                    &u32_arr(&sv[r * top..(r + 1) * top], &[1, top]),
                    &arr(&wv[r * top..(r + 1) * top], &[1, top]),
                    lim,
                )
                .unwrap();
            let batch_b: Vec<u32> = got_v[r * self.h..(r + 1) * self.h]
                .iter()
                .map(|v| v.to_bits())
                .collect();
            assert_eq!(batch_b, read_bits(&one), "{tag} row={r} vs M=1");
        }
    }
}

struct SortedTiles {
    sorted_slots: Vec<u32>,
    n_tiles: usize,
    tr_arr: MxArray,
    tl_arr: MxArray,
    slots_arr: MxArray,
    inv_arr: MxArray,
}

fn sort_and_tile(ids: &[u32], n_tok: usize, top: usize, tile_m: usize) -> SortedTiles {
    let pairs = n_tok * top;
    let mut order: Vec<usize> = (0..pairs).collect();
    order.sort_by_key(|&i| ids[i]);
    let mut inv = vec![0u32; pairs];
    for (pos, &orig) in order.iter().enumerate() {
        inv[orig] = pos as u32;
    }
    let tr: Vec<u32> = order.iter().map(|&i| (i / top) as u32).collect();
    let sl: Vec<u32> = order.iter().map(|&i| ids[i]).collect();
    let tiles = build_prefill_tiles_bm(&sl, tile_m);
    let n_tiles = tiles.len() / 3;
    SortedTiles {
        tr_arr: u32_arr(&tr, &[pairs]),
        tl_arr: u32_arr(&tiles, &[n_tiles, 3]),
        slots_arr: u32_arr(&sl, &[pairs]),
        inv_arr: u32_arr(&inv, &[n_tok, top]),
        sorted_slots: sl,
        n_tiles,
    }
}

fn rand_weights(rng: &mut Rng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|_| rng.f32_range(lo, hi)).collect()
}

/// Covers Q4_K / Q6_K raw-GGUF decode parity (1D/2D pages, 2-byte Q6_K offset 66,
/// H tails, OOB slots), multi-row dedup + bit-exact row invariance (ROWS=1..8 and
/// unchunked M=32 512-assign), and asymmetric SwiGLU gate clamping (gate << -lim).
#[test]
fn gguf_routed_decode_parity_dedup_and_clamp() {
    let slots_r1: Vec<u32> = (0..16).map(|i| i % 4).collect();
    let slots_r2: Vec<u32> = (0..32).map(|i| (i * 3) % 4).collect();
    let mut slots_r2_mix: Vec<u32> = (0..16).chain(10..26).map(|i| i % 24).collect();
    slots_r2_mix[15] = 24;
    let mut slots_r4: Vec<u32> = (0..64)
        .map(|i| ((i / 16) * 5 + (i % 16) * 2) % 12)
        .collect();
    (slots_r4[7], slots_r4[39]) = (12, 12);

    let basic: &[(
        u64,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        &[u32],
        f32,
        bool,
    )] = &[
        (100, 6, 768, 256, 259, 1, 4, &[0, 2, 3, 5], 6.0, false),
        (
            200,
            6,
            512,
            256,
            256,
            2,
            4,
            &[1, 3, 4, 6, 1, 2, 3, 5],
            8.0,
            true,
        ),
        (
            300,
            8,
            512,
            256,
            256,
            4,
            4,
            &[0, 1, 2, 3, 1, 2, 4, 5, 0, 2, 6, 7, 1, 2, 3, 7],
            6.0,
            true,
        ),
        (400, 4, 512, 512, 512, 1, 16, &slots_r1, 4.0, true),
        (500, 4, 512, 512, 512, 2, 16, &slots_r2, 4.0, true),
        (550, 24, 512, 512, 512, 2, 16, &slots_r2_mix, 4.0, true),
        (600, 12, 512, 512, 512, 4, 16, &slots_r4, 4.0, true),
    ];

    for bits in [4, 6] {
        for &(seed, ne, k, m, h, rows, top, slots, scale, pages) in basic {
            let mut rng = Rng::new(seed + bits as u64);
            let rig = GgufExpertsRig::new(&mut rng, bits, ne, k, m, h, pages);
            let (xv, wv) = (
                rng.vec_f32(rows * k, scale),
                rand_weights(&mut rng, rows * top, 0.05, 0.35),
            );
            rig.check_decode(
                &xv,
                slots,
                &wv,
                (rows, top, 7.0, None),
                &[],
                &format!("basic b={bits} K={k} H={h} R={rows}"),
            );
        }

        // Multi-row dedup (>4 chunk splitting with counts 1..=8 + OOB) & row invariance.
        for &(ne, k, m, h, top, seed) in &[
            (24usize, 768usize, 256usize, 259usize, 8usize, 7000u64),
            (24, 512, 512, 512, 16, 8000),
        ] {
            let mut rng = Rng::new(seed + bits as u64);
            let rig = GgufExpertsRig::new(&mut rng, bits, ne, k, m, h, true);
            let (xv, wv) = (
                rng.vec_f32(8 * k, 5.0),
                rand_weights(&mut rng, 8 * top, 0.05, 0.35),
            );
            let mut sids = vec![0u32; 8 * top];
            for r in 0..8 {
                let row = &mut sids[r * top..(r + 1) * top];
                row[0] = 0;
                for (j, (cut, base)) in [(6, 10), (5, 12), (4, 14), (3, 15), (2, 16)]
                    .into_iter()
                    .enumerate()
                {
                    row[j + 1] = if r < cut {
                        j as u32 + 1
                    } else {
                        (base + r) as u32
                    };
                }
                for (j, slot) in row.iter_mut().enumerate().skip(6) {
                    *slot = ((6 + r * 3 + j) % ne) as u32;
                }
                if r == 1 || r == 5 {
                    row[top - 1] = ne as u32;
                }
            }
            for rows in [1usize, 2, 3, 4, 8] {
                let inv: Vec<usize> = (0..rows).collect();
                rig.check_decode(
                    &xv[..rows * k],
                    &sids[..rows * top],
                    &wv[..rows * top],
                    (rows, top, 7.0, None),
                    &inv,
                    &format!("dedup b={bits} H={h} R={rows}"),
                );
            }
        }

        // Unchunked M=32 (n_assign=512, 10-bit packing for assign >= 256, OOB > 256).
        let (ne, k, m, h, rows, top) = (28usize, 512usize, 256usize, 259usize, 32usize, 16usize);
        let mut rng = Rng::new(9100 + bits as u64);
        let rig = GgufExpertsRig::new(&mut rng, bits, ne, k, m, h, true);
        let (xv, wv) = (
            rng.vec_f32(rows * k, 4.0),
            rand_weights(&mut rng, rows * top, 0.05, 0.35),
        );
        let sids: Vec<u32> = (0..rows)
            .flat_map(|r| {
                (0..top).map(move |t| {
                    if r == 25 && t == 15 {
                        ne as u32
                    } else if t < 3 {
                        t as u32
                    } else if t < 6 {
                        (3 + (r / 4) * 3 + (t - 3)) as u32
                    } else {
                        ((12 + r * 5 + t * 3) % ne) as u32
                    }
                })
            })
            .collect();
        rig.check_decode(
            &xv,
            &sids,
            &wv,
            (rows, top, 7.0, None),
            &[0, 15, 16, 25, 31],
            &format!("m32 b={bits}"),
        );
    }

    // Explicit asymmetric clamp: gate upper-clamped at +lim, NEVER lower-clamped at -lim.
    let mut rng = Rng::new(777);
    let rig = GgufExpertsRig::new(&mut rng, 4, 1, 256, 256, 256, false);
    let xv: Vec<f32> = (0..256).map(|i| -12.0 - (i % 7) as f32 * 1.5).collect();
    rig.check_decode(
        &xv,
        &[0],
        &[1.0],
        (1, 1, 1.0, Some(1e-4)),
        &[],
        "asymmetric_clamp",
    );
}

#[test]
fn gguf_prefill_nax_matches_kq_prefill_q4k_and_q6k() {
    let gu_hilo = std::env::var("MLX_HY4_GU_HILO").as_deref() == Ok("1");
    let down_f32 = std::env::var("MLX_HY4_DOWN_F32").as_deref() == Ok("1");
    let (gu_tol, dn_tol) = (
        if gu_hilo { 5e-4 } else { 2e-3 },
        if gu_hilo && down_f32 { 5e-4 } else { 2.5e-3 },
    );
    let cases: &[(u64, usize, usize, usize, usize, usize, usize, f32)] = &[
        (1000, 5, 512, 256, 256, 20, 4, 6.0),
        (2000, 4, 512, 512, 512, 8, 16, 4.0),
    ];

    for bits in [4, 6] {
        for &(seed, ne, k, m, h, n_tok, top, scale) in cases {
            let mut rng = Rng::new(seed + bits as u64);
            let rig = GgufExpertsRig::new(&mut rng, bits, ne, k, m, h, true);
            let pairs = n_tok * top;
            let x = arr(&rng.vec_f32(n_tok * k, scale), &[n_tok, k]);
            let ids: Vec<u32> = (0..n_tok)
                .flat_map(|t| {
                    (0..top).map(move |j| {
                        if (t + j) % 5 == 0 {
                            0
                        } else {
                            ((t * 7 + j * 3) % ne) as u32
                        }
                    })
                })
                .collect();
            let st = sort_and_tile(&ids, n_tok, top, 32);
            let w = arr(&rand_weights(&mut rng, pairs, 0.05, 0.3), &[n_tok, top]);

            let Some(ref_hid) = rig.prefill_gu_kq(&x, &st.tr_arr, &st.tl_arr, 7.0) else {
                assert!(!require_nax(), "Metal/NAX required");
                return;
            };
            let got_hid = rig
                .prefill_gu_gguf(&x, &st.tr_arr, &st.tl_arr, 7.0)
                .expect("gu None");
            let (fp32_hid, fp32_dn) =
                rig.fp32_prefill_ref(&x, &st.tr_arr, &st.slots_arr, pairs, 7.0);
            assert_rel(
                &got_hid,
                &fp32_hid,
                gu_tol,
                &format!("GU/FP32 b={bits} K={k}"),
            );
            assert_rel(
                &got_hid,
                &ref_hid,
                2.5e-3,
                &format!("GU/TF32 b={bits} K={k}"),
            );

            let ref_dn = rig.nax_down_ref(&ref_hid, &st, pairs);
            let got_dn = rig
                .prefill_down_gguf(&got_hid, &st.tl_arr)
                .expect("down None");
            assert_eq!(
                got_dn.dtype().unwrap(),
                if down_f32 {
                    DType::Float32
                } else {
                    DType::Float16
                }
            );
            assert_rel(
                &got_dn,
                &fp32_dn,
                dn_tol,
                &format!("DN/FP32 b={bits} K={k}"),
            );
            assert_rel(&got_dn, &ref_dn, 3e-3, &format!("DN/TF32 b={bits} K={k}"));

            let got_c = kq_sorted_combine(&got_dn, &w, &st.inv_arr);
            assert_rel(
                &got_c,
                &kq_sorted_combine(&fp32_dn, &w, &st.inv_arr),
                dn_tol,
                "COMB/FP32",
            );
            assert_rel(
                &got_c,
                &kq_sorted_combine(&ref_dn, &w, &st.inv_arr),
                3e-3,
                "COMB/TF32",
            );
        }
    }
}

#[test]
fn gguf_prefill_oob_tail_invalid_and_strided() {
    // 1. Invalid args return None (runs even without Metal).
    {
        let buf = MxArray::from_uint8(&vec![0u8; 65536], &[65536]).unwrap();
        let (x, sl, w) = (
            arr(&vec![0.0; 256], &[1, 256]),
            u32_arr(&[0], &[1, 1]),
            arr(&[1.0], &[1, 1]),
        );
        let (tr, tl) = (u32_arr(&[0], &[1]), u32_arr(&[0, 1, 0], &[1, 3]));
        for (g_off, d_off, bits, lim) in [
            (2u64, 0u64, 4i32, 7.0f32),
            (0, 1, 6, 7.0),
            (0, 0, 5, 7.0),
            (0, 0, 4, f32::NAN),
        ] {
            assert!(
                gguf_routed_decode(
                    &x, &sl, &w, &buf, g_off, &buf, 0, &buf, d_off, 1, 256, 256, bits, lim
                )
                .unwrap()
                .is_none()
            );
        }
        assert!(
            gguf_prefill_indirect_gu(&x, &tr, &tl, &buf, 40000, &buf, 0, 1, 256, 7.0)
                .unwrap()
                .is_none()
        );
        assert!(
            gguf_prefill_indirect_down(&x, &tl, &buf, 0, 1, 256, 8)
                .unwrap()
                .is_none()
        );
    }

    // 2. OOB slot >= n_experts writes exact zeros.
    {
        let mut rng = Rng::new(909);
        let rig = GgufExpertsRig::new(&mut rng, 6, 2, 256, 256, 256, false);
        let x = arr(&vec![1.0f32; 4 * 256], &[4, 256]);
        let (tr, tl) = (u32_arr(&[0, 1, 2, 3], &[4]), u32_arr(&[0, 4, 99], &[1, 3]));
        if let Some(gu) = rig.prefill_gu_gguf(&x, &tr, &tl, 7.0) {
            assert!(read_f32(&gu).iter().all(|&v| v == 0.0), "OOB GU");
            assert!(
                read_f32(&rig.prefill_down_gguf(&x, &tl).unwrap())
                    .iter()
                    .all(|&v| v == 0.0),
                "OOB DN"
            );
        } else {
            assert!(!require_nax(), "Metal/NAX required");
        }
    }

    // 3. Non-multiple-of-64 H tail (H=260) matches decode & NAX ref.
    for bits in [4, 6] {
        let (ne, k, m, h, n_tok, top) = (3usize, 512usize, 256usize, 260usize, 5usize, 2usize);
        let mut rng = Rng::new(3100 + bits as u64);
        let rig = GgufExpertsRig::new(&mut rng, bits, ne, k, m, h, false);
        let x = arr(&rng.vec_f32(n_tok * k, 4.0), &[n_tok, k]);
        let ids = [0u32, 1, 0, 2, 1, 2, 0, 2, 1, 2];
        let w = arr(&rand_weights(&mut rng, 10, 0.1, 0.4), &[n_tok, top]);
        let Some(dec) = rig.decode_gguf(&x, &u32_arr(&ids, &[n_tok, top]), &w, 7.0) else {
            assert!(!require_metal(), "Metal required");
            return;
        };
        let st = sort_and_tile(&ids, n_tok, top, 32);
        if let Some(gu) = rig.prefill_gu_gguf(&x, &st.tr_arr, &st.tl_arr, 7.0) {
            let dn = rig.prefill_down_gguf(&gu, &st.tl_arr).unwrap();
            assert_rel(
                &dn,
                &rig.nax_down_ref(&gu, &st, n_tok * top),
                1.5e-3,
                "tail DOWN",
            );
            assert_rel(
                &kq_sorted_combine(&dn, &w, &st.inv_arr),
                &dec,
                2e-3,
                "tail vs decode",
            );
        } else {
            assert!(!require_nax(), "Metal/NAX required");
        }
    }

    // 4. Strided X bit-exact + kq_sorted_combine F16 vs F32 on VEC4 (H=256) and tail (H=259).
    let (ne, k, m, n_tok, top) = (3usize, 512usize, 256usize, 8usize, 4usize);
    let mut rng = Rng::new(4101);
    let rig = GgufExpertsRig::new(&mut rng, 4, ne, k, m, 256, false);
    let xs = arr(&rng.vec_f32(n_tok * 3 * k, 4.0), &[n_tok, 3 * k])
        .slice(&[0, 0], &[n_tok as i64, k as i64])
        .unwrap();
    let xc = arr(&read_f32(&xs), &[n_tok, k]);
    let pairs = n_tok * top;
    let ids: Vec<u32> = (0..pairs).map(|i| ((i * 5 + 1) % ne) as u32).collect();
    let st = sort_and_tile(&ids, n_tok, top, 32);
    if let Some(gu_s) = rig.prefill_gu_gguf(&xs, &st.tr_arr, &st.tl_arr, 7.0) {
        let gu_c = rig
            .prefill_gu_gguf(&xc, &st.tr_arr, &st.tl_arr, 7.0)
            .unwrap();
        assert_eq!(read_bits(&gu_s), read_bits(&gu_c), "strided x mismatch");
    } else {
        assert!(!require_nax(), "Metal/NAX required");
    }
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        assert!(!require_metal(), "Metal required");
        return;
    }
    let w = arr(&rand_weights(&mut rng, pairs, 0.05, 0.4), &[n_tok, top]);
    for h in [256usize, 259usize] {
        let d16 = arr(&rng.vec_f32(pairs * h, 3.0), &[pairs, h])
            .astype(DType::Float16)
            .unwrap();
        let o16 = kq_sorted_combine(&d16, &w, &st.inv_arr);
        let o32 = kq_sorted_combine(&d16.astype(DType::Float32).unwrap(), &w, &st.inv_arr);
        assert_eq!(o16.dtype().unwrap(), DType::Float32);
        let (mabs, rel) = error_stats(&read_f32(&o16), &read_f32(&o32));
        assert!(
            mabs <= 1e-6 && rel <= 1e-6,
            "combine F16 H={h}: {mabs:e}/{rel:e}"
        );
    }
}

#[test]
fn moe_prefill_sort_tiles_and_bm64_multi_tile_parity() {
    // 1. GPU sort + tile across single-TG, multi-TG, OOB, BM=32 and BM=64.
    for &(rows, k, ne, oob) in &[
        (16usize, 4usize, 17usize, false),
        (128, 16, 64, true),
        (129, 16, 64, false),
        (256, 16, 128, true),
        (4096, 16, 512, false),
    ] {
        let mut rng = Rng::new(7700 + (rows * 31 + ne) as u64);
        let r = rows * k;
        let ids: Vec<u32> = (0..r)
            .map(|i| {
                if oob && i % 97 == 13 {
                    ne as u32 + (i as u32 % 5)
                } else if i % 11 == 0 {
                    (i % 3) as u32
                } else {
                    rng.usize(ne) as u32
                }
            })
            .collect();
        let ids_arr = u32_arr(&ids, &[rows, k]);
        for &tm in &[32usize, 64] {
            let Some((tr_a, tl_a, inv_a, sl_a)) = moe_prefill_sort_tiles(&ids_arr, ne, tm).unwrap()
            else {
                assert!(!require_metal(), "Metal required");
                return;
            };
            let (tr, tl, inv, sl) = (
                read_u32(&tr_a),
                read_u32(&tl_a),
                read_u32(&inv_a),
                read_u32(&sl_a),
            );
            assert_eq!(
                (tr.len(), inv.len(), sl.len(), tl.len()),
                (r, r, r, r.min(r.div_ceil(tm) + ne) * 3)
            );
            let valid = ids.iter().filter(|&&e| (e as usize) < ne).count();
            let mut seen = vec![false; valid];
            for (i, &e) in ids.iter().enumerate() {
                let pos = inv[i] as usize;
                if (e as usize) < ne {
                    assert!(pos < valid && !seen[pos] && (tr[pos], sl[pos]) == ((i / k) as u32, e));
                    seen[pos] = true;
                } else {
                    assert_eq!(pos, r);
                }
            }
            assert!(seen.iter().all(|&s| s) && sl[..valid].windows(2).all(|w| w[0] <= w[1]));
            assert!(tr[valid..].iter().all(|&v| v == 0) && sl[valid..].iter().all(|&v| v == 0));
            let want_tl = build_prefill_tiles_bm(&sl[..valid], tm);
            assert_eq!(&tl[..want_tl.len()], want_tl.as_slice());
            assert!(tl[want_tl.len()..].iter().all(|&v| v == 0));
        }
    }

    // 2. BM=64 runs spanning 1, 17, 32, 33, 63, 64, 65, 130 rows per expert (Q4_K & Q6_K).
    let run_lens = [1usize, 17, 32, 33, 63, 64, 65, 130];
    let (ne, pairs, k, m, h, n_tok) = (
        run_lens.len(),
        run_lens.iter().sum::<usize>(),
        512usize,
        256usize,
        256usize,
        64usize,
    );
    let (mut sl, mut tr) = (Vec::with_capacity(pairs), Vec::with_capacity(pairs));
    for (e, &len) in run_lens.iter().enumerate() {
        for j in 0..len {
            sl.push(e as u32);
            tr.push(((e * 13 + j * 5) % n_tok) as u32);
        }
    }
    let tl = build_prefill_tiles_bm(&sl, 64);
    let (tr_a, tl_a, sl_a) = (
        u32_arr(&tr, &[pairs]),
        u32_arr(&tl, &[tl.len() / 3, 3]),
        u32_arr(&sl, &[pairs]),
    );
    for bits in [4, 6] {
        let mut rng = Rng::new(8200 + bits as u64);
        let rig = GgufExpertsRig::new(&mut rng, bits, ne, k, m, h, true);
        let x = arr(&rng.vec_f32(n_tok * k, 5.0), &[n_tok, k]);
        let Some(got_gu) = rig.prefill_gu_gguf(&x, &tr_a, &tl_a, 7.0) else {
            assert!(!require_nax(), "Metal/NAX required");
            return;
        };
        let got_dn = rig.prefill_down_gguf(&got_gu, &tl_a).unwrap();
        let (ref_gu, ref_dn) = rig.fp32_prefill_ref(&x, &tr_a, &sl_a, pairs, 7.0);
        assert_rel(&got_gu, &ref_gu, 2e-3, &format!("BM64 GU b={bits}"));
        assert_rel(&got_dn, &ref_dn, 2.5e-3, &format!("BM64 DN b={bits}"));
    }
}

fn cpu_dequant_q4k_block(blk: &[u8]) -> [f32; 256] {
    let d = half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
    let dmin = half::f16::from_bits(u16::from_le_bytes([blk[2], blk[3]])).to_f32();
    let (sc_b, qs, mut out) = (&blk[4..16], &blk[16..144], [0.0f32; 256]);
    let sm = |g: usize| {
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
    for c in 0..4 {
        let ((d0, m0), (d1, m1)) = (sm(2 * c), sm(2 * c + 1));
        for l in 0..32 {
            let b = qs[c * 32 + l];
            out[c * 64 + l] = f32::from(b & 0x0F).mul_add(d0, m0);
            out[c * 64 + 32 + l] = f32::from(b >> 4).mul_add(d1, m1);
        }
    }
    out
}

fn cpu_dequant_q6k_block(blk: &[u8]) -> [f32; 256] {
    let d = half::f16::from_bits(u16::from_le_bytes([blk[208], blk[209]])).to_f32();
    let mut out = [0.0f32; 256];
    for ip in 0..2 {
        let (ql, qh, sc) = (
            &blk[ip * 64..ip * 64 + 64],
            &blk[128 + ip * 32..160 + ip * 32],
            &blk[192 + ip * 8..200 + ip * 8],
        );
        for l in 0..32 {
            let is = l / 16;
            let qs = [
                i32::from((ql[l] & 0x0F) | ((qh[l] & 0x03) << 4)) - 32,
                i32::from((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4)) - 32,
                i32::from((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) - 32,
                i32::from((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) - 32,
            ];
            for (j, q) in qs.into_iter().enumerate() {
                out[ip * 128 + j * 32 + l] = (q as f32) * d * f32::from(sc[is + j * 2] as i8);
            }
        }
    }
    out
}

fn cpu_ref_matmul(
    tiles: &[u32],
    tr: Option<&[u32]>,
    x: &[f32],
    w0: &[f32],
    w1_lim: Option<(&[f32], f32)>,
    (pairs, k, od): (usize, usize, usize),
) -> Vec<f32> {
    let mut out = vec![0.0f32; pairs * od];
    for tile in tiles.as_chunks::<3>().0 {
        let (begin, end, slot) = (tile[0] as usize, tile[1] as usize, tile[2] as usize);
        for p in begin..end {
            let xr = &x[tr.map_or(p, |t| t[p] as usize) * k..][..k];
            for o in 0..od {
                let off = (slot * od + o) * k;
                let v0: f32 = xr.iter().zip(&w0[off..off + k]).map(|(a, b)| a * b).sum();
                out[p * od + o] = if let Some((w1, lim)) = w1_lim {
                    let g = v0.min(lim);
                    let u = xr
                        .iter()
                        .zip(&w1[off..off + k])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        .clamp(-lim, lim);
                    (g / (1.0 + (-g).exp())) * u
                } else {
                    v0
                };
            }
        }
    }
    out
}

fn build_mixed_blocks(
    rng: &mut Rng,
    fmt: KQuantFormat,
    (ne, rows, k): (usize, usize, usize),
    special: &[u8],
) -> (Vec<u8>, Vec<f32>) {
    let (bb, nb) = (fmt.block_bytes(), k / 256);
    let deq = if matches!(fmt, KQuantFormat::Q6K) {
        cpu_dequant_q6k_block
    } else {
        cpu_dequant_q4k_block
    };
    let normal = gen_raw(rng, ne, rows, 256, fmt);
    let (mut raw, mut ref_w) = (vec![0u8; ne * rows * nb * bb], vec![0.0f32; ne * rows * k]);
    for er in 0..ne * rows {
        for b in 0..nb {
            let dst = &mut raw[(er * nb + b) * bb..(er * nb + b + 1) * bb];
            dst.copy_from_slice(if b == 0 {
                special
            } else {
                &normal[er * bb..(er + 1) * bb]
            });
            ref_w[er * k + b * 256..er * k + (b + 1) * 256].copy_from_slice(&deq(dst));
        }
    }
    (raw, ref_w)
}

/// Regression test for FP16 early-narrowing NaNs:
/// - Q4_K cancellation (`d = dmin = 2048, sc = m = 63, q = 1 -> 0`) in GU and DOWN
/// - Q6_K zero-code (`d = 2048, sc = 64, q = 0 -> 0`) in DOWN at 2-byte offset 66
#[test]
fn gguf_prefill_indirect_no_fp16_early_narrowing_nan() {
    crate::models::hy_v4_flash::exact_f32();
    let mut rng = Rng::new(20261123);
    let d_2048 = half::f16::from_f32(2048.0).to_bits().to_le_bytes();

    let mut q4_blk = [0u8; 144];
    q4_blk[0..2].copy_from_slice(&d_2048);
    q4_blk[2..4].copy_from_slice(&d_2048);
    q4_blk[4..16].fill(0xFF);
    q4_blk[16..144].fill(0x11);
    assert!(cpu_dequant_q4k_block(&q4_blk).iter().all(|&v| v == 0.0));

    let mut q6_blk = [0u8; 210];
    q6_blk[128..192].fill(0xAA);
    q6_blk[192..208].fill(64u8);
    q6_blk[208..210].copy_from_slice(&d_2048);
    assert!(cpu_dequant_q6k_block(&q6_blk).iter().all(|&v| v == 0.0));

    let (ne, m, h, n_tok, pairs) = (2usize, 64usize, 64usize, 16usize, 32usize);
    let (mut sl, mut tr_v) = (Vec::with_capacity(pairs), Vec::with_capacity(pairs));
    for e in 0..ne {
        for j in 0..n_tok {
            sl.push(e as u32);
            tr_v.push(((e * 7 + j) % n_tok) as u32);
        }
    }
    let tl_v = build_prefill_tiles_bm(&sl, 64);
    let (tr_a, tl_a) = (
        u32_arr(&tr_v, &[pairs]),
        u32_arr(&tl_v, &[tl_v.len() / 3, 3]),
    );

    for &k in &[256usize, 512usize] {
        let (rg, ref_g) = build_mixed_blocks(&mut rng, KQuantFormat::Q4K, (ne, m, k), &q4_blk);
        let (ru, ref_u) = build_mixed_blocks(&mut rng, KQuantFormat::Q4K, (ne, m, k), &q4_blk);
        let (gb, ub) = (
            make_padded_u8_buf(&rg, 32, false),
            make_padded_u8_buf(&ru, 64, false),
        );
        let xv = rng.vec_f32(n_tok * k, 1.5);
        let Some(got_gu) = gguf_prefill_indirect_gu(
            &arr(&xv, &[n_tok, k]),
            &tr_a,
            &tl_a,
            &gb,
            32,
            &ub,
            64,
            ne,
            m,
            7.0,
        )
        .unwrap() else {
            assert!(!require_nax(), "Metal/NAX required");
            return;
        };
        let want_gu = cpu_ref_matmul(
            &tl_v,
            Some(&tr_v),
            &xv,
            &ref_g,
            Some((&ref_u, 7.0)),
            (pairs, k, m),
        );
        let (mabs, rel) = error_stats(&read_f32(&got_gu), &want_gu);
        assert!(
            (k == 256 && mabs == 0.0) || (k > 256 && rel <= 5e-3),
            "Q4K gu K={k}: {mabs:e}/{rel:e}"
        );

        let hv = rng.vec_f32(pairs * k, 1.5);
        let ha = arr(&hv, &[pairs, k]);
        for (bits, off, fmt, blk) in [
            (4i32, 48u64, KQuantFormat::Q4K, &q4_blk[..]),
            (6, 66, KQuantFormat::Q6K, &q6_blk[..]),
        ] {
            let (rd, ref_d) = build_mixed_blocks(&mut rng, fmt, (ne, h, k), blk);
            let db = make_padded_u8_buf(&rd, off, false);
            let got = gguf_prefill_indirect_down(&ha, &tl_a, &db, off, ne, h, bits)
                .unwrap()
                .expect("down must run");
            let want_d = cpu_ref_matmul(&tl_v, None, &hv, &ref_d, None, (pairs, k, h));
            let (mabs, rel) = error_stats(&read_f32(&got), &want_d);
            assert!(
                (k == 256 && mabs == 0.0) || (k > 256 && rel <= 5e-3),
                "Q{bits}K down K={k}: {mabs:e}/{rel:e}"
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
    for (bits, ne, m, h, r) in [
        (4, 4usize, 512usize, 256usize, 53usize),
        (6, 4, 512, 256, 53),
        (4, 4, 512, 260, 53),
        (6, 4, 512, 260, 53),
        (6, 1, 256, 1, 5),
    ] {
        let mut rng = Rng::new(9300 + (bits as u64) * 97 + h as u64);
        let (ty, fmt) = fmt_ty(bits);
        let d_kq = repack_to_proj_arrays(&gen_raw(&mut rng, ne, h, m, fmt), ty, ne, h, m);
        let h32 = arr(&rng.vec_f32(r * m, 3.0), &[r, m]);
        let h16 = h32.astype(DType::Float16).unwrap();
        let sl: Vec<u32> = (0..r).map(|i| ((i * ne) / r) as u32).collect();
        let tl_v = build_prefill_tiles(&sl);
        let tl = u32_arr(&tl_v, &[tl_v.len() / 3, 3]);
        let ref_dn = d_kq
            .gather(
                &h32.reshape(&[r as i64, 1, 1, m as i64]).unwrap(),
                &u32_arr(&sl, &[r, 1]),
            )
            .unwrap()
            .astype(DType::Float32)
            .unwrap()
            .reshape(&[r as i64, h as i64])
            .unwrap();

        for (tag, hid) in [("f32", &h32), ("f16", &h16)] {
            let got = kq_prefill_indirect_down(hid, &tl, &d_kq.w, &d_kq.s, &d_kq.b, bits)
                .unwrap()
                .expect("m3 down");
            assert_eq!(got.dtype().unwrap(), DType::Float32);
            assert_rel(&got, &ref_dn, 2e-3, &format!("m3 b={bits} H={h} {tag}"));
        }
        let h4 = h32.slice(&[0, 0], &[4, m as i64]).unwrap();
        let oob = kq_prefill_indirect_down(
            &h4,
            &u32_arr(&[0, 4, 99], &[1, 3]),
            &d_kq.w,
            &d_kq.s,
            &d_kq.b,
            bits,
        )
        .unwrap()
        .unwrap();
        assert!(read_f32(&oob).iter().all(|&v| v == 0.0), "m3 OOB not zero");
    }
}
