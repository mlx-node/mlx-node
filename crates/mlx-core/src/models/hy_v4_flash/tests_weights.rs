use std::collections::HashMap;
#[cfg(unix)]
use std::path::PathBuf;

use super::attention::{grouped_mm_t_mgh, grouped_mm_t_mgh_out};
use super::moe::{
    prefill_q8_grouped_nax_enabled, qlinear, qlinear_out, try_qlinear_sigmoid_mul,
    try_qlinear_sigmoid_mul_out, with_prefill_q8_nax,
};
use super::scan::Role;
use super::test_support::{
    Rng, arr, assert_row_invariant, error_stats, fixture, j_u64, make_qmm, make_sym_q8,
    max_abs_err, read_bits, read_f32, real_scan, rel_l2_err,
};
use super::weights::{
    Dense, Form, OUTPUT, Resident, Residual, TOKEN_EMBD, bf16_into_f32, dense_from_raw, form,
    fuse_qmm_rows, held_shape, le_f32_into, le_u16_into, q8_0_into_f32, quantize_affine4_array,
    quantize_affine4_from_bf16_raw, resident_bytes, role_resident,
};
use crate::array::{DType, MxArray};
use crate::models::deepseek_v41::store::Weight;
use crate::moe_stream::io::ShardSet;
use crate::nn::Activations;
use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};

fn info(name: &str, dims: &[u64], ty: GgufTensorType) -> GgufTensorInfo {
    GgufTensorInfo {
        name: name.into(),
        n_dims: dims.len() as u32,
        dims: dims.to_vec(),
        tensor_type: ty,
        offset: 0,
    }
}

fn q8_0_raw(n: usize) -> Vec<u8> {
    let mut raw = Vec::with_capacity(n / 32 * 34);
    for b in 0..n / 32 {
        let d = half::f16::from_f32(2f32.powi(-((b % 7) as i32 + 3)));
        raw.extend(d.to_bits().to_le_bytes());
        raw.extend((0..32).map(|i| ((((b * 37 + i * 11) % 256) as i32 - 128) as i8) as u8));
    }
    raw
}

fn f32_bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}
fn f32_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn as_dt(x: &MxArray, dt: DType) -> MxArray {
    x.astype(dt).unwrap()
}
fn i32_arr(v: &[i32], s: &[i64]) -> MxArray {
    MxArray::from_int32(v, s).unwrap()
}
fn load_arr(i: &GgufTensorInfo, raw: &[u8], f: Form) -> MxArray {
    dense_from_raw(i, raw, f).unwrap().array().unwrap()
}
fn slice_axis(x: &MxArray, last: bool, lo: usize, hi: usize) -> MxArray {
    let mut stop: Vec<i64> = x.shape().unwrap().as_ref().to_vec();
    let mut start = vec![0i64; stop.len()];
    let ax = if last { stop.len() - 1 } else { 0 };
    start[ax] = lo as i64;
    stop[ax] = hi as i64;
    x.slice(&start, &stop).unwrap()
}
fn ql_nax(nax: bool, x: &MxArray, w: &Dense, f16: bool) -> MxArray {
    with_prefill_q8_nax(nax, || qlinear_out(x, w, f16).unwrap())
}
fn grp_nax(nax: bool, w: &Dense, x: &MxArray, g: usize, f16: bool) -> MxArray {
    with_prefill_q8_nax(nax, || grouped_mm_t_mgh_out(w, x, g as i64, f16).unwrap())
}

fn sym_to_explicit_qmm(d: &Dense) -> Dense {
    let Dense::SymQ8 { w, scales } = d else {
        panic!("expected SymQ8")
    };
    let s = as_dt(scales, DType::Float32);
    let b = s.mul_scalar(-128.0).unwrap();
    MxArray::eval_arrays_with_context(&[w, &s, &b], "test explicit qmm").unwrap();
    Dense::Qmm {
        w: w.clone(),
        scales: s,
        biases: b,
    }
}

fn q8_prefill_tol(grouped: bool) -> f32 {
    if std::env::var("MLX_HY4_Q8_PREFILL_1MMA").as_deref() == Ok("0")
        || (grouped && std::env::var("MLX_HY4_Q8_GROUPED_1MMA").as_deref() == Ok("0"))
    {
        5e-6
    } else {
        5e-4
    }
}

fn grouped_ref(w: &Dense, x: &MxArray, g: usize, k: usize, n: usize) -> Vec<f32> {
    let wf = w
        .f32()
        .unwrap()
        .reshape(&[g as i64, n as i64, k as i64])
        .unwrap();
    let wt = wf.transpose(Some(&[0, 2, 1])).unwrap();
    let y = x.transpose(Some(&[1, 0, 2])).unwrap().matmul(&wt).unwrap();
    read_f32(&y.transpose(Some(&[1, 0, 2])).unwrap())
}

fn check_decode_err(got: &MxArray, want: &[f32], strict: bool, tag: &str) {
    let (mabs, rel) = error_stats(&read_f32(got), want);
    assert!(
        if strict {
            mabs <= 1e-5
        } else {
            rel <= 1e-4 || mabs <= 1e-5
        },
        "{tag}: {mabs:e}/{rel:e}"
    );
}

#[test]
fn decoders_forms_bytes_and_shapes() {
    let (mut w, mut f) = ([0u16; 2], [0f32; 2]);
    le_u16_into(&[0x34, 0x12, 0x80, 0x3f], &mut w).unwrap();
    assert_eq!(w, [0x1234, 0x3f80]);
    bf16_into_f32(&[0x80, 0x3f, 0x40, 0xc0], &mut f).unwrap();
    assert_eq!(f, [1.0, -3.0]);
    let raw = f32_le(&[1.5, -0.25]);
    le_f32_into(&raw, &mut f).unwrap();
    assert_eq!(f, [1.5, -0.25]);
    assert!(le_u16_into(&[0; 3], &mut w).is_err());
    assert!(bf16_into_f32(&[0; 2], &mut f).is_err());
    assert!(le_f32_into(&raw[..7], &mut f).is_err());

    let qraw = q8_0_raw(64);
    let mut out = vec![0f32; 64];
    q8_0_into_f32(&qraw, &mut out).unwrap();
    for b in 0..2 {
        let d = 2f32.powi(-(b as i32 + 3));
        for i in 0..32 {
            let q = ((b * 37 + i * 11) % 256) as i32 - 128;
            assert_eq!(out[b * 32 + i].to_bits(), (d * q as f32).to_bits());
        }
    }
    assert!(q8_0_into_f32(&qraw[..67], &mut out).is_err());
    assert!(q8_0_into_f32(&qraw, &mut out[..32]).is_err());

    use GgufTensorType::{BF16, F32, MXFP8, Q4K, Q8_0};
    use Role::{Dense as RDense, EngramTable, Expert, Mtp};
    let form_cases = [
        (RDense, TOKEN_EMBD, Q8_0, Some(Form::Codes)),
        (RDense, "blk.1.attn_q_b.weight", Q8_0, Some(Form::Qmm)),
        (
            RDense,
            "blk.1.indexer_compressor_value.weight",
            Q8_0,
            Some(Form::F32),
        ),
        (
            RDense,
            "blk.1.indexer_compressor_gate.weight",
            Q8_0,
            Some(Form::F32),
        ),
        (
            RDense,
            "blk.1.xindexer_compressor_gate.weight",
            Q8_0,
            Some(Form::Qmm),
        ),
        (RDense, "blk.1.attn_norm.weight", F32, Some(Form::F32)),
        (RDense, "blk.1.ffn_gate_inp.weight", F32, Some(Form::F32)),
        (RDense, OUTPUT, BF16, Some(Form::Bf16)),
        (RDense, "blk.1.attn_q_b.weight", BF16, None),
        (RDense, "blk.1.attn_q_b.weight", Q4K, None),
        (Mtp, "blk.42.nextn.eh_proj.weight", BF16, Some(Form::Bf16)),
        (
            Mtp,
            "blk.42.indexer_compressor_value.weight",
            BF16,
            Some(Form::F32),
        ),
        (Mtp, "blk.42.nextn.enorm.weight", F32, Some(Form::F32)),
        (Expert, "blk.1.ffn_up_exps.weight", Q4K, None),
        (EngramTable, "per_layer_token_embd.weight", MXFP8, None),
    ];
    for (role, name, ty, want) in form_cases {
        assert_eq!(
            form(role, &info(name, &[32, 4], ty)).ok(),
            want,
            "{role:?} {name}"
        );
    }

    let bytes_cases: &[(&str, &[u64], GgufTensorType, u64)] = &[
        ("blk.0.attn_q_a.weight", &[64, 4], Q8_0, 272),
        (TOKEN_EMBD, &[64, 4], Q8_0, 272),
        (
            "blk.0.indexer_compressor_gate.weight",
            &[128, 128],
            Q8_0,
            65536,
        ),
        (OUTPUT, &[8, 16], BF16, 256),
        ("output_norm.weight", &[3], F32, 12),
    ];
    for &(name, dims, ty, want) in bytes_cases {
        assert_eq!(
            resident_bytes(RDense, &info(name, dims, ty)).unwrap(),
            want,
            "{name}"
        );
    }

    let shape_cases: &[(&str, &[u64], GgufTensorType, &[i64])] = &[
        (
            "blk.0.attn_k_b.weight",
            &[192, 512, 32],
            Q8_0,
            &[32 * 512, 192],
        ),
        (
            "blk.0.attn_v_b.weight",
            &[512, 256, 64],
            Q8_0,
            &[64 * 256, 512],
        ),
        (
            "blk.0.attn_kv_a_mqa.weight",
            &[3072, 576],
            Q8_0,
            &[576, 3072],
        ),
        ("output_norm.weight", &[3072], F32, &[3072]),
    ];
    for &(name, dims, ty, want) in shape_cases {
        assert_eq!(held_shape(&info(name, dims, ty)).unwrap(), want, "{name}");
    }
}

#[test]
fn rank_three_q8_and_dense_forms_hold_file_values() {
    let (k, rows, h, t) = (64usize, 8usize, 3usize, 5usize);
    let i3 = info(
        "blk.0.attn_k_b.weight",
        &[k as u64, rows as u64, h as u64],
        GgufTensorType::Q8_0,
    );
    let raw3 = q8_0_raw(k * rows * h);
    let d = dense_from_raw(&i3, &raw3, Form::Qmm).unwrap();
    assert!(matches!(d, Dense::SymQ8 { .. }));
    assert_eq!(d.nbytes() as u64, resident_bytes(Role::Dense, &i3).unwrap());
    assert_eq!(
        (d.shape_at(0).unwrap(), d.shape_at(1).unwrap()),
        ((h * rows) as i64, k as i64)
    );
    let mut exact3 = vec![0f32; k * rows * h];
    q8_0_into_f32(&raw3, &mut exact3).unwrap();
    assert_eq!(read_bits(&d.f32().unwrap()), f32_bits(&exact3));
    let xs = Rng::new(7).vec_f32(h * t * k, 1.0);
    let got = read_f32(
        &d.grouped_qmm_t(&arr(&xs, &[h, t, k]), h as i64)
            .unwrap()
            .unwrap(),
    );
    let mut want = vec![0f32; h * t * rows];
    for g in 0..h {
        for ti in 0..t {
            for r in 0..rows {
                let w = &exact3[(g * rows + r) * k..(g * rows + r + 1) * k];
                let xv = &xs[(g * t + ti) * k..(g * t + ti + 1) * k];
                want[(g * t + ti) * rows + r] = w
                    .iter()
                    .zip(xv)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum::<f64>() as f32;
            }
        }
    }
    assert!(rel_l2_err(&got, &want) < 1e-5);

    // Codes, F32, BF16, F16 forms.
    let (k2, r2) = (64usize, 6usize);
    let raw = q8_0_raw(k2 * r2);
    let mut exact = vec![0f32; k2 * r2];
    q8_0_into_f32(&raw, &mut exact).unwrap();
    let emb = info(TOKEN_EMBD, &[k2 as u64, r2 as u64], GgufTensorType::Q8_0);
    let codes = dense_from_raw(&emb, &raw, Form::Codes).unwrap();
    assert!(matches!(codes, Dense::Q8 { .. }));
    assert_eq!(
        codes.nbytes() as u64,
        resident_bytes(Role::Dense, &emb).unwrap()
    );
    let gathered: Vec<f32> = [4, 1]
        .iter()
        .flat_map(|&r| exact[r * k2..(r + 1) * k2].to_vec())
        .collect();
    assert_eq!(
        read_bits(&codes.rows(&i32_arr(&[4, 1], &[2])).unwrap()),
        f32_bits(&gathered)
    );

    let comp = info(
        "blk.0.indexer_compressor_value.weight",
        &[k2 as u64, r2 as u64],
        GgufTensorType::Q8_0,
    );
    let f = load_arr(&comp, &raw, Form::F32);
    assert_eq!(
        (f.dtype().unwrap(), read_bits(&f)),
        (DType::Float32, f32_bits(&exact))
    );

    let words: Vec<u16> = (0..k2 * r2)
        .map(|i| 0x3f80 ^ ((i as u16 * 97) & 0x80ff))
        .collect();
    let braw: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let out = info(OUTPUT, &[k2 as u64, r2 as u64], GgufTensorType::BF16);
    let b = load_arr(&out, &braw, Form::Bf16);
    assert_eq!(
        (b.dtype().unwrap(), b.to_uint16_native().unwrap()),
        (DType::BFloat16, words)
    );
    let mut wide = vec![0f32; k2 * r2];
    bf16_into_f32(&braw, &mut wide).unwrap();
    assert_eq!(
        read_bits(&load_arr(&out, &braw, Form::F32)),
        f32_bits(&wide)
    );

    let fraw = f32_le(&Rng::new(3).vec_f32(12, 1.0));
    let mut nv = vec![0f32; 12];
    le_f32_into(&fraw, &mut nv).unwrap();
    let n = load_arr(
        &info("output_norm.weight", &[12], GgufTensorType::F32),
        &fraw,
        Form::F32,
    );
    assert_eq!(read_bits(&n), f32_bits(&nv));
    assert!(dense_from_raw(&emb, &raw[..raw.len() - 1], Form::Codes).is_err());
    assert!(dense_from_raw(&out, &braw, Form::Qmm).is_err());

    let rvals = Rng::new(9).vec_f32(128, 1.0);
    let rinfo = info("blk.1.ffn_gate_inp.weight", &[32, 4], GgufTensorType::F32);
    let rf16 = load_arr(&rinfo, &f32_le(&rvals), Form::F16);
    assert_eq!(
        (rf16.dtype().unwrap(), rf16.nbytes()),
        (DType::Float16, 256)
    );
    let wf16: Vec<f32> = rvals
        .iter()
        .map(|&v| half::f16::from_f32(v).to_f32())
        .collect();
    assert_eq!(read_bits(&rf16), f32_bits(&wf16));
}

fn form_key(role: Role, i: &GgufTensorInfo) -> &'static str {
    match (form(role, i).unwrap(), i.tensor_type) {
        (Form::Codes, _) => "q8_codes",
        (Form::Qmm, _) => "qmm_affine8",
        (Form::F32, GgufTensorType::F32) => "f32",
        (Form::F32, _) => "f32_dequant",
        (Form::F16, _) => "f16",
        (Form::Bf16, _) => "bf16",
    }
}

#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards"]
fn real_resident_bytes_match_the_oracle() {
    let s = real_scan();
    let want = fixture("resident.json");
    let trunk = &want["trunk_dense"];
    assert_eq!(
        role_resident(&s, Role::Dense).unwrap(),
        j_u64(&trunk["resident_bytes"])
    );
    let mut by: HashMap<&str, (u64, u64, u64)> = HashMap::new();
    for (_, loc) in s.with_role(Role::Dense) {
        let e = by.entry(form_key(Role::Dense, &loc.info)).or_default();
        e.0 += 1;
        e.1 += loc.info.data_size().unwrap();
        e.2 += resident_bytes(Role::Dense, &loc.info).unwrap();
    }
    let forms = trunk["by_form"].as_object().unwrap();
    assert_eq!(by.len(), forms.len());
    for (k, v) in forms {
        let exp = (
            j_u64(&v["tensors"]),
            j_u64(&v["file_bytes"]),
            j_u64(&v["resident_bytes"]),
        );
        assert_eq!(by[k.as_str()], exp, "{k}");
    }
    let mtp_want: u64 = want["mtp_blk"]["list"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|t| !t["name"].as_str().unwrap().ends_with("_exps.weight"))
        .map(|t| {
            let (nm, b) = (t["name"].as_str().unwrap(), j_u64(&t["bytes"]));
            if nm.contains("indexer_compressor_value") || nm.contains("indexer_compressor_gate") {
                2 * b
            } else {
                b
            }
        })
        .sum();
    assert_eq!(role_resident(&s, Role::Mtp).unwrap(), mtp_want);
}

#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards and the GPU"]
fn real_dense_load_matches_the_oracle() {
    crate::models::hy_v4_flash::exact_f32();
    let s = real_scan();
    let want = fixture("resident.json");
    let shards = ShardSet::open(&s.paths).unwrap();
    let mut r = Resident::load(&s, &shards).unwrap();
    let trunk = &want["trunk_dense"];
    assert_eq!(
        (r.read, r.bytes),
        (j_u64(&trunk["file_bytes"]), j_u64(&trunk["resident_bytes"]))
    );
    assert_eq!((r.layers.len(), r.mtp.is_empty()), (s.cfg.n_layers, true));
    let (kv_lora, nope) = (s.cfg.mla.kv_lora_rank, s.cfg.mla.qk_nope_dim);
    for il in [0usize, 3] {
        let h = s.cfg.layers[il].n_heads;
        let k_b = &r.layers[il].attn.k_b;
        assert_eq!(
            (k_b.shape_at(0).unwrap(), k_b.shape_at(1).unwrap()),
            ((h * kv_lora) as i64, nope as i64)
        );
        let t = 4usize;
        let x = arr(
            &Rng::new(il as u64 + 11).vec_f32(h * t * nope, 1.0),
            &[h, t, nope],
        );
        let got = read_f32(&k_b.grouped_qmm_t(&x, h as i64).unwrap().unwrap());
        let mut want_v = Vec::with_capacity(got.len());
        for g in 0..h {
            let lo = (g * kv_lora) as i64;
            let w = k_b
                .row_range(lo, lo + kv_lora as i64)
                .unwrap()
                .f32()
                .unwrap();
            let xg = slice_axis(&x, false, g, g + 1)
                .reshape(&[t as i64, nope as i64])
                .unwrap();
            want_v.extend(read_f32(
                &xg.matmul(&w.transpose(Some(&[1, 0])).unwrap()).unwrap(),
            ));
        }
        assert!(rel_l2_err(&got, &want_v) < 1e-5, "layer {il}");
    }
    let before = r.bytes;
    r.load_mtp(&s, &shards).unwrap();
    assert_eq!(r.mtp.len(), s.cfg.layers.len() - s.cfg.n_layers);
    assert!(matches!(r.mtp[0].residual, Residual::Mtp(_)));
    assert_eq!(r.bytes - before, role_resident(&s, Role::Mtp).unwrap());
    let ex = s.experts[s.cfg.n_layers].as_ref().unwrap();
    let mut reader = super::weights::Reader::new(&s, &shards);
    let slab = reader.bf16_experts(&ex.gate, 0, 2).unwrap();
    assert_eq!(
        slab.shape().unwrap().as_ref(),
        &[2, s.cfg.moe.ff as i64, s.cfg.moe.latent as i64]
    );
    assert_eq!(
        reader.read,
        ex.gate.info.data_size().unwrap() / s.cfg.moe.n_experts as u64 * 2
    );
    assert!(reader.bf16_experts(&ex.gate, 2, 2).is_err());
    assert!(
        reader
            .bf16_experts(&s.experts[1].as_ref().unwrap().gate, 0, 1)
            .is_err()
    );
}

#[test]
fn fuse_qmm_rows_matches_separate_qlinear_bit_for_bit() {
    let mut rng = Rng::new(101);
    let (k, r0, r1, r2) = (64usize, 32usize, 48usize, 64usize);
    let (a, b, c) = (
        make_sym_q8(&mut rng, r0, k, 1.0),
        make_sym_q8(&mut rng, r1, k, 1.0),
        make_sym_q8(&mut rng, r2, k, 1.0),
    );
    let plain = Dense::Array(arr(&rng.vec_f32(r0 * k, 1.0), &[r0, k]));
    assert!(
        fuse_qmm_rows(&[&a, &plain]).unwrap().is_none() && fuse_qmm_rows(&[&a]).unwrap().is_none()
    );

    let (fused, views) = fuse_qmm_rows(&[&a, &b, &c]).unwrap().unwrap();
    assert_eq!(
        (fused.shape_at(0).unwrap(), fused.shape_at(1).unwrap()),
        ((r0 + r1 + r2) as i64, k as i64)
    );
    assert_eq!(
        views
            .iter()
            .map(|v| v.shape_at(0).unwrap())
            .collect::<Vec<_>>(),
        [r0 as i64, r1 as i64, r2 as i64]
    );
    for m in [1usize, 2, 4, 12, 256] {
        let x = arr(&rng.vec_f32(m * k, 1.0), &[m, k]);
        let y_all = qlinear(&x, &fused).unwrap();
        for ((p, v), (lo, hi)) in [&a, &b, &c].into_iter().zip(&views).zip([
            (0, r0),
            (r0, r0 + r1),
            (r0 + r1, r0 + r1 + r2),
        ]) {
            let want = read_bits(&qlinear(&x, p).unwrap());
            assert_eq!(read_bits(&slice_axis(&y_all, true, lo, hi)), want);
            assert_eq!(read_bits(&qlinear(&x, v).unwrap()), want);
        }
    }
}

fn check_dense_decode(
    rng: &mut Rng,
    w: &Dense,
    (k, n): (usize, usize),
    dec_ms: &[usize],
    nax_ms: &[usize],
    strict: bool,
    tag: &str,
) {
    let max_m = dec_ms.iter().chain(nax_ms).copied().max().unwrap_or(1);
    let wt = w.f32().unwrap().transpose(Some(&[1, 0])).unwrap();
    let x_all = arr(&rng.vec_f32(max_m * k, 1.25), &[max_m, k]);
    let run1 = |r| qlinear(&slice_axis(&x_all, false, r, r + 1), w).unwrap();

    assert_row_invariant(
        run1,
        |m| {
            let xm = slice_axis(&x_all, false, 0, m);
            let y = qlinear(&xm, w).unwrap();
            if strict || m == 1 || m == 8 {
                check_decode_err(
                    &y,
                    &read_f32(&xm.matmul(&wt).unwrap()),
                    strict,
                    &format!("{tag} ({k},{n}) M={m}"),
                );
            }
            y
        },
        dec_ms,
        &format!("{tag} ({k},{n})"),
    );
    if !nax_ms.is_empty() {
        assert_row_invariant(
            run1,
            |m| {
                let xm = slice_axis(&x_all, false, 0, m);
                let rel = rel_l2_err(
                    &read_f32(&qlinear(&xm, w).unwrap()),
                    &read_f32(&xm.matmul(&wt).unwrap()),
                );
                assert!(rel <= 5e-3, "{tag} NAX ({k},{n}) M={m}: {rel:e}");
                ql_nax(false, &xm, w, false)
            },
            nax_ms,
            &format!("{tag} fb ({k},{n})"),
        );
    }
}

fn check_grouped_decode(
    rng: &mut Rng,
    w: &Dense,
    (g, k, n): (usize, usize, usize),
    ms: &[usize],
    strict: bool,
    tag: &str,
) {
    let max_m = ms.iter().copied().max().unwrap_or(1);
    let x_all = arr(&rng.vec_f32(max_m * g * k, 1.25), &[max_m, g, k]);
    assert_row_invariant(
        |r| grouped_mm_t_mgh(w, &slice_axis(&x_all, false, r, r + 1), g as i64).unwrap(),
        |m| {
            let xm = slice_axis(&x_all, false, 0, m);
            let y = grouped_mm_t_mgh(w, &xm, g as i64).unwrap();
            if strict || m == 1 || m == max_m {
                check_decode_err(
                    &y,
                    &grouped_ref(w, &xm, g, k, n),
                    strict,
                    &format!("{tag} ({g},{k},{n}) M={m}"),
                );
            }
            y
        },
        ms,
        &format!("{tag} ({g},{k},{n})"),
    );
}

#[test]
fn sym_q8_decode_row_invariant_and_matches_f32_ref() {
    crate::models::hy_v4_flash::exact_f32();
    let mut rng = Rng::new(701);

    let (k, n) = (256usize, 96usize);
    let sym = make_sym_q8(&mut rng, n, k, 1.0 / (k as f32).sqrt());
    assert!(matches!(sym, Dense::SymQ8 { .. }));
    assert_eq!(sym.nbytes(), n * k + (n * k) / 16);
    let qmm = sym_to_explicit_qmm(&sym);
    assert_eq!(qmm.nbytes(), n * k + (n * k) / 4);
    let xs = arr(&rng.vec_f32(11 * k, 1.0), &[11, k]);
    for m in 1..=11usize {
        let xm = slice_axis(&xs, false, 0, m);
        assert!(
            max_abs_err(
                &read_f32(&qlinear(&xm, &sym).unwrap()),
                &read_f32(&qlinear(&xm, &qmm).unwrap())
            ) <= 1e-5
        );
    }
    check_dense_decode(
        &mut rng,
        &sym,
        (k, n),
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        &[12, 13, 14, 15, 16],
        true,
        "sym",
    );

    for &(k, n) in &[
        (1024usize, 256usize),
        (1536, 384),
        (1024, 261),
        (2048, 64),
        (192, 67),
    ] {
        let w = make_sym_q8(&mut rng, n, k, 1.0 / (k as f32).sqrt());
        check_dense_decode(
            &mut rng,
            &w,
            (k, n),
            &[1, 2, 3, 4, 8],
            &[12],
            true,
            "sym_v2",
        );
    }
    for &(k, n) in &[(1024usize, 512usize), (512, 256), (2048, 64), (512, 2048)] {
        let w = make_qmm(&mut rng, n, k, 1.5 / (k as f32).sqrt());
        check_dense_decode(
            &mut rng,
            &w,
            (k, n),
            &[1, 2, 4, 8, 11],
            &[12, 16],
            false,
            "qmm_dense",
        );
    }

    let (g, gk, gn) = (4usize, 192usize, 64usize);
    let gsym = make_sym_q8(&mut rng, g * gn, gk, 1.0 / (gk as f32).sqrt());
    let gqmm = sym_to_explicit_qmm(&gsym);
    let gx1 = arr(&rng.vec_f32(g * gk, 1.0), &[1, g, gk]);
    assert!(
        max_abs_err(
            &read_f32(&grouped_mm_t_mgh(&gsym, &gx1, 4).unwrap()),
            &read_f32(&grouped_mm_t_mgh(&gqmm, &gx1, 4).unwrap())
        ) <= 1e-5
    );
    check_grouped_decode(
        &mut rng,
        &gsym,
        (g, gk, gn),
        &[1, 2, 3, 4, 8, 12, 16],
        true,
        "sym_grp",
    );
    for &(g, k, n) in &[
        (16usize, 192usize, 512usize),
        (32, 512, 256),
        (4, 192, 37),
        (4, 512, 19),
    ] {
        let w = make_sym_q8(&mut rng, g * n, k, 1.0 / (k as f32).sqrt());
        check_grouped_decode(&mut rng, &w, (g, k, n), &[1, 2, 3, 4], true, "sym_short");
    }
    for &(g, k, n) in &[(16usize, 192usize, 512usize), (16, 512, 256)] {
        let w = make_qmm(&mut rng, g * n, k, 1.5 / (k as f32).sqrt());
        check_grouped_decode(
            &mut rng,
            &w,
            (g, k, n),
            &[1, 2, 4, 12, 16],
            false,
            "qmm_grp",
        );
    }
}

fn check_q8_dense(rng: &mut Rng, (m, k, n): (usize, usize, usize), check_fb: bool) {
    let sym = make_sym_q8(rng, n, k, 1.0 / (k as f32).sqrt());
    let qmm = sym_to_explicit_qmm(&sym);
    let x = arr(&rng.vec_f32(m * k, 1.0), &[m, k]);
    let y_ref = read_f32(
        &x.matmul(&sym.f32().unwrap().transpose(Some(&[1, 0])).unwrap())
            .unwrap(),
    );
    let tol = q8_prefill_tol(false);
    assert!(
        rel_l2_err(&read_f32(&ql_nax(true, &x, &sym, false)), &y_ref) <= tol,
        "dense SymQ8 ({m},{k},{n})"
    );
    assert!(
        rel_l2_err(&read_f32(&ql_nax(true, &x, &qmm, false)), &y_ref) <= 5e-6,
        "dense Qmm ({m},{k},{n})"
    );
    if check_fb {
        assert!(
            rel_l2_err(&read_f32(&ql_nax(false, &x, &sym, false)), &y_ref) < 1e-4,
            "dense fb ({m},{k},{n})"
        );
    }
}

fn check_q8_grouped(rng: &mut Rng, (m, g, k, n): (usize, usize, usize, usize), check_qmm: bool) {
    let sym = make_sym_q8(rng, g * n, k, 1.0 / (k as f32).sqrt());
    let x = arr(&rng.vec_f32(m * g * k, 1.0), &[m, g, k]);
    let y_ref = grouped_ref(&sym, &x, g, k, n);
    let tol = q8_prefill_tol(true);
    assert!(
        rel_l2_err(&read_f32(&grp_nax(true, &sym, &x, g, false)), &y_ref) <= tol,
        "grp SymQ8 ({m},{g},{k},{n})"
    );
    if check_qmm {
        let qmm = sym_to_explicit_qmm(&sym);
        assert!(
            rel_l2_err(&read_f32(&grp_nax(true, &qmm, &x, g, false)), &y_ref) <= 5e-6,
            "grp Qmm ({m},{g},{k},{n})"
        );
    }
}

fn assert_f16_io(
    x16: &MxArray,
    x32: &MxArray,
    check_32_16: bool,
    run: impl Fn(&MxArray, bool) -> MxArray,
    tag: &str,
) -> MxArray {
    let (y16, y32) = (run(x16, true), run(x32, false));
    assert_eq!(
        (y16.dtype().unwrap(), y32.dtype().unwrap()),
        (DType::Float16, DType::Float32)
    );
    let want16 = read_bits(&as_dt(&y32, DType::Float16));
    assert_eq!(read_bits(&y16), want16, "{tag} f16->f16");
    if check_32_16 {
        let ym = run(x32, true);
        assert_eq!(
            (ym.dtype().unwrap(), read_bits(&ym)),
            (DType::Float16, want16),
            "{tag} f32->f16"
        );
    }
    let y_in = run(x16, false);
    assert_eq!(
        (y_in.dtype().unwrap(), read_bits(&y_in)),
        (DType::Float32, read_bits(&y32)),
        "{tag} f16->f32"
    );
    y32
}

fn assert_no_metal_when_none(what: &str) {
    assert!(
        !unsafe { mlx_sys::mlx_metal_is_available() }
            && std::env::var("MLX_TEST_REQUIRE_NAX").as_deref() != Ok("1"),
        "expected {what} when Metal is available"
    );
}

#[test]
fn q8_dense_and_grouped_prefill_nax_parity() {
    crate::models::hy_v4_flash::exact_f32();
    let mut rng = Rng::new(800);
    let dense_shapes: &[(usize, usize, usize)] = &[
        (17, 64, 48),
        (45, 192, 96),
        (77, 512, 128),
        (64, 192, 128),
        (64, 512, 96),
        (256, 1024, 256),
        (512, 1024, 192),
        (64, 4096, 128),
        (128, 2048, 384),
    ];
    for &s in dense_shapes {
        check_q8_dense(&mut rng, s, s.0 == 64 && s.1 == 192);
    }
    if std::env::var("MLX_HY4_PREFILL_Q8_GROUPED_NAX").as_deref() != Ok("0")
        && std::env::var("MLX_HY4_PREFILL_Q8_NAX").as_deref() != Ok("0")
    {
        assert!(
            prefill_q8_grouped_nax_enabled(),
            "grouped Q8 NAX should be enabled by default"
        );
    }
    let grp_shapes: &[(usize, usize, usize, usize)] = &[
        (17, 4, 192, 512),
        (64, 4, 192, 512),
        (256, 4, 192, 512),
        (512, 2, 192, 512),
        (64, 4, 512, 256),
        (512, 2, 512, 256),
    ];
    for &s in grp_shapes {
        check_q8_grouped(&mut rng, s, s.0 == 64);
    }
}

#[test]
fn q8_prefill_strided_epilogue_and_f16_io_parity() {
    crate::models::hy_v4_flash::exact_f32();
    let mut rng = Rng::new(1400);

    // 1. Strided 2D dense + sigmoid_mul epilogue.
    for &(m, k, n) in &[
        (17usize, 192usize, 96usize),
        (64, 512, 256),
        (77, 512, 160),
        (256, 1024, 256),
    ] {
        let sym = make_sym_q8(&mut rng, n, k, 1.0 / (k as f32).sqrt());
        let xs = slice_axis(
            &arr(&rng.vec_f32(m * (k + 128), 1.0), &[m, k + 128]),
            true,
            64,
            64 + k,
        );
        let xc = arr(&read_f32(&xs), &[m, k]);
        let yc = ql_nax(true, &xc, &sym, false);
        assert_eq!(read_bits(&ql_nax(true, &xs, &sym, false)), read_bits(&yc));
        let aux = arr(&rng.vec_f32(m * n, 1.0), &[m, n]);
        if let Some(fused) =
            with_prefill_q8_nax(true, || try_qlinear_sigmoid_mul(&xs, &sym, &aux).unwrap())
        {
            let want = Activations::sigmoid_mul_compiled(&yc, &aux).unwrap();
            assert!(rel_l2_err(&read_f32(&fused), &read_f32(&want)) <= 1e-6);
        } else {
            assert_no_metal_when_none("q8_dense_prefill_epilogue");
        }
    }

    // 2. Strided grouped `q_nope` + gated epilogue (`v_b` * `sigmoid(gate_lin)`).
    for &(m, g, k, n) in &[(17usize, 4usize, 192usize, 128usize), (64, 4, 512, 128)] {
        let sym = make_sym_q8(&mut rng, g * n, k, 1.0 / (k as f32).sqrt());
        let xs = slice_axis(
            &arr(&rng.vec_f32(m * g * (k + 64), 1.0), &[m, g, k + 64]),
            true,
            0,
            k,
        );
        let xc = arr(&read_f32(&xs), &[m, g, k]);
        let yc = grp_nax(true, &sym, &xc, g, false);
        assert_eq!(
            read_bits(&grp_nax(true, &sym, &xs, g, false)),
            read_bits(&yc)
        );

        let width = g * n;
        let gl_s = slice_axis(
            &arr(&rng.vec_f32(m * (width + 96), 1.0), &[m, width + 96]),
            true,
            96,
            96 + width,
        );
        let (qw, sc, bi) = sym.qmm_parts().unwrap();
        let bp = bi.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
        let h = unsafe {
            mlx_sys::mlx_hy4_q8_grouped_prefill_gated(
                xs.as_raw_ptr(),
                qw.as_raw_ptr(),
                sc.as_raw_ptr(),
                bp,
                gl_s.as_raw_ptr(),
                false,
            )
        };
        if !h.is_null() {
            let gated = MxArray::from_handle(h, "test q8_grouped_prefill_gated").unwrap();
            let want = Activations::sigmoid_mul_compiled(
                &gl_s,
                &yc.reshape(&[m as i64, width as i64]).unwrap(),
            )
            .unwrap();
            assert!(rel_l2_err(&read_f32(&gated), &read_f32(&want)) <= 1e-6);
        } else {
            assert_no_metal_when_none("q8_grouped_prefill_gated");
        }
    }

    // 3. Dense FP16 IO + FP16 epilogue across NAX and fallback.
    for &(m, k, n) in &[
        (4usize, 192usize, 96usize),
        (12, 192, 96),
        (17, 192, 96),
        (77, 512, 256),
        (96, 512, 384),
    ] {
        let sym = make_sym_q8(&mut rng, n, k, 1.0 / (k as f32).sqrt());
        let x16 = as_dt(&arr(&rng.vec_f32(m * k, 1.0), &[m, k]), DType::Float16);
        let x32 = as_dt(&x16, DType::Float32);
        for nax in [true, false] {
            let y32 = assert_f16_io(
                &x16,
                &x32,
                true,
                |x, f16| ql_nax(nax, x, &sym, f16),
                &format!("dense nax={nax} ({m},{k},{n})"),
            );
            if nax && m >= 12 {
                let aux16 = as_dt(&arr(&rng.vec_f32(m * n, 1.0), &[m, n]), DType::Float16);
                if let Some(fused) = with_prefill_q8_nax(true, || {
                    try_qlinear_sigmoid_mul_out(&x32, &sym, &aux16, true).unwrap()
                }) {
                    assert_eq!(fused.dtype().unwrap(), DType::Float16);
                    let want = as_dt(
                        &Activations::sigmoid_mul_compiled(&y32, &as_dt(&aux16, DType::Float32))
                            .unwrap(),
                        DType::Float16,
                    );
                    assert!(rel_l2_err(&read_f32(&fused), &read_f32(&want)) <= 1e-3);
                } else {
                    assert_no_metal_when_none("FP16 epilogue");
                }
            }
        }
    }

    // 4. Strided grouped FP16 IO across decode & prefill, NAX & fallback.
    for &(m, g, k, n) in &[(4usize, 4usize, 192usize, 128usize), (77, 4, 192, 512)] {
        let sym = make_sym_q8(&mut rng, g * n, k, 1.0 / (k as f32).sqrt());
        let wide = as_dt(
            &arr(&rng.vec_f32(m * g * (k + 64), 1.0), &[m, g, k + 64]),
            DType::Float16,
        );
        let x16 = slice_axis(&wide, true, 0, k);
        let x32 = as_dt(&x16, DType::Float32);
        for nax in [true, false] {
            assert_f16_io(
                &x16,
                &x32,
                false,
                |x, f16| grp_nax(nax, &sym, x, g, f16),
                &format!("grp nax={nax} M={m}"),
            );
        }
    }

    // 5. Unevaluated 1-column-offset lazy FP16 slice (2-byte offset, odd stride).
    let sym_mis = make_sym_q8(&mut rng, 64, 64, 0.125);
    let lazy_mis = slice_axis(
        &as_dt(&arr(&rng.vec_f32(17 * 65, 1.0), &[17, 65]), DType::Float16),
        true,
        1,
        65,
    );
    let y_lazy = ql_nax(true, &lazy_mis, &sym_mis, true);
    let contig_mis = as_dt(&arr(&read_f32(&lazy_mis), &[17, 64]), DType::Float16);
    assert_eq!(
        read_bits(&y_lazy),
        read_bits(&ql_nax(true, &contig_mis, &sym_mis, true)),
        "lazy misaligned FP16"
    );
}

#[cfg(unix)]
struct TempMappedBuf {
    path: PathBuf,
    ptr: *mut std::ffi::c_void,
    len: usize,
}

#[cfg(unix)]
impl TempMappedBuf {
    fn create(name: &str, byte_offset: usize, payload: &[u8]) -> (Self, MxArray) {
        use std::io::Write;
        use std::os::fd::AsRawFd;
        let page = crate::array::page_size();
        let total = (byte_offset + payload.len()).div_ceil(page).max(1) * page;
        let path = std::env::temp_dir().join(format!(
            "hy4-weights-mmap-{name}-{}.bin",
            uuid::Uuid::new_v4()
        ));
        let mut bytes = vec![0xA5u8; total];
        bytes[byte_offset..byte_offset + payload.len()].copy_from_slice(payload);
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&bytes).unwrap();
            f.sync_all().unwrap();
        }
        let f = std::fs::File::open(&path).unwrap();
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                total,
                libc::PROT_READ,
                libc::MAP_SHARED,
                f.as_raw_fd(),
                0,
            )
        };
        assert_ne!(ptr, libc::MAP_FAILED);
        unsafe extern "C" fn noop(_: *mut std::ffi::c_void) {}
        let buf =
            unsafe { MxArray::from_mmap_readonly(ptr, total, noop, std::ptr::null_mut()).unwrap() };
        (
            Self {
                path,
                ptr,
                len: total,
            },
            buf,
        )
    }
}

#[cfg(unix)]
impl Drop for TempMappedBuf {
    fn drop(&mut self) {
        crate::array::synchronize();
        unsafe {
            let _ = libc::munmap(self.ptr, self.len);
        }
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(unix)]
#[test]
fn mmap_bf16_lm_head_and_q8_0_embed_match_dense_bit_exact() {
    use super::model::head_rows;
    use super::ops::{head_logits, head_logits_chunked};
    use super::weights::{quantize_sym_q8_array, quantize_sym_q8_from_bf16_raw};
    use crate::nn::Embedding;

    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        return;
    }
    crate::models::hy_v4_flash::exact_f32();
    let mut rng = Rng::new(2000);
    let check_meta = |d: &Dense, r: usize, c: usize| {
        assert_eq!(
            (d.nbytes(), d.shape_at(0).unwrap(), d.shape_at(1).unwrap()),
            (0, r as i64, c as i64)
        );
    };

    // 1. Mmap BF16 LM head: FAST_K (256) and non-FAST_K (64), non-zero offsets, requant.
    for (ci, &(vocab, k, off)) in [(67usize, 256usize, 64usize), (53, 64, 128)]
        .iter()
        .enumerate()
    {
        let raw: Vec<u8> = (0..vocab * k)
            .map(|i| (((((i * 37 + ci * 13) % 101) as f32 - 50.0) * 0.015).to_bits() >> 16) as u16)
            .flat_map(u16::to_le_bytes)
            .collect();
        let dense_arr = dense_from_raw(
            &info(OUTPUT, &[k as u64, vocab as u64], GgufTensorType::BF16),
            &raw,
            Form::Bf16,
        )
        .unwrap();
        let (_m, buf) = TempMappedBuf::create(&format!("bf16-{ci}"), off, &raw);
        let dense_mmap = Dense::MmapBf16 {
            buf,
            byte_offset: off as u64,
            rows: vocab as i64,
            cols: k as i64,
        };
        check_meta(&dense_mmap, vocab, k);

        for &m in &[1usize, 2, 7, 12, 16, 24] {
            let x = arr(&rng.vec_f32(m * k, 1.0), &[m, k]);
            assert_eq!(
                read_bits(&head_logits(&x, &dense_mmap).unwrap()),
                read_bits(&head_logits(&x, &dense_arr).unwrap())
            );
            assert_eq!(
                read_bits(&head_rows(&x, &dense_mmap).unwrap()),
                read_bits(&head_rows(&x, &dense_arr).unwrap())
            );
            if m == 24 {
                assert_eq!(
                    read_bits(&head_logits_chunked(&x, &dense_mmap, 17).unwrap()),
                    read_bits(&head_logits_chunked(&x, &dense_arr, 17).unwrap())
                );
            }
        }
        let Dense::Array(ref bf16_arr) = dense_arr else {
            panic!("expected Dense::Array")
        };
        let (raw_slice, r, c) = dense_mmap.mmap_bf16_raw_slice().unwrap().unwrap();
        let q_raw = quantize_sym_q8_from_bf16_raw(raw_slice, r, c)
            .unwrap()
            .f32()
            .unwrap();
        assert_eq!(
            read_bits(&q_raw),
            read_bits(&quantize_sym_q8_array(bf16_arr).unwrap().f32().unwrap())
        );
    }

    // 2. Mmap Q8_0 embedding with 2-byte aligned offset 34, OOB -> 0, MTP Embedding path.
    let (vocab, dim, off) = (29usize, 128usize, 34usize);
    let raw = q8_0_raw(vocab * dim);
    let mut exact = vec![0f32; vocab * dim];
    q8_0_into_f32(&raw, &mut exact).unwrap();
    let dense_q8 = dense_from_raw(
        &info(
            TOKEN_EMBD,
            &[dim as u64, vocab as u64],
            GgufTensorType::Q8_0,
        ),
        &raw,
        Form::Codes,
    )
    .unwrap();
    let (_m, buf) = TempMappedBuf::create("q8-0-embed", off, &raw);
    let dense_mmap = Dense::MmapQ8_0 {
        buf: buf.clone(),
        byte_offset: off as u64,
        rows: vocab as i64,
        cols: dim as i64,
    };
    check_meta(&dense_mmap, vocab, dim);
    assert_eq!(read_bits(&dense_mmap.f32().unwrap()), f32_bits(&exact));

    for ids in [
        &[0u32][..],
        &[28, 0, 17, 5, 17],
        &[3, 19, 0, 28, 12, 7, 1, 25, 14, 9, 28, 4],
    ] {
        let m = ids.len() as i64;
        let idx_u = MxArray::from_uint32(ids, &[m]).unwrap();
        let idx_i = i32_arr(&ids.iter().map(|&x| x as i32).collect::<Vec<_>>(), &[m]);
        let want = read_bits(&dense_q8.rows(&idx_u).unwrap());
        assert_eq!(read_bits(&dense_mmap.rows(&idx_u).unwrap()), want);
        assert_eq!(read_bits(&dense_mmap.rows(&idx_i).unwrap()), want);
    }

    let mut emb = Embedding::new_uninitialized(vocab as u32, dim as u32).unwrap();
    emb.load_gguf_q8_0_mmap(&buf, off as u64).unwrap();
    assert!(emb.is_quantized());
    let (idx1, idx2d, idx_flat) = (
        i32_arr(&[11], &[1]),
        i32_arr(&[0, 5, 28, 7, 13, 2], &[2, 3]),
        i32_arr(&[0, 5, 28, 7, 13, 2], &[6]),
    );
    let (got1, got2d) = (emb.forward(&idx1).unwrap(), emb.forward(&idx2d).unwrap());
    assert_eq!(got1.shape().unwrap().as_ref(), &[1, dim as i64]);
    assert_eq!(read_bits(&got1), read_bits(&dense_q8.rows(&idx1).unwrap()));
    assert_eq!(got2d.shape().unwrap().as_ref(), &[2, 3, dim as i64]);
    assert_eq!(
        read_bits(&got2d),
        read_bits(&dense_q8.rows(&idx_flat).unwrap())
    );
    assert_eq!(
        read_f32(&emb.forward(&i32_arr(&[-1, vocab as i32], &[2])).unwrap()),
        vec![0.0f32; 2 * dim]
    );
    assert!(emb.as_linear(&arr(&vec![0.0; dim], &[1, dim])).is_err());
}

#[test]
fn dense_qmm4_loader_and_row_invariant() {
    crate::models::hy_v4_flash::exact_f32();
    let mut rng = Rng::new(3100);
    for &(rows, k) in &[(35usize, 64usize), (47, 192), (51, 512)] {
        let bf16_u16: Vec<u16> = rng
            .vec_f32(rows * k, 0.5)
            .into_iter()
            .map(|v| (v.to_bits() >> 16) as u16)
            .collect();
        let bf16_raw: Vec<u8> = bf16_u16.iter().flat_map(|w| w.to_le_bytes()).collect();
        let bf16_f32: Vec<f32> = bf16_u16
            .iter()
            .map(|&w| f32::from_bits(u32::from(w) << 16))
            .collect();
        let q4_raw = quantize_affine4_from_bf16_raw(&bf16_raw, rows, k).unwrap();
        let q4_arr =
            quantize_affine4_array(&as_dt(&arr(&bf16_f32, &[rows, k]), DType::BFloat16)).unwrap();
        assert_eq!(
            read_bits(&q4_raw.f32().unwrap()),
            read_bits(&q4_arr.f32().unwrap())
        );

        let x_all = arr(&rng.vec_f32(4 * k, 1.0), &[4, k]);
        assert_row_invariant(
            |r| {
                let xr = slice_axis(&x_all, false, r, r + 1);
                let y = q4_raw.mm_t(&xr).unwrap();
                assert_eq!(read_bits(&y), read_bits(&q4_arr.mm_t(&xr).unwrap()));
                y
            },
            |m| q4_raw.mm_t(&slice_axis(&x_all, false, 0, m)).unwrap(),
            &[2, 4],
            &format!("Dense::Qmm4 ({rows},{k})"),
        );
    }
}
