use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::scan::{Role, Scan};
use super::weights::{
    Dense, Form, OUTPUT, Resident, Residual, TOKEN_EMBD, bf16_into_f32, dense_from_raw, form,
    held_shape, le_f32_into, le_u16_into, q8_0_into_f32, resident_bytes, role_resident,
};
use crate::array::{DType, MxArray};
use crate::models::deepseek_v41::store::Weight;
use crate::moe_stream::io::ShardSet;
use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};

fn info(name: &str, dims: &[u64], ty: GgufTensorType) -> GgufTensorInfo {
    GgufTensorInfo {
        name: name.to_string(),
        n_dims: dims.len() as u32,
        dims: dims.to_vec(),
        tensor_type: ty,
        offset: 0,
    }
}

/// Deterministic values in [-1, 1).
fn lcg(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Q8_0 blocks for `n` values: block `b` has scale 2^-(b % 7 + 3) and codes
/// running over the full i8 range.
fn q8_0_raw(n: usize) -> Vec<u8> {
    let mut raw = Vec::with_capacity(n / 32 * 34);
    for b in 0..n / 32 {
        let d = half::f16::from_f32(2f32.powi(-((b % 7) as i32 + 3)));
        raw.extend(d.to_bits().to_le_bytes());
        for i in 0..32 {
            let q = ((b * 37 + i * 11) % 256) as i32 - 128;
            raw.extend((q as i8).to_le_bytes());
        }
    }
    raw
}

fn read(x: &MxArray) -> Vec<f32> {
    x.astype(DType::Float32)
        .unwrap()
        .to_float32()
        .unwrap()
        .as_ref()
        .to_vec()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (num, den) = got.iter().zip(want).fold((0f64, 0f64), |(n, d), (a, b)| {
        (n + f64::from(a - b).powi(2), d + f64::from(*b).powi(2))
    });
    (num / den).sqrt()
}

#[test]
fn le_decoders_keep_byte_order() {
    let mut w = [0u16; 2];
    le_u16_into(&[0x34, 0x12, 0x80, 0x3f], &mut w).unwrap();
    assert_eq!(w, [0x1234, 0x3f80]);
    let mut f = [0f32; 2];
    bf16_into_f32(&[0x80, 0x3f, 0x40, 0xc0], &mut f).unwrap();
    assert_eq!(f, [1.0, -3.0]);
    let raw: Vec<u8> = [1.5f32, -0.25]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    le_f32_into(&raw, &mut f).unwrap();
    assert_eq!(f, [1.5, -0.25]);
    assert!(le_u16_into(&[0; 3], &mut w).is_err());
    assert!(bf16_into_f32(&[0; 2], &mut f).is_err());
    assert!(le_f32_into(&raw[..7], &mut f).is_err());
}

#[test]
fn q8_0_decodes_to_scale_times_code() {
    let raw = q8_0_raw(64);
    let mut out = vec![0f32; 64];
    q8_0_into_f32(&raw, &mut out).unwrap();
    for b in 0..2 {
        let d = 2f32.powi(-(b as i32 + 3));
        for i in 0..32 {
            let q = ((b * 37 + i * 11) % 256) as i32 - 128;
            assert_eq!(
                out[b * 32 + i].to_bits(),
                (d * q as f32).to_bits(),
                "{b} {i}"
            );
        }
    }
    assert!(q8_0_into_f32(&raw[..67], &mut out).is_err());
    assert!(q8_0_into_f32(&raw, &mut out[..32]).is_err());
}

#[test]
fn forms_follow_role_and_type() {
    use GgufTensorType::{BF16, F32, Q4K, Q8_0};
    let cases: [(Role, &str, GgufTensorType, Option<Form>); 15] = [
        (Role::Dense, TOKEN_EMBD, Q8_0, Some(Form::Codes)),
        (Role::Dense, "blk.1.attn_q_b.weight", Q8_0, Some(Form::Qmm)),
        (
            Role::Dense,
            "blk.1.indexer_compressor_value.weight",
            Q8_0,
            Some(Form::F32),
        ),
        (
            Role::Dense,
            "blk.1.indexer_compressor_gate.weight",
            Q8_0,
            Some(Form::F32),
        ),
        (
            Role::Dense,
            "blk.1.xindexer_compressor_gate.weight",
            Q8_0,
            Some(Form::Qmm),
        ),
        (Role::Dense, "blk.1.attn_norm.weight", F32, Some(Form::F32)),
        (
            Role::Dense,
            "blk.1.ffn_gate_inp.weight",
            F32,
            Some(Form::F32),
        ),
        (Role::Dense, OUTPUT, BF16, Some(Form::Bf16)),
        (Role::Dense, "blk.1.attn_q_b.weight", BF16, None),
        (Role::Dense, "blk.1.attn_q_b.weight", Q4K, None),
        (
            Role::Mtp,
            "blk.42.nextn.eh_proj.weight",
            BF16,
            Some(Form::Bf16),
        ),
        (
            Role::Mtp,
            "blk.42.indexer_compressor_value.weight",
            BF16,
            Some(Form::F32),
        ),
        (Role::Mtp, "blk.42.nextn.enorm.weight", F32, Some(Form::F32)),
        (Role::Expert, "blk.1.ffn_up_exps.weight", Q4K, None),
        (
            Role::EngramTable,
            "per_layer_token_embd.weight",
            GgufTensorType::MXFP8,
            None,
        ),
    ];
    for (role, name, ty, want) in cases {
        let got = form(role, &info(name, &[32, 4], ty)).ok();
        assert_eq!(got, want, "{role:?} {name} {ty:?}");
    }
}

#[test]
fn resident_bytes_follow_the_form() {
    use GgufTensorType::{BF16, F32, Q8_0};
    let cases: [(&str, &[u64], GgufTensorType, u64); 5] = [
        ("blk.0.attn_q_a.weight", &[64, 4], Q8_0, 256 + 16),
        (TOKEN_EMBD, &[64, 4], Q8_0, 256 + 16),
        (
            "blk.0.indexer_compressor_gate.weight",
            &[128, 128],
            Q8_0,
            65536,
        ),
        (OUTPUT, &[8, 16], BF16, 256),
        ("output_norm.weight", &[3], F32, 12),
    ];
    for (name, dims, ty, want) in cases {
        assert_eq!(
            resident_bytes(Role::Dense, &info(name, dims, ty)).unwrap(),
            want,
            "{name}"
        );
    }
}

#[test]
fn rank_three_tensors_fold_into_rows() {
    let k_b = info(
        "blk.0.attn_k_b.weight",
        &[192, 512, 32],
        GgufTensorType::Q8_0,
    );
    assert_eq!(held_shape(&k_b).unwrap(), vec![32 * 512, 192]);
    let v_b = info(
        "blk.0.attn_v_b.weight",
        &[512, 256, 64],
        GgufTensorType::Q8_0,
    );
    assert_eq!(held_shape(&v_b).unwrap(), vec![64 * 256, 512]);
    let m = info(
        "blk.0.attn_kv_a_mqa.weight",
        &[3072, 576],
        GgufTensorType::Q8_0,
    );
    assert_eq!(held_shape(&m).unwrap(), vec![576, 3072]);
    let v = info("output_norm.weight", &[3072], GgufTensorType::F32);
    assert_eq!(held_shape(&v).unwrap(), vec![3072]);
}

/// A rank-3 Q8_0 `k_b` as `Qmm`: the f32 form is the exact `d·q`, and
/// `grouped_qmm_t(x, H)` == per-head `x_h @ W_h^T` in f32.
#[test]
fn rank_three_q8_qmm_groups_by_head() {
    let (k, rows, h, t) = (64usize, 8usize, 3usize, 5usize);
    let i = info(
        "blk.0.attn_k_b.weight",
        &[k as u64, rows as u64, h as u64],
        GgufTensorType::Q8_0,
    );
    let raw = q8_0_raw(k * rows * h);
    let d = dense_from_raw(&i, &raw, Form::Qmm).unwrap();
    assert!(matches!(d, Dense::SymQ8 { .. }));
    assert_eq!(d.nbytes() as u64, resident_bytes(Role::Dense, &i).unwrap());
    assert_eq!(d.shape_at(0).unwrap(), (h * rows) as i64);
    assert_eq!(d.shape_at(1).unwrap(), k as i64);
    let mut exact = vec![0f32; k * rows * h];
    q8_0_into_f32(&raw, &mut exact).unwrap();
    assert_eq!(bits(&read(&d.f32().unwrap())), bits(&exact));
    let xs = lcg(7, h * t * k);
    let x = MxArray::from_float32(&xs, &[h as i64, t as i64, k as i64]).unwrap();
    let got = read(&d.grouped_qmm_t(&x, h as i64).unwrap().unwrap());
    let mut want = vec![0f32; h * t * rows];
    for g in 0..h {
        for ti in 0..t {
            for r in 0..rows {
                let w = &exact[(g * rows + r) * k..(g * rows + r + 1) * k];
                let xv = &xs[(g * t + ti) * k..(g * t + ti + 1) * k];
                let dot: f64 = w
                    .iter()
                    .zip(xv)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum();
                want[(g * t + ti) * rows + r] = dot as f32;
            }
        }
    }
    let rel = rel_l2(&got, &want);
    assert!(rel < 1e-5, "grouped Qmm relL2 {rel:e}");
}

/// `Codes` keeps f16 scales and gathers exact rows; `F32` and `Bf16` hold
/// the file values.
#[test]
fn codes_f32_and_bf16_forms_hold_the_file_values() {
    let (k, rows) = (64usize, 6usize);
    let raw = q8_0_raw(k * rows);
    let mut exact = vec![0f32; k * rows];
    q8_0_into_f32(&raw, &mut exact).unwrap();
    let emb = info(TOKEN_EMBD, &[k as u64, rows as u64], GgufTensorType::Q8_0);
    let codes = dense_from_raw(&emb, &raw, Form::Codes).unwrap();
    assert!(matches!(codes, Dense::Q8 { .. }));
    assert_eq!(
        codes.nbytes() as u64,
        resident_bytes(Role::Dense, &emb).unwrap()
    );
    let idx = MxArray::from_int32(&[4, 1], &[2]).unwrap();
    let got = read(&codes.rows(&idx).unwrap());
    let want: Vec<f32> = [4, 1]
        .iter()
        .flat_map(|&r| exact[r * k..(r + 1) * k].to_vec())
        .collect();
    assert_eq!(bits(&got), bits(&want));

    let comp = info(
        "blk.0.indexer_compressor_value.weight",
        &[k as u64, rows as u64],
        GgufTensorType::Q8_0,
    );
    let f = dense_from_raw(&comp, &raw, Form::F32)
        .unwrap()
        .array()
        .unwrap();
    assert_eq!(f.dtype().unwrap(), DType::Float32);
    assert_eq!(bits(&read(&f)), bits(&exact));

    let words: Vec<u16> = (0..k * rows)
        .map(|i| 0x3f80 ^ ((i as u16 * 97) & 0x80ff))
        .collect();
    let braw: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let out = info(OUTPUT, &[k as u64, rows as u64], GgufTensorType::BF16);
    let b = dense_from_raw(&out, &braw, Form::Bf16)
        .unwrap()
        .array()
        .unwrap();
    assert_eq!(b.dtype().unwrap(), DType::BFloat16);
    assert_eq!(b.to_uint16_native().unwrap(), words);
    let bf = dense_from_raw(&out, &braw, Form::F32)
        .unwrap()
        .array()
        .unwrap();
    let mut wide = vec![0f32; k * rows];
    bf16_into_f32(&braw, &mut wide).unwrap();
    assert_eq!(bits(&read(&bf)), bits(&wide));

    let fraw: Vec<u8> = lcg(3, 12).iter().flat_map(|x| x.to_le_bytes()).collect();
    let norm = info("output_norm.weight", &[12], GgufTensorType::F32);
    let n = dense_from_raw(&norm, &fraw, Form::F32)
        .unwrap()
        .array()
        .unwrap();
    let mut nv = vec![0f32; 12];
    le_f32_into(&fraw, &mut nv).unwrap();
    assert_eq!(bits(&read(&n)), bits(&nv));

    assert!(dense_from_raw(&emb, &raw[..raw.len() - 1], Form::Codes).is_err());
    assert!(dense_from_raw(&out, &braw, Form::Qmm).is_err());

    let rvals = lcg(9, 128);
    let rraw: Vec<u8> = rvals.iter().flat_map(|x| x.to_le_bytes()).collect();
    let rinfo = info("blk.1.ffn_gate_inp.weight", &[32, 4], GgufTensorType::F32);
    let rf16 = dense_from_raw(&rinfo, &rraw, Form::F16)
        .unwrap()
        .array()
        .unwrap();
    assert_eq!(rf16.dtype().unwrap(), DType::Float16);
    assert_eq!(rf16.nbytes(), 128 * 2);
    let want_f16: Vec<f32> = rvals
        .iter()
        .map(|&v| half::f16::from_f32(v).to_f32())
        .collect();
    assert_eq!(bits(&read(&rf16)), bits(&want_f16));
}

fn fixture(name: &str) -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/hy-v4-flash")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap()
}

fn real_scan() -> Scan {
    let dir = std::env::var("HY4_DIR").expect("HY4_DIR = directory of the two Hy4 shards");
    let paths = crate::models::deepseek_v41::chat::shard_paths(Path::new(&dir)).unwrap();
    assert_eq!(paths.len(), 2);
    Scan::open(&paths).unwrap()
}

/// `resident.json` `by_form` key of a trunk tensor.
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

/// Headers only: resident bytes per form == the oracle's, and the MTP
/// layer's non-expert bytes == the file bytes with the two compressor
/// matrices widened from bf16 to f32.
#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards"]
fn real_resident_bytes_match_the_oracle() {
    let s = real_scan();
    let want = fixture("resident.json");
    let trunk = &want["trunk_dense"];
    assert_eq!(
        role_resident(&s, Role::Dense).unwrap(),
        trunk["resident_bytes"].as_u64().unwrap()
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
        let got = by[k.as_str()];
        let exp = (
            v["tensors"].as_u64().unwrap(),
            v["file_bytes"].as_u64().unwrap(),
            v["resident_bytes"].as_u64().unwrap(),
        );
        assert_eq!(got, exp, "{k}");
    }
    let mut mtp_want = 0u64;
    for t in want["mtp_blk"]["list"].as_array().unwrap() {
        let name = t["name"].as_str().unwrap();
        if name.ends_with("_exps.weight") {
            continue;
        }
        let b = t["bytes"].as_u64().unwrap();
        mtp_want += if name.contains("indexer_compressor_value")
            || name.contains("indexer_compressor_gate")
        {
            2 * b
        } else {
            b
        };
    }
    assert_eq!(role_resident(&s, Role::Mtp).unwrap(), mtp_want);
}

/// Real dense load: bytes read and held == the oracle; `k_b` of an indexed
/// (H 32) and a sliding (H 64) layer grouped == per-head f32 matmul; the
/// MTP layer loads on request with the planned bytes.
#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards and the GPU"]
fn real_dense_load_matches_the_oracle() {
    crate::models::hy_v4_flash::exact_f32();
    let s = real_scan();
    let want = fixture("resident.json");
    let shards = ShardSet::open(&s.paths).unwrap();
    let mut r = Resident::load(&s, &shards).unwrap();
    let trunk = &want["trunk_dense"];
    assert_eq!(r.read, trunk["file_bytes"].as_u64().unwrap());
    assert_eq!(r.bytes, trunk["resident_bytes"].as_u64().unwrap());
    assert_eq!(r.layers.len(), s.cfg.n_layers);
    assert!(r.mtp.is_empty());
    let (kv_lora, nope) = (s.cfg.mla.kv_lora_rank, s.cfg.mla.qk_nope_dim);
    for il in [0usize, 3] {
        let h = s.cfg.layers[il].n_heads;
        let k_b = &r.layers[il].attn.k_b;
        assert_eq!(k_b.shape_at(0).unwrap(), (h * kv_lora) as i64);
        assert_eq!(k_b.shape_at(1).unwrap(), nope as i64);
        let t = 4usize;
        let xs = lcg(il as u64 + 11, h * t * nope);
        let x = MxArray::from_float32(&xs, &[h as i64, t as i64, nope as i64]).unwrap();
        let got = read(&k_b.grouped_qmm_t(&x, h as i64).unwrap().unwrap());
        let mut want_v = Vec::with_capacity(got.len());
        for g in 0..h {
            let lo = (g * kv_lora) as i64;
            let w = k_b
                .row_range(lo, lo + kv_lora as i64)
                .unwrap()
                .f32()
                .unwrap();
            let xg = x
                .slice(&[g as i64, 0, 0], &[g as i64 + 1, t as i64, nope as i64])
                .unwrap();
            let xg = xg.reshape(&[t as i64, nope as i64]).unwrap();
            want_v.extend(read(
                &xg.matmul(&w.transpose(Some(&[1, 0])).unwrap()).unwrap(),
            ));
        }
        let rel = rel_l2(&got, &want_v);
        println!("layer {il} k_b grouped relL2 {rel:e}");
        assert!(rel < 1e-5, "layer {il}: {rel}");
    }
    let before = r.bytes;
    r.load_mtp(&s, &shards).unwrap();
    assert_eq!(r.mtp.len(), s.cfg.layers.len() - s.cfg.n_layers);
    assert!(matches!(r.mtp[0].residual, Residual::Mtp(_)));
    assert_eq!(r.bytes - before, role_resident(&s, Role::Mtp).unwrap());
    let il = s.cfg.n_layers;
    let ex = s.experts[il].as_ref().unwrap();
    let mut reader = super::weights::Reader::new(&s, &shards);
    let slab = reader.bf16_experts(&ex.gate, 0, 2).unwrap();
    let (ff, latent) = (s.cfg.moe.ff as i64, s.cfg.moe.latent as i64);
    assert_eq!(slab.shape_at(0).unwrap(), 2);
    assert_eq!(slab.shape_at(1).unwrap(), ff);
    assert_eq!(slab.shape_at(2).unwrap(), latent);
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
    use super::moe::qlinear;
    use super::weights::fuse_qmm_rows;

    let k = 64usize;
    let make_qmm = |rows: usize, seed: u64| -> Dense {
        let n = rows * k;
        let mut raw = Vec::with_capacity(n / 32 * 34);
        let vals = lcg(seed, n);
        for blk in vals.as_chunks::<32>().0 {
            let amax = blk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let d = half::f16::from_f32((amax / 127.0).max(1e-6));
            let df = d.to_f32();
            raw.extend(d.to_bits().to_le_bytes());
            for &v in blk {
                let q = (v / df).round().clamp(-127.0, 127.0) as i8;
                raw.extend(q.to_le_bytes());
            }
        }
        let inf = info("w", &[k as u64, rows as u64], GgufTensorType::Q8_0);
        dense_from_raw(&inf, &raw, Form::Qmm).unwrap()
    };

    let (r0, r1, r2) = (32usize, 48usize, 64usize);
    let a = make_qmm(r0, 101);
    let b = make_qmm(r1, 202);
    let c = make_qmm(r2, 303);

    // Non-Qmm returns None
    let plain =
        Dense::Array(MxArray::from_float32(&lcg(404, r0 * k), &[r0 as i64, k as i64]).unwrap());
    assert!(fuse_qmm_rows(&[&a, &plain]).unwrap().is_none());
    assert!(fuse_qmm_rows(&[&a]).unwrap().is_none());

    let (fused, views) = fuse_qmm_rows(&[&a, &b, &c]).unwrap().unwrap();
    assert_eq!(fused.shape_at(0).unwrap(), (r0 + r1 + r2) as i64);
    assert_eq!(fused.shape_at(1).unwrap(), k as i64);
    assert_eq!(views.len(), 3);
    assert_eq!(views[0].shape_at(0).unwrap(), r0 as i64);
    assert_eq!(views[1].shape_at(0).unwrap(), r1 as i64);
    assert_eq!(views[2].shape_at(0).unwrap(), r2 as i64);

    for m in [1usize, 2, 4, 12, 256] {
        let xs = lcg(500 + m as u64, m * k);
        let x = MxArray::from_float32(&xs, &[m as i64, k as i64]).unwrap();

        let ya = read(&qlinear(&x, &a).unwrap());
        let yb = read(&qlinear(&x, &b).unwrap());
        let yc = read(&qlinear(&x, &c).unwrap());

        let y_all = qlinear(&x, &fused).unwrap();
        let (r0i, r1i, r2i) = (r0 as i64, r1 as i64, r2 as i64);
        let fa = read(&y_all.slice(&[0, 0], &[m as i64, r0i]).unwrap());
        let fb = read(&y_all.slice(&[0, r0i], &[m as i64, r0i + r1i]).unwrap());
        let fc = read(
            &y_all
                .slice(&[0, r0i + r1i], &[m as i64, r0i + r1i + r2i])
                .unwrap(),
        );

        assert_eq!(bits(&fa), bits(&ya), "M={m} fused slice 0");
        assert_eq!(bits(&fb), bits(&yb), "M={m} fused slice 1");
        assert_eq!(bits(&fc), bits(&yc), "M={m} fused slice 2");

        // Zero-copy views also match original separate matrices bit-for-bit
        assert_eq!(
            bits(&read(&qlinear(&x, &views[0]).unwrap())),
            bits(&ya),
            "M={m} view 0"
        );
        assert_eq!(
            bits(&read(&qlinear(&x, &views[1]).unwrap())),
            bits(&yb),
            "M={m} view 1"
        );
        assert_eq!(
            bits(&read(&qlinear(&x, &views[2]).unwrap())),
            bits(&yc),
            "M={m} view 2"
        );
    }
}

fn make_sym_q8(rows: usize, k: usize, seed: u64) -> Dense {
    let n = rows * k;
    let mut raw = Vec::with_capacity(n / 32 * 34);
    let vals = lcg(seed, n);
    let scale_norm = 1.0 / (k as f32).sqrt();
    for blk in vals.as_chunks::<32>().0 {
        let amax = blk
            .iter()
            .map(|v| (v * scale_norm).abs())
            .fold(0.0f32, f32::max);
        let d = half::f16::from_f32((amax / 127.0).max(1e-6));
        let df = d.to_f32();
        raw.extend(d.to_bits().to_le_bytes());
        for &v in blk {
            let q = ((v * scale_norm) / df).round().clamp(-127.0, 127.0) as i8;
            raw.extend(q.to_le_bytes());
        }
    }
    let inf = info("w", &[k as u64, rows as u64], GgufTensorType::Q8_0);
    dense_from_raw(&inf, &raw, Form::Qmm).unwrap()
}

fn sym_to_explicit_qmm(d: &Dense) -> Dense {
    let Dense::SymQ8 { w, scales } = d else {
        panic!("expected SymQ8");
    };
    let s = scales.astype(DType::Float32).unwrap();
    let b = s.mul_scalar(-128.0).unwrap();
    MxArray::eval_arrays_with_context(&[w, &s, &b], "test explicit qmm").unwrap();
    Dense::Qmm {
        w: w.clone(),
        scales: s,
        biases: b,
    }
}

#[test]
fn sym_q8_decode_row_invariant_and_matches_f32_ref() {
    use super::attention::grouped_mm_t_mgh;
    use super::moe::{qlinear, with_prefill_q8_nax};

    crate::models::hy_v4_flash::exact_f32();

    // Dense SymQ8: M = 1..=11 decode row-invariance and <= 1e-5 vs F32 ref;
    // M = 12..=16 routes to NAX prefill by default and stays row-invariant when NAX is disabled.
    let (k, n) = (256usize, 96usize);
    let sym = make_sym_q8(n, k, 701);
    assert!(matches!(sym, Dense::SymQ8 { .. }));
    assert_eq!(sym.nbytes(), n * k + (n * k) / 16);
    let qmm = sym_to_explicit_qmm(&sym);
    assert_eq!(qmm.nbytes(), n * k + (n * k) / 4);

    let wf = sym.f32().unwrap();
    let wt = wf.transpose(Some(&[1, 0])).unwrap();
    let all_xs = lcg(702, 16 * k);
    let mut single_bits = Vec::with_capacity(16);
    for r in 0..16 {
        let xr = MxArray::from_float32(&all_xs[r * k..(r + 1) * k], &[1, k as i64]).unwrap();
        let y_sym = read(&qlinear(&xr, &sym).unwrap());
        let y_qmm = read(&qlinear(&xr, &qmm).unwrap());
        let diff_qmm = y_sym
            .iter()
            .zip(&y_qmm)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff_qmm <= 1e-5, "M=1 row {r} SymQ8 vs Qmm {diff_qmm:e}");
        let y_ref = read(&xr.matmul(&wt).unwrap());
        let max_abs = y_sym
            .iter()
            .zip(&y_ref)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs <= 1e-5, "M=1 row {r} max_abs {max_abs:e}");
        single_bits.push(bits(&y_sym));
    }
    for m in 2..=11usize {
        let xm = MxArray::from_float32(&all_xs[..m * k], &[m as i64, k as i64]).unwrap();
        let ym = read(&qlinear(&xm, &sym).unwrap());
        let ym_qmm = read(&qlinear(&xm, &qmm).unwrap());
        let diff_qmm = ym
            .iter()
            .zip(&ym_qmm)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff_qmm <= 1e-5, "M={m} SymQ8 vs Qmm {diff_qmm:e}");
        for (r, want) in single_bits.iter().enumerate().take(m) {
            assert_eq!(&bits(&ym[r * n..(r + 1) * n]), want, "M={m} row {r}");
        }
    }
    for m in 12..=16usize {
        let xm = MxArray::from_float32(&all_xs[..m * k], &[m as i64, k as i64]).unwrap();
        let ym_nax = read(&qlinear(&xm, &sym).unwrap());
        let y_ref = read(&xm.matmul(&wt).unwrap());
        let rel = rel_l2(&ym_nax, &y_ref);
        assert!(rel < 1e-3, "M={m} dense NAX rel_l2={rel:e}");
        let ym_dec = with_prefill_q8_nax(false, || read(&qlinear(&xm, &sym).unwrap()));
        for (r, want) in single_bits.iter().enumerate().take(m) {
            assert_eq!(
                &bits(&ym_dec[r * n..(r + 1) * n]),
                want,
                "M={m} decode fallback row {r}"
            );
        }
    }

    // Grouped SymQ8: M = 1..=16 row-invariance and <= 1e-5 vs F32 ref.
    let (g, gk, gn) = (4usize, 192usize, 64usize);
    let gsym = make_sym_q8(g * gn, gk, 703);
    let gqmm = sym_to_explicit_qmm(&gsym);
    let gwf = gsym
        .f32()
        .unwrap()
        .reshape(&[g as i64, gn as i64, gk as i64])
        .unwrap()
        .transpose(Some(&[0, 2, 1]))
        .unwrap();
    let gxs = lcg(704, 16 * g * gk);
    let mut g_single_bits = Vec::with_capacity(16);
    for r in 0..16 {
        let xr = MxArray::from_float32(
            &gxs[r * g * gk..(r + 1) * g * gk],
            &[1, g as i64, gk as i64],
        )
        .unwrap();
        let y_sym = read(&grouped_mm_t_mgh(&gsym, &xr, g as i64).unwrap());
        let y_qmm = read(&grouped_mm_t_mgh(&gqmm, &xr, g as i64).unwrap());
        let diff_qmm = y_sym
            .iter()
            .zip(&y_qmm)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            diff_qmm <= 1e-5,
            "grouped M=1 row {r} SymQ8 vs Qmm {diff_qmm:e}"
        );
        let y_ref = read(
            &xr.transpose(Some(&[1, 0, 2]))
                .unwrap()
                .matmul(&gwf)
                .unwrap()
                .transpose(Some(&[1, 0, 2]))
                .unwrap(),
        );
        let max_abs = y_sym
            .iter()
            .zip(&y_ref)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs <= 1e-5, "grouped M=1 row {r} max_abs {max_abs:e}");
        g_single_bits.push(bits(&y_sym));
    }
    for m in 2..=16usize {
        let xm =
            MxArray::from_float32(&gxs[..m * g * gk], &[m as i64, g as i64, gk as i64]).unwrap();
        let ym = read(&grouped_mm_t_mgh(&gsym, &xm, g as i64).unwrap());
        for (r, want) in g_single_bits.iter().enumerate().take(m) {
            assert_eq!(
                &bits(&ym[r * g * gn..(r + 1) * g * gn]),
                want,
                "grouped M={m} row {r}"
            );
        }
    }
}

#[test]
fn q8_dense_prefill_nax_parity() {
    use super::moe::{qlinear, with_prefill_q8_nax};

    crate::models::hy_v4_flash::exact_f32();

    // Cover M in {17, 45, 64, 77, 256, 2048} and K in {192, 512, 1024, 3072, 16384},
    // including unaligned M and unaligned N.
    let cases: [(usize, usize, usize); 11] = [
        (17, 64, 48),
        (45, 192, 96),
        (77, 512, 128),
        (64, 192, 128),
        (64, 512, 96),
        (256, 1024, 256),
        (256, 3072, 192),
        (2048, 512, 128),
        (2048, 3072, 256),
        (64, 16384, 128),
        (256, 16384, 64),
    ];
    for (ci, &(m, k, n)) in cases.iter().enumerate() {
        let sym = make_sym_q8(n, k, 800 + ci as u64 * 17);
        let qmm = sym_to_explicit_qmm(&sym);
        let xs = lcg(900 + ci as u64 * 19, m * k);
        let x = MxArray::from_float32(&xs, &[m as i64, k as i64]).unwrap();

        let wf = sym.f32().unwrap();
        let y_ref = read(&x.matmul(&wf.transpose(Some(&[1, 0])).unwrap()).unwrap());

        let y_nax = with_prefill_q8_nax(true, || read(&qlinear(&x, &sym).unwrap()));
        let rel_sym = rel_l2(&y_nax, &y_ref);
        assert!(
            rel_sym <= 5e-6,
            "dense prefill SymQ8 M={m} K={k} N={n}: rel_l2={rel_sym:e}"
        );

        let y_qmm = with_prefill_q8_nax(true, || read(&qlinear(&x, &qmm).unwrap()));
        let rel_qmm = rel_l2(&y_qmm, &y_ref);
        assert!(
            rel_qmm <= 5e-6,
            "dense prefill Qmm M={m} K={k} N={n}: rel_l2={rel_qmm:e}"
        );

        if m == 64 && k == 192 {
            let y_fb = with_prefill_q8_nax(false, || read(&qlinear(&x, &sym).unwrap()));
            let rel_fb = rel_l2(&y_fb, &y_ref);
            assert!(
                rel_fb < 1e-4,
                "fallback M={m} K={k} N={n}: rel_l2={rel_fb:e}"
            );
        }
    }
}

#[test]
fn q8_grouped_prefill_nax_parity() {
    use super::attention::grouped_mm_t_mgh;
    use super::moe::{prefill_q8_grouped_nax_enabled, with_prefill_q8_nax};

    crate::models::hy_v4_flash::exact_f32();
    if std::env::var("MLX_HY4_PREFILL_Q8_GROUPED_NAX").as_deref() != Ok("0")
        && std::env::var("MLX_HY4_PREFILL_Q8_NAX").as_deref() != Ok("0")
    {
        assert!(
            prefill_q8_grouped_nax_enabled(),
            "3-MMA grouped Q8 NAX should be enabled by default"
        );
    }

    // k_b (K=192, N=512) and v_b (K=512, N=256) across M = 17, 64, 256, 2048.
    let cases: [(usize, usize, usize, usize); 7] = [
        (17, 4, 192, 512),
        (64, 4, 192, 512),
        (256, 4, 192, 512),
        (2048, 2, 192, 512),
        (64, 4, 512, 256),
        (256, 4, 512, 256),
        (2048, 2, 512, 256),
    ];
    for (ci, &(m, g, k, n)) in cases.iter().enumerate() {
        let sym = make_sym_q8(g * n, k, 1000 + ci as u64 * 23);
        let qmm = sym_to_explicit_qmm(&sym);
        let xs = lcg(1100 + ci as u64 * 29, m * g * k);
        let x = MxArray::from_float32(&xs, &[m as i64, g as i64, k as i64]).unwrap();

        let wf = sym
            .f32()
            .unwrap()
            .reshape(&[g as i64, n as i64, k as i64])
            .unwrap()
            .transpose(Some(&[0, 2, 1]))
            .unwrap();
        let y_ref = read(
            &x.transpose(Some(&[1, 0, 2]))
                .unwrap()
                .matmul(&wf)
                .unwrap()
                .transpose(Some(&[1, 0, 2]))
                .unwrap(),
        );

        let y_sym = with_prefill_q8_nax(true, || {
            read(&grouped_mm_t_mgh(&sym, &x, g as i64).unwrap())
        });
        let rel_sym = rel_l2(&y_sym, &y_ref);
        let max_abs_sym = y_sym
            .iter()
            .zip(&y_ref)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "[grouped_q8_nax] SymQ8 M={m} G={g} K={k} N={n}: rel_l2={rel_sym:.3e} max_abs={max_abs_sym:.3e}"
        );
        assert!(
            rel_sym <= 5e-6,
            "grouped prefill SymQ8 M={m} G={g} K={k} N={n}: rel_l2={rel_sym:e} max_abs={max_abs_sym:e}"
        );

        if m == 64 {
            let y_qmm = with_prefill_q8_nax(true, || {
                read(&grouped_mm_t_mgh(&qmm, &x, g as i64).unwrap())
            });
            let rel_qmm = rel_l2(&y_qmm, &y_ref);
            let max_abs_qmm = y_qmm
                .iter()
                .zip(&y_ref)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "[grouped_q8_nax] Qmm   M={m} G={g} K={k} N={n}: rel_l2={rel_qmm:.3e} max_abs={max_abs_qmm:.3e}"
            );
            assert!(
                rel_qmm <= 5e-6,
                "grouped prefill Qmm M={m} G={g} K={k} N={n}: rel_l2={rel_qmm:e} max_abs={max_abs_qmm:e}"
            );
        }
    }
}
