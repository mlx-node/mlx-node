use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::scan::{Role, Scan};
use super::weights::{
    Form, OUTPUT, Resident, Residual, TOKEN_EMBD, bf16_into_f32, dense_from_raw, form, held_shape,
    le_f32_into, le_u16_into, q8_0_into_f32, resident_bytes, role_resident,
};
use crate::array::{DType, MxArray};
use crate::models::deepseek_v41::store::{Dense, Weight};
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
    let cases: [(Role, &str, GgufTensorType, Option<Form>); 14] = [
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
        ("blk.0.attn_q_a.weight", &[64, 4], Q8_0, 256 + 64),
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
    assert!(matches!(d, Dense::Qmm { .. }));
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
