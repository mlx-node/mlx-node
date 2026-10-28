// Tiny end to end vs the oracle (`oracle.npz`: logits of every position for
// c1/c2/c3/pad in four modes; routes, engram rows, top-k, candidate masks and
// block outputs for c1/pad). Included into `tests.rs` so the gate filters
// read `deepseek_v41::tests::<name>`.

use super::attention::FakeQuant;
use super::engram::ResidentTable;
use super::store::Dense;
use super::model::{Model, Trace, record_fp32_sites};
use super::moe::ExpertBank;
use super::test_fixture::{Npz, TinyGguf, read, rel_l2};
use crate::array::alloc::OutLayout;
use crate::array::{DType, MxArray};
use crate::utils::gguf::GgufTensorType;
use crate::utils::gguf_cpu_repack::decode_affine_band_into;
use crate::utils::gguf_gpu::RepackDst;

const VOCAB: usize = 512;
const CASES: [&str; 4] = ["c1", "c2", "c3", "pad"];

/// pad fp32_on: the ORACLE sits on fp8 code edges (`oracle/s9_pad_probe.py`:
/// a 1e-6 relative change of the L0 `o` or L1 `kv` round-trip input flips
/// its L5 row-21 index top-k to mlx-node's), so that run's floats differ
/// from L1 on (router weights ~1e-4) and its selections from L5 on: those
/// are reported, not gated (plan S7 near-edge excuse).
fn near_edge_run(case: &str, mode: &str) -> bool {
    case == "pad" && mode == "fp32_on"
}

/// First layer whose router weights the oracle itself moves past 1e-6 under
/// one random-sign 1-ulp change of its embedding output or block outputs (a
/// fake-quant code flip): from there the weights depend on the GPU's f32
/// accumulation order and are reported, not gated. Route ids stay exact.
fn near_edge_weights(case: &str, mode: &str) -> usize {
    match (case, mode) {
        ("pad", "fp32_on") => 1,
        ("c1", "fp32_on") => 2,
        _ => usize::MAX,
    }
}

pub(super) fn mode_of(m: &str) -> (DType, FakeQuant) {
    let dt = if m.starts_with("bf16") { DType::BFloat16 } else { DType::Float32 };
    (dt, if m.ends_with("_on") { FakeQuant::ON } else { FakeQuant::OFF })
}

/// The production Q8_0 load: affine 8-bit repack (`decode_block`, code =
/// q + 128) with the bias plane -128 * d, dequantized by MLX in f32.
fn q8_affine(g: &TinyGguf, name: &str) -> napi::Result<MxArray> {
    let (info, raw) = g.raw(name);
    if info.tensor_type != GgufTensorType::Q8_0 {
        return g.dense(name);
    }
    let (dst, ptrs) = RepackDst::alloc(info, OutLayout::RowMajor)?;
    decode_affine_band_into(info.tensor_type, &raw, 0, ptrs.rows, ptrs.k, &ptrs)?;
    let scales = dst.scales.unwrap().astype(DType::Float32)?;
    let biases = scales.mul_scalar(-128.0)?;
    let mode = std::ffi::CString::new("affine").unwrap();
    // SAFETY: live arrays for the call; the bridge returns a new array or null.
    let h = unsafe {
        mlx_sys::mlx_dequantize(
            dst.weight.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
            32,
            8,
            -1,
            mode.as_ptr(),
        )
    };
    MxArray::from_handle(h, "q8_0 affine dequantize")
}

pub(super) fn tiny(g: &TinyGguf, act: DType, fq: FakeQuant, affine: bool) -> Model {
    let cfg = DeepseekV41Config::from_gguf(&g.gguf).unwrap();
    let (_, table) = g.raw("per_layer_token_embd.weight");
    let table = ResidentTable::new(table, cfg.engram.as_ref().unwrap().head_dim).unwrap();
    let bank = |n: &str| {
        let (info, raw) = g.raw(n);
        ExpertBank::from_mxfp4(info, &raw)
    };
    let get = |n: &str| if affine { q8_affine(g, n) } else { g.dense(n) }.map(Dense::from);
    Model::from_gguf(cfg, act, fq, get, bank, Some(Box::new(table)), None).unwrap()
}

pub(super) fn case_ids(npz: &Npz, case: &str) -> Vec<u32> {
    npz.get(&format!("ids.{case}")).1.iter().map(|&x| x as u32).collect()
}

/// The oracle's call split of a case (decode teacher-forced).
pub(super) fn split(case: &str) -> Vec<usize> {
    match case {
        "c2" => std::iter::once(12).chain(std::iter::repeat_n(1, 12)).collect(),
        "c3" => vec![1; 24],
        _ => vec![24],
    }
}

/// Logits of every position `[24 * VOCAB]` and one trace per call.
pub(super) fn run(m: &Model, ids: &[u32], sizes: &[usize]) -> (Vec<f32>, Vec<Trace>) {
    let mut sess = m.new_session();
    let (mut out, mut traces, mut at) = (Vec::new(), Vec::new(), 0);
    for &n in sizes {
        let mut t = Trace {
            keep_out: true,
            ..Default::default()
        };
        let l = m.forward(&mut sess, &ids[at..at + n], None, Some(&mut t)).unwrap();
        out.extend(read(&l));
        traces.push(t);
        at += n;
    }
    (out, traces)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// Mean over rows of KL(p || q) of the softmaxes of logits rows.
fn mean_kl(p: &[f32], q: &[f32]) -> f64 {
    let lsm = |r: &[f32]| -> Vec<f64> {
        let mx = r.iter().fold(f64::MIN, |m, &x| m.max(f64::from(x)));
        let z = r.iter().map(|&x| (f64::from(x) - mx).exp()).sum::<f64>().ln() + mx;
        r.iter().map(|&x| f64::from(x) - z).collect()
    };
    let rows = p.len() / VOCAB;
    (0..rows)
        .map(|i| {
            let (a, b) = (lsm(&p[i * VOCAB..(i + 1) * VOCAB]), lsm(&q[i * VOCAB..(i + 1) * VOCAB]));
            a.iter().zip(&b).map(|(x, y)| x.exp() * (x - y)).sum::<f64>()
        })
        .sum::<f64>()
        / rows as f64
}

fn f32_logits_gate(mode: &str, tol: f32) {
    let (g, npz) = (TinyGguf::open(), Npz::open("oracle.npz"));
    let (act, fq) = mode_of(mode);
    let m = tiny(&g, act, fq, false);
    for case in CASES {
        let (got, _) = run(&m, &case_ids(&npz, case), &split(case));
        let want = &npz.get(&format!("{case}.{mode}.logits")).1;
        let d = max_abs(&got, want);
        println!("{case} {mode}: max|dlogit| {d:e} (tol {tol:e})");
        assert!(d < tol || near_edge_run(case, mode), "{case} {mode}: max|dlogit| {d:e}");
    }
}

/// G9.1 f32, fake quant off: every position of c1-c3 and pad within 1e-4.
#[test]
fn tiny_logits_f32_fq_off() {
    f32_logits_gate("fp32_off", 1e-4);
}

/// G9.1 f32, fake quant on: within 1e-3 (near-edge fp8 codes, S7-5).
#[test]
fn tiny_logits_f32_fq_on() {
    f32_logits_gate("fp32_on", 1e-3);
}

/// §3 near-tie test of one route row: `m32` = the f32 margin between the
/// last kept and first dropped expert, `g` = the larger bf16-vs-f32 drift of
/// those two scores; a mismatch is a near tie iff `m32 < 2 g`.
fn router_near_tie(s32: &[f32], s16: &[f32], k: usize) -> (bool, f32, f32) {
    let mut o: Vec<usize> = (0..s32.len()).collect();
    o.sort_by(|&i, &j| s32[j].total_cmp(&s32[i]).then(i.cmp(&j)));
    let (a, b) = (o[k - 1], o[k]);
    let m32 = s32[a] - s32[b];
    let g = (s16[a] - s32[a]).abs().max((s16[b] - s32[b]).abs());
    (m32 < 2.0 * g, m32, g)
}

/// G9.1 bf16. Routing: up to the first layer where mlx-node's bf16 route
/// sets differ from the oracle's bf16 ones, every differing row is a §3
/// near tie (margins from mlx-node's own f32 run, which equals the oracle's
/// f32 within G9.1). Floats: a case with no route difference has logits
/// relL2 and mean KL vs the oracle's fp32 run within 2x the oracle's own
/// bf16-vs-fp32 drift (§3); after an excused flip, within 2x the largest
/// oracle drift of the mode (its own flipped cases). At least one case is
/// gated on the per-case floor.
#[test]
fn tiny_logits_bf16_within_floor() {
    let (g, npz) = (TinyGguf::open(), Npz::open("oracle.npz"));
    let n = 12;
    let mut gated = 0;
    for mode in ["bf16_on", "bf16_off"] {
        let (act, fq) = mode_of(mode);
        let (m16, m32) = (tiny(&g, act, fq, false), tiny(&g, DType::Float32, fq, false));
        let f32_mode = mode.replace("bf16", "fp32");
        let max_floor = CASES
            .iter()
            .map(|c| {
                let o32 = &npz.get(&format!("{c}.{f32_mode}.logits")).1;
                rel_l2(&npz.get(&format!("{c}.{mode}.logits")).1, o32)
            })
            .fold(0f64, f64::max);
        for case in CASES {
            let ids = case_ids(&npz, case);
            let (got, tr16) = run(&m16, &ids, &split(case));
            let (_, tr32) = run(&m32, &ids, &split(case));
            let want = &npz.get(&format!("{case}.{mode}.router_idx")).1;
            let mut flips = 0;
            for l in 0..8 {
                let (mut rows, mut row0) = (Vec::new(), 0);
                for (t16, t32) in tr16.iter().zip(&tr32) {
                    let (r16, r32) = (&t16.routes[l], &t32.routes[l]);
                    for r in 0..r16.rows() {
                        let p = row0 + r;
                        let mut a: Vec<u32> = r16.ids[r * 6..r * 6 + 6].to_vec();
                        let mut b: Vec<u32> = want[(l * 24 + p) * 6..(l * 24 + p) * 6 + 6].iter().map(|&x| x as u32).collect();
                        a.sort_unstable();
                        b.sort_unstable();
                        if a != b {
                            let near = router_near_tie(&r32.biased[r * n..(r + 1) * n], &r16.biased[r * n..(r + 1) * n], 6);
                            rows.push((p, near));
                        }
                    }
                    row0 += r16.rows();
                }
                if rows.is_empty() {
                    continue;
                }
                println!("{case} {mode} L{l}: route sets differ from the oracle at (row, (near, m32, g)) {rows:?}");
                if flips == 0 {
                    assert!(rows.iter().all(|r| r.1.0), "{case} {mode} L{l}: a route flip that is not a near tie");
                }
                flips += rows.len();
            }
            let o32 = &npz.get(&format!("{case}.{f32_mode}.logits")).1;
            let o16 = &npz.get(&format!("{case}.{mode}.logits")).1;
            let (r, floor) = (rel_l2(&got, o32), rel_l2(o16, o32));
            let (kl, kl_floor) = (mean_kl(o32, &got), mean_kl(o32, o16));
            println!("{case} {mode}: relL2 {r:.3e} floor {floor:.3e}; KL {kl:.3e} floor {kl_floor:.3e}; route flips {flips}");
            if flips == 0 {
                assert!(r <= (2.0 * floor).max(1e-3), "{case} {mode}: relL2 {r:e} > 2 x {floor:e}");
                assert!(kl <= (2.0 * kl_floor).max(1e-4), "{case} {mode}: KL {kl:e} > 2 x {kl_floor:e}");
                gated += 1;
            } else {
                // the oracle's own drift when its bf16 routes flip
                assert!(r <= 2.0 * max_floor, "{case} {mode}: relL2 {r:e} > 2 x {max_floor:e} (largest oracle floor)");
            }
            if case == "c1" && mode == "bf16_on" {
                for (l, x) in tr16[0].out.iter().enumerate() {
                    let w32 = &npz.get(&format!("c1.fp32_on.L{l}.out")).1;
                    let w16 = &npz.get(&format!("c1.bf16_on.L{l}.out@bf16")).1;
                    let (a, b) = (rel_l2(&read(x), w32), rel_l2(w16, w32));
                    println!("  L{l}: relL2 vs ora32 {a:.3e}, oracle floor {b:.3e}, r {:.2}", a / b.max(1e-6));
                }
            }
        }
    }
    assert!(gated >= 1, "no case gated on floats");
}

/// G9.2: the production Q8_0 load (affine 8-bit + bias plane, MLX
/// dequantize) gives the host `d * q` weights bit for bit, so the model on
/// it gives the same logits as G9.1's, bit for bit, in every case.
#[test]
fn tiny_q8_0_affine_load_matches() {
    let (g, npz) = (TinyGguf::open(), Npz::open("oracle.npz"));
    let mut n_q8 = 0;
    for t in g.gguf.tensors.iter().filter(|t| t.tensor_type == GgufTensorType::Q8_0) {
        let (a, b) = (read(&q8_affine(&g, &t.name).unwrap()), read(&g.dense(&t.name).unwrap()));
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "{}", t.name);
        n_q8 += 1;
    }
    assert!(n_q8 >= 70, "{n_q8} Q8_0 tensors");
    for mode in ["fp32_on", "bf16_on"] {
        let (act, fq) = mode_of(mode);
        let (host, aff) = (tiny(&g, act, fq, false), tiny(&g, act, fq, true));
        for case in CASES {
            let ids = case_ids(&npz, case);
            let (a, b) = (run(&aff, &ids, &split(case)).0, run(&host, &ids, &split(case)).0);
            assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "{case} {mode}");
            let d = max_abs(&a, &npz.get(&format!("{case}.{mode}.logits")).1);
            println!("{case} {mode}: affine == host bitwise; max|dlogit| vs oracle {d:e}");
        }
    }
    println!("{n_q8} Q8_0 tensors bitwise");
}

/// G9.3: a new session (reset) replaying a sequence equals the first run
/// bit for bit, also after the model served another sequence.
#[test]
fn invariants_reset_replay_bitwise() {
    let (g, npz) = (TinyGguf::open(), Npz::open("oracle.npz"));
    for mode in ["fp32_on", "bf16_on"] {
        let (act, fq) = mode_of(mode);
        let m = tiny(&g, act, fq, false);
        let ids = case_ids(&npz, "c2");
        let first = run(&m, &ids, &split("c2")).0;
        let _ = run(&m, &case_ids(&npz, "pad"), &[24]);
        let again = run(&m, &ids, &split("c2")).0;
        assert!(first.iter().zip(&again).all(|(x, y)| x.to_bits() == y.to_bits()), "{mode}");
    }
}

/// G9.3: chunk sizes {1, 7, all} agree within 1e-5 (relL2 per position) in
/// f32 with fake quant off; fake quant on is printed (near-edge codes).
#[test]
fn invariants_chunk_sizes() {
    let (g, npz) = (TinyGguf::open(), Npz::open("oracle.npz"));
    let ids = case_ids(&npz, "c1");
    for mode in ["fp32_off", "fp32_on"] {
        let (act, fq) = mode_of(mode);
        let m = tiny(&g, act, fq, false);
        let all = run(&m, &ids, &[24]).0;
        for sizes in [vec![1; 24], vec![7, 7, 7, 3]] {
            let got = run(&m, &ids, &sizes).0;
            let worst = (0..24)
                .map(|p| rel_l2(&got[p * VOCAB..(p + 1) * VOCAB], &all[p * VOCAB..(p + 1) * VOCAB]))
                .fold(0f64, f64::max);
            println!("{mode} chunks {}: worst per-position relL2 {worst:e}", sizes[0]);
            if mode == "fp32_off" {
                assert!(worst <= 1e-5, "chunks {}: {worst:e}", sizes[0]);
            }
        }
    }
}

/// G9.3: a call cancelled (or failing) mid-stack leaves no partial layer
/// state: the session continues bit for bit as if the call never happened.
#[test]
fn invariants_cancel_restores() {
    let (g, npz) = (TinyGguf::open(), Npz::open("oracle.npz"));
    let m = tiny(&g, DType::Float32, FakeQuant::ON, false);
    let ids = case_ids(&npz, "c2");
    let clean = run(&m, &ids, &split("c2")).0;
    let mut sess = m.new_session();
    let stop_at = |l: usize| move |x: usize| x == l;
    let err = m.forward(&mut sess, &ids[..12], Some(&stop_at(3)), None);
    assert!(err.is_err() && sess.tokens.is_empty());
    let vocab = DeepseekV41Config::from_gguf(&g.gguf).unwrap().vocab_size as u32;
    for bad in [vocab, u32::MAX] {
        let err = m.forward(&mut sess, &[ids[0], bad], None, None).err().unwrap();
        assert!(err.reason.contains("outside the"), "{}", err.reason);
        assert!(sess.tokens.is_empty());
    }
    let mut out = read(&m.forward(&mut sess, &ids[..12], None, None).unwrap());
    for p in 12..24 {
        // layer 6 runs after the KV source 5 advanced its compressed cache
        assert!(m.forward(&mut sess, &ids[p..=p], Some(&stop_at(6)), None).is_err());
        assert_eq!(sess.tokens.len(), p);
        out.extend(read(&m.forward(&mut sess, &ids[p..=p], None, None).unwrap()));
    }
    assert!(clean.iter().zip(&out).all(|(x, y)| x.to_bits() == y.to_bits()));
}

/// S14c: `forward_last` returns the last row of `forward`'s logits (the head
/// GEMM has another M, so within 1e-6) and leaves the same session: the next
/// call's logits are bitwise equal.
#[test]
fn forward_last_is_the_last_row() {
    let (g, npz) = (TinyGguf::open(), Npz::open("oracle.npz"));
    let m = tiny(&g, DType::Float32, FakeQuant::ON, false);
    let ids = case_ids(&npz, "c2");
    let (mut a, mut b) = (m.new_session(), m.new_session());
    let full = read(&m.forward(&mut a, &ids[..23], None, None).unwrap());
    let last = m.forward_last(&mut b, &ids[..23], None, None).unwrap();
    assert_eq!(last.shape().unwrap().as_ref(), [1, VOCAB as i64]);
    let r = rel_l2(&read(&last), &full[22 * VOCAB..]);
    println!("forward_last vs forward row 22: relL2 {r:e}");
    assert!(r <= 1e-6, "{r:e}");
    let na = read(&m.forward(&mut a, &ids[23..], None, None).unwrap());
    let nb = read(&m.forward(&mut b, &ids[23..], None, None).unwrap());
    assert!(na.iter().zip(&nb).all(|(x, y)| x.to_bits() == y.to_bits()));
}

/// G9.3: routes, engram rows, index top-k and candidate blocks equal the
/// oracle's bit for bit (f32 modes; every f32 boundary is >= 10x its noise).
#[test]
fn invariants_routes_rows_topk_match_oracle() {
    let (g, npz) = (TinyGguf::open(), Npz::open("oracle.npz"));
    let cfg = DeepseekV41Config::from_gguf(&g.gguf).unwrap();
    let hasher = super::engram_hash::EngramHasher::new(cfg.engram.as_ref().unwrap()).unwrap();
    let (k, n_l) = (cfg.n_activated_experts, cfg.n_layers);
    for mode in ["fp32_on", "fp32_off"] {
        let (act, fq) = mode_of(mode);
        let m = tiny(&g, act, fq, false);
        for case in CASES {
            let (_, tr) = run(&m, &case_ids(&npz, case), &split(case));
            // router_idx [layers, 24, k] across calls, rank order
            let want = &npz.get(&format!("{case}.{mode}.router_idx")).1;
            // pad fp32_on: bitwise through L4, reported from L5 (near_edge_run)
            let gated = if near_edge_run(case, mode) { 5 } else { n_l };
            for l in 0..n_l {
                let got: Vec<f32> = tr.iter().flat_map(|t| t.routes[l].ids.iter().map(|&e| e as f32)).collect();
                if l >= gated {
                    let same = got == want[l * 24 * k..(l + 1) * 24 * k];
                    println!("{case} {mode} L{l}: routes equal {same} (reported only)");
                    continue;
                }
                assert_eq!(got, want[l * 24 * k..(l + 1) * 24 * k], "{case} {mode} L{l} routes");
            }
            if !matches!(case, "c1" | "pad") {
                continue;
            }
            let t = &tr[0];
            let key = |l: usize, w: &str| format!("{case}.{mode}.L{l}.{w}");
            for l in 0..gated {
                let w = &npz.get(&key(l, "router_w")).1;
                let d = max_abs(&t.routes[l].weights, w);
                if l >= near_edge_weights(case, mode) {
                    println!("{case} {mode} L{l}: weights {d:e} (reported only)");
                } else {
                    assert!(d <= 1e-6, "{case} {mode} L{l} weights {d:e}");
                }
                if let Some(tk) = &t.topk[l] {
                    // indexer output = compressed row + window offset (24), -1 = none
                    let want: Vec<i32> = npz.get(&key(l, "index_topk")).1.iter()
                        .map(|&x| if x < 0.0 { -1 } else { x as i32 - 24 })
                        .collect();
                    assert_eq!(tk.ids, want, "{case} {mode} L{l} top-k");
                }
                if let Some(c) = &t.candidates[l] {
                    let (shape, want) = npz.get(&key(l, "cand_mask"));
                    let w = shape[1] as usize;
                    let got: Vec<f32> = (0..24)
                        .flat_map(|r| (0..w).map(move |j| (r, j)))
                        .map(|(r, j)| f32::from(u8::from(c.keep[r * c.n_blocks + j / c.block_size])))
                        .collect();
                    assert_eq!(&got, want, "{case} {mode} L{l} candidates");
                }
            }
            let cols = hasher.n_cols();
            let got: Vec<f32> = t.engram_rows.iter().enumerate()
                .map(|(i, &r)| (u64::from(r) - hasher.layer_base((i / cols) % hasher.n_layers())) as f32)
                .collect();
            assert_eq!(&got, &npz.get(&format!("{case}.{mode}.hash_ids")).1, "{case} {mode} engram rows");
        }
    }
}

/// G9.4: in bf16 mode every §3 fp32 site is recorded, always as f32.
#[test]
fn fp32_sites_stay_f32_in_bf16() {
    // §3: MoE sum, hc mixes, compressor ratio 2, head input, RMSNorm, router
    // logits, attention scores
    const SITES: [&str; 7] = [
        "attn_scores",
        "compressor_ratio2",
        "head_input",
        "hc_mixes",
        "moe_sum",
        "rms_norm",
        "router_logits",
    ];
    let (g, npz) = (TinyGguf::open(), Npz::open("oracle.npz"));
    let m = tiny(&g, DType::BFloat16, FakeQuant::ON, false);
    let ids = case_ids(&npz, "c2");
    let ((), seen) = record_fp32_sites(|| {
        run(&m, &ids[..13], &[12, 1]);
    });
    let mut want = SITES.to_vec();
    want.sort_unstable();
    let mut names: Vec<&str> = seen.iter().map(|s| s.0).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names, want);
    for (site, layer, dt) in &seen {
        assert_eq!(*dt, DType::Float32, "{site} at layer {layer}");
    }
    println!("{} fp32 site records, all f32", seen.len());
}
