//! The fused GDN prefill on its row-split scan against the
//! one-value-head-per-threadgroup kernel (`HEAD_SCAN_PREFILL`), bit for bit:
//! every Qwen3.5 head family, window lengths around the 16-row scan block and
//! the 32-row reference chunk, carried and zero states, states chained over
//! calls, capture rungs on block and chunk edges, and the layer forward.

use napi::bindgen_prelude::Either;
use napi::{Error, Result};

use crate::array::{DType, MxArray};

use super::arrays_cache::ArraysCache;
use super::config::Qwen3_5Config;
use super::gated_delta::{
    GdnCarry, GdnPrefill, GdnTail, HEAD_SCAN_PREFILL, gated_delta_fused_prefill,
    gated_delta_split_at,
};
use super::gated_delta_net::{GatedDeltaNet, GdnCapturePlan};

const D: i64 = 128;

/// (key heads, value heads, tiled GGUF head map, what has the shape)
const FAMILIES: [(i64, i64, bool, &str); 9] = [
    (16, 48, true, "27B GGUF"),
    (16, 48, false, "27B"),
    (16, 32, true, "35B-A3B MoE GGUF"),
    (16, 32, false, "35B-A3B MoE / 9B / 4B"),
    (16, 16, false, "2B / 0.8B"),
    (16, 64, false, "122B-A10B / 397B-A17B"),
    (8, 40, true, "Hk 8 / Hv 40"),
    (8, 80, true, "Hk 8 / Hv 80"),
    (1, 2, false, "one key head"),
];

const ROWS_27B: [usize; 7] = [1, 31, 32, 33, 2047, 2048, 2049];
const ROWS_OTHER: [usize; 4] = [1, 17, 33, 2049];

fn normal(shape: &[i64], std: f64, dtype: DType) -> MxArray {
    MxArray::random_normal(shape, 0.0, std, Some(DType::Float32))
        .unwrap()
        .astype(dtype)
        .unwrap()
}

/// Bits of every element (bf16 widens to f32 exactly).
fn bits(a: &MxArray) -> Vec<u32> {
    a.astype(DType::Float32)
        .unwrap()
        .to_float32()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

fn differing(a: &MxArray, b: &MxArray) -> usize {
    let (a, b) = (bits(a), bits(b));
    assert_eq!(a.len(), b.len(), "compared arrays differ in size");
    a.iter().zip(&b).filter(|(x, y)| x != y).count()
}

/// Runs `f` with the head kernel (`head`) or the product scan.
fn on_kernel<T>(head: bool, f: impl FnOnce() -> T) -> T {
    HEAD_SCAN_PREFILL.with(|c| c.set(head));
    let out = f();
    HEAD_SCAN_PREFILL.with(|c| c.set(false));
    out
}

fn launches() -> bool {
    super::gated_delta::tests::fused_prefill_launches()
        && on_kernel(true, super::gated_delta::tests::fused_prefill_launches)
}

/// Production views of one window: pre-norm q/k/v split out of the conv row,
/// b/a out of the padded ba row, z out of the qkvz row (all strided).
struct Window {
    q: MxArray,
    k: MxArray,
    v: MxArray,
    a: MxArray,
    b: MxArray,
    z: MxArray,
}

struct Weights {
    hk: i64,
    hv: i64,
    tiled: bool,
    w: MxArray,
    a_log: MxArray,
    dt_bias: MxArray,
    wq: MxArray,
    wk: MxArray,
}

fn weights(hk: i64, hv: i64, tiled: bool) -> Weights {
    let inv = (D as f64).powf(-0.5);
    Weights {
        hk,
        hv,
        tiled,
        w: normal(&[D], 0.3, DType::BFloat16).add_scalar(1.0).unwrap(),
        a_log: normal(&[hv], 0.3, DType::Float32),
        dt_bias: normal(&[hv], 0.3, DType::Float32),
        wq: MxArray::full(&[D], Either::A(inv * inv), Some(DType::BFloat16)).unwrap(),
        wk: MxArray::full(&[D], Either::A(inv), Some(DType::BFloat16)).unwrap(),
    }
}

fn window(t: usize, wt: &Weights) -> Window {
    let (hk, hv, t) = (wt.hk, wt.hv, t as i64);
    let conv = normal(&[1, t, 2 * hk * D + hv * D], 2.0, DType::BFloat16);
    let head = |s: i64, e: i64, heads: i64| {
        conv.slice_axis(2, s, e)
            .unwrap()
            .reshape(&[1, t, heads, D])
            .unwrap()
    };
    let ba = normal(&[1, t, 2 * hv + 32], 0.3, DType::BFloat16);
    let z = normal(&[1, t, 2 * hk * D + 2 * hv * D + 96], 2.0, DType::BFloat16)
        .slice_axis(2, 2 * hk * D + hv * D, 2 * hk * D + 2 * hv * D)
        .unwrap();
    Window {
        q: head(0, hk * D, hk),
        k: head(hk * D, 2 * hk * D, hk),
        v: head(2 * hk * D, 2 * hk * D + hv * D, hv),
        a: ba.slice_axis(2, hv, 2 * hv).unwrap(),
        b: ba.slice_axis(2, 0, hv).unwrap(),
        z,
    }
}

fn state(wt: &Weights) -> MxArray {
    normal(&[1, wt.hv, D, D], 0.3, DType::Float32)
}

fn rows(x: &MxArray, s: usize, e: usize) -> MxArray {
    x.slice_axis(1, s as i64, e as i64).unwrap()
}

/// The fused prefill on rows `[s, e)` of `win`.
fn fused(
    win: &Window,
    wt: &Weights,
    s: usize,
    e: usize,
    state: Option<&MxArray>,
) -> Result<(MxArray, MxArray)> {
    let (a, b, z) = (rows(&win.a, s, e), rows(&win.b, s, e), rows(&win.z, s, e));
    gated_delta_fused_prefill(
        &rows(&win.q, s, e),
        &rows(&win.k, s, e),
        &rows(&win.v, s, e),
        state,
        &GdnPrefill {
            a: &a,
            b: &b,
            a_log: &wt.a_log,
            dt_bias: &wt.dt_bias,
            wq: &wt.wq,
            wk: &wt.wk,
            qk_eps: 1e-6,
            tiled: wt.tiled,
            tail: GdnTail {
                z: &z,
                norm_weight: &wt.w,
                eps: 1e-6,
                state_dst: None,
            },
        },
    )?
    .ok_or_else(|| Error::from_reason("the fused GDN prefill declined"))
}

/// Both kernels on the same window and state: (differing out, differing
/// state). Asserts the head kernel's result is not degenerate.
fn compare(
    label: &str,
    win: &Window,
    wt: &Weights,
    t: usize,
    head_state: Option<&MxArray>,
    row_state: Option<&MxArray>,
) -> (usize, usize, MxArray, MxArray) {
    let (yh, sh) = on_kernel(true, || fused(win, wt, 0, t, head_state)).unwrap();
    let (yr, sr) = on_kernel(false, || fused(win, wt, 0, t, row_state)).unwrap();
    MxArray::eval_arrays(&[&yh, &sh, &yr, &sr]).unwrap();
    let y_bits = bits(&yh);
    let nonzero = y_bits.iter().filter(|&&u| u & 0x7fff_ffff != 0).count();
    let nan = yh
        .astype(DType::Float32)
        .unwrap()
        .to_float32()
        .unwrap()
        .iter()
        .filter(|x| !x.is_finite())
        .count();
    assert!(
        nonzero * 2 > y_bits.len() && nan == 0,
        "{label}: degenerate head-kernel output ({nonzero} of {} non-zero, {nan} not finite)",
        y_bits.len()
    );
    let (dy, ds) = (differing(&yh, &yr), differing(&sh, &sr));
    eprintln!(
        "E4_BITS {label}: out {} elements, {nonzero} non-zero, differ {dy}; state {} elements, differ {ds}",
        y_bits.len(),
        bits(&sh).len()
    );
    (dy, ds, sh, sr)
}

/// Every head family and window length the fused route can hand the
/// kernel (a capture split hands it segments of any length), from a carried
/// state and from the in-kernel zero state.
#[test]
fn row_scan_keeps_the_head_kernel_bits_on_every_family() {
    if !launches() {
        eprintln!("SKIP row_scan_keeps_the_head_kernel_bits_on_every_family: no fused prefill");
        return;
    }
    let mut bad = Vec::new();
    for (i, &(hk, hv, tiled, name)) in FAMILIES.iter().enumerate() {
        let wt = weights(hk, hv, tiled);
        let ts: &[usize] = if i < 2 { &ROWS_27B } else { &ROWS_OTHER };
        for &t in ts {
            let win = window(t, &wt);
            let s0 = state(&wt);
            for carried in [true, false] {
                if !carried && t != 33 && t != 2049 {
                    continue;
                }
                let label = format!("{name} hk={hk} hv={hv} tiled={tiled} T={t} carried={carried}");
                let st = carried.then_some(&s0);
                let (dy, ds, sh, _) = compare(&label, &win, &wt, t, st, st);
                if carried && t > 1 {
                    assert!(
                        differing(&s0, &sh) * 2 > bits(&s0).len(),
                        "{label}: state did not move"
                    );
                }
                if dy + ds != 0 {
                    bad.push((label, dy, ds));
                }
            }
        }
    }
    assert!(
        bad.is_empty(),
        "row scan bits differ from the head kernel: {bad:?}"
    );
}

/// A state carried through three windows (2,048 rows, 37, 2,049), each
/// kernel feeding its own state into its next call.
#[test]
fn row_scan_state_chains_over_three_calls() {
    if !launches() {
        eprintln!("SKIP row_scan_state_chains_over_three_calls: no fused prefill");
        return;
    }
    for &(hk, hv, tiled, name) in &FAMILIES[..4] {
        let wt = weights(hk, hv, tiled);
        let s0 = state(&wt);
        let (mut head, mut row) = (s0.clone(), s0);
        for (call, t) in [2048usize, 37, 2049].into_iter().enumerate() {
            let win = window(t, &wt);
            let label = format!("{name} chain call {call} T={t}");
            let (dy, ds, sh, sr) = compare(&label, &win, &wt, t, Some(&head), Some(&row));
            assert_eq!((dy, ds), (0, 0), "{label}: bits differ");
            (head, row) = (sh, sr);
        }
    }
}

/// Capture rungs on the scan's 16-row block edges and the head kernel's
/// 32-row chunk edges: the split recurrence gives the same output rows, final
/// state and every captured state on both kernels.
#[test]
fn capture_rungs_on_chunk_edges_keep_the_head_kernel_bits() {
    if !launches() {
        eprintln!("SKIP capture_rungs_on_chunk_edges_keep_the_head_kernel_bits: no fused prefill");
        return;
    }
    let cuts = [15usize, 16, 17, 32, 48, 64, 1024, 2031, 2032, 2048];
    for &(hk, hv, tiled, name) in &FAMILIES[..3] {
        let wt = weights(hk, hv, tiled);
        let t = 2049;
        let win = window(t, &wt);
        let init = GdnCarry {
            state: Some(state(&wt)),
            history: None,
        };
        let split = |head: bool| {
            on_kernel(head, || {
                gated_delta_split_at(t, &cuts, init.clone(), |s, e, c| {
                    let (y, st) = fused(&win, &wt, s, e, c.state.as_ref())?;
                    Ok((
                        y,
                        GdnCarry {
                            state: Some(st),
                            history: None,
                        },
                    ))
                })
            })
            .unwrap()
        };
        let (h, r) = (split(true), split(false));
        let label = format!("{name} T={t} cuts={cuts:?}");
        let dy = differing(&h.y, &r.y);
        let state_of = |c: &GdnCarry| c.state.clone().expect("carried state");
        let ds = differing(&state_of(&h.carry), &state_of(&r.carry));
        eprintln!("E4_RUNGS {label}: out differ {dy}, final state differ {ds}");
        assert_eq!((dy, ds), (0, 0), "{label}");
        assert_eq!(h.captured.len(), cuts.len());
        for ((c, hc), rc) in cuts.iter().zip(&h.captured).zip(&r.captured) {
            let d = differing(&state_of(hc), &state_of(rc));
            assert_eq!(d, 0, "{label}: captured state at row {c} differs");
        }
    }
}

fn net(hidden: i64, hk: i64, hv: i64, tiled: bool) -> GatedDeltaNet {
    let config = Qwen3_5Config {
        rope_scaling: None,
        qwen35_gguf_gdn_layout: tiled.then(|| "tiled".to_string()),
        kv_format: None,
        vocab_size: 32,
        hidden_size: hidden as i32,
        num_layers: 4,
        num_heads: 2,
        num_kv_heads: 1,
        intermediate_size: 32,
        rms_norm_eps: 1e-6,
        head_dim: 8,
        tie_word_embeddings: true,
        attention_bias: false,
        max_position_embeddings: 4096,
        pad_token_id: 0,
        eos_token_id: 1,
        bos_token_id: 2,
        linear_num_value_heads: hv as i32,
        linear_num_key_heads: hk as i32,
        linear_key_head_dim: D as i32,
        linear_value_head_dim: D as i32,
        linear_conv_kernel_dim: 4,
        full_attention_interval: 2,
        partial_rotary_factor: 0.25,
        rope_theta: 10_000.0,
        paged_cache_memory_mb: None,
        paged_block_size: None,
        n_mtp_layers: 0,
    };
    let mut net = GatedDeltaNet::new(&config).unwrap();
    let (keys, values) = (hk * D, hv * D);
    let scale = (hidden as f64).powf(-0.5);
    net.set_in_proj_qkvz_weight(&normal(
        &[2 * keys + 2 * values, hidden],
        scale,
        DType::BFloat16,
    ))
    .unwrap();
    net.set_in_proj_ba_weight(&normal(&[2 * hv, hidden], scale, DType::BFloat16))
        .unwrap();
    net.set_conv1d_weight(
        &normal(&[2 * keys + values, 1, 4], 0.5, DType::BFloat16),
        DType::BFloat16,
    )
    .unwrap();
    net.set_norm_weight(
        &normal(&[D], 0.3, DType::BFloat16).add_scalar(1.0).unwrap(),
        DType::BFloat16,
    )
    .unwrap();
    net.set_out_proj_weight(&normal(&[hidden, values], 0.02, DType::BFloat16))
        .unwrap();
    net.set_dt_bias(&normal(&[hv], 0.3, DType::Float32));
    net.set_a_log(&normal(&[hv], 0.3, DType::Float32)).unwrap();
    net
}

fn cache_copy(cache: &ArraysCache) -> ArraysCache {
    let mut copy = ArraysCache::new(2);
    for slot in 0..2 {
        if let Some(x) = cache.get(slot) {
            copy.set(slot, x.clone()).unwrap();
        }
    }
    copy
}

/// The layer forward on both kernels: a warm cache (5 rows first), then a
/// captured prompt chunk and a plain one after it. Output, both cache slots
/// and every captured state keep their bits; the 27B shape (hidden 64 here)
/// and the 35B-A3B MoE's hidden 2048 with 32 value heads.
#[test]
fn layer_forward_keeps_the_head_kernel_bits() -> Result<()> {
    if !launches() {
        eprintln!("SKIP layer_forward_keeps_the_head_kernel_bits: no fused prefill");
        return Ok(());
    }
    for (hidden, hk, hv, tiled, t, cuts) in [
        (
            64i64,
            16i64,
            48i64,
            true,
            2049usize,
            &[16usize, 32, 2032, 2048][..],
        ),
        (2048, 16, 32, false, 1000, &[16, 33, 992][..]),
    ] {
        let net = net(hidden, hk, hv, tiled);
        let warm = normal(&[1, 5, hidden], 1.0, DType::BFloat16);
        let chunk = normal(&[1, t as i64, hidden], 1.0, DType::BFloat16);
        let next = normal(&[1, 130, hidden], 1.0, DType::BFloat16);
        let mut base = ArraysCache::new(2);
        net.forward(&warm, None, Some(&mut base), true)?;
        MxArray::eval_arrays(&[base.get(0).unwrap(), base.get(1).unwrap()])?;
        let run = |head: bool| -> Result<(Vec<MxArray>, ArraysCache)> {
            on_kernel(head, || {
                let mut cache = cache_copy(&base);
                let (out, captured) = net.forward_captured(
                    &chunk,
                    Some(&mut cache),
                    true,
                    None,
                    &GdnCapturePlan { rows: cuts },
                )?;
                let after = net.forward(&next, None, Some(&mut cache), true)?;
                let mut keep = vec![out, after];
                keep.extend(captured.into_iter().map(|c| c.rec));
                Ok((keep, cache))
            })
        };
        let (h, hc) = run(true)?;
        let (r, rc) = run(false)?;
        let label = format!("layer hidden={hidden} hk={hk} hv={hv} tiled={tiled} T={t}");
        for (i, (a, b)) in h.iter().zip(&r).enumerate() {
            let d = differing(a, b);
            eprintln!("E4_LAYER {label} array {i}: differ {d}");
            assert_eq!(d, 0, "{label}: array {i} (0 out, 1 next out, 2.. captures)");
        }
        for slot in 0..2 {
            assert_eq!(
                differing(hc.get(slot).unwrap(), rc.get(slot).unwrap()),
                0,
                "{label}: cache slot {slot}"
            );
        }
    }
    Ok(())
}

fn family_count(name: &std::ffi::CStr) -> u64 {
    // SAFETY: `name` is NUL-terminated; test-only counter.
    unsafe { mlx_sys::mlx_test_kquant_family_count(name.as_ptr()) }
}

fn counted(f: impl FnOnce()) -> (u64, u64) {
    // SAFETY: test-only thread-local counter switch.
    unsafe { mlx_sys::mlx_test_kquant_counting(true) };
    f();
    let counts = (
        family_count(c"gdn_prefill_row_scan"),
        family_count(c"gdn_prefill_head_scan"),
    );
    // SAFETY: as above.
    unsafe { mlx_sys::mlx_test_kquant_counting(false) };
    counts
}

/// Route: every window the fused prefill takes runs the row scan, on every
/// family; the head kernel runs only under the test switch; decode, verify
/// and short windows (T < 64) keep their own kernels.
#[test]
fn every_fused_prefill_window_takes_the_row_scan() -> Result<()> {
    if !launches() {
        eprintln!("SKIP every_fused_prefill_window_takes_the_row_scan: no fused prefill");
        return Ok(());
    }
    for &(hk, hv, tiled, name) in &FAMILIES {
        let wt = weights(hk, hv, tiled);
        for t in [1usize, 64, 2049] {
            let win = window(t, &wt);
            let s0 = state(&wt);
            for head in [false, true] {
                let counts = counted(|| {
                    let (y, s) = on_kernel(head, || fused(&win, &wt, 0, t, Some(&s0))).unwrap();
                    MxArray::eval_arrays(&[&y, &s]).unwrap();
                });
                let want = if head { (0, 1) } else { (1, 0) };
                assert_eq!(
                    counts, want,
                    "{name} T={t} head={head}: (row, head) dispatches"
                );
            }
        }
    }
    let net = net(64, 16, 48, true);
    for (t, fused_route) in [
        (1usize, false),
        (8, false),
        (16, false),
        (63, false),
        (64, true),
        (2049, true),
    ] {
        let x = normal(&[1, t as i64, 64], 1.0, DType::BFloat16);
        let counts = counted(|| {
            let mut cache = ArraysCache::new(2);
            let y = net.forward(&x, None, Some(&mut cache), true).unwrap();
            MxArray::eval_arrays(&[&y, cache.get(1).unwrap()]).unwrap();
        });
        let want = if fused_route { (1, 0) } else { (0, 0) };
        assert_eq!(counts, want, "layer T={t}: (row, head) dispatches");
    }
    Ok(())
}

#[path = "gdn_row_split_scan_tests.rs"]
mod gdn_row_split_scan_tests;
