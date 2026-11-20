//! The fused GDN prefill with bf16 `a_log`/`dt_bias` (what a GGUF load of
//! the 27B keeps: the F32 tensors are stored bf16) against the chain the
//! real model ran before (`rms_norm_scaled` q/k -> `kernel_inputs` with
//! `fused_gdn_gating` -> per-step kernel -> gated RMSNorm + z gate), bit for
//! bit: kernel level over head families, window lengths, carried and zero
//! states, chained calls; and the layer forward (output, both cache slots,
//! captured states, the next window).

use napi::bindgen_prelude::Either;
use napi::{Error, Result};

use crate::array::{DType, MxArray};

use super::{GdnPrefill, GdnTail, gated_delta_fused_prefill, gated_delta_kernel, kernel_inputs};
use crate::models::qwen3_5::arrays_cache::ArraysCache;
use crate::models::qwen3_5::config::Qwen3_5Config;
use crate::models::qwen3_5::gated_delta_net::{GatedDeltaNet, GdnCapturePlan, GdnLayerTape};

const D: i64 = 128;

fn normal(shape: &[i64], std: f64, dtype: DType) -> MxArray {
    MxArray::random_normal(shape, 0.0, std, Some(DType::Float32))
        .unwrap()
        .astype(dtype)
        .unwrap()
}

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
    assert_eq!(a.dtype().unwrap(), b.dtype().unwrap(), "compared dtypes");
    let (a, b) = (bits(a), bits(b));
    assert_eq!(a.len(), b.len(), "compared arrays differ in size");
    a.iter().zip(&b).filter(|(x, y)| x != y).count()
}

fn launches() -> bool {
    super::tests::fused_prefill_launches()
}

fn family_count(name: &std::ffi::CStr) -> u64 {
    // SAFETY: `name` is NUL-terminated; test-only counter.
    unsafe { mlx_sys::mlx_test_kquant_family_count(name.as_ptr()) }
}

/// Row-scan dispatches inside `f`.
fn row_scans(f: impl FnOnce()) -> u64 {
    // SAFETY: test-only thread-local counter switch.
    unsafe { mlx_sys::mlx_test_kquant_counting(true) };
    f();
    let n = family_count(c"gdn_prefill_row_scan");
    // SAFETY: as above.
    unsafe { mlx_sys::mlx_test_kquant_counting(false) };
    n
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

fn weights(hk: i64, hv: i64, tiled: bool, scalars: DType) -> Weights {
    let inv = (D as f64).powf(-0.5);
    Weights {
        hk,
        hv,
        tiled,
        w: normal(&[D], 0.3, DType::BFloat16).add_scalar(1.0).unwrap(),
        a_log: normal(&[hv], 0.3, scalars),
        dt_bias: normal(&[hv], 0.3, scalars),
        wq: MxArray::full(&[D], Either::A(inv * inv), Some(DType::BFloat16)).unwrap(),
        wk: MxArray::full(&[D], Either::A(inv), Some(DType::BFloat16)).unwrap(),
    }
}

/// Production views of one window (all strided): pre-norm q/k/v out of the
/// conv row, b/a out of the padded ba row, z out of the qkvz row.
struct Window {
    q: MxArray,
    k: MxArray,
    v: MxArray,
    a: MxArray,
    b: MxArray,
    z: MxArray,
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
    let ba = normal(&[1, t, 2 * hv + 32], 1.7, DType::BFloat16);
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

fn rows(x: &MxArray, s: usize, e: usize) -> MxArray {
    x.slice_axis(1, s as i64, e as i64).unwrap()
}

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

/// The chain the real model ran on rows `[s, e)`.
fn chain(
    win: &Window,
    wt: &Weights,
    s: usize,
    e: usize,
    state: Option<&MxArray>,
) -> Result<(MxArray, MxArray)> {
    let qn = crate::nn::rms_norm_scaled(&rows(&win.q, s, e), &wt.wq, 1e-6)?;
    let kn = crate::nn::rms_norm_scaled(&rows(&win.k, s, e), &wt.wk, 1e-6)?;
    let v = rows(&win.v, s, e);
    let (q, k, g, beta, init) = kernel_inputs(
        &qn,
        &kn,
        &v,
        &rows(&win.a, s, e),
        &rows(&win.b, s, e),
        &wt.a_log,
        &wt.dt_bias,
        state,
        wt.tiled,
        None,
    )?;
    let (y, new_state) = gated_delta_kernel(&q, &k, &v, &g, &beta, &init, None)?;
    let t = (e - s) as i64;
    let mut norm = crate::nn::RMSNormGated::new(D as u32, Some(1e-6))?;
    norm.set_weight(&wt.w)?;
    let z = rows(&win.z, s, e).reshape(&[1, t, wt.hv, D])?;
    let out = norm.forward(&y, Some(&z))?.reshape(&[1, t, wt.hv * D])?;
    Ok((out, new_state))
}

const FAMILIES: [(i64, i64, bool, &str); 6] = [
    (16, 48, true, "27B GGUF"),
    (16, 48, false, "27B"),
    (16, 32, true, "35B-A3B MoE GGUF"),
    (16, 32, false, "35B-A3B MoE / 9B / 4B"),
    (16, 16, false, "2B / 0.8B"),
    (16, 64, false, "122B-A10B / 397B-A17B"),
];

/// bf16 a_log/dt_bias: the fused prefill equals the chain on every family,
/// window length (around the 16-row scan block and the 2048-row route
/// chunk), carried and zero state, and chained calls (2048 -> 37 -> 2049
/// rows, the state carried by each side itself); the output is not
/// degenerate. f32 scalars on the same windows too (the old contract).
#[test]
fn bf16_scalars_keep_the_chain_bits() -> Result<()> {
    if !launches() {
        eprintln!("SKIP bf16_scalars_keep_the_chain_bits: no fused prefill");
        return Ok(());
    }
    let mut total = 0usize;
    for scalars in [DType::BFloat16, DType::Float32] {
        for &(hk, hv, tiled, name) in &FAMILIES {
            let wt = weights(hk, hv, tiled, scalars);
            let lengths: &[usize] = if hv == 48 {
                &[1, 15, 16, 17, 33, 64, 2047, 2048, 2049]
            } else {
                &[17, 64, 2049]
            };
            for &t in lengths {
                let win = window(t, &wt);
                let s0 = normal(&[1, hv, D, D], 0.3, DType::Float32);
                for carried in [true, false] {
                    let st = carried.then_some(&s0);
                    let (yf, sf) = fused(&win, &wt, 0, t, st)?;
                    let (yc, sc) = chain(&win, &wt, 0, t, st)?;
                    MxArray::eval_arrays(&[&yf, &sf, &yc, &sc])?;
                    let y_bits = bits(&yc);
                    let nonzero = y_bits.iter().filter(|&&u| u & 0x7fff_ffff != 0).count();
                    assert!(
                        nonzero * 2 > y_bits.len(),
                        "{name} T={t}: degenerate chain output"
                    );
                    let (dy, ds) = (differing(&yc, &yf), differing(&sc, &sf));
                    eprintln!(
                        "X3_SCALARS {scalars:?} {name} T={t} carried={carried}: out {} differ {dy}; state differ {ds}",
                        y_bits.len()
                    );
                    assert_eq!(
                        (dy, ds),
                        (0, 0),
                        "{scalars:?} {name} T={t} carried={carried}"
                    );
                    total += 1;
                }
            }
            let (mut sf, mut sc): (Option<MxArray>, Option<MxArray>) = (None, None);
            for (call, t) in [2048usize, 37, 2049].into_iter().enumerate() {
                let win = window(t, &wt);
                let (yf, nf) = fused(&win, &wt, 0, t, sf.as_ref())?;
                let (yc, nc) = chain(&win, &wt, 0, t, sc.as_ref())?;
                MxArray::eval_arrays(&[&yf, &nf, &yc, &nc])?;
                let (dy, ds) = (differing(&yc, &yf), differing(&nc, &nf));
                eprintln!(
                    "X3_CHAIN {scalars:?} {name} call {call} T={t}: out differ {dy}, state differ {ds}"
                );
                assert_eq!((dy, ds), (0, 0), "{scalars:?} {name} call {call}");
                sf = Some(nf);
                sc = Some(nc);
            }
        }
    }
    eprintln!("X3_SCALARS points {total}");
    Ok(())
}

/// Mixed a_log/dt_bias dtypes and f16 scalars stay off the fused entry.
#[test]
fn other_scalar_dtypes_decline() -> Result<()> {
    if !launches() {
        eprintln!("SKIP other_scalar_dtypes_decline: no fused prefill");
        return Ok(());
    }
    let wt = weights(16, 48, true, DType::BFloat16);
    let win = window(64, &wt);
    for (a_log, dt_bias) in [
        (wt.a_log.astype(DType::Float32)?, wt.dt_bias.clone()),
        (wt.a_log.clone(), wt.dt_bias.astype(DType::Float32)?),
        (
            wt.a_log.astype(DType::Float16)?,
            wt.dt_bias.astype(DType::Float16)?,
        ),
    ] {
        let out = gated_delta_fused_prefill(
            &win.q,
            &win.k,
            &win.v,
            None,
            &GdnPrefill {
                a: &win.a,
                b: &win.b,
                a_log: &a_log,
                dt_bias: &dt_bias,
                wq: &wt.wq,
                wk: &wt.wk,
                qk_eps: 1e-6,
                tiled: true,
                tail: GdnTail {
                    z: &win.z,
                    norm_weight: &wt.w,
                    eps: 1e-6,
                    state_dst: None,
                },
            },
        )?;
        assert!(
            out.is_none(),
            "a_log {:?} dt_bias {:?} must decline",
            a_log.dtype()?,
            dt_bias.dtype()?
        );
    }
    Ok(())
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
    net.set_in_proj_ba_weight(&normal(&[2 * hv, hidden], 4.0 * scale, DType::BFloat16))
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
    // F32 in the checkpoint, stored bf16 as the GGUF load stores them.
    net.set_dt_bias(
        &normal(&[hv], 0.3, DType::Float32)
            .astype(DType::BFloat16)
            .unwrap(),
    );
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

/// The chain forward of the layer (a tape sink keeps the window off the
/// fused prefill; the tape only records lazy clones).
fn chain_forward(net: &GatedDeltaNet, x: &MxArray, cache: &mut ArraysCache) -> Result<MxArray> {
    let mut sink: Option<GdnLayerTape> = None;
    net.forward_with_tape(x, None, Some(cache), true, Some(&mut sink))
}

/// Layer forward with bf16 a_log/dt_bias: the window takes the row scan
/// (one dispatch) and keeps the chain's bits in the output, both cache
/// slots, every captured state and the next window. 27B heads (hidden 64,
/// tiled, T 2049) and the MoE's hidden 2048 / 16/32 (untiled, T 1000).
#[test]
fn layer_with_bf16_scalars_takes_the_scan_and_keeps_the_chain_bits() -> Result<()> {
    if !launches() {
        eprintln!(
            "SKIP layer_with_bf16_scalars_takes_the_scan_and_keeps_the_chain_bits: no fused prefill"
        );
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
        let label = format!("hidden={hidden} hk={hk} hv={hv} tiled={tiled} T={t}");
        let net = net(hidden, hk, hv, tiled);
        assert_eq!(
            (net.get_a_log().dtype()?, net.get_dt_bias().dtype()?),
            (DType::BFloat16, DType::BFloat16)
        );
        let warm = normal(&[1, 5, hidden], 1.0, DType::BFloat16);
        let chunk = normal(&[1, t as i64, hidden], 1.0, DType::BFloat16);
        let next = normal(&[1, 130, hidden], 1.0, DType::BFloat16);
        let mut base = ArraysCache::new(2);
        net.forward(&warm, None, Some(&mut base), true)?;
        MxArray::eval_arrays(&[base.get(0).unwrap(), base.get(1).unwrap()])?;

        let mut fc = cache_copy(&base);
        let mut yf = None;
        let scans = row_scans(|| {
            let y = net.forward(&chunk, None, Some(&mut fc), true).unwrap();
            MxArray::eval_arrays(&[&y, fc.get(1).unwrap()]).unwrap();
            yf = Some(y);
        });
        let yf = yf.unwrap();
        assert_eq!(scans, 1, "{label}: row-scan dispatches of the fused window");

        let mut cc = cache_copy(&base);
        let mut yc = None;
        let chain_scans = row_scans(|| {
            let y = chain_forward(&net, &chunk, &mut cc).unwrap();
            MxArray::eval_arrays(&[&y, cc.get(1).unwrap()]).unwrap();
            yc = Some(y);
        });
        let yc = yc.unwrap();
        assert_eq!(
            chain_scans, 0,
            "{label}: the chain reference ran no row scan"
        );

        let d = differing(&yc, &yf);
        eprintln!("X3_LAYER {label} out differ {d}");
        assert_eq!(d, 0, "{label}: out");
        for slot in 0..2 {
            let d = differing(cc.get(slot).unwrap(), fc.get(slot).unwrap());
            eprintln!("X3_LAYER {label} slot {slot} differ {d}");
            assert_eq!(d, 0, "{label}: cache slot {slot}");
        }
        let nf = net.forward(&next, None, Some(&mut fc), true)?;
        let nc = chain_forward(&net, &next, &mut cc)?;
        MxArray::eval_arrays(&[&nf, &nc])?;
        assert_eq!(differing(&nc, &nf), 0, "{label}: next window");
        assert_eq!(
            differing(cc.get(1).unwrap(), fc.get(1).unwrap()),
            0,
            "{label}: state after the next window"
        );

        let mut kc = cache_copy(&base);
        let (yk, captured) = net.forward_captured(
            &chunk,
            Some(&mut kc),
            true,
            None,
            &GdnCapturePlan { rows: cuts },
        )?;
        MxArray::eval_arrays(&[&yk])?;
        assert_eq!(differing(&yc, &yk), 0, "{label}: captured forward out");
        assert_eq!(captured.len(), cuts.len());
        // A capture is the window's own arithmetic on rows [0, c); the
        // chain reference of a c-row window takes that route only from 64
        // rows (shorter windows go to gdn_prepare).
        for (&c, cap) in cuts.iter().zip(&captured).filter(|(c, _)| **c >= 64) {
            let mut pc = cache_copy(&base);
            chain_forward(&net, &chunk.slice_axis(1, 0, c as i64)?, &mut pc)?;
            let d = differing(pc.get(1).unwrap(), &cap.rec);
            let dc = differing(pc.get(0).unwrap(), &cap.conv);
            eprintln!("X3_LAYER {label} capture {c}: state differ {d}, conv differ {dc}");
            assert_eq!((d, dc), (0, 0), "{label}: capture at {c}");
        }
    }
    Ok(())
}
