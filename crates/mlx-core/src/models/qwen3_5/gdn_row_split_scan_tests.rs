//! Checks of the row-split GDN prefill scan against the
//! one-value-head-per-threadgroup kernel, on raw bits (NaN payloads and
//! signed zeros included): every window length around the scan block, two
//! different epsilons, hostile values, long state chains that cross from one
//! kernel to the other, sidecars bound at an offset, and the layer route at
//! its edge.

use super::*;

const QK_EPS: f32 = 1e-6;
const TAIL_EPS: f32 = 3e-5;

fn raw16(a: &MxArray) -> Vec<u16> {
    a.eval();
    a.to_uint16_native().expect("bf16 bits")
}

fn raw32(a: &MxArray) -> Vec<u32> {
    a.eval();
    a.to_float32()
        .expect("f32 values")
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

/// The fused prefill on rows `[s, e)` with the q/k epsilon and the tail
/// epsilon apart, so a kernel that reads the wrong one shows.
fn call(
    win: &Window,
    wt: &Weights,
    s: usize,
    e: usize,
    state: Option<&MxArray>,
    head: bool,
) -> (Vec<u16>, Vec<u32>, MxArray) {
    let (a, b, z) = (rows(&win.a, s, e), rows(&win.b, s, e), rows(&win.z, s, e));
    let (y, st) = on_kernel(head, || {
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
                qk_eps: QK_EPS,
                tiled: wt.tiled,
                tail: GdnTail {
                    z: &z,
                    norm_weight: &wt.w,
                    eps: TAIL_EPS,
                    state_dst: None,
                },
            },
        )
    })
    .expect("fused prefill")
    .expect("the fused GDN prefill declined");
    MxArray::eval_arrays(&[&y, &st]).expect("eval");
    (raw16(&y), raw32(&st), st)
}

fn count_differing<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    assert_eq!(a.len(), b.len(), "compared arrays differ in size");
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// Every window length from 1 to 70 rows, lengths around the 16-row block
/// and the 32-row reference chunk up to 6,000 rows (past a 2,048-row chunk),
/// each starting at a different row of a longer window, from a carried state
/// and from the zero state, on both 27B head maps: 0 differing raw bits in
/// the output and the state.
/// Mutations v-e4-stride (the scan walks v with the dense row stride),
/// v-e4-tilemap (the scan takes the other head map) and v-e4-eps (the
/// prologue and the tail swap their epsilons) turn it red.
#[test]
fn every_window_length_keeps_the_head_kernel_bits() {
    if !launches() {
        eprintln!("SKIP every_window_length_keeps_the_head_kernel_bits: no fused prefill");
        return;
    }
    let lengths: Vec<usize> = (1..=70)
        .chain([
            95, 96, 97, 127, 128, 129, 255, 257, 1023, 1025, 4095, 4096, 4097, 6000,
        ])
        .collect();
    let mut points = 0;
    for &(hk, hv, tiled, name) in &FAMILIES[..2] {
        let wt = weights(hk, hv, tiled);
        let win = window(6016, &wt);
        let carried = state(&wt);
        carried.eval();
        for &t in &lengths {
            let s = (t * 5) % 16;
            for st in [Some(&carried), None] {
                let (yh, sh, _) = call(&win, &wt, s, s + t, st, true);
                let (yr, sr, _) = call(&win, &wt, s, s + t, st, false);
                let (dy, ds) = (count_differing(&yh, &yr), count_differing(&sh, &sr));
                assert_eq!(
                    (dy, ds),
                    (0, 0),
                    "{name} T={t} from row {s}, carried={}: (out, state) raw bits differ",
                    st.is_some()
                );
                points += 1;
            }
        }
    }
    eprintln!("V2_E4 lengths: {points} points, 0 differing");
}

/// `[1, t, width]` bf16 rows by row class: normal, x 1e18, zeros, x 1e-38
/// (subnormal bf16), normal, and a row with an inf, a NaN and a -inf lane.
/// Before the last eighth of the rows the three lanes fall in the first
/// `early` columns only (none for 0), so a part of the outputs stays finite.
fn hostile(t: usize, width: usize, std: f32, seed: u64, early: usize) -> MxArray {
    // SAFETY: a plain global setter in the MLX FFI.
    unsafe { mlx_sys::mlx_seed(seed) };
    let base = MxArray::random_normal(&[t as i64, width as i64], 0.0, 1.0, Some(DType::Float32))
        .expect("normal");
    let mut v = base.to_float32().expect("values").to_vec();
    for r in 0..t {
        let scale = match r % 6 {
            1 => std * 1e18,
            2 => 0.0,
            3 => 1e-38,
            _ => std,
        };
        for x in &mut v[r * width..(r + 1) * width] {
            *x *= scale;
        }
        let lanes = if r >= t - t / 8 { width } else { early };
        if r % 6 == 5 && lanes > 0 {
            v[r * width + (r * 13) % lanes] = f32::INFINITY;
            v[r * width + (r * 29 + 7) % lanes] = f32::NAN;
            v[r * width + (r * 31 + 11) % lanes] = f32::NEG_INFINITY;
        }
    }
    let x = MxArray::from_float32(&v, &[1, t as i64, width as i64])
        .expect("array")
        .astype(DType::BFloat16)
        .expect("bf16");
    x.eval();
    x
}

/// [`window`] over hostile rows.
fn hostile_window(t: usize, wt: &Weights, seed: u64) -> Window {
    let (hk, hv, tl) = (wt.hk, wt.hv, t as i64);
    // Early non-finite lanes in q only: they reach the output rows, not the
    // state.
    let conv = hostile(
        t,
        (2 * hk * D + hv * D) as usize,
        2.0,
        seed,
        (hk * D) as usize,
    );
    let head = |s: i64, e: i64, heads: i64| {
        conv.slice_axis(2, s, e)
            .unwrap()
            .reshape(&[1, tl, heads, D])
            .unwrap()
    };
    let ba = hostile(t, (2 * hv + 32) as usize, 0.3, seed + 1, 0);
    let z_width = (2 * hk * D + 2 * hv * D + 96) as usize;
    let z = hostile(t, z_width, 2.0, seed + 2, z_width)
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

/// A carried state with subnormal, huge and negative-zero rows among normal
/// ones in every head, and NaN and infinite rows in the odd heads.
fn hostile_state(wt: &Weights) -> MxArray {
    let n = (wt.hv * D * D) as usize;
    let base = normal(&[1, wt.hv, D, D], 0.3, DType::Float32);
    let mut v = base.to_float32().expect("values").to_vec();
    let row = D as usize;
    for r in 0..n / row {
        let cell = &mut v[r * row..(r + 1) * row];
        match r % 11 {
            3 => cell.iter_mut().for_each(|x| *x *= 1e-40),
            5 => cell.iter_mut().for_each(|x| *x *= 1e30),
            7 => cell.iter_mut().for_each(|x| *x = -0.0),
            8 if (r / row) % 2 == 1 => cell[r % row] = f32::NAN,
            9 if (r / row) % 2 == 1 => cell[r % row] = f32::INFINITY,
            _ => {}
        }
    }
    let state = MxArray::from_float32(&v, &[1, wt.hv, D, D]).expect("state");
    state.eval();
    state
}

/// Zeros, overflowing rows, subnormals, infinities and NaNs in every input
/// and in the carried state: the head kernel gives the same raw bits twice,
/// and the row scan gives those bits (NaN payloads and signs included).
#[test]
fn hostile_values_keep_the_head_kernel_bits() {
    if !launches() {
        eprintln!("SKIP hostile_values_keep_the_head_kernel_bits: no fused prefill");
        return;
    }
    for &(hk, hv, tiled, name) in &[FAMILIES[0], FAMILIES[3]] {
        let wt = weights(hk, hv, tiled);
        for (t, seed) in [(100usize, 41u64), (2049, 43)] {
            let win = hostile_window(t, &wt, seed);
            let carried = hostile_state(&wt);
            for st in [Some(&carried), None] {
                let (yh, sh, _) = call(&win, &wt, 0, t, st, true);
                let (yh2, sh2, _) = call(&win, &wt, 0, t, st, true);
                assert_eq!(
                    (count_differing(&yh, &yh2), count_differing(&sh, &sh2)),
                    (0, 0),
                    "{name} T={t}: the head kernel does not repeat its own bits"
                );
                let (yr, sr, _) = call(&win, &wt, 0, t, st, false);
                let non_finite = yh.iter().filter(|&&u| u & 0x7f80 == 0x7f80).count();
                let finite_nonzero = yh
                    .iter()
                    .filter(|&&u| u & 0x7f80 != 0x7f80 && u & 0x7fff != 0)
                    .count();
                assert!(
                    non_finite > 0 && finite_nonzero > 0,
                    "{name} T={t}: the case holds {non_finite} non-finite and \
                     {finite_nonzero} finite non-zero outputs"
                );
                let (dy, ds) = (count_differing(&yh, &yr), count_differing(&sh, &sr));
                eprintln!(
                    "V2_E4 hostile {name} T={t} carried={}: {non_finite} non-finite, \
                     {finite_nonzero} finite non-zero outputs; differ out {dy}, state {ds}",
                    st.is_some()
                );
                assert_eq!((dy, ds), (0, 0), "{name} T={t}: raw bits differ");
            }
        }
    }
}

/// Twenty calls chained through the state (1 to 2,048 rows each): the row
/// scan's chain, the head kernel's chain and two chains that switch kernel
/// at every call (a state written by one kernel read by the other, both
/// ways) give the same output at every call and the same final state.
/// Mutation v-e4-nostate (the scan starts from zeros whatever it is handed)
/// turns it red.
#[test]
fn states_chain_across_many_calls_and_both_kernels() {
    if !launches() {
        eprintln!("SKIP states_chain_across_many_calls_and_both_kernels: no fused prefill");
        return;
    }
    let lengths = [
        1usize, 2, 5, 16, 17, 31, 64, 100, 33, 7, 2048, 3, 64, 65, 129, 48, 15, 1, 1, 300,
    ];
    let total: usize = lengths.iter().sum();
    for &(hk, hv, tiled, name) in &[FAMILIES[0], FAMILIES[3], FAMILIES[6]] {
        let wt = weights(hk, hv, tiled);
        let win = window(total, &wt);
        let first = state(&wt);
        first.eval();
        let chain = |pick: &dyn Fn(usize) -> bool| {
            let mut st = first.clone();
            let mut outs = Vec::new();
            let mut at = 0;
            for (i, &t) in lengths.iter().enumerate() {
                let (y, _, next) = call(&win, &wt, at, at + t, Some(&st), pick(i));
                outs.push(y);
                st = next;
                at += t;
            }
            (outs, raw32(&st))
        };
        let head = chain(&|_| true);
        for (tag, picked) in [
            ("row scan", chain(&|_| false)),
            ("head first, alternating", chain(&|i| i % 2 == 0)),
            ("row first, alternating", chain(&|i| i % 2 == 1)),
        ] {
            for (i, (a, b)) in head.0.iter().zip(&picked.0).enumerate() {
                assert_eq!(
                    count_differing(a, b),
                    0,
                    "{name} {tag}: call {i} ({} rows) output differs",
                    lengths[i]
                );
            }
            assert_eq!(
                count_differing(&head.1, &picked.1),
                0,
                "{name} {tag}: final state differs"
            );
        }
        eprintln!("V2_E4 chain {name}: 20 calls x 4 chains, 0 differing");
    }
}

/// The carried state, the gate scalars and the norm weights bound at a
/// non-zero offset inside a larger buffer give the bits of their own
/// copies, on both kernels.
#[test]
fn sidecars_at_an_offset_keep_the_bits() {
    if !launches() {
        eprintln!("SKIP sidecars_at_an_offset_keep_the_bits: no fused prefill");
        return;
    }
    let (hk, hv, tiled, name) = FAMILIES[0];
    let wt = weights(hk, hv, tiled);
    let t = 200;
    let win = window(t, &wt);
    let plain_state = state(&wt);
    plain_state.eval();
    let copy = |x: &MxArray| {
        let host = x.to_float32().expect("values").to_vec();
        let shape = x.shape().expect("shape").to_vec();
        let dtype = x.dtype().expect("dtype");
        MxArray::from_float32(&host, &shape)
            .expect("array")
            .astype(dtype)
            .expect("dtype")
    };
    // `x` as rows [3, 3 + n) of a longer buffer of the same dtype.
    let inside = |x: &MxArray, axis_len: i64| {
        let pad_shape: Vec<i64> = {
            let mut s = x.shape().expect("shape").to_vec();
            s[0] = 3;
            s
        };
        let pad = normal(&pad_shape, 1.0, x.dtype().expect("dtype"));
        let joined =
            MxArray::concatenate(&MxArray::concatenate(&pad, x, 0).unwrap(), &pad, 0).unwrap();
        joined.eval();
        joined.slice_axis(0, 3, 3 + axis_len).unwrap()
    };
    let offset = Weights {
        hk,
        hv,
        tiled,
        w: inside(&wt.w, D),
        a_log: inside(&wt.a_log, hv),
        dt_bias: inside(&wt.dt_bias, hv),
        wq: inside(&wt.wq, D),
        wk: inside(&wt.wk, D),
    };
    let offset_state = inside(&plain_state, 1);
    assert_eq!(
        raw32(&offset_state),
        raw32(&copy(&plain_state)),
        "the offset state does not hold the state's values"
    );
    let (y0, s0, _) = call(&win, &wt, 0, t, Some(&plain_state), true);
    for head in [true, false] {
        let (y, s, _) = call(&win, &offset, 0, t, Some(&offset_state), head);
        assert_eq!(
            (count_differing(&y0, &y), count_differing(&s0, &s)),
            (0, 0),
            "{name} head={head}: sidecars at an offset changed the bits"
        );
    }
}

/// The layer forward at the fused route's edge: 63 rows stay off the fused
/// prefill, 64 / 65 / 80 rows take the row scan and keep the head kernel's
/// output and cache bits.
#[test]
fn the_layer_route_edge_keeps_the_bits() -> Result<()> {
    if !launches() {
        eprintln!("SKIP the_layer_route_edge_keeps_the_bits: no fused prefill");
        return Ok(());
    }
    let net = net(64, 16, 48, true);
    let warm = normal(&[1, 5, 64], 1.0, DType::BFloat16);
    let mut base = ArraysCache::new(2);
    net.forward(&warm, None, Some(&mut base), true)?;
    MxArray::eval_arrays(&[base.get(0).unwrap(), base.get(1).unwrap()])?;
    for (t, fused_route) in [(63usize, false), (64, true), (65, true), (80, true)] {
        let x = normal(&[1, t as i64, 64], 1.0, DType::BFloat16);
        x.eval();
        let run = |head: bool| -> Result<(Vec<u16>, Vec<u32>, (u64, u64))> {
            let mut out = None;
            let counts = counted(|| {
                out = Some(on_kernel(head, || {
                    let mut cache = cache_copy(&base);
                    let y = net.forward(&x, None, Some(&mut cache), true)?;
                    Ok::<_, Error>((raw16(&y), raw32(cache.get(1).unwrap())))
                }));
            });
            let (y, st) = out.expect("ran")?;
            Ok((y, st, counts))
        };
        let (yh, sh, head_counts) = run(true)?;
        let (yr, sr, row_counts) = run(false)?;
        let want = |head: bool| match (fused_route, head) {
            (false, _) => (0, 0),
            (true, true) => (0, 1),
            (true, false) => (1, 0),
        };
        assert_eq!(head_counts, want(true), "T={t}: head switch (row, head)");
        assert_eq!(row_counts, want(false), "T={t}: product (row, head)");
        assert_eq!(
            (count_differing(&yh, &yr), count_differing(&sh, &sr)),
            (0, 0),
            "layer T={t}: (out, state) raw bits differ"
        );
    }
    Ok(())
}
