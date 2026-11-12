//! Deferred prior-prefix settlement must preserve every current-window bit.
use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
struct Environment(Vec<(&'static str, Option<std::ffi::OsString>)>);
impl Environment {
    fn new() -> Self {
        Self(
            ["MLX_GDN_SCAN8", "MLX_GDN_T8_STORAGE"]
                .into_iter()
                .map(|name| (name, std::env::var_os(name)))
                .collect(),
        )
    }
    fn set(scan: &str, storage: &str) {
        unsafe {
            std::env::set_var("MLX_GDN_SCAN8", scan);
            std::env::set_var("MLX_GDN_T8_STORAGE", storage);
        }
    }
}
impl Drop for Environment {
    fn drop(&mut self) {
        for (name, value) in &self.0 {
            unsafe {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }
}
fn array(shape: &[i64], seed: usize, dtype: DType) -> MxArray {
    let size: usize = shape.iter().map(|&d| d as usize).product();
    let data = (0..size)
        .map(|i| (((i * 17 + seed * 53) % 251) as f32 - 125.0) / 501.0)
        .collect::<Vec<_>>();
    MxArray::from_float32(&data, shape)
        .unwrap()
        .astype(dtype)
        .unwrap()
}
struct Window {
    a: Vec<MxArray>,
}
impl Window {
    fn new(rows: i64, hv: i64, seed: usize, strided: bool) -> Self {
        Self::with_heads(rows, hv / 2, hv, seed, strided)
    }
    fn with_heads(rows: i64, hk: i64, hv: i64, seed: usize, strided: bool) -> Self {
        let width = 2 * hk * 128 + hv * 128;
        // Odd element offsets and padded row strides exercise the actual split views.
        let packed = array(&[1, rows, 1 + width + hv * 130 + 3], seed, DType::BFloat16);
        let take = |start: i64, end: i64| {
            let view = packed
                .slice(&[0, 0, 1 + start], &[1, rows, 1 + end])
                .unwrap();
            if strided {
                view
            } else {
                array(
                    &[1, rows, end - start],
                    seed + start as usize,
                    DType::BFloat16,
                )
            }
        };
        Self {
            a: vec![
                take(0, width),
                take(width + hv * 128, width + hv * 129),
                take(width + hv * 129, width + hv * 130),
                array(&[width, 4], seed + 2, DType::Float32),
                array(&[3, width], seed + 3, DType::BFloat16),
                MxArray::from_float32(&vec![-0.3; hv as usize], &[hv]).unwrap(),
                array(&[hv], seed + 4, DType::Float32),
                array(&[1, hv, 128, 128], seed + 5, DType::Float32),
                take(width, width + hv * 128),
                MxArray::ones(&[128], Some(DType::BFloat16)).unwrap(),
            ],
        }
    }
    fn complete(
        &self,
        prepared: bool,
        dst: Option<(&MxArray, &MxArray)>,
        pending: Option<[&MxArray; 6]>,
    ) -> Option<GdnCompleteOut> {
        self.complete_heads(prepared, dst, pending.map(|p| (p, true)))
    }
    fn complete_heads(
        &self,
        prepared: bool,
        dst: Option<(&MxArray, &MxArray)>,
        pending: Option<([&MxArray; 6], bool)>,
    ) -> Option<GdnCompleteOut> {
        let a = &self.a;
        let p = GdnPrologue {
            qkv: &a[0],
            a: &a[1],
            b: &a[2],
            conv: &a[3],
            history: &a[4],
            scale: &a[5],
            dt_bias: &a[6],
            history_dst: dst.map(|d| d.1),
        };
        let t = GdnTail {
            z: &a[8],
            norm_weight: &a[9],
            eps: 1e-6,
            state_dst: dst.map(|d| d.0),
        };
        gated_delta_fused_complete_impl(&p, &a[7], &t, prepared, pending)
    }
}
fn bits(a: &MxArray) -> Vec<u32> {
    if a.dtype().unwrap() == DType::Uint32 {
        return a.to_uint32().unwrap().as_ref().to_vec();
    }
    a.astype(DType::Float32)
        .unwrap()
        .to_float32()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}
fn exact(a: &MxArray, b: &MxArray, label: &str) {
    assert_eq!(
        a.shape().unwrap().as_ref(),
        b.shape().unwrap().as_ref(),
        "{label} shape"
    );
    assert_eq!(bits(a), bits(b), "{label}");
}
fn settle_t1(
    tape: &GdnKernelTape,
    state: &MxArray,
    history: &MxArray,
    raw: &MxArray,
    keep: usize,
) -> (MxArray, MxArray) {
    let mut state = state.clone();
    for step in 0..keep as i64 {
        let row = |a: &MxArray| a.slice_axis(1, step, step + 1).unwrap();
        state = gated_delta_kernel(
            &row(&tape.q),
            &row(&tape.k),
            &row(&tape.v),
            &row(&tape.g),
            &row(&tape.beta),
            &state,
            None,
        )
        .unwrap()
        .1;
    }
    let history =
        MxArray::concatenate_many(vec![history, &raw.squeeze(Some(&[0])).unwrap()], Some(0))
            .unwrap()
            .slice_axis(0, keep as i64, keep as i64 + 3)
            .unwrap();
    (state, history)
}
fn compare(actual: &GdnCompleteOut, expected: &GdnCompleteOut, settled: (&MxArray, &MxArray)) {
    exact(&actual.out, &expected.out, "current gated output");
    exact(&actual.state, settled.0, "settled recurrent state");
    exact(&actual.history, settled.1, "settled convolution history");
    for (a, b) in [
        (&actual.tape.q, &expected.tape.q),
        (&actual.tape.k, &expected.tape.k),
        (&actual.tape.v, &expected.tape.v),
        (&actual.tape.g, &expected.tape.g),
        (&actual.tape.beta, &expected.tape.beta),
    ] {
        exact(a, b, "current tape");
    }
    match (&actual.prepared_sg8, &expected.prepared_sg8) {
        (Some(a), Some(b)) => {
            exact(&a.packed, &b.packed, "native SG8 packed words");
            exact(&a.sums, &b.sums, "native SG8 FP32 sums");
        }
        (None, None) => {}
        _ => panic!("prepared output mismatch"),
    }
}
fn pending<'a>(tape: &'a GdnKernelTape, raw: &'a MxArray, keep: &'a MxArray) -> [&'a MxArray; 6] {
    [&tape.k, &tape.v, &tape.g, &tape.beta, raw, keep]
}

#[test]
fn deferred_gdn_matches_independent_t1_prefix_for_every_count_and_width() {
    let _serial = SERIAL.lock().unwrap();
    let _env = Environment::new();
    if !crate::engine::persistence::compiled_forward_backend_available() {
        return;
    }
    for scan in ["0", "1"] {
        for storage in ["0", "1"] {
            Environment::set(scan, storage);
            let prior = Window::new(8, 4, 11, true);
            let prior_out = prior
                .complete(false, None, None)
                .expect("ordinary complete GDN");
            for keep in 0..=8 {
                let settled =
                    settle_t1(&prior_out.tape, &prior.a[7], &prior.a[4], &prior.a[0], keep);
                let keep_array = MxArray::from_int32(&[keep as i32], &[1]).unwrap();
                for rows in 1..=8 {
                    let mut current =
                        Window::new(rows, 4, 31 + keep + rows as usize, rows % 2 == 1);
                    current.a[4] = prior.a[4].clone();
                    current.a[7] = prior.a[7].clone();
                    let state_dst = MxArray::zeros(
                        current.a[7].shape().unwrap().as_ref(),
                        Some(DType::Float32),
                    )
                    .unwrap();
                    let history_dst = MxArray::zeros(
                        current.a[4].shape().unwrap().as_ref(),
                        Some(DType::BFloat16),
                    )
                    .unwrap();
                    let prepared = rows == 8;
                    let actual = current
                        .complete(
                            prepared,
                            Some((&state_dst, &history_dst)),
                            Some(pending(&prior_out.tape, &prior.a[0], &keep_array)),
                        )
                        .unwrap();
                    current.a[4] = settled.1.clone();
                    current.a[7] = settled.0.clone();
                    let expected = current.complete(prepared, None, None).unwrap();
                    compare(&actual, &expected, (&settled.0, &settled.1));
                    assert!(unsafe {
                        sys::mlx_array_aliases(actual.state.as_raw_ptr(), state_dst.as_raw_ptr())
                    });
                    assert!(unsafe {
                        sys::mlx_array_aliases(
                            actual.history.as_raw_ptr(),
                            history_dst.as_raw_ptr(),
                        )
                    });
                }
            }
        }
    }
}

struct Builder {
    traces: AtomicUsize,
    prepared: bool,
}
unsafe extern "C" fn build(
    context: *mut std::ffi::c_void,
    inputs: *const *const sys::mlx_array,
    count: usize,
    outputs: *mut *mut sys::mlx_array,
    n_outputs: usize,
) -> bool {
    if context.is_null() || count != 18 {
        return false;
    }
    let ctx = unsafe { &*context.cast::<Builder>() };
    if n_outputs != if ctx.prepared { 11 } else { 9 } {
        return false;
    }
    ctx.traces.fetch_add(1, Ordering::Relaxed);
    let owned = unsafe { std::slice::from_raw_parts(inputs, count) }
        .iter()
        .map(|&a| MxArray::from_handle(a.cast_mut(), "deferred trace input").unwrap())
        .collect::<Vec<_>>();
    let a = owned.iter().map(MxArray::as_raw_ptr).collect::<Vec<_>>();
    unsafe {
        sys::mlx_gdn_fused_complete_deferred(
            a[0],
            a[1],
            a[2],
            a[3],
            a[4],
            a[5],
            a[6],
            a[7],
            a[8],
            a[9],
            1e-6,
            a[16],
            a[17],
            a[10..16].as_ptr(),
            ctx.prepared,
            true,
            outputs,
        )
    }
}
fn from_outputs(mut outputs: Vec<MxArray>) -> GdnCompleteOut {
    let raw = outputs.pop();
    let prepared_sg8 = if outputs.len() == 10 {
        let sums = outputs.pop().unwrap();
        let packed = outputs.pop().unwrap();
        Some(crate::models::quantized_linear::PreparedSg8 { packed, sums })
    } else {
        None
    };
    let mut iter = outputs.into_iter();
    GdnCompleteOut {
        out: iter.next().unwrap(),
        state: iter.next().unwrap(),
        tape: GdnKernelTape {
            q: iter.next().unwrap(),
            k: iter.next().unwrap(),
            v: iter.next().unwrap(),
            g: iter.next().unwrap(),
            beta: iter.next().unwrap(),
        },
        history: iter.next().unwrap(),
        prepared_sg8,
        raw,
    }
}
#[test]
fn deferred_gdn_one_trace_changes_counts_widths_strides_and_parity() {
    let _serial = SERIAL.lock().unwrap();
    let _env = Environment::new();
    if !crate::engine::persistence::compiled_forward_backend_available() {
        return;
    }
    for (index, (scan, storage, prepared)) in [
        ("0", "0", false),
        ("1", "1", false),
        ("0", "1", true),
        ("1", "0", true),
    ]
    .into_iter()
    .enumerate()
    {
        Environment::set(scan, storage);
        let key = 0x6DD0_DEFE_0000_0000 + index as u64;
        let mut ctx = Builder {
            traces: AtomicUsize::new(0),
            prepared,
        };
        for (cycle, keep) in [0, 8, 1, 3, 7, 2, 6, 4, 5, 8].into_iter().enumerate() {
            let prior = Window::new(8, 32, 71 + cycle, cycle % 2 == 0);
            let prior_out = prior.complete(false, None, None).unwrap();
            let rows = if prepared {
                8
            } else {
                [8, 1, 7, 3, 8][cycle % 5]
            };
            let mut current = Window::new(rows, 32, 151 + cycle, cycle % 2 == 1);
            current.a[4] = prior.a[4].clone();
            current.a[7] = prior.a[7].clone();
            let count = MxArray::from_int32(&[keep as i32], &[1]).unwrap();
            let state_dst =
                MxArray::zeros(current.a[7].shape().unwrap().as_ref(), Some(DType::Float32))
                    .unwrap();
            let hist_dst = MxArray::zeros(
                current.a[4].shape().unwrap().as_ref(),
                Some(DType::BFloat16),
            )
            .unwrap();
            let mut input = current.a.iter().collect::<Vec<_>>();
            input.extend(pending(&prior_out.tape, &prior.a[0], &count));
            input.extend([&state_dst, &hist_dst]);
            MxArray::eval_arrays(&input).unwrap();
            let raw = input
                .iter()
                .map(|a| a.as_raw_ptr().cast_const())
                .collect::<Vec<_>>();
            let mut output = vec![std::ptr::null_mut(); if prepared { 11 } else { 9 }];
            assert!(unsafe {
                sys::mlx_compiled_graph_invoke(
                    key,
                    Some(build),
                    (&mut ctx as *mut Builder).cast(),
                    raw.as_ptr(),
                    raw.len(),
                    output.as_mut_ptr(),
                    output.len(),
                    true,
                )
            });
            let actual = from_outputs(
                output
                    .into_iter()
                    .map(|p| MxArray::from_handle(p, "deferred graph output").unwrap())
                    .collect(),
            );
            let settled = settle_t1(&prior_out.tape, &prior.a[7], &prior.a[4], &prior.a[0], keep);
            current.a[4] = settled.1.clone();
            current.a[7] = settled.0.clone();
            let expected = current.complete(prepared, None, None).unwrap();
            compare(&actual, &expected, (&settled.0, &settled.1));
        }
        assert_eq!(ctx.traces.load(Ordering::Relaxed), 1);
        unsafe { sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
    }
}

#[test]
fn deferred_gdn_rejects_invalid_pending_shapes_and_destination_aliases() {
    let _serial = SERIAL.lock().unwrap();
    let _env = Environment::new();
    if !crate::engine::persistence::compiled_forward_backend_available() {
        return;
    }
    let prior = Window::new(8, 4, 301, true);
    let prior_out = prior.complete(false, None, None).unwrap();
    let current = Window::new(8, 4, 311, true);
    let count = MxArray::from_int32(&[3], &[1]).unwrap();
    let invalid = MxArray::from_int32(&[3, 4], &[2]).unwrap();
    assert!(
        current
            .complete(
                false,
                None,
                Some(pending(&prior_out.tape, &prior.a[0], &invalid))
            )
            .is_none()
    );
    for (raw_count, bounded) in [(-1, 0), (9, 8), (i32::MAX, 8)] {
        let count = MxArray::from_int32(&[raw_count], &[1]).unwrap();
        let actual = current
            .complete(
                false,
                None,
                Some(pending(&prior_out.tape, &prior.a[0], &count)),
            )
            .unwrap();
        let settled = settle_t1(
            &prior_out.tape,
            &current.a[7],
            &current.a[4],
            &prior.a[0],
            bounded,
        );
        exact(
            &actual.state,
            &settled.0,
            "raw count is bounded before every tape read",
        );
        exact(
            &actual.history,
            &settled.1,
            "raw count bounds virtual history",
        );
    }
    let original_state = bits(&current.a[7]);
    let original_history = bits(&current.a[4]);
    // Failure occurs before a destination write, after evaluated storage is known.
    let aliased = current
        .complete(
            false,
            Some((&current.a[7], &current.a[4])),
            Some(pending(&prior_out.tape, &prior.a[0], &count)),
        )
        .unwrap();
    assert!(MxArray::eval_arrays(&[&aliased.out, &aliased.state, &aliased.history]).is_err());
    assert_eq!(bits(&current.a[7]), original_state);
    assert_eq!(bits(&current.a[4]), original_history);
    // A fresh valid graph must recover after the rejected invocation.
    let recovered = current
        .complete(
            false,
            None,
            Some(pending(&prior_out.tape, &prior.a[0], &count)),
        )
        .unwrap();
    MxArray::eval_arrays(&[&recovered.out, &recovered.state, &recovered.history]).unwrap();
}

#[test]
fn deferred_gdn_chain_settles_every_accepted_prefix_before_publication() {
    let _serial = SERIAL.lock().unwrap();
    let _env = Environment::new();
    if !crate::engine::persistence::compiled_forward_backend_available() {
        return;
    }
    Environment::set("1", "1");
    let first = Window::new(8, 4, 401, false);
    let empty = first.complete(false, None, None).unwrap();
    let mut physical = (first.a[7].clone(), first.a[4].clone());
    let mut logical = physical.clone();
    let mut prior_tape = empty.tape;
    let mut prior_raw = first.a[0].clone();
    let mut prior_keep = 0;
    for (cycle, keep) in [8, 1, 8, 3, 7, 2, 6, 4, 5, 8].into_iter().enumerate() {
        let mut current = Window::new(8, 4, 431 + cycle, cycle % 2 == 1);
        current.a[7] = physical.0.clone();
        current.a[4] = physical.1.clone();
        let count = MxArray::from_int32(&[prior_keep as i32], &[1]).unwrap();
        let actual = current
            .complete(true, None, Some(pending(&prior_tape, &prior_raw, &count)))
            .unwrap();
        current.a[7] = logical.0.clone();
        current.a[4] = logical.1.clone();
        let ordinary = current.complete(true, None, None).unwrap();
        compare(&actual, &ordinary, (&logical.0, &logical.1));
        logical = settle_t1(&ordinary.tape, &logical.0, &logical.1, &current.a[0], keep);
        physical = (actual.state, actual.history);
        prior_tape = actual.tape;
        prior_raw = current.a[0].clone();
        prior_keep = keep;
    }
    let published = settle_t1(
        &prior_tape,
        &physical.0,
        &physical.1,
        &prior_raw,
        prior_keep,
    );
    exact(&published.0, &logical.0, "finish recurrent frontier");
    exact(&published.1, &logical.1, "finish convolution frontier");
}

/// Rows `>= keep` of a tape array replaced by NaN (Splash
/// `replayAcrossBatches` poisons the bytes past the retained prefix).
fn poison(a: &MxArray, keep: usize) -> MxArray {
    let rows = a.shape_at(1).unwrap();
    let keep = keep as i64;
    if keep == rows {
        return a.clone();
    }
    let mut shape = a.shape().unwrap().as_ref().to_vec();
    shape[1] = rows - keep;
    let nan = MxArray::full(&shape, Either::A(f64::NAN), Some(a.dtype().unwrap())).unwrap();
    MxArray::concatenate(&a.slice_axis(1, 0, keep).unwrap(), &nan, 1).unwrap()
}

/// The zero one-row tape and keep 0 for `hk`/`hv` heads of a `width` conv.
fn zero_pending(hk: i64, hv: i64, width: i64) -> [MxArray; 6] {
    [
        MxArray::zeros(&[1, 1, hk, 128], Some(DType::BFloat16)).unwrap(),
        MxArray::zeros(&[1, 1, hv, 128], Some(DType::BFloat16)).unwrap(),
        MxArray::zeros(&[1, 1, hv], Some(DType::Float32)).unwrap(),
        MxArray::zeros(&[1, 1, hv], Some(DType::BFloat16)).unwrap(),
        MxArray::zeros(&[1, 1, width], Some(DType::BFloat16)).unwrap(),
        MxArray::from_int32(&[0], &[1]).unwrap(),
    ]
}

fn refs(a: &[MxArray; 6]) -> [&MxArray; 6] {
    [&a[0], &a[1], &a[2], &a[3], &a[4], &a[5]]
}

/// q/k one per value head for kernels that map `hv % Hk`.
fn grouped_tape(tape: &GdnKernelTape, hk: i64, hv: i64) -> GdnKernelTape {
    let expand = |a: &MxArray| a.repeat((hv / hk) as i32, 2).unwrap();
    GdnKernelTape {
        q: expand(&tape.q),
        k: expand(&tape.k),
        v: tape.v.clone(),
        g: tape.g.clone(),
        beta: tape.beta.clone(),
    }
}

/// The fold on the 27B (Hk 16, Hv 48, W 10240) and 35B (Hk 16, Hv 32,
/// W 8192) geometries, both head orders, both storage schedules: for every
/// keep the deferred call's state and history equal the T=1 chain and
/// `GdnLayerTape::replay_into` from the same start, the tape rows past the
/// keep poisoned with NaN, and its output, tape and raw inputs equal a
/// keep-0 run from the settled state. Grouped heads have no ordinary
/// complete, so their producer and reference are the deferred entry with
/// the zero tape. Mutation F1 (the kernel folds all 8 rows) turns it red.
#[test]
fn deferred_fold_matches_t1_settle_on_model_shapes() {
    let _serial = SERIAL.lock().unwrap();
    let _env = Environment::new();
    if !crate::engine::persistence::compiled_forward_backend_available() {
        return;
    }
    for (scan, storage) in [("0", "0"), ("1", "1")] {
        Environment::set(scan, storage);
        for (hk, hv) in [(16i64, 48i64), (16, 32)] {
            let width = 2 * hk * 128 + hv * 128;
            let zero = zero_pending(hk, hv, width);
            let prepared = hv == 32;
            for tiled in [true, false] {
                let run = |w: &Window, dst, pending: Option<[&MxArray; 6]>| {
                    if tiled {
                        w.complete(prepared, dst, pending)
                    } else {
                        w.complete_heads(
                            prepared,
                            dst,
                            Some((pending.unwrap_or(refs(&zero)), false)),
                        )
                    }
                };
                let prior = Window::with_heads(8, hk, hv, 501 + hv as usize, true);
                let prior_out = run(&prior, None, None).expect("producer");
                let prior_raw = prior_out.raw.clone().unwrap_or_else(|| prior.a[0].clone());
                let chain_tape = if tiled {
                    prior_out.tape.clone()
                } else {
                    grouped_tape(&prior_out.tape, hk, hv)
                };
                for keep in 0..=8usize {
                    let tag = format!(
                        "hk={hk} hv={hv} tiled={tiled} keep={keep} scan={scan} storage={storage}"
                    );
                    let settled =
                        settle_t1(&chain_tape, &prior.a[7], &prior.a[4], &prior_raw, keep);
                    let mut replayed = crate::models::qwen3_5::arrays_cache::ArraysCache::new(2);
                    crate::models::qwen3_5::gated_delta_net::GdnLayerTape {
                        kernel: chain_tape.clone(),
                        qkv: prior_raw.clone(),
                        conv_kernel_dim: 4,
                    }
                    .replay_into(
                        &mut replayed,
                        Some(&prior.a[4].reshape(&[1, 3, width]).unwrap()),
                        Some(&prior.a[7]),
                        keep,
                    )
                    .unwrap();
                    exact(
                        replayed.get(1).unwrap(),
                        &settled.0,
                        &format!("{tag}: replay state"),
                    );
                    exact(
                        &replayed.get(0).unwrap().squeeze(Some(&[0])).unwrap(),
                        &settled.1,
                        &format!("{tag}: replay history"),
                    );
                    let keep_array = MxArray::from_int32(&[keep as i32], &[1]).unwrap();
                    let poisoned = [
                        poison(&prior_out.tape.k, keep),
                        poison(&prior_out.tape.v, keep),
                        poison(&prior_out.tape.g, keep),
                        poison(&prior_out.tape.beta, keep),
                        poison(&prior_raw, keep),
                        keep_array,
                    ];
                    let mut current = Window::with_heads(8, hk, hv, 611 + keep, keep % 2 == 0);
                    current.a[4] = prior.a[4].clone();
                    current.a[7] = prior.a[7].clone();
                    let state_dst = MxArray::zeros(
                        current.a[7].shape().unwrap().as_ref(),
                        Some(DType::Float32),
                    )
                    .unwrap();
                    let history_dst = MxArray::zeros(
                        current.a[4].shape().unwrap().as_ref(),
                        Some(DType::BFloat16),
                    )
                    .unwrap();
                    let actual = current
                        .complete_heads(
                            prepared,
                            Some((&state_dst, &history_dst)),
                            Some((refs(&poisoned), tiled)),
                        )
                        .unwrap_or_else(|| panic!("{tag}: deferred declined"));
                    current.a[4] = settled.1.clone();
                    current.a[7] = settled.0.clone();
                    let expected = run(&current, None, None).unwrap();
                    compare(&actual, &expected, (&settled.0, &settled.1));
                    exact(
                        actual.raw.as_ref().expect("deferred raw inputs"),
                        &current.a[0],
                        &format!("{tag}: raw inputs"),
                    );
                }
            }
        }
    }
}

/// Grouped heads on the 27B geometry: a keep-0 deferred call with the zero
/// tape gives the `qwen4_gdn_prepare` -> fused tail (`tiled_gqa = false`)
/// output and tape bits, with the incoming state and history untouched;
/// folding its own tape at keep = T then gives the chain's window-final
/// state and history. Mutation G1 (the deferred entry maps tiled heads)
/// turns it red.
#[test]
fn grouped_deferred_matches_the_prepare_chain() {
    let _serial = SERIAL.lock().unwrap();
    let _env = Environment::new();
    if !crate::engine::persistence::compiled_forward_backend_available() {
        return;
    }
    let (hk, hv, width) = (16i64, 48i64, 10240i64);
    let zero = zero_pending(hk, hv, width);
    for rows in [1i64, 8] {
        let tag = format!("rows={rows}");
        let w = Window::with_heads(rows, hk, hv, 701 + rows as usize, true);
        let a = &w.a;
        let mut outputs = [std::ptr::null_mut(); 6];
        assert!(unsafe {
            sys::mlx_qwen4_gdn_prepare(
                a[0].as_raw_ptr(),
                a[1].as_raw_ptr(),
                a[2].as_raw_ptr(),
                a[3].as_raw_ptr(),
                a[4].as_raw_ptr(),
                a[5].as_raw_ptr(),
                a[6].as_raw_ptr(),
                true,
                true,
                outputs.as_mut_ptr(),
            )
        });
        let [q, k, v, decay, beta, history] =
            outputs.map(|o| MxArray::from_handle(o, "gdn_prepare").unwrap());
        let mut sink = None;
        let Some((out, state)) = gated_delta_update_fused(
            &q,
            &k,
            &v,
            &a[1],
            &a[2],
            &a[5],
            &a[6],
            Some(&a[7]),
            None,
            true,
            false,
            Some((&decay, &beta)),
            GdnTail {
                z: &a[8],
                norm_weight: &a[9],
                eps: 1e-6,
                state_dst: None,
            },
            Some(&mut sink),
        )
        .unwrap() else {
            assert!(
                !super::tests::fused_step_launches(),
                "the grouped geometry takes the fused step"
            );
            eprintln!("SKIP grouped deferred: this device cannot launch the fused step kernel");
            return;
        };
        let first = w
            .complete_heads(false, None, Some((refs(&zero), false)))
            .expect("grouped deferred");
        exact(&first.out, &out, &format!("{tag}: output"));
        exact(&first.tape.k, &k, &format!("{tag}: compact k"));
        exact(&first.tape.v, &v, &format!("{tag}: v"));
        exact(&first.tape.g, &decay, &format!("{tag}: decay"));
        exact(&first.tape.beta, &beta, &format!("{tag}: beta"));
        exact(
            &first.state,
            &a[7],
            &format!("{tag}: keep 0 leaves the state"),
        );
        exact(
            &first.history,
            &a[4],
            &format!("{tag}: keep 0 leaves the history"),
        );
        let keep = MxArray::from_int32(&[rows as i32], &[1]).unwrap();
        let raw = first.raw.clone().expect("raw inputs");
        let mut next = Window::with_heads(8, hk, hv, 777, false);
        next.a[4] = a[4].clone();
        next.a[7] = a[7].clone();
        let folded = next
            .complete_heads(
                false,
                None,
                Some((
                    [
                        &first.tape.k,
                        &first.tape.v,
                        &first.tape.g,
                        &first.tape.beta,
                        &raw,
                        &keep,
                    ],
                    false,
                )),
            )
            .expect("grouped fold");
        exact(&folded.state, &state, &format!("{tag}: window-final state"));
        exact(
            &folded.history,
            &history,
            &format!("{tag}: window-final history"),
        );
    }
}
