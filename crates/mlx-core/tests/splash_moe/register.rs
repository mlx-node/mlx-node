use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
const FLAG: &str = "MLX_SPLASH_MOE_REGISTER";
unsafe extern "C" {
    fn mlx_test_splash_moe_sg8_gather(
        input: *mut mlx_sys::mlx_array,
        routes: *mut mlx_sys::mlx_array,
        count: *mut mlx_sys::mlx_array,
    ) -> *mut mlx_sys::mlx_array;
    fn mlx_test_splash_moe_sg8_pack(
        input: *mut mlx_sys::mlx_array,
        bits: i32,
    ) -> *mut mlx_sys::mlx_array;
}
struct Env(Option<std::ffi::OsString>);
impl Env {
    fn on() -> Self {
        let v = std::env::var_os(FLAG);
        unsafe { std::env::set_var(FLAG, "1") };
        Self(v)
    }
}
impl Drop for Env {
    fn drop(&mut self) {
        unsafe {
            match &self.0 {
                Some(v) => std::env::set_var(FLAG, v),
                None => std::env::remove_var(FLAG),
            }
        }
    }
}
fn flag(on: bool) {
    unsafe { std::env::set_var(FLAG, if on { "1" } else { "0" }) }
}
fn words(handle: *mut mlx_sys::mlx_array) -> Vec<u32> {
    assert!(!handle.is_null());
    let mut h = handle;
    assert!(unsafe { mlx_sys::mlx_eval(&mut h, 1) });
    let mut v = vec![0u32; unsafe { mlx_sys::mlx_array_size(h) }];
    assert!(unsafe { mlx_sys::mlx_array_to_uint32(h, v.as_mut_ptr(), v.len()) });
    unsafe { mlx_sys::mlx_array_delete(h) };
    v
}
fn pair(a: f32, b: f32) -> u32 {
    u32::from(half::bf16::from_f32(a).to_bits())
        | (u32::from(half::bf16::from_f32(b).to_bits()) << 16)
}
fn tree(mut v: [f32; 8]) -> f32 {
    for xor in [1, 2, 4] {
        let old = v;
        for i in 0..8 {
            v[i] = old[i] + old[i ^ xor]
        }
    }
    v[0]
}
#[test]
#[ignore = "exclusive GPU, grouped-register raw packing oracle"]
fn register_raw_q4_q6_pack_and_sum_bits_match_native_contract() {
    let _guard = SERIAL.lock().unwrap();
    let _env = Env::on();
    for k in [256usize, 512, 2048] {
        for live in [0usize, 1, 2, 7, 8] {
            for bits in [0i32, 4, 6] {
                let mut input = vec![0f32; 8 * k];
                for r in 0..live {
                    for c in 0..k {
                        input[r * k + c] = bf(if c % 7 == 0 {
                            16.0
                        } else if c % 7 == 1 {
                            -16.0
                        } else {
                            ((r * 37 + c * 11) % 97) as f32 * 0.013 - 0.4
                        });
                    }
                }
                let x = MxArray::from_float32(&input, &[8, k as i64])
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap();
                let actual = if bits == 0 {
                    let routes: Vec<u32> = (0..8)
                        .map(|r| {
                            if r < live {
                                (live - 1 - r) as u32
                            } else {
                                u32::MAX
                            }
                        })
                        .collect();
                    let route_array = MxArray::from_uint32(&routes, &[8]).unwrap();
                    let count = MxArray::from_uint32(&[1], &[1]).unwrap();
                    let original = input.clone();
                    for r in 0..live {
                        input[r * k..(r + 1) * k]
                            .copy_from_slice(&original[(live - 1 - r) * k..(live - r) * k]);
                    }
                    words(unsafe {
                        mlx_test_splash_moe_sg8_gather(
                            x.as_raw_ptr(),
                            route_array.as_raw_ptr(),
                            count.as_raw_ptr(),
                        )
                    })
                } else {
                    words(unsafe { mlx_test_splash_moe_sg8_pack(x.as_raw_ptr(), bits) })
                };
                let bits = if bits == 0 { 4 } else { bits };
                let mut expected = vec![0u32; actual.len()];
                for u in 0..k / 32 {
                    for lane in 0..32 {
                        let fm = ((lane >> 2) & 4) | ((lane >> 1) & 3);
                        let row = (((lane >> 2) & 2) << 1) | ((lane & 1) << 1);
                        for j in 0..4 {
                            let c = u * 32
                                + if bits == 4 {
                                    4 * fm + j
                                } else {
                                    2 * fm + j % 2 + if j >= 2 { 16 } else { 0 }
                                };
                            expected[(u * 32 + lane) * 4 + j] =
                                pair(input[row * k + c], input[(row + 1) * k + c]);
                        }
                    }
                    for half in 0..if bits == 6 { 2 } else { 1 } {
                        for row in 0..8 {
                            let mut partial = [0.0; 8];
                            for fm in 0..8 {
                                let start = row * k
                                    + u * 32
                                    + if bits == 4 {
                                        4 * fm
                                    } else {
                                        half * 16 + 2 * fm
                                    };
                                partial[fm] = if bits == 4 {
                                    (input[start] + input[start + 1])
                                        + (input[start + 2] + input[start + 3])
                                } else {
                                    input[start] + input[start + 1]
                                };
                            }
                            expected
                                [k * 4 + (u * if bits == 6 { 2 } else { 1 } + half) * 8 + row] =
                                tree(partial).to_bits();
                        }
                    }
                }
                assert_eq!(actual, expected, "K{k} live{live} Q{bits}");
            }
        }
    }
}
struct Fixture {
    packed: Vec<Weights>,
    decoded: Vec<Vec<f32>>,
    modes: Vec<u32>,
}
impl Fixture {
    fn new(routed_down: &str, shared_down: &str, tile: bool) -> Self {
        Self::from_formats(["q4k", "q4k", routed_down, "q4k", "q4k", shared_down], tile)
    }
    fn from_formats(names: [&str; 6], tile: bool) -> Self {
        let formats: Vec<_> = names.into_iter().map(kquant).collect();
        let mut packed = vec![];
        let mut decoded = vec![];
        for (j, q) in formats.iter().enumerate() {
            let (n, k) = if j % 3 == 2 { (H, I) } else { (I, H) };
            let shape = if j < 3 {
                vec![E as i64, n as i64]
            } else {
                vec![n as i64]
            };
            let mut w = weights(q, &shape, k as i64, 0xDA00 + j as u32);
            w.biases = w
                .biases
                .mul_scalar(0.001)
                .unwrap()
                .astype(DType::Float16)
                .unwrap();
            decoded.push(
                read_bits(
                    "register F64 decode",
                    dequantize(&w, DType::Float32, q, CPU),
                )
                .2
                .into_iter()
                .map(f32::from_bits)
                .collect(),
            );
            if tile {
                w = Weights {
                    w: tiled(&w.w, q.bits as i64),
                    scales: tiled(&w.scales, 16),
                    biases: tiled(&w.biases, if q.bits == 6 { 1 } else { 2 }),
                };
            }
            packed.push(w);
        }
        let modes = formats
            .iter()
            .map(|q| q.bits as u32 | if tile { 256 } else { 0 })
            .collect();
        Self {
            packed,
            decoded,
            modes,
        }
    }
    fn inputs(&self, m: i64, seed: u32) -> Vec<MxArray> {
        let x = activation(&[m, H as i64], seed, DType::BFloat16);
        let mut log = vec![-8.0; m as usize * E];
        for r in 0..m as usize {
            let first = if r % 8 == 7 { 1 } else { 0 };
            let second = 2 + r % (E - 2);
            log[r * E + first] = 4.0;
            log[r * E + second] = 3.5;
        }
        let logits = MxArray::from_float32(&log, &[m, E as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let shared = activation(&[m, 1], seed + 1, DType::BFloat16);
        let mut a = vec![x, logits, shared];
        for w in &self.packed {
            a.extend([w.w.clone(), w.scales.clone(), w.biases.clone()])
        }
        a
    }
    fn call(&self, a: &[MxArray]) -> Vec<f32> {
        let handles: Vec<_> = a.iter().map(MxArray::as_raw_ptr).collect();
        read_output("register grouped", unsafe {
            mlx_sys::mlx_splash_moe(
                handles.as_ptr(),
                handles.len(),
                self.modes.as_ptr(),
                TOP as i32,
                true,
            )
        })
        .2
        .into_iter()
        .map(|b| f32::from_bits(b << 16))
        .collect()
    }
    fn oracle(&self, a: &[MxArray]) -> Vec<f32> {
        let x = a[0].astype(DType::Float32).unwrap().to_float32().unwrap();
        let (coeff, ids) = topk_from_logits(
            &a[1],
            E as i32,
            TOP as i32,
            RoutingMode::Qwen35 {
                renormalize_topk: true,
            },
        )
        .unwrap();
        let coeff = coeff.astype(DType::Float32).unwrap().to_float32().unwrap();
        let ids = ids.to_uint32().unwrap();
        let shared = Activations::sigmoid(&a[2])
            .unwrap()
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap();
        let mut out = vec![0.0; x.len()];
        for (r, input) in x.chunks_exact(H).enumerate() {
            for slot in 0..=TOP {
                let (e, j, c) = if slot == TOP {
                    (0, 3, shared[r])
                } else {
                    (ids[r * TOP + slot] as usize, 0, coeff[r * TOP + slot])
                };
                let g = project(input, &self.decoded[j][e * I * H..], I, 0);
                let u = project(input, &self.decoded[j + 1][e * I * H..], I, 0);
                let act: Vec<_> = g
                    .iter()
                    .zip(u)
                    .map(|(&g, u)| bf(bf(bf_sigmoid(g) * g) * u))
                    .collect();
                let d = project(&act, &self.decoded[j + 2][e * H * I..], H, 0);
                for cidx in 0..H {
                    out[r * H + cidx] = bf(out[r * H + cidx] + bf(c * d[cidx]));
                }
            }
        }
        out
    }
}
#[test]
#[ignore = "exclusive GPU, register experts independent F64 and mixed Q4/Q6 layouts"]
fn register_mixed_down_formats_budget_and_f64() {
    let _guard = SERIAL.lock().unwrap();
    let _env = Env::on();
    for (down, shared) in [
        ("q4k", "q4k"),
        ("q6k", "q6k"),
        ("q4k", "q6k"),
        ("q6k", "q4k"),
    ] {
        for tile in [false, true] {
            let f = Fixture::new(down, shared, tile);
            let mut a = f.inputs(8, 991);
            let reference = f.oracle(&a);
            let full = f.call(&a);
            let peak = reference.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            for budget in [1i32, 3, 8] {
                a.truncate(21);
                a.push(MxArray::from_int32(&[budget], &[]).unwrap());
                start_counting();
                let result = f.call(&a);
                assert_eq!(family_count("splash_moe_register_gate_up"), 1);
                assert_eq!(family_count("splash_moe_register_down"), 1);
                stop_counting();
                let mut max = 0.0f32;
                let mut square = 0.0f64;
                let mut norm = 0.0f64;
                for (i, (&got, &want)) in result.iter().zip(&reference).enumerate() {
                    assert!(got.is_finite());
                    if i >= budget as usize * H {
                        assert_eq!(got.to_bits(), 0);
                        continue;
                    }
                    assert_eq!(
                        got.to_bits(),
                        full[i].to_bits(),
                        "live prefix independent of occupancy"
                    );
                    max = max.max((got - want).abs());
                    square += (f64::from(got) - f64::from(want)).powi(2);
                    norm += f64::from(want).powi(2);
                }
                eprintln!(
                    "register down={down}/{shared} tiled={tile} budget={budget} max/peak={} relative_rms={}",
                    max / peak,
                    (square / norm.max(1e-30)).sqrt()
                );
                assert!(max <= peak * 0.01 + 1e-6, "independent F64 limit unchanged");
            }
        }
    }
}
struct Builder {
    modes: Vec<u32>,
    traces: AtomicUsize,
}
unsafe extern "C" fn build(
    context: *mut std::ffi::c_void,
    inputs: *const *const mlx_sys::mlx_array,
    n: usize,
    outputs: *mut *mut mlx_sys::mlx_array,
    m: usize,
) -> bool {
    if context.is_null() || n != 22 || m != 1 {
        return false;
    }
    let b = unsafe { &*context.cast::<Builder>() };
    b.traces.fetch_add(1, Ordering::Relaxed);
    let a: Vec<_> = unsafe { std::slice::from_raw_parts(inputs, n) }
        .iter()
        .map(|h| h.cast_mut())
        .collect();
    let out = unsafe { mlx_sys::mlx_splash_moe(a.as_ptr(), n, b.modes.as_ptr(), TOP as i32, true) };
    for h in a {
        unsafe { mlx_sys::mlx_array_delete(h) }
    }
    unsafe { *outputs = out };
    !out.is_null()
}
#[test]
#[ignore = "exclusive GPU, one-trace M1/M8/M33 fallback and masked register replay"]
fn register_shapeless_width_budget_replay_and_fallback() {
    let _guard = SERIAL.lock().unwrap();
    let _env = Env::on();
    let f = Fixture::new("q6k", "q4k", true);
    let mut builder = Builder {
        modes: f.modes.clone(),
        traces: AtomicUsize::new(0),
    };
    let key = 0x6d6f655347380001;
    unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
    for (i, (m, budget)) in [(1, 1), (8, 1), (8, 3), (33, 33), (8, 8), (1, 1)]
        .into_iter()
        .enumerate()
    {
        let mut a = f.inputs(m, 301 + i as u32);
        a.push(MxArray::from_int32(&[budget], &[]).unwrap());
        flag(false);
        let off = if m != 8 { Some(f.call(&a)) } else { None };
        flag(true);
        let expected = f.call(&a);
        if let Some(off) = off {
            assert_eq!(expected, off, "M{m} fallback")
        }
        let handles: Vec<_> = a.iter().map(|a| a.as_raw_ptr().cast_const()).collect();
        let mut out = std::ptr::null_mut();
        assert!(unsafe {
            mlx_sys::mlx_compiled_graph_invoke(
                key,
                Some(build),
                (&mut builder as *mut Builder).cast(),
                handles.as_ptr(),
                handles.len(),
                &mut out,
                1,
                true,
            )
        });
        start_counting();
        let result = read_output("register compiled", out)
            .2
            .into_iter()
            .map(|b| f32::from_bits(b << 16))
            .collect::<Vec<_>>();
        assert_eq!(
            family_count("splash_moe_register_gate_up"),
            u64::from(m == 8)
        );
        stop_counting();
        assert_eq!(result, expected);
        assert_eq!(builder.traces.load(Ordering::Relaxed), 1);
    }
    unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, key) };
}

#[test]
#[ignore = "exclusive GPU, all six unsupported projections preserve original grouped math"]
fn register_unsupported_projection_formats_fall_back_exactly() {
    let _guard = SERIAL.lock().unwrap();
    let _env = Env::on();
    for tile in [false, true] {
        for changed in 0..6 {
            let mut formats = ["q4k", "q4k", "q6k", "q4k", "q4k", "q4k"];
            formats[changed] = "q5k";
            let f = Fixture::from_formats(formats, tile);
            let inputs = f.inputs(8, 701 + changed as u32);
            flag(false);
            let expected = f.call(&inputs);
            flag(true);
            start_counting();
            let actual = f.call(&inputs);
            assert_eq!(family_count("splash_moe_register_gate_up"), 0);
            assert_eq!(family_count("splash_moe_register_down"), 0);
            stop_counting();
            assert_eq!(actual, expected, "projection {changed} tile={tile}");
        }
    }
}

fn typed_output(f: &Fixture, arrays: &[MxArray]) -> (Vec<i64>, DType, Vec<u32>) {
    let handles: Vec<_> = arrays.iter().map(MxArray::as_raw_ptr).collect();
    read_output("register typed output", unsafe {
        mlx_sys::mlx_splash_moe(
            handles.as_ptr(),
            handles.len(),
            f.modes.as_ptr(),
            TOP as i32,
            true,
        )
    })
}

fn output_values(dtype: DType, words: &[u32]) -> Vec<f32> {
    assert!(matches!(dtype, DType::BFloat16 | DType::Float32));
    words
        .iter()
        .map(|&word| {
            if dtype == DType::BFloat16 {
                half::bf16::from_bits(word as u16).to_f32()
            } else {
                f32::from_bits(word)
            }
        })
        .collect()
}

fn offset_contiguous_plane(input: &MxArray) -> MxArray {
    let shape = input.shape().unwrap();
    let dtype = input.dtype().unwrap();
    let prefix = MxArray::zeros(&[1], Some(dtype)).unwrap();
    let storage = MxArray::concatenate(&prefix, &input.reshape(&[-1]).unwrap(), 0).unwrap();
    storage.eval();
    let view = storage
        .slice(&[1], &[input.size().unwrap() as i64 + 1])
        .unwrap()
        .reshape(&shape)
        .unwrap();
    // Lazy Slice does not publish its physical buffer offset until evaluation.
    view.eval();
    let offset = unsafe { mlx_sys::mlx_array_get_buffer_offset(view.as_raw_ptr()) };
    assert_ne!(offset % 16, 0, "the test must retain a misaligned view");
    assert_eq!(view.shape().unwrap().as_ref(), shape.as_ref());
    assert_eq!(view.dtype().unwrap(), dtype);
    view
}

#[test]
#[ignore = "exclusive GPU, eligible register lane all-plane misalignment"]
fn register_misaligned_all_planes_match_aligned() {
    let _guard = SERIAL.lock().unwrap();
    let _env = Env::on();
    assert!(
        gpu_gen() >= 17,
        "test requires the register expert device tier"
    );
    for tile in [false, true] {
        for (down, shared) in [("q4k", "q6k"), ("q6k", "q4k")] {
            let f = Fixture::new(down, shared, tile);
            for dtype in [DType::BFloat16, DType::Float32] {
                let mut aligned = f.inputs(8, 1907);
                aligned[1] = aligned[1].astype(dtype).unwrap();
                aligned[2] = aligned[2].astype(dtype).unwrap();
                // Include the optional scalar among the independently offset
                // planes; three live rows also exercise padded prepared rows.
                aligned.push(MxArray::from_int32(&[3], &[]).unwrap());
                let expected = typed_output(&f, &aligned);
                assert_eq!(expected.0, [8, H as i64]);
                assert_eq!(expected.1, dtype);
                assert!(expected.2[3 * H..].iter().all(|&v| v == 0));
                let views: Vec<_> = aligned.iter().map(offset_contiguous_plane).collect();
                // Every activation, routing, packed-weight, subscale, metadata
                // and budget plane separately, then all 22 planes together.
                for index in 0..=aligned.len() {
                    let mut inputs = aligned.clone();
                    if index == aligned.len() {
                        inputs.clone_from(&views);
                    } else {
                        inputs[index] = views[index].clone();
                    }
                    start_counting();
                    let actual = typed_output(&f, &inputs);
                    assert_eq!(family_count("splash_moe_register_gate_up"), 1);
                    assert_eq!(family_count("splash_moe_register_down"), 1);
                    stop_counting();
                    assert_eq!(
                        actual, expected,
                        "plane={index} tile={tile} down={down}/{shared} router={dtype:?}"
                    );
                }
            }
        }
    }
}

// Independent decoded-weight F64 projections, with the native router and
// shared dtype boundaries kept separately. The final output is F32 unless
// both coefficient paths are BF16; this catches accidental F32 truncation.
fn mixed_router_oracle(f: &Fixture, arrays: &[MxArray]) -> Vec<f32> {
    let router_bf16 = arrays[1].dtype().unwrap() == DType::BFloat16;
    let shared_bf16 = arrays[2].dtype().unwrap() == DType::BFloat16;
    let x = arrays[0]
        .astype(DType::Float32)
        .unwrap()
        .to_float32()
        .unwrap();
    let (coefficients, ids) = topk_from_logits(
        &arrays[1],
        E as i32,
        TOP as i32,
        RoutingMode::Qwen35 {
            renormalize_topk: true,
        },
    )
    .unwrap();
    let coefficients = coefficients
        .astype(DType::Float32)
        .unwrap()
        .to_float32()
        .unwrap();
    let ids = ids.to_uint32().unwrap();
    let shared = Activations::sigmoid(&arrays[2])
        .unwrap()
        .astype(DType::Float32)
        .unwrap()
        .to_float32()
        .unwrap();
    let mut result = vec![0.0f32; x.len()];
    for (r, input) in x.chunks_exact(H).enumerate() {
        for slot in 0..=TOP {
            let (expert, j, coefficient, round_product) = if slot == TOP {
                (0, 3, shared[r], shared_bf16)
            } else {
                (
                    ids[r * TOP + slot] as usize,
                    0,
                    coefficients[r * TOP + slot],
                    router_bf16,
                )
            };
            let gate = project(input, &f.decoded[j][expert * I * H..], I, 0);
            let up = project(input, &f.decoded[j + 1][expert * I * H..], I, 0);
            let hidden: Vec<_> = gate
                .iter()
                .zip(up)
                .map(|(&g, u)| bf(bf(bf_sigmoid(g) * g) * u))
                .collect();
            let down = project(&hidden, &f.decoded[j + 2][expert * H * I..], H, 0);
            for (column, value) in down.into_iter().enumerate() {
                let product = coefficient * value;
                let product = if round_product { bf(product) } else { product };
                let output = &mut result[r * H + column];
                *output += product;
                if (slot < TOP && router_bf16) || (slot == TOP && router_bf16 && shared_bf16) {
                    *output = bf(*output);
                }
            }
        }
    }
    result
}

#[test]
#[ignore = "exclusive GPU, eligible register F32 router/shared coefficients and F64 oracle"]
fn register_f32_router_and_shared_preserve_dtype_and_f64() {
    let _guard = SERIAL.lock().unwrap();
    let _env = Env::on();
    assert!(
        gpu_gen() >= 17,
        "test requires the register expert device tier"
    );
    for tile in [false, true] {
        for (down, shared_down) in [("q4k", "q6k"), ("q6k", "q4k")] {
            let f = Fixture::new(down, shared_down, tile);
            for (router_dtype, shared_dtype) in [
                (DType::Float32, DType::Float32),
                (DType::Float32, DType::BFloat16),
                (DType::BFloat16, DType::Float32),
            ] {
                let mut arrays = f.inputs(8, 2017);
                // These offsets are intentionally not BF16-representable.
                arrays[1] = arrays[1]
                    .astype(DType::Float32)
                    .unwrap()
                    .add_scalar(0.00317)
                    .unwrap()
                    .astype(router_dtype)
                    .unwrap();
                arrays[2] = arrays[2]
                    .astype(DType::Float32)
                    .unwrap()
                    .add_scalar(0.00131)
                    .unwrap()
                    .astype(shared_dtype)
                    .unwrap();
                let oracle = mixed_router_oracle(&f, &arrays);
                let full = typed_output(&f, &arrays);
                assert_eq!(full.0, [8, H as i64]);
                assert_eq!(full.1, DType::Float32);
                assert!(
                    full.2.iter().any(|bits| bits & 0xffff != 0),
                    "fixture must exercise F32 output precision"
                );
                for budget in [1i32, 3, 8] {
                    arrays.truncate(21);
                    arrays.push(MxArray::from_int32(&[budget], &[]).unwrap());
                    start_counting();
                    let actual = typed_output(&f, &arrays);
                    assert_eq!(family_count("splash_moe_register_gate_up"), 1);
                    assert_eq!(family_count("splash_moe_register_down"), 1);
                    stop_counting();
                    assert_eq!(actual.0, full.0);
                    assert_eq!(actual.1, DType::Float32);
                    let live = budget as usize * H;
                    assert_eq!(
                        actual.2[..live],
                        full.2[..live],
                        "budget changed the live prefix"
                    );
                    assert!(actual.2[live..].iter().all(|&v| v == 0));
                    let values = output_values(actual.1, &actual.2);
                    assert!(values.iter().all(|v| v.is_finite()));
                    let peak = oracle[..live]
                        .iter()
                        .map(|v| v.abs())
                        .fold(0.0f32, f32::max);
                    let error = values[..live]
                        .iter()
                        .zip(&oracle[..live])
                        .map(|(actual, expected)| (actual - expected).abs())
                        .fold(0.0f32, f32::max);
                    eprintln!(
                        "register F32 tile={tile} down={down}/{shared_down} router={router_dtype:?} shared={shared_dtype:?} budget={budget} max/peak={}",
                        error / peak.max(1e-30)
                    );
                    assert!(
                        error <= peak * 0.01 + 1e-6,
                        "independent F64 limit unchanged"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "exclusive GPU, indirect default/optout exact output and dispatch"]
fn indirect_default_and_optout_preserve_register_and_staged_outputs() {
    let _guard = SERIAL.lock().unwrap();
    let _register = Env::on();
    assert!(gpu_gen() >= 17);
    struct Restore(Option<std::ffi::OsString>);
    impl Drop for Restore {
        fn drop(&mut self) {
            unsafe {
                match &self.0 {
                    Some(v) => std::env::set_var("MLX_SPLASH_MOE_INDIRECT", v),
                    None => std::env::remove_var("MLX_SPLASH_MOE_INDIRECT"),
                }
            }
        }
    }
    let _indirect = Restore(std::env::var_os("MLX_SPLASH_MOE_INDIRECT"));
    let fixture = Fixture::new("q6k", "q4k", true);
    for register in [false, true] {
        flag(register);
        for budget in [1, 3, 8] {
            let mut arrays = fixture.inputs(8, 2317);
            arrays.push(MxArray::from_int32(&[budget], &[]).unwrap());
            let mut reference = None;
            for value in [None, Some("1"), Some("0"), Some("invalid")] {
                unsafe {
                    match value {
                        Some(v) => std::env::set_var("MLX_SPLASH_MOE_INDIRECT", v),
                        None => std::env::remove_var("MLX_SPLASH_MOE_INDIRECT"),
                    }
                }
                start_counting();
                let output = fixture.call(&arrays);
                assert_eq!(
                    family_count("splash_moe_indirect"),
                    u64::from(value.is_none_or(|v| v == "1"))
                );
                assert_eq!(
                    family_count("splash_moe_register_gate_up"),
                    u64::from(register)
                );
                stop_counting();
                if let Some(expected) = &reference {
                    assert_eq!(
                        &output, expected,
                        "register={register} budget={budget} flag={value:?}"
                    );
                } else {
                    reference = Some(output);
                }
            }
        }
    }
}

include!("hybrid.rs");
