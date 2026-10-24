//! End-to-end grouped GGUF MoE against a CPU F64 accumulation oracle.
//! Covers all six independently mixed quant formats, both layouts, sparse
//! column tiles, 8/16/32 matrix tiers, partial tiles, FP32 route weights and
//! both norm_topk_prob settings. No model checkpoint is required.
mod kquant_support;
use kquant_support::*;
use mlx_core::array::{DType, MxArray};
use mlx_core::models::quant_dispatch::kquant_tile_rows;
use mlx_core::models::quantized_linear::{MLPVariant, QuantizedLinear, QuantizedSwitchLinear};
use mlx_core::models::qwen3_5_moe::switch_glu::SwitchGLU;
use mlx_core::moe::{RoutingMode, topk_from_logits};
use mlx_core::nn::Activations;

// Non-square and multiple native K blocks in both projection directions.
const H: usize = 1024;
const I: usize = 512;
const E: usize = 8;
const TOP: usize = 2;
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn bf(v: f32) -> f32 {
    half::bf16::from_f32(v).to_f32()
}

fn bf_sigmoid(x: f32) -> f32 {
    let y = bf(1.0 / bf(1.0 + bf(x.abs().exp())));
    if x < 0.0 { y } else { bf(1.0 - y) }
}

fn project(x: &[f32], w: &[f32], n: usize, precision: u8) -> Vec<f32> {
    w.chunks_exact(x.len())
        .take(n)
        .map(|row| {
            bf(row
                .iter()
                .zip(x)
                .map(|(a, b)| {
                    let w = match precision {
                        1 => half::f16::from_f32(*a).to_f32(),
                        2 => bf(*a),
                        _ => *a,
                    };
                    f64::from(w) * f64::from(*b)
                })
                .sum::<f64>() as f32)
        })
        .collect()
}

fn expected(
    x: &[f32],
    routing_weights: &[f32],
    routing_ids: &[u32],
    shared_weights: &[f32],
    weights: &[Vec<f32>],
    router_bf16: bool,
    tiled: bool,
) -> Vec<f32> {
    let mut result = Vec::with_capacity(x.len());
    let rows = x.len() / H;
    let precision: Vec<u8> = (0..6)
        .map(|j| {
            if j < 3 {
                if rows * TOP >= 64 && rows * TOP / E >= 4 {
                    2
                } else {
                    0
                }
            } else {
                let (n, k) = if j % 3 == 2 { (H, I) } else { (I, H) };
                if rows as i64 >= qmv_vector_limit(k as i64, n as i64) {
                    2
                } else if tiled && rows == 8 {
                    1
                } else {
                    0
                }
            }
        })
        .collect();
    for (r, input) in x.as_chunks::<H>().0.iter().enumerate() {
        let mut output = vec![0.0f32; H];
        for slot in 0..=TOP {
            let (expert, offset, coefficient) = if slot == TOP {
                (0, 3, shared_weights[r])
            } else {
                (
                    routing_ids[r * TOP + slot] as usize,
                    0,
                    routing_weights[r * TOP + slot],
                )
            };
            let coefficient = if router_bf16 {
                bf(coefficient)
            } else {
                coefficient
            };
            let gate = project(
                input,
                &weights[offset][expert * I * H..],
                I,
                precision[offset],
            );
            let up = project(
                input,
                &weights[offset + 1][expert * I * H..],
                I,
                precision[offset + 1],
            );
            let hidden: Vec<f32> = gate
                .iter()
                .zip(up)
                .map(|(g, u)| bf(bf(bf_sigmoid(*g) * g) * u))
                .collect();
            let down = project(
                &hidden,
                &weights[offset + 2][expert * H * I..],
                H,
                precision[offset + 2],
            );
            for (o, d) in output.iter_mut().zip(down) {
                let product = coefficient * d;
                let product = if router_bf16 { bf(product) } else { product };
                *o += product;
            }
            // Routed reduction and final shared addition are separate ops.
            if router_bf16 && (slot == TOP - 1 || slot == TOP) {
                output.iter_mut().for_each(|v| *v = bf(*v));
            }
        }
        result.extend(output);
    }
    result
}

fn native_reference(
    x: &MxArray,
    logits: &MxArray,
    shared_logits: &MxArray,
    packed: &[Weights],
    formats: &[&KQuant],
    normalize: bool,
    tiled: bool,
) -> MxArray {
    let qs = |j: usize| {
        QuantizedSwitchLinear::new(
            packed[j].w.clone(),
            packed[j].scales.clone(),
            Some(packed[j].biases.clone()),
            formats[j].group_size,
            formats[j].bits,
            format!("{}{}", formats[j].mode, if tiled { "@t64" } else { "" }),
        )
    };
    let ql = |j: usize| {
        QuantizedLinear::new(
            packed[j].w.clone(),
            packed[j].scales.clone(),
            Some(packed[j].biases.clone()),
            None,
            formats[j].group_size,
            formats[j].bits,
            format!("{}{}", formats[j].mode, if tiled { "@t64" } else { "" }),
        )
    };
    let switch = SwitchGLU::new_quantized(qs(0), qs(1), qs(2));
    let shared = MLPVariant::Quantized {
        gate_proj: ql(3),
        up_proj: ql(4),
        down_proj: ql(5),
        gate_up: None,
    };
    let (weights, ids) = topk_from_logits(
        logits,
        E as i32,
        TOP as i32,
        RoutingMode::Qwen35 {
            renormalize_topk: normalize,
        },
    )
    .unwrap();
    let rows = x.shape().unwrap()[0];
    let routed = switch
        .forward(x, &ids)
        .unwrap()
        .mul(&weights.reshape(&[rows, TOP as i64, 1]).unwrap())
        .unwrap()
        .sum(Some(&[1]), None)
        .unwrap();
    let shared = shared
        .forward(x)
        .unwrap()
        .mul(&Activations::sigmoid(shared_logits).unwrap())
        .unwrap();
    routed.add(&shared).unwrap()
}

fn tiled(a: &MxArray, unit: i64) -> MxArray {
    let shape = a.shape().unwrap();
    let width = *shape.last().unwrap();
    kquant_tile_rows(&a.reshape(&[-1, width]).unwrap(), unit)
        .unwrap()
        .reshape(&shape)
        .unwrap()
}

#[test]
fn grouped_mixed_kquants_match_cpu_oracle() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !unsafe { mlx_sys::mlx_splash_moe_available() } {
        eprintln!("grouped MoE unavailable; requires a tensor-op Metal GPU");
        return;
    }
    let formats: Vec<&KQuant> = ["q4k", "q5k", "q6k", "q6k", "q4k", "q5k"]
        .into_iter()
        .map(|mode| KQUANTS.iter().find(|k| k.mode == mode).unwrap())
        .collect();
    let mut packed = Vec::new();
    let mut decoded = Vec::new();
    for (j, k) in formats.iter().enumerate() {
        let (n, input) = if j % 3 == 2 { (H, I) } else { (I, H) };
        let leading = if j < 3 {
            vec![E as i64, n as i64]
        } else {
            vec![n as i64]
        };
        let mut w = weights(k, &leading, input as i64, 71 + j as u32);
        // Keep all projections well-conditioned through three BF16 boundaries.
        w.biases = w
            .biases
            .mul_scalar(0.001)
            .unwrap()
            .astype(DType::Float16)
            .unwrap();
        let (_, _, bits) = read_bits("CPU dequantization", dequantize(&w, DType::Float32, k, CPU));
        decoded.push(bits.into_iter().map(f32::from_bits).collect::<Vec<_>>());
        packed.push(w);
    }
    for tile in [false, true] {
        let tiled_weights: Vec<Weights> = packed
            .iter()
            .zip(&formats)
            .map(|(w, k)| {
                if tile {
                    Weights {
                        w: tiled(&w.w, i64::from(k.bits)),
                        scales: tiled(&w.scales, 16),
                        biases: tiled(&w.biases, k.biases_cols),
                    }
                } else {
                    Weights {
                        w: w.w.clone(),
                        scales: w.scales.clone(),
                        biases: w.biases.clone(),
                    }
                }
            })
            .collect();
        for rows in [1, 2, 3, 8, 9, 17, 33, 129] {
            let x = activation(&[rows, H as i64], 345, DType::BFloat16);
            let xv = x
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap()
                .to_vec();
            // Deliberately skew routes to expose multi-tile tail handling;
            // equal logits exercise ascending expert-id tie breaking.
            let lv: Vec<f32> = (0..rows as usize)
                .flat_map(|r| {
                    (0..E).map(move |e| {
                        if e < 2 {
                            2.0
                        } else {
                            -1.0 - e as f32 * 0.2 + (r % 3) as f32 * 0.01
                        }
                    })
                })
                .collect();
            let sv: Vec<f32> = (0..rows).map(|r| (r % 5) as f32 * 0.3 - 0.6).collect();
            for router_bf16 in [false, true] {
                let dtype = if router_bf16 {
                    DType::BFloat16
                } else {
                    DType::Float32
                };
                let logits = MxArray::from_float32(&lv, &[rows, E as i64])
                    .unwrap()
                    .astype(dtype)
                    .unwrap();
                let shared = MxArray::from_float32(&sv, &[rows, 1])
                    .unwrap()
                    .astype(dtype)
                    .unwrap();
                for normalize in [false, true] {
                    let mut arrays = vec![x.as_raw_ptr(), logits.as_raw_ptr(), shared.as_raw_ptr()];
                    let mut modes = Vec::new();
                    for (w, k) in tiled_weights.iter().zip(&formats) {
                        arrays.extend([
                            w.w.as_raw_ptr(),
                            w.scales.as_raw_ptr(),
                            w.biases.as_raw_ptr(),
                        ]);
                        modes.push(k.bits as u32 | if tile { 256 } else { 0 });
                    }
                    let call = || unsafe {
                        mlx_sys::mlx_splash_moe(
                            arrays.as_ptr(),
                            arrays.len(),
                            modes.as_ptr(),
                            TOP as i32,
                            normalize,
                        )
                    };
                    let (_, actual_dtype, bits) = read_output("grouped MoE", call());
                    assert_eq!(actual_dtype, dtype);
                    let (route_weights, route_ids) = topk_from_logits(
                        &logits,
                        E as i32,
                        TOP as i32,
                        RoutingMode::Qwen35 {
                            renormalize_topk: normalize,
                        },
                    )
                    .unwrap();
                    let route_weights = route_weights
                        .astype(DType::Float32)
                        .unwrap()
                        .to_float32()
                        .unwrap();
                    let route_ids = route_ids.to_uint32().unwrap();
                    let shared_weights = Activations::sigmoid(&shared)
                        .unwrap()
                        .astype(DType::Float32)
                        .unwrap()
                        .to_float32()
                        .unwrap();
                    let reference = expected(
                        &xv,
                        &route_weights,
                        &route_ids,
                        &shared_weights,
                        &decoded,
                        router_bf16,
                        tile,
                    );
                    let values: Vec<f32> = bits
                        .iter()
                        .map(|v| f32::from_bits(if router_bf16 { v << 16 } else { *v }))
                        .collect();
                    let native = native_reference(
                        &x,
                        &logits,
                        &shared,
                        &tiled_weights,
                        &formats,
                        normalize,
                        tile,
                    );
                    assert_eq!(native.dtype().unwrap(), dtype);
                    let native = native
                        .astype(DType::Float32)
                        .unwrap()
                        .to_float32()
                        .unwrap()
                        .to_vec();
                    let native_peak = native.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                    let native_worst = values
                        .iter()
                        .zip(&native)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    let peak = reference.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                    let worst = values
                        .iter()
                        .zip(&reference)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    eprintln!(
                        "rows={rows} tiled={tile} normalized={normalize} dtype={dtype:?} cpu_rel={} native_rel={}",
                        worst / peak,
                        native_worst / native_peak
                    );
                    assert!(values.iter().all(|v| v.is_finite()));
                    assert!(
                        native_worst <= 0.01 * native_peak + 1e-6,
                        "native rows={rows} tiled={tile} normalized={normalize} dtype={dtype:?}: worst={native_worst} peak={native_peak}"
                    );
                    assert!(
                        worst <= 0.025 * peak + 1e-6,
                        "rows={rows} tiled={tile} normalized={normalize}: worst={worst} peak={peak}"
                    );
                    let (_, _, repeat) = read_output("grouped MoE repeat", call());
                    assert_eq!(bits, repeat, "repeat differed at rows={rows} tiled={tile}");
                }
            }
        }
    }
}

#[test]
fn grouped_misaligned_contiguous_inputs_match_aligned() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !unsafe { mlx_sys::mlx_splash_moe_available() } {
        return;
    }
    let formats: Vec<&KQuant> = ["q4k", "q5k", "q6k", "q6k", "q4k", "q5k"]
        .into_iter()
        .map(|mode| KQUANTS.iter().find(|k| k.mode == mode).unwrap())
        .collect();
    let packed: Vec<Weights> = formats
        .iter()
        .enumerate()
        .map(|(j, k)| {
            let (n, input) = if j % 3 == 2 { (H, I) } else { (I, H) };
            let leading = if j < 3 {
                vec![E as i64, n as i64]
            } else {
                vec![n as i64]
            };
            let mut w = weights(k, &leading, input as i64, 194 + j as u32);
            w.biases = w
                .biases
                .mul_scalar(0.001)
                .unwrap()
                .astype(DType::Float16)
                .unwrap();
            w
        })
        .collect();
    let misaligned = |input: &MxArray| {
        let shape = input.shape().unwrap();
        let flat = input.reshape(&[-1]).unwrap();
        let prefix = MxArray::zeros(&[1], Some(input.dtype().unwrap())).unwrap();
        let storage = MxArray::concatenate(&prefix, &flat, 0).unwrap();
        storage.eval();
        let view = storage
            .slice(&[1], &[input.size().unwrap() as i64 + 1])
            .unwrap()
            .reshape(&shape)
            .unwrap();
        view.eval();
        let offset = unsafe { mlx_sys::mlx_array_get_buffer_offset(view.as_raw_ptr()) };
        assert_ne!(offset % 16, 0, "fixture must retain its misaligned offset");
        view
    };
    for tile in [false, true] {
        for router_dtype in [DType::BFloat16, DType::Float32] {
            let mut inputs = vec![
                activation(&[8, H as i64], 159, DType::BFloat16),
                activation(&[8, E as i64], 160, router_dtype),
                activation(&[8, 1], 161, router_dtype),
            ];
            let mut modes = Vec::new();
            for (w, k) in packed.iter().zip(&formats) {
                if tile {
                    inputs.extend([
                        tiled(&w.w, i64::from(k.bits)),
                        tiled(&w.scales, 16),
                        tiled(&w.biases, k.biases_cols),
                    ]);
                } else {
                    inputs.extend([w.w.clone(), w.scales.clone(), w.biases.clone()]);
                }
                modes.push(k.bits as u32 | if tile { 256 } else { 0 });
            }
            let call = |arrays: &[MxArray]| {
                let handles: Vec<_> = arrays.iter().map(MxArray::as_raw_ptr).collect();
                read_output("grouped aligned-view parity", unsafe {
                    mlx_sys::mlx_splash_moe(
                        handles.as_ptr(),
                        handles.len(),
                        modes.as_ptr(),
                        TOP as i32,
                        true,
                    )
                })
            };
            let expected = call(&inputs);
            let views: Vec<MxArray> = inputs.iter().map(misaligned).collect();
            let actual = call(&views);
            assert_eq!(actual, expected, "tiled={tile} router={router_dtype:?}");
        }
    }
}

#[test]
fn route_group_maximum_geometry_and_inverse() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !unsafe { mlx_sys::mlx_splash_moe_available() } {
        return;
    }
    const EXPERTS: usize = 256;
    const K: usize = 8;
    for rows in [1usize, 8, 33, 2048] {
        let scores: Vec<f32> = (0..rows)
            .flat_map(|r| (0..EXPERTS).map(move |e| (((e + r * 17) % EXPERTS) / 2) as f32 * 0.02))
            .collect();
        let shared: Vec<f32> = (0..rows).map(|r| (r % 17) as f32 * 0.1 - 0.8).collect();
        let logits = MxArray::from_float32(&scores, &[rows as i64, EXPERTS as i64]).unwrap();
        let shared_logits = MxArray::from_float32(&shared, &[rows as i64]).unwrap();
        let tile_rows = if rows * K <= EXPERTS { 8 } else { 32 };
        let capacity =
            (rows * K).min((rows * K).div_ceil(tile_rows) + EXPERTS - 1) + rows.div_ceil(tile_rows);
        let routes = rows * (K + 1);
        let tile_offset = 2 * routes;
        let count_offset = tile_offset + 2 * capacity;
        let grouped_offset = count_offset + 7;
        let inverse_offset = grouped_offset + capacity * tile_rows;
        for normalize in [false, true] {
            let mut raw = unsafe {
                mlx_sys::mlx_test_splash_moe_route_group(
                    logits.as_raw_ptr(),
                    shared_logits.as_raw_ptr(),
                    K as i32,
                    normalize,
                )
            };
            assert!(!raw.is_null());
            let mut error = [0i8; 1024];
            assert!(
                unsafe {
                    mlx_sys::mlx_eval_with_error(&mut raw, 1, error.as_mut_ptr(), error.len())
                },
                "route/group evaluation failed"
            );
            let mut words = vec![0u32; inverse_offset + routes];
            assert!(unsafe { mlx_sys::mlx_array_to_uint32(raw, words.as_mut_ptr(), words.len()) });
            unsafe { mlx_sys::mlx_array_delete(raw) };
            let live = words[count_offset] as usize;
            assert!(live <= capacity);
            assert_eq!(
                &words[count_offset + 1..count_offset + 7],
                &[16, live as u32, 1, 16, live as u32, 1]
            );
            let mut count_by_expert = vec![0usize; EXPERTS + 1];
            for t in 0..live {
                let expert = words[tile_offset + 2 * t] as usize;
                let used = words[tile_offset + 2 * t + 1] as usize;
                assert!(expert <= EXPERTS && used > 0 && used <= tile_rows);
                count_by_expert[expert] += used;
                if t > 0 {
                    assert!(words[tile_offset + 2 * (t - 1)] <= expert as u32);
                }
            }
            assert_eq!(count_by_expert.iter().sum::<usize>(), routes);
            assert_eq!(count_by_expert[EXPERTS], rows);
            let mut seen = vec![false; capacity * tile_rows];
            for r in 0..rows {
                let row_scores = &scores[r * EXPERTS..(r + 1) * EXPERTS];
                let mut order: Vec<usize> = (0..EXPERTS).collect();
                order.sort_by(|a, b| row_scores[*a].total_cmp(&row_scores[*b]).then(a.cmp(b)));
                let max = row_scores[order[EXPERTS - 1]];
                let denominator: f32 = order[if normalize { EXPERTS - K } else { 0 }..]
                    .iter()
                    .map(|e| (row_scores[*e] - max).exp())
                    .sum();
                for slot in 0..=K {
                    let route = r * (K + 1) + slot;
                    let expected = if slot == K {
                        EXPERTS
                    } else {
                        order[EXPERTS - K + slot]
                    };
                    assert_eq!(
                        words[route], expected as u32,
                        "selected id at row {r} slot {slot}"
                    );
                    let probability = if slot == K {
                        1.0 / (1.0 + (-shared[r]).exp())
                    } else {
                        (row_scores[expected] - max).exp() / denominator
                    };
                    assert!((f32::from_bits(words[routes + route]) - probability).abs() < 2e-6);
                    let grouped = words[inverse_offset + route] as usize;
                    assert!(grouped < live * tile_rows);
                    assert!(!seen[grouped]);
                    seen[grouped] = true;
                    assert_eq!(words[grouped_offset + grouped], route as u32);
                    let tile = grouped / tile_rows;
                    assert_eq!(words[tile_offset + 2 * tile], expected as u32);
                    assert!(grouped % tile_rows < words[tile_offset + 2 * tile + 1] as usize);
                }
            }
        }
    }
}

#[test]
fn routing_matches_native_softmax_partition_with_bf16_ties() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !unsafe { mlx_sys::mlx_splash_moe_available() } {
        return;
    }
    const EXPERTS: usize = 256;
    const K: usize = 8;
    for dtype in [DType::BFloat16, DType::Float32] {
        for rows in [1usize, 8, 33] {
            // Exact ties, BF16 probability-rounding ties, and permuted order.
            let scores: Vec<f32> = (0..rows)
                .flat_map(|r| {
                    (0..EXPERTS).map(move |e| match r % 4 {
                        0 => 0.0,
                        1 => (((e + r * 17) % EXPERTS) / 2) as f32 * 0.0001,
                        2 => ((e + r * 17) % EXPERTS) as f32 * 0.02 - 1.0,
                        _ => {
                            let n = (e as u32)
                                .wrapping_mul(1664525)
                                .wrapping_add((r as u32).wrapping_mul(1013904223));
                            ((n >> 8) as f32 / 16777216.0 - 0.5) * 12.0
                        }
                    })
                })
                .collect();
            let logits = MxArray::from_float32(&scores, &[rows as i64, EXPERTS as i64])
                .unwrap()
                .astype(dtype)
                .unwrap();
            let shared_values: Vec<f32> = (0..rows)
                .map(|r| {
                    let boundaries = [
                        f32::from_bits(0x3c01_0000),
                        -f32::from_bits(0x3c01_0000),
                        0.7,
                        -0.7,
                        1.0,
                        -1.0,
                        88.0,
                        -88.0,
                    ];
                    boundaries[r % boundaries.len()]
                })
                .collect();
            let shared = MxArray::from_float32(&shared_values, &[rows as i64, 1])
                .unwrap()
                .astype(dtype)
                .unwrap();
            for normalize in [false, true] {
                let (native_weights, native_ids) = topk_from_logits(
                    &logits,
                    EXPERTS as i32,
                    K as i32,
                    RoutingMode::Qwen35 {
                        renormalize_topk: normalize,
                    },
                )
                .unwrap();
                let native_ids = native_ids.to_uint32().unwrap();
                let native_weights = native_weights
                    .astype(DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap();
                let native_shared = Activations::sigmoid(&shared)
                    .unwrap()
                    .astype(DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap();
                let mut raw = unsafe {
                    mlx_sys::mlx_test_splash_moe_route_group(
                        logits.as_raw_ptr(),
                        shared.as_raw_ptr(),
                        K as i32,
                        normalize,
                    )
                };
                assert!(!raw.is_null());
                let mut error = [0i8; 1024];
                assert!(unsafe {
                    mlx_sys::mlx_eval_with_error(&mut raw, 1, error.as_mut_ptr(), error.len())
                });
                let mut words = vec![0u32; unsafe { mlx_sys::mlx_array_size(raw) }];
                assert!(unsafe {
                    mlx_sys::mlx_array_to_uint32(raw, words.as_mut_ptr(), words.len())
                });
                unsafe { mlx_sys::mlx_array_delete(raw) };
                let routes = rows * (K + 1);
                for r in 0..rows {
                    for k in 0..=K {
                        let route = r * (K + 1) + k;
                        let expected = if k == K {
                            EXPERTS as u32
                        } else {
                            native_ids[r * K + k]
                        };
                        assert_eq!(
                            words[route], expected,
                            "dtype={dtype:?} rows={rows} norm={normalize} row={r} k={k}"
                        );
                        let expected = if k == K {
                            native_shared[r]
                        } else {
                            native_weights[r * K + k]
                        };
                        let actual = f32::from_bits(words[routes + route]);
                        if dtype == DType::BFloat16 {
                            assert_eq!(
                                actual.to_bits(),
                                expected.to_bits(),
                                "dtype={dtype:?} rows={rows} norm={normalize} row={r} k={k}"
                            );
                        } else {
                            assert!((actual - expected).abs() < 2e-7);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn nonfinite_routing_never_escapes_or_repeats_experts() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !unsafe { mlx_sys::mlx_splash_moe_available() } {
        return;
    }
    let scores = [
        [f32::NAN; 8],
        [f32::NEG_INFINITY; 8],
        [
            f32::INFINITY,
            1.0,
            -1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            2.0,
            3.0,
            0.0,
        ],
        [f32::NAN, 2.0, 3.0, 4.0, 1.0, f32::NEG_INFINITY, -4.0, 0.0],
    ];
    let logits = MxArray::from_float32(&scores.concat(), &[4, 8]).unwrap();
    let shared = MxArray::from_float32(&[0.0; 4], &[4]).unwrap();
    let mut raw = unsafe {
        mlx_sys::mlx_test_splash_moe_route_group(logits.as_raw_ptr(), shared.as_raw_ptr(), 8, true)
    };
    assert!(!raw.is_null());
    let mut error = [0i8; 1024];
    assert!(unsafe { mlx_sys::mlx_eval_with_error(&mut raw, 1, error.as_mut_ptr(), error.len()) });
    let mut words = vec![0u32; unsafe { mlx_sys::mlx_array_size(raw) }];
    assert!(unsafe { mlx_sys::mlx_array_to_uint32(raw, words.as_mut_ptr(), words.len()) });
    unsafe { mlx_sys::mlx_array_delete(raw) };
    for row in 0..4 {
        let mut selected = words[row * 9..row * 9 + 8].to_vec();
        selected.sort_unstable();
        assert_eq!(selected, (0..8).collect::<Vec<u32>>());
        assert_eq!(words[row * 9 + 8], 8);
        for slot in 0..8 {
            assert!(f32::from_bits(words[36 + row * 9 + slot]).is_nan());
        }
    }
    // 8 routed experts + shared, each with four live rows in one tile.
    let tile_capacity = 9;
    let tiles = 72;
    let count = tiles + tile_capacity * 2;
    let grouped = count + 7;
    let inverse = grouped + tile_capacity * 32;
    assert_eq!(words[count], 9);
    for route in 0..36 {
        let position = words[inverse + route] as usize;
        assert!(position < 9 * 32);
        assert_eq!(words[grouped + position], route as u32);
        assert_eq!(words[tiles + 2 * (position / 32)], words[route]);
    }
}

#[test]
fn explicit_cpu_device_keeps_the_generic_path() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let previous = unsafe { mlx_sys::mlx_default_device() };
    unsafe { mlx_sys::mlx_set_default_device(0) };
    let available = unsafe { mlx_sys::mlx_splash_moe_available() };
    unsafe { mlx_sys::mlx_set_default_device(previous) };
    assert!(
        !available,
        "grouped GPU MoE must not override the caller's CPU device"
    );
}

/// Bounded diagnostic using one real Ornith layer, not the full model. Set
/// SPLASH_MOE_WEIGHTS to the prepared model.safetensors; defaults to rows 8,33,
/// and 2047 (the actual teacher prefill width). No weights are dequantized.
#[test]
#[ignore = "requires an explicitly supplied local Ornith checkpoint and GPU slot"]
fn grouped_actual_layer_stage_comparison() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    assert!(unsafe { mlx_sys::mlx_splash_moe_available() });
    let path = std::env::var("SPLASH_MOE_WEIGHTS").expect("set SPLASH_MOE_WEIGHTS");
    let config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(&path)
                .parent()
                .unwrap()
                .join("config.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let layer: usize = std::env::var("SPLASH_MOE_LAYER")
        .unwrap_or_else(|_| "0".to_string())
        .parse()
        .unwrap();
    eprintln!("ACTUAL_LAYER layer={layer}");
    let source_math = std::env::var("MLX_SPLASH_MOE_SOURCE_MATH").as_deref() == Ok("1");
    let source_reference = source_math.then(|| {
        let directory = std::path::PathBuf::from(
            std::env::var("SPLASH_MOE_SOURCE_ORACLE_DIR")
                .expect("source-math diagnostic requires an independently replayed Splash oracle"),
        );
        let run: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("run.json")).unwrap()).unwrap();
        assert_eq!(run["layer"].as_u64(), Some(layer as u64));
        assert_eq!(run["routing_mode"], "native");
        assert_eq!(run["zero_residual"], true);
        assert_eq!(run["router_bf16"], true);
        assert_eq!(run["shared_bf16"], true);
        assert_eq!(
            run["snapshot"].as_str(),
            Some(std::env::var("SPLASH_MOE_INPUT").unwrap().as_str())
        );
        directory
    });
    let mut tensors = mlx_core::utils::load_safetensors_lazy(path).unwrap();
    let mut get = |name: &str| {
        tensors
            .remove(&format!("model.layers.{layer}.mlp.{name}"))
            .unwrap()
    };
    let router = get("gate.weight");
    let shared_router = get("shared_expert_gate.weight");
    let mut packed = Vec::new();
    for group in ["switch_mlp", "shared_expert"] {
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            packed.push(Weights {
                w: get(&format!("{group}.{projection}.weight")),
                scales: get(&format!("{group}.{projection}.scales")),
                biases: get(&format!("{group}.{projection}.biases")),
            });
        }
    }
    drop(tensors); // Never evaluate the other 39 layers.
    let (h, intermediate, experts, top) = (2048i64, 512i64, 256i32, 8i32);
    let formats: Vec<&KQuant> = (0..6)
        .map(|j| {
            let group = if j < 3 { "switch_mlp" } else { "shared_expert" };
            let projection = ["gate_proj", "up_proj", "down_proj"][j % 3];
            let key = format!("language_model.model.layers.{layer}.mlp.{group}.{projection}");
            let mode = config["quantization"][key]["mode"].as_str().unwrap();
            KQUANTS.iter().find(|q| q.mode == mode).unwrap()
        })
        .collect();
    let tiled_weights: Vec<Weights> = packed
        .iter()
        .zip(&formats)
        .enumerate()
        .map(|(j, (w, q))| {
            let group = if j < 3 { "switch_mlp" } else { "shared_expert" };
            let projection = ["gate_proj", "up_proj", "down_proj"][j % 3];
            let key = format!("language_model.model.layers.{layer}.mlp.{group}.{projection}");
            let entry = &config["quantization"][&key];
            assert_eq!(entry["mode"].as_str(), Some(q.mode));
            let already_tiled = entry["layout"].as_str() == Some("t64");
            assert_eq!(
                already_tiled,
                j >= 3,
                "this diagnostic expects native routed row-major / shared Tiled64 storage"
            );
            if already_tiled {
                Weights {
                    w: w.w.clone(),
                    scales: w.scales.clone(),
                    biases: w.biases.clone(),
                }
            } else {
                Weights {
                    w: tiled(&w.w, i64::from(q.bits)),
                    scales: tiled(&w.scales, 16),
                    biases: tiled(&w.biases, if q.bits == 6 { 1 } else { 2 }),
                }
            }
        })
        .collect();
    let qs = |j: usize| {
        QuantizedSwitchLinear::new(
            packed[j].w.clone(),
            packed[j].scales.clone(),
            Some(packed[j].biases.clone()),
            formats[j].group_size,
            formats[j].bits,
            formats[j].mode.to_string(),
        )
    };
    let ql = |j: usize| {
        QuantizedLinear::new(
            tiled_weights[j].w.clone(),
            tiled_weights[j].scales.clone(),
            Some(tiled_weights[j].biases.clone()),
            None,
            formats[j].group_size,
            formats[j].bits,
            format!("{}@t64", formats[j].mode),
        )
    };
    let gate_q = qs(0);
    let up_q = qs(1);
    let down_q = qs(2);
    let shared_gate_up = ql(3).concat_rows(&ql(4)).unwrap().unwrap();
    let shared_down = ql(5);
    let snapshot = std::env::var("SPLASH_MOE_INPUT").ok().map(|path| {
        let mut saved = mlx_core::utils::load_safetensors_lazy(path).unwrap();
        if let Ok(row) = std::env::var("SPLASH_MOE_INPUT_ROW") {
            let row: i64 = row.parse().unwrap();
            for key in ["x", "router_logits", "shared_logits"] {
                let Some(value) = saved.get(key) else {
                    continue;
                };
                assert!(row >= 0 && row < value.shape().unwrap()[0]);
                let view = value
                    .slice(&[row, 0], &[row + 1, value.shape().unwrap()[1]])
                    .unwrap();
                saved.insert(key.to_string(), view);
            }
        }
        saved
    });
    let rows: Vec<i64> = std::env::var("SPLASH_MOE_ROWS")
        .unwrap_or_else(|_| "8,33,2047".to_string())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let rows = if let Some(saved) = &snapshot {
        vec![saved["x"].shape().unwrap()[0]]
    } else {
        rows
    };
    let mut within_tolerance = true;
    for m in rows {
        if source_math {
            assert_eq!(m, 8, "source diagnostic is M8 only");
        }
        let mut seed = 991u32;
        let xv: Vec<f32> = (0..m * h)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                bf(((seed >> 8) as f32 / 16777216.0 - 0.5) * 3.4641016)
            })
            .collect();
        let x = if let Some(saved) = &snapshot {
            saved["x"].clone()
        } else {
            MxArray::from_float32(&xv, &[m, h])
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap()
        };
        assert_eq!(x.shape().unwrap().as_ref(), [m, h]);
        let xv = if snapshot.is_some() {
            x.astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap()
                .to_vec()
        } else {
            xv
        };
        if std::env::var_os("SPLASH_MOE_SOURCE_ROUTING_ORACLE_DIR").is_some() {
            assert_eq!(m, 8);
            let source_router = snapshot
                .as_ref()
                .and_then(|saved| saved.get("source_router_weight"))
                .unwrap_or(&router);
            let source_shared = snapshot
                .as_ref()
                .and_then(|saved| saved.get("source_shared_gate_weight"))
                .unwrap_or(&shared_router);
            let mut owned = vec![x.clone(), source_router.clone(), source_shared.clone()];
            for w in &tiled_weights {
                owned.extend([w.w.clone(), w.scales.clone(), w.biases.clone()]);
            }
            let modes: Vec<u32> = formats.iter().map(|q| q.bits as u32 | 256).collect();
            source_routing_tests::actual(&owned, &modes, layer);
            continue;
        }
        let logits = if let Some(saved) = &snapshot {
            saved["router_logits"].clone()
        } else if std::env::var("SPLASH_MOE_ROUTE_PATTERN").as_deref() == Ok("skew") {
            let scores: Vec<f32> = (0..m)
                .flat_map(|r| {
                    (0..experts).map(move |e| {
                        let origin = if r == m - 1 { top } else { 0 };
                        if e >= origin && e < origin + top {
                            1.0 + (e - origin) as f32 * 0.13
                        } else {
                            -8.0
                        }
                    })
                })
                .collect();
            MxArray::from_float32(&scores, &[m, i64::from(experts)])
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap()
        } else {
            x.matmul(&router.transpose(None).unwrap()).unwrap()
        };
        let shared_logits = if let Some(saved) = &snapshot {
            saved["shared_logits"].clone()
        } else {
            x.matmul(&shared_router.transpose(None).unwrap()).unwrap()
        };
        if std::env::var_os("SPLASH_MOE_BENCH").is_some() {
            assert!(
                [1, 8].contains(&m),
                "bounded benchmark supports physical M1 or M8"
            );
            let batch: usize = std::env::var("SPLASH_MOE_BENCH_BATCH")
                .ok()
                .map_or(if m == 1 { 1 } else { 16 }, |v| v.parse().unwrap());
            assert!(
                [1, 16].contains(&batch),
                "use batch1 latency or batch16 throughput"
            );
            let switch = SwitchGLU::new_quantized(qs(0), qs(1), qs(2));
            let t64 = |j: usize| {
                QuantizedSwitchLinear::new(
                    tiled_weights[j].w.clone(),
                    tiled_weights[j].scales.clone(),
                    Some(tiled_weights[j].biases.clone()),
                    formats[j].group_size,
                    formats[j].bits,
                    format!("{}@t64", formats[j].mode),
                )
            };
            let switch_t64 = SwitchGLU::new_quantized(t64(0), t64(1), t64(2));
            let native = |switch: &SwitchGLU| {
                let (coeff, selected) = topk_from_logits(
                    &logits,
                    experts,
                    top,
                    RoutingMode::Qwen35 {
                        renormalize_topk: true,
                    },
                )
                .unwrap();
                let routed = switch
                    .forward(&x, &selected)
                    .unwrap()
                    .mul(&coeff.reshape(&[m, i64::from(top), 1]).unwrap())
                    .unwrap()
                    .sum(Some(&[1]), None)
                    .unwrap();
                let gu = shared_gate_up.forward(&x).unwrap();
                let gate = gu.slice(&[0, 0], &[m, intermediate]).unwrap();
                let up = gu
                    .slice(&[0, intermediate], &[m, 2 * intermediate])
                    .unwrap();
                let a = Activations::swiglu_compiled(&gate, &up).unwrap();
                let shared = shared_down
                    .forward(&a)
                    .unwrap()
                    .mul(&Activations::sigmoid(&shared_logits).unwrap())
                    .unwrap();
                let out = routed.add(&shared).unwrap();
                unsafe { mlx_sys::mlx_array_clone_handle(out.as_raw_ptr()) }
            };
            let mut args = vec![
                x.as_raw_ptr(),
                logits.as_raw_ptr(),
                shared_logits.as_raw_ptr(),
            ];
            for w in &tiled_weights {
                args.extend([
                    w.w.as_raw_ptr(),
                    w.scales.as_raw_ptr(),
                    w.biases.as_raw_ptr(),
                ]);
            }
            let modes: Vec<u32> = formats.iter().map(|q| q.bits as u32 | 256).collect();
            let budget = std::env::var("SPLASH_MOE_BUDGET").ok().map(|v| {
                let rows: i32 = v.parse().unwrap();
                assert!((1..=m as i32).contains(&rows));
                MxArray::from_int32(&[rows], &[1]).unwrap()
            });
            if let Some(budget) = &budget {
                args.push(budget.as_raw_ptr());
            }
            let live = std::env::var("SPLASH_MOE_BUDGET")
                .ok()
                .map_or(m as usize, |v| v.parse::<usize>().unwrap());
            let (_, ids) = topk_from_logits(
                &logits,
                experts,
                top,
                RoutingMode::Qwen35 {
                    renormalize_topk: true,
                },
            )
            .unwrap();
            let ids = ids.to_uint32().unwrap();
            let mut occupancy = vec![0usize; experts as usize];
            for &e in ids.iter().take(live * top as usize) {
                occupancy[e as usize] += 1;
            }
            let mut histogram = vec![0usize; 9];
            for &n in &occupancy {
                histogram[n] += 1;
            }
            let routed_tiles: usize = occupancy.iter().map(|n| n.div_ceil(8)).sum();
            let staged_rows: usize = occupancy.iter().map(|&n| if n <= 2 { n } else { 8 }).sum();
            let routes = m as usize * top as usize;
            let max_tiles = routes.min(routes.div_ceil(8) + experts as usize - 1) + 1;
            eprintln!(
                "ACTUAL_MOE_GEOMETRY layer={layer} M={m} budget={live} routes={} routed_tiles={routed_tiles} shared_tiles=1 max_grid_tiles={max_tiles} gathered_rows={} allocated_rows={} occupancy_0_to_8={histogram:?}",
                live * top as usize,
                staged_rows + 8,
                max_tiles * 8
            );
            let grouped = || unsafe {
                mlx_sys::mlx_splash_moe(args.as_ptr(), args.len(), modes.as_ptr(), top, true)
            };
            // Batch1 measures one complete layer with synchronization; batch16
            // measures independent-grid throughput. Inputs/weights are resident
            // after warmup. Neither is a full-model performance estimate.
            let mut stages = if std::env::var_os("SPLASH_MOE_PROFILE").is_some() {
                vec![0, 1, 2, 3, 4, 5, 6, 7]
            } else {
                vec![0, 7]
            };
            if std::env::var_os("SPLASH_MOE_BENCH_T64_M1").is_some() {
                assert_eq!(m, 1);
                assert_eq!(std::env::var("MLX_MOE_T64_M1").as_deref(), Ok("1"));
                stages.push(8);
            }
            for stage in stages {
                let name = [
                    "native",
                    "route",
                    "group",
                    "gather",
                    "gate_up",
                    "down",
                    "combine",
                    "grouped",
                    "native_t64",
                ][stage as usize];
                let build = || {
                    if stage == 0 {
                        native(&switch)
                    } else if stage == 8 {
                        native(&switch_t64)
                    } else if stage == 7 {
                        grouped()
                    } else {
                        unsafe {
                            mlx_sys::mlx_test_splash_moe_profile(
                                args.as_ptr(),
                                args.len(),
                                modes.as_ptr(),
                                top,
                                true,
                                stage,
                            )
                        }
                    }
                };
                if stage == 8 {
                    start_counting();
                    let got = read_output("native T64 route proof", build());
                    assert_eq!(
                        family_count("gather_t64_m1"),
                        3,
                        "gate/up/down must all use the adapter without untile fallback"
                    );
                    stop_counting();
                    let expected = read_output("native row-major route proof", native(&switch));
                    assert_eq!(
                        got, expected,
                        "T64 M1 layer must preserve baseline output bits"
                    );
                }
                if (1..=6).contains(&stage) {
                    let mut raw = build();
                    assert!(!raw.is_null());
                    let mut error = [0i8; 1024];
                    assert!(unsafe {
                        mlx_sys::mlx_eval_with_error(&mut raw, 1, error.as_mut_ptr(), error.len())
                    });
                    assert_eq!(unsafe { mlx_sys::mlx_array_size(raw) }, 1);
                    let mut value = 0u32;
                    assert!(unsafe { mlx_sys::mlx_array_to_uint32(raw, &mut value, 1) });
                    unsafe { mlx_sys::mlx_array_delete(raw) };
                    if stage == 1 {
                        assert!(value < experts as u32);
                    } else {
                        assert_eq!(value as usize, routed_tiles + 1);
                    }
                } else if let Ok(directory) = std::env::var("SPLASH_MOE_OUTPUT_DIR") {
                    let (_, _, bits) = read_output("benchmark output", build());
                    let bytes: Vec<_> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
                    std::fs::write(
                        std::path::Path::new(&directory).join(format!("{name}.u32")),
                        bytes,
                    )
                    .unwrap();
                }
                let mut times = Vec::new();
                let mut build_times = Vec::new();
                for sample in 0..47 {
                    let t = std::time::Instant::now();
                    let mut roots: Vec<_> = (0..batch).map(|_| build()).collect();
                    assert!(roots.iter().all(|p| !p.is_null()));
                    let constructed = t.elapsed().as_secs_f64() * 1e6 / batch as f64;
                    let mut error = [0i8; 1024];
                    let t = std::time::Instant::now();
                    let ok = unsafe {
                        mlx_sys::mlx_eval_with_error(
                            roots.as_mut_ptr(),
                            roots.len(),
                            error.as_mut_ptr(),
                            error.len(),
                        )
                    };
                    let elapsed = t.elapsed().as_secs_f64() * 1e6 / batch as f64;
                    for p in roots {
                        unsafe { mlx_sys::mlx_array_delete(p) };
                    }
                    assert!(ok, "batched layer evaluation failed");
                    if sample >= 16 {
                        times.push(elapsed);
                        build_times.push(constructed);
                    }
                }
                times.sort_by(f64::total_cmp);
                build_times.sort_by(f64::total_cmp);
                eprintln!(
                    "ACTUAL_MOE_BENCH layer={layer} M={m} path={name} budget={live} batch={batch} warmup=16 samples=31 eval_median_us={:.3} eval_p10_us={:.3} graph_build_median_us={:.3}",
                    times[15], times[3], build_times[15]
                );
            }
            continue;
        }
        let (coefficients, ids) = topk_from_logits(
            &logits,
            experts,
            top,
            RoutingMode::Qwen35 {
                renormalize_topk: true,
            },
        )
        .unwrap();
        // Canonical native gather-sort policy (all cases here have >=64 routes).
        let flat = ids.reshape(&[-1]).unwrap();
        let order = flat.argsort(Some(-1)).unwrap();
        let inverse = order.argsort(Some(-1)).unwrap();
        let sorted_ids = flat.take(&order, 0).unwrap();
        let token_ids = order
            .floor_divide(&MxArray::scalar_int(top).unwrap())
            .unwrap();
        let sorted_x = x.reshape(&[-1, 1, h]).unwrap().take(&token_ids, 0).unwrap();
        let gate = gate_q.forward(&sorted_x, &sorted_ids, true).unwrap();
        let up = up_q.forward(&sorted_x, &sorted_ids, true).unwrap();
        let activation = Activations::swiglu_compiled(&gate, &up).unwrap();
        let down = down_q.forward(&activation, &sorted_ids, true).unwrap();
        let gu = shared_gate_up.forward(&x).unwrap();
        let sg = gu.slice(&[0, 0], &[m, intermediate]).unwrap();
        let su = gu
            .slice(&[0, intermediate], &[m, 2 * intermediate])
            .unwrap();
        let sa = Activations::swiglu_compiled(&sg, &su).unwrap();
        let sd = shared_down.forward(&sa).unwrap();
        let mut inputs = vec![
            x.as_raw_ptr(),
            logits.as_raw_ptr(),
            shared_logits.as_raw_ptr(),
        ];
        for w in &tiled_weights {
            inputs.extend([
                w.w.as_raw_ptr(),
                w.scales.as_raw_ptr(),
                w.biases.as_raw_ptr(),
            ]);
        }
        let modes: Vec<u32> = formats.iter().map(|q| q.bits as u32 | 256).collect();
        if std::env::var_os("SPLASH_MOE_WORKSPACE_CHECK").is_some() {
            let mut owned = vec![x.clone(), logits.clone(), shared_logits.clone()];
            for w in &tiled_weights {
                owned.extend([w.w.clone(), w.scales.clone(), w.biases.clone()]);
            }
            workspace_tests::actual_workspace_replay(&owned, &modes, &router, &shared_router);
        }
        if source_math && std::env::var_os("SPLASH_MOE_SOURCE_CHECK_REPLAY").is_some() {
            let mut owned = vec![x.clone(), logits.clone(), shared_logits.clone()];
            for w in &tiled_weights {
                owned.extend([w.w.clone(), w.scales.clone(), w.biases.clone()]);
            }
            source_math_tests::replay_and_fallbacks(&owned, &modes);
        }
        let native_ids = ids.to_uint32().unwrap();
        let mut occupancy = vec![0usize; experts as usize];
        for &expert in native_ids.iter() {
            occupancy[expert as usize] += 1;
        }
        eprintln!(
            "ACTUAL_LAYER M={m} column_experts={} max_routes={}",
            occupancy.iter().filter(|&&n| n > 0 && n <= 2).count(),
            occupancy.iter().max().unwrap()
        );
        let native_coeff = coefficients
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap();
        let mut route_raw = unsafe {
            mlx_sys::mlx_test_splash_moe_route_group(
                logits.as_raw_ptr(),
                shared_logits.as_raw_ptr(),
                top,
                true,
            )
        };
        assert!(!route_raw.is_null());
        let mut route_error = [0i8; 1024];
        assert!(unsafe {
            mlx_sys::mlx_eval_with_error(
                &mut route_raw,
                1,
                route_error.as_mut_ptr(),
                route_error.len(),
            )
        });
        let mut route = vec![0u32; unsafe { mlx_sys::mlx_array_size(route_raw) }];
        assert!(unsafe {
            mlx_sys::mlx_array_to_uint32(route_raw, route.as_mut_ptr(), route.len())
        });
        unsafe { mlx_sys::mlx_array_delete(route_raw) };
        let route_count = m as usize * (top as usize + 1);
        if let Some(directory) = &source_reference {
            let source: serde_json::Value =
                serde_json::from_slice(&std::fs::read(directory.join("routes.json")).unwrap())
                    .unwrap();
            assert_eq!(source["live_rows"], 8);
            for i in 0..route_count {
                assert_eq!(
                    source["selected"][i].as_u64(),
                    Some(u64::from(route[i])),
                    "source route id{i}"
                );
                let coefficient = source["coefficients"][i].as_f64().unwrap() as f32;
                assert_eq!(
                    coefficient.to_bits(),
                    route[route_count + i],
                    "source route coefficient{i}"
                );
            }
        }
        let mut route_mismatches = 0usize;
        for r in 0..m as usize {
            for slot in 0..top as usize {
                let p = r * (top as usize + 1) + slot;
                let n = r * top as usize + slot;
                assert_eq!(route[p], native_ids[n], "actual router IDs M={m}");
                route_mismatches +=
                    usize::from(f32::from_bits(route[route_count + p]) != native_coeff[n]);
            }
        }
        eprintln!(
            "ACTUAL_LAYER M={m} route_coefficient_mismatches={route_mismatches}/{}",
            m * top as i64
        );
        within_tolerance &= route_mismatches == 0;
        if let Ok(directory) = std::env::var("SPLASH_MOE_ROUTE_DUMP") {
            let mut unnormalized = unsafe {
                mlx_sys::mlx_test_splash_moe_route_group(
                    logits.as_raw_ptr(),
                    shared_logits.as_raw_ptr(),
                    top,
                    false,
                )
            };
            assert!(unsafe {
                mlx_sys::mlx_eval_with_error(
                    &mut unnormalized,
                    1,
                    route_error.as_mut_ptr(),
                    route_error.len(),
                )
            });
            let mut words = vec![0u32; unsafe { mlx_sys::mlx_array_size(unnormalized) }];
            assert!(unsafe {
                mlx_sys::mlx_array_to_uint32(unnormalized, words.as_mut_ptr(), words.len())
            });
            unsafe { mlx_sys::mlx_array_delete(unnormalized) };
            let plain: Vec<f32> = words[route_count..route_count * 2]
                .iter()
                .map(|b| f32::from_bits(*b))
                .collect();
            let normalized: Vec<f32> = route[route_count..route_count * 2]
                .iter()
                .map(|b| f32::from_bits(*b))
                .collect();
            let data = serde_json::json!({"M":m,"logits":logits.astype(DType::Float32).unwrap().to_float32().unwrap().to_vec(),"native_probabilities":Activations::softmax(&logits,Some(-1)).unwrap().astype(DType::Float32).unwrap().to_float32().unwrap().to_vec(),"ids":native_ids.to_vec(),"native_coefficients":native_coeff.to_vec(),"grouped_probabilities":plain,"grouped_coefficients":normalized});
            std::fs::write(
                std::path::Path::new(&directory).join(format!("actual-router-{m}.json")),
                serde_json::to_vec(&data).unwrap(),
            )
            .unwrap();
        }
        let native_down = down
            .take(&inverse, 0)
            .unwrap()
            .reshape(&[m, top as i64, h])
            .unwrap();
        let generic = native_down
            .mul(&coefficients.reshape(&[m, top as i64, 1]).unwrap())
            .unwrap()
            .sum(Some(&[1]), None)
            .unwrap()
            .add(
                &sd.mul(&Activations::sigmoid(&shared_logits).unwrap())
                    .unwrap(),
            )
            .unwrap();
        let (first_row, first_slot) = if std::env::var_os("SPLASH_MOE_ORACLE_STAGED").is_some() {
            let (index, &expert) = native_ids
                .iter()
                .enumerate()
                .max_by_key(|(_, expert)| occupancy[**expert as usize])
                .unwrap();
            assert!(
                occupancy[expert as usize] > 2,
                "fixture has no staged routed expert"
            );
            (index as i64 / i64::from(top), index as i64 % i64::from(top))
        } else {
            (0, 0)
        };
        let mut first_actual = Vec::<Vec<f32>>::new();
        let mut first_native = Vec::<Vec<f32>>::new();
        let mut last_actual = Vec::<Vec<f32>>::new();
        let mut last_native = Vec::<Vec<f32>>::new();
        let native_gemm = m * i64::from(top) >= 64 && m * i64::from(top) / i64::from(experts) >= 4;
        for (stage, (routed, shared)) in
            [(&gate, &sg), (&up, &su), (&activation, &sa), (&down, &sd)]
                .into_iter()
                .enumerate()
        {
            let width = if stage == 3 { h } else { intermediate };
            let restored = routed
                .take(&inverse, 0)
                .unwrap()
                .reshape(&[m, top as i64, width])
                .unwrap();
            let native =
                MxArray::concatenate(&restored, &shared.reshape(&[m, 1, width]).unwrap(), 1)
                    .unwrap();
            let reference = native.astype(DType::Float32).unwrap().to_float32().unwrap();
            let (_, dtype, bits) =
                paired_stage_tests::maybe_compare("actual grouped stage", || unsafe {
                    mlx_sys::mlx_test_splash_moe_stage(
                        inputs.as_ptr(),
                        inputs.len(),
                        modes.as_ptr(),
                        top,
                        true,
                        stage as i32 + 1,
                    )
                });
            assert_eq!(dtype, DType::BFloat16);
            let actual: Vec<f32> = bits.iter().map(|v| f32::from_bits(v << 16)).collect();
            let first = ((first_row * (i64::from(top) + 1) + first_slot) * width) as usize;
            if let Some(directory) = &source_reference
                && stage >= 2
            {
                let name = if stage == 2 {
                    "activation-route.bf16"
                } else {
                    "down-route.bf16"
                };
                source_math_tests::require_source_bits(directory, name, &bits);
            }
            first_actual.push(actual[first..first + width as usize].to_vec());
            first_native.push(reference[first..first + width as usize].to_vec());
            let last =
                ((m - 1) * (i64::from(top) + 1) + i64::from(top) - 1) as usize * width as usize;
            last_actual.push(actual[last..last + width as usize].to_vec());
            last_native.push(reference[last..last + width as usize].to_vec());
            let routed_ok = report_stage_delta(
                m,
                stage + 1,
                "routed",
                &actual,
                &reference,
                (top as usize, width as usize),
                false,
            );
            let shared_ok = report_stage_delta(
                m,
                stage + 1,
                "shared",
                &actual,
                &reference,
                (top as usize, width as usize),
                true,
            );
            // Native vector qmv intentionally differs: its BF16 activation
            // partial sums enter the affine bias correction. Compare that
            // tier to the independent decoded-weight F64 oracle below.
            if native_gemm {
                within_tolerance &= routed_ok && shared_ok;
            }
        }
        if !native_gemm || std::env::var_os("SPLASH_MOE_CPU_ORACLE").is_some() {
            for (sample_row, sample_slot, sample_actual, sample_native) in [
                (first_row, first_slot, &first_actual, &first_native),
                (m - 1, i64::from(top) - 1, &last_actual, &last_native),
            ] {
                let expert =
                    i64::from(native_ids[(sample_row * i64::from(top) + sample_slot) as usize]);
                eprintln!(
                    "ACTUAL_LAYER M={m} CPU_ORACLE row={sample_row} slot={sample_slot} expert={expert} occupancy={}",
                    occupancy[expert as usize]
                );
                let mut decoded = Vec::new();
                for j in 0..3 {
                    let n = if j == 2 { h } else { intermediate };
                    let slice = |a: &MxArray| {
                        let width = a.shape().unwrap()[2];
                        a.slice(&[expert, 0, 0], &[expert + 1, n, width])
                            .unwrap()
                            .reshape(&[n, width])
                            .unwrap()
                    };
                    let weight = Weights {
                        w: slice(&packed[j].w),
                        scales: slice(&packed[j].scales),
                        biases: slice(&packed[j].biases),
                    };
                    let (_, _, bits) = read_bits(
                        "actual CPU dequantization",
                        dequantize(&weight, DType::Float32, formats[j], CPU),
                    );
                    decoded.push(bits.into_iter().map(f32::from_bits).collect::<Vec<_>>());
                }
                let input = &xv[(sample_row * h) as usize..((sample_row + 1) * h) as usize];
                let precision = if source_math {
                    1
                } else if native_gemm {
                    2
                } else {
                    0
                };
                let oracle_gate = project(input, &decoded[0], intermediate as usize, precision);
                let oracle_up = project(input, &decoded[1], intermediate as usize, precision);
                let oracle_activation: Vec<f32> = oracle_gate
                    .iter()
                    .zip(&oracle_up)
                    .map(|(&g, &u)| {
                        if source_math {
                            bf(u * (g / (1.0 + (-g).exp())))
                        } else {
                            bf(bf(bf_sigmoid(g) * g) * u)
                        }
                    })
                    .collect();
                let oracle_down = project(&oracle_activation, &decoded[2], h as usize, precision);
                for (j, reference) in [oracle_gate, oracle_up, oracle_activation, oracle_down]
                    .iter()
                    .enumerate()
                {
                    within_tolerance &= report_stage_delta(
                        m,
                        j + 1,
                        "grouped_vs_cpu_f64",
                        &sample_actual[j],
                        reference,
                        (0, reference.len()),
                        true,
                    );
                    let _ = report_stage_delta(
                        m,
                        j + 1,
                        "native_vs_cpu_f64",
                        &sample_native[j],
                        reference,
                        (0, reference.len()),
                        true,
                    );
                }
            }
        }
        let reference = generic
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap();
        let (_, _, bits) =
            paired_stage_tests::maybe_compare("actual grouped combined", || unsafe {
                mlx_sys::mlx_splash_moe(inputs.as_ptr(), inputs.len(), modes.as_ptr(), top, true)
            });
        let actual: Vec<f32> = bits.iter().map(|v| f32::from_bits(v << 16)).collect();
        let native_delta = report_stage_delta(
            m,
            5,
            "combined_native_comparison",
            &actual,
            &reference,
            (0, h as usize),
            true,
        );
        if let Some(directory) = &source_reference {
            source_math_tests::require_source_bits(directory, "output.bf16", &bits);
        } else {
            within_tolerance &= native_delta;
        }
    }
    assert!(
        within_tolerance,
        "actual-layer independent oracle, routing, or full-output bound failed"
    );
}

fn report_stage_delta(
    rows: i64,
    stage: usize,
    part: &str,
    actual: &[f32],
    reference: &[f32],
    layout: (usize, usize),
    shared: bool,
) -> bool {
    let (top, width) = layout;
    assert_eq!(actual.len(), reference.len());
    let mut count = 0usize;
    let mut exact = 0usize;
    let mut square = 0f64;
    let mut ref_square = 0f64;
    let mut worst = 0f32;
    let mut peak = 0f32;
    for (i, (&a, &r)) in actual.iter().zip(reference).enumerate() {
        if ((i / width) % (top + 1) == top) != shared {
            continue;
        }
        assert!(a.is_finite() && r.is_finite());
        count += 1;
        exact += usize::from(a == r);
        square += (f64::from(a) - f64::from(r)).powi(2);
        ref_square += f64::from(r).powi(2);
        worst = worst.max((a - r).abs());
        peak = peak.max(r.abs());
    }
    let relative_rms = (square / ref_square.max(1e-30)).sqrt();
    eprintln!(
        "ACTUAL_LAYER M={rows} stage={stage} part={part} elements={count} exact={:.8} relative_rms={relative_rms:.8} max_abs={worst:.8} peak={peak:.8}",
        exact as f64 / count as f64
    );
    // Same strict peak-normalized limit as the synthetic native reference.
    worst <= peak * 0.01 + 1e-6
}

#[path = "splash_moe/register.rs"]
mod register_expert_tests;

#[path = "splash_moe/paired_stage.rs"]
mod paired_stage_tests;

#[path = "splash_moe/source_math.rs"]
mod source_math_tests;

#[path = "splash_moe/source_routing.rs"]
mod source_routing_tests;

#[path = "splash_moe/workspace.rs"]
mod workspace_tests;
