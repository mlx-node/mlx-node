include!("full_width_context_tests.rs");
include!("execution_policy_tests.rs");
use super::{
    CandidateSelector, DFlash2Config, DFlash2ContextCache, SelectorPath, normalized_selector_probs,
    parallel_query_ids,
};
use crate::array::{DType, MxArray};
use crate::models::quantized_linear::LinearProj;
use crate::nn::{Embedding, Linear};

/// The smallest geometry whose every draft projection is tileable
/// (`N % 64 == 0`, `K % 256 == 0`): o_proj's K is `heads * head_dim`.
fn checkpoint_inventory_config() -> DFlash2Config {
    DFlash2Config {
        block_size: 7,
        mask_token_id: 0,
        target_layers: vec![0, 1],
        target_num_layers: 2,
        hidden_size: 256,
        intermediate_size: 256,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        num_key_value_heads: 1,
        head_dim: 64,
        vocab_size: 32,
        rms_norm_eps: 1e-5,
        max_position_embeddings: 128,
        sliding_window: 8,
        conv_group_size: 16,
        conv_kernel_size: 2,
        selector_rank: 8,
        selector_top_k: 4,
        rope_theta: 10_000.0,
        final_logit_softcapping: None,
        valid_vocab_size: None,
        capture_final_norm: None,
        embedding_normalization: None,
    }
}

fn checkpoint_inventory() -> std::collections::HashMap<String, MxArray> {
    super::expected_tensor_shapes(&checkpoint_inventory_config())
        .into_iter()
        .map(|(name, shape)| {
            let array = MxArray::zeros(&shape, Some(DType::BFloat16)).unwrap();
            (name, array)
        })
        .collect()
}

/// The ring holds `sliding_window` slots and a block's rows past them.
/// Mutation: drop the block rows from the ring rows → red.
#[test]
fn draft_kv_ring_bytes_match_the_window_buffer_cap() {
    let config = DFlash2Config {
        num_hidden_layers: 5,
        num_key_value_heads: 8,
        head_dim: 128,
        sliding_window: 2048,
        block_size: 7,
        ..checkpoint_inventory_config()
    };
    assert_eq!(config.kv_ring_bytes(), (2048 + 8) * 5 * 2 * 8 * 128 * 2);
    assert_eq!(config.kv_ring_bytes(), 42_106_880);
}

#[test]
fn dense_draft_inventory_rejects_malformed_tensors_before_evaluation() {
    let config = checkpoint_inventory_config();
    let mut params = checkpoint_inventory();
    super::validate_tensor_inventory(&params, &config).unwrap();
    for (name, shape) in super::expected_tensor_shapes(&config) {
        let saved = params.remove(&name).unwrap();
        assert!(
            super::validate_tensor_inventory(&params, &config)
                .unwrap_err()
                .to_string()
                .contains(&name)
        );
        params.insert(
            name.clone(),
            MxArray::zeros(&[1], Some(DType::BFloat16)).unwrap(),
        );
        assert!(
            super::validate_tensor_inventory(&params, &config)
                .unwrap_err()
                .to_string()
                .contains("shape")
        );
        params.insert(
            name.clone(),
            MxArray::zeros(&shape, Some(DType::Uint32)).unwrap(),
        );
        assert!(
            super::validate_tensor_inventory(&params, &config)
                .unwrap_err()
                .to_string()
                .contains("dtype")
        );
        params.insert(name, saved);
    }
    params.insert(
        "fc.scales".into(),
        MxArray::zeros(&[256, 2], Some(DType::BFloat16)).unwrap(),
    );
    assert!(
        super::validate_tensor_inventory(&params, &config)
            .unwrap_err()
            .to_string()
            .contains("fc.scales")
    );
}

#[test]
fn selector_linear_keeps_checkpoint_precision() {
    let mut params = checkpoint_inventory();
    let projection = super::dense_linear(&mut params, "fc", 512, 256).unwrap();
    assert!(matches!(projection, LinearProj::Standard(_)));
    assert_eq!(projection.get_weight().dtype().unwrap(), DType::BFloat16);
}

/// The mode a draft projection carries on this host: the affine arrays
/// tiled into the K-quant contract where the `_t64` kernels run, MLX's
/// own affine route elsewhere.
fn draft_mode() -> &'static str {
    if crate::models::quant_dispatch::kquant_tiled_enabled() {
        "a4g64@t64"
    } else {
        "affine"
    }
}

/// Asserts an affine Q4/group64 projection of a bf16 `[rows, cols]`
/// weight: 8 codes per u32 and one bf16 scale/offset pair per 64 inputs
/// (the Tiled64 permutation keeps the 2-D shapes).
fn assert_q4_group64(projection: &LinearProj, rows: i64, cols: i64, name: &str) {
    let LinearProj::Quantized(packed) = projection else {
        panic!("{name}: draft projection must load quantized");
    };
    assert_eq!(packed.mode(), draft_mode(), "{name}");
    assert_eq!(packed.bits(), 4, "{name}");
    assert_eq!(
        packed.get_weight().dtype().unwrap(),
        DType::Uint32,
        "{name}"
    );
    assert_eq!(
        packed.get_weight().shape().unwrap().as_ref(),
        &[rows, cols / 8],
        "{name}: packed codes"
    );
    assert_eq!(
        packed.get_scales().dtype().unwrap(),
        DType::BFloat16,
        "{name}"
    );
    assert_eq!(
        packed.get_scales().shape().unwrap().as_ref(),
        &[rows, cols / 64],
        "{name}: scales"
    );
    let biases = packed.get_biases().expect("affine Q4 has offsets");
    assert_eq!(biases.dtype().unwrap(), DType::BFloat16, "{name}");
    assert_eq!(
        biases.shape().unwrap().as_ref(),
        &[rows, cols / 64],
        "{name}: offsets"
    );
}

/// `draft_linear` keeps the affine Q4/group64 numerics through the
/// tiled contract: integer endpoints spanning exactly 15 steps decode
/// exactly, in rows of either sign (stored offsets), on every route a
/// decode block reaches (M = 1, 8 and a prefill height).
#[test]
fn draft_linear_is_q4_group64_with_exact_endpoints_and_residency() {
    let (rows, cols) = (64i64, 256i64);
    let values = (0..rows * cols)
        .map(|i| {
            let (row, col) = (i / cols, i % cols);
            if col % 2 == 0 {
                0.0
            } else if row % 2 == 0 {
                15.0
            } else {
                -15.0
            }
        })
        .collect::<Vec<_>>();
    let weight = MxArray::from_float32(&values, &[rows, cols])
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap();
    let mut params = std::collections::HashMap::from([("fc.weight".into(), weight.clone())]);
    let mut savings = 0;
    let projection = super::draft_linear(
        &mut params,
        "fc",
        cols as usize,
        rows as usize,
        &mut savings,
    )
    .unwrap();
    assert_q4_group64(&projection, rows, cols, "fc");
    let LinearProj::Quantized(packed) = &projection else {
        unreachable!();
    };
    let resident = packed.get_weight().nbytes()
        + packed.get_scales().nbytes()
        + packed.get_biases().map_or(0, |biases| biases.nbytes());
    // Each row owns 128 packed bytes and four bf16 scale/offset pairs.
    assert_eq!(resident as i64, rows * (cols / 2 + cols / 64 * 4));
    assert_eq!(savings, weight.nbytes() as u64 - resident as u64);
    assert!(params.is_empty());

    // Every row sums its 128 non-zero entries: +-1920, exact in bf16.
    let want = (0..rows)
        .map(|row| if row % 2 == 0 { 1920.0 } else { -1920.0 })
        .collect::<Vec<f32>>();
    for m in [1i64, 8, 87] {
        let input = MxArray::ones(&[1, m, cols], Some(DType::BFloat16)).unwrap();
        let output = projection.forward(&input).unwrap().to_float32().unwrap();
        assert_eq!(output.len() as i64, m * rows, "M={m}");
        for (row, got) in output.chunks_exact(rows as usize).enumerate() {
            assert_eq!(got, want.as_slice(), "M={m} input row {row}");
        }
    }

    // An untileable geometry fails loud rather than loading slower.
    let odd = MxArray::zeros(&[2, 128], Some(DType::BFloat16)).unwrap();
    let Err(err) = super::quantize_draft_weight(&odd, "odd") else {
        panic!("an untileable draft weight must be refused");
    };
    assert!(err.to_string().contains("N % 64 == 0"), "{err}");
}

#[test]
fn load_quantizes_every_draft_projection_and_keeps_the_selector_dense() {
    let config = checkpoint_inventory_config();
    let dir = std::env::temp_dir().join(format!(
        "mlx-dflash2-load-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        serde_json::json!({
            "architectures": ["DFlash2DraftModel"],
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_target_layers": config.target_num_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "vocab_size": config.vocab_size,
            "rms_norm_eps": config.rms_norm_eps,
            "max_position_embeddings": config.max_position_embeddings,
            "sliding_window": config.sliding_window,
            "layer_types": ["sliding_attention"],
            "rope_theta": config.rope_theta,
            "dflash_config": {
                "block_size": config.block_size + 1,
                "conv_group_size": config.conv_group_size,
                "conv_kernel_size": config.conv_kernel_size,
                "mask_token_id": config.mask_token_id,
                "selector_rank": config.selector_rank,
                "selector_top_k": config.selector_top_k,
                "target_layer_ids": config.target_layers,
            },
        })
        .to_string(),
    )
    .unwrap();
    let shapes = super::expected_tensor_shapes(&config);
    let mut tensors = shapes
        .iter()
        .map(|(name, shape)| {
            let array = MxArray::random_normal(shape, 0.0, 0.02, Some(DType::BFloat16)).unwrap();
            (name.clone(), array)
        })
        .collect::<std::collections::HashMap<_, _>>();
    crate::utils::safetensors::save_safetensors(dir.join("model.safetensors"), &mut tensors, None)
        .unwrap();
    let loaded = super::load_dflash2(&dir);
    std::fs::remove_dir_all(&dir).unwrap();
    let (model, bytes) = loaded.unwrap();

    let projection = |name: &str| -> bool {
        name == "fc.weight"
            || name.ends_with("_proj.weight")
            || name.ends_with(".kernel_projection.weight")
    };
    let packed_bytes = |elements: u64| elements / 2 + elements / 64 * 4;
    let mut expected_bytes = shapes
        .iter()
        .map(|(name, shape)| {
            let elements = shape.iter().product::<i64>() as u64;
            if projection(name) {
                // u32 codes at 4 bits plus bf16 scale and offset per 64.
                packed_bytes(elements)
            } else {
                elements * 2
            }
        })
        .sum::<u64>();
    // The merges follow the tiled contract: on a Metal host every
    // projection tiles, and on CUDA affine Q4/g64 draft projections also
    // merge rows, so q|k|v, gate|up and the cross-layer k|v (a second
    // resident copy of k/v) all exist.
    let merges_rows = crate::models::quant_dispatch::kquant_tiled_enabled()
        || crate::accelerator::cuda_ops_available();
    assert_eq!(model.context_kv.is_some(), merges_rows);
    assert_eq!(model.layers[0].attention.qkv_proj.is_some(), merges_rows);
    assert_eq!(model.layers[0].mlp.gate_up.is_some(), merges_rows);
    if model.context_kv.is_some() {
        let kv = (config.num_key_value_heads * config.head_dim * config.hidden_size) as u64;
        expected_bytes += 2 * packed_bytes(kv) * config.num_hidden_layers as u64;
    }
    assert_eq!(bytes, expected_bytes);
    assert_eq!(model.weight_bytes, expected_bytes);

    let hidden = config.hidden_size as i64;
    let intermediate = config.intermediate_size as i64;
    let attention = (config.num_attention_heads * config.head_dim) as i64;
    let kv = (config.num_key_value_heads * config.head_dim) as i64;
    let conv = (2 * config.conv_kernel_size * (config.hidden_size / config.conv_group_size)) as i64;
    let taps = config.target_layers.len() as i64;
    assert_q4_group64(&model.fc, hidden, hidden * taps, "fc");
    assert_eq!(model.layers.len(), 1);
    let layer = &model.layers[0];
    for (name, projection, rows, cols) in [
        ("q_proj", &layer.attention.q_proj, attention, hidden),
        ("k_proj", &layer.attention.k_proj, kv, hidden),
        ("v_proj", &layer.attention.v_proj, kv, hidden),
        ("o_proj", &layer.attention.o_proj, hidden, attention),
        ("gate_proj", &layer.mlp.gate_proj, intermediate, hidden),
        ("up_proj", &layer.mlp.up_proj, intermediate, hidden),
        ("down_proj", &layer.mlp.down_proj, hidden, intermediate),
        (
            "attention_conv",
            &layer.attention_conv.kernel_projection,
            conv,
            hidden,
        ),
        ("mlp_conv", &layer.mlp_conv.kernel_projection, conv, hidden),
    ] {
        assert_q4_group64(projection, rows, cols, name);
    }
    let selector = &model.selector.hidden_projection;
    assert!(matches!(selector, LinearProj::Standard(_)));
    assert_eq!(selector.get_weight().dtype().unwrap(), DType::BFloat16);
}

fn test_selector(vocab: usize, top_k: usize, rank: usize, hidden: usize) -> CandidateSelector {
    let normal = |rows: usize, cols: usize| {
        MxArray::random_normal(&[rows as i64, cols as i64], 0.0, 0.5, Some(DType::Float32)).unwrap()
    };
    CandidateSelector {
        predecessor_codebook: Embedding::from_weight(&normal(vocab, rank)).unwrap(),
        successor_codebook: Embedding::from_weight(&normal(vocab, rank)).unwrap(),
        hidden_projection: LinearProj::Standard(
            Linear::from_weights(&normal(rank, hidden), None).unwrap(),
        ),
        top_k,
        rank,
        vocab_size: vocab,
        active_vocab_size: vocab,
        final_logit_softcapping: None,
    }
}

#[test]
#[ignore = "run with MLX_DFLASH2_SELECTOR_EDGES_FP32=1 to validate the opt-in production hook"]
fn selector_fp32_edges_hook_widens_native_bf16_unary() {
    assert_eq!(
        std::env::var("MLX_DFLASH2_SELECTOR_EDGES_FP32").as_deref(),
        Ok("1")
    );
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        return;
    }
    let mut selector = test_selector(32, 16, 256, 64);
    let predecessor = selector
        .predecessor_codebook
        .get_weight()
        .astype(DType::BFloat16)
        .unwrap();
    let successor = selector
        .successor_codebook
        .get_weight()
        .astype(DType::BFloat16)
        .unwrap();
    selector.predecessor_codebook = Embedding::from_weight(&predecessor).unwrap();
    selector.successor_codebook = Embedding::from_weight(&successor).unwrap();
    let candidates = MxArray::from_int32(
        &(0..112).map(|i| (i * 7 + 3) % 32).collect::<Vec<_>>(),
        &[1, 7, 16],
    )
    .unwrap();
    let unary = MxArray::random_normal(&[1, 7, 16], 0.0, 0.5, Some(DType::BFloat16)).unwrap();
    let hidden = MxArray::random_normal(&[7, 1, 256], 0.0, 0.5, Some(DType::BFloat16)).unwrap();
    let anchor = MxArray::from_int32(&[9], &[1]).unwrap();
    let widened = unary.astype(DType::Float32).unwrap();
    let expected = MxArray::from_handle(
        unsafe {
            mlx_sys::mlx_dflash2_selector_edges(
                candidates.as_raw_ptr(),
                widened.as_raw_ptr(),
                hidden.as_raw_ptr(),
                predecessor.as_raw_ptr(),
                successor.as_raw_ptr(),
                anchor.as_raw_ptr(),
                0.0,
            )
        },
        "test selector FP32 edges",
    )
    .unwrap();
    let actual = selector
        .edge_scores(&candidates, &unary, &hidden, &anchor, 7)
        .unwrap();
    let actual = actual.to_float32().unwrap();
    assert_eq!(actual.as_ref(), expected.to_float32().unwrap().as_ref());
    assert!(
        actual.iter().any(|v| v.to_bits() & 0xffff != 0),
        "native BF16 fallback would erase low score bits"
    );
}

fn direct_fused_greedy(candidates: &MxArray, scores: &MxArray) -> Option<Vec<i32>> {
    let metal = unsafe { mlx_sys::mlx_metal_is_available() };
    let cuda = crate::accelerator::cuda_ops_available();
    if !metal && !cuda {
        return None;
    }
    let mut path = std::ptr::null_mut();
    let ok = if metal {
        unsafe {
            mlx_sys::mlx_dflash2_greedy_path(
                candidates.as_raw_ptr(),
                scores.as_raw_ptr(),
                &mut path,
            )
        }
    } else {
        unsafe {
            mlx_sys::mlx_dflash2_greedy_path_cuda(
                candidates.as_raw_ptr(),
                scores.as_raw_ptr(),
                &mut path,
            )
        }
    };
    assert!(ok);
    assert!(!path.is_null());
    let path = MxArray::from_handle(path, "test dflash2 greedy path").unwrap();
    Some(path.to_int32().unwrap().as_ref().to_vec())
}

fn lazy_greedy(
    selector: &CandidateSelector,
    candidates: &MxArray,
    scores: &MxArray,
    length: usize,
) -> Vec<i32> {
    selector
        .lazy_greedy_path(candidates, scores, length)
        .unwrap()
        .to_int32()
        .unwrap()
        .as_ref()
        .to_vec()
}

#[test]
fn fused_greedy_path_matches_lazy_random_and_branching_walks() {
    if !unsafe { mlx_sys::mlx_metal_is_available() } && !crate::accelerator::cuda_ops_available() {
        eprintln!("skipping: fused DFlash2 greedy path needs a supported GPU");
        return;
    }
    for top_k in [4usize, 16] {
        let selector = test_selector(32, top_k, 4, 4);
        for length in 1usize..=8 {
            let candidate_values = (0..length * top_k)
                .map(|i| 10_000 + i as i32)
                .collect::<Vec<_>>();
            let candidates =
                MxArray::from_int32(&candidate_values, &[1, length as i64, top_k as i64]).unwrap();

            // Finite random scores cover arbitrary predecessor-dependent
            // routes. Exact output indices, not score tolerances, are the
            // contract under test.
            let random_scores = MxArray::random_normal(
                &[length as i64, top_k as i64, top_k as i64],
                0.0,
                1.0,
                Some(DType::Float32),
            )
            .unwrap();
            assert_eq!(
                direct_fused_greedy(&candidates, &random_scores).unwrap(),
                lazy_greedy(&selector, &candidates, &random_scores, length),
                "random L={length} K={top_k}",
            );

            // Each predecessor row deliberately has a different winner;
            // a kernel that forgets to feed the prior selection into the
            // next position fails this expected path.
            let mut branching = vec![-100.0f32; length * top_k * top_k];
            for position in 0..length {
                for predecessor in 0..top_k {
                    let winner = (predecessor * 3 + position * 5 + 1) % top_k;
                    let row = (position * top_k + predecessor) * top_k;
                    branching[row + winner] = 100.0 + position as f32;
                }
            }
            let scores =
                MxArray::from_float32(&branching, &[length as i64, top_k as i64, top_k as i64])
                    .unwrap();
            let mut predecessor = 0usize;
            let mut expected = Vec::with_capacity(length);
            for position in 0..length {
                predecessor = (predecessor * 3 + position * 5 + 1) % top_k;
                expected.push(candidate_values[position * top_k + predecessor]);
            }
            assert_eq!(
                direct_fused_greedy(&candidates, &scores).unwrap(),
                expected,
                "branching L={length} K={top_k}",
            );
        }
    }
}

#[test]
fn fused_greedy_path_matches_lazy_ties_infinities_signed_zero_and_nan() {
    if !unsafe { mlx_sys::mlx_metal_is_available() } && !crate::accelerator::cuda_ops_available() {
        eprintln!("skipping: fused DFlash2 greedy path needs a supported GPU");
        return;
    }
    for top_k in [4usize, 16] {
        let length = 12usize;
        let selector = test_selector(32, top_k, 4, 4);
        let candidate_values = (0..length * top_k)
            .map(|i| 20_000 + i as i32)
            .collect::<Vec<_>>();
        let candidates =
            MxArray::from_int32(&candidate_values, &[1, length as i64, top_k as i64]).unwrap();
        let mut values = vec![f32::NEG_INFINITY; length * top_k * top_k];
        for position in 0..length {
            for predecessor in 0..top_k {
                let row = (position * top_k + predecessor) * top_k;
                match position % 7 {
                    0 => {
                        values[row + 1] = 7.0;
                        values[row + top_k - 2] = 7.0; // last equal max wins
                    }
                    1 => {
                        // IEEE `>` treats both zero signs as equal, so a
                        // trailing -0.0 must beat an earlier +0.0 under
                        // the reversed last-maximum walk. `total_cmp`
                        // would choose the +0.0 and is deliberately not
                        // the device argmax contract exercised here.
                        values[row..row + top_k].fill(-0.0);
                        values[row + 1] = 0.0;
                    }
                    2 => {
                        // A NaN beats every number, and the reverse walk
                        // meets the last NaN slot first.
                        values[row..row + top_k].fill(f32::NAN);
                        values[row] = 3.0;
                        values[row + top_k / 2] = 3.0;
                    }
                    3 => values[row..row + top_k].fill(f32::NAN),
                    4 => {
                        values[row + 1] = f32::INFINITY;
                        values[row + top_k - 2] = f32::INFINITY;
                    }
                    5 => {
                        // The walk meets 9.0 before the one NaN; the NaN
                        // still wins.
                        values[row + 1] = f32::NAN;
                        values[row + top_k - 1] = 9.0;
                    }
                    _ => {
                        // Leave the row entirely -inf: no value exceeds
                        // the initial -inf, so the reverse walk retains
                        // the final slot.
                    }
                }
            }
        }
        let scores =
            MxArray::from_float32(&values, &[length as i64, top_k as i64, top_k as i64]).unwrap();
        let expected_slots = [
            top_k - 2,
            top_k - 1,
            top_k - 1,
            top_k - 1,
            top_k - 2,
            1,
            top_k - 1,
        ]
        .into_iter()
        .cycle()
        .take(length)
        .collect::<Vec<_>>();
        let expected = expected_slots
            .iter()
            .enumerate()
            .map(|(position, &slot)| candidate_values[position * top_k + slot])
            .collect::<Vec<_>>();
        let fused = direct_fused_greedy(&candidates, &scores).unwrap();
        assert_eq!(fused, expected, "special values K={top_k}");
        assert_eq!(
            fused,
            lazy_greedy(&selector, &candidates, &scores, length),
            "device argmax semantics K={top_k}",
        );
    }
}

#[test]
fn fused_greedy_path_accepts_noncontiguous_views() {
    if !unsafe { mlx_sys::mlx_metal_is_available() } && !crate::accelerator::cuda_ops_available() {
        eprintln!("skipping: fused DFlash2 greedy path needs a supported GPU");
        return;
    }
    let (length, top_k) = (7usize, 16usize);
    let selector = test_selector(32, top_k, 4, 4);
    let candidates = MxArray::from_int32(
        &(0..length * top_k).map(|i| i as i32).collect::<Vec<_>>(),
        &[1, top_k as i64, length as i64],
    )
    .unwrap()
    .transpose(Some(&[0, 2, 1]))
    .unwrap();
    let scores = MxArray::random_normal(
        &[top_k as i64, length as i64, top_k as i64],
        0.0,
        1.0,
        Some(DType::Float32),
    )
    .unwrap()
    .transpose(Some(&[1, 0, 2]))
    .unwrap();
    assert_eq!(
        direct_fused_greedy(&candidates, &scores).unwrap(),
        lazy_greedy(&selector, &candidates, &scores, length),
    );
}

#[test]
fn fused_greedy_confidence_matches_selected_softmax_probability() {
    if !unsafe { mlx_sys::mlx_metal_is_available() } && !crate::accelerator::cuda_ops_available() {
        return;
    }
    let (length, top_k) = (7usize, 16usize);
    let selector = test_selector(32, top_k, 4, 4);
    let candidates = MxArray::from_int32(
        &(0..length * top_k).map(|i| i as i32).collect::<Vec<_>>(),
        &[1, length as i64, top_k as i64],
    )
    .unwrap();
    let scores = MxArray::random_normal(
        &[length as i64, top_k as i64, top_k as i64],
        0.0,
        2.0,
        Some(DType::Float32),
    )
    .unwrap();
    let score_values = scores.to_float32().unwrap().to_vec();
    let (path, confidence) = selector
        .fused_greedy_path_confidence(&candidates, &scores)
        .unwrap();
    MxArray::eval_arrays(&[&path, &confidence]).unwrap();
    assert_eq!(
        path.to_int32().unwrap().as_ref(),
        lazy_greedy(&selector, &candidates, &scores, length)
    );
    let actual = confidence.to_float32().unwrap();
    let mut predecessor = 0usize;
    for position in 0..length {
        let row = (position * top_k + predecessor) * top_k;
        let mut selected = top_k - 1;
        let mut best = f32::NEG_INFINITY;
        for candidate in (0..top_k).rev() {
            let value = score_values[row + candidate];
            if (!best.is_nan() && value.is_nan()) || value > best {
                best = value;
                selected = candidate;
            }
        }
        let normalizer = (0..top_k)
            .map(|candidate| (score_values[row + candidate] - best).exp())
            .sum::<f32>();
        let expected = 1.0 / normalizer;
        assert!(
            (actual[position] - expected).abs() <= 2e-6 * expected.max(1.0),
            "confidence position {position}: {} vs {expected}",
            actual[position]
        );
        predecessor = selected;
    }
}

#[test]
fn fused_greedy_path_ffi_rejects_malformed_inputs() {
    let good_candidates = MxArray::from_int32(&[0; 8], &[1, 2, 4]).unwrap();
    let good_scores = MxArray::from_float32(&[0.0; 32], &[2, 4, 4]).unwrap();
    let bad_candidate_dtype = MxArray::from_float32(&[0.0; 8], &[1, 2, 4]).unwrap();
    let bad_score_dtype = good_scores.astype(DType::BFloat16).unwrap();
    let wrong_length = MxArray::from_float32(&[0.0; 16], &[1, 4, 4]).unwrap();
    let nonsquare = MxArray::from_float32(&[0.0; 40], &[2, 4, 5]).unwrap();
    for (candidates, scores) in [
        (&bad_candidate_dtype, &good_scores),
        (&good_candidates, &bad_score_dtype),
        (&good_candidates, &wrong_length),
        (&good_candidates, &nonsquare),
    ] {
        let mut path = std::ptr::null_mut();
        assert!(!unsafe {
            mlx_sys::mlx_dflash2_greedy_path(
                candidates.as_raw_ptr(),
                scores.as_raw_ptr(),
                &mut path,
            )
        });
        assert!(path.is_null());
    }
    assert!(!unsafe {
        mlx_sys::mlx_dflash2_greedy_path(
            good_candidates.as_raw_ptr(),
            good_scores.as_raw_ptr(),
            std::ptr::null_mut(),
        )
    });
}

/// Diagnostic only: isolates the terminal device-resident predecessor walk
/// at the production DFlash2 shape. Each timed iteration constructs a fresh
/// graph, evaluates it, and reads the path; shared inputs are materialized
/// before warmup. Alternating batches limit order/thermal bias. This prints
/// observations and deliberately asserts no timing threshold.
#[test]
#[ignore = "diagnostic GPU timing; run manually on an idle machine"]
fn benchmark_fused_greedy_path_vs_lazy_graph() {
    if !unsafe { mlx_sys::mlx_metal_is_available() } && !crate::accelerator::cuda_ops_available() {
        eprintln!("skipping: DFlash2 greedy-path benchmark needs a supported GPU");
        return;
    }
    const LENGTH: usize = 7;
    const TOP_K: usize = 16;
    const BATCH_ITERS: usize = 50;
    const SAMPLES: usize = 12;
    let selector = test_selector(32, TOP_K, 4, 4);
    let candidates = MxArray::from_int32(
        &(0..LENGTH * TOP_K).map(|i| i as i32).collect::<Vec<_>>(),
        &[1, LENGTH as i64, TOP_K as i64],
    )
    .unwrap();
    let scores = MxArray::random_normal(
        &[LENGTH as i64, TOP_K as i64, TOP_K as i64],
        0.0,
        1.0,
        Some(DType::Float32),
    )
    .unwrap();
    MxArray::eval_arrays(&[&candidates, &scores]).unwrap();

    let expected = lazy_greedy(&selector, &candidates, &scores, LENGTH);
    assert_eq!(direct_fused_greedy(&candidates, &scores).unwrap(), expected);
    for _ in 0..3 {
        assert_eq!(
            lazy_greedy(&selector, &candidates, &scores, LENGTH),
            expected
        );
        assert_eq!(direct_fused_greedy(&candidates, &scores).unwrap(), expected);
    }

    let measure = |fused: bool| {
        let start = std::time::Instant::now();
        for _ in 0..BATCH_ITERS {
            let path = if fused {
                direct_fused_greedy(&candidates, &scores).unwrap()
            } else {
                lazy_greedy(&selector, &candidates, &scores, LENGTH)
            };
            std::hint::black_box(path);
        }
        start.elapsed().as_secs_f64() * 1e6 / BATCH_ITERS as f64
    };
    let (mut lazy_us, mut fused_us) = (Vec::with_capacity(SAMPLES), Vec::with_capacity(SAMPLES));
    for sample in 0..SAMPLES {
        let (lazy, fused) = if sample % 2 == 0 {
            (measure(false), measure(true))
        } else {
            let fused = measure(true);
            (measure(false), fused)
        };
        lazy_us.push(lazy);
        fused_us.push(fused);
        eprintln!("[greedy-path] sample={sample} lazy={lazy:.2}us fused={fused:.2}us");
    }
    eprintln!("[greedy-path] lazy_us={lazy_us:?} fused_us={fused_us:?}");
}

#[test]
fn selector_device_walk_matches_host_greedy_walk() {
    let (vocab, top_k, rank, hidden, len) = (32usize, 4usize, 8usize, 8usize, 6i64);
    let selector = test_selector(vocab, top_k, rank, hidden);
    let hidden =
        MxArray::random_normal(&[1, len, hidden as i64], 0.0, 1.0, Some(DType::Float32)).unwrap();
    let logits =
        MxArray::random_normal(&[1, len, vocab as i64], 0.0, 1.0, Some(DType::Float32)).unwrap();

    let (host_path, _) = selector
        .select(&hidden, &logits, 3, 0.0, false, None, &mut rand::rng())
        .expect("host select");
    let SelectorPath::Host(host_ids) = host_path else {
        panic!("host walk must return host ids");
    };
    let (device_path, sparse) = selector
        .select(&hidden, &logits, 3, 0.0, true, None, &mut rand::rng())
        .expect("device select");
    assert!(sparse.is_empty(), "greedy device walk emits no sparse rows");
    let SelectorPath::Device(ids) = device_path else {
        panic!("greedy device walk must return device ids");
    };
    let device_ids: Vec<i32> = ids.to_int32().unwrap().as_ref().to_vec();
    assert_eq!(device_ids, host_ids);
}

/// Zeroing the hidden projection zeroes the edge term, so every top-k
/// score row degenerates to the unary logits and duplicate maxima become
/// EXACT f32 ties. The host walk's `max_by` keeps the last maximum; the
/// device walk must agree (reversed argmax), or the greedy proposal path
/// would silently diverge on ties.
#[test]
fn selector_device_walk_matches_host_last_max_on_ties() {
    let (vocab, top_k, rank, hidden, len) = (16usize, 4usize, 4usize, 4usize, 4i64);
    let mut selector = test_selector(vocab, top_k, rank, hidden);
    selector.hidden_projection = LinearProj::Standard(
        Linear::from_weights(
            &MxArray::zeros(&[rank as i64, hidden as i64], Some(DType::Float32)).unwrap(),
            None,
        )
        .unwrap(),
    );
    let hidden = MxArray::zeros(&[1, len, hidden as i64], Some(DType::Float32)).unwrap();
    // Two equal maxima inside the top-4 of every row: ties at different
    // candidate slots always discriminate first- vs last-max policies.
    let row: Vec<f32> = (0..vocab)
        .map(|i| {
            if i == 5 || i == 9 {
                2.0
            } else {
                i as f32 * 0.1
            }
        })
        .collect();
    let flat = row.repeat(len as usize);
    let logits = MxArray::from_float32(&flat, &[1, len, vocab as i64]).unwrap();

    let (host_path, _) = selector
        .select(&hidden, &logits, 0, 0.0, false, None, &mut rand::rng())
        .unwrap();
    let SelectorPath::Host(host_ids) = host_path else {
        panic!("host walk must return host ids");
    };
    let (device_path, _) = selector
        .select(&hidden, &logits, 0, 0.0, true, None, &mut rand::rng())
        .unwrap();
    let SelectorPath::Device(ids) = device_path else {
        panic!("greedy device walk must return device ids");
    };
    let device_ids: Vec<i32> = ids.to_int32().unwrap().as_ref().to_vec();
    assert_eq!(device_ids, host_ids);
    assert!(
        device_ids.iter().all(|&id| id == 5 || id == 9),
        "tie rows must select one of the tied maxima, got {device_ids:?}"
    );
}

#[test]
fn fused_topk16_rejects_empty_and_undersized_shapes() -> napi::Result<()> {
    // Rejected before shader construction or evaluation, including zero
    // vocabulary (which must not reach the row-count division).
    for shape in [[1, 2, 0], [1, 0, 16], [1, 2, 15]] {
        let logits = MxArray::zeros(&shape, Some(DType::Float32))?;
        let mut ids = logits.as_raw_ptr();
        let mut values = logits.as_raw_ptr();
        assert!(!unsafe {
            mlx_sys::mlx_dflash2_topk16(logits.as_raw_ptr(), &mut ids, &mut values)
        });
        assert!(ids.is_null() && values.is_null());
    }
    Ok(())
}

/// The sharded top-16 kernel must return exactly the argpartition
/// candidate SET per row — same ids, matching logits values, ascending
/// value order. Called through the FFI directly so the test can never
/// pass vacuously on the fallback path.
#[test]
fn fused_topk16_matches_argpartition_candidate_set() {
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        eprintln!("SKIP fused topk16 parity: Metal backend unavailable");
        return;
    }
    // 248320 is the vocabulary in both the target GGUF text config and
    // its DFlash2 companion. Two rows bound each fixture below 2 MiB of
    // host logits while covering the production BF16 specialization.
    const ROWS: usize = 2;
    for (dtype, vocab) in [
        (DType::Float32, 8192_i64),
        (DType::BFloat16, 248320),
        (DType::Float16, 248320),
    ] {
        let logits = if dtype == DType::Float32 {
            MxArray::random_normal(&[1, ROWS as i64, vocab], 0.0, 1.0, Some(dtype)).unwrap()
        } else {
            let mut data = (0..ROWS * vocab as usize)
                .map(|i| -1.0 - (i % 97) as f32)
                .collect::<Vec<_>>();
            // Distinct exactly representable winners avoid BF16/F16
            // cutoff ties and exercise every scan shard.
            for row in 0..ROWS {
                for rank in 0..16usize {
                    let token = rank * (vocab as usize / 16) + row * 13 + 5;
                    data[row * vocab as usize + token] = (rank + 1) as f32;
                }
            }
            MxArray::from_float32(&data, &[1, ROWS as i64, vocab])
                .unwrap()
                .astype(dtype)
                .unwrap()
        };
        let mut ids = std::ptr::null_mut();
        let mut values = std::ptr::null_mut();
        let status =
            unsafe { mlx_sys::mlx_dflash2_topk16_test(logits.as_raw_ptr(), &mut ids, &mut values) };
        if status == 0 {
            assert!(ids.is_null() && values.is_null());
            eprintln!(
                "SKIP fused topk16 parity dtype={dtype:?} vocab={vocab}: valid fixture \
                     is unsupported by pipeline SIMD/threadgroup/shared-memory capabilities"
            );
            continue;
        }
        assert_eq!(
            status, 1,
            "fused topk16 construction/preflight failed for dtype={dtype:?} vocab={vocab}; \
                 see native error"
        );
        let ids = MxArray::from_handle(ids, "topk16 ids").unwrap();
        let values = MxArray::from_handle(values, "topk16 values").unwrap();
        let ids: Vec<i32> = ids.to_int32().unwrap().as_ref().to_vec();
        let values: Vec<f32> = values.to_float32().unwrap().as_ref().to_vec();
        let logits_f: Vec<f32> = logits
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap()
            .as_ref()
            .to_vec();

        let reference = logits
            .argpartition(-16, Some(-1))
            .unwrap()
            .slice_axis(2, vocab - 16, vocab)
            .unwrap();
        let ref_ids: Vec<i32> = reference.to_int32().unwrap().as_ref().to_vec();

        for row in 0..ROWS {
            let mine = &ids[row * 16..(row + 1) * 16];
            let mine_v = &values[row * 16..(row + 1) * 16];
            let theirs = &ref_ids[row * 16..(row + 1) * 16];
            let mut a = mine.to_vec();
            let mut b = theirs.to_vec();
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(
                a, b,
                "dtype={dtype:?} vocab={vocab} row={row}: candidate sets differ"
            );
            assert!(
                mine_v.windows(2).all(|w| w[0] <= w[1]),
                "dtype={dtype:?} vocab={vocab} row={row}: values not ascending: {mine_v:?}"
            );
            for (i, (&id, &v)) in mine.iter().zip(mine_v.iter()).enumerate() {
                assert!(
                    id >= 0 && (id as i64) < vocab,
                    "dtype={dtype:?} vocab={vocab} row={row} slot={i}: bad id {id}"
                );
                let expected = logits_f[row * vocab as usize + id as usize];
                assert_eq!(v, expected, "row {row} slot {i}: value mismatch");
            }
        }
        eprintln!(
            "PASS fused topk16 parity: native scan+merge ran for {ROWS} rows, \
                 dtype={dtype:?}, vocab={vocab}"
        );
    }
}

#[test]
fn cuda_topk16_matches_argpartition_candidate_set() {
    if !crate::accelerator::cuda_ops_available() {
        return;
    }
    const ROWS: usize = 3;
    const VOCAB: i64 = 257;
    let mut data = (0..ROWS * VOCAB as usize)
        .map(|i| -1.0 - (i % 97) as f32)
        .collect::<Vec<_>>();
    for row in 0..ROWS {
        for rank in 0..16usize {
            let token = rank * 16 + row + 1;
            data[row * VOCAB as usize + token] = (rank + 1) as f32;
        }
    }
    let logits = MxArray::from_float32(&data, &[1, ROWS as i64, VOCAB])
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap();
    let mut ids = std::ptr::null_mut();
    let mut values = std::ptr::null_mut();
    let status =
        unsafe { mlx_sys::mlx_dflash2_topk16_cuda(logits.as_raw_ptr(), &mut ids, &mut values) };
    assert_eq!(status, 1);
    let ids = MxArray::from_handle(ids, "CUDA topk16 ids").unwrap();
    let values = MxArray::from_handle(values, "CUDA topk16 values").unwrap();
    let ids: Vec<i32> = ids.to_int32().unwrap().as_ref().to_vec();
    let values: Vec<f32> = values.to_float32().unwrap().as_ref().to_vec();
    let reference = logits
        .argpartition(-16, Some(-1))
        .unwrap()
        .slice_axis(2, VOCAB - 16, VOCAB)
        .unwrap();
    let ref_ids: Vec<i32> = reference.to_int32().unwrap().as_ref().to_vec();
    let logits_f: Vec<f32> = logits
        .astype(DType::Float32)
        .unwrap()
        .to_float32()
        .unwrap()
        .as_ref()
        .to_vec();
    for row in 0..ROWS {
        let mine = &ids[row * 16..(row + 1) * 16];
        let mine_v = &values[row * 16..(row + 1) * 16];
        let theirs = &ref_ids[row * 16..(row + 1) * 16];
        let mut a = mine.to_vec();
        let mut b = theirs.to_vec();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "row {row}: candidate sets differ");
        assert!(mine_v.windows(2).all(|w| w[0] <= w[1]));
        for (&id, &value) in mine.iter().zip(mine_v.iter()) {
            assert!(id >= 0 && i64::from(id) < VOCAB);
            assert_eq!(value, logits_f[row * VOCAB as usize + id as usize]);
        }
    }
}

/// `record_only` advances logical length and provenance identically to
/// `append` — just without the (dead) K/V write. The `logical_len ==
/// token_history.len()` invariant is what `append` validates on the next
/// retained segment.
#[test]
fn record_only_advances_context_without_kv() {
    let config = DFlash2Config {
        block_size: 7,
        mask_token_id: 0,
        target_layers: vec![0],
        target_num_layers: 1,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        vocab_size: 32,
        rms_norm_eps: 1e-5,
        max_position_embeddings: 128,
        sliding_window: 8,
        conv_group_size: 1,
        conv_kernel_size: 2,
        selector_rank: 4,
        selector_top_k: 4,
        rope_theta: 10_000.0,
        final_logit_softcapping: None,
        valid_vocab_size: None,
        capture_final_norm: None,
        embedding_normalization: None,
    };
    let mut context = DFlash2ContextCache::new(&config);
    context.record_only(&[10, 11, 12]).unwrap();
    context.record_only(&[13, 14]).unwrap();
    assert_eq!(context.logical_len(), 5);
    assert_eq!(context.token_history(), &[10, 11, 12, 13, 14]);
}

#[test]
fn block_has_one_anchor_and_mask_rows() {
    assert_eq!(
        parallel_query_ids(7, 248_070, 3),
        vec![7, 248_070, 248_070, 248_070]
    );
}

#[test]
fn selector_softmax_is_normalized_and_temperature_scaled() {
    let probs = normalized_selector_probs(&[0.0, 1.0, 2.0], 0.5).expect("selector probs");
    assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!(probs[2] > probs[1] && probs[1] > probs[0]);
}

#[test]
fn fused_convolve_matches_elementwise_chain() {
    use super::GroupedDynamicCausalConv;
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        eprintln!("skipping: fused conv requires a Metal back-end");
        return;
    }
    let (hidden_size, group_size, kernel_size) = (64usize, 16i64, 2usize);
    let groups = hidden_size / group_size as usize;
    let conv = GroupedDynamicCausalConv {
        base_kernel: MxArray::random_normal(
            &[2, kernel_size as i64, hidden_size as i64],
            0.0,
            0.5,
            Some(DType::BFloat16),
        )
        .expect("base"),
        kernel_projection: LinearProj::Standard(
            Linear::from_weights(
                &MxArray::zeros(
                    &[2 * kernel_size as i64 * groups as i64, hidden_size as i64],
                    Some(DType::BFloat16),
                )
                .expect("proj weight"),
                None,
            )
            .expect("proj"),
        ),
        kernel_size,
        group_size: group_size as usize,
    };
    for length in [1i64, 5, 33] {
        let hidden = MxArray::random_normal(
            &[1, length, hidden_size as i64],
            0.0,
            1.0,
            Some(DType::BFloat16),
        )
        .expect("hidden");
        let dynamic = MxArray::random_normal(
            &[1, length, 2, kernel_size as i64, groups as i64],
            0.0,
            0.5,
            Some(DType::BFloat16),
        )
        .expect("dynamic");
        for side in [0usize, 1] {
            let fused = conv
                .fused_convolve(&hidden, &dynamic, side)
                .expect("fused conv must accept well-formed bf16 inputs");
            let reference = conv
                .convolve_elementwise(&hidden, &dynamic, side)
                .expect("reference conv");
            let max_abs = fused
                .sub(&reference)
                .and_then(|d| d.abs())
                .and_then(|d| d.astype(DType::Float32))
                .and_then(|d| d.reshape(&[-1]))
                .and_then(|d| d.max(Some(&[0]), Some(false)))
                .expect("diff");
            max_abs.eval();
            assert_eq!(
                max_abs.item_at_float32(0).expect("item"),
                0.0,
                "fused conv diverged at L={length} side={side}"
            );
        }
    }
}

#[test]
#[ignore = "requires the external 3.6 GB DFlash2 checkpoint"]
fn loads_real_qwen38_dflash2_checkpoint_strictly() {
    let path =
        std::env::var("MLX_TEST_QWEN38_DFLASH2_PATH").expect("set MLX_TEST_QWEN38_DFLASH2_PATH");
    let (model, bytes) = super::load_dflash2(std::path::Path::new(&path))
        .expect("real DFlash2 checkpoint must load");
    assert_eq!(model.config.block_size, 7);
    assert_eq!(model.config.target_layers, vec![5, 19, 33, 47, 61]);
    // Q4/group64 projections: ~1.27 GB resident versus 3.85 GB bf16
    // and ~2.17 GB at Q8/group64, plus the cross-layer k|v copy.
    assert!(
        bytes > 1_000_000_000 && bytes < 1_500_000_000,
        "resident draft bytes {bytes}"
    );
    // Every projection of the published geometry tiles (hidden 5120,
    // 32/8 heads of 128, intermediate 17408, fc K = 25600, conv N =
    // 1280) and the merges follow.
    let tiled = crate::models::quant_dispatch::kquant_tiled_enabled();
    assert_q4_group64(&model.fc, 5120, 25600, "fc");
    for layer in &model.layers {
        assert_eq!(layer.attention.qkv_proj.is_some(), tiled, "q|k|v merge");
        assert_eq!(layer.mlp.gate_up.is_some(), tiled, "gate|up merge");
        assert_q4_group64(&layer.attention.o_proj, 5120, 4096, "o_proj");
        assert_q4_group64(&layer.mlp.down_proj, 5120, 17408, "down_proj");
        assert_q4_group64(
            &layer.attention_conv.kernel_projection,
            1280,
            5120,
            "attention_conv",
        );
    }
    assert_eq!(model.context_kv.is_some(), tiled, "cross-layer k|v merge");
}

fn three_layer_tiny_draft() -> super::DFlash2Model {
    let target = super::super::config::Qwen3_5Config {
        rope_scaling: None,
        qwen35_gguf_gdn_layout: None,
        kv_format: None,
        vocab_size: 32,
        hidden_size: 64,
        num_layers: 4,
        num_heads: 4,
        num_kv_heads: 2,
        intermediate_size: 128,
        rms_norm_eps: 1e-6,
        head_dim: 16,
        tie_word_embeddings: true,
        attention_bias: false,
        max_position_embeddings: 256,
        pad_token_id: 0,
        eos_token_id: 1,
        bos_token_id: 2,
        linear_num_value_heads: 2,
        linear_num_key_heads: 2,
        linear_key_head_dim: 16,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        full_attention_interval: 2,
        partial_rotary_factor: 0.25,
        rope_theta: 10_000.0,
        paged_cache_memory_mb: None,
        paged_block_size: None,
        n_mtp_layers: 0,
    };
    unsafe { mlx_sys::mlx_seed(0xDF1A_5A11) };
    let mut model = super::tiny_dflash2_model_for_stepper_test(&target).unwrap();
    for _ in 0..2 {
        let extra = super::tiny_dflash2_model_for_stepper_test(&target).unwrap();
        model.layers.extend(extra.layers);
    }
    model.config.num_hidden_layers = model.layers.len();
    // Non-unit norm weights so a skipped or misplaced norm changes bits.
    for layer in &mut model.layers {
        for norm in [&mut layer.input_norm, &mut layer.post_attention_norm] {
            norm.set_weight(
                &MxArray::random_normal(&[64], 1.0, 0.2, Some(DType::BFloat16)).unwrap(),
            )
            .unwrap();
        }
    }
    model
        .norm
        .set_weight(&MxArray::random_normal(&[64], 1.0, 0.2, Some(DType::BFloat16)).unwrap())
        .unwrap();
    model
}

/// The draft forward as it was before residual sums moved into the next
/// norm and before the sliding mask was shared: plain adds, separate
/// norms and a mask built per layer.
fn reference_forward_hidden(
    model: &super::DFlash2Model,
    embedding: &Embedding,
    block_ids: &MxArray,
    query_base: i32,
    context: &super::DFlash2ContextCache,
) -> MxArray {
    let mut hidden = embedding.forward(block_ids).unwrap();
    for (index, layer) in model.layers.iter().enumerate() {
        let cached = context.layers[index].view();
        let live_len = cached
            .as_ref()
            .map(|(keys, _)| keys.shape_at(2).unwrap())
            .unwrap_or(0) as i32;
        let context_base = context.logical_len.saturating_sub(live_len);
        let residual = hidden.clone();
        let (prepared, dynamic) = layer
            .attention_conv
            .prepare(&layer.input_norm.forward(&hidden).unwrap())
            .unwrap();
        let mut fresh = None;
        // The pre-window path: the block's K/V concatenated after a copy
        // of the context, the three projections and the four-op
        // norm/rope chain.
        let attention = &layer.attention;
        let (batch, seq) = (prepared.shape_at(0).unwrap(), prepared.shape_at(1).unwrap());
        let queries = attention
            .q_proj
            .forward(&prepared)
            .unwrap()
            .reshape(&[batch, seq, attention.num_heads, attention.head_dim])
            .unwrap();
        let queries = attention
            .rope
            .forward(
                &attention
                    .q_norm
                    .forward(&queries)
                    .unwrap()
                    .transpose(Some(&[0, 2, 1, 3]))
                    .unwrap(),
                Some(query_base),
            )
            .unwrap();
        let (block_keys, block_values) = attention.project_context(&prepared, query_base).unwrap();
        let (keys, values, key_base) = match &cached {
            Some((context_keys, context_values)) => (
                MxArray::concatenate(context_keys, &block_keys, 2).unwrap(),
                MxArray::concatenate(context_values, &block_values, 2).unwrap(),
                context_base,
            ),
            None => (block_keys, block_values, query_base),
        };
        let attention = attention
            .attend(
                &queries,
                &keys,
                &values,
                key_base,
                query_base,
                model.config.sliding_window as i64,
                &mut fresh,
            )
            .unwrap();
        hidden = residual
            .add(&layer.attention_conv.finish(&attention, &dynamic).unwrap())
            .unwrap();
        let (prepared, dynamic) = layer
            .mlp_conv
            .prepare(&layer.post_attention_norm.forward(&hidden).unwrap())
            .unwrap();
        hidden = hidden
            .add(
                &layer
                    .mlp_conv
                    .finish(&layer.mlp.forward(&prepared, 8).unwrap(), &dynamic)
                    .unwrap(),
            )
            .unwrap();
    }
    model.norm.forward(&hidden).unwrap()
}

#[test]
#[ignore = "run with MLX_DFLASH2_ATTENTION_NAX=1 to validate the production attention hook"]
fn draft_tensor_attention_serves_fused_query_layout_and_shared_hook() {
    assert_eq!(
        std::env::var("MLX_DFLASH2_ATTENTION_NAX").as_deref(),
        Ok("1")
    );
    if !unsafe { mlx_sys::mlx_metal_is_nax_available() } {
        return;
    }
    use super::DFlash2Attention;
    use crate::nn::{Linear, RMSNorm, RoPE};
    let unused = || {
        LinearProj::Standard(
            Linear::from_weights(
                &MxArray::zeros(&[1, 1], Some(DType::BFloat16)).unwrap(),
                None,
            )
            .unwrap(),
        )
    };
    let norm = || {
        RMSNorm::from_weight(
            &MxArray::ones(&[128], Some(DType::BFloat16)).unwrap(),
            Some(1e-6),
        )
        .unwrap()
    };
    let attention = DFlash2Attention {
        tensor_attention_enabled: true,
        q_proj: unused(),
        k_proj: unused(),
        v_proj: unused(),
        o_proj: LinearProj::Standard(
            Linear::from_weights(
                &MxArray::random_normal(&[4, 4096], 0.0, 0.02, Some(DType::BFloat16)).unwrap(),
                None,
            )
            .unwrap(),
        ),
        qkv_proj: None,
        q_norm: norm(),
        k_norm: norm(),
        rope: RoPE::new(128, Some(false), Some(10_000_000.0), None),
        num_heads: 32,
        num_kv_heads: 8,
        head_dim: 128,
    };
    let packed = |heads| {
        MxArray::random_normal(&[1, 8, heads, 128], 0.0, 0.5, Some(DType::BFloat16)).unwrap()
    };
    let (queries, keys) = attention
        .fused_qk_norm_rope(&packed(32), &packed(8), 32768)
        .unwrap();
    let values = MxArray::random_normal(&[1, 8, 8, 128], 0.0, 0.5, Some(DType::BFloat16)).unwrap();
    let mut raw = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            mlx_sys::mlx_dflash2_attention(
                queries.as_raw_ptr(),
                keys.as_raw_ptr(),
                values.as_raw_ptr(),
                std::ptr::null_mut(),
                1.0f32 / 128.0f32.sqrt(),
                &mut raw,
            )
        },
        1,
        "fused QK norm/RoPE outputs must satisfy the production tensor-op layout"
    );
    let raw = MxArray::from_handle(raw, "test draft tensor attention").unwrap();
    let expected = attention
        .o_proj
        .forward(
            &raw.transpose(Some(&[0, 2, 1, 3]))
                .unwrap()
                .reshape(&[1, 8, 4096])
                .unwrap(),
        )
        .unwrap();
    let actual = attention
        .attend_masked(&queries, &keys, &values, None)
        .unwrap();
    assert_eq!(bits(&actual), bits(&expected));
}

/// A draft at the published Qwen3.8 attention geometry (hidden 5120,
/// 32/8 heads of 128, affine Q4/g64 as `draft_linear` loads it) with
/// random weights.
fn quantized_draft_model(layers: usize, intermediate: usize) -> super::DFlash2Model {
    use super::{DFlash2Attention, DFlash2Layer, DFlash2Mlp, GroupedDynamicCausalConv};
    use crate::nn::{RMSNorm, RoPE};
    let (hidden, heads, kv_heads, head_dim) = (5120usize, 32i64, 8i64, 128i64);
    let eps = 1e-6;
    let quantized = |input: usize, output: usize| -> LinearProj {
        let weight = MxArray::random_normal(
            &[output as i64, input as i64],
            0.0,
            0.02,
            Some(DType::BFloat16),
        )
        .unwrap();
        LinearProj::Quantized(super::quantize_draft_weight(&weight, "test").unwrap())
    };
    let norm = |size: i64| {
        RMSNorm::from_weight(
            &MxArray::random_normal(&[size], 1.0, 0.2, Some(DType::BFloat16)).unwrap(),
            Some(eps),
        )
        .unwrap()
    };
    let conv = || GroupedDynamicCausalConv {
        base_kernel: MxArray::random_normal(
            &[2, 2, hidden as i64],
            0.0,
            0.02,
            Some(DType::BFloat16),
        )
        .unwrap(),
        kernel_projection: quantized(hidden, 4 * (hidden / 16)),
        kernel_size: 2,
        group_size: 16,
    };
    let config = super::DFlash2Config {
        block_size: 7,
        mask_token_id: 0,
        target_layers: vec![1, 3],
        target_num_layers: 4,
        hidden_size: hidden,
        intermediate_size: intermediate,
        num_hidden_layers: layers,
        num_attention_heads: heads as usize,
        num_key_value_heads: kv_heads as usize,
        head_dim: head_dim as usize,
        vocab_size: 64,
        rms_norm_eps: eps,
        max_position_embeddings: 262_144,
        sliding_window: 2048,
        conv_group_size: 16,
        conv_kernel_size: 2,
        selector_rank: 8,
        selector_top_k: 4,
        rope_theta: 10_000_000.0,
        final_logit_softcapping: None,
        valid_vocab_size: None,
        capture_final_norm: None,
        embedding_normalization: None,
    };
    let execution_policy = super::DraftExecutionPolicy::capture();
    super::DFlash2Model {
        execution_policy,
        fc: quantized(hidden * 2, hidden),
        hidden_norm: norm(hidden as i64),
        norm: norm(hidden as i64),
        selector: test_selector(64, 4, 8, hidden),
        layers: (0..layers)
            .map(|_| DFlash2Layer {
                attention: DFlash2Attention {
                    tensor_attention_enabled: execution_policy.tensor_attention(),
                    q_proj: quantized(hidden, (heads * head_dim) as usize),
                    k_proj: quantized(hidden, (kv_heads * head_dim) as usize),
                    v_proj: quantized(hidden, (kv_heads * head_dim) as usize),
                    o_proj: quantized((heads * head_dim) as usize, hidden),
                    qkv_proj: None,
                    q_norm: norm(head_dim),
                    k_norm: norm(head_dim),
                    rope: RoPE::new(head_dim as i32, Some(false), Some(config.rope_theta), None),
                    num_heads: heads,
                    num_kv_heads: kv_heads,
                    head_dim,
                },
                mlp: DFlash2Mlp {
                    gate_proj: quantized(hidden, intermediate),
                    up_proj: quantized(hidden, intermediate),
                    down_proj: quantized(intermediate, hidden),
                    gate_up: None,
                },
                input_norm: norm(hidden as i64),
                post_attention_norm: norm(hidden as i64),
                attention_conv: conv(),
                mlp_conv: conv(),
            })
            .collect(),
        config,
        weight_bytes: 0,
        context_kv: None,
        mask_embedding: None,
        compiled: super::CompiledDraft::default(),
    }
}

fn bits(array: &MxArray) -> Vec<u16> {
    array.eval();
    array.to_uint16_native().unwrap()
}

/// The merge gate admits K-quant projections (a tiled one only at whole
/// tile widths) and declines MLX's affine route and dense projections.
#[test]
fn merge_rows_gate_follows_the_packed_contract() {
    use super::kquant_rows_merge;
    use crate::models::quantized_linear::QuantizedLinear;
    let packed = |mode: &str| {
        QuantizedLinear::new(
            MxArray::zeros(&[64, 32], Some(DType::Uint32)).unwrap(),
            MxArray::zeros(&[64, 4], Some(DType::BFloat16)).unwrap(),
            Some(MxArray::zeros(&[64, 4], Some(DType::BFloat16)).unwrap()),
            None,
            64,
            4,
            mode.to_string(),
        )
    };
    assert!(kquant_rows_merge(&packed("a4g64@t64"), &[64, 1024]));
    assert!(!kquant_rows_merge(&packed("a4g64@t64"), &[64, 96]));
    assert!(kquant_rows_merge(&packed("q4k@t64"), &[128]));
    assert!(!kquant_rows_merge(&packed("q4k"), &[64]));
    assert_eq!(
        kquant_rows_merge(&packed("affine"), &[64]),
        crate::accelerator::cuda_ops_available()
    );
    assert!(!kquant_rows_merge(&packed("mxfp4"), &[64]));
}

/// The flat window holds exactly the newest `window` appended rows in
/// order through growth, reallocation and over-window appends, and
/// `with_block` presents the window followed by the block, bit for bit
/// like a concatenation — while the block rows it parks past the window
/// never leak into later views.
/// Rows appended in chunks of 1, 3 and 8 over three wraps of an
/// 8-slot ring, each row at slot p % 8: the live rows read in row order
/// equal the newest 7 rows of the whole stream, bit for bit; a gap
/// restarts the window.
/// Mutation RG1 (an append writes row p at slot (p + 1) % 8) → red.
#[test]
fn ring_after_n_cycles_holds_the_sliding_rows() {
    let (heads, dim, slots) = (2i64, 16i64, 8i64);
    let rows = |t: i64| {
        (
            MxArray::random_normal(&[1, heads, t, dim], 0.0, 1.0, Some(DType::BFloat16)).unwrap(),
            MxArray::random_normal(&[1, t, heads, dim], 0.0, 1.0, Some(DType::BFloat16))
                .unwrap()
                .transpose(Some(&[0, 2, 1, 3]))
                .unwrap(),
        )
    };
    let mut ring = super::DraftKvWindow::new(slots, 3);
    let mut all: Option<(MxArray, MxArray)> = None;
    let mut base = 0i64;
    for (append, t) in [1i64, 3, 8, 1, 3, 1, 8, 3, 3, 1, 1, 8, 3]
        .into_iter()
        .enumerate()
    {
        let (keys, values) = rows(t);
        ring.append(base, &keys, &values).unwrap();
        base += t;
        all = Some(match all {
            Some((k, v)) => (
                MxArray::concatenate(&k, &keys, 2).unwrap(),
                MxArray::concatenate(&v, &values, 2).unwrap(),
            ),
            None => (keys, values),
        });
        let (all_keys, all_values) = all.as_ref().unwrap();
        let n = all_keys.shape_at(2).unwrap();
        let from = (n - (slots - 1)).max(0);
        assert_eq!((ring.start, ring.end), (from, n), "append {append}");
        let (got_keys, got_values) = ring.view().unwrap();
        assert_eq!(
            bits(&got_keys),
            bits(&all_keys.slice_axis(2, from, n).unwrap()),
            "append {append}: keys"
        );
        assert_eq!(
            bits(&got_values),
            bits(&all_values.slice_axis(2, from, n).unwrap()),
            "append {append}: values"
        );
    }
    assert!(base > 3 * slots, "three wraps");
    let (keys, values) = rows(2);
    ring.append(base + 5, &keys, &values).unwrap();
    assert_eq!((ring.start, ring.end), (base + 5, base + 7));
    assert_eq!(bits(&ring.view().unwrap().0), bits(&keys));
}

/// The draft over a wrapped ring (attention in slot order with the
/// per-row mask) against the sliding path (the live rows copied in
/// order): 40 one-row commits on an 8-slot ring, then blocks of one row
/// and of the block size, every block row's hidden state within
/// `assert_bits_match`'s bf16 summation tolerance.
/// Mutation DM1 (the ring mask hides for every row only what row 0
/// hides) → red.
#[test]
fn ring_attention_matches_the_sliding_window() {
    let mut model = three_layer_tiny_draft();
    model.config.sliding_window = 8;
    model.config.block_size = 7;
    let mut embedding = Embedding::new(32, 64).unwrap();
    embedding
        .load_weight(&MxArray::random_normal(&[32, 64], 0.0, 0.5, Some(DType::BFloat16)).unwrap())
        .unwrap();
    let mut context = super::DFlash2ContextCache::new(&model.config);
    for row in 0..40u32 {
        let fused = MxArray::random_normal(&[1, 1, 64], 0.0, 1.0, Some(DType::BFloat16)).unwrap();
        context
            .append(&model, &fused, row as i32, &[3 + row % 20])
            .unwrap();
    }
    assert!(context.layers[0].wrapped());
    let block_size = model.config.block_size as i32;
    for rows in [1, block_size] {
        let ids: Vec<i32> = (0..rows).map(|i| 7 + i % 5).collect();
        let block = MxArray::from_int32(&ids, &[1, rows as i64]).unwrap();
        let base = context.logical_len;
        let want = reference_forward_hidden(&model, &embedding, &block, base, &context);
        let got = model
            .forward_hidden(&embedding, &block, base, &mut context)
            .unwrap();
        got.eval();
        want.eval();
        assert_bits_match(
            &got.to_uint16_native().unwrap(),
            &want.to_uint16_native().unwrap(),
            false,
            &format!("block of {rows}"),
        );
    }
}

/// bf16 bit patterns `got` against `want`: identical when `exact`, else
/// (the M = 8 heights, where the merged width changes the tensor-op
/// split-K and so the fp32 summation order) every element within 2^-6
/// of the tensor's largest magnitude and at most 5% of them differing
/// at all — last-bit flips, not a wrong row or slice.
fn assert_bits_match(got: &[u16], want: &[u16], exact: bool, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    if exact {
        assert!(got == want, "{label}: bits differ");
        return;
    }
    let f = |b: u16| half::bf16::from_bits(b).to_f32();
    let scale = want.iter().map(|&w| f(w).abs()).fold(0f32, f32::max);
    let worst = got
        .iter()
        .zip(want)
        .map(|(&g, &w)| (f(g) - f(w)).abs())
        .fold(0f32, f32::max);
    let differing = got.iter().zip(want).filter(|(g, w)| g != w).count();
    assert!(
        worst <= scale / 64.0 && differing * 20 <= got.len(),
        "{label}: max |diff| {worst} of scale {scale}, {differing}/{} elements differ",
        got.len()
    );
}

/// The merged `q|k|v`, `gate|up` and cross-layer `k|v` matmuls, the
/// fused q/k norm + RoPE kernel and the one RoPE over all layers' keys
/// must reproduce the separate projections and the four-op chain at the
/// real head geometry. Bit for bit at 1..=7 rows (the tiled `qmv_t64` /
/// `qmv_wide_t64` kernels reduce each row the same way at any width) and
/// at a prefill-high 87 rows (the merge is bypassed: the GEMM split-K
/// follows N). At 8 rows the `qmm_m8_nax_t64` split-K count follows N
/// too (`qmm_m8_nax_splits`: k/v at N = 1024 split 8 ways, the merged
/// q|k|v at 6144 4 ways), so the merged widths round the fp32 partial
/// sums differently in a few last bits: `assert_bits_match` bounds it at
/// the merged matmuls' outputs. The MLP is bounded at `gate|up` (the
/// swiglu input to `down_proj`), not at its output: `down_proj` spreads
/// each flipped intermediate over all 5120 outputs, so the share of
/// differing outputs follows the draw (0.01%..8% over 1000 seeds) while
/// the swiglu share stays at or under 0.1%.
/// Mutation: swap the `gate|up` halves in `DFlash2Mlp::gated` -> red
/// ("mlp gate|up: bits differ" at T=1).
#[test]
fn merged_draft_projections_match_separate_bits() {
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        eprintln!("skipping: Metal only");
        return;
    }
    unsafe { mlx_sys::mlx_seed(0xDF1A_3E26) };
    let mut model = quantized_draft_model(3, 4096);
    let heights = [1i64, 2, 3, 4, 5, 6, 7, 8, 87];
    let inputs = heights
        .iter()
        .map(|&t| MxArray::random_normal(&[1, t, 5120], 0.0, 1.0, Some(DType::BFloat16)).unwrap())
        .collect::<Vec<_>>();
    let base = 1234;
    // References from the unmerged model: the four-op norm/rope chain,
    // separate matmuls, per-layer context projections.
    let mut want = Vec::new();
    for x in &inputs {
        let seq = x.shape_at(1).unwrap();
        let mut per_layer = Vec::new();
        for layer in &model.layers {
            let attention = &layer.attention;
            let queries = attention
                .q_proj
                .forward(x)
                .unwrap()
                .reshape(&[1, seq, attention.num_heads, attention.head_dim])
                .unwrap();
            let queries = attention
                .rope
                .forward(
                    &attention
                        .q_norm
                        .forward(&queries)
                        .unwrap()
                        .transpose(Some(&[0, 2, 1, 3]))
                        .unwrap(),
                    Some(base),
                )
                .unwrap();
            let (keys, values) = attention.project_context(x, base).unwrap();
            per_layer.push((
                bits(&queries),
                bits(&keys),
                bits(&values),
                bits(&layer.mlp.gated(x, 0).unwrap()),
                bits(&layer.mlp.forward(x, 0).unwrap()),
            ));
        }
        let context = model
            .project_context(x, base)
            .unwrap()
            .iter()
            .map(|(k, v)| (bits(k), bits(v)))
            .collect::<Vec<_>>();
        want.push((per_layer, context));
    }
    model.merge_projections().unwrap();
    assert!(model.context_kv.is_some(), "cross-layer k|v merge");
    let probe = |heads: i64| {
        MxArray::random_normal(&[1, 8, heads, 128], 0.0, 1.0, Some(DType::BFloat16)).unwrap()
    };
    assert!(
        model.layers[0]
            .attention
            .fused_qk_norm_rope(&probe(32), &probe(8), base)
            .is_some(),
        "the fused norm/rope kernel must serve the draft geometry"
    );
    // Bitwise equality between the merged and separate projections
    // holds only where the width change keeps the reduction order:
    // below the GEMM limit that pairing was calibrated on gen-17
    // (`qmv_t64` / `qmv_wide_t64` reduce each row identically at any
    // width; the M = 8 NAX tier is the documented exception). Other
    // GPUs pair different kernels at the merged width and differ in
    // ulps — a draw can land either side of that marginal equality,
    // as the CI VM GPU showed at T = 1. The CPU reference is
    // bit-identical on every device.
    let nax_exact = !unsafe { mlx_sys::mlx_metal_is_available() }
        || unsafe { mlx_sys::mlx_gpu_architecture_gen() } == 17;
    for (x, (per_layer, context)) in inputs.iter().zip(&want) {
        let seq = x.shape_at(1).unwrap();
        let exact = nax_exact && seq != model.merged_rows();
        for (index, layer) in model.layers.iter().enumerate() {
            assert!(layer.attention.qkv_proj.is_some(), "q|k|v merge");
            assert!(layer.mlp.gate_up.is_some(), "gate|up merge");
            let (queries, keys, values) = layer
                .attention
                .project_block(x, base, model.merged_rows())
                .unwrap();
            let (want_q, want_k, want_v, want_gated, want_mlp) = &per_layer[index];
            let label = |what: &str| format!("T={seq} layer {index}: {what}");
            assert_bits_match(&bits(&queries), want_q, exact, &label("queries"));
            assert_bits_match(&bits(&keys), want_k, exact, &label("block keys"));
            assert_bits_match(&bits(&values), want_v, exact, &label("block values"));
            let gated = layer.mlp.gated(x, model.merged_rows()).unwrap();
            assert_bits_match(&bits(&gated), want_gated, exact, &label("mlp gate|up"));
            let mlp = bits(&layer.mlp.forward(x, model.merged_rows()).unwrap());
            let want_mlp = if exact {
                want_mlp.clone()
            } else {
                bits(&layer.mlp.down_proj.forward(&gated).unwrap())
            };
            assert!(mlp == want_mlp, "{}: bits differ", label("mlp"));
        }
        let got = model.project_context(x, base).unwrap();
        assert_eq!(got.len(), context.len());
        for (index, ((k, v), (want_k, want_v))) in got.iter().zip(context).enumerate() {
            let label = |what: &str| format!("T={seq} layer {index}: {what}");
            assert_bits_match(&bits(k), want_k, exact, &label("context keys"));
            assert_bits_match(&bits(v), want_v, exact, &label("context values"));
        }
    }
}

/// A draft at the published Qwen3.8 DFlash2 geometry, merged as
/// `load_dflash2` merges it: five layers (merged context `k|v` of
/// N = 10240) fed by five 5120-wide target taps (fc `[5120, 25600]`).
fn qwen38_geometry_draft() -> super::DFlash2Model {
    let mut model = quantized_draft_model(5, 256);
    model.config.target_layers = vec![5, 19, 33, 47, 61];
    model.config.target_num_layers = 64;
    let fc = MxArray::random_normal(&[5120, 5 * 5120], 0.0, 0.02, Some(DType::BFloat16)).unwrap();
    model.fc = LinearProj::Quantized(super::quantize_draft_weight(&fc, "probe fc").unwrap());
    model.merge_projections().unwrap();
    model
}

/// Row-major bits of per-layer `[1, HK, T, D]` context K/V: row `t` is
/// the contiguous slice `[t * width, (t + 1) * width)` holding every
/// layer's K then V heads.
fn context_rows(pairs: &[(MxArray, MxArray)]) -> (Vec<u16>, usize) {
    let mut parts = Vec::new();
    for (keys, values) in pairs {
        for array in [keys, values] {
            let rows = array.shape_at(2).unwrap();
            let width = array.shape_at(1).unwrap() * array.shape_at(3).unwrap();
            parts.push(
                array
                    .transpose(Some(&[0, 2, 1, 3]))
                    .unwrap()
                    .reshape(&[rows, width])
                    .unwrap(),
            );
        }
    }
    let joined = MxArray::concatenate_many(parts.iter().collect(), Some(1)).unwrap();
    let width = joined.shape_at(1).unwrap() as usize;
    (bits(&joined), width)
}

/// Partition `widths` into classes whose shared rows are bit-identical
/// (`rows[i]` holds whole rows of `width` elements; `suffix` compares
/// the newest shared rows, else the oldest).
fn bit_classes(widths: &[i64], rows: &[Vec<u16>], suffix: bool) -> Vec<Vec<i64>> {
    let agree = |a: &[u16], b: &[u16]| {
        let m = a.len().min(b.len());
        if suffix {
            a[a.len() - m..] == b[b.len() - m..]
        } else {
            a[..m] == b[..m]
        }
    };
    let mut classes: Vec<(usize, Vec<i64>)> = Vec::new();
    for (index, &width) in widths.iter().enumerate() {
        match classes
            .iter_mut()
            .find(|(first, _)| agree(&rows[*first], &rows[index]))
        {
            Some((_, members)) => members.push(width),
            None => classes.push((index, vec![width])),
        }
    }
    classes.into_iter().map(|(_, members)| members).collect()
}

/// Whether a row's draft K/V bits depend on (i) the append width M,
/// (ii) the projection route (per-layer above `merged_rows()`, merged
/// `context_kv` at or below it, full-width eight-row prefix), (iii) the
/// row's slot in its block, (iv) the values of the other (padding) rows.
///
/// Slot and padding never matter on any GPU: every kernel reduces a row
/// alone. Width does: the kernel tier follows M (`qmv_t64` at 1,
/// `qmv_wide_t64`, the tensor-op tier, then `qmm_t_splitk`, whose split
/// count is `512 / (32 * ceil(M / 32))` clamped to whole K groups), so
/// the fc runs unsplit from M = 33 and each layer's `k|v` (N = 1024)
/// from M = 257. A cold-prefill tail append narrower than that is not
/// bit-reproducible at another width, and a rebuilt window must project
/// each segment at its recorded (route, width), padding allowed. Which
/// widths differ depends on the GPU's kernel tiers and core count, so
/// that part is asserted on the calibrated host only (gen 17 NAX, 40
/// cores) and only printed elsewhere.
#[test]
fn draft_context_kv_bits_follow_append_width_not_slot_or_padding() {
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        eprintln!("skipping: Metal only");
        return;
    }
    unsafe { mlx_sys::mlx_seed(0x51_0001) };
    let model = qwen38_geometry_draft();
    assert!(model.context_kv.is_some(), "cross-layer k|v merge");
    assert_eq!(model.merged_rows(), 8);
    let n = 2048i64;
    let taps = (0..5)
        .map(|_| MxArray::random_normal(&[1, n, 5120], 0.0, 0.5, Some(DType::BFloat16)).unwrap())
        .collect::<Vec<_>>();
    let tap_rows = |from: i64, to: i64| {
        taps.iter()
            .map(|tap| tap.slice_axis(1, from, to).unwrap())
            .collect::<Vec<_>>()
    };
    let base = 40_000i32;
    let fused_all = model.fuse_context(&taps).unwrap();
    let widths = [
        1i64, 2, 3, 7, 8, 9, 15, 16, 17, 32, 33, 64, 65, 128, 129, 224, 225, 255, 256, 257, 288,
        548, 1024, 1500, 2047, 2048,
    ];

    // (i) Width alone: fc over taps rows [0, M); projection of fixed
    // fused rows [0, M); end to end as a cold prefill tail append makes
    // it (rows [n - M, n) appended at their absolute positions).
    let mut fc = Vec::new();
    let mut projection = Vec::new();
    let mut tail = Vec::new();
    for &width in &widths {
        fc.push(bits(&model.fuse_context(&tap_rows(0, width)).unwrap()));
        projection.push(
            context_rows(
                &model
                    .project_context(&fused_all.slice_axis(1, 0, width).unwrap(), base)
                    .unwrap(),
            )
            .0,
        );
        let start = n - width;
        let mut context = DFlash2ContextCache::new(&model.config);
        context
            .record_only(&vec![0; (base as i64 + start) as usize])
            .unwrap();
        context
            .append(
                &model,
                &model.fuse_context(&tap_rows(start, n)).unwrap(),
                base + start as i32,
                &vec![1; width as usize],
            )
            .unwrap();
        let views = context
            .layers
            .iter()
            .map(|window| window.view().unwrap())
            .collect::<Vec<_>>();
        tail.push(context_rows(&views).0);
    }
    let fc_classes = bit_classes(&widths, &fc, false);
    let projection_classes = bit_classes(&widths, &projection, false);
    let tail_classes = bit_classes(&widths, &tail, true);
    eprintln!("probe (i) fc width classes: {fc_classes:?}");
    eprintln!("probe (i) projection width classes: {projection_classes:?}");
    eprintln!("probe (i) end-to-end tail width classes: {tail_classes:?}");

    // (ii) Routes at <= 8 rows: merged exact width (dense commit and a
    // short prefill tail), full-width eight-row prefix (MoE commit),
    // per-layer exact width (no merge), per-layer at 9 and 2048 rows.
    let width_index = |width: i64| widths.iter().position(|&w| w == width).unwrap();
    let (_, row) = context_rows(
        &model
            .project_context(&fused_all.slice_axis(1, 0, 1).unwrap(), base)
            .unwrap(),
    );
    let mut route_table = Vec::new();
    for keep in 1..=8i64 {
        let fused = fused_all.slice_axis(1, 0, keep).unwrap();
        let merged = context_rows(&model.project_context(&fused, base).unwrap()).0;
        let full_width = context_rows(
            &model
                .project_context_rows(&fused_all.slice_axis(1, 0, 8).unwrap(), base, Some(keep))
                .unwrap(),
        )
        .0;
        let per_layer = context_rows(
            &model
                .layers
                .iter()
                .map(|layer| layer.attention.project_context(&fused, base).unwrap())
                .collect::<Vec<_>>(),
        )
        .0;
        let rows = keep as usize * row;
        let per_layer_9 = &projection[width_index(9)][..rows];
        let per_layer_2048 = &projection[width_index(2048)][..rows];
        route_table.push((
            keep,
            merged == full_width,
            merged == per_layer,
            per_layer == per_layer_9,
            per_layer == per_layer_2048,
        ));
    }
    eprintln!(
        "probe (ii) keep: merged==full_width, merged==per_layer, per_layer==per_layer@9, per_layer==per_layer@2048: {route_table:?}"
    );

    // (iii) Slot: one row at slot p of an M-row block, its absolute
    // position held fixed (base - p), other rows unchanged.
    let slot_widths = [8i64, 16, 32, 64, 256];
    let mut slot_table = Vec::new();
    for &width in &slot_widths {
        let others_fused = fused_all.slice_axis(1, 100, 100 + width).unwrap();
        let others_taps = tap_rows(100, 100 + width);
        let x_fused = fused_all.slice_axis(1, 5, 6).unwrap();
        let x_taps = tap_rows(5, 6);
        let place = |others: &MxArray, x: &MxArray, slot: i64| {
            let mut parts = Vec::new();
            if slot > 0 {
                parts.push(others.slice_axis(1, 0, slot).unwrap());
            }
            parts.push(x.clone());
            if slot + 1 < width {
                parts.push(others.slice_axis(1, slot + 1, width).unwrap());
            }
            MxArray::concatenate_many(parts.iter().collect(), Some(1)).unwrap()
        };
        let mut projected = Vec::new();
        let mut fused_end_to_end = Vec::new();
        for slot in [0, 1, width / 2, width - 1] {
            let at = base - slot as i32;
            let pick =
                |rows: Vec<u16>| rows[slot as usize * row..(slot as usize + 1) * row].to_vec();
            projected.push(pick(
                context_rows(
                    &model
                        .project_context(&place(&others_fused, &x_fused, slot), at)
                        .unwrap(),
                )
                .0,
            ));
            let block_taps = others_taps
                .iter()
                .zip(&x_taps)
                .map(|(others, x)| place(others, x, slot))
                .collect::<Vec<_>>();
            fused_end_to_end.push(pick(
                context_rows(
                    &model
                        .project_context(&model.fuse_context(&block_taps).unwrap(), at)
                        .unwrap(),
                )
                .0,
            ));
        }
        slot_table.push((
            width,
            projected.windows(2).all(|pair| pair[0] == pair[1]),
            fused_end_to_end.windows(2).all(|pair| pair[0] == pair[1]),
        ));
    }
    eprintln!(
        "probe (iii) M: projection slot-invariant, end-to-end slot-invariant: {slot_table:?}"
    );

    // (iv) Padding: retained rows [0, k) of an M-row block, the rest
    // the real rows, zeros, or large random rows.
    let mut padding_table = Vec::new();
    for &width in &slot_widths {
        for keep in [1, width / 2, width - 1] {
            let variants = [0u8, 1, 2];
            let pad = |source: &MxArray, variant: u8| -> MxArray {
                let rows = width - keep;
                let shape = [1, rows, source.shape_at(2).unwrap()];
                let padding = match variant {
                    0 => source.slice_axis(1, keep, width).unwrap(),
                    1 => MxArray::zeros(&shape, Some(DType::BFloat16)).unwrap(),
                    _ => MxArray::random_normal(&shape, 0.0, 8.0, Some(DType::BFloat16)).unwrap(),
                };
                MxArray::concatenate(&source.slice_axis(1, 0, keep).unwrap(), &padding, 1).unwrap()
            };
            let retained = keep as usize * row;
            let mut projected = Vec::new();
            let mut fused_end_to_end = Vec::new();
            for variant in variants {
                projected.push(
                    context_rows(
                        &model
                            .project_context(
                                &pad(&fused_all.slice_axis(1, 0, width).unwrap(), variant),
                                base,
                            )
                            .unwrap(),
                    )
                    .0[..retained]
                        .to_vec(),
                );
                let padded_taps = tap_rows(0, width)
                    .iter()
                    .map(|tap| pad(tap, variant))
                    .collect::<Vec<_>>();
                fused_end_to_end.push(
                    context_rows(
                        &model
                            .project_context(&model.fuse_context(&padded_taps).unwrap(), base)
                            .unwrap(),
                    )
                    .0[..retained]
                        .to_vec(),
                );
            }
            padding_table.push((
                width,
                keep,
                projected.windows(2).all(|pair| pair[0] == pair[1]),
                fused_end_to_end.windows(2).all(|pair| pair[0] == pair[1]),
            ));
        }
    }
    eprintln!(
        "probe (iv) M, keep: projection padding-invariant, end-to-end padding-invariant: {padding_table:?}"
    );

    assert!(
        slot_table
            .iter()
            .all(|&(_, projected, fused)| projected && fused),
        "a row's draft K/V must not depend on its slot: {slot_table:?}"
    );
    assert!(
        padding_table
            .iter()
            .all(|&(_, _, projected, fused)| projected && fused),
        "a row's draft K/V must not depend on the other rows: {padding_table:?}"
    );
    let class_of = |classes: &[Vec<i64>], width: i64| {
        classes
            .iter()
            .position(|class| class.contains(&width))
            .unwrap()
    };
    let width_changes_bits = tail_classes.len() > 1;
    let per_layer_classes = widths
        .iter()
        .filter(|&&width| width > model.merged_rows())
        .map(|&width| class_of(&projection_classes, width))
        .collect::<std::collections::BTreeSet<_>>();
    let segment_key = if per_layer_classes.len() > 1 {
        "(route, width)"
    } else {
        "route class"
    };
    eprintln!(
        "probe decision: (1) a changed append width changes draft K/V bits: {}; (2) the segment map needs {segment_key}",
        if width_changes_bits { "yes" } else { "no" }
    );

    let calibrated = unsafe {
        mlx_sys::mlx_gpu_architecture_gen() == 17
            && mlx_sys::mlx_metal_is_nax_available()
            && mlx_sys::mlx_gpu_core_count() == 40
    };
    if !calibrated {
        return;
    }
    assert!(
        width_changes_bits && per_layer_classes.len() > 1,
        "the tiled kernels' reduction follows M: a width-independent draft projection \
             would let a rebuilt window ignore append widths"
    );
    assert_eq!(
        fc_classes,
        [
            vec![1],
            vec![2, 3, 7],
            vec![8, 9, 15, 16, 17, 32],
            vec![
                33, 64, 65, 128, 129, 224, 225, 255, 256, 257, 288, 548, 1024, 1500, 2047, 2048
            ],
        ]
    );
    assert_eq!(
        projection_classes,
        [
            vec![1],
            vec![2, 3, 7, 9],
            vec![8],
            vec![15, 16, 17, 32],
            vec![33, 64],
            vec![65],
            vec![128],
            vec![129, 224, 225, 255, 256],
            vec![257, 288, 548, 1024, 1500, 2047, 2048],
        ]
    );
    assert_eq!(
        tail_classes,
        [
            vec![1],
            vec![2, 3, 7],
            vec![8],
            vec![9],
            vec![15, 16, 17, 32],
            vec![33, 64],
            vec![65],
            vec![128],
            vec![129, 224, 225, 255, 256],
            vec![257, 288, 548, 1024, 1500, 2047, 2048],
        ]
    );
    // keep: merged == full-width prefix, merged == per-layer,
    // per-layer == per-layer at 9 rows, per-layer == per-layer at 2048.
    let mut want_routes = vec![(1, false, false, false, false)];
    want_routes.extend((2..=7).map(|keep| (keep, false, true, true, false)));
    want_routes.push((8, true, false, true, false));
    assert_eq!(route_table, want_routes);
}

/// Splash bounds the draft by its sliding window only: a proposal past
/// the draft's trained context (256 here) runs. Mutation: the
/// query-end check back in `propose` → red.
#[test]
fn draft_proposes_past_its_trained_context() {
    let mut model = three_layer_tiny_draft();
    model.config.sliding_window = 8;
    assert_eq!(model.config.max_position_embeddings, 256);
    let embedding = Embedding::from_weight(
        &MxArray::random_normal(&[32, 64], 0.0, 0.5, Some(DType::BFloat16)).unwrap(),
    )
    .unwrap();
    let mut context = super::DFlash2ContextCache::new(&model.config);
    context.logical_len = 300;
    for layer in &mut context.layers {
        layer.keys =
            Some(MxArray::random_normal(&[1, 2, 16, 16], 0.0, 0.5, Some(DType::BFloat16)).unwrap());
        layer.values =
            Some(MxArray::random_normal(&[1, 2, 16, 16], 0.0, 0.5, Some(DType::BFloat16)).unwrap());
        layer.start = 293;
        layer.end = 300;
    }
    model
        .propose(
            &embedding,
            None,
            &mut context,
            7,
            2,
            0.0,
            true,
            None,
            &mut rand::rng(),
        )
        .expect("a proposal past the draft's trained context");
}

#[test]
fn compiled_draft_replays_positions_windows_and_masks() {
    if !unsafe { mlx_sys::mlx_metal_is_available() } && !crate::accelerator::cuda_ops_available() {
        return;
    }
    let mut model = three_layer_tiny_draft();
    model.config.sliding_window = 8;
    model.selector.top_k = 16;
    let mut embedding = Embedding::from_weight(
        &MxArray::random_normal(&[32, 64], 0.0, 0.5, Some(DType::BFloat16)).unwrap(),
    )
    .unwrap();
    let mut previous_path = None;
    for (case, (live, capacity, position, anchor)) in [
        (3, 16, 3, 7),
        (5, 24, 100, 19),
        (7, 16, 7, 2),
        (7, 32, 211, 23),
        (3, 16, 37, 4),
    ]
    .into_iter()
    .enumerate()
    {
        if case == 4 {
            embedding
                .load_weight(
                    &MxArray::random_normal(&[32, 64], 0.0, 0.7, Some(DType::BFloat16)).unwrap(),
                )
                .unwrap();
            model.invalidate_compiled();
            assert!(!model.compiled.prepared.load(super::Ordering::Relaxed));
        }
        let mut eager = super::DFlash2ContextCache::new(&model.config);
        let mut compiled = super::DFlash2ContextCache::new(&model.config);
        eager.logical_len = position;
        compiled.logical_len = position;
        for (left, right) in eager.layers.iter_mut().zip(&mut compiled.layers) {
            for is_key in [true, false] {
                let data =
                    MxArray::random_normal(&[1, 2, capacity, 16], 0.0, 0.5, Some(DType::BFloat16))
                        .unwrap();
                let copy = data.deep_copy().unwrap();
                MxArray::eval_arrays(&[&data, &copy]).unwrap();
                if is_key {
                    left.keys = Some(data);
                    right.keys = Some(copy);
                } else {
                    left.values = Some(data);
                    right.values = Some(copy);
                }
            }
            left.start = i64::from(position) - live;
            left.end = i64::from(position);
            right.start = i64::from(position) - live;
            right.end = i64::from(position);
        }
        let mut confidence = None;
        let (expected, _) = model
            .propose(
                &embedding,
                None,
                &mut eager,
                anchor,
                2,
                0.0,
                true,
                Some(&mut confidence),
                &mut rand::rng(),
            )
            .unwrap();
        let super::SelectorPath::Device(expected) = expected else {
            panic!("host selector");
        };
        // Force the eager run first: captured weights are materialized,
        // exactly as the first proposal does for the production opt-in.
        MxArray::eval_arrays(&[&expected]).unwrap();
        let ids = MxArray::from_int32(&[anchor as i32, 0, 0], &[1, 3]).unwrap();
        let (path, probability) = model
            .propose_compiled(&embedding, None, None, &mut compiled, &ids, true)
            .unwrap()
            .unwrap();
        MxArray::eval_arrays(&[&path]).unwrap();
        assert_eq!(
            path.to_int32().unwrap().as_ref(),
            expected.to_int32().unwrap().as_ref(),
            "case {case}"
        );
        assert_eq!(
            probability.unwrap().to_float32().unwrap().as_ref(),
            confidence.unwrap().to_float32().unwrap().as_ref(),
            "confidence case {case}"
        );
        for (left, right) in eager.layers.iter().zip(&compiled.layers) {
            assert_eq!(
                bits(right.keys.as_ref().unwrap()),
                bits(left.keys.as_ref().unwrap()),
                "K case {case}"
            );
            assert_eq!(
                bits(right.values.as_ref().unwrap()),
                bits(left.values.as_ref().unwrap()),
                "V case {case}"
            );
            assert_eq!(
                (right.start, right.end),
                (i64::from(position) - live, i64::from(position))
            );
        }
        previous_path = Some(path);
    }
    drop(previous_path);
    model
        .compiled
        .disabled
        .store(true, super::Ordering::Relaxed);
    model.invalidate_compiled();
    assert!(model.compiled.disabled.load(super::Ordering::Relaxed));
    assert_eq!(
        model.compiled.traces.load(super::Ordering::Relaxed),
        3,
        "two mask modes plus one invalidated target weight variant"
    );
}

#[test]
fn draft_window_tail_rejects_unrelated_view_before_any_write() {
    if !unsafe { mlx_sys::mlx_metal_is_available() } && !crate::accelerator::cuda_ops_available() {
        return;
    }
    let make = |rows| MxArray::zeros(&[1, 2, rows, 16], Some(DType::BFloat16)).unwrap();
    let keys = make(16);
    let values = make(16);
    let unrelated = make(6);
    let values_view = values.slice_axis(2, 2, 8).unwrap();
    let block = MxArray::ones(&[1, 2, 3, 16], Some(DType::BFloat16)).unwrap();
    let outputs = super::store_window_tail(
        &[keys.clone(), values.clone(), unrelated, values_view],
        &block,
        &block,
    )
    .unwrap();
    let error = MxArray::eval_arrays(&outputs.iter().collect::<Vec<_>>()).unwrap_err();
    assert!(
        error.to_string().contains("invalid window alias"),
        "{error}"
    );
    assert!(bits(&keys).iter().all(|&x| x == 0));
    assert!(bits(&values).iter().all(|&x| x == 0));
}

#[test]
fn draft_window_tail_rejects_cross_aliased_blocks_before_any_write() {
    if !unsafe { mlx_sys::mlx_metal_is_available() } && !crate::accelerator::cuda_ops_available() {
        return;
    }
    let make = |rows| MxArray::zeros(&[1, 2, rows, 16], Some(DType::BFloat16)).unwrap();
    for key_block_aliases_values in [true, false] {
        let keys = make(16);
        let values = make(16);
        let keys_view = keys.slice_axis(2, 0, 8).unwrap();
        let values_view = values.slice_axis(2, 0, 8).unwrap();
        let block = MxArray::ones(&[1, 2, 3, 16], Some(DType::BFloat16)).unwrap();
        let (key_block, value_block) = if key_block_aliases_values {
            (values.slice_axis(2, 5, 8).unwrap(), block)
        } else {
            (block, keys.slice_axis(2, 5, 8).unwrap())
        };
        let outputs = super::store_window_tail(
            &[keys.clone(), values.clone(), keys_view, values_view],
            &key_block,
            &value_block,
        )
        .unwrap();
        let error = MxArray::eval_arrays(&outputs.iter().collect::<Vec<_>>()).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("K/V geometry or ownership differs"),
            "key_block_aliases_values={key_block_aliases_values}: {error}"
        );
        assert!(bits(&keys).iter().all(|&x| x == 0));
        assert!(bits(&values).iter().all(|&x| x == 0));
    }
}

#[test]
fn draft_window_tail_preserves_same_buffer_prefix_views() {
    if !unsafe { mlx_sys::mlx_metal_is_available() } && !crate::accelerator::cuda_ops_available() {
        return;
    }
    let make = |rows| MxArray::zeros(&[1, 2, rows, 16], Some(DType::BFloat16)).unwrap();
    let keys = make(16);
    let values = make(16);
    let keys_view = keys.slice_axis(2, 0, 8).unwrap();
    let values_view = values.slice_axis(2, 0, 8).unwrap();
    let blocks = MxArray::ones(&[1, 2, 6, 16], Some(DType::BFloat16)).unwrap();
    let key_block = blocks.slice_axis(2, 0, 3).unwrap();
    let value_block = blocks.slice_axis(2, 3, 6).unwrap();
    let outputs = super::store_window_tail(
        &[keys.clone(), values.clone(), keys_view, values_view],
        &key_block,
        &value_block,
    )
    .unwrap();
    MxArray::eval_arrays(&outputs.iter().collect::<Vec<_>>()).unwrap();
    let mut expected = vec![0_u16; 2 * 16 * 16];
    for head in 0..2 {
        for row in 5..8 {
            let at = (head * 16 + row) * 16;
            expected[at..at + 16].fill(0x3f80);
        }
    }
    assert_eq!(bits(&keys), expected);
    assert_eq!(bits(&values), expected);
}

/// `forward_hidden` fuses each residual sum into the next norm and shares
/// one sliding mask across layers; with and without a mask (context
/// shorter and longer than the window) it must equal the reference bits.
#[test]
fn forward_hidden_fused_residuals_and_shared_mask_are_bit_identical() {
    let model = three_layer_tiny_draft();
    let mut embedding = Embedding::new(32, 64).unwrap();
    embedding
        .load_weight(&MxArray::random_normal(&[32, 64], 0.0, 0.5, Some(DType::BFloat16)).unwrap())
        .unwrap();
    let mut cases = 0;
    for context_rows in [3usize, 12, 30] {
        let mut context = super::DFlash2ContextCache::new(&model.config);
        let fused = MxArray::random_normal(
            &[1, context_rows as i64, 64],
            0.0,
            1.0,
            Some(DType::BFloat16),
        )
        .unwrap();
        let tokens: Vec<u32> = (0..context_rows as u32).map(|t| 3 + t % 20).collect();
        context.append(&model, &fused, 0, &tokens).unwrap();
        let base = context.logical_len;
        let block = MxArray::from_int32(&[7, 0, 0], &[1, 3]).unwrap();
        let got = model
            .forward_hidden(&embedding, &block, base, &mut context)
            .unwrap();
        let want = reference_forward_hidden(&model, &embedding, &block, base, &context);
        got.eval();
        want.eval();
        assert_eq!(
            got.to_uint16_native().unwrap(),
            want.to_uint16_native().unwrap(),
            "context_rows={context_rows}"
        );
        cases += 1;
    }
    assert_eq!(cases, 3);
}

#[test]
fn propose_tree_lattice_matches_selector_and_handles_16_row_context_append() {
    let mut model = three_layer_tiny_draft();
    model.config.sliding_window = 64;
    model.config.block_size = 7;
    model.selector.top_k = 16;
    let mut embedding = Embedding::new(32, 64).unwrap();
    embedding
        .load_weight(&MxArray::random_normal(&[32, 64], 0.0, 0.5, Some(DType::BFloat16)).unwrap())
        .unwrap();

    let mut ctx_lattice = super::DFlash2ContextCache::new(&model.config);
    let mut ctx_propose = super::DFlash2ContextCache::new(&model.config);

    // Verify `DFlash2ContextCache::append` handles `keep = 16` cleanly.
    let fused_16 = MxArray::random_normal(&[1, 16, 64], 0.0, 1.0, Some(DType::BFloat16)).unwrap();
    let tokens_16: Vec<u32> = (0..16).map(|t| 1 + (t % 20)).collect();
    ctx_lattice
        .append(&model, &fused_16, 0, &tokens_16)
        .unwrap();
    ctx_propose
        .append(&model, &fused_16, 0, &tokens_16)
        .unwrap();
    assert_eq!(ctx_lattice.logical_len(), 16);

    // max_len == 0 returns an empty lattice.
    let empty = model
        .propose_tree_lattice(&embedding, None, &mut ctx_lattice, 5, 0)
        .unwrap();
    assert_eq!(empty.slots, 0);

    // Full-width lattice proposal matches `propose(...)` greedy Markov path.
    let lattice = model
        .propose_tree_lattice(&embedding, None, &mut ctx_lattice, 5, 7)
        .unwrap();
    assert_eq!(lattice.slots, 7);
    assert_eq!(lattice.top_k, 16);
    let (path, _) = model
        .propose(
            &embedding,
            None,
            &mut ctx_propose,
            5,
            7,
            0.0,
            false,
            None,
            &mut rand::rng(),
        )
        .unwrap();
    let super::SelectorPath::Host(expected_path) = path else {
        panic!("expected Host selector path");
    };
    let lattice_top_path: Vec<i32> = (0..7)
        .map(|pos| lattice.token(pos, lattice.top_candidate[pos]) as i32)
        .collect();
    assert_eq!(
        lattice_top_path, expected_path,
        "DraftLattice greedy Markov chain must match CandidateSelector::select_with_anchor"
    );
}

#[test]
fn propose_tree_lattice_embedded_compiled_matches_eager() {
    use crate::models::quantized_linear::QuantizedLinear;
    if !crate::accelerator::cuda_ops_available() {
        return;
    }
    struct EmbeddedQLinearView<'a> {
        embedding: &'a Embedding,
        lm_head: &'a QuantizedLinear,
    }
    impl super::DFlash2TargetView for EmbeddedQLinearView<'_> {
        fn embed(&mut self, block: &MxArray, _ids: &[i32]) -> napi::Result<MxArray> {
            self.embedding.forward(block)
        }
        fn logits(&mut self, hidden: &MxArray) -> napi::Result<MxArray> {
            self.lm_head.forward(hidden)
        }
        fn traced(&self) -> Option<(&Embedding, Option<&LinearProj>)> {
            None
        }
        fn target_lm_head_qlinear(&self) -> Option<&QuantizedLinear> {
            Some(self.lm_head)
        }
    }

    let target = crate::models::qwen3_5::config::Qwen3_5Config {
        rope_scaling: None,
        qwen35_gguf_gdn_layout: None,
        kv_format: None,
        vocab_size: 64,
        hidden_size: 256,
        num_layers: 4,
        num_heads: 4,
        num_kv_heads: 2,
        intermediate_size: 512,
        rms_norm_eps: 1e-6,
        head_dim: 128,
        tie_word_embeddings: false,
        attention_bias: false,
        max_position_embeddings: 512,
        pad_token_id: 0,
        eos_token_id: 2,
        bos_token_id: 1,
        linear_num_value_heads: 2,
        linear_num_key_heads: 2,
        linear_key_head_dim: 8,
        linear_value_head_dim: 8,
        linear_conv_kernel_dim: 4,
        full_attention_interval: 4,
        partial_rotary_factor: 0.25,
        rope_theta: 10_000.0,
        paged_cache_memory_mb: None,
        paged_block_size: None,
        n_mtp_layers: 0,
    };
    let mut model = super::tiny_dflash2_model_with_block(&target, 7).unwrap();
    model.config.sliding_window = 64;
    model.selector.top_k = 16;
    model.configure_dense_execution();

    let mut embedding = Embedding::new(64, 256).unwrap();
    embedding
        .load_weight(&MxArray::random_normal(&[64, 256], 0.0, 0.5, Some(DType::BFloat16)).unwrap())
        .unwrap();
    let dense_head = MxArray::random_normal(&[64, 256], 0.0, 0.2, Some(DType::BFloat16)).unwrap();
    let mut qw = std::ptr::null_mut();
    let mut qs = std::ptr::null_mut();
    let mut qb = std::ptr::null_mut();
    let mode = std::ffi::CString::new("affine").unwrap();
    assert!(unsafe {
        mlx_sys::mlx_quantize(
            dense_head.as_raw_ptr(),
            32,
            8,
            mode.as_ptr(),
            &mut qw,
            &mut qs,
            &mut qb,
        )
    });
    let lm_head = QuantizedLinear::new(
        MxArray::from_handle(qw, "qw").unwrap(),
        MxArray::from_handle(qs, "qs").unwrap(),
        Some(MxArray::from_handle(qb, "qb").unwrap()),
        None,
        32,
        8,
        "affine".to_string(),
    );
    lm_head.eval_packed_arrays("test_embedded_lm_head").unwrap();

    let fused = MxArray::random_normal(&[1, 16, 256], 0.0, 1.0, Some(DType::BFloat16)).unwrap();
    let tokens: Vec<u32> = (0..16).map(|t| 1 + (t % 20)).collect();
    let mut ctx_compiled = super::DFlash2ContextCache::new(&model.config);
    let mut ctx_eager = super::DFlash2ContextCache::new(&model.config);
    ctx_compiled.append(&model, &fused, 0, &tokens).unwrap();
    ctx_eager.append(&model, &fused, 0, &tokens).unwrap();
    ctx_compiled.eval().unwrap();
    ctx_eager.eval().unwrap();

    let mut view = EmbeddedQLinearView {
        embedding: &embedding,
        lm_head: &lm_head,
    };
    // Cycle 0: warms up `compiled.prepared = true`.
    let (c0, s0) = model
        .propose_tree_lattice_arrays_embedded(&mut view, &mut ctx_compiled, 5, 7)
        .unwrap();
    MxArray::eval_arrays(&[&c0, &s0]).unwrap();
    assert!(model.compiled.prepared.load(super::Ordering::Relaxed));

    // Cycle 1 & 2: compiled replay must match eager bit-for-bit and trace once!
    for anchor in [7u32, 11u32] {
        let (cc, cs) = model
            .propose_tree_lattice_arrays_embedded(&mut view, &mut ctx_compiled, anchor, 7)
            .unwrap();
        MxArray::eval_arrays(&[&cc, &cs]).unwrap();

        model
            .compiled
            .disabled
            .store(true, super::Ordering::Relaxed);
        let (ec, es) = model
            .propose_tree_lattice_arrays_embedded(&mut view, &mut ctx_eager, anchor, 7)
            .unwrap();
        MxArray::eval_arrays(&[&ec, &es]).unwrap();
        model
            .compiled
            .disabled
            .store(false, super::Ordering::Relaxed);

        assert_eq!(
            cc.to_int32().unwrap().as_ref(),
            ec.to_int32().unwrap().as_ref(),
            "compiled embedded lattice candidates mismatch for anchor={anchor}"
        );
        assert_eq!(
            cs.to_float32().unwrap().as_ref(),
            es.to_float32().unwrap().as_ref(),
            "compiled embedded lattice scores mismatch for anchor={anchor}"
        );
    }
    assert_eq!(
        model.compiled.traces.load(super::Ordering::Relaxed),
        1,
        "fixed-width compiled embedded lattice must trace once and replay without retracing"
    );
}

mod adaptive_budget_tests {
    use mlx_sys as sys;
    use napi::bindgen_prelude::*;

    use crate::array::MxArray;

    #[test]
    fn confidence_preserves_greedy_path_and_drives_lazy_budget() -> Result<()> {
        if !unsafe { sys::mlx_metal_is_available() } {
            return Ok(());
        }
        let candidates = MxArray::from_int32(&(0..7 * 16).collect::<Vec<_>>(), &[1, 7, 16])?;
        let scores = (0..7 * 16 * 16)
            .map(|i| ((i * 17 % 83) as f32 - 41.0) * 0.07)
            .collect::<Vec<_>>();
        let table = MxArray::from_float32(&scores, &[7, 16, 16])?;
        let mut old = std::ptr::null_mut();
        let mut path = std::ptr::null_mut();
        let mut confidence = std::ptr::null_mut();
        assert!(unsafe {
            sys::mlx_dflash2_greedy_path(candidates.as_raw_ptr(), table.as_raw_ptr(), &mut old)
        });
        assert!(unsafe {
            sys::mlx_dflash2_greedy_path_confidence(
                candidates.as_raw_ptr(),
                table.as_raw_ptr(),
                &mut path,
                &mut confidence,
            )
        });
        let old = MxArray::from_handle(old, "old")?;
        let path = MxArray::from_handle(path, "path")?;
        let confidence = MxArray::from_handle(confidence, "confidence")?;
        let params =
            crate::models::qwen3_5::verify_budget::VerifyBudget::default().parameters(false, true);
        let mut values = vec![params.acceptance.slope];
        values.extend(params.acceptance.intercepts);
        values.extend(params.costs);
        values.push(0.0);
        let parameters = MxArray::from_float32(&values, &[18])?;
        let budget = MxArray::from_handle(
            unsafe {
                sys::mlx_dflash2_verify_budget(confidence.as_raw_ptr(), parameters.as_raw_ptr())
            },
            "budget",
        )?;
        // One root evaluation traverses selector -> confidence -> budget.
        MxArray::eval_arrays(&[&old, &path, &budget])?;
        assert_eq!(old.to_int32()?.as_ref(), path.to_int32()?.as_ref());
        let confidence_values = confidence.to_float32()?;
        let selected = path.to_int32()?;
        let mut predecessor = 0;
        for position in 0..7 {
            let row =
                &scores[(position * 16 + predecessor) * 16..(position * 16 + predecessor + 1) * 16];
            let chosen = selected[position] as usize % 16;
            let expected = 1.0 / row.iter().map(|v| (*v - row[chosen]).exp()).sum::<f32>();
            assert!((confidence_values[position] - expected).abs() < 2e-6);
            predecessor = chosen;
        }
        let mut expected = 1.0f32;
        let mut survival = 1.0f32;
        let mut best = expected / params.costs[1];
        let mut rows = 1;
        for position in 0..7 {
            let q = confidence_values[position].clamp(1e-6, 1.0 - 1e-6);
            let z = params.acceptance.slope * (q / (1.0 - q)).ln()
                + params.acceptance.intercepts[position];
            survival *= 1.0 / (1.0 + (-z).exp());
            expected += survival;
            let value = expected / params.costs[position + 2];
            if value > best {
                best = value;
                rows = position + 2;
            }
        }
        assert_eq!(budget.item_at_int32(0)?, rows as i32);
        values[17] = 1.0;
        let parameters = MxArray::from_float32(&values, &[18])?;
        let all = MxArray::from_handle(
            unsafe {
                sys::mlx_dflash2_verify_budget(confidence.as_raw_ptr(), parameters.as_raw_ptr())
            },
            "all",
        )?;
        assert_eq!(all.item_at_int32(0)?, 8);
        Ok(())
    }
}
