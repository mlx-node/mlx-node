use super::*;
fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp")
}

#[test]
fn singleton_decoder_matches_mlx_vlm_across_sparse_attention_and_eos() {
    let path = fixture();
    let raw: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("config.json")).unwrap()).unwrap();
    let config = config::Config::parse(&raw).unwrap();
    let oracle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("oracle.json")).unwrap()).unwrap();
    let store = weights::Store::open(&path).unwrap();
    assert_eq!(store.bytes_read, 0);
    let mut model = decoder::Decoder::new(config, store).unwrap();
    for (i, t) in oracle["tokens"].as_array().unwrap().iter().enumerate() {
        let actual = model
            .step(t.as_u64().unwrap() as u32)
            .unwrap()
            .to_float32()
            .unwrap();
        let expected = oracle["logits"][i].as_array().unwrap();
        let diff = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b.as_f64().unwrap() as f32).abs())
            .fold(0f32, f32::max);
        assert!(diff < 0.0001, "token {i}: max absolute logit error {diff}");
    }
    assert!(model.weights.peak_cache_bytes <= weights::CACHE_BYTES);
    model.reset();
    assert!(model.history.is_empty());
    let first = model.step(3).unwrap().to_float32().unwrap();
    let expected = oracle["logits"][0].as_array().unwrap();
    assert!(
        first
            .iter()
            .zip(expected)
            .all(|(a, b)| (a - b.as_f64().unwrap() as f32).abs() < 0.0001)
    );
}

#[test]
fn interrupted_token_discards_partial_layer_state_and_can_restart() {
    let path = fixture();
    let raw = serde_json::from_slice(&std::fs::read(path.join("config.json")).unwrap()).unwrap();
    let mut model = decoder::Decoder::new(
        config::Config::parse(&raw).unwrap(),
        weights::Store::open(&path).unwrap(),
    )
    .unwrap();
    let expected = model.step(3).unwrap().to_float32().unwrap();
    // Fail after layer 0 has already advanced, with a populated prior prefix.
    let key = "layers.1.attn_hyper_connection.hc_norm.weight";
    let removed = model.weights.tensors.remove(key).expect("fixture norm");
    // A shape-independent missing descriptor still fails even if weights were cached.
    assert!(model.step(9).is_err());
    assert!(model.history.is_empty());
    model.weights.tensors.insert(key.into(), removed);
    let restarted = model.step(3).unwrap().to_float32().unwrap();
    assert_eq!(&*expected, &*restarted);

    let flag = Arc::new(AtomicBool::new(true));
    model.cancelled = Some(flag.clone());
    assert!(model.step(9).is_err());
    assert!(model.history.is_empty());
    flag.store(false, std::sync::atomic::Ordering::Relaxed);
    assert_eq!(&*model.step(3).unwrap().to_float32().unwrap(), &*expected);
}

#[test]
fn rejects_invalid_dimensions_before_model_construction() {
    let mut raw: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture().join("config.json")).unwrap()).unwrap();
    raw["text_config"]["num_experts_per_tok"] = serde_json::json!(100);
    assert!(config::Config::parse(&raw).is_err());
    raw["text_config"]["num_experts_per_tok"] = serde_json::json!(2);
    raw["text_config"]["linear_num_key_heads"] = serde_json::json!(0);
    assert!(config::Config::parse(&raw).is_err());
}

#[test]
fn rejects_out_of_range_ssd_reads() {
    let mut store = weights::Store::open(&fixture()).unwrap();
    assert!(store.read("embed_tokens.weight", 64, 1).is_err());
    assert!(store.read("embed_tokens.weight", usize::MAX, 2).is_err());
    assert_eq!(store.bytes_read, 0);
}

#[test]
fn gguf_layout_matches_safetensors_oracle() {
    let path = fixture();
    let store = weights::Store::open(&path.join("model.gguf")).unwrap();
    let config = gguf::config(&store).unwrap();
    let mut model = decoder::Decoder::new(config, store).unwrap();
    let oracle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("oracle.json")).unwrap()).unwrap();
    for (i, t) in oracle["tokens"].as_array().unwrap().iter().enumerate() {
        let actual = model
            .step(t.as_u64().unwrap() as u32)
            .unwrap()
            .to_float32()
            .unwrap();
        let expected = oracle["logits"][i].as_array().unwrap();
        let diff = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b.as_f64().unwrap() as f32).abs())
            .fold(0f32, f32::max);
        assert!(
            diff < 0.0001,
            "GGUF token {i}: max absolute logit error {diff}"
        );
    }
}

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!("mlx-qwen4-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn gguf_string(bytes: &mut Vec<u8>, s: &str) {
    bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
    bytes.extend_from_slice(s.as_bytes());
}
#[test]
fn q5_1_ssd_rows_preserve_high_bits_and_stored_minimum() {
    let dir = Temp::new();
    let path = dir.0.join("test.gguf");
    let mut b = b"GGUF".to_vec();
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    gguf_string(&mut b, "general.architecture");
    b.extend_from_slice(&8u32.to_le_bytes());
    gguf_string(&mut b, "qwen4exp");
    gguf_string(&mut b, "expert.weight");
    b.extend_from_slice(&2u32.to_le_bytes());
    b.extend_from_slice(&32u64.to_le_bytes());
    b.extend_from_slice(&2u64.to_le_bytes());
    b.extend_from_slice(&7u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    while !b.len().is_multiple_of(32) {
        b.push(0);
    }
    // Read the second row only; first row has a distinct scale/minimum.
    for (scale, min) in [(0.25f32, 1.0f32), (0.5, -2.0)] {
        b.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
        b.extend_from_slice(&half::f16::from_f32(min).to_bits().to_le_bytes());
        b.extend_from_slice(&0xffff0000u32.to_le_bytes());
        for i in 0..16 {
            b.push(i | (i << 4));
        }
    }
    std::fs::write(&path, b).unwrap();
    let mut store = weights::Store::open(&path).unwrap();
    let w = store.read("expert.weight", 1, 1).unwrap();
    assert_eq!(&*w.values.shape().unwrap(), [1, 5]);
    let a = w.dense().unwrap().to_float32().unwrap();
    for i in 0..32 {
        assert_eq!(a[i], i as f32 * 0.5 - 2.0);
    }
    assert_eq!(store.bytes_read, 24);
    let x = MxArray::ones(&[1, 32], Some(crate::array::DType::BFloat16)).unwrap();
    assert_eq!(w.linear(&x).unwrap().item_at_float32(0).unwrap(), 184.0);
    let descriptor = store.descriptor("expert.weight").unwrap().clone();
    store.tensors.insert("token_embd.weight".into(), descriptor);
    let row = store.read("token_embd.weight", 1, 1).unwrap();
    assert!(row.scales.is_none(), "retain the computed lookup row");
    assert_eq!(&*row.dense().unwrap().to_float32().unwrap(), &*a);
    let bytes = store.bytes_read;
    assert!(Arc::ptr_eq(
        &row,
        &store.read("token_embd.weight", 1, 1).unwrap()
    ));
    assert_eq!(store.bytes_read, bytes, "warm lookup must not read SSD");
}

#[test]
fn full_bank_preparation_reuses_streamed_expert_chunks_after_restart() {
    let dir = Temp::new();
    let path = dir.0.join("test.gguf");
    let mut b = b"GGUF".to_vec();
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    gguf_string(&mut b, "general.architecture");
    b.extend_from_slice(&8u32.to_le_bytes());
    gguf_string(&mut b, "qwen4exp");
    let name = "blk.0.ffn_gate_exps.weight";
    gguf_string(&mut b, name);
    b.extend_from_slice(&3u32.to_le_bytes());
    for d in [32u64, 256, 2] {
        b.extend_from_slice(&d.to_le_bytes());
    }
    b.extend_from_slice(&7u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    while !b.len().is_multiple_of(32) {
        b.push(0);
    }
    for row in 0..512 {
        b.extend_from_slice(&half::f16::from_f32(0.25).to_bits().to_le_bytes());
        b.extend_from_slice(&half::f16::from_f32(-2.0).to_bits().to_le_bytes());
        b.extend_from_slice(&(row as u32 * 7919).to_le_bytes());
        b.extend_from_slice(&[row as u8; 16]);
    }
    std::fs::write(&path, b).unwrap();
    let mut streaming = weights::Store::open_with_packed_root(&path, Some(&dir.0)).unwrap();
    let expected = streaming
        .read(name, 0, 256)
        .unwrap()
        .values
        .to_uint32()
        .unwrap()
        .to_vec();
    assert_eq!(streaming.bytes_read, 256 * 24);
    drop(streaming);
    let mut resident = weights::Store::open_with_packed_root(&path, Some(&dir.0)).unwrap();
    resident.plan.resident = true;
    resident.prepare_hot().unwrap();
    assert_eq!(
        resident.packed_hits, 1,
        "the previously prepared expert must be reused"
    );
    assert_eq!(
        resident.bytes_read,
        256 * 24,
        "only the cold expert should be imported"
    );
    assert_eq!(
        &*resident
            .read(name, 0, 256)
            .unwrap()
            .values
            .to_uint32()
            .unwrap(),
        &expected
    );
    drop(resident);
    let mut restarted = weights::Store::open_with_packed_root(&path, Some(&dir.0)).unwrap();
    restarted.plan.resident = true;
    restarted.prepare_hot().unwrap();
    assert_eq!(restarted.packed_hits, 2);
    assert_eq!(
        restarted.bytes_read, 0,
        "resident restart must not repeat imports"
    );
}

#[test]
fn chunked_prefill_preserves_singleton_logits_and_continuation() {
    for source in [fixture(), fixture().join("bf16")] {
        for path in [source.clone(), source.join("model.gguf")] {
            for width in [2, 7, 16, 32] {
                let raw =
                    serde_json::from_slice(&std::fs::read(source.join("config.json")).unwrap())
                        .unwrap();
                let config = config::Config::parse(&raw).unwrap();
                let mut single =
                    decoder::Decoder::new(config.clone(), weights::Store::open(&path).unwrap())
                        .unwrap();
                let mut chunked =
                    decoder::Decoder::new(config, weights::Store::open(&path).unwrap()).unwrap();
                chunked.batch_prefill = false;
                let tokens = [3, 9, 4, 7, 1, 5, 8, 2, 3, 6, 9, 4, 7, 1, 3, 5, 6, 8].repeat(2);
                for chunk in tokens.chunks(width) {
                    let mut want = None;
                    for &token in chunk {
                        want = Some(single.step(token).unwrap());
                    }
                    let got = chunked.prefill_chunk(chunk, None, true).unwrap();
                    assert_eq!(
                        &*got.to_float32().unwrap(),
                        &*want.unwrap().to_float32().unwrap(),
                        "{path:?}, chunk width {width}"
                    );
                    assert_eq!(chunked.history, single.history);
                }
                assert_eq!(
                    &*chunked.step(11).unwrap().to_float32().unwrap(),
                    &*single.step(11).unwrap().to_float32().unwrap()
                );
            }
        }
    }
}

#[test]
fn batched_prefill_matches_reference_across_chunks_sparse_boundaries_and_continuation() {
    for source in [
        fixture(),
        fixture().join("bf16"),
        fixture().join("bf16/paged"),
    ] {
        for path in [source.clone(), source.join("model.gguf")] {
            for width in [2, 7, 32, 64, 128] {
                let raw =
                    serde_json::from_slice(&std::fs::read(source.join("config.json")).unwrap())
                        .unwrap();
                let config = config::Config::parse(&raw).unwrap();
                let mut scalar =
                    decoder::Decoder::new(config.clone(), weights::Store::open(&path).unwrap())
                        .unwrap();
                let mut batched =
                    decoder::Decoder::new(config, weights::Store::open(&path).unwrap()).unwrap();
                scalar.batch_prefill = false;
                if width >= 64 {
                    batched.weights.plan.resident = true;
                    batched.weights.prepare_hot().unwrap();
                } else if width == 32 {
                    // Assembled fixed projections with demand-loaded experts
                    // must preserve the same prompt/continuation contract.
                    batched.weights.plan.resident = false;
                    batched.weights.plan.policy = "auto".into();
                    batched.weights.prepare_hot().unwrap();
                }
                let tokens = [3, 9, 4, 7, 1, 5, 8, 2, 3, 6, 9, 4, 7, 1, 3, 5, 6, 8]
                    .repeat(if width >= 64 { 7 } else { 2 });
                let tolerance = if source == fixture() { 0.0001 } else { 0.04 };
                for chunk in tokens.chunks(width) {
                    let want = scalar.prefill_chunk(chunk, None, true).unwrap();
                    let got = batched.prefill_chunk(chunk, None, true).unwrap();
                    let a = want.to_float32().unwrap();
                    let b = got.to_float32().unwrap();
                    let error = a
                        .iter()
                        .zip(b.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    assert!(error < tolerance, "{path:?}, width={width}, error={error}");
                    assert_eq!(scalar.history, batched.history);
                }
                let a = scalar.step(11).unwrap().to_float32().unwrap();
                let b = batched.step(11).unwrap().to_float32().unwrap();
                let error = a
                    .iter()
                    .zip(b.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    error < tolerance,
                    "continuation {path:?}, width={width}, error={error}"
                );
            }
        }
    }
}

#[test]
fn contiguous_prompt_windows_preserve_rows_frontiers_and_continuation() {
    for source in [
        fixture(),
        fixture().join("bf16"),
        fixture().join("bf16/paged"),
    ] {
        for path in [source.clone(), source.join("model.gguf")] {
            for width in [2, 7, 32, 128] {
                let raw =
                    serde_json::from_slice(&std::fs::read(source.join("config.json")).unwrap())
                        .unwrap();
                let config = config::Config::parse(&raw).unwrap();
                let mut rows =
                    decoder::Decoder::new(config.clone(), weights::Store::open(&path).unwrap())
                        .unwrap();
                let mut window =
                    decoder::Decoder::new(config, weights::Store::open(&path).unwrap()).unwrap();
                rows.window_carry = false;
                rows.async_prefill = false;
                window.async_prefill = true;
                let tokens = [3, 9, 4, 7, 1, 5, 8, 2, 3, 6, 9, 4, 7, 1, 3, 5, 6, 8].repeat(7);
                for chunk in tokens.chunks(width) {
                    let expected = rows.prefill_chunk(chunk, None, true).unwrap();
                    let actual = window.prefill_chunk(chunk, None, true).unwrap();
                    assert_eq!(
                        &*expected.to_float32().unwrap(),
                        &*actual.to_float32().unwrap(),
                        "{path:?}, width={width}"
                    );
                    assert_eq!(rows.history, window.history);
                    assert_eq!(rows.last_chunk_hidden.len(), window.last_chunk_hidden.len());
                    for (a, b) in rows.last_chunk_hidden.iter().zip(&window.last_chunk_hidden) {
                        assert_eq!(&*a.to_float32().unwrap(), &*b.to_float32().unwrap());
                    }
                }
                assert_eq!(
                    &*rows.step(11).unwrap().to_float32().unwrap(),
                    &*window.step(11).unwrap().to_float32().unwrap()
                );
            }
        }
    }
}

#[test]
fn batched_rotary_preserves_split_indexer_blocks_and_owner_changes() {
    let path = fixture().join("bf16");
    let raw = serde_json::from_slice(&std::fs::read(path.join("config.json")).unwrap()).unwrap();
    let mut config = config::Config::parse(&raw).unwrap();
    // Cross the dense-to-sparse attention boundary after forming batched blocks.
    config.indexer_budget = 32;
    let mut reference =
        decoder::Decoder::new(config.clone(), weights::Store::open(&path).unwrap()).unwrap();
    reference.batch_rotary = false;
    let mut candidate =
        decoder::Decoder::new(config, weights::Store::open(&path).unwrap()).unwrap();
    for owner in 0..2 {
        reference.reset();
        candidate.reset();
        let positions: Vec<_> = (0..96)
            .map(|p| [p + owner * 100, p / 3 + owner * 10, p / 5 + 3])
            .collect();
        reference.positions = positions.clone();
        candidate.positions = positions;
        let tokens = [3, 9, 4, 7, 1, 5, 8, 2].repeat(12);
        let mut offset = 0;
        for count in [3, 7, 2, 17, 3, 4, 28, 32] {
            let chunk = &tokens[offset..offset + count];
            let expected = reference.prefill_chunk(chunk, None, true).unwrap();
            let got = candidate.prefill_chunk(chunk, None, true).unwrap();
            assert_eq!(
                &*got.to_float32().unwrap(),
                &*expected.to_float32().unwrap(),
                "owner={owner}, offset={offset}"
            );
            offset += count;
        }
        // Restoring a frontier must not keep the next owner's rotary tables.
        let saved = candidate.snapshot();
        assert_eq!(
            &*candidate.step(11).unwrap().to_float32().unwrap(),
            &*reference.step(11).unwrap().to_float32().unwrap()
        );
        candidate.restore_state(saved);
    }
}

#[test]
fn prepared_ssd_matrix_survives_store_reopen_and_recovers_from_corruption() {
    let dir = Temp::new();
    let path = dir.0.join("test.gguf");
    let mut b = b"GGUF".to_vec();
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    gguf_string(&mut b, "general.architecture");
    b.extend_from_slice(&8u32.to_le_bytes());
    gguf_string(&mut b, "qwen4exp");
    gguf_string(&mut b, "projection.weight");
    b.extend_from_slice(&2u32.to_le_bytes());
    b.extend_from_slice(&32u64.to_le_bytes());
    b.extend_from_slice(&256u64.to_le_bytes());
    b.extend_from_slice(&7u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    while !b.len().is_multiple_of(32) {
        b.push(0);
    }
    for row in 0..256 {
        b.extend_from_slice(&half::f16::from_f32(0.25).to_bits().to_le_bytes());
        b.extend_from_slice(&half::f16::from_f32(-2.0).to_bits().to_le_bytes());
        b.extend_from_slice(&(row as u32 * 7919).to_le_bytes());
        b.extend_from_slice(&[row as u8; 16]);
    }
    std::fs::write(&path, b).unwrap();
    let mut first = weights::Store::open_with_packed_root(&path, Some(&dir.0)).unwrap();
    let want = first
        .read("projection.weight", 0, 256)
        .unwrap()
        .values
        .to_uint32()
        .unwrap()
        .to_vec();
    assert_eq!(first.bytes_read, 256 * 24);
    drop(first);
    let mut reopened = weights::Store::open_with_packed_root(&path, Some(&dir.0)).unwrap();
    assert_eq!(
        reopened
            .read("projection.weight", 0, 256)
            .unwrap()
            .values
            .to_uint32()
            .unwrap()
            .as_ref(),
        want.as_slice()
    );
    assert_eq!(reopened.bytes_read, 0);
    assert_eq!(reopened.packed_hits, 1);
    drop(reopened);
    let alias = dir.0.join("alias.gguf");
    std::os::unix::fs::symlink(&path, &alias).unwrap();
    let mut aliased = weights::Store::open_with_packed_root(&alias, Some(&dir.0)).unwrap();
    aliased.read("projection.weight", 0, 256).unwrap();
    assert_eq!(
        aliased.bytes_read, 0,
        "symlink alias must reuse prepared matrices"
    );
    assert_eq!(aliased.packed_hits, 1);
    drop(aliased);
    let root = std::fs::read_dir(&dir.0)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.is_dir())
        .unwrap();
    let entry = std::fs::read_dir(root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|s| s == "pack"))
        .unwrap();
    let mut bytes = std::fs::read(&entry).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    std::fs::write(entry, bytes).unwrap();
    let mut repaired = weights::Store::open_with_packed_root(&path, Some(&dir.0)).unwrap();
    assert_eq!(
        repaired
            .read("projection.weight", 0, 256)
            .unwrap()
            .values
            .to_uint32()
            .unwrap()
            .as_ref(),
        want.as_slice()
    );
    assert_eq!(repaired.bytes_read, 256 * 24);
    assert_eq!(repaired.packed_hits, 0);
}

#[test]
fn fused_recurrence_matches_f32_reference_and_preserves_snapshot_input() {
    use crate::array::DType;
    let make = |n: usize, shape: &[i64], phase: f32, scale: f32| {
        MxArray::from_float32(
            &(0..n)
                .map(|i| ((i as f32 + phase) * 0.13).sin() * scale)
                .collect::<Vec<_>>(),
            shape,
        )
        .unwrap()
    };
    for kd in [32, 128] {
        let q = make(4 * kd, &[1, 4, 1, kd as i64], 0.0, 0.07);
        let k = make(4 * kd, &[1, 4, 1, kd as i64], 3.0, 0.07);
        let v = make(4 * 32, &[1, 4, 32], 5.0, 0.3);
        let decay =
            MxArray::full(&[1, 4, 1, 1], napi::Either::A(0.97), Some(DType::Float32)).unwrap();
        let beta = MxArray::full(&[1, 4, 1], napi::Either::A(0.4), Some(DType::Float32)).unwrap();
        let initial = make(4 * 32 * kd, &[1, 4, 32, kd as i64], 7.0, 0.1);
        let snapshot = initial.to_float32().unwrap().to_vec();
        let mut fused = initial.clone();
        let mut reference = initial.clone();
        for _ in 0..64 {
            let (y, next) = math::recurrent_step(&q, &k, &v, &decay, &beta, &fused).unwrap();
            let (want, next_ref) =
                math::recurrent_step_reference(&q, &k, &v, &decay, &beta, &reference).unwrap();
            for (a, b) in [(&y, &want), (&next, &next_ref)] {
                let a = a.to_float32().unwrap();
                let b = b.to_float32().unwrap();
                let max = a
                    .iter()
                    .zip(b.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(max < 1e-5, "Dk={kd}, recurrence error {max}");
            }
            fused = next;
            reference = next_ref;
        }
        assert_eq!(initial.to_float32().unwrap().as_ref(), snapshot.as_slice());
    }
}

#[test]
fn fused_recurrence_sequence_matches_varying_stepwise_inputs_and_continuation() {
    use crate::array::DType;
    let make = |shape: &[i64], phase: f32, scale: f32| {
        MxArray::from_float32(
            &(0..shape.iter().product::<i64>())
                .map(|i| ((i as f32 + phase) * 0.17).sin() * scale)
                .collect::<Vec<_>>(),
            shape,
        )
        .unwrap()
    };
    let mut reference = MxArray::zeros(&[1, 4, 32, 128], Some(DType::Float32)).unwrap();
    let mut fused = reference.clone();
    for round in 0..2 {
        let q = make(&[1, 32, 4, 128], round as f32, 0.05);
        let k = make(&[1, 32, 4, 128], 3.0 + round as f32, 0.06);
        let v = make(&[1, 32, 4, 32], 9.0, 0.2);
        let g = make(&[1, 32, 4], 2.0, 0.03).add_scalar(0.95).unwrap();
        let b = make(&[1, 32, 4], 5.0, 0.2).add_scalar(0.5).unwrap();
        let (out, next) = math::recurrent_sequence(&q, &k, &v, &g, &b, &fused).unwrap();
        for t in 0..32 {
            let (want, state) = math::recurrent_step_reference(
                &q.slice_axis(1, t, t + 1)
                    .unwrap()
                    .reshape(&[1, 4, 1, 128])
                    .unwrap(),
                &k.slice_axis(1, t, t + 1)
                    .unwrap()
                    .reshape(&[1, 4, 1, 128])
                    .unwrap(),
                &v.slice_axis(1, t, t + 1)
                    .unwrap()
                    .reshape(&[1, 4, 32])
                    .unwrap(),
                &g.slice_axis(1, t, t + 1)
                    .unwrap()
                    .reshape(&[1, 4, 1, 1])
                    .unwrap(),
                &b.slice_axis(1, t, t + 1)
                    .unwrap()
                    .reshape(&[1, 4, 1])
                    .unwrap(),
                &reference,
            )
            .unwrap();
            let a = out.slice_axis(1, t, t + 1).unwrap().to_float32().unwrap();
            let b = want.to_float32().unwrap();
            assert!(a.iter().zip(b.iter()).all(|(a, b)| (a - b).abs() < 1e-5));
            reference = state;
        }
        let a = next.to_float32().unwrap();
        let b = reference.to_float32().unwrap();
        assert!(a.iter().zip(b.iter()).all(|(a, b)| (a - b).abs() < 1e-5));
        fused = next;
    }
}

#[test]
fn batched_causal_convolution_matches_stepwise_history() {
    use crate::array::DType;
    for dtype in [DType::Float32, DType::BFloat16] {
        let weight = MxArray::from_float32(
            &(0..64).map(|i| (i as f32 * 0.2).cos()).collect::<Vec<_>>(),
            &[16, 4],
        )
        .unwrap()
        .astype(dtype)
        .unwrap();
        let mut a = None;
        let mut b = None;
        for tokens in [3, 32, 1, 7] {
            let input = MxArray::from_float32(
                &(0..tokens * 16)
                    .map(|i| (i as f32 * 0.17).sin())
                    .collect::<Vec<_>>(),
                &[1, tokens, 16],
            )
            .unwrap()
            .astype(dtype)
            .unwrap();
            let actual = math::conv_sequence(&input, &weight, &mut a, 4).unwrap();
            for t in 0..tokens {
                let want = math::conv(
                    &input.slice_axis(1, t, t + 1).unwrap(),
                    &weight,
                    &mut b,
                    4,
                    1,
                )
                .unwrap()
                .to_float32()
                .unwrap();
                let got = actual
                    .slice_axis(1, t, t + 1)
                    .unwrap()
                    .to_float32()
                    .unwrap();
                assert!(
                    got.iter()
                        .zip(want.iter())
                        .all(|(a, b)| (a - b).abs() < 1e-6)
                );
            }
            assert_eq!(
                a.as_ref().unwrap().to_float32().unwrap().as_ref(),
                b.as_ref().unwrap().to_float32().unwrap().as_ref()
            );
        }
    }
}

#[test]
fn dilated_ple_window_preserves_history_across_short_and_long_chunks() {
    use crate::array::DType;
    for dtype in [DType::Float32, DType::BFloat16] {
        let weight = MxArray::from_float32(
            &(0..64).map(|i| (i as f32 * 0.2).cos()).collect::<Vec<_>>(),
            &[16, 4],
        )
        .unwrap()
        .astype(dtype)
        .unwrap();
        let mut batched = None;
        let mut scalar = None;
        for tokens in [2, 32, 1, 7, 3] {
            let x = MxArray::from_float32(
                &(0..tokens * 16)
                    .map(|i| (i as f32 * 0.17).sin())
                    .collect::<Vec<_>>(),
                &[1, tokens, 16],
            )
            .unwrap()
            .astype(dtype)
            .unwrap();
            let actual = math::conv_window(&x, &weight, &mut batched, 4, 3).unwrap();
            for t in 0..tokens {
                let want = math::conv(
                    &x.slice_axis(1, t, t + 1).unwrap(),
                    &weight,
                    &mut scalar,
                    4,
                    3,
                )
                .unwrap()
                .to_float32()
                .unwrap();
                let got = actual
                    .slice_axis(1, t, t + 1)
                    .unwrap()
                    .to_float32()
                    .unwrap();
                assert_eq!(&*got, &*want);
            }
            assert_eq!(&*batched.as_ref().unwrap().shape().unwrap(), &[9, 16]);
            assert_eq!(
                &*batched.as_ref().unwrap().to_float32().unwrap(),
                &*scalar.as_ref().unwrap().to_float32().unwrap()
            );
        }
    }
}

#[test]
fn giant_sparse_file_is_indexed_without_materializing_and_reads_are_capped() {
    use std::io::Write;
    let dir = Temp::new();
    let path = dir.0.join("model.safetensors");
    let header=serde_json::to_vec(&serde_json::json!({"table.weight":{"dtype":"BF16","shape":[1_000_000_000u64,160],"data_offsets":[0,320_000_000_000u64]}})).unwrap();
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.set_len(8 + header.len() as u64 + 320_000_000_000)
        .unwrap();
    drop(file);
    let mut store = weights::Store::open(&dir.0).unwrap();
    assert_eq!(store.bytes_read, 0);
    assert!(store.read("table.weight", 0, 1_000_000_000).is_err());
    assert_eq!(store.bytes_read, 0);
    let row = store
        .read("table.weight", 999_999_999, 1)
        .unwrap()
        .dense()
        .unwrap();
    assert_eq!(row.size().unwrap(), 160);
    assert_eq!(store.bytes_read, 320);
}

#[test]
fn truncated_weight_file_is_rejected_during_descriptor_preflight() {
    use std::io::Write;
    let dir = Temp::new();
    let header = br#"{"table.weight":{"dtype":"BF16","shape":[16,32],"data_offsets":[0,1024]}}"#;
    let mut f = std::fs::File::create(dir.0.join("model.safetensors")).unwrap();
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header).unwrap();
    assert!(weights::Store::open(&dir.0).is_err());
}

#[test]
fn vision_merger_is_assembled_across_bounded_reads() {
    use std::io::{Seek, SeekFrom, Write};
    let dir = Temp::new();
    let width = 4608usize;
    let elements = width * width;
    let key = "visual.merger.linear_fc1.weight";
    let header = serde_json::to_vec(&serde_json::json!({
        key: {"dtype": "BF16", "shape": [width, width], "data_offsets": [0, elements * 2]}
    }))
    .unwrap();
    let mut file = std::fs::File::create(dir.0.join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    let offset = 8 + header.len() as u64;
    file.set_len(offset + elements as u64 * 2).unwrap();
    let boundary = (super::weights::MAX_READ_BYTES as usize / 4 / width) * width;
    for (index, value) in [
        (0, 1.0),
        (boundary - 1, 2.0),
        (boundary, 3.0),
        (elements - 1, 4.0),
    ] {
        file.seek(SeekFrom::Start(offset + index as u64 * 2))
            .unwrap();
        file.write_all(&half::bf16::from_f32(value).to_bits().to_le_bytes())
            .unwrap();
    }
    drop(file);
    let mut store = weights::Store::open(&dir.0).unwrap();
    assert!(store.dense(key).is_err());
    assert_eq!(store.bytes_read, 0);
    let value = media::load_vision_tensor(&mut store, key).unwrap();
    assert_eq!(&*value.shape().unwrap(), &[width as i64, width as i64]);
    assert_eq!(value.dtype().unwrap(), crate::array::DType::BFloat16);
    assert_eq!(store.bytes_read, (elements * 2) as u64);
    let value = value.reshape(&[-1]).unwrap();
    for (index, want) in [
        (0, 1.0),
        (boundary - 1, 2.0),
        (boundary, 3.0),
        (elements - 1, 4.0),
    ] {
        assert_eq!(value.item_at_float32(index).unwrap(), want);
    }
    assert_eq!(value.item_at_float32(boundary + 1).unwrap(), 0.0);
}

#[test]
fn bf16_decoder_matches_mlx_vlm() {
    let path = fixture().join("bf16");
    let raw: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("config.json")).unwrap()).unwrap();
    let oracle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("oracle.json")).unwrap()).unwrap();
    for source in [&path, path.join("model.gguf").as_path()] {
        let store = weights::Store::open(source).unwrap();
        let config = if store.gguf {
            gguf::config(&store).unwrap()
        } else {
            config::Config::parse(&raw).unwrap()
        };
        let mut model = decoder::Decoder::new(config, store).unwrap();
        let mut maximum = 0f32;
        for (i, t) in oracle["tokens"].as_array().unwrap().iter().enumerate() {
            let a = model
                .step(t.as_u64().unwrap() as u32)
                .unwrap()
                .to_float32()
                .unwrap();
            let expected = oracle["logits"][i].as_array().unwrap();
            let diff = a
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b.as_f64().unwrap() as f32).abs())
                .fold(0f32, f32::max);
            maximum = maximum.max(diff);
            assert!(
                diff < 0.04,
                "BF16 {} token {i}: max absolute logit error {diff}",
                source.display()
            );
        }
        eprintln!(
            "BF16 oracle {} maximum absolute logit error {maximum}",
            source.display()
        );
    }
}

#[test]
fn paged_qsa_matches_flat_across_page_boundaries_sparse_selection_and_owner_switches() {
    let path = fixture().join("bf16/paged");
    let raw = serde_json::from_slice(&std::fs::read(path.join("config.json")).unwrap()).unwrap();
    let config = config::Config::parse(&raw).unwrap();
    let make =
        || decoder::Decoder::new(config.clone(), weights::Store::open(&path).unwrap()).unwrap();
    let mut flat_a = make();
    let mut flat_b = make();
    let mut paged = make();
    paged.paged = Some(super::paged::create(&config).unwrap());
    let mut states = [None, None];
    for round in 0..5 {
        for (owner, saved_state) in states.iter_mut().enumerate() {
            let seq = owner as u32 + 1;
            let adapter = paged.paged.as_mut().unwrap();
            if round == 0 {
                adapter.begin_request(seq).unwrap();
            } else {
                adapter.activate_request(seq).unwrap();
            }
            if let Some(state) = saved_state.take() {
                paged.restore_state(state);
            }
            let tokens: Vec<u32> = (0..9)
                .map(|i| ((round * 9 + i + owner * 7) % 19 + 3) as u32)
                .collect();
            let flat = if owner == 0 { &mut flat_a } else { &mut flat_b };
            let expected = flat
                .prefill_chunk(&tokens, None, true)
                .unwrap()
                .to_float32()
                .unwrap();
            let actual = paged
                .prefill_chunk(&tokens, None, true)
                .unwrap()
                .to_float32()
                .unwrap();
            assert_eq!(&*expected, &*actual, "round {round}, owner {owner}");
            assert_eq!(paged.history, flat.history);
            assert_eq!(
                paged.paged.as_ref().unwrap().request_tokens(),
                paged.history
            );
            assert_eq!(
                paged
                    .paged
                    .as_ref()
                    .unwrap()
                    .block_table()
                    .unwrap()
                    .num_blocks(),
                paged.history.len().div_ceil(32)
            );
            *saved_state = Some(paged.take_state());
        }
    }
}

#[test]
fn mtp_head_matches_reference_hc_norm_and_shifted_history() {
    let path = fixture().join("bf16/paged");
    let raw = serde_json::from_slice(&std::fs::read(path.join("config.json")).unwrap()).unwrap();
    let c = config::Config::parse(&raw).unwrap();
    let store = weights::Store::open(&path).unwrap();
    assert!(auxiliary::validate_mtp(&store, &c).unwrap());
    let mut decoder = decoder::Decoder::new(c.clone(), store).unwrap();
    let oracle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("mtp-oracle.json")).unwrap()).unwrap();
    let mut cache = decoder::LayerCache::default();
    for row in oracle["rows"].as_array().unwrap() {
        let hidden: Vec<f32> = row["input_hidden"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let hidden = MxArray::from_float32(&hidden, &[1, 1, (c.hc_count * c.hidden_size) as i64])
            .unwrap()
            .astype(crate::array::DType::BFloat16)
            .unwrap();
        let (h, logits) = decoder
            .draft_step(
                &hidden,
                row["token"].as_u64().unwrap() as u32,
                row["position"].as_u64().unwrap() as usize,
                &mut cache,
                true,
            )
            .unwrap();
        for (actual, key) in [(h, "hidden"), (logits.unwrap(), "logits")] {
            let diff = actual
                .to_float32()
                .unwrap()
                .iter()
                .zip(row[key].as_array().unwrap())
                .map(|(a, b)| (a - b.as_f64().unwrap() as f32).abs())
                .fold(0f32, f32::max);
            assert!(diff < 0.04, "position {} {key}: {diff}", row["position"]);
        }
    }
    assert!(
        decoder.history.is_empty(),
        "head must not mutate target history"
    );
}

#[test]
fn exclusive_owner_admission_bounds_state_and_preserves_existing_sessions() {
    use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
    let mut inner = tiny_inner();
    for seq in 1..=4 {
        inner.activate_exclusive_seq(seq).unwrap();
        inner
            .decoder
            .prefill_chunk(&[seq + 2], None, false)
            .unwrap();
    }
    let before = inner.decoder.paged.as_ref().unwrap().live_seq_ids();
    let bytes = inner.decoder.weights.bytes_read;
    assert!(inner.activate_exclusive_seq(5).is_err());
    assert_eq!(inner.active_seq, Some(4));
    assert_eq!(inner.decoder.history, [6]);
    assert_eq!(inner.decoder.weights.bytes_read, bytes);
    assert_eq!(inner.decoder.paged.as_ref().unwrap().live_seq_ids(), before);
    inner.activate_exclusive_seq(2).unwrap();
    assert_eq!(inner.decoder.history, [4]);
    inner
        .decoder
        .paged
        .as_mut()
        .unwrap()
        .release_request_for(1)
        .unwrap();
    inner.release_scheduled_recurrent_for(1);
    inner.activate_exclusive_seq(5).unwrap();
    assert!(inner.decoder.history.is_empty());
    for seq in 2..=4 {
        inner.activate_exclusive_seq(seq).unwrap();
        assert_eq!(inner.decoder.history, [seq + 2]);
    }
}

#[test]
fn media_mrope_matches_reference_including_compressed_indexer_block_positions() {
    let path = fixture().join("bf16/paged");
    let raw = serde_json::from_slice(&std::fs::read(path.join("config.json")).unwrap()).unwrap();
    let c = config::Config::parse(&raw).unwrap();
    let oracle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("mrope-oracle.json")).unwrap()).unwrap();
    for pages in [false, true] {
        let mut decoder =
            decoder::Decoder::new(c.clone(), weights::Store::open(&path).unwrap()).unwrap();
        if pages {
            let mut a = super::paged::create(&c).unwrap();
            a.begin_request(1).unwrap();
            decoder.paged = Some(a);
        }
        decoder.positions = oracle["positions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| std::array::from_fn(|i| v[i].as_i64().unwrap()))
            .collect();
        decoder.rope_delta = -4;
        for (i, t) in oracle["tokens"].as_array().unwrap().iter().enumerate() {
            let actual = decoder
                .step(t.as_u64().unwrap() as u32)
                .unwrap()
                .to_float32()
                .unwrap();
            let diff = actual
                .iter()
                .zip(oracle["logits"][i].as_array().unwrap())
                .map(|(a, b)| (a - b.as_f64().unwrap() as f32).abs())
                .fold(0f32, f32::max);
            assert!(diff < 0.04, "paged={pages}, token {i}, error {diff}");
        }
    }
}

fn tiny_inner() -> Inner {
    let path = fixture().join("bf16/paged");
    let raw = serde_json::from_slice(&std::fs::read(path.join("config.json")).unwrap()).unwrap();
    let c = config::Config::parse(&raw).unwrap();
    let store = weights::Store::open(&path).unwrap();
    let vision = media::Vision::metadata(&serde_json::Value::Null, &store, &c).unwrap();
    let tokenizer_path =
        std::env::temp_dir().join(format!("qwen4-test-tokenizer-{}.json", std::process::id()));
    std::fs::write(&tokenizer_path,r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"a":0,"b":1,"<unk>":2},"unk_token":"<unk>"}}"#).unwrap();
    let tokenizer =
        Arc::new(Qwen3Tokenizer::load_from_file_sync(tokenizer_path.to_str().unwrap()).unwrap());
    std::fs::remove_file(tokenizer_path).unwrap();
    let mut decoder = decoder::Decoder::new(c.clone(), store).unwrap();
    decoder.paged = Some(super::paged::create(&c).unwrap());
    Inner {
        decoder,
        tokenizer,
        defaults: Default::default(),
        saved_history: Vec::new(),
        rows: Default::default(),
        active_seq: None,
        mtp: Default::default(),
        has_mtp: true,
        vision,
        media_prefill: None,
        _pool_guard: None,
        _cache_limit_guard: crate::cache_limit::coordinator().register(1 << 20),
    }
}

#[test]
fn completed_paged_turn_keeps_auxiliary_state_and_partial_page_for_continuation() {
    use crate::engine::backend::{PagedBackend, PagedPrefix};
    use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
    let mut inner = tiny_inner();
    inner.activate_paged_seq(1).unwrap();
    let prompt = [3, 9, 5, 2, 7];
    let prefix = inner
        .prime_prefix_state(&prompt, false, 32, &[], 0)
        .unwrap();
    inner
        .paged_prefill(
            &prompt,
            &prefix,
            Stream::default(crate::stream::DeviceType::Gpu),
        )
        .unwrap();
    inner.decoder.step(11).unwrap();
    inner.finalize_paged_turn(true, 0);
    inner
        .save_paged_history(&prompt, &[11, 12], false, true)
        .unwrap();
    let state = inner.capture_owner_state(1);
    assert_eq!(state, [3, 9, 5, 2, 7, 11]);
    inner.park_active_scheduled_recurrent().unwrap();
    inner.activate_paged_seq(2).unwrap();
    inner.decoder.prefill_chunk(&[8, 4], None, false).unwrap();
    inner.install_owner_state(1, &state);
    let next = [3, 9, 5, 2, 7, 11, 12, 6];
    let prefix = inner.prime_prefix_state(&next, true, 32, &[], 0).unwrap();
    assert_eq!(prefix.effective_cached_prefix_len(), 6);
    let actual = inner
        .paged_prefill(
            &next[6..],
            &prefix,
            Stream::default(crate::stream::DeviceType::Gpu),
        )
        .unwrap();
    let mut reference = tiny_inner();
    reference.activate_paged_seq(1).unwrap();
    let expected = reference.decoder.prefill_chunk(&next, None, true).unwrap();
    assert_eq!(
        &*actual.to_float32().unwrap(),
        &*expected.to_float32().unwrap()
    );
    assert_eq!(
        inner
            .decoder
            .paged
            .as_ref()
            .unwrap()
            .block_table()
            .unwrap()
            .num_blocks(),
        1
    );
}

#[test]
fn scheduled_owner_switch_does_not_inherit_another_requests_cancellation() {
    use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
    let mut inner = tiny_inner();
    inner.activate_paged_seq(1).unwrap();
    inner.set_turn_cancel_flag(Some(Arc::new(AtomicBool::new(false))));
    inner.decoder.prefill_chunk(&[3, 9], None, false).unwrap();
    inner.activate_paged_seq(2).unwrap();
    let cancel = Arc::new(AtomicBool::new(false));
    inner.set_turn_cancel_flag(Some(cancel.clone()));
    inner.decoder.prefill_chunk(&[8, 4], None, false).unwrap();
    cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    inner.run_paged_decode_step_batched(&[(1, 7)]).unwrap();
    assert_eq!(inner.decoder.history, [3, 9, 7]);
    inner.activate_paged_seq(2).unwrap();
    assert!(inner.decoder.check_cancelled().is_err());
}

#[test]
fn scheduled_decode_preserves_sampling_shape_and_owner_logits() {
    use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
    let mut inner = tiny_inner();
    let mut expected = Vec::new();
    for (seq, prompt, token) in [(1, vec![3, 9, 5], 7), (2, vec![8, 2, 4, 3], 11)] {
        inner.activate_paged_seq(seq).unwrap();
        inner.decoder.prefill_chunk(&prompt, None, false).unwrap();
        let mut reference = tiny_inner();
        reference.activate_paged_seq(seq).unwrap();
        reference
            .decoder
            .prefill_chunk(&prompt, None, false)
            .unwrap();
        expected.extend_from_slice(&reference.decoder.step(token).unwrap().to_float32().unwrap());
    }
    assert!(inner.run_paged_decode_step_batched(&[]).is_err());
    assert!(
        inner
            .run_paged_decode_step_batched(&[(1, 7), (1, 7)])
            .is_err()
    );
    let logits = inner
        .run_paged_decode_step_batched(&[(1, 7), (2, 11)])
        .unwrap();
    assert_eq!(&*logits.shape().unwrap(), [2, 1, 64]);
    assert_eq!(&*logits.to_float32().unwrap(), &expected);
    // Exercise the same singleton-axis removal used by the scheduler sampler.
    assert_eq!(
        &*logits.squeeze(Some(&[1])).unwrap().shape().unwrap(),
        [2, 64]
    );
}

#[test]
fn mtp_confidence_and_adaptive_policy_preserve_the_target_frontier() {
    use crate::engine::backend::PagedBackend;
    use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
    use rand::SeedableRng;
    let mut inner = tiny_inner();
    let config = crate::engine::types::ChatConfig {
        temperature: Some(0.0),
        enable_mtp: Some(true),
        mtp_depth: Some(5),
        mtp_adaptive_depth: Some(true),
        ..Default::default()
    };
    let params = inner.resolve_params(&config);
    assert!(params.mtp_adaptive_depth);
    assert_eq!(params.mtp_depth, 3);
    let sampled = inner.resolve_params(&crate::engine::types::ChatConfig {
        temperature: Some(0.7),
        ..config
    });
    assert!(sampled.enable_mtp);
    assert!(!sampled.mtp_adaptive_depth);
    inner.activate_paged_seq(1).unwrap();
    inner.begin_mtp(1, 0).unwrap();
    let prompt = [3, 9, 5, 7];
    let prefix = inner
        .prime_prefix_state(&prompt, false, 32, &[], 0)
        .unwrap();
    let logits = inner
        .paged_prefill(
            &prompt,
            &prefix,
            Stream::default(crate::stream::DeviceType::Gpu),
        )
        .unwrap();
    let anchor = logits.argmax(-1, None).unwrap().item_at_int32(0).unwrap() as u32;
    let mut rng = rand::rngs::StdRng::seed_from_u64(19);
    let proposal = inner
        .propose_mtp(1, anchor, 3, &params, &mut rng, true)
        .unwrap();
    assert_eq!(proposal.draft_ids.len(), 3);
    let confidence = proposal.keep_probabilities.unwrap();
    assert_eq!(confidence.len(), 3);
    assert!(
        confidence
            .iter()
            .all(|p| p.is_finite() && *p > 0.0 && *p <= 1.0)
    );
    let stochastic = inner
        .propose_mtp(1, anchor, 3, &sampled, &mut rng, false)
        .unwrap();
    assert_eq!(stochastic.draft_dists.len(), 3);
    for distribution in stochastic.draft_dists {
        let sum: f32 = distribution.to_float32().unwrap().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
    assert_eq!(inner.decoder.history, prompt);
    assert_eq!(
        inner.decoder.paged.as_ref().unwrap().request_tokens(),
        prompt
    );
    inner.release_scheduled_speculation(1);
    assert!(
        inner.begin_mtp(1, 0).unwrap(),
        "a cold restart replaces the previous head frontier"
    );
}

#[test]
fn mtp_warm_resume_and_ar_fallback_do_not_replay_untrusted_head_state() {
    use crate::engine::backend::PagedBackend;
    use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
    let mut inner = tiny_inner();
    inner.activate_paged_seq(1).unwrap();
    inner.begin_mtp(1, 0).unwrap();
    let prompt = [3, 9, 5, 7];
    let prefix = inner
        .prime_prefix_state(&prompt, false, 32, &[], 0)
        .unwrap();
    inner
        .paged_prefill(
            &prompt,
            &prefix,
            Stream::default(crate::stream::DeviceType::Gpu),
        )
        .unwrap();
    inner.release_scheduled_speculation(1);
    assert!(
        inner.begin_mtp(1, 4).unwrap(),
        "matching completed head can resume"
    );
    inner.release_scheduled_speculation(1);
    // A target-only fallback must stop touching the head. A missing descriptor
    // makes any accidental head advance fail even when its arrays were cached.
    inner
        .decoder
        .weights
        .tensors
        .remove("mtp.pre_fc_norm_hidden.weight")
        .unwrap();
    inner.run_paged_decode_step_batched(&[(1, 11)]).unwrap();
    assert_eq!(inner.decoder.history, [3, 9, 5, 7, 11]);
    assert!(
        !inner.begin_mtp(1, 5).unwrap(),
        "AR advanced past the retained draft frontier"
    );
}

#[test]
fn mtp_transaction_restores_every_accepted_frontier_and_continues_exactly() {
    use crate::engine::backend::PagedBackend;
    use crate::engine::hybrid_scheduler::{
        HybridSchedulerBackend, ScheduledVerifyCommit, ScheduledVerifyRow,
    };
    let prompt = [3, 9, 5, 2, 7, 11, 12, 8, 10, 9, 13, 7, 4, 3, 11, 8, 6, 5];
    let verify = [4, 11, 7, 9];
    for keep in 0..=4 {
        let mut inner = tiny_inner();
        inner.activate_paged_seq(1).unwrap();
        inner.begin_mtp(1, 0).unwrap();
        let prefix = inner
            .prime_prefix_state(&prompt, false, 32, &[], 0)
            .unwrap();
        inner
            .paged_prefill(
                &prompt,
                &prefix,
                Stream::default(crate::stream::DeviceType::Gpu),
            )
            .unwrap();
        let mut reference = tiny_inner();
        reference.activate_paged_seq(1).unwrap();
        reference
            .decoder
            .prefill_chunk(&prompt, None, false)
            .unwrap();
        let expected: Vec<_> = verify
            .iter()
            .map(|&t| {
                reference
                    .decoder
                    .step(t)
                    .unwrap()
                    .to_float32()
                    .unwrap()
                    .to_vec()
            })
            .collect();
        let actual = inner
            .verify_mtp(&[ScheduledVerifyRow {
                seq_id: 1,
                first_position: prompt.len() as u32,
                tokens: verify.to_vec(),
                speculative: true,
            }])
            .unwrap();
        assert_eq!(&*actual.shape().unwrap(), [4, 1, 64]);
        let actual = actual
            .transpose(Some(&[1, 0, 2]))
            .unwrap()
            .to_float32()
            .unwrap();
        assert_eq!(
            &*actual,
            expected.concat(),
            "layer-major verification must preserve singleton logits"
        );
        inner
            .commit_mtp(&[ScheduledVerifyCommit { seq_id: 1, keep }])
            .unwrap()
            .remove(0)
            .unwrap();
        assert_eq!(
            inner.decoder.history,
            [&prompt[..], &verify[..keep]].concat()
        );
        assert_eq!(
            inner.decoder.paged.as_ref().unwrap().request_tokens(),
            inner.decoder.history
        );
        let mut clean = tiny_inner();
        clean.activate_paged_seq(1).unwrap();
        for &t in prompt.iter().chain(verify[..keep].iter()) {
            clean.decoder.step(t).unwrap();
        }
        assert_eq!(
            &*inner.decoder.step(15).unwrap().to_float32().unwrap(),
            &*clean.decoder.step(15).unwrap().to_float32().unwrap(),
            "keep {keep}"
        );
    }
}

#[test]
fn mtp_verification_keeps_ragged_owners_isolated() {
    use crate::engine::backend::PagedBackend;
    use crate::engine::hybrid_scheduler::{
        HybridSchedulerBackend, ScheduledVerifyCommit, ScheduledVerifyRow,
    };
    let mut inner = tiny_inner();
    let prompts = [vec![3, 9, 7, 8, 11], vec![12, 7, 3, 13, 9, 2, 6, 7, 8]];
    for (i, prompt) in prompts.iter().enumerate() {
        let seq = i as u32 + 1;
        inner.activate_paged_seq(seq).unwrap();
        inner.begin_mtp(seq, 0).unwrap();
        let prefix = inner.prime_prefix_state(prompt, false, 32, &[], 0).unwrap();
        inner
            .paged_prefill(
                prompt,
                &prefix,
                Stream::default(crate::stream::DeviceType::Gpu),
            )
            .unwrap();
    }
    inner
        .verify_mtp(&[
            ScheduledVerifyRow {
                seq_id: 1,
                first_position: 5,
                tokens: vec![3, 4, 5, 6],
                speculative: true,
            },
            ScheduledVerifyRow {
                seq_id: 2,
                first_position: 9,
                tokens: vec![11, 12],
                speculative: true,
            },
        ])
        .unwrap();
    for result in inner
        .commit_mtp(&[
            ScheduledVerifyCommit { seq_id: 2, keep: 1 },
            ScheduledVerifyCommit { seq_id: 1, keep: 2 },
        ])
        .unwrap()
    {
        result.unwrap();
    }
    for (i, prompt) in prompts.iter().enumerate() {
        let seq = i as u32 + 1;
        inner.activate_paged_seq(seq).unwrap();
        let mut clean = tiny_inner();
        clean.activate_paged_seq(1).unwrap();
        let kept: &[u32] = if seq == 1 { &[3, 4] } else { &[11] };
        for &token in prompt.iter().chain(kept) {
            clean.decoder.step(token).unwrap();
        }
        assert_eq!(
            &*inner.decoder.step(15).unwrap().to_float32().unwrap(),
            &*clean.decoder.step(15).unwrap().to_float32().unwrap()
        );
    }
}

#[test]
fn incomplete_mtp_companion_is_rejected_before_payload_reads() {
    let path = fixture().join("bf16/paged");
    let raw = serde_json::from_slice(&std::fs::read(path.join("config.json")).unwrap()).unwrap();
    let c = config::Config::parse(&raw).unwrap();
    let mut store = weights::Store::open(&path).unwrap();
    store.tensors.remove("mtp.fc_hidden.weight");
    assert!(auxiliary::validate_mtp(&store, &c).is_err());
    assert_eq!(store.bytes_read, 0);
}

#[test]
fn fused_routed_experts_preserve_mixed_formats_and_changing_inputs() {
    if !crate::engine::persistence::compiled_forward_backend_available() {
        return;
    }
    use crate::array::DType;
    use std::sync::Arc;
    let (e, h, m) = (3usize, 2560usize, 640usize);
    for (gb, db) in [(4, 5), (4, 8), (5, 8)] {
        for tokens in [1usize, 2, 8] {
            for mut seed in [71u32, 197] {
                let mut next = || {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    seed
                };
                let mut bank = |n: usize, k: usize, bits: usize, affine: bool| {
                    let values: Vec<u32> = (0..e * n * k * bits / 32).map(|_| next()).collect();
                    let scales = if affine {
                        let values: Vec<u16> = (0..e * n * k / 32)
                            .map(|_| {
                                half::f16::from_f32((next() % 97 + 1) as f32 / 16384.).to_bits()
                            })
                            .collect();
                        MxArray::from_float16(&values, &[(e * n) as i64, (k / 32) as i64]).unwrap()
                    } else {
                        let values: Vec<u8> =
                            (0..e * n * k / 16).map(|_| (next() % 64) as u8).collect();
                        MxArray::from_uint8(&values, &[(e * n) as i64, (k / 16) as i64]).unwrap()
                    };
                    let side = if affine { k / 32 } else { k / 128 };
                    let biases: Vec<u16> = (0..e * n * side)
                        .map(|_| {
                            half::f16::from_f32(
                                (next() % 97 + 1) as f32 / 16384. * if affine { -1. } else { 1. },
                            )
                            .to_bits()
                        })
                        .collect();
                    Arc::new(weights::Weight {
                        dense_bf16: None,
                        values: MxArray::from_uint32(
                            &values,
                            &[(e * n) as i64, (k * bits / 32) as i64],
                        )
                        .unwrap(),
                        scales: Some(scales),
                        biases: Some(
                            MxArray::from_float16(&biases, &[(e * n) as i64, side as i64]).unwrap(),
                        ),
                        group: 32,
                        bits: bits as i32,
                        mode: if affine {
                            "affine"
                        } else if bits == 4 {
                            "q4k"
                        } else {
                            "q5k"
                        }
                        .into(),
                    })
                };
                let banks = [
                    bank(m, h, gb, false),
                    bank(m, h, gb, false),
                    bank(h, m, db, true),
                ];
                let input: Vec<f32> = (0..tokens * h)
                    .map(|_| (next() % 1024) as f32 / 256. - 2.)
                    .collect();
                let x = MxArray::from_float32(&input, &[1, tokens as i64, h as i64])
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap();
                let selected: Vec<u32> = (0..tokens * 10).map(|_| next() % e as u32).collect();
                let ids = MxArray::from_uint32(&selected, &[(tokens * 10) as i64]).unwrap();
                let scores: Vec<f32> = (0..tokens * 10)
                    .map(|_| (next() % 19 + 1) as f32 / 100.)
                    .collect();
                let scores = MxArray::from_float32(&scores, &[1, tokens as i64, 10]).unwrap();
                let rows: Vec<i32> = (0..tokens * 10).map(|i| (i / 10) as i32).collect();
                let expanded = x
                    .reshape(&[tokens as i64, h as i64])
                    .unwrap()
                    .take(
                        &MxArray::from_int32(&rows, &[(tokens * 10) as i64]).unwrap(),
                        0,
                    )
                    .unwrap()
                    .reshape(&[(tokens * 10) as i64, 1, h as i64])
                    .unwrap();
                let gate = banks[0].expert_rows(&expanded, &ids, e, false).unwrap();
                let up = banks[1].expert_rows(&expanded, &ids, e, false).unwrap();
                let hidden = crate::nn::Activations::swiglu_compiled(&gate, &up).unwrap();
                let down = banks[2]
                    .expert_rows(&hidden, &ids, e, false)
                    .unwrap()
                    .reshape(&[(tokens * 10) as i64, h as i64])
                    .unwrap();
                for score_type in [DType::Float32, DType::BFloat16] {
                    let scores = scores.astype(score_type).unwrap();
                    let expected = math::combine_expert_rows(&down, &scores, None, 10)
                        .unwrap()
                        .to_float32()
                        .unwrap();
                    let output = math::routed_experts(&x, &ids, &scores, &banks)
                        .unwrap()
                        .expect("mixed-format fast path");
                    assert_eq!(output.dtype().unwrap(), score_type);
                    assert_eq!(
                        &*expected,
                        &*output.to_float32().unwrap(),
                        "gate={gb}, down={db}, tokens={tokens}, scores={score_type:?}"
                    );
                }
                assert!(
                    math::routed_experts(&x.astype(DType::Float32).unwrap(), &ids, &scores, &banks)
                        .unwrap()
                        .is_none()
                );
                assert!(
                    math::routed_experts(&x, &ids, &scores.astype(DType::Float16).unwrap(), &banks)
                        .unwrap()
                        .is_none()
                );
            }
        }
    }
}

#[test]
fn kquant_nax_gathers_match_dequantized_experts_with_tail_tiles() {
    use crate::array::DType;
    // More than four rows per expert reaches the sorted RHS matrix path;
    // an explicit 79-row matrix reaches the general gathered matrix path.
    // Neither M=79 nor N=96 fills the NAX 64x64 tile.
    for (mode, bits, group, ratio, mins) in [
        ("q4k", 4usize, 32usize, 8usize, true),
        ("q5k", 5, 32, 8, true),
        ("q6k", 6, 16, 16, false),
        ("q3k", 3, 16, 16, false),
        ("iq4nl", 4, 32, 1, false),
        ("iq4xs", 4, 32, 8, false),
        ("iq3s", 8, 32, 8, false),
    ] {
        let (experts, n, k, m) = (3usize, 96usize, 256usize, 79usize);
        let side = if mins { 2 } else { 1 };
        let values: Vec<u32> = (0..experts * n * k * bits / 32)
            .map(|i| (i as u32).wrapping_mul(0x9e3779b9).wrapping_add(0x12345678))
            .collect();
        let scales: Vec<u8> = (0..experts * n * k / group * side)
            .map(|i| (i % 3 + 1) as u8)
            .collect();
        let bias = vec![
            half::f16::from_f32(1.0 / 1024.0).to_bits();
            experts * n * k / group / ratio * side
        ];
        let weight = weights::Weight {
            dense_bf16: None,
            values: MxArray::from_uint32(&values, &[(experts * n) as i64, (k * bits / 32) as i64])
                .unwrap(),
            scales: Some(
                MxArray::from_uint8(&scales, &[(experts * n) as i64, (k / group * side) as i64])
                    .unwrap()
                    .astype(if mins { DType::Uint8 } else { DType::Int8 })
                    .unwrap(),
            ),
            biases: Some(
                MxArray::from_float16(
                    &bias,
                    &[(experts * n) as i64, (k / group / ratio * side) as i64],
                )
                .unwrap(),
            ),
            group: group as i32,
            bits: bits as i32,
            mode: mode.into(),
        };
        let w = weight
            .values
            .reshape(&[experts as i64, n as i64, (k * bits / 32) as i64])
            .unwrap();
        let s = weight
            .scales
            .as_ref()
            .unwrap()
            .reshape(&[experts as i64, n as i64, (k / group * side) as i64])
            .unwrap();
        let b = weight
            .biases
            .as_ref()
            .unwrap()
            .reshape(&[experts as i64, n as i64, (k / group / ratio * side) as i64])
            .unwrap();
        let dense = weight.dense().unwrap().to_float32().unwrap();
        for factor in [1i64, 2] {
            let reshape = |a: &MxArray| {
                let shape = a.shape().unwrap();
                a.reshape(&[shape[0] / factor, shape[1] * factor]).unwrap()
            };
            let narrow = weights::Weight {
                dense_bf16: None,
                values: reshape(&weight.values),
                scales: weight.scales.as_ref().map(reshape),
                biases: weight.biases.as_ref().map(reshape),
                ..weight.clone()
            };
            let ids = MxArray::from_uint32(&[2, 1, 0, 2, 1, 0, 2, 0, 0, 1], &[10]).unwrap();
            for dtype in [DType::Float32, DType::Float16, DType::BFloat16] {
                let data: Vec<f32> = (0..10 * k * factor as usize)
                    .map(|i| (i % 7) as f32 / 64. - 0.046875)
                    .collect();
                let input = MxArray::from_float32(&data, &[10, 1, k as i64 * factor])
                    .unwrap()
                    .astype(dtype)
                    .unwrap();
                let expected = narrow
                    .gather_rows(&input, &ids, experts, false)
                    .unwrap()
                    .to_float32()
                    .unwrap()
                    .to_vec();
                let direct = narrow
                    .expert_rows(&input, &ids, experts, false)
                    .unwrap()
                    .to_float32()
                    .unwrap();
                assert_eq!(
                    &*direct, &expected,
                    "direct GEMV {mode}, factor={factor}, dtype={dtype:?}"
                );
            }
        }
        for sorted in [false, true] {
            let batches = if sorted { m } else { 3 };
            let rows = if sorted { 1 } else { m };
            let input: Vec<f32> = (0..batches * rows * k)
                .map(|i| (i % 7) as f32 / 64.0 - 0.046875)
                .collect();
            let x = MxArray::from_float32(&input, &[batches as i64, rows as i64, k as i64])
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            let ids: Vec<u32> = (0..batches)
                .map(|i| {
                    if sorted {
                        (i * experts / batches) as u32
                    } else {
                        (2 - i) as u32
                    }
                })
                .collect();
            let indices = MxArray::from_uint32(&ids, &[batches as i64]).unwrap();
            let mode_c = std::ffi::CString::new(mode).unwrap();
            let out = unsafe {
                mlx_sys::mlx_gather_qmm(
                    x.as_raw_ptr(),
                    w.as_raw_ptr(),
                    s.as_raw_ptr(),
                    b.as_raw_ptr(),
                    std::ptr::null_mut(),
                    indices.as_raw_ptr(),
                    true,
                    group as i32,
                    bits as i32,
                    mode_c.as_ptr(),
                    sorted,
                )
            };
            let out = MxArray::from_handle(out, "K-quant gathered matrix test")
                .unwrap()
                .to_float32()
                .unwrap();
            let mut error = 0f32;
            for batch in 0..batches {
                for row in 0..rows {
                    for col in 0..n {
                        let want: f64 = (0..k)
                            .map(|j| {
                                input[(batch * rows + row) * k + j] as f64
                                    * dense[(ids[batch] as usize * n + col) * k + j] as f64
                            })
                            .sum();
                        let got = out[(batch * rows + row) * n + col];
                        assert!(got.is_finite(), "{mode} sorted={sorted}: nonfinite result");
                        error = error.max((got - want as f32).abs());
                    }
                }
            }
            assert!(error < 0.003, "{mode} sorted={sorted}: error={error}");
        }
    }
}

#[test]
fn sorted_expert_combine_keeps_product_rounding_and_reduction_order() {
    use crate::array::DType;
    if !crate::engine::persistence::compiled_forward_backend_available() {
        return;
    }
    let (tokens, top) = (17usize, 10usize);
    let rows = tokens * top;
    let inverse = MxArray::from_uint32(
        &(0..rows)
            .map(|i| ((i * 37) % rows) as u32)
            .collect::<Vec<_>>(),
        &[rows as i64],
    )
    .unwrap();
    for hidden in [128usize, 2560] {
        for (value_type, score_type) in [
            (DType::BFloat16, DType::BFloat16),
            (DType::BFloat16, DType::Float32),
            (DType::Float32, DType::Float32),
        ] {
            let values = MxArray::from_float32(
                &(0..rows * hidden)
                    .map(|i| {
                        ((i * 997 % 253) as f32 - 126.) * if i % 7 == 0 { 16.0 } else { 0.0013 }
                    })
                    .collect::<Vec<_>>(),
                &[rows as i64, hidden as i64],
            )
            .unwrap()
            .astype(value_type)
            .unwrap();
            let scores = MxArray::from_float32(
                &(0..rows)
                    .map(|i| (i % 11 + 1) as f32 * 0.0113)
                    .collect::<Vec<_>>(),
                &[1, tokens as i64, top as i64],
            )
            .unwrap()
            .astype(score_type)
            .unwrap();
            let reference = values
                .take(&inverse, 0)
                .unwrap()
                .reshape(&[1, tokens as i64, top as i64, hidden as i64])
                .unwrap()
                .mul(&scores.expand_dims(-1).unwrap())
                .unwrap()
                .sum(Some(&[2]), Some(false))
                .unwrap()
                .to_float32()
                .unwrap();
            let raw = unsafe {
                mlx_sys::mlx_qwen4_sorted_combine(
                    values.as_raw_ptr(),
                    scores.as_raw_ptr(),
                    inverse.as_raw_ptr(),
                    top as i32,
                )
            };
            let actual = MxArray::from_handle(raw, "sorted combine test")
                .unwrap()
                .to_float32()
                .unwrap();
            assert_eq!(
                &*actual, &*reference,
                "{hidden} {value_type:?} {score_type:?}"
            );
        }
    }
}

#[test]
fn expert_aligned_prefill_preserves_quantized_projections_and_tile_tails() {
    use crate::array::DType;
    let counts = [0usize, 31, 32, 33, 63, 64, 65, 0, 19];
    let ids: Vec<u32> = counts
        .iter()
        .enumerate()
        .flat_map(|(e, &n)| std::iter::repeat_n(e as u32, n))
        .collect();
    let rows = ids.len();
    let indices = MxArray::from_uint32(&ids, &[rows as i64]).unwrap();
    let Some(tiles) = math::expert_tiles(&indices, counts.len()).unwrap() else {
        return; // The specialized path is unavailable on pre-NAX hardware.
    };
    let spans = tiles.to_uint32().unwrap();
    let mut covered = Vec::new();
    for span in spans.as_chunks::<2>().0.iter().filter(|v| v[0] != v[1]) {
        let (a, b) = (span[0] as usize, span[1] as usize);
        assert!(a < b && b <= rows && b - a <= 64);
        assert!(ids[a..b].iter().all(|&id| id == ids[a]));
        covered.extend(a..b);
    }
    assert_eq!(covered, (0..rows).collect::<Vec<_>>());
    for (mode, bits) in [("q4k", 4usize), ("q5k", 5), ("affine", 5), ("affine", 8)] {
        let (e, n, k) = (counts.len(), 128usize, 256usize);
        let affine = mode == "affine";
        let w = weights::Weight {
            dense_bf16: None,
            values: MxArray::from_uint32(
                &(0..e * n * k * bits / 32)
                    .map(|i| (i as u32).wrapping_mul(0x9e3779b9))
                    .collect::<Vec<_>>(),
                &[(e * n) as i64, (k * bits / 32) as i64],
            )
            .unwrap(),
            scales: Some(if affine {
                MxArray::from_float16(
                    &vec![half::f16::from_f32(0.0013).to_bits(); e * n * k / 32],
                    &[(e * n) as i64, (k / 32) as i64],
                )
                .unwrap()
            } else {
                MxArray::from_uint8(
                    &(0..e * n * k / 32 * 2)
                        .map(|i| (i % 17 + 1) as u8)
                        .collect::<Vec<_>>(),
                    &[(e * n) as i64, (k / 32 * 2) as i64],
                )
                .unwrap()
            }),
            biases: Some(
                MxArray::from_float16(
                    &(0..e * n * if affine { k / 32 } else { k / 256 * 2 })
                        .map(|i| {
                            half::f16::from_f32(if affine {
                                -0.02 + (i % 7) as f32 * 0.001
                            } else {
                                0.00013 + (i % 7) as f32 * 0.0001
                            })
                            .to_bits()
                        })
                        .collect::<Vec<_>>(),
                    &[
                        (e * n) as i64,
                        if affine {
                            (k / 32) as i64
                        } else {
                            (k / 256 * 2) as i64
                        },
                    ],
                )
                .unwrap(),
            ),
            group: 32,
            bits: bits as i32,
            mode: mode.into(),
        };
        let x = MxArray::from_float32(
            &(0..rows * k)
                .map(|i| ((i * 13 % 251) as f32 - 125.) / 128.)
                .collect::<Vec<_>>(),
            &[rows as i64, 1, k as i64],
        )
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap();
        let reference = w
            .expert_rows(&x, &indices, e, true)
            .unwrap()
            .to_float32()
            .unwrap();
        let tiled = w
            .tiled_expert_rows(&x, &indices, &tiles, e)
            .unwrap()
            .expect("supported prefill format");
        let actual = tiled.to_float32().unwrap();
        assert_eq!(&*actual, &*reference, "expert-aligned {mode}/{bits}");
    }
}

#[test]
fn stable_gpu_routing_preserves_duplicates_tails_and_inverse() {
    use crate::array::MxArray;
    for (experts, rows) in [(7, 257), (256, 1024), (512, 10240), (1000, 2049)] {
        let ids: Vec<u32> = (0..rows)
            .map(|i| ((i * 7919 + i / 11) % experts) as u32)
            .collect();
        let input = MxArray::from_uint32(&ids, &[rows as i64]).unwrap();
        let (order, inverse, sorted) = math::route_sort(&input, experts).unwrap().unwrap();
        let mut want: Vec<u32> = (0..rows as u32).collect();
        want.sort_by_key(|&i| ids[i as usize]);
        assert_eq!(&*order.to_uint32().unwrap(), &want);
        assert_eq!(
            &*sorted.to_uint32().unwrap(),
            &want.iter().map(|&i| ids[i as usize]).collect::<Vec<_>>()
        );
        let inverse = inverse.to_uint32().unwrap();
        for (rank, &row) in want.iter().enumerate() {
            assert_eq!(inverse[row as usize], rank as u32);
        }
    }
}

#[test]
fn prefill_bootstrap_uses_dimensions_and_live_headroom() {
    let raw =
        serde_json::from_slice(&std::fs::read(fixture().join("config.json")).unwrap()).unwrap();
    let mut config = config::Config::parse(&raw).unwrap();
    let mut plan = weights::Store::open_metadata(&fixture(), None)
        .unwrap()
        .plan;
    plan.budget = 32 << 30;
    plan.available_bytes = Some(64 << 30);
    plan.physical_bytes = Some(128 << 30);
    let small = memory::prefill_window(&config, &plan).unwrap();
    config.hidden_size = 8192;
    config.hc_count = 4;
    config.moe_intermediate_size = 4096;
    config.num_attention_heads = 32;
    config.head_dim = 256;
    config.num_experts_per_tok = 10;
    config.indexer_budget = 2048;
    let large = memory::prefill_window(&config, &plan).unwrap();
    assert!(large < small);
    let old_ple = config.ple_conv_kernel_size;
    config.ple_conv_kernel_size = 64;
    assert!(
        memory::prefill_window(&config, &plan).unwrap() < large,
        "A dominant PLE stage must reduce the admitted window"
    );
    config.ple_conv_kernel_size = old_ple;
    plan.available_bytes = Some((51 << 30) + (128 << 20));
    assert!(memory::prefill_window(&config, &plan).unwrap() < large);
}

#[test]
fn mtp_cache_only_prefill_matches_full_head_history_and_keeps_target_state() {
    let path = fixture().join("bf16/paged");
    let raw = serde_json::from_slice(&std::fs::read(path.join("config.json")).unwrap()).unwrap();
    let c = config::Config::parse(&raw).unwrap();
    let oracle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("mtp-oracle.json")).unwrap()).unwrap();
    let rows = oracle["rows"].as_array().unwrap();
    let tokens: Vec<u32> = rows
        .iter()
        .map(|r| r["token"].as_u64().unwrap() as u32)
        .collect();
    let hidden: Vec<MxArray> = rows
        .iter()
        .map(|r| {
            let values: Vec<f32> = r["input_hidden"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            MxArray::from_float32(&values, &[1, 1, (c.hc_count * c.hidden_size) as i64])
                .unwrap()
                .astype(crate::array::DType::BFloat16)
                .unwrap()
        })
        .collect();
    for width in [1, 3, 8, 16] {
        let mut reference =
            decoder::Decoder::new(c.clone(), weights::Store::open(&path).unwrap()).unwrap();
        let mut candidate =
            decoder::Decoder::new(c.clone(), weights::Store::open(&path).unwrap()).unwrap();
        candidate.history = vec![3, 9, 7];
        let mut full_cache = decoder::LayerCache::default();
        let mut fast_cache = decoder::LayerCache::default();
        for i in 0..16 {
            reference
                .draft_step(&hidden[i], tokens[i], i, &mut full_cache, false)
                .unwrap();
        }
        for start in (0..16).step_by(width) {
            let end = (start + width).min(16);
            candidate
                .draft_prefill(
                    &hidden[start..end],
                    &tokens[start..end],
                    start,
                    &mut fast_cache,
                )
                .unwrap();
        }
        let (_, a) = reference
            .draft_step(&hidden[16], tokens[16], 16, &mut full_cache, true)
            .unwrap();
        let (_, b) = candidate
            .draft_step(&hidden[16], tokens[16], 16, &mut fast_cache, true)
            .unwrap();
        let a = a.unwrap().to_float32().unwrap();
        let b = b.unwrap().to_float32().unwrap();
        let error = a
            .iter()
            .zip(b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(error < 0.04, "MTP history width={width}, error={error}");
        assert_eq!(candidate.history, [3, 9, 7]);
    }
}

struct CompleteGdnInputs {
    qkv: MxArray,
    z: MxArray,
    a: MxArray,
    b: MxArray,
    conv: MxArray,
    scale: MxArray,
    dt: MxArray,
    norm: MxArray,
    kernel: usize,
}

impl CompleteGdnInputs {
    fn new(kernel: usize, seed: u32) -> Self {
        use crate::array::DType;
        let mut seed = seed;
        let mut make = |shape: &[i64], scale: f32, shift: f32, dtype| {
            let values: Vec<_> = (0..shape.iter().product::<i64>())
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    (seed as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32 * scale + shift
                })
                .collect();
            MxArray::from_float32(&values, shape)
                .unwrap()
                .astype(dtype)
                .unwrap()
        };
        let out = Self {
            qkv: make(&[1, 1, 10240], 2.0, 0., DType::BFloat16),
            z: make(&[1, 1, 48, 128], 4.0, 0., DType::BFloat16),
            a: make(&[1, 1, 48], 20., 0., DType::BFloat16)
                .astype(DType::Float32)
                .unwrap(),
            b: make(&[1, 1, 48], 20., 0., DType::BFloat16)
                .astype(DType::Float32)
                .unwrap(),
            conv: make(&[10240, kernel as i64], 0.25, 0., DType::Float32),
            scale: make(&[48], 0.49, -0.5, DType::Float32),
            dt: make(&[48], 1., 0., DType::Float32),
            norm: make(&[128], 0.5, 1., DType::BFloat16),
            kernel,
        };
        MxArray::eval_arrays(&[
            &out.qkv, &out.z, &out.a, &out.b, &out.conv, &out.scale, &out.dt, &out.norm,
        ])
        .unwrap();
        out
    }

    fn run(
        &self,
        compiled: bool,
        history: &MxArray,
        state: &MxArray,
    ) -> (MxArray, MxArray, MxArray) {
        use crate::array::DType;
        if compiled {
            return math::complete_gdn(
                &self.qkv,
                &self.z,
                &self.a,
                &self.b,
                &self.conv,
                history,
                &self.scale,
                &self.dt,
                state,
                &self.norm,
                1e-6,
            )
            .unwrap();
        }
        let mut history = Some(history.clone());
        let qkv = math::conv(&self.qkv, &self.conv, &mut history, self.kernel, 1).unwrap();
        let q = math::l2(
            &qkv.slice_axis(2, 0, 2048)
                .unwrap()
                .reshape(&[1, 16, 128])
                .unwrap(),
        )
        .unwrap()
        .mul_scalar(128f64.powf(-0.5))
        .unwrap()
        .astype(DType::Float32)
        .unwrap()
        .reshape(&[1, 16, 1, 128])
        .unwrap();
        let k = math::l2(
            &qkv.slice_axis(2, 2048, 4096)
                .unwrap()
                .reshape(&[1, 16, 128])
                .unwrap(),
        )
        .unwrap()
        .astype(DType::Float32)
        .unwrap()
        .reshape(&[1, 16, 1, 128])
        .unwrap();
        let v = qkv
            .slice_axis(2, 4096, 10240)
            .unwrap()
            .reshape(&[1, 48, 128])
            .unwrap()
            .astype(DType::Float32)
            .unwrap();
        let (decay, beta) = math::gdn_gates(&self.a, &self.b, &self.scale, &self.dt).unwrap();
        let (out, next) = math::recurrent_step(
            &q,
            &k,
            &v,
            &decay.reshape(&[1, 48, 1, 1]).unwrap(),
            &beta.reshape(&[1, 48, 1]).unwrap(),
            state,
        )
        .unwrap();
        let out = math::norm(
            &out.astype(DType::BFloat16).unwrap(),
            &self.norm,
            128,
            1e-6,
            false,
        )
        .unwrap()
        .astype(DType::Float32)
        .unwrap();
        let out = math::sigmoid_mul(&self.z.astype(DType::Float32).unwrap(), &out)
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap()
            .reshape(&[1, 1, 6144])
            .unwrap();
        (out, next, history.unwrap())
    }
}

#[test]
fn complete_gdn_replay_preserves_outputs_and_independent_histories() {
    use crate::array::DType;
    if !crate::engine::persistence::compiled_forward_backend_available() {
        return;
    }
    for kernel in [2, 4, 8] {
        let inputs = [
            CompleteGdnInputs::new(kernel, 71),
            CompleteGdnInputs::new(kernel, 197),
        ];
        let state = MxArray::zeros(&[1, 48, 128, 128], Some(DType::Float32)).unwrap();
        let history = MxArray::zeros(&[kernel as i64 - 1, 10240], Some(DType::BFloat16)).unwrap();
        let mut refs = [
            (history.clone(), state.clone()),
            (history.clone(), state.clone()),
        ];
        let mut compiled = refs.clone();
        // Interleave two owners through one compiled trace. Changing both weights
        // and inputs also detects accidental capture of a prior layer's arrays.
        for step in 0..24 {
            let owner = step % 2;
            let x = &inputs[(step / 2) % 2];
            let (a, sa, ha) = x.run(false, &refs[owner].0, &refs[owner].1);
            let (b, sb, hb) = x.run(true, &compiled[owner].0, &compiled[owner].1);
            for (name, a, b) in [
                ("output", &a, &b),
                ("state", &sa, &sb),
                ("history", &ha, &hb),
            ] {
                let a = a.to_float32().unwrap();
                let b = b.to_float32().unwrap();
                let error = a
                    .iter()
                    .zip(b.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                let different = a.iter().zip(b.iter()).filter(|(a, b)| a != b).count();
                assert!(a.iter().chain(b.iter()).all(|v| v.is_finite()));
                assert_eq!(
                    different, 0,
                    "kernel={kernel}, step={step}, {name}, max error={error}"
                );
            }
            refs[owner] = (ha, sa);
            compiled[owner] = (hb, sb);
        }
        // Published outputs did not mutate the input snapshot retained above.
        assert!(state.to_float32().unwrap().iter().all(|&v| v == 0.));
        assert!(history.to_float32().unwrap().iter().all(|&v| v == 0.));
    }
}

#[test]
fn wide_causal_windows_match_singleton_with_pages_media_and_retained_prefix() {
    let path = fixture().join("bf16/paged");
    let raw = serde_json::from_slice(&std::fs::read(path.join("config.json")).unwrap()).unwrap();
    let mut c = config::Config::parse(&raw).unwrap();
    c.indexer_budget = 2048;
    c.max_position_embeddings = 2048;
    let mut reference =
        decoder::Decoder::new(c.clone(), weights::Store::open(&path).unwrap()).unwrap();
    reference.batch_prefill = false;
    let mut candidate =
        decoder::Decoder::new(c.clone(), weights::Store::open(&path).unwrap()).unwrap();
    let mut pages = paged::create(&c).unwrap();
    pages.begin_request(1).unwrap();
    candidate.paged = Some(pages);
    let positions: Vec<_> = (0..256)
        .map(|i| [i as i64 / 3, i as i64 / 7, i as i64 % 11])
        .collect();
    reference.positions = positions.clone();
    candidate.positions = positions;
    reference.rope_delta = -4;
    candidate.rope_delta = -4;
    for count in [7, 65, 128, 3] {
        let tokens: Vec<u32> = (0..count).map(|i| (i % 11 + 1) as u32).collect();
        let a = reference
            .prefill_chunk(&tokens, None, true)
            .unwrap()
            .to_float32()
            .unwrap();
        let b = candidate
            .prefill_chunk(&tokens, None, true)
            .unwrap()
            .to_float32()
            .unwrap();
        let error = a
            .iter()
            .zip(b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            error < 0.04,
            "wide causal window={count}, max logit error={error}"
        );
        assert_eq!(
            candidate.paged.as_ref().unwrap().request_tokens(),
            reference.history
        );
    }
    let a = reference.step(9).unwrap().to_float32().unwrap();
    let b = candidate.step(9).unwrap().to_float32().unwrap();
    let error = a
        .iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(error < 0.04, "wide-window continuation error={error}");
}

#[test]
#[cfg(target_os = "macos")]
fn paged_compiled_writes_retain_offset_projection_views() {
    assert_eq!(
        unsafe { mlx_sys::mlx_paged_kv_write_compile_offset_views_check() },
        1
    );
}

#[test]
fn paged_writes_accept_offset_projection_views() {
    let path = fixture().join("bf16/paged");
    let raw = serde_json::from_slice(&std::fs::read(path.join("config.json")).unwrap()).unwrap();
    let c = config::Config::parse(&raw).unwrap();
    let mut pages = paged::create(&c).unwrap();
    pages.begin_request(1).unwrap();
    let h = c.num_key_value_heads;
    let d = c.head_dim;
    for n in [1usize, 33] {
        let base = pages.request_tokens().len();
        pages.record_tokens(&vec![3; n]).unwrap();
        let backing = MxArray::from_float32(
            &(0..2 * n * h * d)
                .map(|i| (i % 127) as f32 / 16.)
                .collect::<Vec<_>>(),
            &[(2 * n) as i64, h as i64, d as i64],
        )
        .unwrap()
        .astype(crate::array::DType::BFloat16)
        .unwrap();
        MxArray::eval_arrays(&[&backing]).unwrap();
        let view = backing.slice_axis(0, n as i64, (2 * n) as i64).unwrap();
        let expected = view
            .transpose(Some(&[1, 0, 2]))
            .unwrap()
            .to_float32()
            .unwrap()
            .to_vec();
        paged::write_rows(&mut pages, 0, &view, &view, base as u32).unwrap();
        drop(view);
        drop(backing);
        pages.eval_pending_pool_writes().unwrap();
        let ids = (base..base + n).map(|i| i as i32).collect::<Vec<_>>();
        let (k, v) = paged::gather_selected(&pages, 0, &ids, h, d).unwrap();
        assert_eq!(&*k.to_float32().unwrap(), &*expected);
        assert_eq!(&*v.to_float32().unwrap(), &*expected);
    }
}

#[test]
fn convolution_history_releases_wide_prefill_storage() {
    // Run serialized with the GPU suite. Check live allocation ownership,
    // not allocator cache size; a view would retain > 2 MiB here.
    crate::array::memory::synchronize();
    let before = crate::array::memory::get_active_memory();
    let mut history = None;
    {
        let x = MxArray::ones(&[1, 512, 1024], Some(crate::array::DType::Float32)).unwrap();
        let weights = MxArray::ones(&[1024, 4], Some(crate::array::DType::Float32)).unwrap();
        let output = math::conv_sequence(&x, &weights, &mut history, 4).unwrap();
        MxArray::eval_arrays(&[&output]).unwrap();
    }
    crate::array::memory::synchronize();
    let retained = crate::array::memory::get_active_memory() - before;
    assert!(
        retained < 256. * 1024.,
        "convolution history retained {retained} bytes"
    );
    assert_eq!(
        history.as_ref().unwrap().shape().unwrap().as_ref(),
        &[3, 1024]
    );
    assert!(
        history
            .unwrap()
            .to_float32()
            .unwrap()
            .iter()
            .all(|&x| x == 1.)
    );
}

#[test]
fn indirect_prefill_preserves_mixed_formats_rows_and_tail_rounding() {
    use crate::array::DType;
    use std::sync::Arc;
    let counts = [0usize, 15, 16, 17, 31, 32, 33, 63, 64, 65];
    let e = counts.len();
    let (h, m, tokens) = (256usize, 128usize, 53usize);
    let ids: Vec<u32> = counts
        .iter()
        .enumerate()
        .flat_map(|(id, &n)| std::iter::repeat_n(id as u32, n))
        .collect();
    let rows = ids.len();
    let ids = MxArray::from_uint32(&ids, &[rows as i64]).unwrap();
    let Some(tiles) = math::expert_tiles(&ids, e).unwrap() else {
        return;
    };
    let source = MxArray::from_float32(
        &(0..tokens * h)
            .map(|i| ((i * 13 % 251) as f32 - 125.) / 128.)
            .collect::<Vec<_>>(),
        &[tokens as i64, h as i64],
    )
    .unwrap()
    .astype(DType::BFloat16)
    .unwrap();
    let token_rows = MxArray::from_uint32(
        &(0..rows)
            .map(|i| ((i * 37 + 11) % tokens) as u32)
            .collect::<Vec<_>>(),
        &[rows as i64],
    )
    .unwrap();
    let expanded = source
        .take(&token_rows, 0)
        .unwrap()
        .reshape(&[rows as i64, 1, h as i64])
        .unwrap();
    let bank = |n: usize, k: usize, bits: usize, affine: bool, seed: u32| {
        Arc::new(weights::Weight {
            values: MxArray::from_uint32(
                &(0..e * n * k * bits / 32)
                    .map(|i| (i as u32).wrapping_mul(0x9e3779b9).wrapping_add(seed))
                    .collect::<Vec<_>>(),
                &[(e * n) as i64, (k * bits / 32) as i64],
            )
            .unwrap(),
            scales: Some(if affine {
                MxArray::from_float16(
                    &vec![half::f16::from_f32(0.0013).to_bits(); e * n * k / 32],
                    &[(e * n) as i64, (k / 32) as i64],
                )
                .unwrap()
            } else {
                MxArray::from_uint8(
                    &(0..e * n * k / 16)
                        .map(|i| ((i + seed as usize) % 63 + 1) as u8)
                        .collect::<Vec<_>>(),
                    &[(e * n) as i64, (k / 16) as i64],
                )
                .unwrap()
            }),
            biases: Some(
                MxArray::from_float16(
                    &(0..e * n * if affine { k / 32 } else { k / 128 })
                        .map(|i| {
                            half::f16::from_f32(if affine {
                                -0.02 + (i % 7) as f32 * 0.001
                            } else {
                                0.00013 + (i % 7) as f32 * 0.0001
                            })
                            .to_bits()
                        })
                        .collect::<Vec<_>>(),
                    &[
                        (e * n) as i64,
                        if affine {
                            (k / 32) as i64
                        } else {
                            (k / 128) as i64
                        },
                    ],
                )
                .unwrap(),
            ),
            dense_bf16: None,
            group: 32,
            bits: bits as i32,
            mode: if affine {
                "affine".into()
            } else {
                format!("q{bits}k")
            },
        })
    };
    for gb in [4, 5] {
        for db in [5, 8] {
            let banks = [
                bank(m, h, gb, false, 71),
                bank(m, h, gb, false, 197),
                bank(h, m, db, true, 13),
            ];
            let gate = banks[0]
                .tiled_expert_rows(&expanded, &ids, &tiles, e)
                .unwrap()
                .unwrap();
            let up = banks[1]
                .tiled_expert_rows(&expanded, &ids, &tiles, e)
                .unwrap()
                .unwrap();
            let hidden = math::swiglu(&gate, &up).unwrap();
            let expected = banks[2]
                .tiled_expert_rows(&hidden, &ids, &tiles, e)
                .unwrap()
                .unwrap()
                .to_float32()
                .unwrap();
            let actual = math::prefill_indirect(&source, &ids, &token_rows, &banks, e)
                .unwrap()
                .expect("supported NAX format")
                .to_float32()
                .unwrap();
            assert_eq!(&*actual, &*expected, "indirect gate={gb} down={db}");
            // The one-bank routing path must preserve duplicate assignments,
            // inactive experts, offset views and the original reduction order.
            // These sizes exercise both indirect and ordinary sorted dispatch.
            for n in [9i64, 27, 53] {
                let top = 10usize;
                let assignments = n as usize * top;
                let routes = MxArray::from_uint32(
                    &(0..assignments * 2)
                        .map(|i| ((i * 7 + i / 13) % (e - 1) + 1) as u32)
                        .collect::<Vec<_>>(),
                    &[(assignments * 2) as i64],
                )
                .unwrap()
                .slice_axis(0, assignments as i64, (assignments * 2) as i64)
                .unwrap();
                let x = source
                    .slice_axis(0, 0, n)
                    .unwrap()
                    .reshape(&[1, n, h as i64])
                    .unwrap();
                let rows = MxArray::from_int32(
                    &(0..assignments)
                        .map(|i| (i / top) as i32)
                        .collect::<Vec<_>>(),
                    &[assignments as i64],
                )
                .unwrap();
                let expected_rows = decoder::Decoder::expert_assignment_rows(
                    &x.reshape(&[n, h as i64]).unwrap().take(&rows, 0).unwrap(),
                    &routes,
                    &banks,
                    e,
                )
                .unwrap();
                let scores = MxArray::from_float32(
                    &(0..assignments)
                        .map(|i| (i % 11 + 1) as f32 * 0.0113)
                        .collect::<Vec<_>>(),
                    &[1, assignments as i64, 1],
                )
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
                let (output, inverse) =
                    decoder::Decoder::single_slot_group_rows(&x, &routes, &banks, e, e - 1, top)
                        .unwrap();
                let expected = math::combine_expert_rows(&expected_rows, &scores, None, top)
                    .unwrap()
                    .to_float32()
                    .unwrap();
                let actual = math::combine_expert_rows(&output, &scores, inverse.as_ref(), top)
                    .unwrap()
                    .to_float32()
                    .unwrap();
                assert_eq!(
                    &*actual, &*expected,
                    "one slot group tokens={n} gate={gb} down={db}"
                );
            }
            assert!(
                math::prefill_indirect(
                    &source.astype(DType::Float32).unwrap(),
                    &ids,
                    &token_rows,
                    &banks,
                    e,
                )
                .unwrap()
                .is_none()
            );
        }
    }
}

#[test]
fn compact_dense_prefill_preserves_affine_rounding_views_and_fallbacks() {
    use crate::array::DType;
    use crate::models::qwen3_5::quantized_linear::QuantizedLinear;
    if std::env::var("MLX_ENABLE_TF32").as_deref() == Ok("0") {
        return;
    }
    let ids = MxArray::from_uint32(&vec![0; 256], &[256]).unwrap();
    if math::expert_tiles(&ids, 1).unwrap().is_none() {
        return; // This primitive requires the NAX backend.
    }
    let (n, k) = (2048usize, 320usize);
    let w = MxArray::from_uint32(
        &(0..n * k / 4)
            .map(|i| (i as u32).wrapping_mul(0x9e3779b9))
            .collect::<Vec<_>>(),
        &[n as i64, (k / 4) as i64],
    )
    .unwrap();
    let scales = MxArray::from_float32(
        &(0..n * k / 32)
            .map(|i| ((i * 13 % 61) as f32 + 1.) / 65536.)
            .collect::<Vec<_>>(),
        &[n as i64, (k / 32) as i64],
    )
    .unwrap()
    .astype(DType::Float16)
    .unwrap();
    let biases = scales
        .mul_scalar(-128.)
        .unwrap()
        .astype(DType::Float16)
        .unwrap();
    let reference = QuantizedLinear::new(
        w.clone(),
        scales.clone(),
        Some(biases.clone()),
        None,
        32,
        8,
        "affine".into(),
    );
    for rows in [256usize, 257, 692, 1024] {
        let full = MxArray::from_float32(
            &(0..(rows + 1) * k)
                .map(|i| ((i * 17 % 251) as f32 - 125.) / 128.)
                .collect::<Vec<_>>(),
            &[(rows + 1) as i64, k as i64],
        )
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap();
        // Nonzero-offset view and a 3D caller exercise the real projection layouts.
        let x = full
            .slice_axis(0, 1, (rows + 1) as i64)
            .unwrap()
            .reshape(&[1, rows as i64, k as i64])
            .unwrap();
        let run = |input: &MxArray| unsafe {
            mlx_sys::mlx_qwen4_dense_prefill(
                input.as_raw_ptr(),
                w.as_raw_ptr(),
                scales.as_raw_ptr(),
                biases.as_raw_ptr(),
            )
        };
        let y = MxArray::from_handle(run(&x), "compact dense test").unwrap();
        assert_eq!(&*y.shape().unwrap(), &[1, rows as i64, n as i64]);
        assert_eq!(y.dtype().unwrap(), DType::BFloat16);
        assert_eq!(
            &*y.to_float32().unwrap(),
            &*reference.forward(&x).unwrap().to_float32().unwrap(),
            "rows={rows}"
        );
        assert!(run(&x.astype(DType::Float32).unwrap()).is_null());
        assert!(run(&x.slice_axis(1, 0, 1).unwrap()).is_null());
    }
    // Existing split-K and smaller shared projections must keep their dispatch.
    let small_w = w.slice_axis(0, 0, 320).unwrap();
    let small_s = scales.slice_axis(0, 0, 320).unwrap();
    let small_b = biases.slice_axis(0, 0, 320).unwrap();
    let x = MxArray::zeros(&[256, k as i64], Some(DType::BFloat16)).unwrap();
    assert!(
        unsafe {
            mlx_sys::mlx_qwen4_dense_prefill(
                x.as_raw_ptr(),
                small_w.as_raw_ptr(),
                small_s.as_raw_ptr(),
                small_b.as_raw_ptr(),
            )
        }
        .is_null()
    );
    let x = MxArray::zeros(&[1024, 8192], Some(DType::BFloat16)).unwrap();
    let w = MxArray::zeros(&[256, 2048], Some(DType::Uint32)).unwrap();
    let sides = MxArray::zeros(&[256, 256], Some(DType::Float16)).unwrap();
    assert!(
        unsafe {
            mlx_sys::mlx_qwen4_dense_prefill(
                x.as_raw_ptr(),
                w.as_raw_ptr(),
                sides.as_raw_ptr(),
                sides.as_raw_ptr(),
            )
        }
        .is_null(),
        "two-partition accumulation must retain the existing fallback"
    );
}
