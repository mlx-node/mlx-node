//! Smoke tests for the block-paged adapter construction on Qwen3.5
//! dense. The forward dispatch lives in `paged_turn_sync_core`
//! / `paged_turn_stream_core`; these tests cover the
//! Inner-construction surface in isolation.
//!
//! Tests that allocate a `LayerKVPool` require Metal. Construction-only
//! cases are `#[ignore]`-marked behind `MLX_TEST_PAGED=1`; forward-path
//! checks are also ignored because no-Metal hosts can abort inside MLX
//! before Rust receives an `Err`.

use super::*;
use crate::array::DType;
use crate::models::qwen3_5::config::Qwen3_5Config;
use crate::models::qwen3_5::decoder_layer::{self, AttentionType};
use crate::models::qwen3_5::quantized_linear::{
    MXFP8_BITS, MXFP8_GROUP_SIZE, MXFP8_MODE, QuantizedLinear,
};

fn tiny_cfg(use_block_paged: bool) -> Qwen3_5Config {
    Qwen3_5Config {
        qwen35_gguf_gdn_layout: None,
        vocab_size: 1024,
        hidden_size: 64,
        num_layers: 8,
        num_heads: 4,
        num_kv_heads: 2,
        intermediate_size: 128,
        rms_norm_eps: 1e-6,
        head_dim: 16,
        tie_word_embeddings: true,
        attention_bias: false,
        max_position_embeddings: 1024,
        pad_token_id: 0,
        eos_token_id: 0,
        bos_token_id: 0,
        linear_num_value_heads: 4,
        linear_num_key_heads: 2,
        linear_key_head_dim: 16,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        full_attention_interval: 4,
        partial_rotary_factor: 0.25,
        rope_theta: 100_000.0,
        paged_cache_memory_mb: Some(64),
        paged_cache_initial_memory_mb: None,
        paged_block_size: Some(16),
        use_block_paged_cache: if use_block_paged { Some(true) } else { None },
        persist_paged_cache: None,
        n_mtp_layers: 0,
    }
}

fn tiny_paged_forward_cfg() -> Qwen3_5Config {
    let mut cfg = tiny_cfg(true);
    // Paged attention's Metal kernels require head_dim=32+, so the
    // production-forward tests use a separate, larger shape.
    cfg.hidden_size = 128;
    cfg.intermediate_size = 256;
    cfg.head_dim = 32;
    cfg.linear_key_head_dim = 32;
    cfg.linear_value_head_dim = 32;
    cfg.paged_cache_memory_mb = Some(256);
    cfg
}

fn expected_scheduler_window(inner: &Qwen35Inner) -> (u32, u32, u32) {
    use crate::engine::hybrid_scheduler::{
        pool_tokens_after_recurrent, scheduled_turn_context, scheduler_per_seq_context_override,
    };
    let trained = inner.config.max_position_embeddings.max(0) as u32;
    let adapter = inner
        .paged_adapter
        .as_ref()
        .expect("test adapter must be installed");
    let pool = adapter.max_capacity_tokens();
    let usable = pool_tokens_after_recurrent(
        pool,
        adapter.block_size(),
        adapter.bytes_per_block().unwrap_or(0),
        inner.config.recurrent_state_bytes(),
    );
    (
        trained,
        scheduled_turn_context(trained, usable, scheduler_per_seq_context_override()),
        pool,
    )
}

#[test]
fn tiny_hybrid_recurrent_bytes_make_usable_window_stricter_than_raw_pool() {
    use crate::engine::hybrid_scheduler::{pool_tokens_after_recurrent, scheduled_turn_context};
    let inner = Qwen35Inner::new(tiny_cfg(false)).expect("construct tiny dense model");
    let rec = inner.config.recurrent_state_bytes();
    assert!(rec > 0, "tiny hybrid config must have GDN state");
    let trained = 1_048_576;
    let pool = 349_520;
    let usable = pool_tokens_after_recurrent(pool, 16, 1024, rec);
    assert!(usable < pool);
    assert_ne!(
        scheduled_turn_context(trained, usable, None),
        trained.min(pool)
    );
    assert_eq!(
        scheduled_turn_context(trained, usable, Some(32_768)),
        32_768
    );
}

#[test]
fn paged_context_limits_and_preflight_use_scheduler_usable_window() {
    let Some(inner) = dense_inner_with_test_adapter_or_skip(
        "paged_context_limits_and_preflight_use_scheduler_usable_window",
    ) else {
        return;
    };
    assert!(inner.config.recurrent_state_bytes() > 0);
    let (trained, expected, pool) = expected_scheduler_window(&inner);
    let (published_trained, effective, _, _) = inner.paged_context_limits();
    assert_eq!(published_trained, trained);
    assert_eq!(effective, expected);
    assert!(
        effective < pool,
        "recurrent charge must shrink the published window below raw pool tokens"
    );
    assert_ne!(effective, trained.min(pool));

    let mut params = extract_chat_params(&crate::engine::types::ChatConfig {
        max_new_tokens: Some(32),
        ..crate::engine::types::ChatConfig::default()
    });
    inner
        .preflight_paged_context(effective as usize, &mut params)
        .expect("prompt at the scheduler window must pass");
    let err = inner
        .preflight_paged_context(pool as usize, &mut params)
        .expect_err("prompt at raw pool tokens must fail after recurrent charge");
    assert!(
        err.reason
            .starts_with("context_length_exceeded: rendered prompt has"),
        "{}",
        err.reason
    );
}

#[test]
fn scheduled_recurrent_cap_counts_the_active_row_and_parked_rows() {
    let mut inner = Qwen35Inner::new(tiny_cfg(false)).expect("construct tiny dense model");
    let bytes = inner.config.recurrent_state_bytes();

    inner
        .activate_scheduled_recurrent(1)
        .expect("activate first row");
    assert_eq!(inner.scheduled_recurrent_units(), 1);
    assert_eq!(inner.scheduled_recurrent_bytes(), bytes);

    inner
        .activate_scheduled_recurrent(2)
        .expect("park first and activate second row");
    assert_eq!(inner.scheduled_recurrent_units(), 2);
    assert_eq!(inner.scheduled_recurrent_bytes(), bytes * 2);
    assert!(
        inner.activate_scheduled_recurrent(3).is_err(),
        "the active row must count toward the two-unit cap"
    );

    inner.release_scheduled_recurrent_for(1);
    inner
        .activate_scheduled_recurrent(3)
        .expect("an idle row release opens exactly one slot");
    assert_eq!(inner.scheduled_recurrent_units(), 2);
    assert_eq!(inner.scheduled_recurrent_bytes(), bytes * 2);
}

#[test]
fn all_full_attention_recurrent_lifecycle_is_a_noop() {
    let mut config = tiny_cfg(false);
    config.full_attention_interval = 1;
    let mut inner = Qwen35Inner::new(config).expect("construct all-full dense model");
    assert_eq!(inner.config.recurrent_state_bytes(), 0);

    inner
        .activate_scheduled_recurrent(7)
        .expect("activate empty recurrent shell");
    assert_eq!(inner.active_scheduled_seq, Some(7));
    inner
        .park_active_scheduled_recurrent()
        .expect("zero-byte recurrent state must park as a no-op");

    assert_eq!(inner.active_scheduled_seq, None);
    assert!(
        inner.caches.is_none(),
        "the empty recurrent shell is dropped"
    );
    assert_eq!(inner.scheduled_recurrent_units(), 0);
    assert_eq!(inner.scheduled_recurrent_bytes(), 0);
    assert!(!inner.has_scheduled_recurrent(7));
    assert!(inner.can_activate_scheduled_recurrent(8));
    inner
        .validate_scheduled_decode_residency(&[(7, 11), (8, 13)])
        .expect("attention-only decode must not require recurrent rows");
    assert!(
        inner
            .scheduled_decode_recurrent_snapshots(&[(7, 11), (8, 13)])
            .expect("attention-only decode must not snapshot recurrent rows")
            .is_empty()
    );
}

#[test]
fn save_model_rejects_quantized_dense_projection_before_creating_destination() {
    let mut inner = Qwen35Inner::new(tiny_cfg(false)).expect("construct tiny dense model");
    let weight = MxArray::zeros(&[128, 16], Some(DType::Uint32)).unwrap();
    let scales = MxArray::zeros(&[128, 2], Some(DType::Uint8)).unwrap();
    let quantized = QuantizedLinear::new(
        weight,
        scales,
        None,
        None,
        MXFP8_GROUP_SIZE,
        MXFP8_BITS,
        MXFP8_MODE.to_string(),
    );
    match &mut inner.layers[3].attn {
        AttentionType::Full(attn) => attn.set_quantized_q_proj(quantized),
        AttentionType::Linear(_) => panic!("layer 3 must be full attention"),
    }

    let destination = std::env::temp_dir().join(format!(
        "mlx_node_dense_quant_save_reject_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    assert!(!destination.exists());
    let err = inner
        .save_model_sync(destination.to_str().unwrap())
        .expect_err("quantized main-model projection must reject dense-only save");
    let message = err.reason.to_string();
    assert!(message.contains("dense/BF16-only"), "{message}");
    assert!(message.contains("layers.3"), "{message}");
    assert!(message.contains("before creating"), "{message}");
    assert!(
        !destination.exists(),
        "rejected save must not create its destination directory"
    );
}

fn paged_inner_or_skip(test_name: &str) -> Option<(Qwen35Inner, Qwen3_5Config)> {
    paged_inner_with_cfg_or_skip(test_name, tiny_paged_forward_cfg())
}

fn paged_inner_with_cfg_or_skip(
    test_name: &str,
    cfg: Qwen3_5Config,
) -> Option<(Qwen35Inner, Qwen3_5Config)> {
    // Only a device-less host licenses a skip. A `LayerKVPool::new: ...
    // must be > 0` means the test config stopped producing a usable pool,
    // and skipping on that is a silent green — see `metal_device_absent`.
    let unavailable = crate::test_support::metal_device_absent;
    match Qwen35Inner::new(cfg.clone()) {
        Ok(mut inner) => match inner.initialize_paged_adapter() {
            Ok(()) => Some((inner, cfg)),
            Err(err) => {
                let msg = err.reason.to_string();
                if unavailable(&msg) {
                    eprintln!("skipping {test_name} (paged adapter unavailable): {msg}");
                    None
                } else {
                    panic!("unexpected paged init failure in {test_name}: {msg}");
                }
            }
        },
        Err(err) => {
            let msg = err.reason.to_string();
            if unavailable(&msg) {
                eprintln!("skipping {test_name} (paged adapter unavailable): {msg}");
                None
            } else {
                panic!("unexpected Qwen35Inner::new failure in {test_name}: {msg}");
            }
        }
    }
}

fn cast_qwen35_inner_weights_bf16(inner: &mut Qwen35Inner) {
    let cast = |a: &MxArray| -> MxArray { a.astype(DType::BFloat16).expect("astype bf16") };

    let w = inner.embedding.get_weight();
    inner.embedding.set_weight(&cast(&w)).expect("set embed");

    let w = inner.final_norm.get_weight();
    inner
        .final_norm
        .set_weight(&cast(&w))
        .expect("set final_norm");

    if let Some(head) = inner.lm_head.as_mut() {
        let w = head.get_weight();
        head.set_weight(&cast(&w), "lm_head").expect("set lm_head");
    }

    for layer in inner.layers.iter_mut() {
        let w = layer.get_input_layernorm_weight();
        layer
            .set_input_layernorm_weight(&cast(&w))
            .expect("set input_layernorm");
        let w = layer.get_post_attention_layernorm_weight();
        layer
            .set_post_attention_layernorm_weight(&cast(&w))
            .expect("set post_attention_layernorm");

        match &mut layer.attn {
            AttentionType::Linear(gdn) => {
                let w = gdn.get_dt_bias();
                gdn.set_dt_bias(&cast(&w));
                let w = gdn.get_a_log();
                gdn.set_a_log(&cast(&w)).expect("set a_log");
                let w = gdn.get_in_proj_qkvz_weight();
                gdn.set_in_proj_qkvz_weight(&cast(&w))
                    .expect("set in_proj_qkvz");
                let w = gdn.get_in_proj_ba_weight();
                gdn.set_in_proj_ba_weight(&cast(&w))
                    .expect("set in_proj_ba");
                let w = gdn.get_conv1d_weight();
                gdn.set_conv1d_weight(&cast(&w)).expect("set conv1d");
                let w = gdn.get_norm_weight();
                gdn.set_norm_weight(&cast(&w)).expect("set gdn norm");
                let w = gdn.get_out_proj_weight();
                gdn.set_out_proj_weight(&cast(&w)).expect("set out_proj");
            }
            AttentionType::Full(attn) => {
                let w = attn.get_q_proj_weight();
                attn.set_q_proj_weight(&cast(&w)).expect("set q_proj");
                let w = attn.get_k_proj_weight();
                attn.set_k_proj_weight(&cast(&w)).expect("set k_proj");
                let w = attn.get_v_proj_weight();
                attn.set_v_proj_weight(&cast(&w)).expect("set v_proj");
                let w = attn.get_o_proj_weight();
                attn.set_o_proj_weight(&cast(&w)).expect("set o_proj");
                let w = attn.get_q_norm_weight();
                attn.set_q_norm_weight(&cast(&w)).expect("set q_norm");
                let w = attn.get_k_norm_weight();
                attn.set_k_norm_weight(&cast(&w)).expect("set k_norm");
            }
        }

        let w = layer.mlp.get_gate_proj_weight();
        layer
            .mlp
            .set_gate_proj_weight(&cast(&w))
            .expect("set gate_proj");
        let w = layer.mlp.get_up_proj_weight();
        layer
            .mlp
            .set_up_proj_weight(&cast(&w))
            .expect("set up_proj");
        let w = layer.mlp.get_down_proj_weight();
        layer
            .mlp
            .set_down_proj_weight(&cast(&w))
            .expect("set down_proj");
    }
}

#[test]
#[ignore = "requires Metal GPU; run with --ignored"]
fn dense_hybrid_n2_batched_decode_matches_scalar_replay() {
    let Some((mut inner, cfg)) =
        paged_inner_or_skip("dense_hybrid_n2_batched_decode_matches_scalar_replay")
    else {
        return;
    };
    cast_qwen35_inner_weights_bf16(&mut inner);
    let prompt = vec![7, 11, 13, 17];
    for seq_id in [101, 202] {
        inner
            .activate_scheduled_recurrent(seq_id)
            .expect("activate recurrent row");
        inner.set_cache_owner_id(&format!("owner-{seq_id}"), None);
        let prefix = inner
            .prime_prefix_state(&prompt, true, 16, &[], seq_id as u64)
            .expect("prime request");
        inner
            .paged_prefill(
                &prompt[prefix.effective_cached_prefix_len..],
                &prefix,
                Stream::new(DeviceType::Gpu),
            )
            .expect("prefill request")
            .eval();
        inner
            .park_active_scheduled_recurrent()
            .expect("park recurrent row");
    }

    let snapshots = [101, 202]
        .into_iter()
        .map(|seq_id| {
            let state = inner
                .scheduled_recurrent
                .live(seq_id)
                .expect("prefilled recurrent row");
            let snapshot =
                crate::models::qwen3_5::paged_forward::snapshot_materialized_linear_layer_caches(
                    state,
                )
                .expect("materialized GDN state");
            (seq_id, snapshot)
        })
        .collect::<Vec<_>>();
    let decode_rows = [(101, 19), (202, 23)];
    let batched_started = Instant::now();
    let batched = inner
        .run_paged_decode_step_batched(&decode_rows)
        .expect("batched hybrid decode");
    assert_eq!(
        batched.shape().unwrap().as_ref(),
        [2, 1, cfg.vocab_size as i64]
    );
    let batched_tokens = batched
        .argmax(-1, Some(false))
        .unwrap()
        .to_uint32()
        .unwrap()
        .to_vec();
    let batched_elapsed = batched_started.elapsed();

    for &(seq_id, _) in decode_rows.iter().rev() {
        let adapter = inner.paged_adapter.as_mut().unwrap();
        adapter.activate_request(seq_id).unwrap();
        adapter.rollback_last_tokens(1).unwrap();
    }
    for (seq_id, snapshot) in snapshots {
        inner
            .scheduled_recurrent
            .insert_live(seq_id, inner.config.recurrent_state_bytes(), snapshot)
            .expect("restore GDN snapshot");
    }

    let serial_started = Instant::now();
    let mut serial_rows = Vec::new();
    for (seq_id, token_id) in decode_rows {
        inner
            .activate_paged_seq(seq_id)
            .expect("activate scalar row");
        let embed = inner.embedding.clone();
        let logits = {
            let caches = inner.caches.as_mut().expect("active GDN state");
            let adapter = inner.paged_adapter.as_mut().expect("paged adapter");
            crate::models::qwen3_5::paged_forward::run_paged_decode_step(
                token_id,
                &embed,
                &mut inner.layers,
                caches,
                &inner.final_norm,
                &inner.lm_head,
                &inner.layer_kinds,
                adapter,
                0,
            )
            .expect("scalar hybrid replay")
        };
        serial_rows.push(logits);
        inner
            .park_active_scheduled_recurrent()
            .expect("park scalar row");
    }
    let serial = MxArray::concatenate_many(serial_rows.iter().collect(), Some(0)).unwrap();
    let serial_tokens = serial
        .argmax(-1, Some(false))
        .unwrap()
        .to_uint32()
        .unwrap()
        .to_vec();
    let serial_elapsed = serial_started.elapsed();
    assert_eq!(batched_tokens, serial_tokens);
    eprintln!(
        "qwen3.5 N=2 decode microbench: fused={:.3}ms exclusive={:.3}ms speedup={:.2}x",
        batched_elapsed.as_secs_f64() * 1_000.0,
        serial_elapsed.as_secs_f64() * 1_000.0,
        serial_elapsed.as_secs_f64() / batched_elapsed.as_secs_f64().max(f64::EPSILON),
    );
}

fn reset_paged_request(inner: &mut Qwen35Inner, prompt: &[u32]) {
    inner.caches = Some(
        (0..inner.config.num_layers as usize)
            .map(|i| {
                if inner.config.is_linear_layer(i) {
                    Qwen3_5LayerCache::new_linear()
                } else {
                    Qwen3_5LayerCache::new_full_attention()
                }
            })
            .collect(),
    );

    let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
    if adapter.block_table().is_some() {
        adapter.release_request().expect("release_request");
    }
    adapter.reset_for_new_request(0).expect("reset request");
    let prefix = adapter
        .find_cached_prefix(prompt, &[], 0, false)
        .expect("find_cached_prefix");
    assert_eq!(
        prefix.cached_token_count, 0,
        "dense chunking tests must start from a cold adapter prefix"
    );
    adapter
        .allocate_suffix_blocks(prompt.len() as u32)
        .expect("allocate suffix blocks");
}

fn run_dense_paged_prefill_with_size(
    inner: &mut Qwen35Inner,
    full_tokens: &[u32],
    suffix_tokens: &[u32],
    cached_prefix_len: u32,
    chunk_size: i32,
) -> Result<MxArray> {
    run_dense_paged_prefill_with_size_and_checkpoint(
        inner,
        full_tokens,
        suffix_tokens,
        cached_prefix_len,
        chunk_size,
    )
    .map(|(logits, _checkpoint)| logits)
}

fn run_dense_paged_prefill_with_size_and_checkpoint(
    inner: &mut Qwen35Inner,
    full_tokens: &[u32],
    suffix_tokens: &[u32],
    cached_prefix_len: u32,
    chunk_size: i32,
) -> Result<(
    MxArray,
    Vec<crate::models::qwen3_5::paged_forward::MaterializedGdnPrefixCheckpoint>,
)> {
    let layer_kinds = decoder_layer::compute_layer_kinds(inner.config.num_layers as usize, |i| {
        inner.config.is_linear_layer(i)
    });
    let embed = inner.embedding.clone();
    let caches = inner.caches.as_mut().expect("qwen35 caches initialized");
    let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");

    crate::models::qwen3_5::paged_forward::run_paged_prefill_chunk_with_size(
        full_tokens,
        suffix_tokens,
        cached_prefix_len,
        false,
        &embed,
        &mut inner.layers,
        caches,
        &inner.final_norm,
        &inner.lm_head,
        &layer_kinds,
        adapter,
        chunk_size,
        /* cached_rope_deltas */ 0,
        None,
    )
}

/// The planned-MTP twin of [`run_dense_paged_prefill_with_size_and_checkpoint`].
///
/// `run_dense_core_paged_prefill` forks on `keep_prompt_hidden_tokens`: an
/// AR turn lands in `run_paged_prefill_chunk_with_size`, a planned-MTP turn
/// in `run_paged_prefill_chunk_with_hidden_with_size`. Those are two
/// separate bodies that each decide their own checkpoint break set, so a
/// helper that only reaches the first cannot see the second regress.
fn run_dense_paged_prefill_with_hidden_and_checkpoint(
    inner: &mut Qwen35Inner,
    full_tokens: &[u32],
    suffix_tokens: &[u32],
    cached_prefix_len: u32,
    chunk_size: i32,
) -> Result<(
    MxArray,
    MxArray,
    Vec<crate::models::qwen3_5::paged_forward::MaterializedGdnPrefixCheckpoint>,
)> {
    let layer_kinds = decoder_layer::compute_layer_kinds(inner.config.num_layers as usize, |i| {
        inner.config.is_linear_layer(i)
    });
    let embed = inner.embedding.clone();
    let keep_tokens = full_tokens.len();
    let caches = inner.caches.as_mut().expect("qwen35 caches initialized");
    let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");

    crate::models::qwen3_5::paged_forward::run_paged_prefill_chunk_with_hidden_with_size(
        full_tokens,
        suffix_tokens,
        cached_prefix_len,
        false,
        &embed,
        &mut inner.layers,
        caches,
        &inner.final_norm,
        &inner.lm_head,
        &layer_kinds,
        adapter,
        chunk_size,
        Some(keep_tokens),
        /* cached_rope_deltas */ 0,
        None,
    )
}

fn logits_to_f32_vec(logits: &MxArray) -> Vec<f32> {
    let f32_arr = logits.astype(DType::Float32).expect("astype f32");
    f32_arr.eval();
    let n = f32_arr.shape_at(0).expect("shape_at(0)") as usize;
    (0..n)
        .map(|i| f32_arr.item_at_float32(i).expect("item_at_float32"))
        .collect()
}

fn batch_vocab_logits_to_f32_vec(logits: &MxArray) -> Vec<f32> {
    assert_eq!(logits.ndim().expect("ndim"), 2, "batch logits ndim");
    assert_eq!(logits.shape_at(0).expect("shape_at(0)"), 1);
    let squeezed = logits.squeeze(Some(&[0])).expect("squeeze batch");
    logits_to_f32_vec(&squeezed)
}

fn assert_finite_vocab_logits(logits: &MxArray, vocab_size: i32, context: &str) {
    assert_eq!(logits.ndim().expect("ndim"), 1, "{context}: logits ndim");
    assert_eq!(
        logits.shape_at(0).expect("shape_at(0)"),
        vocab_size as i64,
        "{context}: logits shape"
    );
    let values = logits_to_f32_vec(logits);
    for (i, v) in values.iter().enumerate() {
        assert!(v.is_finite(), "{context}: logits[{i}] is not finite: {v}");
    }
}

fn assert_finite_batch_vocab_logits(logits: &MxArray, vocab_size: i32, context: &str) {
    assert_eq!(logits.ndim().expect("ndim"), 2, "{context}: logits ndim");
    assert_eq!(
        logits.shape_at(0).expect("shape_at(0)"),
        1,
        "{context}: logits batch"
    );
    assert_eq!(
        logits.shape_at(1).expect("shape_at(1)"),
        vocab_size as i64,
        "{context}: logits vocab"
    );
    let values = batch_vocab_logits_to_f32_vec(logits);
    for (i, v) in values.iter().enumerate() {
        assert!(v.is_finite(), "{context}: logits[{i}] is not finite: {v}");
    }
}

fn assert_close_batch_vocab_logits(left: &MxArray, right: &MxArray, abs_tol: f32, context: &str) {
    let left = batch_vocab_logits_to_f32_vec(left);
    let right = batch_vocab_logits_to_f32_vec(right);
    assert_eq!(left.len(), right.len(), "{context}: logits len");
    for (i, (a, b)) in left.iter().zip(right.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff <= abs_tol,
            "{context}: logits[{i}] differ: left={a}, right={b}, abs_diff={diff}, tol={abs_tol}"
        );
    }
}

fn assert_finite_hidden(hidden: &MxArray, context: &str) {
    let f32_arr = hidden.astype(DType::Float32).expect("astype hidden f32");
    f32_arr.eval();
    let total = (0..f32_arr.ndim().expect("hidden ndim"))
        .map(|axis| f32_arr.shape_at(axis).expect("hidden shape") as usize)
        .product::<usize>();
    for i in 0..total {
        let value = f32_arr.item_at_float32(i).expect("hidden item");
        assert!(
            value.is_finite(),
            "{context}: hidden[{i}] is not finite: {value}"
        );
    }
}

fn reset_dense_caches(inner: &mut Qwen35Inner) {
    inner.caches = Some(fresh_dense_layer_caches(&inner.config));
}

fn run_dense_final_logits_legacy_chunked_projection(
    inner: &mut Qwen35Inner,
    prompt: &MxArray,
    embedding: &Embedding,
    chunk_size: i64,
) -> Result<MxArray> {
    reset_dense_caches(inner);
    let total_len = prompt.shape_at(1)?;
    let chunk_size = if chunk_size <= 0 {
        total_len
    } else {
        chunk_size
    };
    let generation_stream = Stream::new(DeviceType::Gpu);
    let mut offset = 0;
    while total_len - offset > chunk_size {
        let chunk = prompt.slice_axis(1, offset, offset + chunk_size)?;
        {
            let _stream_ctx = StreamContext::new(generation_stream);
            let _logits = forward_inner(
                &chunk,
                embedding,
                &mut inner.layers,
                &mut inner.caches,
                &inner.final_norm,
                &inner.lm_head,
            )?;
        }
        eval_layer_caches(&inner.caches)?;
        crate::array::clear_cache();
        offset += chunk_size;
    }

    let remaining = prompt.slice_axis(1, offset, total_len)?;
    let logits = {
        let _stream_ctx = StreamContext::new(generation_stream);
        forward_inner(
            &remaining,
            embedding,
            &mut inner.layers,
            &mut inner.caches,
            &inner.final_norm,
            &inner.lm_head,
        )?
    };
    let seq_len = logits.shape_at(1)?;
    logits
        .slice_axis(1, seq_len - 1, seq_len)?
        .squeeze(Some(&[1]))
}

fn run_dense_final_logits_chunked(
    inner: &mut Qwen35Inner,
    prompt: &MxArray,
    embedding: &Embedding,
    chunk_size: i64,
) -> Result<MxArray> {
    reset_dense_caches(inner);
    chunked_prefill_with_size(
        prompt,
        embedding,
        &mut inner.layers,
        &mut inner.caches,
        &inner.final_norm,
        &inner.lm_head,
        Stream::new(DeviceType::Gpu),
        chunk_size,
        None,
    )
}

/// `use_block_paged_cache` defaults to `None` and round-trips
/// through serde.
#[test]
fn test_use_block_paged_cache_serde_default_none() {
    let json = serde_json::json!({
        "vocab_size": 1024,
        "hidden_size": 64,
        "num_layers": 8,
        "num_heads": 4,
        "num_kv_heads": 2,
        "intermediate_size": 128,
        "rms_norm_eps": 1e-6,
        "head_dim": 16,
        "tie_word_embeddings": true,
        "max_position_embeddings": 1024,
        "pad_token_id": 0,
        "eos_token_id": 0,
        "bos_token_id": 0,
    });
    let cfg: Qwen3_5Config = serde_json::from_value(json).unwrap();
    assert_eq!(
        cfg.use_block_paged_cache, None,
        "use_block_paged_cache must default to None on JSON without the key"
    );
    assert_eq!(cfg.paged_block_size, None);
    assert_eq!(cfg.paged_cache_memory_mb, None);
}

#[test]
fn test_use_block_paged_cache_serde_true_round_trip() {
    let json = serde_json::json!({
        "vocab_size": 1024,
        "hidden_size": 64,
        "num_layers": 8,
        "num_heads": 4,
        "num_kv_heads": 2,
        "intermediate_size": 128,
        "rms_norm_eps": 1e-6,
        "head_dim": 16,
        "tie_word_embeddings": true,
        "max_position_embeddings": 1024,
        "pad_token_id": 0,
        "eos_token_id": 0,
        "bos_token_id": 0,
        "use_block_paged_cache": true,
        "paged_block_size": 16,
        "paged_cache_memory_mb": 256,
    });
    let cfg: Qwen3_5Config = serde_json::from_value(json).unwrap();
    assert_eq!(cfg.use_block_paged_cache, Some(true));
    assert_eq!(cfg.paged_block_size, Some(16));
    assert_eq!(cfg.paged_cache_memory_mb, Some(256));
}

#[test]
fn test_full_attention_layer_count() {
    let cfg = tiny_cfg(false);
    // 8 layers, full_attention_interval=4 → layers 3 and 7 are
    // full-attention (2 layers).
    assert_eq!(cfg.full_attention_layer_count(), 2);
}

#[test]
fn test_dense_gdn_root_rotation_and_legacy_retention_policy() {
    fn push_checkpoint(
        inner: &mut Qwen35Inner,
        owner_id: &str,
        root_owner_id: Option<&str>,
        marker: u32,
    ) {
        crate::engine::backend::ChatBackend::set_cache_owner_id(inner, owner_id, root_owner_id);
        let tokens: Vec<u32> = (0..16).map(|offset| marker * 100 + offset).collect();
        inner
            .gdn_prefix_checkpoints
            .push_back(DenseGdnPrefixCheckpoint {
                owner_id: inner.active_cache_owner_id.clone(),
                prefix_len: 16,
                block_size: 16,
                final_block_hash: marker as u64,
                block_hashes: vec![marker as u64],
                tokens,
                caches: Vec::new(),
            });
        inner.prune_dense_gdn_prefix_checkpoints();
    }

    let mut inner = Qwen35Inner::new(tiny_cfg(false)).expect("construct tiny dense model");
    push_checkpoint(&mut inner, "root-0", Some("root-0"), 1);
    push_checkpoint(&mut inner, "root-1", Some("root-1"), 2);
    for (index, owner) in ["child-0", "child-1", "child-2", "child-3"]
        .into_iter()
        .enumerate()
    {
        push_checkpoint(&mut inner, owner, Some("root-1"), 10 + index as u32);
    }
    assert_eq!(inner.gdn_root_cache_owner_id.as_deref(), Some("root-1"));
    assert!(inner.gdn_root_cache_owner_is_explicit);
    assert_eq!(inner.gdn_prefix_checkpoints.len(), 5);
    assert!(
        !inner
            .gdn_prefix_checkpoints
            .iter()
            .any(|checkpoint| checkpoint.owner_id == "root-0")
    );
    assert!(
        inner
            .gdn_prefix_checkpoints
            .iter()
            .any(|checkpoint| checkpoint.owner_id == "root-1")
    );

    // Without an explicit root there is no separate global budget, so the
    // owner cap is also the global cap. This model has no paged adapter and
    // therefore no cold GDN policy, so the owner cap is the PRE-LADDER one:
    // nothing here could ever read a ladder, and widening the store on that
    // arm changes which checkpoint a later persistence-OFF turn lands on.
    let mut legacy = Qwen35Inner::new(tiny_cfg(false)).expect("construct legacy dense model");
    assert!(
        !legacy.wants_gdn_checkpoint_ladder(),
        "a model with no paged adapter cannot have a cold GDN sidecar policy"
    );
    for marker in 1..=(GDN_PREFIX_CHECKPOINTS_PER_OWNER as u32 + 1) {
        push_checkpoint(&mut legacy, "", None, marker);
    }
    assert!(!legacy.gdn_root_cache_owner_is_explicit);
    assert_eq!(
        legacy.gdn_prefix_checkpoints.len(),
        GDN_PREFIX_CHECKPOINTS_PER_OWNER_NO_LADDER
    );
}

/// The retention cap is chosen at this call site, not inside the store, and
/// the two arms are reachable only from here: `wants_gdn_checkpoint_ladder`
/// reads the adapter's installed cold-tier policy, which no store-level test
/// can see.
///
/// Push the SAME four-rung ladder both ways. With a GDN sidecar policy
/// installed the whole ladder must survive — that is what the ladder is for.
/// With the policy removed the same push must collapse to the pre-ladder
/// number, because a turn that publishes no ladder must retain what it
/// retained before the ladder existed.
///
/// Catches: hardcoding either arm at the call site. `true` regresses every
/// persistence-OFF request's emitted tokens; `false` silently reverts
/// persist-ON to single-endpoint retention.
#[test]
fn dense_gdn_retention_follows_the_installed_cold_sidecar_policy() {
    let Some(mut inner) = dense_inner_with_test_adapter_or_skip(
        "dense_gdn_retention_follows_the_installed_cold_sidecar_policy",
    ) else {
        return;
    };
    let tokens: Vec<u32> = (0..64u32).collect();
    let extra_keys = vec![Vec::new(); 4];

    let push_ladder = |inner: &mut Qwen35Inner| {
        inner.gdn_prefix_checkpoints.clear();
        crate::engine::backend::ChatBackend::set_cache_owner_id(inner, "", None);
        for rung in 1..=4u32 {
            let prefix_len = rung * 16;
            let block_hashes =
                compute_paged_prefix_block_hashes(&tokens, prefix_len, 16, &extra_keys, 0)
                    .expect("block-aligned rung must hash");
            inner
                .gdn_prefix_checkpoints
                .push_back(DenseGdnPrefixCheckpoint {
                    owner_id: inner.active_cache_owner_id.clone(),
                    prefix_len,
                    block_size: 16,
                    final_block_hash: block_hashes.last().copied().unwrap_or_default(),
                    block_hashes,
                    tokens: tokens[..prefix_len as usize].to_vec(),
                    caches: Vec::new(),
                });
            inner.prune_dense_gdn_prefix_checkpoints();
        }
        inner.gdn_prefix_checkpoints.len()
    };

    assert!(!inner.wants_gdn_checkpoint_ladder());
    assert_eq!(
        push_ladder(&mut inner),
        GDN_PREFIX_CHECKPOINTS_PER_OWNER_NO_LADDER,
        "with no cold GDN policy the store must stay at the pre-ladder cap"
    );

    let root = temp_cold_root("dense-gdn-retention-policy");
    install_gdn_cold_tier(&mut inner, &root);
    assert!(inner.wants_gdn_checkpoint_ladder());
    assert_eq!(
        push_ladder(&mut inner),
        GDN_PREFIX_CHECKPOINTS_PER_OWNER,
        "a cold GDN sidecar policy must keep the whole published ladder"
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// The retention policy lives in `gdn_checkpoint_store`, but WHICH owner it
/// protects is decided here, by what this call site passes. A twin that
/// hands `prune_gdn_checkpoints` the root where the active owner belongs
/// collapses a subagent's ladder to its endpoint rung while every
/// store-level test stays green, so drive it through the model.
///
/// The root is seeded with a spare rung on purpose: that is the redundancy
/// the first arm spends, and without it there is nothing for the arm to
/// prefer and the mutation is invisible.
///
/// The cold tier is installed for the same reason. Publisher deferral is a
/// `GdnRetentionPolicy::Ladder` behaviour and `active_owner_id` is read on
/// no other arm, so a turn with no GDN sidecar policy cannot express the
/// mutation at all.
/// The persistence-OFF push below pins the other arm's shape rather than
/// leaving it unmeasured.
#[test]
fn test_dense_ladder_survives_sibling_owners_through_the_model_call_site() {
    fn push(inner: &mut Qwen35Inner, owner_id: &str, root_owner_id: &str, blocks: u32) {
        crate::engine::backend::ChatBackend::set_cache_owner_id(
            inner,
            owner_id,
            Some(root_owner_id),
        );
        inner
            .gdn_prefix_checkpoints
            .push_back(DenseGdnPrefixCheckpoint {
                owner_id: inner.active_cache_owner_id.clone(),
                prefix_len: blocks * 16,
                block_size: 16,
                final_block_hash: u64::from(blocks),
                block_hashes: (1..=u64::from(blocks)).collect(),
                tokens: (0..blocks * 16).collect(),
                caches: Vec::new(),
            });
        inner.prune_dense_gdn_prefix_checkpoints();
    }

    /// Root seeded with a spare rung, two siblings holding one each, then a
    /// third subagent publishing a whole quartered ladder rung by rung the
    /// way `publish_dense_gdn_materialized_prefix_checkpoint_with_keys`
    /// does. Every slot is taken before the ladder starts.
    fn run_fleet(inner: &mut Qwen35Inner) -> Vec<(String, u32)> {
        inner.gdn_prefix_checkpoints.clear();
        push(inner, "root", "root", 1);
        push(inner, "root", "root", 64);
        for sibling in ["child-0", "child-1"] {
            push(inner, sibling, "root", 64);
        }
        for blocks in [1, 4, 16, 64] {
            push(inner, "child-3", "root", blocks);
        }
        inner
            .gdn_prefix_checkpoints
            .iter()
            .map(|checkpoint| (checkpoint.owner_id.clone(), checkpoint.prefix_len))
            .collect()
    }

    let Some(mut inner) = dense_inner_with_test_adapter_or_skip(
        "test_dense_ladder_survives_sibling_owners_through_the_model_call_site",
    ) else {
        return;
    };

    // Persistence OFF: no deferral, and the pre-ladder per-owner cap of 2.
    assert!(!inner.wants_gdn_checkpoint_ladder());
    assert_eq!(
        run_fleet(&mut inner),
        vec![
            ("root".to_string(), 1024),
            ("child-0".to_string(), 1024),
            ("child-1".to_string(), 1024),
            ("child-3".to_string(), 256),
            ("child-3".to_string(), 1024),
        ],
        "a persistence-OFF turn takes the pre-ladder victim order, which \
         keeps the publisher's DEEPEST rungs and defers nobody"
    );

    // Persistence ON: the publisher's shallow rung is what this turn's cold
    // capture anchors on, so it outlives the root's spare rung.
    let root = temp_cold_root("dense-ladder-sibling-call-site");
    install_gdn_cold_tier(&mut inner, &root);
    assert!(inner.wants_gdn_checkpoint_ladder());
    assert_eq!(
        run_fleet(&mut inner),
        vec![
            ("root".to_string(), 1024),
            ("child-0".to_string(), 1024),
            ("child-1".to_string(), 1024),
            ("child-3".to_string(), 16),
            ("child-3".to_string(), 1024),
        ],
        "the publishing subagent must keep the shallow rung this turn's \
         capture anchors on, paid for by the root's spare rung, and no \
         sibling may be pushed to zero"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn test_qwen35_media_plan_requires_complete_paged_vision_stack() {
    for has_encoder in [false, true] {
        for has_processor in [false, true] {
            for has_paged in [false, true] {
                let plan = qwen35_dense_media_plan(has_encoder, has_processor, has_paged);
                let available = has_encoder && has_processor && has_paged;
                assert_eq!(plan.available.images, available);
                assert!(!plan.available.audio);
                assert_eq!(plan.backend_validated.images, !available);
                assert!(!plan.backend_validated.audio);
                assert!(plan.admitted().images);
            }
        }
    }
}

#[test]
fn test_qwen35_session_media_survives_paged_image_key_clear() {
    assert_eq!(
        qwen35_dense_session_media(true, false),
        MediaCapabilities::IMAGES
    );
    assert_eq!(
        qwen35_dense_session_media(false, true),
        MediaCapabilities::IMAGES,
        "a retained M-RoPE delta proves successive paged text turns still extend image KV"
    );
    assert_eq!(
        qwen35_dense_session_media(false, false),
        MediaCapabilities::NONE
    );
}

#[test]
fn test_qwen35_session_media_payload_identity() {
    let images = vec![vec![1, 2, 3]];
    let cached_key = Some(engine::compute_image_cache_key(&images));

    assert!(qwen35_dense_session_media_matches_payloads(
        cached_key,
        &images,
        &[]
    ));
    assert!(!qwen35_dense_session_media_matches_payloads(
        cached_key,
        &[vec![1, 2, 4]],
        &[]
    ));
    assert!(!qwen35_dense_session_media_matches_payloads(
        cached_key,
        &images,
        &[vec![9]]
    ));
    assert!(!qwen35_dense_session_media_matches_payloads(
        None,
        &images,
        &[]
    ));
}

#[test]
fn test_dense_paged_finalize_failure_cannot_republish_image_session() {
    let mut inner = Qwen35Inner::new(tiny_cfg(false)).expect("construct tiny dense model");
    inner.caches = Some(fresh_dense_layer_caches(&inner.config));
    inner.cached_token_history = vec![7, 248_056, 248_056, 8];
    inner.cached_image_key = Some(0xD35E);
    inner.cached_paged_image_token_positions = vec![(1, 0xA11C), (2, 0xA11C)];
    inner.cached_rope_deltas = Some(-2);

    // A missing adapter exercises the same infallible-hook downgrade used
    // for registration/evaluation/release failures without allocating a
    // Metal LayerKVPool in this unit test.
    <Qwen35Inner as crate::engine::backend::PagedBackend>::finalize_paged_turn(&mut inner, true, 0);
    assert!(inner.paged_finalize_failed);
    assert!(inner.caches.is_none());
    assert!(inner.cached_token_history.is_empty());
    assert!(inner.cached_image_key.is_none());
    assert!(inner.cached_paged_image_token_positions.is_empty());
    assert!(inner.cached_rope_deltas.is_none());

    // The engine always calls save after the infallible finalize hook. It
    // must consume the failure latch rather than reviving the session from
    // expanded image-placeholder ids.
    <Qwen35Inner as crate::engine::backend::PagedBackend>::save_paged_history(
        &mut inner,
        &[7, 248_056, 248_056, 8],
        &[9],
        false,
        true,
    )
    .expect("failed finalization must downgrade history save");
    assert!(!inner.paged_finalize_failed);
    assert!(inner.cached_token_history.is_empty());
    assert!(!crate::engine::backend::ChatBackend::has_live_session(
        &inner
    ));
    assert_eq!(
        crate::engine::backend::ChatBackend::session_media(&inner),
        MediaCapabilities::NONE
    );
}

fn seed_dense_paged_image_session(inner: &mut Qwen35Inner) {
    inner.caches = Some(fresh_dense_layer_caches(&inner.config));
    inner.cached_token_history = vec![7, 248_056, 248_056, 8];
    inner.cached_image_key = Some(0xD35E);
    inner.cached_paged_image_token_positions = vec![(1, 0xA11C), (2, 0xA11C)];
    inner.cached_rope_deltas = Some(-2);
    inner.paged_full_attn_caches_dirty = true;
    inner.flat_mtp_caches_desynced = true;
}

#[test]
fn test_dense_manual_paged_finalize_failure_invalidates_session() {
    let mut inner = Qwen35Inner::new(tiny_cfg(false)).expect("construct tiny dense model");
    seed_dense_paged_image_session(&mut inner);

    let error = inner
        .finalize_dense_manual_paged_turn(&[(1, 0xA11C), (2, 0xA11C)], 0, 4)
        .expect_err("missing adapter must fail manual finalization");
    assert!(error.to_string().contains("paged finalization failed"));
    assert!(inner.caches.is_none());
    assert!(inner.cached_token_history.is_empty());
    assert!(inner.cached_image_key.is_none());
    assert!(inner.cached_paged_image_token_positions.is_empty());
    assert!(inner.cached_rope_deltas.is_none());
    assert!(!inner.paged_full_attn_caches_dirty);
    assert!(!inner.flat_mtp_caches_desynced);
}

#[test]
fn test_dense_vlm_prefix_prepare_failure_invalidates_session() {
    let mut inner = Qwen35Inner::new(tiny_cfg(false)).expect("construct tiny dense model");
    seed_dense_paged_image_session(&mut inner);

    inner
        .prepare_dense_vlm_paged_prefix(
            &[7, 248_056, 248_056, 8],
            4,
            16,
            &[vec![0xA11C]],
            true,
            true,
            0xD35E,
            0,
        )
        .expect_err("missing adapter must fail VLM prefix preparation");
    assert!(inner.caches.is_none());
    assert!(inner.cached_token_history.is_empty());
    assert!(inner.cached_image_key.is_none());
    assert!(inner.cached_paged_image_token_positions.is_empty());
    assert!(inner.cached_rope_deltas.is_none());
    assert!(!inner.paged_full_attn_caches_dirty);
    assert!(!crate::engine::backend::ChatBackend::has_live_session(
        &inner
    ));
}

#[test]
fn test_dense_generic_paged_abort_invalidates_session() {
    let mut inner = Qwen35Inner::new(tiny_cfg(false)).expect("construct tiny dense model");
    seed_dense_paged_image_session(&mut inner);

    <Qwen35Inner as crate::engine::backend::PagedBackend>::abort_paged_turn(&mut inner);
    assert!(inner.caches.is_none());
    assert!(inner.cached_token_history.is_empty());
    assert!(inner.cached_image_key.is_none());
    assert!(inner.cached_paged_image_token_positions.is_empty());
    assert!(inner.cached_rope_deltas.is_none());
    assert!(!inner.paged_full_attn_caches_dirty);
    assert!(!inner.flat_mtp_caches_desynced);
}

#[test]
fn test_qwen35_planned_decoder_overrides_raw_mtp_flag() {
    let mut config = ChatConfig {
        cache_salt: None,
        cache_owner_id: None,
        cache_root_owner_id: None,
        enable_mtp: Some(true),
        ..ChatConfig::default()
    };
    assert!(!apply_qwen35_dense_planned_decoder(
        &mut config,
        DecoderPlan::Autoregressive
    ));
    assert_eq!(config.enable_mtp, Some(false));

    assert!(apply_qwen35_dense_planned_decoder(
        &mut config,
        DecoderPlan::Speculative(SpeculativeKind::NativeMtp)
    ));
    assert_eq!(config.enable_mtp, Some(true));

    assert!(!apply_qwen35_dense_planned_decoder(
        &mut config,
        DecoderPlan::Speculative(SpeculativeKind::DraftModel)
    ));
    assert_eq!(config.enable_mtp, Some(false));
}

/// When `use_block_paged_cache` is `None`, `paged_adapter` is None.
#[test]
fn test_inner_no_paged_adapter_when_flag_is_none() {
    let cfg = tiny_cfg(false);
    let inner = Qwen35Inner::new(cfg).expect("Qwen35Inner::new must succeed without paged adapter");
    assert!(
        inner.paged_adapter.is_none(),
        "paged_adapter must be None when use_block_paged_cache is None"
    );
}

#[test]
fn test_fresh_dense_layer_caches_are_not_gdn_reuse_ready() {
    let cfg = tiny_cfg(true);
    let caches = fresh_dense_layer_caches(&cfg);
    assert_eq!(caches.len(), cfg.num_layers as usize);
    assert!(
        !dense_paged_linear_caches_ready(&cfg, Some(&caches)),
        "fresh linear caches have empty conv/recurrent slots, so a live continuation must replay GDN"
    );
    assert!(matches!(caches[0], Qwen3_5LayerCache::Linear(_)));
    assert!(matches!(caches[3], Qwen3_5LayerCache::FullAttention(_)));
}

fn run_dense_chunked_prefill_final_logits_only_matches_legacy_chunking() -> Result<()> {
    let mut inner = Qwen35Inner::new(tiny_cfg(false))?;
    cast_qwen35_inner_weights_bf16(&mut inner);

    let prompt_tokens: Vec<u32> = (0u32..33).map(|i| (i * 17 + 5) % 997).collect();
    let prompt = MxArray::from_uint32(&prompt_tokens, &[1, prompt_tokens.len() as i64])?;
    let embedding = inner.embedding.clone();

    let expected =
        run_dense_final_logits_legacy_chunked_projection(&mut inner, &prompt, &embedding, 16)?;
    assert_finite_batch_vocab_logits(
        &expected,
        inner.config.vocab_size,
        "legacy chunked final logits",
    );

    let chunked = run_dense_final_logits_chunked(&mut inner, &prompt, &embedding, 16)?;
    assert_finite_batch_vocab_logits(&chunked, inner.config.vocab_size, "chunked final logits");
    assert_close_batch_vocab_logits(
        &expected,
        &chunked,
        1e-6,
        "chunked final logits vs legacy chunking",
    );

    Ok(())
}

#[test]
fn test_dense_chunked_prefill_final_logits_only_matches_legacy_chunking() {
    if let Err(err) = run_dense_chunked_prefill_final_logits_only_matches_legacy_chunking() {
        let msg = err.reason.to_string();
        if msg.contains("Metal") || msg.contains("device") {
            eprintln!(
                "skipping test_dense_chunked_prefill_final_logits_only_matches_legacy_chunking: {msg}"
            );
            return;
        }
        panic!("unexpected dense chunked prefill failure: {msg}");
    }
}

fn run_dense_chunked_prefill_with_hidden_keeps_tail_contract() -> Result<()> {
    let mut inner = Qwen35Inner::new(tiny_cfg(false))?;
    cast_qwen35_inner_weights_bf16(&mut inner);

    let prompt_tokens: Vec<u32> = (0u32..35).map(|i| (i * 23 + 3) % 997).collect();
    let prompt = MxArray::from_uint32(&prompt_tokens, &[1, prompt_tokens.len() as i64])?;
    let embedding = inner.embedding.clone();

    reset_dense_caches(&mut inner);
    let (logits, hidden) = chunked_prefill_with_hidden_with_size(
        &prompt,
        &embedding,
        &mut inner.layers,
        &mut inner.caches,
        &inner.final_norm,
        &inner.lm_head,
        Stream::new(DeviceType::Gpu),
        Some(5),
        16,
        None,
    )?;
    assert_finite_batch_vocab_logits(
        &logits,
        inner.config.vocab_size,
        "chunked hidden final logits",
    );
    assert_eq!(hidden.ndim()?, 3, "prompt hidden ndim");
    assert_eq!(hidden.shape_at(0)?, 1, "prompt hidden batch");
    assert_eq!(hidden.shape_at(1)?, 5, "prompt hidden tail len");
    assert_eq!(
        hidden.shape_at(2)?,
        inner.config.hidden_size as i64,
        "prompt hidden width"
    );
    assert_finite_hidden(&hidden, "chunked hidden tail");

    let logits_without_hidden =
        run_dense_final_logits_chunked(&mut inner, &prompt, &embedding, 16)?;
    assert_close_batch_vocab_logits(
        &logits,
        &logits_without_hidden,
        1e-6,
        "hidden and logits-only chunked final logits",
    );

    reset_dense_caches(&mut inner);
    let (_logits, full_tail_hidden) = chunked_prefill_with_hidden_with_size(
        &prompt,
        &embedding,
        &mut inner.layers,
        &mut inner.caches,
        &inner.final_norm,
        &inner.lm_head,
        Stream::new(DeviceType::Gpu),
        Some(100),
        16,
        None,
    )?;
    assert_eq!(
        full_tail_hidden.shape_at(1)?,
        prompt_tokens.len() as i64,
        "oversized keep window keeps the whole prompt"
    );
    assert_finite_hidden(&full_tail_hidden, "chunked full prompt hidden");

    Ok(())
}

#[test]
fn test_dense_chunked_prefill_with_hidden_keeps_tail_contract() {
    if let Err(err) = run_dense_chunked_prefill_with_hidden_keeps_tail_contract() {
        let msg = err.reason.to_string();
        if msg.contains("Metal") || msg.contains("device") {
            eprintln!("skipping test_dense_chunked_prefill_with_hidden_keeps_tail_contract: {msg}");
            return;
        }
        panic!("unexpected dense chunked prefill hidden failure: {msg}");
    }
}

#[test]
fn test_dense_paged_prefix_block_hash_matches_allocator_chain() {
    let tokens: Vec<u32> = (1..=12).collect();
    let per_block = vec![vec![11], vec![], vec![33, 44]];

    let h0 = mlx_paged_attn::hash_tokens(&tokens[0..4], 0, &per_block[0]);
    let h1 = mlx_paged_attn::hash_tokens(&tokens[4..8], h0, &per_block[1]);
    let h2 = mlx_paged_attn::hash_tokens(&tokens[8..12], h1, &per_block[2]);

    assert_eq!(
        compute_paged_prefix_block_hash(&tokens, 12, 4, &per_block, 0),
        Some(h2)
    );
}

#[test]
fn test_dense_paged_prefix_block_hash_applies_salt_to_first_block_only() {
    let tokens: Vec<u32> = (1..=8).collect();
    let per_block = vec![vec![11], vec![22]];
    let salt = 99;

    let mut first_block_keys = per_block[0].clone();
    first_block_keys.push(salt);
    let h0 = mlx_paged_attn::hash_tokens(&tokens[0..4], 0, &first_block_keys);
    let h1 = mlx_paged_attn::hash_tokens(&tokens[4..8], h0, &per_block[1]);

    assert_eq!(
        compute_paged_prefix_block_hash(&tokens, 8, 4, &per_block, salt),
        Some(h1)
    );
}

#[test]
fn test_dense_paged_prefix_block_hash_rejects_non_full_or_unkeyed_prefix() {
    let tokens: Vec<u32> = (1..=8).collect();
    let per_block = vec![vec![]];

    assert_eq!(
        compute_paged_prefix_block_hash(&tokens, 6, 4, &per_block, 0),
        None
    );
    assert_eq!(
        compute_paged_prefix_block_hash(&tokens, 8, 4, &per_block, 0),
        None
    );
}

/// Allocates a `LayerKVPool`. Requires Metal; gate on
/// `MLX_TEST_PAGED=1`.
#[test]
#[ignore = "Allocates Metal LayerKVPool; gate on MLX_TEST_PAGED=1"]
fn test_inner_constructs_paged_adapter_when_flag_is_true() {
    if std::env::var_os("MLX_TEST_PAGED").is_none() {
        return;
    }
    let cfg = tiny_cfg(true);
    let mut inner = Qwen35Inner::new(cfg).expect(
        "Qwen35Inner::new with use_block_paged_cache=true must succeed on Metal-capable host",
    );
    inner
        .initialize_paged_adapter()
        .expect("post-load paged adapter initialization must succeed");
    assert!(
        inner.paged_adapter.is_some(),
        "paged_adapter must be Some when use_block_paged_cache = Some(true)"
    );
}

/// VLM checkpoints are accepted under paged dispatch. The media plan is
/// not executable with only an encoder; it becomes available only after
/// the processor completes the paged vision stack.
#[test]
#[ignore = "Allocates Metal LayerKVPool; gate on MLX_TEST_PAGED=1"]
fn test_vlm_loads_when_paged_enabled() {
    if std::env::var_os("MLX_TEST_PAGED").is_none() {
        return;
    }
    use crate::models::qwen3_5::vision::Qwen3_5VisionConfig;
    use crate::models::qwen3_5::vision::Qwen3_5VisionEncoder;

    let cfg = tiny_cfg(true);
    let mut inner = Qwen35Inner::new(cfg).unwrap();
    inner
        .initialize_paged_adapter()
        .expect("post-load paged adapter initialization");
    let vision_cfg = Qwen3_5VisionConfig {
        hidden_size: 64,
        intermediate_size: 256,
        num_heads: 4,
        num_layers: 2,
        patch_size: 16,
        spatial_merge_size: 2,
        image_size: 256,
        out_hidden_size: 64,
    };
    let vision_enc = Qwen3_5VisionEncoder::new(vision_cfg).expect("vision encoder construction");
    let result = inner.set_vision_encoder(vision_enc);
    assert!(
        result.is_ok(),
        "set_vision_encoder must succeed when paged_adapter is Some so VLM \
         checkpoints can complete their paged media stack; got {result:?}"
    );
    assert!(
        inner.vision_encoder.is_some(),
        "vision_encoder field must be populated after a successful set"
    );
    let incomplete = inner.execution_plan().media;
    assert_eq!(incomplete.available, MediaCapabilities::NONE);
    assert_eq!(incomplete.backend_validated, MediaCapabilities::IMAGES);

    inner.set_image_processor(Qwen35VLImageProcessor::new(None));
    let complete = inner.execution_plan().media;
    assert_eq!(complete.available, MediaCapabilities::IMAGES);
    assert_eq!(complete.backend_validated, MediaCapabilities::NONE);
}

/// Dense Qwen3.5 paged-prefill chunking state test. This drives the
/// production chunk-size worker once and asserts the adapter cursor,
/// request token log, and block table cover the whole prompt after all
/// chunks have been recorded.
#[test]
#[ignore = "requires Metal GPU; run with --ignored"]
fn test_dense_paged_prefill_chunks_advance_adapter_state() {
    let Some((mut inner, cfg)) =
        paged_inner_or_skip("test_dense_paged_prefill_chunks_advance_adapter_state")
    else {
        return;
    };
    cast_qwen35_inner_weights_bf16(&mut inner);

    let prompt: Vec<u32> = (0u32..64).map(|i| (i * 5 + 7) % 257).collect();
    reset_paged_request(&mut inner, &prompt);

    let logits = match run_dense_paged_prefill_with_size(
        &mut inner, &prompt, &prompt, 0, /* chunk_size */ 16,
    ) {
        Ok(logits) => logits,
        Err(err) => {
            let msg = err.reason.to_string();
            if msg.contains("Metal GPU not available") || msg.contains("No Metal device") {
                eprintln!("skipping test_dense_paged_prefill_chunks_advance_adapter_state: {msg}");
                return;
            }
            panic!("unexpected dense paged chunk failure: {msg}");
        }
    };

    let adapter = inner.paged_adapter.as_ref().expect("paged_adapter");
    assert_eq!(
        adapter.current_token_count() as usize,
        prompt.len(),
        "adapter cursor after chunked prefill"
    );
    assert_eq!(
        adapter.request_tokens(),
        prompt.as_slice(),
        "request token log after chunked prefill"
    );
    let block_table = adapter.block_table().expect("block_table");
    let expected_min_blocks = prompt.len().div_ceil(adapter.block_size() as usize);
    assert!(
        block_table.num_blocks() >= expected_min_blocks,
        "block table has {} blocks, expected at least {expected_min_blocks}",
        block_table.num_blocks()
    );
    assert_finite_vocab_logits(&logits, cfg.vocab_size, "final dense paged chunk prefill");

    let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
    let _ = adapter.register_full_blocks_for_reuse(&[], 0);
    adapter.release_request().expect("release_request");
}

/// Uneven-tail coverage for dense Qwen3.5 paged prefill: a 33-token
/// prompt with chunk_size=16 must record two full chunks plus a
/// one-token tail and return valid logits for the tail chunk.
#[test]
#[ignore = "requires Metal GPU; run with --ignored"]
fn test_dense_paged_prefill_chunks_handle_uneven_tail() {
    let Some((mut inner, cfg)) =
        paged_inner_or_skip("test_dense_paged_prefill_chunks_handle_uneven_tail")
    else {
        return;
    };
    cast_qwen35_inner_weights_bf16(&mut inner);

    let prompt: Vec<u32> = (0u32..33).map(|i| (i * 11 + 3) % 257).collect();
    reset_paged_request(&mut inner, &prompt);

    let final_logits = run_dense_paged_prefill_with_size(
        &mut inner, &prompt, &prompt, 0, /* chunk_size */ 16,
    )
    .expect("dense paged uneven-tail chunked prefill");

    assert_eq!(
        prompt.len(),
        33,
        "test setup must exercise a one-token tail"
    );
    let adapter = inner.paged_adapter.as_ref().expect("paged_adapter");
    assert_eq!(
        adapter.current_token_count(),
        33,
        "adapter cursor must include the uneven tail"
    );
    assert_eq!(
        adapter.request_tokens(),
        prompt.as_slice(),
        "request token log must include the uneven tail"
    );
    assert_finite_vocab_logits(
        &final_logits,
        cfg.vocab_size,
        "uneven-tail dense paged chunk prefill",
    );

    let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
    let _ = adapter.register_full_blocks_for_reuse(&[], 0);
    adapter.release_request().expect("release_request");
}

/// Regression coverage for Pi-style agent turns: a block-aligned prefill
/// must retain GDN state at the largest reusable paged-block boundary (16
/// of 32 tokens here). A one-block rollback restores exactly, while a
/// full-boundary hit replays only one GDN block.
#[test]
#[ignore = "requires Metal GPU; run with --ignored"]
fn test_dense_paged_prefill_captures_exact_gdn_block_checkpoint() {
    let Some((mut inner, cfg)) =
        paged_inner_or_skip("test_dense_paged_prefill_captures_exact_gdn_block_checkpoint")
    else {
        return;
    };
    cast_qwen35_inner_weights_bf16(&mut inner);

    let prompt: Vec<u32> = (0u32..32).map(|i| (i * 13 + 9) % 257).collect();
    reset_paged_request(&mut inner, &prompt);

    let (logits, checkpoint) = run_dense_paged_prefill_with_size_and_checkpoint(
        &mut inner, &prompt, &prompt, 0, /* chunk_size */ 16,
    )
    .expect("dense paged checkpoint prefill");
    assert_finite_vocab_logits(&logits, cfg.vocab_size, "checkpoint prefill logits");

    let mut checkpoint = checkpoint;
    let checkpoint = checkpoint
        .pop()
        .expect("stable reusable-block GDN checkpoint");
    assert_eq!(checkpoint.prefix_len, 16);
    assert!(dense_paged_linear_caches_ready(
        &inner.config,
        Some(&checkpoint.caches)
    ));

    let block_size = inner
        .paged_adapter
        .as_ref()
        .expect("paged_adapter")
        .block_size();
    let extra_keys = engine::build_paged_extra_keys(prompt.len(), block_size, &[]);
    let cache_salt = 59;
    inner.publish_dense_gdn_materialized_prefix_checkpoint(&prompt, cache_salt, vec![checkpoint]);
    inner.active_cache_owner_id = "child-session".to_owned();
    assert!(
        inner
            .find_dense_gdn_prefix_checkpoint(&prompt, 16, block_size, &extra_keys, cache_salt,)
            .is_none(),
        "an exact token/hash checkpoint owned by another session must not restore"
    );
    inner.active_cache_owner_id.clear();
    assert!(
        inner
            .find_dense_gdn_prefix_checkpoint(&prompt, 16, block_size, &extra_keys, 0)
            .is_none(),
        "the salted publisher must not silently store the checkpoint in domain zero"
    );
    let restored = inner
        .find_dense_gdn_prefix_checkpoint(&prompt, 16, block_size, &extra_keys, cache_salt)
        .expect("exact checkpoint restore");
    assert_eq!(restored.0, 16);
    assert!(dense_paged_linear_caches_ready(
        &inner.config,
        Some(&restored.1)
    ));
    let prepared = inner
        .prepare_dense_gdn_prefix_state(&prompt, 16, block_size, &extra_keys, cache_salt, false)
        .expect("prepare one-block rollback checkpoint");
    assert_eq!(prepared.state, "checkpoint");
    assert!(prepared.already_primed);
    assert_eq!(prepared.restored_prefix_tokens, 16);
    assert_eq!(prepared.replayed_prefix_tokens, 0);

    let prepared = inner
        .prepare_dense_gdn_prefix_state(&prompt, 32, block_size, &extra_keys, cache_salt, false)
        .expect("prepare full-boundary hit from stable checkpoint");
    assert_eq!(prepared.state, "checkpoint_replay_materialized");
    assert!(prepared.already_primed);
    assert_eq!(prepared.restored_prefix_tokens, 16);
    assert_eq!(prepared.replayed_prefix_tokens, 16);

    let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
    let _ = adapter.register_full_blocks_for_reuse(&[], cache_salt);
    adapter.release_request().expect("release_request");
}

/// A cancel flag observed between materialized GDN
/// replay chunks must abort the replay with the distinguished error,
/// and the staged-commit wrapper must drop the partial state (no warm
/// publish). A one-chunk replay stays single-shot and ignores the flag.
#[test]
#[ignore = "requires Metal GPU; run with --ignored"]
fn test_dense_materialized_gdn_replay_cancels_between_chunks() {
    let Some((mut inner, cfg)) =
        paged_inner_or_skip("test_dense_materialized_gdn_replay_cancels_between_chunks")
    else {
        return;
    };
    cast_qwen35_inner_weights_bf16(&mut inner);
    let prompt: Vec<u32> = (0u32..37).map(|i| (i * 19 + 11) % 128).collect();
    let embed = inner.embedding.clone();
    let cancelled = std::sync::Arc::new(AtomicBool::new(true));

    // Multi-chunk replay with the flag pre-set: chunk 1 runs, the
    // between-chunk poll aborts before chunk 2, and the staged cache is
    // dropped — the active cache is never published.
    {
        let mut active: Option<Vec<Qwen3_5LayerCache>> = None;
        let staged = fresh_dense_layer_caches(&cfg);
        let layers = &mut inner.layers;
        let err = replay_gdn_cache_and_commit(&mut active, staged, |staged| {
            crate::models::qwen3_5::paged_forward::run_gdn_only_prefill_materialized_with_chunk_size(
                &prompt,
                &embed,
                layers,
                staged,
                7,
                Some(cancelled.as_ref()),
            )
        })
        .expect_err("cancelled multi-chunk replay must abort");
        assert!(
            err.to_string().contains("prefill cancelled"),
            "replay abort must carry the distinguished error, got: {err}",
        );
        assert!(
            active.is_none(),
            "a cancelled replay must never publish the staged cache"
        );
    }

    // One-chunk exception: the same pre-set flag is ignored when the
    // whole prefix fits in a single replay chunk (single-shot contract).
    {
        let mut active: Option<Vec<Qwen3_5LayerCache>> = None;
        let staged = fresh_dense_layer_caches(&cfg);
        let layers = &mut inner.layers;
        replay_gdn_cache_and_commit(&mut active, staged, |staged| {
            crate::models::qwen3_5::paged_forward::run_gdn_only_prefill_materialized_with_chunk_size(
                &prompt,
                &embed,
                layers,
                staged,
                prompt.len(),
                Some(cancelled.as_ref()),
            )
        })
        .expect("single-chunk replay stays uncancellable");
        assert!(active.is_some(), "a completed replay must publish");
    }

    // Un-set flag: the multi-chunk replay completes and publishes.
    cancelled.store(false, Ordering::Relaxed);
    {
        let mut active: Option<Vec<Qwen3_5LayerCache>> = None;
        let staged = fresh_dense_layer_caches(&cfg);
        let layers = &mut inner.layers;
        replay_gdn_cache_and_commit(&mut active, staged, |staged| {
            crate::models::qwen3_5::paged_forward::run_gdn_only_prefill_materialized_with_chunk_size(
                &prompt,
                &embed,
                layers,
                staged,
                7,
                Some(cancelled.as_ref()),
            )
        })
        .expect("uncancelled multi-chunk replay must complete");
        assert!(active.is_some(), "a completed replay must publish");
    }

    let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
    adapter.release_request().expect("release_request");
}

/// Compatibility guard for the current/default dense Qwen3.5 paged
/// prefill behavior: a full suffix passed in one call remains a valid
/// single-shot prefill and is stable across a fresh adapter reset.
#[test]
#[ignore = "requires Metal GPU; run with --ignored"]
fn test_dense_paged_prefill_single_shot_default_still_works() {
    let Some((mut inner, cfg)) =
        paged_inner_or_skip("test_dense_paged_prefill_single_shot_default_still_works")
    else {
        return;
    };
    cast_qwen35_inner_weights_bf16(&mut inner);

    // Longer than two 16-token blocks: if chunk_size=0 accidentally starts
    // splitting at checkpoint boundaries, this test observes a sidecar.
    let prompt: Vec<u32> = (0u32..33).map(|i| (i * 17 + 5) % 257).collect();

    reset_paged_request(&mut inner, &prompt);
    let (logits_a, checkpoint_a) = run_dense_paged_prefill_with_size_and_checkpoint(
        &mut inner, &prompt, &prompt, 0, /* chunk_size */ 0,
    )
    .expect("single-shot A");
    assert!(
        checkpoint_a.is_empty(),
        "chunk_size=0 must not split to capture a GDN checkpoint"
    );
    assert_finite_vocab_logits(&logits_a, cfg.vocab_size, "single-shot A");
    {
        let adapter = inner.paged_adapter.as_ref().expect("paged_adapter");
        assert_eq!(
            adapter.current_token_count() as usize,
            prompt.len(),
            "single-shot cursor"
        );
        assert_eq!(adapter.request_tokens(), prompt.as_slice());
    }

    reset_paged_request(&mut inner, &prompt);
    let (logits_b, checkpoint_b) = run_dense_paged_prefill_with_size_and_checkpoint(
        &mut inner, &prompt, &prompt, 0, /* chunk_size */ 0,
    )
    .expect("single-shot B");
    assert!(
        checkpoint_b.is_empty(),
        "chunk_size=0 must stay single-shot after adapter reset"
    );
    let a = logits_to_f32_vec(&logits_a);
    let b = logits_to_f32_vec(&logits_b);
    assert_eq!(a.len(), cfg.vocab_size as usize);
    assert_eq!(b.len(), cfg.vocab_size as usize);
    for (i, (left, right)) in a.iter().zip(b.iter()).enumerate() {
        let abs_diff = (left - right).abs();
        assert!(
            abs_diff <= 1e-6,
            "single-shot dense paged prefill changed after fresh reset at index {i}: \
             first={left}, second={right}, abs_diff={abs_diff}"
        );
    }

    let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
    let _ = adapter.register_full_blocks_for_reuse(&[], 0);
    adapter.release_request().expect("release_request");
}

/// Dynamic (grow-on-demand) dense pool end to end: a tiny initial pool
/// must grow to hold a prefill that exceeds it, fire the growth notifier
/// with the new total bytes, stop growing once the prompt fits, and
/// produce byte-identical logits to a repeat run on the grown pool.
#[test]
#[ignore = "requires Metal GPU; run with --ignored"]
fn test_dense_initial_pool_grows_to_hold_a_long_prefill() {
    let mut cfg = tiny_paged_forward_cfg();
    // Geometry: 2 attn layers × 2 KV heads × 32 head × 16 tokens × 2
    // sides × 2 bytes = 8 KiB per block. 1 MiB initial → 128 blocks; the
    // 3000-token prompt needs 188. The max stays at the config's 256 MiB
    // (PagedAttentionConfig::validate floors gpu_memory_mb at 256), so
    // the first grow lands on min(2×128, 128+60) = 256 blocks.
    cfg.max_position_embeddings = 8192;
    cfg.paged_cache_initial_memory_mb = Some(1);
    let Some((mut inner, cfg)) =
        paged_inner_with_cfg_or_skip("test_dense_initial_pool_grows_to_hold_a_long_prefill", cfg)
    else {
        return;
    };
    cast_qwen35_inner_weights_bf16(&mut inner);

    let (initial_blocks, max_blocks) = {
        let adapter = inner.paged_adapter.as_ref().expect("paged_adapter");
        let pool = adapter.layer_kv_pool();
        let max = pool.max_num_blocks();
        let live = adapter.block_capacity();
        assert!(
            live < max,
            "initial pool must sit below the max ceiling: {live} vs {max}"
        );
        (live, max)
    };
    assert!(initial_blocks >= 1);

    let grown_bytes = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    {
        let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
        let observed = grown_bytes.clone();
        adapter.set_pool_growth_notifier(Some(std::sync::Arc::new(move |bytes| {
            observed.store(bytes, std::sync::atomic::Ordering::Relaxed);
        })));
    }

    // Longer than two initial blocks: growth must happen before eviction.
    let prompt: Vec<u32> = (0u32..3000).map(|i| (i * 17 + 5) % 257).collect();
    reset_paged_request(&mut inner, &prompt);
    let (logits_a, checkpoint_a) = run_dense_paged_prefill_with_size_and_checkpoint(
        &mut inner, &prompt, &prompt, 0, /* chunk_size */ 0,
    )
    .expect("dynamic-pool single-shot A");
    assert!(
        checkpoint_a.is_empty(),
        "chunk_size=0 must not split to capture a GDN checkpoint"
    );
    assert_finite_vocab_logits(&logits_a, cfg.vocab_size, "dynamic-pool A");

    {
        let adapter = inner.paged_adapter.as_ref().expect("paged_adapter");
        let pool = adapter.layer_kv_pool();
        let live = adapter.block_capacity();
        assert!(
            live > initial_blocks,
            "the pool must have grown past its initial size: {live} vs {initial_blocks}"
        );
        assert!(
            live <= max_blocks,
            "growth must stay capped at the max ceiling: {live} vs {max_blocks}"
        );
        assert_eq!(
            pool.max_num_blocks(),
            max_blocks,
            "the max ceiling is fixed for the pool's lifetime"
        );
        assert_eq!(
            grown_bytes.load(std::sync::atomic::Ordering::Relaxed),
            pool.total_bytes(),
            "the notifier must report the new TOTAL pool bytes"
        );
        assert_eq!(pool.num_blocks(), live);
        assert_eq!(
            adapter.current_token_count() as usize,
            prompt.len(),
            "dynamic-pool cursor"
        );
    }

    // Second run on the grown pool: no further growth, identical logits.
    let bytes_after_a = grown_bytes.load(std::sync::atomic::Ordering::Relaxed);
    reset_paged_request(&mut inner, &prompt);
    let (logits_b, _checkpoint_b) = run_dense_paged_prefill_with_size_and_checkpoint(
        &mut inner, &prompt, &prompt, 0, /* chunk_size */ 0,
    )
    .expect("dynamic-pool single-shot B");
    assert_finite_vocab_logits(&logits_b, cfg.vocab_size, "dynamic-pool B");
    assert_eq!(
        grown_bytes.load(std::sync::atomic::Ordering::Relaxed),
        bytes_after_a,
        "a prompt that already fits must not grow the pool again"
    );

    let a = logits_to_f32_vec(&logits_a);
    let b = logits_to_f32_vec(&logits_b);
    assert_eq!(a.len(), cfg.vocab_size as usize);
    assert_eq!(b.len(), cfg.vocab_size as usize);
    for (i, (left, right)) in a.iter().zip(b.iter()).enumerate() {
        let abs_diff = (left - right).abs();
        assert!(
            abs_diff <= 1e-6,
            "the grown pool must decode byte-identically to the initial run at index {i}: \
             first={left}, second={right}, abs_diff={abs_diff}"
        );
    }

    let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
    adapter.release_request().expect("release_request");
}

/// Dense inner carrying a real adapter over a test-sized `LayerKVPool`.
///
/// `initialize_paged_adapter` sizes a pool for a real checkpoint and needs a
/// production block geometry; the lifecycle below dispatches no kernel, so a
/// two-layer placeholder pool is enough and runs wherever the adapter's own
/// lifecycle tests run.
fn dense_inner_with_test_adapter_or_skip(test_name: &str) -> Option<Qwen35Inner> {
    const BLOCK_SIZE: u32 = 16;
    const NUM_BLOCKS: u32 = 8;
    let pool_config = mlx_paged_attn::PagedAttentionConfig {
        block_size: BLOCK_SIZE,
        num_kv_heads: 1,
        head_size: 32,
        num_layers: 2,
        ..mlx_paged_attn::PagedAttentionConfig::default()
    };
    let pool = match mlx_paged_attn::LayerKVPool::new_for_test(
        pool_config,
        NUM_BLOCKS,
        2,
        mlx_paged_attn::metal::MetalDtype::Float16,
    ) {
        Ok(pool) => Arc::new(pool),
        // Routed through `test_support::metal_device_absent` rather than
        // matching the string inline, so `MLX_TEST_REQUIRE_METAL=1` can
        // turn a self-skip into a failure here too. libtest counts a
        // skipped-and-returned test as passed, and these two are the only
        // gates that can see a hardcoded arm at the prune call site.
        Err(err) if crate::test_support::metal_device_absent(&err) => {
            eprintln!("skipping {test_name} (no Metal device): {err}");
            return None;
        }
        Err(err) => {
            panic!("unexpected LayerKVPool::new_for_test failure in {test_name}: {err}")
        }
    };
    let allocator = Arc::new(Mutex::new(mlx_paged_attn::BlockAllocator::new(
        NUM_BLOCKS, NUM_BLOCKS, BLOCK_SIZE,
    )));
    let adapter = PagedKVCacheAdapter::new(allocator, pool, BLOCK_SIZE)
        .expect("test paged adapter must construct");
    let mut inner = Qwen35Inner::new(tiny_cfg(true)).expect("construct tiny dense model");
    inner.paged_adapter = Some(adapter);
    Some(inner)
}

/// Attach a cold tier carrying the GDN sidecar policy — the exact shape
/// `build_cold_tier_context` installs when persistence is on.
fn install_gdn_cold_tier(inner: &mut Qwen35Inner, root: &std::path::Path) {
    let manager = mlx_paged_attn::ColdCacheManager::open_default_at(root.to_path_buf())
        .expect("temp-dir cold cache must open");
    let cache_dtype = {
        let adapter = inner.paged_adapter.as_ref().expect("paged_adapter");
        format!("{:?}", adapter.layer_kv_pool().cache_dtype())
    };
    let policy = crate::models::qwen3_5::gdn_sidecar::policy(&inner.config, &cache_dtype)
        .expect("a hybrid config must yield a GDN sidecar policy");
    let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
    adapter.set_cold_tier(
        crate::transformer::paged_kv_cache_adapter::ColdTierContext {
            manager: Arc::new(manager),
            fingerprint: mlx_paged_attn::ColdCacheFingerprint::from_components([
                b"qwen3-5-dense-sidecar-test".as_slice(),
            ]),
            sidecar_policy: Some(policy),
        },
    );
}

fn temp_cold_root(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "mlx-{name}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ))
}

fn record_whole_paged_request(inner: &mut Qwen35Inner, prompt: &[u32]) {
    let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
    adapter.reset_for_new_request(0).expect("reset request");
    adapter
        .allocate_suffix_blocks(prompt.len() as u32)
        .expect("allocate suffix blocks");
    adapter.record_tokens(prompt).expect("record tokens");
}

/// The hand-written cores finalize through `finalize_dense_manual_paged_turn`
/// — every planned-MTP turn ends there and never reaches the engine hook. It
/// must run the GDN sidecar capture, otherwise an MTP session persists K/V
/// blocks whose recurrent half no restore can reconstruct.
///
/// Catches: deleting `capture_dense_gdn_cold_sidecar` from the `Ok` arm of
/// `finalize_dense_manual_paged_turn` — both deltas drop to zero.
#[test]
fn dense_manual_paged_finalize_reaches_the_gdn_sidecar_capture() {
    let Some(mut inner) = dense_inner_with_test_adapter_or_skip(
        "dense_manual_paged_finalize_reaches_the_gdn_sidecar_capture",
    ) else {
        return;
    };
    let root = temp_cold_root("dense-manual-finalize-capture");
    install_gdn_cold_tier(&mut inner, &root);
    let prompt: Vec<u32> = (0u32..32).collect();
    record_whole_paged_request(&mut inner, &prompt);

    let _serialized = crate::cold_tier::sidecar_counter_test_lock();
    let before = crate::cold_tier::cold_sidecar_telemetry();
    inner
        .finalize_dense_manual_paged_turn(&[], 0, prompt.len())
        .expect("manual paged finalization must succeed");
    let after = crate::cold_tier::cold_sidecar_telemetry();

    assert_eq!(
        after.capture_reached,
        before.capture_reached + 1,
        "the manual finalize must enter the GDN sidecar capture exactly once"
    );
    // The chain really carried both blocks off the GPU. Asserted here
    // because everything below depends on it and nothing below can see it:
    // a pool whose buffers are too small makes `read_block_all_layers`
    // return `Err`, the chain capture stops at its first block, and the
    // boundary assertion after this one becomes unreachable rather than
    // false. Without this line that failure reads as "the capture did not
    // reach its boundary selection", three layers from the real cause.
    assert_eq!(
        inner
            .paged_adapter
            .as_ref()
            .expect("paged_adapter")
            .cold_captured_blocks(),
        2,
        "both whole blocks must have been captured off the pool"
    );
    // The finalize published two whole blocks to the cold chain, so the
    // capture ran all the way to the boundary selection and found no
    // in-memory checkpoint to anchor on. Reaching THAT arm proves it got
    // past the policy and media guards rather than bailing at its first line.
    assert_eq!(
        after.boundary_skips,
        before.boundary_skips + 1,
        "the capture must reach its boundary selection under a GdnState policy"
    );

    let _ = std::fs::remove_dir_all(&root);
}

/// The v1 text-only guard reads the turn's OWN media map. The VLM cores clear
/// `cached_paged_image_token_positions` before prefill and reassign it only
/// after finalization returns, so a `self`-read guard sees an empty vec on
/// exactly the media turns it exists to refuse.
///
/// Catches: reverting the guard to `self.cached_paged_image_token_positions`
/// — that field is empty here, so the capture would run on past it and bump
/// `chain_empty`.
#[test]
fn dense_gdn_sidecar_capture_refuses_the_turns_own_media_map() {
    let Some(mut inner) = dense_inner_with_test_adapter_or_skip(
        "dense_gdn_sidecar_capture_refuses_the_turns_own_media_map",
    ) else {
        return;
    };
    let root = temp_cold_root("dense-manual-finalize-media");
    install_gdn_cold_tier(&mut inner, &root);
    let prompt: Vec<u32> = (0u32..32).collect();
    record_whole_paged_request(&mut inner, &prompt);
    assert!(
        inner.cached_paged_image_token_positions.is_empty(),
        "the model-level media map is empty mid-turn; only the argument carries the truth"
    );

    let _serialized = crate::cold_tier::sidecar_counter_test_lock();
    let before = crate::cold_tier::cold_sidecar_telemetry();
    inner
        .finalize_dense_manual_paged_turn(&[(4, 99)], 0, prompt.len())
        .expect("manual paged finalization must succeed");
    let after = crate::cold_tier::cold_sidecar_telemetry();

    assert_eq!(
        after.capture_reached,
        before.capture_reached + 1,
        "a media turn still enters the capture"
    );
    assert_eq!(
        after.boundary_skips, before.boundary_skips,
        "a media turn must be refused before the capture inspects the cold chain"
    );
    assert_eq!(
        after.chain_empty, before.chain_empty,
        "a media turn must be refused before the capture inspects the cold chain"
    );

    let _ = std::fs::remove_dir_all(&root);
}

/// Tripwire: a publish site that keys GDN state on a history
/// disagreeing with the paged frontier — without first running
/// `check_dense_paged_frontier`, which would arm the refuse-to-persist
/// latch — must trip the debug assert inside
/// `remember_dense_gdn_history_checkpoint` instead of storing state that
/// does not match its token key.
///
/// Catches: removing that assert — the publish then proceeds silently
/// and this test fails on the missing panic. (`catch_unwind` rather than
/// `#[should_panic]` so the no-Metal self-skip can still pass.)
#[test]
fn history_checkpoint_publish_asserts_the_paged_frontier() {
    let Some(mut inner) = dense_inner_with_test_adapter_or_skip(
        "history_checkpoint_publish_asserts_the_paged_frontier",
    ) else {
        return;
    };
    let prompt: Vec<u32> = (0u32..32).collect();
    record_whole_paged_request(&mut inner, &prompt);
    // One token short of the adapter's 32 recorded rows with the latch
    // NOT armed — the exact shape a future check-less publish site would
    // hand the store.
    inner.cached_token_history = prompt[..prompt.len() - 1].to_vec();
    assert!(!inner.paged_gdn_state_dirty);

    let publish = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = inner.remember_dense_gdn_history_checkpoint();
    }));
    assert!(
        publish.is_err(),
        "a checkpoint publish keyed off the paged frontier must trip the L-KEY debug assert"
    );
}

/// The chunk size every paged prefill reads. Single-shot materializes no GDN
/// checkpoint, so a sidecar has nothing to anchor on; an installed GdnState
/// policy has to turn the ladder's rungs into real splits.
///
/// Catches: dropping the policy probe from `cold_gdn_prefill_chunk_size`
/// (returns 0 with a policy installed) — the defect that leaves a persist-cold
/// turn publishing no rung at all.
#[test]
fn dense_cold_gdn_prefill_chunk_size_follows_the_installed_sidecar_policy() {
    let Some(mut inner) = dense_inner_with_test_adapter_or_skip(
        "dense_cold_gdn_prefill_chunk_size_follows_the_installed_sidecar_policy",
    ) else {
        return;
    };
    assert_eq!(
        inner.cold_gdn_prefill_chunk_size(),
        0,
        "without a cold GDN policy the prefill stays single-shot"
    );

    let root = temp_cold_root("dense-chunk-size-policy");
    install_gdn_cold_tier(&mut inner, &root);
    assert!(
        inner.cold_gdn_prefill_chunk_size() > 0,
        "an installed GdnState policy must make the prefill split at the ladder"
    );

    let _ = std::fs::remove_dir_all(&root);
}

/// Both hand-written cores prefill through `run_dense_core_paged_prefill`, so
/// this drives that shared body directly: with a GdnState policy installed it
/// must publish checkpoint ladder rungs, and with no cold tier it must stay
/// single-shot.
///
/// Catches: passing a literal `0` or `crate::array::paged_prefill_chunk_size()`
/// instead of `self.cold_gdn_prefill_chunk_size()` inside
/// `run_dense_core_paged_prefill` — the cold arm then returns no rungs, which
/// is exactly the MTP-turn defect.
///
/// The last two arms use an EXPLICIT positive chunk size, which is the only
/// way to reach the break-set decision with no policy installed — the shape
/// a persist-off `mlx agent` turn has, since `run-agent.ts` seeds
/// `MLX_PAGED_PREFILL_CHUNK_SIZE=2048` before any persistence decision. They
/// catch hardcoding `gdn_cold_sidecar_ladder_wanted`'s result at the prefill
/// body in either direction, which the arms above cannot: at 64 tokens the
/// ladder is already a single rung, so both arms agree there.
#[test]
#[ignore = "requires Metal GPU; run with --ignored"]
fn dense_core_paged_prefill_publishes_ladder_rungs_under_a_cold_policy() {
    let Some((mut inner, _cfg)) =
        paged_inner_or_skip("dense_core_paged_prefill_publishes_ladder_rungs_under_a_cold_policy")
    else {
        return;
    };
    cast_qwen35_inner_weights_bf16(&mut inner);
    let layer_kinds = decoder_layer::compute_layer_kinds(inner.config.num_layers as usize, |i| {
        inner.config.is_linear_layer(i)
    });
    let prompt: Vec<u32> = (0u32..64).map(|i| (i * 5 + 7) % 257).collect();

    reset_paged_request(&mut inner, &prompt);
    let (_, hidden, cold_checkpoints) = inner
        .run_dense_core_paged_prefill(
            &prompt,
            &prompt,
            0,
            false,
            None,
            &layer_kinds,
            "cold-policy prefill",
        )
        .expect("prefill with no cold tier");
    assert!(hidden.is_none());
    assert!(
        cold_checkpoints.is_empty(),
        "with no cold GDN policy the cores must stay single-shot"
    );

    // A prompt long enough that the two break sets DIFFER: ladder
    // [48, 192] against the single pre-ladder boundary [192]. Run BEFORE
    // the cold tier is installed — the adapter has no way to drop one.
    let long_prompt: Vec<u32> = (0u32..200).map(|i| (i * 5 + 7) % 257).collect();
    reset_paged_request(&mut inner, &long_prompt);
    let (_, no_policy_rungs) = run_dense_paged_prefill_with_size_and_checkpoint(
        &mut inner,
        &long_prompt,
        &long_prompt,
        0,
        2048,
    )
    .expect("explicit-chunk prefill with no cold tier");
    assert_eq!(
        no_policy_rungs
            .iter()
            .map(|c| c.prefix_len)
            .collect::<Vec<_>>(),
        vec![192],
        "with no cold GDN policy an explicit chunk size must take the single pre-ladder \
         boundary, not the ladder — the extra breaks change the prefill GEMM's M and so the \
         sampled tokens of a persistence-off request"
    );
    inner
        .paged_adapter
        .as_mut()
        .expect("paged_adapter")
        .release_request()
        .expect("release_request");

    // The SAME claim on the planned-MTP body. `run_dense_core_paged_prefill`
    // forks on `keep_prompt_hidden_tokens` into two separate prefill
    // functions that each pick their own break set, and every arm above
    // takes the AR fork only.
    // `ChatSession::mergeConfig` auto-defaults `enableMtp` to true on any
    // checkpoint carrying an MTP head, so on those checkpoints the MTP fork
    // is the one an `mlx agent` turn actually takes.
    reset_paged_request(&mut inner, &long_prompt);
    let (_, _, no_policy_mtp_rungs) = run_dense_paged_prefill_with_hidden_and_checkpoint(
        &mut inner,
        &long_prompt,
        &long_prompt,
        0,
        2048,
    )
    .expect("explicit-chunk MTP prefill with no cold tier");
    assert_eq!(
        no_policy_mtp_rungs
            .iter()
            .map(|c| c.prefix_len)
            .collect::<Vec<_>>(),
        vec![192],
        "the planned-MTP prefill body must take the same single pre-ladder boundary with no \
         cold GDN policy as the AR body does"
    );
    // Release WITHOUT registering: publishing this prompt's blocks would
    // give the next arm's shared 64-token prefix a cache hit, and every arm
    // here has to start cold.
    inner
        .paged_adapter
        .as_mut()
        .expect("paged_adapter")
        .release_request()
        .expect("release_request");

    let root = temp_cold_root("dense-core-ladder");
    install_gdn_cold_tier(&mut inner, &root);
    reset_paged_request(&mut inner, &prompt);
    let (_, _, ladder) = inner
        .run_dense_core_paged_prefill(
            &prompt,
            &prompt,
            0,
            false,
            None,
            &layer_kinds,
            "cold-policy prefill",
        )
        .expect("prefill under a cold GDN policy");
    assert!(
        !ladder.is_empty(),
        "a persist-cold turn must materialize the GDN checkpoint ladder the sidecar anchors on"
    );
    for checkpoint in &ladder {
        assert!(
            checkpoint.prefix_len > 0 && (checkpoint.prefix_len as usize) < prompt.len(),
            "a rung must sit strictly inside the prompt, got {}",
            checkpoint.prefix_len
        );
    }

    // Same long prompt, same explicit chunk size, policy now installed:
    // every rung of the ladder is a real break.
    reset_paged_request(&mut inner, &long_prompt);
    let (_, cold_rungs) = run_dense_paged_prefill_with_size_and_checkpoint(
        &mut inner,
        &long_prompt,
        &long_prompt,
        0,
        2048,
    )
    .expect("explicit-chunk prefill under a cold GDN policy");
    assert_eq!(
        cold_rungs.iter().map(|c| c.prefix_len).collect::<Vec<_>>(),
        vec![48, 192],
        "an explicit chunk size under a GDN cold policy still publishes the whole ladder"
    );
    inner
        .paged_adapter
        .as_mut()
        .expect("paged_adapter")
        .release_request()
        .expect("release_request");

    // And the MTP fork under the policy: the ladder is what the sidecar anchors
    // on, so an MTP turn publishing only the deep rung means silent zero reuse.
    reset_paged_request(&mut inner, &long_prompt);
    let (_, _, cold_mtp_rungs) = run_dense_paged_prefill_with_hidden_and_checkpoint(
        &mut inner,
        &long_prompt,
        &long_prompt,
        0,
        2048,
    )
    .expect("explicit-chunk MTP prefill under a cold GDN policy");
    assert_eq!(
        cold_mtp_rungs
            .iter()
            .map(|c| c.prefix_len)
            .collect::<Vec<_>>(),
        vec![48, 192],
        "the planned-MTP prefill body must publish the whole ladder under a cold GDN policy"
    );

    let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
    let _ = adapter.register_full_blocks_for_reuse(&[], 0);
    adapter.release_request().expect("release_request");
    let _ = std::fs::remove_dir_all(&root);
}

/// A paged speculative turn ADDRESSES the model's adapter, it
/// does not take it. Three things follow, and each is one assertion here:
///
///   * the adapter stays reachable through `inner.paged_adapter` for the
///     whole turn, and the turn's writes land on THAT adapter — a stepper
///     that moved a copy out would leave the model's cursor at the prompt;
///   * a facade write addressed to any other sequence is refused, so a
///     turn can never drive a cache that has moved to another request;
///   * a paged turn whose adapter names no active request is refused AT
///     ENTRY with the adapter untouched, instead of running Step A against
///     it and failing at the verify write.
#[test]
#[ignore = "requires Metal GPU; run with --ignored"]
fn paged_mtp_turn_addresses_the_model_adapter_instead_of_taking_it() {
    use crate::engine::spec_paged::SpecPagedCache;

    let mut cfg = tiny_paged_forward_cfg();
    cfg.n_mtp_layers = 1;
    let Some((mut inner, _cfg)) = paged_inner_with_cfg_or_skip(
        "paged_mtp_turn_addresses_the_model_adapter_instead_of_taking_it",
        cfg,
    ) else {
        return;
    };
    cast_qwen35_inner_weights_bf16(&mut inner);
    inner.mtp_weights_loaded = true;

    let prompt: Vec<u32> = (0..24u32).map(|i| (i * 5 + 7) % 257).collect();
    reset_paged_request(&mut inner, &prompt);
    run_dense_paged_prefill_with_size(&mut inner, &prompt, &prompt, 0, 2048)
        .expect("dense paged prefill")
        .eval();

    let depth = 2usize;
    let lookahead = inner
        .execution_plan()
        .speculative
        .expect("dense MTP checkpoint must publish a speculative plan")
        .lookahead_rows(depth);
    let setup = MtpTurnSetup {
        prompt_hidden: None,
        prompt_hidden_ids: None,
        first_sampled_token: 21,
        lookahead_rows: lookahead,
    };

    let seq_id = {
        let mut step = inner.begin_mtp_decode(&setup).expect("begin_mtp_decode");
        let seq_id = step
            .owned_adapter()
            .active_seq_id()
            .expect("active request");

        // A facade WRITE addressed to a foreign sequence is refused. The
        // frontier read is already gated elsewhere; this is the write half,
        // and it is what stops a re-pointed cache from being driven.
        let err = SpecPagedCache::reserve_lookahead(&mut step, seq_id + 1, lookahead)
            .expect_err("a foreign sequence must be refused by the facade");
        assert!(
            err.contains("not this turn's owner"),
            "the refusal must name the owner mismatch, got: {err}"
        );

        // One real Step-A forward, so the turn actually writes a row.
        let ids = MxArray::from_int32(&[21], &[1, 1]).expect("step-A ids");
        let embedding = step.embedding().clone();
        let (logits, _hidden, _squeeze) = step
            .forward_with_hidden(&ids, &embedding)
            .expect("Step-A forward on the model's adapter");
        logits.eval();
        seq_id
    };

    // The write landed on the MODEL's adapter, not on a copy the stepper
    // owned and handed back.
    let adapter = inner
        .paged_adapter
        .as_ref()
        .expect("adapter stays on model");
    assert_eq!(adapter.active_seq_id(), Some(seq_id));
    assert_eq!(
        adapter.current_token_count(),
        prompt.len() as u32 + 1,
        "the turn's Step-A row must be visible on the model's own adapter"
    );

    // Entry refusal: no active request means no addressable sequence, so
    // the turn never starts and the adapter is left exactly as it was.
    let blocks_before = adapter.num_allocated_blocks();
    inner
        .paged_adapter
        .as_mut()
        .expect("adapter")
        .release_request()
        .expect("release_request");
    let err = inner
        .begin_mtp_decode(&setup)
        .err()
        .expect("a paged turn with no active request must be refused at entry");
    assert!(
        err.reason.contains("no active request"),
        "the entry refusal must name the missing request, got: {}",
        err.reason
    );
    let adapter = inner
        .paged_adapter
        .as_ref()
        .expect("the refused turn must not consume the adapter");
    assert!(
        adapter.num_allocated_blocks() <= blocks_before,
        "a refused turn must not reserve lookahead blocks"
    );
}

/// Cross-module gate for the reserve→verify seam and
/// for the facade cycle the dense paged commit is routed through
/// (`engine::spec_paged`). Every cycle here is a REAL verify forward, so
/// every row the block table holds was written by one.
///
/// Geometry: exactly one block of prompt, then production cycles until a
/// cycle's `+lookahead` margin crosses out of the allocated tail. The
/// TURN-ENTRY reservation is pinned by the first crossing (prompt block
/// → prompt + lookahead), the PER-CYCLE facade reservation by the second
/// — a reservation that never reaches the allocator then shows up as a
/// mid-verify allocation on that cycle instead of passing vacuously.
///
/// Catches: I1 drift (the plan property diverging from what verify
/// actually writes), a dead reserve→record seam (reservation not
/// reaching the adapter, or advancing the cursor), a verify/rollback
/// pair whose net adapter growth escapes the reserved margin, and a
/// ONE-SIDED commit — an adapter rollback that does not land the
/// recurrent state with it, which the full `SpecFrontier` equality after
/// every cycle pins. That last shape is invisible to the release
/// backstop `check_dense_paged_frontier`, which compares the adapter
/// against the history and never against the GDN count.
///
/// The tail segment catches one more: `record_verify` / `record_rows`
/// answering anything but `Err` on this family, which would hand a
/// driver a cycle whose kept rows the recurrent state never saw.
#[test]
#[ignore = "requires Metal GPU; run with --ignored"]
fn paged_mtp_lookahead_reservation_covers_verify() {
    use crate::engine::spec_paged::{NoSettleInCycle, SpecPagedCache};

    /// `(allocated blocks, recorded rows)` of the turn's paged adapter.
    fn adapter_stats(step: &DenseMtpStepper<'_>) -> (usize, u32) {
        let adapter = step.owned_adapter();
        (
            adapter.num_allocated_blocks(),
            adapter.current_token_count(),
        )
    }

    /// One production cycle in the engine's order: per-cycle facade
    /// reservation → snapshot → verify over `depth + 1` real ids →
    /// partial-accept rollback keeping the anchor and one draft. Asserts
    /// what must hold on EVERY cycle, and returns the block count before
    /// and after the reservation so the caller can see which cycle's
    /// margin crossed a block boundary.
    fn run_cycle(
        step: &mut DenseMtpStepper<'_>,
        seq_id: u32,
        embedding: &Embedding,
        depth: usize,
        lookahead: usize,
        block_size: u32,
        first_id: i32,
    ) -> (usize, usize) {
        use crate::engine::spec_paged::SpecPagedCache;

        let (pre_blocks, pre_cursor) = adapter_stats(step);
        assert!(
            SpecPagedCache::reserve_lookahead(step, seq_id, lookahead)
                .expect("per-cycle facade reservation"),
            "a reservation the pool can hold must cover the cycle"
        );
        let (reserved_blocks, reserved_cursor) = adapter_stats(step);
        assert_eq!(
            reserved_cursor, pre_cursor,
            "the reservation must not advance the cursor"
        );

        step.snapshot_main_linear();
        let ids: Vec<i32> = (0..=depth as i32).map(|i| first_id + i).collect();
        let verify_ids = ids.iter().map(|&id| id as u32).collect::<Vec<_>>();
        step.verify_step(&verify_ids, embedding, depth)
            .expect("paged verify step")
            .hiddens
            .eval();
        assert!(
            step.open_cycle.is_some(),
            "the verify write must OPEN the cycle its rollback closes — an \
             un-ticketed write leaves the retraction outside the facade"
        );
        let (verify_blocks, verify_cursor) = adapter_stats(step);
        assert_eq!(
            verify_blocks, reserved_blocks,
            "the verify write must allocate ZERO new blocks after the reservation"
        );
        assert_eq!(
            verify_blocks,
            (pre_cursor as usize + lookahead).div_ceil(block_size as usize),
            "the reservation must cover exactly cursor + lookahead rows"
        );
        assert_eq!(
            verify_cursor,
            pre_cursor + lookahead as u32,
            "the verify write must record exactly the reserved rows"
        );

        step.rollback(/* accepted_drafts */ 1, depth);
        assert!(
            step.take_replay_error().is_none(),
            "the GDN tape replay must succeed on the partial accept"
        );
        assert!(
            step.open_cycle.is_none(),
            "the rollback must CONSUME the cycle's ticket"
        );
        let (committed_blocks, committed_cursor) = adapter_stats(step);
        assert_eq!(
            committed_blocks, reserved_blocks,
            "rollback is bookkeeping-only — no block may move"
        );
        let growth = committed_cursor - pre_cursor;
        assert_eq!(growth, 2, "committed rows = accepted drafts + anchor");
        assert!(
            (growth as usize) <= lookahead,
            "total adapter growth must fit the reserved lookahead region"
        );
        // The facade commit is the ADAPTER half of `rollback`; the GDN
        // tape replay is the other half. A commit that retracts only the
        // adapter leaves the recurrent side `depth - accepted` rows
        // ahead, which is exactly what one frontier catches.
        assert_eq!(
            MtpStepper::frontier(step),
            Some(SpecFrontier {
                attn_tokens: u64::from(committed_cursor),
                recurrent_tokens: Some(u64::from(committed_cursor)),
            }),
            "the cycle's commit must land the adapter AND the recurrent state on \
             ONE frontier"
        );
        (pre_blocks, reserved_blocks)
    }

    let mut cfg = tiny_paged_forward_cfg();
    cfg.n_mtp_layers = 1;
    let Some((mut inner, _cfg)) =
        paged_inner_with_cfg_or_skip("paged_mtp_lookahead_reservation_covers_verify", cfg)
    else {
        return;
    };
    cast_qwen35_inner_weights_bf16(&mut inner);
    // Random-init MTP module + loaded marker so `execution_plan()`
    // publishes the speculative plan (the drafter itself never runs
    // here — every driven cycle is snapshot → verify → rollback).
    inner.mtp_weights_loaded = true;

    // Exactly one full block of prompt: the lookahead region then starts
    // ON a block boundary, the worst case for a mid-verify allocation.
    let block_size = inner.paged_adapter.as_ref().expect("adapter").block_size();
    let prompt: Vec<u32> = (0..block_size).map(|i| (i * 5 + 7) % 257).collect();
    reset_paged_request(&mut inner, &prompt);
    run_dense_paged_prefill_with_size(&mut inner, &prompt, &prompt, 0, 2048)
        .expect("dense paged prefill")
        .eval();

    let depth = 3usize;
    let plan = inner
        .execution_plan()
        .speculative
        .expect("dense MTP checkpoint must publish a speculative plan");
    let lookahead = plan.lookahead_rows(depth);
    assert_eq!(lookahead, depth + 1);

    let (pre_blocks, prompt_len) = {
        let adapter = inner.paged_adapter.as_ref().expect("adapter");
        (
            adapter.num_allocated_blocks(),
            adapter.current_token_count(),
        )
    };
    assert_eq!(prompt_len, prompt.len() as u32);

    let setup = MtpTurnSetup {
        prompt_hidden: None,
        prompt_hidden_ids: None,
        first_sampled_token: 21,
        lookahead_rows: lookahead,
    };
    let mut step = inner.begin_mtp_decode(&setup).expect("begin_mtp_decode");

    // Turn-entry reservation: blocks grow to cover prompt + lookahead,
    // the cursor does not move.
    let (entry_blocks, entry_cursor) = adapter_stats(&step);
    assert_eq!(
        entry_blocks,
        (prompt.len() + lookahead).div_ceil(block_size as usize),
        "the turn-entry reservation must cover prompt + lookahead rows"
    );
    assert!(
        entry_blocks > pre_blocks,
        "the prompt ends ON a block boundary, so the turn-entry reservation must \
         cross into a new block"
    );
    assert_eq!(
        entry_cursor, prompt_len,
        "the reservation must not advance the cursor"
    );

    let seq_id = step
        .owned_adapter()
        .active_seq_id()
        .expect("active request");
    assert_eq!(
        SpecPagedCache::frontier(&step, seq_id),
        MtpStepper::frontier(&step),
        "the facade frontier must be the stepper's frontier accessor, reused"
    );
    assert_eq!(
        SpecPagedCache::frontier(&step, seq_id + 1),
        None,
        "a sequence that is not the turn's must have no facade frontier"
    );

    // Decode real cycles until one cycle's `+lookahead` margin crosses
    // out of the allocated tail. Only the PER-CYCLE facade reservation
    // covers that block — the turn-entry one is long spent — so that
    // cycle is where a dead reservation surfaces as a mid-verify
    // allocation.
    let embedding = step.embedding().clone();
    let mut crossing_cycle = None;
    for cycle in 0..block_size as usize {
        let (before, after) = run_cycle(
            &mut step,
            seq_id,
            &embedding,
            depth,
            lookahead,
            block_size,
            21 + 10 * cycle as i32,
        );
        if after > before {
            crossing_cycle = Some(cycle);
            break;
        }
    }
    let crossing_cycle = crossing_cycle.expect(
        "some cycle's lookahead margin must cross a block boundary, or the \
         per-cycle reservation is never exercised",
    );
    assert!(
        crossing_cycle > 0,
        "the crossing must be paid for by a PER-CYCLE reservation, not by the \
         turn-entry one"
    );
    let (committed_blocks, committed_cursor) = adapter_stats(&step);

    // The facade's own writers are REFUSED on this family — the verify
    // core records its rows itself, and a facade row a commit kept would
    // advance the adapter while the recurrent state stood still.
    let mut cache = NoSettleInCycle::new(step);
    let kv_rows: [u32; 2] = [71, 72];
    let err = cache
        .record_verify(seq_id, &kv_rows)
        .expect_err("record_verify must be refused on the dense stepper");
    assert!(
        err.contains("open_core_write_cycle"),
        "the refusal must name the opener that replaces it: {err}"
    );
    let err = cache
        .record_rows(seq_id, &kv_rows)
        .expect_err("record_rows must be refused on the dense stepper");
    assert!(
        err.contains("open_core_write_cycle"),
        "the refusal must name the opener that replaces it: {err}"
    );
    assert_eq!(
        adapter_stats(cache.inner()),
        (committed_blocks, committed_cursor),
        "a refused facade write must leave the adapter exactly where it was"
    );

    // So the law runs on the PRODUCTION shape: the cycle
    // `open_core_write_cycle` opens around a write the core performs —
    // here the adapter `record_tokens` the paged verify core makes —
    // committed with a keep of ZERO. The rows are retracted whole, which
    // keeps this segment frontier-neutral and leaves every surviving
    // block-table row backed by a real verify write.
    assert!(
        cache
            .reserve_lookahead(seq_id, kv_rows.len())
            .expect("KV-half reservation"),
        "two rows fit the tail the crossing cycle allocated"
    );
    let (kv_blocks, _) = adapter_stats(cache.inner());
    let ticket = cache
        .open_core_write_cycle(seq_id, kv_rows.len())
        .expect("open the cycle around the core write");
    cache
        .inner_mut()
        .owned_adapter_mut()
        .record_tokens(&kv_rows)
        .expect("the core write the cycle was opened around");
    assert_eq!(
        adapter_stats(cache.inner()),
        (kv_blocks, committed_cursor + kv_rows.len() as u32),
        "the core write must land in the reserved tail"
    );

    // The ordering law is executable on the real stepper. This family's
    // identity settle is lawful at exactly this basis once the cycle is
    // closed (asserted below), so the refusal here is the checker's, not
    // the implementation's argument validation.
    let err = cache
        .settle_committed(seq_id, u64::from(committed_cursor))
        .expect_err("an in-cycle settle must trip the order check");
    assert!(err.contains("L-SETTLE"), "unexpected error text: {err}");

    cache
        .commit_cycle(seq_id, ticket, 0)
        .expect("a keep of zero retracts the whole core write");
    assert!(
        !cache.settle_captures_durable_state(),
        "the identity settle captures nothing; this family's durable surfaces run \
         in the turn epilogue, and a `true` here would bar the in-cycle settle a \
         permissive call-order checker admits"
    );
    cache
        .settle_committed(seq_id, u64::from(committed_cursor))
        .expect("the identity settle is lawful once the cycle is closed");
    cache
        .settle_committed(seq_id, u64::from(committed_cursor) + 1)
        .expect_err("a committed frontier past the cursor must be refused");

    let mut step = cache.into_inner();
    assert_eq!(
        adapter_stats(&step),
        (kv_blocks, committed_cursor),
        "a fully retracted out-of-band cycle must leave the block table and the \
         cursor exactly where it found them"
    );
    assert_eq!(
        MtpStepper::frontier(&step),
        Some(SpecFrontier {
            attn_tokens: u64::from(committed_cursor),
            recurrent_tokens: Some(u64::from(committed_cursor)),
        }),
        "an out-of-band write touches no recurrent state, so a fully retracted \
         cycle must be frontier-neutral"
    );

    // Real decode still advances on the same request, and both sides
    // move together.
    let next_ids = MxArray::from_int32(&[29], &[1, 1]).expect("ar ids");
    let (logits, _hidden, _needs_squeeze) = step
        .forward_with_hidden(&next_ids, &embedding)
        .expect("AR forward after the facade cycles");
    logits.eval();
    let (ar_blocks, ar_cursor) = adapter_stats(&step);
    assert_eq!(
        ar_cursor,
        committed_cursor + 1,
        "AR decode must advance one row past the committed frontier"
    );
    assert_eq!(ar_blocks, committed_blocks);
    assert_eq!(
        MtpStepper::frontier(&step),
        Some(SpecFrontier {
            attn_tokens: u64::from(ar_cursor),
            recurrent_tokens: Some(u64::from(ar_cursor)),
        }),
        "an AR step must move the adapter and the recurrent state together"
    );

    drop(step);
    assert!(
        inner.paged_adapter.is_some(),
        "the adapter stays on the model for the whole speculative turn"
    );
}

/// Failure path of the same seam: a reservation the pool cannot hold
/// must signal AR fallback (`Ok(false)`) with adapter state untouched —
/// never a turn error — and plain AR decode must still make progress on
/// the same request afterwards.
///
/// Catches: exhaustion routed to `Err` (turn error → session
/// invalidation), and a failed reservation corrupting the request's
/// block table or cursor.
#[test]
#[ignore = "requires Metal GPU; run with --ignored"]
fn paged_mtp_lookahead_exhaustion_falls_back_to_ar() {
    use crate::engine::types::ChatConfig;

    let mut cfg = tiny_paged_forward_cfg();
    cfg.n_mtp_layers = 1;
    let Some((mut inner, _cfg)) =
        paged_inner_with_cfg_or_skip("paged_mtp_lookahead_exhaustion_falls_back_to_ar", cfg)
    else {
        return;
    };
    cast_qwen35_inner_weights_bf16(&mut inner);
    inner.mtp_weights_loaded = true;

    let block_size = inner.paged_adapter.as_ref().expect("adapter").block_size();
    let prompt: Vec<u32> = (0..block_size).map(|i| (i * 5 + 7) % 257).collect();
    reset_paged_request(&mut inner, &prompt);
    run_dense_paged_prefill_with_size(&mut inner, &prompt, &prompt, 0, 2048)
        .expect("dense paged prefill")
        .eval();

    let mut p = extract_chat_params(&ChatConfig {
        enable_mtp: Some(true),
        ..ChatConfig::default()
    });
    // The public surface clamps `mtp_depth` to [1, 5]; drive the params
    // struct directly so the reservation exceeds `max_capacity_tokens`.
    p.mtp_depth = inner
        .paged_adapter
        .as_ref()
        .expect("adapter")
        .max_capacity_tokens() as usize;

    let (blocks_before, cursor_before) = {
        let adapter = inner.paged_adapter.as_ref().expect("adapter");
        (
            adapter.num_allocated_blocks(),
            adapter.current_token_count(),
        )
    };
    let admitted = inner
        .reserve_paged_mtp_lookahead(&p, "exhaustion_test")
        .expect("capacity exhaustion must not error the turn");
    assert!(
        !admitted,
        "an over-capacity reservation must signal AR fallback"
    );
    {
        let adapter = inner.paged_adapter.as_ref().expect("adapter");
        assert_eq!(adapter.num_allocated_blocks(), blocks_before);
        assert_eq!(adapter.current_token_count(), cursor_before);
    }

    // A cycle-sized reservation still fits and admits MTP.
    p.mtp_depth = 3;
    assert!(
        inner
            .reserve_paged_mtp_lookahead(&p, "exhaustion_test")
            .expect("in-capacity reservation"),
        "a reservation the pool can hold must admit the MTP arm"
    );

    // The AR fallback can proceed: one real paged decode step on the
    // same request.
    let logits = {
        let layer_kinds = inner.layer_kinds.clone();
        let embed = inner.embedding.clone();
        let caches_ref = inner.caches.as_mut().expect("caches");
        let adapter = inner.paged_adapter.as_mut().expect("adapter");
        crate::models::qwen3_5::paged_forward::run_paged_decode_step(
            21,
            &embed,
            &mut inner.layers,
            caches_ref,
            &inner.final_norm,
            &inner.lm_head,
            &layer_kinds,
            adapter,
            0,
        )
        .expect("AR decode step after fallback")
    };
    logits.eval();
    assert_eq!(
        inner
            .paged_adapter
            .as_ref()
            .expect("adapter")
            .current_token_count(),
        cursor_before + 1,
        "AR decode must advance one row past the prompt"
    );
}

/// Cycle-2 twin of the exhaustion gate, driving the SAME stepper seam
/// the engine loop calls (`MtpStepper::reserve_cycle_lookahead`) on a
/// REAL paged model. After a first verify cycle + partial-accept
/// rollback has moved the frontier, the per-cycle reservation must
/// (a) grow the block table to cover the NEW cursor + lookahead before
/// cycle 2's verify — the prompt is sized so that margin CROSSES a block
/// boundary the turn-entry reservation never covered, so a
/// "reservation skipped on cycle ≥ 2" mutation fails the pre-verify
/// block count — (b) report AR fallback (`Ok(false)`) with untouched
/// adapter state on a reservation the pool cannot hold — never a turn
/// error — and (c) leave Step-A-style AR decode advancing on the same
/// request afterwards.
#[test]
#[ignore = "requires Metal GPU; run with --ignored"]
fn paged_mtp_lookahead_cycle2_exhaustion_falls_back_to_ar() {
    let mut cfg = tiny_paged_forward_cfg();
    cfg.n_mtp_layers = 1;
    let Some((mut inner, _cfg)) = paged_inner_with_cfg_or_skip(
        "paged_mtp_lookahead_cycle2_exhaustion_falls_back_to_ar",
        cfg,
    ) else {
        return;
    };
    cast_qwen35_inner_weights_bf16(&mut inner);
    inner.mtp_weights_loaded = true;

    // `prompt = 2 * block_size - 4`: the turn-entry reservation covers
    // exactly 2 blocks (cursor + lookahead = 2 * block_size), cycle 1's
    // verify fills them, and the partial-accept rollback leaves the
    // cursor at `2 * block_size - 2` — cycle 2's `+lookahead` margin then
    // needs a 3rd block only a per-cycle reservation can pre-allocate.
    let block_size = inner.paged_adapter.as_ref().expect("adapter").block_size();
    assert!(block_size >= 5, "prompt sizing needs block_size >= 5");
    let prompt: Vec<u32> = (0..(2 * block_size - 4))
        .map(|i| (i * 5 + 7) % 257)
        .collect();
    reset_paged_request(&mut inner, &prompt);
    run_dense_paged_prefill_with_size(&mut inner, &prompt, &prompt, 0, 2048)
        .expect("dense paged prefill")
        .eval();

    let depth = 3usize;
    let plan = inner
        .execution_plan()
        .speculative
        .expect("dense MTP checkpoint must publish a speculative plan");
    let lookahead = plan.lookahead_rows(depth);
    assert_eq!(lookahead, depth + 1);
    let prompt_len = prompt.len() as u32;

    let setup = MtpTurnSetup {
        prompt_hidden: None,
        prompt_hidden_ids: None,
        first_sampled_token: 21,
        lookahead_rows: lookahead,
    };
    let mut step = inner.begin_mtp_decode(&setup).expect("begin_mtp_decode");

    // Cycle 1 in the engine's order: snapshot → verify → partial accept.
    let embedding = step.embedding().clone();
    step.snapshot_main_linear();
    let verify_ids = [21, 22, 23, 24];
    step.verify_step(&verify_ids, &embedding, depth)
        .expect("cycle-1 verify")
        .hiddens
        .eval();
    step.rollback(/* accepted_drafts */ 1, depth);
    assert!(
        step.take_replay_error().is_none(),
        "cycle-1 replay must succeed"
    );
    let (blocks_after_cycle1, cursor_after_cycle1) = {
        let adapter = step.owned_adapter();
        (
            adapter.num_allocated_blocks(),
            adapter.current_token_count(),
        )
    };
    assert_eq!(cursor_after_cycle1, prompt_len + 2);
    assert_eq!(
        blocks_after_cycle1,
        (prompt.len() + lookahead).div_ceil(block_size as usize),
        "cycle 1 stays inside the turn-entry reservation"
    );

    // (a) Cycle 2's reservation covers the MOVED cursor + lookahead.
    assert!(
        step.reserve_cycle_lookahead(lookahead)
            .expect("in-capacity per-cycle reservation"),
        "a reservation the pool can hold must admit cycle 2"
    );
    let reserved_blocks = {
        let adapter = step.owned_adapter();
        assert_eq!(
            adapter.current_token_count(),
            cursor_after_cycle1,
            "the reservation must not advance the cursor"
        );
        adapter.num_allocated_blocks()
    };
    assert_eq!(
        reserved_blocks,
        (cursor_after_cycle1 as usize + lookahead).div_ceil(block_size as usize),
        "cycle 2's reservation must cover cursor + lookahead rows"
    );
    assert!(
        reserved_blocks > blocks_after_cycle1,
        "cycle 2's margin crosses a block boundary — skipping the \
         per-cycle reservation would leave the turn-entry block count"
    );

    // Cycle 2's verify then writes into pre-allocated blocks only.
    step.snapshot_main_linear();
    let verify_ids2 = [25, 26, 27, 28];
    step.verify_step(&verify_ids2, &embedding, depth)
        .expect("cycle-2 verify")
        .hiddens
        .eval();
    {
        let adapter = step.owned_adapter();
        assert_eq!(
            adapter.num_allocated_blocks(),
            reserved_blocks,
            "cycle 2's verify must allocate ZERO new blocks after its reservation"
        );
    }
    step.rollback(/* accepted_drafts */ 1, depth);
    assert!(
        step.take_replay_error().is_none(),
        "cycle-2 replay must succeed"
    );

    // (b) A reservation the pool cannot hold: AR fallback, state untouched.
    let (blocks_before, cursor_before, over_capacity) = {
        let adapter = step.owned_adapter();
        (
            adapter.num_allocated_blocks(),
            adapter.current_token_count(),
            adapter.max_capacity_tokens() as usize,
        )
    };
    let admitted = step
        .reserve_cycle_lookahead(over_capacity)
        .expect("capacity exhaustion must not error the cycle");
    assert!(
        !admitted,
        "an over-capacity per-cycle reservation must signal AR fallback"
    );
    {
        let adapter = step.owned_adapter();
        assert_eq!(adapter.num_allocated_blocks(), blocks_before);
        assert_eq!(adapter.current_token_count(), cursor_before);
    }

    // (c) The AR fallback still advances: one Step-A-style forward on the
    // SAME stepper (the engine's next-iteration path after the skip).
    let next_ids = MxArray::from_int32(&[29], &[1, 1]).expect("ar ids");
    let (logits, _hidden, _needs_squeeze) = step
        .forward_with_hidden(&next_ids, &embedding)
        .expect("AR forward after fallback");
    logits.eval();
    {
        let adapter = step.owned_adapter();
        assert_eq!(
            adapter.current_token_count(),
            cursor_before + 1,
            "AR decode must advance one row past the rewound frontier"
        );
    }

    drop(step);
    assert!(
        inner.paged_adapter.is_some(),
        "the adapter stays on the model for the whole speculative turn"
    );
}
