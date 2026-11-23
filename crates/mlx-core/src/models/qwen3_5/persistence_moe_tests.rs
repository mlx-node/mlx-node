use crate::models::mtp_drafter::strip_wrapper_prefix;
use serde_json::json;

/// The MoE config refuses a scaled rope and hands its YaRN to the dense
/// config every attention layer is built from. Mutation:
/// `to_dense_config` drops the scaling → red.
#[test]
fn moe_dense_config_carries_the_scaling() {
    let raw = json!({
        "model_type": "qwen3_5_moe",
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "intermediate_size": 128,
        "num_experts": 2,
        "max_position_embeddings": 262_144,
        "rope_parameters": {"rope_type": "default"},
    });
    let mut config = super::parse_moe_config(&raw).unwrap();
    config.rope_scaling = Some(
        crate::nn::rope_scaling::RopeScaling::served_for(
            "Qwen3.5 MoE",
            262_144,
            config.scaling(),
            Some(409_600),
        )
        .unwrap(),
    );
    assert!(config.scaling().yarn());
    assert_eq!(config.to_dense_config().scaling(), config.scaling());
    assert_eq!(config.served_context_tokens(), 409_600);
    let mut scaled = raw.clone();
    scaled["rope_parameters"] = json!({"rope_type": "yarn"});
    let error = super::parse_moe_config(&scaled).unwrap_err();
    assert!(
        error.reason.starts_with("Qwen3.5 MoE config:"),
        "{}",
        error.reason
    );
}

/// The preload equals what the directory loader reads from the
/// converter's output: the converter's tensors (the checked-in golden
/// manifest) plus the biases the loader derives for symmetric blocks.
/// Not a live conversion: that flips MLX's process-wide default device to
/// CPU under every test running beside it.
#[tokio::test]
async fn moe_loader_uses_preload_not_files() {
    use std::collections::{BTreeMap, HashMap};

    use super::{CheckpointSource, load_moe_weights};
    use crate::array::MxArray;
    use crate::engine::persistence::expand_symmetric_affine_biases;
    use crate::utils::gguf::NativeGgufFamily;
    use crate::utils::gguf_import::gate::manifest;
    use crate::utils::gguf_test_fixtures::{import_golden_root, sha256_hex, write_import_fixture};

    const FIXTURE: &str = "qwen35moe-3d-experts";
    let golden = import_golden_root().join(FIXTURE);
    let read_json = |name: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(golden.join(name)).unwrap())
            .unwrap_or_else(|e| panic!("{FIXTURE}/{name}: {e}"))
    };
    let work = std::env::temp_dir().join(format!("mlx-moe-preload-{}", uuid::Uuid::new_v4()));
    let fx = write_import_fixture(FIXTURE, &work.join("src"));
    assert_eq!(
        read_json("fixture.json")["main"].as_str(),
        Some(sha256_hex(&std::fs::read(&fx.main).unwrap()).as_str()),
        "fixture builder drifted from the golden"
    );
    let converted: BTreeMap<String, serde_json::Value> =
        serde_json::from_value(read_json("model.manifest.json")).unwrap();

    let source = CheckpointSource::resolve_in(
        &fx.main,
        NativeGgufFamily::Qwen35Moe,
        Some(&work.join("cache")),
    )
    .await
    .unwrap();
    let CheckpointSource::Gguf(ckpt) = &source else {
        panic!("the MoE GGUF resolved to a plain directory");
    };
    ckpt.check_asset_config().unwrap();
    let weight_files: Vec<_> = std::fs::read_dir(&ckpt.asset_dir)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .ends_with(".safetensors")
        })
        .map(|entry| entry.file_name())
        .collect();
    assert!(
        weight_files.is_empty(),
        "the asset directory holds weight files: {weight_files:?}"
    );

    let loaded = load_moe_weights(&ckpt.asset_dir, Some(&ckpt.preload_input(false)));
    assert!(
        loaded.is_ok(),
        "the MoE loader did not take its weights from the preload: {}",
        loaded
            .as_ref()
            .err()
            .map(|e| e.reason.as_str())
            .unwrap_or_default()
    );
    let loaded = loaded.unwrap();
    let mut dir_load: HashMap<String, MxArray> = converted
        .keys()
        .map(|name| {
            let array = loaded
                .get(name)
                .unwrap_or_else(|| panic!("the preload lacks converted tensor {name}"));
            (name.clone(), array.clone())
        })
        .collect();
    for (name, entry) in &manifest(&dir_load) {
        assert_eq!(Some(entry), converted.get(name), "tensor {name} differs");
    }
    expand_symmetric_affine_biases(&golden, &mut dir_load).unwrap();
    let want = manifest(&dir_load);
    let got = manifest(&loaded);
    assert_eq!(
        got.len(),
        want.len(),
        "preload tensor count differs from the converted checkpoint"
    );
    for (name, entry) in &want {
        assert_eq!(got.get(name), Some(entry), "tensor {name} differs");
    }
    source.release_lease();
    std::fs::remove_dir_all(&work).ok();
}

/// Gate: an unset MoE format is the geometry default (int8 on Metal at
/// 16 x 2 x 256 with BF16 compute), and an explicit value wins both ways.
/// Mutation: restore `.or_else(|| Some("bf16".to_string()))` in
/// `to_dense_config` → red (dense `kv_format` is `Some("bf16")`; on Metal
/// the default is Bf16, not Int8).
#[test]
fn moe_kv_format_follows_the_geometry_default_and_explicit_format() {
    use crate::transformer::KvFormat;
    let mut raw = json!({
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "moe_intermediate_size": 512,
        "num_experts": 256
    });
    let default = parse_moe_config(&raw).unwrap();
    assert_eq!(default.to_dense_config().kv_format, None);
    assert_eq!(
        default.kv_format_for(DType::BFloat16),
        KvFormat::default_for_geometry(256, 16, 2, DType::BFloat16)
    );
    let metal = unsafe { mlx_sys::mlx_metal_is_available() }
        && unsafe { mlx_sys::mlx_default_device() } == 1;
    if metal {
        assert_eq!(default.kv_format_for(DType::BFloat16), KvFormat::Int8);
    }
    for (value, format) in [("bf16", KvFormat::Bf16), ("int8", KvFormat::Int8)] {
        raw["kv_format"] = json!(value);
        let explicit = parse_moe_config(&raw).unwrap();
        assert_eq!(explicit.kv_format(), format);
        assert_eq!(explicit.kv_format_for(DType::Float16), format);
        assert_eq!(explicit.to_dense_config().kv_format(), format);
        // A runtime option, never saved (Splash `--kv-format`).
        let restored: Qwen3_5MoeConfig =
            serde_json::from_value(serde_json::to_value(&explicit).unwrap()).unwrap();
        assert_eq!(restored.kv_format, None);
    }
}

/// Gate: the 35B-A3B DFlash2 lane is priced in the int8 rows its load
/// pins for BF16 compute: 10,400 B/token over its 10 full-attention
/// layers; BF16 rows are 20,480.
/// Mutation: restore `.or_else(|| Some("bf16".to_string()))` in
/// `to_dense_config` → red (20,480).
#[test]
fn moe_dflash2_lane_prices_the_default_format() {
    use crate::array::DType;
    use crate::engine::memory_plan::KvLayout;
    use crate::transformer::KvFormat;
    let metal = unsafe { mlx_sys::mlx_metal_is_available() }
        && unsafe { mlx_sys::mlx_default_device() } == 1;
    if !metal {
        return;
    }
    let raw = json!({
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "moe_intermediate_size": 512,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262144
    });
    let mut cfg = parse_moe_config(&raw).unwrap();
    assert_eq!(cfg.recurrent_state_bytes(), 64_389_120);
    cfg.pin_kv_format(None, DType::BFloat16);
    let dense = cfg.to_dense_config();
    let row_bytes = |format| {
        KvLayout::of(
            dense.full_attention_layer_count(),
            dense.num_kv_heads,
            dense.head_dim,
            DType::BFloat16,
            format,
        )
        .row_bytes()
    };
    assert_eq!(row_bytes(dense.kv_format()), 10_400);
    assert_eq!(row_bytes(KvFormat::Bf16), 20_480);
}

#[test]
fn paged_config_reads_root_options_without_losing_absence() {
    let mut raw = json!({
        "text_config": {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 128,
            "num_experts": 2,
            "paged_cache_memory_mb": 999,
            "paged_block_size": 999
        }
    });
    let absent = parse_moe_config(&raw).unwrap();
    assert_eq!(absent.paged_cache_memory_mb, None);
    assert_eq!(absent.paged_block_size, None);
    raw["paged_cache_memory_mb"] = json!(4294967360_u64);
    raw["paged_block_size"] = json!(-1);
    raw["pagedBlockSize"] = json!(32);
    raw["use_block_paged_cache"] = json!(false);
    let config = parse_moe_config(&raw).unwrap();
    assert_eq!(config.paged_cache_memory_mb, Some(64));
    assert_eq!(config.paged_block_size, None);
}

/// The MoE body strip (`sanitize_weights`) delegates to the shared
/// longest-first `strip_wrapper_prefix`, so a raw, un-converted HF
/// VLM-wrapped checkpoint's triple-prefixed inline-MTP key is normalized to
/// the canonical `mtp.*` form instead of being silently dropped (the
/// shorter `model.language_model.` strip would have left
/// `model.mtp.layers.0...`).
#[test]
fn moe_body_strip_survives_triple_wrapped_mtp_key() {
    assert_eq!(
        strip_wrapper_prefix("model.language_model.model.mtp.layers.0.input_layernorm.weight"),
        "mtp.layers.0.input_layernorm.weight"
    );
}

use super::{
    DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE, DType, MxArray, PerLayerMode, PerLayerQuant,
    Qwen3_5MoeConfig, Qwen35Inner, align_affine_embedding_dtype, apply_weights_moe_inner,
    declared_residual_dtype, default_per_layer_quant, load_vision_encoder_moe, parse_moe_config,
    residual_stream_dtype, sanitize_weights_moe,
};
use crate::models::qwen3_5::decoder_layer::{AttentionType, MLPType};
use std::collections::HashMap;

/// `dtype` is the current Transformers spelling and `torch_dtype` the
/// legacy one; the current key wins, `null` and unknown spellings declare
/// nothing, and a flat (GGUF-synthesized) config is read too.
#[test]
fn declared_residual_dtype_reads_dtype_before_torch_dtype() {
    assert_eq!(
        declared_residual_dtype(&serde_json::json!({
            "text_config": {"dtype": "bfloat16", "torch_dtype": "float16"}
        })),
        Some(DType::BFloat16)
    );
    assert_eq!(
        declared_residual_dtype(&serde_json::json!({"text_config": {"torch_dtype": "F16"}})),
        Some(DType::Float16)
    );
    assert_eq!(
        declared_residual_dtype(&serde_json::json!({"dtype": "BFloat16"})),
        Some(DType::BFloat16)
    );
    assert_eq!(
        declared_residual_dtype(&serde_json::json!({
            "text_config": {"dtype": null, "torch_dtype": "float8_e4m3fn"}
        })),
        None
    );
    assert_eq!(declared_residual_dtype(&serde_json::json!({})), None);
}

/// Regression for the AgentWorld UD-Q4_K_XL import: a Q8_0 `token_embd`
/// lands as an affine-8 group whose ggml block scales are fp16, while every
/// dense tensor the native import wrote is bf16. The dequantized table is
/// fp16, `fast::rms_norm` promotes its output to `result_type(x, weight)` =
/// fp32, and the residual stream never returns to a 2-byte dtype — the
/// first paged KV write then aborts with "input dtype Float32 not supported
/// by LayerKVPool". Aligning the group's sidecars at load keeps the stream
/// in the checkpoint's dtype.
#[test]
fn affine_fp16_embedding_sidecars_cannot_promote_the_residual_stream() {
    let (vocab, hidden) = (8u32, 64u32);
    let table = MxArray::from_float32(
        &vec![0.01f32; (vocab * hidden) as usize],
        &[vocab as i64, hidden as i64],
    )
    .unwrap()
    .astype(DType::BFloat16)
    .unwrap();
    let (mut qw, mut qs, mut qb) = (
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    );
    assert!(unsafe {
        mlx_sys::mlx_quantize(
            table.as_raw_ptr(),
            32,
            8,
            c"affine".as_ptr(),
            &mut qw,
            &mut qs,
            &mut qb,
        )
    });
    let weight = MxArray::from_handle(qw, "qw").unwrap();
    // ggml stores the block scale in fp16 and MLX reports
    // `result_type(scales, biases)` as the group's dequant dtype.
    let scales = MxArray::from_handle(qs, "qs")
        .unwrap()
        .astype(DType::Float16)
        .unwrap();
    let biases = MxArray::from_handle(qb, "qb")
        .unwrap()
        .astype(DType::Float16)
        .unwrap();
    // Every dense tensor the import wrote is bf16.
    let norm_weight = MxArray::from_float32(&vec![1.0f32; hidden as usize], &[hidden as i64])
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap();

    let stream_dtype = |scales: &MxArray, biases: &MxArray| -> DType {
        let mut embedding = crate::nn::Embedding::new(vocab, hidden).unwrap();
        embedding
            .load_quantized(&weight, scales, Some(biases), 32, 8, "affine")
            .unwrap();
        let ids = MxArray::from_uint32(&[0, 1, 2], &[1, 3]).unwrap();
        let embedded = embedding.forward(&ids).unwrap();
        let mut norm = crate::nn::RMSNorm::new(hidden, Some(1e-6)).unwrap();
        norm.set_weight(&norm_weight).unwrap();
        norm.forward(&embedded).unwrap().dtype().unwrap()
    };

    // As the import lands today: fp16 sidecars → fp16 table → the bf16 norm
    // weight promotes the residual stream to fp32.
    assert_eq!(stream_dtype(&scales, &biases), DType::Float32);

    let mut params: HashMap<String, MxArray> = HashMap::new();
    params.insert("embedding.weight".to_string(), weight.clone());
    params.insert("embedding.scales".to_string(), scales);
    params.insert("embedding.biases".to_string(), biases);
    params.insert(
        "layers.0.input_layernorm.weight".to_string(),
        norm_weight.clone(),
    );
    // A GGUF-synthesized config declares no dtype; the checkpoint's own
    // dense weights are the reference.
    let target = residual_stream_dtype(&serde_json::json!({}), &params);
    assert_eq!(target, Some(DType::BFloat16));
    assert!(align_affine_embedding_dtype(&mut params, target).unwrap());
    assert_eq!(params["embedding.scales"].dtype().unwrap(), DType::BFloat16);
    assert_eq!(params["embedding.biases"].dtype().unwrap(), DType::BFloat16);

    let aligned_scales = params["embedding.scales"].clone();
    let aligned_biases = params["embedding.biases"].clone();
    assert_eq!(
        stream_dtype(&aligned_scales, &aligned_biases),
        DType::BFloat16
    );
}

/// Regression for `mlx convert --dtype float16`: convert casts every
/// float tensor — norms included — but copies config.json verbatim, so
/// the output still declares `torch_dtype: bfloat16`. Declared-first
/// precedence picked bf16, `align_affine_embedding_dtype` cast the
/// embedding sidecars to bf16, the first f16 RMSNorm promoted the
/// residual stream to f32, and the paged KV pool aborted with "input
/// dtype Float32 not supported by LayerKVPool". The loaded norm dtype is
/// the ground truth; the declaration is only the fallback for a config
/// that declares nothing (the GGUF-synthesized standalone config).
#[test]
fn residual_stream_dtype_prefers_the_loaded_norm_over_a_stale_declaration() {
    let hidden = 64u32;
    let mut params: HashMap<String, MxArray> = HashMap::new();
    params.insert(
        "layers.0.input_layernorm.weight".to_string(),
        MxArray::from_float32(&vec![1.0f32; hidden as usize], &[hidden as i64])
            .unwrap()
            .astype(DType::Float16)
            .unwrap(),
    );
    params.insert(
        "embedding.weight".to_string(),
        MxArray::from_uint32(&[0u32; 16], &[8, 2]).unwrap(),
    );
    params.insert(
        "embedding.scales".to_string(),
        MxArray::from_float32(&[1.0f32; 16], &[8, 2]).unwrap(),
    );
    params.insert(
        "embedding.biases".to_string(),
        MxArray::from_float32(&[0.0f32; 16], &[8, 2]).unwrap(),
    );

    // The verbatim-copied config still declares the SOURCE dtype; the f16
    // norm wins, and the fp32 sidecars follow it — not the declaration.
    let target = residual_stream_dtype(&serde_json::json!({"torch_dtype": "bfloat16"}), &params);
    assert_eq!(target, Some(DType::Float16));
    assert!(align_affine_embedding_dtype(&mut params, target).unwrap());
    assert_eq!(params["embedding.scales"].dtype().unwrap(), DType::Float16);
    assert_eq!(params["embedding.biases"].dtype().unwrap(), DType::Float16);

    // A config that declares nothing still resolves from the probe alone.
    assert_eq!(
        residual_stream_dtype(&serde_json::json!({}), &params),
        Some(DType::Float16)
    );
    // And with nothing to probe, the declaration is the only signal left.
    let declared_only: HashMap<String, MxArray> = HashMap::new();
    assert_eq!(
        residual_stream_dtype(&serde_json::json!({"dtype": "bfloat16"}), &declared_only),
        Some(DType::BFloat16)
    );
}

/// Gate: the load pins the K/V format for the checkpoint's compute
/// dtype. An `mlx convert --dtype float16` MoE at 16 x 2 x 256 (f16 norms,
/// fp32 affine embedding sidecars, a stale `torch_dtype: bfloat16`) gets
/// BF16 K/V — the int8 verify needs BF16 queries — while the same
/// checkpoint with BF16 norms gets int8 on Metal. An explicit int8 (load
/// option or config.json) still wins over f16 compute.
/// Mutations: pass `DType::BFloat16` to `pin_kv_format` in
/// `settle_compute_dtype` → red (f16 gives Int8); pin before
/// `align_affine_embedding_dtype` → red (the BF16-norm checkpoint reads
/// its fp32 sidecars and gives Bf16); apply the compute check to an
/// explicit format → red (explicit int8 gives Bf16).
#[test]
fn moe_load_pins_the_kv_format_for_the_compute_dtype() {
    use super::settle_compute_dtype;
    use crate::transformer::KvFormat;
    let metal = unsafe { mlx_sys::mlx_metal_is_available() }
        && unsafe { mlx_sys::mlx_default_device() } == 1;
    let mut raw = serde_json::json!({
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "moe_intermediate_size": 16,
        "num_experts": 2,
        "torch_dtype": "bfloat16"
    });
    let checkpoint = |norm: DType| {
        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert(
            "layers.0.input_layernorm.weight".to_string(),
            MxArray::from_float32(&[1.0f32; 64], &[64])
                .unwrap()
                .astype(norm)
                .unwrap(),
        );
        params.insert(
            "embedding.weight".to_string(),
            MxArray::from_uint32(&[0u32; 16], &[8, 2]).unwrap(),
        );
        params.insert(
            "embedding.scales".to_string(),
            MxArray::from_float32(&[1.0f32; 16], &[8, 2]).unwrap(),
        );
        params.insert(
            "embedding.biases".to_string(),
            MxArray::from_float32(&[0.0f32; 16], &[8, 2]).unwrap(),
        );
        params
    };
    let settle = |raw: &serde_json::Value, norm: DType, requested: Option<KvFormat>| {
        let mut config = parse_moe_config(raw).unwrap();
        let mut params = checkpoint(norm);
        let format = settle_compute_dtype(raw, &mut params, &mut config, requested).unwrap();
        assert_eq!(config.kv_format.as_deref(), Some(format.as_str()));
        assert_eq!(config.kv_format(), format);
        assert_eq!(config.to_dense_config().kv_format(), format);
        format
    };

    assert_eq!(settle(&raw, DType::Float16, None), KvFormat::Bf16, "f16");
    let bf16_default = if metal {
        KvFormat::Int8
    } else {
        KvFormat::Bf16
    };
    assert_eq!(settle(&raw, DType::BFloat16, None), bf16_default, "bf16");
    assert_eq!(
        settle(&raw, DType::Float16, Some(KvFormat::Int8)),
        KvFormat::Int8,
        "load option int8 at f16"
    );
    raw["kv_format"] = serde_json::json!("int8");
    assert_eq!(
        settle(&raw, DType::Float16, None),
        KvFormat::Int8,
        "config.json int8 at f16"
    );
    assert_eq!(
        settle(&raw, DType::BFloat16, Some(KvFormat::Bf16)),
        KvFormat::Bf16,
        "load option over config.json"
    );
}

/// Integer sidecars mean a K/IQ group whatever the quantization mode says,
/// and those decode to bf16 on their own; a group that already matches the
/// checkpoint dtype is left byte-identical.
#[test]
fn affine_embedding_alignment_ignores_integer_and_matching_sidecars() {
    let mut kquant: HashMap<String, MxArray> = HashMap::new();
    kquant.insert(
        "embedding.weight".to_string(),
        MxArray::from_uint32(&[0u32; 16], &[8, 2]).unwrap(),
    );
    kquant.insert(
        "embedding.scales".to_string(),
        MxArray::from_float32(&[1.0f32; 16], &[8, 2])
            .unwrap()
            .astype(DType::Int8)
            .unwrap(),
    );
    kquant.insert(
        "embedding.biases".to_string(),
        MxArray::from_float16(&[0u16; 8], &[8, 1]).unwrap(),
    );
    assert!(!align_affine_embedding_dtype(&mut kquant, Some(DType::BFloat16)).unwrap());
    assert_eq!(
        kquant["embedding.scales"].dtype().unwrap(),
        DType::Int8,
        "a K/IQ group's sidecars are not affine floats"
    );

    let mut aligned: HashMap<String, MxArray> = HashMap::new();
    aligned.insert(
        "embedding.weight".to_string(),
        MxArray::from_uint32(&[0u32; 16], &[8, 2]).unwrap(),
    );
    aligned.insert(
        "embedding.scales".to_string(),
        MxArray::from_float32(&[1.0f32; 16], &[8, 2])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap(),
    );
    assert!(!align_affine_embedding_dtype(&mut aligned, Some(DType::BFloat16)).unwrap());

    // A dense embedding (no group at all) and a target the paged KV pool
    // cannot hold are both no-ops.
    let mut dense: HashMap<String, MxArray> = HashMap::new();
    dense.insert(
        "embedding.weight".to_string(),
        MxArray::from_float32(&[0.0f32; 16], &[8, 2]).unwrap(),
    );
    assert!(!align_affine_embedding_dtype(&mut dense, Some(DType::BFloat16)).unwrap());
    assert!(!align_affine_embedding_dtype(&mut kquant, Some(DType::Float32)).unwrap());
}

#[test]
fn native_tiled_gdn_layout_survives_moe_parse_and_dense_projection() {
    let config = super::parse_moe_config(&serde_json::json!({
        "vocab_size": 128,
        "hidden_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 128,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "qwen35_gguf_gdn_layout": "tiled"
    }))
    .unwrap();

    assert_eq!(config.qwen35_gguf_gdn_layout.as_deref(), Some("tiled"));
    assert_eq!(
        config.to_dense_config().qwen35_gguf_gdn_layout.as_deref(),
        Some("tiled")
    );
}

/// Paged, tied, non-MTP, non-VLM `Qwen3_5MoeConfig` fixture. `head_dim = 32`
/// (smallest valid block-paged pool head size); `hidden_size = num_heads *
/// head_dim = 128`. Mirrors `paged_forward::tests::moe_paged_tiny_config`.
fn moe_paged_tiny_cfg() -> Qwen3_5MoeConfig {
    Qwen3_5MoeConfig {
        rope_scaling: None,
        vocab_size: 128,
        hidden_size: 128,
        num_layers: 8,
        num_heads: 8,
        num_kv_heads: 2,
        intermediate_size: 128,
        rms_norm_eps: 1e-6,
        head_dim: 128,
        tie_word_embeddings: true,
        attention_bias: false,
        max_position_embeddings: 256,
        pad_token_id: 0,
        eos_token_id: 0,
        bos_token_id: 0,
        linear_num_value_heads: 4,
        linear_num_key_heads: 2,
        linear_key_head_dim: 32,
        linear_value_head_dim: 32,
        linear_conv_kernel_dim: 4,
        full_attention_interval: 4,
        partial_rotary_factor: 0.25,
        rope_theta: 100_000.0,
        num_experts: 4,
        num_experts_per_tok: 2,
        decoder_sparse_step: 1,
        shared_expert_intermediate_size: None,
        moe_intermediate_size: None,
        norm_topk_prob: true,
        mlp_only_layers: None,
        paged_cache_memory_mb: Some(256),
        paged_block_size: Some(16),
        n_mtp_layers: 0,
        qwen35_gguf_gdn_layout: None,
        kv_format: None,
    }
}

#[test]
fn mixed_q4k_dense_split_gdn_projections_install_through_moe_loader() {
    let label = "mixed_q4k_dense_split_gdn_projections_install_through_moe_loader";
    let mut config = moe_paged_tiny_cfg();
    config.paged_cache_memory_mb = None;
    let mut inner = match Qwen35Inner::new_moe(config.clone()) {
        Ok(inner) => inner,
        Err(error) => {
            let message = error.reason.to_string();
            if message.contains("Metal")
                || message.contains("device")
                || message.contains("LayerKVPool")
            {
                eprintln!("skipping {label} (MLX/Metal unavailable): {message}");
                return;
            }
            panic!("unexpected Qwen35Inner::new failure in {label}: {message}");
        }
    };

    let dense_bf16 = |rows: i64, cols: i64| {
        MxArray::from_float32(&vec![0.01; (rows * cols) as usize], &[rows, cols])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap()
    };
    let insert_q4k = |params: &mut HashMap<String, MxArray>, prefix: &str, rows: i64| {
        params.insert(
            format!("{prefix}.weight"),
            MxArray::from_uint32(&vec![0; (rows * 4) as usize], &[rows, 4]).unwrap(),
        );
        params.insert(
            format!("{prefix}.scales"),
            MxArray::from_float32(&vec![1.0; (rows * 2) as usize], &[rows, 2])
                .unwrap()
                .astype(DType::Uint8)
                .unwrap(),
        );
        params.insert(
            format!("{prefix}.biases"),
            MxArray::from_float16(
                &vec![half::f16::from_f32(0.5).to_bits(); rows as usize],
                &[rows, 1],
            )
            .unwrap(),
        );
    };

    let prefix = "layers.0.linear_attn";
    let qkv_prefix = format!("{prefix}.in_proj_qkv");
    let z_prefix = format!("{prefix}.in_proj_z");
    let b_prefix = format!("{prefix}.in_proj_b");
    let a_prefix = format!("{prefix}.in_proj_a");
    let mut params = HashMap::new();
    insert_q4k(&mut params, &qkv_prefix, 256);
    params.insert(format!("{z_prefix}.weight"), dense_bf16(128, 128));
    insert_q4k(&mut params, &b_prefix, 4);
    params.insert(format!("{a_prefix}.weight"), dense_bf16(4, 128));

    let q4k = PerLayerQuant {
        bits: 4,
        group_size: 32,
        mode: PerLayerMode::Q4K,
        input_amax: None,
        layout: Default::default(),
    };
    let per_layer_quant = HashMap::from([(qkv_prefix, q4k), (b_prefix, q4k)]);

    let error = apply_weights_moe_inner(
        &mut inner,
        &params,
        &config,
        DEFAULT_QUANT_BITS,
        DEFAULT_QUANT_GROUP_SIZE,
        None,
        &per_layer_quant,
        false,
    )
    .expect_err("the intentionally partial checkpoint must fail completeness validation");
    assert!(
        error.reason.contains("missing mandatory weights"),
        "mixed split installation must reach the final completeness gate: {}",
        error.reason
    );

    match &inner.layers[0].attn {
        AttentionType::Linear(gdn) => assert_eq!(
            gdn.split_in_proj_quantized_sides(),
            (Some((true, false)), Some((true, false))),
            "each Q4_K half must stay packed while its BF16 peer stays dense"
        ),
        AttentionType::Full(_) => panic!("layer 0 must be a GDN layer"),
    }
}

/// MoE mirror of the dense `tied_quantized_embedding_loads_via_packed_path`:
/// a quantized `embedding.*` sidecar on a flat, declared-MTP, VLM
/// `Qwen3_5MoeConfig` must load via `Embedding::load_quantized_packed`
/// (packed-resident, so the tied lm_head runs `quantized_matmul` via
/// `as_linear` on the paged path) — NOT the legacy full-table pre-dequant
/// `Embedding::load_quantized`. Guards `apply_weights_moe_inner` directly
/// and covers all three former packed-load exclusions in one fixture.
#[test]
fn tied_quantized_embedding_loads_via_packed_path_moe() {
    let label = "tied_quantized_embedding_loads_via_packed_path_moe";
    let mut cfg = moe_paged_tiny_cfg();
    cfg.paged_cache_memory_mb = None;
    cfg.n_mtp_layers = 1;

    let mut inner = match Qwen35Inner::new_moe(cfg.clone()) {
        Ok(inner) => inner,
        Err(err) => {
            let msg = err.reason.to_string();
            // Pool allocation (`LayerKVPool`) requires Metal; skip cleanly
            // when the GPU/device is unavailable (CI without Metal).
            if msg.contains("Metal") || msg.contains("device") || msg.contains("LayerKVPool") {
                eprintln!("skipping {label} (MLX/Metal unavailable): {msg}");
                return;
            }
            panic!("unexpected Qwen35Inner::new failure in {label}: {msg}");
        }
    };

    let vocab = cfg.vocab_size as i64;
    let hidden = cfg.hidden_size as i64;
    let n = (vocab * hidden) as usize;
    let data: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
    let dense = match MxArray::from_float32(&data, &[vocab, hidden]) {
        Ok(a) => match a.astype(DType::BFloat16) {
            Ok(a) => a,
            Err(err) => panic!("unexpected astype failure in {label}: {}", err.reason),
        },
        Err(err) => {
            let msg = err.reason.to_string();
            if msg.contains("Metal") || msg.contains("device") {
                eprintln!("skipping {label} (MLX/Metal unavailable): {msg}");
                return;
            }
            panic!("unexpected MxArray::from_float32 failure in {label}: {msg}");
        }
    };

    let group_size = 32;
    let bits = 4;
    let mut out_q: *mut mlx_sys::mlx_array = std::ptr::null_mut();
    let mut out_s: *mut mlx_sys::mlx_array = std::ptr::null_mut();
    let mut out_b: *mut mlx_sys::mlx_array = std::ptr::null_mut();
    let ok = unsafe {
        mlx_sys::mlx_quantize(
            dense.as_raw_ptr(),
            group_size,
            bits,
            c"affine".as_ptr(),
            &mut out_q,
            &mut out_s,
            &mut out_b,
        )
    };
    assert!(ok, "mlx_quantize affine failed");
    let qw = MxArray::from_handle(out_q, "qw").expect("qw");
    let qs = MxArray::from_handle(out_s, "qs").expect("qs");
    let qb = MxArray::from_handle(out_b, "qb").expect("qb");

    let mut params: HashMap<String, MxArray> = HashMap::new();
    params.insert("embedding.weight".to_string(), qw);
    params.insert("embedding.scales".to_string(), qs);
    params.insert("embedding.biases".to_string(), qb);

    let mut per_layer_quant: HashMap<String, PerLayerQuant> = HashMap::new();
    per_layer_quant.insert(
        "embed_tokens".to_string(),
        PerLayerQuant {
            bits,
            group_size,
            mode: PerLayerMode::Affine,
            input_amax: None,
            layout: Default::default(),
        },
    );

    // Same tolerate-completeness rationale as the dense mirror: the
    // embedding loads UP FRONT, then `apply_weights_moe_inner`'s end-of-
    // function completeness gate rejects this embedding-only fixture. The
    // Err fires after the embedding backend is installed, so the packed
    // assertion below still observes the real load decision.
    match apply_weights_moe_inner(
        &mut inner,
        &params,
        &cfg,
        DEFAULT_QUANT_BITS,
        DEFAULT_QUANT_GROUP_SIZE,
        None,
        &per_layer_quant,
        /* has_vision */ true,
    ) {
        Ok(()) => {}
        Err(err) => {
            let msg = err.reason.to_string();
            assert!(
                msg.contains("missing mandatory weights"),
                "unexpected apply_weights_moe_inner error in {label}: {msg}"
            );
        }
    }

    assert!(
        inner.embedding.is_packed_quantized(),
        "tied+quantized MoE embedding.* must stay packed across flat, declared-MTP, and VLM configs"
    );
    assert!(
        !inner.has_mtp_weights(),
        "declaring MoE MTP in config without loading its weights must not enable runtime MTP"
    );
}

// ===== sym8 (per-output-channel symmetric int8) loader dispatch =====
//
// qwen3_5_moe's non-expert sublayers (attention q/k/v/o, GDN
// in_proj_qkvz/in_proj_ba/out_proj, shared-expert MLP body) reuse dense
// qwen3_5's LinearProj/QuantizedLinear machinery, so the sym8 int8 W8A8
// backend is family-agnostic. These tests pin the LOADER seam: a
// sym8-default checkpoint must install the raw int8 backend on non-expert
// linears, the per-expert switch_mlp gather path (no sym8 kernel) must
// fail loud on a sym8 override, and an MTP-capable sym8 checkpoint must
// load with the speculative MTP head disabled.

/// Flat (non-paged), tied, non-MTP tiny MoE config for the sym8 loader
/// tests. `full_attention_interval = 1` makes layer 0 Full attention and
/// `decoder_sparse_step = 1` makes it MoE; `hidden_size = num_heads *
/// head_dim = 64`, so q_proj's K = 64 satisfies the sym8 kernel's
/// K % 16 == 0 contract.
fn tiny_sym8_moe_cfg() -> Qwen3_5MoeConfig {
    Qwen3_5MoeConfig {
        rope_scaling: None,
        vocab_size: 8,
        hidden_size: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 2,
        intermediate_size: 64,
        rms_norm_eps: 1e-6,
        head_dim: 16,
        tie_word_embeddings: true,
        attention_bias: false,
        max_position_embeddings: 256,
        pad_token_id: 0,
        eos_token_id: 0,
        bos_token_id: 0,
        linear_num_value_heads: 4,
        linear_num_key_heads: 2,
        linear_key_head_dim: 32,
        linear_value_head_dim: 32,
        linear_conv_kernel_dim: 4,
        full_attention_interval: 1,
        partial_rotary_factor: 0.25,
        rope_theta: 100_000.0,
        num_experts: 4,
        num_experts_per_tok: 2,
        decoder_sparse_step: 1,
        shared_expert_intermediate_size: None,
        moe_intermediate_size: None,
        norm_topk_prob: true,
        mlp_only_layers: None,
        paged_cache_memory_mb: None,
        paged_block_size: None,
        n_mtp_layers: 0,
        qwen35_gguf_gdn_layout: None,
        kv_format: None,
    }
}

/// Synthesize a well-formed sym8 group under `{base}.*`: int8 `[n,k]`
/// weight (values in [-127,127]) + positive f32 `[n]` scales. Mirrors the
/// dense/lfm2 sym8 test helper of the same name.
fn synth_sym8_group(p: &mut HashMap<String, MxArray>, base: &str, n: i64, k: i64) {
    let q: Vec<f32> = (0..n * k).map(|i| ((i % 255) - 127) as f32).collect();
    let w = MxArray::from_float32(&q, &[n, k])
        .expect("from_float32")
        .astype(DType::Int8)
        .expect("astype int8");
    let s: Vec<f32> = (0..n).map(|i| 0.001 + (i as f32) * 1e-4).collect();
    p.insert(format!("{base}.weight"), w);
    p.insert(
        format!("{base}.scales"),
        MxArray::from_float32(&s, &[n]).expect("scales"),
    );
}

#[test]
fn plain_fp8_expert_trio_dispatches_through_moe_loader() {
    let config = tiny_sym8_moe_cfg();
    let mut inner = Qwen35Inner::new_moe(config.clone()).expect("Qwen35Inner::new must succeed");

    let mut params: HashMap<String, MxArray> = HashMap::new();
    params.insert(
        "embedding.weight".to_string(),
        MxArray::from_float32(&vec![0.0f32; 8 * 64], &[8, 64]).expect("embedding"),
    );
    params.insert(
        "final_norm.weight".to_string(),
        MxArray::from_float32(&vec![1.0f32; 64], &[64]).expect("final_norm"),
    );
    // A dense q_proj is enough for the loader's attention completeness
    // gate; the FP8 expert sidecars below make the checkpoint quantized.
    params.insert(
        "layers.0.self_attn.q_proj.weight".to_string(),
        MxArray::from_float32(&vec![0.0f32; 128 * 64], &[128, 64]).expect("q_proj"),
    );
    params.insert(
        "layers.0.mlp.gate.weight".to_string(),
        MxArray::from_float32(&vec![0.0f32; 4 * 64], &[4, 64]).expect("router gate"),
    );

    let mut per_layer_quant = HashMap::new();
    for suffix in ["gate_proj", "up_proj", "down_proj"] {
        let base = format!("layers.0.mlp.switch_mlp.{suffix}");
        let source = MxArray::from_float32(&vec![0.25f32; 4 * 64 * 64], &[4, 64, 64])
            .expect("expert source")
            .astype(DType::BFloat16)
            .expect("expert bf16");
        let (weight, scales) =
            crate::quant::fp8_weight::quantize_per_output_channel(&source, &base)
                .expect("plain fp8 expert quantization");
        params.insert(format!("{base}.weight"), weight);
        params.insert(format!("{base}.scales"), scales);
        per_layer_quant.insert(
            base,
            default_per_layer_quant(
                crate::quant::fp8_weight::FP8_E4M3_BITS,
                crate::quant::fp8_weight::FP8_E4M3_GROUP_SIZE,
                PerLayerMode::Fp8E4m3,
            ),
        );
    }

    apply_weights_moe_inner(
        &mut inner,
        &params,
        &config,
        4,
        32,
        None,
        &per_layer_quant,
        /* has_vision */ false,
    )
    .expect("plain FP8 switch_mlp expert trio must load");

    match &inner.layers[0].mlp {
        MLPType::MoE(moe) => {
            let switch = moe.get_switch_mlp();
            assert!(
                switch.is_quantized(),
                "plain FP8 trio must install the quantized SwitchGLU backend"
            );
            let weight = switch.get_gate_proj_weight();
            assert_eq!(weight.dtype().unwrap(), DType::Uint8);
            assert_eq!(weight.shape().unwrap().to_vec(), vec![4, 64, 64]);
        }
        _ => panic!("layer 0 must be MoE (decoder_sparse_step = 1)"),
    }
}

/// A pre-stacked 3-D K-quant `switch_mlp.*` trio — exactly what the native
/// GGUF importer writes for `ffn_{gate,up,down}_exps` — must load through
/// the quantized switch backend, and a truncated group (`.scales` without
/// `.biases`) must fail loud instead of installing packed bytes as dense
/// expert weights.
#[test]
fn prestacked_q4k_expert_trio_installs_through_moe_loader() {
    let mut config = tiny_sym8_moe_cfg();
    // Expert width = hidden = 256 so every K-quant companion carries its
    // real ggml geometry: Q4_K packs 256 values per super-block, so
    // `.weight` keeps k*4/32 columns, `.scales` 2*(k/32) and `.biases`
    // 2*(k/256).
    config.hidden_size = 256;
    config.intermediate_size = 256;
    config.num_heads = 4;
    config.head_dim = 64;
    let mut inner = Qwen35Inner::new_moe(config.clone()).expect("Qwen35Inner::new must succeed");

    let q4k_group = |params: &mut HashMap<String, MxArray>, prefix: &str| {
        params.insert(
            format!("{prefix}.weight"),
            MxArray::from_uint32(&vec![0u32; 4 * 256 * 32], &[4, 256, 32]).expect("weight"),
        );
        params.insert(
            format!("{prefix}.scales"),
            MxArray::from_float32(&vec![1.0f32; 4 * 256 * 16], &[4, 256, 16])
                .expect("scales")
                .astype(DType::Uint8)
                .expect("uint8 scales"),
        );
        params.insert(
            format!("{prefix}.biases"),
            MxArray::from_float16(
                &vec![half::f16::from_f32(0.5).to_bits(); 4 * 256 * 2],
                &[4, 256, 2],
            )
            .expect("biases"),
        );
    };

    let mut params: HashMap<String, MxArray> = HashMap::new();
    params.insert(
        "embedding.weight".to_string(),
        MxArray::from_float32(&vec![0.0f32; 8 * 256], &[8, 256]).expect("embedding"),
    );
    params.insert(
        "final_norm.weight".to_string(),
        MxArray::from_float32(&vec![1.0f32; 256], &[256]).expect("final_norm"),
    );
    params.insert(
        "layers.0.self_attn.q_proj.weight".to_string(),
        // The attended projection carries the query gate: 2 * heads * head_dim.
        MxArray::from_float32(&vec![0.0f32; 512 * 256], &[512, 256]).expect("q_proj"),
    );
    params.insert(
        "layers.0.mlp.gate.weight".to_string(),
        MxArray::from_float32(&vec![0.0f32; 4 * 256], &[4, 256]).expect("router gate"),
    );
    for proj in ["gate_proj", "up_proj", "down_proj"] {
        q4k_group(&mut params, &format!("layers.0.mlp.switch_mlp.{proj}"));
    }

    let mut per_layer_quant: HashMap<String, PerLayerQuant> = HashMap::new();
    for proj in ["gate_proj", "up_proj", "down_proj"] {
        per_layer_quant.insert(
            format!("layers.0.mlp.switch_mlp.{proj}"),
            PerLayerQuant {
                bits: 4,
                group_size: 32,
                mode: PerLayerMode::Q4K,
                input_amax: None,
                layout: Default::default(),
            },
        );
    }

    apply_weights_moe_inner(
        &mut inner,
        &params,
        &config,
        4,
        32,
        None,
        &per_layer_quant,
        /* has_vision */ false,
    )
    .expect("a pre-stacked Q4_K expert trio must load");

    match &inner.layers[0].mlp {
        MLPType::MoE(moe) => {
            let switch = moe.get_switch_mlp();
            assert!(
                switch.is_quantized(),
                "the Q4_K trio must install the quantized SwitchGLU backend"
            );
            for (proj, weight) in [
                ("gate_proj", switch.get_gate_proj_weight()),
                ("up_proj", switch.get_up_proj_weight()),
                ("down_proj", switch.get_down_proj_weight()),
            ] {
                assert_eq!(
                    weight.dtype().unwrap(),
                    DType::Uint32,
                    "{proj} must stay packed"
                );
                assert_eq!(weight.shape().unwrap().to_vec(), vec![4, 256, 32]);
            }
        }
        _ => panic!("layer 0 must be MoE (decoder_sparse_step = 1)"),
    }

    params.remove("layers.0.mlp.switch_mlp.up_proj.biases");
    let error = apply_weights_moe_inner(
        &mut inner,
        &params,
        &config,
        4,
        32,
        None,
        &per_layer_quant,
        /* has_vision */ false,
    )
    .expect_err("a K-quant group without its ggml `d` sidecar must fail loud");
    assert!(
        error.reason.contains("biases missing"),
        "the truncated group must name the missing companion: {}",
        error.reason
    );
}

#[test]
fn grouped_expert_tiling_preserves_registered_weight_bytes() {
    use crate::models::quant_dispatch::kquant_untile_rows;
    use crate::moe::SwitchProj;

    if !unsafe { mlx_sys::mlx_splash_moe_available() } {
        eprintln!("[skip] grouped MoE tiling requires the grouped Metal backend");
        return;
    }
    let mut config = tiny_sym8_moe_cfg();
    config.hidden_size = 256;
    config.intermediate_size = 256;
    config.num_heads = 4;
    config.head_dim = 64;
    let mut inner = Qwen35Inner::new_moe(config.clone()).expect("Qwen35Inner::new must succeed");

    let insert_q4k = |params: &mut HashMap<String, MxArray>, prefix: &str, shape: &[i64]| {
        let rows = shape[..shape.len() - 1].iter().product::<i64>() as usize;
        let weights = shape.iter().product::<i64>() as usize;
        let mut scales_shape = shape[..shape.len() - 1].to_vec();
        scales_shape.push(16);
        let mut biases_shape = shape[..shape.len() - 1].to_vec();
        biases_shape.push(2);
        params.insert(
            format!("{prefix}.weight"),
            MxArray::from_uint32(
                &(0..weights)
                    .map(|i| {
                        (i as u32)
                            .wrapping_mul(1_664_525)
                            .wrapping_add(1_013_904_223)
                    })
                    .collect::<Vec<_>>(),
                shape,
            )
            .expect("weight"),
        );
        params.insert(
            format!("{prefix}.scales"),
            MxArray::from_uint8(
                &(0..rows * 16)
                    .map(|i| ((i * 37 + 11) % 251 + 1) as u8)
                    .collect::<Vec<_>>(),
                &scales_shape,
            )
            .expect("scales"),
        );
        params.insert(
            format!("{prefix}.biases"),
            MxArray::from_float16(
                &(0..rows * 2)
                    .map(|i| [0x2800u16, 0x2c00, 0x3000, 0x3200][i % 4])
                    .collect::<Vec<_>>(),
                &biases_shape,
            )
            .expect("biases"),
        );
    };

    let mut params: HashMap<String, MxArray> = HashMap::new();
    params.insert(
        "embedding.weight".to_string(),
        MxArray::from_float32(&vec![0.0f32; 8 * 256], &[8, 256]).expect("embedding"),
    );
    params.insert(
        "final_norm.weight".to_string(),
        MxArray::from_float32(&vec![1.0f32; 256], &[256]).expect("final_norm"),
    );
    params.insert(
        "layers.0.self_attn.q_proj.weight".to_string(),
        MxArray::from_float32(&vec![0.0f32; 512 * 256], &[512, 256]).expect("q_proj"),
    );
    params.insert(
        "layers.0.mlp.gate.weight".to_string(),
        MxArray::from_float32(&vec![0.0f32; 4 * 256], &[4, 256]).expect("router gate"),
    );

    let expert_prefixes = [
        "layers.0.mlp.switch_mlp.gate_proj",
        "layers.0.mlp.switch_mlp.up_proj",
        "layers.0.mlp.switch_mlp.down_proj",
    ];
    let shared_prefixes = [
        "layers.0.mlp.shared_expert.gate_proj",
        "layers.0.mlp.shared_expert.up_proj",
        "layers.0.mlp.shared_expert.down_proj",
    ];
    let mut per_layer_quant = HashMap::new();
    for prefix in expert_prefixes.into_iter().chain(shared_prefixes) {
        let shape: &[i64] = if prefix.contains(".switch_mlp.") {
            &[4, 256, 32]
        } else {
            &[256, 32]
        };
        insert_q4k(&mut params, prefix, shape);
        per_layer_quant.insert(
            prefix.to_string(),
            PerLayerQuant {
                bits: 4,
                group_size: 32,
                mode: PerLayerMode::Q4K,
                input_amax: None,
                layout: Default::default(),
            },
        );
    }

    let bytes = |arrays: &HashMap<String, MxArray>| {
        arrays
            .values()
            .map(|array| array.nbytes() as u64)
            .sum::<u64>()
    };
    let expected_text_bytes = bytes(&params);
    let routed_expert_bytes = params
        .iter()
        .filter(|(key, _)| key.contains(".mlp.switch_mlp."))
        .map(|(_, array)| array.nbytes() as u64)
        .sum::<u64>();
    let quantized_source_bytes = params
        .iter()
        .filter(|(key, _)| key.contains(".mlp.switch_mlp.") || key.contains(".mlp.shared_expert."))
        .map(|(_, array)| array.nbytes() as u64)
        .sum::<u64>();
    let mut live = params.clone();
    let (_fp8_residency, registered_text_bytes) = super::apply_weights_moe_inner_with_residency(
        &mut inner,
        &mut live,
        &config,
        DEFAULT_QUANT_BITS,
        DEFAULT_QUANT_GROUP_SIZE,
        None,
        &per_layer_quant,
        false,
    )
    .expect("grouped Q4_K experts must load");

    assert_eq!(registered_text_bytes, expected_text_bytes);
    assert_eq!(
        expected_text_bytes - bytes(&live),
        quantized_source_bytes,
        "the post-install map must lose every released tiled source"
    );
    assert_eq!(
        registered_text_bytes,
        bytes(&live) + quantized_source_bytes,
        "registration must not use the pruned post-install map"
    );

    let MLPType::MoE(moe) = &inner.layers[0].mlp else {
        panic!("layer 0 must be MoE");
    };
    let switch = moe.get_switch_mlp();
    let mut installed_expert_bytes = 0;
    for (projection, suffix) in [
        (&switch.gate_proj, "gate_proj"),
        (&switch.up_proj, "up_proj"),
        (&switch.down_proj, "down_proj"),
    ] {
        let SwitchProj::Quantized(projection) = projection else {
            panic!("grouped experts must stay quantized");
        };
        assert_eq!(projection.mode(), "q4k@t64");
        let prefix = format!("layers.0.mlp.switch_mlp.{suffix}");
        for companion in ["weight", "scales", "biases"] {
            assert!(!live.contains_key(&format!("{prefix}.{companion}")));
        }
        installed_expert_bytes += [
            projection.get_weight(),
            projection.get_scales(),
            projection.get_biases().expect("Q4_K super-scales"),
        ]
        .iter()
        .map(|array| array.nbytes() as u64)
        .sum::<u64>();

        let untile = |array: &MxArray, unit: i64| {
            let shape = array.shape().expect("shape");
            kquant_untile_rows(&array.reshape(&[-1, shape[2]]).expect("flatten"), unit)
                .expect("untile")
                .reshape(&shape)
                .expect("restore expert axis")
        };
        let weight = untile(projection.get_weight(), 4);
        let scales = untile(projection.get_scales(), 16);
        let biases = untile(projection.get_biases().expect("Q4_K super-scales"), 2);
        assert_eq!(
            weight.to_uint32().expect("tiled weight").to_vec(),
            params[&format!("{prefix}.weight")]
                .to_uint32()
                .expect("source weight")
                .to_vec()
        );
        assert_eq!(
            scales.to_uint8().expect("tiled scales"),
            params[&format!("{prefix}.scales")]
                .to_uint8()
                .expect("source scales")
        );
        assert_eq!(
            biases.to_uint16_native().expect("tiled biases"),
            params[&format!("{prefix}.biases")]
                .to_uint16_native()
                .expect("source biases")
        );
    }
    assert_eq!(installed_expert_bytes, routed_expert_bytes);
}

/// T13: the MoE expert stacker keys per-expert companions on the three
/// canonical suffixes (.weight / .scales / .biases), so a K-quant expert
/// set stacks into THREE separate `switch_mlp.{proj}` companions with dtypes
/// preserved — the reused suffixes mean no stacker change is needed for
/// K-quants. This pins that invariant: if the stacker ever folded an unknown
/// suffix into the `.weight` group, `.scales`/`.biases` would go missing.
#[test]
fn kquant_expert_set_stacks_three_companions_separately() {
    let config = tiny_sym8_moe_cfg(); // num_experts = 4
    let mut params: HashMap<String, MxArray> = HashMap::new();
    // Q6_K-shaped companions: uint32 packed .weight, int8 .scales, float16
    // .biases (ggml `d`). Shapes are nominal — the stacker only concatenates
    // the per-expert arrays over a new leading axis.
    for e in 0..4 {
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let base = format!("layers.0.mlp.experts.{e}.{proj}");
            params.insert(
                format!("{base}.weight"),
                MxArray::from_uint32(&[0u32; 8 * 3], &[8, 3]).expect("weight"),
            );
            params.insert(
                format!("{base}.scales"),
                MxArray::from_float32(&[1.0f32; 8 * 2], &[8, 2])
                    .expect("scales")
                    .astype(DType::Int8)
                    .expect("int8 scales"),
            );
            params.insert(
                format!("{base}.biases"),
                MxArray::from_float16(&[half::f16::from_f32(0.5).to_bits(); 8], &[8, 1])
                    .expect("biases"),
            );
        }
    }

    let out = sanitize_weights_moe(params, &config, &HashMap::new())
        .expect("K-quant expert stacking must succeed");

    for proj in ["gate_proj", "up_proj", "down_proj"] {
        let w = out
            .get(&format!("layers.0.mlp.switch_mlp.{proj}.weight"))
            .unwrap_or_else(|| panic!("{proj}.weight stacked companion missing"));
        let s = out
            .get(&format!("layers.0.mlp.switch_mlp.{proj}.scales"))
            .unwrap_or_else(|| panic!("{proj}.scales missing — folded into weight group?"));
        let b = out
            .get(&format!("layers.0.mlp.switch_mlp.{proj}.biases"))
            .unwrap_or_else(|| panic!("{proj}.biases missing — folded into weight group?"));
        assert_eq!(w.shape().unwrap().to_vec(), vec![4, 8, 3]);
        assert_eq!(w.dtype().unwrap(), DType::Uint32);
        assert_eq!(s.shape().unwrap().to_vec(), vec![4, 8, 2]);
        assert_eq!(s.dtype().unwrap(), DType::Int8);
        assert_eq!(b.shape().unwrap().to_vec(), vec![4, 8, 1]);
        assert_eq!(b.dtype().unwrap(), DType::Float16);
    }
}

#[test]
fn sym8_default_installs_int8_backend_on_attention_q_proj() {
    if unsafe { mlx_sys::mlx_gpu_architecture_gen() } < 17 {
        eprintln!("[skip] sym8 MoE loader test: int8 kernels need an M5+ GPU (gen >= 17)");
        return;
    }
    let config = tiny_sym8_moe_cfg();
    let mut inner = Qwen35Inner::new_moe(config.clone()).expect("Qwen35Inner::new must succeed");

    let mut params: HashMap<String, MxArray> = HashMap::new();
    params.insert(
        "embedding.weight".to_string(),
        MxArray::from_float32(&vec![0.0f32; 8 * 64], &[8, 64]).expect("embedding"),
    );
    params.insert(
        "final_norm.weight".to_string(),
        MxArray::from_float32(&vec![1.0f32; 64], &[64]).expect("final_norm"),
    );
    // hidden_size=64, num_heads=4, head_dim=16 -> q_proj is [64, 64]; K=64
    // satisfies the sym8 kernel's K % 16 == 0 contract.
    synth_sym8_group(&mut params, "layers.0.self_attn.q_proj", 64, 64);
    params.insert(
        "layers.0.mlp.gate.weight".to_string(),
        MxArray::from_float32(&vec![0.0f32; 4 * 64], &[4, 64]).expect("gate"),
    );

    // Mirrors what convert.rs actually emits for a real sym8 checkpoint:
    // 3-D switch_mlp.* experts always get a forced affine-8 override.
    let mut per_layer_quant = HashMap::new();
    for suffix in ["gate_proj", "up_proj", "down_proj"] {
        per_layer_quant.insert(
            format!("layers.0.mlp.switch_mlp.{suffix}"),
            default_per_layer_quant(8, 64, PerLayerMode::Affine),
        );
    }

    apply_weights_moe_inner(
        &mut inner,
        &params,
        &config,
        4,
        32,
        Some(PerLayerMode::Sym8),
        &per_layer_quant,
        /* has_vision */ false,
    )
    .expect("sym8-default checkpoint must load: attention is a non-expert sublayer");

    match &inner.layers[0].attn {
        AttentionType::Full(attn) => {
            assert_eq!(
                attn.get_q_proj_weight().dtype().unwrap(),
                DType::Int8,
                "q_proj must install the raw int8 sym8 backend, not affine-packed Uint32"
            );
        }
        AttentionType::Linear(_) => {
            panic!("tiny_sym8_moe_cfg (full_attention_interval=1) must assign Full attention")
        }
    }
}

/// RED-first regression guard for the MoE activation-fp8 plumbing gap: the
/// MoE loader's `try_build_ql` must thread the resolved
/// `PerLayerQuant::input_amax` onto the built attention `QuantizedLinear`,
/// exactly like the dense qwen3_5 loader (`persistence.rs` ~:970). Before
/// the fix `try_build_ql` returned the raw builder result (amax dropped),
/// so a MoE (agentworld-style) nvidia checkpoint with a calibrated
/// per-tensor `input_amax` on `self_attn.q_proj` silently loaded it as
/// `None`. Drives the real `apply_weights_moe_inner` loader seam (which
/// installs q_proj via `try_build_ql` + `set_quantized_q_proj`) and reads
/// the amax back off the loaded attention block. This assertion is RED on
/// the unfixed closure (observes `None`) and GREEN once the amax is
/// threaded (observes `Some(AMAX)`).
#[test]
fn mxfp8_attention_q_proj_threads_input_amax_through_moe_loader() {
    const AMAX: f32 = 37.5;
    let config = tiny_sym8_moe_cfg();
    let mut inner = Qwen35Inner::new_moe(config.clone()).expect("Qwen35Inner::new must succeed");

    // Minimal in-memory mxfp8 q_proj group: packed fp8 weight (Uint8) +
    // E8M0 Uint8 scales. `try_build_mxfp8_quantized_linear` only clones
    // weight/scales into a QuantizedLinear (no load-time eval/kernel), so
    // arbitrary bytes and the [64,64] / [64,2] shapes are load-inert.
    let mut params: HashMap<String, MxArray> = HashMap::new();
    params.insert(
        "embedding.weight".to_string(),
        MxArray::from_float32(&vec![0.0f32; 8 * 64], &[8, 64]).expect("embedding"),
    );
    params.insert(
        "final_norm.weight".to_string(),
        MxArray::from_float32(&vec![1.0f32; 64], &[64]).expect("final_norm"),
    );
    params.insert(
        "layers.0.self_attn.q_proj.weight".to_string(),
        MxArray::from_float32(&vec![0.0f32; 64 * 64], &[64, 64])
            .expect("q_proj weight")
            .astype(DType::Uint8)
            .expect("astype uint8 weight"),
    );
    params.insert(
        "layers.0.self_attn.q_proj.scales".to_string(),
        MxArray::from_float32(&vec![1.0f32; 64 * 2], &[64, 2])
            .expect("q_proj scales")
            .astype(DType::Uint8)
            .expect("astype uint8 scales"),
    );
    // Satisfies the loader's per-layer has_mlp completeness gate.
    params.insert(
        "layers.0.mlp.gate.weight".to_string(),
        MxArray::from_float32(&vec![0.0f32; 4 * 64], &[4, 64]).expect("gate"),
    );

    // The calibrated activation-fp8 override that must survive the load.
    let mut per_layer_quant = HashMap::new();
    per_layer_quant.insert(
        "layers.0.self_attn.q_proj".to_string(),
        PerLayerQuant {
            bits: 8,
            group_size: 32,
            mode: PerLayerMode::Mxfp8,
            input_amax: Some(AMAX),
            layout: Default::default(),
        },
    );

    apply_weights_moe_inner(
        &mut inner,
        &params,
        &config,
        4,
        32,
        None,
        &per_layer_quant,
        /* has_vision */ false,
    )
    .expect("mxfp8 q_proj MoE checkpoint must load");

    match &inner.layers[0].attn {
        AttentionType::Full(attn) => {
            assert_eq!(
                attn.q_proj_input_amax(),
                Some(AMAX),
                "MoE loader must thread PerLayerQuant::input_amax onto the built q_proj \
                 QuantizedLinear (was dropped before the try_build_ql fix)"
            );
        }
        AttentionType::Linear(_) => {
            panic!("tiny_sym8_moe_cfg (full_attention_interval=1) must assign Full attention")
        }
    }
}

/// RED-first NEGATIVE sibling to
/// `mxfp8_attention_q_proj_threads_input_amax_through_moe_loader`: the MoE
/// `try_build_ql` closure gated the recorded `amax_key` behind
/// `is_activation_fp8_site` but threaded the CONSUMED `input_amax`
/// UNCONDITIONALLY. A stale / hand-edited / future-recipe config that puts
/// `input_amax` on a NON-activation-site mxfp8 projection (here `lm_head`)
/// would carry it into `QuantizedLinear::forward` FP8 fake-quant — violating
/// "activation FP8 only on attn/GDN sites". After the symmetric-gate fix a
/// non-site drops `input_amax` to `None`.
///
/// `lm_head` normalizes to `"lm_head"` (NOT a site) yet installs through the
/// exact closure under test (`inner.lm_head = LinearProj::Quantized(ql)`):
/// pre-fix this observes `Some(2.0)` (RED), post-fix `None` (GREEN).
#[test]
fn mxfp8_non_site_lm_head_drops_input_amax_through_moe_loader() {
    use crate::models::quantized_linear::{LinearProj, MXFP8_BITS, MXFP8_GROUP_SIZE};
    let config = tiny_sym8_moe_cfg();
    let mut inner = Qwen35Inner::new_moe(config.clone()).expect("Qwen35Inner::new must succeed");

    let u8 = |v: f32, shape: &[i64]| {
        let n: i64 = shape.iter().product();
        MxArray::from_float32(&vec![v; n as usize], shape)
            .expect("from_float32")
            .astype(DType::Uint8)
            .expect("astype uint8")
    };

    let mut params: HashMap<String, MxArray> = HashMap::new();
    params.insert(
        "embedding.weight".to_string(),
        MxArray::from_float32(&vec![0.0f32; 8 * 64], &[8, 64]).expect("embedding"),
    );
    params.insert(
        "final_norm.weight".to_string(),
        MxArray::from_float32(&vec![1.0f32; 64], &[64]).expect("final_norm"),
    );
    // Attention q_proj + router gate keep the loader's completeness gate
    // satisfied (mirrors the positive sibling); q_proj carries NO input_amax.
    params.insert(
        "layers.0.self_attn.q_proj.weight".to_string(),
        u8(0.0, &[64, 64]),
    );
    params.insert(
        "layers.0.self_attn.q_proj.scales".to_string(),
        u8(1.0, &[64, 2]),
    );
    params.insert(
        "layers.0.mlp.gate.weight".to_string(),
        MxArray::from_float32(&vec![0.0f32; 4 * 64], &[4, 64]).expect("gate"),
    );
    // The non-activation-site projection under test: mxfp8 lm_head carrying
    // a stale per-tensor activation amax that must NOT survive the load.
    params.insert("lm_head.weight".to_string(), u8(0.0, &[8, 64]));
    params.insert("lm_head.scales".to_string(), u8(1.0, &[8, 2]));

    let mut per_layer_quant = HashMap::new();
    per_layer_quant.insert(
        "layers.0.self_attn.q_proj".to_string(),
        PerLayerQuant {
            bits: MXFP8_BITS,
            group_size: MXFP8_GROUP_SIZE,
            mode: PerLayerMode::Mxfp8,
            input_amax: None,
            layout: Default::default(),
        },
    );
    per_layer_quant.insert(
        "lm_head".to_string(),
        PerLayerQuant {
            bits: MXFP8_BITS,
            group_size: MXFP8_GROUP_SIZE,
            mode: PerLayerMode::Mxfp8,
            input_amax: Some(2.0),
            layout: Default::default(),
        },
    );

    apply_weights_moe_inner(
        &mut inner,
        &params,
        &config,
        4,
        32,
        None,
        &per_layer_quant,
        /* has_vision */ false,
    )
    .expect("mxfp8 lm_head MoE checkpoint must load");

    assert!(
        matches!(inner.lm_head, Some(LinearProj::Quantized(_))),
        "mxfp8 lm_head must install as LinearProj::Quantized"
    );
    if let Some(LinearProj::Quantized(ref ql)) = inner.lm_head {
        assert_eq!(
            ql.input_amax(),
            None,
            "MoE loader must DROP input_amax on a non-activation-fp8-site mxfp8 projection \
             (lm_head); it was threaded unconditionally before the try_build_ql gate fix"
        );
    }
}

#[test]
fn sym8_switch_mlp_expert_projection_fails_loud_never_silently_affine() {
    let config = tiny_sym8_moe_cfg();
    let mut inner = Qwen35Inner::new_moe(config.clone()).expect("Qwen35Inner::new must succeed");

    let mut params: HashMap<String, MxArray> = HashMap::new();
    params.insert(
        "layers.0.mlp.switch_mlp.gate_proj.scales".to_string(),
        MxArray::from_float32(&[0.0f32], &[1]).expect("scales placeholder"),
    );

    let mut per_layer_quant = HashMap::new();
    per_layer_quant.insert(
        "layers.0.mlp.switch_mlp.gate_proj".to_string(),
        default_per_layer_quant(8, -1, PerLayerMode::Sym8),
    );

    let err = apply_weights_moe_inner(
        &mut inner,
        &params,
        &config,
        4,
        32,
        None,
        &per_layer_quant,
        /* has_vision */ false,
    )
    .expect_err("3-D stacked experts have no sym8 kernel; must fail loud");
    let msg = format!("{err}");
    assert!(
        msg.contains("switch_mlp.gate_proj") && msg.contains("sym8"),
        "error must name the expert projection and mention sym8, got: {msg}"
    );
}

/// An EXPLICIT per-layer sym8 override on a switch_mlp expert projection
/// is lying metadata: convert never emits one (`sym8_eligible` forces
/// every quantized switch_mlp.* tensor to an affine-8 override, and the
/// group-coherence pass emits forced-dense trios with NO override — a
/// sym8 entry can only come from hand-edited/stale config.json). Even
/// when the trio carries float weights with no `.scales` sidecar, the
/// load must fail loud on the explicit override rather than silently
/// installing the weights dense — only the top-level sym8 DEFAULT
/// falling through (no explicit entry) may take the forced-dense
/// `Ok(None)` hand-off.
#[test]
fn sym8_explicit_switch_mlp_override_without_scales_fails_loud() {
    let config = tiny_sym8_moe_cfg();
    let mut inner = Qwen35Inner::new_moe(config.clone()).expect("Qwen35Inner::new must succeed");

    let mut params: HashMap<String, MxArray> = HashMap::new();
    params.insert(
        "embedding.weight".to_string(),
        MxArray::from_float32(&vec![0.0f32; 8 * 64], &[8, 64]).expect("embedding"),
    );
    params.insert(
        "final_norm.weight".to_string(),
        MxArray::from_float32(&vec![1.0f32; 64], &[64]).expect("final_norm"),
    );
    // One OTHER genuinely-quantized tensor (affine pack; never forwarded,
    // so the pack contents are irrelevant): makes `is_quantized_checkpoint`
    // true so the loader takes the quantized MoE arm — the path that calls
    // `try_build_qsl`.
    params.insert(
        "layers.0.self_attn.q_proj.weight".to_string(),
        MxArray::from_float32(&vec![0.0f32; 64 * 8], &[64, 8])
            .expect("q_proj pack")
            .astype(DType::Uint32)
            .expect("astype uint32"),
    );
    params.insert(
        "layers.0.self_attn.q_proj.scales".to_string(),
        MxArray::from_float32(&vec![1.0f32; 64 * 2], &[64, 2]).expect("q_proj scales"),
    );
    params.insert(
        "layers.0.self_attn.q_proj.biases".to_string(),
        MxArray::from_float32(&vec![0.0f32; 64 * 2], &[64, 2]).expect("q_proj biases"),
    );
    // The lying trio: float 3-D expert weights, NO `.scales`, but an
    // explicit sym8 per-layer override for gate_proj.
    for suffix in ["gate_proj", "up_proj", "down_proj"] {
        params.insert(
            format!("layers.0.mlp.switch_mlp.{suffix}.weight"),
            MxArray::from_float32(&vec![0.5f32; 4 * 8 * 64], &[4, 8, 64]).expect("trio"),
        );
    }

    let mut per_layer_quant = HashMap::new();
    per_layer_quant.insert(
        "layers.0.mlp.switch_mlp.gate_proj".to_string(),
        default_per_layer_quant(8, -1, PerLayerMode::Sym8),
    );

    let err = apply_weights_moe_inner(
        &mut inner,
        &params,
        &config,
        4,
        32,
        None,
        &per_layer_quant,
        /* has_vision */ false,
    )
    .expect_err(
        "an explicit sym8 override on a switch_mlp projection must fail loud even without .scales",
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("switch_mlp.gate_proj") && msg.contains("sym8"),
        "error must name the expert projection and mention sym8, got: {msg}"
    );
}

/// A checkpoint whose top-level mode is sym8 can legitimately carry a
/// DENSE switch_mlp trio: convert's group-coherence pass forces the whole
/// trio dense when any member is unquantizable (e.g. K not divisible by
/// the affine group size), emitting f32 weights with NO `.scales` sidecar
/// and NO per-layer override. The prefix then resolves to the sym8
/// DEFAULT PLQ, and `try_build_qsl` must treat the absent `.scales` as
/// "this layer is not quantized" (`Ok(None)`, the contract shared with
/// `try_build_sym8_quantized_linear` and the 2-D `try_build_ql`) so the
/// dtype-guarded dense fallback installs the weights — NOT fail the whole
/// load with the re-convert error.
#[test]
fn sym8_default_loads_forced_dense_switch_mlp_trio() {
    if unsafe { mlx_sys::mlx_gpu_architecture_gen() } < 17 {
        eprintln!(
            "[skip] sym8 MoE forced-dense trio test: int8 kernels need an M5+ GPU (gen >= 17)"
        );
        return;
    }
    let config = tiny_sym8_moe_cfg();
    let mut inner = Qwen35Inner::new_moe(config.clone()).expect("Qwen35Inner::new must succeed");

    let mut params: HashMap<String, MxArray> = HashMap::new();
    params.insert(
        "embedding.weight".to_string(),
        MxArray::from_float32(&vec![0.0f32; 8 * 64], &[8, 64]).expect("embedding"),
    );
    params.insert(
        "final_norm.weight".to_string(),
        MxArray::from_float32(&vec![1.0f32; 64], &[64]).expect("final_norm"),
    );
    // One OTHER tensor genuinely sym8-quantized: makes
    // `is_quantized_checkpoint` true so the loader takes the quantized
    // MoE arm (the path that calls `try_build_qsl`).
    synth_sym8_group(&mut params, "layers.0.self_attn.q_proj", 64, 64);
    params.insert(
        "layers.0.mlp.gate.weight".to_string(),
        MxArray::from_float32(&vec![0.0f32; 4 * 64], &[4, 64]).expect("gate"),
    );
    // The forced-dense trio: 3-D f32 [experts, N, K] with K = 60 (fails
    // the affine group gate, so convert's coherence pass emitted the
    // whole trio dense), no .scales, no per-layer override.
    for suffix in ["gate_proj", "up_proj", "down_proj"] {
        params.insert(
            format!("layers.0.mlp.switch_mlp.{suffix}.weight"),
            MxArray::from_float32(&vec![0.5f32; 4 * 8 * 60], &[4, 8, 60]).expect("trio"),
        );
    }

    apply_weights_moe_inner(
        &mut inner,
        &params,
        &config,
        4,
        32,
        Some(PerLayerMode::Sym8),
        &HashMap::new(),
        /* has_vision */ false,
    )
    .expect("forced-dense switch_mlp trio under a sym8 default must load dense");

    match &inner.layers[0].mlp {
        MLPType::MoE(moe) => {
            let switch = moe.get_switch_mlp();
            assert!(
                !switch.is_quantized(),
                "trio must land on the dense SwitchGLU path"
            );
            let w = switch.get_gate_proj_weight();
            assert_eq!(
                w.dtype().unwrap(),
                DType::Float32,
                "dense fallback must keep the f32 checkpoint dtype"
            );
            assert_eq!(
                &w.shape().unwrap()[..],
                &[4i64, 8, 60],
                "dense fallback must install the checkpoint tensor, not the ctor placeholder"
            );
        }
        _ => panic!("layer 0 must be MoE (decoder_sparse_step = 1)"),
    }
}

/// A sym8 checkpoint on an MTP-capable config must LOAD (the non-expert
/// sublayers dispatch sym8) but leave the speculative MTP head DISABLED:
/// mtp.rs's own builder closures have no sym8 wiring (`Sym8 => None`), so
/// without the call-site gate the head would silently install nothing.
/// Mirrors dense qwen3_5's MTP-disable-under-sym8 branch.
#[test]
fn sym8_checkpoint_disables_mtp_head_load() {
    if unsafe { mlx_sys::mlx_gpu_architecture_gen() } < 17 {
        eprintln!("[skip] sym8 MoE MTP-disable test: int8 kernels need an M5+ GPU (gen >= 17)");
        return;
    }
    let config = Qwen3_5MoeConfig {
        n_mtp_layers: 1,
        ..tiny_sym8_moe_cfg()
    };
    let mut inner = Qwen35Inner::new_moe(config.clone()).expect("Qwen35Inner::new must succeed");
    assert!(
        inner.mtp.is_some(),
        "config with n_mtp_layers=1 must construct the MTP module"
    );

    let mut params: HashMap<String, MxArray> = HashMap::new();
    params.insert(
        "embedding.weight".to_string(),
        MxArray::from_float32(&vec![0.0f32; 8 * 64], &[8, 64]).expect("embedding"),
    );
    params.insert(
        "final_norm.weight".to_string(),
        MxArray::from_float32(&vec![1.0f32; 64], &[64]).expect("final_norm"),
    );
    synth_sym8_group(&mut params, "layers.0.self_attn.q_proj", 64, 64);
    params.insert(
        "layers.0.mlp.gate.weight".to_string(),
        MxArray::from_float32(&vec![0.0f32; 4 * 64], &[4, 64]).expect("gate"),
    );

    let mut per_layer_quant = HashMap::new();
    for suffix in ["gate_proj", "up_proj", "down_proj"] {
        per_layer_quant.insert(
            format!("layers.0.mlp.switch_mlp.{suffix}"),
            default_per_layer_quant(8, 64, PerLayerMode::Affine),
        );
    }

    apply_weights_moe_inner(
        &mut inner,
        &params,
        &config,
        4,
        32,
        Some(PerLayerMode::Sym8),
        &per_layer_quant,
        /* has_vision */ false,
    )
    .expect("sym8 MTP-capable checkpoint must still load (plain AR decode)");

    assert!(
        !inner.mtp_weights_loaded,
        "sym8 checkpoint must disable the speculative MTP head load"
    );
    assert!(
        !inner.has_mtp_weights(),
        "has_mtp_weights() must report inactive under sym8"
    );
}

/// A sym8 checkpoint that ships a vision tower must LOAD (text-only)
/// with the vision encoder NOT installed: the eager VLM prefill path is
/// not wired for sym8 int8 operands, so `load_vision_encoder_moe` strips
/// the vision params under sym8 (loud warn, image turns then fail loud
/// with "vision encoder/processor not loaded"). Mirrors dense qwen3_5's
/// vision-gate-under-sym8 branch.
#[test]
fn sym8_checkpoint_skips_vision_encoder_load() {
    if unsafe { mlx_sys::mlx_gpu_architecture_gen() } < 17 {
        eprintln!("[skip] sym8 MoE vision-gate test: int8 kernels need an M5+ GPU (gen >= 17)");
        return;
    }
    let config = tiny_sym8_moe_cfg();
    let mut inner = Qwen35Inner::new_moe(config.clone()).expect("Qwen35Inner::new must succeed");

    let mut params: HashMap<String, MxArray> = HashMap::new();
    params.insert(
        "embedding.weight".to_string(),
        MxArray::from_float32(&vec![0.0f32; 8 * 64], &[8, 64]).expect("embedding"),
    );
    params.insert(
        "final_norm.weight".to_string(),
        MxArray::from_float32(&vec![1.0f32; 64], &[64]).expect("final_norm"),
    );
    synth_sym8_group(&mut params, "layers.0.self_attn.q_proj", 64, 64);
    params.insert(
        "layers.0.mlp.gate.weight".to_string(),
        MxArray::from_float32(&vec![0.0f32; 4 * 64], &[4, 64]).expect("gate"),
    );

    let mut per_layer_quant = HashMap::new();
    for suffix in ["gate_proj", "up_proj", "down_proj"] {
        per_layer_quant.insert(
            format!("layers.0.mlp.switch_mlp.{suffix}"),
            default_per_layer_quant(8, 64, PerLayerMode::Affine),
        );
    }

    // Text load succeeds under sym8 (has_vision mirrors the real loader,
    // which passes true when the checkpoint ships vision tensors).
    apply_weights_moe_inner(
        &mut inner,
        &params,
        &config,
        4,
        32,
        Some(PerLayerMode::Sym8),
        &per_layer_quant,
        /* has_vision */ true,
    )
    .expect("sym8 checkpoint with a vision tower must still load (text-only)");

    // The checkpoint ships vision tensors; under sym8 the vision step
    // must strip them WITHOUT erroring and WITHOUT installing the
    // encoder (a real tower would need real weights — the gate must
    // reject BEFORE any vision weight is read).
    let mut vision_params: HashMap<String, MxArray> = HashMap::new();
    vision_params.insert(
        "patch_embed.proj.weight".to_string(),
        MxArray::from_float32(&[0.0f32], &[1]).expect("vision placeholder"),
    );
    let raw = serde_json::json!({});
    let retained = load_vision_encoder_moe(
        &mut inner,
        Some(vision_params),
        &raw,
        &config,
        Some(PerLayerMode::Sym8),
        &per_layer_quant,
    )
    .expect("sym8 vision gate must fail soft (strip + warn), not error");

    assert!(
        retained.is_none(),
        "sym8 must strip the vision params (weight-byte accounting must not count them)"
    );
    assert!(
        inner.vision_encoder.is_none(),
        "sym8 checkpoint must NOT install the vision encoder"
    );
    assert!(
        inner.image_processor.is_none(),
        "sym8 checkpoint must NOT install the image processor"
    );
}

/// A packed (non-float) member of a PARTIAL quant group must fail loud
/// on the dense fallback, never silently install. When a `.scales`
/// sidecar is missing, `try_build_ql`/`try_build_qsl` return `Ok(None)`
/// and the whole trio drops to the dense setters — the switch_mlp
/// setters are infallible, so without the dtype guard a packed Uint32
/// payload would enter dense expert math (garbage logits, no error).
/// Mirrors dense qwen3_5's `ensure_dense_weight_floating` fallback
/// contract.
#[test]
fn packed_weight_without_sidecars_fails_loud_on_dense_fallback() {
    let config = tiny_sym8_moe_cfg();
    let scales_placeholder = || {
        // Lone `.scales` on an unrelated projection flips
        // `is_quantized_checkpoint` true without forming any buildable
        // quant group (same trick as the sym8 fail-loud test above).
        MxArray::from_float32(&[0.0f32], &[1]).expect("scales placeholder")
    };

    // (a) switch_mlp trio member: packed Uint32 `.weight`, no sidecars.
    let mut inner = Qwen35Inner::new_moe(config.clone()).expect("Qwen35Inner::new must succeed");
    let mut params: HashMap<String, MxArray> = HashMap::new();
    params.insert(
        "layers.0.self_attn.q_proj.scales".to_string(),
        scales_placeholder(),
    );
    params.insert(
        "layers.0.mlp.switch_mlp.gate_proj.weight".to_string(),
        MxArray::from_uint32(&vec![0u32; 2 * 16 * 8], &[2, 16, 8]).expect("packed uint32"),
    );
    let per_layer_quant: HashMap<String, PerLayerQuant> = HashMap::new();
    let err = apply_weights_moe_inner(
        &mut inner,
        &params,
        &config,
        4,
        32,
        None,
        &per_layer_quant,
        /* has_vision */ false,
    )
    .expect_err("packed switch_mlp member without sidecars must fail loud");
    let msg = format!("{err}");
    assert!(
        msg.contains("layers.0.mlp.switch_mlp.gate_proj.weight") && msg.contains("non-float"),
        "error must name the key and the dtype problem, got: {msg}"
    );

    // (b) shared_expert trio member: same stripped-sidecar shape.
    let mut inner = Qwen35Inner::new_moe(config.clone()).expect("Qwen35Inner::new must succeed");
    let mut params: HashMap<String, MxArray> = HashMap::new();
    params.insert(
        "layers.0.self_attn.q_proj.scales".to_string(),
        scales_placeholder(),
    );
    params.insert(
        "layers.0.mlp.shared_expert.up_proj.weight".to_string(),
        MxArray::from_uint32(&vec![0u32; 8 * 8], &[8, 8]).expect("packed uint32"),
    );
    let err = apply_weights_moe_inner(
        &mut inner,
        &params,
        &config,
        4,
        32,
        None,
        &per_layer_quant,
        /* has_vision */ false,
    )
    .expect_err("packed shared_expert member without sidecars must fail loud");
    let msg = format!("{err}");
    assert!(
        msg.contains("layers.0.mlp.shared_expert.up_proj.weight") && msg.contains("non-float"),
        "error must name the key and the dtype problem, got: {msg}"
    );
}
