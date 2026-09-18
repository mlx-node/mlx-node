//! K2-Horizon weight loading: config parse → safetensors → FP8 dequant →
//! sanitize → apply → materialize → paged pool → cold tier → tokenizer.
//!
//! Two ingest shapes:
//!   * **Converted mxfp8** (the intended path — `mlx convert` emits packed
//!     `.weight` + u8 E8M0 `.scales` + a `quantization` config block, and
//!     the generic quant dispatch installs `QuantizedLinear` backends).
//!   * **Raw compressed-tensors FP8** (the shipped checkpoint): `.weight`
//!     in F8E4M3 storage + `.weight_scale` `[N/128, K/128]` BF16 block
//!     scales. [`dequant_fp8_block_scale`] dequantizes to bf16 at load —
//!     byte-identical math to `dequant_fp8`'s `weight_scale_inv` branch,
//!     differing only in the sidecar suffix. The block-FP8 → mxfp8 scale
//!     repack is intentionally NOT done on-device (128×128 source groups
//!     cannot express 32-column E8M0 groups losslessly); convert offline.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use tracing::info;

use crate::array::{DType, MxArray};
use crate::cold_tier::{CheckpointLoadGuard, resolve_persist_cold};
use crate::engine::persistence::{
    KeyRule, RenameSpec, apply_rename_spec, cast_f32_tensors_to_bf16, dequant_fp8_block_scale,
    dequant_fp8_weights, load_all_safetensors, prewarm_checkpoint_pages,
};
use crate::models::quant_dispatch::{
    PerLayerMode, PerLayerQuant, default_per_layer_quant, load_dense_mlp_variant,
    load_embedding_affine_or_bf16, load_linear_proj_quantized_or_bf16,
    load_quant_settings_from_disk, resolve_default_mode,
};
use crate::models::quantized_linear::{
    DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE, is_mxfp8_checkpoint,
};
use crate::tokenizer::Qwen3Tokenizer;

use super::config::K2HorizonConfig;
use super::model::{K2HorizonModel, K2Inner};

/// Parse config.json into `K2HorizonConfig` (+ structural validation).
fn parse_config(model_path: &Path) -> Result<K2HorizonConfig> {
    let config_path = model_path.join("config.json");
    let raw_str = fs::read_to_string(&config_path)
        .map_err(|e| Error::from_reason(format!("Failed to read config.json: {e}")))?;
    let config: K2HorizonConfig = serde_json::from_str(&raw_str)
        .map_err(|e| Error::from_reason(format!("Failed to deserialize K2HorizonConfig: {e}")))?;
    config.validate()?;
    Ok(config)
}

/// Declarative half of `sanitize_weights` — strip the `model.` prefix and
/// drop the rotary-embedding buffers the fused path recomputes.
static K2_RENAME_SPEC: RenameSpec<'static> = RenameSpec {
    raw_rules: &[],
    strip_prefixes: &["model."],
    rules_require_strip: false,
    rules: &[KeyRule::DropContains("rotary_emb")],
    reject_duplicate_keys_as: None,
};

/// Sanitize HF weight keys to internal format: strip the `model.` prefix,
/// drop rotary buffers (computed at runtime), keep the untied `lm_head`.
///
/// The f32→bf16 cast keeps sym8 `.scales` (mandatory f32 `[N]`) at full
/// precision — identified by an Int8 sibling `.weight` — and skips
/// compressed-tensors `.weight_scale` leftovers (the dequant pass already
/// consumed them; a survivor is malformed but harmless, keep it f32 so the
/// mandatory-weights validator reports it untouched).
fn sanitize_weights(params: &mut HashMap<String, MxArray>) -> Result<HashMap<String, MxArray>> {
    let mut sanitized = apply_rename_spec(std::mem::take(params), &K2_RENAME_SPEC)?;

    // Value hook: cast f32 tensors to bf16 to avoid dtype promotion issues —
    // EXCLUDING sym8 `.scales` (f32 [N] is the sym8 storage contract).
    cast_f32_tensors_to_bf16(&mut sanitized, |_| false);

    Ok(sanitized)
}

/// K2 family tag for the shared non-MoE loader helpers in
/// [`crate::models::quant_dispatch`] — keeps error strings naming this
/// family after the verbatim helper copies were deduplicated.
const FAMILY: &str = "k2_horizon";

/// Validate all mandatory K2 tensors are present in the sanitized param
/// map — load-time failure beats silent random-init garbage.
fn validate_mandatory_weights(
    params: &HashMap<String, MxArray>,
    config: &K2HorizonConfig,
    num_layers: usize,
) -> Result<()> {
    let mut missing: Vec<String> = Vec::new();

    if !params.contains_key("embed_tokens.weight") {
        missing.push("embed_tokens.weight".to_string());
    }
    if !params.contains_key("norm.weight") {
        missing.push("norm.weight".to_string());
    }
    if !config.tie_word_embeddings && !params.contains_key("lm_head.weight") {
        missing.push("lm_head.weight".to_string());
    }

    for i in 0..num_layers {
        let prefix = format!("layers.{i}");
        for suffix in [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            let key = format!("{prefix}.{suffix}");
            if !params.contains_key(&key) {
                missing.push(key);
            }
        }

        // Quantized-group integrity: a `.scales` companion without its
        // packed `.weight` (or vice versa) must fail here, not surface as
        // a random-init projection in `apply_weights`.
        for base in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ] {
            let scales_key = format!("{prefix}.{base}.scales");
            let weight_key = format!("{prefix}.{base}.weight");
            if params.contains_key(&scales_key) && !params.contains_key(&weight_key) {
                missing.push(format!("{weight_key} (paired with {scales_key})"));
            }
        }
    }

    if !missing.is_empty() {
        return Err(Error::from_reason(format!(
            "k2_horizon: missing {} mandatory weight tensor(s): {}",
            missing.len(),
            missing.join(", ")
        )));
    }
    Ok(())
}

/// Install every loaded tensor onto `inner` (dense or quantized).
fn apply_weights(
    inner: &mut K2Inner,
    params: &HashMap<String, MxArray>,
    quant_bits: i32,
    quant_group_size: i32,
    top_level_mode: Option<PerLayerMode>,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
) -> Result<()> {
    validate_mandatory_weights(params, &inner.config, inner.layers.len())?;
    info!("Applying weights: {} tensors", params.len());

    let is_mxfp8 = is_mxfp8_checkpoint(params);
    let default_mode = resolve_default_mode(top_level_mode, is_mxfp8);
    let default_plq = default_per_layer_quant(quant_bits, quant_group_size, default_mode);

    load_embedding_affine_or_bf16(
        &mut inner.embed_tokens,
        params,
        "embed_tokens",
        per_layer_quant,
        default_plq,
        FAMILY,
    )?;

    if let Some(w) = params.get("norm.weight") {
        inner.norm.set_weight(w)?;
    }

    // Untied lm_head — LinearProj so an mxfp8 head can install (the
    // shipped checkpoint keeps it dense; `should_quantize` skips it).
    // Skipped entirely when tied: logits route through
    // `embed_tokens.as_linear` and a stray `lm_head.weight` would only
    // waste residency.
    if !inner.config.tie_word_embeddings {
        load_linear_proj_quantized_or_bf16(
            &mut inner.lm_head,
            params,
            "lm_head",
            per_layer_quant,
            default_plq,
            FAMILY,
        )?;
    }

    for (i, layer) in inner.layers.iter_mut().enumerate() {
        let prefix = format!("layers.{i}");

        if let Some(w) = params.get(&format!("{prefix}.input_layernorm.weight")) {
            layer.input_layernorm.set_weight(w)?;
        }
        if let Some(w) = params.get(&format!("{prefix}.post_attention_layernorm.weight")) {
            layer.post_attention_layernorm.set_weight(w)?;
        }

        let attn_prefix = format!("{prefix}.self_attn");
        load_linear_proj_quantized_or_bf16(
            layer.attention.q_proj_mut(),
            params,
            &format!("{attn_prefix}.q_proj"),
            per_layer_quant,
            default_plq,
            FAMILY,
        )?;
        load_linear_proj_quantized_or_bf16(
            layer.attention.k_proj_mut(),
            params,
            &format!("{attn_prefix}.k_proj"),
            per_layer_quant,
            default_plq,
            FAMILY,
        )?;
        load_linear_proj_quantized_or_bf16(
            layer.attention.v_proj_mut(),
            params,
            &format!("{attn_prefix}.v_proj"),
            per_layer_quant,
            default_plq,
            FAMILY,
        )?;
        load_linear_proj_quantized_or_bf16(
            layer.attention.o_proj_mut(),
            params,
            &format!("{attn_prefix}.o_proj"),
            per_layer_quant,
            default_plq,
            FAMILY,
        )?;

        load_dense_mlp_variant(
            &mut layer.mlp,
            params,
            &format!("{prefix}.mlp"),
            per_layer_quant,
            default_plq,
            FAMILY,
        )?;
    }

    info!("All weights applied successfully");
    Ok(())
}

/// Resident weight footprint: the packed/dense bytes of every checkpoint
/// tensor. mxfp8/sym8 tensors stay packed-resident; the FP8-dequantized
/// path has already replaced `.weight` entries with their bf16 expansion,
/// so the sum matches live memory either way.
fn compute_weight_bytes(params: &HashMap<String, MxArray>) -> u64 {
    params
        .values()
        .map(|a| a.nbytes() as u64)
        .fold(0u64, |acc, v| acc.saturating_add(v))
}

impl K2Inner {
    /// Load a `K2Inner` from a directory containing safetensors +
    /// config.json. Runs synchronously on the model thread; returns the
    /// inner plus the deterministic `(weight_bytes, pool_bytes)` totals
    /// for the cache-limit coordinator.
    pub fn load_from_dir(model_path: &str) -> Result<(Self, u64, u64)> {
        let path = Path::new(model_path);

        let mut config = parse_config(path)?;
        info!(
            "K2-Horizon config: {}L, h={}, heads={}, kv_heads={}, head_dim={}, ff={}, groups={}, vocab={}",
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim(),
            config.intermediate_size,
            config.layernorm_num_groups,
            config.vocab_size,
        );

        // Durable cold-tier intent resolved before the weight mmap so the
        // shard-identity guard brackets the complete load. Pure standard-KV
        // — no sidecar policy.
        let persist_env = std::env::var("MLX_PERSIST_PAGED_CACHE").ok();
        let persist_cold = resolve_persist_cold(
            "k2_horizon",
            persist_env.as_deref(),
            config.persist_paged_cache,
        );
        let mut checkpoint_load = CheckpointLoadGuard::before_mmap(path, persist_cold);

        let (quant_bits, quant_group_size, top_level_mode, per_layer_quant) =
            load_quant_settings_from_disk(path, DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE)?;

        let mut params = load_all_safetensors(path, false)?;
        checkpoint_load.record_mmap();

        // Watchdog pre-warm before the FIRST GPU eval on mmap-backed
        // weights (FP8 dequant is the first touch on the raw checkpoint).
        prewarm_checkpoint_pages(path);
        info!("Loaded {} tensors from safetensors", params.len());

        // FP8 dequant: DeepSeek-style `weight_scale_inv` first (generic
        // pass — a no-op for K2's layout), then compressed-tensors
        // `weight_scale` block scales.
        dequant_fp8_weights(&mut params, DType::BFloat16)?;
        dequant_fp8_block_scale(&mut params, DType::BFloat16)?;

        let params = sanitize_weights(&mut params)?;
        info!("Sanitized to {} tensors", params.len());

        // Authoritative quant signal: `.scales` tensors post-sanitize.
        // Paged is the default for K2 regardless (pure standard-KV); an
        // explicit config false still opts out.
        config.use_block_paged_cache =
            K2HorizonConfig::resolve_use_block_paged_default(config.use_block_paged_cache);

        let mut inner = K2Inner::new(config)?;
        inner.set_gen_defaults(crate::engine::persistence::parse_generation_defaults(path));

        apply_weights(
            &mut inner,
            &params,
            quant_bits,
            quant_group_size,
            top_level_mode,
            &per_layer_quant,
        )?;

        let weights_resident = {
            let weight_refs: Vec<&MxArray> = params.values().collect();
            crate::array::memory::materialize_weights(&weight_refs)?
        };
        inner.size_paged_pool_after_weight_load()?;

        if persist_cold
            && let Some(context) = inner.build_cold_tier_context(model_path, &weights_resident)
        {
            if checkpoint_load.stable_after_fingerprint() {
                inner.attach_cold_tier(context, &weights_resident);
            } else {
                tracing::warn!(
                    "cold-tier persistence disabled for {model_path}: model directory changed \
                     during load (shard identity mismatch); KV persistence stays off for safety"
                );
            }
        }

        let tokenizer_path = path.join("tokenizer.json");
        if tokenizer_path.exists() {
            let tokenizer = Qwen3Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| Error::from_reason(format!("Failed to load tokenizer: {e}")))?;
            inner.set_tokenizer(Arc::new(tokenizer));
            info!("Tokenizer loaded");
        }

        let weight_bytes: u64 = compute_weight_bytes(&params);
        let pool_bytes = inner.paged_pool_allocated_bytes()?;

        Ok((inner, weight_bytes, pool_bytes))
    }
}

impl K2HorizonModel {
    /// Load a K2-Horizon model: spawns the dedicated model thread, runs
    /// all weight loading inside it, then hands the thread the hybrid
    /// scheduler loop.
    pub async fn load_from_dir(model_path: &str) -> Result<Self> {
        let model_path = model_path.to_string();

        let (thread, init_rx) = crate::model_thread::ModelThread::spawn_with_scheduler(
            move || {
                let (inner, weight_bytes, pool_bytes) = K2Inner::load_from_dir(&model_path)?;
                let cache_limit_guard = crate::cache_limit::coordinator().register(weight_bytes);
                let pool_cache_limit_guard = (pool_bytes != 0)
                    .then(|| crate::cache_limit::coordinator().register_pool(pool_bytes));
                let config = inner.config.clone();
                let paged_active = inner.paged_adapter.is_some();
                Ok((
                    super::model::K2SchedulerState::new(inner)?,
                    (
                        config,
                        cache_limit_guard,
                        pool_cache_limit_guard,
                        paged_active,
                    ),
                ))
            },
            |state, receiver| state.drive(receiver),
        );

        let (config, cache_limit_guard, pool_cache_limit_guard, paged_active) = init_rx
            .await
            .map_err(|_| napi::Error::from_reason("Model thread exited during load"))??;

        Ok(K2HorizonModel {
            thread,
            config,
            paged_active,
            _cache_limit_guard: cache_limit_guard,
            _pool_cache_limit_guard: pool_cache_limit_guard,
        })
    }
}
