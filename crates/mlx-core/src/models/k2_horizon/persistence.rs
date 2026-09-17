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
use crate::cold_tier::{resolve_persist_cold, shard_identities_stable, snapshot_shard_identities};
use crate::engine::persistence::{
    dequant_fp8_block_scale, dequant_fp8_weights, load_all_safetensors, prewarm_checkpoint_pages,
};
use crate::models::quant_dispatch::{
    PerLayerMode, PerLayerQuant, default_per_layer_quant, effective_plq_for,
    ensure_affine_biases_present, ensure_dense_weight_floating, ensure_int8_storage_resolves_sym8,
    ensure_kquant_storage_resolves_kquant, ensure_plain_fp8_storage_resolves_fp8_e4m3,
    load_quant_settings_from_disk, resolve_default_mode,
};
use crate::models::qwen3_5_moe::quantized_linear::{
    DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE, LinearProj, MLPVariant, QuantizedLinear,
    is_mxfp8_checkpoint, try_build_kquant_quantized_linear, try_build_mxfp4_quantized_linear,
    try_build_mxfp8_quantized_linear, try_build_nvfp4_quantized_linear,
    try_build_quantized_linear, try_build_sym8_quantized_linear,
};
use crate::tokenizer::Qwen3Tokenizer;

use super::config::K2HorizonConfig;
use super::model::{K2HorizonModel, K2Inner};

/// Parse config.json into `K2HorizonConfig` (+ structural validation +
/// `generation_config.json` EOS override).
fn parse_config(model_path: &Path) -> Result<K2HorizonConfig> {
    let config_path = model_path.join("config.json");
    let raw_str = fs::read_to_string(&config_path)
        .map_err(|e| Error::from_reason(format!("Failed to read config.json: {e}")))?;
    let config: K2HorizonConfig = serde_json::from_str(&raw_str)
        .map_err(|e| Error::from_reason(format!("Failed to deserialize K2HorizonConfig: {e}")))?;
    config.validate()?;
    Ok(config)
}

/// Sanitize HF weight keys to internal format: strip the `model.` prefix,
/// drop rotary buffers (computed at runtime), keep the untied `lm_head`.
///
/// The f32→bf16 cast keeps sym8 `.scales` (mandatory f32 `[N]`) at full
/// precision — identified by an Int8 sibling `.weight` — and skips
/// compressed-tensors `.weight_scale` leftovers (the dequant pass already
/// consumed them; a survivor is malformed but harmless, keep it f32 so the
/// mandatory-weights validator reports it untouched).
fn sanitize_weights(params: &mut HashMap<String, MxArray>) -> Result<HashMap<String, MxArray>> {
    let mut sanitized = HashMap::new();

    let keys: Vec<String> = params.keys().cloned().collect();
    for key in keys {
        let value = params.remove(&key).unwrap();
        let clean_key = key.strip_prefix("model.").unwrap_or(&key).to_string();

        if clean_key.contains("rotary_emb") {
            continue;
        }
        sanitized.insert(clean_key, value);
    }

    // Cast f32 tensors to bf16 to avoid dtype promotion issues — EXCLUDING
    // sym8 `.scales` (f32 [N] is the sym8 storage contract).
    let sym8_scales: std::collections::HashSet<String> = sanitized
        .keys()
        .filter_map(|k| {
            let prefix = k.strip_suffix(".scales")?;
            let w = sanitized.get(&format!("{prefix}.weight"))?;
            (w.dtype().ok()? == DType::Int8).then(|| k.clone())
        })
        .collect();
    for (k, value) in sanitized.iter_mut() {
        if sym8_scales.contains(k) {
            continue;
        }
        if value.dtype().is_ok_and(|dt| dt == DType::Float32)
            && let Ok(casted) = value.astype(DType::BFloat16)
        {
            *value = casted;
        }
    }

    Ok(sanitized)
}

/// Build a quantized `QuantizedLinear` for `base`, dispatching on the
/// resolved per-layer mode. `Ok(None)` = incomplete `.weight`/`.scales`
/// group (caller fails loud); `Err` = a mode/storage skew this builder
/// must never silently skip.
fn build_k2_non_moe_ql(
    params: &HashMap<String, MxArray>,
    base: &str,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    default_plq: PerLayerQuant,
) -> Result<Option<QuantizedLinear>> {
    let plq = effective_plq_for(base, per_layer_quant, default_plq, None);
    ensure_int8_storage_resolves_sym8(params, base, plq.mode, "k2_horizon")?;
    ensure_plain_fp8_storage_resolves_fp8_e4m3(params, base, plq.mode, "k2_horizon")?;
    ensure_kquant_storage_resolves_kquant(params, base, plq.mode, "k2_horizon")?;
    ensure_affine_biases_present(params, base, plq.mode, "k2_horizon")?;
    Ok(match plq.mode {
        PerLayerMode::Mxfp4 => try_build_mxfp4_quantized_linear(params, base),
        PerLayerMode::Mxfp8 => try_build_mxfp8_quantized_linear(params, base),
        PerLayerMode::Nvfp4 => try_build_nvfp4_quantized_linear(params, base),
        PerLayerMode::Fp8E4m3 => {
            return Err(Error::from_reason(format!(
                "k2_horizon: projection '{base}' resolved to fp8_e4m3, but plain per-output \
                 E4M3 storage is supported only by Qwen3.5 DGX artifacts"
            )));
        }
        PerLayerMode::Affine => {
            try_build_quantized_linear(params, base, plq.group_size, plq.bits)
        }
        PerLayerMode::Sym8 => try_build_sym8_quantized_linear(params, base)?,
        PerLayerMode::Q6K
        | PerLayerMode::Q4K
        | PerLayerMode::Q5K
        | PerLayerMode::Q3K
        | PerLayerMode::IQ4NL
        | PerLayerMode::IQ4XS
        | PerLayerMode::IQ3S => {
            try_build_kquant_quantized_linear(params, base, plq.mode, "k2_horizon")?
        }
    })
}

/// Load one `LinearProj` either quantized (ANY mode) or plain bf16, keyed
/// off `{base}.scales`.
fn load_linear_proj_quantized_or_bf16(
    proj: &mut LinearProj,
    params: &HashMap<String, MxArray>,
    base: &str,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    default_plq: PerLayerQuant,
) -> Result<()> {
    if params.contains_key(&format!("{base}.scales")) {
        let ql = build_k2_non_moe_ql(params, base, per_layer_quant, default_plq)?.ok_or_else(
            || {
                Error::from_reason(format!(
                    "k2_horizon: quantized tensor '{base}' has '.scales' but its packed \
                     '.weight' could not be resolved (missing weight/scales) — refusing to load \
                     with random init"
                ))
            },
        )?;
        proj.set_quantized(ql);
    } else if let Some(w) = params.get(&format!("{base}.weight")) {
        ensure_dense_weight_floating(&format!("{base}.weight"), w)?;
        proj.set_weight(w, base)?;
    }
    Ok(())
}

/// Whether any of a dense MLP's three projections carries `.scales`.
fn dense_mlp_is_quantized(params: &HashMap<String, MxArray>, prefix: &str) -> bool {
    ["gate_proj", "up_proj", "down_proj"]
        .iter()
        .any(|p| params.contains_key(&format!("{prefix}.{p}.scales")))
}

/// Load a dense `MLPVariant` (gate/up/down) quantized (ANY mode) or bf16.
fn load_dense_mlp_variant(
    ff: &mut MLPVariant,
    params: &HashMap<String, MxArray>,
    prefix: &str,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    default_plq: PerLayerQuant,
) -> Result<()> {
    let gate_base = format!("{prefix}.gate_proj");
    let up_base = format!("{prefix}.up_proj");
    let down_base = format!("{prefix}.down_proj");

    if dense_mlp_is_quantized(params, prefix) {
        let gate_proj = build_k2_non_moe_ql(params, &gate_base, per_layer_quant, default_plq)?
            .ok_or_else(|| {
                Error::from_reason(format!(
                    "k2_horizon: quantized dense-MLP projection '{gate_base}' could not be built \
                     (missing weight/scales)"
                ))
            })?;
        let up_proj = build_k2_non_moe_ql(params, &up_base, per_layer_quant, default_plq)?
            .ok_or_else(|| {
                Error::from_reason(format!(
                    "k2_horizon: quantized dense-MLP projection '{up_base}' could not be built \
                     (missing weight/scales)"
                ))
            })?;
        let down_proj = build_k2_non_moe_ql(params, &down_base, per_layer_quant, default_plq)?
            .ok_or_else(|| {
                Error::from_reason(format!(
                    "k2_horizon: quantized dense-MLP projection '{down_base}' could not be built \
                     (missing weight/scales)"
                ))
            })?;
        *ff = MLPVariant::Quantized {
            gate_proj,
            up_proj,
            down_proj,
        };
    } else {
        for (base, set) in [
            (gate_base.as_str(), MLPVariant::set_gate_proj_weight as fn(&mut MLPVariant, &MxArray) -> Result<()>),
            (up_base.as_str(), MLPVariant::set_up_proj_weight as fn(&mut MLPVariant, &MxArray) -> Result<()>),
            (down_base.as_str(), MLPVariant::set_down_proj_weight as fn(&mut MLPVariant, &MxArray) -> Result<()>),
        ] {
            if let Some(w) = params.get(&format!("{base}.weight")) {
                ensure_dense_weight_floating(&format!("{base}.weight"), w)?;
                set(ff, w)?;
            }
        }
    }
    Ok(())
}

/// Load `embed_tokens` PACKED-quantized (ANY mode) or bf16, keyed off
/// `embed_tokens.scales`. Packed stays resident — `forward` gather-then-
/// dequantizes per row, so the dense `vocab × hidden` table never
/// materializes.
fn load_embedding_affine_or_bf16(
    embedding: &mut crate::nn::Embedding,
    params: &HashMap<String, MxArray>,
    base: &str,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    default_plq: PerLayerQuant,
) -> Result<()> {
    if let Some(scales) = params.get(&format!("{base}.scales")) {
        let weight = params.get(&format!("{base}.weight")).ok_or_else(|| {
            Error::from_reason(format!(
                "k2_horizon: quantized embedding '{base}' has '.scales' but is missing its \
                 packed '.weight'"
            ))
        })?;
        let plq = effective_plq_for(base, per_layer_quant, default_plq, None);
        let (group_size, bits, mode) = match plq.mode {
            PerLayerMode::Affine => (plq.group_size, plq.bits, "affine"),
            PerLayerMode::Mxfp8 => (
                crate::models::qwen3_5_moe::quantized_linear::MXFP8_GROUP_SIZE,
                crate::models::qwen3_5_moe::quantized_linear::MXFP8_BITS,
                "mxfp8",
            ),
            PerLayerMode::Mxfp4 => (
                crate::models::qwen3_5_moe::quantized_linear::MXFP4_GROUP_SIZE,
                crate::models::qwen3_5_moe::quantized_linear::MXFP4_BITS,
                "mxfp4",
            ),
            PerLayerMode::Nvfp4 => (
                crate::models::qwen3_5_moe::quantized_linear::NVFP4_GROUP_SIZE,
                crate::models::qwen3_5_moe::quantized_linear::NVFP4_BITS,
                "nvfp4",
            ),
            other => {
                return Err(Error::from_reason(format!(
                    "k2_horizon: embedding '{base}' resolved to mode {other:?}, but the packed \
                     embedding path supports only affine/mxfp4/mxfp8/nvfp4"
                )));
            }
        };
        let biases = params.get(&format!("{base}.biases"));
        embedding.load_quantized_packed(weight, scales, biases, group_size, bits, mode)?;
    } else if let Some(w) = params.get(&format!("{base}.weight")) {
        ensure_dense_weight_floating(&format!("{base}.weight"), w)?;
        embedding.load_weight(w)?;
    }
    Ok(())
}

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
        )?;
        load_linear_proj_quantized_or_bf16(
            layer.attention.k_proj_mut(),
            params,
            &format!("{attn_prefix}.k_proj"),
            per_layer_quant,
            default_plq,
        )?;
        load_linear_proj_quantized_or_bf16(
            layer.attention.v_proj_mut(),
            params,
            &format!("{attn_prefix}.v_proj"),
            per_layer_quant,
            default_plq,
        )?;
        load_linear_proj_quantized_or_bf16(
            layer.attention.o_proj_mut(),
            params,
            &format!("{attn_prefix}.o_proj"),
            per_layer_quant,
            default_plq,
        )?;

        load_dense_mlp_variant(
            &mut layer.mlp,
            params,
            &format!("{prefix}.mlp"),
            per_layer_quant,
            default_plq,
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
        let persist_cold =
            resolve_persist_cold("k2_horizon", persist_env.as_deref(), config.persist_paged_cache);
        let shard_snapshot_before_mmap = if persist_cold {
            snapshot_shard_identities(path)
        } else {
            None
        };

        let (quant_bits, quant_group_size, top_level_mode, per_layer_quant) =
            load_quant_settings_from_disk(path, DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE)?;

        let mut params = load_all_safetensors(path, false)?;
        let shard_snapshot_at_mmap = if persist_cold {
            snapshot_shard_identities(path)
        } else {
            None
        };

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
            let after_fingerprint = snapshot_shard_identities(path);
            if shard_identities_stable(
                &shard_snapshot_before_mmap,
                &shard_snapshot_at_mmap,
                &after_fingerprint,
            ) {
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
