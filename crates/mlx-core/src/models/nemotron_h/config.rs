use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::Value;

/// NVIDIA Nemotron 3.5 Lightning ("nemotron_h") model configuration.
///
/// Hybrid MoE: every layer is one pre-RMSNorm + ONE mixer + a residual, closed
/// by `norm_f` and an untied `lm_head`. Parsed fail-closed — unknown block
/// types, missing fields and unsupported features are rejected at load.
#[napi(object)]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NemotronHConfig {
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub max_position_embeddings: i32,
    pub layer_norm_epsilon: f64,
    /// Per-layer mixer kind, remapped from the checkpoint's
    /// `layers_block_type` to the HF `MIXER_TYPES` names.
    pub layers_block_type: Vec<String>,

    // --- Mamba-2 mixer ---
    pub mamba_num_heads: i32,
    pub mamba_head_dim: i32,
    /// SSM state size, per head per group.
    pub ssm_state_size: i32,
    /// Number of SSM groups; head `h` belongs to group `h / (H / G)`.
    pub n_groups: i32,
    /// Depthwise causal conv1d kernel size.
    pub conv_kernel: i32,
    pub chunk_size: i32,
    /// Declared minimum discretized time step.
    ///
    /// UNUSED BY THE RUNTIME. No served reference clamps dt to it - only HF's
    /// torch fallback does. `time_step_limit_pair()` is the real clamp.
    pub time_step_min: f64,
    /// Optional `[min, max]` bounds for the discretized time step. `None` is
    /// the reference default `(0.0, +inf)`, i.e. no clamp.
    #[napi(ts_type = "number[]")]
    pub time_step_limit: Option<Vec<f64>>,

    // --- MoE mixer ---
    pub n_routed_experts: i32,
    pub num_experts_per_tok: i32,
    /// Routing weight scale, applied after normalization.
    pub routed_scaling_factor: f64,
    /// Renormalize the gathered top-k weights to sum to 1 before scaling.
    pub norm_topk_prob: bool,
    /// Per-expert MLP intermediate size (non-gated up -> relu2 -> down).
    pub intermediate_size: i32,
    /// Shared-expert MLP intermediate size; it runs on ALL tokens.
    pub moe_shared_expert_intermediate_size: i32,

    // --- Token ids ---
    /// EOS token ids, from the config.json scalar or array.
    #[napi(ts_type = "number[]")]
    pub eos_token_ids: Vec<i32>,

    // --- MTP head ---
    /// MTP layer kinds, remapped like `layers_block_type`.
    #[serde(default)]
    #[napi(ts_type = "string[]")]
    pub mtp_layers_block_type: Vec<String>,
    /// Number of MTP predictor steps (`num_nextn_predict_layers`).
    #[serde(default)]
    pub n_mtp_layers: i32,

    // --- Block-paged KV cache ---
    /// Optional block-paged KV cache memory cap in MiB. `None` requests MiB
    /// for one `max_position_embeddings` sequence (6 GiB on Lightning
    /// 30B-A3B), not 2048; explicit values are honored.
    #[serde(default)]
    pub paged_cache_memory_mb: Option<u32>,
    /// Optional paged block size in tokens (default 16).
    #[serde(default)]
    pub paged_block_size: Option<u32>,
    /// Opt in/out of the block-paged KV adapter. `None` enables the paged pool
    /// and the continuous-batching lane; `Some(false)` reverts to whole-turn.
    #[serde(default)]
    pub use_block_paged_cache: Option<bool>,
}

impl NemotronHConfig {
    /// The `(min, max)` bounds the Mamba-2 mixer clips `softplus(dt + dt_bias)`
    /// to. Defaults to `(0.0, +inf)` - no clamp - as mlx-lm does.
    /// `time_step_min` is deliberately NOT consulted; see its doc comment.
    pub fn time_step_limit_pair(&self) -> (f64, f64) {
        match self.time_step_limit.as_deref() {
            Some([lo, hi]) => (*lo, *hi),
            _ => (0.0, f64::INFINITY),
        }
    }

    /// Mamba mixer intermediate size = mamba_num_heads * mamba_head_dim.
    pub fn mamba_intermediate_size(&self) -> i32 {
        self.mamba_num_heads * self.mamba_head_dim
    }

    /// The Mamba conv dim: intermediate + 2 * n_groups * ssm_state_size.
    pub fn mamba_conv_dim(&self) -> i32 {
        self.mamba_intermediate_size() + 2 * self.n_groups * self.ssm_state_size
    }

    /// in_proj output size: intermediate + conv_dim + num_heads (gate | xBC | dt).
    ///
    /// TEST-ONLY, and gated so it stays that way: `new_mamba_mixer` inlines
    /// the same sum, and two definitions of it would drift.
    #[cfg(test)]
    pub fn mamba_in_proj_size(&self) -> i32 {
        self.mamba_intermediate_size() + self.mamba_conv_dim() + self.mamba_num_heads
    }

    pub fn is_mamba_layer(&self, idx: usize) -> bool {
        self.layer_kind(idx) == "linear_attention"
    }

    pub fn is_attention_layer(&self, idx: usize) -> bool {
        self.layer_kind(idx) == "full_attention"
    }

    pub fn is_moe_layer(&self, idx: usize) -> bool {
        self.layer_kind(idx) == "moe"
    }

    /// The remapped block type of layer `idx`; panics out of range, which the
    /// parse-time layer-count check rules out.
    pub fn layer_kind(&self, idx: usize) -> &str {
        &self.layers_block_type[idx]
    }

    /// Indices of the attention layers, in order.
    pub fn attention_layer_idxs(&self) -> Vec<usize> {
        (0..self.num_hidden_layers as usize)
            .filter(|&i| self.is_attention_layer(i))
            .collect()
    }

    /// Rough resident-memory estimate for the wired-limit context.
    pub fn estimate_memory_bytes(&self) -> u64 {
        let h = self.hidden_size as u64;
        let v = self.vocab_size as u64;
        let n = self.num_hidden_layers as u64;

        let e = self.n_routed_experts as u64;
        let moe_i = self.intermediate_size as u64;
        let shared_i = self.moe_shared_expert_intermediate_size as u64;
        let mamba_i = self.mamba_intermediate_size() as u64;
        let conv_dim = self.mamba_conv_dim() as u64;

        let embed = v * h;
        // Expert weights + shared expert + router (non-gated: 2 projections)
        let moe_params = e * 2 * h * moe_i + 2 * h * shared_i + e * h;
        let attn_params = h * (self.num_attention_heads as u64 * self.head_dim as u64) * 2
            + h * (self.num_key_value_heads as u64 * self.head_dim as u64) * 2;
        let mamba_params = h * (mamba_i + conv_dim + self.mamba_num_heads as u64)
            + mamba_i * h
            + conv_dim * self.conv_kernel as u64
            + 3 * mamba_i;
        let per_layer = moe_params + attn_params + mamba_params + h;
        let total_params = embed * 2 + n * per_layer + h;

        // 2 bytes per param (bf16)
        total_params * 2
    }
}

/// Remap a `layers_block_type` entry to the HF `MIXER_TYPES` name.
fn remap_block_type(kind: &str, ctx: &str) -> Result<String> {
    match kind {
        "mamba" => Ok("linear_attention".to_string()),
        "attention" => Ok("full_attention".to_string()),
        "moe" => Ok("moe".to_string()),
        "mlp" => Ok("mlp".to_string()),
        other => Err(Error::from_reason(format!(
            "Unknown {ctx} entry '{other}': expected 'mamba', 'moe', 'attention', or 'mlp'"
        ))),
    }
}

/// Parse a required i32 config field, failing closed when missing or invalid.
fn req_i32(raw: &Value, key: &str) -> Result<i32> {
    raw.get(key)
        .and_then(Value::as_i64)
        .map(|v| v as i32)
        .ok_or_else(|| {
            Error::from_reason(format!(
                "config.json missing or invalid required field '{key}'"
            ))
        })
}

/// Parse a required f64 config field, failing closed when missing or invalid.
fn req_f64(raw: &Value, key: &str) -> Result<f64> {
    raw.get(key).and_then(Value::as_f64).ok_or_else(|| {
        Error::from_reason(format!(
            "config.json missing or invalid required field '{key}'"
        ))
    })
}

/// Parse an optional f64 config field, falling back to `default`.
fn opt_f64(raw: &Value, key: &str, default: f64) -> f64 {
    raw.get(key).and_then(Value::as_f64).unwrap_or(default)
}

/// Parse the optional `time_step_limit`. Absent is `None` (no clamp); anything
/// mis-shaped is rejected, since ignoring it changes every SSM time step.
fn parse_time_step_limit(raw: &Value) -> Result<Option<Vec<f64>>> {
    let Some(v) = raw.get("time_step_limit") else {
        return Ok(None);
    };
    if v.is_null() {
        return Ok(None);
    }
    let pair: Vec<f64> = v
        .as_array()
        .map(|items| items.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_default();
    if pair.len() != 2 || v.as_array().map(Vec::len) != Some(2) {
        return Err(Error::from_reason(format!(
            "config.json time_step_limit must be a [min, max] pair of numbers, got {v}"
        )));
    }
    // `partial_cmp` rather than `!(a <= b)` so a NaN bound is rejected.
    let ordered = matches!(
        pair[0].partial_cmp(&pair[1]),
        Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
    );
    if !ordered {
        return Err(Error::from_reason(format!(
            "config.json time_step_limit min must be <= max, got {v}"
        )));
    }
    Ok(Some(pair))
}

/// Parse a required bool config field, failing closed when missing or invalid.
fn req_bool(raw: &Value, key: &str) -> Result<bool> {
    raw.get(key).and_then(Value::as_bool).ok_or_else(|| {
        Error::from_reason(format!(
            "config.json missing or invalid required field '{key}'"
        ))
    })
}

/// Parse the optional `eos_token_id`, scalar or array, into a flat id list.
fn parse_eos_ids(raw: &Value) -> Result<Vec<i32>> {
    match raw.get("eos_token_id") {
        None => Ok(Vec::new()),
        Some(v) if v.is_array() => v
            .as_array()
            .expect("checked is_array")
            .iter()
            .map(|item| {
                item.as_i64().map(|v| v as i32).ok_or_else(|| {
                    Error::from_reason(format!(
                        "config.json eos_token_id array contains a non-integer: {item}"
                    ))
                })
            })
            .collect(),
        Some(v) => v.as_i64().map(|v| vec![v as i32]).ok_or_else(|| {
            Error::from_reason(format!(
                "config.json eos_token_id must be an integer or an array, got {v}"
            ))
        }),
    }
}

/// Parse a `NemotronHConfig` from a checkpoint `config.json`. Fail-closed:
/// every field the model math depends on is required and validated.
pub fn parse_config(raw: &Value) -> Result<NemotronHConfig> {
    // The architecture is authoritative, matching the TS registry probe: a
    // config declaring NemotronHForCausalLM is this family regardless.
    let model_type = raw.get("model_type").and_then(Value::as_str).unwrap_or("");
    // Both forms: TS `normalizeConfig` converts a bare string into a
    // single-element architecture set, so this parser must accept it too.
    let architectures_declare_nemotron = match raw.get("architectures") {
        Some(serde_json::Value::Array(values)) => values
            .iter()
            .any(|a| a.as_str() == Some("NemotronHForCausalLM")),
        Some(serde_json::Value::String(s)) => s == "NemotronHForCausalLM",
        _ => false,
    };
    if model_type != "nemotron_h" && !architectures_declare_nemotron {
        return Err(Error::from_reason(format!(
            "config.json model_type must be 'nemotron_h' (or architectures must declare 'NemotronHForCausalLM'), got model_type '{model_type}'"
        )));
    }
    if raw.get("moe_latent_size").is_some_and(|v| !v.is_null()) {
        return Err(Error::from_reason(
            "config.json moe_latent_size is non-null: latent-projected experts are not supported",
        ));
    }

    let block_raw = raw.get("layers_block_type").ok_or_else(|| {
        Error::from_reason("config.json missing required field 'layers_block_type'")
    })?;
    let block_arr = block_raw.as_array().ok_or_else(|| {
        Error::from_reason("config.json layers_block_type must be an array of strings")
    })?;
    let mut layers_block_type = Vec::with_capacity(block_arr.len());
    for (i, entry) in block_arr.iter().enumerate() {
        let kind = entry.as_str().ok_or_else(|| {
            Error::from_reason(format!(
                "config.json layers_block_type[{i}] is not a string: {entry}"
            ))
        })?;
        layers_block_type.push(remap_block_type(kind, &format!("layers_block_type[{i}]"))?);
    }

    let num_hidden_layers = req_i32(raw, "num_hidden_layers")?;
    if num_hidden_layers <= 0 {
        return Err(Error::from_reason(format!(
            "config.json num_hidden_layers must be > 0, got {num_hidden_layers}"
        )));
    }
    if layers_block_type.len() != num_hidden_layers as usize {
        return Err(Error::from_reason(format!(
            "config.json layers_block_type has {} entries but num_hidden_layers={num_hidden_layers}",
            layers_block_type.len()
        )));
    }

    // MTP layer kinds (optional; absent means no MTP head).
    let mut mtp_layers_block_type = Vec::new();
    if let Some(mtp_raw) = raw.get("mtp_layers_block_type") {
        let mtp_arr = mtp_raw.as_array().ok_or_else(|| {
            Error::from_reason("config.json mtp_layers_block_type must be an array of strings")
        })?;
        for (i, entry) in mtp_arr.iter().enumerate() {
            let kind = entry.as_str().ok_or_else(|| {
                Error::from_reason(format!(
                    "config.json mtp_layers_block_type[{i}] is not a string: {entry}"
                ))
            })?;
            mtp_layers_block_type.push(remap_block_type(
                kind,
                &format!("mtp_layers_block_type[{i}]"),
            )?);
        }
    }

    let num_attention_heads = req_i32(raw, "num_attention_heads")?;
    let num_key_value_heads = req_i32(raw, "num_key_value_heads")?;
    if num_attention_heads <= 0 || num_key_value_heads <= 0 {
        return Err(Error::from_reason(format!(
            "config.json attention heads must be > 0 (heads={num_attention_heads}, kv_heads={num_key_value_heads})"
        )));
    }
    if num_attention_heads % num_key_value_heads != 0 {
        return Err(Error::from_reason(format!(
            "config.json num_attention_heads ({num_attention_heads}) must be divisible by \
             num_key_value_heads ({num_key_value_heads})"
        )));
    }
    let head_dim = req_i32(raw, "head_dim")?;
    if head_dim <= 0 {
        return Err(Error::from_reason(format!(
            "config.json head_dim must be > 0, got {head_dim}"
        )));
    }

    let mamba_num_heads = req_i32(raw, "mamba_num_heads")?;
    let mamba_head_dim = req_i32(raw, "mamba_head_dim")?;
    let ssm_state_size = req_i32(raw, "ssm_state_size")?;
    let n_groups = req_i32(raw, "n_groups")?;
    if mamba_num_heads <= 0 || mamba_head_dim <= 0 || ssm_state_size <= 0 || n_groups <= 0 {
        return Err(Error::from_reason(format!(
            "config.json mamba sizes must be > 0 (heads={mamba_num_heads}, head_dim={mamba_head_dim}, \
             state={ssm_state_size}, groups={n_groups})"
        )));
    }
    if mamba_num_heads % n_groups != 0 {
        return Err(Error::from_reason(format!(
            "config.json mamba_num_heads ({mamba_num_heads}) must be divisible by n_groups ({n_groups})"
        )));
    }

    let n_routed_experts = req_i32(raw, "n_routed_experts")?;
    let num_experts_per_tok = req_i32(raw, "num_experts_per_tok")?;
    if n_routed_experts <= 0 || num_experts_per_tok <= 0 || num_experts_per_tok > n_routed_experts {
        return Err(Error::from_reason(format!(
            "config.json invalid MoE routing (experts={n_routed_experts}, experts_per_tok={num_experts_per_tok})"
        )));
    }
    let n_group = raw.get("n_group").and_then(Value::as_i64).unwrap_or(1) as i32;
    let topk_group = raw.get("topk_group").and_then(Value::as_i64).unwrap_or(1) as i32;
    if n_group <= 0 || topk_group <= 0 || topk_group > n_group {
        return Err(Error::from_reason(format!(
            "config.json invalid expert grouping (n_group={n_group}, topk_group={topk_group})"
        )));
    }
    if n_routed_experts % n_group != 0 {
        return Err(Error::from_reason(format!(
            "config.json n_routed_experts ({n_routed_experts}) must be divisible by n_group ({n_group})"
        )));
    }
    if n_group != 1 || topk_group != 1 {
        return Err(Error::from_reason(format!(
            "config.json expert grouping n_group={n_group}/topk_group={topk_group} is not the \
             supported degenerate (1/1) configuration"
        )));
    }

    let n_mtp_layers = raw
        .get("num_nextn_predict_layers")
        .and_then(Value::as_i64)
        .unwrap_or(0) as i32;
    if n_mtp_layers < 0 {
        return Err(Error::from_reason(format!(
            "config.json num_nextn_predict_layers must be >= 0, got {n_mtp_layers}"
        )));
    }
    if n_mtp_layers > 0 && mtp_layers_block_type.is_empty() {
        return Err(Error::from_reason(
            "config.json declares num_nextn_predict_layers > 0 but mtp_layers_block_type is missing",
        ));
    }
    if n_mtp_layers > 1 {
        return Err(Error::from_reason(format!(
            "config.json num_nextn_predict_layers={n_mtp_layers}: only a single-step MTP head is supported"
        )));
    }

    // Fail closed on a tied head: the loader has no embedding-tied path, so
    // accepting it only defers the failure to a missing-tensor error deep in
    // weight loading. Reject it where the reason is still legible.
    if req_bool(raw, "tie_word_embeddings")? {
        return Err(Error::from_reason(
            "config.json tie_word_embeddings=true: this family supports only an UNTIED lm_head. \
             The weight loader requires an explicit 'lm_head.weight' tensor and has no \
             embedding-tied path.",
        ));
    }

    let eos_token_ids = parse_eos_ids(raw)?;
    if eos_token_ids.is_empty() {
        return Err(Error::from_reason(
            "config.json missing eos_token_id (Nemotron requires at least one EOS id)",
        ));
    }

    Ok(NemotronHConfig {
        vocab_size: req_i32(raw, "vocab_size")?,
        hidden_size: req_i32(raw, "hidden_size")?,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        max_position_embeddings: req_i32(raw, "max_position_embeddings")?,
        layer_norm_epsilon: req_f64(raw, "layer_norm_epsilon")?,
        layers_block_type,
        mamba_num_heads,
        mamba_head_dim,
        ssm_state_size,
        n_groups,
        conv_kernel: req_i32(raw, "conv_kernel")?,
        chunk_size: {
            // Fail-closed: the chunk scan divides by chunk_size, so a
            // non-positive value would panic the model thread on the first
            // multi-token forward instead of failing the load.
            let chunk_size = req_i32(raw, "chunk_size")?;
            if chunk_size <= 0 {
                return Err(Error::from_reason(format!(
                    "config.json chunk_size must be positive, got {chunk_size}"
                )));
            }
            chunk_size
        },
        // Optional: nothing clips to it, so a config that omits it must still
        // load. The runtime clamp comes from `time_step_limit` alone.
        time_step_min: opt_f64(raw, "time_step_min", 0.001),
        time_step_limit: parse_time_step_limit(raw)?,
        n_routed_experts,
        num_experts_per_tok,
        routed_scaling_factor: req_f64(raw, "routed_scaling_factor")?,
        norm_topk_prob: req_bool(raw, "norm_topk_prob")?,
        intermediate_size: req_i32(raw, "intermediate_size")?,
        moe_shared_expert_intermediate_size: req_i32(raw, "moe_shared_expert_intermediate_size")?,
        eos_token_ids,
        mtp_layers_block_type,
        n_mtp_layers,
        paged_cache_memory_mb: raw
            .get("paged_cache_memory_mb")
            .and_then(Value::as_u64)
            .map(|v| v as u32),
        paged_block_size: raw
            .get("paged_block_size")
            .and_then(Value::as_u64)
            .map(|v| v as u32),
        use_block_paged_cache: raw.get("use_block_paged_cache").and_then(Value::as_bool),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn lightning_json() -> Value {
        let block: Vec<Value> = (0..52)
            .map(|i| {
                json!(match i {
                    5 | 12 | 19 | 26 | 33 | 42 => "attention",
                    i if i % 2 == 0 => "mamba",
                    _ => "moe",
                })
            })
            .collect();
        json!({
            "model_type": "nemotron_h",
            "architectures": ["NemotronHForCausalLM"],
            "vocab_size": 131072,
            "hidden_size": 2688,
            "num_hidden_layers": 52,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "max_position_embeddings": 1048576,
            "layer_norm_epsilon": 1e-5,
            "layers_block_type": block,
            "mamba_num_heads": 64,
            "mamba_head_dim": 64,
            "ssm_state_size": 128,
            "n_groups": 8,
            "conv_kernel": 4,
            "chunk_size": 128,
            "time_step_min": 0.001,
            "n_routed_experts": 128,
            "num_experts_per_tok": 6,
            "routed_scaling_factor": 2.5,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": true,
            "intermediate_size": 1856,
            "moe_shared_expert_intermediate_size": 3712,
            "tie_word_embeddings": false,
            "eos_token_id": 2,
            "mtp_layers_block_type": ["attention", "moe"],
            "num_nextn_predict_layers": 1,
        })
    }

    /// An absent `time_step_limit` must resolve to `(0.0, +inf)`.
    ///
    /// Mutation caught: deriving it from `time_step_min`, which silently
    /// bounds real SSM heads.
    #[test]
    fn absent_time_step_limit_is_unbounded() {
        let raw = lightning_json();
        assert!(
            raw.get("time_step_limit").is_none(),
            "the released config declares no time_step_limit"
        );
        let cfg = parse_config(&raw).expect("lightning config parses");
        assert_eq!(cfg.time_step_min, 0.001, "still parsed, just unused");
        assert_eq!(cfg.time_step_limit, None);
        assert_eq!(cfg.time_step_limit_pair(), (0.0, f64::INFINITY));
    }

    /// A declared `time_step_limit` pair is honoured verbatim.
    #[test]
    fn declared_time_step_limit_is_parsed() {
        let mut raw = lightning_json();
        raw.as_object_mut()
            .unwrap()
            .insert("time_step_limit".into(), json!([0.0, 0.1]));
        let cfg = parse_config(&raw).expect("config with time_step_limit parses");
        assert_eq!(cfg.time_step_limit, Some(vec![0.0, 0.1]));
        assert_eq!(cfg.time_step_limit_pair(), (0.0, 0.1));
    }

    /// A mis-shaped `time_step_limit` must fail the load, not be ignored.
    #[test]
    fn malformed_time_step_limit_is_rejected() {
        for bad in [
            json!([0.1]),
            json!([0.0, 0.1, 0.2]),
            json!(0.1),
            json!(["a", "b"]),
        ] {
            let mut raw = lightning_json();
            raw.as_object_mut()
                .unwrap()
                .insert("time_step_limit".into(), bad.clone());
            assert!(
                parse_config(&raw).is_err(),
                "time_step_limit {bad} must be rejected"
            );
        }
        // min > max is nonsense too.
        let mut raw = lightning_json();
        raw.as_object_mut()
            .unwrap()
            .insert("time_step_limit".into(), json!([1.0, 0.5]));
        assert!(parse_config(&raw).is_err(), "min > max must be rejected");
    }

    /// Nothing reads `time_step_min`, so a checkpoint that omits it must still
    /// load. Regression against parsing it with `req_f64`.
    #[test]
    fn time_step_min_is_optional() {
        let mut raw = lightning_json();
        raw.as_object_mut().unwrap().remove("time_step_min");
        let cfg = parse_config(&raw).expect("config without time_step_min parses");
        assert_eq!(cfg.time_step_min, 0.001);
        assert_eq!(cfg.time_step_limit_pair(), (0.0, f64::INFINITY));
    }

    #[test]
    fn parses_lightning_checkpoint_config() {
        let cfg = parse_config(&lightning_json()).expect("lightning config parses");
        assert_eq!(cfg.vocab_size, 131072);
        assert_eq!(cfg.hidden_size, 2688);
        assert_eq!(cfg.num_hidden_layers, 52);
        assert_eq!(cfg.mamba_intermediate_size(), 4096);
        assert_eq!(cfg.mamba_conv_dim(), 4096 + 2 * 8 * 128);
        assert_eq!(cfg.mamba_in_proj_size(), 4096 + 6144 + 64);
        assert_eq!(cfg.eos_token_ids, vec![2]);
        assert_eq!(cfg.n_mtp_layers, 1);
        assert_eq!(cfg.mtp_layers_block_type, vec!["full_attention", "moe"]);
        assert_eq!(cfg.layers_block_type.len(), 52);
        assert_eq!(
            cfg.layers_block_type
                .iter()
                .filter(|k| k.as_str() == "linear_attention")
                .count(),
            23
        );
        assert_eq!(
            cfg.layers_block_type
                .iter()
                .filter(|k| k.as_str() == "moe")
                .count(),
            23
        );
        assert_eq!(
            cfg.layers_block_type
                .iter()
                .filter(|k| k.as_str() == "full_attention")
                .count(),
            6
        );
        assert!(cfg.is_mamba_layer(0));
        assert!(cfg.is_moe_layer(1));
        assert!(cfg.is_attention_layer(5));
        assert_eq!(cfg.attention_layer_idxs(), vec![5, 12, 19, 26, 33, 42]);
    }

    #[test]
    fn eos_token_id_array_parses() {
        let mut v = lightning_json();
        v["eos_token_id"] = json!([2, 11]);
        let cfg = parse_config(&v).unwrap();
        assert_eq!(cfg.eos_token_ids, vec![2, 11]);
    }

    #[test]
    fn architecture_is_authoritative_over_wrong_model_type() {
        // The TS registry's architecture probe selects this family for such a
        // config, so the native loader must agree.
        let mut v = lightning_json();
        v["model_type"] = json!("qwen3");
        assert!(parse_config(&v).is_ok());
    }

    #[test]
    fn accepts_bare_string_architecture_form() {
        // TS `normalizeConfig` blesses this form, so the native parser must
        // agree or `detectModelType()` picks a family `load()` rejects.
        let mut v = lightning_json();
        v.as_object_mut().unwrap().remove("model_type");
        v["architectures"] = json!("NemotronHForCausalLM");
        assert!(parse_config(&v).is_ok());
    }

    #[test]
    fn rejects_wrong_model_type_without_architecture() {
        let mut v = lightning_json();
        v["model_type"] = json!("qwen3");
        v.as_object_mut().unwrap().remove("architectures");
        let err = parse_config(&v).unwrap_err();
        assert!(err.reason.contains("model_type"), "{}", err.reason);
    }

    #[test]
    fn rejects_missing_required_field() {
        let mut v = lightning_json();
        v.as_object_mut().unwrap().remove("hidden_size");
        let err = parse_config(&v).unwrap_err();
        assert!(err.reason.contains("hidden_size"), "{}", err.reason);
    }

    #[test]
    fn rejects_unknown_block_type() {
        let mut v = lightning_json();
        v["layers_block_type"][0] = json!("ssm");
        let err = parse_config(&v).unwrap_err();
        assert!(err.reason.contains("ssm"), "{}", err.reason);
    }

    #[test]
    fn rejects_block_count_mismatch() {
        let mut v = lightning_json();
        v["layers_block_type"] = json!(["mamba", "moe"]);
        let err = parse_config(&v).unwrap_err();
        assert!(err.reason.contains("num_hidden_layers"), "{}", err.reason);
    }

    #[test]
    fn rejects_zero_chunk_size() {
        let mut v = lightning_json();
        v["chunk_size"] = json!(0);
        let err = parse_config(&v).unwrap_err();
        assert!(err.reason.contains("chunk_size"), "{}", err.reason);
    }

    #[test]
    fn rejects_non_degenerate_grouping() {
        let mut v = lightning_json();
        v["n_group"] = json!(2);
        v["topk_group"] = json!(1);
        let err = parse_config(&v).unwrap_err();
        assert!(err.reason.contains("n_group"), "{}", err.reason);
    }

    #[test]
    fn rejects_moe_latent_size() {
        let mut v = lightning_json();
        v["moe_latent_size"] = json!(512);
        let err = parse_config(&v).unwrap_err();
        assert!(err.reason.contains("moe_latent_size"), "{}", err.reason);
    }

    /// A tied-head checkpoint must be rejected AT PARSE TIME, or the failure
    /// only defers to a missing-tensor error that reads like a bad download.
    ///
    /// Mutation caught: dropping the guard for a bare `req_bool`.
    #[test]
    fn rejects_tied_word_embeddings() {
        let mut v = lightning_json();
        v["tie_word_embeddings"] = json!(true);
        let err = parse_config(&v).unwrap_err();
        assert!(
            err.reason.contains("tie_word_embeddings"),
            "the message must name the field: {}",
            err.reason
        );
        assert!(
            err.reason.contains("lm_head.weight"),
            "the message must name the tensor the loader would have failed on: {}",
            err.reason
        );
        // ...and the field is still REQUIRED, not a silent `false`.
        let mut missing = lightning_json();
        missing
            .as_object_mut()
            .unwrap()
            .remove("tie_word_embeddings");
        let err = parse_config(&missing).unwrap_err();
        assert!(err.reason.contains("tie_word_embeddings"), "{}", err.reason);
    }

    #[test]
    fn rejects_multi_step_mtp() {
        let mut v = lightning_json();
        v["num_nextn_predict_layers"] = json!(2);
        let err = parse_config(&v).unwrap_err();
        assert!(err.reason.contains("single-step"), "{}", err.reason);
    }

    #[test]
    fn mtp_absent_when_no_predict_layers() {
        let mut v = lightning_json();
        v.as_object_mut()
            .unwrap()
            .remove("num_nextn_predict_layers");
        v.as_object_mut().unwrap().remove("mtp_layers_block_type");
        let cfg = parse_config(&v).unwrap();
        assert_eq!(cfg.n_mtp_layers, 0);
        assert!(cfg.mtp_layers_block_type.is_empty());
    }
}
