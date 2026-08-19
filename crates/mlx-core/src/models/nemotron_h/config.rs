use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::Value;

/// NVIDIA Nemotron 3.5 Lightning ("nemotron_h") model configuration.
///
/// Hybrid MoE architecture: 52 layers alternating between three mixer kinds
/// (`layers_block_type`): `mamba` (Mamba-2 SSM), `moe` (pure MoE-FFN), and
/// `attention` (GQA). Each layer is a single pre-RMSNorm + ONE mixer + a
/// residual connection; the final `norm_f` + a separate untied `lm_head`
/// complete the stack.
///
/// Parsed fail-closed from the checkpoint `config.json`; unknown layer block
/// types, missing mandatory fields, and unsupported optional features
/// (e.g. `moe_latent_size`) are rejected at load time rather than silently
/// mis-decoded.
#[napi(object)]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NemotronHConfig {
    /// Vocabulary size (131072 for the 30B Lightning checkpoint).
    pub vocab_size: i32,
    /// Hidden dimension (2688).
    pub hidden_size: i32,
    /// Total number of decoder layers (52).
    pub num_hidden_layers: i32,
    /// Number of query heads in the GQA attention layers (32).
    pub num_attention_heads: i32,
    /// Number of KV heads in the GQA attention layers (2).
    pub num_key_value_heads: i32,
    /// Per-head attention dimension (128).
    pub head_dim: i32,
    /// Maximum position embeddings (context length).
    pub max_position_embeddings: i32,
    /// RMSNorm epsilon for every norm in the model (1e-5).
    pub layer_norm_epsilon: f64,
    /// RoPE base frequency for the attention layers (10000).
    pub rope_theta: f64,
    /// Per-layer mixer kind, remapped from the checkpoint's
    /// `layers_block_type`: "mamba" -> "linear_attention", "attention" ->
    /// "full_attention", "moe" -> "moe" (the HF `MIXER_TYPES` names).
    pub layers_block_type: Vec<String>,

    // --- Mamba-2 mixer ---
    /// Number of SSM heads (64).
    pub mamba_num_heads: i32,
    /// SSM head dimension (64).
    pub mamba_head_dim: i32,
    /// SSM state size per head per group (128).
    pub ssm_state_size: i32,
    /// Number of SSM groups (8); each head belongs to group h / (H / G).
    pub n_groups: i32,
    /// Depthwise causal conv1d kernel size (4).
    pub conv_kernel: i32,
    /// Mamba-2 chunk-scan chunk size (128).
    pub chunk_size: i32,
    /// Minimum discretized time step (1e-3); softplus(dt + dt_bias) is
    /// clamped to [time_step_min, inf).
    pub time_step_min: f64,

    // --- MoE mixer ---
    /// Total routed experts (128).
    pub n_routed_experts: i32,
    /// Experts selected per token (6).
    pub num_experts_per_tok: i32,
    /// Post-normalization routing weight scale (2.5).
    pub routed_scaling_factor: f64,
    /// Number of expert groups (1 - grouping degenerates to a flat top-k).
    pub n_group: i32,
    /// Number of top groups selected (1).
    pub topk_group: i32,
    /// Renormalize the gathered top-k weights to sum to 1 before scaling.
    pub norm_topk_prob: bool,
    /// Per-expert MLP intermediate size (1856; non-gated up->relu2->down).
    pub intermediate_size: i32,
    /// Shared-expert MLP intermediate size (3712), applied on ALL tokens.
    pub moe_shared_expert_intermediate_size: i32,

    // --- Token ids / heads ---
    /// Tie the lm_head to the embedding table. False for Nemotron.
    pub tie_word_embeddings: bool,
    pub bos_token_id: i32,
    /// EOS token ids (config.json scalar; generation_config carries {2, 11}).
    #[napi(ts_type = "number[]")]
    pub eos_token_ids: Vec<i32>,
    pub pad_token_id: i32,
    /// Compute only the last `num_logits_to_keep` logits (1).
    pub num_logits_to_keep: i32,

    // --- MTP head ---
    /// MTP layer kinds, remapped like `layers_block_type` (["attention","moe"]).
    #[serde(default)]
    #[napi(ts_type = "string[]")]
    pub mtp_layers_block_type: Vec<String>,
    /// Number of MTP predictor steps (`num_nextn_predict_layers`; 1).
    #[serde(default)]
    pub n_mtp_layers: i32,

    // --- Block-paged KV cache ---
    /// Optional block-paged KV cache memory cap in MiB. None resolves to the
    /// default 2048 MiB pool; explicit values are honored.
    #[serde(default)]
    pub paged_cache_memory_mb: Option<u32>,
    /// Optional paged block size in tokens (default 16).
    #[serde(default)]
    pub paged_block_size: Option<u32>,
    /// Opt in/out of the block-paged KV adapter. None (the default) enables
    /// the paged pool and the continuous-batching scheduler lane; explicit
    /// `Some(false)` reverts to the flat-cache whole-turn path.
    #[serde(default)]
    pub use_block_paged_cache: Option<bool>,
}

impl NemotronHConfig {
    /// Mamba mixer intermediate size = mamba_num_heads * mamba_head_dim.
    pub fn mamba_intermediate_size(&self) -> i32 {
        self.mamba_num_heads * self.mamba_head_dim
    }

    /// The Mamba conv dim: intermediate + 2 * n_groups * ssm_state_size.
    pub fn mamba_conv_dim(&self) -> i32 {
        self.mamba_intermediate_size() + 2 * self.n_groups * self.ssm_state_size
    }

    /// in_proj output size: intermediate + conv_dim + num_heads (gate | xBC | dt).
    pub fn mamba_in_proj_size(&self) -> i32 {
        self.mamba_intermediate_size() + self.mamba_conv_dim() + self.mamba_num_heads
    }

    /// Whether layer `idx` uses the Mamba-2 SSM mixer.
    pub fn is_mamba_layer(&self, idx: usize) -> bool {
        self.layer_kind(idx) == "linear_attention"
    }

    /// Whether layer `idx` uses the GQA attention mixer.
    pub fn is_attention_layer(&self, idx: usize) -> bool {
        self.layer_kind(idx) == "full_attention"
    }

    /// Whether layer `idx` uses the pure MoE-FFN mixer.
    pub fn is_moe_layer(&self, idx: usize) -> bool {
        self.layer_kind(idx) == "moe"
    }

    /// The remapped block type of layer `idx`. Panics on an out-of-range
    /// index - the layer count is validated at parse time.
    pub fn layer_kind(&self, idx: usize) -> &str {
        &self.layers_block_type[idx]
    }

    /// Indices of the attention layers, in order.
    pub fn attention_layer_idxs(&self) -> Vec<usize> {
        (0..self.num_hidden_layers as usize)
            .filter(|&i| self.is_attention_layer(i))
            .collect()
    }

    /// The LAST attention layer index - the backbone attention layer whose
    /// flat K/V cache the MTP head's attention reads as "target KV".
    pub fn last_attention_idx(&self) -> Option<usize> {
        self.attention_layer_idxs().pop()
    }

    /// Rough resident-memory estimate in bytes (bf16 weights + packed
    /// quantized approximations), used for the wired-limit context.
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

/// Remap a checkpoint `layers_block_type` entry to the internal HF
/// `MIXER_TYPES` name. Fails closed on unknown kinds.
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

/// Parse a required bool config field, failing closed when missing or invalid.
fn req_bool(raw: &Value, key: &str) -> Result<bool> {
    raw.get(key).and_then(Value::as_bool).ok_or_else(|| {
        Error::from_reason(format!(
            "config.json missing or invalid required field '{key}'"
        ))
    })
}

/// Parse the optional `eos_token_id` as a scalar or array, returning the
/// flattened id list. Empty when absent.
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

/// Parse a `NemotronHConfig` from a parsed checkpoint `config.json` value.
///
/// Fail-closed contract: every field the model math depends on is required
/// and validated; the `model_type` must be `nemotron_h`, `layers_block_type`
/// entries must be known mixer kinds, the layer count must equal the block
/// list length, and unsupported optional features (`moe_latent_size`) are
/// rejected.
pub fn parse_config(raw: &Value) -> Result<NemotronHConfig> {
    // The architecture is authoritative, matching the TypeScript registry's
    // architecture probe and the native gemma4 loader: a config that
    // declares NemotronHForCausalLM is this family even when model_type is
    // absent or malformed. Without the architecture, model_type must match.
    let model_type = raw.get("model_type").and_then(Value::as_str).unwrap_or("");
    // Accept BOTH registry-blessed forms: the array and the bare string
    // (the TS normalizeConfig contract explicitly converts the string form
    // into a single-element architecture set, so this parser must agree).
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
        rope_theta: req_f64(raw, "rope_theta")?,
        layers_block_type,
        mamba_num_heads,
        mamba_head_dim,
        ssm_state_size,
        n_groups,
        conv_kernel: req_i32(raw, "conv_kernel")?,
        chunk_size: {
            // Fail-closed: the Mamba-2 chunk scan divides by chunk_size, so a
            // zero (or negative) value from checkpoint metadata would panic
            // the model thread on the first multi-token forward instead of
            // failing the load.
            let chunk_size = req_i32(raw, "chunk_size")?;
            if chunk_size <= 0 {
                return Err(Error::from_reason(format!(
                    "config.json chunk_size must be positive, got {chunk_size}"
                )));
            }
            chunk_size
        },
        time_step_min: req_f64(raw, "time_step_min")?,
        n_routed_experts,
        num_experts_per_tok,
        routed_scaling_factor: req_f64(raw, "routed_scaling_factor")?,
        n_group,
        topk_group,
        norm_topk_prob: req_bool(raw, "norm_topk_prob")?,
        intermediate_size: req_i32(raw, "intermediate_size")?,
        moe_shared_expert_intermediate_size: req_i32(raw, "moe_shared_expert_intermediate_size")?,
        tie_word_embeddings: req_bool(raw, "tie_word_embeddings")?,
        bos_token_id: raw.get("bos_token_id").and_then(Value::as_i64).unwrap_or(0) as i32,
        eos_token_ids,
        pad_token_id: raw.get("pad_token_id").and_then(Value::as_i64).unwrap_or(0) as i32,
        num_logits_to_keep: raw
            .get("num_logits_to_keep")
            .and_then(Value::as_i64)
            .unwrap_or(1) as i32,
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
            "rope_theta": 10000.0,
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
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "num_logits_to_keep": 1,
            "mtp_layers_block_type": ["attention", "moe"],
            "num_nextn_predict_layers": 1,
        })
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
        // Layer classification: 23 mamba + 23 moe + 6 attention.
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
        assert_eq!(cfg.last_attention_idx(), Some(42));
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
        // Wrong model_type WITH the architecture present is accepted (the
        // registry's architecture probe selects this family for the same
        // config, so the native loader must agree).
        let mut v = lightning_json();
        v["model_type"] = json!("qwen3");
        assert!(parse_config(&v).is_ok());
    }

    #[test]
    fn accepts_bare_string_architecture_form() {
        // The TS registry's normalizeConfig blesses the bare-string
        // architectures form as a single-element set; the native parser must
        // agree or detectModelType() would select a family load() rejects.
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
