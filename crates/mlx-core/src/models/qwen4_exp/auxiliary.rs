use super::{config::Config, weights::Store};
use napi::{Error, Result};
use std::path::Path;

pub fn attach(store: &mut Store, config: &mut Config, path: &Path) -> Result<serde_json::Value> {
    let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
    let source = Config::parse(&raw)?;
    // GGUF represents the same PLE as a single table and rounds epsilon to
    // f32. Compare execution geometry, rather than serialization artifacts.
    let target = serde_json::to_value(&*config)?;
    let auxiliary = serde_json::to_value(&source)?;
    for key in [
        "hidden_size",
        "vocab_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "hc_count",
        "hc_lowrank",
        "indexer_n_heads",
        "indexer_head_dim",
        "indexer_budget",
        "indexer_compress_ratio",
        "eos_token_id",
        "ple_embed_dim",
        "ngram_size",
        "heads_per_ngram",
    ] {
        if target[key] != auxiliary[key] {
            return Err(Error::from_reason(format!(
                "Qwen4 auxiliary checkpoint {key} does not match the target"
            )));
        }
    }
    if config.rope_dims() != source.rope_dims() || config.rope_theta() != source.rope_theta() {
        return Err(Error::from_reason(
            "Qwen4 auxiliary rotary geometry does not match target",
        ));
    }
    let source_store = Store::open_metadata(path, None)?;
    store.attach_auxiliary(source_store)?;
    config.rope_parameters = source.rope_parameters;
    Ok(raw)
}

pub fn validate_mtp(store: &Store, c: &Config) -> Result<bool> {
    if !store.tensors.keys().any(|k| k.starts_with("mtp.")) {
        return Ok(false);
    }
    let h = c.hidden_size;
    let hc = h * c.hc_count;
    let mut shapes = vec![
        ("pre_fc_norm_embedding.weight".to_string(), vec![h]),
        ("pre_fc_norm_hidden.weight".into(), vec![hc]),
        ("fc_embedding.weight".into(), vec![h, h]),
        ("fc_hidden.weight".into(), vec![h, h]),
    ];
    for prefix in [
        "hyper_connection_mixer",
        "layers.0.attn_hyper_connection",
        "layers.0.mlp_hyper_connection",
    ] {
        shapes.extend([
            (format!("{prefix}.hc_norm.weight"), vec![hc]),
            (
                format!("{prefix}.input_mix_weight_down.weight"),
                vec![c.hc_lowrank, hc],
            ),
            (
                format!("{prefix}.input_mix_weight_up.weight"),
                vec![hc, c.hc_lowrank],
            ),
        ]);
        if prefix.starts_with("layers") {
            shapes.push((
                format!("{prefix}.block_inject_weight.weight"),
                vec![c.hc_count, hc],
            ));
        }
    }
    let q = c.num_attention_heads * c.head_dim;
    let k = c.num_key_value_heads * c.head_dim;
    for (name, shape) in [
        ("self_attn.q_proj.weight", vec![2 * q, h]),
        ("self_attn.k_proj.weight", vec![k, h]),
        ("self_attn.v_proj.weight", vec![k, h]),
        ("self_attn.o_proj.weight", vec![h, q]),
        ("self_attn.q_norm.weight", vec![c.head_dim]),
        ("self_attn.k_norm.weight", vec![c.head_dim]),
        (
            "self_attn.indexer.index_qk_proj.weight",
            vec![(c.indexer_n_heads + 1) * c.indexer_head_dim, h],
        ),
        (
            "self_attn.indexer.q_layernorm.weight",
            vec![c.indexer_head_dim],
        ),
        (
            "self_attn.indexer.k_layernorm.weight",
            vec![c.indexer_head_dim],
        ),
        ("mlp.gate.weight", vec![c.num_experts, h]),
        (
            "mlp.experts.gate_up_proj",
            vec![c.num_experts, 2 * c.moe_intermediate_size, h],
        ),
        (
            "mlp.experts.down_proj",
            vec![c.num_experts, h, c.moe_intermediate_size],
        ),
        (
            "mlp.shared_expert.gate_proj.weight",
            vec![c.shared_expert_intermediate_size, h],
        ),
        (
            "mlp.shared_expert.up_proj.weight",
            vec![c.shared_expert_intermediate_size, h],
        ),
        (
            "mlp.shared_expert.down_proj.weight",
            vec![h, c.shared_expert_intermediate_size],
        ),
        ("mlp.shared_expert_gate.weight", vec![1, h]),
    ] {
        shapes.push((format!("layers.0.{name}"), shape));
    }
    for (name, shape) in shapes {
        let key = format!("mtp.{name}");
        if store.descriptor(&key)?.shape != shape {
            return Err(Error::from_reason(format!(
                "Qwen4 incompatible MTP tensor {key}"
            )));
        }
    }
    Ok(true)
}
