/**
 * Parameter Management Utilities
 *
 * Provides utilities to convert between different parameter representations:
 * - Structured dictionaries (HashMap<String, MxArray>) for functional forward pass
 * - Validation utilities for debugging
 *
 * Note: The actual gradient flow uses HashMap-based lookup (not positional ordering):
 * 1. model.get_parameters() returns HashMap<String, MxArray>
 * 2. autograd.rs sorts keys alphabetically for flat param vector
 * 3. Gradients are mapped back to HashMap by name
 * 4. apply_gradients() looks up parameters by name
 */
use crate::array::MxArray;
use crate::models::qwen3::Qwen3Config;
use napi::bindgen_prelude::*;
use std::collections::HashMap;

/// Map flat parameter vector to structured dictionary
///
/// Converts a flat vector of parameters (from autograd) back to a structured
/// dictionary that can be used by the functional forward pass.
///
/// # Arguments
/// * `params` - Flat vector of parameter arrays
/// * `param_names` - Names corresponding to each parameter
///
/// # Returns
/// * HashMap mapping parameter names to arrays
///
/// # Panics
/// * If params.len() != param_names.len()
pub fn map_params_to_dict(
    params: &[MxArray],
    param_names: &[String],
) -> Result<HashMap<String, MxArray>> {
    if params.len() != param_names.len() {
        return Err(Error::from_reason(format!(
            "Parameter count mismatch: {} arrays but {} names",
            params.len(),
            param_names.len()
        )));
    }

    let mut param_dict = HashMap::new();

    for (param, name) in params.iter().zip(param_names.iter()) {
        // Use clone() not copy() - clone just increments Arc reference count (O(1))
        // while copy() creates a full tensor copy with eval() (O(n))
        param_dict.insert(name.clone(), param.clone());
    }

    Ok(param_dict)
}

/// Get the total number of parameters in a Qwen3 model
///
/// This computes the expected parameter count based on model configuration.
/// Useful for validation and debugging.
///
/// # Arguments
/// * `config` - Model configuration
///
/// # Returns
/// * Total number of parameters (scalar count, not tensor count)
pub fn count_total_parameters(config: &Qwen3Config) -> i64 {
    let hidden_size = config.hidden_size as i64;
    let vocab_size = config.vocab_size as i64;
    let intermediate_size = config.intermediate_size as i64;
    let num_layers = config.num_layers as i64;

    let mut total = 0i64;

    // Embedding: vocab_size * hidden_size
    total += vocab_size * hidden_size;

    // Each transformer layer
    for _ in 0..num_layers {
        // Attention: Q, K, V, O projections
        // Q: hidden_size * hidden_size
        // K: hidden_size * (num_kv_heads / num_heads * hidden_size)
        // V: hidden_size * (num_kv_heads / num_heads * hidden_size)
        // O: hidden_size * hidden_size
        // For simplicity, assume num_kv_heads == num_heads (full attention)
        total += hidden_size * hidden_size * 4;

        // MLP: gate, up, down
        total += hidden_size * intermediate_size; // gate
        total += hidden_size * intermediate_size; // up
        total += intermediate_size * hidden_size; // down

        // Norms: input_layernorm + post_attention_layernorm
        total += hidden_size * 2;

        // Optional QK norms
        if config.use_qk_norm {
            let head_dim = hidden_size / config.num_heads as i64;
            total += head_dim * 2; // q_norm + k_norm
        }
    }

    // Final norm: hidden_size
    total += hidden_size;

    // LM head: hidden_size * vocab_size
    total += hidden_size * vocab_size;

    total
}

/// Validate that parameter names match expected model structure
///
/// Checks that all required parameters are present and named correctly.
/// Useful for debugging parameter extraction/mapping issues.
///
/// # Arguments
/// * `param_names` - List of parameter names to validate
/// * `config` - Expected model configuration
///
/// # Returns
/// * Ok(()) if valid, Err with description if invalid
pub fn validate_param_names(param_names: &[String], config: &Qwen3Config) -> Result<()> {
    let mut expected_names = Vec::new();

    // Embedding
    expected_names.push("embedding.weight".to_string());

    // Layers
    for layer_idx in 0..config.num_layers {
        let prefix = format!("layers.{}", layer_idx);

        expected_names.push(format!("{}.self_attn.q_proj.weight", prefix));
        expected_names.push(format!("{}.self_attn.k_proj.weight", prefix));
        expected_names.push(format!("{}.self_attn.v_proj.weight", prefix));
        expected_names.push(format!("{}.self_attn.o_proj.weight", prefix));

        if config.use_qk_norm {
            expected_names.push(format!("{}.self_attn.q_norm.weight", prefix));
            expected_names.push(format!("{}.self_attn.k_norm.weight", prefix));
        }

        expected_names.push(format!("{}.mlp.gate_proj.weight", prefix));
        expected_names.push(format!("{}.mlp.up_proj.weight", prefix));
        expected_names.push(format!("{}.mlp.down_proj.weight", prefix));

        expected_names.push(format!("{}.input_layernorm.weight", prefix));
        expected_names.push(format!("{}.post_attention_layernorm.weight", prefix));
    }

    // Final norm and LM head
    expected_names.push("final_norm.weight".to_string());
    expected_names.push("lm_head.weight".to_string());

    // Check that all expected names are present
    for expected_name in &expected_names {
        if !param_names.contains(expected_name) {
            return Err(Error::from_reason(format!(
                "Missing required parameter: {}",
                expected_name
            )));
        }
    }

    // Check that no unexpected names are present
    for param_name in param_names {
        if !expected_names.contains(param_name) {
            return Err(Error::from_reason(format!(
                "Unexpected parameter: {}",
                param_name
            )));
        }
    }

    Ok(())
}
