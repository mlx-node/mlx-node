// GRPO Training with MLX Autograd
//
// This module provides autograd-based training for GRPO, replacing manual gradients
// with automatic differentiation via MLX's value_and_grad.
//
// ## Architecture
//
// The challenge is that MLX autograd requires a pure function, but our model is stateful.
// We solve this by:
//
// 1. Extracting all trainable parameters into a flat Vec<MxArray>
// 2. Creating a loss closure that:
//    - Receives updated parameter values from MLX
//    - Maps them to a structured dictionary
//    - Recomputes forward pass using functional components
//    - Computes GRPO loss from recomputed logprobs
//    - Returns scalar loss
// 3. Calling autograd::value_and_grad to get loss + gradients
// 4. Mapping gradients back to parameter names
// 5. Applying gradients via existing optimizers
//
// ## Key Innovation
//
// Unlike the previous broken implementation, this version ACTUALLY recomputes the
// forward pass from parameters, creating a proper computation graph for autograd.

use std::collections::HashMap;

use napi::bindgen_prelude::*;

use crate::array::{MxArray, pad_float_sequences, pad_sequences};
use crate::autograd;
use crate::grpo::{advantages::compute_advantages, loss as grpo_loss};
use crate::models::qwen3::Qwen3Config;
use crate::nn::Activations;
use crate::param_manager;
use crate::utils::functional;

/// Compute GRPO loss with automatic differentiation using functional forward pass
///
/// This function computes both the loss value and gradients with respect to
/// all trainable parameters using MLX's automatic differentiation.
///
/// **KEY DIFFERENCE from previous implementation**: This version ACTUALLY recomputes
/// the forward pass from parameters, creating a proper computation graph.
///
/// # Arguments
/// * `model_config` - Model configuration (for functional forward pass)
/// * `model_params` - Current model parameters (will not be modified)
/// * `prompt_tokens` - Tokenized prompts (1D arrays)
/// * `completion_tokens` - Generated completions (1D arrays)
/// * `old_logprobs` - Log probabilities from old policy (for importance sampling)
/// * `rewards` - Reward values for each completion
/// * `group_size` - Number of completions per prompt
/// * `loss_config` - GRPO loss configuration
///
/// # Returns
/// * `(loss_value, gradients)` - Scalar loss and gradients for each parameter
///
/// # Example
///
/// ```no_run
/// # use mlx_node::grpo::compute_loss_and_gradients_autograd;
/// # use mlx_node::models::qwen3::Qwen3Config;
/// # use mlx_node::grpo::GRPOLossConfig;
/// # use std::collections::HashMap;
/// # let config = Qwen3Config {
/// #     vocab_size: 151936,
/// #     hidden_size: 1024,
/// #     num_layers: 28,
/// #     num_heads: 16,
/// #     num_kv_heads: 8,
/// #     intermediate_size: 3072,
/// #     rms_norm_eps: 1e-6,
/// #     rope_theta: 1000000.0,
/// #     max_position_embeddings: 40960,
/// #     head_dim: 64,
/// #     use_qk_norm: true,
/// #     tie_word_embeddings: true,
/// #     pad_token_id: 151643,
/// #     eos_token_id: 151645,
/// #     bos_token_id: 151643,
/// # };
/// # let params = HashMap::new();
/// # let prompt_tokens = vec![];
/// # let completion_tokens = vec![];
/// # let old_logprobs = vec![];
/// # let rewards = vec![];
/// # let group_size = 4;
/// # let loss_config = GRPOLossConfig::default();
/// let (loss, grads) = compute_loss_and_gradients_autograd(
///     &config,
///     &params,
///     &prompt_tokens,
///     &completion_tokens,
///     &old_logprobs,
///     &rewards,
///     group_size,
///     loss_config,
/// ).unwrap();
/// ```
pub fn compute_loss_and_gradients_autograd(
    model_config: &Qwen3Config,
    model_params: &HashMap<String, MxArray>,
    prompt_tokens: &[&MxArray],
    completion_tokens: &[&MxArray],
    old_logprobs: &[&MxArray], // Changed: use old_logprobs instead of recomputing
    rewards: &[f64],
    group_size: i32,
    loss_config: grpo_loss::GRPOLossConfig,
) -> Result<(f64, HashMap<String, MxArray>)> {
    // 1. Flatten parameters into ordered list
    let mut param_names: Vec<String> = model_params.keys().cloned().collect();
    param_names.sort(); // Ensure consistent ordering

    // CRITICAL FIX: Upcast ALL parameters to float32 BEFORE autograd
    //
    // Why this is necessary:
    // - Pretrained models use bfloat16 (dtype=3) for memory efficiency
    // - bfloat16 has only ~3 decimal digits of precision
    // - When computing gradients through 28 transformer layers, precision loss
    //   compounds through backpropagation, causing NaN gradients
    // - Random weights (float32) work fine, pretrained weights (bfloat16) produce
    //   307/311 NaN gradients without this fix
    //
    // The parameters are upcasted HERE (before value_and_grad), not inside
    // the closure, because MLX computes gradients with respect to the arrays
    // passed to value_and_grad. If we upcast inside the closure, the gradients
    // still flow to the original bfloat16 parameters.
    //
    // We store the original dtype to convert gradients back at the end.
    let original_dtypes: Vec<crate::array::DType> = param_names
        .iter()
        .map(|name| {
            model_params
                .get(name)
                .ok_or_else(|| Error::from_reason(format!("Parameter not found: {}", name)))
                .and_then(|p| p.dtype())
        })
        .collect::<Result<Vec<_>>>()?;

    // Upcast parameters to float32
    let param_arrays_f32: Vec<MxArray> = param_names
        .iter()
        .map(|name| {
            model_params
                .get(name)
                .ok_or_else(|| Error::from_reason(format!("Parameter not found: {}", name)))
                .and_then(|p| p.astype(crate::array::DType::Float32))
        })
        .collect::<Result<Vec<_>>>()?;

    let param_arrays: Vec<&MxArray> = param_arrays_f32.iter().collect();

    // 2. Prepare data for loss computation
    let rewards_f32: Vec<f32> = rewards.iter().map(|&x| x as f32).collect();
    let rewards_array = MxArray::from_float32(&rewards_f32, &[rewards.len() as i64])?;

    // Compute advantages (this doesn't need gradients)
    let advantages_array = compute_advantages(&rewards_array, group_size, "group".to_string())?;

    // 3. Pad sequences
    let prompts_expanded: Vec<&MxArray> = prompt_tokens
        .iter()
        .flat_map(|p| std::iter::repeat_n(*p, group_size as usize))
        .collect();

    // Use the model's actual pad_token_id for padding sequences
    // Using wrong padding (like 0) can cause extreme logits and NaN
    let pad_token_id = model_config.pad_token_id;

    let padded_prompts_result = pad_sequences(prompts_expanded, pad_token_id)?;
    let padded_prompts = padded_prompts_result.get_padded()?;

    let padded_completions_result = pad_sequences(completion_tokens.to_vec(), pad_token_id)?;
    let padded_completions = padded_completions_result.get_padded()?;
    let completion_masks = padded_completions_result.get_masks()?;

    // CRITICAL: Pad old_logprobs with a reasonable negative value, NOT 0.0
    //
    // Why 0.0 is problematic:
    // - Log prob of 0 means probability = 1 (100% certainty)
    // - If new_logprob at padded position is -50 (low prob), ratio = exp(-50 - 0) = tiny
    // - But if new_logprob has NaN (from model numerical issues), 0 * NaN = NaN!
    // - The mask doesn't help because NaN * 0 = NaN, not 0
    //
    // Why -5.0 works:
    // - Represents ~0.7% probability, a reasonable "unlikely but not extreme" value
    // - log_ratio at padded positions: new (-X) - (-5) = -(X-5), clamped to [-88, 88]
    // - Even if new is -100, log_ratio = -95, clamped to -88, exp(-88) ≈ 0
    // - Critical: if new_logprob is NaN, we need to handle it separately (see below)
    let padded_old_logprobs_raw = pad_float_sequences(old_logprobs.to_vec(), -5.0)?;

    // Clamp old_logprobs to reasonable range to prevent ratio explosion
    // This is defensive against very negative values from generation
    let padded_old_logprobs = padded_old_logprobs_raw.clip(Some(-20.0), Some(0.0))?;

    // 4. Concatenate prompts + completions for full sequence
    let input_ids = MxArray::concatenate(&padded_prompts, &padded_completions, 1)?;

    // Clone data needed by closure (must be owned)
    let param_names_clone = param_names.clone();
    let input_ids_clone = input_ids.clone();
    let padded_completions_clone = padded_completions.clone();
    let padded_old_logprobs_clone = padded_old_logprobs.clone();
    let advantages_clone = advantages_array.clone();
    let completion_masks_clone = completion_masks.clone();
    let config_clone = model_config.clone();
    let loss_config_clone = loss_config.clone();

    // 5. Define loss function for autograd
    // This closure will be called by MLX with updated parameter values
    // NOTE: Parameters are ALREADY in float32 (upcasted before value_and_grad call)
    let loss_fn = move |params: &[MxArray]| -> Result<MxArray> {
        // Map params to structured dictionary
        let param_dict = param_manager::map_params_to_dict(params, &param_names_clone)?;

        // Recompute forward pass with parameters
        // All computation happens in float32 for numerical stability
        let logits =
            functional::qwen3_forward_functional(&config_clone, &param_dict, &input_ids_clone)?;

        // Extract logits for completion part
        // Use shape_at() to avoid allocating full shape vector
        let batch_size = input_ids_clone.shape_at(0)?;
        let total_seq_len = input_ids_clone.shape_at(1)?;

        let completion_len = padded_completions_clone.shape_at(1)?;

        let prompt_len = total_seq_len - completion_len;

        // Get logits for completion tokens only
        let completion_logits = logits.slice(
            &[0, prompt_len, 0],
            &[batch_size, total_seq_len, config_clone.vocab_size as i64],
        )?;

        // Following TRL/transformers pattern: upcast to float32 for numerical stability
        // TRL's selective_log_softmax uses float32 for the computation
        // No clamping - TRL doesn't clamp logits before log_softmax
        let logits_f32 = completion_logits.astype(crate::array::DType::Float32)?;

        // Compute log probabilities in float32
        let logprobs_3d = Activations::log_softmax(&logits_f32, Some(-1))?;

        // Extract per-token logprobs using completion token IDs as indices
        let completion_logprobs = logprobs_3d
            .take_along_axis(&padded_completions_clone.expand_dims(-1)?, -1)?
            .squeeze(Some(&[-1]))?;

        // Clamp log probs to reasonable range to prevent numerical issues:
        // - Lower bound -20: prevents extreme ratios (exp(-20 - (-5)) = exp(-15) is tiny but stable)
        // - Upper bound 0: log probs are always <= 0
        // This also handles potential NaN by replacing with clamped values
        // Note: MLX clip replaces NaN with the nearest bound
        let clamped_completion_logprobs = completion_logprobs.clip(Some(-20.0), Some(0.0))?;

        // Compute GRPO loss
        let loss = grpo_loss::grpo_loss(
            &clamped_completion_logprobs,
            &padded_old_logprobs_clone,
            &advantages_clone,
            &completion_masks_clone,
            loss_config_clone.clone(),
            None, // no reference model
        )?;

        Ok(loss)
    };

    // 6. Compute value and gradients using MLX autograd
    let (loss_array, grad_arrays) = autograd::value_and_grad(param_arrays.clone(), loss_fn)?;

    // 7. Extract loss value and force evaluation
    loss_array.eval();
    let loss_value = loss_array.item_at_float32(0)? as f64;

    // 8. Map gradients back to parameter names and force evaluation
    // Gradients are computed in float32 for numerical stability.
    // KEEP gradients in float32 - do NOT convert back to bfloat16 as that can
    // reintroduce numerical issues. The optimizer should handle mixed precision.
    let gradients = param_names
        .into_iter()
        .enumerate()
        .map(|(i, param_name)| {
            let grad = grad_arrays[i].clone();
            grad.eval(); // Force evaluation to release computation graph
            (param_name, grad)
        })
        .collect::<HashMap<_, _>>();

    // Note: original_dtypes is kept for future use if needed
    let _ = original_dtypes;

    Ok((loss_value, gradients))
}
