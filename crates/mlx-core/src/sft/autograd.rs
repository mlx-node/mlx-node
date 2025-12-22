// SFT Training with MLX Autograd
//
// This module provides autograd-based training for Supervised Fine-Tuning (SFT).
// Much simpler than GRPO - just forward pass + cross-entropy loss on completions.
//
// ## Architecture
//
// 1. Extract trainable parameters into flat Vec<MxArray>
// 2. Create loss closure that:
//    - Maps params to structured dictionary
//    - Runs functional forward pass
//    - Computes cross-entropy loss on completion tokens
//    - Returns scalar loss
// 3. Call autograd::value_and_grad
// 4. Map gradients back to parameter names

use std::collections::HashMap;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::autograd;
use crate::models::qwen3::Qwen3Config;
use crate::nn::Losses;
use crate::param_manager;
use crate::utils::functional;

use super::SftLossConfig;

/// Compute SFT loss and gradients using autograd
///
/// This is much simpler than GRPO autograd:
/// - No importance sampling or policy ratios
/// - No advantage computation
/// - Just cross-entropy loss on completion tokens
///
/// # Arguments
/// * `model_config` - Model configuration (for functional forward pass)
/// * `model_params` - Current model parameters (will not be modified)
/// * `input_ids` - Full input sequences [batch, seq_len] (prompts + completions)
/// * `labels` - Target labels [batch, seq_len] with -100 for prompt/padding tokens
/// * `loss_config` - SFT loss configuration
///
/// # Returns
/// * `(loss_value, gradients)` - Scalar loss and gradients for each parameter
pub fn compute_sft_loss_and_gradients(
    model_config: &Qwen3Config,
    model_params: &HashMap<String, MxArray>,
    input_ids: &MxArray,
    labels: &MxArray,
    loss_config: SftLossConfig,
) -> Result<(f64, HashMap<String, MxArray>)> {
    // 1. Flatten parameters into ordered list (keep native dtype - bfloat16)
    let mut param_names: Vec<String> = model_params.keys().cloned().collect();
    param_names.sort(); // Ensure consistent ordering

    let param_arrays: Vec<&MxArray> = param_names
        .iter()
        .map(|name| {
            model_params
                .get(name)
                .ok_or_else(|| Error::from_reason(format!("Parameter not found: {}", name)))
        })
        .collect::<Result<Vec<_>>>()?;

    // 2. Clone data needed by closure
    let param_names_clone = param_names.clone();
    let input_ids_clone = input_ids.clone();
    let labels_clone = labels.clone();
    let config_clone = model_config.clone();
    let loss_config_clone = loss_config.clone();

    // 3. Define loss function for autograd
    let loss_fn = move |params: &[MxArray]| -> Result<MxArray> {
        // Map params to structured dictionary
        let param_dict = param_manager::map_params_to_dict(params, &param_names_clone)?;

        // Forward pass using functional implementation
        let logits =
            functional::qwen3_forward_functional(&config_clone, &param_dict, &input_ids_clone)?;

        // Get shapes
        let batch_size = logits.shape_at(0)?;
        let seq_len = logits.shape_at(1)?;
        let vocab_size = logits.shape_at(2)?;

        // For autograd-compatible cross-entropy, we need to:
        // 1. Shift logits and labels for next-token prediction
        // 2. Reshape for cross_entropy
        // 3. Apply ignore_index masking

        // Shift: logits[:-1] predicts labels[1:]
        // This is standard causal LM training
        let shift_logits = logits.slice(&[0, 0, 0], &[batch_size, seq_len - 1, vocab_size])?;
        let shift_labels = labels_clone.slice(&[0, 1], &[batch_size, seq_len])?;

        // Reshape for cross_entropy
        let logits_flat = shift_logits.reshape(&[(batch_size) * (seq_len - 1), vocab_size])?;
        let labels_flat = shift_labels.reshape(&[(batch_size) * (seq_len - 1)])?;

        // Compute cross-entropy loss
        // The Losses::cross_entropy handles:
        // - Numerical stability via logsumexp
        // - ignore_index masking
        // - Label smoothing
        let ignore_idx = loss_config_clone.ignore_index.unwrap_or(-100);
        let label_smoothing = loss_config_clone.label_smoothing.unwrap_or(0.0);

        Losses::cross_entropy(
            &logits_flat,
            &labels_flat,
            None,
            Some(ignore_idx),
            Some(label_smoothing),
        )
    };

    // 4. Compute value and gradients using MLX autograd
    let (loss_array, grad_arrays) = autograd::value_and_grad(param_arrays, loss_fn)?;

    // 5. Eval all outputs and cleanup
    loss_array.eval();
    for grad in &grad_arrays {
        grad.eval();
    }
    crate::array::heavy_cleanup();

    // 6. Extract loss value
    let loss_value = loss_array.item_at_float32(0)? as f64;

    // 7. Map gradients back to parameter names
    let gradients = param_names
        .into_iter()
        .enumerate()
        .map(|(i, param_name)| (param_name, grad_arrays[i].clone()))
        .collect::<HashMap<_, _>>();

    Ok((loss_value, gradients))
}

/// Compute token-level accuracy for SFT training
///
/// Returns the fraction of correctly predicted tokens (excluding ignored tokens).
/// This requires an extra forward pass but provides useful training metrics.
///
/// # Arguments
/// * `model_config` - Model configuration
/// * `model_params` - Current model parameters
/// * `input_ids` - Input sequences [batch, seq_len]
/// * `labels` - Target labels [batch, seq_len] with -100 for ignored tokens
///
/// # Returns
/// * Accuracy value between 0.0 and 1.0
pub fn compute_token_accuracy(
    model_config: &Qwen3Config,
    model_params: &HashMap<String, MxArray>,
    input_ids: &MxArray,
    labels: &MxArray,
) -> Result<f64> {
    // Forward pass
    let logits = functional::qwen3_forward_functional(model_config, model_params, input_ids)?;

    // Get shapes
    let batch_size = logits.shape_at(0)?;
    let seq_len = logits.shape_at(1)?;
    let vocab_size = logits.shape_at(2)?;

    // Shift for next-token prediction (same as loss computation)
    let shift_logits = logits.slice(&[0, 0, 0], &[batch_size, seq_len - 1, vocab_size])?;
    let shift_labels = labels.slice(&[0, 1], &[batch_size, seq_len])?;

    // Get predictions (argmax over vocab)
    let predictions = shift_logits.argmax(-1, Some(false))?;
    let labels_flat = shift_labels.reshape(&[(batch_size) * (seq_len - 1)])?;
    let preds_flat = predictions.reshape(&[(batch_size) * (seq_len - 1)])?;

    // Create mask for valid (non-ignored) tokens
    let ignore_val = MxArray::scalar_int(-100)?;
    let valid_mask = labels_flat.not_equal(&ignore_val)?;

    // Compute correct predictions (where pred == label AND label != -100)
    let correct = preds_flat.equal(&labels_flat)?;
    let correct_and_valid = correct.logical_and(&valid_mask)?;

    // Sum correct and total valid
    let num_correct = correct_and_valid.sum(None, Some(false))?;
    let num_valid = valid_mask.sum(None, Some(false))?;

    num_correct.eval();
    num_valid.eval();

    let correct_val = num_correct.item_at_float32(0)? as f64;
    let valid_val = num_valid.item_at_float32(0)? as f64;

    if valid_val == 0.0 {
        Ok(0.0)
    } else {
        Ok(correct_val / valid_val)
    }
}

/// Compute SFT loss only (no gradients)
///
/// Useful for evaluation/validation without the overhead of gradient computation.
pub fn compute_sft_loss_only(
    model_config: &Qwen3Config,
    model_params: &HashMap<String, MxArray>,
    input_ids: &MxArray,
    labels: &MxArray,
    loss_config: SftLossConfig,
) -> Result<f64> {
    // Build param dict
    let param_dict: HashMap<String, MxArray> = model_params.clone();

    // Forward pass
    let logits = functional::qwen3_forward_functional(model_config, &param_dict, input_ids)?;

    // Get shapes
    let batch_size = logits.shape_at(0)?;
    let seq_len = logits.shape_at(1)?;
    let vocab_size = logits.shape_at(2)?;

    // Shift for next-token prediction
    let shift_logits = logits.slice(&[0, 0, 0], &[batch_size, seq_len - 1, vocab_size])?;
    let shift_labels = labels.slice(&[0, 1], &[batch_size, seq_len])?;

    // Reshape
    let logits_flat = shift_logits.reshape(&[(batch_size) * (seq_len - 1), vocab_size])?;
    let labels_flat = shift_labels.reshape(&[(batch_size) * (seq_len - 1)])?;

    // Compute loss
    let ignore_idx = loss_config.ignore_index.unwrap_or(-100);
    let label_smoothing = loss_config.label_smoothing.unwrap_or(0.0);

    let loss = Losses::cross_entropy(
        &logits_flat,
        &labels_flat,
        None,
        Some(ignore_idx),
        Some(label_smoothing),
    )?;

    loss.eval();
    let loss_value = loss.item_at_float32(0)? as f64;

    Ok(loss_value)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full autograd tests require a model, which is tested at integration level
    // Here we just verify the function signatures compile correctly

    #[test]
    fn test_loss_config_default() {
        let config = SftLossConfig::default();
        assert_eq!(config.ignore_index, Some(-100));
        assert_eq!(config.label_smoothing, Some(0.0));
    }
}
