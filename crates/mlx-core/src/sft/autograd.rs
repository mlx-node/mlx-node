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
use crate::nn::Losses;
use crate::param_manager;
use crate::training_model::ModelType;
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
pub(crate) fn compute_sft_loss_and_gradients(
    model_type: &ModelType,
    model_params: &HashMap<String, MxArray>,
    input_ids: &MxArray,
    labels: &MxArray,
    loss_config: SftLossConfig,
    use_checkpointing: bool,
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

    // 2. Compute gradients in inner scope so closure captures are dropped before cleanup
    let (loss_value, gradients) = {
        // Clone data needed by closure
        let param_names_clone = param_names.clone();
        let input_ids_clone = input_ids.clone();
        let labels_clone = labels.clone();
        let config_clone = model_type.clone();
        let loss_config_clone = loss_config.clone();

        // Define loss function for autograd
        let ckpt_contexts =
            std::rc::Rc::new(std::cell::RefCell::new(autograd::CheckpointContexts::new()));
        let ckpt_ctx = ckpt_contexts.clone();
        let loss_fn = move |params: &[MxArray]| -> Result<MxArray> {
            // Map params to structured dictionary
            let param_dict = param_manager::map_params_to_dict(params, &param_names_clone)?;

            // Forward pass using functional implementation
            let logits = functional::forward_functional_dispatch(
                &config_clone,
                &param_dict,
                &input_ids_clone,
                use_checkpointing,
                &mut ckpt_ctx.borrow_mut(),
            )?;

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

        // Compute value and gradients using MLX autograd
        let (loss_array, grad_arrays) = autograd::value_and_grad(param_arrays, loss_fn)?;

        // Eval all outputs INSIDE the scope
        loss_array.eval();
        for grad in &grad_arrays {
            grad.eval();
        }

        // Reclaim checkpoint contexts now that the graph is eval'd
        ckpt_contexts.borrow_mut().reclaim();

        // Extract values before scope ends
        let loss_val = loss_array.item_at_float32(0)? as f64;
        let grads: HashMap<String, MxArray> = param_names
            .into_iter()
            .enumerate()
            .map(|(i, name)| (name, grad_arrays[i].clone()))
            .collect();

        (loss_val, grads)
    }; // Closure and its captures (input_ids_clone, labels_clone, etc.) are dropped here!

    // 3. NOW do cleanup - computation graph should be fully releasable
    crate::array::heavy_cleanup();

    Ok((loss_value, gradients))
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
