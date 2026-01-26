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
use crate::nn::efficient_selective_log_softmax;
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
/// # use mlx_core::grpo::compute_loss_and_gradients_autograd;
/// # use mlx_core::models::qwen3::Qwen3Config;
/// # use mlx_core::grpo::GRPOLossConfig;
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
/// #     use_paged_attention: None,
/// #     paged_cache_memory_mb: None,
/// #     paged_block_size: None,
/// #     use_fp8_cache: None,
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
    // Early validation: KL penalty (beta > 0) requires reference model which isn't supported
    if loss_config.beta > 0.0 {
        return Err(Error::from_reason(
            "KL penalty (beta > 0) requires reference model logprobs which is not yet supported in autograd mode. \
             Set klCoef: 0.0 in your config to disable KL penalty.",
        ));
    }

    // 1. Flatten parameters into ordered list (keep native dtype - bfloat16)
    //
    // NOTE: We no longer upcast parameters to float32. MLX-LM's official trainer
    // runs in native bfloat16 throughout forward and backward passes. bfloat16
    // has the same exponent range as float32, so it won't overflow like fp16.
    // Only the loss scalar needs float32 for summation accuracy.
    //
    // This saves ~50% memory per training step (no float32 copies of all params).
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

    // 2. Prepare data for loss computation
    let rewards_f32: Vec<f32> = rewards.iter().map(|&x| x as f32).collect();
    let rewards_array = MxArray::from_float32(&rewards_f32, &[rewards.len() as i64])?;

    // Compute advantages (this doesn't need gradients)
    let advantages_array = compute_advantages(&rewards_array, group_size, "group".to_string())?;

    // 3. Truncate completions to max_completion_length if specified
    // This prevents OOM from degenerate outputs that hit the max token limit
    let max_completion_length = loss_config.max_completion_length.unwrap_or(1024);
    let completion_tokens_truncated: Vec<MxArray> = completion_tokens
        .iter()
        .map(|tokens| {
            let len = tokens.shape_at(0).unwrap_or(0);
            if len > max_completion_length {
                // Truncate to max_completion_length
                tokens
                    .slice_axis(0, 0, max_completion_length)
                    .unwrap_or_else(|_| (*tokens).clone())
            } else {
                (*tokens).clone()
            }
        })
        .collect();
    let completion_tokens_refs: Vec<&MxArray> = completion_tokens_truncated.iter().collect();

    // Also truncate old_logprobs to match
    let old_logprobs_truncated: Vec<MxArray> = old_logprobs
        .iter()
        .map(|logprobs| {
            let len = logprobs.shape_at(0).unwrap_or(0);
            if len > max_completion_length {
                logprobs
                    .slice_axis(0, 0, max_completion_length)
                    .unwrap_or_else(|_| (*logprobs).clone())
            } else {
                (*logprobs).clone()
            }
        })
        .collect();
    let old_logprobs_refs: Vec<&MxArray> = old_logprobs_truncated.iter().collect();

    // 4. Pad sequences
    let prompts_expanded: Vec<&MxArray> = prompt_tokens
        .iter()
        .flat_map(|p| std::iter::repeat_n(*p, group_size as usize))
        .collect();

    // Use the model's actual pad_token_id for padding sequences
    // Using wrong padding (like 0) can cause extreme logits and NaN
    let pad_token_id = model_config.pad_token_id;

    let padded_prompts_result = pad_sequences(prompts_expanded, pad_token_id)?;
    let padded_prompts = padded_prompts_result.get_padded()?;

    let padded_completions_result = pad_sequences(completion_tokens_refs, pad_token_id)?;
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
    let padded_old_logprobs_raw = pad_float_sequences(old_logprobs_refs, -5.0)?;

    // Clamp old_logprobs to reasonable range to prevent ratio explosion
    // This is defensive against very negative values from generation
    let padded_old_logprobs = padded_old_logprobs_raw.clip(Some(-20.0), Some(0.0))?;

    // 4. Concatenate prompts + completions for full sequence
    let input_ids = MxArray::concatenate(&padded_prompts, &padded_completions, 1)?;

    // Check if we should use chunked autograd (separate autograd per chunk)
    // This is different from chunked forward - it runs autograd multiple times
    // and accumulates gradients, allowing memory to be freed between chunks.
    let forward_chunk_size = loss_config.forward_chunk_size;
    let batch_size = input_ids.shape_at(0)?;

    if let Some(chunk_size) = forward_chunk_size
        && batch_size > chunk_size
    {
        // CHUNKED AUTOGRAD PATH: Run autograd per chunk and accumulate gradients
        // This allows MLX to release the computation graph after each chunk.
        return compute_loss_and_gradients_chunked_autograd(
            model_config,
            &param_names,
            &param_arrays,
            &input_ids,
            &padded_completions,
            &padded_old_logprobs,
            &advantages_array,
            &completion_masks,
            &loss_config,
            chunk_size,
        );
    }

    // STANDARD PATH: Single autograd call for entire batch
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
    // Parameters are in native dtype (bfloat16 for pretrained models)
    let lm_head_chunk_size = loss_config.lm_head_chunk_size;
    let loss_fn = move |params: &[MxArray]| -> Result<MxArray> {
        // Map params to structured dictionary
        let param_dict = param_manager::map_params_to_dict(params, &param_names_clone)?;

        // Use shape_at() to avoid allocating full shape vector
        let batch_size = input_ids_clone.shape_at(0)?;
        let total_seq_len = input_ids_clone.shape_at(1)?;
        let completion_len = padded_completions_clone.shape_at(1)?;
        let prompt_len = total_seq_len - completion_len;

        // Get completion logprobs using either chunked LM head or full computation
        // Note: forward_chunk_size is handled by chunked autograd path above
        let completion_logprobs = if let Some(chunk_size) = lm_head_chunk_size {
            // CHUNKED LM HEAD PATH: Memory-efficient computation for large batches
            //
            // Instead of computing full [B, T, V] logits tensor (1.2GB for Qwen3),
            // we process the LM head in chunks to reduce peak memory.

            // Get hidden states (before LM head)
            let hidden_states = functional::qwen3_forward_hidden_states(
                &config_clone,
                &param_dict,
                &input_ids_clone,
            )?;

            // Extract completion hidden states: [B, prompt_len:, :]
            let completion_hidden = hidden_states.slice(
                &[0, prompt_len, 0],
                &[batch_size, total_seq_len, config_clone.hidden_size as i64],
            )?;

            // Get LM head weight
            let lm_head_weight = if config_clone.tie_word_embeddings {
                param_dict
                    .get("embedding.weight")
                    .ok_or_else(|| Error::from_reason("Missing embedding.weight"))?
            } else {
                param_dict
                    .get("lm_head.weight")
                    .ok_or_else(|| Error::from_reason("Missing lm_head.weight"))?
            };

            // Compute chunked log-softmax
            functional::chunked_lm_head_selective_logprobs(
                &completion_hidden,
                lm_head_weight,
                &padded_completions_clone,
                chunk_size,
                config_clone.tie_word_embeddings,
            )?
        } else {
            // FULL PATH: Standard computation (for small batches or testing)
            //
            // Recompute forward pass with parameters in native dtype
            let logits =
                functional::qwen3_forward_functional(&config_clone, &param_dict, &input_ids_clone)?;

            // Get logits for completion tokens only
            let completion_logits = logits.slice(
                &[0, prompt_len, 0],
                &[batch_size, total_seq_len, config_clone.vocab_size as i64],
            )?;

            // Memory-efficient log-softmax using logsumexp decomposition
            // Avoids materializing full [B, T, V] tensor - 99.99% memory savings
            //
            // Math: log_softmax(x_i) = x_i - logsumexp(x)
            // Instead of: [B,T,V] log_softmax -> [B,T,1] gather = 1.2GB intermediate
            // We compute: [B,T,1] logsumexp + [B,T,1] gather = 16KB intermediate
            efficient_selective_log_softmax(
                &completion_logits,
                &padded_completions_clone,
                loss_config_clone.vocab_chunk_size,
                None,
                None,
            )?
        };

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

    // === CRITICAL MEMORY OPTIMIZATION: Eval ALL tensors IMMEDIATELY, then cleanup ===
    //
    // MLX builds a computation graph during value_and_grad. This graph holds references
    // to ALL intermediate tensors. Calling eval() materializes the results and allows
    // the graph nodes to be released. We MUST:
    // Eval all outputs before extracting values
    loss_array.eval();

    for grad in &grad_arrays {
        grad.eval();
    }

    // CRITICAL: heavy_cleanup releases the autograd computation graph
    // synchronize_and_clear_cache only clears MLX cache, not the graph
    crate::array::heavy_cleanup();

    // Extract the loss value
    let loss_value = loss_array.item_at_float32(0)? as f64;

    // Step 5: Map gradients back to parameter names
    // Gradients are in the same dtype as parameters (bfloat16 for pretrained models).
    // The optimizer handles mixed precision - it can accumulate in float32 internally.
    let gradients = param_names
        .into_iter()
        .enumerate()
        .map(|(i, param_name)| {
            let grad = grad_arrays[i].clone();
            (param_name, grad)
        })
        .collect::<HashMap<_, _>>();

    Ok((loss_value, gradients))
}

/// Compute GRPO loss and gradients using chunked autograd
///
/// This function runs autograd separately for each chunk of sequences and accumulates
/// gradients. This allows MLX to release the computation graph after each chunk,
/// dramatically reducing peak memory usage.
///
/// # Memory Savings
/// For batch=4, groupSize=4 (16 sequences), chunk_size=4:
/// - Standard autograd: builds graph for all 16 sequences at once (~20GB)
/// - Chunked autograd: builds graph for 4 sequences at a time (~5GB peak)
///
/// # Arguments
/// * `model_config` - Model configuration
/// * `param_names` - Sorted list of parameter names
/// * `param_arrays` - Parameter arrays (references)
/// * `input_ids` - Full input IDs [batch, seq_len]
/// * `padded_completions` - Padded completion tokens [batch, completion_len]
/// * `padded_old_logprobs` - Old log probabilities [batch, completion_len]
/// * `advantages` - Pre-computed advantages [batch]
/// * `completion_masks` - Masks for valid completion tokens [batch, completion_len]
/// * `loss_config` - GRPO loss configuration
/// * `chunk_size` - Number of sequences per chunk
fn compute_loss_and_gradients_chunked_autograd(
    model_config: &Qwen3Config,
    param_names: &[String],
    param_arrays: &[&MxArray],
    input_ids: &MxArray,
    padded_completions: &MxArray,
    padded_old_logprobs: &MxArray,
    advantages: &MxArray,
    completion_masks: &MxArray,
    loss_config: &grpo_loss::GRPOLossConfig,
    chunk_size: i64,
) -> Result<(f64, HashMap<String, MxArray>)> {
    let batch_size = input_ids.shape_at(0)?;
    let total_seq_len = input_ids.shape_at(1)?;
    let completion_len = padded_completions.shape_at(1)?;

    // Track accumulated loss and gradients
    let mut total_loss = 0.0f64;
    let mut accumulated_gradients: Option<Vec<MxArray>> = None;

    // Process batch in chunks
    let mut start = 0i64;
    while start < batch_size {
        let end = (start + chunk_size).min(batch_size);
        let chunk_batch_size = end - start;

        // Slice data for this chunk
        let chunk_input_ids = input_ids.slice(&[start, 0], &[end, total_seq_len])?;
        let chunk_completions = padded_completions.slice(&[start, 0], &[end, completion_len])?;
        let chunk_old_logprobs = padded_old_logprobs.slice(&[start, 0], &[end, completion_len])?;
        let chunk_advantages = advantages.slice(&[start], &[end])?;
        let chunk_masks = completion_masks.slice(&[start, 0], &[end, completion_len])?;

        // Clone data for closure
        let param_names_clone = param_names.to_vec();
        let chunk_input_ids_clone = chunk_input_ids.clone();
        let chunk_completions_clone = chunk_completions.clone();
        let chunk_old_logprobs_clone = chunk_old_logprobs.clone();
        let chunk_advantages_clone = chunk_advantages.clone();
        let chunk_masks_clone = chunk_masks.clone();
        let config_clone = model_config.clone();
        let loss_config_clone = loss_config.clone();
        let lm_head_chunk_size = loss_config.lm_head_chunk_size;

        // Define loss function for this chunk
        let chunk_loss_fn = move |params: &[MxArray]| -> Result<MxArray> {
            let param_dict = param_manager::map_params_to_dict(params, &param_names_clone)?;

            let chunk_batch = chunk_input_ids_clone.shape_at(0)?;
            let chunk_total_seq = chunk_input_ids_clone.shape_at(1)?;
            let chunk_comp_len = chunk_completions_clone.shape_at(1)?;
            let chunk_prompt_len = chunk_total_seq - chunk_comp_len;

            // Compute logprobs for this chunk
            let completion_logprobs = if let Some(lm_chunk) = lm_head_chunk_size {
                // Chunked LM head path
                let hidden_states = functional::qwen3_forward_hidden_states(
                    &config_clone,
                    &param_dict,
                    &chunk_input_ids_clone,
                )?;

                let completion_hidden = hidden_states.slice(
                    &[0, chunk_prompt_len, 0],
                    &[
                        chunk_batch,
                        chunk_total_seq,
                        config_clone.hidden_size as i64,
                    ],
                )?;

                let lm_head_weight = if config_clone.tie_word_embeddings {
                    param_dict
                        .get("embedding.weight")
                        .ok_or_else(|| Error::from_reason("Missing embedding.weight"))?
                } else {
                    param_dict
                        .get("lm_head.weight")
                        .ok_or_else(|| Error::from_reason("Missing lm_head.weight"))?
                };

                functional::chunked_lm_head_selective_logprobs(
                    &completion_hidden,
                    lm_head_weight,
                    &chunk_completions_clone,
                    lm_chunk,
                    config_clone.tie_word_embeddings,
                )?
            } else {
                // Full path
                let logits = functional::qwen3_forward_functional(
                    &config_clone,
                    &param_dict,
                    &chunk_input_ids_clone,
                )?;

                let completion_logits = logits.slice(
                    &[0, chunk_prompt_len, 0],
                    &[chunk_batch, chunk_total_seq, config_clone.vocab_size as i64],
                )?;

                efficient_selective_log_softmax(
                    &completion_logits,
                    &chunk_completions_clone,
                    loss_config_clone.vocab_chunk_size,
                    None,
                    None,
                )?
            };

            // Clamp logprobs
            let clamped_logprobs = completion_logprobs.clip(Some(-20.0), Some(0.0))?;

            // Compute GRPO loss for this chunk
            // Note: We use the chunk's advantages which were pre-computed across the full batch
            let loss = grpo_loss::grpo_loss(
                &clamped_logprobs,
                &chunk_old_logprobs_clone,
                &chunk_advantages_clone,
                &chunk_masks_clone,
                loss_config_clone.clone(),
                None,
            )?;

            Ok(loss)
        };

        // Run autograd for this chunk
        let (chunk_loss_array, chunk_grad_arrays) =
            autograd::value_and_grad(param_arrays.to_vec(), chunk_loss_fn)?;

        // Eval immediately to materialize results
        chunk_loss_array.eval();
        for grad in &chunk_grad_arrays {
            grad.eval();
        }

        // Extract loss value
        let chunk_loss_value = chunk_loss_array.item_at_float32(0)? as f64;

        // Accumulate loss (weighted by chunk size for proper averaging)
        total_loss += chunk_loss_value * (chunk_batch_size as f64);

        // Accumulate gradients
        if let Some(ref mut acc_grads) = accumulated_gradients {
            // Add to existing gradients
            for (i, grad) in chunk_grad_arrays.into_iter().enumerate() {
                acc_grads[i] = acc_grads[i].add(&grad)?;
            }
        } else {
            // First chunk - initialize accumulated gradients
            accumulated_gradients = Some(chunk_grad_arrays);
        }

        // CRITICAL: Release computation graph for this chunk
        crate::array::heavy_cleanup();

        start = end;
    }

    // Average the loss
    let avg_loss = total_loss / (batch_size as f64);

    // Get accumulated gradients (already summed, no need to average for GRPO)
    let grad_arrays =
        accumulated_gradients.ok_or_else(|| Error::from_reason("No gradients computed"))?;

    // Map gradients back to parameter names
    let gradients = param_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), grad_arrays[i].clone()))
        .collect::<HashMap<_, _>>();

    Ok((avg_loss, gradients))
}
