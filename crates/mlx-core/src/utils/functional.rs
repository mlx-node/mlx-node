/**
 * Functional Transformer Components
 *
 * Stateless, functional implementations of all transformer building blocks.
 * These functions take parameters explicitly as arguments rather than accessing
 * them from self, enabling automatic differentiation through the computation graph.
 *
 * This is the foundation for autograd-based training, allowing MLX to trace
 * gradients through the entire forward pass from parameters to loss.
 */
use crate::array::{MxArray, scaled_dot_product_attention};
use crate::models::qwen3::Qwen3Config;
use crate::nn::Activations;
use napi::bindgen_prelude::*;
use std::collections::HashMap;

// ============================================
// Basic Layer Functions
// ============================================

/// Functional embedding lookup
///
/// # Arguments
/// * `weight` - Embedding weight matrix, shape: [vocab_size, hidden_size]
/// * `input_ids` - Token IDs, shape: [batch_size, seq_len]
///
/// # Returns
/// * Embedding vectors, shape: [batch_size, seq_len, hidden_size]
pub fn embedding_functional(weight: &MxArray, input_ids: &MxArray) -> Result<MxArray> {
    // Embedding is just a lookup: weight[input_ids] along axis 0 (vocab dimension)
    weight.take(input_ids, 0)
}

/// Functional linear layer (matrix multiplication with optional bias)
///
/// # Arguments
/// * `input` - Input tensor, shape: [..., in_features]
/// * `weight` - Weight matrix, shape: [out_features, in_features]
/// * `bias` - Optional bias vector, shape: [out_features]
///
/// # Returns
/// * Output tensor, shape: [..., out_features]
pub fn linear_functional(
    input: &MxArray,
    weight: &MxArray,
    bias: Option<&MxArray>,
) -> Result<MxArray> {
    // Linear: output = input @ weight^T + bias
    let weight_t = weight.transpose(None)?;
    let output = input.matmul(&weight_t)?;

    if let Some(b) = bias {
        output.add(b)
    } else {
        Ok(output)
    }
}

/// Functional RMS normalization
///
/// Implements RMSNorm following the transformers pattern:
/// - Upcast to float32 for variance computation (numerical stability)
/// - Use rsqrt instead of 1/sqrt (more efficient)
/// - No clamping (TRL/transformers don't clamp)
///
/// # Arguments
/// * `input` - Input tensor, shape: [...]
/// * `weight` - Scale parameter, shape: [normalized_shape]
/// * `eps` - Epsilon for numerical stability
///
/// # Returns
/// * Normalized tensor, shape: [...]
pub fn rms_norm_functional(input: &MxArray, weight: &MxArray, eps: f64) -> Result<MxArray> {
    // RMSNorm: output = input * rsqrt(mean(input^2) + eps) * weight
    // Following transformers pattern: compute variance in float32 for numerical stability

    // Store original dtype for later
    let original_dtype = input.dtype()?;

    // Upcast to float32 for variance computation (critical for numerical stability)
    let input_f32 = input.astype(crate::array::DType::Float32)?;

    // Compute variance: mean(x^2)
    let squared = input_f32.square()?;
    let mean_squared = squared.mean(Some(&[-1]), Some(true))?;

    // Add epsilon and compute 1/sqrt(var + eps)
    let variance_plus_eps = mean_squared.add_scalar(eps)?;
    let sqrt_var = variance_plus_eps.sqrt()?;

    // Normalize: input / sqrt(var + eps)
    let normalized = input_f32.div(&sqrt_var)?;

    // Cast back to original dtype
    let normalized_orig = normalized.astype(original_dtype)?;

    // Scale by weight (weight stays in original dtype)
    normalized_orig.mul(weight)
}

/// Functional MLP with SwiGLU activation
///
/// Architecture: down_proj(silu(gate_proj(x)) * up_proj(x))
/// No clamping - following TRL/transformers pattern
///
/// # Arguments
/// * `input` - Input tensor, shape: [batch, seq_len, hidden_size]
/// * `gate_weight` - Gate projection weight, shape: [intermediate_size, hidden_size]
/// * `up_weight` - Up projection weight, shape: [intermediate_size, hidden_size]
/// * `down_weight` - Down projection weight, shape: [hidden_size, intermediate_size]
///
/// # Returns
/// * Output tensor, shape: [batch, seq_len, hidden_size]
pub fn mlp_functional(
    input: &MxArray,
    gate_weight: &MxArray,
    up_weight: &MxArray,
    down_weight: &MxArray,
) -> Result<MxArray> {
    // Gate projection
    let gate = linear_functional(input, gate_weight, None)?;

    // Up projection
    let up = linear_functional(input, up_weight, None)?;

    // Apply SiLU to gate - uses autograd-compatible version that preserves computation graph
    let gate_act = Activations::silu_for_autograd(&gate)?;

    // Element-wise multiplication
    let gated = gate_act.mul(&up)?;

    // Down projection (no clamping - TRL/transformers don't clamp)
    linear_functional(&gated, down_weight, None)
}

/// Parameters for a single attention layer (uses references to avoid cloning)
pub struct AttentionParams<'a> {
    pub q_proj_weight: &'a MxArray,
    pub k_proj_weight: &'a MxArray,
    pub v_proj_weight: &'a MxArray,
    pub o_proj_weight: &'a MxArray,
    pub q_norm_weight: Option<&'a MxArray>,
    pub k_norm_weight: Option<&'a MxArray>,
}

/// Functional multi-head attention
///
/// Supports:
/// - Grouped Query Attention (GQA)
/// - Optional QK normalization
/// - RoPE (Rotary Position Embeddings)
///
/// # Arguments
/// * `input` - Input tensor, shape: [batch, seq_len, hidden_size]
/// * `params` - Attention parameters (Q/K/V/O projection weights, optional norms)
/// * `num_heads` - Number of query heads
/// * `num_kv_heads` - Number of key/value heads (for GQA)
/// * `head_dim` - Dimension per head
/// * `rope_theta` - RoPE base frequency
/// * `use_qk_norm` - Whether to use QK normalization
/// * `qk_norm_eps` - Epsilon for QK normalization
/// * `offset` - Position offset for RoPE (for KV caching)
///
/// # Returns
/// * Output tensor, shape: [batch, seq_len, hidden_size]
pub fn attention_functional(
    input: &MxArray,
    params: &AttentionParams<'_>,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rope_theta: f64,
    use_qk_norm: bool,
    qk_norm_eps: f64,
    offset: i32,
) -> Result<MxArray> {
    // Use shape_at() to avoid allocating full shape vector
    let batch = input.shape_at(0)?;
    let seq_len = input.shape_at(1)?;

    // 1. Project to Q, K, V
    let queries = linear_functional(input, params.q_proj_weight, None)?;
    let keys = linear_functional(input, params.k_proj_weight, None)?;
    let values = linear_functional(input, params.v_proj_weight, None)?;

    // 2. Reshape to multi-head format: [B, L, n_heads, head_dim]
    let queries = queries.reshape(&[batch, seq_len, num_heads as i64, head_dim as i64])?;
    let keys = keys.reshape(&[batch, seq_len, num_kv_heads as i64, head_dim as i64])?;
    let values = values.reshape(&[batch, seq_len, num_kv_heads as i64, head_dim as i64])?;

    // 3. Apply QK normalization (if enabled)
    let queries = if use_qk_norm {
        if let Some(q_norm_weight) = params.q_norm_weight {
            rms_norm_functional(&queries, q_norm_weight, qk_norm_eps)?
        } else {
            queries
        }
    } else {
        queries
    };

    let keys = if use_qk_norm {
        if let Some(k_norm_weight) = params.k_norm_weight {
            rms_norm_functional(&keys, k_norm_weight, qk_norm_eps)?
        } else {
            keys
        }
    } else {
        keys
    };

    // 4. Transpose to [B, n_heads, L, head_dim] for attention
    let queries = queries.transpose(Some(&[0, 2, 1, 3]))?;
    let keys = keys.transpose(Some(&[0, 2, 1, 3]))?;
    let values = values.transpose(Some(&[0, 2, 1, 3]))?;

    // 5. Apply RoPE
    let queries = apply_rope(&queries, head_dim, rope_theta, offset)?;
    let keys = apply_rope(&keys, head_dim, rope_theta, offset)?;

    // 6. Scaled dot-product attention with causal masking
    // CRITICAL: Use "causal" mode instead of explicit mask to enable Flash Attention VJP
    // Explicit masks force materialization of full attention matrices (~100GB for batch=4, seq=2048)
    // "causal" mode uses optimized STEEL kernel with logsumexp output (~5GB)
    let scale = 1.0 / (head_dim as f64).sqrt();
    let output = if seq_len > 1 {
        // Use causal mode for autoregressive training
        use crate::array::attention::scaled_dot_product_attention_causal;
        scaled_dot_product_attention_causal(&queries, &keys, &values, scale)?
    } else {
        // Single token doesn't need masking
        scaled_dot_product_attention(&queries, &keys, &values, scale, None)?
    };

    // 8. Transpose back to [B, L, n_heads, head_dim]
    let output = output.transpose(Some(&[0, 2, 1, 3]))?;

    // 9. Reshape to [B, L, n_heads * head_dim]
    let output = output.reshape(&[batch, seq_len, (num_heads * head_dim) as i64])?;

    // 10. Output projection (no clamping - TRL/transformers don't clamp)
    linear_functional(&output, params.o_proj_weight, None)
}

/// Apply RoPE (Rotary Position Embeddings) to query or key tensors
///
/// # Arguments
/// * `x` - Input tensor, shape: [batch, n_heads, seq_len, head_dim]
/// * `head_dim` - Dimension per head
/// * `theta` - Base frequency
/// * `offset` - Position offset (for KV caching)
///
/// # Returns
/// * Tensor with RoPE applied, shape: [batch, n_heads, seq_len, head_dim]
fn apply_rope(x: &MxArray, head_dim: u32, theta: f64, offset: i32) -> Result<MxArray> {
    // This is a simplified version - in production, use the RoPE class from nn.rs
    // For now, use the existing RoPE::forward method
    use crate::nn::RoPE;
    let rope = RoPE::new(head_dim as i32, Some(false), Some(theta), Some(1.0))?;
    rope.forward(x, Some(offset))
}

/// Parameters for a single transformer block (uses references to avoid cloning)
pub struct TransformerBlockParams<'a> {
    pub attn_params: AttentionParams<'a>,
    pub mlp_gate_weight: &'a MxArray,
    pub mlp_up_weight: &'a MxArray,
    pub mlp_down_weight: &'a MxArray,
    pub input_norm_weight: &'a MxArray,
    pub post_attn_norm_weight: &'a MxArray,
}

/// Functional transformer block
///
/// Architecture (pre-norm with residual connections):
/// 1. x = x + self_attn(norm(x))
/// 2. x = x + mlp(norm(x))
///
/// # Arguments
/// * `input` - Input tensor, shape: [batch, seq_len, hidden_size]
/// * `params` - Transformer block parameters
/// * `num_heads` - Number of attention heads
/// * `num_kv_heads` - Number of key/value heads
/// * `head_dim` - Dimension per head
/// * `rope_theta` - RoPE base frequency
/// * `use_qk_norm` - Whether to use QK normalization
/// * `rms_norm_eps` - Epsilon for RMS normalization
/// * `offset` - Position offset for RoPE
///
/// # Returns
/// * Output tensor, shape: [batch, seq_len, hidden_size]
pub fn transformer_block_functional(
    input: &MxArray,
    params: &TransformerBlockParams<'_>,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rope_theta: f64,
    use_qk_norm: bool,
    rms_norm_eps: f64,
    offset: i32,
) -> Result<MxArray> {
    // 1. Self-attention with pre-norm and residual
    let normed = rms_norm_functional(input, params.input_norm_weight, rms_norm_eps)?;
    let attn_out = attention_functional(
        &normed,
        &params.attn_params,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_theta,
        use_qk_norm,
        rms_norm_eps, // Use same eps for QK norm
        offset,
    )?;
    let h = input.add(&attn_out)?;

    // 2. MLP with pre-norm and residual
    let normed = rms_norm_functional(&h, params.post_attn_norm_weight, rms_norm_eps)?;
    let mlp_out = mlp_functional(
        &normed,
        params.mlp_gate_weight,
        params.mlp_up_weight,
        params.mlp_down_weight,
    )?;
    h.add(&mlp_out)
}

/// Functional Qwen3 forward pass returning hidden states (before LM head)
///
/// This function runs the transformer layers and final norm but stops before the LM head.
/// Used for chunked training where we want to process the LM head in smaller batches
/// to avoid memory issues with large vocabulary sizes.
///
/// # Arguments
/// * `config` - Model configuration
/// * `params` - All model parameters as a dictionary (name -> MxArray)
/// * `input_ids` - Token IDs, shape: [batch_size, seq_len]
///
/// # Returns
/// * Hidden states, shape: [batch_size, seq_len, hidden_size]
pub fn qwen3_forward_hidden_states(
    config: &Qwen3Config,
    params: &HashMap<String, MxArray>,
    input_ids: &MxArray,
) -> Result<MxArray> {
    // 1. Embedding lookup
    let embedding_weight = params
        .get("embedding.weight")
        .ok_or_else(|| Error::from_reason("Missing embedding.weight"))?;

    let mut hidden_states = embedding_functional(embedding_weight, input_ids)?;

    // 2. Transformer layers
    let num_layers = config.num_layers as usize;
    for layer_idx in 0..num_layers {
        // Get layer parameters
        let layer_params = get_layer_params(params, layer_idx, config)?;

        // Forward through transformer block
        hidden_states = transformer_block_functional(
            &hidden_states,
            &layer_params,
            config.num_heads as u32,
            config.num_kv_heads as u32,
            config.head_dim as u32, // Use explicit head_dim from config (critical for Qwen3!)
            config.rope_theta,
            config.use_qk_norm,
            config.rms_norm_eps,
            0, // offset (no KV caching for training)
        )?;
    }

    // 3. Final layer norm
    let final_norm_weight = params
        .get("final_norm.weight")
        .ok_or_else(|| Error::from_reason("Missing final_norm.weight"))?;

    rms_norm_functional(&hidden_states, final_norm_weight, config.rms_norm_eps)
}

/// Functional Qwen3 forward pass returning hidden states with batch chunking (memory optimization)
///
/// This function processes the transformer layers in batch chunks to reduce peak memory
/// from O(batch × heads × seq²) for attention. Instead of processing all 16 sequences
/// (batch=4 × group=4) at once, it processes 4 sequences at a time.
///
/// # Memory Savings
/// For batch=4, groupSize=4, seq=1024, hidden=1024, heads=16:
/// - Full batch: 16 × 16 × 1024² × 4 bytes ≈ 1 GB per attention layer
/// - Chunked (chunk_size=4): 4 × 16 × 1024² × 4 bytes ≈ 256 MB per attention layer
/// - 24 layers total: ~24 GB → ~6 GB peak (75% reduction)
///
/// # Arguments
/// * `config` - Model configuration
/// * `params` - All model parameters as a dictionary (name -> MxArray)
/// * `input_ids` - Token IDs, shape: [batch_size, seq_len]
/// * `chunk_size` - Number of sequences to process at once (default: 4 if <= 0)
///
/// # Returns
/// * Hidden states, shape: [batch_size, seq_len, hidden_size]
pub fn qwen3_forward_hidden_states_chunked(
    config: &Qwen3Config,
    params: &HashMap<String, MxArray>,
    input_ids: &MxArray,
    chunk_size: i64,
) -> Result<MxArray> {
    let batch_size = input_ids.shape_at(0)?;
    let seq_len = input_ids.shape_at(1)?;

    // Default chunk_size to 4 if invalid
    let chunk_size = if chunk_size <= 0 { 4 } else { chunk_size };

    // Skip chunking for small batches (no memory benefit)
    if batch_size <= chunk_size {
        return qwen3_forward_hidden_states(config, params, input_ids);
    }

    // Get embedding and final norm weights
    let embedding_weight = params
        .get("embedding.weight")
        .ok_or_else(|| Error::from_reason("Missing embedding.weight"))?;
    let final_norm_weight = params
        .get("final_norm.weight")
        .ok_or_else(|| Error::from_reason("Missing final_norm.weight"))?;

    // Pre-extract layer params (avoid repeated HashMap lookups in chunks)
    let num_layers = config.num_layers as usize;
    let layer_params: Vec<TransformerBlockParams<'_>> = (0..num_layers)
        .map(|i| get_layer_params(params, i, config))
        .collect::<Result<Vec<_>>>()?;

    // Process batch in chunks
    let mut chunk_results: Vec<MxArray> = Vec::new();
    let mut start = 0i64;

    while start < batch_size {
        let end = (start + chunk_size).min(batch_size);

        // Slice input_ids for this chunk: [start:end, :]
        let chunk_input = input_ids.slice(&[start, 0], &[end, seq_len])?;

        // 1. Embedding lookup
        let mut hidden = embedding_functional(embedding_weight, &chunk_input)?;

        // 2. Process all transformer layers (this is where memory is saved)
        for layer_param in layer_params.iter().take(num_layers) {
            hidden = transformer_block_functional(
                &hidden,
                layer_param,
                config.num_heads as u32,
                config.num_kv_heads as u32,
                config.head_dim as u32,
                config.rope_theta,
                config.use_qk_norm,
                config.rms_norm_eps,
                0, // offset (no KV caching for training)
            )?;
        }

        // 3. Final layer norm
        hidden = rms_norm_functional(&hidden, final_norm_weight, config.rms_norm_eps)?;

        // Note: We do NOT call eval() here during autograd.
        // The computation graph must be preserved for backpropagation.
        // Memory savings come from processing smaller tensors, not from eval().

        chunk_results.push(hidden);
        start = end;
    }

    // Concatenate all chunks: Vec<[chunk, T, H]> -> [B, T, H]
    if chunk_results.len() == 1 {
        Ok(chunk_results.into_iter().next().unwrap())
    } else {
        let refs: Vec<&MxArray> = chunk_results.iter().collect();
        MxArray::concatenate_many(refs, Some(0))
    }
}

/// Functional Qwen3 forward pass
///
/// This is the core function for autograd - it traces the entire forward pass
/// from parameters to logits, allowing MLX to compute gradients automatically.
///
/// # Arguments
/// * `config` - Model configuration
/// * `params` - All model parameters as a dictionary (name -> MxArray)
/// * `input_ids` - Token IDs, shape: [batch_size, seq_len]
///
/// # Returns
/// * Logits, shape: [batch_size, seq_len, vocab_size]
pub fn qwen3_forward_functional(
    config: &Qwen3Config,
    params: &HashMap<String, MxArray>,
    input_ids: &MxArray,
) -> Result<MxArray> {
    // 1. Embedding lookup
    let embedding_weight = params
        .get("embedding.weight")
        .ok_or_else(|| Error::from_reason("Missing embedding.weight"))?;

    let mut hidden_states = embedding_functional(embedding_weight, input_ids)?;

    // 2. Transformer layers
    let num_layers = config.num_layers as usize;
    for layer_idx in 0..num_layers {
        // Get layer parameters
        let layer_params = get_layer_params(params, layer_idx, config)?;

        // Forward through transformer block
        hidden_states = transformer_block_functional(
            &hidden_states,
            &layer_params,
            config.num_heads as u32,
            config.num_kv_heads as u32,
            config.head_dim as u32, // Use explicit head_dim from config (critical for Qwen3!)
            config.rope_theta,
            config.use_qk_norm,
            config.rms_norm_eps,
            0, // offset (no KV caching for training)
        )?;
    }

    // 3. Final layer norm
    let final_norm_weight = params
        .get("final_norm.weight")
        .ok_or_else(|| Error::from_reason("Missing final_norm.weight"))?;

    hidden_states = rms_norm_functional(&hidden_states, final_norm_weight, config.rms_norm_eps)?;

    // 4. LM head (handle tied embeddings like stateful model)
    let logits = if config.tie_word_embeddings {
        // When tie_word_embeddings=true, use embedding.weight transposed
        // This matches model.rs: hidden_states.matmul(&embedding_weight.transpose([1, 0]))
        let embedding_weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
        hidden_states.matmul(&embedding_weight_t)?
    } else {
        // When tie_word_embeddings=false, use separate lm_head weight
        let lm_head_weight = params
            .get("lm_head.weight")
            .ok_or_else(|| Error::from_reason("Missing lm_head.weight"))?;
        linear_functional(&hidden_states, lm_head_weight, None)?
    };

    // Clamp logits to prevent overflow in cross-entropy loss computation
    // TRL uses dtype-aware log_softmax (logsumexp unstable in BF16), but clamping
    // provides an additional safety margin. Range [-100, 100] is safe for both
    // FP32 (exp(88.7) = max) and BF16 precision.
    logits.clip(Some(-100.0), Some(100.0))
}

/// Chunked LM head with selective log-softmax for memory efficiency.
///
/// Reduces peak memory from ~1.2GB to ~150MB for Qwen3 (vocab=151936) by processing
/// the batch dimension in chunks. This is critical for GRPO training which
/// needs multiple samples per prompt.
///
/// Memory savings: For batch=8, seq=256, vocab=151936:
/// - Full computation: [8, 256, 151936] = 1.2 GB intermediate
/// - Chunked (chunk_size=2): [2, 256, 151936] = 300 MB peak
///
/// # Arguments
/// * `hidden_states` - Hidden states after final norm, shape: [B, T, H]
/// * `lm_head_weight` - LM head weight (embedding.weight or lm_head.weight), shape: [V, H]
/// * `target_ids` - Token IDs to extract probabilities for, shape: [B, T]
/// * `chunk_size` - Batch chunk size (default: 2)
/// * `tie_word_embeddings` - Whether weight needs transpose (true for tied embeddings)
///
/// # Returns
/// * Log probabilities for selected tokens, shape: [B, T]
pub fn chunked_lm_head_selective_logprobs(
    hidden_states: &MxArray,
    lm_head_weight: &MxArray,
    target_ids: &MxArray,
    chunk_size: i64,
    tie_word_embeddings: bool,
) -> Result<MxArray> {
    use crate::nn::efficient_selective_log_softmax;

    let batch_size = hidden_states.shape_at(0)?;
    let seq_len = hidden_states.shape_at(1)?;

    // Validate chunk_size
    let chunk_size = if chunk_size <= 0 {
        2 // Default to 2 if invalid
    } else {
        chunk_size
    };

    // If batch_size is small enough, skip chunking
    if batch_size <= chunk_size {
        // Full computation without chunking
        let logits = if tie_word_embeddings {
            let weight_t = lm_head_weight.transpose(Some(&[1, 0]))?;
            hidden_states.matmul(&weight_t)?
        } else {
            linear_functional(hidden_states, lm_head_weight, None)?
        };
        let logits_clamped = logits.clip(Some(-100.0), Some(100.0))?;
        return efficient_selective_log_softmax(&logits_clamped, target_ids, None, None, None);
    }

    // Transpose weight once (outside the loop)
    let weight_t = if tie_word_embeddings {
        lm_head_weight.transpose(Some(&[1, 0]))?
    } else {
        // For non-tied weights, shape is [out_features, in_features]
        // linear_functional expects weight in this format and transposes internally
        // So we need to pre-transpose to avoid repeated transposition in the loop
        lm_head_weight.transpose(Some(&[1, 0]))?
    };

    // Collect chunk results
    let mut chunk_logprobs: Vec<MxArray> = Vec::new();

    // Process in chunks
    let mut start = 0i64;
    while start < batch_size {
        let end = (start + chunk_size).min(batch_size);

        // Slice hidden_states: [start:end, :, :] -> [chunk, T, H]
        let chunk_hidden =
            hidden_states.slice(&[start, 0, 0], &[end, seq_len, hidden_states.shape_at(2)?])?;

        // Slice target_ids: [start:end, :] -> [chunk, T]
        let chunk_targets = target_ids.slice(&[start, 0], &[end, seq_len])?;

        // Compute chunk logits: [chunk, T, H] @ [H, V] -> [chunk, T, V]
        let chunk_logits = chunk_hidden.matmul(&weight_t)?;

        // Clamp logits to prevent overflow
        let chunk_logits_clamped = chunk_logits.clip(Some(-100.0), Some(100.0))?;

        // Compute selective log-softmax: [chunk, T, V] + [chunk, T] -> [chunk, T]
        let chunk_result = efficient_selective_log_softmax(
            &chunk_logits_clamped,
            &chunk_targets,
            None,
            None,
            None,
        )?;

        // Note: We intentionally do NOT call eval() here during autograd.
        // While eval() would materialize values, MLX autograd still needs to keep
        // the computation graph for backpropagation. Calling eval() doesn't help
        // reduce memory during autograd - it just adds unnecessary synchronization.
        //
        // The real memory savings come from:
        // 1. Computing logits in smaller chunks (reducing peak allocation)
        // 2. Immediately reducing [chunk, T, V] to [chunk, T] via selective_log_softmax

        chunk_logprobs.push(chunk_result);

        start = end;
    }

    // Concatenate all chunks: Vec<[chunk, T]> -> [B, T]
    if chunk_logprobs.len() == 1 {
        Ok(chunk_logprobs.into_iter().next().unwrap())
    } else {
        let refs: Vec<&MxArray> = chunk_logprobs.iter().collect();
        MxArray::concatenate_many(refs, Some(0))
    }
}

/// Extract parameters for a specific transformer layer
///
/// Returns references to the parameters in the HashMap, avoiding clones.
/// This reduces memory allocation from 576 Arc clones to 0 per forward pass
/// (12 params × 48 layers = 576 clones previously).
///
/// # Arguments
/// * `params` - All model parameters
/// * `layer_idx` - Layer index (0-based)
/// * `config` - Model configuration (for QK norm check)
///
/// # Returns
/// * TransformerBlockParams with references to the specified layer's parameters
fn get_layer_params<'a>(
    params: &'a HashMap<String, MxArray>,
    layer_idx: usize,
    config: &Qwen3Config,
) -> Result<TransformerBlockParams<'a>> {
    let prefix = format!("layers.{}", layer_idx);

    // Attention parameters - get references, no cloning
    let q_proj_weight = params
        .get(&format!("{}.self_attn.q_proj.weight", prefix))
        .ok_or_else(|| Error::from_reason(format!("Missing {}.self_attn.q_proj.weight", prefix)))?;

    let k_proj_weight = params
        .get(&format!("{}.self_attn.k_proj.weight", prefix))
        .ok_or_else(|| Error::from_reason(format!("Missing {}.self_attn.k_proj.weight", prefix)))?;

    let v_proj_weight = params
        .get(&format!("{}.self_attn.v_proj.weight", prefix))
        .ok_or_else(|| Error::from_reason(format!("Missing {}.self_attn.v_proj.weight", prefix)))?;

    let o_proj_weight = params
        .get(&format!("{}.self_attn.o_proj.weight", prefix))
        .ok_or_else(|| Error::from_reason(format!("Missing {}.self_attn.o_proj.weight", prefix)))?;

    // Optional QK norm weights - get references if present
    let q_norm_weight = if config.use_qk_norm {
        params.get(&format!("{}.self_attn.q_norm.weight", prefix))
    } else {
        None
    };

    let k_norm_weight = if config.use_qk_norm {
        params.get(&format!("{}.self_attn.k_norm.weight", prefix))
    } else {
        None
    };

    let attn_params = AttentionParams {
        q_proj_weight,
        k_proj_weight,
        v_proj_weight,
        o_proj_weight,
        q_norm_weight,
        k_norm_weight,
    };

    // MLP parameters - get references, no cloning
    let mlp_gate_weight = params
        .get(&format!("{}.mlp.gate_proj.weight", prefix))
        .ok_or_else(|| Error::from_reason(format!("Missing {}.mlp.gate_proj.weight", prefix)))?;

    let mlp_up_weight = params
        .get(&format!("{}.mlp.up_proj.weight", prefix))
        .ok_or_else(|| Error::from_reason(format!("Missing {}.mlp.up_proj.weight", prefix)))?;

    let mlp_down_weight = params
        .get(&format!("{}.mlp.down_proj.weight", prefix))
        .ok_or_else(|| Error::from_reason(format!("Missing {}.mlp.down_proj.weight", prefix)))?;

    // Normalization parameters - get references, no cloning
    let input_norm_weight = params
        .get(&format!("{}.input_layernorm.weight", prefix))
        .ok_or_else(|| Error::from_reason(format!("Missing {}.input_layernorm.weight", prefix)))?;

    let post_attn_norm_weight = params
        .get(&format!("{}.post_attention_layernorm.weight", prefix))
        .ok_or_else(|| {
            Error::from_reason(format!(
                "Missing {}.post_attention_layernorm.weight",
                prefix
            ))
        })?;

    Ok(TransformerBlockParams {
        attn_params,
        mlp_gate_weight,
        mlp_up_weight,
        mlp_down_weight,
        input_norm_weight,
        post_attn_norm_weight,
    })
}

#[cfg(test)]
mod forward_pass_equivalence_tests {
    use super::*;
    use crate::models::qwen3::{Qwen3Config, Qwen3Model};

    /// Helper to check if two arrays are close within tolerance
    fn arrays_close(a: &[f32], b: &[f32], atol: f32, rtol: f32) -> bool {
        if a.len() != b.len() {
            return false;
        }
        for (x, y) in a.iter().zip(b.iter()) {
            let tol = atol + rtol * y.abs();
            if (x - y).abs() > tol {
                return false;
            }
        }
        true
    }

    /// Create a tiny model config for fast testing
    fn tiny_config() -> Qwen3Config {
        Qwen3Config {
            vocab_size: 100,
            hidden_size: 32,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            max_position_embeddings: 128,
            head_dim: 8, // 32 / 4 = 8
            use_qk_norm: true,
            tie_word_embeddings: false,
            pad_token_id: 0,
            eos_token_id: 1,
            bos_token_id: 0,
            // Paged attention options (disabled for test)
            use_paged_attention: None,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            use_fp8_cache: None,
        }
    }

    #[test]
    fn test_stateful_and_functional_forward_produce_identical_outputs() {
        // Create model with tiny config
        let config = tiny_config();
        let model = Qwen3Model::new(config.clone()).unwrap();

        // Get parameters as HashMap
        let params = model.get_parameters();

        // Create input tokens [batch=1, seq_len=5]
        let input_ids = MxArray::from_int32(&[0, 1, 2, 3, 4], &[1, 5]).unwrap();

        // Run stateful forward pass
        let stateful_output = model.forward(&input_ids).unwrap();
        stateful_output.eval();

        // Run functional forward pass
        let functional_output = qwen3_forward_functional(&config, &params, &input_ids).unwrap();
        functional_output.eval();

        // Compare shapes (convert BigInt64Array to Vec<i64> for comparison)
        let stateful_shape: Vec<i64> = stateful_output.shape().unwrap().to_vec();
        let functional_shape: Vec<i64> = functional_output.shape().unwrap().to_vec();
        assert_eq!(stateful_shape, functional_shape, "Shapes must match");

        // Compare values with tolerance
        let stateful_data = stateful_output.to_float32().unwrap();
        let functional_data = functional_output.to_float32().unwrap();

        assert!(
            arrays_close(&stateful_data, &functional_data, 1e-4, 1e-4),
            "Forward pass outputs must be close. Max diff: {}",
            stateful_data
                .iter()
                .zip(functional_data.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        );
    }

    #[test]
    fn test_equivalence_across_batch_sizes() {
        let config = tiny_config();
        let model = Qwen3Model::new(config.clone()).unwrap();
        let params = model.get_parameters();

        for batch_size in [1, 2, 4] {
            // Create input tokens [batch, seq_len=4]
            let seq_len = 4;
            let total = batch_size * seq_len;
            let input_data: Vec<i32> = (0..total).map(|i| i % 50).collect();
            let input_ids =
                MxArray::from_int32(&input_data, &[batch_size as i64, seq_len as i64]).unwrap();

            // Run both forward passes
            let stateful_output = model.forward(&input_ids).unwrap();
            stateful_output.eval();

            let functional_output = qwen3_forward_functional(&config, &params, &input_ids).unwrap();
            functional_output.eval();

            // Compare
            let stateful_data = stateful_output.to_float32().unwrap();
            let functional_data = functional_output.to_float32().unwrap();

            assert!(
                arrays_close(&stateful_data, &functional_data, 1e-4, 1e-4),
                "Batch size {} failed. Max diff: {}",
                batch_size,
                stateful_data
                    .iter()
                    .zip(functional_data.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max)
            );
        }
    }

    #[test]
    fn test_equivalence_across_sequence_lengths() {
        let config = tiny_config();
        let model = Qwen3Model::new(config.clone()).unwrap();
        let params = model.get_parameters();

        for seq_len in [1, 4, 16] {
            // Create input tokens [batch=1, seq_len]
            let input_data: Vec<i32> = (0..seq_len).map(|i| i % 50).collect();
            let input_ids = MxArray::from_int32(&input_data, &[1, seq_len as i64]).unwrap();

            // Run both forward passes
            let stateful_output = model.forward(&input_ids).unwrap();
            stateful_output.eval();

            let functional_output = qwen3_forward_functional(&config, &params, &input_ids).unwrap();
            functional_output.eval();

            // Compare
            let stateful_data = stateful_output.to_float32().unwrap();
            let functional_data = functional_output.to_float32().unwrap();

            assert!(
                arrays_close(&stateful_data, &functional_data, 1e-4, 1e-4),
                "Seq len {} failed. Max diff: {}",
                seq_len,
                stateful_data
                    .iter()
                    .zip(functional_data.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max)
            );
        }
    }

    #[test]
    fn test_equivalence_with_tied_embeddings() {
        // Test with tied word embeddings enabled
        let mut config = tiny_config();
        config.tie_word_embeddings = true;

        let model = Qwen3Model::new(config.clone()).unwrap();
        let params = model.get_parameters();

        // Create input
        let input_ids = MxArray::from_int32(&[0, 1, 2, 3], &[1, 4]).unwrap();

        // Run both forward passes
        let stateful_output = model.forward(&input_ids).unwrap();
        stateful_output.eval();

        let functional_output = qwen3_forward_functional(&config, &params, &input_ids).unwrap();
        functional_output.eval();

        // Compare
        let stateful_data = stateful_output.to_float32().unwrap();
        let functional_data = functional_output.to_float32().unwrap();

        assert!(
            arrays_close(&stateful_data, &functional_data, 1e-4, 1e-4),
            "Tied embeddings test failed. Max diff: {}",
            stateful_data
                .iter()
                .zip(functional_data.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        );
    }
}

#[cfg(test)]
mod chunked_lm_head_tests {
    use super::*;
    use crate::nn::efficient_selective_log_softmax;

    /// Helper to check if two arrays are close within tolerance
    fn arrays_close(a: &[f32], b: &[f32], atol: f32, rtol: f32) -> bool {
        if a.len() != b.len() {
            return false;
        }
        for (x, y) in a.iter().zip(b.iter()) {
            let tol = atol + rtol * y.abs();
            if (x - y).abs() > tol {
                return false;
            }
        }
        true
    }

    /// Reference implementation: full (non-chunked) LM head + selective log-softmax
    fn full_lm_head_selective_logprobs(
        hidden_states: &MxArray,
        lm_head_weight: &MxArray,
        target_ids: &MxArray,
        tie_word_embeddings: bool,
    ) -> Result<MxArray> {
        // Full computation without chunking
        let logits = if tie_word_embeddings {
            let weight_t = lm_head_weight.transpose(Some(&[1, 0]))?;
            hidden_states.matmul(&weight_t)?
        } else {
            linear_functional(hidden_states, lm_head_weight, None)?
        };
        let logits_clamped = logits.clip(Some(-100.0), Some(100.0))?;
        efficient_selective_log_softmax(&logits_clamped, target_ids, None, None, None)
    }

    #[test]
    fn test_chunked_matches_full_small() {
        // Small test case for debugging
        let batch = 4;
        let seq = 8;
        let hidden = 32;
        let vocab = 100;

        // Create random hidden states and weight
        let hidden_states = MxArray::random_normal(&[batch, seq, hidden], 0.0, 1.0, None).unwrap();
        let weight = MxArray::random_normal(&[vocab, hidden], 0.0, 0.1, None).unwrap();
        let targets = MxArray::randint(&[batch, seq], 0, vocab as i32).unwrap();

        // Full computation
        let full_result =
            full_lm_head_selective_logprobs(&hidden_states, &weight, &targets, true).unwrap();
        full_result.eval();

        // Chunked computation with chunk_size=2
        let chunked_result =
            chunked_lm_head_selective_logprobs(&hidden_states, &weight, &targets, 2, true).unwrap();
        chunked_result.eval();

        // Compare
        let full_data = full_result.to_float32().unwrap();
        let chunked_data = chunked_result.to_float32().unwrap();

        assert_eq!(full_data.len(), chunked_data.len());
        assert!(
            arrays_close(&full_data, &chunked_data, 1e-5, 1e-5),
            "Chunked should match full. Max diff: {}",
            full_data
                .iter()
                .zip(chunked_data.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        );
    }

    #[test]
    fn test_chunked_matches_full_various_chunk_sizes() {
        let batch = 8;
        let seq = 4;
        let hidden = 16;
        let vocab = 50;

        let hidden_states = MxArray::random_normal(&[batch, seq, hidden], 0.0, 1.0, None).unwrap();
        let weight = MxArray::random_normal(&[vocab, hidden], 0.0, 0.1, None).unwrap();
        let targets = MxArray::randint(&[batch, seq], 0, vocab as i32).unwrap();

        // Full computation
        let full_result =
            full_lm_head_selective_logprobs(&hidden_states, &weight, &targets, true).unwrap();
        full_result.eval();
        let full_data = full_result.to_float32().unwrap();

        // Test various chunk sizes
        for chunk_size in [1, 2, 3, 4, 8, 16] {
            let chunked_result = chunked_lm_head_selective_logprobs(
                &hidden_states,
                &weight,
                &targets,
                chunk_size,
                true,
            )
            .unwrap();
            chunked_result.eval();
            let chunked_data = chunked_result.to_float32().unwrap();

            assert!(
                arrays_close(&full_data, &chunked_data, 1e-5, 1e-5),
                "Chunk size {} failed. Max diff: {}",
                chunk_size,
                full_data
                    .iter()
                    .zip(chunked_data.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max)
            );
        }
    }

    #[test]
    fn test_chunked_with_non_tied_embeddings() {
        let batch = 6;
        let seq = 4;
        let hidden = 16;
        let vocab = 50;

        let hidden_states = MxArray::random_normal(&[batch, seq, hidden], 0.0, 1.0, None).unwrap();
        // For non-tied embeddings, weight shape is [vocab, hidden]
        let weight = MxArray::random_normal(&[vocab, hidden], 0.0, 0.1, None).unwrap();
        let targets = MxArray::randint(&[batch, seq], 0, vocab as i32).unwrap();

        // Full computation with tie_word_embeddings=false
        let full_result =
            full_lm_head_selective_logprobs(&hidden_states, &weight, &targets, false).unwrap();
        full_result.eval();

        // Chunked computation
        let chunked_result =
            chunked_lm_head_selective_logprobs(&hidden_states, &weight, &targets, 2, false)
                .unwrap();
        chunked_result.eval();

        let full_data = full_result.to_float32().unwrap();
        let chunked_data = chunked_result.to_float32().unwrap();

        assert!(
            arrays_close(&full_data, &chunked_data, 1e-5, 1e-5),
            "Non-tied embeddings test failed. Max diff: {}",
            full_data
                .iter()
                .zip(chunked_data.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        );
    }

    #[test]
    fn test_chunked_batch_smaller_than_chunk_size() {
        // When batch_size <= chunk_size, should skip chunking
        let batch = 2;
        let seq = 4;
        let hidden = 16;
        let vocab = 50;

        let hidden_states = MxArray::random_normal(&[batch, seq, hidden], 0.0, 1.0, None).unwrap();
        let weight = MxArray::random_normal(&[vocab, hidden], 0.0, 0.1, None).unwrap();
        let targets = MxArray::randint(&[batch, seq], 0, vocab as i32).unwrap();

        // Full computation
        let full_result =
            full_lm_head_selective_logprobs(&hidden_states, &weight, &targets, true).unwrap();
        full_result.eval();

        // Chunked with chunk_size > batch_size (should skip chunking)
        let chunked_result =
            chunked_lm_head_selective_logprobs(&hidden_states, &weight, &targets, 4, true).unwrap();
        chunked_result.eval();

        let full_data = full_result.to_float32().unwrap();
        let chunked_data = chunked_result.to_float32().unwrap();

        assert!(
            arrays_close(&full_data, &chunked_data, 1e-5, 1e-5),
            "Small batch test failed"
        );
    }

    #[test]
    fn test_chunked_output_shape() {
        let batch = 8;
        let seq = 16;
        let hidden = 32;
        let vocab = 100;

        let hidden_states = MxArray::random_normal(&[batch, seq, hidden], 0.0, 1.0, None).unwrap();
        let weight = MxArray::random_normal(&[vocab, hidden], 0.0, 0.1, None).unwrap();
        let targets = MxArray::randint(&[batch, seq], 0, vocab as i32).unwrap();

        let result =
            chunked_lm_head_selective_logprobs(&hidden_states, &weight, &targets, 2, true).unwrap();
        result.eval();

        let shape = result.shape().unwrap();
        assert_eq!(shape.len(), 2, "Output should be 2D");
        assert_eq!(shape[0], batch, "Batch size should match");
        assert_eq!(shape[1], seq, "Sequence length should match");
    }

    #[test]
    fn test_chunked_numerical_stability() {
        // Test with extreme values to ensure numerical stability
        let batch = 4;
        let seq = 4;
        let hidden = 16;
        let vocab = 50;

        // Create hidden states with extreme values
        let hidden_states = MxArray::random_normal(&[batch, seq, hidden], 0.0, 10.0, None).unwrap();
        let weight = MxArray::random_normal(&[vocab, hidden], 0.0, 0.1, None).unwrap();
        let targets = MxArray::randint(&[batch, seq], 0, vocab as i32).unwrap();

        let result =
            chunked_lm_head_selective_logprobs(&hidden_states, &weight, &targets, 2, true).unwrap();
        result.eval();

        let data = result.to_float32().unwrap();

        // Check no NaN or Inf
        for (i, &v) in data.iter().enumerate() {
            assert!(v.is_finite(), "Value at index {} is not finite: {}", i, v);
            // Log probs should be <= 0
            assert!(v <= 0.0, "Log prob at {} should be <= 0, got {}", i, v);
        }
    }
}

#[cfg(test)]
mod chunked_forward_tests {
    use super::*;
    use crate::models::qwen3::{Qwen3Config, Qwen3Model};

    /// Helper to check if two arrays are close within tolerance
    fn arrays_close(a: &[f32], b: &[f32], atol: f32, rtol: f32) -> bool {
        if a.len() != b.len() {
            return false;
        }
        for (x, y) in a.iter().zip(b.iter()) {
            let tol = atol + rtol * y.abs();
            if (x - y).abs() > tol {
                return false;
            }
        }
        true
    }

    /// Create a tiny model config for fast testing
    fn tiny_config() -> Qwen3Config {
        Qwen3Config {
            vocab_size: 100,
            hidden_size: 32,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            max_position_embeddings: 128,
            head_dim: 8, // 32 / 4 = 8
            use_qk_norm: true,
            tie_word_embeddings: false,
            pad_token_id: 0,
            eos_token_id: 1,
            bos_token_id: 0,
            use_paged_attention: None,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            use_fp8_cache: None,
        }
    }

    #[test]
    fn test_chunked_forward_matches_full() {
        // Chunked forward should produce identical results to full forward
        let config = tiny_config();
        let model = Qwen3Model::new(config.clone()).unwrap();
        let params = model.get_parameters();

        // Create input with batch_size > chunk_size to trigger chunking
        // batch=8, seq=4, chunk_size=2 -> 4 chunks
        let batch = 8;
        let seq = 4;
        let input_data: Vec<i32> = (0..batch * seq).map(|i| i % 50).collect();
        let input_ids = MxArray::from_int32(&input_data, &[batch as i64, seq as i64]).unwrap();

        // Full forward (no chunking)
        let full_result = qwen3_forward_hidden_states(&config, &params, &input_ids).unwrap();
        full_result.eval();

        // Chunked forward with chunk_size=2
        let chunked_result =
            qwen3_forward_hidden_states_chunked(&config, &params, &input_ids, 2).unwrap();
        chunked_result.eval();

        // Compare shapes (convert BigInt64Array to Vec for comparison)
        let full_shape: Vec<i64> = full_result.shape().unwrap().to_vec();
        let chunked_shape: Vec<i64> = chunked_result.shape().unwrap().to_vec();
        assert_eq!(
            full_shape, chunked_shape,
            "Shapes must match: {:?} vs {:?}",
            full_shape, chunked_shape
        );

        // Compare values
        let full_data = full_result.to_float32().unwrap();
        let chunked_data = chunked_result.to_float32().unwrap();

        assert!(
            arrays_close(&full_data, &chunked_data, 1e-4, 1e-4),
            "Chunked should match full. Max diff: {}",
            full_data
                .iter()
                .zip(chunked_data.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        );
    }

    #[test]
    fn test_chunked_forward_various_chunk_sizes() {
        let config = tiny_config();
        let model = Qwen3Model::new(config.clone()).unwrap();
        let params = model.get_parameters();

        let batch = 12;
        let seq = 4;
        let input_data: Vec<i32> = (0..batch * seq).map(|i| i % 50).collect();
        let input_ids = MxArray::from_int32(&input_data, &[batch as i64, seq as i64]).unwrap();

        // Reference: full forward
        let full_result = qwen3_forward_hidden_states(&config, &params, &input_ids).unwrap();
        full_result.eval();
        let full_data = full_result.to_float32().unwrap();

        // Test various chunk sizes
        for chunk_size in [1, 2, 3, 4, 6, 12] {
            let chunked_result =
                qwen3_forward_hidden_states_chunked(&config, &params, &input_ids, chunk_size)
                    .unwrap();
            chunked_result.eval();
            let chunked_data = chunked_result.to_float32().unwrap();

            assert!(
                arrays_close(&full_data, &chunked_data, 1e-4, 1e-4),
                "Chunk size {} failed. Max diff: {}",
                chunk_size,
                full_data
                    .iter()
                    .zip(chunked_data.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max)
            );
        }
    }

    #[test]
    fn test_chunked_forward_small_batch_skips_chunking() {
        // When batch_size <= chunk_size, should skip chunking (identical path)
        let config = tiny_config();
        let model = Qwen3Model::new(config.clone()).unwrap();
        let params = model.get_parameters();

        let batch = 2;
        let seq = 4;
        let input_data: Vec<i32> = (0..batch * seq).map(|i| i % 50).collect();
        let input_ids = MxArray::from_int32(&input_data, &[batch as i64, seq as i64]).unwrap();

        // Full forward
        let full_result = qwen3_forward_hidden_states(&config, &params, &input_ids).unwrap();
        full_result.eval();

        // Chunked with chunk_size > batch_size (should skip chunking)
        let chunked_result =
            qwen3_forward_hidden_states_chunked(&config, &params, &input_ids, 4).unwrap();
        chunked_result.eval();

        let full_data = full_result.to_float32().unwrap();
        let chunked_data = chunked_result.to_float32().unwrap();

        assert!(
            arrays_close(&full_data, &chunked_data, 1e-5, 1e-5),
            "Small batch test failed"
        );
    }

    #[test]
    fn test_chunked_forward_output_shape() {
        let config = tiny_config();
        let model = Qwen3Model::new(config.clone()).unwrap();
        let params = model.get_parameters();

        let batch = 8;
        let seq = 6;
        let input_data: Vec<i32> = (0..batch * seq).map(|i| i % 50).collect();
        let input_ids = MxArray::from_int32(&input_data, &[batch as i64, seq as i64]).unwrap();

        let result = qwen3_forward_hidden_states_chunked(&config, &params, &input_ids, 2).unwrap();
        result.eval();

        let shape: Vec<i64> = result.shape().unwrap().to_vec();
        assert_eq!(shape.len(), 3, "Output should be 3D");
        assert_eq!(shape[0], batch as i64, "Batch size should match");
        assert_eq!(shape[1], seq as i64, "Sequence length should match");
        assert_eq!(
            shape[2], config.hidden_size as i64,
            "Hidden size should match"
        );
    }

    #[test]
    fn test_chunked_forward_with_tied_embeddings() {
        // Test with tied word embeddings (affects LM head path but not hidden states)
        let mut config = tiny_config();
        config.tie_word_embeddings = true;

        let model = Qwen3Model::new(config.clone()).unwrap();
        let params = model.get_parameters();

        let batch = 6;
        let seq = 4;
        let input_data: Vec<i32> = (0..batch * seq).map(|i| i % 50).collect();
        let input_ids = MxArray::from_int32(&input_data, &[batch as i64, seq as i64]).unwrap();

        // Full forward
        let full_result = qwen3_forward_hidden_states(&config, &params, &input_ids).unwrap();
        full_result.eval();

        // Chunked forward
        let chunked_result =
            qwen3_forward_hidden_states_chunked(&config, &params, &input_ids, 2).unwrap();
        chunked_result.eval();

        let full_data = full_result.to_float32().unwrap();
        let chunked_data = chunked_result.to_float32().unwrap();

        assert!(
            arrays_close(&full_data, &chunked_data, 1e-4, 1e-4),
            "Tied embeddings test failed. Max diff: {}",
            full_data
                .iter()
                .zip(chunked_data.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        );
    }
}
