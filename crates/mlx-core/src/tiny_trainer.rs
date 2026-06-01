/*!
 * Tiny in-browser trainable transformer.
 *
 * Exposes a minimal end-to-end training loop for a small Qwen3-shaped
 * transformer to JS via `#[napi]`. It reuses the existing MLX autograd
 * (`autograd::value_and_grad`), the stateless functional forward pass
 * (`utils::functional::qwen3_forward_functional`), the `AdamW` optimizer, and
 * the cross-entropy loss — exactly the same primitives the native SFT trainer
 * uses, but wired up so the whole thing compiles and runs inside the WASM /
 * WebGPU browser backend.
 *
 * The trainer:
 *   - generates every weight tensor itself (fp32) so it never depends on the
 *     `Embedding::new` / `Linear::new` lazy-zero-init path that yields all-zero
 *     weights on WASM,
 *   - keeps a long-lived `AdamW` whose moment state persists across steps,
 *   - and performs teacher-forced next-token training on a single sequence per
 *     `train_step` call.
 *
 * Everything here is fp32 and avoids the wasm-unsafe training stack
 * (`training_model`, `sft`, `grpo`, `sqlx`, `tokio`, `mlx_db`, ...).
 */

use std::collections::HashMap;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::array::MxArray;
use crate::autograd;
use crate::models::qwen3::Qwen3Config;
use crate::nn::Losses;
use crate::optimizers::AdamW;
use crate::param_manager;
use crate::sampling::{self, SamplingConfig};

/// A tiny, fully in-process trainable Qwen3-shaped transformer.
///
/// All weights are fp32 and live in `params`, keyed by the same names the
/// functional forward pass (`qwen3_forward_functional`) expects. Training
/// state (the AdamW moment estimates) is held inside `optimizer` and persists
/// across `train_step` calls.
#[napi]
pub struct TinyTrainer {
    config: Qwen3Config,
    /// Stable, sorted parameter names. Index `i` here lines up positionally
    /// with the gradient returned by `value_and_grad` and the param returned by
    /// `AdamW::update_batch`.
    param_names: Vec<String>,
    /// Mutable fp32 weights, persisted across calls.
    params: HashMap<String, MxArray>,
    optimizer: AdamW,
    learning_rate: f64,
    seed: u64,
    step: i64,
}

#[napi]
impl TinyTrainer {
    /// Create a new trainer.
    ///
    /// * `config` - Qwen3 model shape. Use a small config; `tie_word_embeddings`
    ///   is honored (when true the lm_head reuses `embedding.weight`).
    /// * `seed` - RNG seed for **weight initialization only**. It does not affect
    ///   `sample()` draws (sampling uses a separate fixed thread-local RNG on
    ///   WASM).
    /// * `learning_rate` - AdamW learning rate (default 3e-3).
    #[napi(constructor)]
    pub fn new(config: Qwen3Config, seed: f64, learning_rate: Option<f64>) -> Result<Self> {
        let learning_rate = learning_rate.unwrap_or(3e-3);
        let optimizer = Self::build_optimizer(learning_rate);
        let mut trainer = Self {
            config,
            param_names: Vec::new(),
            params: HashMap::new(),
            optimizer,
            learning_rate,
            seed: seed as u64,
            step: 0,
        };
        trainer.init_params(seed)?;
        Ok(trainer)
    }

    /// Run a single teacher-forced training step over `token_ids`.
    ///
    /// The sequence is fed as both input and labels; inside the loss closure the
    /// logits/labels are shifted by one position (predict-next-token), matching
    /// the native SFT loop. Returns the (pre-update) scalar loss for this step.
    ///
    /// Requires `token_ids.len() >= 2` (at least one shift target).
    #[napi]
    pub fn train_step(&mut self, token_ids: Vec<i32>) -> Result<f64> {
        if token_ids.len() < 2 {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "train_step requires at least 2 token ids (got {}); need one shift target",
                    token_ids.len()
                ),
            ));
        }

        let seq_len = token_ids.len() as i64;
        let input_ids = MxArray::from_int32(&token_ids, &[1, seq_len])?;

        // Collect param arrays in the stable sorted order.
        let param_arrays: Vec<&MxArray> = self
            .param_names
            .iter()
            .map(|name| {
                self.params
                    .get(name)
                    .ok_or_else(|| Error::from_reason(format!("Parameter not found: {name}")))
            })
            .collect::<Result<Vec<_>>>()?;

        // Data owned by the 'static loss closure.
        let config = self.config.clone();
        let names = self.param_names.clone();
        let labels = input_ids.clone();
        let inputs = input_ids.clone();

        let loss_fn = move |params: &[MxArray]| -> Result<MxArray> {
            let param_dict = param_manager::map_params_to_dict(params, &names)?;

            // [batch, seq, vocab]
            let logits =
                crate::utils::functional::qwen3_forward_functional(&config, &param_dict, &inputs)?;

            let b = logits.shape_at(0)?;
            let t = logits.shape_at(1)?;
            let v = logits.shape_at(2)?;

            // Shift: logits[:, :-1] predicts labels[:, 1:] (causal LM).
            let shift_logits = logits.slice(&[0, 0, 0], &[b, t - 1, v])?;
            let shift_labels = labels.slice(&[0, 1], &[b, t])?;

            let logits_flat = shift_logits.reshape(&[b * (t - 1), v])?;
            let labels_flat = shift_labels.reshape(&[b * (t - 1)])?;

            // Mean cross-entropy; no ignore_index, no label smoothing.
            Losses::cross_entropy(&logits_flat, &labels_flat, None, None, Some(0.0))
        };

        let (loss_array, grad_arrays) = autograd::value_and_grad(param_arrays, loss_fn)?;
        loss_array.eval();
        for g in &grad_arrays {
            g.eval();
        }
        let loss_val = loss_array.item_at_float32(0)? as f64;

        // Assemble params + grads in the same (sorted) order for the optimizer.
        let names = self.param_names.clone();
        let params_vec: Vec<&MxArray> = names
            .iter()
            .map(|name| {
                self.params
                    .get(name)
                    .ok_or_else(|| Error::from_reason(format!("Parameter not found: {name}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let grads_vec: Vec<&MxArray> = grad_arrays.iter().collect();

        let updated = self
            .optimizer
            .update_batch(names.clone(), params_vec, grads_vec)?;

        // Write updated params back by name (same order we passed in) and
        // materialize them so the graph for this step can be released.
        for (name, new_param) in names.into_iter().zip(updated) {
            new_param.eval();
            self.params.insert(name, new_param);
        }

        self.step += 1;

        // Memory: this forward/backward path is uncompiled, so nothing
        // accumulates in MLX's compiled-graph cache, and each step's graph and
        // gradients are released as they drop (params are eval'd and overwritten
        // above). As cheap insurance for a long-lived browser tab, periodically
        // release MLX's retained buffer cache back to the system. The loss read
        // above already synchronized, so this adds no extra stall.
        if self.step % 64 == 0 {
            crate::array::heavy_cleanup();
        }

        Ok(loss_val)
    }

    /// Autoregressively sample `n` tokens continuing `prompt_ids`.
    ///
    /// Returns only the newly generated tail (ids after the prompt). Stops early
    /// if the configured EOS id is produced. `temperature <= 0` is greedy.
    ///
    /// Note: generation draws use a fixed thread-local RNG (not seeded by the
    /// trainer `seed`), so only weight init is reproducible via `seed`.
    #[napi]
    pub fn sample(&self, prompt_ids: Vec<i32>, n: u32, temperature: f64) -> Result<Vec<i32>> {
        if prompt_ids.is_empty() {
            return Err(Error::new(
                Status::InvalidArg,
                "sample requires a non-empty prompt".to_string(),
            ));
        }

        let mut ids = prompt_ids;
        let prompt_len = ids.len();
        let sampling_cfg = SamplingConfig {
            temperature: Some(temperature),
            top_k: Some(0),
            top_p: Some(1.0),
            min_p: Some(0.0),
        };

        // No KV cache: each step re-runs the full forward over the whole
        // (growing) sequence, so this is O(n · T²) in the sequence length.
        // Fine for the tiny model and short generations this playground uses.
        for _ in 0..n {
            let seq_len = ids.len() as i64;
            let input = MxArray::from_int32(&ids, &[1, seq_len])?;

            // [1, T, vocab]
            let logits = crate::utils::functional::qwen3_forward_functional(
                &self.config,
                &self.params,
                &input,
            )?;
            let t = logits.shape_at(1)?;
            let v = logits.shape_at(2)?;

            // Last-position logits -> [vocab]
            let last = logits.slice(&[0, t - 1, 0], &[1, t, v])?.reshape(&[v])?;

            let tok = sampling::sample(&last, Some(sampling_cfg))?;
            tok.eval();
            let id = tok.item_at_float32(0)? as i32;
            ids.push(id);

            if id == self.config.eos_token_id {
                break;
            }
        }

        Ok(ids[prompt_len..].to_vec())
    }

    /// Re-initialize all weights from `seed` and reset the optimizer + step.
    #[napi]
    pub fn reset(&mut self, seed: f64) -> Result<()> {
        self.init_params(seed)?;
        self.optimizer = Self::build_optimizer(self.learning_rate);
        self.step = 0;
        Ok(())
    }

    /// Number of training steps performed since construction / last `reset`.
    #[napi(getter)]
    pub fn step(&self) -> i64 {
        self.step
    }

    /// The RNG seed used for the current weight initialization (set at
    /// construction or the last `reset`). Useful for the UI to display and
    /// reproduce a run. Seeds weight init only — not `sample()` draws.
    #[napi(getter)]
    pub fn seed(&self) -> f64 {
        self.seed as f64
    }

    /// Build a fresh AdamW configured for fast convergence on a tiny model.
    fn build_optimizer(learning_rate: f64) -> AdamW {
        AdamW::new(
            Some(learning_rate),
            None,       // beta1
            None,       // beta2
            None,       // eps
            Some(0.0),  // weight_decay
            Some(true), // bias_correction
        )
    }

    /// Generate every weight tensor (fp32) and populate `params` / `param_names`.
    ///
    /// Initialization recipe (standard transformer init):
    ///   - all linear projections + embedding: Normal(0, 0.02)
    ///   - all RMS-norm / layernorm weights: ones
    ///
    /// `mlx_seed` is set once up front so weight init is reproducible.
    fn init_params(&mut self, seed: f64) -> Result<()> {
        unsafe {
            mlx_sys::mlx_seed(seed as u64);
        }

        let hidden = self.config.hidden_size as i64;
        let vocab = self.config.vocab_size as i64;
        let intermediate = self.config.intermediate_size as i64;
        let head_dim = self.config.head_dim as i64;
        let num_heads = self.config.num_heads as i64;
        let num_kv_heads = self.config.num_kv_heads as i64;
        let num_layers = self.config.num_layers;

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let mut params: HashMap<String, MxArray> = HashMap::new();

        let linear = |out: i64, in_: i64| -> Result<MxArray> {
            MxArray::random_normal(&[out, in_], 0.0, 0.02, None)
        };

        // Token embedding (also reused as the lm_head when tie_word_embeddings).
        params.insert("embedding.weight".to_string(), linear(vocab, hidden)?);

        for layer_idx in 0..num_layers {
            let prefix = format!("layers.{layer_idx}");

            // Attention projections.
            params.insert(
                format!("{prefix}.self_attn.q_proj.weight"),
                linear(q_dim, hidden)?,
            );
            params.insert(
                format!("{prefix}.self_attn.k_proj.weight"),
                linear(kv_dim, hidden)?,
            );
            params.insert(
                format!("{prefix}.self_attn.v_proj.weight"),
                linear(kv_dim, hidden)?,
            );
            params.insert(
                format!("{prefix}.self_attn.o_proj.weight"),
                linear(hidden, q_dim)?,
            );

            // Optional QK norms.
            if self.config.use_qk_norm {
                params.insert(
                    format!("{prefix}.self_attn.q_norm.weight"),
                    MxArray::ones(&[head_dim], None)?,
                );
                params.insert(
                    format!("{prefix}.self_attn.k_norm.weight"),
                    MxArray::ones(&[head_dim], None)?,
                );
            }

            // MLP.
            params.insert(
                format!("{prefix}.mlp.gate_proj.weight"),
                linear(intermediate, hidden)?,
            );
            params.insert(
                format!("{prefix}.mlp.up_proj.weight"),
                linear(intermediate, hidden)?,
            );
            params.insert(
                format!("{prefix}.mlp.down_proj.weight"),
                linear(hidden, intermediate)?,
            );

            // Norms.
            params.insert(
                format!("{prefix}.input_layernorm.weight"),
                MxArray::ones(&[hidden], None)?,
            );
            params.insert(
                format!("{prefix}.post_attention_layernorm.weight"),
                MxArray::ones(&[hidden], None)?,
            );
        }

        // Final norm.
        params.insert(
            "final_norm.weight".to_string(),
            MxArray::ones(&[hidden], None)?,
        );

        // Separate lm_head only when embeddings are NOT tied.
        if !self.config.tie_word_embeddings {
            params.insert("lm_head.weight".to_string(), linear(vocab, hidden)?);
        }

        // Materialize so the random draws are realized deterministically now.
        for v in params.values() {
            v.eval();
        }

        let mut param_names: Vec<String> = params.keys().cloned().collect();
        param_names.sort();

        self.params = params;
        self.param_names = param_names;
        self.seed = seed as u64;
        Ok(())
    }
}

/// Direct correctness probe for the WebGPU scatter_add (Gather::vjp) kernel.
///
/// Computes d/dtable of `sum(take(table, [3, 3, 3], axis=0))` for
/// `table = zeros([5, 1])`. The three gathers all collide on row 3, so the
/// gradient must ACCUMULATE them into that row => `3.0`. A broken
/// last-write-wins scatter would instead yield `1.0` there.
///
/// `take` is a `Gather` whose vjp is exactly the `Scatter` Sum-reduce path,
/// the same op the embedding-table gradient (`embedding_functional`) exercises
/// during training. Running it through `value_and_grad` + `eval` on whatever
/// device is active means this validates the shipped WGSL kernel end-to-end
/// (the WebGPU device in the browser), not just the training loss curve.
///
/// Returns the full gradient column flattened to `f64` (length 5). For a
/// correct kernel that is `[0.0, 0.0, 0.0, 3.0, 0.0]`; a broken
/// last-write-wins kernel yields `[0.0, 0.0, 0.0, 1.0, 0.0]`.
#[napi]
pub fn webgpu_scatter_add_selftest() -> napi::Result<Vec<f64>> {
    // table = zeros([5, 1]) f32; the leaf we differentiate w.r.t.
    let table = MxArray::zeros(&[5, 1], None)?;

    // Three colliding gathers of row 3 (shape [3]).
    let indices = MxArray::from_int32(&[3, 3, 3], &[3])?;

    // loss(table) = sum(take(table, [3,3,3], axis=0)). `take` is a Gather, so
    // its vjp is the Scatter Sum-reduce kernel under test; colliding rows must
    // accumulate.
    let loss_fn = move |params: &[MxArray]| -> Result<MxArray> {
        let gathered = params[0].take(&indices, 0)?;
        gathered.sum(None, None)
    };

    let (_loss, grad_arrays) = autograd::value_and_grad(vec![&table], loss_fn)?;
    let grad = &grad_arrays[0];
    grad.eval();

    // Read the full gradient column back to host and widen to f64.
    let grad_f32 = grad.to_float32_vec()?;
    Ok(grad_f32.into_iter().map(|v| v as f64).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small but real Qwen3-shaped config. `num_heads * head_dim == hidden`
    /// and `tie_word_embeddings = true` (so no separate lm_head tensor).
    fn tiny_config() -> Qwen3Config {
        Qwen3Config {
            vocab_size: 16,
            hidden_size: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            max_position_embeddings: 64,
            head_dim: 16, // 4 * 16 == 64 == hidden_size
            use_qk_norm: true,
            tie_word_embeddings: true,
            pad_token_id: 0,
            eos_token_id: 1,
            bos_token_id: 0,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            use_block_paged_cache: Some(false),
        }
    }

    #[test]
    fn test_tiny_trainer_converges() {
        let config = tiny_config();
        let mut trainer = TinyTrainer::new(config, 42.0, Some(3e-3)).unwrap();

        // Fixed cyclic pattern, length 16, all ids within vocab (0..16).
        // Includes a deliberately repeated id (the two 3s) so the embedding
        // gradient-accumulation path for a token appearing multiple times in
        // one sequence is exercised.
        let seq: Vec<i32> = vec![2, 5, 3, 7, 3, 9, 4, 6, 8, 10, 11, 2, 5, 7, 9, 4];

        let total_steps = 250usize;
        let mut losses: Vec<f64> = Vec::with_capacity(total_steps);
        for _ in 0..total_steps {
            let loss = trainer.train_step(seq.clone()).unwrap();
            assert!(
                loss.is_finite(),
                "loss became non-finite during training: {loss}"
            );
            losses.push(loss);
        }

        assert_eq!(trainer.step(), total_steps as i64);

        // Compare the mean of the first window vs the last window.
        let window = 10usize;
        let initial: f64 = losses[..window].iter().sum::<f64>() / window as f64;
        let final_: f64 = losses[total_steps - window..].iter().sum::<f64>() / window as f64;

        println!(
            "tiny_trainer convergence: initial(mean of first {window}) = {initial:.4}, \
             final(mean of last {window}) = {final_:.4}, steps = {total_steps}"
        );

        assert!(
            final_ < 0.5 * initial,
            "loss did not drop substantially: initial={initial:.4} final={final_:.4}"
        );
        assert!(
            final_ < 1.0,
            "final loss too high for a memorizable tiny sequence: {final_:.4}"
        );

        // After memorizing the pattern, greedy continuation from a prefix should
        // reproduce the next tokens of the training sequence.
        let prefix: Vec<i32> = seq[..5].to_vec();
        let generated = trainer.sample(prefix, 4, 0.0).unwrap();
        let expected: Vec<i32> = seq[5..9].to_vec();
        println!("tiny_trainer greedy continuation: got {generated:?}, expected {expected:?}");
        let matches = generated
            .iter()
            .zip(expected.iter())
            .filter(|(a, b)| a == b)
            .count();
        assert!(
            matches >= 3,
            "greedy continuation should mostly match the trained pattern: got {generated:?}, expected {expected:?}"
        );
    }

    #[test]
    fn test_train_step_requires_two_tokens() {
        let mut trainer = TinyTrainer::new(tiny_config(), 1.0, None).unwrap();
        assert!(trainer.train_step(vec![3]).is_err());
    }

    #[test]
    fn test_reset_restores_initial_loss() {
        let config = tiny_config();
        let mut trainer = TinyTrainer::new(config, 7.0, Some(3e-3)).unwrap();
        let seq: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let first_loss = trainer.train_step(seq.clone()).unwrap();
        for _ in 0..40 {
            trainer.train_step(seq.clone()).unwrap();
        }
        let trained_loss = trainer.train_step(seq.clone()).unwrap();
        assert!(trained_loss < first_loss, "loss should drop with training");

        // Reset with the same seed -> first post-reset loss should match the
        // original first loss (deterministic weight init), and step resets.
        trainer.reset(7.0).unwrap();
        assert_eq!(trainer.step(), 0);
        let post_reset_loss = trainer.train_step(seq.clone()).unwrap();
        assert!(
            (post_reset_loss - first_loss).abs() < 1e-3,
            "post-reset loss {post_reset_loss} should match initial {first_loss} (same seed)"
        );
    }
}
