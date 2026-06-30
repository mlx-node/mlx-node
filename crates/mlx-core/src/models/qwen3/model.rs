/**
 * Qwen3 Model - Core Model Implementation
 *
 * Contains the model structure, forward passes, and core model methods.
 */
use std::cell::Cell;
use std::collections::HashMap;
use std::iter;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use tracing::{debug, info, warn};

use crate::array::{MxArray, heavy_cleanup, synchronize_and_clear_cache};
use crate::engine::backend::{
    ChatBackend, DecodeStep, PagedBackend, PagedPrefix, PagedTurnSetup, ResetScope, SaveStateArgs,
    TrainBackend, TurnOutput, TurnSetup, WholeTurnArgs,
};
use crate::engine::cmd::{
    ChatCmd, FromChatCmd, FromTrainCmd, TrainCmd, handle_chat_cmd, handle_train_cmd,
};
use crate::model_thread::{ModelThread, ResponseTx, send_and_await};
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::sampling::{
    SamplingConfig, apply_frequency_penalty, apply_presence_penalty, apply_repetition_penalty,
    check_repetition_cutoff, sample, sample_and_logprobs,
};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer, ToolDefinition};
use crate::training_model::ModelType;
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
use crate::transformer::{KVCache, TransformerBlock};

use super::{BatchGenerationResult, GenerationConfig, GenerationResult, Qwen3Config};
use crate::engine;

/// Internal model state owned exclusively by the dedicated model thread.
///
/// No `Arc<RwLock<>>` — the model thread has sole ownership of all inference
/// and training state. Training commands are routed via `TrainingDispatch`.
pub(crate) struct Qwen3Inner {
    pub(crate) config: Qwen3Config,
    pub(crate) embedding: Embedding,
    pub(crate) layers: Vec<TransformerBlock>,
    pub(crate) final_norm: RMSNorm,
    pub(crate) lm_head: Linear,
    pub(crate) kv_caches: Option<Vec<KVCache>>,
    pub(crate) tokenizer: Option<Arc<Qwen3Tokenizer>>,
    /// Block-paged KV adapter (vLLM-style refcounted prefix cache).
    ///
    /// **Opt-in via `Qwen3Config::use_block_paged_cache`**. When `Some`,
    /// chat-session methods route through `forward_paged_adapter` for
    /// cross-request prefix reuse. Defaults to `None` so the existing flat
    /// `Vec<KVCache>` path stays untouched.
    pub(crate) paged_adapter: Option<PagedKVCacheAdapter>,
    pub(crate) cached_kv_keys: Vec<Option<MxArray>>,
    pub(crate) cached_kv_values: Vec<Option<MxArray>>,
    pub(crate) cached_cache_idx: i32,
    pub(crate) cached_token_history: Vec<u32>,
    /// Structural uniformity with VLM-capable models. Always `None` for
    /// text-only Qwen3 — the field exists so that future session helpers
    /// share the same shape as dense/MoE/VLM inner structs.
    pub(crate) cached_image_key: Option<u64>,
    /// Turn-scratch KV state for the flat engine flow. The `ChatBackend`
    /// split (`prefill` → `begin_decode` → `save_cache_state`) needs the
    /// in-flight clones of `cached_kv_keys` / `cached_kv_values` /
    /// `cached_cache_idx` to live on `self` between hook calls. Seeded by
    /// [`Qwen3Inner::flat_prefill`], advanced by [`Qwen3Decode`],
    /// persisted (moved into `cached_*`) by
    /// [`ChatBackend::save_cache_state`].
    pub(crate) turn_kv_keys: Vec<Option<MxArray>>,
    pub(crate) turn_kv_values: Vec<Option<MxArray>>,
    pub(crate) turn_cache_idx: i32,
    /// Sanctioned pure-KV exact-match rewind (see
    /// [`ChatBackend::verify_cache_prefix`]'s rustdoc): set by
    /// `verify_cache_prefix` when the rendered prompt EXACTLY equals the
    /// cached history (it then returns `cached_len - 1`), consumed by
    /// the next `flat_prefill`, which backs the write cursor up one slot
    /// and re-forwards the final token (the "zero delta — re-run last
    /// token" case). `Cell` because the verify hook takes `&self`; the
    /// inner runs single-threaded on its model thread.
    pending_exact_match_rewind: Cell<bool>,
    /// Whether the in-flight generic-flow turn is streaming. Recorded by
    /// [`ChatBackend::profiler_label`] (the one pre-`begin_decode` hook
    /// that receives `is_streaming`; `TurnSetup` does not carry it) so
    /// `begin_decode` can bake the streaming profiler relabel
    /// (`"qwen3_chat_stream[_delta]_rust"`) into the stepper.
    turn_is_streaming: Cell<bool>,
    /// Training state owned by the model thread.
    /// Created when `InitTraining` command is received, destroyed when training ends.
    pub(crate) training_state: Option<crate::training_state::ModelThreadTrainingState>,
    /// Sampling + stop-token defaults parsed from the checkpoint's
    /// `generation_config.json` at load time. Empty for checkpoints that
    /// ship no such file. Consumed by the [`ChatBackend`] sampling/EOS
    /// hooks and the raw `generate` loop.
    gen_defaults: crate::engine::ModelGenerationDefaults,
}

/// Commands dispatched from NAPI methods to the dedicated model thread.
pub(crate) enum Qwen3Cmd {
    /// Chat-session commands (start / continue / tool + streaming twins +
    /// reset-caches), wrapping the model-neutral engine's [`ChatCmd`].
    /// The thread loop routes these to
    /// [`crate::engine::cmd::handle_chat_cmd`], which drives the
    /// [`ChatBackend`] impl on [`Qwen3Inner`]; the per-variant
    /// behavioural contracts live on [`ChatCmd`] and the generic session
    /// cores in [`crate::engine::session`].
    Chat(ChatCmd),
    Generate {
        messages: Vec<ChatMessage>,
        config: Option<GenerationConfig>,
        reply: ResponseTx<GenerationResult>,
    },
    GenerateBatch {
        prompts: Vec<Vec<ChatMessage>>,
        group_size: u32,
        config: Option<GenerationConfig>,
        reply: ResponseTx<BatchGenerationResult>,
    },
    /// Training-session commands, wrapping the model-neutral engine's
    /// [`TrainCmd`]. The thread loop routes these to
    /// [`crate::engine::cmd::handle_train_cmd`], which drives the
    /// [`TrainBackend`] impl on [`Qwen3Inner`].
    Train(TrainCmd),
    SaveModel {
        save_path: String,
        reply: ResponseTx<()>,
    },
}

impl FromChatCmd for Qwen3Cmd {
    #[inline]
    fn from_chat(cmd: ChatCmd) -> Self {
        Qwen3Cmd::Chat(cmd)
    }
}

impl FromTrainCmd for Qwen3Cmd {
    #[inline]
    fn from_train(cmd: TrainCmd) -> Self {
        Qwen3Cmd::Train(cmd)
    }
}

/// Training backend the model-neutral [`handle_train_cmd`] drives. Each
/// method forwards to the inherent `*_sync_impl` body on [`Qwen3Inner`].
impl TrainBackend for Qwen3Inner {
    fn training_state_mut(
        &mut self,
    ) -> &mut Option<crate::training_state::ModelThreadTrainingState> {
        &mut self.training_state
    }

    fn init_training_sync(
        &mut self,
        config: Box<crate::grpo::engine::GRPOEngineConfig>,
        model_type: crate::training_model::ModelType,
    ) -> Result<()> {
        self.init_training_sync_impl(*config, model_type)
    }

    fn generate_for_training_thread_sync(
        &mut self,
        prompts: Vec<Vec<ChatMessage>>,
        group_size: usize,
        gen_config: super::GenerationConfig,
        enable_thinking: Option<bool>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Result<crate::training_model::GenerationPlainData> {
        self.generate_for_training_thread_sync_impl(
            prompts,
            group_size,
            gen_config,
            enable_thinking,
            tools,
        )
    }

    fn train_step_grpo_sync(
        &mut self,
        rewards: Vec<f64>,
        group_size: i32,
        loss_config: crate::grpo::loss::GRPOLossConfig,
        valid_indices: Option<Vec<usize>>,
    ) -> Result<crate::training_model::TrainStepPlainMetrics> {
        self.train_step_grpo_sync_impl(rewards, group_size, loss_config, valid_indices)
    }

    fn train_step_sft_sync(
        &mut self,
        input_ids: Vec<i32>,
        input_shape: Vec<i64>,
        labels: Vec<i32>,
        labels_shape: Vec<i64>,
        config: crate::sft::engine::SftEngineConfig,
    ) -> Result<crate::training_model::TrainStepPlainMetrics> {
        self.train_step_sft_sync_impl(input_ids, input_shape, labels, labels_shape, config)
    }

    fn save_optimizer_state_sync(&self, path: String) -> Result<()> {
        self.save_optimizer_state_sync_impl(path)
    }

    fn load_optimizer_state_sync(&mut self, path: String) -> Result<()> {
        self.load_optimizer_state_sync_impl(path)
    }
}

/// Command handler for the dedicated model thread.
pub(crate) fn handle_qwen3_cmd(inner: &mut Qwen3Inner, cmd: Qwen3Cmd) {
    match cmd {
        Qwen3Cmd::Chat(chat_cmd) => {
            // NOTE: no per-request cache drain here. On a multi-model
            // server the MLX allocator free-pool is process-wide, so
            // flushing after a request on model A discards blocks
            // about to be reused by model B. Between-turn drain is
            // handled by the TS idle sweeper in `@mlx-node/server`.
            handle_chat_cmd(inner, chat_cmd);
        }
        Qwen3Cmd::Generate {
            messages,
            config,
            reply,
        } => {
            let _ = reply.send(inner.generate_sync(messages, config));
        }
        Qwen3Cmd::GenerateBatch {
            prompts,
            group_size,
            config,
            reply,
        } => {
            let _ = reply.send(inner.generate_batch_sync(prompts, group_size, config));
        }
        // --- Training commands ---
        Qwen3Cmd::Train(train_cmd) => {
            handle_train_cmd(inner, train_cmd);
        }
        Qwen3Cmd::SaveModel { save_path, reply } => {
            let _ = reply.send(inner.save_model_sync(&save_path));
        }
    }
}

/// Pure prefix-match check — exposed for unit testing so the invariant
/// "never returns an intermediate value" can be exercised without
/// instantiating a full [`Qwen3Inner`] (which requires model weights).
///
/// See [`Qwen3Inner::verify_cache_prefix`] for the doc contract.
fn verify_cache_prefix_pure(
    tokens: &[u32],
    cached_token_history: &[u32],
    has_kv_caches: bool,
    reuse_cache: bool,
) -> usize {
    if !reuse_cache {
        return 0;
    }
    if cached_token_history.is_empty() {
        return 0;
    }
    if tokens.len() < cached_token_history.len() {
        return 0;
    }
    if !tokens.starts_with(cached_token_history) {
        return 0;
    }
    if !has_kv_caches {
        return 0;
    }
    cached_token_history.len()
}

// ========== Qwen3Inner implementation ==========
// All these methods run on the dedicated model thread (synchronous, no locks).

impl Qwen3Inner {
    /// Create a new Qwen3Inner with the given configuration.
    pub(crate) fn new(config: Qwen3Config) -> Result<Self> {
        let embedding = Embedding::new(config.vocab_size as u32, config.hidden_size as u32)?;

        let layers = (0..config.num_layers)
            .map(|_| {
                TransformerBlock::new(
                    config.hidden_size as u32,
                    config.num_heads as u32,
                    config.num_kv_heads as u32,
                    config.intermediate_size as u32,
                    config.rms_norm_eps,
                    Some(config.rope_theta),
                    Some(config.use_qk_norm),
                    Some(config.head_dim as u32),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let final_norm = RMSNorm::new(config.hidden_size as u32, Some(config.rms_norm_eps))?;

        let lm_head = Linear::new(
            config.hidden_size as u32,
            config.vocab_size as u32,
            Some(false),
        )?;

        // Block-paged KV adapter — opt-in via `use_block_paged_cache`.
        //
        // The adapter pairs an `Arc<Mutex<BlockAllocator>>` (logical
        // refcounted block lifecycle + prefix hash table) with an
        // `Arc<LayerKVPool>` (per-layer Metal K/V buffers). Together they
        // supersede the flat `Vec<KVCache>` storage when wired through the
        // forward path.
        //
        // Memory budget: derived from `paged_cache_memory_mb` — divided by
        // per-block size to compute `num_blocks`.
        //
        // Cache dtype: BFloat16 (Qwen3's production dtype). FP16 is also
        // supported by `LayerKVPool` for non-Qwen3 callers, but Qwen3 weights
        // ship as BF16 so we hard-code that here. FP8 mode is intentionally
        // not yet plumbed through this path — `KvScaleManager` integration
        // is a follow-up.
        // Default to ON: paged-vs-flat parity verified via
        // `qwen3_paged_vs_flat_parity` integration test (greedy byte-equal +
        // prefix-reuse byte-equal at BF16 against real Qwen3-0.6B weights).
        // Callers can still opt out with `use_block_paged_cache: Some(false)`.
        // The block-paged KV path uses Metal-only kernels; on a non-Metal
        // backend (the CUDA/Linux build) its write/gather methods are throwing
        // stubs. Force flat eager there by leaving the adapter None, so the
        // `paged_adapter.is_some()` routing falls through to the flat path.
        // macOS is unaffected — the probe is always true, so the default wins.
        let paged_adapter = if config.use_block_paged_cache.unwrap_or(true)
            && crate::engine::persistence::compiled_forward_backend_available()
        {
            let block_size = config.paged_block_size.unwrap_or(16);
            let gpu_memory_mb = config.paged_cache_memory_mb.unwrap_or(2048);
            let pa_config = mlx_paged_attn::PagedAttentionConfig {
                block_size,
                gpu_memory_mb,
                head_size: config.head_dim as u32,
                num_kv_heads: config.num_kv_heads as u32,
                num_layers: config.num_layers as u32,
                // FP8 mode for the adapter is gated separately on a follow-up
                // (KvScaleManager); always false here.
                use_fp8_cache: Some(false),
                max_seq_len: Some(config.max_position_embeddings as u32),
                max_batch_size: Some(32),
            };

            let num_blocks = pa_config.calculate_num_blocks();
            if num_blocks == 0 {
                return Err(napi::Error::from_reason(format!(
                    "Block-paged adapter: gpu_memory_mb={gpu_memory_mb} too small to hold any \
                     blocks (head_size={}, num_kv_heads={}, block_size={}, num_layers={})",
                    pa_config.head_size,
                    pa_config.num_kv_heads,
                    pa_config.block_size,
                    pa_config.num_layers,
                )));
            }

            let allocator = Arc::new(std::sync::Mutex::new(mlx_paged_attn::BlockAllocator::new(
                num_blocks, block_size,
            )));

            // BFloat16 = Qwen3 production dtype.
            let cache_dtype = mlx_paged_attn::metal::MetalDtype::BFloat16;
            let pool = mlx_paged_attn::LayerKVPool::new(pa_config, num_blocks, cache_dtype)
                .map_err(|e| {
                    napi::Error::from_reason(format!(
                        "Failed to construct LayerKVPool for block-paged adapter: {e}"
                    ))
                })?;

            let adapter =
                PagedKVCacheAdapter::new(allocator, Arc::new(pool), block_size).map_err(|e| {
                    napi::Error::from_reason(format!(
                        "Failed to construct PagedKVCacheAdapter: {e}"
                    ))
                })?;

            info!(
                "Block-paged adapter enabled: num_blocks={num_blocks}, block_size={block_size}, \
                 gpu_memory_mb={gpu_memory_mb}, cache_dtype=BFloat16"
            );
            Some(adapter)
        } else {
            None
        };

        Ok(Self {
            config,
            embedding,
            layers,
            final_norm,
            lm_head,
            kv_caches: None,
            tokenizer: None,
            paged_adapter,
            cached_kv_keys: Vec::new(),
            cached_kv_values: Vec::new(),
            cached_cache_idx: 0,
            cached_token_history: Vec::new(),
            cached_image_key: None,
            turn_kv_keys: Vec::new(),
            turn_kv_values: Vec::new(),
            turn_cache_idx: 0,
            pending_exact_match_rewind: Cell::new(false),
            turn_is_streaming: Cell::new(false),
            training_state: None,
            gen_defaults: crate::engine::ModelGenerationDefaults::default(),
        })
    }

    pub(crate) fn set_tokenizer(&mut self, tokenizer: Arc<Qwen3Tokenizer>) {
        self.tokenizer = Some(tokenizer);
    }

    /// Store the checkpoint's parsed `generation_config.json` defaults.
    /// Called once at load time after construction.
    pub(crate) fn set_gen_defaults(&mut self, defaults: crate::engine::ModelGenerationDefaults) {
        self.gen_defaults = defaults;
    }

    fn reset_kv_caches_sync(&mut self) -> Result<()> {
        if let Some(caches) = self.kv_caches.as_mut() {
            for cache in caches.iter_mut() {
                cache.reset();
            }
        }
        self.cached_kv_keys.clear();
        self.cached_kv_values.clear();
        self.cached_cache_idx = 0;
        self.cached_token_history.clear();
        self.cached_image_key = None;
        self.turn_kv_keys.clear();
        self.turn_kv_values.clear();
        self.turn_cache_idx = 0;
        self.pending_exact_match_rewind.set(false);
        // Drop any live paged-adapter request. Without this, a
        // finalize_turn_keep_live call from a prior session would leave
        // block_table populated and a subsequent `paged_turn_sync_core`
        // could mistakenly take the warm-continue path against stale
        // tokens. NOTE: this alone is NOT a fully cold reset — released
        // full blocks stay content-addressed in the allocator's prefix
        // cache; the EXPLICIT command reset purges them on top of this
        // (`ResetScope::Command` branch of `ChatBackend::reset_caches`).
        if let Some(adapter) = self.paged_adapter.as_mut() {
            let _ = adapter.release_request();
        }
        Ok(())
    }

    /// Run a paged-attention prefill chunk over the layer stack.
    ///
    /// `suffix_tokens` is the chunk of NEW tokens (already excluded from
    /// the prefix-cache hit). `first_logical_position` is the logical
    /// position at which `suffix_tokens[0]` lives in the full request (so
    /// `cached_prefix_len` for a normal cache-hit prefill, or `0` for a
    /// fresh request). Records the chunk into the adapter and writes K/V
    /// through the pool via `forward_paged_adapter`. Returns the last
    /// position's logits squeezed to `[vocab]`.
    ///
    /// When `MLX_PAGED_PREFILL_CHUNK_SIZE` is set to a positive value AND
    /// `suffix_tokens.len()` exceeds that value, the suffix is sliced into
    /// `<chunk_size>`-token sub-chunks. Each sub-chunk runs through every
    /// layer with the existing `forward_paged_adapter` (which already
    /// supports `cached_prefix_len > 0` + `Q < K` via the explicit causal
    /// mask path). Between sub-chunks we `synchronize_and_clear_cache()` so
    /// MLX's lazy graph + caching allocator do not pile up the entire
    /// suffix's intermediates simultaneously. Memory peak is then bounded
    /// by `chunk_len * hidden_dim` instead of `suffix_len * hidden_dim`.
    /// `final_norm` + `lm_head` only run on the last sub-chunk (vocab
    /// projection is throwaway work for non-final chunks; matches vLLM's
    /// `is_prefill_chunk` skip).
    fn run_paged_prefill_chunk(
        &mut self,
        suffix_tokens: &[u32],
        first_logical_position: u32,
        num_layers: usize,
        positions: &MxArray,
    ) -> Result<MxArray> {
        let chunk_size = crate::array::paged_prefill_chunk_size();
        self.run_paged_prefill_chunk_with_size(
            suffix_tokens,
            first_logical_position,
            num_layers,
            positions,
            chunk_size,
        )
    }

    /// Chunk-size-parameterized worker for `run_paged_prefill_chunk`. The
    /// public entry point is a thin wrapper that reads
    /// `MLX_PAGED_PREFILL_CHUNK_SIZE` once via `OnceLock` and forwards. We
    /// expose this private helper so tests can drive both the single-shot
    /// path (`chunk_size <= 0`) and the chunked path (>0) without
    /// process-wide env mutation, and so we can directly verify numerical
    /// parity between them in the same test binary.
    ///
    /// `chunk_size <= 0` OR `suffix_tokens.len() <= chunk_size` takes the
    /// single-shot path. Anything else loops over `chunks(chunk_size)`.
    fn run_paged_prefill_chunk_with_size(
        &mut self,
        suffix_tokens: &[u32],
        first_logical_position: u32,
        num_layers: usize,
        positions: &MxArray,
        chunk_size: i32,
    ) -> Result<MxArray> {
        if suffix_tokens.is_empty() {
            return Err(napi::Error::from_reason(
                "run_paged_prefill_chunk called with empty suffix",
            ));
        }

        // Single-shot path: chunking disabled or suffix already small
        // enough that a single forward fits within the existing memory
        // budget.
        if chunk_size <= 0 || suffix_tokens.len() <= chunk_size as usize {
            return self.run_paged_prefill_single_shot(
                suffix_tokens,
                first_logical_position,
                num_layers,
                positions,
            );
        }

        // Chunked path. We slice the suffix into `chunk_size`-token chunks
        // and process each through ALL layers. The K/V is written into the
        // paged pool by `forward_paged_adapter` per chunk; later chunks
        // attend to the cumulative `[0, total_ctx)` via the
        // `cached_prefix_len > 0` branch of `forward_paged_adapter` (which
        // builds an explicit causal mask aligned at the suffix).
        let chunk_size_usize = chunk_size as usize;
        let total_chunks = suffix_tokens.len().div_ceil(chunk_size_usize);
        let mut last_hidden: Option<MxArray> = None;
        let mut tokens_consumed: u32 = 0;
        for (chunk_idx, chunk) in suffix_tokens.chunks(chunk_size_usize).enumerate() {
            let chunk_start_pos = first_logical_position + tokens_consumed;
            let is_last_chunk = chunk_idx + 1 == total_chunks;
            let hidden =
                self.run_paged_prefill_one_chunk(chunk, chunk_start_pos, num_layers, positions)?;
            tokens_consumed += chunk.len() as u32;

            if is_last_chunk {
                last_hidden = Some(hidden);
            } else {
                // Materialize the residual stream so MLX can release every
                // upstream node (embedding + per-layer attention/MLP
                // intermediates) before we start building the next chunk's
                // graph. Without this the lazy DAG accumulates across
                // chunks and defeats the entire memory-bounding purpose.
                hidden.eval();
                synchronize_and_clear_cache();
            }
        }
        let hidden_states = last_hidden.expect("chunked loop processed at least one chunk");

        // Final norm + lm_head ONLY on the last chunk's residual stream
        // (we only need the last token's logits to sample the first decode
        // token; intermediate chunks' vocab-projections would be discarded
        // anyway, and skipping them saves [chunk_len, vocab] worth of FLOPs
        // per non-final chunk).
        self.project_last_token_logits(&hidden_states)
    }

    /// Single-shot prefill: feed the entire suffix through every layer in
    /// one forward pass. Used both by the non-chunked path (chunk_size <= 0)
    /// and the chunked-path's "small enough to skip chunking" fast path.
    fn run_paged_prefill_single_shot(
        &mut self,
        suffix_tokens: &[u32],
        first_logical_position: u32,
        num_layers: usize,
        positions: &MxArray,
    ) -> Result<MxArray> {
        let hidden_states = self.run_paged_prefill_one_chunk(
            suffix_tokens,
            first_logical_position,
            num_layers,
            positions,
        )?;
        self.project_last_token_logits(&hidden_states)
    }

    /// Run a single prefill chunk through `record_tokens` + every layer's
    /// `forward_paged_adapter`. Returns the post-last-layer residual
    /// stream (NOT logits — caller decides whether to project to vocab).
    ///
    /// This is the per-chunk inner loop shared between the single-shot
    /// path and the chunked driver. It must NOT touch `final_norm` /
    /// `lm_head` so the chunked path can skip those on intermediate
    /// chunks.
    fn run_paged_prefill_one_chunk(
        &mut self,
        chunk_tokens: &[u32],
        chunk_first_position: u32,
        num_layers: usize,
        positions: &MxArray,
    ) -> Result<MxArray> {
        let chunk_len = chunk_tokens.len() as u32;
        // 1. record_tokens BEFORE forward (forward_paged_adapter expects
        //    the cursor to be advanced by the chunk so update_keys_values
        //    aligns).
        {
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                napi::Error::from_reason("run_paged_prefill_chunk: paged_adapter is None")
            })?;
            adapter
                .record_tokens(chunk_tokens)
                .map_err(napi::Error::from_reason)?;
        }

        // 2. Embed input ids.
        let input_ids = MxArray::from_uint32(chunk_tokens, &[1, chunk_len as i64])?;
        let mut hidden_states = self.embedding.forward(&input_ids)?;

        // 3. Run forward through every layer, dispatching the adapter.
        //    We have to split the borrow of `self.layers` (immutable) from
        //    `self.paged_adapter` (mutable) — the layer slice is captured
        //    as a separate reference and passed in, while the adapter is
        //    accessed via `self.paged_adapter` per-layer.
        //
        //    `cached_prefix_len = chunk_first_position`: every token
        //    already in the paged pool (prior cache hit + prior chunks
        //    written by earlier iterations of the chunked driver) lives at
        //    logical positions `[0, chunk_first_position)`. The new chunk
        //    occupies `[chunk_first_position, chunk_first_position+chunk_len)`.
        //    `forward_paged_adapter` already handles the `cached_prefix_len
        //    > 0` branch (Q = chunk only, K/V = cumulative `[0,
        //    total_ctx)`, with `create_causal_mask(num_tokens=chunk_len,
        //    offset=cached_prefix_len)` aligning the mask at the suffix).
        let cached_prefix_len = chunk_first_position;
        for layer_idx in 0..num_layers {
            // Re-borrow per layer to avoid holding the mutable borrow
            // across the immutable layer access. `self.layers[idx]` and
            // `self.paged_adapter` are disjoint fields, so we use a
            // raw-ptr split borrow trick: read the layer reference up
            // front, then re-borrow the adapter.
            let layer: &TransformerBlock = unsafe {
                let ptr = self.layers.as_ptr().add(layer_idx);
                &*ptr
            };
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                napi::Error::from_reason("run_paged_prefill_chunk: paged_adapter dropped")
            })?;
            hidden_states = layer.forward_paged_adapter(
                &hidden_states,
                adapter,
                layer_idx as u32,
                chunk_first_position,
                cached_prefix_len,
                self.config.num_heads as u32,
                positions,
                /* num_seqs */ 1,
                /* seq_len */ chunk_len as i64,
                /* is_prefill */ true,
            )?;
            // Smooth the prefill memory peak: every K layers, materialize the
            // residual stream so MLX can release the upstream graph nodes
            // (embedding + every prior layer's attention/MLP intermediates)
            // from the cache pool. Without this the in-flight lazy graph
            // accumulates ~50 GB on long contexts before the post-prefill
            // sync fires. Cadence is `MLX_PAGED_PREFILL_EVAL_INTERVAL` (default 8).
            crate::array::maybe_eval_clear_for_paged_prefill_layer(layer_idx, &hidden_states)?;
        }
        Ok(hidden_states)
    }

    /// Project the per-token residual stream through `final_norm` + the
    /// LM head and slice the last position's logits down to `[vocab]`.
    /// Shared between the single-shot path and the chunked path's
    /// final-chunk return path. Hidden state shape on entry: `[1,
    /// chunk_len, hidden]`.
    fn project_last_token_logits(&self, hidden_states: &MxArray) -> Result<MxArray> {
        let normed = self.final_norm.forward(hidden_states)?;
        let logits = if self.config.tie_word_embeddings {
            let embedding_weight = self.embedding.get_weight();
            normed.matmul(&embedding_weight.transpose(Some(&[1, 0]))?)?
        } else {
            self.lm_head.forward(&normed)?
        };
        // Slice last token: logits shape [1, chunk_len, vocab] -> [vocab].
        let seq_len = logits.shape_at(1)?;
        let last = logits
            .slice_axis(1, seq_len - 1, seq_len)?
            .squeeze(Some(&[0, 1]))?;
        Ok(last)
    }

    /// Run one decode step through the paged forward path. Mirrors
    /// `run_paged_prefill_chunk` for a single-token chunk.
    fn run_paged_decode_step(
        &mut self,
        token_value: u32,
        num_layers: usize,
        positions: &MxArray,
    ) -> Result<MxArray> {
        // 1. record_tokens for the new token. The decode-step's logical
        //    position is `current_token_count` BEFORE record (=
        //    after-record - 1).
        let first_logical_position = {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                napi::Error::from_reason("run_paged_decode_step: paged_adapter is None")
            })?;
            adapter.current_token_count()
        };
        {
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                napi::Error::from_reason("run_paged_decode_step: paged_adapter dropped")
            })?;
            adapter
                .record_tokens(&[token_value])
                .map_err(napi::Error::from_reason)?;
        }

        // 2. Embed and forward.
        let input_ids = MxArray::from_uint32(&[token_value], &[1, 1])?;
        let mut hidden_states = self.embedding.forward(&input_ids)?;

        for layer_idx in 0..num_layers {
            let layer: &TransformerBlock = unsafe {
                let ptr = self.layers.as_ptr().add(layer_idx);
                &*ptr
            };
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                napi::Error::from_reason("run_paged_decode_step: paged_adapter dropped")
            })?;
            hidden_states = layer.forward_paged_adapter(
                &hidden_states,
                adapter,
                layer_idx as u32,
                first_logical_position,
                /* cached_prefix_len */ 0, // unused in decode path
                self.config.num_heads as u32,
                positions,
                /* num_seqs */ 1,
                /* seq_len */ 1,
                /* is_prefill */ false,
            )?;
        }

        // 3. Final norm + lm_head.
        hidden_states = self.final_norm.forward(&hidden_states)?;
        let logits = if self.config.tie_word_embeddings {
            let embedding_weight = self.embedding.get_weight();
            hidden_states.matmul(&embedding_weight.transpose(Some(&[1, 0]))?)?
        } else {
            self.lm_head.forward(&hidden_states)?
        };
        Ok(logits)
    }

    /// Generate synchronous (runs on model thread).
    fn generate_sync(
        &mut self,
        messages: Vec<ChatMessage>,
        config: Option<GenerationConfig>,
    ) -> Result<GenerationResult> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("Tokenizer not available."))?
            .clone();

        let formatted = messages
            .iter()
            .map(|msg| format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content))
            .chain(iter::once("<|im_start|>assistant\n".to_string()))
            .collect::<String>();

        let token_ids = tokenizer.encode_sync(&formatted, Some(false))?;
        let input_ids = MxArray::from_uint32(&token_ids, &[1, token_ids.len() as i64])?;

        // Use generate_for_training_sync on the NAPI model (which uses Arc<RwLock<>>)
        // But since we're on the model thread, we do a direct implementation here.
        //
        // `GenerationConfig::default()` bakes `Some(builtin)` sampling values, so
        // unwrapping an OMITTED config would shadow the checkpoint's
        // generation_config.json defaults (a baked `Some(1.0)` would win the
        // `.or(gen_defaults)` fold). Capture the caller's ACTUAL sampling fields
        // first — `None` whenever the field, or the whole config, was omitted —
        // so an unspecified field still falls back to the model default and only
        // an explicit request value wins. Non-sampling fields keep using the
        // builtin-filled `config` below.
        let req_temperature = config.as_ref().and_then(|c| c.temperature);
        let req_top_k = config.as_ref().and_then(|c| c.top_k);
        let req_top_p = config.as_ref().and_then(|c| c.top_p);
        let req_min_p = config.as_ref().and_then(|c| c.min_p);
        let req_repetition_penalty = config.as_ref().and_then(|c| c.repetition_penalty);
        let config = config.unwrap_or_default();
        let max_new_tokens = config.max_new_tokens.unwrap_or(2048);
        // Reject a nonpositive budget on the public `generate()` API (parity
        // with qwen3_5's `generate` guard): without this a negative `i32`
        // reaches `Vec::with_capacity(.. as usize)` below
        // (`-1i32 as usize == usize::MAX` → capacity-overflow abort).
        if max_new_tokens <= 0 {
            return Err(Error::from_reason(format!(
                "max_new_tokens must be > 0, got {}",
                max_new_tokens
            )));
        }
        // Request value wins; otherwise fall back to the checkpoint's
        // generation_config.json default; otherwise the sampler's builtin.
        // When the request omits temperature, a `do_sample:false` in
        // generation_config.json forces greedy decoding (temperature 0),
        // overriding any gen-config temperature (HuggingFace transformers
        // semantics) — `effective_temperature()` folds that rule in.
        let temperature = req_temperature
            .or(self.gen_defaults.effective_temperature())
            .unwrap_or(1.0);
        let top_k = req_top_k.or(self.gen_defaults.top_k).unwrap_or(0);
        let top_p = req_top_p.or(self.gen_defaults.top_p).unwrap_or(1.0);
        let min_p = req_min_p.or(self.gen_defaults.min_p).unwrap_or(0.0);
        let repetition_penalty = req_repetition_penalty
            .or(self.gen_defaults.repetition_penalty)
            .unwrap_or(1.0);
        let repetition_context_size = config.repetition_context_size.unwrap_or(256);
        let presence_penalty = config.presence_penalty.unwrap_or(0.0);
        let presence_context_size = config.presence_context_size.unwrap_or(20);
        let frequency_penalty = config.frequency_penalty.unwrap_or(0.0);
        let frequency_context_size = config.frequency_context_size.unwrap_or(20);
        let max_consecutive_tokens = config
            .max_consecutive_tokens
            .unwrap_or(crate::sampling::DEFAULT_MAX_CONSECUTIVE_TOKENS);
        let max_ngram_repeats = config
            .max_ngram_repeats
            .unwrap_or(crate::sampling::DEFAULT_MAX_NGRAM_REPEATS);
        let ngram_size = config
            .ngram_size
            .unwrap_or(crate::sampling::DEFAULT_NGRAM_SIZE);
        let eos_token_id = config.eos_token_id.or(Some(self.config.eos_token_id));
        let return_logprobs = config.return_logprobs.unwrap_or(true);
        let prefill_step_size = config.prefill_step_size.unwrap_or(2048) as usize;

        let embedding_weight = self.embedding.get_weight();
        let layers = &self.layers;
        let final_norm = &self.final_norm;
        let lm_head = &self.lm_head;
        let model_config = &self.config;

        let generation_stream = Stream::new(DeviceType::Gpu);

        let num_layers = layers.len();
        let mut kv_keys: Vec<Option<MxArray>> = vec![None; num_layers];
        let mut kv_values: Vec<Option<MxArray>> = vec![None; num_layers];
        let mut cache_idx: i32 = 0;

        let mut rope_offsets = MxArray::from_int32(&[0], &[1])?;
        let left_padding = MxArray::from_int32(&[0], &[1])?;

        let input_tokens = input_ids.to_uint32()?;
        let current_ids = input_ids;
        let mut generated_tokens: Vec<u32> =
            Vec::with_capacity(engine::generated_capacity_hint(max_new_tokens));
        let mut generated_logprobs: Vec<f32> = if return_logprobs {
            Vec::with_capacity(engine::generated_capacity_hint(max_new_tokens))
        } else {
            Vec::new()
        };
        let mut finish_reason = "length";

        // Extra stop ids from generation_config.json (e.g. a second EOS).
        // Captured into a local so the decode loop's `&mut self` borrow does
        // not conflict with reading `self.gen_defaults`.
        let extra_eos_ids = self.gen_defaults.eos_token_ids.clone();

        let sampling_config = SamplingConfig {
            temperature: Some(temperature),
            top_k: Some(top_k),
            top_p: Some(top_p),
            min_p: Some(min_p),
        };

        // PREFILL
        let total_seq_len = current_ids.shape_at(1)? as usize;
        let use_chunked_prefill = prefill_step_size > 0 && total_seq_len > prefill_step_size;
        let mut last_logits = if use_chunked_prefill {
            let mut offset = 0usize;
            while offset + prefill_step_size < total_seq_len {
                let chunk_end = offset + prefill_step_size;
                let chunk = current_ids.slice(&[0, offset as i64], &[1, chunk_end as i64])?;
                rope_offsets = MxArray::from_int32(&[offset as i32], &[1])?;
                {
                    let _stream_ctx = StreamContext::new(generation_stream);
                    let _ = Qwen3Model::forward_fused(
                        &chunk,
                        &embedding_weight,
                        layers,
                        final_norm,
                        lm_head,
                        model_config,
                        &mut kv_keys,
                        &mut kv_values,
                        &mut cache_idx,
                        &rope_offsets,
                        &left_padding,
                    )?;
                }
                for kv_key in kv_keys.iter().flatten() {
                    kv_key.eval();
                }
                for kv_value in kv_values.iter().flatten() {
                    kv_value.eval();
                }
                synchronize_and_clear_cache();
                offset = chunk_end;
            }
            let final_chunk = current_ids.slice(&[0, offset as i64], &[1, total_seq_len as i64])?;
            rope_offsets = MxArray::from_int32(&[offset as i32], &[1])?;
            let logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                Qwen3Model::forward_fused(
                    &final_chunk,
                    &embedding_weight,
                    layers,
                    final_norm,
                    lm_head,
                    model_config,
                    &mut kv_keys,
                    &mut kv_values,
                    &mut cache_idx,
                    &rope_offsets,
                    &left_padding,
                )?
            };
            let chunk_seq_len = logits.shape_at(1)?;
            logits
                .slice_axis(1, chunk_seq_len - 1, chunk_seq_len)?
                .squeeze(Some(&[0, 1]))?
        } else {
            let logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                Qwen3Model::forward_fused(
                    &current_ids,
                    &embedding_weight,
                    layers,
                    final_norm,
                    lm_head,
                    model_config,
                    &mut kv_keys,
                    &mut kv_values,
                    &mut cache_idx,
                    &rope_offsets,
                    &left_padding,
                )?
            };
            let seq_len = logits.shape_at(1)?;
            logits
                .slice_axis(1, seq_len - 1, seq_len)?
                .squeeze(Some(&[0, 1]))?
        };

        rope_offsets = MxArray::from_int32(&[cache_idx], &[1])?;

        if repetition_penalty != 1.0 && !input_tokens.is_empty() {
            last_logits = apply_repetition_penalty(
                &last_logits,
                &input_tokens,
                repetition_penalty,
                Some(repetition_context_size),
            )?;
        }
        if presence_penalty != 0.0 {
            last_logits = apply_presence_penalty(
                &last_logits,
                &input_tokens,
                presence_penalty,
                Some(presence_context_size),
            )?;
        }
        if frequency_penalty != 0.0 {
            last_logits = apply_frequency_penalty(
                &last_logits,
                &input_tokens,
                frequency_penalty,
                Some(frequency_context_size),
            )?;
        }

        let (mut token, mut logprobs) = if return_logprobs {
            let (tok, lp) = sample_and_logprobs(&last_logits, Some(sampling_config))?;
            (tok, Some(lp))
        } else {
            (sample(&last_logits, Some(sampling_config))?, None)
        };

        // DECODE
        const DECODE_CLEANUP_INTERVAL: i32 = 256;
        let one_arr = MxArray::from_int32(&[1], &[1])?;
        for step in 0..max_new_tokens {
            let _stream_ctx = StreamContext::new(generation_stream);
            token.eval();
            if step > 0 && step % DECODE_CLEANUP_INTERVAL == 0 {
                synchronize_and_clear_cache();
            }
            let token_value = token.item_at_int32(0)? as u32;
            generated_tokens.push(token_value);
            if return_logprobs && let Some(ref lp) = logprobs {
                lp.eval();
                let token_logprob = lp.item_at_float32(token_value as usize)?;
                generated_logprobs.push(token_logprob);
            }
            if let Some(reason) = check_repetition_cutoff(
                &generated_tokens,
                max_consecutive_tokens,
                max_ngram_repeats,
                ngram_size,
            ) {
                finish_reason = reason;
                break;
            }
            if let Some(eos_id) = eos_token_id
                && token_value == eos_id as u32
            {
                finish_reason = "stop";
                break;
            }
            // Honor extra stop ids declared in generation_config.json.
            if extra_eos_ids.contains(&token_value) {
                finish_reason = "stop";
                break;
            }
            let next_input = MxArray::from_uint32(&[token_value], &[1, 1])?;
            let next_logits = Qwen3Model::forward_fused(
                &next_input,
                &embedding_weight,
                layers,
                final_norm,
                lm_head,
                model_config,
                &mut kv_keys,
                &mut kv_values,
                &mut cache_idx,
                &rope_offsets,
                &left_padding,
            )?;
            rope_offsets = rope_offsets.add(&one_arr)?;
            let next_last_logits = next_logits.slice_axis(1, 0, 1)?.squeeze(Some(&[0, 1]))?;
            last_logits = next_last_logits;
            if repetition_penalty != 1.0 || presence_penalty != 0.0 || frequency_penalty != 0.0 {
                let context_tokens: Vec<u32> = input_tokens
                    .iter()
                    .copied()
                    .chain(generated_tokens.iter().copied())
                    .collect();
                if repetition_penalty != 1.0 {
                    last_logits = apply_repetition_penalty(
                        &last_logits,
                        &context_tokens,
                        repetition_penalty,
                        Some(repetition_context_size),
                    )?;
                }
                if presence_penalty != 0.0 {
                    last_logits = apply_presence_penalty(
                        &last_logits,
                        &context_tokens,
                        presence_penalty,
                        Some(presence_context_size),
                    )?;
                }
                if frequency_penalty != 0.0 {
                    last_logits = apply_frequency_penalty(
                        &last_logits,
                        &context_tokens,
                        frequency_penalty,
                        Some(frequency_context_size),
                    )?;
                }
            }
            let (next_tok, next_lp) = if return_logprobs {
                let (tok, lp) = sample_and_logprobs(&last_logits, Some(sampling_config))?;
                (tok, Some(lp))
            } else {
                (sample(&last_logits, Some(sampling_config))?, None)
            };
            token = next_tok;
            logprobs = next_lp;
        }

        let tokens_array =
            MxArray::from_uint32(&generated_tokens, &[generated_tokens.len() as i64])?;
        let logprobs_array = if return_logprobs {
            MxArray::from_float32(&generated_logprobs, &[generated_logprobs.len() as i64])?
        } else {
            MxArray::from_float32(&[], &[0])?
        };

        let generated_ids_vec: Vec<u32> = generated_tokens.clone();
        let text = tokenizer.decode_sync(&generated_ids_vec, true)?;

        Ok(GenerationResult {
            text,
            tokens: tokens_array,
            logprobs: logprobs_array,
            finish_reason: finish_reason.to_string(),
            num_tokens: generated_tokens.len(),
        })
    }

    /// Generate batch synchronous (runs on model thread).
    fn generate_batch_sync(
        &mut self,
        prompts: Vec<Vec<ChatMessage>>,
        group_size: u32,
        config: Option<GenerationConfig>,
    ) -> Result<BatchGenerationResult> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("Tokenizer not available."))?
            .clone();

        let num_prompts = prompts.len();
        let group_size_usize = group_size as usize;

        // Tokenize all prompts
        let mut prompt_token_arrays = Vec::with_capacity(num_prompts);
        for messages in &prompts {
            let formatted = messages
                .iter()
                .map(|msg| format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content))
                .chain(iter::once("<|im_start|>assistant\n".to_string()))
                .collect::<String>();
            let token_ids = tokenizer.encode_sync(&formatted, Some(false))?;
            let prompt_tokens = MxArray::from_uint32(&token_ids, &[1, token_ids.len() as i64])?;
            prompt_token_arrays.push(prompt_tokens);
        }

        // Pre-build lightweight message copies (ChatMessage has Uint8Array which can't Clone)
        let lightweight_prompts: Vec<Vec<ChatMessage>> = prompts
            .iter()
            .map(|msgs| {
                msgs.iter()
                    .map(|m| ChatMessage {
                        role: m.role.clone(),
                        content: m.content.clone(),
                        tool_calls: m.tool_calls.clone(),
                        tool_call_id: m.tool_call_id.clone(),
                        is_error: m.is_error,
                        reasoning_content: m.reasoning_content.clone(),
                        images: None,
                        audio: None,
                    })
                    .collect()
            })
            .collect();

        // Generate N*G completions sequentially (using the existing generate_sync logic)
        let mut all_tokens = Vec::with_capacity(num_prompts * group_size_usize);
        let mut all_logprobs = Vec::with_capacity(num_prompts * group_size_usize);
        let mut all_finish_reasons = Vec::with_capacity(num_prompts);
        let mut all_token_counts = Vec::with_capacity(num_prompts);

        for (prompt_idx, _prompt_tokens) in prompt_token_arrays.iter().enumerate() {
            let mut prompt_finish_reasons = Vec::with_capacity(group_size_usize);
            let mut prompt_token_counts = Vec::with_capacity(group_size_usize);

            for _group_idx in 0..group_size {
                // Reconstruct lightweight messages for each call (generate_sync only uses role+content)
                let msgs = lightweight_prompts[prompt_idx]
                    .iter()
                    .map(|m| ChatMessage {
                        role: m.role.clone(),
                        content: m.content.clone(),
                        tool_calls: m.tool_calls.clone(),
                        tool_call_id: m.tool_call_id.clone(),
                        is_error: m.is_error,
                        reasoning_content: m.reasoning_content.clone(),
                        images: None,
                        audio: None,
                    })
                    .collect();
                let result = self.generate_sync(msgs, config.clone())?;
                all_tokens.push(result.tokens);
                all_logprobs.push(result.logprobs);
                prompt_finish_reasons.push(result.finish_reason);
                prompt_token_counts.push(result.num_tokens as u32);
            }

            all_finish_reasons.push(prompt_finish_reasons);
            all_token_counts.push(prompt_token_counts);
        }

        // Decode all texts
        let mut decoded_texts = Vec::with_capacity(all_tokens.len());
        for token_array in &all_tokens {
            let generated_ids = token_array.to_uint32()?;
            let decoded = tokenizer.decode_sync(&generated_ids, true)?;
            decoded_texts.push(decoded);
        }

        Ok(BatchGenerationResult {
            tokens: all_tokens,
            logprobs: all_logprobs,
            texts: decoded_texts,
            finish_reasons: all_finish_reasons,
            token_counts: all_token_counts,
            num_prompts,
            group_size,
        })
    }

    // ========== Training methods (run on model thread) ==========

    /// Initialize training state with optimizer and configuration.
    /// Inherent body of [`TrainBackend::init_training_sync`].
    fn init_training_sync_impl(
        &mut self,
        config: crate::grpo::engine::GRPOEngineConfig,
        _model_type: ModelType,
    ) -> Result<()> {
        if self.training_state.is_some() {
            return Err(napi::Error::from_reason(
                "Training state already initialized. A single model thread can host only one active training run.",
            ));
        }
        let optimizer = if config.optimizer_type.as_deref().unwrap_or("adamw") == "adamw" {
            Some(crate::optimizers::AdamW::new(
                config.learning_rate,
                config.adamw_beta1,
                config.adamw_beta2,
                config.adamw_eps,
                config.weight_decay,
                Some(true), // bias correction
            ))
        } else {
            None
        };

        self.training_state = Some(crate::training_state::ModelThreadTrainingState::new(
            config.learning_rate.unwrap_or(1e-6),
            config.gradient_accumulation_steps.unwrap_or(1),
            config.gradient_clip_norm,
            config.gradient_clip_value,
            config.max_nan_gradients.unwrap_or(100),
            config.emergency_save_threshold.unwrap_or(5),
            config.verbose_nan_detection.unwrap_or(false),
            config.gradient_checkpointing.unwrap_or(true),
            optimizer,
        ));
        info!("Training state initialized on model thread");
        Ok(())
    }

    /// Inherent body of [`TrainBackend::save_optimizer_state_sync`].
    fn save_optimizer_state_sync_impl(&self, path: String) -> Result<()> {
        let ts = self.training_state.as_ref().ok_or_else(|| {
            napi::Error::from_reason("Training state not initialized. Call InitTraining first.")
        })?;
        ts.save_optimizer_state_sync(&path)
    }

    /// Save the model weights and configuration to disk (runs on model thread).
    ///
    /// Collects all parameters directly from this `Qwen3Inner`'s fields (which
    /// are owned by the dedicated model thread), validates them for NaN/Inf,
    /// writes them as SafeTensors, and emits a `config.json` tagged with
    /// `model_type: "qwen3"` for `detectModelType` on reload.
    pub(crate) fn save_model_sync(&self, save_path: &str) -> Result<()> {
        let mut params: HashMap<String, MxArray> = HashMap::new();

        // Embedding
        params.insert("embedding.weight".to_string(), self.embedding.get_weight());

        // Transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{}", i);

            let attn = &layer.self_attn;
            params.insert(
                format!("{}.self_attn.q_proj.weight", prefix),
                attn.get_q_proj_weight(),
            );
            params.insert(
                format!("{}.self_attn.k_proj.weight", prefix),
                attn.get_k_proj_weight(),
            );
            params.insert(
                format!("{}.self_attn.v_proj.weight", prefix),
                attn.get_v_proj_weight(),
            );
            params.insert(
                format!("{}.self_attn.o_proj.weight", prefix),
                attn.get_o_proj_weight(),
            );

            // QK norm parameters (if enabled)
            if self.config.use_qk_norm {
                if let Some(q_norm_weight) = attn.get_q_norm_weight() {
                    params.insert(format!("{}.self_attn.q_norm.weight", prefix), q_norm_weight);
                }
                if let Some(k_norm_weight) = attn.get_k_norm_weight() {
                    params.insert(format!("{}.self_attn.k_norm.weight", prefix), k_norm_weight);
                }
            }

            let mlp = &layer.mlp;
            params.insert(
                format!("{}.mlp.gate_proj.weight", prefix),
                mlp.get_gate_proj_weight(),
            );
            params.insert(
                format!("{}.mlp.up_proj.weight", prefix),
                mlp.get_up_proj_weight(),
            );
            params.insert(
                format!("{}.mlp.down_proj.weight", prefix),
                mlp.get_down_proj_weight(),
            );

            params.insert(
                format!("{}.input_layernorm.weight", prefix),
                layer.get_input_layernorm_weight(),
            );
            params.insert(
                format!("{}.post_attention_layernorm.weight", prefix),
                layer.get_post_attention_layernorm_weight(),
            );
        }

        // Final norm
        params.insert(
            "final_norm.weight".to_string(),
            self.final_norm.get_weight(),
        );

        // LM head (only if not tied to embeddings)
        if !self.config.tie_word_embeddings {
            params.insert("lm_head.weight".to_string(), self.lm_head.get_weight());
        }

        // Validate every parameter for NaN/Inf before touching the filesystem.
        for (name, param) in params.iter() {
            let data = param.to_float32()?;
            let invalid_count = data
                .iter()
                .filter(|v| v.is_nan() || v.is_infinite())
                .count();
            if invalid_count > 0 {
                return Err(napi::Error::new(
                    napi::Status::GenericFailure,
                    format!(
                        "Cannot save model: parameter '{}' contains {} NaN/Inf values. \
                        Model weights are corrupted, likely due to training instability. \
                        Consider reducing learning rate or using an earlier checkpoint.",
                        name, invalid_count
                    ),
                ));
            }
        }

        let mut params_clone: HashMap<String, MxArray> =
            params.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

        // Build weights.mlx metadata (shape + dtype only; full data is in safetensors).
        let mut weights_metadata = serde_json::Map::new();
        for (key, array) in params.iter() {
            let shape_data = array.shape()?;
            let shape: Vec<i64> = shape_data.as_ref().to_vec();
            let dtype = array.dtype()?;
            let mut param_info = serde_json::Map::new();
            param_info.insert("shape".to_string(), serde_json::json!(shape));
            param_info.insert("dtype".to_string(), serde_json::json!(dtype as i32));
            weights_metadata.insert(key.clone(), serde_json::Value::Object(param_info));
        }

        // Config JSON — inject `model_type: "qwen3"` so `detectModelType`
        // routes the saved directory back to the Qwen3 loader.
        let mut config_value = serde_json::to_value(&self.config).map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Failed to serialize config: {e}"),
            )
        })?;
        if let serde_json::Value::Object(ref mut map) = config_value {
            map.insert("model_type".to_string(), serde_json::json!("qwen3"));
        }

        let weights_json = serde_json::json!({
            "version": "1.0",
            "config": config_value,
            "weights": weights_metadata,
            "note": "Full weights are in weights.safetensors"
        });

        let path = std::path::Path::new(save_path);
        std::fs::create_dir_all(path)?;

        info!("Saving model to {}", save_path);

        let config_path = path.join("config.json");
        let config_json = serde_json::to_string_pretty(&config_value)?;
        std::fs::write(&config_path, config_json)?;
        info!("Saved config.json");

        let safetensors_path = path.join("weights.safetensors");
        let metadata = Some(serde_json::json!({
            "format": "mlx-node",
            "version": "1.0"
        }));
        crate::utils::safetensors::save_safetensors(
            &safetensors_path,
            &mut params_clone,
            metadata,
        )?;
        info!("Saved weights.safetensors");

        let weights_str = serde_json::to_string_pretty(&weights_json)?;
        let weights_path = path.join("weights.mlx");
        std::fs::write(&weights_path, weights_str)?;
        info!("Saved weights.mlx metadata");

        Ok(())
    }

    /// Inherent body of [`TrainBackend::load_optimizer_state_sync`].
    fn load_optimizer_state_sync_impl(&mut self, path: String) -> Result<()> {
        let ts = self.training_state.as_mut().ok_or_else(|| {
            napi::Error::from_reason("Training state not initialized. Call InitTraining first.")
        })?;
        ts.load_optimizer_state_sync(&path)
    }

    /// Generate completions for training.
    ///
    /// Tokenizes prompts using Jinja2 chat template, generates completions,
    /// caches MxArray results in training_state for the subsequent training step,
    /// and returns plain data across the thread boundary.
    /// Inherent body of [`TrainBackend::generate_for_training_thread_sync`].
    fn generate_for_training_thread_sync_impl(
        &mut self,
        prompts: Vec<Vec<ChatMessage>>,
        group_size: usize,
        gen_config: super::GenerationConfig,
        enable_thinking: Option<bool>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Result<crate::training_model::GenerationPlainData> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("Tokenizer not available."))?
            .clone();

        let num_prompts = prompts.len();
        let total_completions = num_prompts * group_size;

        let mut completion_texts = Vec::with_capacity(total_completions);
        let mut prompt_texts = Vec::with_capacity(total_completions);
        let mut completion_tokens_plain = Vec::with_capacity(total_completions);
        let mut completion_logprobs_plain = Vec::with_capacity(total_completions);
        let mut token_counts = Vec::with_capacity(total_completions);
        let mut finish_reasons = Vec::with_capacity(total_completions);

        // Cache MxArrays for the training step
        let mut cached_prompt_tokens: Vec<MxArray> = Vec::with_capacity(num_prompts);
        let mut cached_completion_tokens: Vec<MxArray> = Vec::with_capacity(total_completions);
        let mut cached_completion_logprobs: Vec<MxArray> = Vec::with_capacity(total_completions);

        for prompt_messages in prompts.iter() {
            // Tokenize the prompt using Jinja2 chat template (supports tools + thinking)
            let prompt_token_ids = tokenizer.apply_chat_template_sync(
                prompt_messages,
                Some(true),
                tools.as_deref(),
                enable_thinking,
            )?;

            let prompt_array =
                MxArray::from_uint32(&prompt_token_ids, &[1, prompt_token_ids.len() as i64])?;
            let prompt_array_1d = prompt_array.squeeze(Some(&[0]))?;
            let prompt_text = tokenizer.decode_sync(&prompt_token_ids, true)?;

            // Generate group_size completions for this prompt
            for _g in 0..group_size {
                let result = self
                    .generate_single_for_training_sync(&prompt_array, Some(gen_config.clone()))?;

                // Extract plain data for crossing thread boundary
                let tok_ids: Vec<i32> = result
                    .tokens
                    .to_uint32()?
                    .iter()
                    .map(|&t| t as i32)
                    .collect();
                let lp_data: Vec<f32> = result.logprobs.to_float32()?.to_vec();
                let decoded = tokenizer
                    .decode_sync(&tok_ids.iter().map(|&t| t as u32).collect::<Vec<_>>(), true)?;

                completion_texts.push(decoded);
                prompt_texts.push(prompt_text.clone());
                completion_tokens_plain.push(tok_ids);
                completion_logprobs_plain.push(lp_data);
                token_counts.push(result.num_tokens as u32);
                finish_reasons.push(result.finish_reason.clone());

                // Cache MxArrays (these stay on the model thread)
                cached_completion_tokens.push(result.tokens);
                cached_completion_logprobs.push(result.logprobs);

                // Clean up between completions to prevent Metal context accumulation
                heavy_cleanup();
            }

            cached_prompt_tokens.push(prompt_array_1d);
        }

        // Store cached MxArrays in training_state (prompt-major layout)
        if let Some(ref mut ts) = self.training_state {
            ts.cached_prompt_tokens = Some(cached_prompt_tokens);
            ts.cached_completion_tokens = Some(cached_completion_tokens);
            ts.cached_completion_logprobs = Some(cached_completion_logprobs);
        }

        Ok(crate::training_model::GenerationPlainData {
            completion_texts,
            prompt_texts,
            completion_tokens: completion_tokens_plain,
            completion_logprobs: completion_logprobs_plain,
            token_counts,
            finish_reasons,
        })
    }

    /// Generate a single completion for training purposes.
    ///
    /// Uses fresh local KV caches (not the shared inference caches).
    /// Returns GenerationResult with MxArray tokens and logprobs.
    fn generate_single_for_training_sync(
        &self,
        input_ids: &MxArray,
        config: Option<super::GenerationConfig>,
    ) -> Result<GenerationResult> {
        let config = config.unwrap_or_default();
        let max_new_tokens = config.max_new_tokens.unwrap_or(2048);
        let temperature = config.temperature.unwrap_or(1.0);
        let top_k = config.top_k.unwrap_or(0);
        let top_p = config.top_p.unwrap_or(1.0);
        let min_p = config.min_p.unwrap_or(0.0);
        let repetition_penalty = config.repetition_penalty.unwrap_or(1.0);
        let repetition_context_size = config.repetition_context_size.unwrap_or(256);
        let presence_penalty = config.presence_penalty.unwrap_or(0.0);
        let presence_context_size = config.presence_context_size.unwrap_or(20);
        let frequency_penalty = config.frequency_penalty.unwrap_or(0.0);
        let frequency_context_size = config.frequency_context_size.unwrap_or(20);
        let max_consecutive_tokens = config
            .max_consecutive_tokens
            .unwrap_or(crate::sampling::DEFAULT_MAX_CONSECUTIVE_TOKENS);
        let max_ngram_repeats = config
            .max_ngram_repeats
            .unwrap_or(crate::sampling::DEFAULT_MAX_NGRAM_REPEATS);
        let ngram_size = config
            .ngram_size
            .unwrap_or(crate::sampling::DEFAULT_NGRAM_SIZE);
        let eos_token_id = config.eos_token_id.or(Some(self.config.eos_token_id));
        let return_logprobs = config.return_logprobs.unwrap_or(true);
        let prefill_step_size = config.prefill_step_size.unwrap_or(2048) as usize;

        let embedding_weight = self.embedding.get_weight();
        let layers = &self.layers;
        let final_norm = &self.final_norm;
        let lm_head = &self.lm_head;
        let model_config = &self.config;

        let generation_stream = Stream::new(DeviceType::Gpu);

        let num_layers = layers.len();
        let mut kv_keys: Vec<Option<MxArray>> = vec![None; num_layers];
        let mut kv_values: Vec<Option<MxArray>> = vec![None; num_layers];
        let mut cache_idx: i32 = 0;

        let mut rope_offsets = MxArray::from_int32(&[0], &[1])?;
        let left_padding = MxArray::from_int32(&[0], &[1])?;

        let input_tokens = input_ids.to_uint32()?;
        let current_ids = input_ids.clone();
        // Clamp the capacity hint to 0: this training-only path takes
        // `max_new_tokens` from training config (SFT/GRPO) where panics are
        // banned; a negative value would make `.. as usize` wrap to
        // `usize::MAX` and abort. The `0..max_new_tokens` loop below already
        // yields nothing for a nonpositive budget, so this only prevents the
        // abort without changing behavior for valid budgets.
        let mut generated_tokens: Vec<u32> =
            Vec::with_capacity(engine::generated_capacity_hint(max_new_tokens));
        let mut generated_logprobs: Vec<f32> = if return_logprobs {
            Vec::with_capacity(engine::generated_capacity_hint(max_new_tokens))
        } else {
            Vec::new()
        };
        let mut finish_reason = "length";

        let sampling_config = SamplingConfig {
            temperature: Some(temperature),
            top_k: Some(top_k),
            top_p: Some(top_p),
            min_p: Some(min_p),
        };

        // PREFILL
        let total_seq_len = current_ids.shape_at(1)? as usize;
        let use_chunked_prefill = prefill_step_size > 0 && total_seq_len > prefill_step_size;
        let mut last_logits = if use_chunked_prefill {
            let mut offset = 0usize;
            while offset + prefill_step_size < total_seq_len {
                let chunk_end = offset + prefill_step_size;
                let chunk = current_ids.slice(&[0, offset as i64], &[1, chunk_end as i64])?;
                rope_offsets = MxArray::from_int32(&[offset as i32], &[1])?;
                {
                    let _stream_ctx = StreamContext::new(generation_stream);
                    let _ = Qwen3Model::forward_fused(
                        &chunk,
                        &embedding_weight,
                        layers,
                        final_norm,
                        lm_head,
                        model_config,
                        &mut kv_keys,
                        &mut kv_values,
                        &mut cache_idx,
                        &rope_offsets,
                        &left_padding,
                    )?;
                }
                for kv_key in kv_keys.iter().flatten() {
                    kv_key.eval();
                }
                for kv_value in kv_values.iter().flatten() {
                    kv_value.eval();
                }
                synchronize_and_clear_cache();
                offset = chunk_end;
            }
            let final_chunk = current_ids.slice(&[0, offset as i64], &[1, total_seq_len as i64])?;
            rope_offsets = MxArray::from_int32(&[offset as i32], &[1])?;
            let logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                Qwen3Model::forward_fused(
                    &final_chunk,
                    &embedding_weight,
                    layers,
                    final_norm,
                    lm_head,
                    model_config,
                    &mut kv_keys,
                    &mut kv_values,
                    &mut cache_idx,
                    &rope_offsets,
                    &left_padding,
                )?
            };
            let chunk_seq_len = logits.shape_at(1)?;
            logits
                .slice_axis(1, chunk_seq_len - 1, chunk_seq_len)?
                .squeeze(Some(&[0, 1]))?
        } else {
            let logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                Qwen3Model::forward_fused(
                    &current_ids,
                    &embedding_weight,
                    layers,
                    final_norm,
                    lm_head,
                    model_config,
                    &mut kv_keys,
                    &mut kv_values,
                    &mut cache_idx,
                    &rope_offsets,
                    &left_padding,
                )?
            };
            let seq_len = logits.shape_at(1)?;
            logits
                .slice_axis(1, seq_len - 1, seq_len)?
                .squeeze(Some(&[0, 1]))?
        };

        rope_offsets = MxArray::from_int32(&[cache_idx], &[1])?;

        if repetition_penalty != 1.0 && !input_tokens.is_empty() {
            last_logits = apply_repetition_penalty(
                &last_logits,
                &input_tokens,
                repetition_penalty,
                Some(repetition_context_size),
            )?;
        }
        if presence_penalty != 0.0 {
            last_logits = apply_presence_penalty(
                &last_logits,
                &input_tokens,
                presence_penalty,
                Some(presence_context_size),
            )?;
        }
        if frequency_penalty != 0.0 {
            last_logits = apply_frequency_penalty(
                &last_logits,
                &input_tokens,
                frequency_penalty,
                Some(frequency_context_size),
            )?;
        }

        let (mut token, mut logprobs) = if return_logprobs {
            let (tok, lp) = sample_and_logprobs(&last_logits, Some(sampling_config))?;
            (tok, Some(lp))
        } else {
            (sample(&last_logits, Some(sampling_config))?, None)
        };

        // DECODE
        const DECODE_CLEANUP_INTERVAL: i32 = 256;
        let one_arr = MxArray::from_int32(&[1], &[1])?;
        for step in 0..max_new_tokens {
            let _stream_ctx = StreamContext::new(generation_stream);
            token.eval();
            if step > 0 && step % DECODE_CLEANUP_INTERVAL == 0 {
                synchronize_and_clear_cache();
            }
            let token_value = token.item_at_int32(0)? as u32;
            generated_tokens.push(token_value);
            if return_logprobs && let Some(ref lp) = logprobs {
                lp.eval();
                let lp_value = lp.item_at_float32(0)?;
                generated_logprobs.push(lp_value);
            }
            if let Some(eos) = eos_token_id
                && token_value == eos as u32
            {
                finish_reason = "stop";
                break;
            }
            if let Some(reason) = check_repetition_cutoff(
                &generated_tokens,
                max_consecutive_tokens,
                max_ngram_repeats,
                ngram_size,
            ) {
                finish_reason = reason;
                break;
            }
            let next_ids = MxArray::from_uint32(&[token_value], &[1, 1])?;
            let next_logits = Qwen3Model::forward_fused(
                &next_ids,
                &embedding_weight,
                layers,
                final_norm,
                lm_head,
                model_config,
                &mut kv_keys,
                &mut kv_values,
                &mut cache_idx,
                &rope_offsets,
                &left_padding,
            )?;
            rope_offsets = rope_offsets.add(&one_arr)?;
            let next_last_logits = next_logits.slice_axis(1, 0, 1)?.squeeze(Some(&[0, 1]))?;
            last_logits = next_last_logits;
            if repetition_penalty != 1.0 || presence_penalty != 0.0 || frequency_penalty != 0.0 {
                let context_tokens: Vec<u32> = input_tokens
                    .iter()
                    .copied()
                    .chain(generated_tokens.iter().copied())
                    .collect();
                if repetition_penalty != 1.0 {
                    last_logits = apply_repetition_penalty(
                        &last_logits,
                        &context_tokens,
                        repetition_penalty,
                        Some(repetition_context_size),
                    )?;
                }
                if presence_penalty != 0.0 {
                    last_logits = apply_presence_penalty(
                        &last_logits,
                        &context_tokens,
                        presence_penalty,
                        Some(presence_context_size),
                    )?;
                }
                if frequency_penalty != 0.0 {
                    last_logits = apply_frequency_penalty(
                        &last_logits,
                        &context_tokens,
                        frequency_penalty,
                        Some(frequency_context_size),
                    )?;
                }
            }
            let (next_tok, next_lp) = if return_logprobs {
                let (tok, lp) = sample_and_logprobs(&last_logits, Some(sampling_config))?;
                (tok, Some(lp))
            } else {
                (sample(&last_logits, Some(sampling_config))?, None)
            };
            token = next_tok;
            logprobs = next_lp;
        }

        let tokens_array =
            MxArray::from_uint32(&generated_tokens, &[generated_tokens.len() as i64])?;
        let logprobs_array = if return_logprobs {
            MxArray::from_float32(&generated_logprobs, &[generated_logprobs.len() as i64])?
        } else {
            MxArray::from_float32(&[], &[0])?
        };

        Ok(GenerationResult {
            text: String::new(), // Text decoding done by caller
            tokens: tokens_array,
            logprobs: logprobs_array,
            finish_reason: finish_reason.to_string(),
            num_tokens: generated_tokens.len(),
        })
    }

    /// GRPO training step: compute loss, gradients, and apply optimizer.
    ///
    /// Consumes cached MxArrays from the generation phase, computes loss and
    /// gradients via autograd, validates and clips gradients, accumulates them,
    /// and applies the optimizer step when accumulation is complete.
    /// Inherent body of [`TrainBackend::train_step_grpo_sync`].
    fn train_step_grpo_sync_impl(
        &mut self,
        rewards: Vec<f64>,
        group_size: i32,
        loss_config: crate::grpo::loss::GRPOLossConfig,
        valid_indices: Option<Vec<usize>>,
    ) -> Result<crate::training_model::TrainStepPlainMetrics> {
        use crate::array::memory::{get_active_memory, get_peak_memory, reset_peak_memory};
        use crate::grpo::advantages::compute_advantages;
        use crate::grpo::autograd::compute_loss_and_gradients_autograd;
        use crate::optimizers::GradientUtils;

        reset_peak_memory();

        // Get cached generation results from training_state
        let ts = self.training_state.as_ref().ok_or_else(|| {
            napi::Error::from_reason("Training state not initialized. Call InitTraining first.")
        })?;

        let prompt_tokens = ts.cached_prompt_tokens.as_ref().ok_or_else(|| {
            napi::Error::from_reason("No cached prompt tokens. Call GenerateForTraining first.")
        })?;
        let completion_tokens = ts.cached_completion_tokens.as_ref().ok_or_else(|| {
            napi::Error::from_reason("No cached completion tokens. Call GenerateForTraining first.")
        })?;
        let completion_logprobs = ts.cached_completion_logprobs.as_ref().ok_or_else(|| {
            napi::Error::from_reason(
                "No cached completion logprobs. Call GenerateForTraining first.",
            )
        })?;

        let use_checkpointing = ts.gradient_checkpointing;
        let gradient_clip_value = ts.gradient_clip_value;
        let gradient_clip_norm = ts.gradient_clip_norm;
        let verbose_nan = ts.verbose_nan_detection;
        let learning_rate = ts.learning_rate;
        let max_nan_gradients = ts.max_nan_gradients;
        let emergency_save_threshold = ts.emergency_save_threshold;

        // Get model parameters
        let params = self.get_parameters_sync()?;
        let model_type = ModelType::Qwen3(self.config.clone());

        // Build completion/logprob refs, optionally filtering by valid_indices from
        // the engine's degenerate-completion filter. prompt_refs always has one
        // entry per prompt — the autograd function expands them to one per
        // completion via repeat_n(group_size), so the `group_size` passed here
        // must be the effective group size after filtering (the engine computes
        // effective_group_size = valid_indices.len() / num_prompts).
        let prompt_refs: Vec<&MxArray> = prompt_tokens.iter().collect();
        let (completion_refs, logprob_refs): (Vec<&MxArray>, Vec<&MxArray>) =
            if let Some(ref indices) = valid_indices {
                let n = completion_tokens.len();
                for &i in indices {
                    if i >= n {
                        return Err(napi::Error::from_reason(format!(
                            "valid_indices contains out-of-range index {} (completion count = {})",
                            i, n
                        )));
                    }
                }
                let c: Vec<&MxArray> = indices.iter().map(|&i| &completion_tokens[i]).collect();
                let l: Vec<&MxArray> = indices.iter().map(|&i| &completion_logprobs[i]).collect();
                (c, l)
            } else {
                (
                    completion_tokens.iter().collect(),
                    completion_logprobs.iter().collect(),
                )
            };

        let (loss_value, gradients) = compute_loss_and_gradients_autograd(
            &model_type,
            &params,
            &prompt_refs,
            &completion_refs,
            &logprob_refs,
            &rewards,
            group_size,
            loss_config,
            use_checkpointing,
        )?;

        // Check for NaN/Inf loss
        if loss_value.is_nan() || loss_value.is_infinite() {
            warn!("Skipping step due to invalid loss: {}", loss_value);
            synchronize_and_clear_cache();
            // Skipped steps must still advance the authoritative step counter
            // (H1) and drop the cached generation so the next cycle starts
            // clean.
            let ts = self.training_state.as_mut().unwrap();
            ts.clear_generation_cache();
            ts.step += 1;
            let new_step = ts.step;
            let nan_count = ts.nan_gradient_count;
            return Ok(crate::training_model::TrainStepPlainMetrics {
                loss: loss_value,
                gradients_applied: false,
                mean_advantage: 0.0,
                std_advantage: 0.0,
                nan_gradient_count: nan_count,
                peak_memory_mb: get_peak_memory() / 1e6,
                active_memory_mb: get_active_memory() / 1e6,
                total_tokens: 0,
                step: new_step,
            });
        }

        // Validate ALL gradients — skip entire step if ANY has NaN/Inf
        for (name, grad) in gradients.iter() {
            grad.eval();
            let has_invalid = grad.has_nan_or_inf()?;
            if has_invalid {
                if verbose_nan {
                    let data = grad.to_float32()?;
                    let invalid_count = data
                        .iter()
                        .filter(|v| v.is_nan() || v.is_infinite())
                        .count();
                    warn!(
                        "Gradient '{}' contains {} invalid values - SKIPPING STEP",
                        name, invalid_count
                    );
                } else {
                    warn!("Gradient '{}' contains NaN/Inf - SKIPPING STEP", name);
                }

                let ts = self.training_state.as_mut().unwrap();
                ts.nan_gradient_count += 1;
                ts.consecutive_nan_count += 1;

                if ts.nan_gradient_count >= max_nan_gradients as u64 {
                    return Err(napi::Error::from_reason(format!(
                        "Training stopped: exceeded max NaN gradient count ({}/{})",
                        ts.nan_gradient_count, max_nan_gradients
                    )));
                }

                if ts.consecutive_nan_count >= emergency_save_threshold as u32 {
                    warn!(
                        "Emergency save triggered: {} consecutive NaN gradients",
                        ts.consecutive_nan_count
                    );
                }

                // Advance the authoritative step counter (H1) and clear the
                // cached generation data so the next cycle starts clean.
                ts.clear_generation_cache();
                ts.step += 1;
                let new_step = ts.step;
                let nan_count = ts.nan_gradient_count;
                synchronize_and_clear_cache();
                return Ok(crate::training_model::TrainStepPlainMetrics {
                    loss: loss_value,
                    gradients_applied: false,
                    mean_advantage: 0.0,
                    std_advantage: 0.0,
                    nan_gradient_count: nan_count,
                    peak_memory_mb: get_peak_memory() / 1e6,
                    active_memory_mb: get_active_memory() / 1e6,
                    total_tokens: 0,
                    step: new_step,
                });
            }
        }

        // Element-wise gradient clipping
        let grad_clip_val = gradient_clip_value.unwrap_or(1.0);
        let mut clamped_gradients: HashMap<String, MxArray> = HashMap::new();
        for (name, grad) in gradients.iter() {
            let clamped = grad.clip(Some(-grad_clip_val), Some(grad_clip_val))?;
            clamped.eval();
            clamped_gradients.insert(name.clone(), clamped);
        }

        // Gradient norm clipping
        let clipped_gradients = if let Some(max_norm) = gradient_clip_norm {
            let grad_refs: HashMap<String, &MxArray> = clamped_gradients
                .iter()
                .map(|(k, v)| (k.clone(), v))
                .collect();
            GradientUtils::clip_grad_norm(grad_refs, max_norm)?
        } else {
            clamped_gradients
        };

        // Accumulate gradients
        let ts = self.training_state.as_mut().unwrap();
        // Reset consecutive NaN count on successful gradient computation
        ts.consecutive_nan_count = 0;

        Self::accumulate_gradients_inner(ts, clipped_gradients)?;
        ts.micro_step += 1;

        let grad_acc_steps = ts.grad_accumulation_steps;
        let gradients_applied = if ts.micro_step >= grad_acc_steps {
            let grads = ts
                .accumulated_gradients
                .take()
                .ok_or_else(|| napi::Error::from_reason("No accumulated gradients"))?;

            // Apply optimizer step
            if let Some(ref mut optimizer) = ts.optimizer {
                // AdamW path
                let mut param_names_vec: Vec<String> = Vec::new();
                let mut param_refs: Vec<&MxArray> = Vec::new();
                let mut grad_refs: Vec<&MxArray> = Vec::new();

                // Scale gradients if using accumulation
                let scaled_grads: HashMap<String, MxArray>;
                let grads_to_use = if grad_acc_steps > 1 {
                    let scale = 1.0 / grad_acc_steps as f32;
                    let scale_arr = MxArray::from_float32(&[scale], &[])?;
                    scaled_grads = grads
                        .iter()
                        .map(|(name, grad)| Ok((name.clone(), grad.mul(&scale_arr)?)))
                        .collect::<Result<HashMap<_, _>>>()?;
                    &scaled_grads
                } else {
                    &grads
                };

                for (name, grad) in grads_to_use {
                    if let Some(param) = params.get(name) {
                        param_names_vec.push(name.clone());
                        param_refs.push(param);
                        grad_refs.push(grad);
                    }
                }

                let updated = optimizer.update_batch(
                    param_names_vec.clone(),
                    param_refs.clone(),
                    grad_refs,
                )?;

                // `update_batch` above has already committed the optimizer's step
                // counter and moment tensors. If this fallible delta build (or the
                // atomic apply below) errors, the optimizer is left one step ahead
                // of the still-unchanged model params (all deltas are built before
                // any are applied). Accepted over panicking; such `?` failures are
                // fatal device errors that abort the run.
                // Create deltas: delta = param - updated (so param - 1.0 * delta = updated)
                let delta_map: HashMap<String, MxArray> = param_names_vec
                    .iter()
                    .enumerate()
                    .map(|(i, name)| {
                        let delta = param_refs[i].sub(&updated[i])?;
                        Ok((name.clone(), delta))
                    })
                    .collect::<Result<HashMap<_, _>>>()?;

                let delta_refs: HashMap<String, &MxArray> =
                    delta_map.iter().map(|(k, v)| (k.clone(), v)).collect();
                self.apply_gradients_inner(delta_refs, 1.0, &params)?;

                debug!(
                    "Applied AdamW update (step={})",
                    self.training_state.as_ref().unwrap().step
                );
            } else {
                // SGD path
                let lr = learning_rate / grad_acc_steps as f64;
                let grads_refs: HashMap<String, &MxArray> =
                    grads.iter().map(|(k, v)| (k.clone(), v)).collect();
                self.apply_gradients_inner(grads_refs, lr, &params)?;
                debug!("Applied SGD gradients with lr: {}", lr);
            }

            let ts = self.training_state.as_mut().unwrap();
            ts.accumulated_gradients = None;
            ts.micro_step = 0;
            ts.step += 1;
            true
        } else {
            ts.step += 1;
            false
        };

        // Compute advantage statistics
        let rewards_f32: Vec<f32> = rewards.iter().map(|&r| r as f32).collect();
        let rewards_array = MxArray::from_float32(&rewards_f32, &[rewards.len() as i64])?;
        let advantages = compute_advantages(&rewards_array, group_size, "group".to_string())?;
        let adv_data = advantages.to_float32()?;
        let mean_advantage =
            adv_data.iter().map(|&a| a as f64).sum::<f64>() / adv_data.len().max(1) as f64;
        let std_advantage = {
            let variance = adv_data
                .iter()
                .map(|&a| {
                    let diff = a as f64 - mean_advantage;
                    diff * diff
                })
                .sum::<f64>()
                / adv_data.len().max(1) as f64;
            variance.sqrt()
        };

        // Count tokens BEFORE clearing the cache — otherwise total_tokens is
        // always zero on the success path.
        let ts = self.training_state.as_ref().unwrap();
        let total_tokens: i32 = if let Some(ref ct) = ts.cached_completion_tokens {
            ct.iter()
                .filter_map(|t| t.shape_at(0).ok())
                .map(|n| n as i32)
                .sum()
        } else {
            0
        };

        // Clear cached generation data
        if let Some(ref mut ts) = self.training_state {
            ts.clear_generation_cache();
        }

        // CRITICAL: heavy_cleanup after autograd to clear compiled graph cache
        heavy_cleanup();

        let ts = self.training_state.as_ref().unwrap();
        Ok(crate::training_model::TrainStepPlainMetrics {
            loss: loss_value,
            gradients_applied,
            mean_advantage,
            std_advantage,
            nan_gradient_count: ts.nan_gradient_count,
            peak_memory_mb: get_peak_memory() / 1e6,
            active_memory_mb: get_active_memory() / 1e6,
            total_tokens,
            step: ts.step,
        })
    }

    /// SFT training step: compute loss, gradients, and apply optimizer.
    ///
    /// Receives plain data (Vec<i32> + shape) from the SFT engine, reconstructs
    /// MxArrays on the model thread, computes SFT loss + gradients, validates,
    /// clips, accumulates, and applies optimizer step when accumulation is complete.
    /// Inherent body of [`TrainBackend::train_step_sft_sync`].
    fn train_step_sft_sync_impl(
        &mut self,
        input_ids: Vec<i32>,
        input_shape: Vec<i64>,
        labels: Vec<i32>,
        labels_shape: Vec<i64>,
        config: crate::sft::engine::SftEngineConfig,
    ) -> Result<crate::training_model::TrainStepPlainMetrics> {
        use crate::array::memory::{get_active_memory, get_peak_memory, reset_peak_memory};
        use crate::optimizers::GradientUtils;

        reset_peak_memory();

        // Ensure training state is initialized
        let ts = self.training_state.as_ref().ok_or_else(|| {
            napi::Error::from_reason("Training state not initialized. Call InitTraining first.")
        })?;
        let _ = ts; // just validating it exists

        // Reconstruct MxArrays from plain data
        let input_ids_arr = MxArray::from_int32(&input_ids, &input_shape)?;
        let labels_arr = MxArray::from_int32(&labels, &labels_shape)?;

        // Get model parameters
        let params = self.get_parameters_sync()?;
        let model_type = crate::training_model::ModelType::Qwen3(self.config.clone());

        // Build loss config from SftEngineConfig
        let loss_config = crate::sft::SftLossConfig {
            ignore_index: Some(-100),
            label_smoothing: config.label_smoothing,
        };

        let use_checkpointing = config.gradient_checkpointing.unwrap_or(true);
        let verbose_nan = config.verbose_nan_detection.unwrap_or(false);
        let max_nan_gradients = config.max_nan_gradients.unwrap_or(100);
        let emergency_save_threshold = config.emergency_save_threshold.unwrap_or(5);

        // Compute loss and gradients
        let (loss_value, gradients) = crate::sft::autograd::compute_sft_loss_and_gradients(
            &model_type,
            &params,
            &input_ids_arr,
            &labels_arr,
            loss_config,
            use_checkpointing,
        )?;

        // Check for NaN/Inf loss
        if loss_value.is_nan() || loss_value.is_infinite() {
            warn!("SFT: Skipping step due to invalid loss: {}", loss_value);
            synchronize_and_clear_cache();
            let ts = self.training_state.as_mut().unwrap();
            ts.nan_gradient_count += 1;
            ts.consecutive_nan_count += 1;

            if ts.nan_gradient_count >= max_nan_gradients as u64 {
                return Err(napi::Error::from_reason(format!(
                    "Training stopped: exceeded max NaN gradient count ({}/{})",
                    ts.nan_gradient_count, max_nan_gradients
                )));
            }

            if ts.consecutive_nan_count >= emergency_save_threshold as u32 {
                warn!(
                    "Emergency save triggered: {} consecutive NaN losses",
                    ts.consecutive_nan_count
                );
            }

            return Ok(crate::training_model::TrainStepPlainMetrics {
                loss: 0.0,
                gradients_applied: false,
                mean_advantage: 0.0,
                std_advantage: 0.0,
                nan_gradient_count: ts.nan_gradient_count,
                peak_memory_mb: get_peak_memory() / 1e6,
                active_memory_mb: get_active_memory() / 1e6,
                total_tokens: 0,
                step: ts.step,
            });
        }

        // Validate ALL gradients — skip entire step if ANY has NaN/Inf
        for (name, grad) in gradients.iter() {
            grad.eval();
            let has_invalid = grad.has_nan_or_inf()?;
            if has_invalid {
                if verbose_nan {
                    let data = grad.to_float32()?;
                    let invalid_count = data
                        .iter()
                        .filter(|v| v.is_nan() || v.is_infinite())
                        .count();
                    warn!(
                        "SFT: Gradient '{}' contains {} invalid values - SKIPPING STEP",
                        name, invalid_count
                    );
                } else {
                    warn!("SFT: Gradient '{}' contains NaN/Inf - SKIPPING STEP", name);
                }

                let ts = self.training_state.as_mut().unwrap();
                ts.nan_gradient_count += 1;
                ts.consecutive_nan_count += 1;

                if ts.nan_gradient_count >= max_nan_gradients as u64 {
                    return Err(napi::Error::from_reason(format!(
                        "Training stopped: exceeded max NaN gradient count ({}/{})",
                        ts.nan_gradient_count, max_nan_gradients
                    )));
                }

                if ts.consecutive_nan_count >= emergency_save_threshold as u32 {
                    warn!(
                        "Emergency save triggered: {} consecutive NaN gradients",
                        ts.consecutive_nan_count
                    );
                }

                synchronize_and_clear_cache();
                return Ok(crate::training_model::TrainStepPlainMetrics {
                    loss: loss_value,
                    gradients_applied: false,
                    mean_advantage: 0.0,
                    std_advantage: 0.0,
                    nan_gradient_count: ts.nan_gradient_count,
                    peak_memory_mb: get_peak_memory() / 1e6,
                    active_memory_mb: get_active_memory() / 1e6,
                    total_tokens: 0,
                    step: ts.step,
                });
            }
        }

        // Element-wise gradient clipping (if configured)
        let clipped_gradients = if let Some(clip_val) = config.gradient_clip_value {
            let mut clamped: HashMap<String, MxArray> = HashMap::new();
            for (name, grad) in gradients.iter() {
                let c = grad.clip(Some(-clip_val), Some(clip_val))?;
                c.eval();
                clamped.insert(name.clone(), c);
            }
            clamped
        } else {
            gradients.clone()
        };

        // Gradient norm clipping (if configured)
        let final_gradients = if let Some(clip_norm) = config.gradient_clip_norm {
            let grad_refs: HashMap<String, &MxArray> = clipped_gradients
                .iter()
                .map(|(k, v)| (k.clone(), v))
                .collect();
            GradientUtils::clip_grad_norm(grad_refs, clip_norm)?
        } else {
            clipped_gradients
        };

        // Accumulate gradients
        let ts = self.training_state.as_mut().unwrap();
        // Reset consecutive NaN count on successful gradient computation
        ts.consecutive_nan_count = 0;

        Self::accumulate_gradients_inner(ts, final_gradients)?;
        ts.micro_step += 1;

        let grad_acc_steps = config.gradient_accumulation_steps.unwrap_or(1);
        let learning_rate = config.learning_rate.unwrap_or(2e-5);
        let weight_decay = config.weight_decay.unwrap_or(0.01);

        let gradients_applied = if ts.micro_step >= grad_acc_steps {
            let grads = ts
                .accumulated_gradients
                .take()
                .ok_or_else(|| napi::Error::from_reason("No accumulated gradients"))?;

            // Apply optimizer step
            if let Some(ref mut optimizer) = ts.optimizer {
                // AdamW path
                let mut param_names_vec: Vec<String> = Vec::new();
                let mut param_refs: Vec<&MxArray> = Vec::new();
                let mut grad_refs: Vec<&MxArray> = Vec::new();

                // Scale gradients if using accumulation
                let scaled_grads: HashMap<String, MxArray>;
                let grads_to_use = if grad_acc_steps > 1 {
                    let scale = 1.0 / grad_acc_steps as f32;
                    let scale_arr = MxArray::from_float32(&[scale], &[])?;
                    scaled_grads = grads
                        .iter()
                        .map(|(name, grad)| Ok((name.clone(), grad.mul(&scale_arr)?)))
                        .collect::<Result<HashMap<_, _>>>()?;
                    &scaled_grads
                } else {
                    &grads
                };

                for (name, grad) in grads_to_use {
                    if let Some(param) = params.get(name) {
                        param_names_vec.push(name.clone());
                        param_refs.push(param);
                        grad_refs.push(grad);
                    }
                }

                let updated = optimizer.update_batch(
                    param_names_vec.clone(),
                    param_refs.clone(),
                    grad_refs,
                )?;

                // `update_batch` above has already committed the optimizer's step
                // counter and moment tensors. If this fallible delta build (or the
                // atomic apply below) errors, the optimizer is left one step ahead
                // of the still-unchanged model params (all deltas are built before
                // any are applied). Accepted over panicking; such `?` failures are
                // fatal device errors that abort the run.
                // Create deltas: delta = param - updated (so param - 1.0 * delta = updated)
                let delta_map: HashMap<String, MxArray> = param_names_vec
                    .iter()
                    .enumerate()
                    .map(|(i, name)| {
                        let delta = param_refs[i].sub(&updated[i])?;
                        Ok((name.clone(), delta))
                    })
                    .collect::<Result<HashMap<_, _>>>()?;

                let delta_refs: HashMap<String, &MxArray> =
                    delta_map.iter().map(|(k, v)| (k.clone(), v)).collect();
                self.apply_gradients_inner(delta_refs, 1.0, &params)?;

                debug!(
                    "SFT: Applied AdamW update (step={})",
                    self.training_state.as_ref().unwrap().step
                );
            } else {
                // SGD path with weight decay
                let lr = learning_rate / grad_acc_steps as f64;

                // Apply weight decay to gradients if configured
                let grads_with_decay = if weight_decay > 0.0 {
                    grads
                        .into_iter()
                        .map(|(name, grad)| {
                            if let Some(param) = params.get(&name) {
                                if let Ok(decay_term) = param.mul_scalar(weight_decay)
                                    && let Ok(new_grad) = grad.add(&decay_term)
                                {
                                    return (name, new_grad);
                                }
                                (name, grad)
                            } else {
                                (name, grad)
                            }
                        })
                        .collect::<HashMap<_, _>>()
                } else {
                    grads
                };

                let grads_refs: HashMap<String, &MxArray> = grads_with_decay
                    .iter()
                    .map(|(k, v)| (k.clone(), v))
                    .collect();
                self.apply_gradients_inner(grads_refs, lr, &params)?;
                debug!("SFT: Applied SGD gradients with lr: {}", lr);
            }

            let ts = self.training_state.as_mut().unwrap();
            ts.accumulated_gradients = None;
            ts.micro_step = 0;
            ts.step += 1;
            true
        } else {
            ts.step += 1;
            false
        };

        // Count valid tokens from the labels
        let total_tokens = {
            let ignore_val = MxArray::scalar_int(-100)?;
            let valid_mask = labels_arr.not_equal(&ignore_val)?;
            let count = valid_mask.sum(None, Some(false))?;
            count.eval();
            count.item_at_int32(0).unwrap_or(0)
        };

        // CRITICAL: heavy_cleanup after autograd to clear compiled graph cache
        heavy_cleanup();

        let ts = self.training_state.as_ref().unwrap();
        Ok(crate::training_model::TrainStepPlainMetrics {
            loss: loss_value,
            gradients_applied,
            mean_advantage: 0.0,
            std_advantage: 0.0,
            nan_gradient_count: ts.nan_gradient_count,
            peak_memory_mb: get_peak_memory() / 1e6,
            active_memory_mb: get_active_memory() / 1e6,
            total_tokens,
            step: ts.step,
        })
    }

    /// Accumulate gradients into training state.
    fn accumulate_gradients_inner(
        ts: &mut crate::training_state::ModelThreadTrainingState,
        new_grads: HashMap<String, MxArray>,
    ) -> Result<()> {
        match &mut ts.accumulated_gradients {
            Some(acc) => {
                for (name, grad) in new_grads {
                    grad.eval();
                    if grad.has_nan_or_inf()? {
                        warn!(
                            "Skipping gradient accumulation for '{}' due to NaN/Inf",
                            name
                        );
                        continue;
                    }
                    if let Some(existing) = acc.get_mut(&name) {
                        let summed = existing.add(&grad)?;
                        summed.eval();
                        *existing = summed;
                    } else {
                        acc.insert(name, grad);
                    }
                }
            }
            None => {
                let mut evaluated_grads = HashMap::with_capacity(new_grads.len());
                for (name, grad) in new_grads {
                    grad.eval();
                    if grad.has_nan_or_inf()? {
                        warn!("Skipping initial gradient for '{}' due to NaN/Inf", name);
                        continue;
                    }
                    evaluated_grads.insert(name, grad);
                }
                ts.accumulated_gradients = Some(evaluated_grads);
            }
        }
        Ok(())
    }

    /// Apply gradients to model weights (SGD or AdamW delta application).
    ///
    /// Direct field access on Qwen3Inner — no locks needed.
    fn apply_gradients_inner(
        &mut self,
        gradients: HashMap<String, &MxArray>,
        learning_rate: f64,
        current_params: &HashMap<String, MxArray>,
    ) -> Result<()> {
        let updated_params =
            crate::training_model::compute_sgd_updates(&gradients, learning_rate, current_params)?;

        // Apply updated parameters directly to model fields
        for (name, updated_param) in updated_params.iter() {
            if name == "lm_head.weight" {
                self.lm_head.set_weight(updated_param)?;
            } else if name == "final_norm.weight" {
                self.final_norm.set_weight(updated_param)?;
            } else if name == "embedding.weight" {
                self.embedding.set_weight(updated_param)?;
            } else if name.starts_with("layers.") {
                let parts: Vec<&str> = name.split('.').collect();
                if parts.len() >= 3
                    && let Ok(layer_idx) = parts[1].parse::<usize>()
                    && layer_idx < self.layers.len()
                {
                    let layer = &mut self.layers[layer_idx];
                    if name.contains(".self_attn.q_proj.weight") {
                        layer.self_attn.set_q_proj_weight(updated_param)?;
                    } else if name.contains(".self_attn.k_proj.weight") {
                        layer.self_attn.set_k_proj_weight(updated_param)?;
                    } else if name.contains(".self_attn.v_proj.weight") {
                        layer.self_attn.set_v_proj_weight(updated_param)?;
                    } else if name.contains(".self_attn.o_proj.weight") {
                        layer.self_attn.set_o_proj_weight(updated_param)?;
                    } else if name.contains(".mlp.gate_proj.weight") {
                        layer.mlp.set_gate_proj_weight(updated_param)?;
                    } else if name.contains(".mlp.up_proj.weight") {
                        layer.mlp.set_up_proj_weight(updated_param)?;
                    } else if name.contains(".mlp.down_proj.weight") {
                        layer.mlp.set_down_proj_weight(updated_param)?;
                    } else if name.contains(".input_layernorm.weight") {
                        layer.set_input_layernorm_weight(updated_param)?;
                    } else if name.contains(".post_attention_layernorm.weight") {
                        layer.set_post_attention_layernorm_weight(updated_param)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Extract all trainable parameters from the model.
    /// Direct field access — no locks needed on model thread.
    pub(crate) fn get_parameters_sync(&self) -> Result<HashMap<String, MxArray>> {
        let mut params = HashMap::new();

        // Embedding
        params.insert("embedding.weight".to_string(), self.embedding.get_weight());

        // Transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{}", i);

            let attn = &layer.self_attn;
            params.insert(
                format!("{}.self_attn.q_proj.weight", prefix),
                attn.get_q_proj_weight(),
            );
            params.insert(
                format!("{}.self_attn.k_proj.weight", prefix),
                attn.get_k_proj_weight(),
            );
            params.insert(
                format!("{}.self_attn.v_proj.weight", prefix),
                attn.get_v_proj_weight(),
            );
            params.insert(
                format!("{}.self_attn.o_proj.weight", prefix),
                attn.get_o_proj_weight(),
            );

            if self.config.use_qk_norm {
                if let Some(q_norm_weight) = attn.get_q_norm_weight() {
                    params.insert(format!("{}.self_attn.q_norm.weight", prefix), q_norm_weight);
                }
                if let Some(k_norm_weight) = attn.get_k_norm_weight() {
                    params.insert(format!("{}.self_attn.k_norm.weight", prefix), k_norm_weight);
                }
            }

            let mlp = &layer.mlp;
            params.insert(
                format!("{}.mlp.gate_proj.weight", prefix),
                mlp.get_gate_proj_weight(),
            );
            params.insert(
                format!("{}.mlp.up_proj.weight", prefix),
                mlp.get_up_proj_weight(),
            );
            params.insert(
                format!("{}.mlp.down_proj.weight", prefix),
                mlp.get_down_proj_weight(),
            );

            params.insert(
                format!("{}.input_layernorm.weight", prefix),
                layer.get_input_layernorm_weight(),
            );
            params.insert(
                format!("{}.post_attention_layernorm.weight", prefix),
                layer.get_post_attention_layernorm_weight(),
            );
        }

        // Final norm
        params.insert(
            "final_norm.weight".to_string(),
            self.final_norm.get_weight(),
        );

        // LM head (only if not tied to embeddings)
        if !self.config.tie_word_embeddings {
            params.insert("lm_head.weight".to_string(), self.lm_head.get_weight());
        }

        Ok(params)
    }
}

// =====================================================================
// Neutral chat-engine backend.
//
// The chat-session surface (start / continue / tool + streaming twins +
// reset-caches) is driven by the model-neutral engine: the thread loop
// routes `Qwen3Cmd::Chat` to `handle_chat_cmd::<Qwen3Inner>`, whose
// generic session cores (`crate::engine::session`) call back into the
// `ChatBackend` impl below. The block-paged production path (default
// ON) runs behind `paged_turn`; the flat (opt-out) path runs the
// engine's generic prefill + `run_decode_loop` flow via `prefill` /
// `begin_decode`.
// =====================================================================

impl Qwen3Inner {
    /// Flat chunked prefill for the engine's generic flow
    /// ([`ChatBackend::prefill`]): 2048-token chunks, per-chunk KV evals
    /// plus `synchronize_and_clear_cache`, last-token logits squeezed to
    /// `[1, vocab]`.
    ///
    /// Seeds the turn-scratch KV state from the saved session state: on
    /// a prefix miss the engine already ran
    /// `reset_caches(ResetScope::PrefixMiss)`, so the cached vectors are
    /// empty and the turn starts cold (`vec![None; num_layers]`); on a
    /// hit / delta the cached handles are cloned. The exact-match rewind
    /// flag (see [`Qwen3Inner::pending_exact_match_rewind`]) backs the
    /// write cursor up one slot so the single re-prefilled token
    /// overwrites the last cached position (the "zero delta — re-run last
    /// token" case).
    fn flat_prefill(&mut self, prompt_tokens: &[u32], stream: Stream) -> Result<MxArray> {
        if prompt_tokens.is_empty() {
            return Err(napi::Error::from_reason(
                "qwen3 flat prefill requires a non-empty prompt",
            ));
        }
        let num_layers = self.layers.len();
        let rewind = self.pending_exact_match_rewind.replace(false);
        self.turn_kv_keys = if self.cached_kv_keys.is_empty() {
            vec![None; num_layers]
        } else {
            self.cached_kv_keys.clone()
        };
        self.turn_kv_values = if self.cached_kv_values.is_empty() {
            vec![None; num_layers]
        } else {
            self.cached_kv_values.clone()
        };
        self.turn_cache_idx = if rewind {
            self.cached_cache_idx - 1
        } else {
            self.cached_cache_idx
        };

        let embedding_weight = self.embedding.get_weight();
        let total_len = prompt_tokens.len();
        let prefill_input = MxArray::from_uint32(prompt_tokens, &[1, total_len as i64])?;
        let prefill_step_size: usize = 2048;
        let use_chunked_prefill = total_len > prefill_step_size;

        let mut rope_offsets = MxArray::from_int32(&[self.turn_cache_idx], &[1])?;
        let left_padding = MxArray::from_int32(&[0], &[1])?;

        let last_logits = if use_chunked_prefill {
            let mut offset = 0usize;
            while offset + prefill_step_size < total_len {
                let chunk_end = offset + prefill_step_size;
                let chunk = prefill_input.slice(&[0, offset as i64], &[1, chunk_end as i64])?;
                rope_offsets = MxArray::from_int32(&[self.turn_cache_idx], &[1])?;
                {
                    let _stream_ctx = StreamContext::new(stream);
                    let _ = Qwen3Model::forward_fused(
                        &chunk,
                        &embedding_weight,
                        &self.layers,
                        &self.final_norm,
                        &self.lm_head,
                        &self.config,
                        &mut self.turn_kv_keys,
                        &mut self.turn_kv_values,
                        &mut self.turn_cache_idx,
                        &rope_offsets,
                        &left_padding,
                    )?;
                }
                for kv_key in self.turn_kv_keys.iter().flatten() {
                    kv_key.eval();
                }
                for kv_value in self.turn_kv_values.iter().flatten() {
                    kv_value.eval();
                }
                synchronize_and_clear_cache();
                offset = chunk_end;
            }
            let final_chunk = prefill_input.slice(&[0, offset as i64], &[1, total_len as i64])?;
            rope_offsets = MxArray::from_int32(&[self.turn_cache_idx], &[1])?;
            let logits = {
                let _stream_ctx = StreamContext::new(stream);
                Qwen3Model::forward_fused(
                    &final_chunk,
                    &embedding_weight,
                    &self.layers,
                    &self.final_norm,
                    &self.lm_head,
                    &self.config,
                    &mut self.turn_kv_keys,
                    &mut self.turn_kv_values,
                    &mut self.turn_cache_idx,
                    &rope_offsets,
                    &left_padding,
                )?
            };
            let chunk_seq_len = logits.shape_at(1)?;
            // Keep as `[1, vocab]` (squeeze only axis 1) so the shape
            // matches dense/MoE streaming and flows cleanly through
            // the shared penalty + sampling pipeline.
            logits
                .slice_axis(1, chunk_seq_len - 1, chunk_seq_len)?
                .squeeze(Some(&[1]))?
        } else {
            let logits = {
                let _stream_ctx = StreamContext::new(stream);
                Qwen3Model::forward_fused(
                    &prefill_input,
                    &embedding_weight,
                    &self.layers,
                    &self.final_norm,
                    &self.lm_head,
                    &self.config,
                    &mut self.turn_kv_keys,
                    &mut self.turn_kv_values,
                    &mut self.turn_cache_idx,
                    &rope_offsets,
                    &left_padding,
                )?
            };
            let seq_len = logits.shape_at(1)?;
            logits
                .slice_axis(1, seq_len - 1, seq_len)?
                .squeeze(Some(&[1]))?
        };

        Ok(last_logits)
    }
}

/// Paged decode stepper for qwen3 (pure-eager — no compiled path, so no
/// lifecycle/reset guard fields). The paged analog of [`Qwen3Decode`]:
/// drives [`crate::engine::decode::run_decode_loop`] through the
/// `forward_paged_adapter` path via [`Qwen3Inner::run_paged_decode_step`].
pub(crate) struct Qwen3PagedDecode<'a> {
    inner: &'a mut Qwen3Inner,
    num_layers: usize,
    /// Fixed `[0]` positions array threaded into `forward_paged_adapter`
    /// (the paged forward derives real positions from the adapter
    /// cursor, so this placeholder is never read for them).
    positions_dummy: MxArray,
}

impl DecodeStep for Qwen3PagedDecode<'_> {
    fn forward(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)> {
        // The paged forward needs the concrete token id (record_tokens +
        // re-embed as [1, 1]); recover it from the [1, 1] input the loop
        // reshaped. `item_at_int32` forces the eval of `y` — idempotent
        // with the loop-top `y.eval()`.
        let token_value = input_ids.item_at_int32(0)? as u32;
        let logits = self.inner.run_paged_decode_step(
            token_value,
            self.num_layers,
            &self.positions_dummy,
        )?;
        // `run_paged_decode_step` returns `[1, 1, vocab]`; `true` requests
        // the engine's squeeze of axis 1 → `[1, vocab]` (the FLAT
        // contract — `sample` / `apply_all_penalties` accept the leading
        // batch axis).
        Ok((logits, true))
    }

    fn eval_step(&mut self, next_token: &MxArray, logits: &MxArray, _budget_forced: bool) {
        // Schedule next_token (+ logits, matching the flat `Qwen3Decode`)
        // for async eval; the loop-top `y.eval()` forces materialization
        // next iteration.
        MxArray::async_eval_arrays(&[next_token, logits]);
    }

    fn maintain_cache(&mut self, step: i32) {
        // Paged cadence — the per-step
        // `maybe_clear_cache_for_paged_step(step)`.
        crate::array::maybe_clear_cache_for_paged_step(step);
    }

    fn materialize_final(&mut self, token_id: u32) -> Result<()> {
        // LENGTH-exit only (the engine gates the call): run ONE more
        // `run_paged_decode_step` for the final committed token so its
        // K/V lands in the paged adapter, then DISCARD the logits. This
        // runs `record_tokens(&[token]) + forward_paged_adapter`, so the
        // adapter's `request_tokens()` / cursor advances by exactly 1 to
        // equal the saved keep-all history.
        //
        // No sample / no `generated_tokens` push / no chunk / no
        // finish_reason change: the produced logits are simply dropped.
        let _logits =
            self.inner
                .run_paged_decode_step(token_id, self.num_layers, &self.positions_dummy)?;
        Ok(())
    }
    // end_decode → default Ok(()).
}

/// qwen3 paged prefix state — the effective prefix/suffix split from
/// `prepare_turn_with_max_cache_hit_tokens`. Identity-style (no held
/// cache handles); the adapter mutated its own internal state during the
/// prime.
pub(crate) struct Qwen3PrefixState {
    effective_cached_prefix_len: usize,
    suffix_len: usize,
}

impl PagedPrefix for Qwen3PrefixState {
    fn effective_cached_prefix_len(&self) -> usize {
        self.effective_cached_prefix_len
    }
    fn suffix_len(&self) -> usize {
        self.suffix_len
    }
}

impl PagedBackend for Qwen3Inner {
    type PagedDecode<'a>
        = Qwen3PagedDecode<'a>
    where
        Self: 'a;
    type PrefixState = Qwen3PrefixState;

    fn prime_prefix_state(
        &mut self,
        plan: &[u32],
        reuse_cache: bool,
        _block_size: usize,
        extra_keys: &[u64],
        cache_salt: u64,
    ) -> Result<Self::PrefixState> {
        // Lazy decode allocation: pass the prompt length only (decode
        // blocks grow on-demand via `record_tokens`). vLLM-style
        // exact-prefix cap: leave at least one prompt token to prefill so
        // the decoder always has something to consume.
        let total_budget = plan.len() as u32;
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
        // Per-turn seq_id: the adapter is single-request and
        // `reset_for_new_request` makes the previous seq_id irrelevant.
        let seq_id: u32 = 0;
        // Adapter-owned warm-continue / cold-start lifecycle (suffix
        // blocks are allocated inside prepare_turn — do NOT call
        // `allocate_suffix_blocks` again).
        let turn_plan = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| {
                napi::Error::from_reason(
                    "prime_prefix_state: paged_adapter is None — caller must check \
                     use_block_paged_cache before dispatch",
                )
            })?
            .prepare_turn_with_max_cache_hit_tokens(
                seq_id,
                plan,
                total_budget,
                reuse_cache,
                extra_keys,
                cache_salt,
                false,
                max_cache_hit_tokens,
            )
            .map_err(napi::Error::from_reason)?;
        Ok(Qwen3PrefixState {
            effective_cached_prefix_len: turn_plan.cached_prefix_len as usize,
            suffix_len: turn_plan.suffix_len as usize,
        })
    }

    fn paged_prefill(
        &mut self,
        suffix_tokens: &[u32],
        prefix: &Self::PrefixState,
        _stream: Stream,
    ) -> Result<MxArray> {
        let positions_dummy = MxArray::from_int32(&[0], &[1])?;
        self.run_paged_prefill_chunk(
            suffix_tokens,
            prefix.effective_cached_prefix_len as u32,
            self.layers.len(),
            &positions_dummy,
        )
    }

    fn begin_paged_decode(&mut self, _setup: &PagedTurnSetup<'_>) -> Result<Self::PagedDecode<'_>> {
        let positions_dummy = MxArray::from_int32(&[0], &[1])?;
        let num_layers = self.layers.len();
        Ok(Qwen3PagedDecode {
            inner: self,
            num_layers,
            positions_dummy,
        })
    }

    fn finalize_paged_turn(&mut self, reuse_cache: bool) {
        // Terminal lifecycle for the paged turn. Success: keep the request
        // live across turns when reuse is on so the next turn's
        // `continue_turn` builds on the partial trailing block's live K/V;
        // otherwise register full blocks for reuse + release. Infallible
        // (`let _ =` every call — a teardown failure must not mask the turn
        // result).
        if let Some(adapter) = self.paged_adapter.as_mut() {
            if reuse_cache {
                let _ = adapter.finalize_turn_keep_live(&[], 0);
            } else {
                let _ = adapter.register_full_blocks_for_reuse(&[], 0);
                let _ = adapter.release_request();
            }
        }
    }

    fn abort_paged_turn(&mut self) {
        // Error-path teardown: release the request fully — partial
        // block_table state is unsafe to keep around. Release ONLY — never
        // register / keep live. Infallible (`let _ =` — must not mask the
        // turn's error).
        if let Some(adapter) = self.paged_adapter.as_mut() {
            let _ = adapter.release_request();
        }
    }

    fn save_paged_history(
        &mut self,
        save_tokens: &[u32],
        generated: &[u32],
        keep_all: bool,
        reuse_cache: bool,
    ) -> Result<()> {
        // Save token history ONLY — the adapter's pool owns the K/V; never
        // touch `cached_kv_keys`/`cached_kv_values`/`cached_cache_idx`.
        // `keep_all` is the flat rule (engine: `finish_reason ==
        // "length"`), so the DROP-LAST trim below matches
        // `save_cache_state`'s (`finish_reason != "length"` => drop the
        // boundary token). The engine reconciles `request_tokens()` to this
        // same trimmed history via `reconcile_paged_request_tokens` before
        // finalize.
        if reuse_cache {
            let mut full_history = save_tokens.to_vec();
            let history_tokens = if keep_all || generated.is_empty() {
                generated
            } else {
                &generated[..generated.len() - 1]
            };
            full_history.extend_from_slice(history_tokens);
            self.cached_token_history = full_history;
            // Qwen3 has no vision path — keep the image cache key None
            // for uniformity with the VLM-capable siblings' branch.
            self.cached_image_key = None;
        } else {
            self.cached_token_history.clear();
            self.cached_image_key = None;
        }
        // Standard-KV: token-history only, never fails (no GDN/recurrent
        // checkpoint). `Ok` satisfies the fallible trait contract.
        Ok(())
    }

    fn reconcile_paged_request_tokens(
        &mut self,
        prompt_len: usize,
        generated: &[u32],
        keep_all: bool,
    ) -> bool {
        // Perf-parity warm-continue restore (see the trait doc). The
        // pipelined decode loop records the stop token into the adapter
        // (its forward ran at the loop top BEFORE the stop-check), but the
        // saved history DROPS it on a non-length exit. Roll the adapter
        // back to the to-be-saved history length so `request_tokens()`
        // matches the persisted history and the next turn warm-continues.
        //
        // `history_len` uses the EXACT same trim as `save_paged_history`
        // above (same `keep_all`), so the two never disagree. The rollback
        // count is the adapter's surplus over that history; `saturating_sub`
        // makes it a no-op when the adapter is already <= the history
        // (length exit: the engine's `materialize_final` already recorded
        // the final token's K/V, so the adapter EQUALS the kept history —
        // surplus 0; final-step stop: forward never ran, nothing
        // over-recorded).
        //
        // Return contract: `true` on reconcile/no-op,
        // `false` ONLY when the rollback itself returned `Err` (then the
        // adapter is over-recorded vs the saved history, so the engine
        // finalizes with reuse=false = release_request, not keep-live).
        // surplus <= request_tokens().len(), so rollback's `n > len`
        // `Err` cannot fire — the `false` arm is a defensive contract,
        // unreachable in practice.
        let Some(adapter) = self.paged_adapter.as_mut() else {
            return true;
        };
        let history_len = if keep_all || generated.is_empty() {
            generated.len()
        } else {
            generated.len() - 1
        };
        let target_len = prompt_len + history_len;
        let surplus = adapter.request_tokens().len().saturating_sub(target_len);
        if surplus > 0
            && let Err(e) = adapter.rollback_last_tokens(surplus as u32)
        {
            tracing::warn!(
                target: "mlx_core::qwen3::paged",
                "reconcile_paged_request_tokens: rollback_last_tokens({surplus}) failed \
                 (finalize releases the request; next turn cold-prefills): {e}",
            );
            return false;
        }
        true
    }
}

/// Eager flat decode stepper for one qwen3 turn
/// ([`ChatBackend::begin_decode`]). Owns the turn-constant arrays
/// (embedding weight, RoPE offset cursor, left padding, the `+1`
/// increment) and threads the turn-scratch KV state on [`Qwen3Inner`]
/// through [`Qwen3Model::forward_fused`].
pub(crate) struct Qwen3Decode<'a> {
    inner: &'a mut Qwen3Inner,
    embedding_weight: MxArray,
    rope_offsets: MxArray,
    left_padding: MxArray,
    one_arr: MxArray,
    /// Decode-path profiler relabel:
    /// `"qwen3_chat_stream[_delta]_rust"` on the streaming paths;
    /// `None` on the sync paths (which have no profiler).
    relabel: Option<&'static str>,
}

impl DecodeStep for Qwen3Decode<'_> {
    fn forward(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)> {
        let inner = &mut *self.inner;
        let logits = Qwen3Model::forward_fused(
            input_ids,
            &self.embedding_weight,
            &inner.layers,
            &inner.final_norm,
            &inner.lm_head,
            &inner.config,
            &mut inner.turn_kv_keys,
            &mut inner.turn_kv_values,
            &mut inner.turn_cache_idx,
            &self.rope_offsets,
            &self.left_padding,
        )?;
        self.rope_offsets = self.rope_offsets.add(&self.one_arr)?;
        // `true` requests the engine's squeeze of axis 1: the eager Rust
        // forward returns `[1, 1, vocab]`.
        Ok((logits, true))
    }

    fn eval_step(&mut self, next_token: &MxArray, logits: &MxArray, _budget_forced: bool) {
        MxArray::async_eval_arrays(&[next_token, logits]);
    }

    fn profiler_relabel(&self) -> Option<&'static str> {
        self.relabel
    }

    fn materialize_final(&mut self, token_id: u32) -> Result<()> {
        // LENGTH-exit only (the engine gates the call): run ONE more
        // `forward_fused` for the final committed token so its K/V lands in
        // the flat turn KV (advancing `turn_cache_idx` + `rope_offsets`
        // exactly as a normal decode step), then DISCARD the logits. This
        // makes `turn_cache_idx` (→ `cached_cache_idx` at save) equal the
        // keep-all-on-length saved history. No sample / push / emit.
        let input_ids = MxArray::from_int32(&[token_id as i32], &[1, 1])?;
        let inner = &mut *self.inner;
        let _logits = Qwen3Model::forward_fused(
            &input_ids,
            &self.embedding_weight,
            &inner.layers,
            &inner.final_norm,
            &inner.lm_head,
            &inner.config,
            &mut inner.turn_kv_keys,
            &mut inner.turn_kv_values,
            &mut inner.turn_cache_idx,
            &self.rope_offsets,
            &self.left_padding,
        )?;
        self.rope_offsets = self.rope_offsets.add(&self.one_arr)?;
        Ok(())
    }
}

impl ChatBackend for Qwen3Inner {
    fn tokenizer(&self) -> Result<Arc<Qwen3Tokenizer>> {
        self.tokenizer
            .clone()
            .ok_or_else(|| napi::Error::from_reason("Tokenizer not loaded"))
    }

    fn family_name(&self) -> &'static str {
        "qwen3"
    }

    fn session_eos_id(&self, tok: &Qwen3Tokenizer) -> Result<u32> {
        tok.im_end_id()
            .ok_or_else(|| napi::Error::from_reason("Tokenizer missing <|im_end|> special token"))
    }

    fn generation_defaults(&self) -> Option<&crate::engine::ModelGenerationDefaults> {
        Some(&self.gen_defaults)
    }

    fn extra_eos_ids(&self) -> Vec<u32> {
        self.gen_defaults.eos_token_ids.clone()
    }

    // thinking: engine default `policy()` == `ThinkingPolicy::TemplateHonoring`
    // → `thinking_setup` resolves to
    // `{enabled: resolve_enable_thinking(config).unwrap_or(true),
    //   budget: config.thinking_token_budget}`.

    fn cached_token_history(&self) -> &[u32] {
        &self.cached_token_history
    }

    fn reset_caches(&mut self, scope: ResetScope) -> Result<()> {
        // qwen3 has no scope-dependent FLAT reset state (no rope deltas /
        // GDN checkpoints): both the turn-internal prefix-miss reset and
        // the explicit command reset run the full clear via
        // `reset_kv_caches_sync`.
        self.reset_kv_caches_sync()?;
        // The EXPLICIT command reset must restore the fully cold state:
        // `release_request` (inside `reset_kv_caches_sync`) leaves the
        // request's full blocks content-addressed in the allocator's
        // prefix cache, so a reset-then-rerun of the same prompt would take
        // the prefix-hit suffix-prefill path — a different bf16 reduction
        // order than the cold full prefill, enough to flip a greedy
        // near-tie. The turn-internal `PrefixMiss` reset keeps the prefix
        // cache.
        if scope == ResetScope::Command
            && let Some(adapter) = self.paged_adapter.as_mut()
        {
            adapter
                .release_request_and_purge_prefix_cache()
                .map_err(|e| {
                    Error::from_reason(format!(
                        "qwen3 reset_caches: paged prefix-cache purge failed: {e}"
                    ))
                })?;
        }
        Ok(())
    }

    /// All-or-nothing prefix match, PLUS the sanctioned qwen3 pure-KV
    /// exception (see the trait rustdoc): on an EXACT match
    /// (`tokens == cached_token_history`) the flat path rewinds one
    /// position and re-forwards the final token ("zero delta — re-run
    /// last token", `cache_idx -= 1`). Expressed here by returning
    /// `cached_len - 1` and arming
    /// [`Qwen3Inner::pending_exact_match_rewind`], which the next
    /// [`Qwen3Inner::flat_prefill`] consumes. Safe ONLY because qwen3's
    /// flat path is a pure standard-KV stack whose last slot can be
    /// overwritten. The `cached_len == 1` corner returns 0 (miss →
    /// reset, then a 1-token prefill — numerically identical to a
    /// rewind from an empty effective cache).
    fn verify_cache_prefix(&self, tokens: &[u32], reuse_cache: bool) -> usize {
        self.pending_exact_match_rewind.set(false);
        let hit = verify_cache_prefix_pure(
            tokens,
            &self.cached_token_history,
            !self.cached_kv_keys.is_empty() && !self.cached_kv_values.is_empty(),
            reuse_cache,
        );
        if hit > 1 && hit == tokens.len() {
            self.pending_exact_match_rewind.set(true);
            return hit - 1;
        }
        hit
    }

    fn save_cache_state(&mut self, args: SaveStateArgs<'_>) {
        // Delta turns persist unconditionally (the engine's delta guards
        // guarantee `reuse_cache`); fresh turns take the
        // `if reuse_cache { .. } else { clear }` split.
        if args.is_delta || args.reuse_cache {
            self.cached_kv_keys = std::mem::take(&mut self.turn_kv_keys);
            self.cached_kv_values = std::mem::take(&mut self.turn_kv_values);
            self.cached_cache_idx = self.turn_cache_idx;
            // Exclude the final generated token when the decode terminated
            // for any reason other than the `length` budget — in every
            // non-length case the final token IS the boundary marker
            // (`<|im_end|>` or the cutoff token) the next delta re-renders
            // itself.
            let history_tokens =
                if args.finish_reason != "length" && !args.generated_tokens.is_empty() {
                    &args.generated_tokens[..args.generated_tokens.len() - 1]
                } else {
                    args.generated_tokens
                };
            let mut full_history = args.save_tokens.to_vec();
            full_history.extend_from_slice(history_tokens);
            self.cached_token_history = full_history;
            // Qwen3 has no vision path — the image cache key is
            // structurally always None, but we reset it here for clarity
            // and uniformity with VLM-capable siblings.
            self.cached_image_key = None;
        } else {
            self.turn_kv_keys = Vec::new();
            self.turn_kv_values = Vec::new();
            self.turn_cache_idx = 0;
            self.cached_kv_keys.clear();
            self.cached_kv_values.clear();
            self.cached_cache_idx = 0;
            self.cached_token_history.clear();
            self.cached_image_key = None;
        }
    }

    fn eval_caches(&self) -> Result<()> {
        // No post-prefill cache sync on qwen3's flat path: the chunked
        // prefill already evals per chunk inside `flat_prefill`, and the
        // decode loop schedules async evals. A blocking sync here would
        // introduce a needless stall.
        Ok(())
    }

    fn prefill(&mut self, prompt_tokens: &[u32], stream: Stream) -> Result<MxArray> {
        self.flat_prefill(prompt_tokens, stream)
    }

    type Decode<'a>
        = Qwen3Decode<'a>
    where
        Self: 'a;

    fn begin_decode(&mut self, turn: &TurnSetup<'_>) -> Result<Self::Decode<'_>> {
        // Streaming turns relabel the profiler to the "_rust" variant;
        // sync turns have no profiler (no relabel).
        let relabel = if self.turn_is_streaming.get() {
            Some(if turn.is_delta {
                "qwen3_chat_stream_delta_rust"
            } else {
                "qwen3_chat_stream_rust"
            })
        } else {
            None
        };
        let embedding_weight = self.embedding.get_weight();
        // First decode step sees position `cache_idx`: seed
        // `rope_offsets` from the post-prefill cursor.
        let rope_offsets = MxArray::from_int32(&[self.turn_cache_idx], &[1])?;
        let left_padding = MxArray::from_int32(&[0], &[1])?;
        let one_arr = MxArray::from_int32(&[1], &[1])?;
        Ok(Qwen3Decode {
            inner: self,
            embedding_weight,
            rope_offsets,
            left_padding,
            one_arr,
            relabel,
        })
    }

    fn has_paged_adapter(&self) -> bool {
        self.paged_adapter.is_some()
    }

    fn wired_limit_bytes(&self) -> Option<usize> {
        // qwen3 never created a WiredLimitContext anywhere — `None`
        // skips the context entirely (constructing one always mutates
        // the device wired limit regardless of the byte argument).
        None
    }

    fn profiler_label(&self, is_delta: bool, is_streaming: bool) -> &'static str {
        // Record the turn's streaming-ness for `begin_decode`'s relabel
        // (`TurnSetup` does not carry it). The session core calls this
        // hook exactly once per generic-flow turn, before
        // `begin_decode`; paged turns return from the probe earlier and
        // never consult either hook.
        self.turn_is_streaming.set(is_streaming);
        match (is_streaming, is_delta) {
            (true, false) => "qwen3_chat_stream",
            (true, true) => "qwen3_chat_stream_delta",
            // Sync turns have no profiler; these labels only surface when
            // profiling is enabled.
            (false, false) => "chat",
            (false, true) => "chat_delta",
        }
    }

    fn session_holds_images(&self) -> bool {
        // Structurally always `None` for text-only qwen3; checked anyway
        // so the default `text_delta_image_guard` keeps its defensive
        // behavior.
        self.cached_image_key.is_some()
    }

    fn paged_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Option<Result<TurnOutput>> {
        // The probe is gated on `has_paged_adapter()`; with the adapter
        // configured EVERY turn (fresh + delta, sync + streaming) takes
        // the generic paged engine. `run_paged_turn` drives the adapter
        // lifecycle via [`PagedBackend`], reusing the shared
        // `run_decode_loop`. The probe runs for delta streaming turns too,
        // routed by the engine's session-core dispatch.
        Some(crate::engine::paged_turn::run_paged_turn(self, args))
    }
}

/// Qwen3 Model with automatic differentiation support
///
/// Uses a dedicated model thread for inference and training commands.
/// Training commands are routed via `TrainingDispatch`.
#[napi]
pub struct Qwen3Model {
    /// Dedicated model thread for inference and training.
    pub(crate) thread: ModelThread<Qwen3Cmd>,
    pub(crate) config: Qwen3Config,
    // Tokenizer for text-to-text generation (loaded via load)
    pub(crate) tokenizer: Option<Arc<Qwen3Tokenizer>>,
    /// Snapshot of `Qwen3Inner::paged_adapter.is_some()` captured at
    /// construction time. The block-paged KV adapter is wired up once at
    /// load (under `config.use_block_paged_cache.unwrap_or(true)`) and
    /// never re-allocated for the life of the model, so a `bool` snapshot
    /// is sufficient — no model-thread roundtrip is needed for callers
    /// that just want to know whether the native cache reuses SYS blocks
    /// across requests via content-addressing. Surfaced through the
    /// `hasBlockPagedCache()` NAPI method so the server-side
    /// `/v1/messages` endpoint can bypass the (now-redundant) JS warm
    /// slot when paged is active.
    pub(crate) paged_active: bool,
    /// RAII: on drop (JS GC'd the wrapper) unregister this model's
    /// baseline from the cache-limit coordinator so the global cap can
    /// shrink back. Held as a field rather than consumed because the
    /// guard has no API — only its Drop does useful work.
    pub(crate) _cache_limit_guard: crate::cache_limit::CacheLimitGuard,
}

#[napi]
impl Qwen3Model {
    /// Whether the block-paged KV cache adapter is active on this model
    /// instance.
    ///
    /// `true` iff `Qwen3Inner::paged_adapter` was successfully constructed
    /// at load time (driven by `Qwen3Config::use_block_paged_cache`,
    /// defaulting to `true` for Qwen3 since paged-vs-flat parity has been
    /// verified). When `true`, the native cache reuses SYS blocks across
    /// `chatSessionStart` calls via content-addressing in
    /// `BlockAllocator`'s prefix-hash table — the JS-side warm slot in
    /// `SessionRegistry.getOrCreateWarmAny` becomes redundant and the
    /// `/v1/messages` server endpoint allocates a fresh `ChatSession` per
    /// request. See `packages/server/src/endpoints/messages.ts` for the
    /// runtime-routing decision.
    #[napi]
    pub fn has_block_paged_cache(&self) -> bool {
        self.paged_active
    }

    /// Fused forward pass using C++ implementation for maximum performance.
    /// Reduces FFI calls from ~300 to 1 per forward pass.
    /// Updates KV cache in-place to avoid allocations (matches mlx-lm's overwrite_descriptor pattern).
    /// Fused forward pass with array offsets for batched generation.
    ///
    /// Uses per-sequence RoPE offsets to enable correct batched generation
    /// when group_size > 1. Each batch element can have a different position
    /// offset for RoPE encoding.
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs [batch, seq_len]
    /// * `cache_idx` - Current write position in KV cache (shared across all layers)
    /// * `rope_offsets` - Per-sequence RoPE offsets [batch]
    /// * `left_padding` - Per-sequence left padding amounts [batch]
    fn forward_fused(
        input_ids: &MxArray,
        embedding_weight: &MxArray,
        layers: &[TransformerBlock],
        final_norm: &RMSNorm,
        lm_head: &Linear,
        config: &Qwen3Config,
        kv_keys: &mut [Option<MxArray>],
        kv_values: &mut [Option<MxArray>],
        cache_idx: &mut i32,
        rope_offsets: &MxArray,
        left_padding: &MxArray,
    ) -> Result<MxArray> {
        use mlx_sys as sys;
        use std::ptr;

        let num_layers = layers.len();

        // Collect layer weights into a flat array (11 weights per layer)
        let mut layer_weights: Vec<*mut sys::mlx_array> = Vec::with_capacity(num_layers * 11);

        for layer in layers.iter() {
            layer_weights.push(layer.get_input_layernorm_weight().handle.0);
            layer_weights.push(layer.get_post_attention_layernorm_weight().handle.0);
            layer_weights.push(layer.self_attn.get_q_proj_weight().handle.0);
            layer_weights.push(layer.self_attn.get_k_proj_weight().handle.0);
            layer_weights.push(layer.self_attn.get_v_proj_weight().handle.0);
            layer_weights.push(layer.self_attn.get_o_proj_weight().handle.0);
            if let Some(q_norm) = layer.self_attn.get_q_norm_weight() {
                layer_weights.push(q_norm.handle.0);
            } else {
                layer_weights.push(ptr::null_mut());
            }
            if let Some(k_norm) = layer.self_attn.get_k_norm_weight() {
                layer_weights.push(k_norm.handle.0);
            } else {
                layer_weights.push(ptr::null_mut());
            }
            layer_weights.push(layer.mlp.get_gate_proj_weight().handle.0);
            layer_weights.push(layer.mlp.get_up_proj_weight().handle.0);
            layer_weights.push(layer.mlp.get_down_proj_weight().handle.0);
        }

        let final_norm_weight = final_norm.get_weight();
        let lm_head_weight_handle = if config.tie_word_embeddings {
            ptr::null_mut()
        } else {
            lm_head.get_weight().handle.0
        };

        // Prepare KV cache input pointers
        let kv_keys_ptrs: Vec<*mut sys::mlx_array> = kv_keys
            .iter()
            .map(|k| k.as_ref().map(|a| a.handle.0).unwrap_or(ptr::null_mut()))
            .collect();
        let kv_values_ptrs: Vec<*mut sys::mlx_array> = kv_values
            .iter()
            .map(|v| v.as_ref().map(|a| a.handle.0).unwrap_or(ptr::null_mut()))
            .collect();

        // Prepare output arrays (will update in place)
        let mut out_logits: *mut sys::mlx_array = ptr::null_mut();
        let mut out_kv_keys: Vec<*mut sys::mlx_array> = vec![ptr::null_mut(); num_layers];
        let mut out_kv_values: Vec<*mut sys::mlx_array> = vec![ptr::null_mut(); num_layers];
        let mut out_cache_idx: i32 = 0;

        // Call the fused FFI function with array offsets
        unsafe {
            sys::mlx_qwen3_forward_step(
                input_ids.handle.0,
                embedding_weight.handle.0,
                layer_weights.as_ptr(),
                num_layers as i32,
                final_norm_weight.handle.0,
                lm_head_weight_handle,
                config.tie_word_embeddings,
                config.hidden_size,
                config.num_heads,
                config.num_kv_heads,
                config.head_dim,
                config.rope_theta as f32,
                config.rms_norm_eps as f32,
                kv_keys_ptrs.as_ptr(),
                kv_values_ptrs.as_ptr(),
                *cache_idx,
                rope_offsets.handle.0,
                left_padding.handle.0,
                &mut out_logits,
                out_kv_keys.as_mut_ptr(),
                out_kv_values.as_mut_ptr(),
                &mut out_cache_idx,
            );
        }

        // Update cache_idx
        *cache_idx = out_cache_idx;

        // Update KV cache in place - reuse existing MxArray handles when possible
        for (i, (existing, new_ptr)) in kv_keys.iter_mut().zip(out_kv_keys).enumerate() {
            if new_ptr.is_null() {
                continue;
            }
            if let Some(arr) = existing {
                // Try to reuse the existing handle
                if let Some(inner) = Arc::get_mut(&mut arr.handle) {
                    unsafe { inner.overwrite(new_ptr) };
                } else {
                    // Fall back to creating a new MxArray (shouldn't happen in fast path)
                    *existing = Some(MxArray::from_handle(new_ptr, "forward_fused kv_keys")?);
                }
            } else {
                // First time - create new MxArray
                *existing = Some(MxArray::from_handle(
                    new_ptr,
                    &format!("forward_fused kv_keys[{}]", i),
                )?);
            }
        }

        for (i, (existing, new_ptr)) in kv_values.iter_mut().zip(out_kv_values).enumerate() {
            if new_ptr.is_null() {
                continue;
            }
            if let Some(arr) = existing {
                if let Some(inner) = Arc::get_mut(&mut arr.handle) {
                    unsafe { inner.overwrite(new_ptr) };
                } else {
                    *existing = Some(MxArray::from_handle(new_ptr, "forward_fused kv_values")?);
                }
            } else {
                *existing = Some(MxArray::from_handle(
                    new_ptr,
                    &format!("forward_fused kv_values[{}]", i),
                )?);
            }
        }

        MxArray::from_handle(out_logits, "forward_fused logits")
    }

    /// Get model configuration
    #[napi]
    pub fn get_config(&self) -> Qwen3Config {
        self.config.clone()
    }
}

#[napi]
impl Qwen3Model {
    /// Text-to-text generation with integrated tokenization
    ///
    /// This is a high-level API that handles chat template formatting, tokenization,
    /// generation, and decoding internally. It takes chat messages, applies the ChatML
    /// template, generates tokens, and decodes them back to text.
    ///
    /// # Arguments
    /// * `messages` - Array of chat messages with role and content
    /// * `config` - Generation configuration
    ///
    /// # Returns
    /// * GenerationResult with text, tokens, logprobs, finish reason, and token count
    ///
    /// # Example
    /// ```typescript
    /// const model = await Qwen3Model.load("path/to/model");
    /// const messages = [
    ///   { role: "user", content: "What is 2+2?" }
    /// ];
    /// const result = await model.generate(messages, {
    ///   maxNewTokens: 50,
    ///   temperature: 0.8,
    ///   topP: 0.95,
    /// });
    /// console.log(result.text); // Decoded text output
    /// console.log(result.tokens); // Token IDs (for GRPO)
    /// console.log(result.logprobs); // Log probabilities (for GRPO)
    /// ```
    #[napi]
    pub async fn generate(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<GenerationConfig>,
    ) -> Result<GenerationResult> {
        send_and_await(&self.thread, |reply| Qwen3Cmd::Generate {
            messages,
            config,
            reply,
        })
        .await
    }

    /// Generate multiple completions for multiple prompts in batch
    ///
    /// This is an optimized method for GRPO training that generates G completions
    /// for each of N prompts. It performs all tokenization, generation, and decoding
    /// in 3 blocking tasks instead of N*(1+2G) tasks.
    ///
    /// # Arguments
    /// * `prompts` - Array of N prompt message arrays
    /// * `group_size` - Number of completions (G) to generate per prompt
    /// * `config` - Generation configuration (sampling params, etc.)
    ///
    /// # Returns
    /// * BatchGenerationResult containing N*G completions with:
    ///   - tokens: Flat array of N*G token arrays
    ///   - logprobs: Flat array of N*G logprob arrays
    ///   - texts: Flat array of N*G decoded texts
    ///   - finish_reasons: N arrays of G finish reasons
    ///   - token_counts: N arrays of G token counts
    ///
    /// # Performance
    /// For N=10 prompts, G=8 completions:
    /// - Old approach: N*(1 tokenize + G generate + G decode) = 10*(1+8+8) = 170 blocking tasks
    /// - New approach: 1 tokenize + N*G generate + 1 decode = 1+80+1 = 82 blocking tasks (2.1x reduction)
    ///
    /// # Example
    /// ```typescript
    /// const result = await model.generateBatch(
    ///   [messages1, messages2, ...], // N prompts
    ///   8,                             // G completions per prompt
    ///   config
    /// );
    /// ```
    #[napi]
    pub async fn generate_batch(
        &self,
        prompts: Vec<Vec<ChatMessage>>,
        group_size: u32,
        config: Option<GenerationConfig>,
    ) -> Result<BatchGenerationResult> {
        send_and_await(&self.thread, |reply| Qwen3Cmd::GenerateBatch {
            prompts,
            group_size,
            config,
            reply,
        })
        .await
    }

    /// Apply chat template and encode to token IDs
    ///
    /// Formats messages using ChatML format (or Jinja2 template with tools) and encodes to tokens.
    /// The model must have been loaded via load() to have a tokenizer available.
    ///
    /// # Arguments
    /// * `messages` - Array of chat messages
    /// * `add_generation_prompt` - Whether to add generation prompt (default: true)
    /// * `tools` - Optional array of tool definitions for function calling
    /// * `enable_thinking` - Optional flag to enable thinking mode (<think> tags)
    ///
    /// # Returns
    /// * Encoded token IDs as Uint32Array
    #[napi]
    pub fn apply_chat_template<'env>(
        &self,
        env: &'env Env,
        messages: Vec<ChatMessage>,
        add_generation_prompt: Option<bool>,
        tools: Option<Vec<ToolDefinition>>,
        enable_thinking: Option<bool>,
    ) -> Result<PromiseRaw<'env, Uint32ArraySlice<'env>>> {
        let tokenizer = self.tokenizer.clone().ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                "Tokenizer not available. Model must be loaded via load().",
            )
        })?;

        // Delegate to tokenizer which handles both simple ChatML and Jinja2 with tools
        tokenizer.apply_chat_template(env, messages, add_generation_prompt, tools, enable_thinking)
    }
}

crate::models::chat_napi::chat_napi_surface! {
    class: Qwen3Model,
    thread_cmd: Qwen3Cmd,
    thread: direct,
    image_guard: text_only,
    ts_stream_start: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue: "userMessage: string, images: Uint8Array[] | null | undefined, audio: Uint8Array[] | null | undefined, config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue_tool: "toolCallId: string, content: string, config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void, isError?: boolean | null | undefined",
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repetition_cutoff_disabled() {
        // When all thresholds are 0, should never trigger
        assert_eq!(check_repetition_cutoff(&[1, 1, 1, 1, 1], 0, 0, 0), None);
        assert_eq!(
            check_repetition_cutoff(&[1, 2, 1, 2, 1, 2, 1, 2], 0, 0, 0),
            None
        );
    }

    #[test]
    fn test_repetition_cutoff_consecutive_triggers() {
        // 5 consecutive tokens with max=5 → triggers
        assert_eq!(
            check_repetition_cutoff(&[1, 1, 1, 1, 1], 5, 3, 64),
            Some("repetition")
        );

        // 6 consecutive tokens with max=5 → triggers
        assert_eq!(
            check_repetition_cutoff(&[1, 1, 1, 1, 1, 1], 5, 3, 64),
            Some("repetition")
        );
    }

    #[test]
    fn test_repetition_cutoff_consecutive_no_trigger() {
        // 4 consecutive tokens with max=5 → no trigger
        assert_eq!(check_repetition_cutoff(&[1, 1, 1, 1], 5, 3, 64), None);

        // Varied tokens → no trigger
        assert_eq!(check_repetition_cutoff(&[1, 2, 3, 4, 5], 5, 3, 64), None);

        // Pattern broken at end → no trigger
        assert_eq!(check_repetition_cutoff(&[1, 1, 1, 1, 2], 5, 3, 64), None);
    }

    #[test]
    fn test_repetition_cutoff_ngram_triggers() {
        // 4 repetitions of 2-gram [1, 2] with max=4 → triggers
        assert_eq!(
            check_repetition_cutoff(&[1, 2, 1, 2, 1, 2, 1, 2], 16, 4, 64),
            Some("repetition")
        );

        // 3 repetitions of 3-gram [1, 2, 3] with max=3 → triggers
        assert_eq!(
            check_repetition_cutoff(&[1, 2, 3, 1, 2, 3, 1, 2, 3], 16, 3, 64),
            Some("repetition")
        );
    }

    #[test]
    fn test_repetition_cutoff_ngram_no_trigger() {
        // Only 3 repetitions of 2-gram with max=4 → no trigger
        assert_eq!(
            check_repetition_cutoff(&[1, 2, 1, 2, 1, 2], 16, 4, 64),
            None
        );

        // Pattern broken → no trigger
        assert_eq!(
            check_repetition_cutoff(&[1, 2, 1, 2, 3, 2, 1, 2], 16, 4, 64),
            None
        );
    }

    #[test]
    fn test_repetition_cutoff_short_sequences() {
        // Very short sequences should not trigger
        assert_eq!(check_repetition_cutoff(&[], 5, 4, 64), None);
        assert_eq!(check_repetition_cutoff(&[1], 5, 4, 64), None);
        assert_eq!(check_repetition_cutoff(&[1, 1], 5, 4, 64), None);
    }

    #[test]
    fn test_repetition_cutoff_default_thresholds() {
        // Test with new defaults (16 consecutive, 3 n-gram repeats, 64 max pattern size)

        // 16 consecutive → triggers
        let tokens: Vec<u32> = vec![1; 16];
        assert_eq!(
            check_repetition_cutoff(&tokens, 16, 3, 64),
            Some("repetition")
        );

        // 15 consecutive → now triggers via range detection (2-gram [1,1] repeats 7x >= 3)
        let tokens: Vec<u32> = vec![1; 15];
        assert_eq!(
            check_repetition_cutoff(&tokens, 16, 3, 64),
            Some("repetition")
        );

        // 5 non-repeating tokens → no trigger
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        assert_eq!(check_repetition_cutoff(&tokens, 16, 3, 64), None);

        // 3 repetitions of 3-gram → triggers (with max_pattern_size=64)
        let tokens: Vec<u32> = (0..9).map(|i| (i % 3) as u32 + 1).collect(); // [1,2,3,1,2,3,1,2,3]
        assert_eq!(
            check_repetition_cutoff(&tokens, 16, 3, 64),
            Some("repetition")
        );
    }

    #[test]
    fn test_repetition_cutoff_long_pattern() {
        // Simulate a long phrase-level repetition (50-token pattern repeated 3 times)
        let pattern: Vec<u32> = (0..50).map(|i| i as u32 + 100).collect();
        let mut tokens = Vec::new();
        for _ in 0..3 {
            tokens.extend_from_slice(&pattern);
        }
        assert_eq!(
            check_repetition_cutoff(&tokens, 16, 3, 64),
            Some("repetition")
        );

        // Only 2 repetitions with min_count=3 → no trigger
        let mut tokens2 = Vec::new();
        for _ in 0..2 {
            tokens2.extend_from_slice(&pattern);
        }
        assert_eq!(check_repetition_cutoff(&tokens2, 16, 3, 64), None);
    }

    // -------------------------------------------------------------
    // verify_cache_prefix invariant tests
    //
    // The prefix-match MUST return either `0` or the full cached
    // length — never an intermediate value. This invariant is what
    // keeps the reset-on-miss refactor safe: a non-zero return
    // always means "the new prompt strictly extends the cached
    // history" so caches can be reused as-is without any
    // mid-sequence rewind (Qwen3 Dense has no GDN state today, but
    // the same invariant future-proofs any recurrent sibling).
    // -------------------------------------------------------------

    #[test]
    fn test_verify_cache_prefix_disabled() {
        let cached = vec![1u32, 2, 3];
        // reuse_cache=false short-circuits to 0 regardless of input.
        assert_eq!(
            verify_cache_prefix_pure(&[1, 2, 3, 4], &cached, true, false),
            0
        );
    }

    #[test]
    fn test_verify_cache_prefix_empty_cache() {
        // Empty cached history — cold start, must return 0 so caller
        // proceeds with a fresh prefill.
        assert_eq!(verify_cache_prefix_pure(&[1, 2, 3], &[], false, true), 0);
        assert_eq!(verify_cache_prefix_pure(&[1, 2, 3], &[], true, true), 0);
    }

    #[test]
    fn test_verify_cache_prefix_append_hit() {
        let cached = vec![1u32, 2, 3];
        // Append: new prompt strictly extends cached history.
        assert_eq!(
            verify_cache_prefix_pure(&[1, 2, 3, 4, 5], &cached, true, true),
            3
        );
        // Exact-match (zero delta): still a "hit" — returns full
        // cached length. Caller is responsible for deciding whether
        // to re-run the last token for logits.
        assert_eq!(verify_cache_prefix_pure(&[1, 2, 3], &cached, true, true), 3);
    }

    #[test]
    fn test_verify_cache_prefix_divergence_miss() {
        let cached = vec![1u32, 2, 3];
        // Byte-divergent at position 2 — never return an
        // intermediate value (i.e., NOT 2), must return 0.
        assert_eq!(
            verify_cache_prefix_pure(&[1, 2, 9, 4], &cached, true, true),
            0
        );
        // Completely different — 0.
        assert_eq!(verify_cache_prefix_pure(&[9, 9, 9], &cached, true, true), 0);
    }

    #[test]
    fn test_verify_cache_prefix_shorter_prompt() {
        let cached = vec![1u32, 2, 3, 4, 5];
        // Prompt shorter than cached history — impossible to be a
        // strict extension. Return 0.
        assert_eq!(verify_cache_prefix_pure(&[1, 2, 3], &cached, true, true), 0);
    }

    #[test]
    fn test_verify_cache_prefix_no_kv_caches() {
        let cached = vec![1u32, 2, 3];
        // `cached_token_history` set but KV handle vectors empty —
        // defensive guard. Must return 0 so caller falls into the
        // miss branch (reset + full prefill).
        assert_eq!(
            verify_cache_prefix_pure(&[1, 2, 3, 4], &cached, false, true),
            0
        );
    }

    #[test]
    fn test_verify_cache_prefix_never_intermediate() {
        // Exhaustive invariant: for every possible prompt up to some
        // length the return value is either 0 or cached.len() — never
        // anything else.
        let cached = vec![1u32, 2, 3, 4];
        let cached_len = cached.len();
        // Try many prompts: random prefixes, random extensions,
        // random divergences.
        let test_prompts: Vec<Vec<u32>> = vec![
            vec![],
            vec![1],
            vec![1, 2],
            vec![1, 2, 3],
            vec![1, 2, 3, 4],
            vec![1, 2, 3, 4, 5],
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            vec![9, 2, 3, 4],    // diverges at 0
            vec![1, 9, 3, 4],    // diverges at 1
            vec![1, 2, 9, 4],    // diverges at 2
            vec![1, 2, 3, 9],    // diverges at 3
            vec![1, 2, 3, 9, 5], // diverges at 3 then extended
        ];
        for prompt in test_prompts {
            let result = verify_cache_prefix_pure(&prompt, &cached, true, true);
            assert!(
                result == 0 || result == cached_len,
                "verify_cache_prefix returned intermediate value {result} for prompt {prompt:?}"
            );
        }
    }

    #[test]
    fn test_repetition_cutoff_range_detection() {
        // Range detection: a 5-token pattern repeated 3 times should be caught
        // even when max_pattern_size is much larger
        let tokens = vec![10, 20, 30, 40, 50, 10, 20, 30, 40, 50, 10, 20, 30, 40, 50];
        assert_eq!(
            check_repetition_cutoff(&tokens, 16, 3, 64),
            Some("repetition")
        );

        // Same pattern but only 2 repeats → no trigger
        let tokens2 = vec![10, 20, 30, 40, 50, 10, 20, 30, 40, 50];
        assert_eq!(check_repetition_cutoff(&tokens2, 16, 3, 64), None);
    }

    /// `use_block_paged_cache` MUST default to `None` (treated as false) when
    /// the JSON config does not set it — otherwise loading any pre-existing
    /// Qwen3 checkpoint would silently switch the storage backend.
    ///
    /// Pure-CPU; no MLX runtime needed.
    #[test]
    fn test_use_block_paged_cache_defaults_to_none_via_serde() {
        // Round-trip a Qwen3Config JSON that omits the new field. Serde
        // `#[serde(default)]` should populate it as `None`.
        let json = serde_json::json!({
            "vocab_size": 0,
            "hidden_size": 0,
            "num_layers": 1,
            "num_heads": 1,
            "num_kv_heads": 1,
            "intermediate_size": 1,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "max_position_embeddings": 2048,
            "head_dim": 128,
            "use_qk_norm": true,
            "tie_word_embeddings": false,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "bos_token_id": 2,
        });
        let cfg: super::Qwen3Config =
            serde_json::from_value(json).expect("deserialize Qwen3Config");
        assert_eq!(
            cfg.use_block_paged_cache, None,
            "use_block_paged_cache must default to None on JSON without the key"
        );
        // After the default flip (parity-verified Qwen3 paged path):
        // unwrap_or(true) yields true when the JSON omits the key.
        // Explicit Some(false) still opts out.
        assert!(
            cfg.use_block_paged_cache.unwrap_or(true),
            "default unwrap_or(true) must yield true (paged on by default)"
        );
    }

    /// Tiny config compatible with the block-paged adapter's
    /// `PagedAttentionConfig::validate` constraints (head_size in the
    /// allowed set, FP8 off, etc.). Mirrors the `tiny_config` helpers in
    /// `utils/functional.rs` but with `head_dim = 32` so the LayerKVPool
    /// constructor accepts it.
    ///
    /// `use_block_paged` is `Option<bool>` so tests can distinguish the
    /// three states the production code now cares about:
    /// * `Some(true)`  — explicit opt-in, paged adapter must allocate.
    /// * `Some(false)` — explicit opt-out, paged adapter must NOT allocate.
    /// * `None`        — default-on under the new policy (`unwrap_or(true)`),
    ///   paged adapter must allocate on Metal hosts.
    #[cfg(test)]
    fn paged_tiny_config(use_block_paged: Option<bool>) -> super::Qwen3Config {
        super::Qwen3Config {
            vocab_size: 100,
            hidden_size: 64, // 2 heads * 32 head_dim
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 2,
            intermediate_size: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            max_position_embeddings: 128,
            head_dim: 32,
            use_qk_norm: true,
            tie_word_embeddings: false,
            pad_token_id: 0,
            eos_token_id: 1,
            bos_token_id: 0,
            paged_cache_memory_mb: Some(256), // smallest valid budget
            paged_block_size: Some(16),
            // The flag under test.
            use_block_paged_cache: use_block_paged,
        }
    }

    /// Explicit opt-out (`Some(false)`) must NOT allocate the block-paged
    /// adapter. Pure path that only relies on the existing MLX runtime
    /// (matches the rest of the `models::qwen3` test suite).
    ///
    /// The previous "None means no adapter" assertion was removed when the
    /// default flipped from `unwrap_or(false)` to `unwrap_or(true)`. The
    /// opt-out path is the new "no adapter" guarantee.
    #[test]
    fn test_qwen3_inner_no_paged_adapter_when_flag_is_explicit_false() {
        let cfg = paged_tiny_config(Some(false));
        let inner = super::Qwen3Inner::new(cfg).expect("Qwen3Inner::new without paged adapter");
        assert!(
            inner.paged_adapter.is_none(),
            "paged_adapter must be None when use_block_paged_cache is Some(false)"
        );
    }

    /// Default-flag construction (`None`) must allocate the block-paged
    /// adapter under the new default-on policy (`unwrap_or(true)`).
    /// Allocates a `LayerKVPool`, so requires Metal — gracefully skips on
    /// no-Metal sandboxes by matching on the LayerKVPool error string.
    #[test]
    fn test_qwen3_inner_paged_adapter_when_flag_is_none_default_on_macos() {
        // Block-paged needs the Metal backend; on a non-Metal build the
        // adapter is gated off (None) and there is nothing to exercise.
        if !crate::engine::persistence::compiled_forward_backend_available() {
            eprintln!("skipping (paged backend unavailable without Metal)");
            return;
        }
        let cfg = paged_tiny_config(None);
        match super::Qwen3Inner::new(cfg) {
            Ok(inner) => {
                assert!(
                    inner.paged_adapter.is_some(),
                    "paged_adapter must be Some when use_block_paged_cache is None \
                     (new default-on policy: unwrap_or(true))"
                );
            }
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Qwen3Inner::new failure: {msg}");
            }
        }
    }

    /// Construction with `use_block_paged_cache: Some(true)` must populate
    /// `paged_adapter`. Allocates a `LayerKVPool`, so requires Metal — gracefully
    /// skips on no-Metal sandboxes by matching on the LayerKVPool error string
    /// (mirrors the pattern used in the `paged_kv_cache_adapter` test module).
    #[test]
    fn test_qwen3_inner_constructs_paged_adapter_when_flag_is_true() {
        // Block-paged needs the Metal backend; on a non-Metal build the
        // adapter is gated off (None) and there is nothing to exercise.
        if !crate::engine::persistence::compiled_forward_backend_available() {
            eprintln!("skipping (paged backend unavailable without Metal)");
            return;
        }
        let cfg = paged_tiny_config(Some(true));
        match super::Qwen3Inner::new(cfg) {
            Ok(inner) => {
                assert!(
                    inner.paged_adapter.is_some(),
                    "paged_adapter must be Some when use_block_paged_cache = Some(true)"
                );
            }
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Qwen3Inner::new failure: {msg}");
            }
        }
    }

    /// **Smoke test for `paged_turn_sync_core`**. Without real weights /
    /// tokenizer we cannot drive the full chat path, but we CAN drive the
    /// underlying `run_paged_prefill_chunk` + `run_paged_decode_step`
    /// helpers that the chat path delegates to. This validates the
    /// adapter lifecycle (reset → find_cached_prefix → allocate_suffix →
    /// record_tokens → forward_paged_adapter), the prefill SDPA path
    /// (no-cache branch), and the decode-loop control flow against a
    /// freshly-constructed Qwen3Inner with random-init weights.
    ///
    /// What we assert:
    /// * Prefill on a 4-token "prompt" produces logits with shape
    ///   `[vocab]` and finite values.
    /// * Two decode steps produce non-empty u32 token ids.
    /// * Adapter's `current_token_count()` matches the cumulative
    ///   prefill + decode tokens.
    /// * No panics during the lifecycle.
    ///
    /// What we do NOT assert: numerical equivalence to the flat path.
    /// Weights are random, so output values are arbitrary. Numerical
    /// validation is covered by an end-to-end test with loaded weights
    /// (gated on tokenizer + checkpoint loading).
    ///
    /// Skips on no-Metal hosts via the `Qwen3Inner::new` Metal-availability
    /// check (existing pattern from
    /// `test_qwen3_inner_constructs_paged_adapter_when_flag_is_true`).
    #[test]
    fn test_paged_turn_sync_core_smoke_via_helpers() {
        // Block-paged needs the Metal backend; on a non-Metal build the
        // adapter is gated off (None) and there is nothing to exercise.
        if !crate::engine::persistence::compiled_forward_backend_available() {
            eprintln!("skipping (paged backend unavailable without Metal)");
            return;
        }
        let cfg = paged_tiny_config(Some(true));
        let mut inner = match super::Qwen3Inner::new(cfg.clone()) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!(
                        "skipping test_paged_turn_sync_core_smoke_via_helpers (no Metal): {msg}"
                    );
                    return;
                }
                panic!("unexpected Qwen3Inner::new failure: {msg}");
            }
        };
        assert!(
            inner.paged_adapter.is_some(),
            "paged_tiny_config(Some(true)) must construct paged_adapter"
        );

        // The default `Embedding::new` / `Linear::new` random-init produces
        // Float32 weights. The block-paged adapter's pool was constructed
        // BFloat16 (Qwen3 production dtype), so the K/V the layers compute
        // would be Float32 and `update_keys_values` would (correctly) reject
        // them. Cast every weight to BFloat16 to match the production
        // configuration the chat path will see at inference time.
        use crate::array::DType;
        let cast = |a: &MxArray| -> MxArray { a.astype(DType::BFloat16).expect("astype BFloat16") };
        // Embedding.
        let w = inner.embedding.get_weight();
        inner.embedding.set_weight(&cast(&w)).expect("set embed");
        // Final norm.
        let w = inner.final_norm.get_weight();
        inner
            .final_norm
            .set_weight(&cast(&w))
            .expect("set final_norm");
        // LM head.
        let w = inner.lm_head.get_weight();
        inner.lm_head.set_weight(&cast(&w)).expect("set lm_head");
        // Per-layer.
        for layer in inner.layers.iter_mut() {
            let w = layer.get_input_layernorm_weight();
            layer
                .set_input_layernorm_weight(&cast(&w))
                .expect("set in ln");
            let w = layer.get_post_attention_layernorm_weight();
            layer
                .set_post_attention_layernorm_weight(&cast(&w))
                .expect("set post ln");
            let w = layer.self_attn.get_q_proj_weight();
            layer.self_attn.set_q_proj_weight(&cast(&w)).expect("set q");
            let w = layer.self_attn.get_k_proj_weight();
            layer.self_attn.set_k_proj_weight(&cast(&w)).expect("set k");
            let w = layer.self_attn.get_v_proj_weight();
            layer.self_attn.set_v_proj_weight(&cast(&w)).expect("set v");
            let w = layer.self_attn.get_o_proj_weight();
            layer.self_attn.set_o_proj_weight(&cast(&w)).expect("set o");
            if let Some(qn) = layer.self_attn.get_q_norm_weight() {
                layer
                    .self_attn
                    .set_q_norm_weight(&cast(&qn))
                    .expect("set qn");
            }
            if let Some(kn) = layer.self_attn.get_k_norm_weight() {
                layer
                    .self_attn
                    .set_k_norm_weight(&cast(&kn))
                    .expect("set kn");
            }
            let w = layer.mlp.get_gate_proj_weight();
            layer.mlp.set_gate_proj_weight(&cast(&w)).expect("set gate");
            let w = layer.mlp.get_up_proj_weight();
            layer.mlp.set_up_proj_weight(&cast(&w)).expect("set up");
            let w = layer.mlp.get_down_proj_weight();
            layer.mlp.set_down_proj_weight(&cast(&w)).expect("set down");
        }

        // Drive the adapter lifecycle the same way `paged_turn_sync_core`
        // does. seq_id is arbitrary (per-request scoping).
        let prompt: Vec<u32> = vec![10, 20, 30, 40];
        let max_decode: u32 = 2;

        {
            let adapter = inner
                .paged_adapter
                .as_mut()
                .expect("paged_adapter constructed above");
            adapter
                .reset_for_new_request(0)
                .expect("reset_for_new_request");
            // First-turn cache miss → cached_prefix_len = 0.
            let prefix = adapter
                .find_cached_prefix(&prompt, &[], 0, false)
                .expect("find_cached_prefix");
            assert_eq!(prefix.cached_token_count, 0);
            adapter
                .allocate_suffix_blocks(prompt.len() as u32 + max_decode)
                .expect("allocate_suffix_blocks");
        }

        let positions = MxArray::from_int32(&[0], &[1]).expect("positions");
        let num_layers = inner.layers.len();

        // Prefill the suffix == full prompt (cached_prefix_len = 0).
        let logits = match inner.run_paged_prefill_chunk(&prompt, 0, num_layers, &positions) {
            Ok(l) => l,
            Err(e) => {
                let msg = e.reason.to_string();
                if msg.contains("Metal GPU not available") || msg.contains("No Metal device") {
                    eprintln!("skipping test_paged_turn_sync_core_smoke_via_helpers: {msg}");
                    return;
                }
                panic!("unexpected run_paged_prefill_chunk failure: {msg}");
            }
        };
        // Logits shape: [vocab].
        assert_eq!(
            logits.ndim().expect("ndim"),
            1,
            "prefill logits must be 1-D"
        );
        assert_eq!(
            logits.shape_at(0).expect("shape_at(0)"),
            cfg.vocab_size as i64,
            "prefill logits must be [vocab]"
        );
        // Spot-check a few values are finite. `item_at_float32` reads after
        // an `eval`; the BF16 logits round-trip through cast on read.
        let logits_f32 = logits
            .astype(crate::array::DType::Float32)
            .expect("astype f32");
        logits_f32.eval();
        let v0 = logits_f32.item_at_float32(0).expect("item_at_float32(0)");
        assert!(v0.is_finite(), "prefill logits[0] must be finite, got {v0}");

        // Adapter cursor should now equal prompt length.
        {
            let adapter = inner.paged_adapter.as_ref().unwrap();
            assert_eq!(adapter.current_token_count(), prompt.len() as u32);
        }

        // Two decode steps with arbitrary token values (the random-weight
        // model would normally pick its own; we just verify the path works
        // for the supplied token ids).
        for (i, tok) in [50u32, 60u32].iter().enumerate() {
            let next_logits = match inner.run_paged_decode_step(*tok, num_layers, &positions) {
                Ok(l) => l,
                Err(e) => {
                    let msg = e.reason.to_string();
                    if msg.contains("Metal GPU not available") || msg.contains("No Metal device") {
                        eprintln!("skipping test_paged_turn_sync_core_smoke_via_helpers: {msg}");
                        return;
                    }
                    panic!("unexpected run_paged_decode_step failure on step {i}: {msg}");
                }
            };
            // Decode logits shape: [1, 1, vocab].
            assert_eq!(next_logits.ndim().expect("ndim"), 3);
            assert_eq!(
                next_logits.shape_at(2).expect("shape_at(2)"),
                cfg.vocab_size as i64
            );
            let next_f32 = next_logits
                .astype(crate::array::DType::Float32)
                .expect("astype f32");
            next_f32.eval();
            let v = next_f32.item_at_float32(0).expect("item_at_float32(0)");
            assert!(
                v.is_finite(),
                "decode logits[0] step {i} must be finite, got {v}"
            );
        }

        // Cursor advanced by 2 decode tokens.
        {
            let adapter = inner.paged_adapter.as_ref().unwrap();
            assert_eq!(
                adapter.current_token_count(),
                prompt.len() as u32 + 2,
                "adapter cursor must reflect 4 prefill + 2 decode tokens"
            );
        }

        // Cleanup mirrors the paged_turn_sync_core success path.
        {
            let adapter = inner.paged_adapter.as_mut().unwrap();
            let _ = adapter.register_full_blocks_for_reuse(&[], 0);
            adapter.release_request().expect("release_request");
        }
    }

    /// `use_block_paged_cache: true` round-trips correctly through serde —
    /// regression guard against a future rename / serde annotation drift.
    #[test]
    fn test_use_block_paged_cache_round_trips_true() {
        let json = serde_json::json!({
            "vocab_size": 0,
            "hidden_size": 0,
            "num_layers": 1,
            "num_heads": 1,
            "num_kv_heads": 1,
            "intermediate_size": 1,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "max_position_embeddings": 2048,
            "head_dim": 128,
            "use_qk_norm": true,
            "tie_word_embeddings": false,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "bos_token_id": 2,
            "use_block_paged_cache": true,
        });
        let cfg: super::Qwen3Config =
            serde_json::from_value(json).expect("deserialize Qwen3Config");
        assert_eq!(cfg.use_block_paged_cache, Some(true));
    }

    /// **Streaming smoke test for the paged path**, structurally analogous
    /// to [`test_paged_turn_sync_core_smoke_via_helpers`]. Drives the
    /// same `run_paged_prefill_chunk` + `run_paged_decode_step` helpers
    /// the streaming variant delegates to (the streaming entry just adds
    /// per-token emit + cancellation), so the helper-level coverage
    /// transitively validates the streaming control flow.
    ///
    /// What we assert:
    /// * Prefill on a 4-token "prompt" produces logits with shape
    ///   `[vocab]` and finite values.
    /// * Two decode steps via `run_paged_decode_step` succeed and produce
    ///   3-D logits.
    /// * Adapter cursor advances correctly across prefill + decode.
    ///
    /// Skips on no-Metal hosts.
    #[test]
    fn test_paged_turn_stream_core_smoke_via_helpers() {
        // Block-paged needs the Metal backend; on a non-Metal build the
        // adapter is gated off (None) and there is nothing to exercise.
        if !crate::engine::persistence::compiled_forward_backend_available() {
            eprintln!("skipping (paged backend unavailable without Metal)");
            return;
        }
        let cfg = paged_tiny_config(Some(true));
        let mut inner = match super::Qwen3Inner::new(cfg.clone()) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!(
                        "skipping test_paged_turn_stream_core_smoke_via_helpers (no Metal): {msg}"
                    );
                    return;
                }
                panic!("unexpected Qwen3Inner::new failure: {msg}");
            }
        };
        assert!(
            inner.paged_adapter.is_some(),
            "paged_tiny_config(Some(true)) must construct paged_adapter"
        );

        // Cast all model weights to BFloat16 (paged pool dtype). Same
        // rationale as `test_paged_turn_sync_core_smoke_via_helpers`.
        use crate::array::DType;
        let cast = |a: &MxArray| -> MxArray { a.astype(DType::BFloat16).expect("astype BFloat16") };
        let w = inner.embedding.get_weight();
        inner.embedding.set_weight(&cast(&w)).expect("set embed");
        let w = inner.final_norm.get_weight();
        inner
            .final_norm
            .set_weight(&cast(&w))
            .expect("set final_norm");
        let w = inner.lm_head.get_weight();
        inner.lm_head.set_weight(&cast(&w)).expect("set lm_head");
        for layer in inner.layers.iter_mut() {
            let w = layer.get_input_layernorm_weight();
            layer
                .set_input_layernorm_weight(&cast(&w))
                .expect("set in ln");
            let w = layer.get_post_attention_layernorm_weight();
            layer
                .set_post_attention_layernorm_weight(&cast(&w))
                .expect("set post ln");
            let w = layer.self_attn.get_q_proj_weight();
            layer.self_attn.set_q_proj_weight(&cast(&w)).expect("set q");
            let w = layer.self_attn.get_k_proj_weight();
            layer.self_attn.set_k_proj_weight(&cast(&w)).expect("set k");
            let w = layer.self_attn.get_v_proj_weight();
            layer.self_attn.set_v_proj_weight(&cast(&w)).expect("set v");
            let w = layer.self_attn.get_o_proj_weight();
            layer.self_attn.set_o_proj_weight(&cast(&w)).expect("set o");
            if let Some(qn) = layer.self_attn.get_q_norm_weight() {
                layer
                    .self_attn
                    .set_q_norm_weight(&cast(&qn))
                    .expect("set qn");
            }
            if let Some(kn) = layer.self_attn.get_k_norm_weight() {
                layer
                    .self_attn
                    .set_k_norm_weight(&cast(&kn))
                    .expect("set kn");
            }
            let w = layer.mlp.get_gate_proj_weight();
            layer.mlp.set_gate_proj_weight(&cast(&w)).expect("set gate");
            let w = layer.mlp.get_up_proj_weight();
            layer.mlp.set_up_proj_weight(&cast(&w)).expect("set up");
            let w = layer.mlp.get_down_proj_weight();
            layer.mlp.set_down_proj_weight(&cast(&w)).expect("set down");
        }

        // Drive the adapter lifecycle the same way
        // `paged_turn_stream_core` does.
        let prompt: Vec<u32> = vec![10, 20, 30, 40];
        let max_decode: u32 = 2;
        {
            let adapter = inner
                .paged_adapter
                .as_mut()
                .expect("paged_adapter constructed above");
            adapter
                .reset_for_new_request(0)
                .expect("reset_for_new_request");
            let prefix = adapter
                .find_cached_prefix(&prompt, &[], 0, false)
                .expect("find_cached_prefix");
            assert_eq!(prefix.cached_token_count, 0);
            adapter
                .allocate_suffix_blocks(prompt.len() as u32 + max_decode)
                .expect("allocate_suffix_blocks");
        }

        let positions = MxArray::from_int32(&[0], &[1]).expect("positions");
        let num_layers = inner.layers.len();

        // Prefill chunk (suffix = full prompt, cached_prefix_len = 0).
        let logits = match inner.run_paged_prefill_chunk(&prompt, 0, num_layers, &positions) {
            Ok(l) => l,
            Err(e) => {
                let msg = e.reason.to_string();
                if msg.contains("Metal GPU not available") || msg.contains("No Metal device") {
                    eprintln!("skipping test_paged_turn_stream_core_smoke_via_helpers: {msg}");
                    return;
                }
                panic!("unexpected run_paged_prefill_chunk failure: {msg}");
            }
        };
        assert_eq!(logits.ndim().expect("ndim"), 1);
        assert_eq!(
            logits.shape_at(0).expect("shape_at(0)"),
            cfg.vocab_size as i64
        );

        let logits_f32 = logits
            .astype(crate::array::DType::Float32)
            .expect("astype f32");
        logits_f32.eval();
        let v0 = logits_f32.item_at_float32(0).expect("item_at_float32(0)");
        assert!(
            v0.is_finite(),
            "stream-paged prefill logits[0] must be finite, got {v0}"
        );

        {
            let adapter = inner.paged_adapter.as_ref().unwrap();
            assert_eq!(adapter.current_token_count(), prompt.len() as u32);
        }

        // Two decode steps (matches the streaming inner loop's per-step
        // dispatch — the only difference vs. the non-streaming variant
        // is the per-token streaming emit, which doesn't change the
        // forward path under test).
        for (i, tok) in [50u32, 60u32].iter().enumerate() {
            let next_logits = match inner.run_paged_decode_step(*tok, num_layers, &positions) {
                Ok(l) => l,
                Err(e) => {
                    let msg = e.reason.to_string();
                    if msg.contains("Metal GPU not available") || msg.contains("No Metal device") {
                        eprintln!("skipping test_paged_turn_stream_core_smoke_via_helpers: {msg}");
                        return;
                    }
                    panic!("unexpected run_paged_decode_step failure on step {i}: {msg}");
                }
            };
            assert_eq!(next_logits.ndim().expect("ndim"), 3);
            assert_eq!(
                next_logits.shape_at(2).expect("shape_at(2)"),
                cfg.vocab_size as i64
            );
            let next_f32 = next_logits
                .astype(crate::array::DType::Float32)
                .expect("astype f32");
            next_f32.eval();
            let v = next_f32.item_at_float32(0).expect("item_at_float32(0)");
            assert!(
                v.is_finite(),
                "stream-paged decode logits[0] step {i} must be finite, got {v}"
            );
        }

        {
            let adapter = inner.paged_adapter.as_ref().unwrap();
            assert_eq!(
                adapter.current_token_count(),
                prompt.len() as u32 + 2,
                "adapter cursor must reflect 4 prefill + 2 decode tokens"
            );
        }

        // Cleanup mirrors the paged_turn_stream_core success path.
        {
            let adapter = inner.paged_adapter.as_mut().unwrap();
            let _ = adapter.register_full_blocks_for_reuse(&[], 0);
            adapter.release_request().expect("release_request");
        }
    }

    /// Helper used by the chunked-prefill tests below: cast every weight in
    /// `inner` to BFloat16 so the K/V the layers compute matches the
    /// paged-pool dtype (mirrors the `test_paged_turn_sync_core_smoke_via_helpers`
    /// pattern). Without this `update_keys_values` rejects the F32 K/V the
    /// random-init weights would otherwise produce.
    #[cfg(test)]
    fn cast_qwen3_inner_weights_bf16(inner: &mut super::Qwen3Inner) {
        use crate::array::DType;
        let cast = |a: &MxArray| -> MxArray { a.astype(DType::BFloat16).expect("astype BFloat16") };
        // Embedding.
        let w = inner.embedding.get_weight();
        inner.embedding.set_weight(&cast(&w)).expect("set embed");
        // Final norm.
        let w = inner.final_norm.get_weight();
        inner
            .final_norm
            .set_weight(&cast(&w))
            .expect("set final_norm");
        // LM head.
        let w = inner.lm_head.get_weight();
        inner.lm_head.set_weight(&cast(&w)).expect("set lm_head");
        // Per-layer.
        for layer in inner.layers.iter_mut() {
            let w = layer.get_input_layernorm_weight();
            layer
                .set_input_layernorm_weight(&cast(&w))
                .expect("set in ln");
            let w = layer.get_post_attention_layernorm_weight();
            layer
                .set_post_attention_layernorm_weight(&cast(&w))
                .expect("set post ln");
            let w = layer.self_attn.get_q_proj_weight();
            layer.self_attn.set_q_proj_weight(&cast(&w)).expect("set q");
            let w = layer.self_attn.get_k_proj_weight();
            layer.self_attn.set_k_proj_weight(&cast(&w)).expect("set k");
            let w = layer.self_attn.get_v_proj_weight();
            layer.self_attn.set_v_proj_weight(&cast(&w)).expect("set v");
            let w = layer.self_attn.get_o_proj_weight();
            layer.self_attn.set_o_proj_weight(&cast(&w)).expect("set o");
            if let Some(qn) = layer.self_attn.get_q_norm_weight() {
                layer
                    .self_attn
                    .set_q_norm_weight(&cast(&qn))
                    .expect("set qn");
            }
            if let Some(kn) = layer.self_attn.get_k_norm_weight() {
                layer
                    .self_attn
                    .set_k_norm_weight(&cast(&kn))
                    .expect("set kn");
            }
            let w = layer.mlp.get_gate_proj_weight();
            layer.mlp.set_gate_proj_weight(&cast(&w)).expect("set gate");
            let w = layer.mlp.get_up_proj_weight();
            layer.mlp.set_up_proj_weight(&cast(&w)).expect("set up");
            let w = layer.mlp.get_down_proj_weight();
            layer.mlp.set_down_proj_weight(&cast(&w)).expect("set down");
        }
    }

    /// Read the full contents of a 1-D `[vocab]` `MxArray` to a host
    /// `Vec<f32>`. Goes via `astype(F32)` + per-element `item_at_float32` so
    /// it works on bf16 logits as well.
    #[cfg(test)]
    fn logits_to_f32_vec(logits: &MxArray) -> Vec<f32> {
        let f32_arr = logits
            .astype(crate::array::DType::Float32)
            .expect("astype f32");
        f32_arr.eval();
        let n = f32_arr.shape_at(0).expect("shape_at(0)") as usize;
        (0..n)
            .map(|i| f32_arr.item_at_float32(i).expect("item_at_float32"))
            .collect()
    }

    /// Chunked-prefill parity test: chunked prefill with the same weights and
    /// the same suffix tokens MUST produce the same final logits as the
    /// single-shot prefill, modulo small bf16 rounding noise.
    ///
    /// Both runs share a single `Qwen3Inner` (so weights are byte-equal)
    /// and the paged-state is reset between them. The same prompt is fed
    /// through `run_paged_prefill_chunk_with_size(..., 0)` (single-shot)
    /// and then `run_paged_prefill_chunk_with_size(..., 16)` (chunked, ~6
    /// chunks for a 96-token prompt). The post-prefill `[vocab]` logits
    /// vectors are compared element-wise.
    ///
    /// Tolerance: `atol=5e-3, rtol=5e-3`. bf16 has only ~3 decimal digits
    /// of precision, and chunked prefill changes the order of GPU
    /// operations (split causal mask reshapes, intermediate evals); empty
    /// reductions / different fma orderings on a vocab-sized matmul push
    /// element-wise differences into the 1e-3 range easily. We're
    /// validating "same answer up to floating-point noise", not bitwise
    /// equality.
    ///
    /// Skips on no-Metal hosts.
    #[test]
    fn test_chunked_prefill_matches_single_shot_logits() {
        // Block-paged needs the Metal backend; on a non-Metal build the
        // adapter is gated off (None) and there is nothing to exercise.
        if !crate::engine::persistence::compiled_forward_backend_available() {
            eprintln!("skipping (paged backend unavailable without Metal)");
            return;
        }
        let cfg = paged_tiny_config(Some(true));
        let mut inner = match super::Qwen3Inner::new(cfg.clone()) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!(
                        "skipping test_chunked_prefill_matches_single_shot_logits (no Metal): \
                         {msg}"
                    );
                    return;
                }
                panic!("unexpected Qwen3Inner::new failure: {msg}");
            }
        };
        cast_qwen3_inner_weights_bf16(&mut inner);

        // 96-token prompt — must be > paged_tiny_config max_position_embeddings/2
        // so a chunk_size of 16 produces 6 chunks (the multi-chunk path is
        // what we actually want to exercise). Tokens are arbitrary modulo
        // vocab_size = 100.
        let prompt: Vec<u32> = (0u32..96).map(|i| (i * 7 + 3) % 100).collect();
        let positions = MxArray::from_int32(&[0], &[1]).expect("positions");
        let num_layers = inner.layers.len();

        // ---- Run 1: single-shot path ----
        {
            let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
            adapter.reset_for_new_request(0).expect("reset");
            let prefix = adapter
                .find_cached_prefix(&prompt, &[], 0, false)
                .expect("find_cached_prefix");
            assert_eq!(prefix.cached_token_count, 0);
            adapter
                .allocate_suffix_blocks(prompt.len() as u32)
                .expect("allocate_suffix_blocks");
        }
        let logits_single = match inner.run_paged_prefill_chunk_with_size(
            &prompt, /* first_logical_position */ 0, num_layers, &positions,
            /* chunk_size */ 0, // single-shot
        ) {
            Ok(l) => l,
            Err(e) => {
                let msg = e.reason.to_string();
                if msg.contains("Metal GPU not available") || msg.contains("No Metal device") {
                    eprintln!("skipping test_chunked_prefill_matches_single_shot_logits: {msg}");
                    return;
                }
                panic!("unexpected single-shot prefill failure: {msg}");
            }
        };
        let single_vec = logits_to_f32_vec(&logits_single);
        assert_eq!(single_vec.len(), cfg.vocab_size as usize);
        for (i, v) in single_vec.iter().enumerate() {
            assert!(v.is_finite(), "single-shot logits[{i}] not finite: {v}");
        }

        // Reset adapter so the second run starts fresh (no prefix-reuse —
        // we want to compare apples to apples).
        {
            let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
            adapter.release_request().expect("release_request");
            adapter.reset_for_new_request(0).expect("reset");
            let prefix = adapter
                .find_cached_prefix(&prompt, &[], 0, false)
                .expect("find_cached_prefix");
            assert_eq!(
                prefix.cached_token_count, 0,
                "second run must not see registered blocks from first run"
            );
            adapter
                .allocate_suffix_blocks(prompt.len() as u32)
                .expect("allocate_suffix_blocks");
        }

        // ---- Run 2: chunked path (chunk_size = 16, so 96/16 = 6 chunks) ----
        let logits_chunked = inner
            .run_paged_prefill_chunk_with_size(
                &prompt, /* first_logical_position */ 0, num_layers, &positions,
                /* chunk_size */ 16,
            )
            .expect("chunked prefill");
        let chunked_vec = logits_to_f32_vec(&logits_chunked);
        assert_eq!(chunked_vec.len(), cfg.vocab_size as usize);

        // Element-wise close-comparison. bf16 + chunk-boundary fma reorderings
        // give ~3 decimals; the assertion is generous to rule out structural
        // bugs without flaking on hardware-numerics jitter.
        let atol = 5e-3f32;
        let rtol = 5e-3f32;
        let mut max_abs_diff = 0.0f32;
        let mut max_rel_diff = 0.0f32;
        for (i, (a, b)) in single_vec.iter().zip(chunked_vec.iter()).enumerate() {
            assert!(b.is_finite(), "chunked logits[{i}] not finite: {b}");
            let abs_diff = (a - b).abs();
            let rel_diff = abs_diff / (a.abs().max(b.abs()).max(1e-6));
            if abs_diff > max_abs_diff {
                max_abs_diff = abs_diff;
            }
            if rel_diff > max_rel_diff {
                max_rel_diff = rel_diff;
            }
            assert!(
                abs_diff <= atol || rel_diff <= rtol,
                "logits diverge at index {i}: single={a}, chunked={b}, \
                 abs_diff={abs_diff}, rel_diff={rel_diff} (max allowed: \
                 atol={atol} or rtol={rtol})"
            );
        }
        eprintln!(
            "chunked-prefill parity max_abs_diff={max_abs_diff}, max_rel_diff={max_rel_diff} \
             (atol={atol}, rtol={rtol}) over {} elements",
            single_vec.len()
        );

        // Cleanup.
        {
            let adapter = inner.paged_adapter.as_mut().unwrap();
            let _ = adapter.register_full_blocks_for_reuse(&[], 0);
            adapter.release_request().expect("release_request");
        }
    }

    /// Per-chunk progression test: drive the chunked
    /// prefill chunk-by-chunk through `run_paged_prefill_one_chunk` and
    /// assert the adapter's bookkeeping advances correctly **after every
    /// chunk**, not just at the end.
    ///
    /// Why this matters: a previous version of this test fed the entire
    /// 64-token prompt through `run_paged_prefill_chunk_with_size` once
    /// and only checked final state. That assertion would still pass if
    /// the implementation silently recorded the whole suffix in one shot
    /// (no real chunking) or never grew `block_table` at chunk
    /// boundaries — exactly the regressions the test is supposed to catch.
    ///
    /// What this test asserts after EACH of the 4 chunks (16 tokens each,
    /// `block_size=16`):
    /// * `adapter.current_token_count()` equals the cumulative tokens fed
    ///   so far (16, 32, 48, 64).
    /// * `adapter.request_tokens().len()` matches that same cumulative
    ///   count (the slice itself byte-equals `prompt[..cumulative]`).
    /// * `adapter.block_table().num_blocks()` is at least
    ///   `ceil(cumulative / block_size)` — i.e. lazy growth never lags
    ///   behind the cursor. With `block_size=16` and 16-token chunks this
    ///   tightens to exactly `cumulative / 16` per chunk (1, 2, 3, 4),
    ///   but the assertion uses `>=` so we don't pin block-table growth
    ///   policy harder than the contract requires.
    ///
    /// Drives `run_paged_prefill_one_chunk` directly (it lives in the
    /// same crate's `pub(crate)` impl block, and this test module sits
    /// inside it). That bypasses the chunked driver loop while exercising
    /// the same per-chunk state-advancement code paths the driver runs.
    #[test]
    fn test_chunked_prefill_advances_adapter_state() {
        // Block-paged needs the Metal backend; on a non-Metal build the
        // adapter is gated off (None) and there is nothing to exercise.
        if !crate::engine::persistence::compiled_forward_backend_available() {
            eprintln!("skipping (paged backend unavailable without Metal)");
            return;
        }
        let cfg = paged_tiny_config(Some(true));
        let mut inner = match super::Qwen3Inner::new(cfg.clone()) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!(
                        "skipping test_chunked_prefill_advances_adapter_state (no Metal): {msg}"
                    );
                    return;
                }
                panic!("unexpected Qwen3Inner::new failure: {msg}");
            }
        };
        cast_qwen3_inner_weights_bf16(&mut inner);

        // 64 tokens, chunk_size 16 → 4 chunks at block_size=16. After
        // chunk N the block table must have ≥ ceil(N*16 / 16) = N blocks.
        let prompt: Vec<u32> = (0u32..64).map(|i| (i * 11 + 5) % 100).collect();
        let positions = MxArray::from_int32(&[0], &[1]).expect("positions");
        let num_layers = inner.layers.len();
        let block_size = cfg
            .paged_block_size
            .expect("block_size in paged_tiny_config");

        {
            let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
            adapter.reset_for_new_request(0).expect("reset");
            let prefix = adapter
                .find_cached_prefix(&prompt, &[], 0, false)
                .expect("find_cached_prefix");
            assert_eq!(prefix.cached_token_count, 0);
            adapter
                .allocate_suffix_blocks(prompt.len() as u32)
                .expect("allocate_suffix_blocks");
        }

        // Pre-flight: nothing fed yet → cursor at 0, no recorded tokens.
        // (block_table may already hold preallocated suffix blocks from
        // `allocate_suffix_blocks` above; we don't pin its starting size
        // here, only verify it grows as cumulative tokens cross block
        // boundaries below.)
        {
            let adapter = inner.paged_adapter.as_ref().expect("paged_adapter");
            assert_eq!(adapter.current_token_count(), 0, "pre-flight cursor");
            assert_eq!(
                adapter.request_tokens().len(),
                0,
                "pre-flight request_tokens"
            );
        }

        // Drive chunk-by-chunk. For each chunk we:
        //   1. Call `run_paged_prefill_one_chunk(chunk, chunk_first_pos, ...)`,
        //      which is the same per-chunk inner the chunked driver loops
        //      over.
        //   2. Materialize the residual stream (matches the driver's
        //      `hidden.eval()` between chunks so the lazy graph doesn't
        //      pile up).
        //   3. Assert cursor / request_tokens / block_table reflect the
        //      cumulative chunks processed so far.
        let chunk_size_usize: usize = 16;
        let mut cumulative: usize = 0;
        for (chunk_idx, chunk) in prompt.chunks(chunk_size_usize).enumerate() {
            let chunk_first_position = cumulative as u32;
            let hidden = match inner.run_paged_prefill_one_chunk(
                chunk,
                chunk_first_position,
                num_layers,
                &positions,
            ) {
                Ok(h) => h,
                Err(e) => {
                    let msg = e.reason.to_string();
                    if msg.contains("Metal GPU not available") || msg.contains("No Metal device") {
                        eprintln!("skipping test_chunked_prefill_advances_adapter_state: {msg}");
                        return;
                    }
                    panic!("unexpected per-chunk prefill failure on chunk {chunk_idx}: {msg}");
                }
            };
            // Mirror the driver loop: eval to release upstream graph
            // nodes before starting the next chunk's forward.
            hidden.eval();
            synchronize_and_clear_cache();

            cumulative += chunk.len();

            // Per-chunk assertions on adapter state.
            let adapter = inner.paged_adapter.as_ref().expect("paged_adapter");
            assert_eq!(
                adapter.current_token_count() as usize,
                cumulative,
                "after chunk {chunk_idx}: cursor must equal cumulative tokens fed \
                 ({cumulative}), got {}",
                adapter.current_token_count()
            );
            assert_eq!(
                adapter.request_tokens().len(),
                cumulative,
                "after chunk {chunk_idx}: request_tokens len must equal cumulative \
                 tokens fed ({cumulative}), got {}",
                adapter.request_tokens().len()
            );
            assert_eq!(
                adapter.request_tokens(),
                &prompt[..cumulative],
                "after chunk {chunk_idx}: request_tokens must byte-equal the cumulative \
                 prefix of the prompt"
            );
            // Lazy growth lower bound: enough blocks to cover every
            // cumulative token. ceil(cumulative / block_size).
            let block_table = adapter.block_table().expect("block_table");
            let required_blocks = (cumulative as u32).div_ceil(block_size) as usize;
            assert!(
                block_table.num_blocks() >= required_blocks,
                "after chunk {chunk_idx}: block_table.num_blocks() ({}) must be ≥ \
                 ceil({cumulative} / {block_size}) = {required_blocks}",
                block_table.num_blocks()
            );
        }

        assert_eq!(
            cumulative,
            prompt.len(),
            "loop must consume every prompt token"
        );

        // Cleanup.
        {
            let adapter = inner.paged_adapter.as_mut().unwrap();
            let _ = adapter.register_full_blocks_for_reuse(&[], 0);
            adapter.release_request().expect("release_request");
        }
    }

    /// Uneven-tail parity test: prove the chunked path handles
    /// a final partial chunk correctly. 97 tokens at chunk_size=16 produces
    /// 6 full chunks of 16 tokens + 1 trailing chunk of 1 token. This is
    /// the worst case for off-by-one bugs at chunk boundaries (the trailing
    /// 1-token chunk's `chunk_first_position` is 96, not aligned to a block
    /// boundary; the explicit causal mask in `forward_paged_adapter` must
    /// be built with `num_tokens=1, offset=96` rather than rounding either
    /// up or down).
    ///
    /// Compares the post-prefill `[vocab]` logits between chunked
    /// (chunk_size=16) and single-shot (chunk_size=0) runs over the same
    /// 97-token prompt. Tolerance budget mirrors the multi-chunk parity
    /// test (atol=rtol=5e-3 for bf16 + chunk-boundary fma reorderings).
    ///
    /// Skips on no-Metal hosts.
    #[test]
    fn test_chunked_prefill_uneven_tail_matches_single_shot() {
        // Block-paged needs the Metal backend; on a non-Metal build the
        // adapter is gated off (None) and there is nothing to exercise.
        if !crate::engine::persistence::compiled_forward_backend_available() {
            eprintln!("skipping (paged backend unavailable without Metal)");
            return;
        }
        let cfg = paged_tiny_config(Some(true));
        let mut inner = match super::Qwen3Inner::new(cfg.clone()) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!(
                        "skipping test_chunked_prefill_uneven_tail_matches_single_shot \
                         (no Metal): {msg}"
                    );
                    return;
                }
                panic!("unexpected Qwen3Inner::new failure: {msg}");
            }
        };
        cast_qwen3_inner_weights_bf16(&mut inner);

        // 97 tokens — 96 = 6*16 full chunks + 1 leftover token. Need
        // max_position_embeddings >= 97; paged_tiny_config gives 128.
        let prompt: Vec<u32> = (0u32..97).map(|i| (i * 13 + 7) % 100).collect();
        let positions = MxArray::from_int32(&[0], &[1]).expect("positions");
        let num_layers = inner.layers.len();

        // ---- Run 1: single-shot ----
        {
            let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
            adapter.reset_for_new_request(0).expect("reset");
            let prefix = adapter
                .find_cached_prefix(&prompt, &[], 0, false)
                .expect("find_cached_prefix");
            assert_eq!(prefix.cached_token_count, 0);
            adapter
                .allocate_suffix_blocks(prompt.len() as u32)
                .expect("allocate_suffix_blocks");
        }
        let logits_single = match inner.run_paged_prefill_chunk_with_size(
            &prompt, /* first_logical_position */ 0, num_layers, &positions,
            /* chunk_size */ 0,
        ) {
            Ok(l) => l,
            Err(e) => {
                let msg = e.reason.to_string();
                if msg.contains("Metal GPU not available") || msg.contains("No Metal device") {
                    eprintln!(
                        "skipping test_chunked_prefill_uneven_tail_matches_single_shot: {msg}"
                    );
                    return;
                }
                panic!("unexpected single-shot prefill failure: {msg}");
            }
        };
        let single_vec = logits_to_f32_vec(&logits_single);
        assert_eq!(single_vec.len(), cfg.vocab_size as usize);

        // Reset adapter for the second run.
        {
            let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
            adapter.release_request().expect("release_request");
            adapter.reset_for_new_request(0).expect("reset");
            let prefix = adapter
                .find_cached_prefix(&prompt, &[], 0, false)
                .expect("find_cached_prefix");
            assert_eq!(
                prefix.cached_token_count, 0,
                "second run must not see registered blocks from first run \
                 (no register_full_blocks call between runs)"
            );
            adapter
                .allocate_suffix_blocks(prompt.len() as u32)
                .expect("allocate_suffix_blocks");
        }

        // ---- Run 2: chunked, 7 chunks (6 of size 16 + 1 of size 1) ----
        let logits_chunked = inner
            .run_paged_prefill_chunk_with_size(
                &prompt, /* first_logical_position */ 0, num_layers, &positions,
                /* chunk_size */ 16,
            )
            .expect("uneven-tail chunked prefill");
        let chunked_vec = logits_to_f32_vec(&logits_chunked);
        assert_eq!(chunked_vec.len(), cfg.vocab_size as usize);

        // Confirm we actually exercised a 1-token trailing chunk: cursor
        // must equal 97 after the run.
        {
            let adapter = inner.paged_adapter.as_ref().expect("paged_adapter");
            assert_eq!(
                adapter.current_token_count(),
                97,
                "uneven-tail chunked prefill must record all 97 tokens \
                 (including the trailing 1-token chunk)"
            );
        }

        let atol = 5e-3f32;
        let rtol = 5e-3f32;
        let mut max_abs_diff = 0.0f32;
        let mut max_rel_diff = 0.0f32;
        for (i, (a, b)) in single_vec.iter().zip(chunked_vec.iter()).enumerate() {
            assert!(
                b.is_finite(),
                "uneven-tail chunked logits[{i}] not finite: {b}"
            );
            let abs_diff = (a - b).abs();
            let rel_diff = abs_diff / (a.abs().max(b.abs()).max(1e-6));
            if abs_diff > max_abs_diff {
                max_abs_diff = abs_diff;
            }
            if rel_diff > max_rel_diff {
                max_rel_diff = rel_diff;
            }
            assert!(
                abs_diff <= atol || rel_diff <= rtol,
                "uneven-tail logits diverge at index {i}: single={a}, chunked={b}, \
                 abs_diff={abs_diff}, rel_diff={rel_diff} (max allowed: \
                 atol={atol} or rtol={rtol})"
            );
        }
        eprintln!(
            "uneven-tail chunked-prefill parity max_abs_diff={max_abs_diff}, \
             max_rel_diff={max_rel_diff} (atol={atol}, rtol={rtol}) over {} elements",
            single_vec.len()
        );

        {
            let adapter = inner.paged_adapter.as_mut().unwrap();
            let _ = adapter.register_full_blocks_for_reuse(&[], 0);
            adapter.release_request().expect("release_request");
        }
    }

    /// Cached-prefix parity test: exercise the
    /// `cached_prefix_len > 0` (Q < K) branch of `forward_paged_adapter`
    /// that the chunked path relies on for chunks N>0 — but with a
    /// genuinely non-zero cached prefix at the START of the prefill
    /// (rather than only synthesized by the chunked driver itself).
    ///
    /// Setup: turn 1 prefills the
    /// first 32 tokens, registers full blocks, releases. Turn 2 then
    /// prefills the full 96-token prompt (whose first 32 tokens are
    /// identical to turn 1's). On turn 2, `find_cached_prefix` returns
    /// `cached_token_count = 32`, so the prefill that follows runs over
    /// a 64-token suffix while the adapter already holds 32 cached
    /// tokens — exactly the state the chunked path's middle chunks see.
    ///
    /// We then run turn 2 twice on a freshly-reset prefix-cache state:
    /// once chunked (chunk_size=16, 4 chunks of 16 tokens each over the
    /// 64-token suffix) and once single-shot (chunk_size=0). Both must
    /// produce the same final-token logits.
    ///
    /// Tolerance: same atol=rtol=5e-3 budget as the other parity tests.
    #[test]
    fn test_chunked_prefill_with_cached_prefix_matches_single_shot() {
        // Block-paged needs the Metal backend; on a non-Metal build the
        // adapter is gated off (None) and there is nothing to exercise.
        if !crate::engine::persistence::compiled_forward_backend_available() {
            eprintln!("skipping (paged backend unavailable without Metal)");
            return;
        }
        let cfg = paged_tiny_config(Some(true));
        let mut inner = match super::Qwen3Inner::new(cfg.clone()) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!(
                        "skipping test_chunked_prefill_with_cached_prefix_matches_single_shot \
                         (no Metal): {msg}"
                    );
                    return;
                }
                panic!("unexpected Qwen3Inner::new failure: {msg}");
            }
        };
        cast_qwen3_inner_weights_bf16(&mut inner);

        // Full prompt: 96 tokens. The first 32 are what turn 1 will
        // prefill + register; the remaining 64 are turn 2's "suffix
        // beyond the cached prefix". 96 = 6*16 so chunk_size=16 → 4
        // chunks of 16 over the 64-token suffix on turn 2.
        let full_prompt: Vec<u32> = (0u32..96).map(|i| (i * 7 + 3) % 100).collect();
        let prefix_len: usize = 32; // == 2 * block_size (16)
        let positions = MxArray::from_int32(&[0], &[1]).expect("positions");
        let num_layers = inner.layers.len();

        // ---- Turn 1: prefill the 32-token prefix and register its full
        // blocks so subsequent `find_cached_prefix` calls see it. ----
        {
            let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
            adapter.reset_for_new_request(0).expect("reset");
            let prefix_lookup = adapter
                .find_cached_prefix(&full_prompt[..prefix_len], &[], 0, false)
                .expect("find_cached_prefix turn 1");
            assert_eq!(
                prefix_lookup.cached_token_count, 0,
                "turn 1 must start with a cold prefix cache"
            );
            adapter
                .allocate_suffix_blocks(prefix_len as u32)
                .expect("allocate_suffix_blocks turn 1");
        }
        match inner.run_paged_prefill_chunk_with_size(
            &full_prompt[..prefix_len],
            /* first_logical_position */ 0,
            num_layers,
            &positions,
            /* chunk_size */ 0, // single-shot is fine for the seeding turn
        ) {
            Ok(_logits) => {}
            Err(e) => {
                let msg = e.reason.to_string();
                if msg.contains("Metal GPU not available") || msg.contains("No Metal device") {
                    eprintln!(
                        "skipping test_chunked_prefill_with_cached_prefix_matches_single_shot \
                         (turn 1): {msg}"
                    );
                    return;
                }
                panic!("unexpected turn-1 prefill failure: {msg}");
            }
        };
        // Register turn 1's full blocks so the BlockAllocator publishes
        // them for prefix lookup. With prefix_len=32 and block_size=16
        // we expect exactly 2 full blocks to register.
        let registered = {
            let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
            let n = adapter
                .register_full_blocks_for_reuse(&[], 0)
                .expect("register_full_blocks_for_reuse turn 1");
            adapter.release_request().expect("release_request turn 1");
            n
        };
        assert_eq!(
            registered, 2,
            "turn 1 must register exactly 2 full blocks for a 32-token prefix at block_size=16"
        );

        // ---- Turn 2 (run A): single-shot prefill of the full 96-token
        // prompt. `find_cached_prefix` should re-find the 32-token prefix
        // turn 1 just registered. ----
        let cached_a = {
            let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
            adapter.reset_for_new_request(0).expect("reset turn 2A");
            let prefix_lookup = adapter
                .find_cached_prefix(&full_prompt, &[], 0, false)
                .expect("find_cached_prefix turn 2A");
            adapter
                .allocate_suffix_blocks(full_prompt.len() as u32)
                .expect("allocate_suffix_blocks turn 2A");
            prefix_lookup.cached_token_count
        };
        assert_eq!(
            cached_a, prefix_len as u32,
            "turn 2A must rediscover the 32-token cached prefix from turn 1"
        );
        let suffix_a = &full_prompt[cached_a as usize..];
        let logits_single = match inner.run_paged_prefill_chunk_with_size(
            suffix_a, /* first_logical_position */ cached_a, num_layers, &positions,
            /* chunk_size */ 0, // single-shot
        ) {
            Ok(l) => l,
            Err(e) => {
                let msg = e.reason.to_string();
                if msg.contains("Metal GPU not available") || msg.contains("No Metal device") {
                    eprintln!(
                        "skipping test_chunked_prefill_with_cached_prefix_matches_single_shot \
                         (turn 2A): {msg}"
                    );
                    return;
                }
                panic!("unexpected turn-2A prefill failure: {msg}");
            }
        };
        let single_vec = logits_to_f32_vec(&logits_single);
        assert_eq!(single_vec.len(), cfg.vocab_size as usize);

        // Release turn 2A WITHOUT registering — we don't want turn 2A's
        // additional blocks to alter the prefix-cache state turn 2B sees.
        {
            let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
            adapter.release_request().expect("release_request turn 2A");
        }

        // ---- Turn 2 (run B): chunked prefill over the same 64-token
        // suffix. cached_prefix_len = 32 > 0 throughout, so every chunk
        // exercises the Q < K branch of `forward_paged_adapter`. ----
        let cached_b = {
            let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
            adapter.reset_for_new_request(0).expect("reset turn 2B");
            let prefix_lookup = adapter
                .find_cached_prefix(&full_prompt, &[], 0, false)
                .expect("find_cached_prefix turn 2B");
            adapter
                .allocate_suffix_blocks(full_prompt.len() as u32)
                .expect("allocate_suffix_blocks turn 2B");
            prefix_lookup.cached_token_count
        };
        assert_eq!(
            cached_b, prefix_len as u32,
            "turn 2B must rediscover the same 32-token cached prefix from turn 1"
        );
        let suffix_b = &full_prompt[cached_b as usize..];
        assert_eq!(suffix_b.len(), 64, "expected a 64-token suffix to chunk");
        let logits_chunked = inner
            .run_paged_prefill_chunk_with_size(
                suffix_b, /* first_logical_position */ cached_b, num_layers, &positions,
                /* chunk_size */ 16, // 4 chunks of 16
            )
            .expect("turn-2B chunked prefill");
        let chunked_vec = logits_to_f32_vec(&logits_chunked);
        assert_eq!(chunked_vec.len(), cfg.vocab_size as usize);

        // Sanity: cumulative state after turn 2B must equal the full 96
        // tokens (32 cached + 64 newly recorded).
        {
            let adapter = inner.paged_adapter.as_ref().expect("paged_adapter");
            assert_eq!(
                adapter.current_token_count() as usize,
                full_prompt.len(),
                "turn 2B cursor must reflect full prompt length (cached + suffix)"
            );
            assert_eq!(
                adapter.request_tokens(),
                full_prompt.as_slice(),
                "turn 2B request_tokens must byte-equal the full prompt"
            );
        }

        // Compare logits.
        let atol = 5e-3f32;
        let rtol = 5e-3f32;
        let mut max_abs_diff = 0.0f32;
        let mut max_rel_diff = 0.0f32;
        for (i, (a, b)) in single_vec.iter().zip(chunked_vec.iter()).enumerate() {
            assert!(
                b.is_finite(),
                "cached-prefix chunked logits[{i}] not finite: {b}"
            );
            let abs_diff = (a - b).abs();
            let rel_diff = abs_diff / (a.abs().max(b.abs()).max(1e-6));
            if abs_diff > max_abs_diff {
                max_abs_diff = abs_diff;
            }
            if rel_diff > max_rel_diff {
                max_rel_diff = rel_diff;
            }
            assert!(
                abs_diff <= atol || rel_diff <= rtol,
                "cached-prefix logits diverge at index {i}: single={a}, chunked={b}, \
                 abs_diff={abs_diff}, rel_diff={rel_diff} (max allowed: \
                 atol={atol} or rtol={rtol})"
            );
        }
        eprintln!(
            "cached-prefix chunked-prefill parity max_abs_diff={max_abs_diff}, \
             max_rel_diff={max_rel_diff} (atol={atol}, rtol={rtol}) over {} elements",
            single_vec.len()
        );

        // Cleanup.
        {
            let adapter = inner.paged_adapter.as_mut().unwrap();
            let _ = adapter.register_full_blocks_for_reuse(&[], 0);
            adapter.release_request().expect("release_request turn 2B");
        }
    }

    /// Env-var fallback test: callers that rely on the env-var
    /// default (chunk_size = 0) MUST still get the byte-equivalent
    /// single-shot path. We exercise this by calling the public
    /// `run_paged_prefill_chunk` (which reads the OnceLock-cached env)
    /// directly and verifying it produces the same `[vocab]` logits as
    /// `run_paged_prefill_chunk_with_size(..., 0)`. Keeps the public
    /// signature stable.
    #[test]
    fn test_run_paged_prefill_chunk_default_matches_single_shot() {
        // Block-paged needs the Metal backend; on a non-Metal build the
        // adapter is gated off (None) and there is nothing to exercise.
        if !crate::engine::persistence::compiled_forward_backend_available() {
            eprintln!("skipping (paged backend unavailable without Metal)");
            return;
        }
        // Skip when MLX_PAGED_PREFILL_CHUNK_SIZE is set in the env: the
        // OnceLock-cached `paged_prefill_chunk_size()` is process-global, and
        // a positive value smaller than the 8-token prompt would route the
        // public wrapper through the chunked branch instead of the
        // single-shot path this test is meant to exercise.
        if crate::array::memory::paged_prefill_chunk_size() != 0 {
            eprintln!(
                "skipping test_run_paged_prefill_chunk_default_matches_single_shot: \
                 MLX_PAGED_PREFILL_CHUNK_SIZE is set, default-path coverage is environment-dependent"
            );
            return;
        }
        let cfg = paged_tiny_config(Some(true));
        let mut inner = match super::Qwen3Inner::new(cfg.clone()) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!(
                        "skipping test_run_paged_prefill_chunk_default_matches_single_shot \
                         (no Metal): {msg}"
                    );
                    return;
                }
                panic!("unexpected Qwen3Inner::new failure: {msg}");
            }
        };
        cast_qwen3_inner_weights_bf16(&mut inner);

        // Short 8-token prompt: small enough that `chunk_size > 0` would
        // also take the single-shot fast path (`suffix_tokens.len() <=
        // chunk_size`). This means the "default chunk_size = 0" case and
        // the "chunk_size > suffix_len" case both produce numerically
        // identical outputs to single-shot — we exercise the first form
        // here. Bigger prompts are covered by the multi-chunk parity test
        // above.
        let prompt: Vec<u32> = vec![5, 11, 21, 33, 47, 60, 71, 83];
        let positions = MxArray::from_int32(&[0], &[1]).expect("positions");
        let num_layers = inner.layers.len();

        {
            let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
            adapter.reset_for_new_request(0).expect("reset");
            let prefix = adapter
                .find_cached_prefix(&prompt, &[], 0, false)
                .expect("find_cached_prefix");
            assert_eq!(prefix.cached_token_count, 0);
            adapter
                .allocate_suffix_blocks(prompt.len() as u32)
                .expect("allocate_suffix_blocks");
        }
        let logits_default = match inner.run_paged_prefill_chunk(&prompt, 0, num_layers, &positions)
        {
            Ok(l) => l,
            Err(e) => {
                let msg = e.reason.to_string();
                if msg.contains("Metal GPU not available") || msg.contains("No Metal device") {
                    eprintln!(
                        "skipping test_run_paged_prefill_chunk_default_matches_single_shot: {msg}"
                    );
                    return;
                }
                panic!("unexpected default prefill failure: {msg}");
            }
        };
        let default_vec = logits_to_f32_vec(&logits_default);
        assert_eq!(default_vec.len(), cfg.vocab_size as usize);
        for (i, v) in default_vec.iter().enumerate() {
            assert!(v.is_finite(), "default-path logits[{i}] not finite: {v}");
        }

        // Reset adapter, run the same prompt explicitly with chunk_size=0
        // (the single-shot path). The two should be byte-equal since the
        // public entry point is a thin wrapper around the _with_size helper
        // at chunk_size=0 (matches the env-default case when
        // MLX_PAGED_PREFILL_CHUNK_SIZE is unset).
        {
            let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
            adapter.release_request().expect("release_request");
            adapter.reset_for_new_request(0).expect("reset");
            let prefix = adapter
                .find_cached_prefix(&prompt, &[], 0, false)
                .expect("find_cached_prefix");
            assert_eq!(prefix.cached_token_count, 0);
            adapter
                .allocate_suffix_blocks(prompt.len() as u32)
                .expect("allocate_suffix_blocks");
        }
        let logits_explicit = inner
            .run_paged_prefill_chunk_with_size(&prompt, 0, num_layers, &positions, 0)
            .expect("explicit chunk_size=0 prefill");
        let explicit_vec = logits_to_f32_vec(&logits_explicit);

        // The default and explicit-zero paths run the SAME code path —
        // bytewise equality is what we expect. Use a vanishingly small
        // tolerance so the assertion still survives if the env var
        // happens to be set to 0 vs. unset (both collapse to chunk_size=0
        // anyway via parse_chunk_size).
        for (i, (a, b)) in default_vec.iter().zip(explicit_vec.iter()).enumerate() {
            let abs_diff = (a - b).abs();
            assert!(
                abs_diff <= 1e-6,
                "default path diverged from explicit chunk_size=0 at index {i}: \
                 default={a}, explicit={b}, abs_diff={abs_diff}"
            );
        }

        {
            let adapter = inner.paged_adapter.as_mut().unwrap();
            let _ = adapter.register_full_blocks_for_reuse(&[], 0);
            adapter.release_request().expect("release_request");
        }
    }
}
