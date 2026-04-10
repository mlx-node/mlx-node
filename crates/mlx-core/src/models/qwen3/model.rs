/**
 * Qwen3 Model - Core Model Implementation
 *
 * Contains the model structure, forward passes, and core model methods.
 */
use std::collections::HashMap;
use std::iter;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use tracing::{debug, info, warn};

use crate::array::{MxArray, heavy_cleanup, synchronize_and_clear_cache};
use crate::model_thread::{ModelThread, ResponseTx, send_and_await, send_and_block};
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::sampling::{
    SamplingConfig, apply_frequency_penalty, apply_presence_penalty, apply_repetition_penalty,
    check_repetition_cutoff, sample, sample_and_logprobs,
};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer, ToolDefinition};
use crate::tools;
use crate::training_model::ModelType;
use crate::transformer::{
    ContinuousBatchingScheduler, KVCache, PagedAttentionConfig, PagedKVCache, PendingRequest,
    SchedulerConfig, TransformerBlock,
};

use super::{BatchGenerationResult, GenerationConfig, GenerationResult, Qwen3Config};
use crate::models::qwen3_5::model::{ChatConfig, ChatResult};

/// Paged attention memory statistics (NAPI-compatible)
#[napi(object)]
#[derive(Debug, Clone)]
pub struct PagedCacheStats {
    /// Total number of blocks in the pool
    pub total_blocks: u32,
    /// Number of free blocks
    pub free_blocks: u32,
    /// Number of allocated blocks
    pub allocated_blocks: u32,
    /// Total memory in MB
    pub total_memory_mb: u32,
    /// Used memory in MB
    pub used_memory_mb: u32,
    /// Utilization percentage
    pub utilization_percent: f64,
}

/// Scheduler statistics (NAPI-compatible)
#[napi(object)]
#[derive(Debug, Clone)]
pub struct SchedulerStatsNapi {
    /// Number of requests waiting to be scheduled
    pub num_waiting: u32,
    /// Number of sequences currently running
    pub num_running: u32,
    /// Number of completed sequences
    pub num_completed: u32,
    /// Number of sequences in prefill phase
    pub num_prefill: u32,
    /// Number of sequences in decode phase
    pub num_decode: u32,
    /// Total tokens across all running sequences
    pub total_running_tokens: u32,
}

/// Output from a single token generation step in paged attention
#[napi(object)]
#[derive(Debug, Clone)]
pub struct PagedTokenOutput {
    /// Sequence ID in the scheduler
    pub seq_id: u32,
    /// Request ID for this sequence
    pub request_id: String,
    /// Generated token ID
    pub token: u32,
    /// Log probability of the token (f64 for NAPI compatibility)
    pub logprob: f64,
    /// Whether this sequence has finished
    pub is_finished: bool,
}

/// Result of a paged generation step
#[napi(object)]
#[derive(Debug, Clone)]
pub struct PagedGenerationStep {
    /// Token outputs for each sequence in the batch
    pub outputs: Vec<PagedTokenOutput>,
    /// Number of sequences that were in prefill phase
    pub num_prefill: u32,
    /// Number of sequences that were in decode phase
    pub num_decode: u32,
}

/// A completed sequence from paged generation
#[napi(object)]
#[derive(Debug, Clone)]
pub struct PagedCompletedSequence {
    /// Original request ID
    pub request_id: String,
    /// All generated tokens (excluding prompt)
    pub tokens: Vec<u32>,
    /// Reason for completion ("stop", "length", "repetition", "tool_calls")
    pub finish_reason: String,
}

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
    pub(crate) paged_cache: Option<PagedKVCache>,
    pub(crate) scheduler: Option<ContinuousBatchingScheduler>,
    pub(crate) cached_kv_keys: Vec<Option<MxArray>>,
    pub(crate) cached_kv_values: Vec<Option<MxArray>>,
    pub(crate) cached_cache_idx: i32,
    pub(crate) cached_token_history: Vec<u32>,
    /// Training state owned by the model thread.
    /// Created when `InitTraining` command is received, destroyed when training ends.
    pub(crate) training_state: Option<crate::training_state::ModelThreadTrainingState>,
}

/// Commands dispatched from NAPI methods to the dedicated model thread.
pub(crate) enum Qwen3Cmd {
    Chat {
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        reply: ResponseTx<ChatResult>,
    },
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
    Decode {
        token_ids: Vec<u32>,
        skip_special_tokens: bool,
        reply: ResponseTx<String>,
    },
    InitKvCaches {
        reply: ResponseTx<()>,
    },
    ResetKvCaches {
        reply: ResponseTx<()>,
    },
    ResetCache {
        reply: ResponseTx<()>,
    },
    ForwardPaged {
        input_ids: MxArray,
        slot_mapping: MxArray,
        seq_ids: Vec<u32>,
        positions: MxArray,
        reply: ResponseTx<MxArray>,
    },
    PrefillPaged {
        prompt_tokens: Vec<u32>,
        seq_id: u32,
        reply: ResponseTx<MxArray>,
    },
    AddPagedRequest {
        request_id: String,
        prompt_tokens: Vec<u32>,
        max_new_tokens: u32,
        priority: Option<i32>,
        reply: ResponseTx<u32>,
    },
    StepPagedGeneration {
        config: Option<GenerationConfig>,
        reply: ResponseTx<Option<PagedGenerationStep>>,
    },
    GetCompletedSequences {
        reply: ResponseTx<Vec<PagedCompletedSequence>>,
    },
    HasPagedWork {
        reply: ResponseTx<bool>,
    },
    PagedCacheStats {
        reply: ResponseTx<Option<PagedCacheStats>>,
    },
    SchedulerStats {
        reply: ResponseTx<Option<SchedulerStatsNapi>>,
    },
    // --- Training commands ---
    InitTraining {
        config: Box<crate::grpo::engine::GRPOEngineConfig>,
        model_type: crate::training_model::ModelType,
        reply: ResponseTx<()>,
    },
    GenerateForTraining {
        prompts: Vec<Vec<crate::tokenizer::ChatMessage>>,
        group_size: usize,
        gen_config: super::GenerationConfig,
        enable_thinking: Option<bool>,
        tools: Option<Vec<crate::tokenizer::ToolDefinition>>,
        reply: ResponseTx<crate::training_model::GenerationPlainData>,
    },
    TrainStepGRPO {
        rewards: Vec<f64>,
        group_size: i32,
        loss_config: crate::grpo::loss::GRPOLossConfig,
        valid_indices: Option<Vec<usize>>,
        reply: ResponseTx<crate::training_model::TrainStepPlainMetrics>,
    },
    /// Bump the training step counter without applying gradients
    /// (used by engine skip paths that abort before training).
    /// Also clears cached generation MxArrays.
    /// Returns the new step.
    BumpSkippedStep {
        reply: ResponseTx<i64>,
    },
    /// Restore the training step counter (for resume from checkpoint).
    /// Does not touch optimizer state — that's loaded via LoadOptimizerState.
    SetTrainingStep {
        step: i64,
        reply: ResponseTx<()>,
    },
    /// Drop the training state on the model thread.
    /// After this, InitTraining can be called again. No-op if no training state.
    ResetTraining {
        reply: ResponseTx<()>,
    },
    TrainStepSFT {
        input_ids: Vec<i32>,
        input_shape: Vec<i64>,
        labels: Vec<i32>,
        labels_shape: Vec<i64>,
        config: crate::sft::engine::SftEngineConfig,
        reply: ResponseTx<crate::training_model::TrainStepPlainMetrics>,
    },
    SaveOptimizerState {
        path: String,
        reply: ResponseTx<()>,
    },
    LoadOptimizerState {
        path: String,
        reply: ResponseTx<()>,
    },
    SaveModel {
        save_path: String,
        reply: ResponseTx<()>,
    },
}

/// Command handler for the dedicated model thread.
pub(crate) fn handle_qwen3_cmd(inner: &mut Qwen3Inner, cmd: Qwen3Cmd) {
    match cmd {
        Qwen3Cmd::Chat {
            messages,
            config,
            reply,
        } => {
            let _ = reply.send(inner.chat_sync(messages, config));
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
        Qwen3Cmd::Decode {
            token_ids,
            skip_special_tokens,
            reply,
        } => {
            let _ = reply.send(inner.decode_sync(&token_ids, skip_special_tokens));
        }
        Qwen3Cmd::InitKvCaches { reply } => {
            let _ = reply.send(inner.init_kv_caches_sync());
        }
        Qwen3Cmd::ResetKvCaches { reply } => {
            let _ = reply.send(inner.reset_kv_caches_sync());
        }
        Qwen3Cmd::ResetCache { reply } => {
            let _ = reply.send(inner.reset_cache_sync());
        }
        Qwen3Cmd::ForwardPaged {
            input_ids,
            slot_mapping,
            seq_ids,
            positions,
            reply,
        } => {
            let _ = reply.send(inner.forward_paged_sync(
                &input_ids,
                &slot_mapping,
                seq_ids,
                &positions,
            ));
        }
        Qwen3Cmd::PrefillPaged {
            prompt_tokens,
            seq_id,
            reply,
        } => {
            let _ = reply.send(inner.prefill_paged_sync(prompt_tokens, seq_id));
        }
        Qwen3Cmd::AddPagedRequest {
            request_id,
            prompt_tokens,
            max_new_tokens,
            priority,
            reply,
        } => {
            let _ = reply.send(inner.add_paged_request_sync(
                request_id,
                prompt_tokens,
                max_new_tokens,
                priority,
            ));
        }
        Qwen3Cmd::StepPagedGeneration { config, reply } => {
            let _ = reply.send(inner.step_paged_generation_sync(config));
        }
        Qwen3Cmd::GetCompletedSequences { reply } => {
            let _ = reply.send(inner.get_completed_sequences_sync());
        }
        Qwen3Cmd::HasPagedWork { reply } => {
            let _ = reply.send(inner.has_paged_work_sync());
        }
        Qwen3Cmd::PagedCacheStats { reply } => {
            let _ = reply.send(Ok(inner.paged_cache_stats_sync()));
        }
        Qwen3Cmd::SchedulerStats { reply } => {
            let _ = reply.send(Ok(inner.scheduler_stats_sync()));
        }
        // --- Training commands ---
        Qwen3Cmd::InitTraining {
            config,
            model_type,
            reply,
        } => {
            let _ = reply.send(inner.init_training_sync(*config, model_type));
        }
        Qwen3Cmd::GenerateForTraining {
            prompts,
            group_size,
            gen_config,
            enable_thinking,
            tools,
            reply,
        } => {
            let _ = reply.send(inner.generate_for_training_thread_sync(
                prompts,
                group_size,
                gen_config,
                enable_thinking,
                tools,
            ));
        }
        Qwen3Cmd::TrainStepGRPO {
            rewards,
            group_size,
            loss_config,
            valid_indices,
            reply,
        } => {
            let _ = reply.send(inner.train_step_grpo_sync(
                rewards,
                group_size,
                loss_config,
                valid_indices,
            ));
        }
        Qwen3Cmd::BumpSkippedStep { reply } => {
            let result = if let Some(ref mut ts) = inner.training_state {
                ts.clear_generation_cache();
                ts.step += 1;
                Ok(ts.step)
            } else {
                Err(napi::Error::from_reason(
                    "Training state not initialized. Call InitTraining first.",
                ))
            };
            let _ = reply.send(result);
        }
        Qwen3Cmd::SetTrainingStep { step, reply } => {
            let result = if let Some(ref mut ts) = inner.training_state {
                ts.step = step;
                Ok(())
            } else {
                Err(napi::Error::from_reason(
                    "Training state not initialized. Call InitTraining first.",
                ))
            };
            let _ = reply.send(result);
        }
        Qwen3Cmd::ResetTraining { reply } => {
            inner.training_state = None;
            let _ = reply.send(Ok(()));
        }
        Qwen3Cmd::TrainStepSFT {
            input_ids,
            input_shape,
            labels,
            labels_shape,
            config,
            reply,
        } => {
            let _ = reply.send(inner.train_step_sft_sync(
                input_ids,
                input_shape,
                labels,
                labels_shape,
                config,
            ));
        }
        Qwen3Cmd::SaveOptimizerState { path, reply } => {
            let _ = reply.send(inner.save_optimizer_state_sync(path));
        }
        Qwen3Cmd::LoadOptimizerState { path, reply } => {
            let _ = reply.send(inner.load_optimizer_state_sync(path));
        }
        Qwen3Cmd::SaveModel { save_path, reply } => {
            let _ = reply.send(inner.save_model_sync(&save_path));
        }
    }
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

        // Initialize paged attention if enabled
        let (paged_cache, scheduler) = if config.use_paged_attention.unwrap_or(false) {
            let paged_config = PagedAttentionConfig {
                block_size: config.paged_block_size.unwrap_or(16),
                gpu_memory_mb: config.paged_cache_memory_mb.unwrap_or(2048),
                head_size: config.head_dim as u32,
                num_kv_heads: config.num_kv_heads as u32,
                num_layers: config.num_layers as u32,
                use_fp8_cache: config.use_fp8_cache,
                max_seq_len: Some(config.max_position_embeddings as u32),
                max_batch_size: Some(32),
            };

            let mut cache = PagedKVCache::new(paged_config.clone()).map_err(|e| {
                napi::Error::from_reason(format!("Failed to create PagedKVCache: {}", e))
            })?;

            #[cfg(target_os = "macos")]
            cache.initialize().map_err(|e| {
                napi::Error::from_reason(format!(
                    "Failed to initialize PagedKVCache GPU buffers: {}",
                    e
                ))
            })?;

            let scheduler_config = SchedulerConfig {
                max_batch_size: 32,
                max_tokens_per_step: Some(4096),
                max_prefill_per_step: Some(1),
                prioritize_decode: Some(true),
                eos_token_id: Some(config.eos_token_id as u32),
            };
            let sched =
                ContinuousBatchingScheduler::new(paged_config.block_size, Some(scheduler_config));

            info!(
                "Paged attention enabled with {}MB cache, block_size={}, fp8={}",
                paged_config.gpu_memory_mb,
                paged_config.block_size,
                paged_config.use_fp8()
            );

            (Some(cache), Some(sched))
        } else {
            (None, None)
        };

        Ok(Self {
            config,
            embedding,
            layers,
            final_norm,
            lm_head,
            kv_caches: None,
            tokenizer: None,
            paged_cache,
            scheduler,
            cached_kv_keys: Vec::new(),
            cached_kv_values: Vec::new(),
            cached_cache_idx: 0,
            cached_token_history: Vec::new(),
            training_state: None,
        })
    }

    pub(crate) fn set_tokenizer(&mut self, tokenizer: Arc<Qwen3Tokenizer>) {
        self.tokenizer = Some(tokenizer);
    }

    fn init_kv_caches_sync(&mut self) -> Result<()> {
        let num_layers = self.layers.len();
        let caches: Vec<KVCache> = (0..num_layers).map(|_| KVCache::new()).collect();
        self.kv_caches = Some(caches);
        Ok(())
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
        Ok(())
    }

    fn reset_cache_sync(&mut self) -> Result<()> {
        self.cached_kv_keys.clear();
        self.cached_kv_values.clear();
        self.cached_cache_idx = 0;
        self.cached_token_history.clear();
        Ok(())
    }

    fn paged_cache_stats_sync(&self) -> Option<PagedCacheStats> {
        let cache = self.paged_cache.as_ref()?;
        let stats = cache.get_memory_stats();
        Some(PagedCacheStats {
            total_blocks: stats.total_blocks,
            free_blocks: stats.free_blocks,
            allocated_blocks: stats.allocated_blocks,
            total_memory_mb: stats.total_memory_mb,
            used_memory_mb: stats.used_memory_mb,
            utilization_percent: stats.utilization_percent,
        })
    }

    fn scheduler_stats_sync(&self) -> Option<SchedulerStatsNapi> {
        let sched = self.scheduler.as_ref()?;
        let stats = sched.get_stats();
        Some(SchedulerStatsNapi {
            num_waiting: stats.num_waiting,
            num_running: stats.num_running,
            num_completed: stats.num_completed,
            num_prefill: stats.num_prefill,
            num_decode: stats.num_decode,
            total_running_tokens: stats.total_running_tokens,
        })
    }

    fn has_paged_work_sync(&self) -> Result<bool> {
        let sched = self
            .scheduler
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("Paged attention not enabled"))?;
        Ok(!sched.is_empty())
    }

    fn get_completed_sequences_sync(&mut self) -> Result<Vec<PagedCompletedSequence>> {
        let sched = self
            .scheduler
            .as_mut()
            .ok_or_else(|| napi::Error::from_reason("Paged attention not enabled"))?;
        let completed = sched.get_completed();
        Ok(completed
            .into_iter()
            .map(|c| PagedCompletedSequence {
                request_id: c.request_id,
                tokens: c.generated_tokens,
                finish_reason: c.finish_reason,
            })
            .collect())
    }

    fn add_paged_request_sync(
        &mut self,
        request_id: String,
        prompt_tokens: Vec<u32>,
        max_new_tokens: u32,
        priority: Option<i32>,
    ) -> Result<u32> {
        if max_new_tokens == 0 {
            return Err(napi::Error::from_reason(
                "max_new_tokens must be > 0 for paged generation",
            ));
        }
        if prompt_tokens.is_empty() {
            return Err(napi::Error::from_reason(
                "prompt_tokens must not be empty for paged generation",
            ));
        }
        let sched = self
            .scheduler
            .as_mut()
            .ok_or_else(|| napi::Error::from_reason("Paged attention not enabled"))?;
        let request = PendingRequest {
            request_id,
            prompt_tokens,
            max_new_tokens,
            priority,
        };
        sched.add_request(request);
        Ok(sched.num_waiting())
    }

    fn forward_paged_sync(
        &self,
        input_ids: &MxArray,
        slot_mapping: &MxArray,
        seq_ids: Vec<u32>,
        positions: &MxArray,
    ) -> Result<MxArray> {
        let paged_cache = self.paged_cache.as_ref().ok_or_else(|| {
            napi::Error::from_reason(
                "Paged attention not enabled. Set use_paged_attention: true in config.",
            )
        })?;
        Self::forward_paged_with_cache(
            &self.config,
            &self.embedding,
            &self.layers,
            &self.final_norm,
            &self.lm_head,
            input_ids,
            slot_mapping,
            seq_ids,
            positions,
            paged_cache,
        )
    }

    /// Forward pass with paged KV cache.
    /// Takes individual field references to avoid borrow conflicts when scheduler/paged_cache
    /// are already mutably borrowed in step_paged_generation_sync.
    fn forward_paged_with_cache(
        config: &Qwen3Config,
        embedding: &Embedding,
        layers: &[TransformerBlock],
        final_norm: &RMSNorm,
        lm_head: &Linear,
        input_ids: &MxArray,
        slot_mapping: &MxArray,
        seq_ids: Vec<u32>,
        positions: &MxArray,
        paged_cache: &mlx_paged_attn::PagedKVCache,
    ) -> Result<MxArray> {
        let mut hidden_states = embedding.forward(input_ids)?;
        let num_seqs = hidden_states.shape_at(0)?;
        let seq_len = hidden_states.shape_at(1)?;
        let num_query_heads = config.num_heads as u32;

        for (layer_idx, layer) in layers.iter().enumerate() {
            hidden_states = layer.forward_paged_metal(
                &hidden_states,
                paged_cache,
                layer_idx as u32,
                slot_mapping,
                &seq_ids,
                num_query_heads,
                positions,
                num_seqs,
                seq_len,
            )?;
        }

        hidden_states = final_norm.forward(&hidden_states)?;

        let logits = if config.tie_word_embeddings {
            let embedding_weight = embedding.get_weight();
            hidden_states.matmul(&embedding_weight.transpose(Some(&[1, 0]))?)?
        } else {
            lm_head.forward(&hidden_states)?
        };

        Ok(logits)
    }

    fn prefill_paged_sync(&mut self, prompt_tokens: Vec<u32>, seq_id: u32) -> Result<MxArray> {
        let prompt_len = prompt_tokens.len();
        if prompt_len == 0 {
            return Err(napi::Error::from_reason("Empty prompt"));
        }

        let input_ids = MxArray::from_uint32(&prompt_tokens, &[1, prompt_len as i64])?;
        let mut hidden_states = self.embedding.forward(&input_ids)?;

        let paged_cache = self
            .paged_cache
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("Paged attention not enabled"))?;

        let slot_mapping = paged_cache
            .get_slot_mapping(seq_id, 0, prompt_len as u32)
            .map_err(napi::Error::from_reason)?;
        let slot_mapping_arr = MxArray::from_int64(&slot_mapping, &[slot_mapping.len() as i64])?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let (output, keys, values) = layer.forward_for_prefill(&hidden_states)?;

            #[cfg(target_os = "macos")]
            unsafe {
                paged_cache
                    .update(
                        layer_idx as u32,
                        keys.handle.0,
                        values.handle.0,
                        slot_mapping_arr.handle.0,
                    )
                    .map_err(napi::Error::from_reason)?;
            }

            #[cfg(not(target_os = "macos"))]
            {
                return Err(napi::Error::from_reason(
                    "Paged attention Metal kernels are only available on macOS",
                ));
            }

            hidden_states = output;
        }

        hidden_states = self.final_norm.forward(&hidden_states)?;

        let logits = if self.config.tie_word_embeddings {
            let embedding_weight = self.embedding.get_weight();
            hidden_states.matmul(&embedding_weight.transpose(Some(&[1, 0]))?)?
        } else {
            self.lm_head.forward(&hidden_states)?
        };

        let vocab_size = logits.shape_at(2)?;
        let last_logits = logits.slice(
            &[0, prompt_len as i64 - 1, 0],
            &[1, prompt_len as i64, vocab_size],
        )?;
        let last_logits = last_logits.reshape(&[1, vocab_size])?;

        Ok(last_logits)
    }

    /// Synchronous step_paged_generation (runs on model thread).
    /// This is a direct port of the old NAPI method but using direct field access.
    fn step_paged_generation_sync(
        &mut self,
        config: Option<GenerationConfig>,
    ) -> Result<Option<PagedGenerationStep>> {
        let sched = self
            .scheduler
            .as_mut()
            .ok_or_else(|| napi::Error::from_reason("Paged attention not enabled"))?;
        let paged_cache = self
            .paged_cache
            .as_mut()
            .ok_or_else(|| napi::Error::from_reason("Paged attention not enabled"))?;

        let config = config.unwrap_or_default();

        let batch = match sched.schedule_step(paged_cache) {
            Some(b) => b,
            None => return Ok(None),
        };

        if batch.seq_ids.is_empty() {
            return Ok(None);
        }

        let sampling_config = SamplingConfig {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            min_p: config.min_p,
        };
        let eos_token_id = config.eos_token_id.unwrap_or(self.config.eos_token_id);

        let repetition_penalty = config.repetition_penalty.unwrap_or(1.0);
        let repetition_context_size = config.repetition_context_size.unwrap_or(256);
        let presence_penalty = config.presence_penalty.unwrap_or(0.0);
        let presence_context_size = config.presence_context_size.unwrap_or(20);
        let frequency_penalty = config.frequency_penalty.unwrap_or(0.0);
        let frequency_context_size = config.frequency_context_size.unwrap_or(20);
        let max_consecutive_tokens = config.max_consecutive_tokens.unwrap_or(16);
        let max_ngram_repeats = config.max_ngram_repeats.unwrap_or(3);
        let ngram_size = config.ngram_size.unwrap_or(64);

        let mut finish_reason_overrides: HashMap<u32, &'static str> = HashMap::new();

        let mut prefill_indices: Vec<usize> = Vec::new();
        let mut decode_indices: Vec<usize> = Vec::new();

        for (i, &is_prefill) in batch.is_prefill.iter().enumerate() {
            if is_prefill {
                prefill_indices.push(i);
            } else {
                decode_indices.push(i);
            }
        }

        let mut outputs = Vec::with_capacity(batch.seq_ids.len());

        // PREFILL PATH
        for &idx in &prefill_indices {
            let seq_id = batch.seq_ids[idx];
            let request_id = &batch.request_ids[idx];
            let prompt_tokens = &batch.input_tokens[idx];
            let prompt_len = prompt_tokens.len();

            if prompt_len == 0 {
                return Err(napi::Error::from_reason(format!(
                    "Empty prompt for sequence {}",
                    seq_id
                )));
            }

            let input_ids = MxArray::from_uint32(prompt_tokens, &[1, prompt_len as i64])?;
            let mut hidden_states = self.embedding.forward(&input_ids)?;

            let slot_mapping = paged_cache
                .get_slot_mapping(seq_id, 0, prompt_len as u32)
                .map_err(napi::Error::from_reason)?;
            let slot_mapping_arr =
                MxArray::from_int64(&slot_mapping, &[slot_mapping.len() as i64])?;

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let (output, keys, values) = layer.forward_for_prefill(&hidden_states)?;

                #[cfg(target_os = "macos")]
                unsafe {
                    paged_cache
                        .update(
                            layer_idx as u32,
                            keys.handle.0,
                            values.handle.0,
                            slot_mapping_arr.handle.0,
                        )
                        .map_err(napi::Error::from_reason)?;
                }

                #[cfg(not(target_os = "macos"))]
                {
                    return Err(napi::Error::from_reason(
                        "Paged attention Metal kernels are only available on macOS",
                    ));
                }

                hidden_states = output;
            }

            hidden_states = self.final_norm.forward(&hidden_states)?;

            let logits = if self.config.tie_word_embeddings {
                let embedding_weight = self.embedding.get_weight();
                hidden_states.matmul(&embedding_weight.transpose(Some(&[1, 0]))?)?
            } else {
                self.lm_head.forward(&hidden_states)?
            };

            let vocab_size = logits.shape_at(2)?;
            logits.eval();
            let logits_data = logits.to_float32()?;
            let start = (prompt_len - 1) * vocab_size as usize;
            let end = start + vocab_size as usize;

            if end > logits_data.len() {
                return Err(napi::Error::from_reason(format!(
                    "Logits buffer size mismatch in prefill: expected {} elements (end), got {}",
                    end,
                    logits_data.len()
                )));
            }

            let logit_slice = &logits_data[start..end];
            let mut logit_arr = MxArray::from_float32(logit_slice, &[1, 1, vocab_size])?;

            if let Some((prompt, generated)) = sched.get_penalty_context(seq_id) {
                let max_ctx = repetition_context_size
                    .max(presence_context_size)
                    .max(frequency_context_size) as usize;
                let total = prompt.len() + generated.len();
                let skip = total.saturating_sub(max_ctx);
                let prompt_skip = skip.min(prompt.len());
                let ctx: Vec<u32> = prompt[prompt_skip..]
                    .iter()
                    .chain(generated.iter())
                    .copied()
                    .collect();
                if !ctx.is_empty() {
                    if repetition_penalty != 1.0 {
                        logit_arr = apply_repetition_penalty(
                            &logit_arr,
                            &ctx,
                            repetition_penalty,
                            Some(repetition_context_size),
                        )?;
                    }
                    if presence_penalty != 0.0 {
                        logit_arr = apply_presence_penalty(
                            &logit_arr,
                            &ctx,
                            presence_penalty,
                            Some(presence_context_size),
                        )?;
                    }
                    if frequency_penalty != 0.0 {
                        logit_arr = apply_frequency_penalty(
                            &logit_arr,
                            &ctx,
                            frequency_penalty,
                            Some(frequency_context_size),
                        )?;
                    }
                }
            }

            let (next_token_arr, logprobs_arr) =
                sample_and_logprobs(&logit_arr, Some(sampling_config))?;

            next_token_arr.eval();
            logprobs_arr.eval();
            let next_token = next_token_arr.item_at_int32(0)? as u32;
            let logprob = logprobs_arr.item_at_float32(next_token as usize)? as f64;
            let is_eos = next_token == eos_token_id as u32;
            let mut finish_reason_override: Option<&'static str> = None;

            if !is_eos && let Some(gen_tokens) = sched.get_generated_tokens(seq_id) {
                let mut history = gen_tokens.to_vec();
                history.push(next_token);
                if let Some(reason) = check_repetition_cutoff(
                    &history,
                    max_consecutive_tokens,
                    max_ngram_repeats,
                    ngram_size,
                ) {
                    finish_reason_override = Some(reason);
                }
            }

            if finish_reason_override.is_none() && !is_eos && sched.would_hit_length_limit(seq_id) {
                finish_reason_override = Some("length");
            }

            let is_finished = is_eos || finish_reason_override.is_some();
            if let Some(reason) = finish_reason_override {
                finish_reason_overrides.insert(seq_id, reason);
            }

            outputs.push(PagedTokenOutput {
                seq_id,
                request_id: request_id.clone(),
                token: next_token,
                logprob,
                is_finished,
            });
        }

        // DECODE PATH
        if !decode_indices.is_empty() {
            let num_decode_seqs = decode_indices.len();
            let decode_seq_ids: Vec<u32> =
                decode_indices.iter().map(|&i| batch.seq_ids[i]).collect();
            let decode_input_tokens: Vec<u32> = decode_indices
                .iter()
                .map(|&i| batch.input_tokens[i].first().copied().unwrap_or(0))
                .collect();
            let decode_context_lens: Vec<u32> = decode_indices
                .iter()
                .map(|&i| batch.context_lens[i])
                .collect();

            for (i, &ctx_len) in decode_context_lens.iter().enumerate() {
                if ctx_len == 0 {
                    return Err(napi::Error::from_reason(format!(
                        "Decode sequence {} (seq_id={}) has context_len=0. Prefill must complete before decode.",
                        i, decode_seq_ids[i]
                    )));
                }
            }

            let input_ids =
                MxArray::from_uint32(&decode_input_tokens, &[num_decode_seqs as i64, 1])?;
            let positions_vec: Vec<i32> = decode_context_lens
                .iter()
                .map(|&ctx| if ctx > 0 { ctx as i32 - 1 } else { 0 })
                .collect();
            let positions_arr = MxArray::from_int32(&positions_vec, &[num_decode_seqs as i64])?;

            let input_lens: Vec<u32> = vec![1; num_decode_seqs];
            let is_prefill_flags: Vec<bool> = vec![false; num_decode_seqs];
            let slot_mapping = paged_cache
                .get_slot_mapping_batch(
                    &decode_seq_ids,
                    &decode_context_lens,
                    &is_prefill_flags,
                    &input_lens,
                )
                .map_err(napi::Error::from_reason)?;
            let slot_mapping_arr =
                MxArray::from_int64(&slot_mapping, &[slot_mapping.len() as i64])?;

            let logits = Self::forward_paged_with_cache(
                &self.config,
                &self.embedding,
                &self.layers,
                &self.final_norm,
                &self.lm_head,
                &input_ids,
                &slot_mapping_arr,
                decode_seq_ids.clone(),
                &positions_arr,
                paged_cache,
            )?;

            let vocab_size = logits.shape_at(2)?;
            logits.eval();
            let logits_data = logits.to_float32()?;

            for (i, &idx) in decode_indices.iter().enumerate() {
                let seq_id = batch.seq_ids[idx];
                let request_id = &batch.request_ids[idx];

                let start = i * vocab_size as usize;
                let end = start + vocab_size as usize;

                if end > logits_data.len() {
                    return Err(napi::Error::from_reason(format!(
                        "Logits buffer size mismatch in decode: expected {} elements (end), got {}",
                        end,
                        logits_data.len()
                    )));
                }

                let logit_slice = &logits_data[start..end];
                let mut logit_arr = MxArray::from_float32(logit_slice, &[1, 1, vocab_size])?;

                if let Some((prompt, generated)) = sched.get_penalty_context(seq_id) {
                    let max_ctx = repetition_context_size
                        .max(presence_context_size)
                        .max(frequency_context_size) as usize;
                    let total = prompt.len() + generated.len();
                    let skip = total.saturating_sub(max_ctx);
                    let prompt_skip = skip.min(prompt.len());
                    let gen_skip = skip.saturating_sub(prompt.len());
                    let ctx: Vec<u32> = prompt[prompt_skip..]
                        .iter()
                        .chain(generated[gen_skip..].iter())
                        .copied()
                        .collect();
                    if !ctx.is_empty() {
                        if repetition_penalty != 1.0 {
                            logit_arr = apply_repetition_penalty(
                                &logit_arr,
                                &ctx,
                                repetition_penalty,
                                Some(repetition_context_size),
                            )?;
                        }
                        if presence_penalty != 0.0 {
                            logit_arr = apply_presence_penalty(
                                &logit_arr,
                                &ctx,
                                presence_penalty,
                                Some(presence_context_size),
                            )?;
                        }
                        if frequency_penalty != 0.0 {
                            logit_arr = apply_frequency_penalty(
                                &logit_arr,
                                &ctx,
                                frequency_penalty,
                                Some(frequency_context_size),
                            )?;
                        }
                    }
                }

                let (next_token_arr, logprobs_arr) =
                    sample_and_logprobs(&logit_arr, Some(sampling_config))?;

                next_token_arr.eval();
                logprobs_arr.eval();
                let next_token = next_token_arr.item_at_int32(0)? as u32;
                let logprob = logprobs_arr.item_at_float32(next_token as usize)? as f64;
                let is_eos = next_token == eos_token_id as u32;
                let mut finish_reason_override: Option<&'static str> = None;

                if !is_eos && let Some(gen_tokens) = sched.get_generated_tokens(seq_id) {
                    let mut history = gen_tokens.to_vec();
                    history.push(next_token);
                    if let Some(reason) = check_repetition_cutoff(
                        &history,
                        max_consecutive_tokens,
                        max_ngram_repeats,
                        ngram_size,
                    ) {
                        finish_reason_override = Some(reason);
                    }
                }

                if finish_reason_override.is_none()
                    && !is_eos
                    && sched.would_hit_length_limit(seq_id)
                {
                    finish_reason_override = Some("length");
                }

                let is_finished = is_eos || finish_reason_override.is_some();
                if let Some(reason) = finish_reason_override {
                    finish_reason_overrides.insert(seq_id, reason);
                }

                outputs.push(PagedTokenOutput {
                    seq_id,
                    request_id: request_id.clone(),
                    token: next_token,
                    logprob,
                    is_finished,
                });
            }
        }

        let token_outputs: Vec<_> = outputs
            .iter()
            .map(|o| {
                let override_reason = finish_reason_overrides.get(&o.seq_id).copied();
                crate::transformer::TokenOutput {
                    seq_id: o.seq_id,
                    token: o.token,
                    is_eos: o.is_finished && override_reason.is_none(),
                    finish_reason_override: override_reason,
                }
            })
            .collect();
        sched
            .process_outputs(token_outputs, paged_cache)
            .map_err(napi::Error::from_reason)?;

        Ok(Some(PagedGenerationStep {
            outputs,
            num_prefill: batch.num_prefill,
            num_decode: batch.num_decode,
        }))
    }

    fn decode_sync(&self, token_ids: &[u32], skip_special: bool) -> Result<String> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            napi::Error::from_reason("Tokenizer not available. Model must be loaded via load().")
        })?;
        tokenizer.decode_sync(token_ids, skip_special)
    }

    /// Chat synchronous (runs on model thread).
    /// Wraps the existing chat logic using direct field access.
    fn chat_sync(&mut self, messages: Vec<ChatMessage>, config: ChatConfig) -> Result<ChatResult> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("Tokenizer not available."))?
            .clone();

        let tools = config.tools.clone();
        let enable_thinking = crate::models::qwen3_5::chat_common::resolve_enable_thinking(&config);
        let report_perf = config.report_performance.unwrap_or(false);
        let reuse_cache = config.reuse_cache.unwrap_or(true);

        let gen_config = GenerationConfig {
            max_new_tokens: config.max_new_tokens.or(Some(2048)),
            temperature: config.temperature.or(Some(0.7)),
            top_k: config.top_k,
            top_p: config.top_p.or(Some(0.9)),
            min_p: config.min_p,
            repetition_penalty: config.repetition_penalty,
            repetition_context_size: config.repetition_context_size,
            presence_penalty: config.presence_penalty,
            presence_context_size: config.presence_context_size,
            frequency_penalty: config.frequency_penalty,
            frequency_context_size: config.frequency_context_size,
            max_consecutive_tokens: config.max_consecutive_tokens,
            max_ngram_repeats: config.max_ngram_repeats,
            ngram_size: config.ngram_size,
            eos_token_id: None,
            return_logprobs: None,
            prefill_step_size: None,
            kv_cache_bits: None,
            kv_cache_group_size: None,
            num_draft_tokens: None,
            report_performance: config.report_performance,
        };

        let gen_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };

        let token_ids_vec = tokenizer.apply_chat_template_sync(
            &messages,
            Some(true),
            tools.as_deref(),
            enable_thinking,
        )?;

        // === Cache reuse: prefix verification ===
        let (initial_kv_keys, initial_kv_values, initial_cache_idx, prefill_input_ids) =
            if reuse_cache {
                let cached = &self.cached_token_history;
                let clen = cached.len();
                let plen = if !cached.is_empty()
                    && token_ids_vec.len() >= clen
                    && token_ids_vec[..clen] == cached[..]
                {
                    clen
                } else {
                    0
                };

                if plen > 0 {
                    let keys = self.cached_kv_keys.clone();
                    let vals = self.cached_kv_values.clone();
                    let idx = self.cached_cache_idx;
                    let delta_tokens = &token_ids_vec[plen..];
                    let delta_ids = if delta_tokens.is_empty() {
                        None
                    } else {
                        Some(MxArray::from_uint32(
                            delta_tokens,
                            &[1, delta_tokens.len() as i64],
                        )?)
                    };
                    info!(
                        "Cache hit: prefix_len={}, delta_tokens={}, cache_idx={}",
                        plen,
                        delta_tokens.len(),
                        idx
                    );
                    (Some(keys), Some(vals), idx, delta_ids)
                } else {
                    if clen > 0 {
                        info!(
                            "Cache miss: cached {} tokens, new {} tokens — full prefill",
                            clen,
                            token_ids_vec.len()
                        );
                    }
                    let input_ids =
                        MxArray::from_uint32(&token_ids_vec, &[1, token_ids_vec.len() as i64])?;
                    (None, None, 0, Some(input_ids))
                }
            } else {
                let input_ids =
                    MxArray::from_uint32(&token_ids_vec, &[1, token_ids_vec.len() as i64])?;
                (None, None, 0, Some(input_ids))
            };

        let actual_prefill_count = match &prefill_input_ids {
            Some(ids) => ids.shape_at(1).unwrap_or(token_ids_vec.len() as i64) as f64,
            None => 1.0,
        };
        let prompt_token_count = actual_prefill_count;

        let embedding_weight = self.embedding.get_weight();
        let layers = &self.layers;
        let final_norm = &self.final_norm;
        let lm_head = &self.lm_head;
        let model_config = &self.config;

        let max_new_tokens = gen_config.max_new_tokens.unwrap_or(2048);
        let temperature = gen_config.temperature.unwrap_or(0.7);
        let top_k = gen_config.top_k.unwrap_or(0);
        let top_p = gen_config.top_p.unwrap_or(0.9);
        let min_p = gen_config.min_p.unwrap_or(0.0);
        let repetition_penalty = gen_config.repetition_penalty.unwrap_or(1.0);
        let repetition_context_size = gen_config.repetition_context_size.unwrap_or(256);
        let presence_penalty = gen_config.presence_penalty.unwrap_or(0.0);
        let presence_context_size = gen_config.presence_context_size.unwrap_or(20);
        let frequency_penalty = gen_config.frequency_penalty.unwrap_or(0.0);
        let frequency_context_size = gen_config.frequency_context_size.unwrap_or(20);
        let max_consecutive_tokens = gen_config.max_consecutive_tokens.unwrap_or(16);
        let max_ngram_repeats = gen_config.max_ngram_repeats.unwrap_or(3);
        let ngram_size = gen_config.ngram_size.unwrap_or(64);
        let eos_token_id = gen_config.eos_token_id.or(Some(model_config.eos_token_id));
        let return_logprobs = gen_config.return_logprobs.unwrap_or(false);
        let prefill_step_size = gen_config.prefill_step_size.unwrap_or(2048) as usize;

        let generation_stream = Stream::new(DeviceType::Gpu);

        let num_layers = layers.len();
        let mut kv_keys = initial_kv_keys.unwrap_or_else(|| vec![None; num_layers]);
        let mut kv_values = initial_kv_values.unwrap_or_else(|| vec![None; num_layers]);
        let mut cache_idx: i32 = initial_cache_idx;

        let mut rope_offsets = MxArray::from_int32(&[cache_idx], &[1])?;
        let left_padding = MxArray::from_int32(&[0], &[1])?;

        let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens as usize);
        let mut generated_logprobs: Vec<f32> = if return_logprobs {
            Vec::with_capacity(max_new_tokens as usize)
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

        let decode_start = if report_perf {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut first_token_elapsed_ms: Option<f64> = None;

        // PREFILL
        let mut last_logits = if let Some(current_ids) = prefill_input_ids {
            let total_seq_len = current_ids.shape_at(1)? as usize;
            let use_chunked_prefill = prefill_step_size > 0 && total_seq_len > prefill_step_size;

            if use_chunked_prefill {
                let mut offset = 0usize;
                while offset + prefill_step_size < total_seq_len {
                    let chunk_end = offset + prefill_step_size;
                    let chunk = current_ids.slice(&[0, offset as i64], &[1, chunk_end as i64])?;
                    rope_offsets = MxArray::from_int32(&[cache_idx], &[1])?;
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
                let final_chunk =
                    current_ids.slice(&[0, offset as i64], &[1, total_seq_len as i64])?;
                rope_offsets = MxArray::from_int32(&[cache_idx], &[1])?;
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
            }
        } else {
            // Zero delta — re-run last token
            let last_token_id = token_ids_vec[token_ids_vec.len() - 1];
            cache_idx -= 1;
            rope_offsets = MxArray::from_int32(&[cache_idx], &[1])?;
            let last_token = MxArray::from_uint32(&[last_token_id], &[1, 1])?;
            let logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                Qwen3Model::forward_fused(
                    &last_token,
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
            logits.squeeze(Some(&[0, 1]))?
        };

        rope_offsets = MxArray::from_int32(&[cache_idx], &[1])?;

        if repetition_penalty != 1.0 && !token_ids_vec.is_empty() {
            last_logits = apply_repetition_penalty(
                &last_logits,
                &token_ids_vec,
                repetition_penalty,
                Some(repetition_context_size),
            )?;
        }
        if presence_penalty != 0.0 {
            last_logits = apply_presence_penalty(
                &last_logits,
                &token_ids_vec,
                presence_penalty,
                Some(presence_context_size),
            )?;
        }
        if frequency_penalty != 0.0 {
            last_logits = apply_frequency_penalty(
                &last_logits,
                &token_ids_vec,
                frequency_penalty,
                Some(frequency_context_size),
            )?;
        }

        let (mut token, mut logprobs_arr) = if return_logprobs {
            let (tok, lp) = sample_and_logprobs(&last_logits, Some(sampling_config))?;
            (tok, Some(lp))
        } else {
            (sample(&last_logits, Some(sampling_config))?, None)
        };

        // DECODE LOOP
        const DECODE_CLEANUP_INTERVAL: i32 = 256;
        let one_arr = MxArray::from_int32(&[1], &[1])?;
        for step in 0..max_new_tokens {
            let _stream_ctx = StreamContext::new(generation_stream);
            token.eval();
            if step > 0 && step % DECODE_CLEANUP_INTERVAL == 0 {
                synchronize_and_clear_cache();
            }
            let token_value = token.item_at_int32(0)? as u32;
            if let Some(ds) = decode_start
                && first_token_elapsed_ms.is_none()
            {
                first_token_elapsed_ms = Some(ds.elapsed().as_secs_f64() * 1000.0);
            }
            generated_tokens.push(token_value);
            if return_logprobs && let Some(ref lp) = logprobs_arr {
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
                let context_tokens: Vec<u32> = token_ids_vec
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
            logprobs_arr = next_lp;
        }

        // Save cache state
        if reuse_cache {
            self.cached_kv_keys = kv_keys;
            self.cached_kv_values = kv_values;
            self.cached_cache_idx = cache_idx;
            let mut full_history = token_ids_vec.clone();
            let history_tokens = if finish_reason != "length" && !generated_tokens.is_empty() {
                &generated_tokens[..generated_tokens.len() - 1]
            } else {
                &generated_tokens
            };
            full_history.extend_from_slice(history_tokens);
            self.cached_token_history = full_history;
        } else {
            self.cached_kv_keys.clear();
            self.cached_kv_values.clear();
            self.cached_cache_idx = 0;
            self.cached_token_history.clear();
        }

        let gen_elapsed = gen_start.map(|s| s.elapsed());

        // Decode text
        let generated_ids_vec: Vec<u32> = generated_tokens.clone();
        let raw_text = tokenizer.decode_sync(&generated_ids_vec, true)?;

        let (cleaned_text, tool_calls, thinking) = tools::parse_generation_output(&raw_text);

        let finish_reason = if tool_calls.iter().any(|tc| tc.status == "ok") {
            "tool_calls".to_string()
        } else {
            finish_reason.to_string()
        };

        let performance = if let (Some(gen_elapsed), Some(first_tok_ms)) =
            (gen_elapsed, first_token_elapsed_ms)
        {
            let total_ms = gen_elapsed.as_secs_f64() * 1000.0;
            let gen_toks = generated_tokens.len() as f64;
            let ttft_ms = first_tok_ms;
            let decode_ms = total_ms - ttft_ms;
            Some(crate::profiling::PerformanceMetrics {
                ttft_ms,
                prefill_tokens_per_second: if ttft_ms > 0.0 {
                    prompt_token_count / (ttft_ms / 1000.0)
                } else {
                    0.0
                },
                decode_tokens_per_second: if decode_ms > 0.0 && gen_toks > 1.0 {
                    (gen_toks - 1.0) / (decode_ms / 1000.0)
                } else {
                    0.0
                },
            })
        } else {
            None
        };

        Ok(ChatResult {
            text: cleaned_text,
            tool_calls,
            thinking,
            num_tokens: generated_tokens.len() as u32,
            finish_reason,
            raw_text,
            performance,
        })
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
        let max_consecutive_tokens = config.max_consecutive_tokens.unwrap_or(16);
        let max_ngram_repeats = config.max_ngram_repeats.unwrap_or(3);
        let ngram_size = config.ngram_size.unwrap_or(64);
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
        let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens as usize);
        let mut generated_logprobs: Vec<f32> = if return_logprobs {
            Vec::with_capacity(max_new_tokens as usize)
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
                        reasoning_content: m.reasoning_content.clone(),
                        images: None,
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
                        reasoning_content: m.reasoning_content.clone(),
                        images: None,
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
    fn init_training_sync(
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

    fn save_optimizer_state_sync(&self, path: String) -> Result<()> {
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

        let params_clone: HashMap<String, MxArray> =
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
        crate::utils::safetensors::save_safetensors(&safetensors_path, &params_clone, metadata)?;
        info!("Saved weights.safetensors");

        let weights_str = serde_json::to_string_pretty(&weights_json)?;
        let weights_path = path.join("weights.mlx");
        std::fs::write(&weights_path, weights_str)?;
        info!("Saved weights.mlx metadata");

        Ok(())
    }

    fn load_optimizer_state_sync(&mut self, path: String) -> Result<()> {
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
    fn generate_for_training_thread_sync(
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
        let max_consecutive_tokens = config.max_consecutive_tokens.unwrap_or(16);
        let max_ngram_repeats = config.max_ngram_repeats.unwrap_or(3);
        let ngram_size = config.ngram_size.unwrap_or(64);
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
        let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens as usize);
        let mut generated_logprobs: Vec<f32> = if return_logprobs {
            Vec::with_capacity(max_new_tokens as usize)
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
    fn train_step_grpo_sync(
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
                        .map(|(name, grad)| (name.clone(), grad.mul(&scale_arr).unwrap()))
                        .collect();
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

                // Create deltas: delta = param - updated (so param - 1.0 * delta = updated)
                let delta_map: HashMap<String, MxArray> = param_names_vec
                    .iter()
                    .enumerate()
                    .map(|(i, name)| {
                        let delta = param_refs[i].sub(&updated[i]).unwrap();
                        (name.clone(), delta)
                    })
                    .collect();

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
    fn train_step_sft_sync(
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
                        .map(|(name, grad)| (name.clone(), grad.mul(&scale_arr).unwrap()))
                        .collect();
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

                // Create deltas: delta = param - updated (so param - 1.0 * delta = updated)
                let delta_map: HashMap<String, MxArray> = param_names_vec
                    .iter()
                    .enumerate()
                    .map(|(i, name)| {
                        let delta = param_refs[i].sub(&updated[i]).unwrap();
                        (name.clone(), delta)
                    })
                    .collect();

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
}

#[napi]
impl Qwen3Model {
    /// Reset the KV cache used for cache reuse across chat() calls.
    /// Call this when starting a new conversation to ensure a full prefill.
    #[napi]
    pub fn reset_cache(&self) -> Result<()> {
        send_and_block(&self.thread, |reply| Qwen3Cmd::ResetCache { reply })
    }

    /// Initialize KV caches for incremental generation
    ///
    /// Creates one KV cache per transformer layer. Call this before starting generation.
    #[napi]
    pub fn init_kv_caches(&self) -> Result<()> {
        send_and_block(&self.thread, |reply| Qwen3Cmd::InitKvCaches { reply })
    }

    /// Reset all KV caches
    ///
    /// Clears cached key-value states. Call this between different generation sequences.
    #[napi]
    pub fn reset_kv_caches(&self) -> Result<()> {
        send_and_block(&self.thread, |reply| Qwen3Cmd::ResetKvCaches { reply })
    }

    /// Get paged attention memory statistics (if enabled)
    ///
    /// Returns memory usage statistics for the paged KV cache.
    #[napi]
    pub fn paged_cache_stats(&self) -> Result<Option<PagedCacheStats>> {
        send_and_block(&self.thread, |reply| Qwen3Cmd::PagedCacheStats { reply })
    }

    /// Get scheduler statistics (if paged attention is enabled)
    ///
    /// Returns the number of waiting, running, and completed sequences.
    #[napi]
    pub fn scheduler_stats(&self) -> Result<Option<SchedulerStatsNapi>> {
        send_and_block(&self.thread, |reply| Qwen3Cmd::SchedulerStats { reply })
    }

    /// Forward pass with paged attention for memory-efficient inference.
    ///
    /// This method uses block-based KV cache management via Metal kernels for:
    /// - Variable-length sequences with efficient memory usage
    /// - Continuous batching with dynamic batch composition
    /// - Long context support beyond GPU memory limits
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs, shape: [num_seqs, 1] for decode
    /// * `slot_mapping` - Slot indices for cache updates, shape: [num_seqs]
    /// * `seq_ids` - Sequence IDs in the batch (for looking up block tables/context lens)
    /// * `positions` - Token positions for RoPE, shape: [num_seqs] (per-sequence positions)
    ///
    /// # Returns
    /// * Logits, shape: [num_seqs, 1, vocab_size] for decode
    #[napi]
    pub fn forward_paged(
        &self,
        input_ids: &MxArray, // [num_seqs, 1] for decode
        slot_mapping: &MxArray,
        seq_ids: Vec<u32>,
        positions: &MxArray, // [num_seqs] - per-sequence RoPE positions
    ) -> Result<MxArray> {
        send_and_block(&self.thread, |reply| Qwen3Cmd::ForwardPaged {
            input_ids: input_ids.clone(),
            slot_mapping: slot_mapping.clone(),
            seq_ids,
            positions: positions.clone(),
            reply,
        })
    }

    /// Prefill a sequence using standard attention and write K/V to paged cache.
    ///
    /// This method should be called before `step_paged_generation()` for each
    /// new prompt. It runs the full forward pass using standard attention
    /// (which is faster for long sequences), then writes the K/V cache to
    /// the paged cache for subsequent decode steps.
    ///
    /// # Arguments
    /// * `prompt_tokens` - Token IDs for the prompt (as u32 array)
    /// * `seq_id` - Sequence ID (obtained from scheduler)
    ///
    /// # Returns
    /// * Logits for the last token, shape: [1, vocab_size]
    #[napi]
    pub fn prefill_paged(&self, prompt_tokens: Vec<u32>, seq_id: u32) -> Result<MxArray> {
        send_and_block(&self.thread, |reply| Qwen3Cmd::PrefillPaged {
            prompt_tokens,
            seq_id,
            reply,
        })
    }

    /// Add a request to the paged attention scheduler.
    ///
    /// The scheduler queues requests and allocates blocks for KV cache.
    /// Use `step_paged_generation()` to process the scheduled batch.
    ///
    /// Note: The actual sequence ID is assigned during scheduling, not when the
    /// request is added. Use the `request_id` to track your requests through
    /// the generation process.
    ///
    /// # Arguments
    /// * `request_id` - Unique identifier for the request (returned in outputs)
    /// * `prompt_tokens` - Token IDs for the prompt
    /// * `max_new_tokens` - Maximum new tokens to generate
    /// * `priority` - Optional priority (higher = scheduled first)
    ///
    /// # Returns
    /// * Number of pending requests in the queue
    #[napi]
    pub fn add_paged_request(
        &self,
        request_id: String,
        prompt_tokens: Vec<u32>,
        max_new_tokens: u32,
        priority: Option<i32>,
    ) -> Result<u32> {
        send_and_block(&self.thread, |reply| Qwen3Cmd::AddPagedRequest {
            request_id,
            prompt_tokens,
            max_new_tokens,
            priority,
            reply,
        })
    }

    /// Schedule and execute one step of paged generation.
    ///
    /// This method:
    /// 1. Schedules the next batch of sequences
    /// 2. Runs forward pass with paged attention
    /// 3. Samples next tokens
    /// 4. Returns the generated tokens for each sequence
    ///
    /// # Arguments
    /// * `config` - Generation configuration (temperature, top_k, etc.)
    ///
    /// # Returns
    /// * `PagedGenerationStep` with token outputs for each sequence
    #[napi]
    pub fn step_paged_generation(
        &self,
        config: Option<GenerationConfig>,
    ) -> Result<Option<PagedGenerationStep>> {
        send_and_block(&self.thread, |reply| Qwen3Cmd::StepPagedGeneration {
            config,
            reply,
        })
    }

    /// Get completed sequences from the scheduler.
    ///
    /// Call this after `step_paged_generation()` returns outputs with `is_finished: true`.
    #[napi]
    pub fn get_completed_sequences(&self) -> Result<Vec<PagedCompletedSequence>> {
        send_and_block(&self.thread, |reply| Qwen3Cmd::GetCompletedSequences {
            reply,
        })
    }

    /// Check if the scheduler has pending work.
    #[napi]
    pub fn has_paged_work(&self) -> Result<bool> {
        send_and_block(&self.thread, |reply| Qwen3Cmd::HasPagedWork { reply })
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
        for (i, (existing, new_ptr)) in kv_keys.iter_mut().zip(out_kv_keys.into_iter()).enumerate()
        {
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

        for (i, (existing, new_ptr)) in kv_values
            .iter_mut()
            .zip(out_kv_values.into_iter())
            .enumerate()
        {
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

// Shell-only dead code (generate_batch_parallel_sync, apply_batch_repetition_penalty)
// removed in P6b — batched generation is handled on the model thread via
// Qwen3Cmd::GenerateBatch -> Qwen3Inner::generate_batch_sync().

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

    /// High-level chat API with structured response parsing
    ///
    /// The primary API for conversational AI. Handles:
    /// - Chat message formatting with Jinja2 templates
    /// - Tool/function calling with structured output
    /// - Thinking extraction from `<think>` tags
    /// - Clean response text with all special tags stripped
    ///
    /// ## `chat()` vs `generate()`
    ///
    /// | Feature | `chat()` | `generate()` |
    /// |---------|----------|--------------|
    /// | **Purpose** | Conversational AI with tools | Raw text generation |
    /// | **Input** | Chat messages | Token IDs (MxArray) |
    /// | **Tool Support** | Built-in parsing | None |
    /// | **Thinking** | Extracts `<think>` content | Raw text only |
    /// | **Output** | Structured `ChatResult` | Basic `GenerationResult` |
    /// | **Use Case** | Chat apps, agents, assistants | Training, low-level control |
    ///
    /// ## When to use `chat()`
    /// - Building conversational applications
    /// - Need tool/function calling
    /// - Want structured responses with thinking separated
    /// - Working with chat message format
    ///
    /// ## When to use `generate()`
    /// - Training and fine-tuning (need raw logprobs)
    /// - Custom tokenization pipeline
    /// - Low-level generation control
    /// - Non-chat use cases
    ///
    /// # Arguments
    /// * `messages` - Array of chat messages (user/assistant/system roles)
    /// * `config` - Chat configuration including optional tools and generation params
    ///
    /// # Returns
    /// * `ChatResult` containing:
    ///   - `text`: Clean response (tool_call and think tags stripped)
    ///   - `thinking`: Extracted chain-of-thought reasoning (or null)
    ///   - `toolCalls`: Parsed tool calls with native JS object arguments
    ///   - `finishReason`: "stop" | "length" | "tool_calls"
    ///   - `rawText`: Original text before processing (for debugging)
    ///
    /// # Example
    /// ```typescript
    /// // Simple chat
    /// const result = await model.chat(messages);
    /// console.log(result.text);
    ///
    /// // With tools
    /// const result = await model.chat(messages, {
    ///   tools: [{ type: 'function', function: { name: 'get_weather' } }],
    ///   maxNewTokens: 2048,
    ///   temperature: 0.7,
    /// });
    ///
    /// // Handle tool calls
    /// for (const call of result.toolCalls) {
    ///   if (call.status === 'ok') {
    ///     console.log(call.name, call.arguments);  // Arguments is a JS object!
    ///   }
    /// }
    ///
    /// // Access thinking (chain-of-thought)
    /// if (result.thinking) {
    ///   console.log('Model reasoning:', result.thinking);
    /// }
    /// ```
    #[napi]
    pub async fn chat(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
    ) -> Result<ChatResult> {
        let config = config.unwrap_or_default();
        send_and_await(&self.thread, |reply| Qwen3Cmd::Chat {
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

    /// Decode token IDs to text using the internal tokenizer
    ///
    /// Helper method for decoding generated tokens. The model must have been loaded
    /// via load() to have a tokenizer available.
    ///
    /// # Arguments
    /// * `token_ids` - Token IDs to decode as Uint32Array
    /// * `skip_special_tokens` - Whether to skip special tokens (default: true)
    ///
    /// # Returns
    /// * Decoded text string
    #[napi]
    pub async fn decode(
        &self,
        token_ids: Uint32Array,
        skip_special_tokens: Option<bool>,
    ) -> Result<String> {
        let skip_special = skip_special_tokens.unwrap_or(true);
        let token_ids_vec = token_ids.to_vec();

        send_and_await(&self.thread, |reply| Qwen3Cmd::Decode {
            token_ids: token_ids_vec,
            skip_special_tokens: skip_special,
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
}
