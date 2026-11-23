//! Thin NAPI facade for `Qwen3_5MoeModel`, backed by the unified
//! `qwen3_5::model::Qwen35Inner` model thread.

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::engine::cmd::ChatCmd;
use crate::engine::types::{ChatConfig, ChatStreamChunk, ChatStreamHandle};
use crate::model_thread::send_and_await;
use crate::models::qwen3_5::model::{Qwen3_5ContextLimits, Qwen35Cmd, Qwen35FamilyCommand};
use crate::models::qwen3_5_moe::config::Qwen3_5MoeConfig;
use crate::models::qwen3_5_moe::persistence;
use crate::tokenizer::ChatMessage;
use crate::vision::qwen::processing::QwenImageProcessor;
use crate::vision::qwen::prompt::expanded_prompt_token_count;

#[cfg(test)]
pub(crate) type Qwen35MoeInner = crate::models::qwen3_5::model::Qwen35Inner;

/// Test-only between-turn snapshot of the MoE paged-MTP GDN bookkeeping, read
/// via [`Qwen35FamilyCommand::MtpPagedGdnStateForTest`]. Serialized behind the
/// model thread, so it observes the fully-finalized preceding turn.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct MoeMtpPagedGdnStateForTest {
    /// Whether the block-paged adapter is installed (the paged MTP lane).
    pub paged_active: bool,
    /// `cached_token_history.len()` — the committed drop-last history.
    pub history_len: usize,
    /// Engine-computed accepted-but-unemitted tail of the most recent paged
    /// MTP turn.
    pub last_rollback_unemitted: usize,
    /// Mid-cycle GDN rewinds performed.
    pub gdn_rewinds: u64,
    /// GDN invalidations: failed rewinds + epilogue frontier disagreements.
    pub gdn_invalidations: u64,
    /// Whether the refuse-to-persist latch is currently armed.
    pub state_dirty: bool,
    /// Whether a GDN history checkpoint is currently stored.
    pub has_history_checkpoint: bool,
}

/// Qwen3.5 MoE Model -- hybrid linear/full attention with Mixture-of-Experts.
///
/// All inference and training state lives on a dedicated OS thread. NAPI methods
/// dispatch commands via channels and await responses. Training commands are
/// routed through `TrainingDispatch` to the model thread.
#[napi]
pub struct Qwen3_5MoeModel {
    /// Dedicated model thread for inference and training.
    pub(crate) thread: crate::model_thread::ModelThread<Qwen35Cmd>,
    /// What the load's warm-up did; `None` for a load with no paged pool.
    pub(crate) warmup: Option<crate::engine::warmup::WarmupReport>,
    /// Cloned from inner for pure-getter NAPI methods (no command dispatch needed).
    pub(crate) config: Qwen3_5MoeConfig,
    /// Snapshot of `Qwen35Inner::paged_adapter.is_some()` captured at
    /// construction time.
    pub(crate) paged_active: bool,
    /// The decode lanes the load's memory plan granted.
    pub(crate) scheduled_lanes: u32,
    /// Snapshot of `Qwen35Inner::has_mtp_weights()` / DFlash2 captured at load.
    pub(crate) mtp_active: bool,
    /// Snapshot of the fully loaded image execution stack.
    pub(crate) vision_active: bool,
    /// Loaded processor and merge-size snapshots retained for CPU-only image
    /// token planning before SSE.
    pub(crate) image_processor: Option<Arc<QwenImageProcessor>>,
    pub(crate) spatial_merge_size: i32,
    pub(crate) context_limits: Qwen3_5ContextLimits,
    /// Directory containing tokenizer/config assets for this loaded model.
    pub(crate) model_assets_path: String,
    /// The model thread's state tier, never held open from here.
    pub(crate) state_tier: crate::disk_tier::TierHandle,
    /// RAII: this model's baseline and its paged pool's debit with the
    /// cache-limit coordinator, released by `dispose` or on drop.
    pub(crate) cache_limit_guards: std::sync::Mutex<
        Option<(
            crate::cache_limit::CacheLimitGuard,
            Option<crate::cache_limit::PoolCacheLimitGuard>,
        )>,
    >,
}

impl Qwen3_5MoeModel {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_loaded(
        thread: crate::model_thread::ModelThread<Qwen35Cmd>,
        config: Qwen3_5MoeConfig,
        warmup: Option<crate::engine::warmup::WarmupReport>,
        paged_active: bool,
        scheduled_lanes: u32,
        mtp_active: bool,
        vision_active: bool,
        image_processor: Option<Arc<QwenImageProcessor>>,
        spatial_merge_size: i32,
        context_limits: Qwen3_5ContextLimits,
        model_assets_path: String,
        state_tier: crate::disk_tier::TierHandle,
        cache_limit: crate::cache_limit::CacheLimitGuard,
        pool_cache_limit: Option<crate::cache_limit::PoolCacheLimitGuard>,
    ) -> Self {
        Self {
            thread,
            warmup,
            config,
            paged_active,
            scheduled_lanes,
            mtp_active,
            vision_active,
            image_processor,
            spatial_merge_size,
            context_limits,
            model_assets_path,
            state_tier,
            cache_limit_guards: std::sync::Mutex::new(Some((cache_limit, pool_cache_limit))),
        }
    }
}

#[napi]
impl Qwen3_5MoeModel {
    /// Frees the weights, caches and state now: the model thread drops them
    /// and exits; every later call fails.
    #[napi]
    pub async fn dispose(&self) -> Result<()> {
        self.thread.dispose().await?;
        self.cache_limit_guards
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        Ok(())
    }
}

#[napi]
impl Qwen3_5MoeModel {
    /// Resolved directory containing this model's tokenizer/config assets.
    /// Streaming wrappers use it after a direct GGUF load so chat templating
    /// reads the reconstructed sidecars from the native-packed cache.
    #[napi]
    pub fn model_assets_path(&self) -> String {
        self.model_assets_path.clone()
    }

    /// The directory name of the model's open persistent SSD cache under
    /// its cache root, read from the cache itself; null without one, for a
    /// temporary cache, and after `closeCache` or `dispose`.
    #[napi]
    pub fn state_tier_namespace(&self) -> Option<String> {
        self.state_tier.namespace()
    }

    /// What the load's warm-up did; `null` for a load with no paged pool,
    /// which runs none.
    #[napi(getter)]
    pub fn warmup(&self) -> Option<crate::engine::warmup::WarmupReport> {
        self.warmup.clone()
    }

    /// Whether the block-paged KV cache adapter is active on this model
    /// instance.
    ///
    /// `true` iff `Qwen35MoeInner::paged_adapter` was successfully
    /// constructed at load time (every load on Metal). On
    /// VLM checkpoints image turns run on the paged-vision core; a load with
    /// no adapter admits no image turn. Surfaced through this NAPI method so
    /// server endpoints can branch on it without round-tripping through
    /// the model thread.
    #[napi]
    pub fn has_block_paged_cache(&self) -> bool {
        self.paged_active
    }

    /// Native admission width for plain text autoregressive turns. Native
    /// MTP, DFlash2 and multimodal turns retain their speculative or exclusive
    /// admission policy instead of using this plain autoregressive width.
    #[napi]
    pub fn max_concurrent_sequences(&self) -> u32 {
        crate::models::qwen3_5::model::qwen35_max_concurrent_sequences(
            self.paged_active,
            self.scheduled_lanes,
        )
    }

    /// Test-only: snapshot the paged-MTP GDN bookkeeping between turns.
    #[doc(hidden)]
    pub async fn mtp_paged_gdn_state_for_test(&self) -> Result<MoeMtpPagedGdnStateForTest> {
        let state = send_and_await(&self.thread, |reply| {
            Qwen35FamilyCommand::MtpPagedGdnStateForTest { reply }
        })
        .await?;
        Ok(MoeMtpPagedGdnStateForTest {
            paged_active: state.paged_active,
            history_len: state.history_len,
            last_rollback_unemitted: state.last_rollback_unemitted,
            gdn_rewinds: state.gdn_rewinds,
            gdn_invalidations: state.gdn_invalidations,
            state_dirty: state.state_dirty,
            has_history_checkpoint: state.has_history_checkpoint,
        })
    }

    /// Test-only state oracle: recompute GDN over the persisted history
    /// checkpoint's own token key from fresh caches and bit-compare against the
    /// checkpoint. `Ok(true)` iff the persisted state equals what its key says.
    #[doc(hidden)]
    pub async fn gdn_history_checkpoint_oracle_for_test(&self) -> Result<bool> {
        send_and_await(&self.thread, |reply| {
            Qwen35FamilyCommand::GdnHistoryCheckpointOracleForTest { reply }
        })
        .await
    }

    /// Whether this instance has a native MTP head or a loaded DFlash2
    /// companion. Snapshotted at load time so the TS `ChatSession` can
    /// auto-default `enableMtp = true` without dispatching a command into the
    /// model thread. Mirrors `Qwen3_5Model::has_mtp_weights`.
    ///
    /// Note: this only reports weight availability. Whether the
    /// speculative-decode path actually runs on a given call also requires
    /// the per-request `enableMtp` flag.
    #[napi]
    pub fn has_mtp_weights(&self) -> bool {
        self.mtp_active
    }

    /// Whether this loaded model instance can execute image-bearing turns.
    ///
    /// This is an authoritative load-time snapshot, not a model-family guess:
    /// it requires the loaded vision encoder, image processor, and block-paged
    /// KV adapter used by the MoE vision path.
    #[napi]
    pub fn supports_images(&self) -> bool {
        self.vision_active
    }

    /// Synchronous active-context snapshot shared with the dense wrapper.
    #[napi]
    pub fn context_limits(&self) -> Qwen3_5ContextLimits {
        self.context_limits.clone()
    }

    /// Exact, non-mutating Qwen image-placeholder expansion count for a fully
    /// rendered prompt and complete message history.
    #[napi]
    pub async fn expanded_prompt_token_count(
        &self,
        prompt_tokens: Uint32Array,
        messages: Vec<ChatMessage>,
    ) -> Result<u32> {
        expanded_prompt_token_count(
            self.image_processor.clone(),
            self.spatial_merge_size,
            prompt_tokens,
            messages,
            None,
        )
        .await
    }

    /// Load a pretrained model from a directory, or a `.gguf` file.
    ///
    /// A direct GGUF file is read into memory and repacked on the model
    /// thread; only its non-weight assets (config, tokenizer) are written to
    /// the native GGUF asset directory, which the model reports as its assets
    /// path.
    #[napi]
    pub async fn load(
        path: String,
        options: Option<crate::models::qwen3_5::model::Qwen35LoadOptions>,
    ) -> Result<Qwen3_5MoeModel> {
        let progress = crate::engine::load_progress::begin(&path);
        let options = options.unwrap_or_default();
        let disk = options.disk_tier("Qwen3.5 MoE")?;
        let draft_model_path = options.draft_model_path;
        let kv_format = options
            .kv_format
            .as_deref()
            .map(|value| crate::transformer::KvFormat::parse(Some(value)))
            .transpose()
            .map_err(|message| Error::from_reason(format!("Qwen3.5 MoE load: {message}")))?;
        let activations = crate::transformer::Activations::parse(
            options.activations.as_deref(),
            crate::models::qwen3_5::load_plan::default_activations(true, false),
        )
        .map_err(|message| Error::from_reason(format!("Qwen3.5 MoE load: {message}")))?;
        let max_memory_bytes = crate::models::qwen3_5::load_plan::max_memory_option(
            "Qwen3.5 MoE",
            options.max_memory_bytes,
        )?;
        let max_context =
            crate::nn::rope_scaling::max_context_option("Qwen3.5 MoE", options.max_context_tokens)?;
        let decode_share = crate::models::qwen3_5::load_plan::decode_share_option(
            "Qwen3.5 MoE",
            options.decode_share,
        )?;
        let source = crate::engine::checkpoint_source::CheckpointSource::resolve(
            std::path::Path::new(&path),
            crate::utils::gguf::NativeGgufFamily::Qwen35Moe,
        )
        .await?;
        let loaded = persistence::load_with_source(
            &source,
            draft_model_path,
            kv_format,
            activations,
            max_memory_bytes,
            max_context,
            decode_share,
            disk,
            progress.cell(),
        )
        .await;
        if loaded.is_err() {
            source.release_lease();
        }
        loaded
    }

    /// Test-only entry point that dispatches `ChatStreamSessionStart`
    /// and returns the raw mpsc receiver the model thread writes into.
    #[doc(hidden)]
    pub fn chat_stream_session_start_for_test(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
    ) -> Result<(
        ChatStreamHandle,
        tokio::sync::mpsc::Receiver<Result<ChatStreamChunk>>,
    )> {
        let config = config.unwrap_or_default();
        let (plumbing, stream_rx) = crate::engine::napi_glue::chat_stream_plumbing();
        let crate::engine::napi_glue::ChatStreamPlumbing {
            cancelled: cancelled_inner,
            stream_tx,
            handle,
        } = plumbing;
        self.thread
            .send(Qwen35Cmd::from_chat(ChatCmd::StreamSessionStart {
                messages,
                config,
                stream_tx,
                cancelled: cancelled_inner,
            }))?;
        Ok((handle, stream_rx))
    }

    /// Test-only entry point that dispatches `ChatStreamSessionContinue`
    /// and returns the raw mpsc receiver the model thread writes into.
    #[doc(hidden)]
    pub fn chat_stream_session_continue_for_test(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
    ) -> Result<(
        ChatStreamHandle,
        tokio::sync::mpsc::Receiver<Result<ChatStreamChunk>>,
    )> {
        let config = config.unwrap_or_default();
        let (plumbing, stream_rx) = crate::engine::napi_glue::chat_stream_plumbing();
        let crate::engine::napi_glue::ChatStreamPlumbing {
            cancelled: cancelled_inner,
            stream_tx,
            handle,
        } = plumbing;
        self.thread
            .send(Qwen35Cmd::from_chat(ChatCmd::StreamSessionContinue {
                messages,
                config,
                stream_tx,
                cancelled: cancelled_inner,
            }))?;
        Ok((handle, stream_rx))
    }

    /// Get the number of parameters in the model.
    ///
    /// Pure config computation -- no model-thread dispatch needed.
    #[napi]
    pub fn num_parameters(&self) -> i64 {
        let h = self.config.hidden_size as i64;
        let v = self.config.vocab_size as i64;
        let n = self.config.num_layers as usize;
        let dense_i = self.config.intermediate_size as i64;

        let mut total = v * h;
        if !self.config.tie_word_embeddings {
            total += v * h;
        }

        let num_experts = self.config.num_experts as i64;
        let moe_i = self
            .config
            .moe_intermediate_size
            .unwrap_or(self.config.intermediate_size) as i64;
        let shared_i = self
            .config
            .shared_expert_intermediate_size
            .unwrap_or(self.config.intermediate_size) as i64;

        let kd = self.config.linear_key_dim() as i64;
        let vd = self.config.linear_value_dim() as i64;

        for layer_idx in 0..n {
            let is_linear = self.config.is_linear_layer(layer_idx);
            let is_moe = self.config.is_moe_layer(layer_idx);

            if is_linear {
                let num_vh = self.config.linear_num_value_heads as i64;
                let vhd = self.config.linear_value_head_dim as i64;
                total += h * (kd * 2 + vd * 2)
                    + h * (num_vh * 2)
                    + (kd * 2 + vd) * self.config.linear_conv_kernel_dim as i64
                    + vd * h
                    + num_vh
                    + num_vh
                    + vhd;
            } else {
                let d = self.config.head_dim as i64;
                total += h * h * 2 + h * (self.config.num_kv_heads as i64 * d) * 2 + h * h + d * 2;
            }

            if is_moe {
                total += h * num_experts + num_experts * 3 * h * moe_i + 3 * h * shared_i + h;
            } else {
                total += 3 * h * dense_i;
            }

            total += h * 2;
        }

        total += h;
        total
    }

    /// Save the model weights and configuration to a directory.
    ///
    /// Dispatches to model thread.
    #[napi]
    pub fn save_model<'env>(
        &self,
        env: &'env Env,
        save_path: String,
    ) -> Result<PromiseRaw<'env, ()>> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.thread.send(Qwen35FamilyCommand::SaveModel {
            save_path,
            reply: tx,
        })?;
        let promise = env.spawn_future(async move {
            rx.await
                .map_err(|_| napi::Error::from_reason("Model thread exited unexpectedly"))?
        })?;
        Ok(promise)
    }

    /// Closes the model's SSD cache at a clean stop (Splash
    /// `closePersistentCache`): a persistent cache first writes every
    /// restore point still waiting, newest first, until `deadlineMs`
    /// (milliseconds on `Date.now()`'s clock, so models closed in turn
    /// share one deadline); a temporary cache is removed. The cache takes no
    /// write afterwards. Resolves false when something missed the deadline.
    ///
    /// Dispatches to model thread.
    #[napi]
    pub async fn close_cache(&self, deadline_ms: f64) -> Result<bool> {
        crate::model_thread::send_and_await(&self.thread, |reply| Qwen35FamilyCommand::CloseCache {
            deadline_ms,
            reply,
        })
        .await
    }
}

crate::models::chat_napi::chat_napi_surface! {
    class: Qwen3_5MoeModel,
    thread_cmd: Qwen35Cmd,
    thread: direct,
    image_guard: none,
    ts_stream_start: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue_tool: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
}
