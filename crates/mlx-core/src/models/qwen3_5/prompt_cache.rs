use napi_derive::napi;

use super::layer_cache::Qwen3_5LayerCache;

/// Opaque handle to KV cache state from a chat-session turn.
///
/// Pass this back via `model.setCache(cache)` before the next
/// chat-session call to enable incremental prefill — only new tokens
/// since the last turn are processed, avoiding redundant computation.
///
/// Created internally by the model during chat-session turns.
/// Extract via `model.takeCache()`, restore via `model.setCache(cache)`.
#[napi]
pub struct PromptCache {
    /// Per-layer KV cache states
    pub(crate) caches: Option<Vec<Qwen3_5LayerCache>>,
    /// Full token sequence that produced this cache state
    pub(crate) token_history: Vec<u32>,
    /// Model type identifier for validation
    pub(crate) model_type: String,
    /// Number of layers — used to validate compatible model on restore
    pub(crate) num_layers: usize,
    /// Image cache key for VLM cache reuse (None for text-only)
    pub(crate) image_cache_key: Option<u64>,
    /// Rope deltas from VLM prefill (None for text-only)
    pub(crate) rope_deltas: Option<i32>,
    /// Model instance ID — prevents restoring cache into a different checkpoint.
    /// Each Qwen3_5Model/MoeModel instance gets a unique ID from a shared counter.
    pub(crate) model_id: u64,
}

#[napi]
impl PromptCache {
    /// Number of tokens stored in this cache.
    #[napi(getter)]
    pub fn token_count(&self) -> u32 {
        self.token_history.len() as u32
    }

    /// Whether this cache has been consumed (caches moved out).
    #[napi(getter)]
    pub fn is_empty(&self) -> bool {
        self.caches.is_none()
    }

    /// Release GPU memory held by this cache.
    #[napi]
    pub fn dispose(&mut self) {
        self.caches = None;
        self.token_history.clear();
        self.image_cache_key = None;
        self.rope_deltas = None;
    }
}

impl PromptCache {
    /// Create a new PromptCache with the given state.
    pub(crate) fn new(
        caches: Vec<Qwen3_5LayerCache>,
        token_history: Vec<u32>,
        model_type: &str,
        num_layers: usize,
        image_cache_key: Option<u64>,
        rope_deltas: Option<i32>,
        model_id: u64,
    ) -> Self {
        Self {
            caches: Some(caches),
            token_history,
            model_type: model_type.to_string(),
            num_layers,
            image_cache_key,
            rope_deltas,
            model_id,
        }
    }

    pub(crate) fn take_caches(&mut self) -> Option<Vec<Qwen3_5LayerCache>> {
        self.caches.take()
    }

    pub(crate) fn token_history(&self) -> &[u32] {
        &self.token_history
    }

    pub(crate) fn model_type(&self) -> &str {
        &self.model_type
    }

    pub(crate) fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub(crate) fn image_cache_key(&self) -> Option<u64> {
        self.image_cache_key
    }

    pub(crate) fn rope_deltas(&self) -> Option<i32> {
        self.rope_deltas
    }

    pub(crate) fn model_id(&self) -> u64 {
        self.model_id
    }
}
