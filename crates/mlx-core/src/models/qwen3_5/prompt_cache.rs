use napi_derive::napi;

use super::layer_cache::Qwen3_5LayerCache;

/// Opaque handle to KV cache state from a previous chat() call.
///
/// Pass this back to the next chat() call via `model.setCache(cache)`
/// to enable incremental prefill — only new tokens since the last turn
/// are processed, avoiding redundant computation.
///
/// Created internally by the model when `reuseCache: true` (default).
/// Extract via `model.takeCache()`, restore via `model.setCache(cache)`.
#[napi]
pub struct PromptCache {
    /// Per-layer KV cache states
    pub(crate) caches: Option<Vec<Qwen3_5LayerCache>>,
    /// Full token sequence that produced this cache state
    pub(crate) token_history: Vec<u32>,
    /// Model type identifier for validation
    pub(crate) model_type: String,
    /// Image cache key for VLM cache reuse (None for text-only)
    pub(crate) image_cache_key: Option<u64>,
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
    }
}

impl PromptCache {
    /// Create a new PromptCache with the given state.
    pub(crate) fn new(
        caches: Vec<Qwen3_5LayerCache>,
        token_history: Vec<u32>,
        model_type: &str,
    ) -> Self {
        Self {
            caches: Some(caches),
            token_history,
            model_type: model_type.to_string(),
            image_cache_key: None,
        }
    }

    /// Take ownership of the caches, leaving this cache empty.
    pub(crate) fn take_caches(&mut self) -> Option<Vec<Qwen3_5LayerCache>> {
        self.caches.take()
    }

    /// Get a reference to the token history.
    pub(crate) fn token_history(&self) -> &[u32] {
        &self.token_history
    }

    /// Get the model type.
    pub(crate) fn model_type(&self) -> &str {
        &self.model_type
    }
}
