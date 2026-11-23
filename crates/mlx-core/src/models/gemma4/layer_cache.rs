//! Gemma4 layer cache aliases backed by [`crate::transformer::hybrid_layer_cache`].

pub use crate::transformer::hybrid_layer_cache::HybridLayerCache as Gemma4LayerCache;
pub(crate) use crate::transformer::hybrid_layer_cache::{
    HybridVerifyRollback as Gemma4VerifyRollback, commit_after_verify, snapshot_before_verify,
};
