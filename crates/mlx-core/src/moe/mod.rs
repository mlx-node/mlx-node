//! Shared Mixture-of-Experts primitives: top-k routing and token dispatch.
pub mod dispatch;
pub mod router;

pub(crate) use dispatch::{gather_sort, scatter_unsort};
pub use router::{RouterConfig, RoutingMode, TopKRouter, topk_from_logits};
