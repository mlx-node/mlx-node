//! Shared Mixture-of-Experts primitives: top-k routing, token dispatch, and switch projections.
pub mod dispatch;
pub mod router;
pub mod switch;

pub(crate) use dispatch::{gather_sort, scatter_unsort};
pub use router::{
    RouterConfig, RoutingMode, TopKRouter, sigmoid_biased_topk_from_logits, topk_from_logits,
};
pub use switch::{SwitchGLU, SwitchLinear, SwitchProj};
