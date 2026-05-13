//! Shared Mixture-of-Experts primitives: top-k routing and token dispatch.
pub mod dispatch;
pub mod router;

pub use router::{RouterConfig, RoutingMode, TopKRouter};
