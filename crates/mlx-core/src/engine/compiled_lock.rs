//! Process-wide unique model-id assignment for Qwen3.5 dense/MoE instances.
//!
//! Holds a single family-neutral counter so dense and MoE share one id space.

use std::sync::atomic::AtomicU64;

/// Monotonically incrementing source of per-instance model IDs for Qwen3.5
/// dense and MoE. Each `Inner::new` claims one id via `fetch_add`. The id is
/// stored on the inner struct and surfaced as a process-unique instance tag;
/// nothing currently keys behavior off it, so it is effectively a stable
/// per-instance identifier rather than a routing key. Shared between dense and
/// MoE so their ids never overlap.
pub(crate) static QWEN35_MODEL_ID_COUNTER: AtomicU64 = AtomicU64::new(1); // 0 = no model
