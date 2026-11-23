//! Re-exports the shared Parker-thread prefix disk writer primitives from
//! [`crate::disk_tier::prefix_disk`] for Hy4 callers (`chat.rs`, `tests_persist.rs`).
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

pub use crate::disk_tier::prefix_disk::{Closer, DiskTelemetry as Telemetry};
