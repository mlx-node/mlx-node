//! Process-wide toggles for the graph-native paged-KV fast paths, shared by
//! every family that drives `PagedKVCacheAdapter`.

use std::sync::OnceLock;

/// When enabled (default), paged K/V is written into the pool with the
/// graph-native, lazily-scheduled `update_keys_values_native`, so the write
/// feeds the same-step attention read through MLX graph dependencies instead
/// of forcing a per-layer host sync. When disabled, the synchronous
/// `update_keys_values` (a raw Metal write outside the graph scheduler) is
/// used; that path is also taken automatically when the native write errors.
/// Only whole-turn decode has that fallback: every family's batched and ragged
/// decode hard-errors when this is off, so disabling it disables continuous
/// batching.
pub(crate) fn native_kv_write_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        crate::inference_trace::env_flag_enabled_or_default("MLX_PAGED_NATIVE_KV_WRITE", true)
    })
}

/// When enabled (default), paged decode gathers historical K/V with the
/// graph-native `gather_kv_for_decode_graph`, which consumes the lazy pool
/// arrays via graph dependencies (no per-layer host eval). When disabled, the
/// synchronous `gather_kv_for_decode` forces a pending-write eval and reads
/// the pool outside the graph; that path is also taken automatically when the
/// graph gather cannot serve the inputs. LFM2 and Nemotron-H batched decode
/// hard-error when this is off rather than falling back.
pub(crate) fn graph_decode_gather_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        crate::inference_trace::env_flag_enabled_or_default("MLX_PAGED_GRAPH_DECODE_GATHER", true)
    })
}
