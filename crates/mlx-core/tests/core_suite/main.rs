//! Consolidated integration test suite for non-isolated `mlx-core` runtime,
//! paged KV adapter, selector, and synthetic MoE/MTP checks.

#[path = "../common/mod.rs"]
mod common;

#[cfg(target_os = "macos")]
mod adapter_e2e_validation;
mod dflash2_selector_edges;
mod gemma4_image_cold_cache_reopen;
#[cfg(target_os = "macos")]
mod gguf_repack_metallib_names;
#[cfg(target_os = "macos")]
mod inkling2_scratch_release;
mod lazy_block_allocation;
mod paged_adapter_reserve_rollback;
mod paged_image_extra_keys_isolation;
mod paged_pool_multi_block_round_trip;
mod qwen3_5_moe_mtp_synthetic;
