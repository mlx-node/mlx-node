//! Consolidated integration test suite for non-isolated `mlx-paged-attn`
//! tests. Binaries that cache process-wide C++ test-probe environment variables
//! (`grouped_gemma4_graph`, `grouped_qwen35_graph`, `paged_int8_read`) remain
//! standalone.

#[cfg(all(target_os = "macos", mlx_node_metal_enabled))]
mod grouped_qwen35_reducer;
#[cfg(all(target_os = "macos", mlx_node_metal_enabled))]
mod layer_kv_pool_array;
#[cfg(all(target_os = "macos", mlx_node_metal_enabled))]
mod paged_attention_budget;
#[cfg(all(target_os = "macos", mlx_node_metal_enabled))]
mod paged_dispatch_stress;
#[cfg(all(target_os = "macos", mlx_node_metal_enabled))]
mod paged_int8_store;
#[cfg(all(target_os = "macos", mlx_node_metal_enabled))]
mod paged_int8_store_ktm;
#[cfg(all(target_os = "macos", mlx_node_metal_enabled))]
mod paged_ops_smoke;
#[cfg(all(target_os = "macos", mlx_node_metal_enabled))]
mod paged_prefill_attention;
#[cfg(all(target_os = "macos", mlx_node_metal_enabled))]
mod paged_q_offset_view;
mod profile_run;
#[cfg(all(target_os = "macos", mlx_node_metal_enabled))]
mod sliding_window_kernel;
#[cfg(all(target_os = "macos", mlx_node_metal_enabled))]
mod varlen_paged_kernel;
