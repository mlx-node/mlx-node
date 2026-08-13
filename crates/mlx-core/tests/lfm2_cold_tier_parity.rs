//! Real-weight restart parity for LFM2's hybrid paged K/V + ShortConv state.
//!
//! A fresh restore instance can report non-zero cached tokens only when both
//! the full-attention block chain and the exact-boundary `ConvState` sidecar
//! survived to disk. The shared harness additionally requires the sidecar
//! install counter to advance and compares deterministic output with a
//! persistence-disabled fresh-prefill baseline.
//!
//! ```shell
//! MLX_COLD_CACHE_DIR=$(mktemp -d) \
//! MLX_TEST_MODEL_PATH=.cache/models/lfm2.5-1.2b-thinking-mlx \
//! cargo test -p mlx-core --release --test lfm2_cold_tier_parity -- \
//!   --ignored --test-threads=1 --nocapture
//! ```

mod cold_tier_parity_harness;

use cold_tier_parity_harness as harness;
use mlx_core::models::lfm2::model::Lfm2Model;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real LFM2/LFM2.5 checkpoint; run with --test-threads=1"]
async fn lfm2_cold_tier_restart_parity() {
    harness::run_restart_parity(
        harness::ColdTierParitySpec::new("lfm2")
            .with_capture_warmup_turns(1)
            .expecting_sidecar_install(),
        |model_dir, messages, config| async move {
            let model = Lfm2Model::load_from_dir(&model_dir.to_string_lossy()).await?;
            model.chat_session_start(messages, Some(config)).await
        },
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_LFM2_MOE_MODEL_PATH pointing to a real LFM2.5 MoE checkpoint"]
async fn lfm2_moe_cold_tier_restart_parity() {
    let mut spec = harness::ColdTierParitySpec::new("lfm2_moe")
        .with_capture_warmup_turns(1)
        .expecting_sidecar_install();
    spec.model_path_env = "MLX_TEST_LFM2_MOE_MODEL_PATH";
    harness::run_restart_parity(spec, |model_dir, messages, config| async move {
        let model = Lfm2Model::load_from_dir(&model_dir.to_string_lossy()).await?;
        model.chat_session_start(messages, Some(config)).await
    })
    .await;
}
