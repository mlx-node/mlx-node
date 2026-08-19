//! Real-weight restart parity for Muse-Glimmer's grouped full + sliding cache.
//!
//! The test is the admission gate for the native/TypeScript cold-tier
//! allowlists: the restart instance must restore a non-zero prefix, install the
//! sliding sidecar, report zero corruptions, and match a persistence-disabled
//! fresh prefill byte for byte.

mod cold_tier_parity_harness;

use cold_tier_parity_harness as harness;
use mlx_core::models::muse_glimmer::MuseGlimmerModel;

const PROMPT: &str = "Explain why a hybrid transformer with full-attention and sliding-window \
    layers cannot safely restore only one cache group from SSD. Describe how a content-addressed \
    full-attention chain, a companion sliding-window sidecar, an exact shared token boundary, \
    model fingerprints, cache salts, checksums, and atomic publication prevent stale or \
    cross-request state from being installed. Keep the response short and deterministic.";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_MUSE_GLIMMER_MODEL_PATH pointing to a converted Muse-Glimmer checkpoint"]
async fn muse_glimmer_cold_tier_restart_parity() {
    let mut spec = harness::ColdTierParitySpec::new("muse-glimmer-grouped")
        .with_pool_memory_mb(2048)
        .with_prompt(PROMPT)
        .with_max_new_tokens(8)
        .expecting_sidecar_install();
    spec.model_path_env = "MLX_TEST_MUSE_GLIMMER_MODEL_PATH";
    harness::run_restart_parity(spec, |model_dir, messages, mut config| async move {
        let owner = String::from("muse-glimmer-cold-tier-parity");
        config.cache_owner_id = Some(owner.clone());
        config.cache_root_owner_id = Some(owner);
        config.enable_mtp = Some(false);
        let model = MuseGlimmerModel::load(model_dir.to_string_lossy().into_owned()).await?;
        model.chat_session_start(messages, Some(config)).await
    })
    .await;
}
