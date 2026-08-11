//! Real-weight restart parity for Gemma4's grouped full + sliding paged cache.
//!
//! The full-attention chain is restored only when the co-keyed sliding-group
//! sidecar validates at the same boundary. The shared harness requires a
//! genuine sidecar install, non-zero SSD hits, zero corruption, and exact
//! deterministic output parity with a persistence-disabled fresh prefill.

mod cold_tier_parity_harness;

use cold_tier_parity_harness as harness;
use mlx_core::models::gemma4::Gemma4Model;

// Long enough to cross the first 64-token grouped checkpoint on every Gemma4
// tokenizer we gate, but deliberately far shorter than the generic 1,200-token
// ladder fixture: this test needs one atomic full+sliding restart boundary, not
// a throughput benchmark.
const GEMMA4_COLD_PROMPT: &str = "Explain how a local inference engine can reuse a block-paged \
    attention prefix after a process restart. Describe why full-attention key and value blocks \
    alone are insufficient for a hybrid model that also has sliding-window layers, why both \
    groups must name the same token boundary, and why a missing or malformed companion should \
    make the engine recompute from the beginning. Then summarize how checksums, model \
    fingerprints, cache salts, atomic file publication, and bounded memory admission keep the \
    restored state isolated from another model, another conversation, or a partially written \
    cache entry. Keep the answer concise and deterministic.";

fn require_deterministic_prefill_shape() {
    assert_eq!(
        std::env::var("MLX_PAGED_PREFILL_CHUNK_SIZE").as_deref(),
        Ok("64"),
        "Gemma4 grouped restart parity must run with \
         MLX_PAGED_PREFILL_CHUNK_SIZE=64 so fresh prefill and the 64-token \
         restore boundary use identical GEMM shapes"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Gemma4 checkpoint"]
async fn gemma4_grouped_cold_tier_restart_parity() {
    require_deterministic_prefill_shape();
    harness::run_restart_parity(
        harness::ColdTierParitySpec::new("gemma4-grouped")
            // E2B needs about 914 MiB for one full-context full-attention
            // lane plus its sliding window. 1 GiB keeps that real geometry
            // while avoiding a 5 GiB test-only pool that can make Metal
            // readback fail on the standard shared runner before SSD capture.
            .with_pool_memory_mb(1024)
            .with_prompt(GEMMA4_COLD_PROMPT)
            .with_max_new_tokens(16)
            .expecting_sidecar_install(),
        |model_dir, messages, config| async move {
            let model = Gemma4Model::load_from_dir(&model_dir.to_string_lossy(), None).await?;
            model.chat_session_start(messages, Some(config)).await
        },
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_GEMMA4_MOE_MODEL_PATH pointing to a real Gemma4 MoE checkpoint"]
async fn gemma4_moe_grouped_cold_tier_restart_parity() {
    require_deterministic_prefill_shape();
    let mut spec = harness::ColdTierParitySpec::new("gemma4-moe-grouped")
        .with_pool_memory_mb(6144)
        .with_prompt(GEMMA4_COLD_PROMPT)
        .with_max_new_tokens(16)
        .expecting_sidecar_install();
    spec.model_path_env = "MLX_TEST_GEMMA4_MOE_MODEL_PATH";
    harness::run_restart_parity(spec, |model_dir, messages, config| async move {
        let model = Gemma4Model::load_from_dir(&model_dir.to_string_lossy(), None).await?;
        model.chat_session_start(messages, Some(config)).await
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_GEMMA4_MOE_MODEL_PATH pointing to a real Gemma4 MoE checkpoint"]
async fn gemma4_moe_paged_fresh_restart_is_deterministic() {
    let Ok(model_path) = std::env::var("MLX_TEST_GEMMA4_MOE_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_GEMMA4_MOE_MODEL_PATH is unset");
        return;
    };
    assert!(
        std::path::Path::new(&model_path).is_dir(),
        "MLX_TEST_GEMMA4_MOE_MODEL_PATH must name a readable checkpoint directory"
    );
    let spec = harness::ColdTierParitySpec::new("gemma4-moe-fresh")
        .with_prompt(GEMMA4_COLD_PROMPT)
        .with_max_new_tokens(16);
    let messages = || vec![harness::user_message(GEMMA4_COLD_PROMPT)];

    let first = {
        let model = Gemma4Model::load_from_dir(&model_path, None)
            .await
            .expect("load first fresh Gemma4 MoE instance");
        model
            .chat_session_start(messages(), Some(harness::parity_chat_config(&spec)))
            .await
            .expect("run first fresh Gemma4 MoE instance")
    };
    let second = {
        let model = Gemma4Model::load_from_dir(&model_path, None)
            .await
            .expect("load second fresh Gemma4 MoE instance");
        model
            .chat_session_start(messages(), Some(harness::parity_chat_config(&spec)))
            .await
            .expect("run second fresh Gemma4 MoE instance")
    };

    assert_eq!(
        first.text, second.text,
        "fresh Gemma4 MoE text changed across model reload"
    );
    assert_eq!(
        first.num_tokens, second.num_tokens,
        "fresh Gemma4 MoE token count changed across model reload"
    );
}
