//! Cheap tests for the MTP acceptance-gate state on `Qwen35Inner` and
//! for what a gated turn does to its plan (no Metal: `new` defers the
//! paged pool, `reset_caches_sync` on a fresh inner touches no GPU state,
//! and an adapter-less paged turn stops at the preflight).

use super::*;
use crate::engine::plan::{MediaInputs, TurnPlan};
use crate::models::qwen3_5::config::Qwen3_5Config;

fn tiny_cfg() -> Qwen3_5Config {
    Qwen3_5Config {
        qwen35_gguf_gdn_layout: None,
        vocab_size: 1024,
        hidden_size: 64,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 2,
        intermediate_size: 128,
        rms_norm_eps: 1e-6,
        head_dim: 16,
        tie_word_embeddings: true,
        attention_bias: false,
        max_position_embeddings: 1024,
        pad_token_id: 0,
        eos_token_id: 0,
        bos_token_id: 0,
        linear_num_value_heads: 4,
        linear_num_key_heads: 2,
        linear_key_head_dim: 16,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        full_attention_interval: 4,
        partial_rotary_factor: 0.25,
        rope_theta: 100_000.0,
        paged_cache_memory_mb: None,
        paged_cache_initial_memory_mb: None,
        paged_block_size: None,
        use_block_paged_cache: None,
        persist_paged_cache: None,
        n_mtp_layers: 0,
    }
}

/// A minimal real tokenizer so `WholeTurnArgs` can be built without a
/// checkpoint (mirrors `engine::paged_turn`'s fixture).
fn tiny_tokenizer() -> Arc<Qwen3Tokenizer> {
    let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": { "a": 0, "b": 1, "c": 2, "<unk>": 3 },
                "unk_token": "<unk>"
            }
        }"#;
    let dir = std::env::temp_dir().join(format!(
        "mlx-node-qwen35-gate-plan-tok-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("fixture dir: {e}"));
    let path = dir.join("tokenizer.json");
    std::fs::write(&path, json).unwrap_or_else(|e| panic!("fixture write: {e}"));
    let tok = Qwen3Tokenizer::from_file(&path).unwrap_or_else(|e| panic!("fixture tokenizer: {e}"));
    let _ = std::fs::remove_dir_all(&dir);
    Arc::new(tok)
}

/// Dense forks its own paged MTP core, so the acceptance gate runs in
/// `paged_whole_turn` instead of in `admit_paged_speculative_decode` the
/// way MoE's does. A gated turn falls through to the generic paged driver,
/// which dispatches on `plan.decoder` — so the gate has to restate the
/// plan, or the fall-through lands on the erroring default hook and fails
/// the turn it was supposed to decode autoregressively.
///
/// MUTATION: drop the `args.plan.decoder = DecoderPlan::Autoregressive`
/// restatement — the final assertion fails.
#[test]
fn a_gated_paged_mtp_turn_leaves_the_speculative_lane() {
    let mut inner = Qwen35Inner::new(tiny_cfg()).expect("construct");

    // Dense implements no paged speculative core, so the generic driver's
    // speculative branch is a turn FAILURE, not an autoregressive
    // fallback. That is what the restatement below has to avoid.
    let probe = extract_chat_params(&ChatConfig::default());
    let refusal = <Qwen35Inner as PagedBackend>::admit_paged_speculative_decode(&mut inner, &probe)
        .expect_err("dense must inherit the erroring default hook");
    assert!(
        refusal
            .reason
            .contains("implements no paged speculative core"),
        "unexpected refusal: {}",
        refusal.reason
    );

    // 0-of-5 accepted first drafts: confidently below break-even.
    assert!(
        mtp_decode::mtp_accept_gate_blocks(0, 5),
        "fixture counts must block the gate"
    );
    inner.mtp_draft_accepted = 0;
    inner.mtp_draft_attempted = 5;

    let tokenizer = tiny_tokenizer();
    let config = ChatConfig {
        enable_mtp: Some(true),
        max_new_tokens: Some(4),
        ..ChatConfig::default()
    };
    let params = extract_chat_params(&config);
    let tokens = [1u32, 2, 3];
    let mut args = WholeTurnArgs {
        tokens: &tokens,
        tokenizer: &tokenizer,
        eos_id: 0,
        config: &config,
        params: &params,
        thinking: ThinkingSetup {
            enabled: false,
            budget: None,
        },
        plan: TurnPlan {
            is_delta: false,
            input_media: MediaCapabilities::NONE,
            context_media: MediaCapabilities::NONE,
            use_paged_attention: true,
            decoder: DecoderPlan::Speculative(SpeculativeKind::NativeMtp),
        },
        sink: None,
        cancelled: None,
        media: MediaInputs {
            images: &[],
            audio: &[],
        },
    };

    // The adapter-less fixture stops the turn at the paged preflight,
    // which runs AFTER the gate — far enough to observe the restated plan
    // without a GPU.
    let err = match inner.paged_whole_turn(&mut args) {
        Ok(_) => panic!("an adapter-less fixture cannot pass the paged preflight"),
        Err(e) => e,
    };
    assert!(
        err.reason.contains("paged cache is not initialized"),
        "expected the preflight refusal, got: {}",
        err.reason
    );
    assert_eq!(
        args.plan.decoder,
        DecoderPlan::Autoregressive,
        "a gated turn must leave the speculative lane before the generic \
         paged driver reads plan.decoder"
    );
}

#[test]
fn reset_caches_clears_mtp_acceptance_gate_state() {
    // A full session reset must clear the MTP acceptance-gate history
    // so a new independent chat on this model probes instead of
    // inheriting the previous chat's rejection.
    let mut inner = Qwen35Inner::new(tiny_cfg()).expect("construct");
    inner.mtp_draft_accepted = 1;
    inner.mtp_draft_attempted = 4;
    inner.mtp_gated_turns = 2;
    inner.reset_caches_sync().expect("reset");
    assert_eq!(inner.mtp_draft_accepted, 0, "gate history cleared");
    assert_eq!(inner.mtp_draft_attempted, 0, "gate history cleared");
    assert_eq!(inner.mtp_gated_turns, 0, "gated-turn streak cleared");
}

#[test]
fn gate_is_depth_1_scoped() {
    // The 0.6 threshold is depth-1 calibrated; a depth>1 turn is never
    // gated even when the aggregated rate is confidently below
    // break-even (the verify cost vs deeper-slot acceptance economics
    // at depth>1 are not captured by a single threshold).
    let mut inner = Qwen35Inner::new(tiny_cfg()).expect("construct");
    inner.mtp_draft_accepted = 0;
    inner.mtp_draft_attempted = 4; // 0/4: confident below break-even
    assert!(
        inner.mtp_gate_allows(2),
        "depth>1 request must not be gated by the depth-1 threshold"
    );
    assert!(
        !inner.mtp_gate_allows(1),
        "a depth-1 request with confidently-low acceptance must be gated"
    );
}

#[test]
fn gate_does_not_act_on_undersampled_or_marginal_rates() {
    // The gate is confidence-aware: a 2-of-4 rate (upper Wilson 95%
    // bound ~0.82) or a 1-of-4 rate (~0.64) is NOT confidently below
    // break-even — a healthy 0.756 head hits 2-of-4 ~25% of the time.
    let mut inner = Qwen35Inner::new(tiny_cfg()).expect("construct");
    inner.mtp_draft_accepted = 2;
    inner.mtp_draft_attempted = 4;
    assert!(inner.mtp_gate_allows(1), "2-of-4 must not gate");
    inner.mtp_draft_accepted = 1;
    assert!(inner.mtp_gate_allows(1), "1-of-4 must not gate");
    inner.mtp_draft_accepted = 0;
    assert!(
        !inner.mtp_gate_allows(1),
        "0-of-4 must gate (~0.35% for a 0.756 head)"
    );
}

#[test]
fn bounded_history_catches_late_degradation() {
    // A long healthy phase must not drown out a later degradation:
    // with the bounded history, a sustained run of rejected drafts
    // pulls the window rate below break-even and the gate blocks.
    let mut inner = Qwen35Inner::new(tiny_cfg()).expect("construct");
    // 10,000 healthy depth-1 drafts, then a sustained bad streak
    // (175 turns x 0-of-4). The history bound keeps the window finite.
    inner.mtp_draft_accepted = 10_000;
    inner.mtp_draft_attempted = 10_000;
    for _ in 0..175 {
        inner.record_turn_mtp_acceptance(0, 4);
    }
    let attempted = inner.mtp_draft_attempted;
    assert!(
        attempted <= mtp_decode::MTP_ACCEPT_GATE_HISTORY_CAP,
        "history must stay bounded, got {attempted}"
    );
    assert!(
        mtp_decode::mtp_accept_gate_blocks(inner.mtp_draft_accepted, attempted),
        "sustained degradation must be confidently below break-even"
    );
    assert!(
        !inner.mtp_gate_allows(1),
        "the gate must block after a sustained degradation"
    );
}
