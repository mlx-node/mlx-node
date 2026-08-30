//! `mlx eval` over a real (tiny, randomly initialized) MoE stack.
//!
//! What only this level can prove is that the MoE adapter feeds the scoring
//! reductions the right thing — that its prefill, its `fa_idx`-threaded layer
//! stack and its tied head line up position for
//! position with the targets the driver derives from the cached ids.

use super::*;
use crate::models::qwen3_5_moe::config::Qwen3_5MoeConfig;
use crate::quality::cache::{EvalCacheMeta, write_meta, write_row};
use crate::quality::runner::{EvalBackend, capture_row, run};
use crate::quality::{EvalOutcome, EvalReport, EvalRequest};

const TOP_K: u32 = 8;
/// Smaller than the scored span, so the head is folded over several chunks
/// on both the capture and the score side.
const LOGIT_CHUNK: u32 = 5;

fn tiny_moe_cfg() -> Qwen3_5MoeConfig {
    Qwen3_5MoeConfig {
        vocab_size: 1024,
        hidden_size: 64,
        num_layers: 8,
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
        num_experts: 4,
        num_experts_per_tok: 2,
        decoder_sparse_step: 1,
        shared_expert_intermediate_size: None,
        moe_intermediate_size: None,
        norm_topk_prob: true,
        mlp_only_layers: None,
        paged_cache_memory_mb: Some(64),
        paged_block_size: Some(16),
        use_block_paged_cache: None,
        persist_paged_cache: None,
        n_mtp_layers: 0,
        qwen35_gguf_gdn_layout: None,
        paged_cache_initial_memory_mb: None,
    }
}

fn scratch_dir() -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "mlx_eval_moe_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn fixture_tokens() -> Vec<u32> {
    (0u32..19).map(|i| (i * 137 + 41) % 1024).collect()
}

/// Write a one-row teacher cache captured from `inner` itself.
fn capture_self(inner: &mut Qwen35MoeInner, dir: &std::path::Path) -> Result<Vec<u32>> {
    let tokens = fixture_tokens();
    std::fs::create_dir_all(dir)
        .map_err(|e| Error::from_reason(format!("create {}: {e}", dir.display())))?;
    let row = capture_row(inner, &tokens, TOP_K, LOGIT_CHUNK)?;
    write_row(dir, 0, &row)?;
    write_meta(
        dir,
        &EvalCacheMeta {
            teacher_path: "tiny-moe".to_string(),
            identity: crate::quality::cache::EvalIdentity::fixture("a"),
            generation: crate::quality::cache::new_generation(),
            teacher_quantized: false,
            vocab_size: inner.vocab_size(),
            seq_len: tokens.len() as u32,
            top_k: TOP_K,
            rows: 1,
            positions: tokens.len() as u64 - 1,
        },
    )?;
    Ok(tokens)
}

fn score_self(inner: &mut Qwen35MoeInner, dir: &std::path::Path) -> Result<EvalReport> {
    match run(
        inner,
        EvalRequest::Score {
            cache_dir: dir.to_path_buf(),
            identity: crate::quality::cache::EvalIdentity::fixture("a"),
            logit_chunk: LOGIT_CHUNK,
        },
    )? {
        EvalOutcome::Scored(report) => Ok(report),
        EvalOutcome::Captured { .. } => Err(Error::from_reason(
            "score returned a capture count — dispatch is inconsistent",
        )),
    }
}

/// Replace the tied lookup/head table so the model is a DIFFERENT
/// checkpoint answering the same token ids.
fn perturb_tied_head(inner: &mut Qwen35MoeInner) -> Result<()> {
    let weight = inner.embedding.weight();
    inner.embedding = Embedding::from_weight(&weight.mul_scalar(1.35)?.add_scalar(0.01)?)?;
    Ok(())
}

fn run_moe_self_identity() -> Result<()> {
    let dir = scratch_dir();
    let mut inner = Qwen35MoeInner::new(tiny_moe_cfg())?;
    let tokens = capture_self(&mut inner, &dir)?;

    let report = score_self(&mut inner, &dir)?;
    assert_eq!(report.rows, 1);
    assert_eq!(report.positions, tokens.len() as u32 - 1);
    assert_eq!(
        report.mean_kl_topk, 0.0,
        "MoE self-KL must be exactly zero, got {}",
        report.mean_kl_topk
    );
    assert_eq!(
        report.top1_agreement, 1.0,
        "a checkpoint must agree with itself at every position"
    );
    assert_eq!(
        report.mean_nll, report.teacher_mean_nll,
        "student NLL must equal the cached teacher NLL bit for bit"
    );
    assert!(
        report.mean_nll.is_finite() && report.mean_nll > 0.0,
        "degenerate MoE logits would make the zero above vacuous, got {}",
        report.mean_nll
    );

    // A perturbed head is a different checkpoint on the same ids: if the
    // score path ignored the student and echoed the cache, this would still
    // report zero.
    perturb_tied_head(&mut inner)?;
    let perturbed = score_self(&mut inner, &dir)?;
    assert_eq!(perturbed.positions, report.positions);
    assert!(
        perturbed.mean_kl_topk > 1e-6,
        "a perturbed MoE head must diverge, got {}",
        perturbed.mean_kl_topk
    );
    assert!(
        perturbed.top1_agreement < 1.0,
        "agreement that never falls is reading the cache, not the student, got {}",
        perturbed.top1_agreement
    );
    assert_ne!(perturbed.mean_nll, perturbed.teacher_mean_nll);
    assert_eq!(
        perturbed.teacher_mean_nll, report.teacher_mean_nll,
        "the teacher's own NLL is read from the cache and cannot move"
    );

    std::fs::remove_dir_all(&dir).ok();
    Ok(())
}

#[test]
fn test_moe_scoring_a_checkpoint_against_its_own_capture_is_exactly_zero() {
    if let Err(err) = run_moe_self_identity() {
        let msg = err.reason.to_string();
        if msg.contains("Metal") || msg.contains("device") {
            eprintln!("skipping test_moe_scoring_a_checkpoint_against_its_own_capture: {msg}");
            return;
        }
        panic!("unexpected MoE teacher-forced eval failure: {msg}");
    }
}
