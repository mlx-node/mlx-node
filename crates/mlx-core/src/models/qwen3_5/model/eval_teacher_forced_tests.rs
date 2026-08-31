//! `mlx eval` over a real (tiny, randomly initialized) dense stack.
//!
//! Dense twin of the MoE `eval_teacher_forced_tests`. The reductions and the
//! driver are covered in `crate::quality`; what only this level can prove is
//! that the dense adapter's chunked prefill and tied head line up position
//! for position with the targets the driver derives from the cached ids.

use super::*;
use crate::models::qwen3_5::config::Qwen3_5Config;
use crate::quality::cache::{EvalCacheMeta, write_meta, write_row};
use crate::quality::runner::{EvalBackend, capture_row, run};
use crate::quality::{EvalOutcome, EvalReport, EvalRequest};

const TOP_K: u32 = 8;
/// Smaller than the scored span, so the head is folded over several chunks
/// on both the capture and the score side.
const LOGIT_CHUNK: u32 = 5;

fn tiny_cfg() -> Qwen3_5Config {
    Qwen3_5Config {
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
        "mlx_eval_dense_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn score_self(inner: &mut Qwen35Inner, dir: &std::path::Path) -> Result<EvalReport> {
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

fn run_dense_self_identity() -> Result<()> {
    let dir = scratch_dir();
    let mut inner = Qwen35Inner::new(tiny_cfg())?;
    let tokens: Vec<u32> = (0u32..19).map(|i| (i * 137 + 41) % 1024).collect();

    std::fs::create_dir_all(&dir)
        .map_err(|e| Error::from_reason(format!("create {}: {e}", dir.display())))?;
    let row = capture_row(&mut inner, &tokens, TOP_K, LOGIT_CHUNK)?;
    write_row(&dir, 0, &row)?;
    write_meta(
        &dir,
        &EvalCacheMeta {
            teacher_path: "tiny-dense".to_string(),
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

    let report = score_self(&mut inner, &dir)?;
    assert_eq!(report.rows, 1);
    assert_eq!(report.positions, tokens.len() as u32 - 1);
    assert_eq!(
        report.mean_kl_topk, 0.0,
        "dense self-KL must be exactly zero, got {}",
        report.mean_kl_topk
    );
    assert_eq!(report.top1_agreement, 1.0);
    assert_eq!(report.mean_nll, report.teacher_mean_nll);
    assert!(
        report.mean_nll.is_finite() && report.mean_nll > 0.0,
        "degenerate logits would make the zero above vacuous, got {}",
        report.mean_nll
    );

    // A perturbed tied head is a different checkpoint on the same ids.
    let weight = inner.embedding.get_weight();
    inner.embedding = Embedding::from_weight(&weight.mul_scalar(1.35)?.add_scalar(0.01)?)?;
    let perturbed = score_self(&mut inner, &dir)?;
    assert!(
        perturbed.mean_kl_topk > 1e-6,
        "a perturbed dense head must diverge, got {}",
        perturbed.mean_kl_topk
    );
    assert!(
        perturbed.top1_agreement < 1.0,
        "agreement that never falls is reading the cache, not the student, got {}",
        perturbed.top1_agreement
    );
    assert_eq!(
        perturbed.teacher_mean_nll, report.teacher_mean_nll,
        "the teacher's own NLL is read from the cache and cannot move"
    );

    std::fs::remove_dir_all(&dir).ok();
    Ok(())
}

#[test]
fn test_dense_scoring_a_checkpoint_against_its_own_capture_is_exactly_zero() {
    if let Err(err) = run_dense_self_identity() {
        let msg = err.reason.to_string();
        if msg.contains("Metal") || msg.contains("device") {
            eprintln!("skipping test_dense_scoring_a_checkpoint_against_its_own_capture: {msg}");
            return;
        }
        panic!("unexpected dense teacher-forced eval failure: {msg}");
    }
}
