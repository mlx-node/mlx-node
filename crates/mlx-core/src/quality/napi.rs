//! NAPI surface for the `mlx eval` CLI.
//!
//! Two exports, both all-in-native one-shots that load the model, dispatch one
//! model-thread command and return: [`capture_teacher_logits`] runs the
//! teacher and writes the reference cache; [`score_against_teacher`] teacher-
//! forces a candidate over the cached token ids and returns its report.
//!
//! `qwen3_5` DENSE and `qwen3_5_moe`, dispatched on `model_type` exactly as
//! `mlx calibrate` does. Both arms hand the same [`EvalRequest`] to the same
//! family-neutral driver; a family costs an [`EvalBackend`] adapter, so the
//! remaining ones are deferred rather than half-wired.
//!
//! [`EvalBackend`]: super::runner::EvalBackend

use std::path::PathBuf;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::calibration::napi::read_model_type;
use crate::models::qwen3_5::model::Qwen35Cmd;
use crate::models::qwen3_5_moe::model::Qwen35MoeCmd;

use super::scoring::EvalReport;
use super::{EvalOutcome, EvalRequest};

/// Process-wide eval mutual-exclusion guard.
///
/// An eval run holds a whole checkpoint resident and, in a capture/score
/// sequence, two of them back to back. `try_lock` so a second caller fails fast
/// rather than queueing another multi-gigabyte load behind the first. Mirrors
/// the `calib_guard` pattern.
fn eval_guard() -> &'static tokio::sync::Mutex<()> {
    static EVAL_GUARD: std::sync::OnceLock<tokio::sync::Mutex<()>> = std::sync::OnceLock::new();
    EVAL_GUARD.get_or_init(|| tokio::sync::Mutex::new(()))
}

/// Pick the loader and model-thread command by `model_type`, then run one eval
/// request on that model's thread. Both families carry the same request type.
async fn run_on_model(model_path: &str, request: EvalRequest) -> Result<EvalOutcome> {
    match read_model_type(model_path)?.as_str() {
        "qwen3_5" => {
            let model =
                crate::models::qwen3_5::persistence::load_with_thread(model_path, None).await?;
            crate::model_thread::send_and_await(&model.thread, |reply| {
                Qwen35Cmd::EvalTeacherForced { request, reply }
            })
            .await
        }
        "qwen3_5_moe" => {
            let model =
                crate::models::qwen3_5_moe::persistence::load_with_thread(model_path).await?;
            crate::model_thread::send_and_await(&model.thread, |reply| {
                Qwen35MoeCmd::EvalTeacherForced { request, reply }
            })
            .await
        }
        other => Err(Error::from_reason(format!(
            "mlx eval supports qwen3_5 / qwen3_5_moe only, got model_type \"{other}\""
        ))),
    }
}

/// Read `<model_path>/config.json` and report whether it declares a
/// quantization block.
///
/// Uses the loader's own alias resolution, so a checkpoint carrying both
/// `quantization` and `quantization_config` is read the same way here as when
/// it is loaded, and a malformed pair fails rather than being guessed at.
fn teacher_is_quantized(model_path: &str) -> Result<bool> {
    let config_path = std::path::Path::new(model_path).join("config.json");
    let data = std::fs::read_to_string(&config_path)
        .map_err(|e| Error::from_reason(format!("read {}: {e}", config_path.display())))?;
    let config: serde_json::Value = serde_json::from_str(&data)
        .map_err(|e| Error::from_reason(format!("parse {}: {e}", config_path.display())))?;
    Ok(crate::models::quant_dispatch::select_quantization_block(&config)?.is_some())
}

/// Run the teacher over `texts` and write its top-`top_k` next-token
/// distribution into `cache_dir`. Returns the number of rows written.
///
/// Each row is tokenized RAW (no chat template, no BOS), truncated to `seq_len`
/// tokens, and prefilled on fresh caches. The head is projected over positions
/// in `logit_chunk`-sized chunks so the full-vocabulary logits never
/// materialize for the whole sequence at once.
///
/// `seq_len` is raised to 2 when lower: the first token primes the forward and
/// has no target of its own, so a shorter row scores nothing. `top_k` is
/// clamped to the teacher's vocabulary, so a wider request degrades to an exact
/// full-vocab KL. The cache records both EFFECTIVE values, not the requested
/// ones — the support width is what says how far a top-K KL can be trusted.
///
/// A quantized teacher is accepted — anchoring on a released quantized
/// checkpoint is a real comparison — but it is warned about and recorded in the
/// cache, because every number then measures divergence from that checkpoint
/// rather than from the bf16 model.
#[napi]
pub async fn capture_teacher_logits(
    teacher_path: String,
    texts: Vec<String>,
    seq_len: u32,
    top_k: u32,
    logit_chunk: u32,
    cache_dir: String,
) -> Result<u32> {
    let _lock = eval_guard()
        .try_lock()
        .map_err(|_| Error::from_reason("another eval run is in progress"))?;

    let identity = crate::quality::cache::EvalIdentity::read(&teacher_path)?;
    let teacher_quantized = teacher_is_quantized(&teacher_path)?;
    if teacher_quantized {
        tracing::warn!(
            teacher = %teacher_path,
            "teacher checkpoint is quantized: every score against this cache measures \
             divergence from it, not from the bf16 model"
        );
    }

    let outcome = run_on_model(
        &teacher_path,
        EvalRequest::Capture {
            teacher_path: teacher_path.clone(),
            teacher_quantized,
            identity,
            texts,
            seq_len,
            top_k,
            logit_chunk,
            cache_dir: PathBuf::from(cache_dir),
        },
    )
    .await?;

    match outcome {
        EvalOutcome::Captured { rows, .. } => Ok(rows),
        EvalOutcome::Scored(_) => Err(Error::from_reason(
            "eval capture returned a score report — model-thread dispatch is inconsistent",
        )),
    }
}

/// Teacher-force `model_path` over the token ids cached in `cache_dir` and
/// report its NLL, perplexity, top-1 agreement and KL against the teacher.
///
/// The candidate is refused when it cannot answer for the cached rows: a
/// different `model_type`, a different tokenizer, or a different vocabulary
/// width. Score reads its token ids FROM THE CACHE, so a tokenizer mismatch
/// would otherwise report a finite, plausible number measured on the wrong
/// text.
#[napi]
pub async fn score_against_teacher(
    model_path: String,
    cache_dir: String,
    logit_chunk: u32,
) -> Result<EvalReport> {
    let _lock = eval_guard()
        .try_lock()
        .map_err(|_| Error::from_reason("another eval run is in progress"))?;

    let identity = crate::quality::cache::EvalIdentity::read(&model_path)?;

    let outcome = run_on_model(
        &model_path,
        EvalRequest::Score {
            cache_dir: PathBuf::from(cache_dir),
            logit_chunk,
            identity,
        },
    )
    .await?;

    match outcome {
        EvalOutcome::Scored(report) => Ok(report),
        EvalOutcome::Captured { .. } => Err(Error::from_reason(
            "eval score returned a capture count — model-thread dispatch is inconsistent",
        )),
    }
}
