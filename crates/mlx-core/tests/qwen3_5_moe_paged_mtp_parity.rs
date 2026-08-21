//! Gated integration gate for the PAGED MoE native-MTP lane (Stage D2).
//!
//! Before D2 the MoE execution plan published `supports_paged_attention:
//! false`, so a paged text turn that asked for MTP silently resolved to plain
//! autoregressive decode — the MTP head loaded and never engaged. These tests
//! pin the lane that replaces that downgrade.
//!
//! Oracle: at T=0 speculative decoding is output-invariant, because every
//! emitted token is re-derived from the MAIN model's own verify forward. So
//! the paged MTP transcript must byte-match BOTH the paged AR transcript on
//! the same checkpoint AND the flat MTP transcript on a `use_block_paged_cache:
//! false` clone of it. Three-way equality separates the two failure classes a
//! two-way comparison confuses: a paged-only bug (paged MTP != paged AR) and a
//! backend-wide one (both paged legs agree with each other but not with flat).
//!
//! A second leg checks the state the transcript cannot see: the GDN recurrent
//! state a paged MTP turn persists is bit-compared against a FRESH recompute
//! over the checkpoint's own token key. A speculative turn that rewound its
//! adapter but not its recurrent state still emits the right tokens this turn
//! and corrupts the next one; only the oracle sees it.
//!
//! Requires a real MoE checkpoint whose MTP head loads AND that runs the
//! block-paged backend (e.g. `.cache/models/qwen3.6-35b-a3b-mxfp8-mtp`). Run:
//!
//! ```shell
//! MLX_TEST_MOE_MTP_MODEL_PATH=/abs/path/to/moe-mtp-checkpoint \
//!     cargo test -p mlx-core --test qwen3_5_moe_paged_mtp_parity \
//!     -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Without the env var every test skips cleanly.

use std::fs;
use std::path::{Path, PathBuf};

use mlx_core::engine::types::{ChatConfig, ChatResult};
use mlx_core::models::qwen3_5_moe::model::Qwen3_5MoeModel;
use mlx_core::tokenizer::ChatMessage;

/// SCREENED fixtures (X8): prompts whose replies carry no bf16 near-tie on
/// this model family, so byte equality measures speculative bookkeeping rather
/// than kernel rounding.
///
/// Screening is not optional here and the cost of skipping it is concrete: an
/// unscreened free-form prompt ("Write a short numbered list of steps to plan a
/// weekend hiking trip") diverges on this checkpoint at the SAME character on
/// the FLAT lane — where MTP is pre-existing code — and even flips between
/// paged AR and flat AR with no speculation involved at all. Every prompt below
/// was screened as paged-MTP == paged-AR AND flat-MTP == flat-AR at depth 1 and
/// depth 4 before being admitted.
///
/// Each also produces a long, fully determined continuation (many cycles: 41,
/// 14 and 28 at depth 1 respectively), so a bookkeeping bug has room to show.
pub const SCREENED_PROMPTS: [&str; 3] = [
    "Count from 1 to 30, space separated.",
    "Write the alphabet in lowercase, space separated.",
    "Write the first 15 Fibonacci numbers, comma separated.",
];

pub const MAX_NEW_TOKENS: i32 = 160;

pub const FOLLOWUP: &str = "Repeat back exactly what you just wrote.";

/// `chat_config` on an ISOLATED cache domain: `salt` doubles as the cache
/// owner, so two legs on one model instance cannot reuse each other's prefix
/// blocks or GDN checkpoints.
pub fn turn_cfg(salt: &str, max_new_tokens: i32, enable_mtp: bool) -> ChatConfig {
    ChatConfig {
        cache_salt: Some(salt.to_string()),
        cache_owner_id: Some(salt.to_string()),
        cache_root_owner_id: None,
        max_new_tokens: Some(max_new_tokens),
        temperature: Some(0.0),
        top_k: None,
        top_p: None,
        min_p: None,
        repetition_penalty: None,
        repetition_context_size: None,
        presence_penalty: None,
        presence_context_size: None,
        frequency_penalty: None,
        frequency_context_size: None,
        max_consecutive_tokens: None,
        max_ngram_repeats: None,
        ngram_size: None,
        tools: None,
        reasoning_effort: Some("none".to_string()),
        thinking_token_budget: Some(0),
        include_reasoning: Some(false),
        report_performance: Some(true),
        reuse_cache: Some(true),
        enable_mtp: Some(enable_mtp),
        mtp_depth: Some(4),
        mtp_adaptive_depth: Some(false),
    }
}

pub fn assistant_message_from_result(result: &ChatResult) -> ChatMessage {
    ChatMessage {
        role: "assistant".to_string(),
        content: result.text.clone(),
        tool_calls: None,
        tool_call_id: None,
        is_error: None,
        reasoning_content: result.thinking.clone(),
        thinking_enabled: Some(result.thinking_enabled),
        images: None,
        audio: None,
    }
}

/// Warm-continue the transcript with pure-AR decode. `cached_tokens > 0` is the
/// anti-vacuity probe: without prefix reuse turn 2 cold-prefills and never
/// reads the state turn 1 left behind.
pub async fn warm_ar_turn2(
    model: &Qwen3_5MoeModel,
    salt: &str,
    transcript: Vec<ChatMessage>,
    context: &str,
) -> ChatResult {
    let r2 = model
        .chat_session_continue(transcript, Some(turn_cfg(salt, 64, false)))
        .await
        .unwrap_or_else(|e| panic!("{context}: warm turn 2 failed: {}", e.reason));
    assert!(
        r2.cached_tokens > 0,
        "{context}: turn 2 must actually warm-continue (cached_tokens=0 means it \
         cold-prefilled and never read turn 1's carried state)"
    );
    r2
}

pub fn user_message(content: &str) -> ChatMessage {
    ChatMessage {
        role: "user".to_string(),
        content: content.to_string(),
        tool_calls: None,
        tool_call_id: None,
        is_error: None,
        reasoning_content: None,
        thinking_enabled: None,
        images: None,
        audio: None,
    }
}

/// Resolve the gated checkpoint, or `None` (with a printed reason) when the
/// environment does not provide one.
pub fn model_path() -> Option<String> {
    let Ok(path) = std::env::var("MLX_TEST_MOE_MTP_MODEL_PATH") else {
        eprintln!(
            "skipping: MLX_TEST_MOE_MTP_MODEL_PATH unset (point it at an ABSOLUTE path to a \
             Qwen3.5-MoE checkpoint that ships an MTP head, e.g. \
             /abs/path/to/qwen3.6-35b-a3b-mxfp8-mtp)"
        );
        return None;
    };
    assert!(
        Path::new(&path).exists(),
        "MLX_TEST_MOE_MTP_MODEL_PATH does not exist: {path}"
    );
    Some(path)
}

/// Clone `src` into the workspace target dir with every weight file SYMLINKED
/// and `config.json` patched to force the FLAT backend. `use_block_paged_cache`
/// is written at BOTH the top level and inside `text_config`, because a VLM
/// export nests the language-model config and the loader reads the merged view.
pub fn flat_clone_model_dir(src: &Path, suffix: &str) -> Result<PathBuf, String> {
    let workspace_target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let manifest = std::env::var("CARGO_MANIFEST_DIR")
                .expect("CARGO_MANIFEST_DIR must be set when running cargo test");
            let mut p = PathBuf::from(manifest);
            p.pop();
            p.pop();
            p.join("target")
        });
    let dst = workspace_target.join(format!("moe-mtp-flat-{}-{suffix}", std::process::id()));
    if dst.exists() {
        let _ = fs::remove_dir_all(&dst);
    }
    fs::create_dir_all(&dst).map_err(|e| format!("create_dir_all({}): {e}", dst.display()))?;

    for entry in fs::read_dir(src).map_err(|e| format!("read_dir({}): {e}", src.display()))? {
        let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
        let from = entry.path();
        if !from.is_file() {
            continue;
        }
        let to = dst.join(entry.file_name());
        if entry.file_name() == "config.json" {
            fs::copy(&from, &to)
                .map_err(|e| format!("copy({} -> {}): {e}", from.display(), to.display()))?;
        } else {
            std::os::unix::fs::symlink(&from, &to)
                .map_err(|e| format!("symlink({} -> {}): {e}", from.display(), to.display()))?;
        }
    }

    let cfg_path = dst.join("config.json");
    let raw = fs::read_to_string(&cfg_path).map_err(|e| format!("read config.json: {e}"))?;
    let mut cfg: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| format!("parse config.json: {e}"))?;
    cfg["use_block_paged_cache"] = serde_json::Value::Bool(false);
    if let Some(text) = cfg.get_mut("text_config") {
        text["use_block_paged_cache"] = serde_json::Value::Bool(false);
    }
    let pretty =
        serde_json::to_string_pretty(&cfg).map_err(|e| format!("serialize config.json: {e}"))?;
    fs::write(&cfg_path, pretty).map_err(|e| format!("write config.json: {e}"))?;
    Ok(dst)
}

/// Best-effort removal of a cloned checkpoint dir, including on panic.
pub struct DirCleanup(pub PathBuf);

impl Drop for DirCleanup {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Three-way T=0 transcript equality: paged MTP == paged AR == flat MTP, over
/// every screened fixture at depth 4.
///
/// The two paged legs share a backend and differ only in the decoder, so their
/// equality isolates the speculative bookkeeping. The flat leg shares the
/// decoder and differs only in the KV backend, so adding it catches a bug that
/// moved BOTH paged legs together (a paged prefill or RoPE-offset regression
/// would do exactly that, and paged-MTP == paged-AR would stay green).
///
/// The MTP legs are also asserted to have actually SPECULATED — an
/// `mtp_cycles > 1` anti-vacuity probe. Without it this test passes trivially
/// on the very downgrade D2 exists to remove: a silent AR fallback matches the
/// AR reference byte-for-byte.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MOE_MTP_MODEL_PATH pointing to a real Qwen3.5-MoE MTP checkpoint"]
async fn paged_mtp_matches_paged_ar_and_flat_mtp() {
    let Some(model_path) = model_path() else {
        return;
    };

    let paged_model = Qwen3_5MoeModel::load(model_path.clone())
        .await
        .expect("failed to load MoE model (paged)");
    assert!(
        paged_model.has_mtp_weights(),
        "checkpoint at {model_path} does not engage the MoE MTP drafter \
         (has_mtp_weights() == false)"
    );
    assert!(
        paged_model.has_block_paged_cache(),
        "this gate targets the PAGED lane, but {model_path} loaded flat"
    );

    let mut paged_mtp_texts = Vec::new();
    for (idx, prompt) in SCREENED_PROMPTS.iter().enumerate() {
        let mtp = paged_model
            .chat_session_start(
                vec![user_message(prompt)],
                Some(turn_cfg(&format!("pm{idx}"), MAX_NEW_TOKENS, true)),
            )
            .await
            .expect("paged MTP chat_session_start failed");
        let perf = mtp
            .performance
            .as_ref()
            .expect("paged MTP performance metrics missing (reportPerformance: true)");
        let cycles = perf.mtp_cycles.unwrap_or(0);
        assert!(
            cycles > 1,
            "ANTI-VACUITY [{prompt}]: the paged turn ran {cycles} MTP cycles — the head \
             loaded but the paged lane decoded autoregressively, which is exactly the \
             downgrade this gate exists to catch and would make every equality below \
             pass for free"
        );
        assert!(
            perf.mtp_mean_accepted_tokens.is_some(),
            "ANTI-VACUITY [{prompt}]: no acceptance metrics were recorded"
        );

        let ar = paged_model
            .chat_session_start(
                vec![user_message(prompt)],
                Some(turn_cfg(&format!("pa{idx}"), MAX_NEW_TOKENS, false)),
            )
            .await
            .expect("paged AR chat_session_start failed");
        assert_eq!(
            ar.performance
                .as_ref()
                .and_then(|p| p.mtp_cycles)
                .unwrap_or(0),
            0,
            "the AR reference must not speculate"
        );
        println!(
            "[{prompt}] paged: cycles={cycles} mean_accepted={:?} tokens={}",
            perf.mtp_mean_accepted_tokens, mtp.num_tokens
        );
        assert_eq!(
            mtp.text, ar.text,
            "[{prompt}] paged MTP diverged from paged AR at T=0 — the speculative \
             bookkeeping (verify rows, cycle commit, GDN tape replay) changed which \
             tokens were emitted"
        );
        assert_eq!(mtp.num_tokens, ar.num_tokens, "[{prompt}]");
        paged_mtp_texts.push(mtp.text);
    }
    drop(paged_model);

    // Flat leg: the SAME weights behind a `use_block_paged_cache: false`
    // config, so only the KV backend differs.
    let flat_dir = flat_clone_model_dir(Path::new(&model_path), "parity")
        .expect("flat clone of the checkpoint failed");
    let _cleanup = DirCleanup(flat_dir.clone());
    let flat_model = Qwen3_5MoeModel::load(flat_dir.to_string_lossy().into_owned())
        .await
        .expect("failed to load MoE model (flat MTP)");
    assert!(
        !flat_model.has_block_paged_cache(),
        "the flat clone still loaded the paged backend — use_block_paged_cache: false did \
         not take, so this leg would silently repeat the paged run"
    );
    for (idx, prompt) in SCREENED_PROMPTS.iter().enumerate() {
        let flat_mtp = flat_model
            .chat_session_start(
                vec![user_message(prompt)],
                Some(turn_cfg(&format!("fm{idx}"), MAX_NEW_TOKENS, true)),
            )
            .await
            .expect("flat MTP chat_session_start failed");
        let flat_cycles = flat_mtp
            .performance
            .as_ref()
            .and_then(|p| p.mtp_cycles)
            .unwrap_or(0);
        assert!(
            flat_cycles > 1,
            "ANTI-VACUITY [{prompt}]: the flat leg ran {flat_cycles} MTP cycles"
        );
        assert_eq!(
            paged_mtp_texts[idx], flat_mtp.text,
            "[{prompt}] the paged and flat MTP lanes disagree at T=0 — both paged legs \
             may have moved together (a paged prefill / RoPE-offset regression), which \
             paged-MTP == paged-AR alone cannot see"
        );
    }
}

/// Carried-state parity: the cache + GDN state a paged MTP turn leaves behind
/// must be indistinguishable from what the SAME transcript decoded pure-AR
/// leaves behind, as seen by an identical warm-continue turn 2.
///
/// This is the assertion the turn-1 transcript comparison cannot make. A
/// speculative turn that rolls its adapter back by the rejected drafts but
/// leaves the recurrent state one cycle ahead emits a correct transcript THIS
/// turn and poisons the next warm continuation; every length-only probe
/// (`history_len`, the stepper frontier) agrees in that state too.
///
/// Turn 2 is decoded AR on BOTH legs and both legs' turn-1 state was built by
/// a decode path, so the comparison is immune to the chunked-prefill-vs-
/// per-token GDN kernel difference that makes a warm-vs-cold byte oracle
/// unsound on quantized checkpoints — see
/// `strict_gdn_state_oracle_is_calibrated_against_a_pure_ar_turn` for the
/// measurement of that difference on this checkpoint.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MOE_MTP_MODEL_PATH pointing to a real Qwen3.5-MoE MTP checkpoint"]
async fn paged_mtp_carries_state_a_warm_ar_continue_cannot_tell_from_ar() {
    let Some(model_path) = model_path() else {
        return;
    };
    let model = Qwen3_5MoeModel::load(model_path.clone())
        .await
        .expect("failed to load MoE model");
    if !model.has_mtp_weights() || !model.has_block_paged_cache() {
        eprintln!("skipping: {model_path} is not a paged MoE MTP checkpoint");
        return;
    }

    let mtp_r1 = model
        .chat_session_start(
            vec![user_message(SCREENED_PROMPTS[0])],
            Some(turn_cfg("state-mtp", MAX_NEW_TOKENS, true)),
        )
        .await
        .expect("paged MTP turn 1 failed");
    let cycles = mtp_r1
        .performance
        .as_ref()
        .and_then(|p| p.mtp_cycles)
        .unwrap_or(0);
    assert!(
        cycles > 1,
        "ANTI-VACUITY: {cycles} MTP cycles — turn 1 decoded autoregressively, so the \
         comparison below would compare AR state against AR state"
    );
    let state = model
        .mtp_paged_gdn_state_for_test()
        .await
        .expect("paged GDN state probe failed");
    assert!(
        state.paged_active,
        "the paged lane must be the one under test"
    );
    assert!(
        !state.state_dirty,
        "the epilogue armed the refuse-to-persist latch: the adapter and the drop-last \
         history disagreed on the frontier after a paged MTP turn"
    );
    assert_eq!(
        state.gdn_invalidations, 0,
        "a paged MTP turn must not invalidate recurrent state"
    );
    assert!(
        state.has_history_checkpoint,
        "a reuse_cache paged turn must publish a GDN history checkpoint"
    );

    let mtp_r2 = warm_ar_turn2(
        &model,
        "state-mtp",
        vec![
            user_message(SCREENED_PROMPTS[0]),
            assistant_message_from_result(&mtp_r1),
            user_message(FOLLOWUP),
        ],
        "MTP leg",
    )
    .await;

    let ar_r1 = model
        .chat_session_start(
            vec![user_message(SCREENED_PROMPTS[0])],
            Some(turn_cfg("state-ar", MAX_NEW_TOKENS, false)),
        )
        .await
        .expect("paged AR turn 1 failed");
    assert_eq!(
        mtp_r1.text, ar_r1.text,
        "MTP turn 1 must byte-match the AR turn (T=0 output invariance) — without it \
         the turn-2 state comparison is not over the same transcript"
    );
    let ar_r2 = warm_ar_turn2(
        &model,
        "state-ar",
        vec![
            user_message(SCREENED_PROMPTS[0]),
            assistant_message_from_result(&ar_r1),
            user_message(FOLLOWUP),
        ],
        "AR leg",
    )
    .await;

    assert_eq!(
        mtp_r2.text, ar_r2.text,
        "the warm continue after a paged MTP turn diverged from the warm continue after \
         the AR turn over the SAME transcript at T=0 — the speculative turn left the \
         carried cache/GDN state at a different point than AR did.\nmtp={:?}\nar ={:?}",
        mtp_r2.text, ar_r2.text
    );
}

/// Calibration + report for the STRICT bit-level GDN state oracle.
///
/// The oracle recomputes the recurrent state over the persisted history's own
/// token key through the CHUNKED prefill kernels, while the live turn built its
/// decode suffix through the per-token kernel. Those two paths are not
/// guaranteed bit-identical on a quantized checkpoint, so the oracle is only
/// meaningful where a pure-AR turn — which performs no speculative rewind at
/// all — already passes it.
///
/// This test measures that control and then, only if the control passes,
/// applies the oracle to a speculative turn. A failing control is REPORTED, not
/// asserted: it is a property of the checkpoint's kernels, not of the
/// speculative bookkeeping, and the behavioural state check above covers the
/// same ground without the artifact.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MOE_MTP_MODEL_PATH pointing to a real Qwen3.5-MoE MTP checkpoint"]
async fn strict_gdn_state_oracle_is_calibrated_against_a_pure_ar_turn() {
    let Some(model_path) = model_path() else {
        return;
    };
    let model = Qwen3_5MoeModel::load(model_path.clone())
        .await
        .expect("failed to load MoE model");
    if !model.has_mtp_weights() || !model.has_block_paged_cache() {
        eprintln!("skipping: {model_path} is not a paged MoE MTP checkpoint");
        return;
    }

    model
        .chat_session_start(
            vec![user_message(SCREENED_PROMPTS[0])],
            Some(turn_cfg("oracle-cal", 96, false)),
        )
        .await
        .expect("oracle calibration (AR) turn failed");
    let control = model
        .gdn_history_checkpoint_oracle_for_test()
        .await
        .expect("AR-control state oracle failed to run");
    println!("strict GDN state oracle, pure-AR control: {control}");
    if !control {
        eprintln!(
            "strict GDN state oracle NOT applicable on this checkpoint: a pure-AR paged \
             turn — which performs no speculative rewind — already diverges from a \
             chunked-prefill recompute over its own token key. The difference is the \
             prefill-vs-decode GDN kernel pair, not the speculative bookkeeping; \
             `paged_mtp_carries_state_a_warm_ar_continue_cannot_tell_from_ar` covers the \
             same ground without the artifact."
        );
        return;
    }

    let mtp = model
        .chat_session_start(
            vec![user_message(SCREENED_PROMPTS[0])],
            Some(turn_cfg("oracle-mtp", 96, true)),
        )
        .await
        .expect("oracle MTP turn failed");
    let cycles = mtp
        .performance
        .as_ref()
        .and_then(|p| p.mtp_cycles)
        .unwrap_or(0);
    assert!(cycles > 1, "ANTI-VACUITY: {cycles} MTP cycles");
    assert!(
        model
            .gdn_history_checkpoint_oracle_for_test()
            .await
            .expect("MTP state oracle failed to run"),
        "the control passed but the GDN state persisted after a paged MTP turn does NOT \
         equal a fresh recompute over its own token key — the speculative rewind left \
         the recurrent side at a different frontier than the transcript it is keyed on"
    );
}
