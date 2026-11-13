//! Real-weights restart gate for the Qwen3.5 state tier on a load without a
//! draft: a conversation resumes after a restart without prefilling what
//! the first process stored (Splash `publishReachedStateBoundaries` →
//! `WriteBehind` → `closePersistentCache` → `Cache::adopt` →
//! `acquireDeepest` → `promotedCopy`), on the default product path and on a
//! plain autoregressive one.
//!
//! The default path is what `mlx serve` runs through `ChatSession` on an
//! MTP-head checkpoint (every official Qwen3.5 one, the CI checkpoint
//! included): a streamed turn with `enableMtp` on, a native-MTP turn. The
//! plain case turns MTP off and runs the non-streaming entry.
//!
//! Four loads of one checkpoint per case, one at a time; each restart's
//! first turn is the one the conversation resumes with:
//!
//! | # | tier | prompts | role |
//! |---|------|---------|------|
//! | C | off  | P2 | cold reference |
//! | A | persistent | P1 | publishes the checkpoint at 4096 and the replay point, closes cleanly |
//! | B | persistent, same root | P2 | takes the states back, resumes P2 at A's replay point, closes cleanly |
//! | D | persistent, same root | P3 | a shorter prompt over the same document: resumes at A's checkpoint |
//!
//! `MLX_E2E_EXPECT_LANE_DISK=1` (the CI leg) asserts the refused-capture
//! path: on the 7 GB VM the host refuses every optional claim, so A's
//! checkpoint goes from the lane to disk, and P3 resumes at it after the
//! restarts. Without it P3 resumes at the checkpoint or prefills cold, and
//! the run prints which.
//!
//! The image case runs a conversation over `examples/ocr.png` (every
//! Qwen3.5 checkpoint loads its vision tower): after the restart the same
//! conversation over a second image of the same size, whose placeholder ids
//! are the first one's, restores nothing, and the one over the first image
//! resumes at A's replay point and decodes as a cold run does.
//!
//! ```shell
//! MLX_TEST_MODEL_PATH=/path/to/qwen3.5-0.8b-mlx-bf16 \
//!     cargo test -p mlx-core --test qwen3_5_state_tier_restart -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mlx_core::engine::types::ChatConfig;
use mlx_core::engine::{MemoryGovernorStatus, memory_governor_status};
use mlx_core::models::qwen3_5::model::{Qwen3_5Model, Qwen35LoadOptions};
use mlx_core::tokenizer::ChatMessage;

const FIXED_ANSWER: &str = "The access code is BLUE-HERON-4471.";
const BLOCK: u32 = 16;
const CHECKPOINT: u32 = 4096;

fn image_message(content: &str, image: &[u8]) -> ChatMessage {
    ChatMessage {
        images: Some(vec![napi::bindgen_prelude::Uint8Array::new(image.to_vec())]),
        ..message("user", content)
    }
}

/// `examples/ocr.png`, and a second PNG of its size with every pixel
/// inverted: the same placeholder ids, another image.
fn images() -> (Vec<u8>, Vec<u8>) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/ocr.png");
    let first = std::fs::read(&path).expect("examples/ocr.png");
    let mut other = image::load_from_memory(&first)
        .expect("decode the image")
        .to_rgb8();
    image::imageops::invert(&mut other);
    let mut encoded = std::io::Cursor::new(Vec::new());
    other
        .write_to(&mut encoded, image::ImageFormat::Png)
        .expect("encode the other image");
    (first, encoded.into_inner())
}

fn message(role: &str, content: &str) -> ChatMessage {
    ChatMessage {
        role: role.to_string(),
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

/// The decode a case's turns run.
#[derive(Clone, Copy)]
enum Decode {
    /// `ChatSession`'s default on an MTP-head checkpoint: a streamed turn
    /// with `enableMtp` on.
    Mtp,
    /// `enableMtp` off, through `chatSessionStart`.
    Autoregressive,
}

/// What a turn's result (or a stream's terminal chunk) reports.
struct Turn {
    raw_text: String,
    num_tokens: u32,
    prompt_tokens: u32,
    cached_tokens: u32,
    mtp_cycles: u32,
}

fn config(decode: Decode) -> ChatConfig {
    ChatConfig {
        max_new_tokens: Some(24),
        temperature: Some(0.0),
        repetition_penalty: Some(1.0),
        presence_penalty: Some(0.0),
        frequency_penalty: Some(0.0),
        include_reasoning: Some(true),
        report_performance: Some(true),
        reuse_cache: Some(true),
        enable_mtp: Some(matches!(decode, Decode::Mtp)),
        ..ChatConfig::default()
    }
}

/// The document's numbered paragraphs, the access code planted near the
/// start.
fn paragraphs() -> Vec<String> {
    let colors = [
        "amber", "slate", "cobalt", "ivory", "russet", "teal", "ochre",
    ];
    let things = [
        "lantern", "harbor", "orchard", "bridge", "archive", "meadow",
    ];
    (0..180)
        .map(|i| {
            if i == 3 {
                return format!("Paragraph {i}. {FIXED_ANSWER} Keep it secret.");
            }
            format!(
                "Paragraph {i}. The {} {} near station {} logged reading {} on day {}, and \
                 the keeper noted it in ledger {}.",
                colors[i % colors.len()],
                things[(i * 5) % things.len()],
                (i * 37) % 1000,
                (i * 91) % 977,
                i % 28 + 1,
                (i * 13) % 101
            )
        })
        .collect()
}

fn document(count: usize) -> String {
    paragraphs()[..count].join("\n")
}

/// The source checkpoint; `None` when `MLX_TEST_MODEL_PATH` is unset.
fn source() -> Option<PathBuf> {
    let Ok(path) = std::env::var("MLX_TEST_MODEL_PATH") else {
        eprintln!("skipping qwen3_5_state_tier_restart: MLX_TEST_MODEL_PATH unset");
        return None;
    };
    let path = PathBuf::from(path.trim());
    assert!(
        path.is_dir(),
        "MLX_TEST_MODEL_PATH is set but is not a directory: {}",
        path.display()
    );
    Some(path)
}

async fn load(path: &std::path::Path, options: Qwen35LoadOptions) -> Qwen3_5Model {
    let model = Qwen3_5Model::load(path.to_string_lossy().to_string(), Some(options))
        .await
        .expect("load the checkpoint");
    assert!(model.has_block_paged_cache(), "the load builds a pool");
    assert!(
        model.has_mtp_weights(),
        "the gate's checkpoint carries MTP heads, so ChatSession turns MTP on"
    );
    model
}

/// One turn. A plain turn runs no MTP cycle; an MTP turn that `drafts`
/// (a model's first turn, before its acceptance gate has history) must
/// have run draft/verify cycles. Later MTP turns may be gated to plain
/// decode by the product's acceptance gate.
async fn turn(
    model: &Qwen3_5Model,
    messages: Vec<ChatMessage>,
    decode: Decode,
    drafts: bool,
) -> Turn {
    let result = match decode {
        Decode::Mtp => {
            let (_handle, mut rx) = model
                .chat_stream_session_start_for_test(messages, Some(config(decode)))
                .expect("a streamed turn");
            let mut last = None;
            while let Some(chunk) = rx.recv().await {
                let chunk = chunk.expect("a stream chunk");
                if chunk.done {
                    last = Some(chunk);
                    break;
                }
            }
            let last = last.expect("the stream ended with a terminal chunk");
            Turn {
                raw_text: last.raw_text.unwrap_or_default(),
                num_tokens: last.num_tokens.unwrap_or(0),
                prompt_tokens: last.prompt_tokens.unwrap_or(0),
                cached_tokens: last.cached_tokens.unwrap_or(0),
                mtp_cycles: last
                    .performance
                    .and_then(|metrics| metrics.mtp_cycles)
                    .unwrap_or(0),
            }
        }
        Decode::Autoregressive => {
            let result = model
                .chat_session_start(messages, Some(config(decode)))
                .await
                .expect("a turn");
            Turn {
                raw_text: result.raw_text,
                num_tokens: result.num_tokens,
                prompt_tokens: result.prompt_tokens,
                cached_tokens: result.cached_tokens,
                mtp_cycles: result
                    .performance
                    .and_then(|metrics| metrics.mtp_cycles)
                    .unwrap_or(0),
            }
        }
    };
    match decode {
        Decode::Mtp if drafts => assert!(
            result.mtp_cycles >= 1,
            "an MTP turn ran no MTP cycle: cached {} of {}",
            result.cached_tokens,
            result.prompt_tokens
        ),
        Decode::Mtp => {}
        Decode::Autoregressive => {
            assert_eq!(result.mtp_cycles, 0, "a plain turn ran MTP cycles")
        }
    }
    assert!(result.num_tokens > 0, "the turn decoded");
    result
}

/// The governor's status once `ready` holds: the model thread publishes its
/// states on a pass at most every `MONITOR_PERIOD_SECONDS` (0.5 s) and
/// sleeps while idle, so each try waits past the period and wakes it.
async fn status_once(
    model: &Qwen3_5Model,
    ready: impl Fn(&MemoryGovernorStatus) -> bool,
) -> MemoryGovernorStatus {
    for _ in 0..20 {
        std::thread::sleep(std::time::Duration::from_millis(600));
        model
            .mtp_paged_gdn_state_for_test()
            .await
            .expect("wake the model thread");
        let status = memory_governor_status().expect("the model is registered");
        if ready(&status) {
            return status;
        }
    }
    memory_governor_status().expect("the model is registered")
}

fn now_ms() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
        * 1000.0
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Qwen3.5 dense checkpoint"]
async fn default_mtp_conversation_resumes_after_a_restart() {
    conversation_resumes_after_a_restart(Decode::Mtp).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Qwen3.5 dense checkpoint"]
async fn plain_conversation_resumes_after_a_restart() {
    conversation_resumes_after_a_restart(Decode::Autoregressive).await;
}

async fn conversation_resumes_after_a_restart(decode: Decode) {
    let Some(path) = source() else {
        return;
    };
    mlx_core::tracing::try_init_tracing();
    let expect_lane_disk = std::env::var("MLX_E2E_EXPECT_LANE_DISK").as_deref() == Ok("1");
    let label = match decode {
        Decode::Mtp => "mtp",
        Decode::Autoregressive => "ar",
    };
    let root = std::env::temp_dir().join(format!(
        "mlx-state-tier-restart-{}-{label}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("cache root");
    let persistent = || Qwen35LoadOptions {
        persist_paged_cache: Some(true),
        cache_dir: Some(root.to_string_lossy().to_string()),
        ..Qwen35LoadOptions::default()
    };
    let full = document(180);
    let q1 = "What is the access code in the document? Answer briefly.";
    let p1 = vec![message("user", &format!("{full}\n\n{q1}"))];
    let p2 = || {
        vec![
            message("user", &format!("{full}\n\n{q1}")),
            message("assistant", FIXED_ANSWER),
            message("user", "Which paragraph holds it? Answer with its number."),
        ]
    };
    let p3 = vec![message(
        "user",
        &format!("{}\n\nWhat does paragraph 3 say?", document(120)),
    )];

    let reference = load(
        &path,
        Qwen35LoadOptions {
            cache_disk: Some("0".to_string()),
            ..Qwen35LoadOptions::default()
        },
    )
    .await;
    let cold = turn(&reference, p2(), decode, true).await;
    assert_eq!(cold.cached_tokens, 0);
    reference.dispose().await.expect("dispose C");
    drop(reference);

    let first = load(&path, persistent()).await;
    let one = turn(&first, p1, decode, true).await;
    let captured = status_once(&first, |status| {
        status.dflash2_states.replay_state_publications >= 1.0
    })
    .await
    .dflash2_states;
    eprintln!(
        "{label} A: prompt {} checkpoints {} replay {} disk {} failures {}/{}/{}",
        one.prompt_tokens,
        captured.checkpoint_publications,
        captured.replay_state_publications,
        captured.disk_state_publications,
        captured.checkpoint_publication_failures,
        captured.replay_state_publication_failures,
        captured.junction_materialization_failures
    );
    assert!(one.prompt_tokens > 6144 + BLOCK, "{}", one.prompt_tokens);
    assert!(captured.checkpoint_publications >= 1.0);
    assert!(captured.replay_state_publications >= 1.0);
    assert_eq!(captured.checkpoint_publication_failures, 0.0);
    assert_eq!(captured.replay_state_publication_failures, 0.0);
    assert_eq!(captured.junction_materialization_failures, 0.0);
    assert_eq!(captured.disk.mode, "persistent");
    assert!(
        first.close_cache(now_ms() + 30_000.0).await.expect("close"),
        "the clean close wrote every waiting point"
    );
    first.dispose().await.expect("dispose A");
    drop(first);

    let second = load(&path, persistent()).await;
    let two = turn(&second, p2(), decode, true).await;
    let restored = status_once(&second, |status| {
        let states = &status.dflash2_states;
        states.disk_promotions + states.disk_promotions_skipped >= 1.0
    })
    .await
    .dflash2_states;
    assert_eq!(restored.disk.mode, "persistent");
    assert!(
        restored.disk.taken_back >= 1.0,
        "{}",
        restored.disk.taken_back
    );
    eprintln!(
        "{label} B: P2 cached {} of {} (A prompt {}); promotions {} skipped {}",
        two.cached_tokens,
        two.prompt_tokens,
        one.prompt_tokens,
        restored.disk_promotions,
        restored.disk_promotions_skipped
    );
    assert_eq!(two.cached_tokens % BLOCK, 0);
    assert!(two.cached_tokens < one.prompt_tokens);
    assert!(
        two.cached_tokens + 64 >= one.prompt_tokens,
        "P2 resumes at A's replay point: {} vs {}",
        two.cached_tokens,
        one.prompt_tokens
    );
    assert_eq!(two.prompt_tokens, cold.prompt_tokens);
    let promoted = restored.disk_promotions + restored.disk_promotions_skipped;
    assert!(promoted >= 1.0, "a disk state was read: {promoted}");
    assert_eq!(restored.offload_failures, 0.0);
    assert_eq!(restored.invalidations, 0.0);
    assert_eq!(
        two.raw_text, cold.raw_text,
        "P2 after the restart == cold P2"
    );
    assert_eq!(two.num_tokens, cold.num_tokens);
    assert!(
        second
            .close_cache(now_ms() + 30_000.0)
            .await
            .expect("close"),
        "the clean close wrote every waiting point"
    );
    second.dispose().await.expect("dispose B");
    drop(second);

    let third = load(&path, persistent()).await;
    let three = turn(&third, p3, decode, true).await;
    eprintln!(
        "{label} D: P3 cached {} of {}",
        three.cached_tokens, three.prompt_tokens
    );
    if expect_lane_disk {
        assert!(captured.disk_state_publications >= 1.0);
        assert_eq!(three.cached_tokens, CHECKPOINT);
    } else {
        assert!(
            three.cached_tokens == 0 || three.cached_tokens == CHECKPOINT,
            "{}",
            three.cached_tokens
        );
        eprintln!(
            "{label} P3 {}",
            if three.cached_tokens == CHECKPOINT {
                "resumed at the checkpoint"
            } else {
                "prefilled cold: the checkpoint stayed in RAM"
            }
        );
    }
    third.dispose().await.expect("dispose D");
    drop(third);
    let _ = std::fs::remove_dir_all(&root);
}

/// The image case: C (cache off) is the cold reference of P2, A publishes
/// P1's replay point over image X and closes cleanly, B takes it back and
/// first runs P2 over image Y (same size, so the same placeholder ids),
/// which restores nothing and leaves X's state, then P2 over X, which
/// resumes at A's replay point and decodes as C did.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Qwen3.5 dense checkpoint"]
async fn image_conversation_resumes_after_a_restart() {
    let Some(path) = source() else {
        return;
    };
    mlx_core::tracing::try_init_tracing();
    let decode = Decode::Autoregressive;
    let root = std::env::temp_dir().join(format!(
        "mlx-state-tier-restart-{}-image",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("cache root");
    let persistent = || Qwen35LoadOptions {
        persist_paged_cache: Some(true),
        cache_dir: Some(root.to_string_lossy().to_string()),
        ..Qwen35LoadOptions::default()
    };
    let (x, y) = images();
    let q1 = format!(
        "{}\n\nRead the text in the image, then tell me the access code. Answer briefly.",
        document(40)
    );
    let p1 = |image: &[u8]| vec![image_message(&q1, image)];
    let p2 = |image: &[u8]| {
        vec![
            image_message(&q1, image),
            message("assistant", FIXED_ANSWER),
            message("user", "Which paragraph holds it? Answer with its number."),
        ]
    };

    let reference = load(
        &path,
        Qwen35LoadOptions {
            cache_disk: Some("0".to_string()),
            ..Qwen35LoadOptions::default()
        },
    )
    .await;
    let cold = turn(&reference, p2(&x), decode, false).await;
    assert_eq!(cold.cached_tokens, 0);
    reference.dispose().await.expect("dispose C");
    drop(reference);

    let first = load(&path, persistent()).await;
    let one = turn(&first, p1(&x), decode, false).await;
    let captured = status_once(&first, |status| {
        status.dflash2_states.replay_state_publications >= 1.0
    })
    .await
    .dflash2_states;
    eprintln!(
        "image A: prompt {} checkpoints {} replay {} failures {}/{}",
        one.prompt_tokens,
        captured.checkpoint_publications,
        captured.replay_state_publications,
        captured.checkpoint_publication_failures,
        captured.replay_state_publication_failures
    );
    assert!(captured.replay_state_publications >= 1.0);
    assert_eq!(captured.replay_state_publication_failures, 0.0);
    assert!(
        first.close_cache(now_ms() + 30_000.0).await.expect("close"),
        "the clean close wrote every waiting point"
    );
    first.dispose().await.expect("dispose A");
    drop(first);

    let second = load(&path, persistent()).await;
    let other = turn(&second, p2(&y), decode, false).await;
    eprintln!(
        "image B: P2 over the other image cached {} of {}",
        other.cached_tokens, other.prompt_tokens
    );
    assert_eq!(
        other.prompt_tokens, cold.prompt_tokens,
        "the same placeholder ids"
    );
    assert_eq!(
        other.cached_tokens, 0,
        "another image at the same placeholder ids restores nothing"
    );
    let two = turn(&second, p2(&x), decode, false).await;
    let restored = status_once(&second, |status| {
        let states = &status.dflash2_states;
        states.disk_promotions + states.disk_promotions_skipped >= 1.0
    })
    .await
    .dflash2_states;
    eprintln!(
        "image B: P2 cached {} of {} (A prompt {}); promotions {} skipped {}",
        two.cached_tokens,
        two.prompt_tokens,
        one.prompt_tokens,
        restored.disk_promotions,
        restored.disk_promotions_skipped
    );
    assert!(
        restored.disk.taken_back >= 1.0,
        "{}",
        restored.disk.taken_back
    );
    assert_eq!(two.cached_tokens % BLOCK, 0);
    assert!(two.cached_tokens < one.prompt_tokens);
    assert!(
        two.cached_tokens + 64 >= one.prompt_tokens,
        "P2 resumes at A's replay point: {} vs {}",
        two.cached_tokens,
        one.prompt_tokens
    );
    assert_eq!(
        restored.invalidations, 0.0,
        "the other image discarded nothing"
    );
    assert_eq!(restored.offload_failures, 0.0);
    assert_eq!(
        two.raw_text, cold.raw_text,
        "P2 after the restart == cold P2"
    );
    assert_eq!(two.num_tokens, cold.num_tokens);
    second.dispose().await.expect("dispose B");
    drop(second);
    let _ = std::fs::remove_dir_all(&root);
}
