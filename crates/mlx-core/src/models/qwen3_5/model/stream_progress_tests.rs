//! The flat streaming cores (the native-MTP turn path) publish their prefill
//! span before prefilling. Each case cancels at the first chunk boundary
//! (prompt > one chunk), so no forward runs and no GPU work is issued.

use super::mtp_gate_state_tests::tiny_cfg;
use super::*;
use crate::engine::progress::TurnProgress;
use crate::model_thread::StreamTx;

const LONG: usize = 2 * super::forward::PREFILL_STEP_SIZE as usize + 5;
const CACHED: usize = 100;

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
                "vocab": { "a": 0, "b": 1, "c": 2, "<unk>": 3, "<|im_end|>": 4 },
                "unk_token": "<unk>"
            }
        }"#;
    let tok = tokenizers::Tokenizer::from_bytes(json)
        .unwrap_or_else(|e| panic!("fixture tokenizer: {e}"));
    let tok = Qwen3Tokenizer::from_tokenizer(tok, None);
    Arc::new(tok)
}

fn tokens(len: usize) -> Vec<u32> {
    (0..len).map(|i| (i % 3) as u32).collect()
}

fn inner_with_history(history: usize) -> Qwen35Inner {
    let mut inner = Qwen35Inner::new(tiny_cfg()).expect("construct");
    inner.tokenizer = Some(tiny_tokenizer());
    inner.turn_cancel = Some(Arc::new(AtomicBool::new(true)));
    if history > 0 {
        inner.cached_token_history = tokens(history);
        inner.caches = Some(Qwen3_5LayerCache::fresh_layer_caches(&inner.config));
    }
    inner
}

fn progress_sink() -> (
    Arc<TurnProgress>,
    StreamTx<ChatStreamChunk>,
    tokio::sync::mpsc::Receiver<Result<ChatStreamChunk>>,
) {
    let progress = Arc::new(TurnProgress::new());
    let (tx, rx) = crate::model_thread::stream_channel_with_progress(8, Arc::clone(&progress));
    (progress, tx, rx)
}

fn thinking_off() -> ThinkingSetup {
    ThinkingSetup {
        enabled: false,
        budget: None,
    }
}

/// `(prefill_total, cached_tokens, prefill_done)` after a stream-start turn
/// over `prompt` that cancels at the first prefill boundary.
fn start_turn_span(inner: &mut Qwen35Inner, prompt: Vec<u32>) -> (u32, u32, u32) {
    let (progress, tx, _rx) = progress_sink();
    let err = inner
        .chat_stream_sync_inner(
            prompt,
            &[],
            ChatConfig::default(),
            4,
            &StreamSender(&tx),
            &AtomicBool::new(false),
            thinking_off(),
        )
        .expect_err("the turn cancels at the first prefill boundary");
    assert_eq!(err.reason, crate::models::forward::PREFILL_CANCELLED);
    span(&progress)
}

fn delta_turn_span(inner: &mut Qwen35Inner, delta: Vec<u32>) -> (u32, u32, u32) {
    let (progress, tx, _rx) = progress_sink();
    let err = inner
        .chat_stream_tokens_delta_sync_inner(
            delta,
            ChatConfig::default(),
            &StreamSender(&tx),
            &AtomicBool::new(false),
            thinking_off(),
        )
        .expect_err("the turn cancels at the first prefill boundary");
    assert_eq!(err.reason, crate::models::forward::PREFILL_CANCELLED);
    span(&progress)
}

fn span(progress: &TurnProgress) -> (u32, u32, u32) {
    let s = progress.snapshot();
    (
        s.prefill_total.expect("prefill total published"),
        s.cached_tokens.expect("cached prefix published"),
        s.prefill_done.expect("prefill done published"),
    )
}

#[test]
fn fresh_stream_start_publishes_the_full_prompt() {
    let mut inner = inner_with_history(0);
    assert_eq!(
        start_turn_span(&mut inner, tokens(LONG)),
        (LONG as u32, 0, 0)
    );
}

#[test]
fn cached_prefix_stream_start_publishes_only_the_suffix() {
    let mut inner = inner_with_history(CACHED);
    assert_eq!(
        start_turn_span(&mut inner, tokens(CACHED + LONG)),
        (LONG as u32, CACHED as u32, 0)
    );
}

#[test]
fn desynced_stream_start_publishes_the_rebuilt_prompt() {
    let mut inner = inner_with_history(CACHED);
    inner.flat_mtp_caches_desynced = true;
    assert_eq!(
        start_turn_span(&mut inner, tokens(CACHED + LONG)),
        ((CACHED + LONG) as u32, 0, 0)
    );
}

#[test]
fn delta_stream_publishes_the_delta_over_the_history() {
    let mut inner = inner_with_history(CACHED);
    assert_eq!(
        delta_turn_span(&mut inner, tokens(LONG)),
        (LONG as u32, CACHED as u32, 0)
    );
}

#[test]
fn rebuilt_delta_stream_publishes_the_full_history() {
    let mut inner = inner_with_history(CACHED);
    inner.flat_mtp_caches_desynced = true;
    assert_eq!(
        delta_turn_span(&mut inner, tokens(LONG)),
        ((CACHED + LONG) as u32, 0, 0)
    );
}
