//! What a streaming turn's progress reads before its prefill in the
//! scheduled lane: the wait freezes at pickup, `prepare` covers the prefix
//! preparation, `slot` only a real lane or memory wait, and the phase never
//! moves back from `prefill`.

use std::sync::Mutex;
use std::time::Duration;

use super::*;
use crate::engine::governor::governor::tests::{G, model};
use crate::engine::napi_glue::chat_stream_plumbing;
use crate::engine::types::{ChatStreamChunk, ChatStreamProgress};
use crate::models::qwen3_5::model::Qwen35Inner;
use crate::models::qwen3_5_moe::model::Qwen35MoeInner;
use crate::tokenizer::ChatMessage;

type Receiver<C> = tokio::sync::mpsc::UnboundedReceiver<ModelCommand<C>>;
type Sender<C> = tokio::sync::mpsc::UnboundedSender<ModelCommand<C>>;

/// Word-level tokens `t0`..`t15`; the template prints each message's text.
fn content_tokenizer() -> Arc<Qwen3Tokenizer> {
    let vocab = (0..16)
        .map(|i| format!("\"t{i}\": {i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let json = format!(
        r#"{{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {{ "type": "Whitespace" }},
            "post_processor": null,
            "decoder": null,
            "model": {{
                "type": "WordLevel",
                "vocab": {{ {vocab}, "<|im_end|>": 16 }},
                "unk_token": "t0"
            }}
        }}"#
    );
    let tokenizer = tokenizers::Tokenizer::from_bytes(json.as_bytes())
        .unwrap_or_else(|error| panic!("fixture tokenizer: {error}"));
    Arc::new(
        Qwen3Tokenizer::from_tokenizer(
            tokenizer,
            Some("{% for m in messages %}{{ m['content'] }} {% endfor %}".to_string()),
        )
        .expect("tokenizer"),
    )
}

fn registration() -> crate::engine::governor::governor::ModelGuard<'static> {
    super::tests::roomy_governor()
        .1
        .register_model(model(16 * G, 0))
        .expect("register")
}

fn dense() -> HybridSchedulerState<Qwen35Inner> {
    let mut inner = crate::models::qwen3_5::model::scheduled_mtp::seeded_inner(0x5730_0101);
    inner.tokenizer = Some(content_tokenizer());
    inner.memory_registration = Some(registration());
    inner.grant_scheduled_lanes(4);
    let mut state = HybridSchedulerState::new(inner).expect("scheduler");
    state.enabled = true;
    state
}

fn moe() -> HybridSchedulerState<Qwen35MoeInner> {
    let mut inner = crate::models::qwen3_5::model::scheduled_mtp::seeded_moe_inner(0x5730_0102);
    inner.tokenizer = Some(content_tokenizer());
    inner.memory_registration = Some(registration());
    inner.grant_scheduled_lanes(4);
    let mut state = HybridSchedulerState::new(inner).expect("scheduler");
    state.enabled = true;
    state
}

/// `len` words starting at `t{first}`.
fn prompt(len: usize, first: usize) -> Vec<ChatMessage> {
    let text = (0..len)
        .map(|i| format!("t{}", (first + i) % 15 + 1))
        .collect::<Vec<_>>()
        .join(" ");
    vec![
        serde_json::from_value(serde_json::json!({"role": "user", "content": text}))
            .expect("message"),
    ]
}

fn config(owner: &str, max_new_tokens: i32) -> ChatConfig {
    ChatConfig {
        max_new_tokens: Some(max_new_tokens),
        temperature: Some(0.0),
        enable_mtp: Some(false),
        max_consecutive_tokens: Some(0),
        max_ngram_repeats: Some(0),
        reasoning_effort: Some("none".to_string()),
        cache_owner_id: Some(owner.to_string()),
        ..ChatConfig::default()
    }
}

struct Stream {
    command: Option<ChatCmd>,
    progress: Arc<TurnProgress>,
    rx: tokio::sync::mpsc::Receiver<Result<ChatStreamChunk>>,
    done: Option<ChatStreamChunk>,
    created: Instant,
}

impl Stream {
    fn new(owner: &str, messages: Vec<ChatMessage>, max_new_tokens: i32) -> Self {
        let (plumbing, rx) = chat_stream_plumbing();
        let command = ChatCmd::StreamSessionStart {
            messages,
            config: config(owner, max_new_tokens),
            stream_tx: plumbing.stream_tx,
            cancelled: plumbing.cancelled,
        };
        Self {
            command: Some(command),
            progress: Arc::clone(&plumbing.handle.progress),
            rx,
            done: None,
            created: Instant::now(),
        }
    }

    fn send<C>(&mut self, sender: &Sender<C>) {
        let command = self.command.take().expect("sent once");
        assert!(sender.send(ModelCommand::from_chat(command)).is_ok());
    }

    fn snapshot(&self) -> ChatStreamProgress {
        self.progress.snapshot()
    }

    /// Reads the chunks so far; whether the turn ended.
    fn poll(&mut self) -> bool {
        while let Ok(chunk) = self.rx.try_recv() {
            let chunk = chunk.unwrap_or_else(|error| panic!("turn failed: {}", error.reason));
            if chunk.done {
                self.done = Some(chunk);
            }
        }
        self.done.is_some()
    }
}

/// One finished sync turn of `owner` over `messages`: its blocks are in the
/// prefix cache afterwards.
fn finish_sync<B: HybridSchedulerBackend>(
    state: &mut HybridSchedulerState<B>,
    owner: &str,
    messages: Vec<ChatMessage>,
) {
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    let (reply, mut result) = tokio::sync::oneshot::channel();
    let command = ChatCmd::SessionStart {
        messages,
        config: config(owner, 2),
        reply,
        cancelled: Arc::new(AtomicBool::new(false)),
    };
    assert!(sender.send(ModelCommand::from_chat(command)).is_ok());
    drop(sender);
    for _ in 0..64 {
        if let Ok(reply) = result.try_recv() {
            reply.unwrap_or_else(|error| panic!("seed turn failed: {}", error.reason));
            return;
        }
        assert!(matches!(state.drive(&mut receiver), LoopControl::Continue));
    }
    panic!("the seed turn did not finish");
}

/// Drives with the channel closed until every stream ended.
fn drive_to_end<B: HybridSchedulerBackend>(
    state: &mut HybridSchedulerState<B>,
    receiver: &mut Receiver<B::FamilyCommand>,
    streams: &mut [&mut Stream],
    mut each_pass: impl FnMut(&[&mut Stream]),
) {
    for _ in 0..256 {
        if streams.iter_mut().all(|stream| stream.poll()) {
            return;
        }
        if matches!(state.drive(receiver), LoopControl::Break) {
            break;
        }
        each_pass(streams);
    }
    assert!(
        streams.iter_mut().all(|stream| stream.poll()),
        "every stream ends"
    );
}

fn skip(test: &str) -> bool {
    crate::test_support::paged_prefill_unsupported(test)
        || !crate::models::qwen3_5::dflash2_decode::tests::metal_is_default_device()
}

/// P-T2: B arrives while A's step runs. Its wait counts from its creation to
/// the pass that takes it off the channel, and stays frozen while B
/// prepares. Mutation P2 (pickup marked at `begin_prefill` only) -> red.
#[test]
fn a_command_sent_during_a_step_waits_until_the_next_pass() {
    if skip("a_command_sent_during_a_step_waits_until_the_next_pass") {
        return;
    }
    let mut state = dense();
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    let mut a = Stream::new("a", prompt(20, 0), 8);
    a.send(&sender);
    assert!(matches!(state.drive(&mut receiver), LoopControl::Continue));
    assert!(!a.poll(), "A still runs");

    let mut b = Stream::new("b", prompt(20, 3), 2);
    let mut command = b.command.take();
    let in_step = sender.clone();
    assert!(state.step_after_snapshot(|| {
        if let Some(command) = command.take() {
            assert!(in_step.send(ModelCommand::from_chat(command)).is_ok());
            std::thread::sleep(Duration::from_millis(30));
        }
    }));
    let unpicked = b.snapshot();
    assert_eq!(unpicked.phase, "queued");
    assert_eq!((unpicked.slot_ms, unpicked.prepare_ms), (None, None));

    let reads = Arc::new(Mutex::new(Vec::new()));
    let hook_reads = Arc::clone(&reads);
    let b_progress = Arc::clone(&b.progress);
    state.after_prefix_prepare = Some(Box::new(move || {
        let first = b_progress.snapshot();
        if first.phase != "prepare" {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
        let second = b_progress.snapshot();
        hook_reads
            .lock()
            .unwrap()
            .push((first.wait_ms, second.wait_ms));
    }));
    drop(sender);
    drive_to_end(&mut state, &mut receiver, &mut [&mut a, &mut b], |_| {});

    let reads = reads.lock().unwrap().clone();
    assert_eq!(reads.len(), 1, "B prepared once: {reads:?}");
    let (first, second) = reads[0];
    assert!(first >= 30.0, "B waited out A's step: {first} ms");
    assert_eq!(first, second, "the wait is frozen at pickup");
    let done = b.snapshot();
    assert_eq!(done.wait_ms, first);
    assert!(done.prepare_ms.is_some_and(|ms| ms >= 20.0), "{done:?}");
}

/// P-T3: A (on a registered prefix) and B arrive in one pass. While A's
/// prefix prepare runs, B waits on the busy thread: `queued`, not `slot`.
/// Mutation P3 (`receive` enters `slot`) -> red.
#[test]
fn a_busy_model_thread_is_not_a_slot_wait() {
    if skip("a_busy_model_thread_is_not_a_slot_wait") {
        return;
    }
    let mut state = moe();
    finish_sync(&mut state, "seed", prompt(40, 0));
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    let mut a = Stream::new("a", prompt(44, 0), 2);
    let mut b = Stream::new("b", prompt(20, 5), 2);
    a.send(&sender);
    b.send(&sender);
    drop(sender);

    let seen = Arc::new(Mutex::new(Vec::new()));
    let hook_seen = Arc::clone(&seen);
    let (a_progress, b_progress) = (Arc::clone(&a.progress), Arc::clone(&b.progress));
    state.after_prefix_prepare = Some(Box::new(move || {
        let a = a_progress.snapshot();
        if a.phase == "prepare" {
            hook_seen.lock().unwrap().push((a, b_progress.snapshot()));
        }
    }));
    drive_to_end(&mut state, &mut receiver, &mut [&mut a, &mut b], |_| {});

    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 1, "A prepared once");
    let (a_seen, b_seen) = &seen[0];
    assert!(
        a_seen.cached_tokens.is_some_and(|n| n > 0),
        "A restores the seed's prefix: {a_seen:?}"
    );
    assert_eq!(b_seen.phase, "queued", "B waits on the busy thread");
    assert_eq!(b_seen.slot_ms, Some(0.0), "B was picked up: {b_seen:?}");
    assert_eq!(b.snapshot().slot_ms, Some(0.0), "B never waited for a slot");
}

/// P-T4: one lane. B waits for it in `slot` while A runs and counts that
/// time. Mutation P4 (no mark at the lane break) -> red.
#[test]
fn a_turn_behind_full_lanes_waits_for_a_slot() {
    if skip("a_turn_behind_full_lanes_waits_for_a_slot") {
        return;
    }
    let mut state = dense();
    state.scheduler.set_max_num_seqs(1);
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    let mut a = Stream::new("a", prompt(20, 0), 6);
    let mut b = Stream::new("b", prompt(20, 4), 2);
    a.send(&sender);
    b.send(&sender);
    drop(sender);
    let mut b_phases = Vec::new();
    drive_to_end(
        &mut state,
        &mut receiver,
        &mut [&mut a, &mut b],
        |streams| {
            if streams[0].done.is_none() {
                b_phases.push(streams[1].snapshot().phase);
            }
        },
    );
    assert!(!b_phases.is_empty());
    assert!(
        b_phases.iter().all(|phase| phase == "slot"),
        "B waits for the lane while A runs: {b_phases:?}"
    );
    let done = b.snapshot();
    assert!(done.slot_ms.is_some_and(|ms| ms > 0.0), "{done:?}");
    assert_eq!(a.snapshot().slot_ms, Some(0.0));
}

/// P-T5: a new owner's turn on a registered prefix restores it while
/// `prepare`, with its cached tokens known, and its timers fit inside its
/// life. Mutation P5 (`install_prepare_progress` starts the turn) -> red.
fn restoring_reads_prepare<B: HybridSchedulerBackend>(mut state: HybridSchedulerState<B>) {
    finish_sync(&mut state, "seed", prompt(40, 0));
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    let mut b = Stream::new("b", prompt(44, 0), 2);
    b.send(&sender);
    drop(sender);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let hook_seen = Arc::clone(&seen);
    let b_progress = Arc::clone(&b.progress);
    state.after_prefix_prepare = Some(Box::new(move || {
        hook_seen.lock().unwrap().push(b_progress.snapshot());
    }));
    drive_to_end(&mut state, &mut receiver, &mut [&mut b], |_| {});
    let elapsed_ms = b.created.elapsed().as_secs_f64() * 1000.0;

    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 1, "{}: one prefix prepare", B::SCHEDULER_NAME);
    let during = &seen[0];
    assert_eq!(during.phase, "prepare", "{}", B::SCHEDULER_NAME);
    let cached = b
        .done
        .as_ref()
        .and_then(|done| done.cached_tokens)
        .expect("the done chunk reports the cached prefix");
    assert!(cached > 0, "{}: the seed's prefix hits", B::SCHEDULER_NAME);
    assert_eq!(during.cached_tokens, Some(cached), "{}", B::SCHEDULER_NAME);

    let done = b.snapshot();
    let prepare = done.prepare_ms.expect("picked up");
    let slot = done.slot_ms.expect("picked up");
    let prefill = done.prefill_elapsed_ms.expect("prefilled");
    assert!(prepare > 0.0, "{}: {done:?}", B::SCHEDULER_NAME);
    assert!(
        done.wait_ms + slot + prepare + prefill <= elapsed_ms,
        "{}: {done:?} within {elapsed_ms} ms",
        B::SCHEDULER_NAME
    );
}

#[test]
fn a_dense_prefix_restore_reads_prepare() {
    if skip("a_dense_prefix_restore_reads_prepare") {
        return;
    }
    restoring_reads_prepare(dense());
}

#[test]
fn a_moe_prefix_restore_reads_prepare() {
    if skip("a_moe_prefix_restore_reads_prepare") {
        return;
    }
    restoring_reads_prepare(moe());
}

/// P-T6: A is preempted after its first prefill slice and replays. Its
/// phase never leaves `prefill`/`decode`, and its prefill clock counts from
/// the first slice. Mutation P6 (`begin_prefill` always restarts the clock)
/// -> red.
#[test]
fn a_preemption_replay_keeps_the_first_prefill_start() {
    if skip("a_preemption_replay_keeps_the_first_prefill_start") {
        return;
    }
    let mut state = dense();
    state.scheduler.set_fixed_prefill_budget(16);
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    let mut a = Stream::new("a", prompt(40, 0), 2);
    a.send(&sender);
    assert!(matches!(state.drive(&mut receiver), LoopControl::Continue));
    let first = a.snapshot();
    assert_eq!(first.phase, "prefill", "{first:?}");
    assert!(
        first.decode_elapsed_ms.is_none(),
        "one slice ran, not the prompt: {first:?}"
    );
    std::thread::sleep(Duration::from_millis(300));
    let seq = state.owner_sequences["a"];
    assert!(state.preempt_running(seq));
    assert_eq!(a.snapshot().phase, "prefill");

    drop(sender);
    let mut phases = Vec::new();
    drive_to_end(&mut state, &mut receiver, &mut [&mut a], |streams| {
        phases.push(streams[0].snapshot().phase);
    });
    assert!(
        phases
            .iter()
            .all(|phase| matches!(phase.as_str(), "prefill" | "decode" | "done")),
        "never back from prefill: {phases:?}"
    );
    let done = a.snapshot();
    assert!(
        done.prefill_elapsed_ms.is_some_and(|ms| ms >= 300.0),
        "the prefill clock runs from the first slice: {done:?}"
    );
}
