//! Local perf bench for dense Qwen3.8 + external DFlash2 drafter.
//!
//! ```shell
//! MLX_TEST_QWEN38_TARGET_PATH=/path/to/target.gguf \
//! MLX_TEST_QWEN38_DFLASH2_PATH=/path/to/dflash2 \
//! cargo test -p mlx-core --release --test qwen3_8_dflash2_bench -- \
//!   --ignored --nocapture --test-threads=1
//! ```
//!
//! Decode lines report the speculative-cycle decomposition:
//! `A` = committed tokens/cycle (numerator) and `C` = ms/cycle
//! (denominator), so `decode_tps ≈ 1000·A/C`. A throughput change is only
//! interpretable alongside both. DFlash2 uses its checkpoint proposal width;
//! `mtp_mean_depth` reports the actual mean proposal width after budget clamping.
//!
//! `qwen38_dflash2_decode_after_restore` decodes 512 tokens on one final
//! prompt after a cold start, an E resume and a C(R) resume, at every context
//! of `MLX_BENCH_RESTORE_CONTEXTS` (default `32768,81920,163840` tokens) and
//! for `MLX_BENCH_RESTORE_PROMPTS` (default 3) decode tasks. Every case runs in
//! a fresh child process of this binary, one model load at a time; the kinds
//! rotate through the slots of each context so no kind always runs first.
//! - cold: the final prompt alone.
//! - E: turn A1 (the document, answered `OK` with thinking off), a turn on
//!   another document that parks A1's end state, then the final prompt, which
//!   extends A1's history.
//! - C(R): A1 with thinking on, so its history leaves the final prompt right
//!   after the generation prefix; the other turn; then the final prompt, which
//!   resumes at A1's replay point.
//!
//! Pass: every resume case has the cold case's `A` and cycle count for the
//! same context and task, and per context the mean `C` over tasks of each
//! resume kind is at most 1.02 x the cold mean. Each kind takes each slot once
//! per context over the 3 tasks, so the means cancel the slot bias: a later
//! process decodes slower on a warmer chip (+9.5% measured for the second of
//! two identical 8K cold turns with no cooldown). A median of per-task ratios
//! would not cancel it. `MLX_BENCH_RESTORE_COOLDOWN_S` (default 60) idles
//! between children. Results
//! go to `$MLX_DFLASH2_GATE_OUT/<MLX_BENCH_STAGE, default S1>/restore/` (under
//! cargo's test tmpdir without `MLX_DFLASH2_GATE_OUT`), one JSON and one
//! inference log per case plus `restore.json`.
//!
//! The two benches are separate tests; run one with its name and `--exact`:
//! ```shell
//! cargo test -p mlx-core --release --test qwen3_8_dflash2_bench -- \
//!   qwen38_dflash2_perf --exact --ignored --nocapture --test-threads=1
//! ```

use std::path::{Path, PathBuf};

use mlx_core::engine::types::ChatConfig;
use mlx_core::models::qwen3_5::model::{Qwen3_5Model, Qwen35LoadOptions};
use mlx_core::profiling::PerformanceMetrics;
use mlx_core::tokenizer::ChatMessage;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

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

fn user_message(content: &str) -> ChatMessage {
    message("user", content)
}

/// `(cycles, A, C)`: speculative cycles, committed tokens/cycle and decode
/// ms/cycle. decode_ms is derived from tokens and decode_tps (the only
/// wall-clock the metrics carry).
fn cycle_split(num_tokens: u32, perf: &PerformanceMetrics) -> Option<(u32, f64, f64)> {
    let (Some(cycles), Some(a)) = (perf.mtp_cycles, perf.mtp_mean_accepted_tokens_total) else {
        return None;
    };
    if cycles == 0 || perf.decode_tokens_per_second <= 0.0 {
        return None;
    }
    let decode_ms = num_tokens as f64 / perf.decode_tokens_per_second * 1000.0;
    Some((cycles, a, decode_ms / cycles as f64))
}

fn print_perf(tag: &str, num_tokens: u32, perf: &PerformanceMetrics) {
    eprintln!(
        "[bench] {tag}: tokens={num_tokens} ttft={:.1}ms prefill={:.1} tok/s decode={:.2} tok/s \
         mtp_cycles={:?} accept_total={:?} accept_drafts={:?} depth={:?} by_pos={:?}",
        perf.ttft_ms,
        perf.prefill_tokens_per_second,
        perf.decode_tokens_per_second,
        perf.mtp_cycles,
        perf.mtp_mean_accepted_tokens_total,
        perf.mtp_mean_accepted_tokens,
        perf.mtp_mean_depth,
        perf.mtp_acceptance_by_position,
    );
    if let Some((cycles, a, c)) = cycle_split(num_tokens, perf) {
        let decode_ms = c * cycles as f64;
        eprintln!(
            "[bench]   cycle: A={a:.3} tok/cycle C={c:.2} ms/cycle (decode_ms={decode_ms:.0})"
        );
    }
    if let Some(phases) = &perf.profile_phases {
        for p in phases {
            eprintln!(
                "[bench]   phase {:<28} total={:>9.1}ms avg={:>9.1}us n={}",
                p.name, p.total_ms, p.avg_us_per_token, p.count
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires real Qwen3.8 target and DFlash2 checkpoints"]
async fn qwen38_dflash2_perf() {
    let target =
        std::env::var("MLX_TEST_QWEN38_TARGET_PATH").expect("set MLX_TEST_QWEN38_TARGET_PATH");
    let draft =
        std::env::var("MLX_TEST_QWEN38_DFLASH2_PATH").expect("set MLX_TEST_QWEN38_DFLASH2_PATH");
    let model = Qwen3_5Model::load(
        target,
        Some(Qwen35LoadOptions {
            draft_model_path: Some(draft),
            kv_format: None,
            max_memory_bytes: None,
            max_context_tokens: None,
            decode_share: None,
        }),
    )
    .await
    .expect("load Qwen3.8 target with DFlash2 companion");

    let chat = |prompt: String, cfg: ChatConfig| {
        let model = &model;
        async move {
            model
                .chat_session_start(vec![user_message(&prompt)], Some(cfg))
                .await
                .expect("chat turn")
        }
    };

    // --- Prefill: long prompt, short generation -------------------------
    // ~1500-token prompt so the draft-context dead-work skip (WS9) and the
    // last-row logits span (WS10) are exercised at scale.
    let long_prompt = format!(
        "Summarize the following in one sentence.\n{}",
        "The quick brown fox jumps over the lazy dog while the moon rises over the quiet harbor town. "
            .repeat(220)
    );
    for rep in 0..3 {
        model.reset_caches().await.expect("reset");
        let r = chat(
            long_prompt.clone(),
            ChatConfig {
                max_new_tokens: Some(8),
                temperature: Some(0.0),
                report_performance: Some(true),
                enable_mtp: Some(true),
                reasoning_effort: Some("none".to_string()),
                ..ChatConfig::default()
            },
        )
        .await;
        print_perf(
            &format!("prefill_long rep{rep}"),
            r.num_tokens,
            r.performance.as_ref().expect("perf"),
        );
    }

    // --- Decode: medium prompt, fixed checkpoint width, long generation ---
    let medium_prompt = std::env::var("MLX_BENCH_PROMPT")
        .map(|p| {
            if let Some(f) = p.strip_prefix('@') {
                let path = std::path::Path::new(f);
                let path = if path.is_absolute() {
                    path.to_path_buf()
                } else {
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("../../")
                        .join(path)
                };
                std::fs::read_to_string(&path).expect("read MLX_BENCH_PROMPT @file")
            } else {
                p
            }
        })
        .unwrap_or_else(|_| "Write a short story about a robot learning to paint.".to_string());
    for rep in 0..3 {
        model.reset_caches().await.expect("reset");
        let r = chat(
            medium_prompt.clone(),
            ChatConfig {
                max_new_tokens: Some(256),
                temperature: Some(0.0),
                report_performance: Some(true),
                enable_mtp: Some(true),
                reasoning_effort: Some("none".to_string()),
                ..ChatConfig::default()
            },
        )
        .await;
        print_perf(
            &format!("decode_dflash2 rep{rep}"),
            r.num_tokens,
            r.performance.as_ref().expect("perf"),
        );
        if rep == 0 {
            // char-boundary truncation: byte-slicing panics when 400
            // lands inside a multibyte char (non-ASCII completions).
            let text: String = r.text.chars().take(400).collect();
            eprintln!("[bench] dflash2 finish={} text: {}", r.finish_reason, text);
        }
    }

    // --- AR baseline: same decode workload, MTP off -----------------------
    for rep in 0..2 {
        model.reset_caches().await.expect("reset");
        let r = chat(
            medium_prompt.clone(),
            ChatConfig {
                max_new_tokens: Some(256),
                temperature: Some(0.0),
                report_performance: Some(true),
                enable_mtp: Some(false),
                reasoning_effort: Some("none".to_string()),
                ..ChatConfig::default()
            },
        )
        .await;
        print_perf(
            &format!("decode_ar rep{rep}"),
            r.num_tokens,
            r.performance.as_ref().expect("perf"),
        );
    }

    model.shutdown().expect("shutdown");
}

// ---------------------------------------------------------------------------
// Decode after restore
// ---------------------------------------------------------------------------

const RESTORE_TEST: &str = "qwen38_dflash2_decode_after_restore";
const RESTORE_CASE_ENV: &str = "MLX_BENCH_RESTORE_CASE";
const RESTORE_PART_ENV: &str = "MLX_BENCH_RESTORE_PART";
const RESTORE_LOG_FILTER: &str = "mlx_core::inference=info";
const RESTORE_DECODE_TOKENS: u32 = 512;
const RESTORE_DEFAULT_CONTEXTS: [usize; 3] = [32_768, 81_920, 163_840];
/// Chat-template and instruction tokens around the document, so the final
/// prompt lands near the nominal context.
const RESTORE_PROMPT_OVERHEAD: usize = 256;
const RESTORE_OTHER_DOC_TOKENS: usize = 2048;
const RESTORE_C_TOLERANCE: f64 = 0.02;
const RESTORE_INTRO: &str = "Read this C++ source excerpt carefully.\n\n";
const RESTORE_ACK: &str = "\n\nReply with exactly the word OK and nothing else.";
const RESTORE_REPLY: &str = "OK";
const OTHER_INTRO: &str = "Name the programming language of these notes in one word.\n\n";
const RESTORE_TASKS: [&str; 3] = [
    "Now write a detailed, section-by-section technical walkthrough of the source excerpt \
     above. Cover every major component and explain how they interact. Write at least 1500 \
     words.",
    "Now write a long design review of the code above: its strengths, its weaknesses, and \
     concrete refactoring proposals with code sketches. Write at least 1500 words.",
    "Now write a tutorial for a new contributor that explains how to extend the code above, \
     with numbered steps and code examples. Write at least 1500 words.",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Resume {
    Cold,
    End,
    Capture,
}

impl Resume {
    const ALL: [Resume; 3] = [Resume::Cold, Resume::End, Resume::Capture];

    fn name(self) -> &'static str {
        match self {
            Resume::Cold => "cold",
            Resume::End => "end",
            Resume::Capture => "capture",
        }
    }

    /// The `hit_kind` of the measured turn's `dflash2_prefix` event.
    fn hit_kind(self) -> &'static str {
        match self {
            Resume::Cold => "none",
            Resume::End => "E",
            Resume::Capture => "C",
        }
    }

    fn parse(name: &str) -> Resume {
        Resume::ALL
            .into_iter()
            .find(|resume| resume.name() == name)
            .unwrap_or_else(|| panic!("unknown resume kind {name}"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RestoreCase {
    context: usize,
    resume: Resume,
    task: usize,
}

impl RestoreCase {
    fn name(self) -> String {
        format!("{}-{}-t{}", self.context, self.resume.name(), self.task)
    }

    fn parse(name: &str) -> RestoreCase {
        let parts: Vec<&str> = name.split('-').collect();
        let [context, resume, task] = parts.as_slice() else {
            panic!("restore case {name} is not <context>-<kind>-t<task>");
        };
        RestoreCase {
            context: context
                .parse()
                .unwrap_or_else(|_| panic!("restore case {name}: bad context")),
            resume: Resume::parse(resume),
            task: task
                .strip_prefix('t')
                .and_then(|task| task.parse().ok())
                .filter(|&task| task < RESTORE_TASKS.len())
                .unwrap_or_else(|| panic!("restore case {name}: bad task")),
        }
    }
}

/// Tasks outermost, then contexts; within each context the kinds rotate by
/// task and context so every kind takes every slot.
fn restore_order(contexts: &[usize], tasks: usize) -> Vec<RestoreCase> {
    let mut order = Vec::with_capacity(contexts.len() * tasks * Resume::ALL.len());
    for task in 0..tasks {
        for (index, &context) in contexts.iter().enumerate() {
            for slot in 0..Resume::ALL.len() {
                order.push(RestoreCase {
                    context,
                    resume: Resume::ALL[(slot + task + index) % Resume::ALL.len()],
                    task,
                });
            }
        }
    }
    order
}

#[derive(Clone, Debug, PartialEq)]
struct RestoreResult {
    case: RestoreCase,
    prompt_tokens: u64,
    hit_kind: String,
    hit_tokens: u64,
    prefilled_tokens: u64,
    ttft_ms: f64,
    num_tokens: u32,
    finish_reason: String,
    cycles: u32,
    a: f64,
    c: f64,
    text_sha256: String,
}

impl RestoreResult {
    fn to_json(&self) -> Value {
        json!({
            "case": self.case.name(),
            "prompt_tokens": self.prompt_tokens,
            "hit_kind": self.hit_kind,
            "hit_tokens": self.hit_tokens,
            "prefilled_tokens": self.prefilled_tokens,
            "ttft_ms": self.ttft_ms,
            "num_tokens": self.num_tokens,
            "finish_reason": self.finish_reason,
            "cycles": self.cycles,
            "a": self.a,
            "c": self.c,
            "text_sha256": self.text_sha256,
        })
    }

    fn from_json(value: &Value) -> RestoreResult {
        let text = |name: &str| {
            value[name]
                .as_str()
                .unwrap_or_else(|| panic!("restore result has no {name}: {value}"))
                .to_string()
        };
        let uint = |name: &str| {
            value[name]
                .as_u64()
                .unwrap_or_else(|| panic!("restore result has no {name}: {value}"))
        };
        let float = |name: &str| {
            value[name]
                .as_f64()
                .unwrap_or_else(|| panic!("restore result has no {name}: {value}"))
        };
        RestoreResult {
            case: RestoreCase::parse(&text("case")),
            prompt_tokens: uint("prompt_tokens"),
            hit_kind: text("hit_kind"),
            hit_tokens: uint("hit_tokens"),
            prefilled_tokens: uint("prefilled_tokens"),
            ttft_ms: float("ttft_ms"),
            num_tokens: uint("num_tokens") as u32,
            finish_reason: text("finish_reason"),
            cycles: uint("cycles") as u32,
            a: float("a"),
            c: float("c"),
            text_sha256: text("text_sha256"),
        }
    }
}

/// Per context and resume kind, the mean `C` over tasks divided by the cold
/// mean over the same tasks.
fn c_ratios(results: &[RestoreResult], contexts: &[usize]) -> Vec<(usize, Resume, f64)> {
    let mut ratios = Vec::new();
    for &context in contexts {
        for resume in [Resume::End, Resume::Capture] {
            let pairs: Vec<(f64, f64)> = results
                .iter()
                .filter(|result| result.case.context == context && result.case.resume == resume)
                .filter_map(|result| {
                    results
                        .iter()
                        .find(|cold| {
                            cold.case
                                == RestoreCase {
                                    resume: Resume::Cold,
                                    ..result.case
                                }
                        })
                        .map(|cold| (result.c, cold.c))
                })
                .collect();
            if !pairs.is_empty() {
                let resumed: f64 = pairs.iter().map(|(resumed, _)| resumed).sum();
                let cold: f64 = pairs.iter().map(|(_, cold)| cold).sum();
                ratios.push((context, resume, resumed / cold));
            }
        }
    }
    ratios
}

/// Every way the results miss the pass rule, empty when they meet it.
fn restore_verdict(results: &[RestoreResult], contexts: &[usize], tasks: usize) -> Vec<String> {
    let mut failures = Vec::new();
    for case in restore_order(contexts, tasks) {
        let Some(result) = results.iter().find(|result| result.case == case) else {
            failures.push(format!("{}: no result", case.name()));
            continue;
        };
        if result.hit_kind != case.resume.hit_kind() {
            failures.push(format!(
                "{}: hit_kind {} (expected {})",
                case.name(),
                result.hit_kind,
                case.resume.hit_kind()
            ));
        }
        if result.num_tokens < RESTORE_DECODE_TOKENS {
            failures.push(format!(
                "{}: decoded {} tokens (finish {}), needs {RESTORE_DECODE_TOKENS}",
                case.name(),
                result.num_tokens,
                result.finish_reason
            ));
        }
        if case.resume == Resume::Cold {
            continue;
        }
        let cold = RestoreCase {
            resume: Resume::Cold,
            ..case
        };
        if let Some(cold) = results.iter().find(|result| result.case == cold)
            && (result.a != cold.a || result.cycles != cold.cycles)
        {
            failures.push(format!(
                "{}: A={} over {} cycles, cold A={} over {} cycles",
                case.name(),
                result.a,
                result.cycles,
                cold.a,
                cold.cycles
            ));
        }
    }
    for (context, resume, ratio) in c_ratios(results, contexts) {
        if ratio > 1.0 + RESTORE_C_TOLERANCE {
            failures.push(format!(
                "{context} {}: mean C / cold mean C = {ratio:.4} > {:.2}",
                resume.name(),
                1.0 + RESTORE_C_TOLERANCE
            ));
        }
    }
    failures
}

struct RestoreOptions {
    contexts: Vec<usize>,
    tasks: usize,
    cooldown: std::time::Duration,
    out_dir: PathBuf,
}

impl RestoreOptions {
    fn from_env() -> RestoreOptions {
        let var = |name: &str| std::env::var(name).ok().filter(|value| !value.is_empty());
        let contexts = var("MLX_BENCH_RESTORE_CONTEXTS").map_or_else(
            || RESTORE_DEFAULT_CONTEXTS.to_vec(),
            |list| {
                list.split(',')
                    .map(|context| {
                        context.trim().parse().unwrap_or_else(|_| {
                            panic!("MLX_BENCH_RESTORE_CONTEXTS: bad context {context:?}")
                        })
                    })
                    .collect()
            },
        );
        let tasks = var("MLX_BENCH_RESTORE_PROMPTS").map_or(RESTORE_TASKS.len(), |tasks| {
            tasks
                .parse()
                .ok()
                .filter(|tasks| (1..=RESTORE_TASKS.len()).contains(tasks))
                .unwrap_or_else(|| {
                    panic!(
                        "MLX_BENCH_RESTORE_PROMPTS must be 1..={}, got {tasks}",
                        RESTORE_TASKS.len()
                    )
                })
        });
        let cooldown = var("MLX_BENCH_RESTORE_COOLDOWN_S").map_or(60, |seconds| {
            seconds
                .parse()
                .unwrap_or_else(|_| panic!("MLX_BENCH_RESTORE_COOLDOWN_S: bad value {seconds}"))
        });
        let stage = var("MLX_BENCH_STAGE").unwrap_or_else(|| "S1".to_string());
        let out_dir = var("MLX_DFLASH2_GATE_OUT")
            .map_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")), PathBuf::from)
            .join(stage)
            .join("restore");
        RestoreOptions {
            contexts,
            tasks,
            cooldown: std::time::Duration::from_secs(cooldown),
            out_dir,
        }
    }
}

/// The parent loads nothing: it runs every case in its own logged child
/// process, one model load at a time, and judges the results.
fn run_restore_parent(options: &RestoreOptions) {
    std::fs::create_dir_all(&options.out_dir)
        .unwrap_or_else(|error| panic!("create {}: {error}", options.out_dir.display()));
    let order = restore_order(&options.contexts, options.tasks);
    let mut results = Vec::with_capacity(order.len());
    let mut failed = Vec::new();
    for (index, case) in order.iter().enumerate() {
        if index > 0 {
            std::thread::sleep(options.cooldown);
        }
        let part = options.out_dir.join(format!("{}.json", case.name()));
        let log = options
            .out_dir
            .join(format!("{}.inference.log", case.name()));
        std::fs::remove_file(&part).ok();
        std::fs::remove_file(&log).ok();
        eprintln!("[restore] {}/{} {}", index + 1, order.len(), case.name());
        let status = std::process::Command::new(std::env::current_exe().expect("test binary path"))
            .args([
                RESTORE_TEST,
                "--exact",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("MLX_NODE_LOG", RESTORE_LOG_FILTER)
            .env("MLX_NODE_LOG_FILE", &log)
            .env(RESTORE_CASE_ENV, case.name())
            .env(RESTORE_PART_ENV, &part)
            .status()
            .unwrap_or_else(|error| panic!("spawn {}: {error}", case.name()));
        if let Ok(text) = std::fs::read_to_string(&part) {
            let value: Value = serde_json::from_str(&text).expect("restore result JSON");
            results.push(RestoreResult::from_json(&value));
        }
        if !status.success() {
            failed.push(format!("{}: {status}", case.name()));
            break;
        }
    }
    let failures = restore_verdict(&results, &options.contexts, options.tasks);
    let ratios: Vec<Value> = c_ratios(&results, &options.contexts)
        .into_iter()
        .map(|(context, resume, ratio)| {
            json!({"context": context, "resume": resume.name(), "mean_c_ratio": ratio})
        })
        .collect();
    let report = json!({
        "contexts": options.contexts,
        "tasks": options.tasks,
        "decode_tokens": RESTORE_DECODE_TOKENS,
        "cooldown_s": options.cooldown.as_secs(),
        "c_tolerance": RESTORE_C_TOLERANCE,
        "results": results.iter().map(RestoreResult::to_json).collect::<Vec<_>>(),
        "mean_c_ratios": ratios,
        "child_failures": failed,
        "verdict_failures": failures,
    });
    let path = options.out_dir.join("restore.json");
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&report).expect("restore report JSON"),
    )
    .unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
    for result in &results {
        eprintln!(
            "[restore] {:<22} prompt={} hit={}:{} prefilled={} ttft={:.0}ms tokens={} A={:.4} \
             cycles={} C={:.2} ms/cycle sha={}",
            result.case.name(),
            result.prompt_tokens,
            result.hit_kind,
            result.hit_tokens,
            result.prefilled_tokens,
            result.ttft_ms,
            result.num_tokens,
            result.a,
            result.cycles,
            result.c,
            &result.text_sha256[..12],
        );
    }
    eprintln!("[restore] mean C ratios: {}", report["mean_c_ratios"]);
    eprintln!("[restore] report: {}", path.display());
    assert!(
        failed.is_empty() && failures.is_empty(),
        "decode-after-restore failed:\n{}",
        failed
            .iter()
            .chain(&failures)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

fn corpus_files(dir: &Path, files: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("read corpus dir {}: {error}", dir.display()));
    for entry in entries {
        let path = entry.expect("corpus dir entry").path();
        if path.is_dir() {
            corpus_files(&path, files);
        } else if matches!(
            path.extension().and_then(|ext| ext.to_str()),
            Some("h" | "cpp")
        ) {
            files.push(path);
        }
    }
}

/// `tokens` tokens of the corpus files, in sorted order or reversed.
fn corpus_text(
    tokenizer: &tokenizers::Tokenizer,
    root: &Path,
    tokens: usize,
    reversed: bool,
) -> String {
    let mut files = Vec::new();
    corpus_files(root, &mut files);
    files.sort();
    if reversed {
        files.reverse();
    }
    let mut text = String::new();
    // Source text runs well under 8 bytes per token, so this many bytes
    // always hold the budget.
    for file in &files {
        if text.len() >= tokens * 8 {
            break;
        }
        let relative = file.strip_prefix(root).unwrap_or(file);
        let source = std::fs::read_to_string(file)
            .unwrap_or_else(|error| panic!("read {}: {error}", file.display()));
        text.push_str(&format!("// file: {}\n{source}\n", relative.display()));
    }
    let encoding = tokenizer
        .encode(text.as_str(), false)
        .unwrap_or_else(|error| panic!("encode corpus: {error}"));
    let ids = encoding.get_ids();
    assert!(
        ids.len() >= tokens,
        "corpus {} holds {} tokens, needs {tokens}",
        root.display(),
        ids.len()
    );
    tokenizer
        .decode(&ids[..tokens], false)
        .unwrap_or_else(|error| panic!("decode corpus: {error}"))
}

fn restore_turn_config(max_new_tokens: u32, reasoning_effort: &str) -> ChatConfig {
    ChatConfig {
        cache_owner_id: Some(uuid::Uuid::new_v4().to_string()),
        max_new_tokens: Some(max_new_tokens as i32),
        temperature: Some(0.0),
        report_performance: Some(true),
        reuse_cache: Some(true),
        enable_mtp: Some(true),
        reasoning_effort: Some(reasoning_effort.to_string()),
        ..ChatConfig::default()
    }
}

/// The `dflash2_prefix` events written to `log` since byte `offset`, and the
/// new offset.
fn prefix_events_since(log: &Path, offset: usize) -> (Vec<Value>, usize) {
    let bytes =
        std::fs::read(log).unwrap_or_else(|error| panic!("read {}: {error}", log.display()));
    let fresh = &bytes[offset.min(bytes.len())..];
    let complete = fresh
        .iter()
        .rposition(|&byte| byte == b'\n')
        .map_or(0, |end| end + 1);
    let events = String::from_utf8_lossy(&fresh[..complete])
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|line| line["target"] == "mlx_core::inference" && line["event"] == "dflash2_prefix")
        .collect();
    (events, offset + complete)
}

fn event_u64(event: &Value, name: &str) -> u64 {
    match &event[name] {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.parse().ok(),
        _ => None,
    }
    .unwrap_or_else(|| panic!("dflash2_prefix has no integer `{name}`: {event}"))
}

async fn run_restore_case(case: RestoreCase) {
    let part = PathBuf::from(
        std::env::var_os(RESTORE_PART_ENV).expect("restore child runs with MLX_BENCH_RESTORE_PART"),
    );
    let log = PathBuf::from(
        std::env::var_os("MLX_NODE_LOG_FILE").expect("restore child runs with MLX_NODE_LOG_FILE"),
    );
    let target =
        std::env::var("MLX_TEST_QWEN38_TARGET_PATH").expect("set MLX_TEST_QWEN38_TARGET_PATH");
    let draft =
        std::env::var("MLX_TEST_QWEN38_DFLASH2_PATH").expect("set MLX_TEST_QWEN38_DFLASH2_PATH");
    let target_dir = Path::new(&target)
        .parent()
        .filter(|_| Path::new(&target).is_file())
        .unwrap_or(Path::new(&target));
    let tokenizer_path = target_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .unwrap_or_else(|error| panic!("load {}: {error}", tokenizer_path.display()));
    let corpus = std::env::var_os("MLX_BENCH_RESTORE_CORPUS").map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("../mlx-sys/mlx/mlx"),
        PathBuf::from,
    );
    let document = corpus_text(
        &tokenizer,
        &corpus,
        case.context - RESTORE_PROMPT_OVERHEAD,
        false,
    );
    let other = corpus_text(&tokenizer, &corpus, RESTORE_OTHER_DOC_TOKENS, true);
    let question = format!("{RESTORE_INTRO}{document}{RESTORE_ACK}");
    let first = vec![user_message(&question)];
    let last = vec![
        user_message(&question),
        message("assistant", RESTORE_REPLY),
        user_message(RESTORE_TASKS[case.task]),
    ];
    let other = vec![user_message(&format!("{OTHER_INTRO}{other}"))];

    let model = Qwen3_5Model::load(
        target,
        Some(Qwen35LoadOptions {
            draft_model_path: Some(draft),
            kv_format: None,
            max_memory_bytes: None,
            max_context_tokens: None,
            decode_share: None,
        }),
    )
    .await
    .expect("load Qwen3.8 target with DFlash2 companion");
    assert!(model.has_mtp_weights(), "DFlash2 companion must be loaded");
    let chat = |messages: Vec<ChatMessage>, cfg: ChatConfig| {
        let model = &model;
        async move {
            model
                .chat_session_start(messages, Some(cfg))
                .await
                .expect("chat turn")
        }
    };

    match case.resume {
        Resume::Cold => {}
        Resume::End => {
            let reply = chat(first, restore_turn_config(8, "none")).await;
            assert_eq!(
                reply.text, RESTORE_REPLY,
                "A1 must answer {RESTORE_REPLY:?} so the final prompt extends its history"
            );
            chat(other, restore_turn_config(8, "none")).await;
        }
        Resume::Capture => {
            // Thinking on: the history leaves the final prompt's
            // `<think>\n\n</think>` right after the generation prefix.
            chat(first, restore_turn_config(8, "medium")).await;
            chat(other, restore_turn_config(8, "none")).await;
        }
    }
    let (_, offset) = prefix_events_since(&log, 0);
    let result = chat(last, restore_turn_config(RESTORE_DECODE_TOKENS, "none")).await;
    let (events, _) = prefix_events_since(&log, offset);
    let perf = result.performance.as_ref().expect("perf");
    print_perf(&case.name(), result.num_tokens, perf);
    let [event] = events.as_slice() else {
        panic!(
            "{}: expected one dflash2_prefix event for the measured turn, got {}",
            case.name(),
            events.len()
        );
    };
    let (cycles, a, c) =
        cycle_split(result.num_tokens, perf).expect("measured turn reports speculative cycles");
    let record = RestoreResult {
        case,
        prompt_tokens: event_u64(event, "prompt_tokens"),
        hit_kind: event["hit_kind"]
            .as_str()
            .expect("dflash2_prefix hit_kind")
            .to_string(),
        hit_tokens: event_u64(event, "hit_tokens"),
        prefilled_tokens: event_u64(event, "prefilled_tokens"),
        ttft_ms: perf.ttft_ms,
        num_tokens: result.num_tokens,
        finish_reason: result.finish_reason.clone(),
        cycles,
        a,
        c,
        text_sha256: Sha256::digest(result.text.as_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
    };
    std::fs::write(
        &part,
        serde_json::to_string_pretty(&record.to_json()).expect("restore result JSON"),
    )
    .unwrap_or_else(|error| panic!("write {}: {error}", part.display()));
    model.shutdown().expect("shutdown");
    assert_eq!(
        record.hit_kind,
        case.resume.hit_kind(),
        "{}: measured turn resumed as {} (miss_reason {})",
        case.name(),
        record.hit_kind,
        event["miss_reason"]
    );
    assert!(
        record.num_tokens >= RESTORE_DECODE_TOKENS,
        "{}: decoded {} tokens (finish {})",
        case.name(),
        record.num_tokens,
        record.finish_reason
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires real Qwen3.8 target and DFlash2 checkpoints"]
async fn qwen38_dflash2_decode_after_restore() {
    match std::env::var(RESTORE_CASE_ENV) {
        Ok(case) => run_restore_case(RestoreCase::parse(&case)).await,
        Err(_) => run_restore_parent(&RestoreOptions::from_env()),
    }
}

// ---------------------------------------------------------------------------
// Model-free checks of the restore harness
// ---------------------------------------------------------------------------

fn sample_result(context: usize, resume: Resume, task: usize, a: f64, c: f64) -> RestoreResult {
    RestoreResult {
        case: RestoreCase {
            context,
            resume,
            task,
        },
        prompt_tokens: context as u64,
        hit_kind: resume.hit_kind().to_string(),
        hit_tokens: 0,
        prefilled_tokens: 0,
        ttft_ms: 0.0,
        num_tokens: RESTORE_DECODE_TOKENS,
        finish_reason: "length".to_string(),
        cycles: 200,
        a,
        c,
        text_sha256: "0".repeat(64),
    }
}

fn passing_results(contexts: &[usize], tasks: usize) -> Vec<RestoreResult> {
    restore_order(contexts, tasks)
        .into_iter()
        .map(|case| sample_result(case.context, case.resume, case.task, 2.5, 60.0))
        .collect()
}

#[test]
fn restore_order_rotates_every_kind_through_every_slot() {
    let contexts = [32_768, 81_920, 163_840];
    let order = restore_order(&contexts, 3);
    assert_eq!(order.len(), 27);
    for task in 0..3 {
        for &context in &contexts {
            let kinds: Vec<Resume> = order
                .iter()
                .filter(|case| case.task == task && case.context == context)
                .map(|case| case.resume)
                .collect();
            assert_eq!(kinds.len(), 3, "{context} t{task}: {kinds:?}");
            for resume in Resume::ALL {
                assert!(kinds.contains(&resume), "{context} t{task}: {kinds:?}");
            }
        }
    }
    for &context in &contexts {
        for slot in 0..3 {
            let mut firsts: Vec<Resume> = (0..3)
                .map(|task| {
                    order
                        .iter()
                        .filter(|case| case.task == task && case.context == context)
                        .nth(slot)
                        .expect("slot")
                        .resume
                })
                .collect();
            firsts.sort_by_key(|resume| resume.name());
            firsts.dedup();
            assert_eq!(firsts.len(), 3, "{context} slot {slot}: {firsts:?}");
        }
    }
    for case in &order {
        assert_eq!(RestoreCase::parse(&case.name()), *case);
    }
}

#[test]
fn restore_verdict_flags_a_drift_c_regression_and_wrong_resume() {
    let contexts = [4096, 8192];
    let ok = passing_results(&contexts, 3);
    assert_eq!(restore_verdict(&ok, &contexts, 3), Vec::<String>::new());

    let mut a_drift = ok.clone();
    let case = a_drift
        .iter_mut()
        .find(|result| result.case.resume == Resume::End)
        .expect("an E case");
    case.a += 1e-9;
    let failures = restore_verdict(&a_drift, &contexts, 3);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(failures[0].contains("A="), "{failures:?}");

    let mut cycles_drift = ok.clone();
    cycles_drift
        .iter_mut()
        .find(|result| result.case.resume == Resume::Capture)
        .expect("a C case")
        .cycles += 1;
    assert_eq!(restore_verdict(&cycles_drift, &contexts, 3).len(), 1);

    // The rule is on means: one task 3% slower is inside, two 3.5% slower
    // are not, and one 10% slower task fails on its own.
    let slow_tasks = |tasks: usize, factor: f64| {
        let mut slow = ok.clone();
        for result in slow.iter_mut().filter(|result| {
            result.case.context == 8192
                && result.case.resume == Resume::Capture
                && result.case.task < tasks
        }) {
            result.c = 60.0 * factor;
        }
        restore_verdict(&slow, &contexts, 3)
    };
    assert_eq!(slow_tasks(1, 1.03), Vec::<String>::new());
    for failures in [slow_tasks(2, 1.035), slow_tasks(1, 1.10)] {
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(failures[0].starts_with("8192 capture"), "{failures:?}");
    }

    // A slot bias that every kind pays once per context cancels in the means.
    let mut rotated = ok.clone();
    for (slot, result) in rotated.iter_mut().enumerate() {
        result.c = 60.0 * [1.0, 1.05, 1.10][slot % 3];
    }
    assert_eq!(
        restore_verdict(&rotated, &contexts, 3),
        Vec::<String>::new()
    );

    let mut inside = ok.clone();
    for result in inside
        .iter_mut()
        .filter(|result| result.case.resume == Resume::End)
    {
        result.c = 60.0 * 1.019;
    }
    assert_eq!(restore_verdict(&inside, &contexts, 3), Vec::<String>::new());

    let mut wrong = ok.clone();
    wrong
        .iter_mut()
        .find(|result| result.case.resume == Resume::End)
        .expect("an E case")
        .hit_kind = "C".to_string();
    let failures = restore_verdict(&wrong, &contexts, 3);
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(failures[0].contains("hit_kind C"), "{failures:?}");

    let mut short = ok.clone();
    short[0].num_tokens = RESTORE_DECODE_TOKENS - 1;
    assert_eq!(restore_verdict(&short, &contexts, 3).len(), 1);

    let missing = &ok[1..];
    let failures = restore_verdict(missing, &contexts, 3);
    assert!(
        failures
            .iter()
            .any(|failure| failure.ends_with("no result")),
        "{failures:?}"
    );
}

#[test]
fn restore_result_round_trips_through_json() {
    let result = sample_result(81_920, Resume::Capture, 2, 2.4190476190476193, 58.21);
    assert_eq!(RestoreResult::from_json(&result.to_json()), result);
}
