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
//! interpretable alongside both. `d<N>` is the configured `mtp_depth`
//! label — the actual proposal width is `mtp_mean_depth`.

use mlx_core::engine::types::ChatConfig;
use mlx_core::models::qwen3_5::model::{Qwen3_5Model, Qwen35LoadOptions};
use mlx_core::profiling::PerformanceMetrics;
use mlx_core::tokenizer::ChatMessage;

fn user_message(content: &str) -> ChatMessage {
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
    // Speculative-cycle decomposition: A = committed tokens/cycle,
    // C = decode ms/cycle; decode_tps ≈ 1000·A/C. decode_ms is derived
    // from tokens and decode_tps (the only wall-clock the metrics carry).
    if let (Some(cycles), Some(a)) = (perf.mtp_cycles, perf.mtp_mean_accepted_tokens_total)
        && cycles > 0
        && perf.decode_tokens_per_second > 0.0
    {
        let decode_ms = num_tokens as f64 / perf.decode_tokens_per_second * 1000.0;
        let c = decode_ms / cycles as f64;
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
                mtp_depth: Some(4),
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

    // --- Decode: medium prompt, long generation, depth sweep -------------
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
    let depths: Vec<i32> = std::env::var("MLX_BENCH_DEPTHS")
        .map(|s| s.split(',').filter_map(|d| d.trim().parse().ok()).collect())
        .unwrap_or_else(|_| vec![4, 8]);
    for depth in depths {
        for rep in 0..3 {
            model.reset_caches().await.expect("reset");
            let r = chat(
                medium_prompt.clone(),
                ChatConfig {
                    max_new_tokens: Some(256),
                    temperature: Some(0.0),
                    report_performance: Some(true),
                    enable_mtp: Some(true),
                    mtp_depth: Some(depth),
                    reasoning_effort: Some("none".to_string()),
                    ..ChatConfig::default()
                },
            )
            .await;
            print_perf(
                &format!("decode_dflash2_d{depth} rep{rep}"),
                r.num_tokens,
                r.performance.as_ref().expect("perf"),
            );
            if rep == 0 {
                // char-boundary truncation: byte-slicing panics when 400
                // lands inside a multibyte char (non-ASCII completions).
                let text: String = r.text.chars().take(400).collect();
                eprintln!("[bench] d{depth} finish={} text: {}", r.finish_reason, text);
            }
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

    model.shutdown_for_test().expect("shutdown");
}
