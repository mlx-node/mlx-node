// S17 product chat lane. Tiny: the backend continues a session on its own
// history (engram rows and logits == a fresh prompt), cancels a prefill
// without touching the session, and runs whole turns through the engine with
// the GGUF template. Real (`continued_turn_engram_ids`): two engine turns in
// chat mode on the 9 shards; the second is a prefix hit whose engram rows equal
// the fresh prompt's. Run with `MLX_ENABLE_TF32=0`.

fn s17_template() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/tokenizer/fixtures/hf/deepseek-v4.1-06cada875e3a.jinja"
    ))
    .unwrap()
}

fn s17_msg(role: &str, content: &str) -> crate::tokenizer::ChatMessage {
    serde_json::from_value(serde_json::json!({"role": role, "content": content})).unwrap()
}

/// The tiny GGUF's embedded tokenizer (it carries no template) with the
/// DeepSeek-V4.1 GGUF template.
fn s17_tiny_inner() -> super::chat::Inner {
    use crate::utils::gguf::embedded_gpt2_tokenizer_json;
    let (act, fq) = mode_of("fp32_off");
    let path = fixture("tiny-q8_0.gguf");
    let store = super::store::Store::open_dense(
        std::slice::from_ref(&path),
        act,
        fq,
        super::store::CachePlan::Off,
        super::store::DenseMode::Exact,
    )
    .unwrap();
    let g = parse_gguf(&path).unwrap();
    let json = embedded_gpt2_tokenizer_json(&g.metadata).unwrap().unwrap();
    let tok = tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&json).unwrap()).unwrap();
    let tok = crate::tokenizer::Qwen3Tokenizer::from_tokenizer(tok, Some(s17_template()));
    let mut inner = super::chat::Inner::new(store, tok, 1, Default::default());
    inner.probe = Some((Vec::new(), Vec::new()));
    inner
}

/// Every call's engram rows equal the rows a fresh prompt of the session's
/// whole history computes at those positions. Returns the checked calls.
fn s17_check_engram(inner: &super::chat::Inner) -> usize {
    let cfg = inner.store.model.cfg.engram.as_ref().unwrap();
    let h = super::engram_hash::EngramHasher::new(cfg).unwrap();
    let (calls, prompts) = inner.probe.as_ref().unwrap();
    let mut n = 0;
    for (k, full) in prompts {
        // the prefill call and the decode calls that follow it in the turn
        let end = prompts
            .iter()
            .map(|(i, _)| *i)
            .find(|i| i > k)
            .unwrap_or(calls.len());
        let mut whole = full.clone();
        for (start, ids, rows) in &calls[*k..end] {
            if *start + ids.len() > whole.len() {
                whole.extend_from_slice(ids);
            }
            assert_eq!(&whole[*start..*start + ids.len()], ids.as_slice());
            let fresh = h.rows(&whole[..*start + ids.len()], 0);
            let w = fresh.len() / (*start + ids.len());
            assert_eq!(rows.as_slice(), &fresh[*start * w..], "engram rows at {start}");
            n += 1;
        }
    }
    n
}

#[test]
fn tiny_chat_lane_continues_session() {
    use crate::engine::backend::{ChatBackend, ResetScope, SaveStateArgs};
    use crate::stream::Stream;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    let mut inner = s17_tiny_inner();
    let flag = Arc::new(AtomicBool::new(false));
    let p1: Vec<u32> = vec![0, 507, 40, 41, 42, 43, 44, 508, 510];
    let generated = [45u32, 46];
    let p2_tail: Vec<u32> = vec![47, 1, 507, 48, 49, 508, 510];

    // turn 1: prompt + two one-token calls, then save (reuse)
    inner.set_turn_cancel_flag(Some(Arc::clone(&flag)));
    inner.prefill(&p1, Stream::generation()).unwrap();
    for t in generated {
        inner.prefill(&[t], Stream::generation()).unwrap();
    }
    inner.save_cache_state(SaveStateArgs {
        reuse_cache: true,
        is_delta: false,
        has_images: false,
        generated_tokens: &generated,
        finish_reason: "length",
        save_tokens: &p1,
        save_expanded_tokens: None,
        image_cache_key: 0,
    });
    inner.set_turn_cancel_flag(None);
    let saved: Vec<u32> = p1.iter().chain(&generated).copied().collect();
    assert_eq!(inner.cached_token_history(), saved.as_slice());

    // turn 2: a hit continues on the saved history
    let full: Vec<u32> = saved.iter().chain(&p2_tail).copied().collect();
    assert_eq!(inner.verify_cache_prefix(&full, true), saved.len());
    assert_eq!(inner.verify_cache_prefix(&full, false), 0);
    assert_eq!(inner.verify_cache_prefix(&saved, true), 0, "exact match is a miss");

    // a cancelled prefill leaves the session as it was
    flag.store(true, Ordering::Relaxed);
    inner.set_turn_cancel_flag(Some(Arc::clone(&flag)));
    let err = inner.prefill(&p2_tail, Stream::generation()).err().unwrap();
    assert_eq!(err.reason, "prefill cancelled");
    assert_eq!(inner.verify_cache_prefix(&full, true), saved.len());
    inner.set_turn_cancel_flag(None);
    flag.store(false, Ordering::Relaxed);

    inner.set_turn_cancel_flag(Some(Arc::clone(&flag)));
    let cont = read(&inner.prefill(&p2_tail, Stream::generation()).unwrap());
    inner.set_turn_cancel_flag(None);
    inner.reset_caches(ResetScope::Command).unwrap();
    assert!(inner.cached_token_history().is_empty());
    inner.set_turn_cancel_flag(Some(Arc::clone(&flag)));
    let fresh = read(&inner.prefill(&full, Stream::generation()).unwrap());
    inner.set_turn_cancel_flag(None);
    let (num, den) = cont.iter().zip(&fresh).fold((0f64, 0f64), |(n, d), (a, b)| {
        (n + f64::from(a - b).powi(2), d + f64::from(*b).powi(2))
    });
    let rel = (num / den).sqrt();
    println!("continued vs fresh last-row logits relL2 {rel:e}");
    assert!(rel < 1e-4, "{rel}");
    let (calls, _) = inner.probe.as_ref().unwrap();
    let cont_call = &calls[calls.len() - 2];
    assert_eq!((cont_call.0, cont_call.1.as_slice()), (saved.len(), p2_tail.as_slice()));
    let fresh_rows = &calls[calls.len() - 1].2;
    let w = fresh_rows.len() / full.len();
    assert_eq!(cont_call.2, fresh_rows[saved.len() * w..], "engram rows");
    assert_eq!(s17_check_engram(&inner), 6);

    // whole turns through the engine (greedy, chat mode)
    inner.reset_caches(ResetScope::Command).unwrap();
    inner.probe = Some((Vec::new(), Vec::new()));
    let cfg = crate::engine::types::ChatConfig {
        max_new_tokens: Some(3),
        temperature: Some(0.0),
        reasoning_effort: Some("none".into()),
        ..Default::default()
    };
    let r1 = crate::engine::session::session_start(
        &mut inner,
        vec![s17_msg("user", "hello")],
        cfg.clone(),
        &flag,
    )
    .unwrap();
    let mut m2 = vec![s17_msg("user", "hello")];
    m2.push(s17_msg("assistant", &r1.text));
    m2.push(s17_msg("user", "more"));
    let r2 = crate::engine::session::session_continue(&mut inner, m2, cfg, &flag).unwrap();
    let (_, prompts) = inner.probe.as_ref().unwrap();
    println!(
        "engine turns: {} + {} tokens, turn-2 prefill from {} of {}",
        r1.num_tokens,
        r2.num_tokens,
        prompts[1].1.len() - inner.probe.as_ref().unwrap().0[prompts[1].0].1.len(),
        prompts[1].1.len()
    );
    assert_eq!(prompts.len(), 2);
    assert!(s17_check_engram(&inner) >= 2);
}

/// Real 9 shards, expert cache 0: two engine turns in chat mode (turn 1 one
/// token, so the saved history is the prompt); turn 2 must be a prefix hit
/// whose call starts at the saved history and whose engram rows equal those
/// of the fresh turn-2 prompt (the template's own render, tokenized).
#[test]
#[ignore = "real 9 shards; MLX_ENABLE_TF32=0 MLX_DSV41_EXPERT_CACHE_GB=0 DSV41_DIR; ~18 GiB"]
fn continued_turn_engram_ids() {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    assert_eq!(std::env::var("MLX_ENABLE_TF32").as_deref(), Ok("0"), "S5-1");
    assert_eq!(std::env::var("MLX_DSV41_EXPERT_CACHE_GB").as_deref(), Ok("0"));
    crate::array::set_cache_limit(super::store::POOL_LIMIT as f64).unwrap();
    let dir = std::path::PathBuf::from(std::env::var("DSV41_DIR").unwrap());
    let t0 = std::time::Instant::now();
    let mut inner = super::chat::Inner::load(&dir, Default::default()).unwrap();
    println!(
        "load {:.1} s, dense {:?}, cache slots {}, admission {:?}",
        t0.elapsed().as_secs_f64(),
        inner.store.dense_mode,
        inner.store.cache_slots(),
        inner.store.admission().map(|a| a.to_string())
    );
    assert_eq!(inner.store.cache_slots(), 0);
    inner.probe = Some((Vec::new(), Vec::new()));
    let flag = Arc::new(AtomicBool::new(false));
    let cfg = crate::engine::types::ChatConfig {
        max_new_tokens: Some(1),
        temperature: Some(0.0),
        reasoning_effort: Some("none".into()),
        ..Default::default()
    };
    let r1 = crate::engine::session::session_start(
        &mut inner,
        vec![s17_msg("user", "Hi")],
        cfg.clone(),
        &flag,
    )
    .unwrap();
    let saved = crate::engine::backend::ChatBackend::cached_token_history(&inner).to_vec();
    let m2 = || {
        vec![
            s17_msg("user", "Hi"),
            s17_msg("assistant", &r1.text),
            s17_msg("user", "Bye"),
        ]
    };
    let tok = crate::engine::backend::ChatBackend::tokenizer(&inner).unwrap();
    let want = tok
        .apply_chat_template_with_config(
            &m2(),
            true,
            &cfg,
            crate::engine::session::SESSION_PRESERVE_THINKING,
        )
        .unwrap();
    let r2 = crate::engine::session::session_continue(&mut inner, m2(), cfg, &flag).unwrap();
    let (calls, prompts) = inner.probe.as_ref().unwrap();
    assert_eq!(prompts.len(), 2);
    let (k, full) = &prompts[1];
    let (start, ids, rows) = &calls[*k];
    println!(
        "turn 1 {:?} ({} prompt ids, {} saved), turn 2 {:?}: call at {start} with {} ids of {}",
        r1.text,
        prompts[0].1.len(),
        saved.len(),
        r2.text,
        ids.len(),
        want.len()
    );
    assert_eq!(*start, saved.len(), "turn 2 continues on the saved history");
    assert!(*start > 0);
    assert_eq!(full, &want, "session history + call == the rendered prompt");
    let cfg_e = inner.store.model.cfg.engram.as_ref().unwrap();
    let h = super::engram_hash::EngramHasher::new(cfg_e).unwrap();
    let fresh = h.rows(&want, 0);
    let w = fresh.len() / want.len();
    assert_eq!(rows.as_slice(), &fresh[start * w..], "engram rows == fresh prompt");
    println!(
        "continued_turn_engram_ids: {} delta rows x {w} engram ids == fresh; {} calls checked",
        ids.len(),
        s17_check_engram(&inner)
    );
}
