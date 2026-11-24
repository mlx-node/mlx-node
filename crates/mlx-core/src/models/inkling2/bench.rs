//! Reusable release-mode roofline benchmark & per-stage GPU micro-profiler for Inkling-2-Mini.
//!
//! Run on M5 Max (release mode so CPU host overhead is real release speed):
//! ```bash
//! cargo test --release -p mlx-core --lib models::inkling2::bench::inkling2_roofline_profile -- --ignored --nocapture
//! ```

use std::path::Path;
use std::time::Instant;

use crate::array::attention::scaled_dot_product_attention;
use crate::array::{DType, MxArray, clear_cache, synchronize, synchronize_and_clear_cache};
use crate::engine::extract_chat_params;
use crate::engine::spec_paged::SpecPagedCache;
use crate::engine::types::ChatConfig;
use crate::nn::Activations;
use crate::stream::WiredResidency;
use crate::tools::inkling::{
    Inkling2RenderOptions, InklingMessage, render_inkling2_messages_to_ids,
};
use crate::transformer::paged_kv_cache_adapter::{PagedKVCacheAdapter, PagedRaggedRow, SeqId};

use super::attention::global_log_scale_tau_for_positions;
use super::attn_res::{ATTN_RES_EPS, apply_sublayer_rms_norm};
use super::dspark::{
    AdaptiveDraftDepth, BatchedDsparkLane, Inkling2DsparkContext, Inkling2DsparkModel,
};
use super::kernels::{
    Inkling2KvDtype, inkling2_block_attn_res, inkling2_gate_sink, inkling2_grouped_moe_available,
    inkling2_lm_head_softcap_argmax, inkling2_markov_fold_softcap_argmax, inkling2_markov_gate,
    inkling2_moe_combine, inkling2_moe_route, inkling2_moe_router_proj, inkling2_paged_attention,
    inkling2_paged_attention_ex, inkling2_qr_clip, inkling2_sconv_k4_kv_prep,
    inkling2_sconv_k4_kv_prep_paged, inkling2_sconv_k4_ragged, inkling2_situ,
};
use super::model::{AttnInputNorm, Inkling2Inner};
use super::persistence::{load_from_dir, test_real_model_path};
use super::short_conv::VisitShortConvState;

const WARMUP_RUNS: usize = 2;
const DECODE_OUT_TOKENS: usize = 128;

const SINGLE_PROMPT: &str = "\
Write a complete, idiomatic Rust implementation of Dijkstra's shortest path algorithm using \
`std::collections::BinaryHeap`. Include a `Graph` struct, edge relaxation, predecessor path \
reconstruction, complexity analysis, and thorough unit tests covering disconnected graphs and \
multi-edge shortest paths.";

const CODE_COMPLETION_PROMPT: &str = "\
Write a complete, idiomatic Rust function `fn binary_search(slice: &[i32], target: i32) -> Option<usize>` \
using iterative bounds, with doc comments and a unit test.";

#[derive(Debug, Clone, Copy)]
enum DsparkDepthMode {
    Fixed(usize),
    Adaptive,
}

const FOUR_LANE_PROMPTS: [&str; 4] = [
    SINGLE_PROMPT,
    "Write a complete, idiomatic Rust implementation of a generic thread-safe LRU cache with \
O(1) get and put operations, capacity eviction, doc comments, and unit tests.",
    "Write a detailed technical explanation and complete Python implementation of an A* pathfinding \
engine on a weighted 2D grid with diagonal movement, tie-breaking heuristics, and path reconstruction.",
    "Write a complete TypeScript implementation of a bounded async work queue with concurrency limits, \
cancellation via AbortSignal, per-task retries with exponential backoff, and full type safety.",
];

#[inline]
fn eval_sync(arrays: &[&MxArray]) {
    MxArray::eval_arrays(arrays).expect("eval_arrays");
    synchronize();
}

fn eval_owned_sync(arrays: &[MxArray]) {
    let refs: Vec<&MxArray> = arrays.iter().collect();
    if !refs.is_empty() {
        MxArray::eval_arrays(&refs).expect("eval_arrays");
    }
    synchronize();
}

fn render_prompt(inner: &Inkling2Inner, text: &str) -> Vec<u32> {
    let tokenizer = inner.tokenizer.as_ref().expect("tokenizer loaded");
    let opts = Inkling2RenderOptions {
        reasoning_effort: 0.0,
        ..Default::default()
    };
    render_inkling2_messages_to_ids(&[InklingMessage::user(text)], tokenizer.as_ref(), &opts)
        .expect("render prompt")
}

fn build_long_prompt(inner: &Inkling2Inner, target_tokens: usize) -> Vec<u32> {
    let mut text = String::with_capacity(target_tokens * 8);
    while text.len() < target_tokens * 8 {
        text.push_str(
            "The Rust borrow checker enforces aliasing XOR mutability at compile time, preventing \
             data races across concurrent threads without runtime garbage collection overhead. ",
        );
    }
    let mut ids = render_prompt(inner, &text);
    assert!(
        ids.len() >= target_tokens,
        "build_long_prompt produced {} < {target_tokens} tokens",
        ids.len()
    );
    ids.truncate(target_tokens);
    ids
}

fn greedy_chat_params() -> crate::engine::params::ChatParams {
    extract_chat_params(&ChatConfig {
        temperature: Some(0.0),
        top_p: Some(1.0),
        top_k: Some(0),
        ..ChatConfig::default()
    })
}

// ============================================================================
// Part 1: End-to-End Release Throughput Benchmarks
// ============================================================================

#[derive(Debug, Clone)]
struct ArSingleResult {
    tokens: Vec<u32>,
    steps: usize,
    total_ms: f64,
    host_build_ms: f64,
    gpu_eval_wait_ms: f64,
    settle_ms: f64,
}

fn run_single_ar(
    inner: &mut Inkling2Inner,
    prompt_ids: &[u32],
    out_tokens: usize,
) -> ArSingleResult {
    let seq_id = 0u32;
    inner.reset_paged_sequence(seq_id).expect("reset seq 0");
    let (prefill_logits, _) = inner
        .paged_prefill_sequence(seq_id, prompt_ids, None)
        .expect("prefill");
    let mut next_tok = prefill_logits
        .argmax(-1, Some(false))
        .expect("argmax")
        .item_at_int32(0)
        .expect("item") as u32;
    synchronize();

    let mut generated = Vec::with_capacity(out_tokens);
    generated.push(next_tok);

    let steps = out_tokens.saturating_sub(1);
    let mut host_build_ms = 0.0;
    let mut gpu_eval_wait_ms = 0.0;
    let mut settle_ms = 0.0;
    let t_total = Instant::now();
    for step in 0..steps {
        let t_build = Instant::now();
        let argmax = inner
            .paged_decode_step_greedy(seq_id, next_tok)
            .expect("ar step");
        host_build_ms += t_build.elapsed().as_secs_f64() * 1000.0;

        let t_eval = Instant::now();
        next_tok = argmax.item_at_int32(0).expect("item") as u32;
        gpu_eval_wait_ms += t_eval.elapsed().as_secs_f64() * 1000.0;

        let t_settle = Instant::now();
        let _ = inner.settle_paged_step(seq_id).expect("settle");
        settle_ms += t_settle.elapsed().as_secs_f64() * 1000.0;

        generated.push(next_tok);
        crate::array::maybe_clear_cache_for_paged_step(step as i32);
    }
    synchronize();
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;

    ArSingleResult {
        tokens: generated,
        steps,
        total_ms,
        host_build_ms,
        gpu_eval_wait_ms,
        settle_ms,
    }
}

#[derive(Debug, Clone)]
struct DsparkSingleResult {
    tokens: Vec<u32>,
    decoded_tokens: usize,
    cycles: usize,
    total_drafted: usize,
    total_accepted_drafts: usize,
    total_ms: f64,
    propose_ms: f64,
    target_verify_ms: f64,
    commit_ms: f64,
}

fn run_single_dspark(
    inner: &mut Inkling2Inner,
    prompt_ids: &[u32],
    out_tokens: usize,
    depth_mode: DsparkDepthMode,
) -> DsparkSingleResult {
    let seq_id = 0u32;
    inner.reset_paged_sequence(seq_id).expect("reset seq 0");
    let (tap_visits, n_draft_layers, max_spec) = {
        let d = inner.dspark.as_ref().expect("dspark loaded");
        (
            d.config.target_layer_ids.clone(),
            d.config.num_hidden_layers,
            d.config.num_draft_tokens(),
        )
    };
    let mut draft_ctx = Inkling2DsparkContext::new(n_draft_layers);
    let mut adaptive_ctrl = AdaptiveDraftDepth::new(max_spec);
    let (first_logits, _) = inner
        .paged_prefill_sequence_with_taps(
            seq_id,
            prompt_ids,
            None,
            &tap_visits,
            |draft, taps, len| draft_ctx.append_taps(draft, taps, len),
        )
        .expect("dspark prefill");
    draft_ctx.eval_state().expect("eval draft ctx");
    let mut anchor = first_logits
        .argmax(-1, Some(false))
        .expect("argmax")
        .item_at_int32(0)
        .expect("item") as u32;
    synchronize();

    let params = greedy_chat_params();
    let mut rng = rand::rng();
    let mut generated = Vec::with_capacity(out_tokens);
    generated.push(anchor);

    let mut cycles = 0usize;
    let mut total_drafted = 0usize;
    let mut total_accepted_drafts = 0usize;
    let mut propose_ms = 0.0;
    let mut target_verify_ms = 0.0;
    let mut commit_ms = 0.0;

    let t_total = Instant::now();
    while generated.len() < out_tokens {
        let remaining = out_tokens - generated.len();
        let hard_cap = max_spec.min(remaining.saturating_sub(1));
        let cap = match depth_mode {
            DsparkDepthMode::Fixed(k) => k.min(hard_cap),
            DsparkDepthMode::Adaptive => adaptive_ctrl.current_depth(hard_cap),
        };

        // 1. Propose (stays device-resident; zero GPU->CPU sync before verify!)
        let t_prop = Instant::now();
        let mut prop_opt = if cap > 0 {
            let draft = inner.dspark.as_ref().expect("dspark");
            Some(
                draft_ctx
                    .propose_block(
                        draft,
                        &inner.embed_tokens,
                        &inner.embed_norm,
                        &inner.lm_head,
                        anchor,
                        cap,
                        &params,
                        &mut rng,
                    )
                    .expect("propose_block"),
            )
        } else {
            None
        };
        let verify_len = 1 + cap;
        let device_verify_ids = if let Some(ref prop) = prop_opt
            && let Some(ref dev_drafts) = prop.device_draft_ids
        {
            let anchor_arr = MxArray::from_int32(&[anchor as i32], &[1]).expect("anchor");
            let v_ids = MxArray::concatenate(&anchor_arr, dev_drafts, 0).expect("concat");
            // Kick ONE async_eval right after propose_block returns (before target verify
            // builds its 48-visit graph) so GPU executes `commit (append_taps) + propose_block`
            // in ONE continuous Metal CB overlapped with CPU graph build!
            MxArray::async_eval_arrays(&[&v_ids]);
            Some(v_ids)
        } else {
            None
        };
        propose_ms += t_prop.elapsed().as_secs_f64() * 1000.0;

        // 2. Fused propose+verify GPU pipeline + fused `inkling2_lm_head_softcap_argmax`
        //    (ONE single eval per cycle, mirroring production `Inkling2DsparkStepper::verify_device_greedy`!)
        let t_ver = Instant::now();
        let (first_pos, ticket, used_placeholders) = {
            let rt = inner.paged.as_mut().expect("paged");
            let pos = rt.coordinator.request_token_count_all(seq_id).expect("pos");
            let (rec_ids, placeholders) = if device_verify_ids.is_some() {
                (vec![u32::MAX; verify_len], true)
            } else {
                let mut ids = Vec::with_capacity(verify_len);
                ids.push(anchor);
                if let Some(ref mut p) = prop_opt {
                    p.materialize_draft_ids().expect("materialize");
                    ids.extend(p.draft_ids.iter().map(|&id| id as u32));
                }
                (ids, false)
            };
            let ticket = rt
                .coordinator
                .record_verify(seq_id, &rec_ids)
                .expect("record_verify");
            (pos, ticket, placeholders)
        };
        let row = PagedRaggedRow {
            seq_id,
            first_logical_position: first_pos,
            query_len: verify_len as u32,
        };
        let (argmax, taps, mut tapes) = if let Some(ref dev_ids) = device_verify_ids {
            inner
                .paged_verify_ragged_taped_device(&[row], dev_ids, &tap_visits, true)
                .expect("verify_device_greedy")
        } else {
            let mut host_ids = Vec::with_capacity(verify_len);
            host_ids.push(anchor);
            if let Some(ref p) = prop_opt {
                host_ids.extend(p.draft_ids.iter().map(|&id| id as u32));
            }
            inner
                .paged_verify_ragged_taped_greedy(&[(row, host_ids)], &tap_visits)
                .expect("verify_greedy")
        };
        argmax.eval();
        target_verify_ms += t_ver.elapsed().as_secs_f64() * 1000.0;

        // 3. Accept + commit + async overlap (zero commit sync!)
        let t_com = Instant::now();
        let mut proposal_ids = Vec::with_capacity(cap);
        if let Some(ref mut prop) = prop_opt {
            prop.materialize_draft_ids().expect("materialize drafts");
            for &id in &prop.draft_ids {
                proposal_ids.push(id as u32);
            }
        }
        let mut k = 0usize;
        while k < proposal_ids.len()
            && argmax.item_at_int32(k).expect("am") as u32 == proposal_ids[k]
        {
            k += 1;
        }
        let boundary = argmax.item_at_int32(k).expect("boundary") as u32;
        let mut emit = Vec::with_capacity(k + 1);
        emit.extend_from_slice(&proposal_ids[..k]);
        emit.push(boundary);

        let emit_n = emit.len().min(remaining);
        let keep = 1 + emit_n.min(k);
        let tape = tapes.pop().expect("tape");
        {
            let rt = inner.paged.as_mut().expect("paged");
            if used_placeholders {
                let mut verified_ids = Vec::with_capacity(verify_len);
                verified_ids.push(anchor);
                verified_ids.extend_from_slice(&proposal_ids);
                rt.coordinator
                    .patch_last_recorded_tokens_all(seq_id, &verified_ids)
                    .expect("patch placeholders");
            }
            rt.coordinator
                .commit_cycle(seq_id, ticket, keep)
                .expect("commit_cycle");
            let conv = rt.conv_states.get_mut(&seq_id).expect("conv");
            tape.commit_prefix(keep, conv).expect("tape commit");
            let committed = rt.coordinator.request_token_count_all(seq_id).expect("pos");
            rt.coordinator
                .settle_committed(seq_id, u64::from(committed))
                .expect("settle_committed");
        }
        let draft = inner.dspark.as_ref().expect("dspark");
        draft_ctx
            .append_taps(draft, &taps, keep)
            .expect("append_taps");
        commit_ms += t_com.elapsed().as_secs_f64() * 1000.0;

        if cap > 0 {
            cycles += 1;
            total_drafted += cap;
            total_accepted_drafts += k;
            if matches!(depth_mode, DsparkDepthMode::Adaptive) {
                adaptive_ctrl.observe(cap, k);
            }
        }
        for &tok in &emit[..emit_n] {
            generated.push(tok);
            anchor = tok;
        }
    }
    synchronize();
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    let decoded_tokens = generated.len().saturating_sub(1);

    DsparkSingleResult {
        tokens: generated,
        decoded_tokens,
        cycles,
        total_drafted,
        total_accepted_drafts,
        total_ms,
        propose_ms,
        target_verify_ms,
        commit_ms,
    }
}

#[derive(Debug, Clone)]
struct Ar4LaneResult {
    tokens: Vec<Vec<u32>>,
    total_decoded_tokens: usize,
    total_ms: f64,
    steps: usize,
}

fn run_4lane_ar(
    inner: &mut Inkling2Inner,
    prompts: &[Vec<u32>],
    out_tokens: usize,
) -> Ar4LaneResult {
    let n = prompts.len();
    let mut anchors = Vec::with_capacity(n);
    let mut generated: Vec<Vec<u32>> = Vec::with_capacity(n);
    for (i, prompt) in prompts.iter().enumerate() {
        let seq_id = i as SeqId;
        inner.reset_paged_sequence(seq_id).expect("reset");
        let (logits, _) = inner
            .paged_prefill_sequence(seq_id, prompt, None)
            .expect("prefill");
        let tok = logits
            .argmax(-1, Some(false))
            .expect("argmax")
            .item_at_int32(0)
            .expect("item") as u32;
        anchors.push(tok);
        let mut g = Vec::with_capacity(out_tokens);
        g.push(tok);
        generated.push(g);
    }
    synchronize();

    let steps = out_tokens.saturating_sub(1);
    let t_total = Instant::now();
    for step in 0..steps {
        let batch_rows: Vec<(SeqId, u32)> = (0..n).map(|i| (i as SeqId, anchors[i])).collect();
        let argmax = inner
            .paged_decode_step_batched_greedy(&batch_rows)
            .expect("batched decode greedy");
        argmax.eval();
        for i in 0..n {
            let tok = argmax.item_at_int32(i).expect("item") as u32;
            anchors[i] = tok;
            generated[i].push(tok);
        }
        {
            let rt = inner.paged.as_mut().expect("paged");
            rt.coordinator
                .eval_pending_pool_writes_all()
                .expect("eval pool writes");
            let mut conv_refs: Vec<&MxArray> = Vec::new();
            for i in 0..n {
                let seq_id = i as SeqId;
                if let Some(conv) = rt.conv_states.get(&seq_id) {
                    for v in &conv.visits {
                        conv_refs.push(&v.k);
                        conv_refs.push(&v.v);
                        conv_refs.push(&v.attn_post);
                        conv_refs.push(&v.mlp_post);
                    }
                }
            }
            if !conv_refs.is_empty() {
                MxArray::async_eval_arrays(&conv_refs);
            }
            for i in 0..n {
                let seq_id = i as SeqId;
                let _ = rt.coordinator.prune_sliding_all(seq_id).expect("prune");
            }
        }
        crate::array::maybe_clear_cache_for_paged_step(step as i32);
    }
    synchronize();
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    for i in 0..n {
        if let Some(rt) = inner.paged.as_mut() {
            rt.release_sequence(i as SeqId);
        }
    }

    Ar4LaneResult {
        tokens: generated,
        total_decoded_tokens: n * steps,
        total_ms,
        steps,
    }
}

#[derive(Debug, Clone)]
struct Dspark4LaneResult {
    tokens: Vec<Vec<u32>>,
    total_decoded_tokens: usize,
    total_ms: f64,
    propose_ms: f64,
    verify_ms: f64,
    commit_ms: f64,
    waves: usize,
    full_m32_waves: usize,
    total_drafted: usize,
    total_accepted_drafts: usize,
}

fn run_4lane_dspark(
    inner: &mut Inkling2Inner,
    prompts: &[Vec<u32>],
    out_tokens: usize,
    depth_mode: DsparkDepthMode,
) -> Dspark4LaneResult {
    let n = prompts.len();
    let (tap_visits, n_draft_layers, max_spec) = {
        let d = inner.dspark.as_ref().expect("dspark loaded");
        (
            d.config.target_layer_ids.clone(),
            d.config.num_hidden_layers,
            d.config.num_draft_tokens(),
        )
    };
    let mut draft_ctxs: Vec<Inkling2DsparkContext> = (0..n)
        .map(|_| Inkling2DsparkContext::new(n_draft_layers))
        .collect();
    let mut lane_ctrls: Vec<AdaptiveDraftDepth> =
        (0..n).map(|_| AdaptiveDraftDepth::new(max_spec)).collect();
    let mut anchors = Vec::with_capacity(n);
    let mut generated: Vec<Vec<u32>> = Vec::with_capacity(n);

    for (i, prompt) in prompts.iter().enumerate() {
        let seq_id = i as SeqId;
        inner.reset_paged_sequence(seq_id).expect("reset");
        let ctx_ref = &mut draft_ctxs[i];
        let (logits, _) = inner
            .paged_prefill_sequence_with_taps(
                seq_id,
                prompt,
                None,
                &tap_visits,
                |draft, taps, len| ctx_ref.append_taps(draft, taps, len),
            )
            .expect("dspark prefill");
        ctx_ref.eval_state().expect("eval ctx");
        let tok = logits
            .argmax(-1, Some(false))
            .expect("argmax")
            .item_at_int32(0)
            .expect("item") as u32;
        anchors.push(tok);
        let mut g = Vec::with_capacity(out_tokens);
        g.push(tok);
        generated.push(g);
    }
    synchronize();

    let params = greedy_chat_params();
    let mut propose_ms = 0.0;
    let mut verify_ms = 0.0;
    let mut commit_ms = 0.0;
    let mut waves = 0usize;
    let mut full_m32_waves = 0usize;
    let mut total_drafted = 0usize;
    let mut total_accepted_drafts = 0usize;
    let d = inner.config.hidden_size as i64;

    let t_total = Instant::now();
    loop {
        let active: Vec<usize> = (0..n)
            .filter(|&i| generated[i].len() < out_tokens)
            .collect();
        if active.is_empty() {
            break;
        }
        waves += 1;

        // 1. Batched propose across all active lanes (stays device-resident, zero sync before verify!)
        let t_prop = Instant::now();
        let mut lane_caps: Vec<usize> = Vec::with_capacity(active.len());
        for &lane_idx in &active {
            let remaining = out_tokens - generated[lane_idx].len();
            let hard_cap = max_spec.min(remaining.saturating_sub(1));
            let cap = match depth_mode {
                DsparkDepthMode::Fixed(k) => k.min(hard_cap),
                DsparkDepthMode::Adaptive => lane_ctrls[lane_idx].current_depth(hard_cap),
            };
            lane_caps.push(cap);
        }
        let mut props = {
            let draft = inner.dspark.as_ref().expect("dspark");
            let mut lane_rngs: Vec<rand::rngs::ThreadRng> =
                (0..active.len()).map(|_| rand::rng()).collect();
            let mut rng_refs: Vec<&mut dyn rand::Rng> = lane_rngs
                .iter_mut()
                .map(|r| r as &mut dyn rand::Rng)
                .collect();
            let batch_lanes: Vec<BatchedDsparkLane<'_>> = active
                .iter()
                .zip(&lane_caps)
                .map(|(&lane_idx, &cap)| BatchedDsparkLane {
                    ctx: &draft_ctxs[lane_idx],
                    anchor_id: anchors[lane_idx],
                    max_drafts: cap,
                    params: &params,
                })
                .collect();
            draft
                .propose_blocks_batched(
                    &inner.embed_tokens,
                    &inner.embed_norm,
                    &inner.lm_head,
                    &batch_lanes,
                    &mut rng_refs,
                )
                .expect("propose_blocks_batched")
        };
        let mut lane_verify_lens: Vec<usize> = Vec::with_capacity(active.len());
        let mut device_segments: Vec<MxArray> = Vec::with_capacity(active.len() * 2);
        for (&lane_idx, (&cap, prop)) in active.iter().zip(lane_caps.iter().zip(props.iter())) {
            let anchor_arr = MxArray::from_int32(&[anchors[lane_idx] as i32], &[1]).expect("anc");
            device_segments.push(anchor_arr);
            if let Some(ref dev_ids) = prop.device_draft_ids {
                let dlen = dev_ids.size().expect("sz") as usize;
                lane_verify_lens.push(1 + dlen);
                device_segments.push(dev_ids.clone());
            } else {
                lane_verify_lens.push(1 + cap.min(prop.draft_ids.len()));
                if !prop.draft_ids.is_empty() {
                    let arr = MxArray::from_int32(&prop.draft_ids, &[prop.draft_ids.len() as i64])
                        .expect("drafts");
                    device_segments.push(arr);
                }
            }
        }
        let seg_refs: Vec<&MxArray> = device_segments.iter().collect();
        let stacked_verify_ids = MxArray::concatenate_many(seg_refs, Some(0)).expect("concat_many");
        // Kick ONE async_eval on `stacked_verify_ids` so GPU runs `commit + propose_blocks_batched`
        // in ONE continuous Metal CB while CPU builds the 48-visit target verify wave!
        MxArray::async_eval_arrays(&[&stacked_verify_ids]);
        propose_ms += t_prop.elapsed().as_secs_f64() * 1000.0;

        // 2. Stacked [N * 8] target verify wave + fused `inkling2_lm_head_softcap_argmax`
        //    (ONE single GPU sync per wave!)
        let t_ver = Instant::now();
        let total_rows: usize = lane_verify_lens.iter().sum();
        if total_rows == 32 {
            full_m32_waves += 1;
        }
        let mut ragged_rows = Vec::with_capacity(active.len());
        let mut tickets = Vec::with_capacity(active.len());
        {
            let rt = inner.paged.as_mut().expect("paged");
            for (slot, &lane_idx) in active.iter().enumerate() {
                let seq_id = lane_idx as SeqId;
                let pos = rt.coordinator.request_token_count_all(seq_id).expect("pos");
                let placeholders = vec![u32::MAX; lane_verify_lens[slot]];
                let ticket = rt
                    .coordinator
                    .record_verify(seq_id, &placeholders)
                    .expect("record_verify");
                ragged_rows.push(PagedRaggedRow {
                    seq_id,
                    first_logical_position: pos,
                    query_len: lane_verify_lens[slot] as u32,
                });
                tickets.push(ticket);
            }
        }
        let (wave_argmax, wave_taps, wave_tapes) = inner
            .paged_verify_ragged_taped_device(&ragged_rows, &stacked_verify_ids, &tap_visits, true)
            .expect("verify wave device greedy");
        wave_argmax.eval();
        verify_ms += t_ver.elapsed().as_secs_f64() * 1000.0;

        // 3. Commit accepted prefixes + ShortConv tapes + batched draft taps (async overlap!)
        let t_com = Instant::now();
        let mut row_off = 0usize;
        let mut pending_taps: Vec<(usize, Vec<MxArray>, usize)> = Vec::with_capacity(active.len());
        for ((((slot, &lane_idx), prop), ticket), tape) in active
            .iter()
            .enumerate()
            .zip(props.iter_mut())
            .zip(tickets)
            .zip(wave_tapes)
        {
            prop.materialize_draft_ids().expect("materialize drafts");
            let seq_id = lane_idx as SeqId;
            let len = lane_verify_lens[slot];
            let prop_ids: Vec<u32> = prop.draft_ids.iter().map(|&id| id as u32).collect();
            let mut k = 0usize;
            while k < prop_ids.len()
                && wave_argmax.item_at_int32(row_off + k).expect("am") as u32 == prop_ids[k]
            {
                k += 1;
            }
            let boundary = wave_argmax.item_at_int32(row_off + k).expect("boundary") as u32;
            let mut emit = Vec::with_capacity(k + 1);
            emit.extend_from_slice(&prop_ids[..k]);
            emit.push(boundary);

            let remaining = out_tokens - generated[lane_idx].len();
            let emit_n = emit.len().min(remaining);
            let keep = 1 + emit_n.min(k);

            let mut verified_ids = Vec::with_capacity(len);
            verified_ids.push(anchors[lane_idx]);
            verified_ids.extend_from_slice(&prop_ids);
            {
                let rt = inner.paged.as_mut().expect("paged");
                rt.coordinator
                    .patch_last_recorded_tokens_all(seq_id, &verified_ids)
                    .expect("patch placeholders");
                rt.coordinator
                    .commit_cycle(seq_id, ticket, keep)
                    .expect("commit_cycle");
                let conv = rt.conv_states.get_mut(&seq_id).expect("conv");
                tape.commit_prefix(keep, conv).expect("tape commit");
                let committed = rt.coordinator.request_token_count_all(seq_id).expect("pos");
                rt.coordinator
                    .settle_committed(seq_id, u64::from(committed))
                    .expect("settle");
            }
            let mut lane_taps = Vec::with_capacity(wave_taps.len());
            for tap in &wave_taps {
                lane_taps.push(
                    tap.slice(&[row_off as i64, 0], &[(row_off + len) as i64, d])
                        .expect("slice tap"),
                );
            }
            pending_taps.push((lane_idx, lane_taps, keep));

            if lane_caps[slot] > 0 {
                total_drafted += lane_caps[slot];
                total_accepted_drafts += k;
                if matches!(depth_mode, DsparkDepthMode::Adaptive) {
                    lane_ctrls[lane_idx].observe(lane_caps[slot], k);
                }
            }
            for &tok in &emit[..emit_n] {
                generated[lane_idx].push(tok);
                anchors[lane_idx] = tok;
            }
            row_off += len;
        }
        {
            let draft = inner.dspark.as_ref().expect("dspark");
            // `active` is strictly ascending, so split `draft_ctxs` cleanly into disjoint `&mut`.
            let mut slice = draft_ctxs.as_mut_slice();
            let mut base = 0usize;
            let mut batch_args: Vec<(&mut Inkling2DsparkContext, &[MxArray], usize)> =
                Vec::with_capacity(pending_taps.len());
            for (lane_idx, taps, keep) in &pending_taps {
                let rel = *lane_idx - base;
                let (_, rest) = slice.split_at_mut(rel);
                let (head, tail) = rest.split_at_mut(1);
                batch_args.push((&mut head[0], taps.as_slice(), *keep));
                slice = tail;
                base = *lane_idx + 1;
            }
            Inkling2DsparkContext::append_taps_batched(draft, &mut batch_args)
                .expect("append_taps_batched");
        }
        commit_ms += t_com.elapsed().as_secs_f64() * 1000.0;
    }
    synchronize();
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    let total_decoded_tokens: usize = generated.iter().map(|g| g.len().saturating_sub(1)).sum();
    for i in 0..n {
        if let Some(rt) = inner.paged.as_mut() {
            rt.release_sequence(i as SeqId);
        }
    }

    Dspark4LaneResult {
        tokens: generated,
        total_decoded_tokens,
        total_ms,
        propose_ms,
        verify_ms,
        commit_ms,
        waves,
        full_m32_waves,
        total_drafted,
        total_accepted_drafts,
    }
}

fn measure_prefill(inner: &mut Inkling2Inner, ids: &[u32]) -> (f64, f64) {
    for _ in 0..WARMUP_RUNS {
        inner.reset_paged_sequence(0).expect("reset");
        let (logits, _) = inner.paged_prefill_sequence(0, ids, None).expect("prefill");
        logits.eval();
        synchronize();
    }
    inner.reset_paged_sequence(0).expect("reset");
    let t = Instant::now();
    let (logits, _) = inner.paged_prefill_sequence(0, ids, None).expect("prefill");
    logits.eval();
    synchronize();
    let secs = t.elapsed().as_secs_f64();
    let tps = ids.len() as f64 / secs;
    (secs * 1000.0, tps)
}

// ============================================================================
// Part 2: Capture-then-Batch Per-Stage GPU Micro-Breakdown (1 sync / stage!)
// ============================================================================
//
// Why Capture-then-Batch instead of syncing inside the 48-visit loop?
// Each Metal `synchronize()` adds ~0.155 ms of command-buffer commit+wait floor.
// Syncing 19 times per visit x 48 visits = ~912 syncs = +152 ms of pure sync fog.
// Instead, we capture exact materialized inputs for all 48 visits once, then replay
// ALL 48 visits of each isolated component in ONE Metal command buffer (1 sync per
// component!), revealing the true pipelined GPU execution time of every stage!

#[derive(Default, Clone, Copy)]
struct MoeBreakdown {
    visits: usize,
    router_ms: f64,
    gate_up_qmm_ms: f64,
    situ_ms: f64,
    down_qmm_ms: f64,
    combine_ms: f64,
}

impl MoeBreakdown {
    fn total_ms(&self) -> f64 {
        self.router_ms + self.gate_up_qmm_ms + self.situ_ms + self.down_qmm_ms + self.combine_ms
    }
}

#[derive(Default, Clone)]
struct StageBreakdown {
    // 1. Block AttnRes + Loop Controllers + Input Norm (+ residual adds)
    attn_res_ctrl_norm_ms: f64,
    residual_add_ms: f64,
    // 2. Dense Q8_0 projections
    qrgkv_sink_proj_ms: f64,
    o_proj_ms: f64,
    // 3. ShortConv
    sconv_kv_ms: f64,
    sconv_attn_post_ms: f64,
    sconv_mlp_post_ms: f64,
    // 4. L2-clip Q/R/K + V RMSNorm + rel_bias matmul
    l2_clip_vnorm_ms: f64,
    rel_bias_matmul_ms: f64,
    // 5. PagedKVWrite + PagedAttention + gate_sink
    kv_write_ms: f64,
    paged_attn_ms: f64,
    gate_sink_ms: f64,
    // 6. MoE L0..19 (NVFP4, 44 visits)
    moe_nvfp4: MoeBreakdown,
    // 7. MoE L20..23 (Q8_0, 4 visits)
    moe_q8: MoeBreakdown,
    // 8. Prologue + Epilogue
    embed_ms: f64,
    output_res_norm_ms: f64,
    lm_head_ms: f64,
    softcap_argmax_ms: f64,
    // Reference unsynced full forward pass for comparison
    unsynced_full_ms: f64,
}

impl StageBreakdown {
    fn total_isolated_ms(&self) -> f64 {
        self.attn_res_ctrl_norm_ms
            + self.residual_add_ms
            + self.qrgkv_sink_proj_ms
            + self.o_proj_ms
            + self.sconv_kv_ms
            + self.sconv_attn_post_ms
            + self.sconv_mlp_post_ms
            + self.l2_clip_vnorm_ms
            + self.rel_bias_matmul_ms
            + self.kv_write_ms
            + self.paged_attn_ms
            + self.gate_sink_ms
            + self.moe_nvfp4.total_ms()
            + self.moe_q8.total_ms()
            + self.embed_ms
            + self.output_res_norm_ms
            + self.lm_head_ms
            + self.softcap_argmax_ms
    }
}

struct SimpleGroupMeta {
    block_table_2d: MxArray,
    seq_lens: MxArray,
    cu_seqlens_q: MxArray,
    slot_mapping: MxArray,
    total_queries: u32,
}

fn build_single_seq_meta(
    adapter: &mut PagedKVCacheAdapter,
    seq_id: SeqId,
    first_pos: u32,
    q_len: u32,
) -> SimpleGroupMeta {
    let table = adapter.block_table_for(seq_id).expect("block_table_for");
    let ctx_len = table.num_tokens() as i32;
    let nb = table.num_blocks().max(1);
    let mut bt = vec![0i32; nb];
    for (idx, b) in table.blocks().iter().enumerate() {
        bt[idx] = b.block_id as i32;
    }
    let block_table_2d = MxArray::from_int32(&bt, &[1, nb as i64]).expect("bt2d");
    let seq_lens = MxArray::from_int32(&[ctx_len], &[1]).expect("seq_lens");
    let cu_seqlens_q = MxArray::from_int32(&[0, q_len as i32], &[2]).expect("cu_q");
    let row = PagedRaggedRow {
        seq_id,
        first_logical_position: first_pos,
        query_len: q_len,
    };
    let (slot_mapping, total_queries) = adapter
        .build_ragged_slot_mapping(&[row])
        .expect("build_ragged_slot_mapping");
    eval_sync(&[&block_table_2d, &seq_lens, &cu_seqlens_q, &slot_mapping]);
    SimpleGroupMeta {
        block_table_2d,
        seq_lens,
        cu_seqlens_q,
        slot_mapping,
        total_queries,
    }
}

/// Captured materialized inputs for one visit so every stage can be replayed across all 48 visits
/// in a single Metal command buffer (1 sync per stage, zero per-visit sync inflation!).
struct VisitCapture {
    visit_idx: usize,
    phys_layer: usize,
    group_id: usize,
    phys_layer_ord: u32,
    blocks_before: Vec<MxArray>,
    partial_before: MxArray,
    bank: Option<Vec<MxArray>>,
    vc_before: VisitShortConvState,
    x_q: MxArray,
    x_k_v: Option<(MxArray, MxArray)>,
    msgs: Option<MxArray>,
    q_raw: MxArray,
    r_raw: MxArray,
    gate_raw: MxArray,
    k_raw: MxArray,
    v_raw: MxArray,
    sink_raw: MxArray,
    q_cur: MxArray,
    r_for_rel: MxArray,
    y: MxArray,
    proj_out: MxArray,
    blocks_after_attn: Vec<MxArray>,
    ffn_inp: MxArray,
    x_mlp: MxArray,
    top_k_indices: MxArray,
    x_for_gather: MxArray,
    idx_for_gather: MxArray,
    inv_order: Option<MxArray>,
    gamma: MxArray,
    gate_up: MxArray,
    act_in: MxArray,
    down_out: MxArray,
    moe_out: MxArray,
}

fn time_stage_ms<F>(warmup: usize, iters: usize, mut f: F) -> f64
where
    F: FnMut(),
{
    for _ in 0..warmup {
        f();
    }
    synchronize();
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    total_ms / iters as f64
}

fn profile_48_visits(
    inner: &mut Inkling2Inner,
    ctx_prompt: &[u32],
    probe_tokens: &[u32],
) -> StageBreakdown {
    let m_rows = probe_tokens.len();
    let seq_id = 0u32;
    inner.reset_paged_sequence(seq_id).expect("reset");
    let (logits, _) = inner
        .paged_prefill_sequence(seq_id, ctx_prompt, None)
        .expect("prefill ctx");
    logits.eval();
    synchronize();

    let base_conv = inner
        .snapshot_paged_conv_state(seq_id)
        .expect("base conv snapshot");
    let base_pos = inner
        .paged
        .as_ref()
        .expect("paged")
        .coordinator
        .request_token_count_all(seq_id)
        .expect("base_pos");

    // 1. Measure unsynced full 48-visit + epilogue latency on these exact probe_tokens (2 warmup, 5 timed)
    let unsynced_full_ms = {
        let mut total = 0.0;
        let iters = 5usize;
        for it in 0..(WARMUP_RUNS + iters) {
            inner.restore_paged_conv_state(seq_id, base_conv.clone());
            let ticket = if m_rows > 1 {
                Some(
                    inner
                        .paged
                        .as_mut()
                        .expect("paged")
                        .coordinator
                        .record_verify(seq_id, probe_tokens)
                        .expect("record_verify"),
                )
            } else {
                None
            };
            let row = PagedRaggedRow {
                seq_id,
                first_logical_position: base_pos,
                query_len: m_rows as u32,
            };
            synchronize();
            let t0 = Instant::now();
            let am = if m_rows == 1 {
                inner
                    .paged_decode_step_greedy(seq_id, probe_tokens[0])
                    .expect("ar greedy")
            } else {
                let (am, _, _) = inner
                    .paged_verify_ragged_taped_greedy(&[(row, probe_tokens.to_vec())], &[])
                    .expect("verify greedy");
                am
            };
            eval_sync(&[&am]);
            let dt = t0.elapsed().as_secs_f64() * 1000.0;
            if it >= WARMUP_RUNS {
                total += dt;
            }
            if let Some(ticket) = ticket {
                inner
                    .paged
                    .as_mut()
                    .expect("paged")
                    .coordinator
                    .commit_cycle(seq_id, ticket, 0)
                    .expect("rollback 0");
            } else {
                inner
                    .paged
                    .as_mut()
                    .expect("paged")
                    .coordinator
                    .rollback_last_tokens_all(seq_id, 1)
                    .expect("rollback 1");
            }
        }
        total / iters as f64
    };

    // 2. Capture exact materialized inputs across all 48 visits
    inner.restore_paged_conv_state(seq_id, base_conv.clone());
    let ticket = inner
        .paged
        .as_mut()
        .expect("paged")
        .coordinator
        .record_verify(seq_id, probe_tokens)
        .expect("record_verify");

    let (group_metas, kv_dtype, page_block_size) = {
        let rt = inner.paged.as_mut().expect("paged");
        let num_groups = rt.coordinator.groups().len();
        let mut metas = Vec::with_capacity(num_groups);
        for gid in 0..num_groups {
            let adapter = rt.coordinator.adapter_mut(gid).expect("adapter");
            metas.push(build_single_seq_meta(
                adapter,
                seq_id,
                base_pos,
                m_rows as u32,
            ));
        }
        (
            metas,
            rt.coordinator.kv_dtype(),
            rt.coordinator.block_size(),
        )
    };

    let positions: Vec<u32> = (0..m_rows as u32).map(|i| base_pos + i).collect();
    let global_tau = global_log_scale_tau_for_positions(
        &positions,
        inner.config.log_scaling_n_floor,
        inner.config.log_scaling_alpha,
    )
    .expect("tau");
    if let Some(ref tau) = global_tau {
        eval_sync(&[tau]);
    }

    let act_dtype = DType::BFloat16;
    let top_k = inner.config.num_experts_per_tok;
    let k_scalar = MxArray::scalar_int(top_k as i32).expect("k_scalar");
    let ids_i32: Vec<i32> = probe_tokens.iter().map(|&id| id as i32).collect();
    let ids_arr = MxArray::from_int32(&ids_i32, &[m_rows as i64]).expect("ids");
    eval_sync(&[&k_scalar, &ids_arr]);

    let raw_emb = inner.embed_tokens.forward(&ids_arr).expect("embed");
    let mut partial = apply_sublayer_rms_norm(
        &raw_emb,
        &inner.embed_norm,
        inner.config.rms_norm_eps as f32,
        act_dtype,
    )
    .expect("embed_norm");
    eval_sync(&[&partial]);

    let visits = inner.config.visits();
    let lc = inner.config.loop_config.clone();
    let core_len = lc.as_ref().map_or(0, |c| c.core_end - c.core_start);
    let block_size_res = inner.config.n_layers_per_attn_res_block;
    let t_i64 = m_rows as i64;
    let d_i64 = inner.config.hidden_size as i64;
    let k_i64 = top_k as i64;
    let total_slots = t_i64 * k_i64;
    let do_sort = total_slots >= 64;

    let mut blocks: Vec<MxArray> = Vec::new();
    let mut prelude_blocks: Vec<MxArray> = Vec::new();
    let mut prelude_partial: Option<MxArray> = None;
    let mut prelude_block_count = 0usize;
    let mut current_bank: Option<Vec<MxArray>> = None;
    let mut captures: Vec<VisitCapture> = Vec::with_capacity(visits.len());

    {
        let layers = &inner.layers;
        let rt = inner.paged.as_mut().expect("paged");
        let mut conv_state = rt.conv_states.remove(&seq_id).expect("conv");

        for (visit_idx, v_meta) in visits.iter().enumerate() {
            if let Some(ref lc_ref) = lc {
                if visit_idx == lc_ref.core_start && prelude_partial.is_none() {
                    prelude_blocks = blocks.clone();
                    prelude_partial = Some(partial.clone());
                    prelude_block_count = prelude_blocks.len();
                } else if v_meta.pass_idx > 0
                    && core_len > 0
                    && visit_idx == lc_ref.core_start + v_meta.pass_idx * core_len
                {
                    let mut bank = Vec::with_capacity(
                        blocks.len().saturating_sub(prelude_block_count + 1) + 1,
                    );
                    bank.extend_from_slice(&blocks[prelude_block_count + 1..]);
                    bank.push(partial.clone());
                    blocks = prelude_blocks.clone();
                    partial = prelude_partial.as_ref().expect("pp").clone();
                    current_bank = Some(bank);
                } else if visit_idx == lc_ref.core_start + lc_ref.num_passes * core_len {
                    current_bank = None;
                }
            }

            let blocks_before = blocks.clone();
            let partial_before = partial.clone();
            let bank_snap = current_bank.clone();

            let layer = &layers[v_meta.phys_layer];
            let route = rt
                .coordinator
                .route_for_visit(visit_idx)
                .expect("route")
                .clone();
            let group_meta = &group_metas[route.group_id];
            let vc = &mut conv_state.visits[visit_idx];
            let vc_before = vc.clone();

            let messages = match (current_bank.as_deref(), &layer.loop_ctrl) {
                (Some(bank), Some(ctrl)) => {
                    let bank_refs: Vec<&MxArray> = bank.iter().collect();
                    let msgs = inkling2_block_attn_res(
                        &bank_refs,
                        None,
                        &ctrl.msg_queries,
                        None,
                        ATTN_RES_EPS,
                        ATTN_RES_EPS,
                        false,
                        DType::Float32,
                    )
                    .expect("msgs");
                    Some((msgs, ctrl))
                }
                _ => None,
            };

            let mut attn_srcs: Vec<&MxArray> = Vec::with_capacity(blocks.len() + 1);
            for b in &blocks {
                attn_srcs.push(b);
            }
            attn_srcs.push(&partial);

            let (x_q, x_k_v_normed) = if let Some((ref msgs, ctrl)) = messages {
                let dest = if let Some(ref q_res) = layer.attn_res_query {
                    inkling2_block_attn_res(
                        &attn_srcs,
                        None,
                        q_res,
                        None,
                        ATTN_RES_EPS,
                        layer.rms_norm_eps,
                        layer.is_local,
                        DType::Float32,
                    )
                    .expect("ar")
                } else {
                    partial.astype(DType::Float32).expect("f32")
                };
                let msgs_qkv = msgs.slice(&[0, 0, 0], &[3, t_i64, d_i64]).expect("s");
                let norm_w = match &layer.attn_inp_norm {
                    AttnInputNorm::Local(w) => w,
                    AttnInputNorm::Global { qkv_norm, .. } => qkv_norm,
                };
                let (xq, xk, xv) = ctrl
                    .forward_qkv_fused_rms_norm(
                        &dest,
                        &msgs_qkv,
                        norm_w,
                        layer.rms_norm_eps,
                        act_dtype,
                    )
                    .expect("qkv_ctrl");
                (xq, Some((xk, xv)))
            } else if let Some(ref q_res) = layer.attn_res_query {
                match &layer.attn_inp_norm {
                    AttnInputNorm::Local(w) => (
                        inkling2_block_attn_res(
                            &attn_srcs,
                            None,
                            q_res,
                            Some(w),
                            ATTN_RES_EPS,
                            layer.rms_norm_eps,
                            true,
                            act_dtype,
                        )
                        .expect("ar"),
                        None,
                    ),
                    AttnInputNorm::Global { qkv_norm, .. } => {
                        let x_qkv = inkling2_block_attn_res(
                            &attn_srcs,
                            None,
                            q_res,
                            Some(qkv_norm),
                            ATTN_RES_EPS,
                            layer.rms_norm_eps,
                            false,
                            act_dtype,
                        )
                        .expect("ar");
                        let xq = x_qkv
                            .slice(&[0, 0, 0], &[1, t_i64, d_i64])
                            .expect("s")
                            .squeeze(Some(&[0]))
                            .expect("sq");
                        let xk = x_qkv
                            .slice(&[1, 0, 0], &[2, t_i64, d_i64])
                            .expect("s")
                            .squeeze(Some(&[0]))
                            .expect("sq");
                        let xv = x_qkv
                            .slice(&[2, 0, 0], &[3, t_i64, d_i64])
                            .expect("s")
                            .squeeze(Some(&[0]))
                            .expect("sq");
                        (xq, Some((xk, xv)))
                    }
                }
            } else {
                match &layer.attn_inp_norm {
                    AttnInputNorm::Local(w) => (
                        apply_sublayer_rms_norm(&partial, w, layer.rms_norm_eps, act_dtype)
                            .expect("n"),
                        None,
                    ),
                    AttnInputNorm::Global {
                        q_norm,
                        k_norm,
                        v_norm,
                        ..
                    } => (
                        apply_sublayer_rms_norm(&partial, q_norm, layer.rms_norm_eps, act_dtype)
                            .expect("qn"),
                        Some((
                            apply_sublayer_rms_norm(
                                &partial,
                                k_norm,
                                layer.rms_norm_eps,
                                act_dtype,
                            )
                            .expect("kn"),
                            apply_sublayer_rms_norm(
                                &partial,
                                v_norm,
                                layer.rms_norm_eps,
                                act_dtype,
                            )
                            .expect("vn"),
                        )),
                    ),
                }
            };

            let attn = &layer.self_attn;
            let q_dim = (attn.num_heads * attn.head_dim) as i64;
            let x_k_v_ref = x_k_v_normed.as_ref().map(|(k, v)| (k, v));
            let (q_raw, r_raw, gate_raw, k_raw, v_raw, sink_raw, dyn_k, dyn_v) = attn
                .project_qrgkv_sink_and_dyn_conv(&x_q, x_k_v_ref)
                .expect("proj");

            let tau_opt = if !attn.is_local {
                global_tau.as_ref()
            } else {
                None
            };
            let (q_cur, r_for_rel) = inkling2_qr_clip(
                &q_raw,
                &r_raw,
                &attn.q_clip,
                &attn.rel_clip,
                tau_opt,
                attn.num_heads,
                attn.head_dim,
                attn.d_rel,
                attn.qk_l2_clip_threshold as f32,
                attn.rel_l2_clip_threshold as f32,
                1e-6,
                DType::BFloat16,
                true,
            )
            .expect("qr_clip");
            let (k_cur, v_cur, kh_next, vh_next) = inkling2_sconv_k4_kv_prep(
                &k_raw,
                &v_raw,
                &attn.k_conv,
                &attn.v_conv,
                dyn_k.as_ref(),
                dyn_v.as_ref(),
                None,
                &[&vc.k],
                &[&vc.v],
                1,
                false,
                attn.num_kv_heads,
                attn.head_dim,
                attn.qk_l2_clip_threshold as f32,
                attn.rms_norm_eps,
                1e-6,
            )
            .expect("kv_prep");
            vc.k = kh_next;
            vc.v = vh_next;
            let rel_bias = r_for_rel.matmul(&attn.rel_bias_t).expect("rb");

            let phys_layer_ord = route.physical_layer_ordinal as u32;
            {
                let adapter = rt.coordinator.adapter_mut(route.group_id).expect("ad");
                adapter
                    .update_keys_values_native_with_slot_mapping(
                        phys_layer_ord,
                        &k_cur,
                        &v_cur,
                        &group_meta.slot_mapping,
                        group_meta.total_queries,
                    )
                    .expect("kvw");
            }
            let (k_pool, v_pool, k_scale, v_scale) = {
                let adapter = rt.coordinator.adapter(route.group_id).expect("ad");
                let kp = adapter.key_pool_array(phys_layer_ord).expect("kp");
                let vp = adapter.value_pool_array(phys_layer_ord).expect("vp");
                let (ks, vs) = adapter.scale_pool_arrays(phys_layer_ord).expect("sp");
                (kp, vp, ks, vs)
            };
            let scale = 1.0_f32 / (attn.head_dim as f32).sqrt();
            let swa_i32 = if attn.is_local {
                attn.sliding_window as i32
            } else {
                0
            };
            let (attn_o, lse) = inkling2_paged_attention(
                &q_cur,
                &k_pool,
                &v_pool,
                &group_meta.block_table_2d,
                &group_meta.seq_lens,
                &group_meta.cu_seqlens_q,
                &k_scale,
                &v_scale,
                Some(&rel_bias),
                scale,
                swa_i32,
                attn.rel_extent as i32,
                page_block_size as i32,
                attn.num_heads as i32,
                attn.num_kv_heads as i32,
                attn.head_dim as i32,
                kv_dtype,
                None,
            )
            .expect("pa");
            let y = inkling2_gate_sink(&attn_o, &gate_raw, &lse, &sink_raw)
                .expect("gs")
                .reshape(&[t_i64, q_dim])
                .expect("r");
            let proj_out = attn.o_proj.forward(&y).expect("op");
            let is_boundary = layer.phys_idx.is_multiple_of(block_size_res);
            let attn_res = if is_boundary { None } else { Some(&partial) };
            let (ffn_inp, ap_hist) = inkling2_sconv_k4_ragged(
                &proj_out,
                &attn.post_conv,
                None,
                None,
                attn_res,
                None,
                &[&vc.attn_post],
                1,
                false,
            )
            .expect("ap");
            vc.attn_post = ap_hist;
            if is_boundary {
                blocks.push(partial.clone());
            }
            let blocks_after_attn = blocks.clone();

            let mut mlp_srcs: Vec<&MxArray> = Vec::with_capacity(blocks.len() + 1);
            for b in &blocks {
                mlp_srcs.push(b);
            }
            mlp_srcs.push(&ffn_inp);
            let x_mlp = if let Some((ref msgs, ctrl)) = messages {
                let dest_mlp = inkling2_block_attn_res(
                    &mlp_srcs,
                    None,
                    &layer.ffn_res_query,
                    None,
                    ATTN_RES_EPS,
                    layer.rms_norm_eps,
                    true,
                    DType::Float32,
                )
                .expect("mar");
                let msg_mlp = msgs
                    .slice(&[3, 0, 0], &[4, t_i64, d_i64])
                    .expect("s")
                    .squeeze(Some(&[0]))
                    .expect("sq");
                ctrl.mlp_ctrl
                    .forward_fused_rms_norm(
                        &dest_mlp,
                        &msg_mlp,
                        &layer.ffn_norm,
                        layer.rms_norm_eps,
                        act_dtype,
                    )
                    .expect("mc")
            } else {
                inkling2_block_attn_res(
                    &mlp_srcs,
                    None,
                    &layer.ffn_res_query,
                    Some(&layer.ffn_norm),
                    ATTN_RES_EPS,
                    layer.rms_norm_eps,
                    true,
                    act_dtype,
                )
                .expect("fn")
            };

            let moe = &layer.moe;
            let x_f32 = x_mlp.astype(DType::Float32).expect("xf");
            let router_logits = x_f32.matmul(&moe.router_weight_t_f32).expect("rl");
            let (top_k_indices, gamma) = inkling2_moe_route(
                &router_logits,
                moe.router_bias_f32.as_ref(),
                moe.down_proj.outer_scale(),
                moe.top_k,
                moe.route_scale as f32,
            )
            .expect("route");
            let (x_for_gather, idx_for_gather, inv_order) = if do_sort {
                let flat_indices = top_k_indices.reshape(&[-1]).expect("fl");
                let order = flat_indices.argsort(Some(-1)).expect("ord");
                let inv = order.argsort(Some(-1)).expect("inv");
                let idx_sorted = flat_indices.take(&order, 0).expect("ts");
                let x_exp = x_mlp.reshape(&[t_i64, 1, d_i64]).expect("xe");
                let tok_idx = order.floor_divide(&k_scalar).expect("fd");
                let x_sorted = x_exp.take(&tok_idx, 0).expect("xs");
                (x_sorted, idx_sorted, Some(inv))
            } else {
                (
                    x_mlp.reshape(&[t_i64, 1, 1, d_i64]).expect("xe"),
                    top_k_indices.clone(),
                    None,
                )
            };
            let gate_up = moe
                .gate_up_proj
                .forward_without_outer_scale(&x_for_gather, &idx_for_gather, do_sort)
                .expect("gu");
            let gu_in = if gate_up.dtype().expect("dt") == act_dtype {
                gate_up.clone()
            } else {
                gate_up.astype(act_dtype).expect("c")
            };
            let outer_arg = moe.gate_up_proj.outer_scale().map(|s| (s, &idx_for_gather));
            let act_in = inkling2_situ(
                &gu_in,
                outer_arg,
                moe.mlp_gate_softcap as f32,
                moe.mlp_up_softcap as f32,
                moe.gate_up_interleaved,
            )
            .expect("situ");
            let down_out = moe
                .down_proj
                .forward_without_outer_scale(&act_in, &idx_for_gather, do_sort)
                .expect("dn");
            let moe_out = inkling2_moe_combine(&down_out, &gamma, inv_order.as_ref(), act_dtype)
                .expect("comb");
            let (next_partial, mp_hist) = inkling2_sconv_k4_ragged(
                &moe_out,
                &moe.post_conv,
                None,
                None,
                Some(&ffn_inp),
                None,
                &[&vc.mlp_post],
                1,
                false,
            )
            .expect("mp");
            vc.mlp_post = mp_hist;
            partial = next_partial;

            // Materialize all captured arrays for this visit in one shot
            let mut to_eval: Vec<&MxArray> = vec![
                &partial,
                &x_q,
                &q_raw,
                &r_raw,
                &gate_raw,
                &k_raw,
                &v_raw,
                &sink_raw,
                &q_cur,
                &r_for_rel,
                &y,
                &proj_out,
                &ffn_inp,
                &x_mlp,
                &top_k_indices,
                &x_for_gather,
                &idx_for_gather,
                &gamma,
                &gate_up,
                &act_in,
                &down_out,
                &moe_out,
                &vc.k,
                &vc.v,
                &vc.attn_post,
                &vc.mlp_post,
            ];
            if let Some((ref xk, ref xv)) = x_k_v_normed {
                to_eval.push(xk);
                to_eval.push(xv);
            }
            if let Some((ref msgs, _)) = messages {
                to_eval.push(msgs);
            }
            if let Some(ref inv) = inv_order {
                to_eval.push(inv);
            }
            MxArray::eval_arrays(&to_eval).expect("eval capture");

            captures.push(VisitCapture {
                visit_idx,
                phys_layer: v_meta.phys_layer,
                group_id: route.group_id,
                phys_layer_ord,
                blocks_before,
                partial_before,
                bank: bank_snap,
                vc_before,
                x_q,
                x_k_v: x_k_v_normed,
                msgs: messages.map(|(m, _)| m),
                q_raw,
                r_raw,
                gate_raw,
                k_raw,
                v_raw,
                sink_raw,
                q_cur,
                r_for_rel,
                y,
                proj_out,
                blocks_after_attn,
                ffn_inp,
                x_mlp,
                top_k_indices,
                x_for_gather,
                idx_for_gather,
                inv_order,
                gamma,
                gate_up,
                act_in,
                down_out,
                moe_out,
            });
        }
        rt.coordinator
            .eval_pending_pool_writes_all()
            .expect("eval writes");
        rt.conv_states.insert(seq_id, conv_state);
    }

    // Epilogue capture
    let mut final_srcs: Vec<&MxArray> = Vec::with_capacity(blocks.len() + 1);
    for b in &blocks {
        final_srcs.push(b);
    }
    final_srcs.push(&partial);
    let normed_out = inkling2_block_attn_res(
        &final_srcs,
        None,
        &inner.output_res_query,
        Some(&inner.output_norm),
        ATTN_RES_EPS,
        inner.config.rms_norm_eps as f32,
        true,
        act_dtype,
    )
    .expect("on");
    let logits_raw = inner
        .lm_head
        .forward_logits_keep_f32(&normed_out)
        .expect("lm");
    let logits_full = if logits_raw.dtype().expect("dt") == DType::Float32 {
        logits_raw
    } else {
        logits_raw.astype(DType::Float32).expect("f32")
    };
    eval_sync(&[&normed_out, &logits_full]);

    // 3. Replay each isolated stage across ALL 48 visits in ONE batch (1 sync/stage!)
    let w = 2usize;
    let iters = 5usize;
    let mut bd = StageBreakdown {
        unsynced_full_ms,
        ..Default::default()
    };
    bd.moe_nvfp4.visits = captures.iter().filter(|c| c.phys_layer < 20).count();
    bd.moe_q8.visits = captures.iter().filter(|c| c.phys_layer >= 20).count();

    // Stage 1a + 1b: Block AttnRes (fused) + Batched Loop Controllers + Input Norm (48 visits, 1 sync)
    bd.attn_res_ctrl_norm_ms = time_stage_ms(w, iters, || {
        let mut outs = Vec::with_capacity(captures.len() * 4);
        for cap in &captures {
            let layer = &inner.layers[cap.phys_layer];
            let messages = match (cap.bank.as_deref(), &layer.loop_ctrl) {
                (Some(bank), Some(ctrl)) => {
                    let refs: Vec<&MxArray> = bank.iter().collect();
                    let msgs = inkling2_block_attn_res(
                        &refs,
                        None,
                        &ctrl.msg_queries,
                        None,
                        ATTN_RES_EPS,
                        ATTN_RES_EPS,
                        false,
                        DType::Float32,
                    )
                    .expect("m");
                    Some((msgs, ctrl))
                }
                _ => None,
            };
            let mut srcs: Vec<&MxArray> = Vec::with_capacity(cap.blocks_before.len() + 1);
            for b in &cap.blocks_before {
                srcs.push(b);
            }
            srcs.push(&cap.partial_before);
            if let Some((ref msgs, ctrl)) = messages {
                let dest = if let Some(ref q_res) = layer.attn_res_query {
                    inkling2_block_attn_res(
                        &srcs,
                        None,
                        q_res,
                        None,
                        ATTN_RES_EPS,
                        layer.rms_norm_eps,
                        layer.is_local,
                        DType::Float32,
                    )
                    .expect("ar")
                } else {
                    cap.partial_before.astype(DType::Float32).expect("f")
                };
                let msgs_qkv = msgs.slice(&[0, 0, 0], &[3, t_i64, d_i64]).expect("s");
                let norm_w = match &layer.attn_inp_norm {
                    AttnInputNorm::Local(w) => w,
                    AttnInputNorm::Global { qkv_norm, .. } => qkv_norm,
                };
                let (xq, xk, xv) = ctrl
                    .forward_qkv_fused_rms_norm(
                        &dest,
                        &msgs_qkv,
                        norm_w,
                        layer.rms_norm_eps,
                        act_dtype,
                    )
                    .expect("qkv");
                outs.extend([xq, xk, xv]);
            } else if let Some(ref q_res) = layer.attn_res_query {
                match &layer.attn_inp_norm {
                    AttnInputNorm::Local(w) => {
                        outs.push(
                            inkling2_block_attn_res(
                                &srcs,
                                None,
                                q_res,
                                Some(w),
                                ATTN_RES_EPS,
                                layer.rms_norm_eps,
                                true,
                                act_dtype,
                            )
                            .expect("ar"),
                        );
                    }
                    AttnInputNorm::Global { qkv_norm, .. } => {
                        outs.push(
                            inkling2_block_attn_res(
                                &srcs,
                                None,
                                q_res,
                                Some(qkv_norm),
                                ATTN_RES_EPS,
                                layer.rms_norm_eps,
                                false,
                                act_dtype,
                            )
                            .expect("ar"),
                        );
                    }
                }
            } else {
                match &layer.attn_inp_norm {
                    AttnInputNorm::Local(w) => {
                        outs.push(
                            apply_sublayer_rms_norm(
                                &cap.partial_before,
                                w,
                                layer.rms_norm_eps,
                                act_dtype,
                            )
                            .expect("n"),
                        );
                    }
                    AttnInputNorm::Global {
                        q_norm,
                        k_norm,
                        v_norm,
                        ..
                    } => {
                        outs.push(
                            apply_sublayer_rms_norm(
                                &cap.partial_before,
                                q_norm,
                                layer.rms_norm_eps,
                                act_dtype,
                            )
                            .expect("q"),
                        );
                        outs.push(
                            apply_sublayer_rms_norm(
                                &cap.partial_before,
                                k_norm,
                                layer.rms_norm_eps,
                                act_dtype,
                            )
                            .expect("k"),
                        );
                        outs.push(
                            apply_sublayer_rms_norm(
                                &cap.partial_before,
                                v_norm,
                                layer.rms_norm_eps,
                                act_dtype,
                            )
                            .expect("v"),
                        );
                    }
                }
            }
            // MLP Block AttnRes + MLP LoopCtrl + FFN Norm
            let mut msrcs: Vec<&MxArray> = Vec::with_capacity(cap.blocks_after_attn.len() + 1);
            for b in &cap.blocks_after_attn {
                msrcs.push(b);
            }
            msrcs.push(&cap.ffn_inp);
            if let Some((ref msgs, ctrl)) = messages {
                let dest_mlp = inkling2_block_attn_res(
                    &msrcs,
                    None,
                    &layer.ffn_res_query,
                    None,
                    ATTN_RES_EPS,
                    layer.rms_norm_eps,
                    true,
                    DType::Float32,
                )
                .expect("mar");
                let mm = msgs
                    .slice(&[3, 0, 0], &[4, t_i64, d_i64])
                    .expect("s")
                    .squeeze(Some(&[0]))
                    .expect("sq");
                outs.push(
                    ctrl.mlp_ctrl
                        .forward_fused_rms_norm(
                            &dest_mlp,
                            &mm,
                            &layer.ffn_norm,
                            layer.rms_norm_eps,
                            act_dtype,
                        )
                        .expect("m"),
                );
            } else {
                outs.push(
                    inkling2_block_attn_res(
                        &msrcs,
                        None,
                        &layer.ffn_res_query,
                        Some(&layer.ffn_norm),
                        ATTN_RES_EPS,
                        layer.rms_norm_eps,
                        true,
                        act_dtype,
                    )
                    .expect("fn"),
                );
            }
        }
        eval_owned_sync(&outs);
    });

    // Stage 1c: FP32 Residual Adds — fused into `sconv_attn_post` & `sconv_mlp_post` in Stage 2!
    bd.residual_add_ms = 0.0;

    // Stage 2a: Dense Q8_0 qrgkv / qrg+k+v + sink (+ fused dyn_conv) projections (48 visits, 1 sync)
    bd.qrgkv_sink_proj_ms = time_stage_ms(w, iters, || {
        let mut outs = Vec::with_capacity(captures.len() * 8);
        for cap in &captures {
            let attn = &inner.layers[cap.phys_layer].self_attn;
            let xkv = cap.x_k_v.as_ref().map(|(k, v)| (k, v));
            let (q, r, g, k, v, s, dk, dv) = attn
                .project_qrgkv_sink_and_dyn_conv(&cap.x_q, xkv)
                .expect("proj");
            outs.extend([q, r, g, k, v, s]);
            if let Some(dk) = dk {
                outs.push(dk);
            }
            if let Some(dv) = dv {
                outs.push(dv);
            }
        }
        eval_owned_sync(&outs);
    });

    // Stage 2b: Dense Q8_0 / BF16 o_proj (48 visits, 1 sync)
    bd.o_proj_ms = time_stage_ms(w, iters, || {
        let mut outs = Vec::with_capacity(captures.len());
        for cap in &captures {
            outs.push(
                super::attention::forward_linear_proj_fast(
                    &inner.layers[cap.phys_layer].self_attn.o_proj,
                    &cap.y,
                )
                .expect("o"),
            );
        }
        eval_owned_sync(&outs);
    });

    // Stage 3a: Fused ShortConv K/V + K L2-clip + V RMSNorm + PagedKVWrite (+ Global dyn_proj)
    bd.sconv_kv_ms = time_stage_ms(w, iters, || {
        let rt = inner.paged.as_mut().expect("paged");
        let mut outs = Vec::with_capacity(captures.len() * 2);
        for cap in &captures {
            let attn = &inner.layers[cap.phys_layer].self_attn;
            #[cfg(target_os = "macos")]
            let gm = &group_metas[cap.group_id];
            let (xk_in, xv_in) = cap
                .x_k_v
                .as_ref()
                .map_or((&cap.x_q, &cap.x_q), |(k, v)| (k, v));
            let dense_bf16 = matches!(
                attn.q_proj,
                crate::models::quantized_linear::LinearProj::Standard(_)
            );
            let dk = attn.k_conv_proj.as_ref().map(|cp| {
                super::attention::forward_linear_proj_fast_ex(cp, xk_in, dense_bf16)
                    .expect("k")
                    .reshape(&[t_i64, attn.num_kv_heads as i64, 4])
                    .expect("r")
            });
            let dv = attn.v_conv_proj.as_ref().map(|cp| {
                super::attention::forward_linear_proj_fast_ex(cp, xv_in, dense_bf16)
                    .expect("v")
                    .reshape(&[t_i64, attn.num_kv_heads as i64, 4])
                    .expect("r")
            });
            #[cfg(target_os = "macos")]
            {
                let adapter = rt.coordinator.adapter_mut(cap.group_id).expect("ad");
                let (kp_in, vp_in, ks_in, vs_in, kv_mode) = adapter
                    .native_fused_write_pool_arrays(cap.phys_layer_ord)
                    .expect("pool");
                let (kp_o, vp_o, ks_o, vs_o, kh, vh) = inkling2_sconv_k4_kv_prep_paged(
                    &cap.k_raw,
                    &cap.v_raw,
                    &attn.k_conv,
                    &attn.v_conv,
                    dk.as_ref(),
                    dv.as_ref(),
                    None,
                    &[&cap.vc_before.k],
                    &[&cap.vc_before.v],
                    &kp_in,
                    &vp_in,
                    &ks_in,
                    &vs_in,
                    &gm.slot_mapping,
                    1,
                    false,
                    attn.num_kv_heads,
                    attn.head_dim,
                    page_block_size as usize,
                    kv_mode,
                    attn.qk_l2_clip_threshold as f32,
                    attn.rms_norm_eps,
                    1e-6,
                )
                .expect("kvp_paged");
                adapter
                    .commit_native_fused_write(cap.phys_layer_ord, kp_o, vp_o, ks_o, vs_o)
                    .expect("commit");
                outs.extend([kh, vh]);
            }
            #[cfg(not(target_os = "macos"))]
            {
                let (kc, vc_o, kh, vh) = inkling2_sconv_k4_kv_prep(
                    &cap.k_raw,
                    &cap.v_raw,
                    &attn.k_conv,
                    &attn.v_conv,
                    dk.as_ref(),
                    dv.as_ref(),
                    None,
                    &[&cap.vc_before.k],
                    &[&cap.vc_before.v],
                    1,
                    false,
                    attn.num_kv_heads,
                    attn.head_dim,
                    attn.qk_l2_clip_threshold as f32,
                    attn.rms_norm_eps,
                    1e-6,
                )
                .expect("kvp");
                outs.extend([kc, vc_o, kh, vh]);
            }
        }
        rt.coordinator.eval_pending_pool_writes_all().expect("ev");
        eval_owned_sync(&outs);
    });

    // Stage 3b: ShortConv attn_post + fused FP32 residual epilogue (48 visits, 1 sync)
    bd.sconv_attn_post_ms = time_stage_ms(w, iters, || {
        let mut outs = Vec::with_capacity(captures.len() * 2);
        for cap in &captures {
            let layer = &inner.layers[cap.phys_layer];
            let attn = &layer.self_attn;
            let res = if layer.phys_idx.is_multiple_of(block_size_res) {
                None
            } else {
                Some(&cap.partial_before)
            };
            let (y, h) = inkling2_sconv_k4_ragged(
                &cap.proj_out,
                &attn.post_conv,
                None,
                None,
                res,
                None,
                &[&cap.vc_before.attn_post],
                1,
                false,
            )
            .expect("ap");
            outs.extend([y, h]);
        }
        eval_owned_sync(&outs);
    });

    // Stage 3c: ShortConv mlp_post + fused FP32 residual epilogue (48 visits, 1 sync)
    bd.sconv_mlp_post_ms = time_stage_ms(w, iters, || {
        let mut outs = Vec::with_capacity(captures.len() * 2);
        for cap in &captures {
            let moe = &inner.layers[cap.phys_layer].moe;
            let (y, h) = inkling2_sconv_k4_ragged(
                &cap.moe_out,
                &moe.post_conv,
                None,
                None,
                Some(&cap.ffn_inp),
                None,
                &[&cap.vc_before.mlp_post],
                1,
                false,
            )
            .expect("mp");
            outs.extend([y, h]);
        }
        eval_owned_sync(&outs);
    });

    // Stage 4a: Fused strided Q + R L2-clip (+ tau) (48 visits, 1 sync)
    bd.l2_clip_vnorm_ms = time_stage_ms(w, iters, || {
        let mut outs = Vec::with_capacity(captures.len() * 2);
        for cap in &captures {
            let attn = &inner.layers[cap.phys_layer].self_attn;
            let tau_opt = if !attn.is_local {
                global_tau.as_ref()
            } else {
                None
            };
            let (qc, rr) = inkling2_qr_clip(
                &cap.q_raw,
                &cap.r_raw,
                &attn.q_clip,
                &attn.rel_clip,
                tau_opt,
                attn.num_heads,
                attn.head_dim,
                attn.d_rel,
                attn.qk_l2_clip_threshold as f32,
                attn.rel_l2_clip_threshold as f32,
                1e-6,
                DType::BFloat16,
                true,
            )
            .expect("qr");
            outs.extend([qc, rr]);
        }
        eval_owned_sync(&outs);
    });

    // Stage 4b: rel_bias matmul — fused on-the-fly into Stage 5b PagedAttention!
    bd.rel_bias_matmul_ms = 0.0;

    // Stage 5a: PagedKVWrite — fused into Stage 3a `sconv_k4_kv_prep_paged`!
    bd.kv_write_ms = 0.0;

    // Stage 5b: Fused PagedAttention + on-the-fly rel_bias + gate_sink (48 visits, 1 sync)
    let pool_views: Vec<(MxArray, MxArray, MxArray, MxArray)> = {
        let rt = inner.paged.as_ref().expect("paged");
        captures
            .iter()
            .map(|cap| {
                let adapter = rt.coordinator.adapter(cap.group_id).expect("ad");
                let kp = adapter.key_pool_array(cap.phys_layer_ord).expect("kp");
                let vp = adapter.value_pool_array(cap.phys_layer_ord).expect("vp");
                let (ks, vs) = adapter.scale_pool_arrays(cap.phys_layer_ord).expect("sp");
                (kp, vp, ks, vs)
            })
            .collect()
    };
    bd.paged_attn_ms = time_stage_ms(w, iters, || {
        let mut outs = Vec::with_capacity(captures.len());
        for (cap, (kp, vp, ks, vs)) in captures.iter().zip(&pool_views) {
            let attn = &inner.layers[cap.phys_layer].self_attn;
            let gm = &group_metas[cap.group_id];
            let scale = 1.0_f32 / (attn.head_dim as f32).sqrt();
            let swa_i32 = if attn.is_local {
                attn.sliding_window as i32
            } else {
                0
            };
            let (y, _) = inkling2_paged_attention_ex(
                &cap.q_cur,
                kp,
                vp,
                &gm.block_table_2d,
                &gm.seq_lens,
                &gm.cu_seqlens_q,
                ks,
                vs,
                Some(&cap.r_for_rel),
                Some(&attn.rel_bias_w),
                Some((&cap.gate_raw, &cap.sink_raw)),
                false,
                scale,
                swa_i32,
                attn.rel_extent as i32,
                page_block_size as i32,
                attn.num_heads as i32,
                attn.num_kv_heads as i32,
                attn.head_dim as i32,
                kv_dtype,
                None,
            )
            .expect("pa");
            outs.push(y);
        }
        eval_owned_sync(&outs);
    });

    // Stage 5c: gate_sink epilogue — fused into Stage 5b PagedAttention!
    bd.gate_sink_ms = 0.0;

    // Direct A/B benchmark: on-the-fly `dot(r, rel_bias_w)` vs sliced `r @ rel_bias_t[:, :eff_ext]`
    let eff_rel_ext = (ctx_prompt.len() + m_rows).min(1024) as i64;
    let ab_sliced_ms = time_stage_ms(w, iters, || {
        let mut outs = Vec::with_capacity(captures.len());
        for (cap, (kp, vp, ks, vs)) in captures.iter().zip(&pool_views) {
            let attn = &inner.layers[cap.phys_layer].self_attn;
            let gm = &group_metas[cap.group_id];
            let scale = 1.0_f32 / (attn.head_dim as f32).sqrt();
            let swa_i32 = if attn.is_local {
                attn.sliding_window as i32
            } else {
                0
            };
            let rb_t_slice = attn
                .rel_bias_t
                .slice(&[0, 0], &[attn.d_rel as i64, eff_rel_ext])
                .expect("sl");
            let rb = cap.r_for_rel.matmul(&rb_t_slice).expect("rb");
            let (y, _) = inkling2_paged_attention_ex(
                &cap.q_cur,
                kp,
                vp,
                &gm.block_table_2d,
                &gm.seq_lens,
                &gm.cu_seqlens_q,
                ks,
                vs,
                Some(&rb),
                None,
                Some((&cap.gate_raw, &cap.sink_raw)),
                false,
                scale,
                swa_i32,
                eff_rel_ext as i32,
                page_block_size as i32,
                attn.num_heads as i32,
                attn.num_kv_heads as i32,
                attn.head_dim as i32,
                kv_dtype,
                None,
            )
            .expect("pa");
            outs.push(y);
        }
        eval_owned_sync(&outs);
    });
    eprintln!(
        "  [A/B rel_bias M={m_rows}, ctx={}]: on-the-fly dot = {:.3} ms | sliced matmul(ext={eff_rel_ext}) + attn = {:.3} ms (delta = {:+.3} ms)",
        ctx_prompt.len(),
        bd.paged_attn_ms,
        ab_sliced_ms,
        bd.paged_attn_ms - ab_sliced_ms,
    );

    // Stage 6 & 7: MoE L0..19 (NVFP4) and L20..23 (Q8_0) — each sub-stage batched in 1 sync!
    let nvfp4_caps: Vec<&VisitCapture> = captures.iter().filter(|c| c.phys_layer < 20).collect();
    let q8_caps: Vec<&VisitCapture> = captures.iter().filter(|c| c.phys_layer >= 20).collect();

    let use_grouped_moe = inkling2_grouped_moe_available(m_rows);
    let time_moe_group = |caps: &[&VisitCapture]| -> MoeBreakdown {
        let mut mb = MoeBreakdown {
            visits: caps.len(),
            ..Default::default()
        };
        let group_uses_grouped = use_grouped_moe
            && !caps.is_empty()
            && caps.iter().all(|cap| {
                inner.layers[cap.phys_layer].moe.supports_grouped_moe(
                    m_rows,
                    d_i64 as usize,
                    act_dtype,
                )
            });
        if group_uses_grouped {
            mb.router_ms = time_stage_ms(w, iters, || {
                let mut outs = Vec::with_capacity(caps.len() * 2);
                for cap in caps {
                    let moe = &inner.layers[cap.phys_layer].moe;
                    let rl = if m_rows <= 32 {
                        inkling2_moe_router_proj(&cap.x_mlp, &moe.router_weight_f32).expect("rp")
                    } else {
                        let xf = cap.x_mlp.astype(DType::Float32).expect("f");
                        xf.matmul(&moe.router_weight_t_f32).expect("mm")
                    };
                    let (topk, gam) = inkling2_moe_route(
                        &rl,
                        moe.router_bias_f32.as_ref(),
                        moe.down_proj.outer_scale(),
                        moe.top_k,
                        moe.route_scale as f32,
                    )
                    .expect("rt");
                    outs.extend([topk, gam]);
                }
                eval_owned_sync(&outs);
            });
            mb.gate_up_qmm_ms = time_stage_ms(w, iters, || {
                let mut outs = Vec::with_capacity(caps.len());
                for cap in caps {
                    let moe = &inner.layers[cap.phys_layer].moe;
                    let out = moe
                        .try_grouped_moe(&cap.x_mlp, &cap.top_k_indices, &cap.gamma, true)
                        .expect("gmoe")
                        .expect("grouped_moe available");
                    outs.push(out);
                }
                eval_owned_sync(&outs);
            });
            return mb;
        }
        mb.router_ms = time_stage_ms(w, iters, || {
            let mut outs = Vec::with_capacity(caps.len() * 4);
            for cap in caps {
                let moe = &inner.layers[cap.phys_layer].moe;
                let xf = cap.x_mlp.astype(DType::Float32).expect("f");
                let rl = xf.matmul(&moe.router_weight_t_f32).expect("mm");
                let (topk, gam) = inkling2_moe_route(
                    &rl,
                    moe.router_bias_f32.as_ref(),
                    moe.down_proj.outer_scale(),
                    moe.top_k,
                    moe.route_scale as f32,
                )
                .expect("rt");
                if do_sort {
                    let fl = topk.reshape(&[-1]).expect("r");
                    let x_exp = cap.x_mlp.reshape(&[t_i64, 1, d_i64]).expect("e");
                    let ord = fl.argsort(Some(-1)).expect("o");
                    let inv = ord.argsort(Some(-1)).expect("i");
                    let idx_s = fl.take(&ord, 0).expect("t");
                    let tid = ord.floor_divide(&k_scalar).expect("f");
                    let xs = x_exp.take(&tid, 0).expect("x");
                    outs.extend([xs, idx_s, inv, gam]);
                } else {
                    let xr = cap.x_mlp.reshape(&[t_i64, 1, 1, d_i64]).expect("e");
                    outs.extend([xr, topk, gam]);
                }
            }
            eval_owned_sync(&outs);
        });
        mb.gate_up_qmm_ms = time_stage_ms(w, iters, || {
            let mut outs = Vec::with_capacity(caps.len());
            for cap in caps {
                let moe = &inner.layers[cap.phys_layer].moe;
                outs.push(
                    moe.gate_up_proj
                        .forward_without_outer_scale(
                            &cap.x_for_gather,
                            &cap.idx_for_gather,
                            do_sort,
                        )
                        .expect("gu"),
                );
            }
            eval_owned_sync(&outs);
        });
        mb.situ_ms = time_stage_ms(w, iters, || {
            let mut outs = Vec::with_capacity(caps.len());
            for cap in caps {
                let moe = &inner.layers[cap.phys_layer].moe;
                let gu_in = if cap.gate_up.dtype().expect("d") == act_dtype {
                    cap.gate_up.clone()
                } else {
                    cap.gate_up.astype(act_dtype).expect("c")
                };
                let idx_i32 = if cap.idx_for_gather.dtype().expect("d") == DType::Int32 {
                    cap.idx_for_gather.clone()
                } else {
                    cap.idx_for_gather.astype(DType::Int32).expect("c")
                };
                let outer = moe.gate_up_proj.outer_scale().map(|s| (s, &idx_i32));
                outs.push(
                    inkling2_situ(
                        &gu_in,
                        outer,
                        moe.mlp_gate_softcap as f32,
                        moe.mlp_up_softcap as f32,
                        moe.gate_up_interleaved,
                    )
                    .expect("situ"),
                );
            }
            eval_owned_sync(&outs);
        });
        mb.down_qmm_ms = time_stage_ms(w, iters, || {
            let mut outs = Vec::with_capacity(caps.len());
            for cap in caps {
                let moe = &inner.layers[cap.phys_layer].moe;
                outs.push(
                    moe.down_proj
                        .forward_without_outer_scale(&cap.act_in, &cap.idx_for_gather, do_sort)
                        .expect("dn"),
                );
            }
            eval_owned_sync(&outs);
        });
        mb.combine_ms = time_stage_ms(w, iters, || {
            let mut outs = Vec::with_capacity(caps.len());
            for cap in caps {
                outs.push(
                    inkling2_moe_combine(
                        &cap.down_out,
                        &cap.gamma,
                        cap.inv_order.as_ref(),
                        act_dtype,
                    )
                    .expect("comb"),
                );
            }
            eval_owned_sync(&outs);
        });
        mb
    };

    bd.moe_nvfp4 = time_moe_group(&nvfp4_caps);
    bd.moe_q8 = time_moe_group(&q8_caps);

    // Stage 8: Prologue + Epilogue (batched K=8 to eliminate 0.15ms sync floor!)
    let epi_k = 8usize;
    bd.embed_ms = time_stage_ms(w, iters, || {
        let mut outs = Vec::with_capacity(epi_k);
        for _ in 0..epi_k {
            let raw = inner.embed_tokens.forward(&ids_arr).expect("e");
            outs.push(
                apply_sublayer_rms_norm(
                    &raw,
                    &inner.embed_norm,
                    inner.config.rms_norm_eps as f32,
                    act_dtype,
                )
                .expect("en"),
            );
        }
        eval_owned_sync(&outs);
    }) / epi_k as f64;

    bd.output_res_norm_ms = time_stage_ms(w, iters, || {
        let mut outs = Vec::with_capacity(epi_k);
        for _ in 0..epi_k {
            outs.push(
                inkling2_block_attn_res(
                    &final_srcs,
                    None,
                    &inner.output_res_query,
                    Some(&inner.output_norm),
                    ATTN_RES_EPS,
                    inner.config.rms_norm_eps as f32,
                    true,
                    act_dtype,
                )
                .expect("on"),
            );
        }
        eval_owned_sync(&outs);
    }) / epi_k as f64;

    let unpadded = inner.config.effective_unpadded_vocab_size();
    let vocab_cols = logits_full.shape().expect("s")[1] as usize;
    let out_cols = unpadded.min(vocab_cols);
    let cap_val = inner.config.final_logit_softcapping as f32;
    if let crate::models::quantized_linear::LinearProj::Quantized(ref ql) = inner.lm_head
        && ql.is_symmetric_q8()
        && m_rows <= 32
    {
        let w_q8 = ql.get_weight();
        let sc_f16 = ql.get_scales();
        bd.lm_head_ms = time_stage_ms(w, iters, || {
            let mut outs = Vec::with_capacity(epi_k);
            for _ in 0..epi_k {
                outs.push(
                    super::kernels::inkling2_lm_head_q8_softcap_argmax(
                        &normed_out,
                        w_q8,
                        sc_f16,
                        out_cols,
                        cap_val,
                    )
                    .expect("q8_lm_am"),
                );
            }
            eval_owned_sync(&outs);
        }) / epi_k as f64;
        bd.softcap_argmax_ms = 0.0;
    } else if let crate::models::quantized_linear::LinearProj::Standard(ref lin) = inner.lm_head
        && m_rows <= 32
    {
        let w_bf16 = lin.get_weight();
        bd.lm_head_ms = time_stage_ms(w, iters, || {
            let mut outs = Vec::with_capacity(epi_k);
            for _ in 0..epi_k {
                outs.push(
                    super::kernels::inkling2_bf16_lm_head_softcap_argmax(
                        &normed_out,
                        &w_bf16,
                        out_cols,
                        cap_val,
                    )
                    .expect("bf16_lm_am"),
                );
            }
            eval_owned_sync(&outs);
        }) / epi_k as f64;
        bd.softcap_argmax_ms = 0.0;
    } else {
        bd.lm_head_ms = time_stage_ms(w, iters, || {
            let mut outs = Vec::with_capacity(epi_k);
            for _ in 0..epi_k {
                let raw = inner
                    .lm_head
                    .forward_logits_keep_f32(&normed_out)
                    .expect("lm");
                if raw.dtype().expect("dt") == DType::Float32 {
                    outs.push(raw);
                } else {
                    outs.push(raw.astype(DType::Float32).expect("f"));
                }
            }
            eval_owned_sync(&outs);
        }) / epi_k as f64;

        bd.softcap_argmax_ms = time_stage_ms(w, iters, || {
            let mut outs = Vec::with_capacity(epi_k);
            for _ in 0..epi_k {
                outs.push(
                    inkling2_lm_head_softcap_argmax(&logits_full, None, out_cols, cap_val, false)
                        .expect("sc_am"),
                );
            }
            eval_owned_sync(&outs);
        }) / epi_k as f64;
    }

    // Roll back probe rows cleanly
    inner
        .paged
        .as_mut()
        .expect("paged")
        .coordinator
        .commit_cycle(seq_id, ticket, 0)
        .expect("rollback");
    // Suppress unused field warning on `visit_idx` / `msgs`
    let _ = captures.first().map(|c| (c.visit_idx, c.msgs.is_some()));

    bd
}

#[derive(Default, Clone, Copy)]
struct DsparkProposeBreakdown {
    embed_mask_ms: f64,
    draft_layers_attn_ms: f64,
    draft_layers_mlp_ms: f64,
    draft_out_norm_ms: f64,
    target_lm_head_ms: f64,
    markov_ctx_precomp_ms: f64,
    markov_rnn_gate_ms: f64,
    markov_fold_matmul_ms: f64,
    markov_softcap_argmax_ms: f64,
    unsynced_total_ms: f64,
}

impl DsparkProposeBreakdown {
    fn draft_layers_total_ms(&self) -> f64 {
        self.draft_layers_attn_ms + self.draft_layers_mlp_ms
    }

    fn markov_7steps_total_ms(&self) -> f64 {
        self.markov_ctx_precomp_ms
            + self.markov_rnn_gate_ms
            + self.markov_fold_matmul_ms
            + self.markov_softcap_argmax_ms
    }

    fn isolated_total_ms(&self) -> f64 {
        self.embed_mask_ms
            + self.draft_layers_total_ms()
            + self.draft_out_norm_ms
            + self.target_lm_head_ms
            + self.markov_7steps_total_ms()
    }
}

fn profile_dspark_propose(inner: &mut Inkling2Inner, ctx_prompt: &[u32]) -> DsparkProposeBreakdown {
    let seq_id = 0u32;
    inner.reset_paged_sequence(seq_id).expect("reset");
    let (tap_visits, n_draft_layers) = {
        let d = inner.dspark.as_ref().expect("dspark");
        (
            d.config.target_layer_ids.clone(),
            d.config.num_hidden_layers,
        )
    };
    let mut draft_ctx = Inkling2DsparkContext::new(n_draft_layers);
    let (first_logits, _) = inner
        .paged_prefill_sequence_with_taps(
            seq_id,
            ctx_prompt,
            None,
            &tap_visits,
            |draft, taps, len| draft_ctx.append_taps(draft, taps, len),
        )
        .expect("prefill");
    draft_ctx.eval_state().expect("eval ctx");
    let anchor = first_logits
        .argmax(-1, Some(false))
        .expect("am")
        .item_at_int32(0)
        .expect("item") as u32;
    synchronize();

    let draft: &Inkling2DsparkModel = inner.dspark.as_ref().expect("dspark");
    let params = greedy_chat_params();
    let mut rng = rand::rng();

    // 1. Unsynced end-to-end propose_block (5 warmup + 10 timed)
    let unsynced_total_ms = time_stage_ms(5, 10, || {
        let mut prop = draft_ctx
            .propose_block(
                draft,
                &inner.embed_tokens,
                &inner.embed_norm,
                &inner.lm_head,
                anchor,
                7,
                &params,
                &mut rng,
            )
            .expect("propose");
        prop.materialize_draft_ids().expect("mat");
        synchronize();
    });

    // 2. Capture intermediates once so each isolated sub-stage can be replayed K=8 in 1 command buffer!
    let f_rnn = draft_ctx.last_f_rnn.as_ref().expect("f_rnn");
    let d = draft.config.hidden_size as i64;
    let n_q = draft.config.num_attention_heads as i64;
    let n_kv = draft.config.num_key_value_heads as i64;
    let hd = draft.config.head_dim as i64;
    let q_dim = n_q * hd;
    let kv_dim = n_kv * hd;
    let ffn = draft.config.intermediate_size as i64;
    let block_i64 = draft.config.block_size as i64;
    let n_draft = draft.config.num_draft_tokens();
    let n_draft_i64 = n_draft as i64;
    let eps = draft.config.rms_norm_eps;
    let offset = draft_ctx.committed_len;
    let attn_scale = 1.0_f64 / (draft.config.head_dim as f64).sqrt();
    let head = &draft.markov_head;
    let rank_i64 = head.markov_rank as i64;
    let softcap = draft.config.final_logit_softcapping;

    let anchor_arr = MxArray::from_int32(&[anchor as i32], &[1]).expect("a");
    let anchor_raw = inner.embed_tokens.forward(&anchor_arr).expect("emb");
    let anchor_emb =
        apply_sublayer_rms_norm(&anchor_raw, &inner.embed_norm, eps, DType::BFloat16).expect("en");
    let masks = draft
        .mask_embd
        .broadcast_to(&[block_i64 - 1, d])
        .expect("m");
    let mut x = MxArray::concatenate(&anchor_emb, &masks, 0).expect("cat");
    let mut layer_attn_in = Vec::with_capacity(draft.layers.len());
    let mut layer_mlp_in = Vec::with_capacity(draft.layers.len());

    for (i, layer) in draft.layers.iter().enumerate() {
        layer_attn_in.push(x.clone());
        let h =
            apply_sublayer_rms_norm(&x, &layer.input_layernorm, eps, DType::BFloat16).expect("in");
        let (q_raw, k_raw, v_raw) = if let Some(ref fused) = layer.fused_qkv {
            let qkv = fused.forward(&h).expect("qkv");
            let mut parts = qkv
                .split_sections(&[q_dim, q_dim + kv_dim], -1)
                .expect("sp");
            let v = parts.pop().expect("v");
            let k = parts.pop().expect("k");
            let q = parts.pop().expect("q");
            (q, k, v)
        } else {
            (
                layer.q_proj.forward(&h).expect("q"),
                layer.k_proj.forward(&h).expect("k"),
                layer.v_proj.forward(&h).expect("v"),
            )
        };
        let qn = apply_sublayer_rms_norm(
            &q_raw.reshape(&[block_i64 * n_q, hd]).expect("r"),
            &layer.q_norm,
            eps,
            DType::BFloat16,
        )
        .expect("qn")
        .reshape(&[1, block_i64, n_q, hd])
        .expect("r")
        .transpose(Some(&[0, 2, 1, 3]))
        .expect("t");
        let kn = apply_sublayer_rms_norm(
            &k_raw.reshape(&[block_i64 * n_kv, hd]).expect("r"),
            &layer.k_norm,
            eps,
            DType::BFloat16,
        )
        .expect("kn")
        .reshape(&[1, block_i64, n_kv, hd])
        .expect("r")
        .transpose(Some(&[0, 2, 1, 3]))
        .expect("t");
        let q_rot = draft.rope.forward(&qn, Some(offset)).expect("rq");
        let k_rot = draft.rope.forward(&kn, Some(offset)).expect("rk");
        let v_4d = v_raw
            .reshape(&[1, block_i64, n_kv, hd])
            .expect("r")
            .transpose(Some(&[0, 2, 1, 3]))
            .expect("t");
        let (k_all, v_all) = match draft_ctx.layers[i].view().expect("view") {
            Some((ck, cv)) => (
                MxArray::concatenate(&ck, &k_rot, 2).expect("ck"),
                MxArray::concatenate(&cv, &v_4d, 2).expect("cv"),
            ),
            None => (k_rot, v_4d),
        };
        let attn_out = scaled_dot_product_attention(&q_rot, &k_all, &v_all, attn_scale, None)
            .expect("sdpa")
            .transpose(Some(&[0, 2, 1, 3]))
            .expect("t")
            .reshape(&[block_i64, q_dim])
            .expect("r");
        let o = layer.o_proj.forward(&attn_out).expect("o");
        x = x
            .astype(DType::Float32)
            .expect("f")
            .add(&o.astype(DType::Float32).expect("f"))
            .expect("a")
            .astype(DType::BFloat16)
            .expect("b");

        layer_mlp_in.push(x.clone());
        let h_mlp =
            apply_sublayer_rms_norm(&x, &layer.post_attention_layernorm, eps, DType::BFloat16)
                .expect("pn");
        let (gate, up) = if let Some(ref fused) = layer.fused_gate_up {
            let gu = fused.forward(&h_mlp).expect("gu");
            let mut parts = gu.split_sections(&[ffn], -1).expect("sp");
            let u = parts.pop().expect("u");
            let g = parts.pop().expect("g");
            (g, u)
        } else {
            (
                layer.gate_proj.forward(&h_mlp).expect("g"),
                layer.up_proj.forward(&h_mlp).expect("u"),
            )
        };
        let act = Activations::silu(&gate.astype(DType::Float32).expect("f"))
            .expect("s")
            .mul(&up.astype(DType::Float32).expect("f"))
            .expect("m")
            .astype(DType::BFloat16)
            .expect("b");
        let mlp_out = layer.down_proj.forward(&act).expect("d");
        x = x
            .astype(DType::Float32)
            .expect("f")
            .add(&mlp_out.astype(DType::Float32).expect("f"))
            .expect("a")
            .astype(DType::BFloat16)
            .expect("b");
    }
    let out_norm_in = x;
    let out_normed =
        apply_sublayer_rms_norm(&out_norm_in, &draft.output_norm, eps, DType::BFloat16)
            .expect("on");
    let hidden = out_normed
        .slice(&[1, 0], &[1 + n_draft_i64, d])
        .expect("sl");
    let base_logits = inner
        .lm_head
        .forward_logits(&hidden)
        .expect("lm")
        .astype(DType::BFloat16)
        .expect("b");
    let padded_vocab = base_logits.shape().expect("s")[1];
    let unpadded_vocab = (draft.config.unpadded_vocab_size as i64).min(padded_vocab);
    let f_rnn_bcast = f_rnn.broadcast_to(&[n_draft_i64, d]).expect("bc");
    let ctx_in = MxArray::concatenate(&hidden, &f_rnn_bcast, -1).expect("cat");
    let ctx_all = ctx_in
        .matmul(&head.rnn_proj_ctx_w_t)
        .expect("mm")
        .astype(DType::Float32)
        .expect("f")
        .add(&head.rnn_bias_f32)
        .expect("a");

    let mut step_states = Vec::with_capacity(n_draft);
    let mut step_prevs = Vec::with_capacity(n_draft);
    let mut step_latents = Vec::with_capacity(n_draft);
    let mut step_biases = Vec::with_capacity(n_draft);
    let mut st = MxArray::zeros(&[1, rank_i64], Some(DType::BFloat16)).expect("st");
    let mut pt = anchor_arr.clone();
    for step in 0..n_draft {
        let s_i64 = step as i64;
        let ctx_i = ctx_all
            .slice(&[s_i64, 0], &[s_i64 + 1, 3 * rank_i64])
            .expect("sl");
        step_states.push(st.clone());
        step_prevs.push(pt.clone());
        let prev_emb = head.markov_w1.take(&pt, 0).expect("tk");
        let rec_in = MxArray::concatenate(&st, &prev_emb, -1).expect("c");
        let rec_mm = rec_in.matmul(&head.rnn_proj_rec_w_t).expect("m");
        let (nst, lat) = inkling2_markov_gate(&rec_mm, Some(&ctx_i), &st).expect("g");
        let base_i = base_logits
            .slice(&[s_i64, 0], &[s_i64 + 1, padded_vocab])
            .expect("sl");
        let chosen = if let Some(ref q8) = head.fold_weight_q8 {
            inkling2_markov_fold_softcap_argmax(
                &lat,
                &q8.weight,
                &q8.scales,
                Some(&q8.biases),
                &head.fold_bias,
                &base_i,
                unpadded_vocab as usize,
                softcap as f32,
            )
            .expect("fold_am")
        } else {
            let fw_t = head.fold_weight_t.as_ref().expect("fw_t");
            let bias = lat
                .matmul(fw_t)
                .expect("m")
                .astype(DType::BFloat16)
                .expect("b")
                .add(&head.fold_bias)
                .expect("a")
                .astype(DType::BFloat16)
                .expect("b");
            let ch = inkling2_lm_head_softcap_argmax(
                &base_i,
                Some(&bias),
                unpadded_vocab as usize,
                softcap as f32,
                true,
            )
            .expect("am");
            step_biases.push(bias);
            ch
        };
        step_latents.push(lat);
        st = nst;
        pt = chosen;
    }
    eval_owned_sync(&layer_attn_in);
    eval_owned_sync(&layer_mlp_in);
    eval_sync(&[&anchor_arr, &out_norm_in, &hidden, &base_logits, &ctx_all]);
    eval_owned_sync(&step_states);
    eval_owned_sync(&step_prevs);
    eval_owned_sync(&step_latents);
    eval_owned_sync(&step_biases);

    // 3. Batch K=8 repeats per sub-stage into 1 sync so sync overhead is <0.02 ms!
    let k_rep = 8usize;
    let w = 2usize;
    let iters = 5usize;

    let embed_mask_ms = time_stage_ms(w, iters, || {
        let mut outs = Vec::with_capacity(k_rep);
        for _ in 0..k_rep {
            let ar = inner.embed_tokens.forward(&anchor_arr).expect("e");
            let ae =
                apply_sublayer_rms_norm(&ar, &inner.embed_norm, eps, DType::BFloat16).expect("n");
            let ms = draft
                .mask_embd
                .broadcast_to(&[block_i64 - 1, d])
                .expect("m");
            outs.push(MxArray::concatenate(&ae, &ms, 0).expect("c"));
        }
        eval_owned_sync(&outs);
    }) / k_rep as f64;

    let draft_layers_attn_ms = time_stage_ms(w, iters, || {
        let mut outs = Vec::with_capacity(k_rep * draft.layers.len());
        for _ in 0..k_rep {
            for (i, layer) in draft.layers.iter().enumerate() {
                let xi = &layer_attn_in[i];
                let h = apply_sublayer_rms_norm(xi, &layer.input_layernorm, eps, DType::BFloat16)
                    .expect("in");
                let (q_raw, k_raw, v_raw) = if let Some(ref fused) = layer.fused_qkv {
                    let qkv = super::attention::forward_linear_proj_fast(fused, &h).expect("qkv");
                    let mut parts = qkv
                        .split_sections(&[q_dim, q_dim + kv_dim], -1)
                        .expect("sp");
                    let v = parts.pop().expect("v");
                    let k = parts.pop().expect("k");
                    let q = parts.pop().expect("q");
                    (q, k, v)
                } else {
                    (
                        super::attention::forward_linear_proj_fast(&layer.q_proj, &h).expect("q"),
                        super::attention::forward_linear_proj_fast(&layer.k_proj, &h).expect("k"),
                        super::attention::forward_linear_proj_fast(&layer.v_proj, &h).expect("v"),
                    )
                };
                let qn = apply_sublayer_rms_norm(
                    &q_raw.reshape(&[block_i64 * n_q, hd]).expect("r"),
                    &layer.q_norm,
                    eps,
                    DType::BFloat16,
                )
                .expect("qn")
                .reshape(&[1, block_i64, n_q, hd])
                .expect("r")
                .transpose(Some(&[0, 2, 1, 3]))
                .expect("t");
                let kn = apply_sublayer_rms_norm(
                    &k_raw.reshape(&[block_i64 * n_kv, hd]).expect("r"),
                    &layer.k_norm,
                    eps,
                    DType::BFloat16,
                )
                .expect("kn")
                .reshape(&[1, block_i64, n_kv, hd])
                .expect("r")
                .transpose(Some(&[0, 2, 1, 3]))
                .expect("t");
                let q_rot = draft.rope.forward(&qn, Some(offset)).expect("rq");
                let k_rot = draft.rope.forward(&kn, Some(offset)).expect("rk");
                let v_4d = v_raw
                    .reshape(&[1, block_i64, n_kv, hd])
                    .expect("r")
                    .transpose(Some(&[0, 2, 1, 3]))
                    .expect("t");
                let (k_all, v_all) = match draft_ctx.layers[i].view().expect("view") {
                    Some((ck, cv)) => (
                        MxArray::concatenate(&ck, &k_rot, 2).expect("ck"),
                        MxArray::concatenate(&cv, &v_4d, 2).expect("cv"),
                    ),
                    None => (k_rot, v_4d),
                };
                let ao = scaled_dot_product_attention(&q_rot, &k_all, &v_all, attn_scale, None)
                    .expect("sdpa")
                    .transpose(Some(&[0, 2, 1, 3]))
                    .expect("t")
                    .reshape(&[block_i64, q_dim])
                    .expect("r");
                let o = super::attention::forward_linear_proj_fast(&layer.o_proj, &ao).expect("o");
                outs.push(
                    xi.astype(DType::Float32)
                        .expect("f")
                        .add(&o.astype(DType::Float32).expect("f"))
                        .expect("a")
                        .astype(DType::BFloat16)
                        .expect("b"),
                );
            }
        }
        eval_owned_sync(&outs);
    }) / k_rep as f64;

    let draft_layers_mlp_ms = time_stage_ms(w, iters, || {
        let mut outs = Vec::with_capacity(k_rep * draft.layers.len());
        for _ in 0..k_rep {
            for (i, layer) in draft.layers.iter().enumerate() {
                let xi = &layer_mlp_in[i];
                let h_mlp = apply_sublayer_rms_norm(
                    xi,
                    &layer.post_attention_layernorm,
                    eps,
                    DType::BFloat16,
                )
                .expect("pn");
                let (gate, up) = if let Some(ref fused) = layer.fused_gate_up {
                    let gu = super::attention::forward_linear_proj_fast(fused, &h_mlp).expect("gu");
                    let mut parts = gu.split_sections(&[ffn], -1).expect("sp");
                    let u = parts.pop().expect("u");
                    let g = parts.pop().expect("g");
                    (g, u)
                } else {
                    (
                        super::attention::forward_linear_proj_fast(&layer.gate_proj, &h_mlp)
                            .expect("g"),
                        super::attention::forward_linear_proj_fast(&layer.up_proj, &h_mlp)
                            .expect("u"),
                    )
                };
                let act = Activations::silu(&gate.astype(DType::Float32).expect("f"))
                    .expect("s")
                    .mul(&up.astype(DType::Float32).expect("f"))
                    .expect("m")
                    .astype(DType::BFloat16)
                    .expect("b");
                let mo =
                    super::attention::forward_linear_proj_fast(&layer.down_proj, &act).expect("d");
                outs.push(
                    xi.astype(DType::Float32)
                        .expect("f")
                        .add(&mo.astype(DType::Float32).expect("f"))
                        .expect("a")
                        .astype(DType::BFloat16)
                        .expect("b"),
                );
            }
        }
        eval_owned_sync(&outs);
    }) / k_rep as f64;

    let draft_out_norm_ms = time_stage_ms(w, iters, || {
        let mut outs = Vec::with_capacity(k_rep);
        for _ in 0..k_rep {
            let on =
                apply_sublayer_rms_norm(&out_norm_in, &draft.output_norm, eps, DType::BFloat16)
                    .expect("on");
            outs.push(on.slice(&[1, 0], &[1 + n_draft_i64, d]).expect("sl"));
        }
        eval_owned_sync(&outs);
    }) / k_rep as f64;

    let target_lm_head_ms = time_stage_ms(w, iters, || {
        let mut outs = Vec::with_capacity(k_rep);
        for _ in 0..k_rep {
            outs.push(
                super::dspark::project_draft_lm_head_bf16(
                    &inner.lm_head,
                    &hidden,
                    draft.config.unpadded_vocab_size,
                )
                .expect("lm"),
            );
        }
        eval_owned_sync(&outs);
    }) / k_rep as f64;

    let markov_ctx_precomp_ms = time_stage_ms(w, iters, || {
        let mut outs = Vec::with_capacity(k_rep);
        for _ in 0..k_rep {
            let fb = f_rnn.broadcast_to(&[n_draft_i64, d]).expect("bc");
            let ci = MxArray::concatenate(&hidden, &fb, -1).expect("c");
            outs.push(
                ci.matmul(&head.rnn_proj_ctx_w_t)
                    .expect("m")
                    .astype(DType::Float32)
                    .expect("f")
                    .add(&head.rnn_bias_f32)
                    .expect("a"),
            );
        }
        eval_owned_sync(&outs);
    }) / k_rep as f64;

    let markov_rnn_gate_ms = time_stage_ms(w, iters, || {
        let mut outs = Vec::with_capacity(k_rep * n_draft * 2);
        for _ in 0..k_rep {
            for step in 0..n_draft {
                let s_i64 = step as i64;
                let ctx_i = ctx_all
                    .slice(&[s_i64, 0], &[s_i64 + 1, 3 * rank_i64])
                    .expect("sl");
                let prev_emb = head.markov_w1.take(&step_prevs[step], 0).expect("tk");
                let rec_in = MxArray::concatenate(&step_states[step], &prev_emb, -1).expect("c");
                let rec_mm = rec_in.matmul(&head.rnn_proj_rec_w_t).expect("m");
                let (nst, lat) =
                    inkling2_markov_gate(&rec_mm, Some(&ctx_i), &step_states[step]).expect("g");
                outs.push(nst);
                outs.push(lat);
            }
        }
        eval_owned_sync(&outs);
    }) / k_rep as f64;

    let (markov_fold_matmul_ms, markov_softcap_argmax_ms) =
        if let Some(ref q8) = head.fold_weight_q8 {
            let fused_ms = time_stage_ms(w, iters, || {
                let mut outs = Vec::with_capacity(k_rep * n_draft);
                for _ in 0..k_rep {
                    for (step, latent) in step_latents.iter().enumerate().take(n_draft) {
                        let s_i64 = step as i64;
                        let base_i = base_logits
                            .slice(&[s_i64, 0], &[s_i64 + 1, padded_vocab])
                            .expect("sl");
                        outs.push(
                            inkling2_markov_fold_softcap_argmax(
                                latent,
                                &q8.weight,
                                &q8.scales,
                                Some(&q8.biases),
                                &head.fold_bias,
                                &base_i,
                                unpadded_vocab as usize,
                                softcap as f32,
                            )
                            .expect("fold_am"),
                        );
                    }
                }
                eval_owned_sync(&outs);
            }) / k_rep as f64;
            (fused_ms, 0.0)
        } else {
            let fw_t = head.fold_weight_t.as_ref().expect("fw_t");
            let mm_ms = time_stage_ms(w, iters, || {
                let mut outs = Vec::with_capacity(k_rep * n_draft);
                for _ in 0..k_rep {
                    for latent in step_latents.iter().take(n_draft) {
                        let bias = latent
                            .matmul(fw_t)
                            .expect("m")
                            .astype(DType::BFloat16)
                            .expect("b")
                            .add(&head.fold_bias)
                            .expect("a")
                            .astype(DType::BFloat16)
                            .expect("b");
                        outs.push(bias);
                    }
                }
                eval_owned_sync(&outs);
            }) / k_rep as f64;

            let sa_ms = time_stage_ms(w, iters, || {
                let mut outs = Vec::with_capacity(k_rep * n_draft);
                for _ in 0..k_rep {
                    for (step, step_bias) in step_biases.iter().enumerate().take(n_draft) {
                        let s_i64 = step as i64;
                        let base_i = base_logits
                            .slice(&[s_i64, 0], &[s_i64 + 1, padded_vocab])
                            .expect("sl");
                        outs.push(
                            inkling2_lm_head_softcap_argmax(
                                &base_i,
                                Some(step_bias),
                                unpadded_vocab as usize,
                                softcap as f32,
                                true,
                            )
                            .expect("am"),
                        );
                    }
                }
                eval_owned_sync(&outs);
            }) / k_rep as f64;
            (mm_ms, sa_ms)
        };

    DsparkProposeBreakdown {
        embed_mask_ms,
        draft_layers_attn_ms,
        draft_layers_mlp_ms,
        draft_out_norm_ms,
        target_lm_head_ms,
        markov_ctx_precomp_ms,
        markov_rnn_gate_ms,
        markov_fold_matmul_ms,
        markov_softcap_argmax_ms,
        unsynced_total_ms,
    }
}

fn print_stage_table(label: &str, m_rows: usize, ctx_len: usize, bd: &StageBreakdown) {
    let total = bd.total_isolated_ms();
    let pct = |v: f64| 100.0 * v / total.max(1e-9);
    let attn_res_all = bd.attn_res_ctrl_norm_ms + bd.residual_add_ms;
    let dense_all = bd.qrgkv_sink_proj_ms + bd.o_proj_ms;
    let sconv_all = bd.sconv_kv_ms + bd.sconv_attn_post_ms + bd.sconv_mlp_post_ms;
    let clip_rel_all = bd.l2_clip_vnorm_ms + bd.rel_bias_matmul_ms;
    let paged_all = bd.kv_write_ms + bd.paged_attn_ms + bd.gate_sink_ms;
    let moe_nvfp4_all = bd.moe_nvfp4.total_ms();
    let moe_q8_all = bd.moe_q8.total_ms();
    let epilogue_all = bd.embed_ms + bd.output_res_norm_ms + bd.lm_head_ms + bd.softcap_argmax_ms;

    eprintln!("\n================================================================================");
    eprintln!(
        "  PER-STAGE GPU MICRO-BREAKDOWN: {label} (M={m_rows}, ctx={ctx_len}, 48 visits, INT8-KTM)"
    );
    eprintln!("  (Capture-then-Batch: 1 sync per isolated stage across all 48 visits)");
    eprintln!("================================================================================");
    eprintln!("  {:<46} {:>9} {:>7}", "Component", "ms", "% sum");
    eprintln!("  ------------------------------------------------------------------");
    eprintln!(
        "  1. Block AttnRes + LoopCtrl + Norm + Resid    {:>8.3} ms {:>6.1}%",
        attn_res_all,
        pct(attn_res_all)
    );
    eprintln!(
        "       AttnRes + LoopCtrl + InputNorm           {:>8.3} ms {:>6.1}%",
        bd.attn_res_ctrl_norm_ms,
        pct(bd.attn_res_ctrl_norm_ms)
    );
    eprintln!(
        "       FP32 Residual Adds                       {:>8.3} ms {:>6.1}%",
        bd.residual_add_ms,
        pct(bd.residual_add_ms)
    );
    eprintln!(
        "  2. Dense Q8_0 Projections (qrgkv/qrg+kv+o)    {:>8.3} ms {:>6.1}%",
        dense_all,
        pct(dense_all)
    );
    eprintln!(
        "       qrgkv / qrg+k+v + sink                   {:>8.3} ms {:>6.1}%",
        bd.qrgkv_sink_proj_ms,
        pct(bd.qrgkv_sink_proj_ms)
    );
    eprintln!(
        "       o_proj                                   {:>8.3} ms {:>6.1}%",
        bd.o_proj_ms,
        pct(bd.o_proj_ms)
    );
    eprintln!(
        "  3. ShortConv (k/v/attn_post/mlp_post)         {:>8.3} ms {:>6.1}%",
        sconv_all,
        pct(sconv_all)
    );
    eprintln!(
        "       k/v ShortConv (+Global dyn_proj)         {:>8.3} ms {:>6.1}%",
        bd.sconv_kv_ms,
        pct(bd.sconv_kv_ms)
    );
    eprintln!(
        "       attn_post ShortConv                      {:>8.3} ms {:>6.1}%",
        bd.sconv_attn_post_ms,
        pct(bd.sconv_attn_post_ms)
    );
    eprintln!(
        "       mlp_post ShortConv                       {:>8.3} ms {:>6.1}%",
        bd.sconv_mlp_post_ms,
        pct(bd.sconv_mlp_post_ms)
    );
    eprintln!(
        "  4. L2-clip Q/R/K + V RMSNorm + rel_bias       {:>8.3} ms {:>6.1}%",
        clip_rel_all,
        pct(clip_rel_all)
    );
    eprintln!(
        "       L2-clip Q/R/K + V RMSNorm                {:>8.3} ms {:>6.1}%",
        bd.l2_clip_vnorm_ms,
        pct(bd.l2_clip_vnorm_ms)
    );
    eprintln!(
        "       rel_bias matmul (R @ W_rel)              {:>8.3} ms {:>6.1}%",
        bd.rel_bias_matmul_ms,
        pct(bd.rel_bias_matmul_ms)
    );
    eprintln!(
        "  5. PagedKVWrite + PagedAttention + gate_sink  {:>8.3} ms {:>6.1}%",
        paged_all,
        pct(paged_all)
    );
    eprintln!(
        "       PagedKVWrite (INT8 KTM)                  {:>8.3} ms {:>6.1}%",
        bd.kv_write_ms,
        pct(bd.kv_write_ms)
    );
    eprintln!(
        "       PagedAttention (INT8 KTM)                {:>8.3} ms {:>6.1}%",
        bd.paged_attn_ms,
        pct(bd.paged_attn_ms)
    );
    eprintln!(
        "       gate_sink epilogue                       {:>8.3} ms {:>6.1}%",
        bd.gate_sink_ms,
        pct(bd.gate_sink_ms)
    );
    eprintln!(
        "  6. MoE L0..19 NVFP4 ({:>2} visits, {:.3}ms/v)   {:>8.3} ms {:>6.1}%",
        bd.moe_nvfp4.visits,
        moe_nvfp4_all / bd.moe_nvfp4.visits.max(1) as f64,
        moe_nvfp4_all,
        pct(moe_nvfp4_all)
    );
    eprintln!(
        "       router + gather prep                     {:>8.3} ms {:>6.1}%",
        bd.moe_nvfp4.router_ms,
        pct(bd.moe_nvfp4.router_ms)
    );
    eprintln!(
        "       gate_up gather_qmm                       {:>8.3} ms {:>6.1}%",
        bd.moe_nvfp4.gate_up_qmm_ms,
        pct(bd.moe_nvfp4.gate_up_qmm_ms)
    );
    eprintln!(
        "       SiTU (fused Metal)                       {:>8.3} ms {:>6.1}%",
        bd.moe_nvfp4.situ_ms,
        pct(bd.moe_nvfp4.situ_ms)
    );
    eprintln!(
        "       down gather_qmm                          {:>8.3} ms {:>6.1}%",
        bd.moe_nvfp4.down_qmm_ms,
        pct(bd.moe_nvfp4.down_qmm_ms)
    );
    eprintln!(
        "       combine (unsort + outer_scale + sum)     {:>8.3} ms {:>6.1}%",
        bd.moe_nvfp4.combine_ms,
        pct(bd.moe_nvfp4.combine_ms)
    );
    eprintln!(
        "  7. MoE L20..23 Q8_0 ({:>2} visits, {:.3}ms/v)   {:>8.3} ms {:>6.1}%",
        bd.moe_q8.visits,
        moe_q8_all / bd.moe_q8.visits.max(1) as f64,
        moe_q8_all,
        pct(moe_q8_all)
    );
    eprintln!(
        "       router + gather prep                     {:>8.3} ms {:>6.1}%",
        bd.moe_q8.router_ms,
        pct(bd.moe_q8.router_ms)
    );
    eprintln!(
        "       gate_up gather_qmm                       {:>8.3} ms {:>6.1}%",
        bd.moe_q8.gate_up_qmm_ms,
        pct(bd.moe_q8.gate_up_qmm_ms)
    );
    eprintln!(
        "       SiTU (fused Metal)                       {:>8.3} ms {:>6.1}%",
        bd.moe_q8.situ_ms,
        pct(bd.moe_q8.situ_ms)
    );
    eprintln!(
        "       down gather_qmm                          {:>8.3} ms {:>6.1}%",
        bd.moe_q8.down_qmm_ms,
        pct(bd.moe_q8.down_qmm_ms)
    );
    eprintln!(
        "       combine (unsort + outer_scale + sum)     {:>8.3} ms {:>6.1}%",
        bd.moe_q8.combine_ms,
        pct(bd.moe_q8.combine_ms)
    );
    eprintln!(
        "  8. Prologue + Epilogue (embed+res+lm+cap)     {:>8.3} ms {:>6.1}%",
        epilogue_all,
        pct(epilogue_all)
    );
    eprintln!(
        "       embed_tokens + embed_norm                {:>8.3} ms {:>6.1}%",
        bd.embed_ms,
        pct(bd.embed_ms)
    );
    eprintln!(
        "       output_res_query + output_norm           {:>8.3} ms {:>6.1}%",
        bd.output_res_norm_ms,
        pct(bd.output_res_norm_ms)
    );
    eprintln!(
        "       lm_head                                  {:>8.3} ms {:>6.1}%",
        bd.lm_head_ms,
        pct(bd.lm_head_ms)
    );
    eprintln!(
        "       softcap + argmax                         {:>8.3} ms {:>6.1}%",
        bd.softcap_argmax_ms,
        pct(bd.softcap_argmax_ms)
    );
    eprintln!("  ------------------------------------------------------------------");
    eprintln!(
        "  Sum of isolated stage blocks                  {:>8.3} ms  100.0%",
        total
    );
    eprintln!(
        "  Unsynced full 48-visit forward + argmax       {:>8.3} ms  (delta: {:+.3} ms)",
        bd.unsynced_full_ms,
        total - bd.unsynced_full_ms
    );
}

#[tokio::test]
#[ignore = "Requires real Inkling-2-Mini + DSpark GGUF on disk (~18 GB); run with --release -- --ignored --nocapture"]
async fn inkling2_roofline_profile() {
    let model_path = test_real_model_path();
    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "Missing Inkling-2-Mini checkpoint at {}",
        model_dir.display()
    );

    let _wired = WiredResidency::new(WiredResidency::working_set_minus_pools);
    let t_load = Instant::now();
    let mut inner = load_from_dir(&model_path).await.expect("load_from_dir");
    assert!(inner.paged.is_some(), "Expected paged runtime enabled");
    assert!(inner.dspark.is_some(), "Expected DSpark draft model loaded");
    let kv_dtype = inner.paged.as_ref().expect("paged").coordinator.kv_dtype();
    assert_eq!(
        kv_dtype,
        Inkling2KvDtype::Int8,
        "Expected default INT8-KTM KV cache"
    );
    synchronize_and_clear_cache();

    eprintln!("\n================================================================================");
    eprintln!("  INKLING-2-MINI M5 MAX RELEASE ROOFLINE & MICRO-PROFILE BASELINE");
    eprintln!("  Model: {model_path}");
    eprintln!(
        "  Loaded in {:.2}s | KV: {:?} | Weights: {:.2} GiB | DSpark: {:.1} MiB | Release: {}",
        t_load.elapsed().as_secs_f64(),
        kv_dtype,
        inner.weight_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
        inner.dspark.as_ref().expect("d").weight_bytes() as f64 / (1024.0 * 1024.0),
        !cfg!(debug_assertions),
    );
    eprintln!("================================================================================");

    let single_prompt_ids = render_prompt(&inner, SINGLE_PROMPT);
    let four_lane_prompt_ids: Vec<Vec<u32>> = FOUR_LANE_PROMPTS
        .iter()
        .map(|p| render_prompt(&inner, p))
        .collect();

    // ========================================================================
    // 1A. Single-seq AR (M=1, 128 output tokens, INT8-KTM)
    // ========================================================================
    for _ in 0..WARMUP_RUNS {
        let _ = run_single_ar(&mut inner, &single_prompt_ids, DECODE_OUT_TOKENS);
    }
    clear_cache();
    let ar1 = run_single_ar(&mut inner, &single_prompt_ids, DECODE_OUT_TOKENS);
    let ar1_ms_tok = ar1.total_ms / ar1.steps as f64;
    let ar1_tps = 1000.0 / ar1_ms_tok;
    let ar1_host_ms_tok = ar1.host_build_ms / ar1.steps as f64;
    let ar1_gpu_ms_tok = ar1.gpu_eval_wait_ms / ar1.steps as f64;
    let ar1_settle_ms_tok = ar1.settle_ms / ar1.steps as f64;

    eprintln!(
        "\n[1.1] Single-seq AR (M=1, {DECODE_OUT_TOKENS} output tokens, prompt={} tok, INT8-KTM):",
        single_prompt_ids.len()
    );
    eprintln!(
        "  Throughput:  {ar1_tps:.2} tok/s  ({ar1_ms_tok:.3} ms/tok over {} decode steps, total={:.1} ms)",
        ar1.steps, ar1.total_ms
    );
    eprintln!(
        "  Split:       host_graph_build = {ar1_host_ms_tok:.3} ms/tok ({:.1}%) | gpu_eval_wait = {ar1_gpu_ms_tok:.3} ms/tok ({:.1}%) | settle = {ar1_settle_ms_tok:.3} ms/tok ({:.1}%)",
        100.0 * ar1_host_ms_tok / ar1_ms_tok,
        100.0 * ar1_gpu_ms_tok / ar1_ms_tok,
        100.0 * ar1_settle_ms_tok / ar1_ms_tok,
    );

    // ========================================================================
    // 1B. Single-seq DSpark Depth Sweep (k=3,4,5,6,7 + Adaptive) on BOTH prompts
    // ========================================================================
    let code_comp_prompt_ids = render_prompt(&inner, CODE_COMPLETION_PROMPT);
    let ar_code = run_single_ar(&mut inner, &code_comp_prompt_ids, DECODE_OUT_TOKENS);
    let ar_code_tps = 1000.0 / (ar_code.total_ms / ar_code.steps as f64);

    let sweep_modes: [(&str, DsparkDepthMode); 6] = [
        ("fixed k=3 (M=4)", DsparkDepthMode::Fixed(3)),
        ("fixed k=4 (M=5)", DsparkDepthMode::Fixed(4)),
        ("fixed k=5 (M=6)", DsparkDepthMode::Fixed(5)),
        ("fixed k=6 (M=7)", DsparkDepthMode::Fixed(6)),
        ("fixed k=7 (M=8)", DsparkDepthMode::Fixed(7)),
        ("adaptive k[3..7]", DsparkDepthMode::Adaptive),
    ];

    for _ in 0..WARMUP_RUNS {
        let _ = run_single_dspark(
            &mut inner,
            &single_prompt_ids,
            DECODE_OUT_TOKENS,
            DsparkDepthMode::Fixed(7),
        );
    }

    for (prompt_label, p_ids, ar_ref, ref_tps) in [
        ("Dijkstra Coding", &single_prompt_ids, &ar1, ar1_tps),
        (
            "BinarySearch CodeComp",
            &code_comp_prompt_ids,
            &ar_code,
            ar_code_tps,
        ),
    ] {
        eprintln!(
            "\n[1.2] Single-seq DSpark Sweep — {prompt_label} ({DECODE_OUT_TOKENS} out tok, AR={ref_tps:.2} tok/s):"
        );
        eprintln!(
            "  {:<18} {:>9} {:>7} {:>8} {:>8} {:>8} {:>7} {:>9}",
            "Mode", "tok/s", "vs AR", "ms/cyc", "accept", "avg_k", "cyc", "exact_ar"
        );
        for &(label, mode) in &sweep_modes {
            clear_cache();
            let dp = run_single_dspark(&mut inner, p_ids, DECODE_OUT_TOKENS, mode);
            let tps = dp.decoded_tokens as f64 / (dp.total_ms / 1000.0);
            let ms_cyc = dp.total_ms / dp.cycles.max(1) as f64;
            let mean_acc = (dp.total_accepted_drafts + dp.cycles) as f64 / dp.cycles.max(1) as f64;
            let avg_k = dp.total_drafted as f64 / dp.cycles.max(1) as f64;
            let exact = dp
                .tokens
                .iter()
                .zip(ar_ref.tokens.iter())
                .take_while(|(a, b)| a == b)
                .count();
            eprintln!(
                "  {:<18} {:>8.2}  {:>6.2}x {:>7.3}  {:>7.3}  {:>7.2}  {:>6}  {:>4}/{:<4}",
                label,
                tps,
                tps / ref_tps.max(1e-6),
                ms_cyc,
                mean_acc,
                avg_k,
                dp.cycles,
                exact,
                ar_ref.tokens.len(),
            );
        }
    }

    clear_cache();
    let dp1 = run_single_dspark(
        &mut inner,
        &single_prompt_ids,
        DECODE_OUT_TOKENS,
        DsparkDepthMode::Adaptive,
    );
    let dp1_tps = dp1.decoded_tokens as f64 / (dp1.total_ms / 1000.0);
    let dp1_ms_cycle = dp1.total_ms / dp1.cycles.max(1) as f64;
    let dp1_propose_cycle = dp1.propose_ms / dp1.cycles.max(1) as f64;
    let dp1_verify_cycle = dp1.target_verify_ms / dp1.cycles.max(1) as f64;
    let dp1_commit_cycle = dp1.commit_ms / dp1.cycles.max(1) as f64;
    let dp1_mean_accept =
        (dp1.total_accepted_drafts + dp1.cycles) as f64 / dp1.cycles.max(1) as f64;
    let dp1_exact_prefix = dp1
        .tokens
        .iter()
        .zip(ar1.tokens.iter())
        .take_while(|(a, b)| a == b)
        .count();

    eprintln!(
        "\n[1.2b] Single-seq DSpark Adaptive Detail ({DECODE_OUT_TOKENS} output tokens, INT8-KTM):"
    );
    eprintln!(
        "  Throughput:  {dp1_tps:.2} tok/s  (speedup vs AR = {:.2}x, total={:.1} ms, cycles={}, exact_ar_prefix={}/{})",
        dp1_tps / ar1_tps.max(1e-6),
        dp1.total_ms,
        dp1.cycles,
        dp1_exact_prefix,
        ar1.tokens.len(),
    );
    eprintln!(
        "  Acceptance:  mean_accept = {dp1_mean_accept:.3} tok/cycle  (accepted_drafts={}/{}, rate={:.1}%)",
        dp1.total_accepted_drafts,
        dp1.total_drafted,
        100.0 * dp1.total_accepted_drafts as f64 / dp1.total_drafted.max(1) as f64,
    );
    eprintln!(
        "  Cycle split: {dp1_ms_cycle:.3} ms/cycle = propose {dp1_propose_cycle:.3} ms ({:.1}%) + target_verify {dp1_verify_cycle:.3} ms ({:.1}%) + commit {dp1_commit_cycle:.3} ms ({:.1}%)",
        100.0 * dp1.propose_ms / dp1.total_ms,
        100.0 * dp1.target_verify_ms / dp1.total_ms,
        100.0 * dp1.commit_ms / dp1.total_ms,
    );

    // ========================================================================
    // 1C. 4-lane concurrent AR (M=4, 128 output tokens/lane)
    // ========================================================================
    for _ in 0..WARMUP_RUNS {
        let _ = run_4lane_ar(&mut inner, &four_lane_prompt_ids, DECODE_OUT_TOKENS);
    }
    clear_cache();
    let ar4 = run_4lane_ar(&mut inner, &four_lane_prompt_ids, DECODE_OUT_TOKENS);
    let ar4_agg_tps = ar4.total_decoded_tokens as f64 / (ar4.total_ms / 1000.0);
    let ar4_lane_tps = ar4_agg_tps / 4.0;
    let ar4_ms_step = ar4.total_ms / ar4.steps as f64;

    eprintln!(
        "\n[1.3] 4-lane Concurrent AR (M=4, {DECODE_OUT_TOKENS} output tokens/lane, INT8-KTM):"
    );
    eprintln!(
        "  Throughput:  aggregate = {ar4_agg_tps:.2} tok/s | per-lane = {ar4_lane_tps:.2} tok/s  ({ar4_ms_step:.3} ms/step, {} tokens in {:.1} ms)",
        ar4.total_decoded_tokens, ar4.total_ms,
    );

    // ========================================================================
    // 1D. 4-lane concurrent DSpark (Fixed k=3..7 vs Adaptive k, 128 output tokens/lane)
    // ========================================================================
    for _ in 0..WARMUP_RUNS {
        let _ = run_4lane_dspark(
            &mut inner,
            &four_lane_prompt_ids,
            DECODE_OUT_TOKENS,
            DsparkDepthMode::Fixed(7),
        );
    }
    eprintln!(
        "\n[1.4] 4-lane Concurrent DSpark Sweep ({DECODE_OUT_TOKENS} out tok/lane, 4-lane AR={ar4_agg_tps:.2} agg tok/s):"
    );
    eprintln!(
        "  {:<18} {:>9} {:>9} {:>7} {:>8} {:>7} {:>8} {:>16}",
        "Mode", "agg tok/s", "lane/s", "vs AR4", "ms/wave", "waves", "acc_rate", "exact/prefixes"
    );
    for &(label, mode) in &[
        ("fixed k=3 (M<=16)", DsparkDepthMode::Fixed(3)),
        ("fixed k=4 (M<=20)", DsparkDepthMode::Fixed(4)),
        ("fixed k=5 (M<=24)", DsparkDepthMode::Fixed(5)),
        ("fixed k=7 (M<=32)", DsparkDepthMode::Fixed(7)),
        ("adaptive k[3..7]", DsparkDepthMode::Adaptive),
    ] {
        clear_cache();
        let dp4 = run_4lane_dspark(&mut inner, &four_lane_prompt_ids, DECODE_OUT_TOKENS, mode);
        let agg = dp4.total_decoded_tokens as f64 / (dp4.total_ms / 1000.0);
        let lane = agg / 4.0;
        let ms_w = dp4.total_ms / dp4.waves.max(1) as f64;
        let acc_rate = 100.0 * dp4.total_accepted_drafts as f64 / dp4.total_drafted.max(1) as f64;
        let exact_lanes = (0..4).filter(|&i| dp4.tokens[i] == ar4.tokens[i]).count();
        let prefixes: Vec<usize> = (0..4)
            .map(|i| {
                dp4.tokens[i]
                    .iter()
                    .zip(ar4.tokens[i].iter())
                    .take_while(|(a, b)| a == b)
                    .count()
            })
            .collect();
        eprintln!(
            "  {:<18} {:>8.2}  {:>8.2}  {:>6.2}x {:>7.3}  {:>6}  {:>6.1}%  {exact_lanes}/4 {prefixes:?}  (p={:.2} v={:.2} c={:.2} m32={})",
            label,
            agg,
            lane,
            agg / ar4_agg_tps.max(1e-6),
            ms_w,
            dp4.waves,
            acc_rate,
            dp4.propose_ms / dp4.waves.max(1) as f64,
            dp4.verify_ms / dp4.waves.max(1) as f64,
            dp4.commit_ms / dp4.waves.max(1) as f64,
            dp4.full_m32_waves,
        );
        if matches!(mode, DsparkDepthMode::Fixed(7) | DsparkDepthMode::Adaptive) {
            let prop_w = dp4.propose_ms / dp4.waves.max(1) as f64;
            let ver_w = dp4.verify_ms / dp4.waves.max(1) as f64;
            let com_w = dp4.commit_ms / dp4.waves.max(1) as f64;
            eprintln!(
                "    -> split: propose={prop_w:.3}ms + verify={ver_w:.3}ms + commit={com_w:.3}ms (full_m32_waves={})",
                dp4.full_m32_waves
            );
        }
    }

    // ========================================================================
    // 1E. Prefill Throughput at exact 2048 and 8192 tokens
    // ========================================================================
    let prefill_2k_ids = build_long_prompt(&inner, 2048);
    let prefill_8k_ids = build_long_prompt(&inner, 8192);
    let (pre2k_ms, pre2k_tps) = measure_prefill(&mut inner, &prefill_2k_ids);
    let (pre8k_ms, pre8k_tps) = measure_prefill(&mut inner, &prefill_8k_ids);
    eprintln!("\n[1.5] Prefill Throughput (INT8-KTM, warmup {WARMUP_RUNS} runs):");
    eprintln!(
        "  2K ({} tok): {pre2k_tps:.1} tok/s  ({pre2k_ms:.1} ms)",
        prefill_2k_ids.len()
    );
    eprintln!(
        "  8K ({} tok): {pre8k_tps:.1} tok/s  ({pre8k_ms:.1} ms)",
        prefill_8k_ids.len()
    );

    // ========================================================================
    // 2. Per-Stage GPU Micro-Breakdown on M=1 and M=8 (exact ctx=512, real gen tokens)
    // ========================================================================
    clear_cache();
    let ctx_512_ids = build_long_prompt(&inner, 512);
    let m1_tok = &ar1.tokens[16..17];
    let m8_toks = &ar1.tokens[16..24];
    let bd_m1 = profile_48_visits(&mut inner, &ctx_512_ids, m1_tok);
    print_stage_table("Single-seq AR Decode", 1, ctx_512_ids.len(), &bd_m1);

    clear_cache();
    let bd_m8 = profile_48_visits(&mut inner, &ctx_512_ids, m8_toks);
    print_stage_table("Single-seq DSpark Verify", 8, ctx_512_ids.len(), &bd_m8);

    // ========================================================================
    // 2B. DSpark propose_block Micro-Breakdown (ctx=512, K=8 batched per stage)
    // ========================================================================
    clear_cache();
    let dp_bd = profile_dspark_propose(&mut inner, &ctx_512_ids);
    let dp_tot = dp_bd.isolated_total_ms();
    let dp_pct = |v: f64| 100.0 * v / dp_tot.max(1e-9);
    eprintln!("\n================================================================================");
    eprintln!(
        "  DSPARK propose_block MICRO-BREAKDOWN (block=8, 7 drafts, ctx={}, K=8 batched)",
        ctx_512_ids.len()
    );
    eprintln!("================================================================================");
    eprintln!(
        "  1. Embed anchor + mask block build            {:>8.3} ms {:>6.1}%",
        dp_bd.embed_mask_ms,
        dp_pct(dp_bd.embed_mask_ms)
    );
    eprintln!(
        "  2. 3 Draft Transformer Layers (total)         {:>8.3} ms {:>6.1}%",
        dp_bd.draft_layers_total_ms(),
        dp_pct(dp_bd.draft_layers_total_ms())
    );
    eprintln!(
        "       3x Attention (QKV+Norm+RoPE+SDPA+O)      {:>8.3} ms {:>6.1}%",
        dp_bd.draft_layers_attn_ms,
        dp_pct(dp_bd.draft_layers_attn_ms)
    );
    eprintln!(
        "       3x MLP (Norm+GateUp+SwiGLU+Down)         {:>8.3} ms {:>6.1}%",
        dp_bd.draft_layers_mlp_ms,
        dp_pct(dp_bd.draft_layers_mlp_ms)
    );
    eprintln!(
        "  3. Draft output_norm + slice [1..8]           {:>8.3} ms {:>6.1}%",
        dp_bd.draft_out_norm_ms,
        dp_pct(dp_bd.draft_out_norm_ms)
    );
    eprintln!(
        "  4. Shared target lm_head ([7, 2304]->V)       {:>8.3} ms {:>6.1}%",
        dp_bd.target_lm_head_ms,
        dp_pct(dp_bd.target_lm_head_ms)
    );
    eprintln!(
        "  5. 7 Markov Recurrent Steps (total)           {:>8.3} ms {:>6.1}%",
        dp_bd.markov_7steps_total_ms(),
        dp_pct(dp_bd.markov_7steps_total_ms())
    );
    eprintln!(
        "       ctx precompute ([7, 4608] @ W_ctx)       {:>8.3} ms {:>6.1}%",
        dp_bd.markov_ctx_precomp_ms,
        dp_pct(dp_bd.markov_ctx_precomp_ms)
    );
    eprintln!(
        "       7x RNN gate (w1+rec_mm+gate)             {:>8.3} ms {:>6.1}%",
        dp_bd.markov_rnn_gate_ms,
        dp_pct(dp_bd.markov_rnn_gate_ms)
    );
    let fold_label = if dp_bd.markov_softcap_argmax_ms == 0.0 {
        "7x fused Q8_0 fold+softcap+argmax"
    } else {
        "7x folded lm_head ([1,256]@[256,V]+b)"
    };
    eprintln!(
        "       {:<39}{:>8.3} ms {:>6.1}%",
        fold_label,
        dp_bd.markov_fold_matmul_ms,
        dp_pct(dp_bd.markov_fold_matmul_ms)
    );
    if dp_bd.markov_softcap_argmax_ms > 0.0 {
        eprintln!(
            "       7x add + slice + softcap + argmax        {:>8.3} ms {:>6.1}%",
            dp_bd.markov_softcap_argmax_ms,
            dp_pct(dp_bd.markov_softcap_argmax_ms)
        );
    }
    eprintln!("  ------------------------------------------------------------------");
    eprintln!(
        "  Sum of isolated stage blocks                  {:>8.3} ms  100.0%",
        dp_tot
    );
    eprintln!(
        "  Unsynced full propose_block + eval            {:>8.3} ms  (delta: {:+.3} ms)",
        dp_bd.unsynced_total_ms,
        dp_tot - dp_bd.unsynced_total_ms
    );
    eprintln!("================================================================================\n");
}
