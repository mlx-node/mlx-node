//! Reusable release-mode roofline benchmark & per-stage GPU micro-profiler for Inkling-2-Mini.
//!
//! Run on M5 Max (release mode so CPU host overhead is real release speed):
//! ```bash
//! cargo test --release -p mlx-core --lib models::inkling2::bench::inkling2_roofline_profile -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use mlx_paged_attn::CacheRecency;

use crate::array::attention::scaled_dot_product_attention;
use crate::array::{DType, MxArray, clear_cache, synchronize, synchronize_and_clear_cache};
use crate::disk_tier::DiskTierOptions;
use crate::engine::extract_chat_params;
use crate::engine::spec_paged::SpecPagedCache;
use crate::engine::types::ChatConfig;
use crate::models::quantized_linear::LinearProj;
use crate::models::qwen3_5::dflash2::{DFlash2Model, DFlash2TargetView};
use crate::models::qwen3_5::verify_tree::{
    LookupIndex, MAX_TREE_NODES,
    cuda::{TreeLookupSource, finalize_tree_greedy_walk},
};
use crate::nn::{Activations, Embedding, Linear};
use crate::stream::WiredResidency;
use crate::tools::inkling::{
    Inkling2RenderOptions, InklingMessage, render_inkling2_messages_to_ids,
};
use crate::transformer::Activations as TransformerActivations;
use crate::transformer::kv_cache::KvFormat;
use crate::transformer::paged_kv_cache_adapter::{PagedKVCacheAdapter, PagedRaggedRow, SeqId};

use super::attention::{Inkling2Attention, global_log_scale_tau_for_positions};
use super::attn_res::{ATTN_RES_EPS, apply_sublayer_rms_norm};
use super::config::Inkling2Config;
use super::dflash2::{
    Inkling2Dflash2Context, Inkling2VerifyTreeMode, Inkling2VerifyTreePolicy,
    is_strict_linear_chain, verify_tree_to_inkling2_rows,
};
use super::dspark::{
    AdaptiveDraftDepth, BatchedDsparkLane, Inkling2DsparkContext, Inkling2DsparkModel,
    resolve_inkling2_dspark_candidates,
};
use super::kernels::{
    Inkling2KvDtype, Inkling2TreeRow, encode_inkling2_tree_rows, inkling2_block_attn_res,
    inkling2_gate_sink, inkling2_grouped_moe_available, inkling2_lm_head_softcap_argmax,
    inkling2_markov_fold_softcap_argmax, inkling2_markov_gate, inkling2_moe_combine,
    inkling2_moe_route, inkling2_moe_router_proj, inkling2_paged_attention,
    inkling2_paged_attention_ex, inkling2_qr_clip, inkling2_sconv_k4_kv_prep,
    inkling2_sconv_k4_kv_prep_paged, inkling2_sconv_k4_ragged, inkling2_situ,
};
use super::kv_cache::Inkling2PagedConfig;
use super::model::{
    AttnInputNorm, Inkling2Inner, Inkling2PhysicalLayer, default_decode_share,
    next_inkling2_model_id,
};
use super::moe::{ExpertSwitchProj, Inkling2MoE};
use super::paged_forward::Inkling2VerifyTreeInput;
use super::persistence::{load_from_dir, test_real_model_path};
use super::scheduled_dspark::Inkling2ScheduledDsparkStates;
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
    render_prompt_with_effort(inner, text, Some(0.0))
}

fn render_prompt_with_effort(
    inner: &Inkling2Inner,
    text: &str,
    reasoning_effort: Option<f64>,
) -> Vec<u32> {
    let tokenizer = inner.tokenizer.as_ref().expect("tokenizer loaded");
    let mut opts = Inkling2RenderOptions::default();
    if let Some(effort) = reasoning_effort {
        opts.reasoning_effort = effort;
    }
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

    // Allow selecting DSpark or DFlash2 explicitly via `MLX_INKLING2_BENCH_DRAFT=dspark|dflash2|auto`.
    let bench_draft_pref = std::env::var("MLX_INKLING2_BENCH_DRAFT")
        .unwrap_or_else(|_| "auto".to_string())
        .to_ascii_lowercase();
    if bench_draft_pref == "dspark" && inner.dspark.is_none() {
        let source_dir = if model_dir.is_file() {
            model_dir.parent().unwrap_or(model_dir)
        } else {
            model_dir
        };
        let is_gguf = model_path.to_ascii_lowercase().ends_with(".gguf");
        let candidates = resolve_inkling2_dspark_candidates(source_dir, source_dir, is_gguf, None)
            .expect("resolve_inkling2_dspark_candidates");
        for path in &candidates {
            if let Ok(draft) = Inkling2DsparkModel::load_auto(path, &inner.config, &inner.lm_head) {
                inner.dspark = Some(draft);
                inner.dflash2 = None;
                break;
            }
        }
    }
    assert!(
        inner.dspark.is_some() || inner.dflash2.is_some(),
        "Expected DSpark or DFlash2 draft model loaded (MLX_INKLING2_BENCH_DRAFT={bench_draft_pref})"
    );
    let kv_dtype = inner.paged.as_ref().expect("paged").coordinator.kv_dtype();
    assert_eq!(
        kv_dtype,
        Inkling2KvDtype::Int8,
        "Expected default INT8-KTM KV cache"
    );
    synchronize_and_clear_cache();

    let (draft_kind, draft_mib) = if let Some(ref d) = inner.dflash2 {
        ("DFlash2", d.weight_bytes as f64 / (1024.0 * 1024.0))
    } else {
        (
            "DSpark",
            inner.dspark.as_ref().expect("d").weight_bytes() as f64 / (1024.0 * 1024.0),
        )
    };

    eprintln!("\n================================================================================");
    eprintln!("  INKLING-2-MINI M5 MAX RELEASE ROOFLINE & MICRO-PROFILE BASELINE");
    eprintln!("  Model: {model_path}");
    eprintln!(
        "  Loaded in {:.2}s | KV: {:?} | Weights: {:.2} GiB | {draft_kind}: {:.1} MiB | Release: {}",
        t_load.elapsed().as_secs_f64(),
        kv_dtype,
        inner.weight_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
        draft_mib,
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
    // 1B. Single-seq Speculative Sweep on BOTH prompts (DSpark or DFlash2)
    // ========================================================================
    let code_comp_prompt_ids = render_prompt(&inner, CODE_COMPLETION_PROMPT);
    let ar_code = run_single_ar(&mut inner, &code_comp_prompt_ids, DECODE_OUT_TOKENS);
    let ar_code_tps = 1000.0 / (ar_code.total_ms / ar_code.steps as f64);

    if inner.dspark.is_some() {
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
                let mean_acc =
                    (dp.total_accepted_drafts + dp.cycles) as f64 / dp.cycles.max(1) as f64;
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
    } else if inner.dflash2.is_some() {
        let chain_policy = Inkling2VerifyTreePolicy {
            mode: Inkling2VerifyTreeMode::Off,
            ..Inkling2VerifyTreePolicy::default()
        };
        let hybrid_policy = Inkling2VerifyTreePolicy {
            mode: Inkling2VerifyTreeMode::Auto,
            ..Inkling2VerifyTreePolicy::from_env()
        };
        for _ in 0..WARMUP_RUNS {
            let _ = run_single_dflash2(
                &mut inner,
                &single_prompt_ids,
                DECODE_OUT_TOKENS,
                chain_policy,
            );
            let _ = run_single_dflash2(
                &mut inner,
                &single_prompt_ids,
                DECODE_OUT_TOKENS,
                hybrid_policy,
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
                "\n[1.2] Single-seq DFlash2 — {prompt_label} ({DECODE_OUT_TOKENS} out tok, AR={ref_tps:.2} tok/s):"
            );
            eprintln!(
                "  {:<22} {:>9} {:>7} {:>8} {:>8} {:>8} {:>7} {:>9}",
                "Mode", "tok/s", "vs AR", "ms/cyc", "accept", "avg_T", "cyc", "exact_ar"
            );
            for (label, pol) in [
                ("DFlash2 1D Chain M=8", chain_policy),
                ("DFlash2 Hybrid (auto)", hybrid_policy),
            ] {
                clear_cache();
                let df = run_single_dflash2(&mut inner, p_ids, DECODE_OUT_TOKENS, pol);
                let tps = df.decoded_tokens as f64 / (df.total_ms / 1000.0);
                let ms_cyc = df.total_ms / df.cycles.max(1) as f64;
                let mean_acc =
                    (df.total_accepted_drafts + df.cycles) as f64 / df.cycles.max(1) as f64;
                let avg_t = df.total_tree_rows as f64 / df.cycles.max(1) as f64;
                let exact = df
                    .tokens
                    .iter()
                    .zip(ar_ref.tokens.iter())
                    .take_while(|(a, b)| a == b)
                    .count();
                eprintln!(
                    "  {:<22} {:>8.2}  {:>6.2}x {:>7.3}  {:>7.3}  {:>7.2}  {:>6}  {:>4}/{:<4}",
                    label,
                    tps,
                    tps / ref_tps.max(1e-6),
                    ms_cyc,
                    mean_acc,
                    avg_t,
                    df.cycles,
                    exact,
                    ar_ref.tokens.len(),
                );
            }
        }
    }

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
    if inner.dspark.is_some() {
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
            "Mode",
            "agg tok/s",
            "lane/s",
            "vs AR4",
            "ms/wave",
            "waves",
            "acc_rate",
            "exact/prefixes"
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
            let acc_rate =
                100.0 * dp4.total_accepted_drafts as f64 / dp4.total_drafted.max(1) as f64;
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
    } else {
        eprintln!(
            "\n[1.4] Skipping 4-lane DSpark sweep (DFlash2 active; set MLX_INKLING2_BENCH_DRAFT=dspark to run DSpark)"
        );
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
    // 2. Per-Stage GPU Micro-Breakdown on M=1, M=8, and M=16 (exact ctx=512)
    // ========================================================================
    clear_cache();
    let ctx_512_ids = build_long_prompt(&inner, 512);
    let m1_tok = &ar1.tokens[16..17];
    let m8_toks = &ar1.tokens[16..24];
    let m16_toks = &ar1.tokens[16..32];
    let bd_m1 = profile_48_visits(&mut inner, &ctx_512_ids, m1_tok);
    print_stage_table("Single-seq AR Decode", 1, ctx_512_ids.len(), &bd_m1);

    clear_cache();
    let bd_m8 = profile_48_visits(&mut inner, &ctx_512_ids, m8_toks);
    print_stage_table("Single-seq Spec Verify", 8, ctx_512_ids.len(), &bd_m8);

    clear_cache();
    let bd_m16 = profile_48_visits(&mut inner, &ctx_512_ids, m16_toks);
    print_stage_table("Single-seq Tree Verify", 16, ctx_512_ids.len(), &bd_m16);

    // ========================================================================
    // 2B. Draft propose_block Micro-Breakdown (DSpark or DFlash2)
    // ========================================================================
    clear_cache();
    if inner.dspark.is_some() {
        let dp_bd = profile_dspark_propose(&mut inner, &ctx_512_ids);
        let dp_tot = dp_bd.isolated_total_ms();
        let dp_pct = |v: f64| 100.0 * v / dp_tot.max(1e-9);
        eprintln!(
            "\n================================================================================"
        );
        eprintln!(
            "  DSPARK propose_block MICRO-BREAKDOWN (block=8, 7 drafts, ctx={}, K=8 batched)",
            ctx_512_ids.len()
        );
        eprintln!(
            "================================================================================"
        );
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
        eprintln!(
            "================================================================================\n"
        );
    } else if inner.dflash2.is_some() {
        let df_bd = profile_dflash2_propose(&mut inner, &ctx_512_ids);
        print_dflash2_propose_table(ctx_512_ids.len(), &df_bd);
    }
}

// ============================================================================
// Part 3: Dedicated Inkling-2-Mini DFlash2 Benchmark + Verify Tree Parity Test
// ============================================================================

const CODE_EDIT_AGENT_PROMPT: &str = "\
You are a code refactoring assistant. Below is an existing Rust ring-buffer module. \
Rewrite the ENTIRE file verbatim, keeping all existing structs, doc comments, methods, \
and unit tests intact, and ONLY add a new method `pub fn peek_back(&self) -> Option<&T>` \
right after `peek_front`, plus one assertion testing `peek_back` at the end of `test_ring_basic`.

```rust
/// Fixed-capacity FIFO ring buffer backed by a contiguous vector.
#[derive(Debug, Clone)]
pub struct RingBuffer<T> {
    data: Vec<Option<T>>,
    head: usize,
    len: usize,
    evicted_count: u64,
}

impl<T> RingBuffer<T> {
    /// Allocate an empty ring buffer with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(capacity > 0, \"capacity must be positive\");
        let mut data = Vec::with_capacity(capacity);
        data.resize_with(capacity, || None);
        Self {
            data,
            head: 0,
            len: 0,
            evicted_count: 0,
        }
    }

    /// Maximum number of elements that can be retained without eviction.
    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    /// Current number of live elements stored in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true when no live elements are stored.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns true when pushing another element will evict the oldest element.
    pub fn is_full(&self) -> bool {
        self.len == self.data.len()
    }

    /// Total number of elements evicted due to capacity overwrite.
    pub fn evicted_count(&self) -> u64 {
        self.evicted_count
    }

    /// Remove all elements while preserving the underlying allocation.
    pub fn clear(&mut self) {
        for slot in &mut self.data {
            *slot = None;
        }
        self.head = 0;
        self.len = 0;
    }

    /// Push a new element at the back, returning any evicted oldest element.
    pub fn push(&mut self, value: T) -> Option<T> {
        let cap = self.data.len();
        if self.len == cap {
            let evicted = self.data[self.head].replace(value);
            self.head = (self.head + 1) % cap;
            self.evicted_count += 1;
            evicted
        } else {
            let tail = (self.head + self.len) % cap;
            self.data[tail] = Some(value);
            self.len += 1;
            None
        }
    }

    /// Pop and return the oldest element from the front of the buffer.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        let value = self.data[self.head].take();
        self.head = (self.head + 1) % self.data.len();
        self.len -= 1;
        value
    }

    /// Inspect the oldest element at the front without removing it.
    pub fn peek_front(&self) -> Option<&T> {
        if self.len == 0 {
            None
        } else {
            self.data[self.head].as_ref()
        }
    }

    /// Drain all elements in FIFO order into a newly allocated vector.
    pub fn drain_fifo(&mut self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.len);
        while let Some(item) = self.pop() {
            out.push(item);
        }
        out
    }

    /// Collect references to all live elements in FIFO order.
    pub fn to_vec_refs(&self) -> Vec<&T> {
        let cap = self.data.len();
        let mut out = Vec::with_capacity(self.len);
        for offset in 0..self.len {
            if let Some(ref item) = self.data[(self.head + offset) % cap] {
                out.push(item);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_basic() {
        let mut rb = RingBuffer::with_capacity(3);
        assert!(rb.is_empty());
        assert_eq!(rb.capacity(), 3);
        assert_eq!(rb.push(10), None);
        assert_eq!(rb.push(20), None);
        assert_eq!(rb.push(30), None);
        assert!(rb.is_full());
        assert_eq!(rb.peek_front(), Some(&10));
        assert_eq!(rb.push(40), Some(10));
        assert_eq!(rb.evicted_count(), 1);
        assert_eq!(rb.to_vec_refs(), vec![&20, &30, &40]);
        assert_eq!(rb.pop(), Some(20));
        assert_eq!(rb.len(), 2);
        assert_eq!(rb.drain_fifo(), vec![30, 40]);
        assert!(rb.is_empty());
    }
}
```";

const MATH_REASONING_PROMPT: &str = "\
Find the number of ordered triples of positive integers (a, b, c) such that \
lcm(a, b) = 2^3 * 3^2 * 5, lcm(b, c) = 2^2 * 3^3 * 5^2, and lcm(c, a) = 2^3 * 3^3 * 5^2, \
and prove your answer completely by analyzing each prime factor independently, verifying \
all boundary cases, and computing the exact final product.";

/// Realistic multi-module Rust async HTTP gateway & cache codebase (~2,048 tokens)
/// properly wrapped in Inkling-2-Mini's chat template (preserving `<|end|><|start|>assistant`
/// framing intact instead of truncating raw token IDs).
fn build_2k_coding_prompt(inner: &Inkling2Inner, reasoning_effort: Option<f64>) -> Vec<u32> {
    const HEADER: &str = "\
You are a senior Rust systems engineer reviewing the existing edge HTTP reverse-proxy crate \
below. Read every module carefully before answering.\n\n```rust\n";

    const MODULES: &[&str] = &[
        "\
// === src/config.rs ===
use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct UpstreamTarget {
    pub name: String,
    pub addr: SocketAddr,
    pub weight: u32,
    pub max_inflight: usize,
    pub connect_timeout: Duration,
    pub read_timeout: Duration,
    pub health_path: String,
}

#[derive(Debug, Clone)]
pub struct RouteRule {
    pub prefix: String,
    pub upstream_group: String,
    pub strip_prefix: bool,
    pub cache_ttl: Option<Duration>,
    pub rate_limit_rps: Option<u32>,
    pub max_retries: u8,
}

#[derive(Debug, Clone)]
pub struct GatewayConfig {
    pub listen_addr: SocketAddr,
    pub max_body_bytes: usize,
    pub idle_pool_timeout: Duration,
    pub max_idle_per_host: usize,
    pub routes: Vec<RouteRule>,
    pub upstreams: HashMap<String, Vec<UpstreamTarget>>,
}

impl GatewayConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.max_body_bytes == 0 {
            return Err(\"max_body_bytes must be > 0\".into());
        }
        for route in &self.routes {
            if !route.prefix.starts_with('/') {
                return Err(format!(\"route prefix '{}' must start with '/'\", route.prefix));
            }
            let targets = self
                .upstreams
                .get(&route.upstream_group)
                .ok_or_else(|| format!(\"unknown upstream group '{}'\", route.upstream_group))?;
            if targets.is_empty() {
                return Err(format!(\"upstream group '{}' is empty\", route.upstream_group));
            }
            if targets.iter().all(|t| t.weight == 0) {
                return Err(format!(\"upstream group '{}' has zero total weight\", route.upstream_group));
            }
        }
        Ok(())
    }
}
",
        "\
// === src/rate_limiter.rs ===
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
struct Bucket {
    tokens: f64,
    capacity: f64,
    refill_per_sec: f64,
    last_refill: Instant,
}

impl Bucket {
    fn new(rps: u32, burst: u32, now: Instant) -> Self {
        let cap = f64::from(burst.max(1));
        Self {
            tokens: cap,
            capacity: cap,
            refill_per_sec: f64::from(rps.max(1)),
            last_refill: now,
        }
    }

    fn try_acquire(&mut self, cost: f64, now: Instant) -> Result<(), Duration> {
        let elapsed = now.saturating_duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_per_sec).min(self.capacity);
        self.last_refill = now;
        if self.tokens >= cost {
            self.tokens -= cost;
            Ok(())
        } else {
            let missing = cost - self.tokens;
            let wait_secs = (missing / self.refill_per_sec).max(0.001);
            Err(Duration::from_secs_f64(wait_secs))
        }
    }
}

pub struct KeyedRateLimiter {
    buckets: HashMap<String, Bucket>,
    max_keys: usize,
    default_burst_factor: u32,
}

impl KeyedRateLimiter {
    pub fn new(max_keys: usize, default_burst_factor: u32) -> Self {
        Self {
            buckets: HashMap::with_capacity(max_keys.min(1024)),
            max_keys: max_keys.max(16),
            default_burst_factor: default_burst_factor.max(1),
        }
    }

    pub fn check(&mut self, key: &str, rps: u32, now: Instant) -> Result<(), Duration> {
        if !self.buckets.contains_key(key) && self.buckets.len() >= self.max_keys {
            self.evict_stale(now, Duration::from_secs(60));
        }
        let burst = rps.saturating_mul(self.default_burst_factor);
        let bucket = self
            .buckets
            .entry(key.to_owned())
            .or_insert_with(|| Bucket::new(rps, burst, now));
        bucket.try_acquire(1.0, now)
    }

    fn evict_stale(&mut self, now: Instant, max_idle: Duration) {
        self.buckets
            .retain(|_, b| now.saturating_duration_since(b.last_refill) < max_idle);
        if self.buckets.len() >= self.max_keys {
            if let Some(oldest) = self
                .buckets
                .iter()
                .min_by_key(|(_, b)| b.last_refill)
                .map(|(k, _)| k.clone())
            {
                self.buckets.remove(&oldest);
            }
        }
    }
}
",
        "\
// === src/cache.rs ===
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct CachedResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    pub stored_at: Instant,
    pub ttl: Duration,
    pub etag: Option<String>,
}

impl CachedResponse {
    pub fn is_fresh(&self, now: Instant) -> bool {
        now.saturating_duration_since(self.stored_at) < self.ttl
    }

    pub fn byte_len(&self) -> usize {
        self.body.len()
            + self
                .headers
                .iter()
                .map(|(k, v)| k.len() + v.len())
                .sum::<usize>()
    }
}

pub struct TtlResponseCache {
    entries: HashMap<String, CachedResponse>,
    order: VecDeque<String>,
    current_bytes: usize,
    max_bytes: usize,
    hits: u64,
    misses: u64,
}

impl TtlResponseCache {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            current_bytes: 0,
            max_bytes: max_bytes.max(4096),
            hits: 0,
            misses: 0,
        }
    }

    pub fn get(&mut self, key: &str, now: Instant) -> Option<&CachedResponse> {
        let expired = self.entries.get(key).is_some_and(|e| !e.is_fresh(now));
        if expired {
            self.remove(key);
            self.misses += 1;
            return None;
        }
        if self.entries.contains_key(key) {
            self.hits += 1;
            self.entries.get(key)
        } else {
            self.misses += 1;
            None
        }
    }

    pub fn insert(&mut self, key: String, resp: CachedResponse) -> bool {
        let size = resp.byte_len();
        if size > self.max_bytes / 4 {
            return false;
        }
        self.remove(&key);
        while self.current_bytes + size > self.max_bytes {
            let Some(victim) = self.order.pop_front() else {
                break;
            };
            if let Some(removed) = self.entries.remove(&victim) {
                self.current_bytes = self.current_bytes.saturating_sub(removed.byte_len());
            }
        }
        self.current_bytes += size;
        self.order.push_back(key.clone());
        self.entries.insert(key, resp);
        true
    }

    pub fn remove(&mut self, key: &str) -> Option<CachedResponse> {
        let removed = self.entries.remove(key)?;
        self.current_bytes = self.current_bytes.saturating_sub(removed.byte_len());
        Some(removed)
    }

    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}
",
        "\
// === src/pool.rs ===
use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct PooledConn {
    pub id: u64,
    pub peer: SocketAddr,
    pub created_at: Instant,
    pub last_used: Instant,
    pub requests_served: u64,
}

pub struct ConnectionPool {
    idle: HashMap<SocketAddr, VecDeque<PooledConn>>,
    active_counts: HashMap<SocketAddr, usize>,
    next_id: u64,
    max_idle_per_host: usize,
    idle_timeout: Duration,
}

impl ConnectionPool {
    pub fn new(max_idle_per_host: usize, idle_timeout: Duration) -> Self {
        Self {
            idle: HashMap::new(),
            active_counts: HashMap::new(),
            next_id: 1,
            max_idle_per_host: max_idle_per_host.max(1),
            idle_timeout,
        }
    }

    pub fn checkout(
        &mut self,
        peer: SocketAddr,
        max_inflight: usize,
        now: Instant,
    ) -> Result<PooledConn, &'static str> {
        let active = self.active_counts.entry(peer).or_insert(0);
        if *active >= max_inflight {
            return Err(\"upstream inflight limit reached\");
        }
        if let Some(queue) = self.idle.get_mut(&peer) {
            while let Some(mut conn) = queue.pop_front() {
                if now.saturating_duration_since(conn.last_used) <= self.idle_timeout {
                    conn.last_used = now;
                    conn.requests_served += 1;
                    *active += 1;
                    return Ok(conn);
                }
            }
        }
        let id = self.next_id;
        self.next_id += 1;
        *active += 1;
        Ok(PooledConn {
            id,
            peer,
            created_at: now,
            last_used: now,
            requests_served: 1,
        })
    }

    pub fn checkin(&mut self, mut conn: PooledConn, reusable: bool, now: Instant) {
        if let Some(active) = self.active_counts.get_mut(&conn.peer) {
            *active = active.saturating_sub(1);
        }
        if !reusable {
            return;
        }
        conn.last_used = now;
        let queue = self.idle.entry(conn.peer).or_default();
        while queue.len() >= self.max_idle_per_host {
            queue.pop_front();
        }
        queue.push_back(conn);
    }

    pub fn active_for(&self, peer: SocketAddr) -> usize {
        self.active_counts.get(&peer).copied().unwrap_or(0)
    }
}
",
        "\
// === src/metrics.rs ===
use std::collections::BTreeMap;

#[derive(Debug, Default, Clone)]
pub struct LatencyHistogram {
    buckets_ms: [u64; 10],
    sum_ms: f64,
    count: u64,
}

impl LatencyHistogram {
    const BOUNDS_MS: [f64; 9] = [1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 1000.0];

    pub fn observe(&mut self, ms: f64) {
        let clamped = ms.max(0.0);
        let idx = Self::BOUNDS_MS
            .iter()
            .position(|&bound| clamped <= bound)
            .unwrap_or(Self::BOUNDS_MS.len());
        self.buckets_ms[idx] += 1;
        self.sum_ms += clamped;
        self.count += 1;
    }

    pub fn mean_ms(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_ms / self.count as f64
        }
    }

    pub fn count(&self) -> u64 {
        self.count
    }
}

#[derive(Debug, Default)]
pub struct MetricsRegistry {
    pub requests_total: BTreeMap<String, u64>,
    pub errors_total: BTreeMap<String, u64>,
    pub retries_total: BTreeMap<String, u64>,
    pub latency_by_upstream: BTreeMap<String, LatencyHistogram>,
}

impl MetricsRegistry {
    pub fn record_request(&mut self, upstream: &str, status: u16, latency_ms: f64) {
        *self.requests_total.entry(upstream.to_owned()).or_insert(0) += 1;
        if status >= 500 {
            *self.errors_total.entry(upstream.to_owned()).or_insert(0) += 1;
        }
        self.latency_by_upstream
            .entry(upstream.to_owned())
            .or_default()
            .observe(latency_ms);
    }

    pub fn record_retry(&mut self, upstream: &str) {
        *self.retries_total.entry(upstream.to_owned()).or_insert(0) += 1;
    }
}
",
        "\
// === src/router.rs ===
use std::collections::HashMap;
use std::time::Instant;

use crate::cache::TtlResponseCache;
use crate::config::{GatewayConfig, RouteRule, UpstreamTarget};
use crate::metrics::MetricsRegistry;
use crate::pool::ConnectionPool;
use crate::rate_limiter::KeyedRateLimiter;

#[derive(Debug, Clone)]
pub struct IncomingRequest {
    pub method: String,
    pub path: String,
    pub client_ip: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteError {
    NotFound,
    PayloadTooLarge,
    RateLimited { retry_after_ms: u64 },
    NoHealthyUpstream,
    UpstreamOverloaded,
}

pub struct ProxyRouter {
    config: GatewayConfig,
    rr_cursors: HashMap<String, usize>,
    pub limiter: KeyedRateLimiter,
    pub cache: TtlResponseCache,
    pub pool: ConnectionPool,
    pub metrics: MetricsRegistry,
}

impl ProxyRouter {
    pub fn new(config: GatewayConfig, cache_bytes: usize) -> Result<Self, String> {
        config.validate()?;
        let max_idle = config.max_idle_per_host;
        let idle_timeout = config.idle_pool_timeout;
        Ok(Self {
            config,
            rr_cursors: HashMap::new(),
            limiter: KeyedRateLimiter::new(4096, 2),
            cache: TtlResponseCache::new(cache_bytes),
            pool: ConnectionPool::new(max_idle, idle_timeout),
            metrics: MetricsRegistry::default(),
        })
    }

    pub fn match_route(&self, path: &str) -> Option<&RouteRule> {
        self.config
            .routes
            .iter()
            .filter(|r| path.starts_with(&r.prefix))
            .max_by_key(|r| r.prefix.len())
    }

    pub fn select_upstream(&mut self, group: &str) -> Option<&UpstreamTarget> {
        let targets = self.config.upstreams.get(group)?;
        let total_weight: u32 = targets.iter().map(|t| t.weight).sum();
        if total_weight == 0 {
            return None;
        }
        let cursor = self.rr_cursors.entry(group.to_owned()).or_insert(0);
        let pick = (*cursor as u32) % total_weight;
        *cursor = cursor.wrapping_add(1);
        let mut acc = 0u32;
        for target in targets {
            acc += target.weight;
            if pick < acc {
                return Some(target);
            }
        }
        targets.first()
    }

    pub fn preflight_request(
        &mut self,
        req: &IncomingRequest,
        now: Instant,
    ) -> Result<(String, Option<String>), RouteError> {
        if req.body.len() > self.config.max_body_bytes {
            return Err(RouteError::PayloadTooLarge);
        }
        let route = self.match_route(&req.path).ok_or(RouteError::NotFound)?.clone();
        if let Some(rps) = route.rate_limit_rps {
            let key = format!(\"{}:{}\", route.prefix, req.client_ip);
            if let Err(wait) = self.limiter.check(&key, rps, now) {
                return Err(RouteError::RateLimited {
                    retry_after_ms: wait.as_millis() as u64,
                });
            }
        }
        let cache_key = if req.method == \"GET\" && route.cache_ttl.is_some() {
            Some(format!(\"{}:{}\", route.upstream_group, req.path))
        } else {
            None
        };
        Ok((route.upstream_group, cache_key))
    }
}
",
        "\
// === src/headers.rs ===
pub fn normalize_forwarded_headers(
    headers: &[(String, String)],
    client_ip: &str,
    request_id: &str,
) -> Vec<(String, String)> {
    const HOP_BY_HOP: &[&str] = &[
        \"connection\",
        \"keep-alive\",
        \"proxy-authenticate\",
        \"proxy-authorization\",
        \"te\",
        \"trailers\",
        \"transfer-encoding\",
        \"upgrade\",
    ];
    let mut out = Vec::with_capacity(headers.len() + 2);
    let mut existing_xff: Option<String> = None;
    for (k, v) in headers {
        let lower = k.to_ascii_lowercase();
        if HOP_BY_HOP.contains(&lower.as_str()) {
            continue;
        }
        if lower == \"x-forwarded-for\" {
            existing_xff = Some(v.clone());
            continue;
        }
        out.push((k.clone(), v.clone()));
    }
    let xff = match existing_xff {
        Some(prev) if !prev.is_empty() => format!(\"{prev}, {client_ip}\"),
        _ => client_ip.to_owned(),
    };
    out.push((\"X-Forwarded-For\".to_owned(), xff));
    out.push((\"X-Request-Id\".to_owned(), request_id.to_owned()));
    out
}

pub fn rewrite_upstream_path(path: &str, prefix: &str, strip_prefix: bool) -> String {
    if !strip_prefix {
        return path.to_owned();
    }
    let rest = path.strip_prefix(prefix).unwrap_or(path);
    if rest.is_empty() {
        \"/\".to_owned()
    } else if rest.starts_with('/') {
        rest.to_owned()
    } else {
        format!(\"/{rest}\")
    }
}
",
    ];

    const TASK: &str = "\
```

Using the exact types and conventions in the crate above (`UpstreamTarget`, `RouteRule`, \
`MetricsRegistry`, `ProxyRouter`), implement a complete new `src/circuit_breaker.rs` module \
with:
1. A three-state `CircuitBreaker` (`Closed`, `Open { opened_at }`, `HalfOpen { probe_inflight }`) \
   tracked per upstream target name (`CircuitBreakerRegistry`), supporting configurable sliding \
   failure-rate thresholds, minimum request volume, open cooldown duration, and single-probe \
   half-open recovery.
2. An `AdaptiveRetryPolicy` that computes capped exponential backoff with deterministic jitter, \
   honors idempotent HTTP methods (`GET`, `HEAD`, `PUT`, `DELETE`, `OPTIONS`), skips tripped \
   upstreams using `CircuitBreakerRegistry`, and records retries on `MetricsRegistry`.
3. A `select_healthy_upstream` helper integrating with `ProxyRouter` and thorough unit tests \
   covering `Closed -> Open -> HalfOpen -> Closed` transitions, half-open failure re-opening, \
   and retry budget exhaustion.";

    // Assemble realistic modules up to ~2,048 rendered tokens (keeping chat framing 100% intact!).
    let mut code = String::with_capacity(10_000);
    for module in MODULES {
        code.push_str(module);
        code.push('\n');
    }
    let mut full = format!("{HEADER}{code}{TASK}");
    let mut ids = render_prompt_with_effort(inner, &full, reasoning_effort);
    // If slightly under 2,048 tokens on this tokenizer, add a realistic integration-test excerpt
    // module before closing the code fence so rendered length sits right around 2,048 tokens.
    let mut extra_idx = 0usize;
    while ids.len() < 2000 && extra_idx < 8 {
        extra_idx += 1;
        code.push_str(&format!(
            "\
// === src/health_probe_{extra_idx}.rs ===
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeVerdict{extra_idx} {{ Healthy, Degraded, Unhealthy }}

pub struct PassiveHealthWindow{extra_idx} {{
    consecutive_failures: u32,
    consecutive_successes: u32,
    failure_threshold: u32,
    recovery_threshold: u32,
}}

impl PassiveHealthWindow{extra_idx} {{
    pub fn new(failure_threshold: u32, recovery_threshold: u32) -> Self {{
        Self {{
            consecutive_failures: 0,
            consecutive_successes: 0,
            failure_threshold: failure_threshold.max(1),
            recovery_threshold: recovery_threshold.max(1),
        }}
    }}

    pub fn on_outcome(&mut self, ok: bool) -> ProbeVerdict{extra_idx} {{
        if ok {{
            self.consecutive_failures = 0;
            self.consecutive_successes = self.consecutive_successes.saturating_add(1);
            if self.consecutive_successes >= self.recovery_threshold {{
                ProbeVerdict{extra_idx}::Healthy
            }} else {{
                ProbeVerdict{extra_idx}::Degraded
            }}
        }} else {{
            self.consecutive_successes = 0;
            self.consecutive_failures = self.consecutive_failures.saturating_add(1);
            if self.consecutive_failures >= self.failure_threshold {{
                ProbeVerdict{extra_idx}::Unhealthy
            }} else {{
                ProbeVerdict{extra_idx}::Degraded
            }}
        }}
    }}
}}
"
        ));
        full = format!("{HEADER}{code}{TASK}");
        ids = render_prompt_with_effort(inner, &full, reasoning_effort);
    }
    // If slightly above 2,064 tokens, trim whole trailing lines from `code` BEFORE rendering
    // so `<|end|><|start|>assistant` framing is never truncated.
    if ids.len() > 2064 {
        let lines: Vec<&str> = code.lines().collect();
        let mut lo = 0usize;
        let mut hi = lines.len();
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            let candidate = format!("{HEADER}{}\n{TASK}", lines[..mid].join("\n"));
            let cand_ids = render_prompt_with_effort(inner, &candidate, reasoning_effort);
            if cand_ids.len() <= 2048 {
                lo = mid;
                ids = cand_ids;
            } else {
                hi = mid - 1;
            }
        }
    }
    ids
}

#[derive(Debug, Clone)]
struct Dflash2SingleResult {
    tokens: Vec<u32>,
    decoded_tokens: usize,
    cycles: usize,
    tree_cycles: usize,
    total_tree_rows: usize,
    total_accepted_drafts: usize,
    total_lookup_accepted: usize,
    non_contiguous_commits: usize,
    total_ms: f64,
    propose_ms: f64,
    target_verify_ms: f64,
    commit_ms: f64,
}

fn run_single_dflash2(
    inner: &mut Inkling2Inner,
    prompt_ids: &[u32],
    out_tokens: usize,
    tree_policy: Inkling2VerifyTreePolicy,
) -> Dflash2SingleResult {
    let seq_id = 0u32;
    inner.reset_paged_sequence(seq_id).expect("reset seq 0");
    let (tap_visits, draft_block_size, mut draft_ctx) = {
        let d = inner.dflash2.as_ref().expect("dflash2 loaded");
        (
            d.config.target_layers.clone(),
            d.config.block_size,
            Inkling2Dflash2Context::new(&d.config),
        )
    };
    let bench_prefix = crate::models::inkling2::paged_forward::Inkling2PrefixState {
        effective_cached_prefix_len: 0,
        suffix_len: prompt_ids.len(),
        full_tokens: prompt_ids.to_vec(),
        cache_salt: 0,
        live_continuation: false,
        reuse_cache: true,
    };
    let first_logits = inner
        .paged_prefill_dflash2_with_taps(seq_id, prompt_ids, &bench_prefix, &mut draft_ctx)
        .expect("dflash2 prefill");
    draft_ctx.eval_state().expect("eval dflash2 ctx");
    let mut anchor = first_logits
        .argmax(-1, Some(false))
        .expect("argmax")
        .item_at_int32(0)
        .expect("item") as u32;
    synchronize();

    let track_lookup = tree_policy.lookup_enabled && tree_policy.tree_path_active();
    if track_lookup {
        draft_ctx.observe_tokens(prompt_ids);
        draft_ctx.observe_tokens(&[anchor]);
    }

    let mut generated = Vec::with_capacity(out_tokens);
    generated.push(anchor);

    let mut cycles = 0usize;
    let mut tree_cycles = 0usize;
    let mut total_tree_rows = 0usize;
    let mut total_accepted_drafts = 0usize;
    let mut total_lookup_accepted = 0usize;
    let mut non_contiguous_commits = 0usize;
    let mut propose_ms = 0.0;
    let mut target_verify_ms = 0.0;
    let mut commit_ms = 0.0;

    let t_total = Instant::now();
    while generated.len() < out_tokens {
        let remaining = out_tokens - generated.len();
        let max_depth = remaining.saturating_sub(1).min(MAX_TREE_NODES - 1);
        let use_tree_cycle =
            tree_policy.should_use_tree_cycle(draft_ctx.peek_lookup_match_len(), max_depth);

        if !use_tree_cycle {
            // Zero-sync overlapped 1D Chain mode (M = 1 + cap <= 8)
            let cap = draft_block_size.min(remaining.saturating_sub(1));
            let t_prop = Instant::now();
            let mut prop_opt = if cap > 0 {
                let draft = inner.dflash2.as_ref().expect("dflash2");
                let mut view = inner.target_view();
                Some(
                    draft_ctx
                        .propose_greedy_chain(draft, &mut view, anchor, cap, true, false)
                        .expect("propose_greedy_chain"),
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
                MxArray::schedule_overlap(&[&v_ids]);
                Some(v_ids)
            } else {
                None
            };
            propose_ms += t_prop.elapsed().as_secs_f64() * 1000.0;

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
            if let Some(ref dev_ids) = device_verify_ids {
                MxArray::eval_arrays(&[&argmax, dev_ids]).expect("eval 1d");
            } else {
                argmax.eval();
            }
            target_verify_ms += t_ver.elapsed().as_secs_f64() * 1000.0;

            let t_com = Instant::now();
            let verified_ids: Vec<u32> = if let Some(ref dev_ids) = device_verify_ids {
                dev_ids
                    .to_int32()
                    .expect("dev_ids")
                    .iter()
                    .map(|&id| id as u32)
                    .collect()
            } else {
                let mut ids = Vec::with_capacity(verify_len);
                ids.push(anchor);
                if let Some(ref mut prop) = prop_opt {
                    prop.materialize_draft_ids().expect("materialize drafts");
                    ids.extend(prop.draft_ids.iter().map(|&id| id as u32));
                }
                ids
            };
            let proposal_ids = &verified_ids[1..];
            let target_argmax: Vec<u32> = argmax
                .to_int32()
                .expect("am")
                .iter()
                .map(|&v| v as u32)
                .collect();
            let mut k = 0usize;
            while k < proposal_ids.len() && target_argmax[k] == proposal_ids[k] {
                k += 1;
            }
            let boundary = target_argmax[k];
            let mut emit = Vec::with_capacity(k + 1);
            emit.extend_from_slice(&proposal_ids[..k]);
            emit.push(boundary);

            let emit_n = emit.len().min(remaining);
            let keep = 1 + emit_n.min(k);
            let tape = tapes.pop().expect("tape");
            {
                let rt = inner.paged.as_mut().expect("paged");
                if used_placeholders {
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
                rt.record_block_boundary_conv_from_tape(seq_id, committed, keep, &tape)
                    .expect("boundary conv");
            }
            let draft = inner.dflash2.as_ref().expect("dflash2");
            draft_ctx
                .append_taps(draft, &taps, &verified_ids[..keep])
                .expect("append_taps");
            if track_lookup {
                draft_ctx.observe_tokens(&emit[..emit_n]);
            }
            commit_ms += t_com.elapsed().as_secs_f64() * 1000.0;

            cycles += 1;
            total_tree_rows += verify_len;
            total_accepted_drafts += emit_n.saturating_sub(1);
            for &tok in &emit[..emit_n] {
                generated.push(tok);
                anchor = tok;
            }
        } else {
            // Verify Tree + SAM Prompt Lookup mode (M <= 16)
            let t_prop = Instant::now();
            let lattice = {
                let draft = inner.dflash2.as_ref().expect("dflash2");
                let mut view = inner.target_view();
                let (candidates, scores) = draft_ctx
                    .propose_tree_lattice(draft, &mut view, anchor, draft_block_size)
                    .expect("propose_tree_lattice");
                draft
                    .lattice_from_arrays(&candidates, &scores, draft_block_size)
                    .expect("lattice_from_arrays")
            };
            let lookup = draft_ctx.prepare_lookup(tree_policy.lookup_enabled, max_depth);
            let tree = tree_policy.build_tree(
                anchor,
                &lattice,
                &lookup,
                MAX_TREE_NODES,
                &draft_ctx.verify_budget,
                &draft_ctx.lookup_acc,
            );
            let tree_len = tree.len();
            let is_chain = is_strict_linear_chain(&tree);
            let encoded_tree = if is_chain {
                None
            } else {
                Some(verify_tree_to_inkling2_rows(&tree).expect("encode tree"))
            };
            propose_ms += t_prop.elapsed().as_secs_f64() * 1000.0;

            let t_ver = Instant::now();
            let (anchor_pos, ticket) = {
                let rt = inner.paged.as_mut().expect("paged");
                let ok = rt
                    .coordinator
                    .reserve_lookahead(seq_id, tree_len)
                    .expect("reserve");
                assert!(ok, "KV pool exhausted reserving {tree_len} tree rows");
                let pos = rt.coordinator.request_token_count_all(seq_id).expect("pos");
                let ticket = rt
                    .coordinator
                    .record_verify(seq_id, &tree.tokens)
                    .expect("record_verify");
                (pos, ticket)
            };
            let tree_input = encoded_tree
                .as_ref()
                .map(|(tree_rows, depths)| Inkling2VerifyTreeInput { tree_rows, depths });
            let row = PagedRaggedRow {
                seq_id,
                first_logical_position: anchor_pos,
                query_len: tree_len as u32,
            };
            let (argmax, taps, mut tapes) = inner
                .paged_verify_ragged_taped_tree(
                    &[(row, tree.tokens.clone())],
                    tree_input,
                    &tap_visits,
                    true,
                )
                .expect("paged_verify_ragged_taped_tree");
            argmax.eval();
            target_verify_ms += t_ver.elapsed().as_secs_f64() * 1000.0;

            let t_com = Instant::now();
            let target_argmax: Vec<u32> = argmax
                .to_int32()
                .expect("argmax i32")
                .iter()
                .map(|&v| v as u32)
                .collect();
            let (walk, lookup_acc) = finalize_tree_greedy_walk(
                &tree,
                &target_argmax,
                &lookup,
                TreeLookupSource::Lattice(&lattice),
                &mut draft_ctx.verify_budget,
                &mut draft_ctx.lookup_acc,
            );
            let emit_n = walk.emitted_tokens.len().min(remaining);
            let keep_nodes = &walk.accepted_nodes[..emit_n];
            let keep_contiguous = keep_nodes.iter().enumerate().all(|(i, &r)| r == i);
            let tape = tapes.pop().expect("tape");
            {
                let rt = inner.paged.as_mut().expect("paged");
                if !keep_contiguous {
                    rt.coordinator
                        .compact_kv_path_all(seq_id, anchor_pos, keep_nodes)
                        .expect("compact_kv_path_all");
                    rt.coordinator
                        .compact_recorded_tree_tokens_all(seq_id, tree_len, keep_nodes)
                        .expect("compact_recorded_tree_tokens_all");
                }
                rt.coordinator
                    .commit_cycle(seq_id, ticket, emit_n)
                    .expect("commit_cycle");
                let conv = rt.conv_states.get_mut(&seq_id).expect("conv");
                if keep_contiguous {
                    tape.commit_prefix(emit_n, conv).expect("commit_prefix");
                } else {
                    tape.commit_path(keep_nodes, conv).expect("commit_path");
                }
                let committed = rt.coordinator.request_token_count_all(seq_id).expect("pos");
                rt.coordinator
                    .settle_committed(seq_id, u64::from(committed))
                    .expect("settle_committed");
                if keep_contiguous {
                    rt.record_block_boundary_conv_from_tape(seq_id, committed, emit_n, &tape)
                        .expect("boundary conv");
                } else {
                    rt.record_block_boundary_conv_from_tape_path(
                        seq_id, committed, keep_nodes, &tape,
                    )
                    .expect("boundary conv path");
                }
            }
            let kept_tokens: Vec<u32> = keep_nodes.iter().map(|&r| tree.tokens[r]).collect();
            let draft = inner.dflash2.as_ref().expect("dflash2");
            draft_ctx
                .append_taps_path(draft, &taps, keep_nodes, &kept_tokens)
                .expect("append_taps_path");
            if track_lookup {
                draft_ctx.observe_tokens(&walk.emitted_tokens[..emit_n]);
            }
            commit_ms += t_com.elapsed().as_secs_f64() * 1000.0;

            cycles += 1;
            tree_cycles += 1;
            total_tree_rows += tree_len;
            total_accepted_drafts += emit_n.saturating_sub(1);
            total_lookup_accepted += lookup_acc;
            if !keep_contiguous {
                non_contiguous_commits += 1;
            }
            for &tok in &walk.emitted_tokens[..emit_n] {
                generated.push(tok);
                anchor = tok;
            }
        }
    }
    synchronize();
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    let decoded_tokens = generated.len().saturating_sub(1);

    Dflash2SingleResult {
        tokens: generated,
        decoded_tokens,
        cycles,
        tree_cycles,
        total_tree_rows,
        total_accepted_drafts,
        total_lookup_accepted,
        non_contiguous_commits,
        total_ms,
        propose_ms,
        target_verify_ms,
        commit_ms,
    }
}

#[derive(Default, Clone, Copy)]
struct Dflash2ProposeBreakdown {
    embed_norm_ms: f64,
    target_lm_head_7_ms: f64,
    chain_propose_total_ms: f64,
    tree_lattice_total_ms: f64,
    append_taps_keep4_ms: f64,
    append_taps_keep8_ms: f64,
    host_tree_build_us: f64,
}

fn profile_dflash2_propose(
    inner: &mut Inkling2Inner,
    ctx_prompt: &[u32],
) -> Dflash2ProposeBreakdown {
    let seq_id = 0u32;
    inner.reset_paged_sequence(seq_id).expect("reset");
    let (tap_visits, block_size, mask_id, mut draft_ctx) = {
        let d = inner.dflash2.as_ref().expect("dflash2");
        (
            d.config.target_layers.clone(),
            d.config.block_size,
            d.config.mask_token_id as i32,
            Inkling2Dflash2Context::new(&d.config),
        )
    };
    let bench_prefix = crate::models::inkling2::paged_forward::Inkling2PrefixState {
        effective_cached_prefix_len: 0,
        suffix_len: ctx_prompt.len(),
        full_tokens: ctx_prompt.to_vec(),
        cache_salt: 0,
        live_continuation: false,
        reuse_cache: true,
    };
    let first_logits = inner
        .paged_prefill_dflash2_with_taps(seq_id, ctx_prompt, &bench_prefix, &mut draft_ctx)
        .expect("dflash2 prefill");
    draft_ctx.eval_state().expect("eval ctx");
    let anchor = first_logits
        .argmax(-1, Some(false))
        .expect("am")
        .item_at_int32(0)
        .expect("item") as u32;
    synchronize();
    draft_ctx.observe_tokens(ctx_prompt);
    draft_ctx.observe_tokens(&[anchor]);

    let draft: &DFlash2Model = inner.dflash2.as_ref().expect("dflash2");

    // 1. Unsynced 1D Chain propose (block=8 -> 7 drafts, device path)
    let chain_propose_total_ms = time_stage_ms(5, 10, || {
        let mut view = inner.target_view();
        let mut prop = draft_ctx
            .propose_greedy_chain(draft, &mut view, anchor, block_size, true, false)
            .expect("propose_greedy_chain");
        prop.materialize_draft_ids().expect("mat");
        synchronize();
    });

    // 2. Unsynced Tree Lattice propose + 7.6 KB readback
    let tree_lattice_total_ms = time_stage_ms(5, 10, || {
        let mut view = inner.target_view();
        let (cands, scores) = draft_ctx
            .propose_tree_lattice(draft, &mut view, anchor, block_size)
            .expect("propose_tree_lattice");
        let _lat = draft
            .lattice_from_arrays(&cands, &scores, block_size)
            .expect("lattice");
        synchronize();
    });

    // 3. Isolated embed_tokens + embed_norm ([1, 8, 2304] BF16, K=8 batched)
    let mut block_ids = vec![mask_id; block_size + 1];
    block_ids[0] = anchor as i32;
    let block_arr = MxArray::from_int32(&block_ids, &[1, (block_size + 1) as i64]).expect("blk");
    let k_batch = 8usize;
    let embed_norm_ms = time_stage_ms(3, 6, || {
        let mut view = inner.target_view();
        let mut outs = Vec::with_capacity(k_batch);
        for _ in 0..k_batch {
            outs.push(view.embed(&block_arr, &block_ids).expect("embed"));
        }
        eval_owned_sync(&outs);
    }) / k_batch as f64;

    // 4. Isolated target shared lm_head ([1, 7, 2304] -> [1, 7, 200058] raw, K=8 batched)
    let sample_hidden = {
        let mut view = inner.target_view();
        let emb = view.embed(&block_arr, &block_ids).expect("embed");
        let h = emb
            .slice_axis(1, 1, (block_size + 1) as i64)
            .expect("slice");
        eval_sync(&[&h]);
        h
    };
    let target_lm_head_7_ms = time_stage_ms(3, 6, || {
        let mut view = inner.target_view();
        let mut outs = Vec::with_capacity(k_batch);
        for _ in 0..k_batch {
            outs.push(view.logits(&sample_hidden).expect("logits"));
        }
        eval_owned_sync(&outs);
    }) / k_batch as f64;

    // 5. Isolated append_taps (fuse_context + context_kv + ring scatter) on real 4-tap [8, 2304]
    let sample_ids = vec![anchor; block_size + 1];
    let (sample_taps, ticket, mut sample_tapes) = {
        let (pos, ticket) = {
            let rt = inner.paged.as_mut().expect("paged");
            let pos = rt.coordinator.request_token_count_all(seq_id).expect("pos");
            let ticket = rt
                .coordinator
                .record_verify(seq_id, &sample_ids)
                .expect("rec");
            (pos, ticket)
        };
        let row = PagedRaggedRow {
            seq_id,
            first_logical_position: pos,
            query_len: sample_ids.len() as u32,
        };
        let (_am, taps, tapes) = inner
            .paged_verify_ragged_taped_greedy(&[(row, sample_ids.clone())], &tap_visits)
            .expect("taps");
        eval_owned_sync(&taps);
        (taps, ticket, tapes)
    };
    if let Some(rt) = inner.paged.as_mut() {
        let _ = rt.coordinator.commit_cycle(seq_id, ticket, 0);
        if let Some(tape) = sample_tapes.pop()
            && let Some(conv) = rt.conv_states.get_mut(&seq_id)
        {
            let _ = tape.commit_prefix(0, conv);
        }
    }

    let draft_ref = inner.dflash2.as_ref().expect("dflash2");
    let append_taps_keep4_ms = time_stage_ms(3, 8, || {
        let mut tmp_ctx = Inkling2Dflash2Context::new(&draft_ref.config);
        tmp_ctx
            .append_taps(draft_ref, &sample_taps, &sample_ids[..4])
            .expect("append 4");
        tmp_ctx.eval_state().expect("eval");
        synchronize();
    });
    let append_taps_keep8_ms = time_stage_ms(3, 8, || {
        let mut tmp_ctx = Inkling2Dflash2Context::new(&draft_ref.config);
        tmp_ctx
            .append_taps(draft_ref, &sample_taps, &sample_ids[..8])
            .expect("append 8");
        tmp_ctx.eval_state().expect("eval");
        synchronize();
    });

    // 6. Host verify-tree build + Inkling2TreeRow encode latency (100 reps)
    let lattice = {
        let mut view = inner.target_view();
        let (cands, scores) = draft_ctx
            .propose_tree_lattice(draft_ref, &mut view, anchor, block_size)
            .expect("lat");
        draft_ref
            .lattice_from_arrays(&cands, &scores, block_size)
            .expect("readback")
    };
    let policy = Inkling2VerifyTreePolicy::default();
    let lookup = draft_ctx.prepare_lookup(true, MAX_TREE_NODES - 1);
    let reps = 100usize;
    let t_host = Instant::now();
    for _ in 0..reps {
        let tree = policy.build_tree(
            anchor,
            &lattice,
            &lookup,
            MAX_TREE_NODES,
            &draft_ctx.verify_budget,
            &draft_ctx.lookup_acc,
        );
        let _ = verify_tree_to_inkling2_rows(&tree).expect("encode");
    }
    let host_tree_build_us = (t_host.elapsed().as_secs_f64() * 1e6) / reps as f64;

    Dflash2ProposeBreakdown {
        embed_norm_ms,
        target_lm_head_7_ms,
        chain_propose_total_ms,
        tree_lattice_total_ms,
        append_taps_keep4_ms,
        append_taps_keep8_ms,
        host_tree_build_us,
    }
}

fn print_dflash2_propose_table(ctx_len: usize, bd: &Dflash2ProposeBreakdown) {
    let body_sel_ms =
        (bd.chain_propose_total_ms - bd.embed_norm_ms - bd.target_lm_head_7_ms).max(0.0);
    let readback_delta_ms = bd.tree_lattice_total_ms - bd.chain_propose_total_ms;
    let pct = |v: f64| 100.0 * v / bd.chain_propose_total_ms.max(1e-9);
    eprintln!("\n================================================================================");
    eprintln!(
        "  DFLASH2 DRAFT PROPOSE MICRO-BREAKDOWN (5 layers, 4-bit g64, block=8, ctx={ctx_len})"
    );
    eprintln!("================================================================================");
    eprintln!(
        "  1. Target embed_tokens + embed_norm (M=8)     {:>8.3} ms {:>6.1}%",
        bd.embed_norm_ms,
        pct(bd.embed_norm_ms)
    );
    eprintln!(
        "  2. Shared target lm_head (M=7 -> 200058)      {:>8.3} ms {:>6.1}%",
        bd.target_lm_head_7_ms,
        pct(bd.target_lm_head_7_ms)
    );
    eprintln!(
        "  3. 5-layer NAX draft body + top-16 selector   {:>8.3} ms {:>6.1}%",
        body_sel_ms,
        pct(body_sel_ms)
    );
    eprintln!("  ------------------------------------------------------------------");
    eprintln!(
        "  Unsynced 1D Chain propose_greedy_chain        {:>8.3} ms  100.0%",
        bd.chain_propose_total_ms
    );
    eprintln!(
        "  Unsynced Tree propose_tree_lattice + readback {:>8.3} ms  (delta: {:+.3} ms)",
        bd.tree_lattice_total_ms, readback_delta_ms
    );
    eprintln!(
        "  Draft append_taps (fuse+KV ring, keep=4 / 8)  {:>8.3} ms / {:>6.3} ms",
        bd.append_taps_keep4_ms, bd.append_taps_keep8_ms
    );
    eprintln!(
        "  Host build_verify_tree + encode_tree_rows     {:>8.1} us",
        bd.host_tree_build_us
    );
    eprintln!("================================================================================\n");
}

/// Dedicated Splash-comparable real-model benchmark for Inkling-2-Mini +
/// `incoai/Inkling-2-Mini-DFlash2`: compares Plain AR (`M=1`), DFlash2 1D Chain (`M=8`),
/// DFlash2 Verify Tree + SAM (`auto`, `M=8..16`), and DFlash2 Verify Tree + SAM (`all`, `M=16`)
/// across 4 realistic prompts (default `512` output tokens via `MLX_INKLING2_BENCH_OUT_TOKENS`),
/// plus M=1/M=8/M=16 GPU micro-breakdowns and DFlash2 draft breakdown.
#[tokio::test]
#[ignore = "Requires real Inkling-2-Mini + Inkling-2-Mini-DFlash2 on disk (~19 GB); run with --release -- --ignored --nocapture"]
async fn inkling2_dflash2_benchmark() {
    let model_path = test_real_model_path();
    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "Missing Inkling-2-Mini checkpoint at {}",
        model_dir.display()
    );

    let out_tokens: usize = std::env::var("MLX_INKLING2_BENCH_OUT_TOKENS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(512);
    let effort_override: Option<f64> = std::env::var("MLX_INKLING2_BENCH_REASONING_EFFORT")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok());

    let _wired = WiredResidency::new(WiredResidency::working_set_minus_pools);
    let t_load = Instant::now();
    let mut inner = load_from_dir(&model_path).await.expect("load_from_dir");
    assert!(inner.paged.is_some(), "Expected paged runtime enabled");
    assert!(
        inner.dflash2.is_some(),
        "Expected incoai/Inkling-2-Mini-DFlash2 companion auto-discovered alongside {model_path}"
    );
    let kv_dtype = inner.paged.as_ref().expect("paged").coordinator.kv_dtype();
    let draft_mib =
        inner.dflash2.as_ref().expect("dflash2").weight_bytes as f64 / (1024.0 * 1024.0);
    synchronize_and_clear_cache();

    let base_tree_policy = Inkling2VerifyTreePolicy::from_env();
    let chain_policy = Inkling2VerifyTreePolicy {
        mode: Inkling2VerifyTreeMode::Off,
        ..Inkling2VerifyTreePolicy::default()
    };
    let hybrid_policy = Inkling2VerifyTreePolicy {
        mode: Inkling2VerifyTreeMode::Auto,
        fixed_rows: None,
        ..base_tree_policy
    };
    let tree_always_policy = Inkling2VerifyTreePolicy {
        mode: Inkling2VerifyTreeMode::Always,
        fixed_rows: None,
        ..base_tree_policy
    };

    eprintln!("\n================================================================================");
    eprintln!("  INKLING-2-MINI + DFLASH2 FULL BENCHMARK (Splash 1.7.2 Comparable)");
    eprintln!("  Model: {model_path}");
    eprintln!(
        "  Loaded in {:.2}s | KV: {:?} | Target: {:.2} GiB | DFlash2: {:.1} MiB | Out: {} tok | Release: {}",
        t_load.elapsed().as_secs_f64(),
        kv_dtype,
        inner.weight_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
        draft_mib,
        out_tokens,
        !cfg!(debug_assertions),
    );
    eprintln!(
        "  Hybrid/TreeCost: base={:.1}ms slope={:.2}ms/row min_nodes={} max_rows={} sam_trigger={}",
        hybrid_policy.cost_model.base_ms,
        hybrid_policy.cost_model.slope_ms,
        hybrid_policy.cost_model.min_nodes,
        hybrid_policy.max_tree_rows,
        hybrid_policy.sam_trigger_len,
    );
    eprintln!("================================================================================");

    // Splash default reasoning template (`reasoning_effort = 0.9`) for 2K Coding, Dijkstra Coding,
    // and Math / Reasoning; direct code output (`0.0`) on Code-Edit Agent so verbatim prompt-lookup
    // starts immediately from token 1 (all overridable via `MLX_INKLING2_BENCH_REASONING_EFFORT`).
    let prompt_2k_ids = build_2k_coding_prompt(&inner, effort_override);
    let dijkstra_ids = render_prompt_with_effort(&inner, SINGLE_PROMPT, effort_override);
    let code_edit_ids = render_prompt_with_effort(
        &inner,
        CODE_EDIT_AGENT_PROMPT,
        effort_override.or(Some(0.0)),
    );
    let math_ids = render_prompt_with_effort(&inner, MATH_REASONING_PROMPT, effort_override);

    // Warmup GPU pipelines (AR, 1D Chain, Hybrid default, Tree always)
    for _ in 0..WARMUP_RUNS {
        let _ = run_single_ar(&mut inner, &dijkstra_ids, 32);
        let _ = run_single_dflash2(&mut inner, &dijkstra_ids, 32, chain_policy);
        let _ = run_single_dflash2(&mut inner, &dijkstra_ids, 32, hybrid_policy);
        let _ = run_single_dflash2(&mut inner, &dijkstra_ids, 32, tree_always_policy);
    }

    let prompts: [(&str, &[u32], usize); 4] = [
        ("2K Context Coding", &prompt_2k_ids, out_tokens),
        ("Dijkstra Coding", &dijkstra_ids, out_tokens),
        ("Code-Edit Agent (SAM)", &code_edit_ids, out_tokens),
        ("Math / Reasoning", &math_ids, out_tokens),
    ];

    eprintln!(
        "\n  {:<24} {:>5} | {:<20} {:>8} {:>7} {:>7} {:>8} {:>7} {:>6} {:>6} {:>9}",
        "Prompt",
        "ctx",
        "Mode",
        "tok/s",
        "ms/tok",
        "vs AR",
        "ms/cyc",
        "accept",
        "avg_T",
        "sam_ac",
        "exact_ar"
    );
    eprintln!("  {}", "-".repeat(120));

    #[derive(Default, Clone, Copy)]
    struct ModeTotals {
        decoded: usize,
        total_ms: f64,
        cycles: usize,
        tree_cycles: usize,
        tree_rows: usize,
        lookup_acc: usize,
    }
    let mut totals = [ModeTotals::default(); 4];
    let mut ar_sample_tokens = Vec::new();

    for (name, p_ids, out_tok) in prompts {
        clear_cache();
        let ar = run_single_ar(&mut inner, p_ids, out_tok);
        let ar_ms_tok = ar.total_ms / ar.steps.max(1) as f64;
        let ar_tps = 1000.0 / ar_ms_tok;
        totals[0].decoded += ar.steps;
        totals[0].total_ms += ar.total_ms;
        totals[0].cycles += ar.steps;
        totals[0].tree_rows += ar.steps;
        if ar_sample_tokens.len() < 32 && ar.tokens.len() >= 32 {
            ar_sample_tokens = ar.tokens.clone();
        }
        eprintln!(
            "  {:<24} {:>5} | {:<20} {:>8.2} {:>7.3} {:>6.2}x {:>8.3} {:>7.3} {:>6.2} {:>6} {:>4}/{:<4}",
            name,
            p_ids.len(),
            "AR (M=1)",
            ar_tps,
            ar_ms_tok,
            1.0,
            ar_ms_tok,
            1.0,
            1.0,
            "-",
            ar.tokens.len(),
            ar.tokens.len(),
        );

        for (slot, (mode_label, policy)) in [
            ("DFlash2 1D (M=8)", chain_policy),
            ("DFlash2 Hybrid", hybrid_policy),
            ("DFlash2 Tree always", tree_always_policy),
        ]
        .into_iter()
        .enumerate()
        {
            clear_cache();
            let df = run_single_dflash2(&mut inner, p_ids, out_tok, policy);
            let tps = df.decoded_tokens as f64 / (df.total_ms / 1000.0);
            let ms_tok = df.total_ms / df.decoded_tokens.max(1) as f64;
            let ms_cyc = df.total_ms / df.cycles.max(1) as f64;
            let mean_acc = (df.total_accepted_drafts + df.cycles) as f64 / df.cycles.max(1) as f64;
            let avg_t = df.total_tree_rows as f64 / df.cycles.max(1) as f64;
            let exact = df
                .tokens
                .iter()
                .zip(ar.tokens.iter())
                .take_while(|(a, b)| a == b)
                .count();
            let t = &mut totals[slot + 1];
            t.decoded += df.decoded_tokens;
            t.total_ms += df.total_ms;
            t.cycles += df.cycles;
            t.tree_cycles += df.tree_cycles;
            t.tree_rows += df.total_tree_rows;
            t.lookup_acc += df.total_lookup_accepted;
            eprintln!(
                "  {:<24} {:>5} | {:<20} {:>8.2} {:>7.3} {:>6.2}x {:>8.3} {:>7.3} {:>6.2} {:>6} {:>4}/{:<4}  (p={:.2} v={:.2} c={:.2} tc={}/{} nc={})",
                "",
                "",
                mode_label,
                tps,
                ms_tok,
                tps / ar_tps.max(1e-6),
                ms_cyc,
                mean_acc,
                avg_t,
                df.total_lookup_accepted,
                exact,
                ar.tokens.len(),
                df.propose_ms / df.cycles.max(1) as f64,
                df.target_verify_ms / df.cycles.max(1) as f64,
                df.commit_ms / df.cycles.max(1) as f64,
                df.tree_cycles,
                df.cycles,
                df.non_contiguous_commits,
            );
        }
        eprintln!("  {}", "-".repeat(120));
    }

    // Aggregate 4-prompt summary (weighted across all decoded tokens)
    let ar_agg_tps = totals[0].decoded as f64 / (totals[0].total_ms / 1000.0).max(1e-9);
    eprintln!("\n  4-PROMPT AGGREGATE SUMMARY ({out_tokens} out tok/prompt):");
    eprintln!(
        "  {:<20} {:>9} {:>8} {:>7} {:>8} {:>7} {:>7} {:>7} {:>10}",
        "Mode", "tok/s", "ms/tok", "vs AR", "ms/cyc", "accept", "avg_T", "sam_ac", "tree_cyc"
    );
    for (label, t) in [
        ("AR (M=1)", totals[0]),
        ("DFlash2 1D (M=8)", totals[1]),
        ("DFlash2 Hybrid", totals[2]),
        ("DFlash2 Tree always", totals[3]),
    ] {
        let tps = t.decoded as f64 / (t.total_ms / 1000.0).max(1e-9);
        let ms_tok = t.total_ms / t.decoded.max(1) as f64;
        let ms_cyc = t.total_ms / t.cycles.max(1) as f64;
        let accept = t.decoded as f64 / t.cycles.max(1) as f64;
        let avg_t = t.tree_rows as f64 / t.cycles.max(1) as f64;
        eprintln!(
            "  {:<20} {:>9.2} {:>8.3} {:>6.2}x {:>8.3} {:>7.3} {:>7.2} {:>7} {:>4}/{:<4}",
            label,
            tps,
            ms_tok,
            tps / ar_agg_tps.max(1e-6),
            ms_cyc,
            accept,
            avg_t,
            t.lookup_acc,
            t.tree_cycles,
            t.cycles,
        );
    }

    // Per-stage GPU micro-breakdown at M=1, M=8, M=16 (exact ctx=512)
    clear_cache();
    let ctx_512_ids = build_long_prompt(&inner, 512);
    let tok_pool: &[u32] = if ar_sample_tokens.len() >= 32 {
        &ar_sample_tokens
    } else {
        &prompt_2k_ids
    };
    let m1_tok = &tok_pool[16..17];
    let m8_toks = &tok_pool[16..24];
    let m16_toks = &tok_pool[16..32];

    let bd_m1 = profile_48_visits(&mut inner, &ctx_512_ids, m1_tok);
    print_stage_table("Inkling2 AR Decode", 1, ctx_512_ids.len(), &bd_m1);

    clear_cache();
    let bd_m8 = profile_48_visits(&mut inner, &ctx_512_ids, m8_toks);
    print_stage_table("Inkling2 DFlash2 1D Verify", 8, ctx_512_ids.len(), &bd_m8);

    clear_cache();
    let bd_m16 = profile_48_visits(&mut inner, &ctx_512_ids, m16_toks);
    print_stage_table(
        "Inkling2 DFlash2 Tree Verify",
        16,
        ctx_512_ids.len(),
        &bd_m16,
    );

    // DFlash2 draft propose micro-breakdown
    clear_cache();
    let df_bd = profile_dflash2_propose(&mut inner, &ctx_512_ids);
    print_dflash2_propose_table(ctx_512_ids.len(), &df_bd);
}

// ============================================================================
// Fast Non-Ignored Synthetic Unit Test: Branching 12-Row Verify Tree Parity
// ============================================================================

fn det_f32_vec(mut state: u64, len: usize, scale: f32, bias: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let u = ((z >> 40) as f32) / ((1u32 << 24) as f32); // [0, 1)
        out.push((u * 2.0 - 1.0) * scale + bias);
    }
    out
}

fn det_f32(seed: u64, shape: &[i64], scale: f32, bias: f32) -> MxArray {
    let len: usize = shape.iter().map(|&d| d as usize).product();
    let vals = det_f32_vec(seed, len, scale, bias);
    MxArray::from_float32(&vals, shape).expect("det_f32")
}

fn det_bf16(seed: u64, shape: &[i64], scale: f32, bias: f32) -> MxArray {
    det_f32(seed, shape, scale, bias)
        .astype(DType::BFloat16)
        .expect("det_bf16")
}

fn build_tiny_synthetic_inkling2_inner() -> Inkling2Inner {
    let d = 64usize;
    let n_heads = 8usize;
    let n_kv = 4usize;
    let head_dim = 128usize;
    let d_rel = 16usize;
    let rel_extent = 64usize;
    let vocab = 128usize;
    let unpadded_vocab = 120usize;
    let num_experts = 4usize;
    let top_k = 2usize;
    let moe_inter = 64usize;
    let inv_sqrt_d = 1.0_f64 / (d as f64).sqrt();

    let config = Inkling2Config {
        model_type: "inkling2".to_string(),
        hidden_size: d,
        num_hidden_layers: 2,
        num_physical_layers: Some(2),
        num_attention_heads: n_heads,
        num_key_value_heads: n_kv,
        head_dim,
        value_head_dim: None,
        vocab_size: vocab,
        padded_vocab_size: Some(vocab),
        unpadded_vocab_size: Some(unpadded_vocab),
        rms_norm_eps: 1e-6,
        max_position_embeddings: 256,
        sliding_window: 64,
        sliding_window_pattern: vec![true, false],
        local_layer_ids: vec![0],
        sconv_kernel_size: 4,
        n_layers_per_attn_res_block: 2,
        d_rel,
        rel_extent,
        qk_l2_clip_threshold: 16.0,
        rel_l2_clip_threshold: 8.0,
        // Small n_floor with non-zero alpha ensures Global tau(anchor + depth) > 1.0
        // and varies strictly with tree depth!
        log_scaling_n_floor: 4,
        log_scaling_alpha: 0.25,
        num_experts,
        num_experts_per_tok: top_k,
        moe_intermediate_size: Some(moe_inter),
        intermediate_size: Some(moe_inter),
        route_scale: 1.0,
        mlp_gate_softcap: 15.0,
        mlp_up_softcap: 15.0,
        final_logit_softcapping: 30.0,
        loop_config: None,
        vision_config: None,
        dspark_config: None,
        eos_token_id: 2,
        pad_token_id: 0,
        gate_up_interleaved: false,
    };

    let make_lin = |seed: u64, out_f: usize, in_f: usize, scale: f32| -> LinearProj {
        let w = det_bf16(seed, &[out_f as i64, in_f as i64], scale, 0.0);
        LinearProj::Standard(Linear::from_weights(&w, None).expect("Linear::from_weights"))
    };

    let emb_w = det_bf16(100, &[vocab as i64, d as i64], 0.35, 0.0);
    let embed_tokens = Embedding::from_weight(&emb_w).expect("Embedding::from_weight");
    let embed_norm = det_f32(101, &[d as i64], 0.1, 1.0);
    let output_res_query = det_f32(102, &[1, d as i64], 0.2, 0.0)
        .mul_scalar(inv_sqrt_d)
        .expect("out_res");
    let output_res_query_t = output_res_query.transpose(None).expect("out_res_t");
    let output_norm = det_f32(103, &[d as i64], 0.1, 1.0);
    let lm_w = det_bf16(104, &[vocab as i64, d as i64], 0.15, 0.0);
    let lm_head = LinearProj::Standard(Linear::from_weights(&lm_w, None).expect("lm_head"));
    let lm_head_bf16_unpadded = Some(
        lm_w.slice_axis(0, 0, unpadded_vocab as i64)
            .expect("lm_unpadded"),
    );

    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv * head_dim;
    let r_dim = n_heads * d_rel;

    let mut layers = Vec::with_capacity(2);
    for (phys, is_local) in [(0usize, true), (1usize, false)] {
        let s = (phys as u64 + 1) * 1000;
        let attn_res_query = if phys == 0 {
            None
        } else {
            let q_rows = if is_local { 1 } else { 3 };
            Some(
                det_f32(s + 1, &[q_rows, d as i64], 0.2, 0.0)
                    .mul_scalar(inv_sqrt_d)
                    .expect("attn_res"),
            )
        };
        let attn_res_query_t = attn_res_query
            .as_ref()
            .map(|q| q.transpose(None).expect("attn_res_t"));
        let ffn_res_query = det_f32(s + 2, &[1, d as i64], 0.2, 0.0)
            .mul_scalar(inv_sqrt_d)
            .expect("ffn_res");
        let ffn_res_query_t = ffn_res_query.transpose(None).expect("ffn_res_t");

        let attn_inp_norm = if is_local {
            AttnInputNorm::Local(det_f32(s + 3, &[d as i64], 0.1, 1.0))
        } else {
            let q_norm = det_f32(s + 3, &[d as i64], 0.1, 1.0);
            let k_norm = det_f32(s + 4, &[d as i64], 0.1, 1.0);
            let v_norm = det_f32(s + 5, &[d as i64], 0.1, 1.0);
            let qkv_norm = MxArray::stack(vec![&q_norm, &k_norm, &v_norm], Some(0)).expect("qkv");
            AttnInputNorm::Global {
                q_norm,
                k_norm,
                v_norm,
                qkv_norm,
            }
        };
        let ffn_norm = det_f32(s + 6, &[d as i64], 0.1, 1.0);

        // Non-zero taps across ALL 4 ShortConv slots so any history corruption is immediately visible.
        let k_conv = det_f32(s + 10, &[kv_dim as i64, 4], 0.3, 0.25);
        let v_conv = det_f32(s + 11, &[kv_dim as i64, 4], 0.3, 0.25);
        let post_conv = det_f32(s + 12, &[d as i64, 4], 0.3, 0.25);
        let (k_conv_proj, v_conv_proj) = if !is_local {
            (
                Some(make_lin(s + 13, n_kv * 4, d, 0.12)),
                Some(make_lin(s + 14, n_kv * 4, d, 0.12)),
            )
        } else {
            (None, None)
        };

        let rel_bias_w = det_f32(s + 15, &[rel_extent as i64, d_rel as i64], 0.2, 0.0);
        let rel_bias_t = rel_bias_w.transpose(None).expect("rb_t");

        let mut self_attn = Inkling2Attention {
            num_heads: n_heads,
            num_kv_heads: n_kv,
            head_dim,
            d_rel,
            rel_extent,
            sliding_window: config.sliding_window,
            is_local,
            qk_l2_clip_threshold: config.qk_l2_clip_threshold,
            rel_l2_clip_threshold: config.rel_l2_clip_threshold,
            log_scaling_n_floor: config.log_scaling_n_floor,
            log_scaling_alpha: config.log_scaling_alpha,
            rms_norm_eps: config.rms_norm_eps as f32,
            q_proj: make_lin(s + 20, q_dim, d, 0.06),
            k_proj: make_lin(s + 21, kv_dim, d, 0.06),
            v_proj: make_lin(s + 22, kv_dim, d, 0.06),
            rel_proj: make_lin(s + 23, r_dim, d, 0.08),
            gate_proj: make_lin(s + 24, q_dim, d, 0.06),
            sink_proj: make_lin(s + 25, n_heads, d, 0.08),
            o_proj: make_lin(s + 26, d, q_dim, 0.04),
            fused_qrg: None,
            fused_qrgkv: None,
            fused_qrg_has_sink: false,
            fused_k_with_conv: None,
            fused_v_with_conv: None,
            k_conv,
            v_conv,
            post_conv,
            k_conv_proj,
            v_conv_proj,
            q_clip: det_f32(s + 27, &[1, n_heads as i64, head_dim as i64], 0.1, 1.0),
            rel_clip: det_f32(s + 28, &[1, n_heads as i64, d_rel as i64], 0.1, 1.0),
            rel_bias_t,
            rel_bias_w,
            qk_thresh_scalar: MxArray::scalar_float(config.qk_l2_clip_threshold).expect("qk"),
            rel_thresh_scalar: MxArray::scalar_float(config.rel_l2_clip_threshold).expect("rel"),
            one_scalar: MxArray::scalar_float(1.0).expect("one"),
        };
        self_attn.try_fuse_projections().expect("fuse proj");

        let router_weight = det_f32(s + 30, &[num_experts as i64, d as i64], 0.15, 0.0);
        let router_bias = Some(det_f32(s + 31, &[num_experts as i64], 0.05, 0.0));
        let gate_up_w = det_bf16(
            s + 32,
            &[num_experts as i64, (2 * moe_inter) as i64, d as i64],
            0.06,
            0.0,
        );
        let down_w = det_bf16(
            s + 33,
            &[num_experts as i64, d as i64, moe_inter as i64],
            0.06,
            0.0,
        );
        let mlp_post_conv = det_f32(s + 34, &[d as i64, 4], 0.3, 0.25);
        let moe = Inkling2MoE::new(
            num_experts,
            top_k,
            moe_inter,
            config.route_scale,
            config.mlp_gate_softcap,
            config.mlp_up_softcap,
            config.gate_up_interleaved,
            router_weight,
            router_bias,
            ExpertSwitchProj::Dense(gate_up_w),
            ExpertSwitchProj::Dense(down_w),
            mlp_post_conv,
        )
        .expect("Inkling2MoE::new");

        layers.push(Inkling2PhysicalLayer {
            phys_idx: phys,
            is_local,
            rms_norm_eps: config.rms_norm_eps as f32,
            attn_res_query,
            attn_res_query_t,
            ffn_res_query,
            ffn_res_query_t,
            attn_inp_norm,
            ffn_norm,
            self_attn,
            moe,
            loop_ctrl: None,
        });
    }

    let mut inner = Inkling2Inner {
        config,
        embed_tokens,
        embed_norm,
        layers,
        output_res_query,
        output_res_query_t,
        output_norm,
        lm_head,
        lm_head_bf16_unpadded,
        vision_encoder: None,
        tokenizer: None,
        dspark: None,
        dflash2: None,
        dflash2_identity: None,
        dspark_turn_contexts: HashMap::new(),
        cuda_dflash2_ctx: None,
        cuda_verify_budget: crate::models::qwen3_5::verify_tree::VerifyBudget::new(),
        cuda_lookup_acceptance: crate::models::qwen3_5::verify_tree::LookupAcceptance::new(),
        cuda_dspark_ctx: None,
        scheduled_dspark: Inkling2ScheduledDsparkStates::default(),
        paged: None,
        flat_cache: None,
        cuda_cache: None,
        cuda_kv_format: crate::models::inkling2::cuda_kv::Inkling2CudaKvFormat::Bf16,
        cached_token_history: Vec::new(),
        cached_image_key: None,
        turn_cancel: None,
        model_id: next_inkling2_model_id(),
        memory_plan: None,
        memory_registration: None,
        wired: None,
        cache_recency: Arc::new(CacheRecency::new()),
        decode_share: default_decode_share(),
        max_context: None,
        activations: TransformerActivations::Auto,
        disk_options: DiskTierOptions::default(),
        prefill_runs: HashMap::new(),
    };
    inner
        .enable_paged(Inkling2PagedConfig {
            kv_format: KvFormat::Int8,
            block_size: 32,
            max_model_len: 128,
            max_chunk: 64,
            max_concurrent_sequences: 2,
            gpu_memory_mb: None,
        })
        .expect("enable_paged INT8-KTM");
    inner
}

/// Fast non-ignored unit test verifying FULL end-to-end branching verify-tree correctness on a
/// tiny 2-layer synthetic Inkling2 model (1 SWA + 1 Global, `head_dim=128`, `block_size=32`,
/// INT8-KTM):
/// - Prefills 29 prompt tokens so the 12-row tree crosses the `block_size=32` physical block
///   boundary (`anchor_pos = 29`, tree spans slots `29..41`, accepted path lands at `29..33`).
/// - Verifies a branching 12-row tree where the accepted path is strictly non-contiguous
///   (`[0, 3, 7, 10]`), checking tree Attention ancestor masking, tree relative bias, Global
///   tree `tau(anchor + depth)`, and all 4 tree ShortConvs (SWA static + Global dynamic).
/// - Compacts KV + recorded tokens + commits ShortConv tape along `[0, 3, 7, 10]`, then runs
///   4 subsequent sequential AR decode steps and asserts logit parity within `1e-4` against
///   plain sequential AR over the accepted path!
#[test]
fn inkling2_verify_tree_branching_matches_sequential_ar() {
    if !crate::engine::persistence::compiled_forward_backend_available() {
        return;
    }

    let mut inner = build_tiny_synthetic_inkling2_inner();
    let prompt: Vec<u32> = (1..=29).collect();

    // Branching 12-row tree in topological order (`parent < row`); accepted path = `[0, 3, 7, 10]`.
    // Every decoy branch uses distinct tokens so any mask, rel_bias, tau, ShortConv, or KV
    // compaction bug leaks directly into logits!
    let tree_tokens: Vec<u32> = vec![31, 41, 42, 51, 61, 62, 63, 52, 71, 72, 53, 81];
    let parents: [i32; 12] = [-1, 0, 1, 0, 3, 4, 3, 3, 7, 8, 7, 10];
    let mut tree_rows = Vec::with_capacity(12);
    let mut depths = Vec::with_capacity(12);
    let mut masks = Vec::with_capacity(12);
    for (i, &p) in parents.iter().enumerate() {
        let (parent_u16, depth_u16, mask_u32) = if i == 0 {
            (u16::MAX, 0u16, 1u32)
        } else {
            let pu = p as usize;
            (p as u16, depths[pu] + 1, masks[pu] | (1u32 << i))
        };
        depths.push(depth_u16);
        masks.push(mask_u32);
        tree_rows.push(Inkling2TreeRow {
            ancestors: mask_u32,
            parent: parent_u16,
            depth: depth_u16,
        });
    }
    let accepted_rows: [usize; 4] = [0, 3, 7, 10];
    let accepted_tokens: Vec<u32> = accepted_rows.iter().map(|&r| tree_tokens[r]).collect();
    let post_tokens: [u32; 4] = [91, 92, 93, 94];

    // 1. Reference sequential AR run on seq 1: prefill prompt, then 1 token at a time over
    //    `accepted_tokens` (`[31, 51, 52, 53]`) followed by `post_tokens` (`[91, 92, 93, 94]`).
    let ref_seq = 1u32;
    inner.reset_paged_sequence(ref_seq).expect("reset ref seq");
    let _ = inner
        .paged_prefill_sequence(ref_seq, &prompt, None)
        .expect("ref prefill");
    let mut ref_logits: Vec<Vec<f32>> = Vec::with_capacity(8);
    for &tok in accepted_tokens.iter().chain(post_tokens.iter()) {
        let logits_2d = inner
            .paged_decode_step(ref_seq, tok)
            .expect("ref decode step");
        let _ = inner.settle_paged_step(ref_seq).expect("ref settle");
        eval_sync(&[&logits_2d]);
        ref_logits.push(logits_2d.to_float32().expect("ref f32").to_vec());
    }

    // 2. Branching tree verify + non-contiguous path commit + 4 post-commit AR steps on seq 0.
    let tree_seq = 0u32;
    inner
        .reset_paged_sequence(tree_seq)
        .expect("reset tree seq");
    let _ = inner
        .paged_prefill_sequence(tree_seq, &prompt, None)
        .expect("tree prefill");

    let (anchor_pos, ticket) = {
        let rt = inner.paged.as_mut().expect("paged");
        assert!(
            rt.coordinator
                .reserve_lookahead(tree_seq, tree_tokens.len())
                .expect("reserve")
        );
        let pos = rt
            .coordinator
            .request_token_count_all(tree_seq)
            .expect("pos");
        assert_eq!(pos, 29);
        let ticket = rt
            .coordinator
            .record_verify(tree_seq, &tree_tokens)
            .expect("record_verify");
        (pos, ticket)
    };
    let encoded_tree = encode_inkling2_tree_rows(&tree_rows).expect("encode_inkling2_tree_rows");
    let row = PagedRaggedRow {
        seq_id: tree_seq,
        first_logical_position: anchor_pos,
        query_len: tree_tokens.len() as u32,
    };
    let (tree_logits, taps, mut tapes) = inner
        .paged_verify_ragged_taped_tree(
            &[(row, tree_tokens.clone())],
            Some(Inkling2VerifyTreeInput {
                tree_rows: &encoded_tree,
                depths: &depths,
            }),
            &[0, 1, 2],
            false,
        )
        .expect("paged_verify_ragged_taped_tree");
    assert_eq!(
        taps.len(),
        3,
        "Expected pre-layer 0, pre-layer 1, and post-final-norm taps"
    );
    eval_sync(&[&tree_logits]);
    let tree_logits_f32 = tree_logits.to_float32().expect("tree logits f32");
    let vocab_cols = tree_logits.shape_at(1).expect("vocab cols") as usize;
    assert_eq!(vocab_cols, 120);

    // 2a. Verify tree logits at every accepted node `[0, 3, 7, 10]` match sequential AR!
    let mut max_tree_diff = 0.0f32;
    for (step_idx, &tree_row) in accepted_rows.iter().enumerate() {
        let start = tree_row * vocab_cols;
        let cand = &tree_logits_f32[start..start + vocab_cols];
        let reference = &ref_logits[step_idx];
        let step_max = cand
            .iter()
            .zip(reference.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        max_tree_diff = max_tree_diff.max(step_max);
        assert!(
            step_max <= 1e-4,
            "Tree verify row {tree_row} (step {step_idx}) diverged from sequential AR: max_abs_diff={step_max}"
        );
    }

    // 2b. Compact KV + recorded tokens + commit cycle + commit ShortConv tape along `[0, 3, 7, 10]`!
    let tape = tapes.pop().expect("ShortConvTape");
    {
        let rt = inner.paged.as_mut().expect("paged");
        rt.coordinator
            .compact_kv_path_all(tree_seq, anchor_pos, &accepted_rows)
            .expect("compact_kv_path_all");
        rt.coordinator
            .compact_recorded_tree_tokens_all(tree_seq, tree_tokens.len(), &accepted_rows)
            .expect("compact_recorded_tree_tokens_all");
        rt.coordinator
            .commit_cycle(tree_seq, ticket, accepted_rows.len())
            .expect("commit_cycle");
        let conv = rt.conv_states.get_mut(&tree_seq).expect("conv state");
        tape.commit_path(&accepted_rows, conv)
            .expect("tape.commit_path");
        let committed = rt
            .coordinator
            .request_token_count_all(tree_seq)
            .expect("committed count");
        assert_eq!(
            committed, 33,
            "Expected 29 prompt + 4 accepted tree tokens = 33 (crossed block_size=32 boundary)"
        );
        rt.coordinator
            .settle_committed(tree_seq, u64::from(committed))
            .expect("settle_committed");
        rt.record_block_boundary_conv_from_tape_path(tree_seq, committed, &accepted_rows, &tape)
            .expect("record_block_boundary_conv_from_tape_path");
    }

    // 2c. Decode 4 sequential AR tokens AFTER non-contiguous tree commit & cross-block compaction!
    let mut max_post_diff = 0.0f32;
    for (post_idx, &tok) in post_tokens.iter().enumerate() {
        let logits_2d = inner
            .paged_decode_step(tree_seq, tok)
            .expect("post-commit decode step");
        let _ = inner.settle_paged_step(tree_seq).expect("post settle");
        eval_sync(&[&logits_2d]);
        let cand = logits_2d.to_float32().expect("post f32");
        let reference = &ref_logits[4 + post_idx];
        let step_max = cand
            .iter()
            .zip(reference.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        max_post_diff = max_post_diff.max(step_max);
        assert!(
            step_max <= 1e-4,
            "Post-commit AR step {post_idx} (tok={tok}) diverged from sequential AR: max_abs_diff={step_max}"
        );
    }

    eprintln!(
        "[inkling2_verify_tree_branching_matches_sequential_ar] PASS: max_tree_diff={max_tree_diff:.3e}, max_post_diff={max_post_diff:.3e}"
    );
}

#[test]
fn inkling2_verify_tree_mode_parsing_and_adaptive_routing() {
    // 1. Env parsing covers auto / always / off spellings cleanly.
    for raw in [
        None,
        Some(""),
        Some("  "),
        Some("auto"),
        Some("AUTO"),
        Some("1"),
        Some("true"),
        Some("on"),
        Some("hybrid"),
    ] {
        assert_eq!(
            Inkling2VerifyTreeMode::parse(raw),
            Inkling2VerifyTreeMode::Auto
        );
    }
    for raw in [
        Some("always"),
        Some("ALWAYS"),
        Some("force"),
        Some("all"),
        Some("  all  "),
    ] {
        assert_eq!(
            Inkling2VerifyTreeMode::parse(raw),
            Inkling2VerifyTreeMode::Always
        );
    }
    for raw in [
        Some("0"),
        Some("false"),
        Some("FALSE"),
        Some("off"),
        Some("  off  "),
    ] {
        assert_eq!(
            Inkling2VerifyTreeMode::parse(raw),
            Inkling2VerifyTreeMode::Off
        );
    }

    // 2. Adaptive routing keeps zero-sync 1D unless SAM finds >= sam_trigger_len (4) with remaining >= 4.
    let auto_policy = Inkling2VerifyTreePolicy::default();
    let always_policy = Inkling2VerifyTreePolicy {
        mode: Inkling2VerifyTreeMode::Always,
        ..auto_policy
    };
    let off_policy = Inkling2VerifyTreePolicy {
        mode: Inkling2VerifyTreeMode::Off,
        ..auto_policy
    };
    assert!(auto_policy.tree_path_active());
    assert!(always_policy.tree_path_active());
    assert!(!off_policy.tree_path_active());

    let mut lookup = LookupIndex::new();
    lookup.append(&[10, 20, 30, 40, 50, 60, 70, 80]);
    let peek = |l: &LookupIndex| -> usize {
        let info = l.match_info();
        if info.match_len >= LookupIndex::MIN_MATCH && !l.proposal_slice(15).is_empty() {
            info.match_len
        } else {
            0
        }
    };
    // No repeated suffix -> Auto stays on zero-sync 1D, Always uses tree, Off stays on 1D.
    assert_eq!(peek(&lookup), 0);
    assert!(!auto_policy.should_use_tree_cycle(peek(&lookup), 15));
    assert!(always_policy.should_use_tree_cycle(peek(&lookup), 15));
    assert!(!off_policy.should_use_tree_cycle(peek(&lookup), 15));

    // 3-token match (`< 4` default trigger) -> Auto still stays on zero-sync 1D.
    lookup.append(&[10, 20, 30]);
    assert_eq!(peek(&lookup), 3);
    assert!(!auto_policy.should_use_tree_cycle(peek(&lookup), 15));

    // 4-token match with continuation `[50, 60, 70, 80]` -> Auto triggers Verify Tree!
    lookup.append(&[40]);
    assert_eq!(peek(&lookup), 4);
    assert!(auto_policy.should_use_tree_cycle(peek(&lookup), 15));
    // Tail of turn (`remaining < 4`, i.e. `max_depth < 3`) stays on 1D even with SAM match.
    assert!(!auto_policy.should_use_tree_cycle(peek(&lookup), 2));
    // Disabling SAM lookup keeps Auto on zero-sync 1D.
    let no_lookup = Inkling2VerifyTreePolicy {
        lookup_enabled: false,
        ..auto_policy
    };
    assert!(!no_lookup.should_use_tree_cycle(peek(&lookup), 15));
}
