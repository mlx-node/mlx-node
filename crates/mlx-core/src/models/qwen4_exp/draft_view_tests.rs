use super::*;
use crate::array::DType;
use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
use crate::models::quantized_linear::LinearProj;
use crate::models::qwen3_5::dflash2::{
    DFLASH2_DRAFT_PROPOSALS, DFlash2ContextCache, DFlash2Model, DFlash2TargetView, SelectorPath,
    tiny_dflash2_model_with_block,
};
use crate::models::qwen3_5::dflash2_state::plan::StatePlan;
use crate::models::qwen3_5::dflash2_turn::DraftTapSink;
use crate::models::qwen3_5::draft_context::DraftContextWindow;
use crate::models::qwen3_5::paged_forward::PrefillTapSink;
use crate::nn::{Embedding, Linear};

/// The draft's taps on the tiny fixture: the PLE layer 1 and the last layer.
const TAPS: [usize; 2] = [1, 3];

/// A one-layer draft of the full block over the tiny fixture's geometry
/// (hidden 32, vocab 64, 4 layers), tapping [`TAPS`], 8 sliding rows.
pub(super) fn tiny_draft() -> DFlash2Model {
    let target = crate::models::qwen3_5::config::Qwen3_5Config {
        rope_scaling: None,
        qwen35_gguf_gdn_layout: None,
        kv_format: None,
        vocab_size: 64,
        hidden_size: 32,
        num_layers: 4,
        num_heads: 4,
        num_kv_heads: 2,
        intermediate_size: 64,
        rms_norm_eps: 1e-6,
        head_dim: 8,
        tie_word_embeddings: false,
        attention_bias: false,
        max_position_embeddings: 128,
        pad_token_id: 0,
        eos_token_id: 2,
        bos_token_id: 1,
        linear_num_value_heads: 2,
        linear_num_key_heads: 2,
        linear_key_head_dim: 8,
        linear_value_head_dim: 8,
        linear_conv_kernel_dim: 4,
        full_attention_interval: 4,
        partial_rotary_factor: 0.25,
        rope_theta: 10_000.0,
        paged_cache_memory_mb: None,
        paged_block_size: None,
        n_mtp_layers: 0,
    };
    unsafe { mlx_sys::mlx_seed(0xD2B0_F1A5) };
    let draft = tiny_dflash2_model_with_block(&target, DFLASH2_DRAFT_PROPOSALS).unwrap();
    assert_eq!(draft.config.target_layers, TAPS);
    assert_eq!(draft.config.sliding_window, 8);
    draft
}

/// The tiny fixture as a Qwen3.5 target reads it: its token table as an
/// `Embedding`, its head as a `LinearProj`.
fn dense_target(decoder: &mut decoder::Decoder) -> (Embedding, LinearProj) {
    let table = decoder.weights.dense("embed_tokens.weight").unwrap();
    let mut embedding = Embedding::new(64, 32).unwrap();
    embedding.load_weight(&table).unwrap();
    let head = decoder.weights.dense("lm_head.weight").unwrap();
    let head = LinearProj::Standard(Linear::from_weights(&head, None).unwrap());
    (embedding, head)
}

fn ids(path: SelectorPath) -> Vec<u32> {
    match path {
        SelectorPath::Device(ids) => ids.to_uint32().unwrap().to_vec(),
        SelectorPath::Host(ids) => ids.into_iter().map(|id| id as u32).collect(),
        SelectorPath::Sampled { ids, .. } => ids.to_uint32().unwrap().to_vec(),
    }
}

/// Both contexts hold the same rows: length, tokens and the bits of every
/// K/V buffer.
fn assert_same_context(a: &DFlash2ContextCache, b: &DFlash2ContextCache, what: &str) {
    assert_eq!(a.logical_len(), b.logical_len(), "{what}: length");
    assert_eq!(a.token_history(), b.token_history(), "{what}: tokens");
    let (mut x, mut y) = (Vec::new(), Vec::new());
    a.collect_arrays(&mut x);
    b.collect_arrays(&mut y);
    assert_eq!(x.len(), y.len(), "{what}: buffers");
    assert!(!x.is_empty(), "{what}: no buffer");
    for (i, (x, y)) in x.iter().zip(&y).enumerate() {
        assert_same_bits(x, y, &format!("{what}: buffer {i}"));
    }
}

/// The tiny fixture with 8-row prefill windows: a 21-row prompt runs 3
/// passes and the draft's 7 ring rows start inside the second.
fn tiny() -> Inner {
    let mut inner = tiny_inner();
    inner.decoder.prefill_window_pin = Some(8);
    inner.decoder.refresh_prefill_window();
    inner
}

fn prompt() -> Vec<u32> {
    (0..21).map(|i| 3 + (i * 7 % 50) as u32).collect()
}

fn transcript(len: usize) -> Vec<u32> {
    (0..len).map(|i| 4 + (i * 11 % 53) as u32).collect()
}

/// The Qwen3.5 path's reference over the same target: the draft's context
/// primed by `DraftTapSink` from the prefill's taps, then per cycle the
/// dense-table proposal, the verify, and its kept rows' taps appended
/// (`Qwen35DFlash2Stepper::append_tapped`). Returns each cycle's proposals
/// and the context.
fn qwen35_reference(
    inner: &mut Inner,
    draft: &DFlash2Model,
    prompt: &[u32],
    transcript: &[u32],
    cycles: usize,
) -> (Vec<Vec<u32>>, DFlash2ContextCache) {
    let (embedding, head) = dense_target(&mut inner.decoder);
    let decoder = &mut inner.decoder;
    decoder.set_hidden_taps(Some(&TAPS[..])).unwrap();
    decoder.prefill_prompt(prompt).unwrap();
    let passes = decoder.take_hidden_taps();
    assert_eq!(passes.len(), prompt.len().div_ceil(8), "prefill passes");
    let mut sink = DraftTapSink::new(
        draft,
        DFlash2ContextCache::new(&draft.config),
        DraftContextWindow::new(64, 1).unwrap(),
        prompt,
        0,
        &StatePlan::default(),
        &[],
    )
    .unwrap();
    for pass in &passes {
        let end = pass.first + pass.rows().unwrap();
        sink.chunk(pass.first, &prompt[pass.first..end], &pass.hidden)
            .unwrap();
    }
    sink.eval().unwrap();
    let (mut context, _) = sink.into_parts();

    let mut proposals = Vec::new();
    let mut at = 0;
    for _ in 0..cycles {
        let position = decoder.history.len();
        let anchor = transcript[at];
        let (path, _) = draft
            .propose(
                &embedding,
                Some(&head),
                &mut context,
                anchor,
                DFLASH2_DRAFT_PROPOSALS,
                0.0,
                true,
                None,
                &mut rand::rng(),
            )
            .unwrap();
        let drafted = ids(path);
        let truth = &transcript[at + 1..];
        let matched = drafted
            .iter()
            .zip(truth)
            .take_while(|(a, b)| a == b)
            .count();
        let verified = 3.min(truth.len());
        let mut tokens = vec![anchor];
        tokens.extend_from_slice(&drafted[..verified]);
        let (_, states) = decoder.verify_chunk(&tokens).unwrap();
        let kept = matched.min(verified) + 1;
        let record = decoder.take_hidden_taps().pop().unwrap();
        assert_eq!(record.first, position);
        let rows: Vec<MxArray> = record
            .hidden
            .iter()
            .map(|tap| tap.slice_axis(1, 0, kept as i64).unwrap())
            .collect();
        decoder
            .paged
            .as_mut()
            .unwrap()
            .rollback_last_tokens((tokens.len() - kept) as u32)
            .unwrap();
        decoder.restore_state(states.into_iter().nth(kept - 1).unwrap());
        let fused = draft.fuse_context(&rows).unwrap();
        context
            .append(draft, &fused, position as i32, &tokens[..kept])
            .unwrap();
        context.eval().unwrap();
        proposals.push(drafted);
        at += kept;
    }
    (proposals, context)
}

/// The draft's context over `prompt`, prefilled from an empty history with
/// the draft's taps set.
fn draft_prime(
    decoder: &mut decoder::Decoder,
    draft: &DFlash2Model,
    prompt: &[u32],
) -> DFlash2ContextCache {
    assert!(decoder.history.is_empty() && !prompt.is_empty());
    decoder.draft_taps_on(draft).unwrap();
    decoder.prefill_prompt(prompt).unwrap();
    let mut context = DFlash2ContextCache::new(&draft.config);
    decoder.draft_follow(draft, &mut context).unwrap();
    context
}

/// Up to `cycles` teacher-forced cycles over `transcript` (`transcript[0]`
/// is the first anchor), each from where the last committed: the draft's
/// whole greedy block, the verify of the anchor and the first
/// `verify_proposals`, and a commit of the anchor and the proposals that
/// equal the transcript, at most the verified ones. Returns each cycle's
/// proposals and the proposals it verified.
fn draft_cycles(
    decoder: &mut decoder::Decoder,
    draft: &DFlash2Model,
    context: &mut DFlash2ContextCache,
    transcript: &[u32],
    cycles: usize,
    verify_proposals: usize,
) -> Vec<(Vec<u32>, usize)> {
    use crate::engine::backend::DsparkStepper;
    let params = crate::engine::params::extract_chat_params(&crate::engine::types::ChatConfig {
        temperature: Some(0.0),
        ..Default::default()
    });
    let mut stepper =
        decoder::DraftStepper::new(decoder, draft, context, verify_proposals + 1).unwrap();
    let mut out = Vec::new();
    let mut at = 0;
    while out.len() < cycles && at + 1 < transcript.len() {
        let anchor = transcript[at];
        let proposals = stepper
            .propose_block(anchor, &params, &mut rand::rng(), false)
            .unwrap()
            .ids;
        let truth = &transcript[at + 1..];
        let matched = proposals
            .iter()
            .zip(truth)
            .take_while(|(got, want)| got == want)
            .count();
        let verified = verify_proposals.min(proposals.len()).min(truth.len());
        let mut tokens = vec![anchor];
        tokens.extend_from_slice(&proposals[..verified]);
        stepper.verify(&tokens).unwrap();
        let kept = matched.min(verified) + 1;
        stepper.commit(kept, tokens.len()).unwrap();
        out.push((proposals, verified));
        at += kept;
    }
    out
}

/// The Flash Next target view hands the draft what the Qwen3.5 path hands
/// it for the same weights: the token rows, the head's logits, the
/// prefill's context rows and each cycle's kept rows, so the proposals,
/// the draft's context and the target's state are the same bits.
#[test]
fn the_target_view_gives_the_draft_the_qwen35_inputs() {
    let draft = tiny_draft();
    let mut view_side = tiny();
    let mut dense_side = tiny();
    for inner in [&mut view_side, &mut dense_side] {
        inner.activate_paged_seq(1).unwrap();
    }
    let (embedding, head) = dense_target(&mut dense_side.decoder);
    assert_eq!(
        view_side.decoder.draft_target_shape(),
        crate::models::qwen3_5::dflash2::DFlash2TargetShape::new(32, 64, 4)
    );
    draft
        .validate_target(view_side.decoder.draft_target_shape())
        .unwrap();

    for anchor in [5u32, 0, 63] {
        let mut block = vec![anchor as i32];
        block.resize(
            DFLASH2_DRAFT_PROPOSALS + 1,
            draft.config.mask_token_id as i32,
        );
        let array = MxArray::from_int32(&block, &[1, block.len() as i64]).unwrap();
        let got = view_side
            .decoder
            .draft_view()
            .embed(&array, &block)
            .unwrap();
        assert_eq!(got.dtype().unwrap(), DType::BFloat16);
        assert_same_bits(
            &got,
            &embedding.forward(&array).unwrap(),
            &format!("anchor {anchor}: token rows"),
        );
    }
    unsafe { mlx_sys::mlx_seed(0xD2B1) };
    let hidden = MxArray::random_normal(&[1, 8, 32], 0.0, 1.0, Some(DType::BFloat16)).unwrap();
    let got = view_side.decoder.draft_view().logits(&hidden).unwrap();
    assert_same_bits(&got, &head.forward_logits(&hidden).unwrap(), "head logits");

    let prompt = prompt();
    let transcript = transcript(40);
    let cycles = 6;
    view_side
        .decoder
        .set_hidden_taps(Some(&draft.config.target_layers))
        .unwrap();
    let mut context = draft_prime(&mut view_side.decoder, &draft, &prompt);
    let (want_proposals, want_context) =
        qwen35_reference(&mut dense_side, &draft, &prompt, &transcript, 0);
    assert_same_context(&context, &want_context, "primed");
    assert_eq!(want_proposals.len(), 0);
    let primed_len = context.logical_len();
    assert_eq!(primed_len, prompt.len() as i32);

    let got = draft_cycles(
        &mut view_side.decoder,
        &draft,
        &mut context,
        &transcript,
        cycles,
        3,
    );
    let mut fresh = tiny();
    fresh.activate_paged_seq(1).unwrap();
    let (want_proposals, want_context) =
        qwen35_reference(&mut fresh, &draft, &prompt, &transcript, cycles);
    assert_eq!(got.len(), cycles);
    for (i, ((proposals, verified), want)) in got.iter().zip(&want_proposals).enumerate() {
        assert_eq!(proposals, want, "cycle {i}: proposals");
        assert_eq!(proposals.len(), DFLASH2_DRAFT_PROPOSALS);
        assert_eq!(*verified, 3, "cycle {i}");
    }
    assert_same_context(&context, &want_context, "after the cycles");
    route_step::assert_same_decoder_state(
        &mut view_side.decoder,
        &mut fresh.decoder,
        "after the cycles",
    );
}
