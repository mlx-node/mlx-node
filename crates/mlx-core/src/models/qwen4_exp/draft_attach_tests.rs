//! The draft as a load attaches it: a draft checkpoint that serves the
//! target attaches, one of another width fails the load, and a load with no
//! draft reports none. Through the scheduler, a sampled drafted turn keeps
//! the target's distribution with the engine's accept rule, a proposal
//! error fails the turn, and a drafted turn reports its cycles.
use super::draft_cycle::{chi_square_p, moe_draft, tilted_row};
use super::draft_switch::{ROWS, chat_model, forced, full_length};
use super::draft_view::tiny_draft;
use super::lanes::direct_inner;
use super::*;
use crate::engine::backend::ChatBackend;
use crate::engine::cmd::ChatCmd;
use crate::engine::hybrid_scheduler::{HybridSchedulerBackend, HybridSchedulerState};
use crate::engine::model_command::ModelCommand;
use crate::engine::params::extract_chat_params;
use crate::engine::types::{ChatConfig, ChatResult};
use crate::model_thread::LoopControl;
use crate::models::qwen3_5::dflash2::{DFlash2Model, write_test_checkpoint};
use crate::sampling::SparseDistribution;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::cell::RefCell;
use std::rc::Rc;

/// A `config.json` of a one-layer DFlash2 draft `hidden` wide over a target
/// of `vocab` tokens and `layers` layers, tapping its second and last layer.
fn draft_json(hidden: usize, kv_heads: usize, vocab: usize, layers: usize) -> serde_json::Value {
    serde_json::json!({
        "architectures": ["DFlash2DraftModel"],
        "hidden_size": hidden,
        "intermediate_size": 256,
        "num_hidden_layers": 1,
        "num_target_layers": layers,
        "num_attention_heads": hidden / 64,
        "num_key_value_heads": kv_heads,
        "head_dim": 64,
        "vocab_size": vocab,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 4096,
        "sliding_window": 16,
        "layer_types": ["sliding_attention"],
        "is_causal": false,
        "rope_theta": 10_000.0,
        "dflash_config": {
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": vocab - 1,
            "selector_rank": 8,
            "selector_top_k": 4,
            "target_layer_ids": [1, layers - 1],
        },
    })
}

/// On the Flash Next-width target a draft checkpoint of the target's width
/// attaches (loaded, priced, offered as the turn's speculation); one 256
/// values wider fails, before and after its weights could load, with the
/// reason, and so does its price.
/// MUTATION: the attach of a draft with another hidden size is accepted →
/// red (the wider draft).
#[test]
fn a_draft_of_the_targets_width_attaches_and_a_wider_one_fails_the_load() {
    let _host = memory::scoped_fixture_host();
    let mut inner = direct_inner();
    let c = inner.decoder.config.clone();
    let temp = Temp::new();
    let wide = temp.0.join("wide");
    let wider = c.hidden_size + 256;
    write_test_checkpoint(
        &wide,
        &draft_json(wider, 4, c.vocab_size, c.num_hidden_layers),
    )
    .unwrap();
    let refused = |error: Option<Error>, what: &str| {
        let reason = error.map(|error| error.reason).unwrap_or_default();
        assert!(
            reason.contains("does not serve this model") && reason.contains("mismatch"),
            "{what}: the wider draft fails with the reason: {reason:?}"
        );
    };
    refused(decoder::load_draft(&wide, &c).err(), "the load");
    refused(
        load_plan::DraftPrice::of_checkpoint("Qwen4Exp", &c, &wide).err(),
        "the estimate",
    );
    let fits = temp.0.join("fits");
    write_test_checkpoint(
        &fits,
        &draft_json(c.hidden_size, 8, c.vocab_size, c.num_hidden_layers),
    )
    .unwrap();
    let draft = decoder::load_draft(&fits, &c).unwrap();
    let price = load_plan::DraftPrice::of(&c, &draft.config, draft.weight_bytes);
    let estimated = load_plan::DraftPrice::of_checkpoint("Qwen4Exp", &c, &fits).unwrap();
    eprintln!(
        "attach: weights {} B loaded, {} B file; lane {} B",
        draft.weight_bytes,
        estimated.weights,
        price.lane()
    );
    assert!(draft.weight_bytes > 0 && price.lane() > 0);
    assert_eq!(estimated.lane(), price.lane());
    assert_eq!(inner.draft_bytes_held(), 0, "no draft: nothing held");
    inner.set_draft(Some(draft)).unwrap();
    assert!(matches!(
        inner.execution_plan().speculative.map(|plan| plan.kind),
        Some(crate::engine::plan::SpeculativeKind::DraftModel)
    ));
    assert_eq!(
        inner.draft_bytes_held(),
        0,
        "an empty context holds nothing"
    );
    let row = memory::row_state_bytes(&c);
    assert_eq!(
        inner.scheduled_state_candidate_bytes(7, 64, true),
        row + decoder::DRAFT_VERIFY_ROWS as u64 * row,
        "a drafted turn's state: its row and one snapshot per verify row"
    );
}

/// A production load of the direct fixture with no draft reports none; the
/// same load with a draft of another width fails, naming the draft and the
/// mismatch.
/// MUTATION: a load with no draft reports a draft → red (the report).
/// MUTATION: the attach of a draft with another hidden size is accepted →
/// red (the refusal).
#[test]
fn a_load_with_no_draft_reports_none_and_a_misfit_draft_fails_it() {
    const CHILD: &str = "MLX_QWEN4_DRAFT_CHILD";
    const MODEL: &str = "MLX_QWEN4_DRAFT_MODEL";
    const DRAFT: &str = "MLX_QWEN4_DRAFT_DIR";
    const REPORT: &str = "qwen4 draft report: ";
    const REFUSED: &str = "qwen4 draft refusal: ";
    if std::env::var_os(CHILD).is_some() {
        let model_path = std::env::var(MODEL).unwrap();
        let draft = std::env::var(DRAFT).unwrap();
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async move {
                let plain = Qwen4ExpModel::load(model_path.clone(), None).await.unwrap();
                println!(
                    "{REPORT}draft {} head {} speculation {}",
                    plain.has_draft,
                    plain.has_mtp,
                    plain.has_mtp_weights()
                );
                plain.thread.shutdown_and_join().unwrap();
                let options = Qwen4ExpLoadOptions {
                    draft_model_path: Some(draft),
                    ..Default::default()
                };
                match Qwen4ExpModel::load(model_path, Some(options)).await {
                    Ok(model) => {
                        model.thread.shutdown_and_join().unwrap();
                        println!("{REFUSED}accepted");
                    }
                    Err(error) => println!("{REFUSED}{}", error.reason),
                }
            });
        return;
    }
    if !crate::utils::gguf_gpu::gpu_repack_available() {
        return;
    }
    let temp = Temp::new();
    let model = temp.0.join("direct.gguf");
    let c = write_direct_gguf_fixture(&model);
    let draft = temp.0.join("draft");
    std::fs::create_dir_all(&draft).unwrap();
    let wider = (c.hidden_size / 64 + 1) * 64;
    std::fs::write(
        draft.join("config.json"),
        draft_json(wider, 1, c.vocab_size, c.num_hidden_layers).to_string(),
    )
    .unwrap();
    let home = temp.0.join("fresh-research-home");
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "models::qwen4_exp::tests::draft_attach::a_load_with_no_draft_reports_none_and_a_misfit_draft_fails_it",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .env(MODEL, &model)
        .env(DRAFT, &draft)
        .env(memory::FIXTURE_HOST_ENV, "1")
        .env("HOME", &home)
        .env("MLX_NATIVE_GGUF_CACHE_DIR", home.join("native-gguf"))
        .env_remove("MLX_QWEN4_RESIDENCY")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "child failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let line = |prefix: &str| {
        stdout
            .lines()
            .find_map(|line| line.split_once(prefix).map(|(_, rest)| rest))
            .unwrap_or_default()
            .to_string()
    };
    let report = line(REPORT);
    eprintln!("{REPORT}{report}");
    assert!(
        report.starts_with("draft false head false speculation false"),
        "a load with no draft reports none: {report:?}"
    );
    let refusal = line(REFUSED);
    eprintln!("{REFUSED}{refusal}");
    assert!(
        refusal.contains("the DFlash2 draft at")
            && refusal.contains("does not serve this model")
            && refusal.contains("mismatch"),
        "a misfit draft fails the load: {refusal:?}"
    );
}

/// The whole-store fixture with its banks filled, serving chat with the
/// word tokenizer.
pub(super) fn direct_chat() -> Inner {
    chat_model(direct_inner())
}

/// A draft that serves [`direct_chat`].
pub(super) fn direct_draft(inner: &Inner) -> Arc<DFlash2Model> {
    let c = &inner.decoder.config;
    Arc::new(moe_draft(c.hidden_size, c.vocab_size, c.num_hidden_layers))
}

fn words(ids: &[u32]) -> String {
    ids.iter()
        .map(|id| format!("t{id}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The ids of a turn's word text (`t12 t5`).
fn ids_of(text: &str) -> Vec<u32> {
    text.split_whitespace()
        .map(|word| word.strip_prefix('t').unwrap().parse().unwrap())
        .collect()
}

type Reply = tokio::sync::oneshot::Receiver<Result<ChatResult>>;

/// A scheduler over one model.
pub(super) struct Driver {
    pub(super) state: HybridSchedulerState<Inner>,
    sender: tokio::sync::mpsc::UnboundedSender<ModelCommand>,
    receiver: tokio::sync::mpsc::UnboundedReceiver<ModelCommand>,
}

impl Driver {
    pub(super) fn new(inner: Inner) -> Self {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        Self {
            state: HybridSchedulerState::new(inner).unwrap(),
            sender,
            receiver,
        }
    }

    fn drive(&mut self, mut rx: Reply) -> Result<ChatResult> {
        for _ in 0..4096 {
            if let Ok(result) = rx.try_recv() {
                return result;
            }
            assert!(matches!(
                self.state.drive(&mut self.receiver),
                LoopControl::Continue
            ));
        }
        panic!("the turn did not finish");
    }

    /// One turn of `owner` on `prompt` under `config`, then its release.
    pub(super) fn turn(
        &mut self,
        owner: &str,
        prompt: &[u32],
        config: ChatConfig,
    ) -> Result<ChatResult> {
        let message: crate::tokenizer::ChatMessage =
            serde_json::from_value(serde_json::json!({"role": "user", "content": words(prompt)}))
                .unwrap();
        let (reply, rx) = tokio::sync::oneshot::channel();
        let command = ChatCmd::SessionStart {
            messages: vec![message],
            config: ChatConfig {
                cache_owner_id: Some(owner.to_string()),
                ..config
            },
            reply,
            cancelled: Arc::new(AtomicBool::new(false)),
        };
        assert!(self.sender.send(ModelCommand::from_chat(command)).is_ok());
        let result = self.drive(rx);
        let (reply, mut rx) = tokio::sync::oneshot::channel();
        let release = ChatCmd::ReleaseCacheOwner {
            owner_id: owner.to_string(),
            reply,
        };
        assert!(self.sender.send(ModelCommand::from_chat(release)).is_ok());
        for _ in 0..64 {
            if rx.try_recv().is_ok() {
                return result;
            }
            assert!(matches!(
                self.state.drive(&mut self.receiver),
                LoopControl::Continue
            ));
        }
        panic!("the owner was not released");
    }
}

/// A turn that asks for the draft's cycle at 4 rows: `max_new` tokens at
/// `temperature` (top 6 when sampled), its performance reported.
fn drafted_config(max_new: i32, temperature: f64) -> ChatConfig {
    ChatConfig {
        max_new_tokens: Some(max_new),
        temperature: Some(temperature),
        top_k: (temperature > 0.0).then_some(6),
        enable_mtp: Some(true),
        mtp_depth: Some(ROWS as i32 - 1),
        mtp_adaptive_depth: Some(false),
        report_performance: Some(true),
        ..Default::default()
    }
}

/// A tiny direct-ready chat model (`hidden = 32`, 4 experts, `plan.direct == true`)
/// opened from a GGUF written by [`write_direct_gguf_fixture`].
fn tiny_direct_chat_at(path: &Path, config: &config::Config) -> Inner {
    let store = weights::Store::open_fixture(path, None).unwrap();
    let vision = media::Vision::metadata(&serde_json::Value::Null, &store, config).unwrap();
    let mut decoder = decoder::Decoder::new(config.clone(), store).unwrap();
    decoder.paged = Some(paged::create_fixture(config).unwrap());
    go_live(&mut decoder, 0);
    assert!(
        decoder.weights.plan.direct,
        "the fixture host admits every bank"
    );
    let probe = MxArray::from_uint32(&[0], &[1]).unwrap();
    for layer in 0..decoder.config.num_hidden_layers {
        decoder
            .weights
            .device_expert_slots(layer, &probe)
            .unwrap()
            .expect("a filled direct layer");
    }
    chat_model(inner_over(decoder, vision, false))
}

/// The target's sampling distribution from a single logit row.
fn target_dist(logits: &MxArray, params: &crate::engine::params::ChatParams) -> Vec<f64> {
    let row = logits.reshape(&[-1]).unwrap();
    f32s(&crate::sampling::sampling_distribution(&row, params.sampling_config).unwrap())
        .iter()
        .map(|&v| f64::from(v))
        .collect()
}

/// The target's sampling distribution after `tokens` stepped from a prefilled
/// snapshot `base`, rolling back `inner` to `base` afterward.
fn target_after_base(
    inner: &mut Inner,
    base: &decoder::DecoderState,
    tokens: &[u32],
    params: &crate::engine::params::ChatParams,
) -> Vec<f64> {
    let mut logits = None;
    for &token in tokens {
        logits = Some(inner.decoder.step(token).unwrap());
    }
    let dist = target_dist(&logits.unwrap(), params);
    inner
        .decoder
        .paged
        .as_mut()
        .unwrap()
        .rollback_last_tokens(tokens.len() as u32)
        .unwrap();
    inner.decoder.restore_state(base.clone());
    dist
}

fn tv(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>() / 2.0
}

fn ranked(p: &[f64]) -> Vec<usize> {
    let mut ranked: Vec<usize> = (0..p.len()).collect();
    ranked.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
    ranked
}

/// 400 sampled turns (temperature 1, top 4, draft width 2) through the
/// scheduler on a tiny direct store (`hidden = 32`, `plan.direct == true`),
/// each drafted with proposals the test draws from tilted rows over the
/// target's likely tokens, and accepted by the engine's rule: the turn's first
/// two tokens match the target's joint distribution by exact Pearson
/// chi-square (`p >= 1e-6`), and so do its second and third (a stop counts as
/// a token), while the second proposal is accepted in some cycles.
/// MUTATION: a refused proposal's token is drawn from the target, not from
/// the residual → red (a token pair follows the target).
/// MUTATION: a proposal is kept with no accept draw → red (a token pair
/// follows the target).
#[test]
fn a_sampled_drafted_turn_keeps_the_target_distribution() {
    let _host = memory::scoped_fixture_host();
    const TRIALS: usize = 400;
    let temp = Temp::new();
    let gguf_path = temp.0.join("direct.gguf");
    let gguf_config = write_direct_gguf_fixture(&gguf_path);
    let prompt = super::prefix_reuse::prompt(8, 3);
    let mut config = drafted_config(4, 1.0);
    config.top_k = Some(4);
    config.mtp_depth = Some(2);
    let params = extract_chat_params(&config);
    let mut plain = tiny_direct_chat_at(&gguf_path, &gguf_config);
    let vocab = plain.decoder.config.vocab_size;
    let stop = vocab;
    let eos: Vec<usize> = {
        let mut ids = plain.extra_eos_ids();
        ids.push(plain.session_eos_id(&plain.tokenizer().unwrap()).unwrap());
        ids.into_iter().map(|id| id as usize).collect()
    };
    plain.activate_paged_seq(1).unwrap();
    let first_logits = plain.decoder.prefill_prompt(&prompt).unwrap();
    let base = plain.decoder.snapshot();
    let first = target_dist(&first_logits, &params);
    let second: Vec<Option<Vec<f64>>> = (0..vocab)
        .map(|x| {
            (first[x] > 0.0).then(|| target_after_base(&mut plain, &base, &[x as u32], &params))
        })
        .collect();
    let mut third = std::collections::HashMap::new();
    for (x, row) in second.iter().enumerate() {
        let Some(row) = row else { continue };
        if eos.contains(&x) {
            continue;
        }
        for (y, &py) in row.iter().enumerate() {
            if py > 0.0 && !eos.contains(&y) {
                third.insert(
                    (x, y),
                    target_after_base(&mut plain, &base, &[x as u32, y as u32], &params),
                );
            }
        }
    }
    let cells = (vocab + 1) * (vocab + 1);
    let cell = |a: usize, b: usize| a * (vocab + 1) + b;
    let as_token = |t: usize| if eos.contains(&t) { stop } else { t };
    let mut joint_a = vec![0.0; cells];
    let mut joint_b = vec![0.0; cells];
    let mut on = 0.0;
    for (x, &px) in first.iter().enumerate() {
        if px == 0.0 {
            continue;
        }
        if eos.contains(&x) {
            joint_a[cell(stop, stop)] += px;
            continue;
        }
        on += px;
        let row = second[x].as_ref().unwrap();
        for (y, &py) in row.iter().enumerate() {
            joint_a[cell(x, as_token(y))] += px * py;
            if eos.contains(&y) {
                joint_b[cell(stop, stop)] += px * py;
            } else if let Some(next) = third.get(&(x, y)) {
                for (z, &pz) in next.iter().enumerate() {
                    joint_b[cell(y, as_token(z))] += px * py * pz;
                }
            }
        }
    }
    joint_b.iter_mut().for_each(|p| *p /= on);
    let tables = Rc::new((second.clone(), third.clone()));
    let rows_for = move |anchor: u32| -> Vec<(Vec<i32>, Vec<f64>)> {
        let (second, third) = &*tables;
        let p = second
            .get(anchor as usize)
            .and_then(Option::as_ref)
            .cloned()
            .unwrap_or_else(|| vec![1.0 / vocab as f64; vocab]);
        let mut marginal = vec![0.0; vocab];
        for (y, &py) in p.iter().enumerate() {
            if let Some(next) = third.get(&(anchor as usize, y)) {
                for (z, &pz) in next.iter().enumerate() {
                    marginal[z] += py * pz;
                }
            }
        }
        if marginal.iter().all(|&m| m == 0.0) {
            marginal = p.clone();
        }
        let (by_first, by_second) = (ranked(&p), ranked(&marginal));
        let row = |ids: &[usize], probs: &[f64]| -> (Vec<i32>, Vec<f64>) {
            (
                ids.iter().map(|&t| t as i32).collect(),
                ids.iter().map(|&t| probs[t].sqrt().max(1e-3)).collect(),
            )
        };
        vec![
            tilted_row(&by_first, &p),
            row(&by_second[..4], &marginal),
            (
                by_second[8..12].iter().map(|&t| t as i32).collect(),
                vec![0.25; 4],
            ),
        ]
    };
    let mut driver = Driver::new(tiny_direct_chat_at(&gguf_path, &gguf_config));
    let draft = Arc::new(tiny_draft());
    driver.state.inner.set_draft(Some(draft)).unwrap();
    // Exercise both (1) a real non-injected sampled turn (`drawn = None`) and
    // (2) a full-width accept-2 + reject-3rd + post-commit continuation turn on
    // the Flash Next MoE shape (`direct_chat`, `hidden = 2560`, `mtp_depth = 3`).
    {
        let mut moe_driver = Driver::new(direct_chat());
        let moe_draft = direct_draft(&moe_driver.state.inner);
        let moe_cfg = &moe_driver.state.inner.decoder.config;
        let moe_vocab = moe_cfg.vocab_size;
        assert_eq!(moe_cfg.hidden_size, 2560);
        assert_eq!(moe_cfg.moe_intermediate_size, 640);
        assert_eq!(moe_cfg.num_experts_per_tok, 10);
        assert!(moe_driver.state.inner.decoder.weights.plan.direct);
        moe_driver
            .state
            .inner
            .set_draft(Some(Arc::clone(&moe_draft)))
            .unwrap();
        let moe_config = drafted_config(8, 1.0); // top_k = 6, mtp_depth = ROWS - 1 (= 3)
        let moe_params = extract_chat_params(&moe_config);
        let mut moe_oracle = direct_chat();
        let mut moe_eos = moe_oracle.extra_eos_ids();
        moe_eos.push(
            moe_oracle
                .session_eos_id(&moe_oracle.tokenizer().unwrap())
                .unwrap(),
        );
        moe_oracle.activate_paged_seq(1).unwrap();
        let moe_first_logits = moe_oracle.decoder.prefill_prompt(&prompt).unwrap();
        let moe_base = moe_oracle.decoder.snapshot();
        let moe_first = target_dist(&moe_first_logits, &moe_params);
        unsafe { mlx_sys::mlx_seed(0x5EED_F4E2) };
        struct LaneSeed;
        impl Drop for LaneSeed {
            fn drop(&mut self) {
                crate::engine::hybrid_scheduler::TEST_LANE_SEED.with(|seed| seed.set(None));
            }
        }
        let _lane_seed = LaneSeed;
        crate::engine::hybrid_scheduler::TEST_LANE_SEED.with(|seed| seed.set(Some(0x5EED_F4E2)));

        // (1) Real non-injected `draft.propose_with` at depth 3 (4-row verify).
        let mut verified_moe_real = false;
        for attempt in 0..8 {
            let before_offered = moe_driver.state.inner.draft_slot().unwrap().offered.len();
            let result = moe_driver
                .turn(&format!("moe-real-{attempt}"), &prompt, moe_config.clone())
                .unwrap();
            let real_offered: Vec<usize> = moe_driver.state.inner.draft_slot().unwrap().offered
                [before_offered..]
                .iter()
                .map(|&(_, _, n)| n)
                .collect();
            if real_offered.contains(&(ROWS - 1)) {
                let committed = ids_of(&result.text);
                assert!(!committed.is_empty());
                assert!(moe_first[committed[0] as usize] > 0.0);
                for i in 1..committed.len() {
                    let dist =
                        target_after_base(&mut moe_oracle, &moe_base, &committed[..i], &moe_params);
                    assert!(
                        dist[committed[i] as usize] > 0.0,
                        "Flash Next MoE real sampled token {} at step {i} outside top-k support",
                        committed[i]
                    );
                }
                verified_moe_real = true;
                break;
            }
        }
        assert!(
            verified_moe_real,
            "Flash Next MoE real sampled turn must verify {} proposals",
            ROWS - 1
        );

        // (2) One-shot accept-2 + reject-3rd + residual-only boundary draw +
        // recovery + real second proposal at hidden = 2560, depth = 3:
        // cycle 0 runs the real `propose_with` then injects proposals 0 & 1
        // with p/q = 2.0 (always accepted) and proposal 2 with q2 covering all
        // top-k tokens of p3 except `only_survivor` (so p3[d2] = 0 guarantees
        // rejection while `(p3 - q2)+` concentrates 100% of its mass on
        // `only_survivor`, distinguishing residual sampling from raw target
        // sampling); cycle 1 runs the REAL `draft.propose_with` on the
        // recovered draft context, whose history and live KV rows are checked
        // against a sequential oracle before cycle 1's proposal.
        let oracle_cell = Rc::new(RefCell::new(moe_oracle));
        let recorded = Rc::new(RefCell::new(None::<(u32, u32, u32, u32, u32)>));
        let mut verified_residual_trials = 0usize;
        let mut verified_moe_recovery = false;
        let accept_config = drafted_config(8, 1.0);
        for attempt in 0..12 {
            *recorded.borrow_mut() = None;
            let oracle_ref = Rc::clone(&oracle_cell);
            let rec_ref = Rc::clone(&recorded);
            let base_clone = moe_base.clone();
            let eos_clone = moe_eos.clone();
            let params_clone = moe_params.clone();
            {
                let slot = moe_driver.state.inner.draft_slot().unwrap();
                slot.capture_context = !verified_moe_recovery;
                slot.captured_contexts.clear();
                slot.drawn_once = Some(Box::new(move |anchor| {
                    let mut oracle = oracle_ref.borrow_mut();
                    let pick_in = |p: &[f64]| {
                        ranked(p)
                            .into_iter()
                            .find(|&t| p[t] > 0.0 && !eos_clone.contains(&(t as u32)))
                            .unwrap() as u32
                    };
                    let pick_out = |p: &[f64]| {
                        (0..moe_vocab)
                            .find(|&t| p[t] == 0.0 && !eos_clone.contains(&(t as u32)))
                            .unwrap() as u32
                    };
                    let always_accept = |tok: u32, p: &[f64]| {
                        let other = pick_out(p);
                        let w = (p[tok as usize] * 0.5).max(1e-6);
                        SparseDistribution::from_parts(
                            vec![tok as i32, other as i32],
                            vec![w, 1.0 - w],
                            moe_vocab,
                        )
                        .unwrap()
                    };
                    let p1 = target_after_base(&mut oracle, &base_clone, &[anchor], &params_clone);
                    let d0 = pick_in(&p1);
                    let p2 =
                        target_after_base(&mut oracle, &base_clone, &[anchor, d0], &params_clone);
                    let d1 = pick_in(&p2);
                    let p3 = target_after_base(
                        &mut oracle,
                        &base_clone,
                        &[anchor, d0, d1],
                        &params_clone,
                    );
                    let d2 = pick_out(&p3);
                    // Suppress every positive token in `p3` except the
                    // lowest-probability non-EOS token `only_survivor`, so
                    // `(p3 - q2)+` has support `{only_survivor}` while `p3`
                    // puts most of its mass on the suppressed tokens.
                    let pos_p3: Vec<usize> =
                        ranked(&p3).into_iter().filter(|&t| p3[t] > 0.0).collect();
                    let only_survivor = *pos_p3
                        .iter()
                        .rev()
                        .find(|&&t| !eos_clone.contains(&(t as u32)))
                        .unwrap();
                    let suppressed: Vec<usize> =
                        pos_p3.into_iter().filter(|&t| t != only_survivor).collect();
                    assert!(
                        !suppressed.is_empty(),
                        "top_k = 6 target distribution must have > 1 positive token"
                    );
                    let margin = p3[only_survivor];
                    let extra = (0.5 * margin) / suppressed.len() as f64;
                    let mut q2_ids: Vec<i32> = suppressed.iter().map(|&t| t as i32).collect();
                    let mut q2_probs: Vec<f64> =
                        suppressed.iter().map(|&t| p3[t] + extra).collect();
                    q2_ids.push(d2 as i32);
                    q2_probs.push(0.5 * margin);
                    let q0 = always_accept(d0, &p1);
                    let q1 = always_accept(d1, &p2);
                    let q2 = SparseDistribution::from_parts(q2_ids, q2_probs, moe_vocab).unwrap();
                    *rec_ref.borrow_mut() = Some((anchor, d0, d1, d2, only_survivor as u32));
                    (vec![d0, d1, d2], vec![q0, q1, q2])
                }));
            }
            let before_offered = moe_driver.state.inner.draft_slot().unwrap().offered.len();
            let result = moe_driver
                .turn(
                    &format!("moe-accept-{attempt}"),
                    &prompt,
                    accept_config.clone(),
                )
                .unwrap();
            let Some((anchor, d0, d1, d2, only_survivor)) = recorded.borrow_mut().take() else {
                // Only valid skip: anchor itself was EOS before cycle 0 drafted.
                assert_eq!(result.finish_reason, "stop");
                continue;
            };
            let (offered, captured) = {
                let slot = moe_driver.state.inner.draft_slot().unwrap();
                slot.capture_context = false;
                (
                    slot.offered[before_offered..]
                        .iter()
                        .map(|&(_, _, n)| n)
                        .collect::<Vec<_>>(),
                    std::mem::take(&mut slot.captured_contexts),
                )
            };
            let committed = ids_of(&result.text);
            // Assert immediately on every injected cycle (never skip early
            // stops, which could hide an EOS draw from the raw target).
            assert_eq!(offered[0], ROWS - 1, "cycle 0 must verify 3 proposals");
            assert!(
                committed.len() >= 4,
                "injected cycle with non-EOS d0, d1, and only_survivor must commit >= 4 tokens, got {committed:?}"
            );
            assert_eq!(committed[0], anchor);
            assert_eq!(committed[1], d0, "first proposal must be accepted");
            assert_eq!(committed[2], d1, "second proposal must be accepted");
            assert_ne!(committed[3], d2, "third proposal must be rejected");
            assert_eq!(
                committed[3], only_survivor,
                "Flash Next MoE boundary token must come from residual (p3 - q2)+, not target p3"
            );
            assert!(
                offered.len() >= 2 && offered[1] == ROWS - 1,
                "cycle 1 must run a real 3-proposal draft after rejection, got offered={offered:?}"
            );
            verified_residual_trials += 1;

            if !verified_moe_recovery && committed.len() >= 5 {
                // Compare the recovered draft context right before cycle 1's real
                // proposal against a fresh sequential oracle stepped on `prompt` +
                // `[anchor, d0, d1]`.
                let mut seq_oracle = direct_chat();
                seq_oracle.activate_paged_seq(1).unwrap();
                seq_oracle.decoder.draft_taps_on(&moe_draft).unwrap();
                seq_oracle.decoder.prefill_prompt(&prompt).unwrap();
                let mut want_ctx =
                    crate::models::qwen3_5::dflash2::DFlash2ContextCache::new(&moe_draft.config);
                seq_oracle
                    .decoder
                    .draft_follow(&moe_draft, &mut want_ctx)
                    .unwrap();
                for &tok in &committed[..3] {
                    seq_oracle.decoder.step(tok).unwrap();
                    seq_oracle
                        .decoder
                        .draft_follow(&moe_draft, &mut want_ctx)
                        .unwrap();
                }
                let got_ctx = &captured[1];
                assert_eq!(
                    got_ctx.logical_len(),
                    want_ctx.logical_len(),
                    "recovered Flash Next draft context length before cycle 1"
                );
                assert_eq!(
                    got_ctx.token_history(),
                    want_ctx.token_history(),
                    "recovered Flash Next draft context token history before cycle 1"
                );
                let got_views = got_ctx.window_views();
                let want_views = want_ctx.window_views();
                assert_eq!(got_views.len(), want_views.len());
                assert!(!got_views.is_empty());
                for (layer, (g, w)) in got_views.iter().zip(&want_views).enumerate() {
                    let (gk, gv) = g.as_ref().expect("recovered draft KV layer populated");
                    let (wk, wv) = w.as_ref().expect("oracle draft KV layer populated");
                    for (name, ga, wa) in [("K", gk, wk), ("V", gv, wv)] {
                        let g_vals = f32s(ga);
                        let w_vals = f32s(wa);
                        assert_eq!(g_vals.len(), w_vals.len(), "draft layer {layer} {name} len");
                        for (elem_idx, (&a, &b)) in g_vals.iter().zip(&w_vals).enumerate() {
                            assert!(
                                a.is_finite() && b.is_finite(),
                                "non-finite recovered Flash Next draft layer {layer} {name}[{elem_idx}]: got={a}, want={b}"
                            );
                        }
                        let (hist, max_ulp) = bf16_ulp_histogram(&g_vals, &w_vals);
                        assert!(
                            max_ulp <= 1,
                            "recovered Flash Next draft layer {layer} {name} drifted from sequential oracle: {max_ulp} bf16 ULP, histogram {hist:?}"
                        );
                    }
                }
                for i in 4..committed.len() {
                    let dist = target_after_base(
                        &mut oracle_cell.borrow_mut(),
                        &moe_base,
                        &committed[..i],
                        &moe_params,
                    );
                    assert!(
                        dist[committed[i] as usize] > 0.0,
                        "post-rejection token {} at step {i} outside sequential target support",
                        committed[i]
                    );
                }
                let perf = result.performance.as_ref().unwrap();
                let cycles = perf.mtp_cycles.unwrap_or(0) as usize;
                let accepted =
                    (perf.mtp_mean_accepted_tokens.unwrap_or(0.0) * cycles as f64).round() as usize;
                assert!(
                    accepted >= 2,
                    "Flash Next MoE turn must record >= 2 accepted proposals, got {accepted}"
                );
                verified_moe_recovery = true;
            }
            if verified_residual_trials >= 4 && verified_moe_recovery {
                break;
            }
        }
        assert!(
            verified_residual_trials >= 4 && verified_moe_recovery,
            "Flash Next MoE residual+recovery check did not complete (trials={verified_residual_trials}, recovery={verified_moe_recovery})"
        );
    }
    let draws = Rc::new(RefCell::new(StdRng::seed_from_u64(0x5EED_D0E5)));
    {
        let draws = Rc::clone(&draws);
        let slot = driver.state.inner.draft_slot().unwrap();
        slot.drawn = Some(Box::new(move |anchor| {
            let mut ids = Vec::new();
            let mut dists = Vec::new();
            for (support, probs) in rows_for(anchor) {
                let total: f64 = probs.iter().sum();
                let mut u = rand::RngExt::random::<f64>(&mut *draws.borrow_mut()) * total;
                let mut pick = support[support.len() - 1];
                for (&t, &w) in support.iter().zip(&probs) {
                    if u < w {
                        pick = t;
                        break;
                    }
                    u -= w;
                }
                ids.push(pick as u32);
                dists.push(SparseDistribution::from_parts(support, probs, vocab).unwrap());
            }
            (ids, dists)
        }));
    }
    let mut pairs_a = vec![0u64; cells];
    let mut pairs_b = vec![0u64; cells];
    let mut n_b = 0usize;
    let mut one_cycle = 0usize;
    for trial in 0..TRIALS {
        let before = driver.state.inner.draft_slot().unwrap().offered.len();
        let result = driver
            .turn(&format!("s{trial}"), &prompt, config.clone())
            .unwrap();
        let cycles = driver.state.inner.draft_slot().unwrap().offered.len() - before;
        let perf_cycles = result
            .performance
            .as_ref()
            .and_then(|p| p.mtp_cycles)
            .unwrap_or(0) as usize;
        let perf_accepted = result
            .performance
            .as_ref()
            .and_then(|p| p.mtp_mean_accepted_tokens)
            .map_or(0, |mean| (mean * perf_cycles as f64).round() as usize);
        let mut seq: Vec<usize> = Vec::new();
        for id in ids_of(&result.text) {
            let t = as_token(id as usize);
            seq.push(t);
            if t == stop {
                break;
            }
        }
        if result.finish_reason == "stop" && seq.last() != Some(&stop) {
            seq.push(stop);
        }
        assert!(
            seq.len() >= 2 || seq.first() == Some(&stop),
            "trial {trial}: {seq:?} ({})",
            result.finish_reason
        );
        if seq[0] == stop {
            pairs_a[cell(stop, stop)] += 1;
            continue;
        }
        pairs_a[cell(seq[0], seq[1])] += 1;
        let b = if seq[1] == stop {
            cell(stop, stop)
        } else {
            cell(seq[1], seq[2])
        };
        pairs_b[b] += 1;
        n_b += 1;
        one_cycle +=
            usize::from(cycles == 1 && perf_accepted >= 2 && seq.len() >= 4 && seq[3] != stop);
    }
    let freq = |pairs: &[u64], n: usize| -> Vec<f64> {
        pairs.iter().map(|&c| c as f64 / n.max(1) as f64).collect()
    };
    let (tv_a, tv_b) = (
        tv(&freq(&pairs_a, TRIALS), &joint_a),
        tv(&freq(&pairs_b, n_b), &joint_b),
    );
    let (stat_a, dof_a, p_a) = chi_square_p(&pairs_a, &joint_a);
    let (stat_b, dof_b, p_b) = chi_square_p(&pairs_b, &joint_b);
    let second_accepted = one_cycle as f64 / TRIALS as f64;
    eprintln!(
        "sampled drafted turns: {TRIALS}; first two chi2 {stat_a:.2} dof {dof_a} p {p_a:.3e} tv {tv_a:.4} over {} cells; second and third chi2 {stat_b:.2} dof {dof_b} p {p_b:.3e} tv {tv_b:.4} over {} cells ({n_b} turns); second proposal accepted {second_accepted:.4}",
        joint_a.iter().filter(|&&p| p > 0.0).count(),
        joint_b.iter().filter(|&&p| p > 0.0).count(),
    );
    assert!(
        second_accepted > 0.05,
        "the second proposal was accepted in some turns: {second_accepted}"
    );
    assert!(
        !p_a.is_nan() && p_a >= 1e-6,
        "the first two tokens follow the target: chi2 {stat_a:.2} dof {dof_a} p {p_a:.3e}"
    );
    assert!(
        !p_b.is_nan() && p_b >= 1e-6,
        "the second and third follow the target: chi2 {stat_b:.2} dof {dof_b} p {p_b:.3e}"
    );
}

/// A proposal that fails ends the drafted turn with an error that names the
/// draft and says the turn fails; the taps go off, and the next turn of
/// another owner gives the plain model's text.
#[test]
fn a_proposal_error_fails_the_turn() {
    let _host = memory::scoped_fixture_host();
    let model = direct_chat;
    let (prompt, plain) = full_length(model, 21, 12, 5);
    let (second, plain_second) = full_length(model, 23, 12, 9);
    let mut driver = Driver::new(model());
    let draft = direct_draft(&driver.state.inner);
    driver.state.inner.set_draft(Some(draft)).unwrap();
    let ks: Rc<RefCell<Vec<usize>>> = Rc::default();
    let vocab = driver.state.inner.decoder.config.vocab_size as u32;
    let mut full = plain.tokens.clone();
    full.push(0);
    {
        let slot = driver.state.inner.draft_slot().unwrap();
        slot.forced = Some(forced(full, vocab, Rc::clone(&ks)));
        slot.fail_at = Some(1);
    }
    let error = driver
        .turn("a", &prompt, drafted_config(12, 0.0))
        .err()
        .map(|error| error.reason)
        .unwrap_or_default();
    eprintln!("proposal error: {error}");
    assert!(
        error.contains("qwen4_exp DFlash2 draft: the proposal failed, so the turn fails"),
        "the turn fails with the draft's error: {error:?}"
    );
    assert_eq!(driver.state.inner.decoder.hidden_tap_layers(), None);
    driver.state.inner.draft_slot().unwrap().fail_at = None;
    let result = driver.turn("b", &second, drafted_config(12, 0.0)).unwrap();
    assert_eq!(result.text, plain_second.text, "the next turn");
}

/// A greedy drafted turn reports its cycles: the cycles that proposed, the
/// proposals they made, the ones they kept, so the trace reads rows per
/// verify and tokens per cycle.
#[test]
fn a_drafted_turn_reports_its_cycles() {
    let _host = memory::scoped_fixture_host();
    let model = direct_chat;
    let (prompt, plain) = full_length(model, 21, 16, 7);
    let mut driver = Driver::new(model());
    let draft = direct_draft(&driver.state.inner);
    driver.state.inner.set_draft(Some(draft)).unwrap();
    let ks: Rc<RefCell<Vec<usize>>> = Rc::default();
    let vocab = driver.state.inner.decoder.config.vocab_size as u32;
    let mut full = plain.tokens.clone();
    full.push(0);
    driver.state.inner.draft_slot().unwrap().forced = Some(forced(full, vocab, Rc::clone(&ks)));
    let result = driver.turn("a", &prompt, drafted_config(16, 0.0)).unwrap();
    assert_eq!(result.text, plain.text, "the drafted turn's text");
    let held = driver.state.inner.draft_bytes_held();
    assert!(held > 0, "the context the turn left is held: {held} B");
    let slot = driver.state.inner.draft_slot().unwrap();
    let offered: Vec<usize> = slot.offered.iter().map(|&(_, _, n)| n).collect();
    let rows: Vec<usize> = slot.proposals.iter().map(|&(_, row, _)| row).collect();
    let kept: Vec<usize> = slot
        .offered
        .iter()
        .filter_map(|&(_, row, _)| {
            rows.iter()
                .find(|&&later| later > row)
                .map(|next| next - row - 1)
        })
        .collect();
    let perf = result
        .performance
        .expect("the turn reports its performance");
    let cycles = perf.mtp_cycles.unwrap_or(0) as usize;
    let proposed = (perf.mtp_mean_depth.unwrap_or(0.0) * cycles as f64).round() as usize;
    let accepted = (perf.mtp_mean_accepted_tokens.unwrap_or(0.0) * cycles as f64).round() as usize;
    eprintln!(
        "drafted turn: cycles {cycles}, proposed {proposed}, accepted {accepted}, tokens per cycle {:?}; offered {offered:?}, kept {kept:?}, forced {:?}",
        perf.mtp_mean_accepted_tokens_total,
        ks.borrow()
    );
    assert!(cycles >= ROWS, "the turn drafted: {cycles} cycles");
    assert_eq!(cycles, offered.len(), "the drafted cycles");
    assert_eq!(proposed, offered.iter().sum::<usize>(), "the proposals");
    let kept_known: usize = kept.iter().sum();
    let last = offered.last().copied().unwrap_or(0);
    assert!(
        (kept_known..=kept_known + last).contains(&accepted),
        "the proposals kept: {accepted} against {kept_known} (+ up to {last} in the last cycle)"
    );
}
