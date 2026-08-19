//! NemotronH Multi-Token Prediction (MTP) head.
//!
//! Single-step predictor matching the vLLM NemotronH-MTP math
//! (vllm/model_executor/models/nemotron_h_mtp.py):
//!
//!   x0 = eh_proj(concat(enorm(emb(last_token)), hnorm(target_hidden)))
//!   layer 0 (attention): h = x0 + attention(norm(x0))  at position t+1
//!                          reading the TARGET KV (the backbone's final
//!                          attention layer cache)
//!   layer 1 (moe):        h = h + moe(norm(h))          (dense bf16 experts)
//!   h = final_layernorm(h)
//!   logits = shared lm_head(h)   (the main model's untied lm_head)
//!
//! The MTP attention is READ-ONLY over the target KV - it never writes
//! K/V, so the draft head is stateless between cycles and depth is clamped
//! to 1 (one draft token per cycle). mtp.* weights are all dense bf16.

use crate::array::MxArray;
use crate::models::qwen3_5_moe::quantized_linear::LinearProj;
use crate::nn::{Linear, RMSNorm};
use crate::transformer::KVCache;
use napi::bindgen_prelude::*;

use super::attention::NemotronHAttention;
use super::config::NemotronHConfig;
use super::sparse_moe::NemotronHMoE;

/// One MTP layer: pre-norm + a single mixer (attention or dense MoE).
pub struct NemotronHMtpLayer {
    pub(crate) norm: RMSNorm,
    pub(crate) mixer: NemotronHMtpMixer,
}

pub enum NemotronHMtpMixer {
    Attention(NemotronHAttention),
    MoE(NemotronHMoE),
}

impl NemotronHMtpLayer {
    pub fn new(config: &NemotronHConfig, kind: &str) -> Result<Self> {
        let mixer = match kind {
            "full_attention" => NemotronHMtpMixer::Attention(NemotronHAttention::new(config)?),
            "moe" => NemotronHMtpMixer::MoE(NemotronHMoE::new(config, /* dense */ true)?),
            other => {
                return Err(Error::from_reason(format!(
                    "NemotronH MTP layer: unsupported block type '{other}'"
                )));
            }
        };
        let norm = RMSNorm::new(config.hidden_size as u32, Some(config.layer_norm_epsilon))?;
        Ok(Self { norm, mixer })
    }
}

/// The MTP predictor module.
pub struct NemotronHMtpModule {
    pub(crate) enorm: RMSNorm,
    pub(crate) hnorm: RMSNorm,
    pub(crate) eh_proj: LinearProj,
    pub(crate) layers: Vec<NemotronHMtpLayer>,
    pub(crate) final_layernorm: RMSNorm,
}

impl NemotronHMtpModule {
    /// Build the MTP module from config. Fails closed unless the checkpoint
    /// declares exactly the supported [attention, moe] pattern.
    pub fn new(config: &NemotronHConfig) -> Result<Self> {
        if config.n_mtp_layers != 1 {
            return Err(Error::from_reason(format!(
                "NemotronHMtpModule::new requires n_mtp_layers == 1, got {}",
                config.n_mtp_layers
            )));
        }
        let kinds = &config.mtp_layers_block_type;
        if kinds.len() != 2 || kinds[0] != "full_attention" || kinds[1] != "moe" {
            return Err(Error::from_reason(format!(
                "NemotronHMtpModule::new supports only the [attention, moe] MTP block pattern,                  got {:?}",
                kinds
            )));
        }
        let h = config.hidden_size as u32;
        let enorm = RMSNorm::new(h, Some(config.layer_norm_epsilon))?;
        let hnorm = RMSNorm::new(h, Some(config.layer_norm_epsilon))?;
        // eh_proj [hidden, 2*hidden], bias-free.
        let eh_proj = LinearProj::Standard(Linear::new(h * 2, h, Some(false))?);
        let layers = kinds
            .iter()
            .map(|kind| NemotronHMtpLayer::new(config, kind))
            .collect::<Result<Vec<_>>>()?;
        let final_layernorm = RMSNorm::new(h, Some(config.layer_norm_epsilon))?;
        Ok(Self {
            enorm,
            hnorm,
            eh_proj,
            layers,
            final_layernorm,
        })
    }

    /// One MTP draft step.
    ///
    /// prev_hidden is the target's post-final-norm hidden [1, 1, hidden]
    /// of the last committed token; prev_emb is its embedding [1, 1,
    /// hidden]. target_kv is the backbone's FINAL attention layer flat KV
    /// cache; position is the next absolute token position (t+1). Returns
    /// the draft hidden [1, 1, hidden] (the caller applies the shared
    /// lm_head for logits).
    pub fn draft_step(
        &self,
        prev_hidden: &MxArray,
        prev_emb: &MxArray,
        target_kv: Option<&KVCache>,
        position: i32,
    ) -> Result<MxArray> {
        let e_norm = self.enorm.forward(prev_emb)?;
        let h_norm = self.hnorm.forward(prev_hidden)?;
        let concat = MxArray::concatenate(&e_norm, &h_norm, -1)?;
        let mut h = self.eh_proj.forward(&concat)?;

        for layer in self.layers.iter() {
            // Pre-norm residual: the skip connection carries the PRE-norm
            // hidden (NemotronHBlock: residual = h; h = norm(h); h =
            // mixer(h); return residual + h). Adding the normed value here
            // would replace the residual stream with its normalized self
            // and corrupt the draft logits.
            let normed = layer.norm.forward(&h)?;
            let out = match &layer.mixer {
                NemotronHMtpMixer::Attention(attn) => {
                    let kv = target_kv.ok_or_else(|| {
                        Error::from_reason(
                            "NemotronH MTP draft: attention layer requires the target KV cache",
                        )
                    })?;
                    attn.forward_read_only(&normed, kv, position)?
                }
                NemotronHMtpMixer::MoE(moe) => moe.forward(&normed)?,
            };
            h = h.add(&out)?;
        }

        self.final_layernorm.forward(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::DType;
    use crate::models::nemotron_h::config::NemotronHConfig;

    fn tiny_cfg() -> NemotronHConfig {
        NemotronHConfig {
            vocab_size: 32,
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 64,
            layer_norm_epsilon: 1e-5,
            rope_theta: 10000.0,
            layers_block_type: vec!["full_attention".into()],
            mamba_num_heads: 2,
            mamba_head_dim: 2,
            ssm_state_size: 2,
            n_groups: 1,
            conv_kernel: 4,
            chunk_size: 2,
            time_step_min: 0.001,
            n_routed_experts: 2,
            num_experts_per_tok: 1,
            routed_scaling_factor: 1.0,
            n_group: 1,
            topk_group: 1,
            norm_topk_prob: true,
            intermediate_size: 8,
            moe_shared_expert_intermediate_size: 8,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_ids: vec![2],
            pad_token_id: 0,
            num_logits_to_keep: 1,
            mtp_layers_block_type: vec!["full_attention".into(), "moe".into()],
            n_mtp_layers: 1,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            use_block_paged_cache: None,
        }
    }

    /// The MTP draft runs end-to-end on a tiny fixture: seed a target KV
    /// cache with the backbone attention, then draft a token's hidden and
    /// apply the shared lm_head to get draft logits.
    #[test]
    fn mtp_draft_produces_logits_from_shared_head() {
        let cfg = tiny_cfg();
        let mtp = NemotronHMtpModule::new(&cfg).expect("mtp builds");
        let h = cfg.hidden_size as usize;

        // Build the target KV with the backbone attention layer.
        let backbone = NemotronHAttention::new(&cfg).expect("attention builds");
        let mut kv = KVCache::new();
        let seq: Vec<f32> = (0..2 * h)
            .map(|i| ((i as f32) * 0.41) % 1.0 - 0.5)
            .collect();
        let mx = MxArray::from_float32(&seq, &[1, 2, h as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let _ = backbone.forward(&mx, None, Some(&mut kv)).unwrap();
        assert_eq!(kv.get_offset(), 2);

        // prev hidden + embedding of the last committed token.
        let prev_hidden = MxArray::from_float32(&seq[h..], &[1, 1, h as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let prev_emb = MxArray::from_float32(
            &(0..h)
                .map(|i| ((i as f32) * 0.17) % 1.0 - 0.5)
                .collect::<Vec<f32>>(),
            &[1, 1, h as i64],
        )
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap();

        // Draft at position 2 (the next absolute position).
        let draft_h = mtp
            .draft_step(&prev_hidden, &prev_emb, Some(&kv), 2)
            .expect("draft step runs");
        let draft_shape = draft_h.shape().unwrap().to_vec();
        assert_eq!(draft_shape, vec![1, 1, h as i64]);
        let vals = draft_h.to_float32().unwrap().to_vec();
        assert!(vals.iter().all(|v| v.is_finite()));

        // Shared lm_head: logits [1, vocab].
        let head = crate::nn::Linear::new(h as u32, 32, Some(false)).unwrap();
        let logits = head.forward(&draft_h).unwrap();
        let logits_shape = logits.shape().unwrap().to_vec();
        assert_eq!(logits_shape, vec![1, 1, 32]);
        let logits_v = logits.to_float32().unwrap().to_vec();
        assert!(logits_v.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mtp_rejects_wrong_block_pattern() {
        let mut cfg = tiny_cfg();
        cfg.mtp_layers_block_type = vec!["moe".into(), "full_attention".into()];
        let err = match NemotronHMtpModule::new(&cfg) {
            Err(e) => e,
            Ok(_) => panic!("wrong pattern must fail"),
        };
        assert!(err.reason.contains("pattern"), "{}", err.reason);

        let mut cfg = tiny_cfg();
        cfg.n_mtp_layers = 0;
        let err = match NemotronHMtpModule::new(&cfg) {
            Err(e) => e,
            Ok(_) => panic!("n_mtp_layers=0 must fail"),
        };
        assert!(err.reason.contains("n_mtp_layers"), "{}", err.reason);
    }
}

#[cfg(test)]
mod mtp_turn_tests {
    use crate::array::MxArray;
    use crate::decode_profiler::DecodeProfiler;
    use crate::engine::backend::{ChatBackend, ThinkingSetup};
    use crate::engine::mtp_turn::{MtpTurnArgs, run_mtp_turn};
    use crate::engine::params::extract_chat_params;
    use crate::engine::penalties::ReasoningTracker;
    use crate::engine::persistence::compiled_forward_backend_available;
    use crate::engine::plan::{MediaCapabilities, TurnPath, TurnPlan, TurnRequest};
    use crate::engine::types::ChatConfig;
    use crate::models::nemotron_h::config::NemotronHConfig;
    use crate::models::nemotron_h::model::NemotronHInner;
    use crate::models::qwen3_5_moe::quantized_linear::LinearProj;
    use crate::nn::Linear;
    use crate::stream::{DeviceType, Stream};
    use napi::bindgen_prelude::{Error, Result};

    /// Tiny flat config: mamba(0) + moe(1) + attention(2), hidden 8, MTP
    /// head [attention, moe], paged adapter OFF.
    fn tiny_mtp_config() -> NemotronHConfig {
        NemotronHConfig {
            vocab_size: 32,
            hidden_size: 8,
            num_hidden_layers: 3,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 64,
            layer_norm_epsilon: 1e-5,
            rope_theta: 10000.0,
            layers_block_type: vec![
                "linear_attention".into(),
                "moe".into(),
                "full_attention".into(),
            ],
            mamba_num_heads: 2,
            mamba_head_dim: 2,
            ssm_state_size: 2,
            n_groups: 1,
            conv_kernel: 4,
            chunk_size: 3,
            time_step_min: 0.001,
            n_routed_experts: 2,
            num_experts_per_tok: 1,
            routed_scaling_factor: 1.0,
            n_group: 1,
            topk_group: 1,
            norm_topk_prob: true,
            intermediate_size: 8,
            moe_shared_expert_intermediate_size: 8,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_ids: vec![2],
            pad_token_id: 0,
            num_logits_to_keep: 1,
            mtp_layers_block_type: vec!["full_attention".into(), "moe".into()],
            n_mtp_layers: 1,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            // Flat-only fixture: use_block_paged_cache None defaults the
            // adapter ON (head_dim 4 is below the pool minimum), so pin it
            // OFF here; tiny_mtp_paged_config opts back in with a valid
            // pool head_dim.
            use_block_paged_cache: Some(false),
        }
    }

    /// Same shape as the scheduler tests' paged fixture (hidden 256, head_dim
    /// 128 - the generic batched paged route) but with the MTP head on, so
    /// the routing test exercises the paged-adapter-present case.
    fn tiny_mtp_paged_config() -> NemotronHConfig {
        NemotronHConfig {
            vocab_size: 32,
            hidden_size: 256,
            num_hidden_layers: 3,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 128,
            max_position_embeddings: 512,
            layer_norm_epsilon: 1e-5,
            rope_theta: 10000.0,
            layers_block_type: vec![
                "linear_attention".into(),
                "moe".into(),
                "full_attention".into(),
            ],
            mamba_num_heads: 2,
            mamba_head_dim: 2,
            ssm_state_size: 2,
            n_groups: 1,
            conv_kernel: 4,
            chunk_size: 3,
            time_step_min: 0.001,
            n_routed_experts: 4,
            num_experts_per_tok: 1,
            routed_scaling_factor: 1.0,
            n_group: 1,
            topk_group: 1,
            norm_topk_prob: true,
            intermediate_size: 6,
            moe_shared_expert_intermediate_size: 8,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_ids: vec![2],
            pad_token_id: 0,
            num_logits_to_keep: 1,
            mtp_layers_block_type: vec!["full_attention".into(), "moe".into()],
            n_mtp_layers: 1,
            paged_cache_memory_mb: Some(256),
            paged_block_size: Some(16),
            use_block_paged_cache: Some(true),
        }
    }

    fn argmax(vec: &[f32]) -> usize {
        vec.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    /// Deterministic dense bf16 expert stack for the backbone MoE layer
    /// (the tiny fixture's default quantized backends have zero payloads).
    fn install_dense_moe(inner: &mut NemotronHInner) -> Result<()> {
        let h = inner.config.hidden_size as i64;
        let e = inner.config.n_routed_experts as i64;
        let inter = inner.config.intermediate_size as i64;
        let up: Vec<f32> = (0..e * inter * h)
            .map(|i| ((i as f32) * 0.017) % 1.0 - 0.5)
            .collect();
        let down: Vec<f32> = (0..e * h * inter)
            .map(|i| ((i as f32) * 0.031) % 1.0 - 0.5)
            .collect();
        let up_w =
            MxArray::from_float32(&up, &[e, inter, h])?.astype(crate::array::DType::BFloat16)?;
        let down_w =
            MxArray::from_float32(&down, &[e, h, inter])?.astype(crate::array::DType::BFloat16)?;
        let moe = inner.layers[1]
            .moe_mut()
            .ok_or_else(|| Error::from_reason("fixture layer 1 must be MoE"))?;
        moe.experts.set_dense(&up_w, &down_w)
    }

    /// Greedy flat AR oracle: prefill the prompt, then argmax-decode up to n
    /// tokens through the same inner (stopping at the config EOS). Returns
    /// the exact token sequence a plain AR turn would emit.
    fn greedy_ar_oracle(inner: &mut NemotronHInner, prompt: &[u32], n: usize) -> Result<Vec<u32>> {
        inner.reset_caches_internal();
        let stream = Stream::new(DeviceType::Gpu);
        let arr = MxArray::from_uint32(prompt, &[1, prompt.len() as i64])?;
        let logits = inner.chunked_prefill(&arr, stream)?;
        let seq_len = logits.shape_at(1)?;
        let mut last = logits
            .slice_axis(1, seq_len - 1, seq_len)?
            .squeeze(Some(&[1]))?;
        let emb = inner.embedding.clone();
        let eos = inner.config.eos_token_ids.first().copied().unwrap_or(2) as u32;
        let mut out = Vec::new();
        for step in 0..n {
            last.eval();
            let v = last.to_float32()?;
            let t = argmax(&v) as u32;
            out.push(t);
            if t == eos {
                break;
            }
            if step + 1 < n {
                let ids = MxArray::from_uint32(&[t], &[1, 1])?;
                let (l, _) = inner.forward_with_hidden(&ids, &emb)?;
                last = l.squeeze(Some(&[1]))?;
            }
        }
        Ok(out)
    }

    /// The engine-owned run_mtp_turn loop driving the real
    /// NemotronHMtpStepper must propose (draft_step) and verify
    /// (verify_step) at least one cycle on a synthetic MTP-capable inner -
    /// proving the draft head, the read-only backbone-KV attention, the
    /// shared-lm_head projection, and the snapshot/rollback path are all
    /// exercised - and the committed output must stay T=0-identical to a
    /// plain greedy AR decode of the same inner.
    #[test]
    fn mtp_turn_runs_draft_verify_cycles() {
        if !compiled_forward_backend_available() {
            eprintln!("skipping (no Metal backend)");
            return;
        }
        // The fixture's weights are random per construction; a greedy run
        // can land on EOS (token 2) before the first MTP cycle, which is a
        // valid zero-cycle exit. Retry with a fresh random inner so the test
        // is robust while still exercising the draft+verify cycle.
        let prompt: Vec<u32> = vec![1, 5, 9, 3];
        let mut attempts = 0;
        loop {
            attempts += 1;
            assert!(attempts <= 8, "failed to get an MTP cycle in 8 attempts");
            let cfg = tiny_mtp_config();
            let mut inner = NemotronHInner::new(cfg).expect("inner builds");
            install_dense_moe(&mut inner).expect("dense backbone MoE");
            let h = inner.config.hidden_size as u32;
            let v = inner.config.vocab_size as u32;
            inner.lm_head = Some(LinearProj::Standard(
                Linear::new(h, v, Some(false)).expect("lm_head builds"),
            ));
            inner.mtp_weights_loaded = true;
            assert!(inner.has_mtp_weights(), "MTP head must be active");

            // Plain greedy AR oracle on the SAME inner: the target greedy
            // tokens the MTP commit path must reproduce exactly.
            let oracle = greedy_ar_oracle(&mut inner, &prompt, 6).expect("AR oracle");
            assert!(!oracle.is_empty(), "oracle must decode at least one token");

            // Fresh flat state for the speculative run.
            inner.reset_caches_internal();
            let stream = Stream::new(DeviceType::Gpu);
            let arr = MxArray::from_uint32(&prompt, &[1, prompt.len() as i64]).expect("prompt arr");
            let logits = inner.chunked_prefill(&arr, stream).expect("prefill");
            let seq_len = logits.shape_at(1).expect("seq len");
            let last = logits
                .slice_axis(1, seq_len - 1, seq_len)
                .expect("last slice")
                .squeeze(Some(&[1]))
                .expect("last squeeze");

            let chat_cfg = ChatConfig {
                temperature: Some(0.0),
                max_new_tokens: Some(6),
                enable_mtp: Some(true),
                ..ChatConfig::default()
            };
            let p = extract_chat_params(&chat_cfg);
            let y = crate::sampling::sample(&last, p.sampling_config).expect("sample y");
            let mut profiler = DecodeProfiler::new("nemotron_mtp_test", "nemotron_h");
            profiler.set_prompt_tokens(prompt.len() as u32);
            let mut tracker = ReasoningTracker::from_setup(
                &ThinkingSetup {
                    enabled: false,
                    budget: None,
                },
                None,
            );
            let mut generated: Vec<u32> = Vec::new();
            let mut hist: Vec<u32> = Vec::new();
            let mut finish = String::new();
            let mut first_tok: Option<std::time::Instant> = None;
            let mut rng = rand::rng();
            let outcome = run_mtp_turn(
                &mut inner,
                &mut rng,
                MtpTurnArgs {
                    y,
                    depth: 1,
                    params: &p,
                    reasoning_tracker: &mut tracker,
                    profiler: &mut profiler,
                    max_new_tokens: 6,
                    eos_id: 2,
                    generated_tokens: &mut generated,
                    token_history: &mut hist,
                    finish_reason: &mut finish,
                    first_token_instant: &mut first_tok,
                    report_perf: true,
                    generation_stream: stream,
                    prompt_hidden: None,
                    prompt_hidden_ids: None,
                    prompt_hidden_position_base: 0,
                    cancel_flag: None,
                },
                None,
            )
            .expect("run_mtp_turn");
            let _ = outcome;

            // Random-weights EOS exits can land before the first cycle:
            // mtp_acceptance_summary() is None then (no cycle recorded), which
            // is a valid zero-cycle exit — retry fresh rather than panicking.
            let Some(summary) = profiler.mtp_acceptance_summary() else {
                continue;
            };
            if summary.2 < 1 || generated.is_empty() {
                continue;
            }
            // STRICTNESS contract at T=0, verified against the plain AR
            // oracle of the same inner: every committed token must equal the
            // target greedy token. An accepted draft therefore equals
            // argmax(target logits) and a rejected draft is replaced by that
            // argmax - otherwise generated would diverge from oracle.
            assert!(
                generated.len() <= oracle.len(),
                "MTP emitted more tokens than AR: {} vs {}",
                generated.len(),
                oracle.len()
            );
            assert_eq!(
                generated, oracle,
                "T=0 MTP output must match plain greedy AR (lossless contract):                  strict accept/commit violated"
            );
            break;
        }
    }

    /// The mid-cycle-stop desync latch must reach the engine through the
    /// ChatBackend trait: the physical flat caches can be ahead of the saved
    /// token history after a partial MTP cycle (EOS/cancel/repetition
    /// cutoff), and cache recovery resets + re-prefills only when
    /// flat_caches_desynced() reports it (the qwen3_5/qwen3_5_moe contract).
    #[test]
    fn flat_cache_desync_latch_reaches_the_backend_trait() {
        let cfg = tiny_mtp_config();
        let mut inner = NemotronHInner::new(cfg).expect("inner builds");
        inner.mtp_weights_loaded = true;
        assert!(
            !inner.flat_caches_desynced(),
            "fresh inner must report in-sync caches"
        );
        inner.flat_mtp_caches_desynced = true;
        assert!(
            inner.flat_caches_desynced(),
            "a mid-cycle MTP stop must surface the latch to the engine"
        );
        inner.clear_flat_caches_desynced();
        assert!(
            !inner.flat_caches_desynced(),
            "the engine's heal must clear the latch"
        );
    }

    /// On a paged model the engine still resolves an MTP-requested turn to
    /// the Paged path (paged attention takes precedence and the family's
    /// SpeculativePlan truthfully declares supports_paged_attention=false),
    /// so run_paged_turn must re-route it to the flat speculative core -
    /// Prefix reuse must require the sequence's recurrent (Mamba) state to
    /// have SURVIVED: a preempted sequence releases its state while its KV
    /// blocks and owner history remain reusable, so a token-only predicate
    /// would skip the Pass-1 reconstruction and resume with KV at the prefix
    /// boundary but Mamba state at position zero.
    #[test]
    fn recurrent_state_survival_gates_prefix_reuse() {
        let cfg = tiny_mtp_paged_config();
        let mut inner = NemotronHInner::new(cfg).expect("inner builds");
        // Fresh activation (no parked state for this seq) must report the
        // state as NOT survived.
        inner.activate_paged_seq(0).expect("activate fresh seq");
        assert!(
            !inner.active_seq_recurrent_survived,
            "fresh zero-state caches must not count as survived state"
        );
        // Park (state survives in the map) and re-activate: now it survived.
        inner.park_active_scheduled_caches();
        inner.activate_paged_seq(0).expect("reactivate parked seq");
        assert!(
            inner.active_seq_recurrent_survived,
            "parked caches restored at the exact boundary must count as survived"
        );
        // Preemption releases the state; the next activation is fresh again.
        inner.park_active_scheduled_caches();
        inner.release_scheduled_caches_for(0);
        inner
            .activate_paged_seq(0)
            .expect("reactivate after preemption");
        assert!(
            !inner.active_seq_recurrent_survived,
            "preemption-released state must force Pass-1 reconstruction"
        );
    }

    /// A zero (or out-of-set) paged block size must fail the load with a
    /// clear error instead of panicking inside the capacity math.
    #[test]
    fn rejects_invalid_paged_block_size() {
        let mut cfg = tiny_mtp_paged_config();
        cfg.paged_block_size = Some(0);
        let err = NemotronHInner::new(cfg).err().expect("must fail");
        assert!(err.reason.contains("paged_block_size"), "{}", err.reason);
        let mut cfg2 = tiny_mtp_paged_config();
        cfg2.paged_block_size = Some(64);
        let err2 = NemotronHInner::new(cfg2).err().expect("must fail");
        assert!(err2.reason.contains("paged_block_size"), "{}", err2.reason);
    }

    /// the exact wiring the real-checkpoint E2E was missing.
    #[test]
    fn mtp_request_on_paged_model_routes_to_flat_core() {
        if !compiled_forward_backend_available() {
            eprintln!("skipping (no Metal backend)");
            return;
        }
        let cfg = tiny_mtp_paged_config();
        let mut inner = NemotronHInner::new(cfg).expect("inner builds");
        assert!(
            inner.paged_adapter.is_some(),
            "gate requires the paged adapter"
        );
        inner.mtp_weights_loaded = true;

        let exec = inner.execution_plan();
        assert!(exec.paged_attention.is_some(), "paged adapter exposed");
        assert!(exec.speculative.is_some(), "MTP plan present");

        // Engine resolution: paged attention wins, the speculative decoder
        // is not admitted, and the turn resolves to the Paged path.
        let plan = TurnPlan::resolve(
            exec,
            TurnRequest {
                is_delta: false,
                input_media: MediaCapabilities::NONE,
                context_media: MediaCapabilities::NONE,
                speculative_requested: true,
            },
        );
        assert_eq!(
            plan.path(),
            TurnPath::Paged,
            "engine resolves paged+MTP to the paged path (the override compensates)"
        );

        // The family's run_paged_turn override compensates: any sync
        // MTP-requested turn on an MTP-capable model routes flat.
        let mut chat_cfg = ChatConfig {
            enable_mtp: Some(true),
            ..ChatConfig::default()
        };
        let p = extract_chat_params(&chat_cfg);
        assert!(
            inner.mtp_flat_routing_required(&p, false),
            "sync enable_mtp turn on an MTP-capable paged model must route flat"
        );
        assert!(
            !inner.mtp_flat_routing_required(&p, true),
            "streaming MTP turns keep the paged AR fallback"
        );
        chat_cfg.enable_mtp = None;
        let p_ar = extract_chat_params(&chat_cfg);
        assert!(
            !inner.mtp_flat_routing_required(&p_ar, false),
            "plain AR turns must stay on the paged lane"
        );
        inner.mtp_weights_loaded = false;
        assert!(
            !inner.mtp_flat_routing_required(&p, false),
            "MTP requested without a loaded head must not route flat"
        );
    }

    /// Real-checkpoint T=0 lossless gate (env-gated: set
    /// MLX_TEST_NEMOTRON_H_MODEL_PATH). Runs plain greedy AR and the MTP
    /// loop from the same prompt and asserts the committed token sequences
    /// are IDENTICAL. Set MLX_MTP_TRACE_ACCEPTANCE=1 to also see the
    /// per-slot accept trace (draft_id / target_argmax / accepted).
    #[test]
    fn real_mtp_t0_lossless_gate() {
        if !compiled_forward_backend_available() {
            eprintln!("skipping (no Metal backend)");
            return;
        }
        let Ok(model_path) = std::env::var("MLX_TEST_NEMOTRON_H_MODEL_PATH") else {
            eprintln!("skipping: MLX_TEST_NEMOTRON_H_MODEL_PATH unset");
            return;
        };
        let (mut inner, _bytes) =
            crate::models::nemotron_h::persistence::load_inner(&model_path).expect("load real");
        if !inner.has_mtp_weights() {
            eprintln!("skipping: no MTP weights in checkpoint");
            return;
        }
        let eos = inner.config.eos_token_ids.first().copied().unwrap_or(2);
        let n = 60;
        // A real tokenized prompt (the model dir ships tokenizer.json) so the
        // draft head sees in-distribution context and the acceptance rate is
        // meaningful; the token ids are still opaque to the forward itself.
        let prompt: Vec<u32> = crate::tokenizer::Qwen3Tokenizer::from_file(
            &std::path::Path::new(&model_path).join("tokenizer.json"),
        )
        .ok()
        .and_then(|tok| {
            tok.encode_sync("What is 2+2? Answer in one sentence.", None)
                .ok()
        })
        .unwrap_or_else(|| vec![1, 5, 9, 3, 7, 13, 21, 34]);

        // ---- plain greedy AR oracle ----
        let ar = greedy_ar_oracle(&mut inner, &prompt, n).expect("AR oracle");

        // ---- MTP loop from the same cold prefix ----
        inner.reset_caches_internal();
        let stream = Stream::new(DeviceType::Gpu);
        let arr = MxArray::from_uint32(&prompt, &[1, prompt.len() as i64]).expect("prompt arr");
        let logits = inner.chunked_prefill(&arr, stream).expect("prefill");
        let seq_len = logits.shape_at(1).expect("seq len");
        let last = logits
            .slice_axis(1, seq_len - 1, seq_len)
            .expect("last slice")
            .squeeze(Some(&[1]))
            .expect("last squeeze");
        let chat_cfg = ChatConfig {
            temperature: Some(0.0),
            max_new_tokens: Some(n as i32),
            enable_mtp: Some(true),
            ..ChatConfig::default()
        };
        let p = extract_chat_params(&chat_cfg);
        let y = crate::sampling::sample(&last, p.sampling_config).expect("sample y");
        let mut profiler = DecodeProfiler::new("nemotron_real_mtp_diag", "nemotron_h");
        let mut tracker = ReasoningTracker::from_setup(
            &ThinkingSetup {
                enabled: false,
                budget: None,
            },
            None,
        );
        let mut generated: Vec<u32> = Vec::new();
        let mut hist: Vec<u32> = Vec::new();
        let mut finish = String::new();
        let mut first_tok: Option<std::time::Instant> = None;
        let mut rng = rand::rng();
        let outcome = run_mtp_turn(
            &mut inner,
            &mut rng,
            MtpTurnArgs {
                y,
                depth: 1,
                params: &p,
                reasoning_tracker: &mut tracker,
                profiler: &mut profiler,
                max_new_tokens: n as i32,
                eos_id: eos as u32,
                generated_tokens: &mut generated,
                token_history: &mut hist,
                finish_reason: &mut finish,
                first_token_instant: &mut first_tok,
                report_perf: true,
                generation_stream: stream,
                prompt_hidden: None,
                prompt_hidden_ids: None,
                prompt_hidden_position_base: 0,
                cancel_flag: None,
            },
            None,
        )
        .expect("run_mtp_turn");
        let _ = outcome;

        let mut first_div = None;
        for i in 0..ar.len().min(generated.len()) {
            if ar[i] != generated[i] {
                first_div = Some(i);
                break;
            }
        }
        eprintln!("DIAG AR({}): {:?}", ar.len(), &ar[..ar.len().min(40)]);
        eprintln!(
            "DIAG MTP({}): {:?}",
            generated.len(),
            &generated[..generated.len().min(40)]
        );
        eprintln!(
            "DIAG first divergence: {:?}",
            first_div.map(|i| (i, ar[i], generated[i]))
        );
        eprintln!("DIAG acceptance: {:?}", profiler.mtp_acceptance_summary());
        assert!(
            first_div.is_none(),
            "T=0 lossless violated: MTP[{}]={} != AR[{}]={}",
            first_div.map(|i| i.to_string()).unwrap_or_default(),
            first_div.map(|i| generated[i]).unwrap_or(0),
            first_div.map(|i| i.to_string()).unwrap_or_default(),
            first_div.map(|i| ar[i]).unwrap_or(0),
        );
    }
}
