//! NemotronH Multi-Token Prediction (MTP) head.
//!
//! Single-step predictor matching the vLLM NemotronH-MTP math
//! (vllm/model_executor/models/nemotron_h_mtp.py):
//!
//!   x0 = eh_proj(concat(enorm(emb(t_{p+1})), hnorm(h_p)))
//!   layer 0 (attention): h = x0 + attention(norm(x0))  over the HEAD's OWN
//!                          K/V history (its own k_proj/v_proj, its own
//!                          causal cache)
//!   layer 1 (moe):        h = h + moe(norm(h))          (dense bf16 experts)
//!   h = final_layernorm(h)
//!   logits = shared lm_head(h)   (the main model's untied lm_head)
//!
//! The head is a real decoder layer with its OWN KV cache group, exactly as
//! vLLM builds it: `NemotronHMTPAttentionDecoderLayer(NemotronHAttentionDecoderLayer)`
//! (nemotron_h_mtp.py:40) whose `mixer` is a full `NemotronHAttention`
//! (nemotron_h.py:497) holding its own `Attention(...)` (nemotron_h.py:457).
//! It is therefore STATEFUL across cycles: [`NemotronHMtpModule::forward`]
//! WRITES K/V into the caller-owned per-layer caches, which the stepper
//! rewinds by cursor `trim` on rejection and extends on commit.
//!
//! Slot convention (vLLM `llm_base_proposer.py:845-867`, the EAGLE
//! `not needs_extra_input_slots` branch this head takes): drafter slot `p`
//! holds `fused(enorm(emb(t_{p+1})), hnorm(h_p))` — token ids shifted LEFT
//! by one against the target hiddens, with the newly sampled token in the
//! final prompt slot. The prompt seed
//! ([`NemotronHInner::chunked_prefill_seeding_mtp`](super::model)) and
//! `commit_mtp` both write that pairing.
//!
//! NemotronH attention is NoPE, so no positions are threaded anywhere;
//! causality comes from the cache offset plus the "causal" SDPA mask, which
//! is bottom-right aligned when `q_len < k_len` (proved by
//! `causal_attention_matches_explicit_offset_mask_when_kv_is_longer` in
//! `array/attention.rs`). mtp.* weights are all dense bf16.

use crate::array::MxArray;
use crate::models::qwen3_5_moe::quantized_linear::LinearProj;
use crate::nn::{Linear, RMSNorm};
use napi::bindgen_prelude::*;

use super::attention::NemotronHAttention;
use super::config::NemotronHConfig;
use super::layer_cache::NemotronHLayerCache;
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

    /// Build a fresh per-layer cache slot for every MTP layer.
    ///
    /// One slot per entry of `config.mtp_layers_block_type`, in order:
    /// `full_attention` gets a flat [`KVCache`](crate::transformer::KVCache),
    /// `moe` is stateless. Mirrors `Qwen3_5MoeMTPModule::fresh_caches`
    /// (qwen3_5_moe/mtp.rs:221). The stepper owns these for the whole turn
    /// and rewinds them by `trim`, never by snapshot.
    pub fn fresh_caches(config: &NemotronHConfig) -> Vec<NemotronHLayerCache> {
        config
            .mtp_layers_block_type
            .iter()
            .map(|kind| {
                if kind == "full_attention" {
                    NemotronHLayerCache::new_attention()
                } else {
                    NemotronHLayerCache::new_moe()
                }
            })
            .collect()
    }

    /// One MTP forward over `[1, L, hidden]` inputs, writing the head's OWN
    /// K/V into `caches`.
    ///
    /// `prev_hidden` holds the target's post-final-norm hiddens `h_p ..
    /// h_{p+L-1}`; `prev_emb` holds the embeddings of the tokens ONE
    /// POSITION LATER (`emb(t_{p+1}) .. emb(t_{p+L})`) — the vLLM EAGLE
    /// left-shift. Returns the draft hidden `[1, L, hidden]`; the caller
    /// applies the shared lm_head for logits.
    ///
    /// `L` is 1 for a draft step and up to the prefill chunk size for the
    /// prompt seed / commit. Causality inside a multi-token call comes from
    /// the "causal" SDPA mode; causality against earlier calls comes from
    /// the cache offset. NemotronH attention is NoPE, so no position is
    /// threaded in.
    pub fn forward(
        &self,
        prev_hidden: &MxArray,
        prev_emb: &MxArray,
        caches: &mut [NemotronHLayerCache],
    ) -> Result<MxArray> {
        if caches.len() != self.layers.len() {
            return Err(Error::from_reason(format!(
                "NemotronHMtpModule::forward: caches length {} != layers length {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let e_norm = self.enorm.forward(prev_emb)?;
        let h_norm = self.hnorm.forward(prev_hidden)?;
        let concat = MxArray::concatenate(&e_norm, &h_norm, -1)?;
        let mut h = self.eh_proj.forward(&concat)?;

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            // Pre-norm residual: the skip connection carries the PRE-norm
            // hidden (NemotronHBlock: residual = h; h = norm(h); h =
            // mixer(h); return residual + h). Adding the normed value here
            // would replace the residual stream with its normalized self
            // and corrupt the draft logits.
            let normed = layer.norm.forward(&h)?;
            let out = match &layer.mixer {
                NemotronHMtpMixer::Attention(attn) => {
                    let kv = cache.as_kv_cache_mut().ok_or_else(|| {
                        Error::from_reason(
                            "NemotronH MTP forward: attention layer needs an Attention cache slot",
                        )
                    })?;
                    attn.forward(&normed, None, Some(kv))?
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
            time_step_limit: None,
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

    /// Deterministic [1, L, hidden] bf16 fixture input.
    fn seq_input(h: usize, len: usize, phase: f32) -> MxArray {
        let v: Vec<f32> = (0..len * h)
            .map(|i| ((i as f32 + phase) * 0.41) % 1.0 - 0.5)
            .collect();
        MxArray::from_float32(&v, &[1, len as i64, h as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap()
    }

    fn attn_offset(caches: &[NemotronHLayerCache]) -> i32 {
        caches
            .iter()
            .find_map(|c| c.as_kv_cache())
            .expect("an attention slot")
            .get_offset()
    }

    /// The MTP draft runs end-to-end on a tiny fixture: draft a token's
    /// hidden from its OWN cache and apply the shared lm_head to get draft
    /// logits.
    #[test]
    fn mtp_draft_produces_logits_from_shared_head() {
        let cfg = tiny_cfg();
        let mtp = NemotronHMtpModule::new(&cfg).expect("mtp builds");
        let h = cfg.hidden_size as usize;
        let mut caches = NemotronHMtpModule::fresh_caches(&cfg);

        let prev_hidden = seq_input(h, 1, 3.0);
        let prev_emb = seq_input(h, 1, 11.0);

        let draft_h = mtp
            .forward(&prev_hidden, &prev_emb, &mut caches)
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

    /// T1: the head WRITES its own K/V. Two L=1 forwards must advance the
    /// attention slot's offset 0 -> 1 -> 2, and a multi-token forward must
    /// advance it by L.
    ///
    /// MUTATION: the pre-port read-only head (`forward_read_only` over the
    /// backbone KV) never wrote K/V, so the offset stayed at 0 forever.
    #[test]
    fn mtp_draft_writes_its_own_kv() {
        let cfg = tiny_cfg();
        let mtp = NemotronHMtpModule::new(&cfg).expect("mtp builds");
        let h = cfg.hidden_size as usize;
        let mut caches = NemotronHMtpModule::fresh_caches(&cfg);
        assert_eq!(caches.len(), cfg.mtp_layers_block_type.len());
        assert_eq!(attn_offset(&caches), 0, "fresh drafter cache is empty");

        mtp.forward(&seq_input(h, 1, 0.0), &seq_input(h, 1, 5.0), &mut caches)
            .expect("first draft");
        assert_eq!(attn_offset(&caches), 1, "one draft token written");

        mtp.forward(&seq_input(h, 1, 1.0), &seq_input(h, 1, 6.0), &mut caches)
            .expect("second draft");
        assert_eq!(attn_offset(&caches), 2, "second draft token written");

        mtp.forward(&seq_input(h, 3, 2.0), &seq_input(h, 3, 7.0), &mut caches)
            .expect("multi-token seed");
        assert_eq!(attn_offset(&caches), 5, "L=3 seed writes three slots");
    }

    /// T2: the head READS its own history. The same input produces a
    /// different output when a different token precedes it in the drafter
    /// cache.
    ///
    /// MUTATION: dropping the cache argument, attending only to the current
    /// token, or reverting to the backbone KV all make out_a == out_b
    /// bit-for-bit. It also dies if k_proj/v_proj are never read (constant
    /// K/V), which is exactly the defect this port fixes.
    #[test]
    fn mtp_draft_reads_its_own_history() {
        let cfg = tiny_cfg();
        let mtp = NemotronHMtpModule::new(&cfg).expect("mtp builds");
        let h = cfg.hidden_size as usize;
        let x_hidden = seq_input(h, 1, 0.0);
        let x_emb = seq_input(h, 1, 5.0);

        let mut fresh = NemotronHMtpModule::fresh_caches(&cfg);
        let out_a = mtp
            .forward(&x_hidden, &x_emb, &mut fresh)
            .expect("cold draft")
            .to_float32()
            .unwrap()
            .to_vec();

        let mut warm = NemotronHMtpModule::fresh_caches(&cfg);
        mtp.forward(&seq_input(h, 1, 9.0), &seq_input(h, 1, 13.0), &mut warm)
            .expect("history token");
        let out_b = mtp
            .forward(&x_hidden, &x_emb, &mut warm)
            .expect("warm draft")
            .to_float32()
            .unwrap()
            .to_vec();

        let max_diff = out_a
            .iter()
            .zip(out_b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-3,
            "the drafter must attend over its own history: max |diff| = {max_diff}"
        );
    }

    /// The head's K/V really come from its OWN k_proj/v_proj: perturbing
    /// those two projections must change the draft output.
    ///
    /// MUTATION: the pre-port head loaded `mtp.*.k_proj` / `v_proj` at
    /// persistence.rs:626-633 and NEVER read them — this test is the one the
    /// old suite could not write.
    #[test]
    fn mtp_kv_projections_change_the_draft() {
        let cfg = tiny_cfg();
        let h = cfg.hidden_size as usize;
        let kv_dim = (cfg.num_key_value_heads * cfg.head_dim) as usize;
        let x_hidden = seq_input(h, 1, 0.0);
        let x_emb = seq_input(h, 1, 5.0);

        let run = |scale: f32| -> Vec<f32> {
            let mtp = NemotronHMtpModule::new(&cfg).expect("mtp builds");
            // Pin both KV projections to a deterministic weight so the two
            // runs differ ONLY in k_proj/v_proj (every other weight comes
            // from the process-global RNG and would otherwise differ).
            let w: Vec<f32> = (0..kv_dim * h)
                .map(|i| (((i * 37) % 23) as f32 * 0.011 - 0.12) * scale)
                .collect();
            let w = MxArray::from_float32(&w, &[kv_dim as i64, h as i64])
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            let mut mtp = mtp;
            match &mut mtp.layers[0].mixer {
                NemotronHMtpMixer::Attention(a) => {
                    a.k_proj.set_weight(&w, "k").unwrap();
                    a.v_proj.set_weight(&w, "v").unwrap();
                }
                _ => panic!("layer 0 is attention"),
            }
            let mut caches = NemotronHMtpModule::fresh_caches(&cfg);
            // Two tokens: the second query attends over the first token's
            // K/V, so a k_proj/v_proj change must move the output.
            mtp.forward(&seq_input(h, 1, 21.0), &seq_input(h, 1, 31.0), &mut caches)
                .expect("history token");
            mtp.forward(&x_hidden, &x_emb, &mut caches)
                .expect("draft")
                .to_float32()
                .unwrap()
                .to_vec()
        };

        // Same RNG draw order in both runs => every non-KV weight matches.
        let a = run(1.0);
        let b = run(-2.5);
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-3,
            "perturbing mtp k_proj/v_proj must change the draft: max |diff| = {max_diff}"
        );
    }

    /// `fresh_caches` must lay out one slot per declared MTP block, in
    /// order, or `forward`'s zip would silently drive the MoE layer with the
    /// attention cache.
    #[test]
    fn fresh_caches_match_the_declared_block_types() {
        let cfg = tiny_cfg();
        let caches = NemotronHMtpModule::fresh_caches(&cfg);
        assert_eq!(caches.len(), 2);
        assert!(caches[0].as_kv_cache().is_some(), "slot 0 is attention");
        assert!(caches[1].as_kv_cache().is_none(), "slot 1 is stateless MoE");

        // A length mismatch must fail loudly rather than half-run.
        let mtp = NemotronHMtpModule::new(&cfg).expect("mtp builds");
        let h = cfg.hidden_size as usize;
        let mut short: Vec<NemotronHLayerCache> = vec![NemotronHLayerCache::new_attention()];
        let err = mtp
            .forward(&seq_input(h, 1, 0.0), &seq_input(h, 1, 1.0), &mut short)
            .err()
            .expect("length mismatch must error");
        assert!(err.reason.contains("caches length"), "{}", err.reason);
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
    use super::NemotronHMtpModule;
    use crate::array::MxArray;
    use crate::decode_profiler::DecodeProfiler;
    use crate::engine::backend::{
        ChatBackend, MtpBackend, MtpStepper, MtpTurnSetup, ThinkingSetup,
    };
    use crate::engine::mtp_turn::{MtpTurnArgs, run_mtp_turn};
    use crate::engine::params::extract_chat_params;
    use crate::engine::penalties::ReasoningTracker;
    use crate::engine::persistence::compiled_forward_backend_available;
    use crate::engine::plan::{MediaCapabilities, TurnPath, TurnPlan, TurnRequest};
    use crate::engine::types::ChatConfig;
    use crate::models::nemotron_h::config::NemotronHConfig;
    use crate::models::nemotron_h::layer_cache::NemotronHLayerCache;
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
            time_step_limit: None,
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
            time_step_limit: None,
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

    /// Prefill the prompt AND seed the MTP drafter over it, exactly as
    /// `run_mtp_whole_turn` does: chunked hidden-emitting prefill feeding the
    /// shifted (hidden, emb) pairs into a fresh drafter cache, sample `y`,
    /// seed the final slot with it, install `pending_mtp_draft_seed`.
    /// Returns the sampled `y` (as `run_mtp_turn` wants it).
    ///
    /// Every MTP turn test must go through this: `begin_mtp_decode` now
    /// hard-errors on an unseeded drafter.
    fn prefill_and_seed_mtp(
        inner: &mut NemotronHInner,
        prompt: &[u32],
        stream: Stream,
        p: &crate::engine::params::ChatParams,
    ) -> Result<MxArray> {
        let arr = MxArray::from_uint32(prompt, &[1, prompt.len() as i64])?;
        let mut caches = super::NemotronHMtpModule::fresh_caches(&inner.config);
        let mut committed_len = 0i32;
        let (logits, h_last) =
            inner.chunked_prefill_seeding_mtp(&arr, stream, &mut caches, &mut committed_len)?;
        let seq_len = logits.shape_at(1)?;
        let last = logits
            .slice_axis(1, seq_len - 1, seq_len)?
            .squeeze(Some(&[1]))?;
        let y = crate::sampling::sample(&last, p.sampling_config)?;
        y.eval();
        let y_id = y.item_at_int32(0)? as u32;
        inner.seed_mtp_final_slot(&h_last, y_id, &mut caches, &mut committed_len)?;
        assert_eq!(
            committed_len as usize,
            prompt.len(),
            "the seed must cover exactly the prompt"
        );
        inner.pending_mtp_draft_seed = Some((caches, committed_len));
        Ok(y)
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

            let chat_cfg = ChatConfig {
                temperature: Some(0.0),
                max_new_tokens: Some(6),
                enable_mtp: Some(true),
                ..ChatConfig::default()
            };
            let p = extract_chat_params(&chat_cfg);
            let y = prefill_and_seed_mtp(&mut inner, &prompt, stream, &p).expect("prefill + seed");
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

            // Random-weights EOS exits can land before the first cycle:
            // mtp_acceptance_summary() is None then (no cycle recorded), which
            // is a valid zero-cycle exit — retry fresh rather than panicking.
            let Some(summary) = profiler.mtp_acceptance_summary() else {
                continue;
            };
            if summary.2 < 1 || generated.is_empty() {
                continue;
            }
            // The outcome is CONSUMED, not discarded (it used to be
            // `let _ = outcome;`): the desync latch and the unemitted
            // remainder must agree, because the latch is the ONLY signal
            // that tells the engine the caches sit ahead of the saved
            // history.
            assert_eq!(
                outcome.desynced,
                outcome.rollback_unemitted > 0,
                "desync latch must fire exactly when a mid-cycle stop left tokens unemitted"
            );
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

    // ===================== drafter-own-KV gates =====================

    /// An MTP-capable tiny inner: dense backbone MoE + a real lm_head + the
    /// MTP head armed.
    fn mtp_ready_inner() -> NemotronHInner {
        let mut inner = NemotronHInner::new(tiny_mtp_config()).expect("inner builds");
        install_dense_moe(&mut inner).expect("dense backbone MoE");
        let h = inner.config.hidden_size as u32;
        let v = inner.config.vocab_size as u32;
        inner.lm_head = Some(LinearProj::Standard(
            Linear::new(h, v, Some(false)).expect("lm_head builds"),
        ));
        inner.mtp_weights_loaded = true;
        assert!(inner.has_mtp_weights());
        inner
    }

    fn greedy_params() -> crate::engine::params::ChatParams {
        extract_chat_params(&ChatConfig {
            temperature: Some(0.0),
            max_new_tokens: Some(6),
            enable_mtp: Some(true),
            ..ChatConfig::default()
        })
    }

    fn det_rows(h: usize, len: usize, phase: f32) -> MxArray {
        let v: Vec<f32> = (0..len * h)
            .map(|i| ((i as f32 + phase) * 0.29) % 1.0 - 0.5)
            .collect();
        MxArray::from_float32(&v, &[1, len as i64, h as i64])
            .unwrap()
            .astype(crate::array::DType::BFloat16)
            .unwrap()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// Draft-probe output of a drafter cache, WITHOUT mutating it further
    /// than one probe write (each caller uses a fresh cache).
    fn probe(mtp: &NemotronHMtpModule, caches: &mut [NemotronHLayerCache], h: usize) -> Vec<f32> {
        mtp.forward(&det_rows(h, 1, 77.0), &det_rows(h, 1, 91.0), caches)
            .expect("probe draft")
            .to_float32()
            .unwrap()
            .to_vec()
    }

    /// T3 — THE shift gate. The prompt seed must pair drafter slot `p` with
    /// `(h_p, emb(t_{p+1}))` and put the newly sampled `y` in the final slot
    /// (vLLM `llm_base_proposer.py:845-867`). Compared against a single
    /// reference `forward` over the whole prompt built with the ids shifted
    /// by hand; the UNSHIFTED pairing must be measurably different.
    ///
    /// MUTATION: no shift, shift by two, or forgetting to substitute `y` in
    /// the last slot. Fails on HEAD, where no seed exists at all.
    #[test]
    fn mtp_prompt_seed_shifts_ids_by_one() {
        if !compiled_forward_backend_available() {
            eprintln!("skipping (no Metal backend)");
            return;
        }
        let mut inner = mtp_ready_inner();
        let h = inner.config.hidden_size as usize;
        let prompt: Vec<u32> = vec![1, 5, 9, 3];
        let y_id: u32 = 7;
        let stream = Stream::new(DeviceType::Gpu);

        // --- the production seed path ---
        inner.reset_caches_internal();
        let arr = MxArray::from_uint32(&prompt, &[1, prompt.len() as i64]).unwrap();
        let mut seeded = NemotronHMtpModule::fresh_caches(&inner.config);
        let mut committed = 0i32;
        let (_logits, h_last) = inner
            .chunked_prefill_seeding_mtp(&arr, stream, &mut seeded, &mut committed)
            .expect("seeding prefill");
        inner
            .seed_mtp_final_slot(&h_last, y_id, &mut seeded, &mut committed)
            .expect("final slot");
        assert_eq!(
            committed,
            prompt.len() as i32,
            "seed covers the whole prompt"
        );
        assert_eq!(
            seeded
                .iter()
                .find_map(|c| c.as_kv_cache())
                .unwrap()
                .get_offset(),
            prompt.len() as i32,
            "one drafter slot per prompt token"
        );

        // --- reference: one backbone forward, ids shifted by hand ---
        inner.reset_caches_internal();
        let embedding = inner.embedding.clone();
        let (_l, hidden) = inner
            .forward_with_hidden_3d(&arr, &embedding)
            .expect("reference prefill");
        let shifted: Vec<u32> = vec![prompt[1], prompt[2], prompt[3], y_id];
        let unshifted: Vec<u32> = prompt.clone();

        let build = |ids: &[u32]| -> Vec<NemotronHLayerCache> {
            let emb = embedding
                .forward(&MxArray::from_uint32(ids, &[1, ids.len() as i64]).unwrap())
                .unwrap();
            let mut caches = NemotronHMtpModule::fresh_caches(&inner.config);
            inner
                .mtp
                .as_ref()
                .unwrap()
                .forward(&hidden, &emb, &mut caches)
                .expect("reference seed");
            caches
        };
        let mut ref_shifted = build(&shifted);
        let mut ref_unshifted = build(&unshifted);

        let mtp = inner.mtp.as_ref().unwrap();
        let got = probe(mtp, &mut seeded, h);
        let want = probe(mtp, &mut ref_shifted, h);
        let wrong = probe(mtp, &mut ref_unshifted, h);

        let d_ok = max_abs_diff(&got, &want);
        let d_bad = max_abs_diff(&got, &wrong);
        // Measured on the random-weight fixture over repeated runs:
        // d_ok 2.6e-5 .. 1.4e-3, d_bad 0.43 .. 1.02 — a ~500x separation.
        // The gates sit an order of magnitude inside both.
        assert!(
            d_ok <= 1e-2,
            "chunked seed must equal the hand-shifted reference: max |diff| = {d_ok}"
        );
        assert!(
            d_bad > 5e-2,
            "the UNSHIFTED pairing must be a different history: max |diff| = {d_bad}"
        );
    }

    /// T4 — the seed must not depend on where the prefill chunk boundaries
    /// fall.
    ///
    /// MUTATION: an off-by-one in the per-chunk id window
    /// `tokens[s+1 .. s+1+L']`, or a wrong `L' = min(e-s, T-1-s)` clamp on
    /// the final chunk (which silently drops or duplicates a pair only when
    /// the prompt spans more than one chunk).
    #[test]
    fn mtp_seed_is_chunk_boundary_invariant() {
        if !compiled_forward_backend_available() {
            eprintln!("skipping (no Metal backend)");
            return;
        }
        let mut inner = mtp_ready_inner();
        let h = inner.config.hidden_size as usize;
        // 7 tokens over chunk_size 3: a step of 3 gives [0,3) [3,6) [6,7).
        let prompt: Vec<u32> = vec![1, 5, 9, 3, 4, 8, 2];
        let y_id: u32 = 6;
        let stream = Stream::new(DeviceType::Gpu);
        let arr = MxArray::from_uint32(&prompt, &[1, prompt.len() as i64]).unwrap();

        let mut seed_with = |step: u32| -> (Vec<NemotronHLayerCache>, i32) {
            inner.reset_caches_internal();
            let mut caches = NemotronHMtpModule::fresh_caches(&inner.config);
            let mut committed = 0i32;
            let (_l, h_last) = inner
                .chunked_prefill_seeding_mtp_stepped(
                    &arr,
                    stream,
                    step,
                    &mut caches,
                    &mut committed,
                )
                .expect("seeding prefill");
            inner
                .seed_mtp_final_slot(&h_last, y_id, &mut caches, &mut committed)
                .expect("final slot");
            (caches, committed)
        };

        let (mut one_chunk, c1) = seed_with(2048);
        let (mut many_chunks, c2) = seed_with(3);
        assert_eq!(c1, prompt.len() as i32);
        assert_eq!(c2, c1, "chunking must not change the committed length");
        assert_eq!(
            many_chunks
                .iter()
                .find_map(|c| c.as_kv_cache())
                .unwrap()
                .get_offset(),
            prompt.len() as i32,
            "multi-chunk seed writes exactly T slots"
        );

        let mtp = inner.mtp.as_ref().unwrap();
        let a = probe(mtp, &mut one_chunk, h);
        let b = probe(mtp, &mut many_chunks, h);
        let d = max_abs_diff(&a, &b);
        // Measured 7.8e-5 .. 7.6e-4 on the random-weight fixture. A real
        // window off-by-one either slices out of range (hard error) or moves
        // the output by ~0.5.
        assert!(d <= 1e-2, "seed is chunk-dependent: max |diff| = {d}");
    }

    /// T8 — `begin_mtp_decode` must REFUSE to run without a seeded drafter
    /// rather than drafting from an empty history.
    ///
    /// MUTATION: a silent `unwrap_or_default()` / `fresh_caches()` fallback,
    /// which is exactly the no-history bug this port removes.
    #[test]
    fn begin_mtp_decode_refuses_an_unseeded_drafter() {
        let mut inner = mtp_ready_inner();
        assert!(inner.pending_mtp_draft_seed.is_none());
        let setup = MtpTurnSetup {
            prompt_hidden: None,
            prompt_hidden_ids: None,
            prompt_hidden_position_base: 0,
            first_sampled_token: 5,
        };
        let err = inner
            .begin_mtp_decode(&setup)
            .err()
            .expect("unseeded drafter must fail closed");
        assert!(
            err.reason.contains("without a seeded drafter cache"),
            "{}",
            err.reason
        );
    }

    /// A cache reset must drop the pending seed: it describes a token stream
    /// that no longer exists.
    ///
    /// MUTATION: leaving `pending_mtp_draft_seed` alive across
    /// `reset_caches_internal`, so the next turn drafts against the previous
    /// turn's history.
    #[test]
    fn reset_clears_the_pending_draft_seed() {
        let mut inner = mtp_ready_inner();
        inner.pending_mtp_draft_seed = Some((NemotronHMtpModule::fresh_caches(&inner.config), 4));
        inner.reset_caches_internal();
        assert!(
            inner.pending_mtp_draft_seed.is_none(),
            "reset must drop the drafter seed"
        );
    }

    /// T5 + the rejection rewind: the drafter's speculative K/V must be
    /// OVERWRITTEN by the commit, never appended past, and a new cycle must
    /// rewind the previous cycle's draft tail.
    ///
    /// MUTATION: dropping the `trim(committed_len)` in `commit_mtp` (or the
    /// trim in `begin_cycle`) — the rejected draft's K/V survives in the
    /// head's history and every later draft conditions on a token that was
    /// never emitted. The offsets catch it structurally; the probe
    /// comparison catches it numerically.
    #[test]
    fn mtp_commit_overwrites_the_rejected_draft_slot() {
        if !compiled_forward_backend_available() {
            eprintln!("skipping (no Metal backend)");
            return;
        }
        let mut inner = mtp_ready_inner();
        let h = inner.config.hidden_size as usize;
        let prompt: Vec<u32> = vec![1, 5, 9, 3];
        let p = greedy_params();
        let stream = Stream::new(DeviceType::Gpu);
        let committed_ids: Vec<u32> = vec![13, 21];

        // Same inner (same weights) run twice: once WITH a rejected draft
        // before the commit, once without. The committed history — and so
        // the next draft — must be identical.
        let mut run = |with_draft: bool| -> Vec<f32> {
            inner.reset_caches_internal();
            let _y = prefill_and_seed_mtp(&mut inner, &prompt, stream, &p).expect("seed");
            let setup = MtpTurnSetup {
                prompt_hidden: None,
                prompt_hidden_ids: None,
                prompt_hidden_position_base: 0,
                first_sampled_token: 7,
            };
            let mut step = inner.begin_mtp_decode(&setup).expect("stepper");
            assert_eq!(step.committed_len(), prompt.len() as i32);
            assert_eq!(step.draft_kv_offset(), prompt.len() as i32);

            step.begin_cycle(false);
            assert_eq!(
                step.draft_kv_offset(),
                prompt.len() as i32,
                "a Step-A cycle anchors at committed_len"
            );

            let seed_h = det_rows(h, 1, 5.0);
            if with_draft {
                let prev_e = det_rows(h, 1, 17.0);
                step.draft_step(&seed_h, &prev_e).expect("draft");
                assert_eq!(
                    step.draft_kv_offset(),
                    prompt.len() as i32 + 1,
                    "the draft writes its own K/V"
                );
            }

            let verify_hiddens = det_rows(h, 2, 33.0);
            let emb = step.embedding().clone();
            step.commit_mtp(
                crate::models::qwen3_5::mtp_decode::MtpCommitAnchor::IncludeAnchor,
                &seed_h,
                &verify_hiddens,
                &committed_ids,
                1,
                &emb,
            )
            .expect("commit");
            let m = committed_ids.len() as i32;
            assert_eq!(
                step.committed_len(),
                prompt.len() as i32 + m,
                "commit advances the committed cursor by M"
            );
            assert_eq!(
                step.draft_kv_offset(),
                prompt.len() as i32 + m,
                "commit must TRIM the rejected draft before writing, not append past it"
            );

            // Probe the post-commit history.
            let (probe_h, probe_e) = (det_rows(h, 1, 77.0), det_rows(h, 1, 91.0));
            step.draft_step(&probe_h, &probe_e)
                .expect("probe draft")
                .0
                .to_float32()
                .unwrap()
                .to_vec()
        };

        let with_reject = run(true);
        let clean = run(false);
        let d = max_abs_diff(&with_reject, &clean);
        assert!(
            d <= 5e-3,
            "a rejected draft leaked into the committed history: max |diff| = {d}"
        );
    }

    /// The drafter cache rewinds WITH the main caches on rejection: a new
    /// cycle re-anchors at the committed cursor, dropping the previous
    /// cycle's speculative tail. Chained cycles anchor one slot lower,
    /// because their draft pair `(h_{p+K}, emb(t_{p+K+1}))` targets the slot
    /// the previous commit already wrote.
    ///
    /// MUTATION: a no-op `begin_cycle` (the pre-port stateless drafter), or
    /// one that resets to a FRESH cache instead of trimming, or one that
    /// ignores `chained_anchor`. `commit_mtp`'s own trim hides all three from
    /// the commit-side test, so this is the gate that sees them.
    #[test]
    fn mtp_begin_cycle_rewinds_the_draft_tail() {
        if !compiled_forward_backend_available() {
            eprintln!("skipping (no Metal backend)");
            return;
        }
        let mut inner = mtp_ready_inner();
        let h = inner.config.hidden_size as usize;
        let prompt: Vec<u32> = vec![1, 5, 9, 3];
        let p = greedy_params();
        let stream = Stream::new(DeviceType::Gpu);
        let t = prompt.len() as i32;

        inner.reset_caches_internal();
        let _y = prefill_and_seed_mtp(&mut inner, &prompt, stream, &p).expect("seed");
        let setup = MtpTurnSetup {
            prompt_hidden: None,
            prompt_hidden_ids: None,
            prompt_hidden_position_base: 0,
            first_sampled_token: 7,
        };
        let mut step = inner.begin_mtp_decode(&setup).expect("stepper");
        assert_eq!(step.draft_kv_offset(), t);

        step.begin_cycle(false);
        assert_eq!(step.draft_kv_offset(), t, "Step-A anchor == committed_len");
        step.draft_step(&det_rows(h, 1, 5.0), &det_rows(h, 1, 17.0))
            .expect("draft");
        assert_eq!(step.draft_kv_offset(), t + 1, "the draft wrote a slot");

        // Next cycle: the speculative slot is dropped, not kept.
        step.begin_cycle(false);
        assert_eq!(
            step.draft_kv_offset(),
            t,
            "a rejected draft's K/V must be rewound before the next cycle"
        );
        assert_eq!(step.committed_len(), t, "trim must not move committed_len");

        // Chained cycles re-anchor one slot lower and overwrite the last
        // committed pair with the identical pair. This is verbatim the
        // engine's call (mtp_turn.rs:1778): `committed_history_active()`
        // returning `false` (its pre-port value) would anchor a chained
        // cycle at `committed_len` and drift the drafter cursor +1 per
        // cycle, on top of re-committing the anchor at mtp_turn.rs:1826.
        let chained_anchor = true && step.committed_history_active();
        assert!(
            chained_anchor,
            "the drafter carries a real committed history now"
        );
        step.begin_cycle(chained_anchor);
        assert_eq!(
            step.draft_kv_offset(),
            t - 1,
            "a chained cycle anchors at committed_len - 1"
        );
        assert_eq!(step.committed_len(), t);
    }

    /// T6 — `rollback_unemitted` rewinds the drafter by CURSOR, with no
    /// snapshot, on the one path where a snapshot is guaranteed to be gone.
    ///
    /// MUTATION: any implementation that reaches for `self.snap` — `rollback`
    /// nulls it on the `accepted_drafts == depth` branch, and depth is
    /// clamped to 1, so EVERY strandable cycle is a full-accept cycle.
    #[test]
    fn mtp_rollback_unemitted_needs_no_snapshot() {
        if !compiled_forward_backend_available() {
            eprintln!("skipping (no Metal backend)");
            return;
        }
        let mut inner = mtp_ready_inner();
        let h = inner.config.hidden_size as usize;
        let prompt: Vec<u32> = vec![1, 5, 9, 3];
        let p = greedy_params();
        let stream = Stream::new(DeviceType::Gpu);

        inner.reset_caches_internal();
        let _y = prefill_and_seed_mtp(&mut inner, &prompt, stream, &p).expect("seed");
        let setup = MtpTurnSetup {
            prompt_hidden: None,
            prompt_hidden_ids: None,
            prompt_hidden_position_base: 0,
            first_sampled_token: 7,
        };
        let mut step = inner.begin_mtp_decode(&setup).expect("stepper");

        // One full cycle: draft, then commit two tokens.
        step.begin_cycle(false);
        let seed_h = det_rows(h, 1, 5.0);
        step.draft_step(&seed_h, &det_rows(h, 1, 17.0))
            .expect("draft");
        let emb = step.embedding().clone();
        step.commit_mtp(
            crate::models::qwen3_5::mtp_decode::MtpCommitAnchor::IncludeAnchor,
            &seed_h,
            &det_rows(h, 2, 33.0),
            &[13u32, 21u32],
            1,
            &emb,
        )
        .expect("commit");
        let before = step.committed_len();
        assert_eq!(before, prompt.len() as i32 + 2);

        // Full-accept rollback NULLS the snapshot — the exact state a
        // snapshot-based unemitted rewind would silently no-op on.
        step.snapshot_main_linear();
        step.rollback(1, 1);
        step.rollback_unemitted(1);
        assert_eq!(
            step.committed_len(),
            before - 1,
            "an unemitted token must leave the committed cursor"
        );
        assert_eq!(
            step.draft_kv_offset(),
            before - 1,
            "the drafter KV must follow the committed cursor"
        );
        assert!(step.into_desynced(), "the desync latch must be set");
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
    ///
    /// `#[ignore]` is load-bearing: without it this test early-returns on an
    /// unset env var and reports "ok", so a skip reads as a pass in the
    /// headline `cargo test -p mlx-core --lib` count.
    #[ignore = "needs MLX_TEST_NEMOTRON_H_MODEL_PATH pointing to a real NemotronH checkpoint WITH an MTP head"]
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
        let chat_cfg = ChatConfig {
            temperature: Some(0.0),
            max_new_tokens: Some(n as i32),
            enable_mtp: Some(true),
            ..ChatConfig::default()
        };
        let p = extract_chat_params(&chat_cfg);
        let y = prefill_and_seed_mtp(&mut inner, &prompt, stream, &p).expect("prefill + seed");
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
        assert_eq!(
            outcome.desynced,
            outcome.rollback_unemitted > 0,
            "desync latch must fire exactly when a mid-cycle stop left tokens unemitted"
        );

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

        // DRAFT QUALITY. Structure can be wired correctly and still draft
        // from the wrong history (wrong shift direction, seed skipped, the
        // head attending someone else's KV) — every such bug passes the
        // lossless gate above, because a rejected draft is simply replaced
        // by the target argmax. Only acceptance can see it. Measured with
        // the pre-port read-only head: ~0.4/cycle, BELOW the 0.6 break-even
        // where drafting stops paying for itself.
        let (mean, _, cycles) = profiler
            .mtp_acceptance_summary()
            .expect("the real-checkpoint gate must run at least one MTP cycle");
        assert!(
            cycles >= 5,
            "too few MTP cycles to judge acceptance: {cycles}"
        );
        assert!(
            mean >= 0.6,
            "MTP acceptance {mean} over {cycles} cycles is below the 0.6 break-even"
        );
    }
}
