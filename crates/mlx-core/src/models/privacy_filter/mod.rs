//! OpenAI Privacy Filter — token-classification PII detector.
pub mod attention;
pub mod classifier;
pub mod config;
pub mod experts;
pub mod forward;
pub mod persistence;
pub mod quantized_linear;
pub mod spans;
pub mod transformer;
pub mod viterbi;
pub mod yarn;
pub use attention::AttentionLayer;
pub use classifier::classifier_forward;
pub use config::{PrivacyFilterConfig, RopeParameters};
pub use experts::GptOssMlp;
pub use forward::PrivacyFilterModel;
pub use persistence::{
    AttnWeights, LayerWeights, LoadedModel, LoadedRouter, MlpWeights, ModelWeights,
    QuantizationConfig, load_from_directory,
};
pub use quantized_linear::{
    LoadedProj, PrivacyFilterQuantizedSwitchLinear, QuantizedLinear, TensorQuantParams, project_2d,
    project_moe,
};
pub use spans::{Entity, extract_spans};
pub use transformer::Block;
pub use viterbi::{Calibration, build_transition_matrix, label_id, viterbi_decode};
pub use yarn::compute_yarn_freqs;

use crate::array::{DType, MxArray};
use crate::nn::Activations;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::path::Path;

/// A privacy entity detected by [`PrivacyFilterModelJs::classify`].
///
/// `start`/`end` are byte offsets into the input string (Hugging Face
/// `tokenizers` convention). `label` is the privacy class without the
/// BIOES prefix (e.g. `"private_email"`). `score` is the mean — across
/// the span's tokens — of the softmax probability of the Viterbi-emitted
/// tag at each token.
#[napi(object)]
pub struct PrivacyEntity {
    pub label: String,
    pub start: u32,
    pub end: u32,
    pub score: f64,
    pub text: String,
}

/// Per-token output emitted when [`PrivacyClassifyOptions::return_tokens`]
/// is `true`. `tag` is the full BIOES tag (`"O"` or `"B-..."`/`"I-..."`/
/// `"E-..."`/`"S-..."`) chosen by the Viterbi decoder. `score` is the
/// softmax probability of that emitted tag at this token, so `tag` and
/// `score` always share decoders (at boundary tokens the Viterbi tag can
/// differ from the local argmax).
#[napi(object)]
pub struct PrivacyToken {
    pub text: String,
    pub tag: String,
    pub score: f64,
    pub start: u32,
    pub end: u32,
}

/// Per-call Viterbi calibration overrides.
///
/// Any field set to `Some(_)` overrides the corresponding bias from the
/// model's default calibration (loaded from `viterbi_calibration.json`
/// at load time). Missing fields fall back to the default.
#[napi(object)]
pub struct PrivacyCalibration {
    pub transition_bias_background_stay: Option<f64>,
    pub transition_bias_background_to_start: Option<f64>,
    pub transition_bias_end_to_background: Option<f64>,
    pub transition_bias_end_to_start: Option<f64>,
    pub transition_bias_inside_to_continue: Option<f64>,
    pub transition_bias_inside_to_end: Option<f64>,
}

/// Options for [`PrivacyFilterModelJs::classify`].
///
/// - `threshold` (default `0.5`): minimum mean per-token probability for
///   an extracted span to be returned.
/// - `calibration`: per-call overrides on top of the checkpoint default.
/// - `return_tokens` (default `false`): when `true`, the result includes
///   a `tokens` array with one entry per input token.
#[napi(object)]
pub struct PrivacyClassifyOptions {
    pub threshold: Option<f64>,
    pub calibration: Option<PrivacyCalibration>,
    pub return_tokens: Option<bool>,
}

/// Result of [`PrivacyFilterModelJs::classify`].
#[napi(object)]
pub struct PrivacyClassifyResult {
    pub entities: Vec<PrivacyEntity>,
    pub tokens: Option<Vec<PrivacyToken>>,
}

/// NAPI-exported view of [`PrivacyFilterModel`].
///
/// Construct via [`PrivacyFilterModelJs::load`] and run end-to-end
/// classification via [`PrivacyFilterModelJs::classify`].
#[napi(js_name = "PrivacyFilterModel")]
pub struct PrivacyFilterModelJs {
    inner: PrivacyFilterModel,
}

#[napi]
impl PrivacyFilterModelJs {
    /// Load a privacy-filter checkpoint from a directory.
    ///
    /// The directory must contain `config.json`, `model.safetensors`,
    /// `tokenizer.json`, and optionally `viterbi_calibration.json` and
    /// `tokenizer_config.json`. Synchronous to match the loader pattern
    /// used by every other model in this crate (e.g. `TextDetModel`).
    #[napi(factory)]
    pub fn load(model_path: String) -> Result<Self> {
        let inner = PrivacyFilterModel::load_from_dir(Path::new(&model_path))?;
        Ok(Self { inner })
    }

    /// Classify `text` and return detected PII entities (and optionally
    /// per-token tags).
    ///
    /// Pipeline:
    /// 1. Tokenize the text with byte offsets, no special tokens.
    /// 2. Run the forward pass to get `[1, T, 33]` logits.
    /// 3. Compute softmax (for per-tag confidences) and log-softmax (for
    ///    Viterbi emissions) over the class axis.
    /// 4. Build the transition matrix from the default calibration
    ///    merged with any per-call overrides.
    /// 5. Viterbi-decode using log-softmax emissions to get the BIOES
    ///    tag sequence.
    /// 6. For each token, take the softmax probability of the
    ///    Viterbi-emitted tag (so per-token `tag` and `score` share
    ///    decoders), then walk the tags + offsets + those probabilities
    ///    to extract coherent spans whose mean probability clears
    ///    `threshold`.
    #[napi]
    pub fn classify(
        &self,
        text: String,
        opts: Option<PrivacyClassifyOptions>,
    ) -> Result<PrivacyClassifyResult> {
        let opts = opts.unwrap_or(PrivacyClassifyOptions {
            threshold: None,
            calibration: None,
            return_tokens: None,
        });
        let threshold = opts.threshold.unwrap_or(0.5) as f32;
        let return_tokens = opts.return_tokens.unwrap_or(false);

        // ---- 1. Tokenize with byte offsets (no special tokens). ----
        //
        // `add_special_tokens=false` matches the HF reference call in
        // `forward.rs`'s test and the gpt-oss / o200k expectation that
        // the token-classification head is fed the raw byte-pair tokens.
        let (ids_u32, offsets) = self
            .inner
            .loaded
            .tokenizer
            .encode_with_offsets_sync(&text, Some(false))?;
        let n_tokens = ids_u32.len();
        let num_classes = self.inner.loaded.label_strs.len();

        // Empty input → empty result. The forward pass would still work
        // on a [1, 0] tensor, but Viterbi would have nothing to do and
        // it's cleaner to short-circuit.
        if n_tokens == 0 {
            return Ok(PrivacyClassifyResult {
                entities: Vec::new(),
                tokens: if return_tokens {
                    Some(Vec::new())
                } else {
                    None
                },
            });
        }

        // ---- 2. Forward pass. ----
        let ids_i32: Vec<i32> = ids_u32.iter().map(|&id| id as i32).collect();
        let input_ids = MxArray::from_int32(&ids_i32, &[1, n_tokens as i64])?;
        let logits = self.inner.forward_logits(&input_ids)?;

        // Promote to f32 so all downstream math has stable headroom.
        // The shipped checkpoint runs in bf16, which makes softmax /
        // log-softmax less precise than the HF reference (which forces
        // f32 internally for its own softmax — see `modeling_opf.py`).
        let logits_f32 = logits.astype(DType::Float32)?;

        // ---- 3. Softmax (for per-tag probabilities, indexed by the
        //         Viterbi pick below) and log-softmax (for Viterbi
        //         emissions). ----
        //
        // Why log-softmax for emissions: the Viterbi decoder adds
        // emission + transition scalars, and transition biases come in
        // additive log-space form (see `viterbi_calibration.json`).
        // Feeding probabilities or raw logits would break that algebra.
        // Log-softmax also gives us numerical stability for free.
        let probs = Activations::softmax(&logits_f32, Some(-1))?;
        let log_probs = Activations::log_softmax(&logits_f32, Some(-1))?;

        // Eval once before two GPU→CPU pulls so we don't replay the
        // forward pass twice.
        probs.eval();
        log_probs.eval();

        let probs_flat: Vec<f32> = probs.to_float32()?.to_vec();
        let log_probs_flat: Vec<f32> = log_probs.to_float32()?.to_vec();
        debug_assert_eq!(probs_flat.len(), n_tokens * num_classes);
        debug_assert_eq!(log_probs_flat.len(), n_tokens * num_classes);

        // Drop GPU-side intermediates. `probs.eval()` / `log_probs.eval()`
        // above already forced the whole forward-pass graph to complete
        // on the GPU before we pulled `probs_flat` / `log_probs_flat` to
        // the host, so these drops just release refcounts.
        drop(probs);
        drop(log_probs);
        drop(logits_f32);
        drop(logits);
        drop(input_ids);

        // Reclaim MLX's caching allocator — but on a cadence, not every
        // call. Without *any* clearing, repeated `classify` calls grow
        // the cache unboundedly (one classify allocates ~per-layer
        // hidden states, attention K/V, MLP gate/up/down for 8 layers +
        // the [1, T, 33] logits + softmax/log-softmax tensors). Clearing
        // on *every* call pairs a full GPU stall with an allocator wipe
        // on top of the actual forward pass, for every single
        // invocation — worse than the paged-decode loop's old,
        // already-rejected cadence=64 (see
        // `crate::array::memory::PAGED_DECODE_CACHE_CLEAR_INTERVAL_DEFAULT`).
        // See `maybe_clear_cache_for_privacy_filter_call`'s doc comment
        // for why no extra `synchronize()` is needed at this call site.
        crate::array::memory::maybe_clear_cache_for_privacy_filter_call();

        // ---- 4. Build the emission matrix [T, num_classes] from
        //         log-softmax. ----
        let mut emit: Vec<Vec<f32>> = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let row = log_probs_flat[t * num_classes..(t + 1) * num_classes].to_vec();
            emit.push(row);
        }

        // ---- 5. Merge default calibration with per-call overrides. ----
        let cal = merge_calibration(
            &self.inner.loaded.calibration_default,
            opts.calibration.as_ref(),
        );
        let transitions = build_transition_matrix(&self.inner.loaded.label_strs, &cal);

        // ---- 6. Viterbi decode. ----
        let tags = viterbi_decode(&emit, &transitions);

        // Per-token confidence aligned to the Viterbi-emitted tag.
        //
        // We deliberately index `probs_flat` by `tags[t]` (the Viterbi
        // pick) rather than by the local argmax. At boundary tokens the
        // Viterbi tag can differ from the argmax tag (transition biases
        // win), so reporting the argmax-class probability would mix
        // semantics: the `tag` would come from Viterbi while the `score`
        // would come from a different class. Using the Viterbi tag's
        // softmax probability keeps both the per-token output and the
        // span-mean score consistent with the emitted tag sequence.
        let mut per_token_tag_probs: Vec<f32> = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let tag_id = tags[t];
            per_token_tag_probs.push(probs_flat[t * num_classes + tag_id]);
        }

        // ---- 7. Extract spans. ----
        let id2label = self.inner.loaded.label_strs.clone();
        let entities_internal = extract_spans(
            &tags,
            &id2label,
            &per_token_tag_probs,
            &offsets,
            &text,
            threshold,
        );

        let entities: Vec<PrivacyEntity> = entities_internal
            .into_iter()
            .map(|e| PrivacyEntity {
                label: e.label,
                start: e.start as u32,
                end: e.end as u32,
                score: e.score as f64,
                text: e.text,
            })
            .collect();

        // ---- 8. Per-token output (optional). ----
        //
        // We report:
        // - `tag`:   the Viterbi-decoded BIOES tag (consistent with the
        //            sequence used to extract entities).
        // - `score`: the softmax probability of that Viterbi-emitted tag
        //            at this token — the same value that feeds into the
        //            span mean, so `tag` and `score` share semantics.
        // - `text`:  the byte slice from `source_text[start..end]`. For
        //            special tokens HF emits `(0, 0)`; we fall back to
        //            the tokenizer's `id_to_token` in that case so the
        //            output is never empty/ambiguous.
        let tokens = if return_tokens {
            let mut out = Vec::with_capacity(n_tokens);
            for t in 0..n_tokens {
                let (start, end) = offsets[t];
                let tok_text = if start == 0 && end == 0 {
                    // Special token or otherwise zero-width — fall back
                    // to the tokenizer's vocab string.
                    self.inner
                        .loaded
                        .tokenizer
                        .id_to_token(ids_u32[t])
                        .unwrap_or_default()
                } else {
                    text.get(start..end).unwrap_or("").to_string()
                };
                let tag_id = tags[t];
                out.push(PrivacyToken {
                    text: tok_text,
                    tag: id2label
                        .get(tag_id)
                        .cloned()
                        .unwrap_or_else(|| "O".to_string()),
                    score: per_token_tag_probs[t] as f64,
                    start: start as u32,
                    end: end as u32,
                });
            }
            Some(out)
        } else {
            None
        };

        Ok(PrivacyClassifyResult { entities, tokens })
    }
}

/// Merge per-call calibration overrides on top of the model's default
/// calibration. Any field left `None` in `overrides` is taken from
/// `default`.
fn merge_calibration(default: &Calibration, overrides: Option<&PrivacyCalibration>) -> Calibration {
    let Some(o) = overrides else {
        return *default;
    };
    Calibration {
        transition_bias_background_stay: o
            .transition_bias_background_stay
            .map(|v| v as f32)
            .unwrap_or(default.transition_bias_background_stay),
        transition_bias_background_to_start: o
            .transition_bias_background_to_start
            .map(|v| v as f32)
            .unwrap_or(default.transition_bias_background_to_start),
        transition_bias_end_to_background: o
            .transition_bias_end_to_background
            .map(|v| v as f32)
            .unwrap_or(default.transition_bias_end_to_background),
        transition_bias_end_to_start: o
            .transition_bias_end_to_start
            .map(|v| v as f32)
            .unwrap_or(default.transition_bias_end_to_start),
        transition_bias_inside_to_continue: o
            .transition_bias_inside_to_continue
            .map(|v| v as f32)
            .unwrap_or(default.transition_bias_inside_to_continue),
        transition_bias_inside_to_end: o
            .transition_bias_inside_to_end
            .map(|v| v as f32)
            .unwrap_or(default.transition_bias_inside_to_end),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_calibration_uses_default_when_no_overrides() {
        let default = Calibration {
            transition_bias_background_stay: 1.5,
            transition_bias_inside_to_end: -0.25,
            ..Calibration::default()
        };
        let merged = merge_calibration(&default, None);
        assert_eq!(merged.transition_bias_background_stay, 1.5);
        assert_eq!(merged.transition_bias_inside_to_end, -0.25);
    }

    #[test]
    fn merge_calibration_overrides_per_field() {
        let default = Calibration {
            transition_bias_background_stay: 1.5,
            transition_bias_inside_to_end: -0.25,
            ..Calibration::default()
        };
        let overrides = PrivacyCalibration {
            transition_bias_background_stay: Some(0.0),
            transition_bias_background_to_start: None,
            transition_bias_end_to_background: None,
            transition_bias_end_to_start: None,
            transition_bias_inside_to_continue: Some(2.5),
            transition_bias_inside_to_end: None,
        };
        let merged = merge_calibration(&default, Some(&overrides));
        // Overridden:
        assert_eq!(merged.transition_bias_background_stay, 0.0);
        assert_eq!(merged.transition_bias_inside_to_continue, 2.5);
        // Inherited from default:
        assert_eq!(merged.transition_bias_inside_to_end, -0.25);
        // Inherited zero-default:
        assert_eq!(merged.transition_bias_end_to_start, 0.0);
    }

    fn checkpoint_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(".cache/models/privacy-filter")
    }

    /// Soak test for the cadence-gated cache clear in `classify()`
    /// (`crate::array::memory::maybe_clear_cache_for_privacy_filter_call`).
    ///
    /// Runs `classify()` well past several cadence boundaries (default
    /// interval is 8; 50 calls crosses it 6 times) and asserts MLX's
    /// cache memory does not grow monotonically call-over-call — i.e.
    /// gating the clear to every Nth call still bounds steady-state
    /// footprint, it just amortizes the stall+wipe cost instead of
    /// paying it on every single call.
    #[test]
    #[ignore = "requires .cache/models/privacy-filter — run with --ignored"]
    fn classify_soak_cache_bounded() {
        use crate::array::memory::get_cache_memory;

        let inner = PrivacyFilterModel::load_from_dir(&checkpoint_dir()).expect("load model");
        let model = PrivacyFilterModelJs { inner };

        let text = "Hi I am Alice Smith, email alice@example.com";

        // Warm up so first-call-only costs (e.g. any one-time kernel
        // compilation) don't skew the "unbounded growth" comparison below.
        model
            .classify(text.to_string(), None)
            .expect("warmup classify");
        let cache_after_warmup = get_cache_memory();

        for _ in 0..50 {
            model.classify(text.to_string(), None).expect("classify");
        }
        let cache_after_soak = get_cache_memory();

        // Bound rather than assert exact equality: allocator fragmentation
        // can shift this a little run to run, but it must not scale with
        // the number of calls (50 calls at unconditional-clear-per-call
        // would stay ~flat too; this guards against a regression back to
        // "never clear" rather than proving the OLD unconditional-clear
        // behavior specifically).
        assert!(
            cache_after_soak <= cache_after_warmup * 3.0 + 16.0 * 1024.0 * 1024.0,
            "cache memory grew unboundedly over 50 classify() calls: \
             after_warmup={cache_after_warmup} after_soak={cache_after_soak}",
        );
    }
}
