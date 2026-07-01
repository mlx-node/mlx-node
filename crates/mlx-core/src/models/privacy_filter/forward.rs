//! End-to-end forward pass for the OpenAI Privacy Filter model.
//!
//! Wires the 8 transformer [`Block`]s together with the input embedding,
//! the final RMSNorm, and the classifier head to produce per-token
//! logits ready for Viterbi decoding.
//!
//! ```text
//!   input_ids [B, T]
//!         │
//!         ├── embed_tokens[input_ids]   → hidden [B, T, 640]
//!         │
//!         ├── for layer in 0..8:
//!         │     Block { input_layernorm → AttentionLayer
//!         │             post_attention_layernorm → GptOssMlp }
//!         │
//!         ├── final RMSNorm (model.norm)
//!         └── classifier head (score.weight / score.bias)
//!                                       → logits [B, T, 33]
//! ```
//!
//! Shared per-model state (YaRN frequencies) is computed once at load
//! time and threaded through every block as a `&MxArray`. The runtime
//! call sequence is identical to mlx-lm's reference forward pass — see
//! `mlx-lm/mlx_lm/models/gpt_oss.py` — modulo bidirectional banded
//! attention (no KV cache) and the per-token classifier output instead
//! of an LM head.

use crate::array::MxArray;
use crate::array::banded_attention::build_band_mask;
use crate::nn::RMSNorm;
use napi::bindgen_prelude::*;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::path::Path;

use super::classifier::classifier_forward;
use super::persistence::{LoadedModel, load_from_directory};
use super::transformer::Block;
use super::yarn::compute_yarn_freqs;

/// A loaded privacy-filter model plus its precomputed YaRN frequencies.
///
/// The frequencies depend only on `head_dim` and `rope_parameters` so we
/// build them once at load time and share the resulting `[head_dim/2]`
/// f32 array across all 8 attention layers.
pub struct PrivacyFilterModel {
    pub loaded: LoadedModel,
    /// Shared YaRN frequencies (`[head_dim / 2]`, f32).
    pub yarn_freqs: MxArray,
}

impl PrivacyFilterModel {
    /// Load a privacy-filter checkpoint from a directory and precompute
    /// the shared YaRN frequencies.
    pub fn load_from_dir(path: &Path) -> Result<Self> {
        let loaded = load_from_directory(path)?;
        let yarn_freqs =
            compute_yarn_freqs(loaded.config.head_dim, &loaded.config.rope_parameters)?;
        Ok(Self { loaded, yarn_freqs })
    }

    /// Run the full forward pass and return per-token logits.
    ///
    /// - `input_ids` shape `[B, T]`, dtype int32 / int64 / uint32 (anything
    ///   `mx.take` accepts as an integer index).
    /// - Output shape `[B, T, num_classes]` (33 for the shipped
    ///   privacy-filter checkpoint).
    /// - Output dtype is the model's native weight dtype (bf16 for the
    ///   shipped checkpoint); callers wanting f32 should `astype` the
    ///   result themselves.
    pub fn forward_logits(&self, input_ids: &MxArray) -> Result<MxArray> {
        let weights = &self.loaded.weights;
        let cfg = &self.loaded.config;

        // 1. Embedding lookup: hidden = embed_tokens[input_ids].
        //    `take(axis=0)` on `[vocab, hidden]` with `[B, T]` indices
        //    produces `[B, T, hidden]`. Same pattern as
        //    `crate::nn::Embedding::forward`.
        let mut hidden = weights.embed_tokens.take(input_ids, 0)?;
        let seq_len = hidden.shape_at(1)?;

        // 2. Run all 8 transformer blocks. Each block needs to know
        //    whether to apply the sliding band or run full bidirectional
        //    attention — gpt-oss alternates by default. `band_for_layer`
        //    only ever returns a handful of distinct values (2 for the
        //    default alternation), so the `[T, T]` band mask is built
        //    once per distinct value here and reused across every layer
        //    that shares it, instead of being rebuilt from scratch on
        //    each of the 8 layers.
        let mut band_mask_cache: HashMap<i32, MxArray> = HashMap::new();
        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            let band = cfg.band_for_layer(layer_idx);
            if let Entry::Vacant(entry) = band_mask_cache.entry(band) {
                entry.insert(build_band_mask(seq_len, band)?);
            }
            let band_mask = band_mask_cache
                .get(&band)
                .expect("band_mask inserted above");

            let block = Block {
                weights: layer,
                config: cfg,
                yarn_freqs: &self.yarn_freqs,
                band_mask,
            };
            hidden = block.forward(&hidden)?;
        }

        // 3. Final RMSNorm (model.norm).
        let final_norm = RMSNorm::from_weight(&weights.final_norm, Some(cfg.rms_norm_eps as f64))?;
        let hidden = final_norm.forward(&hidden)?;

        // 4. Classifier head — `score.weight` `[33, hidden]`,
        //    `score.bias` `[33]`. See [`classifier_forward`].
        classifier_forward(&hidden, &weights.score_weight, &weights.score_bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::DType;

    fn checkpoint_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(".cache/models/privacy-filter")
    }

    /// Shape + finiteness check on a tiny real-tokenized input.
    /// Guards against accidental shape regressions and any NaN/Inf
    /// escaping the 8-block pipeline.
    #[test]
    #[ignore = "requires .cache/models/privacy-filter — run with --ignored"]
    fn forward_logits_shape_and_finite() {
        let model = PrivacyFilterModel::load_from_dir(&checkpoint_dir()).expect("load model");

        let text = "Hi I am Alice.";
        let ids_u32 = model
            .loaded
            .tokenizer
            .encode_sync(text, Some(false))
            .expect("encode");
        let ids: Vec<i32> = ids_u32.iter().map(|&id| id as i32).collect();
        let t = ids.len() as i64;
        let input_ids = MxArray::from_int32(&ids, &[1, t]).expect("from_int32");

        let logits = model.forward_logits(&input_ids).expect("forward");
        let shape: Vec<i64> = logits.shape().unwrap().to_vec();
        assert_eq!(shape, vec![1, t, model.loaded.label_strs.len() as i64]);

        // Finiteness check via GPU reduction (promote to f32 first so
        // bf16's wider exponent range doesn't paper over an actual Inf).
        let logits_f32 = logits.astype(DType::Float32).expect("astype");
        assert!(
            !logits_f32.has_nan_or_inf().expect("has_nan_or_inf"),
            "forward_logits produced NaN or Inf"
        );
    }

    /// Tokenize `text`, run the forward pass through `model`, and
    /// return per-token (string, argmax_tag) pairs. Shared by the
    /// canonical Alice helper and the multi-fixture cross-mode parity
    /// test so every checkpoint sees byte-for-byte identical
    /// tokenization + decoding.
    fn argmax_tags_for_text(model: &PrivacyFilterModel, text: &str) -> (Vec<String>, Vec<String>) {
        let ids_u32 = model
            .loaded
            .tokenizer
            .encode_sync(text, Some(false))
            .expect("encode");
        let ids: Vec<i32> = ids_u32.iter().map(|&id| id as i32).collect();
        let t = ids.len() as i64;
        let input_ids = MxArray::from_int32(&ids, &[1, t]).expect("from_int32");
        let logits = model.forward_logits(&input_ids).expect("forward");
        let pred = logits.argmax(-1, Some(false)).expect("argmax");
        let pred_i32 = pred.to_int32().expect("to_int32");
        let tok_strs: Vec<String> = ids_u32
            .iter()
            .map(|&id| {
                model
                    .loaded
                    .tokenizer
                    .decode_sync(&[id], false)
                    .unwrap_or_default()
            })
            .collect();
        let pred_tags: Vec<String> = pred_i32
            .iter()
            .map(|&i| model.loaded.label_strs[i as usize].clone())
            .collect();
        (tok_strs, pred_tags)
    }

    /// Run the same `forward_classify_alice_smith_email` body against
    /// an arbitrary checkpoint directory and return the (tokens, tags)
    /// pair. Shared by the bf16 reference and quantized variants so
    /// they all use byte-for-byte identical tokenization + decoding.
    fn run_classify_alice_smith_email(ckpt: &std::path::Path) -> (Vec<String>, Vec<String>) {
        let model = PrivacyFilterModel::load_from_dir(ckpt).expect("load model");
        argmax_tags_for_text(&model, "Hi I am Alice Smith, email alice@example.com")
    }

    /// The named-entity positions in the canonical test sentence.
    /// Quantized variants are allowed per-token confidence boundary
    /// noise OUTSIDE this set but must label these positions exactly
    /// as the bf16 reference does (per the Phase C correctness gate).
    const ALICE_NER_POSITIONS: &[(usize, &str)] = &[
        (3, "B-private_person"),
        (4, "E-private_person"),
        (7, "B-private_email"),
        (8, "I-private_email"),
        (9, "E-private_email"),
    ];

    /// End-to-end correctness: classify a PII-laden sentence and verify
    /// that the predicted tag sequence contains the expected entity
    /// starts (`B-private_person` for the name, `B-private_email` for
    /// the email).
    ///
    /// On failure the test prints the actual tokens and tags so the
    /// caller can diagnose whether the bug lives in YaRN, attention
    /// sinks, MoE routing, etc. **Do not loosen this assertion** — a
    /// failure here is a real algorithmic bug.
    #[test]
    #[ignore = "requires .cache/models/privacy-filter — run with --ignored"]
    fn forward_classify_alice_smith_email() {
        let model = PrivacyFilterModel::load_from_dir(&checkpoint_dir()).expect("load model");

        let text = "Hi I am Alice Smith, email alice@example.com";
        let ids_u32 = model
            .loaded
            .tokenizer
            .encode_sync(text, Some(false))
            .expect("encode");
        let ids: Vec<i32> = ids_u32.iter().map(|&id| id as i32).collect();
        let t = ids.len() as i64;
        let input_ids = MxArray::from_int32(&ids, &[1, t]).expect("from_int32");

        let logits = model.forward_logits(&input_ids).expect("forward");

        // argmax over the last axis. `argmax(axis=-1)` reduces the
        // `num_classes` axis, leaving `[1, T]` int indices.
        let pred = logits.argmax(-1, Some(false)).expect("argmax");
        let pred_i32 = pred.to_int32().expect("to_int32");

        // Decode each token id back to a string for the debug print.
        let tok_strs: Vec<String> = ids_u32
            .iter()
            .map(|&id| {
                model
                    .loaded
                    .tokenizer
                    .decode_sync(&[id], false)
                    .unwrap_or_default()
            })
            .collect();

        let pred_tags: Vec<&str> = pred_i32
            .iter()
            .map(|&i| model.loaded.label_strs[i as usize].as_str())
            .collect();

        println!("Tokens: {:?}", tok_strs);
        println!("Tags:   {:?}", pred_tags);

        assert!(
            pred_tags.contains(&"B-private_person"),
            "expected B-private_person in tag sequence; got {:?}",
            pred_tags
        );
        assert!(
            pred_tags.contains(&"B-private_email"),
            "expected B-private_email in tag sequence; got {:?}",
            pred_tags
        );
    }

    /// Run the canonical PII sentence through a quantized variant and
    /// assert the named-entity positions still match the bf16
    /// reference. Per-token confidence boundary noise on
    /// background-token positions is allowed; label flips at the
    /// NER positions are not.
    fn assert_quantized_ner_matches(variant: &str) {
        let ckpt = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(format!(".cache/models/privacy-filter-{variant}"));
        if !ckpt.exists() {
            eprintln!(
                "skipping quantized {variant} test — checkpoint not present at {}",
                ckpt.display()
            );
            return;
        }
        let (tokens, tags) = run_classify_alice_smith_email(&ckpt);
        println!("[{variant}] Tokens: {:?}", tokens);
        println!("[{variant}] Tags:   {:?}", tags);
        for &(pos, expected) in ALICE_NER_POSITIONS {
            assert_eq!(
                tags.get(pos).map(String::as_str),
                Some(expected),
                "[{variant}] tag at NER position {pos} flipped: expected {expected}, got {:?}",
                tags.get(pos)
            );
        }
    }

    #[test]
    #[ignore = "requires .cache/models/privacy-filter-mxfp8 — run with --include-ignored"]
    fn forward_classify_alice_smith_email_mxfp8() {
        assert_quantized_ner_matches("mxfp8");
    }

    #[test]
    #[ignore = "requires .cache/models/privacy-filter-mxfp4 — run with --include-ignored"]
    fn forward_classify_alice_smith_email_mxfp4() {
        assert_quantized_ner_matches("mxfp4");
    }

    #[test]
    #[ignore = "requires .cache/models/privacy-filter-nvfp4 — run with --include-ignored"]
    fn forward_classify_alice_smith_email_nvfp4() {
        assert_quantized_ner_matches("nvfp4");
    }

    #[test]
    #[ignore = "requires .cache/models/privacy-filter-affine — run with --include-ignored"]
    fn forward_classify_alice_smith_email_affine() {
        assert_quantized_ner_matches("affine");
    }

    /// Five-fixture cross-mode parity sweep.
    ///
    /// The bf16 checkpoint produces the reference argmax tag sequence
    /// for each of five PII-laden inputs spanning all eight label
    /// classes (person, email, phone, address, date, url, account,
    /// secret). Every quantized variant is then loaded in turn and
    /// must produce **the same tag** at every position where the bf16
    /// reference predicted a non-`O` tag. Per-token boundary noise on
    /// background positions is allowed; label flips at entity
    /// positions are not.
    ///
    /// This is a stronger version of the per-mode single-input
    /// `forward_classify_alice_smith_email_<mode>` tests above. It
    /// runs five inputs and compares the argmax decision (not Viterbi
    /// output) because raw argmax is the simpler invariant to verify
    /// — the decoder operates downstream on whatever logits the
    /// quantized forward emits, so quantization-induced tag flips
    /// will already show up here.
    #[test]
    #[ignore = "requires .cache/models/privacy-filter-{mxfp4,mxfp8,nvfp4,affine} — run with --include-ignored"]
    fn parity_across_quantized_modes() {
        const INPUTS: &[&str] = &[
            "Hi I am Alice Smith, email alice@example.com",
            "Call me at +1 555 123 4567 anytime",
            "Ship the package to 742 Evergreen Terrace, Springfield",
            "Born 1990-03-14, profile at https://example.org/profile/jdoe",
            "Account 1234567890; api key sk-ZZZZZZZZZZZZZZZZZZZZZZZZZZZZ",
        ];

        let cache_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(".cache/models");

        // Reference pass: bf16 argmax tags for each input.
        let bf16_dir = cache_dir.join("privacy-filter");
        if !bf16_dir.exists() {
            eprintln!(
                "skipping cross-mode parity — bf16 checkpoint missing at {}",
                bf16_dir.display()
            );
            return;
        }
        let bf16 = PrivacyFilterModel::load_from_dir(&bf16_dir).expect("load bf16");
        let mut bf16_tags: Vec<Vec<String>> = Vec::with_capacity(INPUTS.len());
        for &text in INPUTS {
            let (_tokens, tags) = argmax_tags_for_text(&bf16, text);
            bf16_tags.push(tags);
        }
        // Drop the bf16 model to free its weights before loading the
        // next checkpoint — keeps peak footprint bounded on a 36GB box.
        drop(bf16);

        let variants = ["mxfp8", "mxfp4", "nvfp4", "affine"];
        // Collect all divergences across every (variant, input) pair so a
        // single test failure reports the full picture instead of bailing
        // on the first mode that flips. Each report line stringifies one
        // (variant, input_idx, position, token, bf16_tag, quant_tag).
        let mut all_failures: Vec<String> = Vec::new();
        for variant in variants {
            let ckpt = cache_dir.join(format!("privacy-filter-{variant}"));
            if !ckpt.exists() {
                eprintln!(
                    "skipping cross-mode parity for {variant} — checkpoint missing at {}",
                    ckpt.display()
                );
                continue;
            }
            let model = PrivacyFilterModel::load_from_dir(&ckpt)
                .unwrap_or_else(|e| panic!("load {variant}: {e:?}"));
            for (i, &text) in INPUTS.iter().enumerate() {
                let (tokens, tags) = argmax_tags_for_text(&model, text);
                let mut mismatches: Vec<(usize, &str, &str, &str)> = Vec::new();
                for (j, (b, q)) in bf16_tags[i].iter().zip(tags.iter()).enumerate() {
                    if b != "O" && b != q {
                        let tok = tokens.get(j).map(String::as_str).unwrap_or("");
                        mismatches.push((j, tok, b.as_str(), q.as_str()));
                    }
                }
                if !mismatches.is_empty() {
                    all_failures.push(format!(
                        "[{variant}] input[{i}]={text:?}: entity-position tag flips vs bf16: \
                         {mismatches:?}\n  bf16 = {:?}\n  {variant} = {:?}",
                        bf16_tags[i], tags,
                    ));
                }
            }
        }
        assert!(
            all_failures.is_empty(),
            "cross-mode parity divergences found:\n{}",
            all_failures.join("\n\n"),
        );
    }
}
