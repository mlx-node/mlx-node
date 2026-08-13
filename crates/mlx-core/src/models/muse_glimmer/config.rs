//! Muse-Glimmer config parsing.
//!
//! Two layers: `Raw*` mirrors the on-disk `config.json` verbatim; the
//! `MuseGlimmer*Config` types are the validated, resolved form the runtime
//! uses. The resolution step is the point of this module — it turns the two
//! parallel per-layer arrays (`layer_types`, `layer_rope_theta`) into tables
//! that cannot disagree, and refuses configs where they do.

use napi::bindgen_prelude::*;
use serde::Deserialize;
use std::path::Path;

/// `qk_scale_factor` when the checkpoint omits it.
fn default_qk_scale_factor() -> f32 {
    3.87
}

/// `output_multiplier` when the checkpoint omits it. The reference config
/// records `0.19611613513818404`; written here at f32 precision because the
/// remaining digits do not survive the cast (and `clippy::excessive_precision`
/// rejects them).
fn default_output_multiplier() -> f32 {
    0.196_116_13
}

/// `post_norm_eps` when the checkpoint omits it. Deliberately NOT
/// `rms_norm_eps` (1e-5) — see [`MuseGlimmerTextConfig::post_norm_eps`].
fn default_post_norm_eps() -> f32 {
    1e-8
}

/// `final_logit_softcapping` when the checkpoint omits it.
fn default_final_logit_softcapping() -> f32 {
    20.0
}

/// `sliding_window` when the checkpoint omits it.
fn default_sliding_window() -> usize {
    2048
}

/// Per-layer attention span. The decoder repeats
/// `[Sliding, Sliding, Sliding, Full]` thirteen times over 52 layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Sliding,
    Full,
}

/// Raw on-disk `text_config`. Unknown keys (`bos_token_id`, `rope_parameters`,
/// `use_cache`, …) are ignored; only what the runtime needs is named.
#[derive(Debug, Clone, Deserialize)]
struct RawTextConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    #[serde(default = "default_sliding_window")]
    sliding_window: usize,
    #[serde(default = "default_post_norm_eps")]
    post_norm_eps: f32,
    #[serde(default = "default_qk_scale_factor")]
    qk_scale_factor: f32,
    #[serde(default = "default_output_multiplier")]
    output_multiplier: f32,
    #[serde(default = "default_final_logit_softcapping")]
    final_logit_softcapping: f32,
    #[serde(default)]
    tie_word_embeddings: bool,
    layer_types: Vec<String>,
    layer_rope_theta: Vec<f32>,
}

/// Raw on-disk top-level config.
#[derive(Debug, Clone, Deserialize)]
struct RawConfig {
    image_token_id: usize,
    video_token_id: usize,
    out_hidden_size: usize,
    projector_hidden_size: usize,
    projector_hidden_act: String,
    text_config: RawTextConfig,
    vision_config: MuseGlimmerVisionConfig,
}

/// Vision tower geometry.
///
/// Deserialized straight from `config.json` with no `Raw` counterpart: unlike
/// the text config, nothing here needs cross-field resolution. The tower's own
/// `layer_types` table (`window_attention` / `full_attention`) is deliberately
/// NOT read here — resolving and validating it belongs with the vision tower
/// that consumes it, and a `#[serde(default)]` empty table would be a silent
/// trap of exactly the kind this module exists to prevent.
#[derive(Debug, Clone, Deserialize)]
pub struct MuseGlimmerVisionConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub patch_size: usize,
    pub merge_size: usize,
    pub patch_temporal: usize,
    pub pos_emb_height: usize,
    pub pos_emb_width: usize,
    pub layer_norm_eps: f32,
}

/// Validated text-decoder config.
#[derive(Debug, Clone)]
pub struct MuseGlimmerTextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Independent of `hidden_size`: 32 x 128 = 4096 while `hidden_size` is
    /// 6656. Never derive one from the other.
    pub head_dim: usize,
    pub vocab_size: usize,
    pub sliding_window: usize,
    /// Epsilon for the input/pre-feedforward RMS norms.
    pub rms_norm_eps: f32,
    /// Epsilon for the POST norms — three orders of magnitude smaller than
    /// `rms_norm_eps`. Reusing `rms_norm_eps` here is a silent accuracy bug.
    pub post_norm_eps: f32,
    /// Extra multiplier on q, applied ON TOP OF 1/sqrt(head_dim). See
    /// [`MuseGlimmerTextConfig::effective_qk_scale`].
    pub qk_scale_factor: f32,
    pub output_multiplier: f32,
    pub final_logit_softcapping: f32,
    pub tie_word_embeddings: bool,
    /// One entry per layer, resolved from `layer_types`; length is guaranteed
    /// to equal `num_hidden_layers`.
    pub layer_kinds: Vec<LayerKind>,
    /// One entry per layer; `0.0` means NoPE. Prefer
    /// [`MuseGlimmerTextConfig::rope_theta_for`] over indexing this directly.
    ///
    /// Validation guarantees the biconditional
    /// `layer_kinds[i] == Full` **iff** `layer_rope_theta[i] == 0.0`, so
    /// `rope_theta_for(i).is_none()` is equivalent to
    /// `layer_kinds[i] == LayerKind::Full` for every in-range `i`.
    pub layer_rope_theta: Vec<f32>,
}

/// Validated Muse-Glimmer config.
///
/// Rust-internal by design: no `#[napi(object)]` surface here. The other
/// families expose napi configs with `i32`/`f64` fields because TypeScript
/// reads `config.json` itself and hands the object across the bridge
/// (`packages/lm/src/models/model-loader.ts`). This family resolves and
/// validates the file in Rust instead, so the resolved type keeps native
/// `usize`/`f32` and never crosses the bridge. The napi-facing loader options
/// are a separate concern owned by the model-loading milestone.
#[derive(Debug, Clone)]
pub struct MuseGlimmerConfig {
    pub image_token_id: usize,
    pub video_token_id: usize,
    pub out_hidden_size: usize,
    pub projector_hidden_size: usize,
    pub projector_hidden_act: String,
    pub text_config: MuseGlimmerTextConfig,
    pub vision_config: MuseGlimmerVisionConfig,
}

impl MuseGlimmerTextConfig {
    /// `None` means NoPE: no rotation is applied to q/k on this layer. A caller
    /// that substitutes theta = 1.0 applies an identity-ish rotation where the
    /// reference applies none, which is a silent correctness bug.
    pub fn rope_theta_for(&self, layer: usize) -> Option<f32> {
        match self.layer_rope_theta.get(layer) {
            Some(t) if *t != 0.0 => Some(*t),
            _ => None,
        }
    }

    /// Net multiplier on the q·k product: `qk_scale_factor` is applied to q ON TOP
    /// OF the standard 1/sqrt(head_dim) SDPA scale, not instead of it.
    pub fn effective_qk_scale(&self) -> f32 {
        self.qk_scale_factor * (self.head_dim as f32).powf(-0.5)
    }
}

fn parse_layer_kind(raw: &str) -> Result<LayerKind> {
    match raw {
        "sliding_attention" => Ok(LayerKind::Sliding),
        "full_attention" => Ok(LayerKind::Full),
        other => Err(Error::from_reason(format!(
            "muse_glimmer: unrecognized layer_types entry {other:?}; \
             expected \"sliding_attention\" or \"full_attention\""
        ))),
    }
}

impl MuseGlimmerConfig {
    /// Read and validate `<dir>/config.json`.
    pub fn from_path(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let json = std::fs::read_to_string(&path).map_err(|e| {
            Error::from_reason(format!(
                "muse_glimmer: failed to read {}: {e}",
                path.display()
            ))
        })?;
        Self::from_json_str(&json)
    }

    /// Validate a `config.json` body. Every check below fails closed on a
    /// documented trap of this architecture rather than resolving it silently.
    pub fn from_json_str(json: &str) -> Result<Self> {
        let raw: RawConfig = serde_json::from_str(json)
            .map_err(|e| Error::from_reason(format!("muse_glimmer: invalid config.json: {e}")))?;
        let text = raw.text_config;
        let layers = text.num_hidden_layers;

        // Resolve the layer kinds first so an unrecognized span names itself.
        let layer_kinds = text
            .layer_types
            .iter()
            .map(|s| parse_layer_kind(s))
            .collect::<Result<Vec<LayerKind>>>()?;

        // Both arity checks precede any per-layer indexing: a short table must
        // be reported as such, never surface as an out-of-bounds panic below.
        if layer_kinds.len() != layers {
            return Err(Error::from_reason(format!(
                "muse_glimmer: layer_types has {} entries but num_hidden_layers is {layers}",
                layer_kinds.len()
            )));
        }
        if text.layer_rope_theta.len() != layers {
            return Err(Error::from_reason(format!(
                "muse_glimmer: layer_rope_theta has {} entries but num_hidden_layers is {layers}",
                text.layer_rope_theta.len()
            )));
        }

        // In this architecture a layer is full_attention if and only if its theta
        // is 0 (NoPE). Either direction of disagreement means the two tables
        // describe different models, and guessing silently mis-rotates a whole
        // layer, so refuse the config. Both directions are checked and reported
        // separately, because the reader needs to know WHICH table is wrong.
        //
        // The `theta != 0` on a Full layer half matters most: it fails open. The
        // model loads, `rope_theta_for` hands back `Some(theta)`, and inference
        // stays numerically valid while rotating a layer the reference does not
        // rotate — fluent output that is quietly wrong.
        for (i, (kind, theta)) in layer_kinds
            .iter()
            .zip(text.layer_rope_theta.iter())
            .enumerate()
        {
            if *theta == 0.0 && *kind != LayerKind::Full {
                return Err(Error::from_reason(format!(
                    "muse_glimmer: layer {i} has layer_rope_theta 0 (NoPE) but layer_types says \
                     {kind:?}; NoPE is expected only on full_attention layers"
                )));
            }
            if *kind == LayerKind::Full && *theta != 0.0 {
                return Err(Error::from_reason(format!(
                    "muse_glimmer: layer {i} is full_attention with layer_rope_theta {theta}; \
                     the full_attention layers of this architecture are NoPE, so their theta \
                     must be 0 — a non-zero theta here would rotate a layer the reference \
                     leaves unrotated"
                )));
            }
        }

        // GQA's only real head-count invariant: the query heads must divide
        // evenly into kv groups (32 / 2 = 16 here). Note there is deliberately
        // NO `head_dim * num_attention_heads == hidden_size` check — that holds
        // in most families but not this one, and would reject the real
        // checkpoint (32 x 128 = 4096 vs hidden_size 6656).
        if text.num_key_value_heads == 0
            || !text
                .num_attention_heads
                .is_multiple_of(text.num_key_value_heads)
        {
            return Err(Error::from_reason(format!(
                "muse_glimmer: num_key_value_heads {} must be non-zero and divide \
                 num_attention_heads {}",
                text.num_key_value_heads, text.num_attention_heads
            )));
        }
        if text.head_dim == 0 {
            return Err(Error::from_reason(
                "muse_glimmer: head_dim must be non-zero",
            ));
        }

        // `sliding_window` leaves this module as the `AttentionKind::SlidingWindow`
        // payload of 39 of the 52 layers, so a bad value here mis-describes three
        // quarters of the decoder. Two traps, both of which fail OPEN without a
        // check here:
        //
        //   * 0 is not "no window" in the direction a reader expects. vLLM's
        //     disable sentinel is `sliding_window = None`
        //     (`vllm/config/model.py`, `disable_sliding_window`) and its
        //     truthiness checks read a literal 0 as "not a sliding layer", so a 0
        //     would silently promote every sliding layer to full attention —
        //     fluent output, wrong receptive field, no error anywhere. Note this
        //     0 is a DIFFERENT namespace from `layer_rope_theta`'s 0, which means
        //     NoPE; the same digit means unrelated things in the two tables and
        //     they must never be conflated.
        //   * a value past `u32::MAX` truncates on the way into the KV-cache
        //     seam, which carries the window as a `u32`. 5_000_000_000 would
        //     arrive as 705_032_704: a plausible-looking window nobody asked for.
        //
        // Validating here rather than at spec derivation is deliberate. The seam
        // does refuse a 0 (`KVCacheSpecError::InvalidSlidingWindow`), but only
        // once grouping runs and without naming the config field that carried it.
        if text.sliding_window == 0 {
            return Err(Error::from_reason(format!(
                "muse_glimmer: sliding_window must be non-zero, got {}; 39 of this \
                 decoder's layers are sliding_attention, and a 0 window would widen \
                 all of them to full attention instead of failing",
                text.sliding_window
            )));
        }
        if u32::try_from(text.sliding_window).is_err() {
            return Err(Error::from_reason(format!(
                "muse_glimmer: sliding_window {} does not fit in a u32; the KV cache \
                 seam carries the window as a u32 and a cast would truncate it",
                text.sliding_window
            )));
        }

        Ok(Self {
            image_token_id: raw.image_token_id,
            video_token_id: raw.video_token_id,
            out_hidden_size: raw.out_hidden_size,
            projector_hidden_size: raw.projector_hidden_size,
            projector_hidden_act: raw.projector_hidden_act,
            text_config: MuseGlimmerTextConfig {
                hidden_size: text.hidden_size,
                intermediate_size: text.intermediate_size,
                num_hidden_layers: layers,
                num_attention_heads: text.num_attention_heads,
                num_key_value_heads: text.num_key_value_heads,
                head_dim: text.head_dim,
                vocab_size: text.vocab_size,
                sliding_window: text.sliding_window,
                rms_norm_eps: text.rms_norm_eps,
                post_norm_eps: text.post_norm_eps,
                qk_scale_factor: text.qk_scale_factor,
                output_multiplier: text.output_multiplier,
                final_logit_softcapping: text.final_logit_softcapping,
                tie_word_embeddings: text.tie_word_embeddings,
                layer_kinds,
                layer_rope_theta: text.layer_rope_theta,
            },
            vision_config: raw.vision_config,
        })
    }
}

/// `config.json` bodies for tests.
///
/// Shared rather than private to `mod tests` so the sibling modules that
/// consume a validated config — [`super::kv_cache`] — derive their fixtures from
/// the SAME JSON this module's tests validate. Two hand-written copies of the
/// checkpoint shape would let one drift into describing a model the parser would
/// reject.
#[cfg(test)]
pub(crate) mod fixtures {
    /// The two parallel per-layer tables at full length, as JSON array bodies.
    pub(crate) fn layer_tables(layers: usize) -> (String, String) {
        let kinds: Vec<String> = (0..layers)
            .map(|i| {
                if (layers - 1 - i).is_multiple_of(4) {
                    "\"full_attention\"".to_string()
                } else {
                    "\"sliding_attention\"".to_string()
                }
            })
            .collect();
        let thetas: Vec<String> = (0..layers)
            .map(|i| {
                if (layers - 1 - i).is_multiple_of(4) {
                    "0".into()
                } else {
                    "500000.0".into()
                }
            })
            .collect();
        (kinds.join(","), thetas.join(","))
    }

    /// The real checkpoint's text_config, trimmed to the fields under test.
    /// layer_types / layer_rope_theta are given at full length by the helper.
    pub(crate) fn text_config_json(layers: usize) -> String {
        let (kinds, thetas) = layer_tables(layers);
        format!(
            r#"{{
              "model_type": "muse_glimmer_text",
              "hidden_size": 6656, "intermediate_size": 19968,
              "num_hidden_layers": {layers},
              "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
              "vocab_size": 202048, "sliding_window": 2048,
              "rms_norm_eps": 1e-5, "post_norm_eps": 1e-8,
              "qk_scale_factor": 3.87,
              "output_multiplier": 0.19611613513818404,
              "final_logit_softcapping": 20.0,
              "tie_word_embeddings": false,
              "layer_types": [{kinds}],
              "layer_rope_theta": [{thetas}]
            }}"#
        )
    }

    /// Only the fields the parser requires: every `#[serde(default)]` field is
    /// omitted so the documented defaults are what gets exercised.
    pub(crate) fn minimal_text_config_json(layers: usize) -> String {
        let (kinds, thetas) = layer_tables(layers);
        format!(
            r#"{{
              "model_type": "muse_glimmer_text",
              "hidden_size": 6656, "intermediate_size": 19968,
              "num_hidden_layers": {layers},
              "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
              "vocab_size": 202048,
              "rms_norm_eps": 1e-5,
              "layer_types": [{kinds}],
              "layer_rope_theta": [{thetas}]
            }}"#
        )
    }

    /// The brief's `write_config` body, returned as a JSON string rather than
    /// written to a temp dir: `tempfile` is not a dependency of this workspace,
    /// so the validating core takes `&str` and `from_path` is a thin I/O shim
    /// over it. Not one asserted value changed.
    pub(crate) fn config_json(text: &str) -> String {
        format!(
            r#"{{"model_type":"muse_glimmer","image_token_id":200092,
                 "video_token_id":200091,"out_hidden_size":6144,
                 "projector_hidden_size":4096,"projector_hidden_act":"gelu",
                 "text_config":{text},
                 "vision_config":{{"model_type":"muse_glimmer_vision",
                   "hidden_size":1536,"num_hidden_layers":50,
                   "intermediate_size":8960,"num_attention_heads":16,
                   "patch_size":14,"merge_size":2,"patch_temporal":2,
                   "pos_emb_height":32,"pos_emb_width":32,
                   "layer_norm_eps":1e-5}}}}"#
        )
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;

    fn parse(text: &str) -> Result<MuseGlimmerConfig> {
        MuseGlimmerConfig::from_json_str(&config_json(text))
    }

    #[test]
    fn parses_the_real_checkpoint_shape() {
        let cfg = parse(&text_config_json(52)).unwrap();
        let t = &cfg.text_config;
        assert_eq!(t.hidden_size, 6656);
        assert_eq!(t.intermediate_size, 19968);
        assert_eq!(t.num_hidden_layers, 52);
        assert_eq!(t.num_attention_heads, 32);
        assert_eq!(t.num_key_value_heads, 2);
        assert_eq!(t.head_dim, 128);
        assert_eq!(t.vocab_size, 202048);
        assert_eq!(t.sliding_window, 2048);
        assert_eq!(t.qk_scale_factor, 3.87);
        assert_eq!(t.final_logit_softcapping, 20.0);
        assert_eq!(cfg.image_token_id, 200092);
        assert_eq!(cfg.video_token_id, 200091);
        assert_eq!(cfg.out_hidden_size, 6144);
        assert_eq!(cfg.projector_hidden_size, 4096);
    }

    #[test]
    fn two_epsilons_are_distinct() {
        let t = parse(&text_config_json(52)).unwrap().text_config;
        assert_eq!(t.rms_norm_eps, 1e-5);
        assert_eq!(t.post_norm_eps, 1e-8);
        assert_ne!(
            t.rms_norm_eps, t.post_norm_eps,
            "the post norms use a different epsilon; collapsing them is a silent bug"
        );
    }

    #[test]
    fn full_attention_layers_are_exactly_every_fourth_counted_from_the_last() {
        let t = parse(&text_config_json(52)).unwrap().text_config;
        let full: Vec<usize> = (0..52)
            .filter(|&i| t.layer_kinds[i] == LayerKind::Full)
            .collect();
        assert_eq!(full, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51]);
        assert_eq!(full.len(), 13);
        assert_eq!(
            t.layer_kinds
                .iter()
                .filter(|k| **k == LayerKind::Sliding)
                .count(),
            39
        );
    }

    #[test]
    fn nope_is_on_exactly_the_full_layers_and_returns_none() {
        let t = parse(&text_config_json(52)).unwrap().text_config;
        for i in 0..52 {
            match t.layer_kinds[i] {
                LayerKind::Full => assert_eq!(
                    t.rope_theta_for(i),
                    None,
                    "layer {i} is full_attention and must be NoPE — theta 0 means NO \
                     rotation applied, never an identity rotation"
                ),
                LayerKind::Sliding => {
                    assert_eq!(t.rope_theta_for(i), Some(500_000.0), "layer {i}")
                }
            }
        }
        assert_eq!(t.layer_rope_theta.iter().filter(|v| **v == 0.0).count(), 13);
    }

    #[test]
    fn rejects_a_layer_types_length_mismatch() {
        // 52 layers declared, 51 entries supplied.
        let bad = text_config_json(52).replace(
            "\"sliding_attention\",\"sliding_attention\",\"sliding_attention\",\"full_attention\"]",
            "\"sliding_attention\",\"sliding_attention\",\"full_attention\"]",
        );
        let err = parse(&bad).unwrap_err().to_string();
        assert!(
            err.contains("layer_types"),
            "expected a layer_types arity error, got: {err}"
        );
    }

    #[test]
    fn rejects_a_nope_layer_that_is_not_full_attention() {
        // Flip layer 0 to theta 0 while leaving it sliding_attention.
        let bad = text_config_json(52).replacen("500000.0", "0", 1);
        let err = parse(&bad).unwrap_err().to_string();
        assert!(
            err.contains("NoPE") || err.contains("full_attention"),
            "a theta-0 sliding layer must fail closed, got: {err}"
        );
    }

    /// The inverse of the test above, and the direction that fails OPEN if it is
    /// missing: a Full layer carrying a real theta loads fine, `rope_theta_for`
    /// returns `Some(500000.0)`, and the decoder rotates q/k on a layer the
    /// reference leaves unrotated — numerically valid, fluent, and wrong.
    #[test]
    fn rejects_a_full_attention_layer_that_is_not_nope() {
        // Give layer 3 (full_attention) the sliding layers' theta instead of 0.
        let bad = text_config_json(52).replacen(
            "500000.0,500000.0,500000.0,0,",
            "500000.0,500000.0,500000.0,500000.0,",
            1,
        );
        let err = parse(&bad).unwrap_err().to_string();
        // Pinned to wording unique to this direction. The other half's message
        // reads "layer {i} has layer_rope_theta 0 (NoPE) but layer_types says
        // …" and can never produce "is full_attention with", so this assert
        // cannot be satisfied by the theta-0-on-sliding error.
        assert!(
            err.contains("layer 3 is full_attention with layer_rope_theta 500000"),
            "a full_attention layer with a non-zero theta must fail closed, got: {err}"
        );
        assert!(
            err.contains("must be 0"),
            "the error must state the theta has to be 0, got: {err}"
        );
    }

    #[test]
    fn accepts_a_head_dim_that_is_not_hidden_size_over_heads() {
        // 32 * 128 = 4096 != 6656. head_dim is independent of hidden_size in this
        // architecture, so a "head_dim * heads == hidden_size" guard would be a
        // FALSE invariant that rejects the real checkpoint.
        let t = parse(&text_config_json(52)).unwrap().text_config;
        assert_eq!(t.head_dim, 128);
        assert_ne!(t.head_dim * t.num_attention_heads, t.hidden_size);
    }

    #[test]
    fn rejects_a_kv_head_count_that_does_not_divide_the_query_heads() {
        let bad = text_config_json(52)
            .replace("\"num_key_value_heads\": 2", "\"num_key_value_heads\": 5");
        let err = parse(&bad).unwrap_err().to_string();
        assert!(
            err.contains("num_key_value_heads"),
            "GQA requires heads % kv_heads == 0, got: {err}"
        );
    }

    #[test]
    fn effective_qk_scale_is_the_product_not_a_replacement() {
        let t = parse(&text_config_json(52)).unwrap().text_config;
        // 3.87 is applied to q ON TOP OF 1/sqrt(128). Treating 3.87 as the whole
        // scale is ~44x off; treating it as gemma's query_pre_attn_scalar
        // (scalar**-0.5) is ~7.6x off.
        let expected = 3.87f32 * (128f32).powf(-0.5);
        assert!(
            (t.effective_qk_scale() - expected).abs() < 1e-9,
            "got {}, expected {expected}",
            t.effective_qk_scale()
        );
        assert!(
            (expected - 0.342_062_9).abs() < 1e-6,
            "expected ~0.3420629, got {expected}"
        );
    }

    /// Every `#[serde(default)]` field's default equals the reference value, so
    /// no assertion on the real checkpoint can tell "read from the file" apart
    /// from "silently defaulted" — a misspelled serde name would be invisible.
    /// Feeding NON-default values is the only thing that pins the key strings.
    #[test]
    fn defaulted_fields_are_read_from_the_file_when_present() {
        let text = text_config_json(52)
            .replace("\"sliding_window\": 2048", "\"sliding_window\": 4096")
            .replace("\"post_norm_eps\": 1e-8", "\"post_norm_eps\": 2e-8")
            .replace("\"qk_scale_factor\": 3.87", "\"qk_scale_factor\": 1.5")
            .replace(
                "\"output_multiplier\": 0.19611613513818404",
                "\"output_multiplier\": 0.5",
            )
            .replace(
                "\"final_logit_softcapping\": 20.0",
                "\"final_logit_softcapping\": 30.0",
            )
            .replace(
                "\"tie_word_embeddings\": false",
                "\"tie_word_embeddings\": true",
            );
        let t = parse(&text).unwrap().text_config;
        assert_eq!(
            t.sliding_window, 4096,
            "sliding_window came from the default"
        );
        assert_eq!(t.post_norm_eps, 2e-8, "post_norm_eps came from the default");
        assert_eq!(
            t.qk_scale_factor, 1.5,
            "qk_scale_factor came from the default"
        );
        assert_eq!(
            t.output_multiplier, 0.5,
            "output_multiplier came from the default"
        );
        assert_eq!(
            t.final_logit_softcapping, 30.0,
            "final_logit_softcapping came from the default"
        );
        assert!(
            t.tie_word_embeddings,
            "tie_word_embeddings came from the default"
        );
    }

    #[test]
    fn documented_defaults_apply_when_the_fields_are_absent() {
        let t = parse(&minimal_text_config_json(52)).unwrap().text_config;
        assert_eq!(t.sliding_window, 2048);
        assert_eq!(t.post_norm_eps, 1e-8);
        assert_eq!(t.qk_scale_factor, 3.87);
        assert_eq!(t.output_multiplier, 0.196_116_13);
        assert_eq!(t.final_logit_softcapping, 20.0);
        assert!(!t.tie_word_embeddings);
        // The default must never be the *other* epsilon.
        assert_ne!(t.post_norm_eps, t.rms_norm_eps);
    }

    #[test]
    fn rejects_an_unrecognized_layer_type() {
        let bad =
            text_config_json(52).replacen("\"sliding_attention\"", "\"chunked_attention\"", 1);
        let err = parse(&bad).unwrap_err().to_string();
        assert!(
            err.contains("chunked_attention"),
            "an unknown span must name itself, got: {err}"
        );
    }

    #[test]
    fn rejects_a_layer_rope_theta_length_mismatch() {
        // 52 layers declared, 51 thetas supplied. Drop the LAST theta (a NoPE 0)
        // so the surviving 51 still line up with the first 51 layer_types: the
        // arity error is then the only reachable one. Dropping an interior theta
        // shifts the tables and trips the NoPE guard instead, whose message also
        // contains the substring "layer_rope_theta" — that version of this test
        // passed even with the arity check deleted.
        let bad = text_config_json(52).replace(
            "500000.0,500000.0,500000.0,0]",
            "500000.0,500000.0,500000.0]",
        );
        let err = parse(&bad).unwrap_err().to_string();
        assert!(
            err.contains("layer_rope_theta") && err.contains("entries but num_hidden_layers"),
            "expected a layer_rope_theta arity error, got: {err}"
        );
    }

    /// `sliding_window` is the window of 39 of this decoder's 52 layers, and it
    /// leaves this module as the `AttentionKind::SlidingWindow` payload of every
    /// one of them. A 0 must not survive parsing: vLLM's disable sentinel is
    /// `sliding_window = None` (`vllm/config/model.py`, `disable_sliding_window`),
    /// and its truthiness checks read a literal 0 as "not a sliding layer", so a
    /// 0 reaching a spec would silently widen three quarters of the decoder to
    /// full attention. The seam does refuse it
    /// (`KVCacheSpecError::InvalidSlidingWindow`), but only at grouping time and
    /// without naming the config field, which is too late to be actionable.
    #[test]
    fn rejects_a_zero_sliding_window() {
        let bad = text_config_json(52).replace("\"sliding_window\": 2048", "\"sliding_window\": 0");
        let err = parse(&bad).unwrap_err().to_string();
        assert!(
            err.contains("sliding_window must be non-zero, got 0"),
            "a zero sliding_window must fail closed at config load and name the field \
             and the value, got: {err}"
        );
    }

    /// The window crosses into the KV-cache seam as a `u32`. A `usize` value
    /// past `u32::MAX` must be an error, never a silent truncation: `as u32`
    /// turns 5_000_000_000 into 705_032_704, a plausible-looking window that is
    /// not the one the checkpoint asked for.
    #[test]
    fn rejects_a_sliding_window_that_does_not_fit_a_u32() {
        let bad = text_config_json(52)
            .replace("\"sliding_window\": 2048", "\"sliding_window\": 5000000000");
        let err = parse(&bad).unwrap_err().to_string();
        assert!(
            err.contains("sliding_window") && err.contains("5000000000"),
            "an out-of-u32 sliding_window must fail closed and print the observed \
             value, got: {err}"
        );
    }

    #[test]
    fn rejects_a_zero_head_dim() {
        let bad = text_config_json(52).replace("\"head_dim\": 128", "\"head_dim\": 0");
        let err = parse(&bad).unwrap_err().to_string();
        assert!(
            err.contains("head_dim"),
            "a zero head_dim would make effective_qk_scale infinite, got: {err}"
        );
    }

    // ── Real checkpoint (gated) ────────────────────────────────────────

    /// Every value in the tests above is a hand transcription of the reference
    /// config, so on their own they prove the parser self-consistent, not
    /// correct. This one parses the actual downloaded `config.json`, which is
    /// what pins the REQUIRED fields' serde names — a misspelling there is a
    /// hard deserialize error. For the `#[serde(default)]` fields it can only
    /// check key presence (see the loop below); their names are pinned by
    /// `defaulted_fields_are_read_from_the_file_when_present` instead.
    #[test]
    #[ignore = "requires the local Muse-Glimmer checkpoint; set MLX_TEST_MUSE_GLIMMER_MODEL_PATH and run with --ignored"]
    fn real_checkpoint_config_parses() {
        let Ok(dir) = std::env::var("MLX_TEST_MUSE_GLIMMER_MODEL_PATH") else {
            eprintln!("skipping: MLX_TEST_MUSE_GLIMMER_MODEL_PATH not set");
            return;
        };
        let cfg = MuseGlimmerConfig::from_path(Path::new(&dir))
            .expect("the real checkpoint's config.json must parse and validate");
        let t = &cfg.text_config;
        assert_eq!(t.num_hidden_layers, 52);
        assert_eq!(t.layer_kinds.len(), 52);
        let full: Vec<usize> = (0..t.num_hidden_layers)
            .filter(|&i| t.layer_kinds[i] == LayerKind::Full)
            .collect();
        assert_eq!(full, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51]);
        assert_eq!(t.rms_norm_eps, 1e-5);
        assert_eq!(t.post_norm_eps, 1e-8);
        assert_eq!(cfg.image_token_id, 200092);

        // The defaulted fields' values coincide with their defaults, so the
        // asserts above cannot prove the file was read for them. What closes
        // the gap is: this checks the real file carries these exact keys, and
        // `defaulted_fields_are_read_from_the_file_when_present` checks the
        // parser reads these exact keys. Together they pin the whole set.
        let raw: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(Path::new(&dir).join("config.json")).unwrap(),
        )
        .unwrap();
        let file_text = raw
            .get("text_config")
            .and_then(serde_json::Value::as_object)
            .expect("text_config object");
        for key in [
            "sliding_window",
            "post_norm_eps",
            "qk_scale_factor",
            "output_multiplier",
            "final_logit_softcapping",
            "tie_word_embeddings",
        ] {
            assert!(
                file_text.contains_key(key),
                "the real config.json's text_config has no {key:?}; the parser's \
                 default would silently stand in for it"
            );
        }

        eprintln!(
            "real checkpoint OK: {} layers, {} full/NoPE, rms_norm_eps={:e}, post_norm_eps={:e}, \
             image_token_id={}, effective_qk_scale={}",
            t.num_hidden_layers,
            full.len(),
            t.rms_norm_eps,
            t.post_norm_eps,
            cfg.image_token_id,
            t.effective_qk_scale()
        );
    }
}
