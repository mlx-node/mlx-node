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
    /// REQUIRED, deliberately with no `#[serde(default)]`. This is the KV-cache
    /// seam's `max_model_len`, so a guessed value silently mis-sizes every KV
    /// pool the model allocates. The checkpoint carries it (131072); an absent
    /// key must be a hard deserialize error, not a resolved one.
    max_position_embeddings: usize,
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
/// the text config, nothing here needs cross-field RESOLUTION — there are no two
/// parallel tables to reconcile. Read "no resolution" narrowly: the fields are
/// still VALIDATED, by [`MuseGlimmerVisionConfig::validate`], which
/// `from_json_str` calls. The earlier wording said only "nothing here needs
/// cross-field resolution", and a reader took that as "nothing here needs
/// checking" — at which point `patch_size: 0`, `merge_size: 0` and
/// `num_attention_heads: 0` all parsed and reached the runtime while the text
/// config next door refused seven separate traps.
///
/// The tower's own `layer_types` table (`window_attention` / `full_attention`)
/// is deliberately NOT read here — resolving and validating it belongs with the
/// vision tower that consumes it, and a `#[serde(default)]` empty table would be
/// a silent trap of exactly the kind this module exists to prevent.
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
    /// Maximum context length. Consumed by the KV-cache seam as `max_model_len`:
    /// it is the full-attention groups' admission bound outright, and the cap on
    /// the sliding groups' `window - 1 + max_chunk`. Never defaulted — see
    /// [`MuseGlimmerConfig::from_json_str`].
    pub max_position_embeddings: usize,
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

impl MuseGlimmerVisionConfig {
    /// Refuse a tower geometry that cannot describe a real vision stack.
    ///
    /// Every field below is a DIVISOR or a DIMENSION of something M2 will build,
    /// and each zero fails in a different place far from the config:
    ///
    ///   * `patch_size` and `merge_size` divide the image grid. `smart_resize`
    ///     rounds a resolution to a multiple of `patch_size * merge_size`; a 0
    ///     there is a division by zero or an infinite loop depending on how the
    ///     rounding is written, in M2's resize helper, on the first image.
    ///   * `patch_temporal` is the temporal stride (2 real frames per patch).
    ///     A 0 makes `grid_t = frames / patch_temporal` a division by zero.
    ///   * `pos_emb_height` / `pos_emb_width` are the interpolation grid the
    ///     position embedding is resampled FROM (32 x 32). A 0 makes the source
    ///     grid empty, so interpolation reads from nothing.
    ///   * `hidden_size % num_attention_heads == 0` is the tower's own head
    ///     split (1536 / 16 = 96). Unlike the text decoder — where `head_dim` is
    ///     an independent field and `32 x 128 != 6656` on purpose — the tower has
    ///     no `head_dim` key, so its head dim IS the quotient and a non-dividing
    ///     pair silently truncates or panics on reshape.
    ///   * `layer_norm_eps` is the only FLOAT here, so the usize table above
    ///     cannot reach it and it needs its own clause. It is the `eps` of
    ///     `sqrt(var + eps)` in every LayerNorm of the tower: a negative one
    ///     drives that argument negative and the activation to NaN, and a `+inf`
    ///     one drives the normalized activation to zero. Required finite and
    ///     strictly positive — a FLOOR, not a sanity check, since `1e-30` passes
    ///     and is still not a usable epsilon.
    ///
    /// Kept as a method rather than inlined so the tower can re-check a config it
    /// did not parse: the type has all-`pub` fields and no private constructor,
    /// so a struct literal bypasses `from_json_str` entirely.
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("intermediate_size", self.intermediate_size),
            ("num_attention_heads", self.num_attention_heads),
            ("patch_size", self.patch_size),
            ("merge_size", self.merge_size),
            ("patch_temporal", self.patch_temporal),
            ("pos_emb_height", self.pos_emb_height),
            ("pos_emb_width", self.pos_emb_width),
        ] {
            if value == 0 {
                return Err(Error::from_reason(format!(
                    "muse_glimmer: vision_config.{name} must be non-zero; it is a divisor \
                     or a dimension of the patch grid, so a 0 surfaces as a division by \
                     zero or an empty tensor inside the tower rather than here"
                )));
            }
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(Error::from_reason(format!(
                "muse_glimmer: vision_config.num_attention_heads {} must divide \
                 hidden_size {} (real checkpoint: 1536 / 16 = 96); the tower has no \
                 head_dim key, so its head dim IS this quotient",
                self.num_attention_heads, self.hidden_size
            )));
        }
        if !self.layer_norm_eps.is_finite() || self.layer_norm_eps <= 0.0 {
            return Err(Error::from_reason(format!(
                "muse_glimmer: vision_config.layer_norm_eps must be finite and strictly \
                 positive, got {}; it is the eps of sqrt(var + eps) in every LayerNorm of \
                 the tower, so a negative one turns the activation NaN and an infinite one \
                 turns it to zero — neither raises anything",
                self.layer_norm_eps
            )));
        }
        Ok(())
    }
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
            // The two clauses above partition on zero-vs-nonzero and say NOTHING
            // about the sign, magnitude or finiteness of the non-zero half. There
            // is no backstop: MLX validates only `dims` on this path
            // (`fast.cpp:411`) and has no base check anywhere. Worse, the two
            // backends disagree about what a bad base does, and the PRODUCTION
            // one is the silent one:
            //
            //   * CPU (`fast.cpp:463`, `exp(arange * log(base) / hd)`): a base of
            //     -500000 gives 1024/1024 NaN. Loud, and not what runs.
            //   * Metal (`rope.cpp:123`, `std::log2(base_)`): the same base gives
            //     0 NaN and 1024/1024 EXACT ZEROS. q and k come back all-zero and
            //     attention degenerates to the plain causal mean of V — measured
            //     max|good - bad| = 1.5308 against the correct output, and
            //     max|bad - causal_mean_of_V| = 1.19e-07. Finite, plausible, no
            //     error: the same fail-open shape as the negative `sliding_window`
            //     documented below.
            //
            // `is_finite` is not redundant with `> 0.0` here, and the reason is
            // the `as f32` narrowing serde does on the way in: `1e400` is refused
            // at parse ("number out of range"), but `1e40` is valid JSON, fits
            // the f64 serde reads it into, and lands in this f32 field as `+inf`
            // with nothing raised. A `+inf` base zeroes 16/1024 lanes on Metal and
            // NaNs on CPU.
            //
            // Only the rotated entries are checked, and that is deliberate:
            // `-0.0 == 0.0` in IEEE 754, so a `-0.0` never reaches this clause —
            // it is NoPE, which is correct, and is pinned by
            // `a_negative_zero_rope_theta_on_a_full_layer_is_still_nope_and_still_accepted`.
            if *theta != 0.0 && (!theta.is_finite() || *theta <= 0.0) {
                return Err(Error::from_reason(format!(
                    "muse_glimmer: layer {i} has layer_rope_theta {theta}; a rotated layer's \
                     RoPE base must be finite and strictly positive. MLX validates only the \
                     `dims` argument, never the base, and on the Metal path a non-positive \
                     base produces all-zero q/k rather than a NaN — attention then returns \
                     the causal mean of V, which is finite, plausible and wrong"
                )));
            }
        }

        // A 0-layer decoder validates without this check and fails OPEN, one
        // layer down. Both per-layer arity checks above pass (0 == 0), the
        // NoPE<->Full loop iterates nothing, and `compute_layer_kv_cache_specs`
        // returns `Ok(vec![])`. The refusal that does eventually fire is
        // `compute_layer_kv_cache_groups`' both-kinds guard, whose message reads
        // "0 layers produced 0 full-attention group(s) and 0 sliding-window
        // group(s)" — it points the reader at grouping when the bad value is a
        // config field. Refuse here, where the field has a name.
        if layers == 0 {
            return Err(Error::from_reason(
                "muse_glimmer: num_hidden_layers must be non-zero; a 0 passes both \
                 per-layer arity checks against empty tables and surfaces much later as \
                 an empty KV-cache grouping, blaming grouping for a config field",
            ));
        }

        // GQA's only real head-count invariant: the query heads must divide
        // evenly into kv groups (32 / 2 = 16 here). Note there is deliberately
        // NO `head_dim * num_attention_heads == hidden_size` check — that holds
        // in most families but not this one, and would reject the real
        // checkpoint (32 x 128 = 4096 vs hidden_size 6656).
        //
        // `num_attention_heads == 0` needs its own clause because divisibility
        // does not catch it: `0.is_multiple_of(1)` is `true`, so a 0 with
        // `num_key_value_heads: 1` passes. And it fails OPEN through the KV seam
        // specifically — the physical layout is built from `num_key_value_heads`
        // and `head_dim`, never from the query heads, so a 0-query-head config
        // sizes a perfectly valid cache for a decoder with no attention.
        if text.num_attention_heads == 0 {
            return Err(Error::from_reason(
                "muse_glimmer: num_attention_heads must be non-zero; the KV-cache layout \
                 is built from num_key_value_heads and head_dim, so a 0 here sizes a valid \
                 cache and is never caught downstream",
            ));
        }
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

        // The three decoder dimensions nothing else reaches. In most families a
        // 0 `hidden_size` would be caught by `head_dim * num_attention_heads ==
        // hidden_size`, and this family deliberately does NOT have that check
        // (see the GQA clause above: 32 x 128 = 4096 against the checkpoint's
        // hidden_size of 6656), so the usual backstop is absent by design. The
        // head tables do not reach these either — they describe per-layer kinds,
        // not widths.
        //
        // Each one fails far from here and in a way that does not name the
        // config: a 0 `hidden_size` is a zero-width residual stream, a 0
        // `intermediate_size` is an FFN projecting to nothing, and a 0
        // `vocab_size` is an empty embedding and an empty logits table. All three
        // surface inside M1's decoder as an invalid tensor construction or a
        // silently unusable model.
        //
        // Named individually rather than in a loop so a dropped clause is a red
        // test rather than a gap that still reads as covered.
        for (field, value) in [
            ("hidden_size", text.hidden_size),
            ("intermediate_size", text.intermediate_size),
            ("vocab_size", text.vocab_size),
        ] {
            if value == 0 {
                return Err(Error::from_reason(format!(
                    "muse_glimmer: {field} must be non-zero; nothing downstream catches a 0 \
                     here because this family has no head_dim * num_attention_heads == \
                     hidden_size invariant, so it would surface as an invalid tensor inside \
                     the decoder rather than as a config error"
                )));
            }
        }

        // Every remaining f32 of the text config. Not one of them has a guard
        // today and not one has a non-test consumer yet, so this is LATENT rather
        // than live — but each entry below is a measured failure, and none of
        // them is caught anywhere downstream.
        //
        // One rule for all five: FINITE and STRICTLY POSITIVE. `is_finite` is not
        // redundant with `> 0.0`, for the same `as f32` reason as the RoPE base
        // above: `1e400` is refused at parse, `1e40` is valid JSON and arrives
        // here as `+inf`. Bare `NaN` / `Infinity` literals are not valid JSON at
        // all, so serde covers those.
        //
        // Strictly positive is a FLOOR, not a sanity check: `1e-30` passes every
        // clause here and is still not a usable epsilon or a usable multiplier.
        // What the guard buys is that the values which provably destroy the
        // forward pass are named here instead of surfacing as NaN, as zeros, or
        // as plausible-looking wrong output. `effective_qk_scale` needs no clause
        // of its own — it is `qk_scale_factor * head_dim^-0.5`, and `head_dim`'s
        // own non-zero check is above, so guarding the factor covers the product.
        //
        // Each field carries its own consequence rather than sharing one message,
        // so dropping a row is a red test rather than a gap that still reads as
        // covered — and so the softcapping row can say what is actually true of
        // it, which is not what is true of the others.
        for (field, value, consequence) in [
            (
                "rms_norm_eps",
                text.rms_norm_eps,
                "it is the eps of the input and pre-feedforward RMS norms; -1.0 measured \
                 64/64 NaN on GPU and CPU alike, and +inf drives the whole normalized \
                 activation to zero. A small negative such as -1e-8 does not even fail \
                 reliably — it survives while mean(x^2) stays larger than its magnitude — \
                 so the harm is data-dependent, which is precisely why it is refused at \
                 load rather than left to the forward pass",
            ),
            (
                "post_norm_eps",
                text.post_norm_eps,
                "same class as rms_norm_eps, on the POST norms; it is a separate field \
                 carrying this family's smaller epsilon (1e-8), so it needs its own clause",
            ),
            (
                "qk_scale_factor",
                text.qk_scale_factor,
                "it multiplies q on top of 1/sqrt(head_dim), so a 0 flattens every \
                 attention logit and makes attention uniform over the whole context, while \
                 a negative one inverts the ranking outright — the model attends hardest to \
                 what it should attend to least. Both stay finite and fluent",
            ),
            (
                "output_multiplier",
                text.output_multiplier,
                "it scales the final hidden state into the logits, so a 0 zeroes every \
                 logit and makes sampling uniform over the vocabulary, while a negative one \
                 flips the sign of every logit and inverts argmax — greedy decoding then \
                 picks the LEAST likely token",
            ),
            (
                "final_logit_softcapping",
                text.final_logit_softcapping,
                "cap * tanh(x / cap) with cap = 0 flattens EVERY logit to +/-0, i.e. a \
                 uniform distribution over the vocabulary — and 0 is not a \"disabled\" \
                 sentinel, because the default is 20.0 and this family has no \
                 representation for disabled at all. A NEGATIVE cap is the other case and \
                 the honest description differs: the formula is EVEN in cap, so -20.0 \
                 computes exactly what +20.0 computes and corrupts nothing. It is refused \
                 because a config that says -20.0 is silently equivalent to its magnitude, \
                 and the magnitude is what should have been written",
            ),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Error::from_reason(format!(
                    "muse_glimmer: {field} must be finite and strictly positive, got \
                     {value}; {consequence}"
                )));
            }
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
        //   * a value past `i32::MAX` reaches a dispatch NEGATIVE. The KV-cache
        //     seam stores the window as a `u32`, so `u32::MAX` looks like the
        //     natural bound, but both things that consume the window take an
        //     `i32`: the Metal kernel's `sliding_window` FFI argument and
        //     `create_causal_mask`'s `window_size` (`array/mask.rs:35`). So the
        //     band `[2^31, 2^32-1]` fits every type on the way down and still
        //     arrives wrong — 3_000_000_000 becomes -1_294_967_296. The kernel
        //     route then fails closed (the C++ validator rejects a negative
        //     window, `mlx_paged_ops.cpp:483`), but the MASK route fails open and
        //     silent: `create_causal_mask` keeps a cell only when
        //     `linds < rinds + window_size`, so a negative width keeps NOTHING,
        //     and the fused SDPA fed an all-false mask returns the uniform mean
        //     of every V row — finite, plausible, no NaN, no error, max|delta|
        //     1.2485 against the windowed reference on real MLX. Beyond
        //     `u32::MAX` the value also truncates outright (5_000_000_000 would
        //     arrive as 705_032_704), which the same check covers.
        //
        // Validating here rather than at spec derivation is deliberate, and the
        // reason is stronger than "earlier is better": the seam's own refusal is
        // NOT where a reader expects it. `validate_layer_kv_cache_specs`
        // (`transformer/kv_cache_spec.rs`) checks duplicate layer indices, layout
        // validity and shared-anchor rules — it never looks at the window. The
        // only `KVCacheSpecError::InvalidSlidingWindow` lives inside
        // `AttentionKind::sliding_window_max_admission_blocks`, i.e. in the
        // ADMISSION-BLOCK ARITHMETIC that grouping happens to call. So the
        // `KVCacheCoordinator::from_groups` ->
        // `derive_layer_kv_cache_routes_from_groups` path, which calls only the
        // `validate_*` functions, catches a 0 window NOWHERE. Neither this family
        // nor gemma4 reaches that path with an unvalidated window today, but the
        // seam is `pub`, so this check plus `compute_layer_kv_cache_specs`' own
        // are the refusals that actually exist.
        if text.sliding_window == 0 {
            return Err(Error::from_reason(format!(
                "muse_glimmer: sliding_window must be non-zero, got {}; 39 of this \
                 decoder's layers are sliding_attention, and a 0 window would widen \
                 all of them to full attention instead of failing",
                text.sliding_window
            )));
        }
        if i32::try_from(text.sliding_window).is_err() {
            return Err(Error::from_reason(format!(
                "muse_glimmer: sliding_window {} exceeds i32::MAX ({}); the window's two \
                 consumers are both i32 — the paged kernel's sliding_window argument and \
                 create_causal_mask's window_size — so a larger value reaches a dispatch \
                 negative, and a negative mask width keeps no keys at all instead of \
                 failing",
                text.sliding_window,
                i32::MAX
            )));
        }

        // `max_position_embeddings` reaches the KV-cache seam as `max_model_len`.
        // A 0 there makes the full-attention bound `div_ceil(0, block_size) == 0`
        // — a group that can admit no blocks — and the same `u32` truncation trap
        // as the window applies. Both fail closed here, where the field has a
        // name, rather than deep inside group construction.
        if text.max_position_embeddings == 0 {
            return Err(Error::from_reason(format!(
                "muse_glimmer: max_position_embeddings must be non-zero, got {}; it is \
                 the KV cache seam's max_model_len and a 0 admits no blocks at all",
                text.max_position_embeddings
            )));
        }
        if u32::try_from(text.max_position_embeddings).is_err() {
            return Err(Error::from_reason(format!(
                "muse_glimmer: max_position_embeddings {} does not fit in a u32; the KV \
                 cache seam carries max_model_len as a u32 and a cast would truncate it",
                text.max_position_embeddings
            )));
        }

        // The two projector dimensions. Both are top-level and REQUIRED (no
        // serde default), and both reach the resolved config completely
        // unchecked: `from_json_str` validates the text config and the vision
        // config, and nothing at all in between.
        //
        // Nothing reads either field yet — the projector module does not exist —
        // so this is hardening ahead of the consumer, not a live bug fix. The
        // reference dimensions all three projector stages from these two ints
        // (`modeling_muse_glimmer.py:990-1004`), and a 0 in either is not
        // symmetric:
        //
        //   * `projector_hidden_size: 0` makes fc1 emit a `[n_tokens, 0]` tensor.
        //     No error whatsoever — a degenerate value that just flows on.
        //   * `out_hidden_size: 0` ABORTS THE PROCESS. The fc1 matmul
        //     `[8, 6144] @ [0, 4096]` throws a C++ exception out of MLX and
        //     across the FFI boundary, where Rust cannot catch it: "fatal runtime
        //     error: Rust cannot catch foreign exceptions", SIGABRT, and no
        //     diagnostic naming the config. Strictly worse than an `Err`.
        //
        // Zero only. `out_hidden_size` happens to equal
        // `vision_config.hidden_size * merge_size^2` (1536 * 4 = 6144) on this
        // checkpoint, and enforcing that equality WOULD catch more — it is
        // deliberately NOT enforced, because the HF reference does not enforce it
        // either and it would reject a variant checkpoint with a different merge
        // geometry. Negatives need no clause: both fields are `usize`, so serde
        // already refuses them with a message that names the field.
        //
        // `projector_hidden_act` is deliberately left alone despite sitting right
        // next to these. No family in this repo validates an activation name at
        // config load — they all resolve it at the use site (see
        // `pp_doclayout_v3/persistence.rs:211`) — and the supported set is not
        // known until the projector exists.
        for (field, value) in [
            ("out_hidden_size", raw.out_hidden_size),
            ("projector_hidden_size", raw.projector_hidden_size),
        ] {
            if value == 0 {
                return Err(Error::from_reason(format!(
                    "muse_glimmer: {field} must be non-zero; both projector dimensions are \
                     unchecked by the text and vision validators, and a 0 here is either a \
                     degenerate zero-width tensor or an MLX C++ exception thrown across the \
                     FFI boundary, which aborts the process instead of returning an error"
                )));
            }
        }

        // The vision tower's geometry, checked here for the same reason as
        // everything above: the fields are `pub` on a type with no private
        // constructor, so this is the one place a real `config.json` is
        // guaranteed to pass through.
        raw.vision_config.validate()?;

        // Media placeholders enter the text token stream after expansion, so
        // both ids must be representable by the decoder embedding table. Catch a
        // malformed checkpoint at load instead of on its first image or video.
        for (field, id) in [
            ("image_token_id", raw.image_token_id),
            ("video_token_id", raw.video_token_id),
        ] {
            if id >= text.vocab_size {
                return Err(Error::from_reason(format!(
                    "muse_glimmer: {field} {id} must be less than text_config.vocab_size {}",
                    text.vocab_size
                )));
            }
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
                max_position_embeddings: text.max_position_embeddings,
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
              "max_position_embeddings": 131072,
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
              "max_position_embeddings": 131072,
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
        assert_eq!(t.max_position_embeddings, 131072);
        assert_eq!(t.qk_scale_factor, 3.87);
        assert_eq!(t.final_logit_softcapping, 20.0);
        assert_eq!(cfg.image_token_id, 200092);
        assert_eq!(cfg.video_token_id, 200091);
        assert_eq!(cfg.out_hidden_size, 6144);
        assert_eq!(cfg.projector_hidden_size, 4096);
    }

    #[test]
    fn rejects_media_token_ids_outside_the_text_vocabulary() {
        for (field, original) in [("image_token_id", "200092"), ("video_token_id", "200091")] {
            for id in [202048, 202049] {
                let good = config_json(&text_config_json(52));
                let bad = good.replace(
                    &format!("\"{field}\":{original}"),
                    &format!("\"{field}\":{id}"),
                );
                assert_ne!(bad, good, "fixture edit missed {field}");
                let err = MuseGlimmerConfig::from_json_str(&bad)
                    .unwrap_err()
                    .to_string();
                assert!(
                    err.contains(field) && err.contains("vocab_size"),
                    "out-of-vocabulary {field}={id} was not refused by name: {err}"
                );
            }
        }
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

    /// `max_position_embeddings` is the KV-cache seam's `max_model_len`: it caps
    /// sliding admission and IS the full group's admission bound (8192 blocks at
    /// block_size 16 for this checkpoint's 131072). There is no defensible
    /// default for a context length — inventing one silently mis-sizes every KV
    /// pool the model allocates — so the field is REQUIRED, and a missing key is
    /// a hard deserialize error rather than a resolved value. A
    /// `#[serde(default)]` here would be indistinguishable from "silently
    /// defaulted", the exact failure mode
    /// `defaulted_fields_are_read_from_the_file_when_present` exists to catch.
    #[test]
    fn requires_max_position_embeddings_with_no_default() {
        let bad = text_config_json(52).replace("\"max_position_embeddings\": 131072,", "");
        let err = parse(&bad).unwrap_err().to_string();
        // Pinned to serde's own wording, which only a genuinely required field
        // can produce. A `#[serde(default)]` mutation would instead surface the
        // `must be non-zero` guard below, whose message also names the field —
        // asserting on the field name alone would let that mutation live.
        assert!(
            err.contains("missing field `max_position_embeddings`"),
            "an absent context length must be a hard deserialize error, got: {err}"
        );
    }

    /// A 0 context length makes the seam's full-attention bound
    /// `div_ceil(0, block_size) == 0`: a full group that can admit no blocks at
    /// all. That is a startup failure with no useful message, or worse a pool
    /// sized to nothing, so refuse it here where the field has a name.
    #[test]
    fn rejects_a_zero_max_position_embeddings() {
        let bad = text_config_json(52).replace(
            "\"max_position_embeddings\": 131072",
            "\"max_position_embeddings\": 0",
        );
        let err = parse(&bad).unwrap_err().to_string();
        assert!(
            err.contains("max_position_embeddings must be non-zero, got 0"),
            "a zero context length must fail closed and name the field and value, got: {err}"
        );
    }

    /// Same `u32` boundary as `sliding_window`: the seam takes `max_model_len` as
    /// a `u32`, so an out-of-range `usize` must be an error, not a truncation.
    #[test]
    fn rejects_a_max_position_embeddings_that_does_not_fit_a_u32() {
        let bad = text_config_json(52).replace(
            "\"max_position_embeddings\": 131072",
            "\"max_position_embeddings\": 5000000000",
        );
        let err = parse(&bad).unwrap_err().to_string();
        assert!(
            err.contains("max_position_embeddings") && err.contains("5000000000"),
            "an out-of-u32 context length must fail closed and print the observed \
             value, got: {err}"
        );
    }

    /// `sliding_window` is the window of 39 of this decoder's 52 layers, and it
    /// leaves this module as the `AttentionKind::SlidingWindow` payload of every
    /// one of them. A 0 must not survive parsing: vLLM's disable sentinel is
    /// `sliding_window = None` (`vllm/config/model.py`, `disable_sliding_window`),
    /// and its truthiness checks read a literal 0 as "not a sliding layer", so a
    /// 0 reaching a spec would silently widen three quarters of the decoder to
    /// full attention. The seam's refusal is weaker than it looks:
    /// `KVCacheSpecError::InvalidSlidingWindow` is raised only inside
    /// `AttentionKind::sliding_window_max_admission_blocks`, so it fires at
    /// grouping time, without naming the config field — and not at all on the
    /// `KVCacheCoordinator::from_groups` path, which calls only
    /// `validate_layer_kv_cache_specs` and that function never inspects the
    /// window.
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

    /// The window's bound is `i32::MAX`, not `u32::MAX`, and the boundary is
    /// walked rather than spot-checked because the dangerous band is exactly
    /// `[2^31, 2^32-1]` — every value there fits the `u32` the KV-cache seam
    /// stores and still arrives at a dispatch NEGATIVE, since both consumers take
    /// an `i32` (the kernel's `sliding_window` FFI argument and
    /// `create_causal_mask`'s `window_size`). Measured: 3_000_000_000 reaches
    /// `PagedWindowSlot::mask_window()` as -1_294_967_296, and a negative mask
    /// width keeps 0 of 96 cells, which the fused SDPA turns into the uniform
    /// mean of every V row — no NaN, no error, max|delta| 1.2485 against the
    /// windowed reference. Past `u32::MAX` the value truncates outright
    /// (5_000_000_000 -> 705_032_704), which the same check covers.
    ///
    /// `i32::MAX` itself must stay ACCEPTED. A window larger than the context is
    /// legal, and gemma4's "the window cannot bite" fast path decides by
    /// comparing the window against the context, which needs it representable
    /// first. So this test pins both directions.
    ///
    /// Mutations caught: reverting the guard to `u32::try_from` (2_147_483_648
    /// and 4_294_967_295 become `Ok`); tightening it below `i32::MAX`
    /// (2_147_483_647 stops parsing). Pinned to this guard's own "exceeds
    /// i32::MAX" wording, not merely to the token `sliding_window`, because the
    /// zero-window guard's message also contains that token and a loose assertion
    /// would survive deleting this one.
    #[test]
    fn bounds_the_sliding_window_at_i32_max_not_u32_max() {
        const ACCEPTED: [usize; 3] = [2048, 1 << 30, i32::MAX as usize];
        const REFUSED: [usize; 3] = [
            i32::MAX as usize + 1, // 2_147_483_648: fits a u32, arrives as i32::MIN
            u32::MAX as usize,     // 4_294_967_295: fits a u32, arrives as -1
            u32::MAX as usize + 1, // 4_294_967_296: does not even fit a u32
        ];

        for window in ACCEPTED {
            let json = text_config_json(52).replace(
                "\"sliding_window\": 2048",
                &format!("\"sliding_window\": {window}"),
            );
            let cfg = parse(&json).unwrap_or_else(|e| {
                panic!("sliding_window {window} is <= i32::MAX and must parse, got: {e}")
            });
            assert_eq!(cfg.text_config.sliding_window, window);
        }

        for window in REFUSED {
            let json = text_config_json(52).replace(
                "\"sliding_window\": 2048",
                &format!("\"sliding_window\": {window}"),
            );
            let err = parse(&json)
                .expect_err("a sliding_window past i32::MAX must fail closed at config load")
                .to_string();
            assert!(
                err.contains("exceeds i32::MAX") && err.contains(&window.to_string()),
                "the refusal must name the i32 bound and print the observed value \
                 {window}, got: {err}"
            );
        }
    }

    /// A literal negative window never reaches our code at all: `sliding_window`
    /// deserializes as `usize`, so serde refuses it before any guard runs.
    /// Recorded as a test rather than as a comment because it is the reason the
    /// guard above only has to check the UPPER bound — if the field's type ever
    /// widened to a signed one, this test goes red and the missing lower bound
    /// becomes visible instead of arriving at the kernel as a negative window.
    #[test]
    fn a_negative_sliding_window_is_refused_by_deserialization_before_any_guard() {
        let bad =
            text_config_json(52).replace("\"sliding_window\": 2048", "\"sliding_window\": -1");
        let err = parse(&bad)
            .expect_err("a negative sliding_window must not deserialize")
            .to_string();
        assert!(
            err.contains("-1"),
            "serde must reject the negative window and name it, got: {err}"
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

    /// A 0-layer decoder is the config-level shape that fails OPEN through every
    /// existing check: both per-layer arity checks pass against empty tables
    /// (0 == 0), the NoPE<->Full loop iterates nothing, and
    /// `compute_layer_kv_cache_specs` returns `Ok(vec![])`. The refusal that
    /// eventually fires blames GROUPING ("0 layers produced 0 full-attention
    /// group(s)") for a config field.
    ///
    /// Mutation caught: deleting the `layers == 0` guard. Note the assertion is
    /// pinned to this guard's own wording — a looser `contains("num_hidden_layers")`
    /// would stay green under the deletion, because the arity messages name the
    /// same field.
    #[test]
    fn rejects_a_zero_num_hidden_layers_because_empty_tables_agree_with_it() {
        let bad = text_config_json(0);
        let err = parse(&bad).unwrap_err().to_string();
        assert!(
            err.contains("num_hidden_layers must be non-zero"),
            "a 0-layer decoder must be refused where the field has a name, got: {err}"
        );
    }

    /// `num_attention_heads: 0` is invisible to the GQA divisibility check —
    /// `0.is_multiple_of(1)` is `true` — and invisible to the KV seam, which
    /// builds its layout from `num_key_value_heads` and `head_dim` and never
    /// looks at the query heads. So it produces a perfectly valid cache for a
    /// decoder with no attention.
    ///
    /// Mutation caught: deleting the dedicated clause and relying on the
    /// divisibility check. The fixture uses `num_key_value_heads: 1` precisely so
    /// divisibility passes; with the checkpoint's 2 it would fail for the wrong
    /// reason and the mutation would survive.
    #[test]
    fn rejects_a_zero_num_attention_heads_that_divisibility_cannot_catch() {
        let bad = text_config_json(52)
            .replace("\"num_attention_heads\": 32", "\"num_attention_heads\": 0")
            .replace("\"num_key_value_heads\": 2", "\"num_key_value_heads\": 1");
        // Sanity: the mutation this test exists to catch would let this through.
        assert!(0usize.is_multiple_of(1usize));
        let err = parse(&bad).unwrap_err().to_string();
        assert!(
            err.contains("num_attention_heads must be non-zero"),
            "a 0 query-head count must be refused on its own clause, got: {err}"
        );
    }

    /// The three decoder dimensions that no other invariant reaches.
    ///
    /// `num_attention_heads` and `head_dim` have their own clauses, and the head
    /// tables catch a bad layer count -- but `hidden_size`, `intermediate_size`
    /// and `vocab_size` are checked by NOTHING today. In most families
    /// `head_dim * num_attention_heads == hidden_size` would catch a 0, and this
    /// family deliberately does NOT have that check (see the comment on the GQA
    /// clause: the real checkpoint is 32 x 128 = 4096 against a hidden_size of
    /// 6656), so the usual backstop is absent by design.
    ///
    /// Each one fails far from the config that carried it: a 0 `hidden_size` is a
    /// zero-width residual stream, a 0 `intermediate_size` is an FFN that
    /// projects to nothing, and a 0 `vocab_size` is an empty embedding and an
    /// empty logits table -- an invalid tensor construction or an unusable model,
    /// surfacing inside M1's decoder rather than at load.
    ///
    /// Mutation caught: deleting any one of the three clauses. Each field is
    /// asserted separately, so dropping one is a red test rather than a silent
    /// gap in a shared loop.
    #[test]
    fn rejects_a_zero_decoder_dimension_that_no_other_invariant_reaches() {
        for (field, original) in [
            ("hidden_size", "6656"),
            ("intermediate_size", "19968"),
            ("vocab_size", "202048"),
        ] {
            let bad = text_config_json(52).replace(
                &format!("\"{field}\": {original}"),
                &format!("\"{field}\": 0"),
            );
            assert!(
                bad.contains(&format!("\"{field}\": 0")),
                "fixture edit missed {field}; the test would pass for the wrong reason",
            );
            let err = parse(&bad).unwrap_err().to_string();
            assert!(
                err.contains(field) && err.contains("non-zero"),
                "a 0 {field} must be refused by name, got: {err}",
            );
        }
    }

    /// The two projector dimensions, which sit between the two sub-configs and
    /// are therefore reached by NEITHER validator. Both are top-level and both
    /// are required, so a 0 in either survives parsing today.
    ///
    /// The two zeros do not fail the same way. `projector_hidden_size: 0` makes
    /// fc1 emit a `[n_tokens, 0]` tensor and raises nothing at all;
    /// `out_hidden_size: 0` makes the fc1 matmul `[8, 6144] @ [0, 4096]` throw a
    /// C++ exception out of MLX across the FFI boundary, which Rust cannot catch
    /// — "fatal runtime error: Rust cannot catch foreign exceptions", SIGABRT,
    /// and nothing naming the config. A refusal here is strictly better than
    /// either.
    ///
    /// No negative case: both fields are `usize`, so serde already refuses a
    /// negative by name. And deliberately no `out_hidden_size ==
    /// vision_config.hidden_size * merge_size^2` check (1536 * 4 = 6144 here) —
    /// see the comment on the guard for why that stronger invariant is out of
    /// scope.
    ///
    /// Mutation caught: deleting either row of the guard's table. Each field is
    /// asserted by name.
    #[test]
    fn rejects_a_zero_projector_dimension_that_neither_sub_config_validator_reaches() {
        for (field, real) in [
            ("out_hidden_size", "6144"),
            ("projector_hidden_size", "4096"),
        ] {
            let good = config_json(&text_config_json(52));
            let needle = format!("\"{field}\":{real}");
            assert!(
                good.contains(&needle),
                "the fixture must spell {field} as {needle:?} for this substitution to bite"
            );
            let bad = good.replace(&needle, &format!("\"{field}\":0"));
            let err = MuseGlimmerConfig::from_json_str(&bad)
                .expect_err("a zero projector dimension must fail closed")
                .to_string();
            assert!(
                err.contains(&format!("{field} must be non-zero")),
                "the refusal must name the projector field, got: {err}"
            );
        }
    }

    /// Every f32 of the text config, none of which had any guard: the existing
    /// theta check partitions on zero-vs-nonzero and nothing else looks at a
    /// float at all.
    ///
    /// Three bad values per field, and the third is the one that motivates
    /// `is_finite`: `1e40` is valid JSON, fits the f64 serde reads it into, and
    /// becomes `+inf` on the `as f32` narrowing with nothing raised. The
    /// assertion pins the DISPLAYED value ("inf") rather than the literal, so it
    /// also records that narrowing actually happens — see
    /// `an_f32_overflowing_literal_arrives_as_infinity_instead_of_a_deserialize_error`.
    ///
    /// Mutation caught: dropping any single row of the guard's table (each field
    /// is asserted by name), and weakening `!is_finite() || <= 0.0` to either
    /// half alone — dropping `is_finite` leaves `1e40` green, dropping `<= 0.0`
    /// leaves `-1.0` and `0` green.
    #[test]
    fn rejects_a_non_finite_or_non_positive_float_in_the_text_config() {
        for (field, real) in [
            ("rms_norm_eps", "1e-5"),
            ("post_norm_eps", "1e-8"),
            ("qk_scale_factor", "3.87"),
            ("output_multiplier", "0.19611613513818404"),
            ("final_logit_softcapping", "20.0"),
        ] {
            // (JSON literal, how f32's Display renders what it becomes)
            for (literal, shown) in [("-1.0", "-1"), ("0", "0"), ("1e40", "inf")] {
                let good = text_config_json(52);
                let needle = format!("\"{field}\": {real}");
                assert!(
                    good.contains(&needle),
                    "the fixture must spell {field} as {needle:?} for this substitution \
                     to bite"
                );
                let bad = good.replace(&needle, &format!("\"{field}\": {literal}"));
                let err = parse(&bad)
                    .err()
                    .unwrap_or_else(|| {
                        panic!("{field} = {literal} must fail closed at config load")
                    })
                    .to_string();
                assert!(
                    err.contains(&format!(
                        "{field} must be finite and strictly positive, got {shown};"
                    )),
                    "{field} = {literal} must be refused by name and print the value the \
                     parser actually holds ({shown}), got: {err}"
                );
            }
        }
    }

    /// Why the guard above tests `is_finite` and not only `> 0.0`.
    ///
    /// Serde's number handling has a seam exactly one order of magnitude wide in
    /// the exponent: `1e400` is out of range for the f64 it parses into and is
    /// refused before any guard runs, while `1e40` is a perfectly ordinary f64
    /// that only overflows on the `as f32` narrowing into the field — silently,
    /// as `+inf`. So an out-of-range literal is serde's problem and an
    /// out-of-f32-range literal is OURS, and the second one has no other reader.
    ///
    /// Measured harm of a `+inf` RoPE base: 16/1024 lanes zeroed on Metal, all
    /// NaN on CPU. For the epsilons it drives the normalized activation to zero.
    ///
    /// Mutation caught: dropping `!value.is_finite()` from the float guard —
    /// `1e40` then parses `Ok` and the first assertion goes red. Recorded as its
    /// own test because the pair of literals is the evidence, and a reader who
    /// sees only `1e400` rejected would conclude serde already handles this.
    #[test]
    fn an_f32_overflowing_literal_arrives_as_infinity_instead_of_a_deserialize_error() {
        let overflows_f32 =
            text_config_json(52).replace("\"rms_norm_eps\": 1e-5", "\"rms_norm_eps\": 1e40");
        let err = parse(&overflows_f32)
            .expect_err("1e40 reaches the f32 field as +inf and only our guard sees it")
            .to_string();
        assert!(
            err.contains("rms_norm_eps must be finite and strictly positive, got inf"),
            "1e40 is valid JSON and in range for the f64 serde reads it into; the `as f32` \
             narrowing is what makes it +inf, got: {err}"
        );

        let overflows_f64 =
            text_config_json(52).replace("\"rms_norm_eps\": 1e-5", "\"rms_norm_eps\": 1e400");
        let err = parse(&overflows_f64)
            .expect_err("1e400 is out of range for the f64 too")
            .to_string();
        assert!(
            err.contains("invalid config.json") && !err.contains("must be finite"),
            "the neighbouring literal is refused by serde BEFORE any guard runs; if this \
             ever became our guard's message instead, the boundary moved, got: {err}"
        );
    }

    /// The rotated half of `layer_rope_theta`, which the biconditional above it
    /// says nothing about: it partitions on zero-vs-nonzero only, so any non-zero
    /// value at all passes on a sliding layer.
    ///
    /// This is the field where the consequence was measured on real MLX, and it
    /// fails OPEN on the path that actually runs. With base -500000: the CPU
    /// kernel (`fast.cpp:463`, `exp(arange * log(base) / hd)`) gives 1024/1024
    /// NaN, but Metal (`rope.cpp:123`, `std::log2(base_)`) gives 0 NaN and
    /// 1024/1024 exact zeros — q and k come back all-zero and attention collapses
    /// to the plain causal mean of V (max|good - bad| = 1.5308 against the
    /// correct output; max|bad - causal_mean_of_V| = 1.19e-07). Finite,
    /// plausible, no error. MLX itself validates only `dims`, never the base.
    ///
    /// Mutation caught: deleting the clause (both literals parse `Ok`), and
    /// dropping `is_finite` from it (the `1e40` case parses `Ok`). Pinned to this
    /// clause's own "must be finite and strictly positive" wording rather than to
    /// the token `layer_rope_theta`, which the two arity messages and both
    /// biconditional messages also contain.
    #[test]
    fn rejects_a_rope_theta_that_is_negative_or_infinite_on_a_rotated_layer() {
        // Layer 0 is sliding_attention, so it is a ROTATED layer and the
        // biconditional cannot fire on it — this clause is the only refusal.
        for (literal, shown) in [("-500000.0", "-500000"), ("1e40", "inf")] {
            let bad = text_config_json(52).replacen("500000.0", literal, 1);
            assert!(
                bad.contains(&format!("[{literal},500000.0")),
                "fixture edit missed layer 0's theta; the test would pass for the wrong \
                 reason"
            );
            let err = parse(&bad)
                .err()
                .unwrap_or_else(|| panic!("a theta of {literal} must fail closed"))
                .to_string();
            assert!(
                err.contains(&format!("layer 0 has layer_rope_theta {shown};"))
                    && err.contains("must be finite and strictly positive"),
                "a rotated layer's theta of {literal} must be refused by layer index and \
                 value, got: {err}"
            );
        }
    }

    /// `-0.0` is NoPE, and must stay accepted.
    ///
    /// IEEE 754 makes `-0.0 == 0.0` true, so a `-0.0` theta satisfies the
    /// NoPE<->Full biconditional and `rope_theta_for` correctly returns `None`
    /// for it. That is not a hole in the sign guard added alongside this test —
    /// it is the reason that guard is written `*theta != 0.0 && …` instead of
    /// `*theta <= 0.0`.
    ///
    /// Mutation caught: writing the theta guard as a bare `*theta <= 0.0`, or
    /// dropping the `!= 0.0` qualifier from it. Either turns a legitimate NoPE
    /// layer into a parse error, and this is the only test that would say so.
    #[test]
    fn a_negative_zero_rope_theta_on_a_full_layer_is_still_nope_and_still_accepted() {
        // Layer 3 is the first full_attention layer, so its theta is a NoPE 0.
        let json = text_config_json(52).replacen(
            "500000.0,500000.0,500000.0,0,",
            "500000.0,500000.0,500000.0,-0.0,",
            1,
        );
        assert!(
            json.contains("500000.0,-0.0,"),
            "fixture edit missed layer 3's theta; the test would pass for the wrong reason"
        );
        let t = parse(&json)
            .expect("-0.0 == 0.0 in IEEE 754, so a -0.0 theta is NoPE and must be accepted")
            .text_config;
        assert!(
            t.layer_rope_theta[3].is_sign_negative(),
            "the -0.0 must reach the resolved config as a NEGATIVE zero, otherwise this \
             test never exercised the sign at all"
        );
        assert_eq!(t.layer_kinds[3], LayerKind::Full);
        assert_eq!(
            t.rope_theta_for(3),
            None,
            "a -0.0 theta is NoPE exactly as a +0.0 theta is"
        );
    }

    /// The tower's only float, and the one field its usize table cannot reach.
    /// `layer_norm_eps` is the `eps` of `sqrt(var + eps)` in every LayerNorm of
    /// the tower: negative turns the activation NaN, `+inf` turns it to zero, and
    /// neither raises. Same class as the two text-side epsilons, checked in
    /// `validate` rather than in `from_json_str` for the reason that method
    /// exists — the type's fields are all `pub` with no private constructor, so
    /// the tower can re-check a config it did not parse.
    ///
    /// Mutation caught: deleting the clause from `validate`. Pinned to
    /// `vision_config.layer_norm_eps`, so the text-side guard cannot satisfy it.
    #[test]
    fn rejects_a_non_finite_or_non_positive_vision_layer_norm_eps() {
        for (literal, shown) in [("-1.0", "-1"), ("0", "0"), ("1e40", "inf")] {
            let good = config_json(&text_config_json(52));
            assert!(
                good.contains("\"layer_norm_eps\":1e-5"),
                "the fixture must spell the tower's epsilon without a space after the colon"
            );
            let bad = good.replace(
                "\"layer_norm_eps\":1e-5",
                &format!("\"layer_norm_eps\":{literal}"),
            );
            let err = MuseGlimmerConfig::from_json_str(&bad)
                .err()
                .unwrap_or_else(|| panic!("a layer_norm_eps of {literal} must fail closed"))
                .to_string();
            assert!(
                err.contains(&format!(
                    "vision_config.layer_norm_eps must be finite and strictly positive, \
                     got {shown};"
                )),
                "the tower's epsilon must be refused by name and value, got: {err}"
            );
        }
    }

    /// The acceptance half of every guard above. A validator that refuses more
    /// than it should is worse than the gap it closed, and every clause added
    /// here sits on the path the REAL config takes, so the fixture has to keep
    /// parsing untouched — in both its forms, since the minimal one exercises the
    /// `#[serde(default)]` values instead of the file's.
    ///
    /// The on-disk file itself is covered by the gated
    /// `real_checkpoint_config_parses` below; this one runs everywhere.
    ///
    /// Mutation caught: any guard written with the comparison inverted, or with
    /// `>=` where `>` belongs — the checkpoint's own values would start failing
    /// and no other test in this module distinguishes "refuses correctly" from
    /// "refuses everything".
    #[test]
    fn the_unmodified_checkpoint_fixture_passes_every_finiteness_and_positivity_guard() {
        for (label, text) in [
            ("full", text_config_json(52)),
            (
                "minimal (every #[serde(default)] field omitted)",
                minimal_text_config_json(52),
            ),
        ] {
            let cfg = parse(&text)
                .unwrap_or_else(|e| panic!("the {label} checkpoint fixture must parse: {e}"));
            let t = &cfg.text_config;
            for (field, value) in [
                ("rms_norm_eps", t.rms_norm_eps),
                ("post_norm_eps", t.post_norm_eps),
                ("qk_scale_factor", t.qk_scale_factor),
                ("output_multiplier", t.output_multiplier),
                ("final_logit_softcapping", t.final_logit_softcapping),
                (
                    "vision_config.layer_norm_eps",
                    cfg.vision_config.layer_norm_eps,
                ),
            ] {
                assert!(
                    value.is_finite() && value > 0.0,
                    "{label}: {field} is {value}, which the new guard would refuse"
                );
            }
            assert_eq!(cfg.out_hidden_size, 6144);
            assert_eq!(cfg.projector_hidden_size, 4096);
            // The real theta distribution: 39 rotated layers at 500000.0 and 13
            // NoPE zeros. The sign/finiteness clause must leave both alone.
            assert_eq!(
                t.layer_rope_theta
                    .iter()
                    .filter(|v| **v == 500_000.0)
                    .count(),
                39,
                "{label}: the rotated layers must survive the theta guard"
            );
            assert_eq!(
                t.layer_rope_theta.iter().filter(|v| **v == 0.0).count(),
                13,
                "{label}: the NoPE layers must survive the theta guard"
            );
        }
    }

    /// The vision tower's geometry is validated, not merely deserialized. Every
    /// field here is a divisor or a dimension of the patch grid, so a 0 surfaces
    /// as a division by zero or an empty tensor inside M2's tower — far from the
    /// config that carried it.
    ///
    /// Mutation caught: deleting the `raw.vision_config.validate()?` call, or
    /// dropping any single field from `validate`'s table. Every field is
    /// exercised, so dropping one is a red test rather than a silent gap.
    #[test]
    fn rejects_a_zero_in_any_vision_tower_dimension() {
        for (key, real) in [
            ("hidden_size", "1536"),
            ("num_hidden_layers", "50"),
            ("intermediate_size", "8960"),
            ("num_attention_heads", "16"),
            ("patch_size", "14"),
            ("merge_size", "2"),
            ("patch_temporal", "2"),
            ("pos_emb_height", "32"),
            ("pos_emb_width", "32"),
        ] {
            let good = config_json(&text_config_json(52));
            let needle = format!("\"{key}\":{real}");
            assert!(
                good.contains(&needle),
                "the fixture must spell {key} as {needle:?} for this substitution to bite"
            );
            let bad = good.replace(&needle, &format!("\"{key}\":0"));
            let err = MuseGlimmerConfig::from_json_str(&bad)
                .expect_err("a zero vision dimension must fail closed")
                .to_string();
            assert!(
                err.contains(&format!("vision_config.{key} must be non-zero")),
                "the error must name the vision field, got: {err}"
            );
        }
    }

    /// The tower has no `head_dim` key, so its head dim IS
    /// `hidden_size / num_attention_heads` (1536 / 16 = 96). A non-dividing pair
    /// truncates or panics on reshape inside the tower. Deliberately a DIFFERENT
    /// rule from the text decoder, where `head_dim` is independent and
    /// `32 x 128 != 6656` on purpose — see
    /// `accepts_a_head_dim_that_is_not_hidden_size_over_heads`.
    ///
    /// Mutation caught: deleting the divisibility check, or copying the text
    /// decoder's deliberate absence of one into the tower.
    #[test]
    fn rejects_a_vision_head_count_that_does_not_divide_its_hidden_size() {
        let bad = config_json(&text_config_json(52))
            .replace("\"num_attention_heads\":16", "\"num_attention_heads\":7");
        let err = MuseGlimmerConfig::from_json_str(&bad)
            .expect_err("1536 / 7 is not an integer head dim")
            .to_string();
        assert!(
            err.contains("must divide") && err.contains("1536"),
            "the error must name the quotient it could not form, got: {err}"
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
        // The four values that determine every byte of the KV cache, read from
        // the real file rather than from the fixture. Without these, a
        // transcription slip (or a differently-shaped Muse-Glimmer revision)
        // would leave the fixture AND `kv_cache.rs`'s hard-coded layout
        // literals agreeing on a model nobody runs: they were transcribed from
        // the same reading of this file, so they cross-check each other and
        // nothing else. `head_dim` 64 would halve the cache, kv_heads 4 would
        // double it, and a 4096 window would move the sliding admission bound
        // from 161 blocks to 289 — in all three cases with zero failing tests.
        assert_eq!(t.head_dim, 128);
        assert_eq!(t.num_key_value_heads, 2);
        assert_eq!(t.num_attention_heads, 32);
        assert_eq!(t.sliding_window, 2048);
        // The seam's max_model_len: 131072 => 8192 full-attention blocks at
        // block_size 16.
        assert_eq!(t.max_position_embeddings, 131072);
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
