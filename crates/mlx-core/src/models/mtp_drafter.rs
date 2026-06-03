//! Shared detection + key-normalization for the mlx-vlm `qwen3_5_mtp`
//! split MTP-drafter directory.
//!
//! `mlx convert --q-mtp split` (Phase 1) emits a checkpoint laid out as a
//! BODY (no `mtp.*` tensors) plus a sibling `mtp-drafter/` subdirectory in
//! mlx-vlm's `qwen3_5_mtp` drafter format:
//!
//! ```text
//! <checkpoint>/
//!   model-0000N-of-...safetensors   (body, NO mtp.* keys)
//!   model.safetensors.index.json
//!   config.json                     (model_type: qwen3_5 / qwen3_5_moe)
//!   mtp-drafter/
//!     model.safetensors             (BARE keys: fc.weight, norm.weight,
//!                                     layers.{i}.self_attn.q_proj.weight, ...,
//!                                     dense layers.{i}.mlp.{gate,up,down}_proj.weight
//!                                     OR MoE layers.{i}.mlp.switch_mlp.* + .gate)
//!     config.json                   (model_type: qwen3_5_mtp, text_config, block_size)
//!     <tokenizer files>
//! ```
//!
//! The drafter `model.safetensors` carries `{"format":"mlx"}` metadata and the
//! +1.0 RMSNorm shift ALREADY baked in (byte-identical to the inline `mtp.*`
//! tensors emitted by the non-split convert). The keys are BARE — they drop the
//! `mtp.` prefix the engine's MTP head modules key off (see
//! `qwen3_5/mtp.rs::apply_weights` / `qwen3_5_moe/mtp.rs::apply_weights`, both
//! of which read strictly `mtp.*`).
//!
//! This module gives BOTH the dense and MoE loaders a single place to:
//!   1. detect the drafter directory ([`detect_drafter_safetensors`]);
//!   2. re-add the `mtp.` prefix to every bare key
//!      ([`reprefix_drafter_key`]) so the existing key handling +
//!      head modules resolve unchanged ([`load_drafter_tensors`]).
//!
//! The +1.0 shift is NOT re-applied here: the on-disk values are already in
//! final (≈1.0) form, and the load-side `mtp_norms_need_shift` probe in each
//! `sanitize_weights` samples `mtp.layers.0.input_layernorm.weight` (≈1.04 for
//! a converted checkpoint) → probe is false → no second shift.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use napi::bindgen_prelude::*;
use serde_json::Value;
use tracing::{info, warn};

use crate::array::{DType, MxArray};
use crate::utils::safetensors::load_safetensors_lazy;

/// `model_type` value that identifies an mlx-vlm split MTP drafter `config.json`.
pub const DRAFTER_MODEL_TYPE: &str = "qwen3_5_mtp";

/// The conventional drafter subdirectory name emitted by `mlx convert --q-mtp split`.
const DRAFTER_SUBDIR: &str = "mtp-drafter";

/// The single-file drafter weights name (mlx-vlm `qwen3_5_mtp` format).
const DRAFTER_WEIGHTS: &str = "model.safetensors";

/// Which backbone a drafter is being loaded for. Dense and MoE bodies require
/// structurally different MTP layer weights (dense `mlp.{gate,up,down}_proj`
/// vs MoE `mlp.switch_mlp.{gate,up,down}_proj` + `mlp.gate`), and the drafter's
/// sibling `config.json` `text_config.model_type` discriminates the two
/// (`...moe...` for MoE, no `moe` for dense). A mismatch — e.g. a dense drafter
/// dropped beside a MoE body — must be rejected so the loader never activates a
/// structurally-incompatible head as if it were valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrafterBodyVariant {
    /// Dense Qwen3.5 backbone (`qwen3_5` / `qwen3_5_text`).
    Dense,
    /// MoE Qwen3.5 backbone (`qwen3_5_moe` / `qwen3_5_moe_text`).
    Moe,
}

impl DrafterBodyVariant {
    fn is_moe(self) -> bool {
        matches!(self, DrafterBodyVariant::Moe)
    }
}

/// Per-MTP-layer attention/norm weights required by BOTH dense and MoE heads.
/// Mirrors the load-side `set_*` calls in `qwen3_5/mtp.rs` and
/// `qwen3_5_moe/mtp.rs::apply_weights`.
const REQUIRED_PER_LAYER_COMMON: [&str; 8] = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
];

/// Dense-only per-layer MLP linears.
const REQUIRED_PER_LAYER_DENSE_MLP: [&str; 3] = [
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
];

/// MoE-only per-layer expert + router + shared-expert weights.
///
/// The shared expert is UNCONDITIONAL in this codebase: `SparseMoeBlock::new`
/// / `new_quantized` always build `shared_expert` + `shared_expert_gate`, and
/// both `forward` and `Qwen3_5MoeMTPModule::apply_weights`/`get_parameters`
/// always consume/emit them. So a MoE-flavored MTP layer requires the four
/// expert/router keys AND the four shared-expert keys — a checkpoint missing
/// only the shared-expert tensors would otherwise pass this gate and load
/// random shared-expert weights, corrupting speculative decode.
const REQUIRED_PER_LAYER_MOE_MLP: [&str; 8] = [
    "mlp.switch_mlp.gate_proj.weight",
    "mlp.switch_mlp.up_proj.weight",
    "mlp.switch_mlp.down_proj.weight",
    "mlp.gate.weight",
    "mlp.shared_expert.gate_proj.weight",
    "mlp.shared_expert.up_proj.weight",
    "mlp.shared_expert.down_proj.weight",
    "mlp.shared_expert_gate.weight",
];

/// Per-MTP-layer QUANTIZABLE linear projections for a **MoE-flavored** MTP
/// layer (bare `<suffix>`, NO `.weight`). This is the single source of truth
/// for "which MoE MTP linears participate in quantization" shared by:
///   * the convert-side quant policy
///     ([`crate::convert::is_mtp_layer_quantizable_prefix`]), which decides
///     WHICH `mtp.layers.{i}.<suffix>.weight` tensors get packed to INT4, and
///   * the MoE load-side quant-metadata augmentation
///     (`qwen3_5_moe/persistence.rs::augment_mtplx_mtp_quantization_moe`), which
///     records the matching per-layer-quant `(bits, group_size, mode)` so the
///     reload resolves the exact same PLQ the convert chose.
///
/// Keeping produce + reload off ONE list means adding/removing a quantizable
/// MoE MTP linear updates both paths atomically (the drift-guard test below
/// ties the MLP entries to [`REQUIRED_PER_LAYER_MOE_MLP`]). The four attention
/// projections are shared verbatim with the dense set
/// ([`crate::models::qwen3_5::persistence::MTP_LAYER_LINEAR_SUFFIXES`]); the MLP
/// entries are the MoE expert/router/shared-expert linears.
pub(crate) const MTP_MOE_LAYER_LINEAR_SUFFIXES: [&str; 12] = [
    // Attention (same as dense MTP).
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    // MoE MLP: experts, router gate, shared expert + its gate.
    "mlp.switch_mlp.gate_proj",
    "mlp.switch_mlp.up_proj",
    "mlp.switch_mlp.down_proj",
    "mlp.gate",
    "mlp.shared_expert.gate_proj",
    "mlp.shared_expert.up_proj",
    "mlp.shared_expert.down_proj",
    "mlp.shared_expert_gate",
];

/// Top-level MTP head weights required by BOTH variants.
const REQUIRED_TOP_LEVEL: [&str; 4] = [
    "mtp.fc.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
];

/// Re-add the `mtp.` prefix to a bare drafter key so the existing MTP head
/// modules (which read strictly `mtp.*`) and the loader's `mtp.*`-keyed
/// sanitize handling resolve unchanged.
///
/// The drafter file stores keys WITHOUT the `mtp.` prefix (e.g. `fc.weight`,
/// `layers.0.self_attn.q_proj.weight`, `layers.0.mlp.switch_mlp.gate_proj.weight`).
/// This is the inverse of convert's `extract_mtp_drafter_tensors`, which strips
/// the prefix before writing.
///
/// Idempotent + prefix-tolerant: a key that ALREADY carries `mtp.` (or any of
/// the wrapper prefixes the loaders strip, e.g. `model.`, `language_model.`,
/// `model.language_model.`, `language_model.model.`, or the longest
/// `model.language_model.model.`) resolves to the canonical `mtp.<bare>` form
/// rather than double-prefixing. The wrapper list + longest-first order are kept
/// in lockstep with the authoritative `normalize_mtp_prefix` in `convert.rs`.
pub fn reprefix_drafter_key(key: &str) -> String {
    // Delegate to the shared `strip_wrapper_prefix` so this stays byte-for-byte
    // in lockstep with `convert.rs::normalize_mtp_prefix` and the loader body
    // strips (one authoritative wrapper list + order).
    let stripped = strip_wrapper_prefix(key);

    if stripped.starts_with("mtp.") {
        // Already canonical (or carried mtp. inside a wrapper prefix).
        stripped.to_string()
    } else {
        format!("mtp.{stripped}")
    }
}

/// Strip the HF VLM wrapper prefixes from a weight key, longest-first.
///
/// This is the SINGLE authoritative wrapper-prefix chain shared by
/// `convert.rs::normalize_mtp_prefix`, [`reprefix_drafter_key`], and the dense
/// and MoE persistence body/MTP strips. The wrapper list and order MUST be kept
/// identical across all of them so a key normalizes to the same bare form on
/// every path.
///
/// ORDER IS LOAD-BEARING: the longest wrapper (`model.language_model.model.`)
/// MUST be tried before the shorter overlapping `model.language_model.`. Per
/// PR #65 review (Cursor Bugbot), omitting the longest variant let the shorter
/// strip fire on `model.language_model.model.mtp.fc.weight`, leaving
/// `model.mtp.fc.weight` — which doesn't start with `mtp.` — so the key was
/// silently dropped (or wrongly re-prefixed). Longest-first is strictly safe
/// for the generic body strips too: a longer match is only attempted before
/// the shorter ones, so the existing double-wrap `model.language_model.`
/// handling is preserved.
pub(crate) fn strip_wrapper_prefix(key: &str) -> &str {
    key.strip_prefix("model.language_model.model.")
        .or_else(|| key.strip_prefix("model.language_model."))
        .or_else(|| key.strip_prefix("language_model.model."))
        .or_else(|| key.strip_prefix("language_model."))
        .or_else(|| key.strip_prefix("model."))
        .unwrap_or(key)
}

/// Resolve the drafter `model.safetensors` for a model directory, if a valid
/// mlx-vlm `qwen3_5_mtp` drafter is present.
///
/// Detection precedence (returns the FIRST match):
///   1. `<model_dir>/mtp-drafter/model.safetensors` with a sibling
///      `mtp-drafter/config.json` whose `model_type == "qwen3_5_mtp"`.
///   2. A sibling directory passed/derived as a draft path:
///      `<model_dir>/../<dirname>-mtp/model.safetensors` with a sibling
///      `config.json` `model_type == "qwen3_5_mtp"`.
///
/// A directory that exists but whose `config.json` is missing / unreadable / not
/// `qwen3_5_mtp` is skipped with a warning rather than erroring — the caller
/// then falls back to inline / legacy-sidecar discovery. Returns `None` when no
/// valid drafter is found (the common backward-compat case for inline
/// checkpoints).
pub fn detect_drafter_safetensors(model_dir: &Path) -> Option<PathBuf> {
    // 1. Conventional `<model_dir>/mtp-drafter/`.
    let primary = model_dir.join(DRAFTER_SUBDIR);
    if let Some(found) = drafter_dir_if_valid(&primary) {
        return Some(found);
    }

    // 2. Sibling `<model_dir>/../<name>-mtp/`.
    if let (Some(parent), Some(name)) = (model_dir.parent(), model_dir.file_name()) {
        let sibling_name = format!("{}-mtp", name.to_string_lossy());
        let sibling = parent.join(sibling_name);
        // Don't treat the model dir itself as its own drafter.
        if sibling != model_dir
            && let Some(found) = drafter_dir_if_valid(&sibling)
        {
            return Some(found);
        }
    }

    None
}

/// Validate a candidate drafter directory and return its `model.safetensors`
/// path when it is a well-formed mlx-vlm `qwen3_5_mtp` drafter.
///
/// Requires BOTH a `model.safetensors` weights file AND a `config.json` whose
/// `model_type == "qwen3_5_mtp"`. The `config.json` gate is the authoritative
/// signal Phase 1 writes; a stray directory without it is ignored.
fn drafter_dir_if_valid(dir: &Path) -> Option<PathBuf> {
    if !dir.is_dir() {
        return None;
    }
    let weights = dir.join(DRAFTER_WEIGHTS);
    if !weights.is_file() {
        return None;
    }
    let config_path = dir.join("config.json");
    if !config_path.is_file() {
        warn!(
            "MTP drafter directory {} has no config.json; ignoring (cannot confirm model_type=={})",
            dir.display(),
            DRAFTER_MODEL_TYPE
        );
        return None;
    }
    let model_type = match std::fs::read_to_string(&config_path) {
        Ok(data) => match serde_json::from_str::<Value>(&data) {
            Ok(v) => v
                .get("model_type")
                .and_then(|m| m.as_str())
                .map(|s| s.to_string()),
            Err(e) => {
                warn!(
                    "MTP drafter config {} is not valid JSON ({}); ignoring",
                    config_path.display(),
                    e
                );
                return None;
            }
        },
        Err(e) => {
            warn!(
                "MTP drafter config {} could not be read ({}); ignoring",
                config_path.display(),
                e
            );
            return None;
        }
    };
    match model_type.as_deref() {
        Some(DRAFTER_MODEL_TYPE) => Some(weights),
        other => {
            warn!(
                "MTP drafter directory {} has model_type={:?}, expected {:?}; ignoring",
                dir.display(),
                other,
                DRAFTER_MODEL_TYPE
            );
            None
        }
    }
}

/// Read the drafter's sibling `config.json` `text_config.model_type` and confirm
/// it matches the body backbone (`...moe...` ⇔ MoE, otherwise dense).
///
/// `detect_drafter_safetensors` only gates on the top-level `model_type ==
/// "qwen3_5_mtp"` — which is identical for dense and MoE drafters — so the
/// `text_config.model_type` is the only on-disk discriminator. A dense drafter
/// dropped beside a MoE body (or vice-versa) would otherwise re-prefix cleanly
/// but be structurally incompatible (missing `switch_mlp.*` / `mlp.gate`, or
/// carrying them when the body wants dense `mlp.*_proj`). Returns an error
/// describing the mismatch so the loader refuses to merge.
///
/// A `text_config` that lacks `model_type` is tolerated (treated as "unknown",
/// not a mismatch) — the structural key-completeness check below is the
/// authoritative gate in that case.
fn validate_drafter_text_config_variant(
    safetensors_path: &Path,
    body: DrafterBodyVariant,
) -> Result<()> {
    let config_path = match safetensors_path.parent() {
        Some(dir) => dir.join("config.json"),
        None => return Ok(()),
    };
    // `detect_drafter_safetensors` already required config.json + model_type; if
    // it has gone missing between detect and load just fall through to the
    // structural check rather than hard-erroring on a transient race.
    let data = match std::fs::read_to_string(&config_path) {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let cfg: Value = match serde_json::from_str(&data) {
        Ok(v) => v,
        Err(_) => return Ok(()),
    };
    let text_model_type = cfg
        .get("text_config")
        .and_then(|tc| tc.get("model_type"))
        .and_then(|m| m.as_str());
    if let Some(mt) = text_model_type {
        let drafter_is_moe = mt.contains("moe");
        if drafter_is_moe != body.is_moe() {
            return Err(Error::from_reason(format!(
                "MTP drafter {} declares text_config.model_type={:?} ({} backbone) but the body \
                 is a {} backbone; refusing to load a structurally-incompatible MTP head. \
                 Re-run `mlx convert --q-mtp split` against the matching base model.",
                config_path.display(),
                mt,
                if drafter_is_moe { "MoE" } else { "dense" },
                if body.is_moe() { "MoE" } else { "dense" },
            )));
        }
    }
    Ok(())
}

/// Confirm a canonical `mtp.*`-keyed param map contains EVERY weight the
/// configured MTP head will read for `n_mtp_layers` layers. A partial set
/// (truncated drafter file, wrong-variant MLP keys, missing top-level
/// `mtp.fc`/norms, or a truncated inline checkpoint) would otherwise leave the
/// head modules with default-initialized (garbage) weights while the model still
/// reports `has_mtp_weights() == true`, silently corrupting speculative decode.
/// Returns the list of missing canonical `mtp.*` keys (empty ⇒ complete).
///
/// Shared by the drafter-merge gate ([`load_drafter_tensors`]) and the MoE
/// loader's post-sanitize completeness gate, so the dense and MoE backbones use
/// one source of truth for "what a complete MTP head looks like".
///
/// DTYPE-AWARE for quantized companions: for every required `.weight` key, if
/// the present array's dtype is `Uint32` (an affine/MXFP/NVFP packed weight),
/// the sibling `.scales` key is ALSO required — mirroring the dense loader's
/// `require_mtp_linear` (`qwen3_5/persistence.rs`). A packed `Uint32` `.weight`
/// without its `.scales` would otherwise load as silent garbage (MoE
/// `SwitchLinear::set_weight` performs no shape/dtype check). For bf16/f16
/// `.weight`s no `.scales` is required, so plain dense checkpoints are
/// unaffected. If a `.weight`'s dtype cannot be read, it is treated
/// conservatively as non-quantized (no extra `.scales` requirement) rather than
/// erroring — matching the dense loader's handling.
pub(crate) fn missing_required_mtp_keys(
    params: &HashMap<String, MxArray>,
    body: DrafterBodyVariant,
    n_mtp_layers: i32,
) -> Vec<String> {
    let mut missing = Vec::new();
    // For a required `<base>.weight` key: flag the `.weight` if absent;
    // otherwise, if the present array is a packed `Uint32` quantized weight,
    // ALSO require its sibling `<base>.scales` (mirrors the dense loader's
    // `require_mtp_linear`). A dtype that cannot be read is treated as
    // non-quantized (conservative, no extra requirement).
    let mut require_weight = |key: String| {
        let Some(weight) = params.get(&key) else {
            missing.push(key);
            return;
        };
        if matches!(weight.dtype(), Ok(DType::Uint32))
            && let Some(base) = key.strip_suffix(".weight")
        {
            let scales_key = format!("{base}.scales");
            if !params.contains_key(&scales_key) {
                missing.push(scales_key);
            }
        }
    };

    for key in REQUIRED_TOP_LEVEL {
        require_weight(key.to_string());
    }
    let mlp_suffixes: &[&str] = if body.is_moe() {
        &REQUIRED_PER_LAYER_MOE_MLP
    } else {
        &REQUIRED_PER_LAYER_DENSE_MLP
    };
    for layer_idx in 0..n_mtp_layers.max(0) {
        let prefix = format!("mtp.layers.{layer_idx}");
        for suffix in REQUIRED_PER_LAYER_COMMON.iter().chain(mlp_suffixes.iter()) {
            require_weight(format!("{prefix}.{suffix}"));
        }
    }
    missing
}

/// Load every tensor from a drafter `model.safetensors`, re-adding the `mtp.`
/// prefix so the returned map slots directly into the loader's raw-params map
/// before `sanitize_weights`.
///
/// Two variants are taken because the BACKBONE flavor and the per-layer MLP
/// flavor can differ:
///   * `backbone` is the model family (`Dense` for a `qwen3_5` body, `Moe` for a
///     `qwen3_5_moe` body). It gates the drafter's sibling `config.json`
///     `text_config.model_type` (which encodes the body family the drafter was
///     emitted for: `...moe...` ⇔ MoE), so a dense drafter dropped beside a MoE
///     body (or vice-versa) is rejected.
///   * `mlp_flavor` is the MLP-key schema the MTP layer actually emits — derived
///     by the caller from `is_moe_layer(fa_idx)` (see
///     `Qwen3_5MoeMTPModule::mtp_mlp_variant`). A MoE backbone whose MTP layer
///     resolves to a DENSE-flavored layer (sparse step not dividing the
///     attention interval, or `fa_idx ∈ mlp_only_layers`) ships dense
///     `mlp.{gate,up,down}_proj` keys, NOT `switch_mlp.* + mlp.gate`. The
///     structural completeness gate MUST use `mlp_flavor`, not `backbone`, so it
///     mirrors what `get_parameters`/`apply_weights` produce. For a dense
///     backbone the two are always identical (`Dense`).
///
/// VALIDATION (both checks reject the merge with a hard error so a broken drafter
/// can never silently activate as a valid head):
///   1. the drafter's sibling `config.json` `text_config.model_type` must match
///      the `backbone` family (dense ⇔ no `moe`, MoE ⇔ contains `moe`);
///   2. the re-prefixed key set must be COMPLETE for `n_mtp_layers` (top-level
///      `mtp.fc`/norms + every per-layer attn/norm/MLP weight for `mlp_flavor`).
///
/// Returns `Ok(None)` when the file yields no tensors (so the caller can warn +
/// fall through rather than silently merging an empty drafter). Bare keys (the
/// normal case) gain the prefix; defensively wrapped or already-prefixed keys
/// normalize to the canonical `mtp.<bare>` form.
pub fn load_drafter_tensors(
    safetensors_path: &Path,
    backbone: DrafterBodyVariant,
    mlp_flavor: DrafterBodyVariant,
    n_mtp_layers: i32,
) -> Result<Option<HashMap<String, MxArray>>> {
    info!(
        "Loading split MTP drafter from: {} (mmap)",
        safetensors_path.display()
    );
    // Variant gate first — fail fast on an obviously-wrong drafter before mmap'ing.
    // This keys off the BACKBONE family (what the drafter's text_config records),
    // not the per-layer MLP flavor.
    validate_drafter_text_config_variant(safetensors_path, backbone)?;

    let raw = load_safetensors_lazy(safetensors_path)?;
    let source_count = raw.len();
    let mut normalized = HashMap::with_capacity(source_count);
    for (name, array) in raw {
        let key = reprefix_drafter_key(&name);
        normalized.insert(key, array);
    }
    if normalized.is_empty() {
        warn!(
            "Ignoring MTP drafter {} because it contained no tensors",
            safetensors_path.display()
        );
        return Ok(None);
    }

    // Structural completeness gate — reject a partial / wrong-variant drafter so
    // the head never loads with default-initialized weights. Keys off the
    // per-layer MLP flavor (which may be Dense even for a MoE backbone).
    let missing = missing_required_mtp_keys(&normalized, mlp_flavor, n_mtp_layers);
    if !missing.is_empty() {
        let shown = &missing[..missing.len().min(12)];
        return Err(Error::from_reason(format!(
            "MTP drafter {} is incomplete for a {} backbone ({}-flavored MTP MLP) with {} MTP \
             layer(s): missing {} required weight(s); first: {:?}. Refusing to load a partial MTP \
             head. Re-run `mlx convert --q-mtp split` against the matching base model.",
            safetensors_path.display(),
            if backbone.is_moe() { "MoE" } else { "dense" },
            if mlp_flavor.is_moe() { "MoE" } else { "dense" },
            n_mtp_layers.max(0),
            missing.len(),
            shown,
        )));
    }

    info!(
        "Loaded {} MTP tensors from drafter {} (re-prefixed mtp.*)",
        normalized.len(),
        safetensors_path.display()
    );
    Ok(Some(normalized))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// DRIFT GUARD: the convert-side quant-policy predicate
    /// ([`crate::convert::is_mtp_layer_quantizable_prefix`]) and the MoE
    /// load-side augmentation both walk [`MTP_MOE_LAYER_LINEAR_SUFFIXES`], while
    /// the load-completeness gate ([`missing_required_mtp_keys`]) walks
    /// [`REQUIRED_PER_LAYER_MOE_MLP`]. If the two lists drift, a convert could
    /// quantize a MoE MTP linear the loader never augments PLQ for (reload reads
    /// the wrong bits/group_size → corrupt head) — or the gate could require a
    /// `.scales` the convert never produced. Tie them at test time:
    ///   * every MoE-MLP suffix in `MTP_MOE_LAYER_LINEAR_SUFFIXES` (i.e. NOT a
    ///     shared `self_attn.*` projection), with `.weight` appended, is a member
    ///     of `REQUIRED_PER_LAYER_MOE_MLP`; and
    ///   * conversely every `REQUIRED_PER_LAYER_MOE_MLP` entry (minus `.weight`)
    ///     is present in `MTP_MOE_LAYER_LINEAR_SUFFIXES`.
    ///
    /// Adding a quantizable MoE MTP linear in one place therefore forces it in
    /// the other or this test fails.
    #[test]
    fn moe_mtp_linear_suffixes_match_required_moe_mlp() {
        // The four attention projections are shared with the dense set and are
        // NOT part of the MoE-MLP required set — strip them out.
        let moe_mlp_suffixes: Vec<&&str> = MTP_MOE_LAYER_LINEAR_SUFFIXES
            .iter()
            .filter(|s| !s.starts_with("self_attn."))
            .collect();

        // Forward direction: every MoE-MLP linear we QUANTIZE is REQUIRED at load.
        for suffix in &moe_mlp_suffixes {
            let weight_key = format!("{suffix}.weight");
            assert!(
                REQUIRED_PER_LAYER_MOE_MLP.contains(&weight_key.as_str()),
                "MoE MTP linear `{suffix}` (quantized by convert) is missing from \
                 REQUIRED_PER_LAYER_MOE_MLP — the load gate would not require its \
                 .scales, allowing a silently-garbage quantized reload",
            );
        }

        // Reverse direction: every REQUIRED MoE-MLP linear is QUANTIZED.
        for required in REQUIRED_PER_LAYER_MOE_MLP {
            let bare = required
                .strip_suffix(".weight")
                .expect("REQUIRED_PER_LAYER_MOE_MLP entries are `.weight` keys");
            assert!(
                MTP_MOE_LAYER_LINEAR_SUFFIXES.contains(&bare),
                "REQUIRED MoE MTP linear `{bare}` is absent from \
                 MTP_MOE_LAYER_LINEAR_SUFFIXES — convert would leave it bf16 while \
                 the load gate demands its .scales",
            );
        }

        // The attention prefixes must stay byte-identical to the dense set so a
        // dense-flavored MoE MTP layer (which uses the dense suffix list) and a
        // MoE-flavored one agree on attention quant.
        for attn in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ] {
            assert!(
                MTP_MOE_LAYER_LINEAR_SUFFIXES.contains(&attn),
                "MoE MTP suffix set must include attention projection `{attn}`",
            );
            assert!(
                crate::models::qwen3_5::persistence::MTP_LAYER_LINEAR_SUFFIXES.contains(&attn),
                "dense MTP suffix set must include attention projection `{attn}`",
            );
        }
    }

    #[test]
    fn reprefix_bare_keys() {
        assert_eq!(reprefix_drafter_key("fc.weight"), "mtp.fc.weight");
        assert_eq!(reprefix_drafter_key("norm.weight"), "mtp.norm.weight");
        assert_eq!(
            reprefix_drafter_key("pre_fc_norm_embedding.weight"),
            "mtp.pre_fc_norm_embedding.weight"
        );
        assert_eq!(
            reprefix_drafter_key("layers.0.self_attn.q_proj.weight"),
            "mtp.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            reprefix_drafter_key("layers.0.mlp.gate_proj.weight"),
            "mtp.layers.0.mlp.gate_proj.weight"
        );
        // MoE drafter key (already-stacked switch_mlp + router gate).
        assert_eq!(
            reprefix_drafter_key("layers.0.mlp.switch_mlp.gate_proj.weight"),
            "mtp.layers.0.mlp.switch_mlp.gate_proj.weight"
        );
        assert_eq!(
            reprefix_drafter_key("layers.0.mlp.gate.weight"),
            "mtp.layers.0.mlp.gate.weight"
        );
    }

    #[test]
    fn reprefix_is_idempotent_and_prefix_tolerant() {
        // Already canonical.
        assert_eq!(reprefix_drafter_key("mtp.fc.weight"), "mtp.fc.weight");
        // Wrapped + already mtp. inside.
        assert_eq!(
            reprefix_drafter_key("model.mtp.norm.weight"),
            "mtp.norm.weight"
        );
        assert_eq!(
            reprefix_drafter_key("language_model.mtp.fc.weight"),
            "mtp.fc.weight"
        );
        // Wrapped bare key gains the prefix after the wrapper strip.
        assert_eq!(
            reprefix_drafter_key("model.layers.0.input_layernorm.weight"),
            "mtp.layers.0.input_layernorm.weight"
        );
    }

    /// PR #65 review (Cursor Bugbot) regression: the longest wrapper
    /// `model.language_model.model.` must be stripped BEFORE the shorter
    /// overlapping `model.language_model.`. Previously the function omitted the
    /// longest variant, so the shorter strip fired first and left
    /// `model.mtp.fc.weight` — which doesn't start with `mtp.` — getting wrongly
    /// reprefixed to `mtp.model.mtp.fc.weight`. The wrapper set + longest-first
    /// order now mirror `normalize_mtp_prefix` in `convert.rs`.
    #[test]
    fn reprefix_strips_longest_wrapper_first() {
        // The exact reported bug case (already-mtp key under the longest wrapper).
        assert_eq!(
            reprefix_drafter_key("model.language_model.model.mtp.fc.weight"),
            "mtp.fc.weight"
        );
        // Bare key under the longest wrapper gains the prefix after the strip.
        assert_eq!(
            reprefix_drafter_key("model.language_model.model.layers.0.self_attn.q_proj.weight"),
            "mtp.layers.0.self_attn.q_proj.weight"
        );
        // Each wrapper variant `normalize_mtp_prefix` handles must normalize to
        // the canonical `mtp.<bare>` form here too (one regression case per
        // prefix, longest → shortest).
        assert_eq!(
            reprefix_drafter_key("model.language_model.norm.weight"),
            "mtp.norm.weight"
        );
        assert_eq!(
            reprefix_drafter_key("language_model.model.mtp.fc.weight"),
            "mtp.fc.weight"
        );
        assert_eq!(
            reprefix_drafter_key("language_model.model.norm.weight"),
            "mtp.norm.weight"
        );
        assert_eq!(
            reprefix_drafter_key("language_model.fc.weight"),
            "mtp.fc.weight"
        );
        assert_eq!(reprefix_drafter_key("model.fc.weight"), "mtp.fc.weight");
    }

    /// Direct coverage of the shared `strip_wrapper_prefix`: all five wrapper
    /// prefixes (longest → shortest), the triple-wrap raw-VLM case, and the
    /// no-prefix passthrough. This is the single authoritative chain delegated
    /// to by `convert.rs::normalize_mtp_prefix`, `reprefix_drafter_key`, and the
    /// dense/MoE persistence body + MTP strips.
    #[test]
    fn strip_wrapper_prefix_all_variants_longest_first() {
        // Triple-wrap (raw HF VLM checkpoint with inline MTP) — longest first.
        assert_eq!(
            strip_wrapper_prefix("model.language_model.model.mtp.fc.weight"),
            "mtp.fc.weight"
        );
        // The remaining four wrappers, longest → shortest.
        assert_eq!(
            strip_wrapper_prefix("model.language_model.mtp.fc.weight"),
            "mtp.fc.weight"
        );
        assert_eq!(
            strip_wrapper_prefix("language_model.model.mtp.fc.weight"),
            "mtp.fc.weight"
        );
        assert_eq!(
            strip_wrapper_prefix("language_model.mtp.fc.weight"),
            "mtp.fc.weight"
        );
        assert_eq!(strip_wrapper_prefix("model.mtp.fc.weight"), "mtp.fc.weight");
        // Generic (non-mtp) body key under the longest wrapper.
        assert_eq!(
            strip_wrapper_prefix("model.language_model.model.layers.0.self_attn.q_proj.weight"),
            "layers.0.self_attn.q_proj.weight"
        );
        // No wrapper → unchanged.
        assert_eq!(strip_wrapper_prefix("mtp.fc.weight"), "mtp.fc.weight");
        assert_eq!(strip_wrapper_prefix("fc.weight"), "fc.weight");
    }

    fn write_config(dir: &Path, model_type: &str) {
        fs::write(
            dir.join("config.json"),
            format!("{{\"model_type\": \"{model_type}\"}}"),
        )
        .expect("write config.json");
    }

    fn touch(path: &Path) {
        fs::write(path, b"").expect("touch file");
    }

    #[test]
    fn detect_primary_mtp_drafter_dir() {
        let tmp =
            std::env::temp_dir().join(format!("mtp_drafter_detect_primary_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let model = tmp.join("qwen-split");
        let drafter = model.join("mtp-drafter");
        fs::create_dir_all(&drafter).expect("mkdir drafter");
        touch(&drafter.join("model.safetensors"));
        write_config(&drafter, DRAFTER_MODEL_TYPE);

        let found = detect_drafter_safetensors(&model).expect("primary drafter detected");
        assert_eq!(found, drafter.join("model.safetensors"));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn detect_sibling_name_mtp_dir() {
        let tmp =
            std::env::temp_dir().join(format!("mtp_drafter_detect_sibling_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let model = tmp.join("qwen-body");
        fs::create_dir_all(&model).expect("mkdir model");
        let sibling = tmp.join("qwen-body-mtp");
        fs::create_dir_all(&sibling).expect("mkdir sibling");
        touch(&sibling.join("model.safetensors"));
        write_config(&sibling, DRAFTER_MODEL_TYPE);

        let found = detect_drafter_safetensors(&model).expect("sibling drafter detected");
        assert_eq!(found, sibling.join("model.safetensors"));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn detect_returns_none_for_plain_checkpoint() {
        let tmp =
            std::env::temp_dir().join(format!("mtp_drafter_detect_none_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let model = tmp.join("qwen-inline");
        fs::create_dir_all(&model).expect("mkdir model");
        // No mtp-drafter subdir, no sibling -mtp dir.
        assert!(detect_drafter_safetensors(&model).is_none());
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn detect_skips_wrong_model_type() {
        let tmp = std::env::temp_dir().join(format!(
            "mtp_drafter_detect_wrongtype_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&tmp);
        let model = tmp.join("qwen-split");
        let drafter = model.join("mtp-drafter");
        fs::create_dir_all(&drafter).expect("mkdir drafter");
        touch(&drafter.join("model.safetensors"));
        // Wrong model_type → must be ignored.
        write_config(&drafter, "qwen3_5");
        assert!(detect_drafter_safetensors(&model).is_none());
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn detect_skips_dir_without_config() {
        let tmp = std::env::temp_dir().join(format!(
            "mtp_drafter_detect_noconfig_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&tmp);
        let model = tmp.join("qwen-split");
        let drafter = model.join("mtp-drafter");
        fs::create_dir_all(&drafter).expect("mkdir drafter");
        touch(&drafter.join("model.safetensors"));
        // No config.json → cannot confirm model_type → ignored.
        assert!(detect_drafter_safetensors(&model).is_none());
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn detect_skips_dir_without_weights() {
        let tmp = std::env::temp_dir().join(format!(
            "mtp_drafter_detect_noweights_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&tmp);
        let model = tmp.join("qwen-split");
        let drafter = model.join("mtp-drafter");
        fs::create_dir_all(&drafter).expect("mkdir drafter");
        write_config(&drafter, DRAFTER_MODEL_TYPE);
        // model.safetensors missing → ignored.
        assert!(detect_drafter_safetensors(&model).is_none());
        let _ = fs::remove_dir_all(&tmp);
    }

    // ── Finding 2 regression: shared drafter validation ───────────────────

    fn scalar() -> MxArray {
        MxArray::scalar_float(1.0).expect("scalar array")
    }

    /// Build the COMPLETE canonical `mtp.*`-keyed param map for `body` with
    /// `n_layers` MTP layers — the exact set `missing_required_mtp_keys`
    /// requires. Tests then delete keys to simulate a partial / wrong-variant
    /// drafter.
    fn complete_mtp_keys(body: DrafterBodyVariant, n_layers: i32) -> HashMap<String, MxArray> {
        let mut m = HashMap::new();
        for key in REQUIRED_TOP_LEVEL {
            m.insert(key.to_string(), scalar());
        }
        let mlp: &[&str] = if body.is_moe() {
            &REQUIRED_PER_LAYER_MOE_MLP
        } else {
            &REQUIRED_PER_LAYER_DENSE_MLP
        };
        for i in 0..n_layers {
            let prefix = format!("mtp.layers.{i}");
            for suffix in REQUIRED_PER_LAYER_COMMON.iter().chain(mlp.iter()) {
                m.insert(format!("{prefix}.{suffix}"), scalar());
            }
        }
        m
    }

    #[test]
    fn complete_dense_drafter_keys_pass() {
        let params = complete_mtp_keys(DrafterBodyVariant::Dense, 1);
        assert!(missing_required_mtp_keys(&params, DrafterBodyVariant::Dense, 1).is_empty());
    }

    #[test]
    fn complete_moe_drafter_keys_pass() {
        let params = complete_mtp_keys(DrafterBodyVariant::Moe, 1);
        assert!(missing_required_mtp_keys(&params, DrafterBodyVariant::Moe, 1).is_empty());
    }

    #[test]
    fn dense_drafter_keys_rejected_for_moe_body() {
        // A dense drafter (mlp.{gate,up,down}_proj) lacks the MoE switch_mlp +
        // router gate weights a MoE body's head reads → must be flagged missing.
        let dense_params = complete_mtp_keys(DrafterBodyVariant::Dense, 1);
        let missing = missing_required_mtp_keys(&dense_params, DrafterBodyVariant::Moe, 1);
        assert!(!missing.is_empty(), "dense keys must be incomplete for MoE");
        assert!(
            missing
                .iter()
                .any(|k| k == "mtp.layers.0.mlp.switch_mlp.gate_proj.weight")
        );
        assert!(missing.iter().any(|k| k == "mtp.layers.0.mlp.gate.weight"));
    }

    #[test]
    fn partial_moe_drafter_missing_switch_mlp_rejected() {
        let mut params = complete_mtp_keys(DrafterBodyVariant::Moe, 1);
        params.remove("mtp.layers.0.mlp.switch_mlp.up_proj.weight");
        params.remove("mtp.layers.0.mlp.gate.weight");
        let missing = missing_required_mtp_keys(&params, DrafterBodyVariant::Moe, 1);
        assert_eq!(missing.len(), 2);
        assert!(missing.contains(&"mtp.layers.0.mlp.switch_mlp.up_proj.weight".to_string()));
        assert!(missing.contains(&"mtp.layers.0.mlp.gate.weight".to_string()));
    }

    /// Fix A regression: the shared expert is UNCONDITIONAL, so a MoE checkpoint
    /// missing ONLY a shared-expert tensor (switch_mlp + router gate all present)
    /// must still be flagged incomplete — otherwise the head loads random
    /// shared-expert weights and corrupts speculative decode.
    #[test]
    fn partial_moe_drafter_missing_shared_expert_rejected() {
        // (a) missing a shared_expert projection weight.
        let mut params = complete_mtp_keys(DrafterBodyVariant::Moe, 1);
        params.remove("mtp.layers.0.mlp.shared_expert.gate_proj.weight");
        let missing = missing_required_mtp_keys(&params, DrafterBodyVariant::Moe, 1);
        assert_eq!(missing.len(), 1, "got: {missing:?}");
        assert!(missing.contains(&"mtp.layers.0.mlp.shared_expert.gate_proj.weight".to_string()));

        // (b) missing the shared_expert_gate weight.
        let mut params = complete_mtp_keys(DrafterBodyVariant::Moe, 1);
        params.remove("mtp.layers.0.mlp.shared_expert_gate.weight");
        let missing = missing_required_mtp_keys(&params, DrafterBodyVariant::Moe, 1);
        assert_eq!(missing.len(), 1, "got: {missing:?}");
        assert!(missing.contains(&"mtp.layers.0.mlp.shared_expert_gate.weight".to_string()));
    }

    #[test]
    fn missing_top_level_mtp_fc_rejected() {
        let mut params = complete_mtp_keys(DrafterBodyVariant::Dense, 1);
        params.remove("mtp.fc.weight");
        let missing = missing_required_mtp_keys(&params, DrafterBodyVariant::Dense, 1);
        assert_eq!(missing, vec!["mtp.fc.weight".to_string()]);
    }

    /// Fix B regression: a packed `Uint32` required `.weight` with NO sibling
    /// `.scales` must be flagged (the missing `.scales`), mirroring the dense
    /// loader's `require_mtp_linear`. The MoE `SwitchLinear::set_weight` performs
    /// no shape/dtype check, so a quantized `.weight` without scales would load
    /// as silent garbage. Needs Metal to allocate the Uint32 array — skip when
    /// MLX/Metal is unavailable (same pattern as the other allocating tests).
    #[test]
    fn quantized_weight_missing_scales_rejected() {
        // A complete bf16 set, then swap one switch_mlp `.weight` for a packed
        // Uint32 array and omit its `.scales`.
        let mut params = complete_mtp_keys(DrafterBodyVariant::Moe, 1);
        let packed = match MxArray::zeros(&[2, 2], Some(DType::Uint32)) {
            Ok(a) => a,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("Metal") || msg.contains("device") {
                    eprintln!(
                        "skipping quantized_weight_missing_scales_rejected (Metal unavailable): {msg}"
                    );
                    return;
                }
                panic!("unexpected Uint32 zeros failure: {msg}");
            }
        };
        params.insert(
            "mtp.layers.0.mlp.switch_mlp.gate_proj.weight".to_string(),
            packed,
        );
        // No `...switch_mlp.gate_proj.scales` inserted.
        let missing = missing_required_mtp_keys(&params, DrafterBodyVariant::Moe, 1);
        assert!(
            missing.contains(&"mtp.layers.0.mlp.switch_mlp.gate_proj.scales".to_string()),
            "a packed Uint32 .weight with no .scales must require the .scales companion; got: {missing:?}"
        );
        // The `.weight` itself is present, so it must NOT be flagged.
        assert!(
            !missing.contains(&"mtp.layers.0.mlp.switch_mlp.gate_proj.weight".to_string()),
            "the present Uint32 .weight must not be flagged missing; got: {missing:?}"
        );
    }

    #[test]
    fn multi_layer_completeness_checks_every_layer() {
        // A 2-layer head that only ships layer 0 must flag every layer-1 key.
        let mut params = complete_mtp_keys(DrafterBodyVariant::Moe, 1);
        // Re-tag as if it were a 2-layer config: layer 1 keys are absent.
        for key in REQUIRED_TOP_LEVEL {
            params.entry(key.to_string()).or_insert_with(scalar);
        }
        let missing = missing_required_mtp_keys(&params, DrafterBodyVariant::Moe, 2);
        assert!(missing.iter().all(|k| k.starts_with("mtp.layers.1.")));
        // 8 common + 8 MoE MLP (switch_mlp.{gate,up,down} + gate + shared_expert.{gate,up,down}
        // + shared_expert_gate) = 16 per-layer keys for the missing layer.
        assert_eq!(missing.len(), 16);
    }

    fn write_text_config(dir: &Path, top_model_type: &str, text_model_type: &str) {
        fs::write(
            dir.join("config.json"),
            format!(
                "{{\"model_type\": \"{top_model_type}\", \
                 \"text_config\": {{\"model_type\": \"{text_model_type}\"}}}}"
            ),
        )
        .expect("write config.json");
    }

    #[test]
    fn variant_check_rejects_dense_drafter_for_moe_body() {
        let tmp =
            std::env::temp_dir().join(format!("mtp_drafter_variant_d2m_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("mkdir");
        touch(&tmp.join("model.safetensors"));
        write_text_config(&tmp, DRAFTER_MODEL_TYPE, "qwen3_5_text");
        let st = tmp.join("model.safetensors");
        let err = validate_drafter_text_config_variant(&st, DrafterBodyVariant::Moe)
            .expect_err("dense drafter must be rejected for a MoE body");
        assert!(err.reason.contains("dense backbone"));
        assert!(err.reason.contains("MoE backbone"));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn variant_check_rejects_moe_drafter_for_dense_body() {
        let tmp =
            std::env::temp_dir().join(format!("mtp_drafter_variant_m2d_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("mkdir");
        touch(&tmp.join("model.safetensors"));
        write_text_config(&tmp, DRAFTER_MODEL_TYPE, "qwen3_5_moe_text");
        let st = tmp.join("model.safetensors");
        let err = validate_drafter_text_config_variant(&st, DrafterBodyVariant::Dense)
            .expect_err("MoE drafter must be rejected for a dense body");
        assert!(err.reason.contains("MoE backbone"));
        assert!(err.reason.contains("dense backbone"));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn variant_check_accepts_matching_variants() {
        let tmp =
            std::env::temp_dir().join(format!("mtp_drafter_variant_ok_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("mkdir");
        touch(&tmp.join("model.safetensors"));
        let st = tmp.join("model.safetensors");

        write_text_config(&tmp, DRAFTER_MODEL_TYPE, "qwen3_5_text");
        assert!(validate_drafter_text_config_variant(&st, DrafterBodyVariant::Dense).is_ok());

        write_text_config(&tmp, DRAFTER_MODEL_TYPE, "qwen3_5_moe_text");
        assert!(validate_drafter_text_config_variant(&st, DrafterBodyVariant::Moe).is_ok());
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn variant_check_tolerates_missing_text_model_type() {
        // No text_config.model_type → unknown → not a mismatch (structural
        // completeness check is the authoritative gate in that case).
        let tmp =
            std::env::temp_dir().join(format!("mtp_drafter_variant_none_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("mkdir");
        touch(&tmp.join("model.safetensors"));
        fs::write(
            tmp.join("config.json"),
            format!("{{\"model_type\": \"{DRAFTER_MODEL_TYPE}\"}}"),
        )
        .expect("write config.json");
        let st = tmp.join("model.safetensors");
        assert!(validate_drafter_text_config_variant(&st, DrafterBodyVariant::Dense).is_ok());
        assert!(validate_drafter_text_config_variant(&st, DrafterBodyVariant::Moe).is_ok());
        let _ = fs::remove_dir_all(&tmp);
    }
}
