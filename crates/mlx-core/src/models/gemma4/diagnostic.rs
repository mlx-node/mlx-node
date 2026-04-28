//! Numerical-diff scaffolding for the Gemma4 paged-vs-flat parity bug.
//!
//! Gated entirely on the `MLX_DEBUG_GEMMA4_DUMP=1` env var. When unset
//! (production / CI default), every helper is a no-op and the model
//! never touches an MxArray for diagnostic purposes.
//!
//! When set, the layer-loop sites in `model.rs` (flat `forward_body` and
//! paged `run_paged_prefill_chunk` / `run_paged_decode_step`) call
//! `dump_norm()` at three points per layer:
//!
//! 1. `kind="hin"`   — input to attention (post `input_layernorm`).
//! 2. `kind="attn"`  — attention output (pre `post_attention_layernorm`).
//! 3. `kind="hout"`  — final layer output (after PLE + scalar tail).
//!
//! The model also threads its own dispatch path label ("flat" or "paged")
//! and the current step index (`step=-1` = prefill, `step=0..N` = decode
//! steps) through `set_path` / `set_step` before kicking off the layer
//! loop, so the resulting log lines can be diffed against each other to
//! localize the first divergent layer.
//!
//! Sample line:
//! ```text
//! [gemma4-dump] path=paged step=1 layer=4 kind=attn norm=0.000123 shape=[1,1,2048]
//! ```

use crate::array::MxArray;
use std::cell::Cell;
use std::sync::OnceLock;

/// Cached env-var read. `OnceLock<bool>` so we hit the env table once per
/// process: tests fork-spawn so this is safe; production agents that do
/// not set the var pay one ENV lookup at first call and then `false`.
static DUMP_ENABLED: OnceLock<bool> = OnceLock::new();

/// Returns `true` iff `MLX_DEBUG_GEMMA4_DUMP=1`. Called at every
/// `dump_norm` site; the OnceLock keeps the cost to a single atomic
/// load on the steady state.
pub(crate) fn dump_enabled() -> bool {
    *DUMP_ENABLED.get_or_init(|| {
        std::env::var("MLX_DEBUG_GEMMA4_DUMP")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

thread_local! {
    /// Dispatch path tag — "flat" for `forward_body` and "paged" for
    /// the paged prefill/decode loops. Set at the top of each layer
    /// loop; included in every dump line so flat and paged dumps from
    /// the same prompt can be diff'd directly.
    static DUMP_PATH: Cell<&'static str> = const { Cell::new("?") };
    /// Current decode step index. `-1` = prefill chunk, `0..max_new_tokens`
    /// = decode steps. Set by the model layer-loop site.
    static DUMP_STEP: Cell<i32> = const { Cell::new(-2) };
    /// Current layer index — set by the model layer-loop site so
    /// `decoder_layer.rs` can dump from inside `forward`/`forward_shared`/
    /// `forward_paged_or_flat` without threading the index through the
    /// call signatures.
    static DUMP_LAYER: Cell<i32> = const { Cell::new(-1) };
}

/// Set the dispatch path tag. Caller passes a `'static` literal.
pub(crate) fn set_path(path: &'static str) {
    if dump_enabled() {
        DUMP_PATH.with(|c| c.set(path));
    }
}

/// Set the current decode step. `-1` for prefill, `0..` for decode.
pub(crate) fn set_step(step: i32) {
    if dump_enabled() {
        DUMP_STEP.with(|c| c.set(step));
    }
}

/// Set the current layer index, read inside `decoder_layer.rs` dumps.
pub(crate) fn set_layer(layer_idx: usize) {
    if dump_enabled() {
        DUMP_LAYER.with(|c| c.set(layer_idx as i32));
    }
}

/// Dump using the thread-local layer index set via `set_layer`.
pub(crate) fn dump_norm_current_layer(kind: &str, arr: &MxArray, extra: Option<&str>) {
    if !dump_enabled() {
        return;
    }
    let layer = DUMP_LAYER.with(|c| c.get()) as usize;
    dump_norm(layer, kind, arr, extra);
}

/// Dump the L2 norm of `arr` to stderr. No-op when the env var is unset.
///
/// `kind` should be one of "hin" / "attn" / "hout" / "shared_keys" so
/// the dumps can be grouped per layer. `extra` lets the caller append
/// additional context (e.g. the anchor's pool offset) to the line.
pub(crate) fn dump_norm(layer_idx: usize, kind: &str, arr: &MxArray, extra: Option<&str>) {
    if !dump_enabled() {
        return;
    }

    // Compute ||arr||_2 := sqrt(sum(square(arr_f32))). Best-effort: any
    // failure here is silently swallowed so a diagnostic op can't crash
    // a real generation run when someone leaves the env var on.
    let norm_str = match compute_l2_norm(arr) {
        Ok(v) => format!("{:.6e}", v),
        Err(e) => format!("err({e})"),
    };

    let shape_str = match arr.shape() {
        Ok(buf) => format!("{:?}", buf.to_vec()),
        Err(_) => "?".to_string(),
    };

    let path = DUMP_PATH.with(|c| c.get());
    let step = DUMP_STEP.with(|c| c.get());
    let extra_str = extra.unwrap_or("");
    eprintln!(
        "[gemma4-dump] path={path} step={step} layer={layer_idx} kind={kind} norm={norm_str} shape={shape_str}{extra_str}"
    );
}

/// Best-effort L2 norm. Casts to f32 first so bf16/f16 tensors don't
/// overflow `square()`. Reduces over all axes by enumerating the
/// dimensions explicitly (an empty `axes` slice is interpreted as
/// "no reduction").
fn compute_l2_norm(arr: &MxArray) -> Result<f32, String> {
    use crate::array::DType;

    let casted = arr
        .astype(DType::Float32)
        .map_err(|e| format!("astype: {e:?}"))?;
    let squared = casted.square().map_err(|e| format!("square: {e:?}"))?;

    let nd = casted.ndim().map_err(|e| format!("ndim: {e:?}"))? as i32;
    let axes: Vec<i32> = (0..nd).collect();
    let summed = squared
        .sum(Some(&axes), Some(false))
        .map_err(|e| format!("sum: {e:?}"))?;
    let rooted = summed.sqrt().map_err(|e| format!("sqrt: {e:?}"))?;
    rooted.eval();
    rooted
        .item_at_float32(0)
        .map_err(|e| format!("item: {e:?}"))
}

/// Dump the top-5 logits + their indices from a `[V]` or `[1, V]` or
/// `[1, 1, V]` logit tensor. Used at the prefill→decode boundary and at
/// each decode step to compare the final argmax-determining values
/// between flat and paged paths.
pub(crate) fn dump_logits(tag: &str, logits: &MxArray) {
    if !dump_enabled() {
        return;
    }
    let path = DUMP_PATH.with(|c| c.get());
    let step = DUMP_STEP.with(|c| c.get());
    match top_k_logits(logits, 5) {
        Ok(pairs) => {
            let entries: Vec<String> = pairs
                .iter()
                .map(|(idx, v)| format!("{idx}={v:.6e}"))
                .collect();
            eprintln!(
                "[gemma4-logits] path={path} step={step} tag={tag} top5=[{}]",
                entries.join(",")
            );
        }
        Err(e) => {
            eprintln!("[gemma4-logits] path={path} step={step} tag={tag} err={e}");
        }
    }
}

/// Returns the top-k logit (index, value) pairs sorted by value desc.
fn top_k_logits(logits: &MxArray, k: usize) -> Result<Vec<(i64, f32)>, String> {
    use crate::array::DType;

    let shape = logits.shape().map_err(|e| format!("shape: {e:?}"))?;
    let dims = shape.to_vec();
    // Flatten to 1-D vocab vector.
    let total: i64 = dims.iter().copied().product::<i64>().max(1);
    let flat = logits
        .reshape(&[total])
        .map_err(|e| format!("reshape: {e:?}"))?;
    let casted = flat
        .astype(DType::Float32)
        .map_err(|e| format!("astype: {e:?}"))?;
    casted.eval();

    // Pull the entire vector to host. Vocab ~256K * 4B = ~1MB which is
    // acceptable for a debug-only env-gated path.
    let vocab = total as usize;
    let mut buf: Vec<f32> = Vec::with_capacity(vocab);
    for i in 0..vocab {
        let v = casted
            .item_at_float32(i)
            .map_err(|e| format!("item({i}): {e:?}"))?;
        buf.push(v);
    }
    let mut indexed: Vec<(i64, f32)> = buf
        .into_iter()
        .enumerate()
        .map(|(i, v)| (i as i64, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    Ok(indexed)
}
