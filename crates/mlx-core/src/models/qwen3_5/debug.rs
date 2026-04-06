use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use crate::array::MxArray;

#[cfg(target_family = "wasm")]
const MAX_LOGS_PER_LABEL: usize = 2;
#[cfg(target_family = "wasm")]
const SAMPLE_VALUES: usize = 8;
#[cfg(target_family = "wasm")]
const TOPK_VALUES: usize = 8;

#[cfg(target_family = "wasm")]
static LOG_COUNTS: LazyLock<Mutex<HashMap<String, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[cfg(target_family = "wasm")]
fn should_log(_label: &str) -> bool {
    false
}

#[cfg(target_family = "wasm")]
pub(crate) fn heap_probe(label: &'static str) {
    eprintln!("[QWEN35_HEAP] probe.start {label}");
    let ok = unsafe { mlx_sys::mlx_heap_probe() };
    eprintln!("[QWEN35_HEAP] probe.end {label} ok={ok}");
}

#[cfg(target_family = "wasm")]
fn collect_shape(arr: &MxArray) -> napi::bindgen_prelude::Result<Vec<i64>> {
    let ndim = arr.ndim()? as usize;
    let mut shape = Vec::with_capacity(ndim);
    for axis in 0..ndim {
        shape.push(arr.shape_at(axis as u32)?);
    }
    Ok(shape)
}

#[cfg(target_family = "wasm")]
fn summarize_values(values: &[f32]) -> String {
    let mut finite = 0usize;
    let mut nan = 0usize;
    let mut inf = 0usize;
    let mut near_zero = 0usize;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;

    for &value in values {
        if value.is_nan() {
            nan += 1;
            continue;
        }
        if !value.is_finite() {
            inf += 1;
            continue;
        }
        finite += 1;
        if value.abs() <= 1e-30 {
            near_zero += 1;
        }
        min = min.min(value);
        max = max.max(value);
        let v = value as f64;
        sum += v;
        sum_sq += v * v;
    }

    let sample = values
        .iter()
        .take(SAMPLE_VALUES)
        .map(|v| format!("{v:.6e}"))
        .collect::<Vec<_>>()
        .join(", ");

    if finite == 0 {
        return format!(
            "finite=0/{} nan={} inf={} near_zero={} samples=[{}]",
            values.len(),
            nan,
            inf,
            near_zero,
            sample
        );
    }

    let mean = sum / finite as f64;
    let variance = (sum_sq / finite as f64) - (mean * mean);
    let std = variance.max(0.0).sqrt();
    format!(
        "finite={}/{} nan={} inf={} near_zero={} min={:.6e} max={:.6e} mean={:.6e} std={:.6e} samples=[{}]",
        finite,
        values.len(),
        nan,
        inf,
        near_zero,
        min,
        max,
        mean,
        std,
        sample
    )
}

#[cfg(target_family = "wasm")]
pub(crate) fn log_tensor_stats(label: &str, arr: &MxArray) {
    if !should_log(label) {
        return;
    }

    let result = (|| -> napi::bindgen_prelude::Result<()> {
        arr.eval();
        let shape = collect_shape(arr)?;
        let dtype = arr.dtype()?;
        let values = arr.to_float32()?.to_vec();
        eprintln!(
            "[QWEN35_DEBUG] {} shape={:?} dtype={:?} {}",
            label,
            shape,
            dtype,
            summarize_values(&values)
        );
        Ok(())
    })();

    if let Err(err) = result {
        eprintln!("[QWEN35_DEBUG] {} logging failed: {}", label, err);
    }
}

#[cfg(target_family = "wasm")]
pub(crate) fn log_logits(label: &str, logits: &MxArray) {
    if !should_log(label) {
        return;
    }

    let result = (|| -> napi::bindgen_prelude::Result<()> {
        logits.eval();
        let shape = collect_shape(logits)?;
        let dtype = logits.dtype()?;
        let values = logits.to_float32()?.to_vec();
        let mut top = values
            .iter()
            .copied()
            .enumerate()
            .collect::<Vec<(usize, f32)>>();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        top.truncate(TOPK_VALUES);
        let topk = top
            .iter()
            .map(|(idx, value)| format!("{idx}:{value:.6e}"))
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!(
            "[QWEN35_DEBUG] {} shape={:?} dtype={:?} {} topk=[{}]",
            label,
            shape,
            dtype,
            summarize_values(&values),
            topk
        );
        Ok(())
    })();

    if let Err(err) = result {
        eprintln!("[QWEN35_DEBUG] {} logging failed: {}", label, err);
    }
}

#[cfg(not(target_family = "wasm"))]
pub(crate) fn log_tensor_stats(_: &str, _: &MxArray) {}

#[cfg(not(target_family = "wasm"))]
pub(crate) fn log_logits(_: &str, _: &MxArray) {}

#[cfg(not(target_family = "wasm"))]
pub(crate) fn heap_probe(_: &'static str) {}
