//! Shared linear kernel helpers (K-quant mode interning).

use std::ffi::CStr;

/// Pre-interned static C-strings for every CUDA K-quant mode so hot-path
/// `QuantizedLinear::forward` performs zero per-op `CString` heap allocations.
#[inline]
pub(crate) fn intern_kquant_mode(mode: &str) -> Option<&'static CStr> {
    match mode {
        "q3k" => Some(c"q3k"),
        "q4k" => Some(c"q4k"),
        "q5k" => Some(c"q5k"),
        "q6k" => Some(c"q6k"),
        "iq4nl" => Some(c"iq4nl"),
        "iq4xs" => Some(c"iq4xs"),
        "q3k@t64" => Some(c"q3k@t64"),
        "q4k@t64" => Some(c"q4k@t64"),
        "q5k@t64" => Some(c"q5k@t64"),
        "q6k@t64" => Some(c"q6k@t64"),
        "iq4nl@t64" => Some(c"iq4nl@t64"),
        "iq4xs@t64" => Some(c"iq4xs@t64"),
        _ => None,
    }
}
