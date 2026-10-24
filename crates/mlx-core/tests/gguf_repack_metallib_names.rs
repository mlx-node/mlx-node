//! The GGUF repack kernels run only from the prebuilt `paged_attn.metallib`;
//! there is no JIT fallback. Two checks, each both ways:
//!
//! - the C++ kernel table (`mlx_gguf_repack.cpp`) against the library's
//!   `mlx_node_gguf_repack_` functions;
//! - the C++ kernel table against `GGUF_REPACK_KERNEL_MARKERS` in
//!   `packages/core/metallib-select.ts`, which the package build uses to reject
//!   a stale metallib.
//!
//!   cargo test -p mlx-core --test gguf_repack_metallib_names

#![cfg(target_os = "macos")]

use std::collections::BTreeSet;
use std::ffi::{CStr, c_char};

const METALLIB_SELECT_TS: &str = include_str!("../../../packages/core/metallib-select.ts");

fn table_names() -> BTreeSet<String> {
    // SAFETY: a null buffer with capacity 0 only asks for the length.
    let len = unsafe { mlx_sys::mlx_gguf_repack_kernel_names(std::ptr::null_mut(), 0) };
    let mut buf = vec![0 as c_char; len + 1];
    // SAFETY: `buf` is writable for `buf.len()` bytes, more than `len`.
    let written = unsafe { mlx_sys::mlx_gguf_repack_kernel_names(buf.as_mut_ptr(), buf.len()) };
    assert_eq!(written, len, "kernel table length changed between calls");
    // SAFETY: the call wrote a NUL-terminated string (cap > len).
    let joined = unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    let names: BTreeSet<String> = joined.split('\n').map(str::to_owned).collect();
    assert_eq!(
        names.len(),
        joined.split('\n').count(),
        "duplicate names in the kernel table: {joined}"
    );
    names
}

/// The quoted strings of `export const GGUF_REPACK_KERNEL_MARKERS = [ ... ]`.
fn ts_marker_names() -> BTreeSet<String> {
    let start_tag = "export const GGUF_REPACK_KERNEL_MARKERS = [";
    let start = METALLIB_SELECT_TS
        .find(start_tag)
        .expect("GGUF_REPACK_KERNEL_MARKERS not found in metallib-select.ts")
        + start_tag.len();
    let end = start
        + METALLIB_SELECT_TS[start..]
            .find(']')
            .expect("GGUF_REPACK_KERNEL_MARKERS is not closed");
    let body = &METALLIB_SELECT_TS[start..end];
    let mut names = BTreeSet::new();
    let mut count = 0;
    for piece in body.split('\'').skip(1).step_by(2) {
        names.insert(piece.to_owned());
        count += 1;
    }
    assert_eq!(
        names.len(),
        count,
        "duplicate GGUF_REPACK_KERNEL_MARKERS entries"
    );
    names
}

fn diff(a: &BTreeSet<String>, b: &BTreeSet<String>) -> (Vec<String>, Vec<String>) {
    (
        a.difference(b).cloned().collect(),
        b.difference(a).cloned().collect(),
    )
}

#[test]
fn metallib_matches_the_kernel_table() {
    let table = table_names();
    assert_eq!(table.len(), 17, "kernel table: {table:?}");
    let mut report = vec![0 as c_char; 1 << 16];
    // SAFETY: `report` is writable for its length.
    let rc =
        unsafe { mlx_sys::mlx_test_gguf_repack_metallib_check(report.as_mut_ptr(), report.len()) };
    assert_eq!(
        rc, 0,
        "mlx_test_gguf_repack_metallib_check failed (no Metal device, or the metallib did not load)"
    );
    // SAFETY: the hook wrote a NUL-terminated string.
    let report = unsafe { CStr::from_ptr(report.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    assert!(
        report.is_empty(),
        "paged_attn.metallib does not match the GGUF repack kernel table:\n{report}"
    );
    // SAFETY: nullary predicate that catches internally.
    assert!(
        unsafe { mlx_sys::mlx_gguf_repack_available() },
        "mlx_gguf_repack_available() is false although the table matches the metallib"
    );
}

#[test]
fn ts_markers_match_the_kernel_table() {
    let table = table_names();
    let markers = ts_marker_names();
    let (only_table, only_ts) = diff(&table, &markers);
    assert!(
        only_table.is_empty() && only_ts.is_empty(),
        "kernel table and GGUF_REPACK_KERNEL_MARKERS differ: {} only in the C++ table {:?}, {} only in metallib-select.ts {:?}",
        only_table.len(),
        only_table,
        only_ts.len(),
        only_ts
    );
}
