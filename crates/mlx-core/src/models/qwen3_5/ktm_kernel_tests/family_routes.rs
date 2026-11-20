//! The paged attention route counters of the whole test process, printed
//! when it exits. Run beside the tests of another family in one process:
//! the keys-token-major entries show the launches that family's pages took.

use std::io::Write;

extern "C" fn print_routes() {
    let prefill = [0, 1, 3, 4].map(|kind| {
        // SAFETY: a read of a process-wide counter.
        unsafe { mlx_sys::mlx_paged_prefill_test_probe_count(kind) }
    });
    let decode = [0, 1, 2, 3].map(|kind| {
        // SAFETY: a read of a process-wide counter.
        unsafe { mlx_sys::mlx_paged_decode_test_probe_count(kind) }
    });
    let _ = writeln!(
        std::io::stderr(),
        "ROW ktm_routes_at_exit prefill [prefill, verify, q8q, ktm] {prefill:?} decode [vllm, ktm striped, ktm generic, ktm v1] {decode:?}"
    );
}

#[test]
fn route_counters_print_at_exit() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // SAFETY: the function holds no state and only reads counters.
        let rc = unsafe { libc::atexit(print_routes) };
        assert_eq!(rc, 0, "atexit");
    });
}
