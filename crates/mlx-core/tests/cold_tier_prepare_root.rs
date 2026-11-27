//! Offline cover for `cold_tier_parity_harness::prepare_cold_root`: a second
//! gate in the same binary must not unlink and replace the tier directory the
//! first gate's live manager is already holding open.
//!
//! Kept in its own test binary on purpose: `prepare_cold_root` pins the
//! process-global capture budget `OnceLock` and asserts that nothing else in
//! the process has already initialized the cold tier or a paged adapter.

#![cfg(unix)]

mod cold_tier_parity_harness;

#[test]
fn preparing_the_root_twice_keeps_the_same_directory() {
    use std::os::unix::fs::MetadataExt;

    let first = cold_tier_parity_harness::prepare_cold_root();
    let ino_first = std::fs::metadata(first.path())
        .expect("tier root must exist after the first prepare")
        .ino();

    let second = cold_tier_parity_harness::prepare_cold_root();
    let ino_second = std::fs::metadata(second.path())
        .expect("tier root must exist after the second prepare")
        .ino();

    assert_eq!(
        first.path(),
        second.path(),
        "both gates must resolve the same pid-scoped root, or this test is not exercising \
         the shared-root case at all"
    );
    assert_eq!(
        ino_first, ino_second,
        "the second prepare replaced the tier directory ({ino_first} -> {ino_second}): the \
         manager installed by the first gate still holds a descriptor for the unlinked one, \
         so every cache write and read in the second gate fails with ENOENT"
    );
}
