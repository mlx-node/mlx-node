//! `install_cold_capture_budget` / `install_cold_cache_root` must tolerate a
//! SECOND identical call in the same process.
//!
//! Both write process-global `OnceLock`s, and the cold-tier harness asserts on
//! their return value — a `false` there panics before the model loads. That is
//! the right behaviour for a genuine conflict (the run would silently use the
//! 128-block default while the ladder arithmetic assumed 12) and the wrong
//! behaviour for a repeat, because `gemma4_cold_tier_parity.rs` ships TWO
//! `#[ignore]`d gates in one binary and its own docstring calls running them
//! together supported:
//!
//!   "Both gates in this binary are safe to run in one process:
//!    `run_restart_parity` honours an already-created root, the cold keys are
//!    content-derived (different prompts => disjoint chains), and every stats
//!    assertion is a delta around its own instance 2."
//!
//! The env-var version this replaced was idempotent for free: a second
//! `set_var` of the same value is a no-op. So the regression is a property
//! only the new API can have, which is why it is pinned here rather than left
//! to the `#[ignore]`d gates that need a real checkpoint to notice.
//!
//! Its OWN test binary on purpose: these calls resolve process-global state,
//! so running them beside anything that reads the cold tier would decide that
//! state for the other tests too.

use std::path::PathBuf;
use std::time::Duration;

use mlx_core::cold_tier::{install_cold_cache_root, install_cold_capture_budget};

#[test]
fn installing_the_same_settings_twice_is_a_no_op_not_a_conflict() {
    let root: PathBuf =
        std::env::temp_dir().join(format!("mlx-cold-install-idem-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("create temp root");

    let budget = Duration::from_millis(60_000);

    // First install: nothing has resolved either global yet.
    assert!(
        install_cold_capture_budget(12, budget),
        "first budget install should win the OnceLock"
    );
    assert!(
        install_cold_cache_root(&root),
        "first root install should open the tier"
    );

    // Second gate in the same binary, same constants. This is what panicked.
    assert!(
        install_cold_capture_budget(12, budget),
        "a REPEAT install of the identical budget must report success — the harness asserts on \
         this, so a false here aborts the second gate in a two-gate binary before it loads a model"
    );
    assert!(
        install_cold_cache_root(&root),
        "a REPEAT install of the identical root must report success for the same reason"
    );

    // The guard still has to bite, or it is worth nothing: a DIFFERENT value
    // means the process is not configured the way the caller believes.
    assert!(
        !install_cold_capture_budget(999, budget),
        "a conflicting block count must be reported, not silently accepted — that is the \
         wrong-green this return value exists to catch"
    );
    assert!(
        !install_cold_capture_budget(12, Duration::from_millis(1)),
        "a conflicting deadline must be reported too"
    );
    assert!(
        !install_cold_cache_root(&root.join("elsewhere")),
        "a conflicting root must be reported: the tier is open somewhere else and this caller's \
         directory is NOT in effect"
    );

    let _ = std::fs::remove_dir_all(&root);
}
