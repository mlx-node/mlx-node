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

    // FIRST, while both globals are still unset: a root that cannot be OPENED
    // must not report success. `open_managed_cold_cache` returns `None` when
    // the managed `mlx-paged-v1` child cannot be opened, and `GLOBAL.set(None)`
    // still succeeds — so reporting on the `set` alone told the caller its
    // cache was installed while the tier was DISABLED. The harness would then
    // run its whole gate and fail with "restore did not engage", blaming the
    // restore path for a setup that never opened.
    //
    // This has to run BEFORE the successful install below, in this same test,
    // rather than as a second `#[test]`: libtest shares one process, the first
    // successful install owns `INSTALLED_ROOT` permanently, and a later bad-root
    // call would then return false via the path-mismatch branch — passing for
    // the wrong reason and proving nothing about the open.
    let blocked = root.join("blocked-parent");
    std::fs::create_dir_all(&blocked).expect("create blocked parent");
    // A regular file at the exact name the opener needs as a directory, so the
    // failure is a real filesystem refusal rather than a synthetic one.
    std::fs::write(blocked.join("mlx-paged-v1"), b"not a directory").expect("write blocker");
    assert!(
        !install_cold_cache_root(&blocked),
        "install reported success for a root whose managed child could not be opened — the \
         caller would proceed with the tier DISABLED and blame the restore path"
    );

    // First install: nothing has resolved either global yet. This ALSO proves
    // the failed attempt above did not poison them — it must return before
    // touching `GLOBAL` or recording a root.
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
