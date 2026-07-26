//! Post-submission inspection of a Metal command buffer.
//!
//! `commit()` followed by `wait_until_completed()` tells the caller nothing
//! about whether the GPU actually ran the submitted work. A buffer can abort
//! on a fault, a watchdog timeout, or device removal, and an aborted blit
//! leaves the destination holding whatever bytes it already had. Without
//! reading the status back, that is indistinguishable from success: a restore
//! silently publishes a wrong KV block, and a capture silently persists a
//! half-read one for a later process to restore.
//!
//! Every production `commit()` + `wait_until_completed()` pair in this crate
//! calls [`observe`] immediately afterwards and propagates its `Err`.
//!
//! # What this does not do
//!
//! Status checking is not a bounds check. A blit whose size exceeds both
//! buffers was measured on this machine to finish with status `Completed` and
//! a nil error, so the explicit length and block-id validation in
//! `LayerKVPool` remains the only guard against a bad transfer range.

use metal::{CommandBuffer, command_buffer_outcome};

/// Report whether a finished command buffer ran its work, naming `context` on
/// failure so the message identifies which submission aborted.
///
/// Call this after `wait_until_completed()`, never between it and `commit()` —
/// waiting on a buffer that was never committed blocks forever.
pub(crate) fn observe(command_buffer: &CommandBuffer, context: &'static str) -> Result<(), String> {
    #[cfg(test)]
    if take_armed_failure(context) {
        return command_buffer_outcome(
            metal::CommandBufferStatus::Error,
            Some("armed test failure".to_string()),
            context,
        );
    }
    command_buffer_outcome(command_buffer.status(), command_buffer.error(), context)
}

#[cfg(test)]
thread_local! {
    /// Context string of the submission whose next `observe` must report a
    /// failed command buffer.
    static ARMED_FAILURE: std::cell::RefCell<Option<&'static str>> =
        const { std::cell::RefCell::new(None) };
}

/// Clears the arm when it drops, so an arm that never fired cannot leak into a
/// later test that happens to reuse the thread.
#[cfg(test)]
pub(crate) struct ArmedFailure;

#[cfg(test)]
impl Drop for ArmedFailure {
    fn drop(&mut self) {
        ARMED_FAILURE.with(|cell| *cell.borrow_mut() = None);
    }
}

/// Make the next [`observe`] call whose `context` matches behave as if the GPU
/// aborted the buffer.
///
/// A real GPU abort cannot be forced from userland: driving one needs a
/// device fault or a watchdog timeout, and probing every API misuse
/// (oversized blit, double commit, uncommitted wait) on this machine produced
/// either `Completed`, a process abort, or a permanent hang — never
/// `MTLCommandBufferStatusError`. So `cb.status()` genuinely reporting a
/// device fault is NOT covered by any test. What this seam covers is
/// everything downstream: that each submission site propagates the resulting
/// `Err`, and that the cold-tier capture and restore paths clean up behind it.
///
/// The arm is thread-local, so tests running in parallel cannot poison each
/// other's submissions, and one-shot, so it cannot outlive the call it
/// targets. Matching on `context` means an arm aimed at a submission that a
/// refactor moved simply never fires — the test then goes red instead of
/// quietly failing some other buffer.
#[cfg(test)]
#[must_use = "the arm is cleared when the returned guard drops"]
pub(crate) fn arm_failure(context: &'static str) -> ArmedFailure {
    ARMED_FAILURE.with(|cell| *cell.borrow_mut() = Some(context));
    ArmedFailure
}

#[cfg(test)]
fn take_armed_failure(context: &str) -> bool {
    ARMED_FAILURE.with(|cell| {
        let mut cell = cell.borrow_mut();
        if cell.is_some_and(|armed| armed == context) {
            *cell = None;
            true
        } else {
            false
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arm_fires_once_for_its_own_context_only() {
        let _guard = arm_failure("wanted");
        assert!(
            !take_armed_failure("other"),
            "a different context must not consume the arm"
        );
        assert!(take_armed_failure("wanted"));
        assert!(!take_armed_failure("wanted"), "the arm is one-shot");
    }

    #[test]
    fn dropping_the_guard_clears_an_unfired_arm() {
        drop(arm_failure("never-reached"));
        assert!(!take_armed_failure("never-reached"));
    }
}
