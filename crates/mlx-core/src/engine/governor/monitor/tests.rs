use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use super::*;
use crate::engine::governor::governor::ModelRegistration;
use crate::engine::governor::governor::tests::{FakeIo, G, host, stats};
use crate::engine::governor::host::Pressure;

/// A model thread that published a wakeup is woken once it falls due,
/// with the host calm (Splash `WriteBehind::nextWakeup`: a due point is
/// copied with no request), and once only.
/// Mutation WD1 (the monitor ignores a due wakeup) → red.
#[test]
fn due_wakeup_wakes_the_model() {
    const MODEL: u64 = 0x5713_3302;
    let io = FakeIo::new(stats(G), host(32 * G, Pressure::Normal));
    let governor: &'static Governor = Box::leak(Box::new(Governor::new(io.clone())));
    let _registration = governor
        .register_model(ModelRegistration {
            model_id: MODEL,
            hard_budget: 16 * G,
            untracked: 0,
            page_bytes: G / 4,
        })
        .expect("register");
    let wakes = Arc::new(AtomicU32::new(0));
    let counted = Arc::clone(&wakes);
    register_waker(
        MODEL,
        Box::new(move || {
            counted.fetch_add(1, Ordering::AcqRel);
            true
        }),
    );
    assert!(!governor.needs_control());
    set_next_wakeup(MODEL, Some(10.0));
    io.with(|state| state.now = 9.9);
    tick(governor);
    assert_eq!(wakes.load(Ordering::Acquire), 0, "not due yet");
    io.with(|state| state.now = 10.0);
    tick(governor);
    assert_eq!(wakes.load(Ordering::Acquire), 1, "due: the model wakes");
    wake_taken(MODEL);
    tick(governor);
    assert_eq!(wakes.load(Ordering::Acquire), 1, "the wakeup was taken");
    forget_waker(MODEL);
}
