//! The one CI-runnable check that Gemma4 actually reaches the session
//! continuation verifier with its own reasoning-close tag.
//!
//! `render_live_continuation` reads `B::REASONING_CLOSE_TAG` and normalizes
//! the template-owned whitespace around exactly that tag. `</think>` never
//! appears in a Gemma4 render, so the default answer makes the splice fail
//! silently and every reasoning turn ends the live session — a whole-session
//! regression that no unit test of the parser would notice.
//!
//! Read through the trait, not from
//! [`crate::models::gemma4::output_parser::reasoning_close_tag`] directly: what has
//! to hold is that the wiring resolves to that tag, and calling the free
//! function would prove only that the free function returns itself.

use crate::engine::backend::{ChatBackend, REASONING_CLOSE_TAG_DEFAULT};
use crate::models::gemma4::model::Gemma4Inner;
use crate::models::gemma4::output_parser;

#[test]
fn gemma4_backend_declares_its_own_reasoning_close_tag() {
    assert_ne!(
        <Gemma4Inner as ChatBackend>::REASONING_CLOSE_TAG,
        REASONING_CLOSE_TAG_DEFAULT,
        "Gemma4 closes its reasoning channel with <channel|>; inheriting the \
         </think> default silently disables warm continuation after every \
         reasoning turn",
    );
    assert_eq!(
        <Gemma4Inner as ChatBackend>::REASONING_CLOSE_TAG,
        output_parser::reasoning_close_tag(),
        "the tag the session verifier normalizes on must be the tag the \
         Gemma4 output parser splits reasoning from content on",
    );
}
