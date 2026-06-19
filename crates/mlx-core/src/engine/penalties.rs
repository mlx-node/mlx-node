//! Sampling penalties and the reasoning/budget tracker shared by the
//! decode loops.

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::engine::params::ChatParams;
use crate::sampling::{apply_frequency_penalty, apply_presence_penalty, apply_repetition_penalty};

/// Apply repetition + presence + frequency penalties to logits.
pub(crate) fn apply_all_penalties(
    mut logits: MxArray,
    token_history: &[u32],
    params: &ChatParams,
) -> Result<MxArray> {
    if params.repetition_penalty != 1.0 && !token_history.is_empty() {
        logits = apply_repetition_penalty(
            &logits,
            token_history,
            params.repetition_penalty,
            Some(params.repetition_context_size),
        )?;
    }
    if params.presence_penalty != 0.0 {
        logits = apply_presence_penalty(
            &logits,
            token_history,
            params.presence_penalty,
            Some(params.presence_context_size),
        )?;
    }
    if params.frequency_penalty != 0.0 {
        logits = apply_frequency_penalty(
            &logits,
            token_history,
            params.frequency_penalty,
            Some(params.frequency_context_size),
        )?;
    }
    Ok(logits)
}

/// Tracks reasoning vs content state during token-by-token generation.
///
/// For Qwen3.5: the template injects `<think>\n` when thinking is enabled.
/// The model generates thinking tokens, then emits `</think>` (think_end_id),
/// then generates content. This tracker detects the transition at the TOKEN
/// level — no text parsing needed during decoding.
pub(crate) struct ReasoningTracker {
    in_thinking: bool,
    thinking_token_count: i32,
    budget: Option<i32>,
    think_end_id: Option<u32>,
    force_think_end: bool,
    /// Set after `should_force_think_end` is consumed, prevents re-triggering
    /// from subsequent `observe_token` calls before the forced token is extracted.
    end_scheduled: bool,
}

impl ReasoningTracker {
    /// Create a new tracker.
    ///
    /// `starts_in_thinking`: true when the template injected `<think>\n` (thinking enabled).
    /// `budget`: maximum thinking tokens before forcing `</think>`. None = unlimited.
    /// `think_end_id`: token ID for `</think>` from the tokenizer vocabulary.
    pub fn new(starts_in_thinking: bool, budget: Option<i32>, think_end_id: Option<u32>) -> Self {
        // Budget=0 means "no thinking tokens at all" — force </think> immediately
        // on the first decode step, before any thinking token is generated.
        let force_immediately = starts_in_thinking && budget == Some(0) && think_end_id.is_some();
        Self {
            in_thinking: starts_in_thinking,
            thinking_token_count: 0,
            budget,
            think_end_id,
            force_think_end: force_immediately,
            end_scheduled: false,
        }
    }

    /// Build a tracker from an already-resolved
    /// [`crate::engine::backend::ThinkingSetup`]. Equivalent to
    /// `Self::new(setup.enabled, setup.budget, think_end_id)`; this is the
    /// ONE construction point the whole-turn cores call, so the
    /// thinking-mode resolution lives in a single place.
    pub fn from_setup(
        setup: &crate::engine::backend::ThinkingSetup,
        think_end_id: Option<u32>,
    ) -> Self {
        Self::new(setup.enabled, setup.budget, think_end_id)
    }

    /// Process a generated token. Returns whether this token is reasoning content.
    ///
    /// Call AFTER extracting the token ID from the GPU each decode step.
    pub fn observe_token(&mut self, token_id: u32) -> bool {
        if !self.in_thinking {
            return false;
        }

        if self.think_end_id == Some(token_id) {
            self.in_thinking = false;
            self.force_think_end = false;
            self.end_scheduled = false;
            return true; // </think> itself is part of reasoning
        }

        self.thinking_token_count += 1;
        if let Some(budget) = self.budget
            && self.thinking_token_count >= budget
            && !self.end_scheduled
        {
            self.force_think_end = true;
        }
        true
    }

    /// Whether the next token should be forced to think_end_id.
    /// Consumes the flag — returns true at most once per budget trigger.
    ///
    /// Check this BEFORE building the next decode step's graph.
    pub fn should_force_think_end(&mut self) -> bool {
        if self.force_think_end && self.think_end_id.is_some() {
            self.force_think_end = false;
            self.end_scheduled = true;
            true
        } else {
            false
        }
    }

    /// Non-consuming peek: whether a think-end force is currently pending.
    /// Unlike `should_force_think_end`, this does NOT clear the flag or set
    /// `end_scheduled`, so it is safe to call during routing/defer decisions.
    /// The single consuming call must remain at the actual token-insertion site.
    pub fn force_think_end_pending(&self) -> bool {
        self.force_think_end && self.think_end_id.is_some()
    }

    /// The think_end token ID to force. Only valid when `should_force_think_end()` returned true.
    pub fn forced_token_id(&self) -> Result<u32> {
        self.think_end_id.ok_or_else(|| {
            napi::Error::from_reason("should_force_think_end was true but think_end_id is None")
        })
    }

    /// Number of tokens generated during reasoning (inside <think>...</think>).
    pub fn reasoning_token_count(&self) -> u32 {
        self.thinking_token_count.max(0) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const THINK_END_ID: u32 = 151668; // example </think> token ID

    #[test]
    fn test_tracker_starts_in_thinking() {
        let mut tracker = ReasoningTracker::new(true, None, Some(THINK_END_ID));
        assert!(tracker.observe_token(100)); // reasoning
        assert!(tracker.observe_token(200)); // reasoning
        assert!(!tracker.should_force_think_end());
    }

    #[test]
    fn test_tracker_transitions_on_think_end() {
        let mut tracker = ReasoningTracker::new(true, None, Some(THINK_END_ID));
        assert!(tracker.observe_token(100)); // reasoning
        assert!(tracker.observe_token(THINK_END_ID)); // </think> is still reasoning
        assert!(!tracker.observe_token(300)); // now content
        assert!(!tracker.observe_token(400)); // still content
    }

    #[test]
    fn test_stream_reasoning_gate_predicate() {
        // Drives the boundary semantics the streaming send-gate relies on:
        // `observe_token` returns true for reasoning tokens INCLUDING the
        // `</think>` closer, and false for the first content token after.
        // The send-gate is `include_reasoning || !is_reasoning`.
        //
        // Token ids are chosen distinct from THINK_END_ID for the
        // reasoning/content tokens.
        let seq = [101u32, 102, THINK_END_ID, 301, 302];

        // include_reasoning == false: suppress the 3 reasoning tokens
        // (including the </think> closer), emit the 2 content tokens.
        {
            let mut tracker = ReasoningTracker::new(true, None, Some(THINK_END_ID));
            let include_reasoning = false;
            let gate: Vec<bool> = seq
                .iter()
                .map(|&tok| {
                    let is_reasoning = tracker.observe_token(tok);
                    include_reasoning || !is_reasoning
                })
                .collect();
            assert_eq!(gate, vec![false, false, false, true, true]);
        }

        // include_reasoning == true: emit everything.
        {
            let mut tracker = ReasoningTracker::new(true, None, Some(THINK_END_ID));
            let include_reasoning = true;
            let gate: Vec<bool> = seq
                .iter()
                .map(|&tok| {
                    let is_reasoning = tracker.observe_token(tok);
                    include_reasoning || !is_reasoning
                })
                .collect();
            assert_eq!(gate, vec![true, true, true, true, true]);
        }
    }

    #[test]
    fn test_tracker_starts_in_content() {
        let mut tracker = ReasoningTracker::new(false, None, Some(THINK_END_ID));
        assert!(!tracker.observe_token(100));
        assert!(!tracker.observe_token(200));
        assert!(!tracker.should_force_think_end());
    }

    #[test]
    fn test_tracker_budget_enforcement() {
        // Budget=3: allows exactly 3 thinking tokens, then forces on the 3rd.
        let mut tracker = ReasoningTracker::new(true, Some(3), Some(THINK_END_ID));
        assert!(tracker.observe_token(100)); // count→1
        assert!(!tracker.should_force_think_end());
        assert!(tracker.observe_token(200)); // count→2
        assert!(!tracker.should_force_think_end());
        assert!(tracker.observe_token(300)); // count→3, 3>=3 → force!
        assert!(tracker.should_force_think_end());
        assert_eq!(tracker.forced_token_id().unwrap(), THINK_END_ID);
    }

    #[test]
    fn test_tracker_budget_zero() {
        // Budget=0: force is set in new() — triggers BEFORE any thinking token.
        let mut tracker = ReasoningTracker::new(true, Some(0), Some(THINK_END_ID));
        assert!(tracker.should_force_think_end()); // immediate, no observe needed
    }

    #[test]
    fn test_tracker_budget_zero_vs_one() {
        // Budget=0: force immediately (0 thinking tokens allowed).
        let mut t0 = ReasoningTracker::new(true, Some(0), Some(THINK_END_ID));
        assert!(t0.should_force_think_end()); // before any observe

        // Budget=1: allows exactly 1 thinking token before forcing.
        let mut t1 = ReasoningTracker::new(true, Some(1), Some(THINK_END_ID));
        assert!(!t1.should_force_think_end()); // not yet
        assert!(t1.observe_token(100)); // count→1, 1>=1 → force!
        assert!(t1.should_force_think_end()); // triggers after 1st token
    }

    #[test]
    fn test_tracker_budget_clears_on_think_end() {
        let mut tracker = ReasoningTracker::new(true, Some(2), Some(THINK_END_ID));
        assert!(tracker.observe_token(100)); // count→1
        assert!(!tracker.should_force_think_end());
        assert!(tracker.observe_token(200)); // count→2, 2>=2 → force!
        assert!(tracker.should_force_think_end());
        // When the forced think_end token is generated:
        assert!(tracker.observe_token(THINK_END_ID)); // transitions to content
        assert!(!tracker.should_force_think_end()); // force cleared
        assert!(!tracker.observe_token(300)); // now content
    }

    #[test]
    fn test_tracker_no_double_force_with_pipeline_lag() {
        // Simulates pipelined decode: after should_force_think_end() is consumed,
        // the pipeline extracts an over-budget token before the forced </think>
        // arrives. The tracker must NOT re-trigger forcing.
        let mut tracker = ReasoningTracker::new(true, Some(3), Some(THINK_END_ID));
        tracker.observe_token(100); // count→1
        tracker.observe_token(200); // count→2
        tracker.observe_token(300); // count→3, 3>=3 → force=true

        // Phase A of step N+1: consume the force flag
        assert!(tracker.should_force_think_end()); // returns true, sets end_scheduled
        assert!(!tracker.should_force_think_end()); // already consumed — must be false

        // Phase B of step N+1: the pipeline extracts the over-budget token (not </think>)
        assert!(tracker.observe_token(400)); // still reasoning, count→4
        // Must NOT re-trigger forcing despite count(4) >= budget(3)
        assert!(!tracker.should_force_think_end());

        // Phase B of step N+2: the forced </think> token is finally extracted
        assert!(tracker.observe_token(THINK_END_ID)); // transitions to content
        assert!(!tracker.should_force_think_end());

        // Phase B of step N+3: normal content token
        assert!(!tracker.observe_token(500)); // content
    }

    #[test]
    fn test_tracker_no_budget() {
        let mut tracker = ReasoningTracker::new(true, None, Some(THINK_END_ID));
        for i in 0..1000 {
            assert!(tracker.observe_token(i));
            assert!(!tracker.should_force_think_end());
        }
    }

    #[test]
    fn test_tracker_no_think_end_id() {
        let mut tracker = ReasoningTracker::new(true, Some(5), None);
        // Without think_end_id, should_force_think_end is always false
        for i in 0..100 {
            tracker.observe_token(i);
            assert!(!tracker.should_force_think_end());
        }
    }

    #[test]
    fn test_tracker_no_think_end_id_labels_as_reasoning() {
        // When thinking is enabled but think_end_id is missing (tokenizer
        // renders </think> as multiple tokens), observe_token should still
        // return true (reasoning) for every token — consistent with the
        // text-level finalization that will find reasoning via parsing.
        let mut tracker = ReasoningTracker::new(true, None, None);
        assert!(tracker.observe_token(100)); // reasoning
        assert!(tracker.observe_token(200)); // reasoning
        assert!(tracker.observe_token(300)); // reasoning
        // Never transitions — no think_end_id to match
        assert!(!tracker.should_force_think_end()); // budget disabled
    }

    #[test]
    fn test_force_think_end_pending_is_non_consuming() {
        // The non-consuming peek used by the MTP routing/defer decisions
        // must report a pending force WITHOUT clearing it. Repeated peeks
        // stay true; only the single consuming call clears it.
        let mut tracker = ReasoningTracker::new(true, Some(2), Some(THINK_END_ID));
        assert!(tracker.observe_token(100)); // count→1
        assert!(!tracker.force_think_end_pending()); // not yet tripped
        assert!(tracker.observe_token(200)); // count→2, 2>=2 → force=true

        // Peek repeatedly — must stay true, never consuming.
        assert!(tracker.force_think_end_pending());
        assert!(tracker.force_think_end_pending());
        assert!(tracker.force_think_end_pending());

        // The single consuming call returns true exactly once and clears.
        assert!(tracker.should_force_think_end());
        assert!(!tracker.should_force_think_end()); // consumed
        assert!(!tracker.force_think_end_pending()); // peek now reflects cleared flag
    }

    #[test]
    fn test_force_think_end_pending_mirrors_chained_routing() {
        // Mirrors the chained-ON MTP path: the budget trips, the routing
        // peek (`do_step_a`) and the defer guard both poll the NON-consuming
        // predicate (possibly many times across cycles), and only the single
        // token-insertion site consumes + forces exactly once.
        let mut tracker = ReasoningTracker::new(true, Some(1), Some(THINK_END_ID));
        assert!(tracker.observe_token(100)); // count→1, 1>=1 → force=true

        // Routing peek for cycle N (do_step_a), then a defer-guard peek, then
        // routing peek for cycle N+1 — all non-consuming, all stay true.
        for _ in 0..5 {
            assert!(
                tracker.force_think_end_pending(),
                "peek must remain true until the single consume fires"
            );
        }

        // Step A's token-insertion site consumes and forces exactly once.
        assert!(tracker.should_force_think_end());
        assert_eq!(tracker.forced_token_id().unwrap(), THINK_END_ID);

        // After the consume, no further force fires from peeks or consumes.
        assert!(!tracker.force_think_end_pending());
        assert!(!tracker.should_force_think_end());
    }
}
