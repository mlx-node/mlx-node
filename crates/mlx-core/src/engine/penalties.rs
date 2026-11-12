//! Sampling penalties and the reasoning/budget tracker shared by the
//! decode loops.

use std::sync::Arc;

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
/// The template opens `<think>` when thinking is on; the model thinks, then
/// emits the close token, then content. When a thinking budget runs out the
/// turn ends thinking with `end`, the chat template's close (Splash
/// `ThinkingBudget.end`, e.g. Qwen's `"\n</think>"`), one token per step, so
/// the next request's re-render of the reply matches the tokens decoded.
///
/// An end token is in flight from [`Self::take_forced_token`] until it is
/// observed, and the next one is handed out only after that (Splash
/// `commitSelected` advances the end on commit). A loop that picks the next
/// token before observing the current one and a loop that observes first both
/// emit the end once, in order.
pub(crate) struct ReasoningTracker {
    in_thinking: bool,
    thinking_token_count: i32,
    budget: Option<i32>,
    /// The tokens a budget ends thinking with, the close token last.
    end: Arc<[u32]>,
    /// Additional token ids that also END reasoning without being the
    /// budget's close. K2-Horizon opens one of three `<ifm|think*>`
    /// blocks per turn but the model may close with a different member
    /// of the family.
    extra_think_end_ids: Vec<u32>,
    /// The budget ran out and thinking has not closed yet.
    cut: bool,
    /// End tokens handed out, and end tokens observed, since the cut.
    taken: usize,
    observed: usize,
}

impl ReasoningTracker {
    fn with_end(
        starts_in_thinking: bool,
        budget: Option<i32>,
        end: Arc<[u32]>,
        extra_think_end_ids: Vec<u32>,
    ) -> Self {
        // Budget 0 ends thinking before any thinking token.
        let cut = starts_in_thinking && budget == Some(0) && !end.is_empty();
        Self {
            in_thinking: starts_in_thinking,
            thinking_token_count: 0,
            budget,
            end,
            extra_think_end_ids,
            cut,
            taken: 0,
            observed: 0,
        }
    }

    /// A tracker over explicit end tokens, for tests.
    #[cfg(test)]
    pub(crate) fn new(starts_in_thinking: bool, budget: Option<i32>, end: &[u32]) -> Self {
        Self::with_end(starts_in_thinking, budget, Arc::from(end), Vec::new())
    }

    /// The one production construction point: the turn's thinking state and
    /// its close of thinking (`ChatBackend::think_end_for_turn`, or
    /// `Qwen3Tokenizer::think_end` where a family core resolves no override).
    pub(crate) fn from_setup(
        setup: &crate::engine::backend::ThinkingSetup,
        close: &crate::engine::backend::ThinkEndResolution,
    ) -> Self {
        Self::with_end(
            setup.enabled,
            setup.budget,
            close.thinking_end.clone(),
            close.think_end_extra_ids.clone(),
        )
    }

    /// Process a generated token. Returns whether this token is reasoning content.
    ///
    /// Call AFTER extracting the token ID from the GPU each decode step.
    pub fn observe_token(&mut self, token_id: u32) -> bool {
        if !self.in_thinking {
            return false;
        }

        if self.end.last() == Some(&token_id) || self.extra_think_end_ids.contains(&token_id) {
            self.in_thinking = false;
            self.cut = false;
            self.taken = 0;
            self.observed = 0;
            return true; // the close itself is part of reasoning
        }

        self.thinking_token_count += 1;
        if self.cut {
            if self.observed < self.taken && self.end[self.observed] == token_id {
                self.observed += 1;
            } else {
                debug_assert!(false, "thinking went past its budget");
            }
        } else if !self.end.is_empty()
            && let Some(budget) = self.budget
            && self.thinking_token_count >= budget
        {
            self.cut = true;
        }
        true
    }

    /// Thinking is being ended: the budget ran out and the close has not been
    /// observed. No token may be drafted or sampled until it closes.
    pub(crate) fn ending(&self) -> bool {
        self.in_thinking && self.cut
    }

    /// The end token the next step must emit, without taking it. `None` while
    /// a taken end token is still in flight.
    pub(crate) fn peek_forced_token(&self) -> Option<u32> {
        if self.ending() && self.taken == self.observed {
            self.end.get(self.taken).copied()
        } else {
            None
        }
    }

    /// Take the end token the next step must emit; it is in flight until
    /// [`Self::observe_token`] sees it.
    pub(crate) fn take_forced_token(&mut self) -> Option<u32> {
        let token = self.peek_forced_token()?;
        self.taken += 1;
        Some(token)
    }

    /// Replace a not-yet-committed sample at the reasoning boundary. Finish its
    /// dependency graph before discarding it so lazy cache writes are retained.
    pub(crate) fn enforce_next_token(&mut self, token: &mut MxArray) -> Result<()> {
        if let Some(forced) = self.take_forced_token() {
            token.eval();
            *token = MxArray::from_int32(&[forced as i32], &[1])?;
        }
        Ok(())
    }

    /// Largest unforced output prefix whose tokens can be emitted before the
    /// budget requires a forced end token. A speculative cycle cannot cross it.
    pub(crate) fn unforced_token_budget(&self) -> Option<usize> {
        if !self.in_thinking || self.end.is_empty() {
            return None;
        }
        if self.cut {
            return Some(0);
        }
        self.budget
            .map(|budget| budget.saturating_sub(self.thinking_token_count).max(0) as usize)
    }

    /// Number of tokens generated during reasoning before the close, the end
    /// tokens a cut forces included (Splash `count_reasoning_tokens`).
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
        let mut tracker = ReasoningTracker::new(true, None, &[THINK_END_ID]);
        assert!(tracker.observe_token(100)); // reasoning
        assert!(tracker.observe_token(200)); // reasoning
        assert!(!tracker.take_forced_token().is_some());
    }

    #[test]
    fn test_tracker_transitions_on_think_end() {
        let mut tracker = ReasoningTracker::new(true, None, &[THINK_END_ID]);
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
            let mut tracker = ReasoningTracker::new(true, None, &[THINK_END_ID]);
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
            let mut tracker = ReasoningTracker::new(true, None, &[THINK_END_ID]);
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
        let mut tracker = ReasoningTracker::new(false, None, &[THINK_END_ID]);
        assert!(!tracker.observe_token(100));
        assert!(!tracker.observe_token(200));
        assert!(!tracker.take_forced_token().is_some());
    }

    #[test]
    fn test_tracker_budget_enforcement() {
        // Budget=3: allows exactly 3 thinking tokens, then forces on the 3rd.
        let mut tracker = ReasoningTracker::new(true, Some(3), &[THINK_END_ID]);
        assert!(tracker.observe_token(100)); // count→1
        assert!(!tracker.take_forced_token().is_some());
        assert!(tracker.observe_token(200)); // count→2
        assert!(!tracker.take_forced_token().is_some());
        assert!(tracker.observe_token(300)); // count→3, 3>=3 → force!
        assert_eq!(tracker.take_forced_token(), Some(THINK_END_ID));
    }

    #[test]
    fn test_tracker_budget_zero() {
        // Budget=0: force is set in new() — triggers BEFORE any thinking token.
        let mut tracker = ReasoningTracker::new(true, Some(0), &[THINK_END_ID]);
        assert!(tracker.take_forced_token().is_some()); // immediate, no observe needed
    }

    #[test]
    fn test_tracker_budget_zero_vs_one() {
        // Budget=0: force immediately (0 thinking tokens allowed).
        let mut t0 = ReasoningTracker::new(true, Some(0), &[THINK_END_ID]);
        assert!(t0.take_forced_token().is_some()); // before any observe

        // Budget=1: allows exactly 1 thinking token before forcing.
        let mut t1 = ReasoningTracker::new(true, Some(1), &[THINK_END_ID]);
        assert!(!t1.take_forced_token().is_some()); // not yet
        assert!(t1.observe_token(100)); // count→1, 1>=1 → force!
        assert!(t1.take_forced_token().is_some()); // triggers after 1st token
    }

    #[test]
    fn test_tracker_budget_clears_on_think_end() {
        let mut tracker = ReasoningTracker::new(true, Some(2), &[THINK_END_ID]);
        assert!(tracker.observe_token(100)); // count→1
        assert!(!tracker.take_forced_token().is_some());
        assert!(tracker.observe_token(200)); // count→2, 2>=2 → force!
        assert!(tracker.take_forced_token().is_some());
        // When the forced think_end token is generated:
        assert!(tracker.observe_token(THINK_END_ID)); // transitions to content
        assert!(!tracker.take_forced_token().is_some()); // force cleared
        assert!(!tracker.observe_token(300)); // now content
    }

    #[test]
    fn test_tracker_no_double_force_with_pipeline_lag() {
        // A forced token is in flight from take until observed: a second take
        // before the observe hands nothing.
        let mut tracker = ReasoningTracker::new(true, Some(3), &[THINK_END_ID]);
        tracker.observe_token(100);
        tracker.observe_token(200);
        tracker.observe_token(300); // count 3 = budget: the cut starts
        assert_eq!(tracker.take_forced_token(), Some(THINK_END_ID));
        assert_eq!(tracker.take_forced_token(), None);
        assert!(tracker.observe_token(THINK_END_ID)); // closes
        assert_eq!(tracker.take_forced_token(), None);
        assert!(!tracker.observe_token(500)); // content
    }

    #[test]
    fn test_tracker_no_budget() {
        let mut tracker = ReasoningTracker::new(true, None, &[THINK_END_ID]);
        for i in 0..1000 {
            assert!(tracker.observe_token(i));
            assert!(!tracker.take_forced_token().is_some());
        }
    }

    #[test]
    fn test_tracker_no_think_end_id() {
        let mut tracker = ReasoningTracker::new(true, Some(5), &[]);
        // Without a close token nothing is ever forced
        for i in 0..100 {
            tracker.observe_token(i);
            assert!(!tracker.take_forced_token().is_some());
        }
    }

    #[test]
    fn test_tracker_no_think_end_id_labels_as_reasoning() {
        // When thinking is enabled but think_end_id is missing (tokenizer
        // renders </think> as multiple tokens), observe_token should still
        // return true (reasoning) for every token — consistent with the
        // text-level finalization that will find reasoning via parsing.
        let mut tracker = ReasoningTracker::new(true, None, &[]);
        assert!(tracker.observe_token(100)); // reasoning
        assert!(tracker.observe_token(200)); // reasoning
        assert!(tracker.observe_token(300)); // reasoning
        // Never transitions — no think_end_id to match
        assert!(!tracker.take_forced_token().is_some()); // budget disabled
    }

    #[test]
    fn test_peek_forced_token_is_non_consuming() {
        let mut tracker = ReasoningTracker::new(true, Some(2), &[THINK_END_ID]);
        assert!(tracker.observe_token(100));
        assert_eq!(tracker.peek_forced_token(), None);
        assert!(!tracker.ending());
        assert!(tracker.observe_token(200)); // the cut starts
        for _ in 0..3 {
            assert_eq!(tracker.peek_forced_token(), Some(THINK_END_ID));
            assert!(tracker.ending());
        }
        assert_eq!(tracker.take_forced_token(), Some(THINK_END_ID));
        assert_eq!(tracker.peek_forced_token(), None); // in flight
        assert!(tracker.ending()); // until the close is observed
    }

    #[test]
    fn tracker_without_a_close_never_cuts() {
        let mut tracker = ReasoningTracker::new(true, Some(1), &[]);
        for i in 0..4 {
            tracker.observe_token(i);
            assert!(!tracker.ending());
            assert_eq!(tracker.unforced_token_budget(), None);
            assert_eq!(tracker.take_forced_token(), None);
        }
        let zero = ReasoningTracker::new(true, Some(0), &[]);
        assert!(!zero.ending());
    }

    #[test]
    fn tracker_ends_thinking_with_the_whole_close() {
        // Budget 2, end "\n</think>" = [8, 9].
        let mut tracker = ReasoningTracker::new(true, Some(2), &[8, 9]);
        assert_eq!(tracker.unforced_token_budget(), Some(2));
        assert!(tracker.observe_token(100));
        assert_eq!(tracker.unforced_token_budget(), Some(1));
        assert!(tracker.observe_token(101)); // the cut starts
        assert!(tracker.ending());
        assert_eq!(tracker.unforced_token_budget(), Some(0));
        assert_eq!(tracker.take_forced_token(), Some(8));
        // 8 is in flight: a loop that takes again before observing gets nothing.
        assert_eq!(tracker.take_forced_token(), None);
        assert!(tracker.observe_token(8));
        // Between the end's tokens nothing may be drafted.
        assert!(tracker.ending());
        assert_eq!(tracker.unforced_token_budget(), Some(0));
        assert_eq!(tracker.take_forced_token(), Some(9));
        assert!(tracker.observe_token(9)); // the close is reasoning
        assert!(!tracker.ending());
        assert_eq!(tracker.unforced_token_budget(), None);
        assert_eq!(tracker.take_forced_token(), None);
        assert!(!tracker.observe_token(5));
        // Splash count_reasoning_tokens: budget + len(end) - 1.
        assert_eq!(tracker.reasoning_token_count(), 3);

        // Budget 0: the first token is the end's first.
        let mut zero = ReasoningTracker::new(true, Some(0), &[8, 9]);
        assert_eq!(zero.take_forced_token(), Some(8));
        assert_eq!(zero.take_forced_token(), None);
        zero.observe_token(8);
        assert_eq!(zero.take_forced_token(), Some(9));
        zero.observe_token(9);
        assert_eq!(zero.reasoning_token_count(), 1);
    }

    #[test]
    #[should_panic(expected = "thinking went past its budget")]
    fn a_token_past_the_budget_is_a_defect() {
        let mut tracker = ReasoningTracker::new(true, Some(1), &[8, 9]);
        tracker.observe_token(100); // the cut starts
        tracker.observe_token(101); // not the end: Splash commitSelected throws
    }
}
