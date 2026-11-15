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

/// The longest cycle the guard looks for, in tokens.
const CYCLE_MAX_PERIOD: usize = 2048;
/// Copies of a cycle, in a row, that make a hit.
const CYCLE_MIN_REPEATS: u32 = 2;
/// The fewest tokens the repeated stretch must span.
const CYCLE_MIN_SPAN: u32 = 512;
/// Thinking tokens a hit still allows before the end is forced. A speculative
/// cycle commits up to 8 tokens (7 drafts and the bonus) after it read the
/// cut; the grace is wider, so a hit inside a cycle never puts the cut among
/// the tokens that cycle already chose.
const CYCLE_CUT_GRACE: i32 = 64;

/// Exact cycles in the thinking tokens: the last [`CYCLE_MAX_PERIOD`] tokens
/// and, for each period `p`, how many tokens in a row equal the token `p`
/// places back.
struct CycleDetector {
    history: Box<[u32]>,
    runs: Box<[u32]>,
    seen: usize,
}

impl CycleDetector {
    fn new() -> Self {
        Self {
            history: vec![0; CYCLE_MAX_PERIOD].into_boxed_slice(),
            runs: vec![0; CYCLE_MAX_PERIOD + 1].into_boxed_slice(),
            seen: 0,
        }
    }

    /// Feed one thinking token; the smallest period whose cycle is now there
    /// [`CYCLE_MIN_REPEATS`] times over at least [`CYCLE_MIN_SPAN`] tokens.
    fn push(&mut self, token: u32) -> Option<u32> {
        let n = self.seen;
        let mut hit = None;
        for p in 1..=n.min(CYCLE_MAX_PERIOD) {
            let run = if self.history[(n - p) % CYCLE_MAX_PERIOD] == token {
                self.runs[p] + 1
            } else {
                0
            };
            self.runs[p] = run;
            let period = p as u32;
            if hit.is_none()
                && run >= (CYCLE_MIN_REPEATS - 1) * period
                && run + period >= CYCLE_MIN_SPAN
            {
                hit = Some(period);
            }
        }
        self.history[n % CYCLE_MAX_PERIOD] = token;
        self.seen += 1;
        hit
    }
}

/// How a turn's thinking ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReasoningClose {
    /// The model wrote the close itself.
    Model,
    /// The budget ran out and the end was forced.
    Budget,
    /// The cycle guard lowered the budget, and the end was forced.
    Cycle,
    /// The turn ended in thinking.
    Open,
}

impl ReasoningClose {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Budget => "budget",
            Self::Cycle => "cycle",
            Self::Open => "open",
        }
    }
}

/// What a turn's thinking did, read from the tracker when the turn ends.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ReasoningOutcome {
    /// Tokens generated during reasoning before the close, the end tokens a
    /// cut forces included (Splash `count_reasoning_tokens`).
    pub tokens: u32,
    /// `None` when the turn did not start in thinking or has no close token.
    pub close: Option<ReasoningClose>,
    /// Thinking tokens when a cut began.
    pub cut_at: Option<u32>,
    /// Period in tokens of the cycle the guard saw.
    pub cycle_period: Option<u32>,
    /// Close tokens observed after thinking ended.
    pub stray_closes: u32,
}

impl ReasoningOutcome {
    /// An outcome with a count only.
    #[cfg(test)]
    pub(crate) fn counted(tokens: u32) -> Self {
        Self {
            tokens,
            ..Self::default()
        }
    }

    /// `ChatResult::reasoning_close`.
    pub(crate) fn close_name(&self) -> Option<String> {
        self.close.map(|close| close.as_str().to_string())
    }

    /// `ChatResult::reasoning_stray_closes`: present with the close.
    pub(crate) fn reported_stray_closes(&self) -> Option<u32> {
        self.close.map(|_| self.stray_closes)
    }
}

/// Tracks reasoning vs content state during token-by-token generation.
///
/// The template opens `<think>` when thinking is on; the model thinks, then
/// emits the close token, then content. When a thinking budget runs out the
/// turn ends thinking with `end` (Splash `ThinkingBudget.end`: an end text when
/// the request set one, then the chat template's close, e.g. Qwen's
/// `"\n</think>"`), by an AR loop one token per step, by a speculative cycle
/// as its rows from the cut row on ([`ThinkingCut`]), so the next request's
/// re-render of the reply matches the tokens decoded.
///
/// With the cycle guard on, an exact repeat in the thinking tokens lowers the
/// budget to [`CYCLE_CUT_GRACE`] tokens past the hit; the cut then runs the
/// same way.
///
/// An end token is in flight from [`Self::take_forced_token`] until it is
/// observed, and the next one is handed out only after that (Splash
/// `commitSelected` advances the end on commit). A loop that picks the next
/// token before observing the current one and a loop that observes first both
/// emit the end once, in order. A speculative cycle takes nothing: it observes
/// the end tokens its cut rows selected.
pub(crate) struct ReasoningTracker {
    in_thinking: bool,
    started_in_thinking: bool,
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
    cycles: Option<Box<CycleDetector>>,
    cycle_period: Option<u32>,
    cycle_lowered: bool,
    cut_at: Option<u32>,
    close: Option<ReasoningClose>,
    stray_closes: u32,
}

impl ReasoningTracker {
    fn with_end(
        setup: &crate::engine::backend::ThinkingSetup,
        end: Arc<[u32]>,
        extra_think_end_ids: Vec<u32>,
    ) -> Self {
        let starts_in_thinking = setup.enabled;
        // Budget 0 ends thinking before any thinking token.
        let cut = starts_in_thinking && setup.budget == Some(0) && !end.is_empty();
        let cycles = (setup.cycle_guard && starts_in_thinking && !end.is_empty() && !cut)
            .then(|| Box::new(CycleDetector::new()));
        Self {
            in_thinking: starts_in_thinking,
            started_in_thinking: starts_in_thinking,
            thinking_token_count: 0,
            budget: setup.budget,
            end,
            extra_think_end_ids,
            cut,
            taken: 0,
            observed: 0,
            cycles,
            cycle_period: None,
            cycle_lowered: false,
            cut_at: cut.then_some(0),
            close: None,
            stray_closes: 0,
        }
    }

    /// A tracker over explicit end tokens, for tests.
    #[cfg(test)]
    pub(crate) fn new(starts_in_thinking: bool, budget: Option<i32>, end: &[u32]) -> Self {
        Self::guarded(starts_in_thinking, budget, end, false)
    }

    /// [`Self::new`] with the cycle guard set, for tests.
    #[cfg(test)]
    pub(crate) fn guarded(
        starts_in_thinking: bool,
        budget: Option<i32>,
        end: &[u32],
        cycle_guard: bool,
    ) -> Self {
        Self::with_end(
            &crate::engine::backend::ThinkingSetup {
                enabled: starts_in_thinking,
                budget,
                cycle_guard,
            },
            Arc::from(end),
            Vec::new(),
        )
    }

    /// The one production construction point: the turn's thinking state and
    /// its close of thinking (`ChatBackend::think_end_for_turn`, or
    /// `Qwen3Tokenizer::think_end` where a family core resolves no override).
    pub(crate) fn from_setup(
        setup: &crate::engine::backend::ThinkingSetup,
        close: &crate::engine::backend::ThinkEndResolution,
    ) -> Self {
        Self::with_end(
            setup,
            close.thinking_end.clone(),
            close.think_end_extra_ids.clone(),
        )
    }

    fn is_close(&self, token_id: u32) -> bool {
        self.end.last() == Some(&token_id) || self.extra_think_end_ids.contains(&token_id)
    }

    /// Process a generated token. Returns whether this token is reasoning content.
    ///
    /// Call AFTER extracting the token ID from the GPU each decode step.
    pub fn observe_token(&mut self, token_id: u32) -> bool {
        if !self.in_thinking {
            if self.started_in_thinking && !self.end.is_empty() && self.is_close(token_id) {
                self.stray_closes += 1;
            }
            return false;
        }

        if self.is_close(token_id) {
            self.close = Some(match (self.cut, self.cycle_lowered) {
                (false, _) => ReasoningClose::Model,
                (true, false) => ReasoningClose::Budget,
                (true, true) => ReasoningClose::Cycle,
            });
            self.in_thinking = false;
            self.cut = false;
            self.taken = 0;
            self.observed = 0;
            self.cycles = None;
            return true; // the close itself is part of reasoning
        }

        self.thinking_token_count += 1;
        if self.cut {
            if self.end.get(self.observed) == Some(&token_id) {
                self.observed += 1;
                self.taken = self.taken.max(self.observed);
            } else {
                debug_assert!(false, "thinking went past its budget");
            }
        } else if !self.end.is_empty() {
            if let Some(cycles) = self.cycles.as_mut()
                && let Some(period) = cycles.push(token_id)
            {
                self.cycles = None;
                self.cycle_period = Some(period);
                let limit = self.thinking_token_count + CYCLE_CUT_GRACE;
                if self.budget.is_none_or(|budget| limit < budget) {
                    self.budget = Some(limit);
                    self.cycle_lowered = true;
                }
            }
            if let Some(budget) = self.budget
                && self.thinking_token_count >= budget
            {
                self.cut = true;
                self.cut_at = Some(self.thinking_token_count as u32);
                self.cycles = None;
            }
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
    /// budget requires a forced end token.
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

    /// The cut a speculative cycle starting now must apply: `None` with no
    /// budget, no close, outside thinking, or once every end token is out.
    /// Read when a cycle starts; a guard hit inside the cycle lowers the
    /// budget at least [`CYCLE_CUT_GRACE`] tokens past the cycle's last row.
    pub(crate) fn thinking_cut(&self) -> Option<ThinkingCut> {
        let row = self.unforced_token_budget()?;
        let from = self.taken.max(self.observed);
        (from < self.end.len()).then(|| ThinkingCut {
            row,
            end: Arc::clone(&self.end),
            from,
            extra: self.extra_think_end_ids.clone(),
        })
    }

    /// The turn's thinking as it stands.
    pub(crate) fn outcome(&self) -> ReasoningOutcome {
        let close = if !self.started_in_thinking || self.end.is_empty() {
            None
        } else if self.in_thinking {
            Some(ReasoningClose::Open)
        } else {
            self.close
        };
        ReasoningOutcome {
            tokens: self.thinking_token_count.max(0) as u32,
            close,
            cut_at: close.and(self.cut_at),
            cycle_period: close.and(self.cycle_period),
            stray_closes: self.stray_closes,
        }
    }
}

/// Where a thinking budget cuts one speculative cycle (Splash
/// `ThinkingCutoff`). The cycle keeps its depth: rows before `row` are the
/// model's, rows from `row` on select the end tokens not yet emitted, one
/// each, and rows after the close are the model's again. At `row == 0` the end
/// tokens also take the place of the cycle's first proposals, so one cycle
/// commits up to `drafts + 1` of them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ThinkingCut {
    /// Thinking tokens left before the end.
    pub row: usize,
    end: Arc<[u32]>,
    /// `end[from..]`: the end tokens not yet emitted, the close last.
    from: usize,
    extra: Vec<u32>,
}

impl ThinkingCut {
    fn left(&self) -> usize {
        self.end.len() - self.from
    }

    fn is_close(&self, token: i32) -> bool {
        u32::try_from(token)
            .is_ok_and(|token| self.end.last() == Some(&token) || self.extra.contains(&token))
    }

    /// The cut row is one of a cycle's `drafts + 1` rows.
    pub(crate) fn touches(&self, drafts: usize) -> bool {
        self.row <= drafts
    }

    /// Proposals the end tokens replace (Splash `replacedProposals`); they
    /// are not counted as drafted.
    pub(crate) fn replaced(&self, drafts: usize) -> usize {
        if self.row != 0 {
            0
        } else {
            self.left().min(drafts)
        }
    }

    /// The end tokens that take the place of a cycle's first proposals.
    pub(crate) fn proposals(&self, drafts: usize) -> &[u32] {
        &self.end[self.from..self.from + self.replaced(drafts)]
    }

    /// The token each verify row must select, for a cycle that verifies
    /// `drafts`; row `drafts.len()` is the bonus row. No row is decided when
    /// a draft before the cut row closes thinking: that row, if accepted,
    /// ends the thought by the model.
    pub(crate) fn targets(&self, drafts: &[i32]) -> CutTargets {
        let before = &drafts[..self.row.min(drafts.len())];
        let closed = before.iter().any(|&d| self.is_close(d));
        CutTargets {
            first: self.row,
            tokens: if closed {
                Vec::new()
            } else {
                self.end[self.from..].to_vec()
            },
        }
    }
}

/// The rows of one cycle that a [`ThinkingCut`] decides.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct CutTargets {
    first: usize,
    tokens: Vec<u32>,
}

impl CutTargets {
    /// The end token row `row` selects; `None` for a row that is the model's.
    pub(crate) fn at(&self, row: usize) -> Option<u32> {
        row.checked_sub(self.first)
            .and_then(|index| self.tokens.get(index).copied())
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
        assert_eq!(tracker.outcome().tokens, 3);

        // Budget 0: the first token is the end's first.
        let mut zero = ReasoningTracker::new(true, Some(0), &[8, 9]);
        assert_eq!(zero.take_forced_token(), Some(8));
        assert_eq!(zero.take_forced_token(), None);
        zero.observe_token(8);
        assert_eq!(zero.take_forced_token(), Some(9));
        zero.observe_token(9);
        assert_eq!(zero.outcome().tokens, 1);
    }

    #[test]
    #[should_panic(expected = "thinking went past its budget")]
    fn a_token_past_the_budget_is_a_defect() {
        let mut tracker = ReasoningTracker::new(true, Some(1), &[8, 9]);
        tracker.observe_token(100); // the cut starts
        tracker.observe_token(101); // not the end: Splash commitSelected throws
    }

    /// 100 distinct tokens, then one 300-token block over and over.
    fn looping_thought() -> impl Iterator<Item = u32> {
        (1000..1100).chain((0..).flat_map(|_| 2000..2300))
    }

    /// A lane: the forced token when one is due, else the next thought token,
    /// until the close or `limit` tokens.
    fn run_lane(
        tracker: &mut ReasoningTracker,
        mut thought: impl Iterator<Item = u32>,
        limit: usize,
    ) -> Vec<u32> {
        let mut emitted = Vec::new();
        while emitted.len() < limit {
            let token = match tracker.take_forced_token() {
                Some(token) => token,
                None => thought.next().unwrap(),
            };
            let reasoning = tracker.observe_token(token);
            emitted.push(token);
            if reasoning
                && !tracker
                    .outcome()
                    .close
                    .is_none_or(|c| c == ReasoningClose::Open)
            {
                break;
            }
        }
        emitted
    }

    #[test]
    fn a_cycle_closes_the_thought_64_tokens_after_two_periods() {
        let mut tracker = ReasoningTracker::guarded(true, None, &[8, 9], true);
        let mut thought = looping_thought();
        for _ in 0..699 {
            tracker.observe_token(thought.next().unwrap());
        }
        assert_eq!(tracker.unforced_token_budget(), None);
        tracker.observe_token(thought.next().unwrap()); // count 700: two periods
        assert_eq!(tracker.unforced_token_budget(), Some(64));

        let mut tracker = ReasoningTracker::guarded(true, None, &[8, 9], true);
        let emitted = run_lane(&mut tracker, looping_thought(), 2000);
        assert_eq!(emitted.len(), 766);
        assert!(
            emitted[..764]
                .iter()
                .copied()
                .eq(looping_thought().take(764))
        );
        assert_eq!(&emitted[764..], &[8, 9]);
        let outcome = tracker.outcome();
        assert_eq!(outcome.close, Some(ReasoningClose::Cycle));
        assert_eq!(outcome.cycle_period, Some(300));
        assert_eq!(outcome.cut_at, Some(764));
        assert_eq!(outcome.tokens, 764 + 2 - 1);
    }

    #[test]
    fn two_periods_are_enough() {
        let mut tracker = ReasoningTracker::guarded(true, None, &[8, 9], true);
        let emitted = run_lane(&mut tracker, looping_thought(), 2000);
        assert_eq!(tracker.outcome().cut_at, Some(764), "not 1,064");
        assert_eq!(emitted.len(), 766);
    }

    #[test]
    fn short_or_broken_repeats_are_not_cycles() {
        let pattern = |len: usize| (0..len).map(|i| 500 + (i % 2) as u32);
        let cases: [(&str, Vec<u32>); 4] = [
            (
                "66-token block x 2.45, then new text",
                (0..162)
                    .map(|i| 3000 + (i % 66) as u32)
                    .chain(4000..4400)
                    .collect(),
            ),
            ("400 distinct tokens", (5000..5400).collect()),
            (
                "a block, 50 other tokens, the block again",
                (6000..6300).chain(7000..7050).chain(6000..6300).collect(),
            ),
            ("a 2-token pattern for 400 tokens", pattern(400).collect()),
        ];
        for (name, thought) in cases {
            let mut tracker = ReasoningTracker::guarded(true, None, &[8, 9], true);
            for token in thought {
                tracker.observe_token(token);
            }
            assert_eq!(tracker.unforced_token_budget(), None, "{name}");
            assert_eq!(tracker.outcome().cycle_period, None, "{name}");
        }
        let mut tracker = ReasoningTracker::guarded(true, None, &[8, 9], true);
        for token in pattern(512) {
            tracker.observe_token(token);
        }
        assert_eq!(tracker.outcome().cycle_period, Some(2));
        assert_eq!(tracker.unforced_token_budget(), Some(64));
    }

    #[test]
    fn a_block_in_flight_at_the_hit_stays_inside_the_grace() {
        let mut tracker = ReasoningTracker::guarded(true, None, &[8, 9], true);
        let mut thought = (0..).map(|i| 500 + (i % 2) as u32);
        for _ in 0..512 {
            tracker.observe_token(thought.next().unwrap());
        }
        for _ in 0..16 {
            tracker.observe_token(thought.next().unwrap());
        }
        assert_eq!(tracker.unforced_token_budget(), Some(48));
        assert!(!tracker.ending());
    }

    #[test]
    fn the_close_names_its_cause() {
        let cut = |budget: Option<i32>| {
            let mut tracker = ReasoningTracker::guarded(true, budget, &[8, 9], true);
            run_lane(&mut tracker, looping_thought(), 2000);
            tracker.outcome()
        };
        let early = cut(Some(500));
        assert_eq!(early.close, Some(ReasoningClose::Budget));
        assert_eq!((early.cut_at, early.cycle_period), (Some(500), None));
        let late = cut(Some(5000));
        assert_eq!(late.close, Some(ReasoningClose::Cycle));
        assert_eq!((late.cut_at, late.cycle_period), (Some(764), Some(300)));
        let near = cut(Some(740));
        assert_eq!(near.close, Some(ReasoningClose::Budget));
        assert_eq!((near.cut_at, near.cycle_period), (Some(740), Some(300)));

        let mut model = ReasoningTracker::guarded(true, None, &[8, 9], true);
        let emitted = run_lane(&mut model, looping_thought().take(720).chain([9]), 2000);
        assert_eq!(emitted.len(), 721);
        let model = model.outcome();
        assert_eq!(model.close, Some(ReasoningClose::Model));
        assert_eq!((model.cut_at, model.cycle_period), (None, Some(300)));

        let mut open = ReasoningTracker::guarded(true, None, &[8, 9], true);
        let emitted = run_lane(&mut open, looping_thought(), 765);
        assert_eq!(emitted[764], 8);
        let open = open.outcome();
        assert_eq!(open.close, Some(ReasoningClose::Open));
        assert_eq!((open.cut_at, open.cycle_period), (Some(764), Some(300)));
    }

    #[test]
    fn a_cycle_hit_cuts_as_a_budget_of_that_size() {
        let mut probe = ReasoningTracker::guarded(true, None, &[8, 9], true);
        let mut thought = looping_thought();
        let mut hit = 0;
        while probe.unforced_token_budget().is_none() {
            probe.observe_token(thought.next().unwrap());
            hit += 1;
        }
        assert_eq!(hit, 700);

        let mut guarded = ReasoningTracker::guarded(true, None, &[8, 9], true);
        let mut budgeted =
            ReasoningTracker::guarded(true, Some(hit + CYCLE_CUT_GRACE), &[8, 9], false);
        let mut thought = looping_thought();
        let mut emitted = 0;
        while guarded.in_thinking || budgeted.in_thinking {
            assert!(emitted < 2000, "the cut never closed the thought");
            let forced = guarded.take_forced_token();
            assert_eq!(forced, budgeted.take_forced_token(), "token {emitted}");
            let token = forced.unwrap_or_else(|| thought.next().unwrap());
            assert_eq!(guarded.observe_token(token), budgeted.observe_token(token));
            emitted += 1;
            if emitted >= hit {
                assert_eq!(
                    guarded.unforced_token_budget(),
                    budgeted.unforced_token_budget(),
                    "token {emitted}"
                );
                assert_eq!(guarded.ending(), budgeted.ending(), "token {emitted}");
                assert_eq!(
                    guarded.peek_forced_token(),
                    budgeted.peek_forced_token(),
                    "token {emitted}"
                );
                assert_eq!(
                    guarded.thinking_cut(),
                    budgeted.thinking_cut(),
                    "token {emitted}"
                );
            }
        }
        assert_eq!(emitted, 766);
        let (guarded, budgeted) = (guarded.outcome(), budgeted.outcome());
        assert_eq!(guarded.close, Some(ReasoningClose::Cycle));
        assert_eq!(budgeted.close, Some(ReasoningClose::Budget));
        assert_eq!(guarded.cut_at, budgeted.cut_at);
        assert_eq!(guarded.tokens, budgeted.tokens);
    }

    #[test]
    fn the_cut_follows_the_budget() {
        let mut tracker = ReasoningTracker::new(true, Some(5), &[8, 9]);
        for token in [100, 101, 102] {
            tracker.observe_token(token);
        }
        let cut = tracker.thinking_cut().expect("a budget cuts");
        assert_eq!((cut.row, cut.from, cut.left()), (2, 0, 2));
        tracker.observe_token(103);
        tracker.observe_token(104); // the cut starts
        let cut = tracker.thinking_cut().expect("the cut");
        assert_eq!((cut.row, cut.from, cut.left()), (0, 0, 2));
        tracker.observe_token(8);
        let cut = tracker.thinking_cut().expect("the close is left");
        assert_eq!((cut.row, cut.from, cut.left()), (0, 1, 1));
        assert_eq!(cut.proposals(3), &[9]);
        tracker.observe_token(9);
        assert_eq!(tracker.thinking_cut(), None);

        let mut taken = ReasoningTracker::new(true, Some(0), &[8, 9]);
        assert_eq!(taken.take_forced_token(), Some(8));
        assert_eq!(
            taken.thinking_cut().map(|cut| cut.from),
            Some(1),
            "an end token in flight is not handed out twice"
        );
        taken.observe_token(8);
        assert_eq!(taken.thinking_cut().map(|cut| cut.from), Some(1));

        assert_eq!(
            ReasoningTracker::new(true, None, &[8, 9]).thinking_cut(),
            None
        );
        assert_eq!(
            ReasoningTracker::new(true, Some(2), &[]).thinking_cut(),
            None
        );
        assert_eq!(
            ReasoningTracker::new(false, Some(2), &[8, 9]).thinking_cut(),
            None
        );
    }

    #[test]
    fn a_cycle_observes_its_end_tokens_without_a_take() {
        let mut cycle = ReasoningTracker::new(true, Some(1), &[7, 8, 9]);
        cycle.observe_token(100); // the cut starts
        assert!(cycle.observe_token(7));
        assert!(cycle.observe_token(8));
        assert_eq!(cycle.take_forced_token(), Some(9));
        assert!(cycle.observe_token(9));
        assert!(!cycle.observe_token(5));
        let outcome = cycle.outcome();
        assert_eq!(outcome.close, Some(ReasoningClose::Budget));
        assert_eq!((outcome.tokens, outcome.cut_at), (3, Some(1)));

        // An AR step takes the first end token, a cycle the rest.
        let mut mixed = ReasoningTracker::new(true, Some(1), &[7, 8, 9]);
        mixed.observe_token(100);
        assert_eq!(mixed.take_forced_token(), Some(7));
        mixed.observe_token(7);
        mixed.observe_token(8);
        assert_eq!(mixed.thinking_cut().map(|cut| cut.from), Some(2));
        assert_eq!(mixed.take_forced_token(), Some(9));
        mixed.observe_token(9);
        assert_eq!(mixed.outcome(), outcome);
    }

    #[test]
    fn cut_targets_stop_at_a_close_before_the_row() {
        let mut tracker = ReasoningTracker::new(true, Some(4), &[8, 9]);
        tracker.observe_token(100);
        tracker.observe_token(101);
        let cut = tracker.thinking_cut().expect("row 2");
        assert!(cut.touches(2) && !cut.touches(1));
        assert_eq!((cut.replaced(3), cut.proposals(3)), (0, &[][..]));
        let open = cut.targets(&[5, 6, 7]);
        assert_eq!(
            (0..5).map(|row| open.at(row)).collect::<Vec<_>>(),
            vec![None, None, Some(8), Some(9), None]
        );
        let closed = cut.targets(&[5, 9, 7]);
        assert!((0..4).all(|row| closed.at(row).is_none()));
        // A close at or after the cut row does not stop the cut.
        assert_eq!(cut.targets(&[5, 6, 9]).at(2), Some(8));

        let mut extra = ReasoningTracker::with_end(
            &crate::engine::backend::ThinkingSetup {
                enabled: true,
                budget: Some(2),
                cycle_guard: false,
            },
            Arc::from(&[8u32, 9][..]),
            vec![11],
        );
        extra.observe_token(100);
        let cut = extra.thinking_cut().expect("row 1");
        assert!((0..3).all(|row| cut.targets(&[11, 5]).at(row).is_none()));

        // Row 0: the end replaces the first proposals, then the model's rows.
        tracker.observe_token(102);
        tracker.observe_token(103);
        tracker.observe_token(8);
        let cut = tracker.thinking_cut().expect("the close is left");
        assert_eq!((cut.replaced(3), cut.proposals(3)), (1, &[9][..]));
        let targets = cut.targets(&[9, 4, 4]);
        assert_eq!(
            (0..4).map(|row| targets.at(row)).collect::<Vec<_>>(),
            vec![Some(9), None, None, None]
        );
    }

    #[test]
    fn closes_after_the_thought_are_stray() {
        let mut tracker = ReasoningTracker::new(true, None, &[8, 9]);
        for token in [100, 9, 5, 9, 6] {
            tracker.observe_token(token);
        }
        let outcome = tracker.outcome();
        assert_eq!(outcome.close, Some(ReasoningClose::Model));
        assert_eq!(outcome.stray_closes, 1);
        assert_eq!(outcome.close_name().as_deref(), Some("model"));
        assert_eq!(outcome.reported_stray_closes(), Some(1));
        assert_eq!((outcome.cut_at, outcome.cycle_period), (None, None));

        let mut content = ReasoningTracker::new(false, None, &[8, 9]);
        for token in [100, 9, 5] {
            content.observe_token(token);
        }
        let content = content.outcome();
        assert_eq!(content.close_name(), None);
        assert_eq!(content.reported_stray_closes(), None);
        assert_eq!((content.cut_at, content.cycle_period), (None, None));
    }

    #[test]
    fn without_the_guard_or_a_close_a_loop_runs_on() {
        for (guard, end) in [(false, &[8u32, 9][..]), (true, &[][..])] {
            let mut tracker = ReasoningTracker::guarded(true, None, end, guard);
            let emitted = run_lane(&mut tracker, looping_thought(), 3000);
            assert!(emitted.iter().copied().eq(looping_thought().take(3000)));
            let outcome = tracker.outcome();
            assert_eq!(outcome.tokens, 3000);
            assert_eq!(outcome.cycle_period, None);
            assert_eq!(outcome.cut_at, None);
            let close = (!end.is_empty()).then_some(ReasoningClose::Open);
            assert_eq!(outcome.close, close);
        }
    }
}
