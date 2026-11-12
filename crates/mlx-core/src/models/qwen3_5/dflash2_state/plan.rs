//! Where a DFlash2 prefill publishes states (Splash
//! `Engine::configureDraftStatePlan`) and which rows its draft window
//! captures for them (Splash `planDraftContext`).

use napi::bindgen_prelude::*;

/// One planned state: a disposable point is a checkpoint (Splash
/// `Request::StateBoundary`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StateBoundary {
    pub(crate) tokens: u32,
    pub(crate) disposable: bool,
}

/// `window` is the draft's sliding window (Splash `draftContextTokens`, the
/// minimum junction gain); `interval` the checkpoint spacing (Splash
/// `kPrefillCheckpointTokens`); `chunk` the checkpoint margin, a constant
/// (Splash `prefillTokenBudget`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct PlanInputs {
    pub(crate) resume: u32,
    pub(crate) junction: u32,
    pub(crate) prompt_replay: u32,
    pub(crate) replay_state: u32,
    pub(crate) window: u32,
    pub(crate) interval: u32,
    pub(crate) chunk: u32,
}

/// The turn's states in token order, each in `(resume, replay_state]`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct StatePlan {
    pub(crate) boundaries: Vec<StateBoundary>,
    pub(crate) prompt_replay: u32,
}

impl StatePlan {
    /// Splash `reserveCaptures` `stateClass` = `StateClass::InUse`.
    fn replays(&self, index: usize) -> bool {
        self.boundaries
            .get(index)
            .is_some_and(|point| !point.disposable && point.tokens == self.prompt_replay)
    }
}

/// Rows `[begin, end)` the draft window captures; `reset` starts the window
/// empty at `begin` (Splash `DraftCaptureSpan`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CaptureSpan {
    pub(crate) begin: u32,
    pub(crate) end: u32,
    pub(crate) reset: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct DraftContextPlan {
    pub(crate) spans: Vec<CaptureSpan>,
    pub(crate) restores_draft_state: bool,
}

/// Rows the prefill ends a chunk at, and the checkpoints it passes without
/// a capture (Splash `reserveCaptures` `passed`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct CommandBreaks {
    pub(crate) breaks: Vec<u32>,
    pub(crate) passed: Vec<u32>,
}

/// Splash `kPrefillCheckpointTokens`: two draft windows, whole KV blocks.
pub(crate) fn checkpoint_interval(window: u32, block: u32) -> u32 {
    window.saturating_mul(2).next_multiple_of(block.max(1))
}

/// Splash `Engine::replayStateBoundary`: the prompt's replay point while
/// the lane replays the prompt alone, else the last whole block before the
/// history's last token, a state no later request shares.
pub(crate) fn replay_state_boundary(
    exact: u32,
    prompt_tokens: u32,
    prompt_replay: u32,
    block: u32,
) -> u32 {
    if exact == prompt_tokens {
        return prompt_replay;
    }
    let block = block.max(1);
    exact.saturating_sub(1) / block * block
}

/// Splash `plannedCheckpoints`.
pub(crate) fn planned_checkpoints(resume: u32, replay: u32, interval: u32, chunk: u32) -> Vec<u32> {
    let mut checkpoints = Vec::new();
    if interval == 0 {
        return checkpoints;
    }
    let interval = u64::from(interval);
    let chunk = u64::from(chunk);
    let mut boundary = (u64::from(resume) + chunk).div_ceil(interval) * interval;
    while boundary + chunk <= u64::from(replay) {
        checkpoints.push(boundary as u32);
        boundary += interval;
    }
    checkpoints
}

/// Splash `Engine::addStateBoundary`; false when `tokens` adds nothing.
fn add_state_boundary(
    plan: &mut Vec<StateBoundary>,
    after: u32,
    tokens: u32,
    disposable: bool,
    complete_after: bool,
    replay_state: u32,
    window: u32,
) -> bool {
    if tokens <= after || tokens > replay_state {
        return false;
    }
    let found = plan.partition_point(|point| point.tokens < tokens);
    if let Some(point) = plan.get_mut(found)
        && point.tokens == tokens
    {
        point.disposable = point.disposable && disposable;
        return false;
    }
    if !complete_after && found == 0 && tokens - after < window {
        return false;
    }
    plan.insert(found, StateBoundary { tokens, disposable });
    true
}

/// Splash `Engine::configureDraftStatePlan`, shared-prefill points aside.
pub(crate) fn configure_draft_state_plan(inputs: &PlanInputs) -> StatePlan {
    let PlanInputs {
        resume,
        junction,
        prompt_replay,
        replay_state,
        window,
        interval,
        chunk,
    } = *inputs;
    let mut boundaries = Vec::new();
    let mut add = |tokens: u32, disposable: bool| {
        add_state_boundary(
            &mut boundaries,
            resume,
            tokens,
            disposable,
            true,
            replay_state,
            window,
        )
    };
    for checkpoint in planned_checkpoints(resume, replay_state, interval, chunk) {
        add(checkpoint, true);
    }
    if u64::from(junction) >= u64::from(resume) + u64::from(window) {
        add(junction, false);
    }
    add(replay_state, replay_state != prompt_replay);
    add(prompt_replay, false);
    StatePlan {
        boundaries,
        prompt_replay,
    }
}

/// Splash `planDraftContext` over the replay `[begin, end)` and the sorted,
/// unique `boundaries` in `(begin, end]`; the active state at `end` is the
/// last point.
pub(crate) fn plan_draft_context(
    begin: u32,
    end: u32,
    boundaries: &[u32],
    window: u32,
) -> Result<DraftContextPlan> {
    if end < begin {
        return Err(Error::from_reason("draft replay range is reversed"));
    }
    let mut previous: Option<u32> = None;
    for &boundary in boundaries {
        if boundary <= begin || boundary > end {
            return Err(Error::from_reason(
                "draft materialization boundary is outside replay range",
            ));
        }
        if previous.is_some_and(|previous| boundary <= previous) {
            return Err(Error::from_reason(
                "draft materialization boundaries are not sorted and unique",
            ));
        }
        previous = Some(boundary);
    }
    let mut plan = DraftContextPlan::default();
    let mut state = begin;
    let mut have_state = begin != 0;
    let mut add = |boundary: u32, plan: &mut DraftContextPlan| {
        let distance = boundary - state;
        if !(have_state && distance == 0) {
            let continue_state = have_state && distance < window;
            let capture_begin = if continue_state {
                state
            } else {
                boundary - boundary.min(window)
            };
            if boundary != capture_begin {
                match plan.spans.last_mut() {
                    Some(last) if continue_state && last.end == capture_begin => {
                        last.end = boundary;
                    }
                    _ => plan.spans.push(CaptureSpan {
                        begin: capture_begin,
                        end: boundary,
                        reset: !continue_state,
                    }),
                }
            }
        }
        state = boundary;
        have_state = true;
    };
    for &boundary in boundaries {
        if boundary < end {
            add(boundary, &mut plan);
        }
    }
    add(end, &mut plan);
    let first = boundaries
        .first()
        .copied()
        .filter(|&first| first < end)
        .unwrap_or(end);
    plan.restores_draft_state = begin != 0 && first - begin < window;
    Ok(plan)
}

/// The prefill's chunk ends from `resume` to `end` with chunks of at most
/// `chunk` rows, cut at every planned point: a checkpoint is passed when
/// the point after it lies in the same chunk and is not the prompt's
/// replay point (Splash `reserveCaptures` `passed`), since publishing that
/// point would retire it at once.
pub(crate) fn command_breaks(resume: u32, end: u32, chunk: u32, plan: &StatePlan) -> CommandBreaks {
    let points = &plan.boundaries;
    let chunk = chunk.max(1);
    let mut out = CommandBreaks::default();
    let mut index = points.partition_point(|point| point.tokens <= resume);
    let mut start = resume;
    while start < end {
        let limit = start.saturating_add(chunk).min(end);
        let Some(point) = points
            .get(index)
            .filter(|point| point.tokens <= limit && point.tokens < end)
        else {
            start = limit;
            continue;
        };
        let passed = point.disposable
            && points
                .get(index + 1)
                .is_some_and(|next| next.tokens <= limit && !plan.replays(index + 1));
        index += 1;
        if passed {
            out.passed.push(point.tokens);
            continue;
        }
        out.breaks.push(point.tokens);
        start = point.tokens;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const WINDOW: u32 = 2048;
    const CHUNK: u32 = 2048;

    fn inputs(resume: u32, junction: u32, prompt_replay: u32, replay_state: u32) -> PlanInputs {
        PlanInputs {
            resume,
            junction,
            prompt_replay,
            replay_state,
            window: WINDOW,
            interval: checkpoint_interval(WINDOW, 16),
            chunk: CHUNK,
        }
    }

    fn points(plan: &StatePlan) -> Vec<(u32, bool)> {
        plan.boundaries
            .iter()
            .map(|point| (point.tokens, point.disposable))
            .collect()
    }

    fn span(begin: u32, end: u32, reset: bool) -> CaptureSpan {
        CaptureSpan { begin, end, reset }
    }

    /// Mutation CK1 (`planned_checkpoints` starts without the `+ chunk`
    /// margin) → red.
    #[test]
    fn planned_checkpoints_match_splash() {
        assert_eq!(checkpoint_interval(WINDOW, 16), 4096);
        assert_eq!(checkpoint_interval(8, 16), 16);
        assert_eq!(
            planned_checkpoints(0, 20_000, 4096, 2048),
            vec![4096, 8192, 12_288, 16_384]
        );
        assert_eq!(
            planned_checkpoints(3000, 9000, 4096, 2048),
            Vec::<u32>::new()
        );
        assert_eq!(planned_checkpoints(0, 6143, 4096, 2048), Vec::<u32>::new());
        assert_eq!(planned_checkpoints(0, 6144, 4096, 2048), vec![4096]);
        assert_eq!(
            planned_checkpoints(4096, 20_000, 4096, 2048),
            vec![8192, 12_288, 16_384]
        );
        assert_eq!(planned_checkpoints(0, 20_000, 0, 2048), Vec::<u32>::new());
    }

    /// Mutations JG1 (junction gain 0: J 1000 kept) and PS1 (`passed`
    /// ignores the replay point: 4096 passed before R) → red.
    #[test]
    fn plan_merges_points_like_configure_draft_state_plan() {
        let plan = configure_draft_state_plan(&inputs(0, 5000, 20_000, 20_000));
        assert_eq!(
            points(&plan),
            vec![
                (4096, true),
                (5000, false),
                (8192, true),
                (12_288, true),
                (16_384, true),
                (20_000, false),
            ]
        );
        assert_eq!(plan.prompt_replay, 20_000);

        let short = configure_draft_state_plan(&inputs(0, 1000, 20_000, 20_000));
        assert!(
            short.boundaries.iter().all(|point| point.tokens != 1000),
            "a junction short of the gain is dropped: {:?}",
            points(&short)
        );

        let merged = configure_draft_state_plan(&inputs(0, 8192, 20_000, 20_000));
        assert!(merged.boundaries.contains(&StateBoundary {
            tokens: 8192,
            disposable: false
        }));
        assert_eq!(merged.boundaries.len(), 5);

        // A suspended lane that replays the prompt alone keeps its replay point.
        let prompt = 20_010;
        let same = replay_state_boundary(prompt, prompt, 20_000, 16);
        assert_eq!(same, 20_000);
        let resumed = configure_draft_state_plan(&inputs(8192, 0, 20_000, same));
        assert_eq!(
            points(&resumed),
            vec![(12_288, true), (16_384, true), (20_000, false)]
        );
        // One that replays generated history ends at a checkpoint of its own.
        let generated = replay_state_boundary(prompt + 50, prompt, 20_000, 16);
        assert_eq!(generated, 20_048);
        let resumed = configure_draft_state_plan(&inputs(8192, 0, 20_000, generated));
        assert_eq!(
            points(&resumed),
            vec![
                (12_288, true),
                (16_384, true),
                (20_000, false),
                (20_048, true)
            ]
        );

        let past = configure_draft_state_plan(&inputs(0, 20_032, 20_000, 20_000));
        assert!(past.boundaries.iter().all(|point| point.tokens <= 20_000));
        assert_eq!(past.boundaries.len(), 5);

        // A checkpoint whose next point lies in the same chunk is passed...
        let junction = StatePlan {
            boundaries: vec![
                StateBoundary {
                    tokens: 4096,
                    disposable: true,
                },
                StateBoundary {
                    tokens: 5000,
                    disposable: false,
                },
                StateBoundary {
                    tokens: 5040,
                    disposable: false,
                },
            ],
            prompt_replay: 5040,
        };
        assert_eq!(
            command_breaks(1000, 6000, CHUNK, &junction),
            CommandBreaks {
                breaks: vec![5000, 5040],
                passed: vec![4096],
            }
        );
        // ...unless that point is the prompt's replay point.
        let replay = StatePlan {
            boundaries: junction.boundaries[..2].to_vec(),
            prompt_replay: 5000,
        };
        assert_eq!(
            command_breaks(1000, 6000, CHUNK, &replay),
            CommandBreaks {
                breaks: vec![4096, 5000],
                passed: Vec::new(),
            }
        );
        // Chunk-aligned checkpoints are never passed.
        assert_eq!(
            command_breaks(0, 20_010, CHUNK, &plan),
            CommandBreaks {
                breaks: vec![4096, 5000, 8192, 12_288, 16_384, 20_000],
                passed: Vec::new(),
            }
        );
    }

    /// Mutation DS1 (`<=` in `restores_draft_state`: a first point one
    /// window past the resume restores) → red.
    #[test]
    fn draft_spans_match_plan_draft_context() -> Result<()> {
        let plan = plan_draft_context(0, 9000, &[4096, 8192], WINDOW)?;
        assert_eq!(
            plan.spans,
            vec![span(2048, 4096, true), span(6144, 9000, true)]
        );
        assert!(!plan.restores_draft_state);

        let plan = plan_draft_context(3000, 5000, &[4096], WINDOW)?;
        assert_eq!(plan.spans, vec![span(3000, 5000, false)]);
        assert!(plan.restores_draft_state);

        let plan = plan_draft_context(3000, 9000, &[8192], WINDOW)?;
        assert_eq!(plan.spans, vec![span(6144, 9000, true)]);
        assert!(!plan.restores_draft_state);

        let plan = plan_draft_context(3000, 3500, &[], WINDOW)?;
        assert_eq!(plan.spans, vec![span(3000, 3500, false)]);
        assert!(plan.restores_draft_state);

        let plan = plan_draft_context(3000, 5048, &[], WINDOW)?;
        assert_eq!(plan.spans, vec![span(3000, 5048, true)]);
        assert!(!plan.restores_draft_state);

        // The active state at a planned point is that point.
        let plan = plan_draft_context(0, 4096, &[4096], WINDOW)?;
        assert_eq!(plan.spans, vec![span(2048, 4096, true)]);

        assert!(plan_draft_context(10, 5, &[], WINDOW).is_err());
        assert!(plan_draft_context(0, 9000, &[0], WINDOW).is_err());
        assert!(plan_draft_context(0, 9000, &[9001], WINDOW).is_err());
        assert!(plan_draft_context(0, 9000, &[4096, 4096], WINDOW).is_err());
        Ok(())
    }
}
