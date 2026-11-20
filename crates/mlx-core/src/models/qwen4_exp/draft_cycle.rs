//! The DFlash2 draft's cycle on one sequence (Splash `DFlashDraft`): the
//! draft proposes its whole block, one row-exact `verify_chunk` runs the
//! anchor and the first `rows - 1` proposals, the engine's rule
//! (`accept_dspark_proposal`) accepts, and the commit keeps the accepted
//! rows in the target and appends their taps to the draft's context.
//! Greedy text equals the target's own: every verify row carries the bits
//! of the one-row step at its position.
use super::*;
use crate::engine::backend::{DsparkProposal, DsparkStepper, DsparkVerifyOutput};
use crate::engine::params::ChatParams;
use crate::engine::verification_budget::choose_verification_lengths;
use crate::models::qwen3_5::dflash2::{
    DFLASH2_DRAFT_PROPOSALS, DFlash2ContextCache, DFlash2Model, SelectorPath, load_dflash2,
    read_dflash2_config,
};
use crate::sampling::{SparseDistribution, is_greedy_temperature};
use crate::transformer::paged_kv_cache_adapter::SeqId;
use std::path::Path;

/// Rows one verify takes: the anchor and up to three proposals.
pub(in super::super) const DRAFT_VERIFY_ROWS: usize = 4;

/// Cost of a verify by its rows (index = rows; one row = the plain step) and
/// of a proposal with its commit. Only their ratios enter the rule.
const VERIFY_COST: [f64; DRAFT_VERIFY_ROWS + 1] = [0.0, 19.293, 25.046, 30.894, 37.861];
const PROPOSE_COST: f64 = 4.593 + 0.446;
/// More than one row must give this many times the plain step's tokens per
/// cost (the engine's own margin).
const MINIMUM_SPEEDUP: f64 = 1.05;
/// The kept rate of each proposal position a drafted turn starts from.
const KEPT_PRIOR: [f64; DRAFT_VERIFY_ROWS - 1] = [0.981, 0.952, 0.941];
const KEPT_ALPHA: f64 = 0.12;
const CONFIDENCE_WEIGHT: f64 = 0.25;

/// The block a drafted cycle proposed: the anchor's row, the anchor, and
/// the first proposals.
#[derive(Clone, Debug, PartialEq)]
struct Offer {
    row: usize,
    anchor: u32,
    block: Vec<u32>,
}

/// What a drafted turn's cycles kept: per proposal position, a moving rate
/// of the target keeping that proposal when it kept the ones before.
#[derive(Clone, Debug, PartialEq)]
pub(in super::super) struct DraftRecord {
    kept: [f64; DRAFT_VERIFY_ROWS - 1],
    last: Option<Offer>,
}

impl DraftRecord {
    pub(in super::super) fn new() -> Self {
        Self {
            kept: KEPT_PRIOR,
            last: None,
        }
    }

    #[cfg(test)]
    pub(in super::super) fn with_kept(kept: [f64; DRAFT_VERIFY_ROWS - 1]) -> Self {
        Self { kept, last: None }
    }

    #[cfg(test)]
    pub(in super::super) fn kept(&self) -> [f64; DRAFT_VERIFY_ROWS - 1] {
        self.kept
    }

    /// The cycle at `row` proposed `block` after `anchor`.
    pub(in super::super) fn offer(&mut self, row: usize, anchor: u32, block: &[u32]) {
        self.last = Some(Offer {
            row,
            anchor,
            block: block.iter().copied().take(self.kept.len()).collect(),
        });
    }

    /// The last offer's labels, read when the next cycle starts: `history`
    /// holds the rows its verify kept and `anchor` is the token the target
    /// picked after them. A proposal equal to the target's pick is kept;
    /// the first one that is not is refused; later ones are unknown.
    pub(in super::super) fn observe(&mut self, history: &[u32], anchor: u32) {
        let Some(offer) = self.last.take() else {
            return;
        };
        if history.get(offer.row) != Some(&offer.anchor)
            || history.len() > offer.row + DRAFT_VERIFY_ROWS
        {
            return;
        }
        let known = history.len() - offer.row;
        let matched = offer
            .block
            .iter()
            .zip(
                history[offer.row + 1..]
                    .iter()
                    .chain(std::iter::once(&anchor)),
            )
            .take_while(|(proposed, picked)| proposed == picked)
            .count();
        for (position, rate) in self.kept.iter_mut().enumerate() {
            let label = if position < matched {
                1.0
            } else if position < known {
                0.0
            } else {
                break;
            };
            *rate = (1.0 - KEPT_ALPHA) * *rate + KEPT_ALPHA * label;
            if position >= matched {
                break;
            }
        }
    }
}

/// The rows of the next verify, 1 to `ceiling` and at most
/// [`DRAFT_VERIFY_ROWS`], from the draft's confidence in its proposals (none
/// on the host walk) and the turn's record only. Each position's kept rate,
/// scaled by the confidence, is the chance the target keeps that proposal
/// after the ones before; the engine's rule picks the rows with the most
/// expected tokens per cost, and more than one row only when it beats the
/// plain step by [`MINIMUM_SPEEDUP`].
pub(in super::super) fn next_verify_rows(
    confidence: Option<&[f32]>,
    record: &DraftRecord,
    ceiling: usize,
) -> usize {
    let positions = ceiling.min(DRAFT_VERIFY_ROWS).saturating_sub(1);
    let conditional: Vec<f32> = record
        .kept
        .iter()
        .take(positions)
        .enumerate()
        .map(|(position, &kept)| {
            let factor = confidence
                .and_then(|values| values.get(position))
                .filter(|value| (0.0..=1.0).contains(*value))
                .map_or(1.0, |&value| {
                    (1.0 + CONFIDENCE_WEIGHT * (2.0 * f64::from(value) - 1.0)).clamp(0.5, 1.5)
                });
            (kept * factor).clamp(0.0, 1.0) as f32
        })
        .collect();
    let costs: Vec<Option<f64>> = VERIFY_COST
        .iter()
        .map(|&cost| (cost > 0.0).then_some(cost))
        .collect();
    choose_verification_lengths(
        &[conditional.as_slice()],
        &costs,
        PROPOSE_COST,
        VERIFY_COST[1],
        MINIMUM_SPEEDUP,
    )
    .and_then(|lengths| lengths.first().copied())
    .map_or(1, |proposals| proposals + 1)
}

/// The draft's whole block: host ids, on a sampled turn the rows each was
/// drawn from, and the draft's confidence in each when it was asked for and
/// the device walk gave it.
pub(in super::super) struct Proposed {
    pub(in super::super) ids: Vec<u32>,
    pub(in super::super) sparse: Vec<SparseDistribution>,
    pub(in super::super) confidence: Option<Vec<f32>>,
}

impl Proposed {
    /// The engine's proposal: the block's first `proposals`.
    fn proposal(self, proposals: usize) -> DsparkProposal {
        let mut proposal = DsparkProposal {
            draft_ids: self.ids.iter().map(|&id| id as i32).collect(),
            device_draft_ids: None,
            device_draft_sparse: None,
            draft_dists: Vec::new(),
            draft_sparse_dists: self.sparse,
            keep_probabilities: None,
        };
        proposal.truncate(proposals);
        proposal
    }
}

/// The rows of the draft's K/V ring: its sliding window less the query row.
pub(in super::super) fn ring_rows(draft: &DFlash2Model) -> usize {
    draft.config.sliding_window.saturating_sub(1)
}

/// Brings `context` to `history` from the taps of `passes` (Qwen3.5
/// `DraftTapSink::chunk`): rows the taps cover inside the ring's reach are
/// fused once per pass and appended; the rows before them only advance the
/// context's length. A context whose rows are not `history`'s first rows
/// starts again from none.
pub(in super::super) fn follow_history(
    draft: &DFlash2Model,
    context: &mut DFlash2ContextCache,
    passes: &[TappedRows],
    history: &[u32],
) -> Result<()> {
    let end = history.len();
    let mut start = usize::try_from(context.logical_len()).unwrap_or(usize::MAX);
    if start > end || context.token_history() != &history[..start] {
        context.reset();
        start = 0;
    }
    let mut covered = end;
    for pass in passes.iter().rev() {
        let pass_end = pass.first + pass.rows()?;
        if pass.first < covered && pass_end >= covered {
            covered = pass.first;
        } else if pass_end < covered {
            break;
        }
    }
    let kv = end.saturating_sub(ring_rows(draft)).max(start).max(covered);
    context.record_only(&history[start..kv])?;
    let mut next = kv;
    for pass in passes {
        let pass_end = (pass.first + pass.rows()?).min(end);
        if pass_end <= next {
            continue;
        }
        let rows = tapped_rows(std::slice::from_ref(pass), next, pass_end)?;
        let fused = draft.fuse_context(&rows)?;
        context.append(draft, &fused, next as i32, &history[next..pass_end])?;
        next = pass_end;
    }
    if next != end {
        return Err(Error::from_reason(format!(
            "Qwen4 draft context: the taps end at row {next} of {end}"
        )));
    }
    context.eval()
}

impl Decoder {
    /// Taps the draft's layers on every pass from here on, the finished
    /// passes bounded to the draft's ring.
    pub(in super::super) fn draft_taps_on(&mut self, draft: &DFlash2Model) -> Result<()> {
        self.set_hidden_taps(Some(&draft.config.target_layers))?;
        self.bound_hidden_taps(Some(ring_rows(draft)));
        Ok(())
    }

    /// No pass is tapped from here on.
    pub(in super::super) fn draft_taps_off(&mut self) {
        self.hidden_taps = None;
    }

    /// [`follow_history`] over the passes finished since the last take.
    pub(in super::super) fn draft_follow(
        &mut self,
        draft: &DFlash2Model,
        context: &mut DFlash2ContextCache,
    ) -> Result<()> {
        let passes = self.take_hidden_taps();
        follow_history(draft, context, &passes, &self.history)
    }
}

/// The verify a commit closes.
struct Verified {
    position: usize,
    tokens: Vec<u32>,
    states: Vec<DecoderState>,
}

/// One sequence's draft cycle over the decoder's active sequence; the
/// engine's speculative loop drives it in the order propose, verify,
/// commit.
pub(in super::super) struct DraftStepper<'a> {
    decoder: &'a mut Decoder,
    draft: &'a DFlash2Model,
    context: &'a mut DFlash2ContextCache,
    rows: usize,
    verified: Option<Verified>,
    /// Tests put their proposals in place of the draft's: (the anchor's
    /// row, the draft's proposals) -> the proposals.
    #[cfg(test)]
    pub(in super::super) forced: Option<Box<dyn FnMut(usize, &[u32]) -> Vec<u32> + 'a>>,
    /// Tests put their drawn proposals and the rows they drew them from in
    /// place of the draft's on a sampled turn.
    #[cfg(test)]
    pub(in super::super) drawn:
        Option<Box<dyn FnMut(u32) -> (Vec<u32>, Vec<SparseDistribution>) + 'a>>,
}

impl<'a> DraftStepper<'a> {
    /// The stepper over the decoder's history, `rows` rows a verify; the
    /// context first follows the passes finished since the last take.
    pub(in super::super) fn new(
        decoder: &'a mut Decoder,
        draft: &'a DFlash2Model,
        context: &'a mut DFlash2ContextCache,
        rows: usize,
    ) -> Result<Self> {
        if !(1..=DRAFT_VERIFY_ROWS).contains(&rows) {
            return Err(Error::from_reason(format!(
                "Qwen4 draft verifies 1-{DRAFT_VERIFY_ROWS} rows, not {rows}"
            )));
        }
        if decoder.hidden_tap_layers() != Some(draft.config.target_layers.as_slice()) {
            return Err(Error::from_reason(
                "Qwen4 draft: the target does not tap the draft's layers",
            ));
        }
        decoder.draft_follow(draft, context)?;
        Ok(Self {
            decoder,
            draft,
            context,
            rows,
            verified: None,
            #[cfg(test)]
            forced: None,
            #[cfg(test)]
            drawn: None,
        })
    }

    #[cfg(test)]
    pub(in super::super) fn decoder(&self) -> &Decoder {
        self.decoder
    }

    /// The draft's whole block after `anchor`, the token at the history's
    /// end; its confidence only when `confident` asks for it.
    pub(in super::super) fn propose_block(
        &mut self,
        anchor: u32,
        params: &ChatParams,
        rng: &mut dyn rand::Rng,
        confident: bool,
    ) -> Result<Proposed> {
        let position = self.decoder.history.len();
        if self.verified.is_some() {
            return Err(Error::from_reason(
                "Qwen4 draft: the last verify was not committed",
            ));
        }
        if usize::try_from(self.context.logical_len()).ok() != Some(position) {
            return Err(Error::from_reason(format!(
                "Qwen4 draft: the draft context ends at {}, the target at {position}",
                self.context.logical_len()
            )));
        }
        let temperature = params
            .sampling_config
            .unwrap_or_default()
            .temperature
            .unwrap_or(1.0);
        let greedy = is_greedy_temperature(temperature);
        let device_path = greedy
            && params.repetition_penalty == 1.0
            && params.presence_penalty == 0.0
            && params.frequency_penalty == 0.0;
        let mut confidence = None;
        let (path, mut rows) = self.draft.propose_with(
            &mut self.decoder.draft_view(),
            self.context,
            anchor,
            DFLASH2_DRAFT_PROPOSALS,
            temperature,
            device_path,
            if confident {
                Some(&mut confidence)
            } else {
                None
            },
            rng,
        )?;
        let ids: Vec<u32> = match path {
            SelectorPath::Device(ids) => {
                if let Some(values) = confidence.as_ref() {
                    MxArray::eval_arrays(&[&ids, values])?;
                }
                ids.to_uint32()?.to_vec()
            }
            SelectorPath::Host(ids) => ids.into_iter().map(|id| id as u32).collect(),
            SelectorPath::Sampled { ids, rows: drawn } => {
                rows = drawn.host()?;
                ids.to_uint32()?.to_vec()
            }
        };
        let confidence = match confidence {
            Some(values) => Some(values.to_float32()?.to_vec()),
            None => None,
        };
        #[cfg(test)]
        let ids = match self.forced.as_mut() {
            Some(forced) => forced(position, &ids),
            None => ids,
        };
        #[cfg(test)]
        let (ids, rows) = match self.drawn.as_mut().filter(|_| !greedy) {
            Some(drawn) => drawn(anchor),
            None => (ids, rows),
        };
        Ok(Proposed {
            ids,
            sparse: if greedy { Vec::new() } else { rows },
            confidence,
        })
    }
}

impl DsparkStepper for DraftStepper<'_> {
    /// The draft's whole block; the engine gets at most `rows - 1` of it.
    fn propose(
        &mut self,
        anchor_id: u32,
        max_len: usize,
        params: &ChatParams,
        rng: &mut dyn rand::Rng,
    ) -> Result<DsparkProposal> {
        let block = self.propose_block(anchor_id, params, rng, false)?;
        Ok(block.proposal(max_len.min(self.rows - 1)))
    }

    fn verify(&mut self, verify_ids: &[u32]) -> Result<DsparkVerifyOutput> {
        if self.verified.is_some() {
            return Err(Error::from_reason(
                "Qwen4 draft: the last verify was not committed",
            ));
        }
        if verify_ids.is_empty() || verify_ids.len() > self.rows {
            return Err(Error::from_reason(format!(
                "Qwen4 draft verifies 1-{} rows, not {}",
                self.rows,
                verify_ids.len()
            )));
        }
        let position = self.decoder.history.len();
        let (logits, states) = self.decoder.verify_chunk(verify_ids)?;
        self.verified = Some(Verified {
            position,
            tokens: verify_ids.to_vec(),
            states,
        });
        Ok(DsparkVerifyOutput {
            logits: logits.expand_dims(0)?,
        })
    }

    /// The target keeps the first `keep` verify rows (their state, the
    /// page cursor rolled back over the rest); their taps join the draft's
    /// context; the rejected rows' taps are dropped.
    fn commit(&mut self, keep: usize, total_written: usize) -> Result<()> {
        let verified = self
            .verified
            .take()
            .ok_or_else(|| Error::from_reason("Qwen4 draft commit has no verify"))?;
        if keep == 0 || keep > total_written || total_written != verified.tokens.len() {
            return Err(Error::from_reason(format!(
                "Qwen4 draft commit keeps {keep} of {total_written} rows of a {}-row verify",
                verified.tokens.len()
            )));
        }
        let passes = self.decoder.take_hidden_taps();
        let rows = tapped_rows(&passes, verified.position, verified.position + keep)?;
        drop(passes);
        let state = verified
            .states
            .into_iter()
            .nth(keep - 1)
            .ok_or_else(|| Error::from_reason("Qwen4 draft verify returned too few states"))?;
        if let Some(adapter) = self.decoder.paged.as_mut() {
            adapter
                .rollback_last_tokens((total_written - keep) as u32)
                .map_err(Error::from_reason)?;
        }
        self.decoder.restore_state(state);
        let fused = self.draft.fuse_context(&rows)?;
        self.context.append(
            self.draft,
            &fused,
            verified.position as i32,
            &verified.tokens[..keep],
        )?;
        self.context.eval()
    }

    fn eval_boundary(&self, token: &MxArray) {
        let mut arrays: Vec<&MxArray> = self
            .decoder
            .caches
            .iter()
            .flat_map(LayerCache::arrays)
            .collect();
        self.context.collect_arrays(&mut arrays);
        arrays.push(token);
        MxArray::async_eval_arrays(&arrays);
    }
}

/// Whether a turn that starts now runs the draft's cycle: a draft is
/// attached, the store holds every expert, its banks are filled, and no
/// other turn is open (a lane step takes no taps, and taps do not tell
/// sequences apart). A prefill while the banks fill runs as one group of
/// the whole prompt, whose taps no plan holds.
pub(in super::super) fn draft_turn(
    attached: bool,
    direct: bool,
    fill_done: bool,
    open_turns: usize,
) -> bool {
    attached && direct && fill_done && open_turns <= 1
}

/// Loads the DFlash2 draft in `dir` for a target of config `c`: its config
/// is checked against the target before any tensor is read, and the loaded
/// model again.
pub(in super::super) fn load_draft(dir: &Path, c: &Config) -> Result<Arc<DFlash2Model>> {
    let refused = |reason: String| {
        Error::from_reason(format!(
            "Qwen4Exp load: the DFlash2 draft at {} does not serve this model: {reason}",
            dir.display()
        ))
    };
    let shape = draft_target_shape_of(c);
    let config = read_dflash2_config(dir).map_err(|error| refused(error.reason))?;
    config
        .validate_target(shape)
        .map_err(|error| refused(error.reason))?;
    let (model, _) = load_dflash2(dir).map_err(|error| refused(error.reason))?;
    model
        .validate_target(shape)
        .map_err(|error| refused(error.reason))?;
    Ok(Arc::new(model))
}

#[cfg(test)]
pub(in super::super) type ChosenRows = (
    SeqId,
    Option<Vec<f32>>,
    [f64; DRAFT_VERIFY_ROWS - 1],
    usize,
    usize,
);

/// The draft beside the model's sequences: the model and one sequence's
/// context.
pub(in super::super) struct DraftSlot {
    model: Arc<DFlash2Model>,
    context: DFlash2ContextCache,
    /// The sequence the context follows.
    seq: Option<SeqId>,
    /// The open turn whose decode runs the cycle; the taps are on only
    /// while it is set.
    turn: Option<SeqId>,
    /// The drafted turn's record; each drafted turn starts its own.
    record: DraftRecord,
    /// Tests put this confidence in place of the draft's.
    #[cfg(test)]
    pub(in super::super) confidence: Option<Vec<f32>>,
    /// Per rows rule call: (sequence, confidence, kept rates, ceiling, rows).
    #[cfg(test)]
    pub(in super::super) chosen: Vec<ChosenRows>,
    /// Tests put their proposals in place of the draft's, as
    /// [`DraftStepper::forced`].
    #[cfg(test)]
    pub(in super::super) forced: Option<Box<dyn FnMut(usize, &[u32]) -> Vec<u32>>>,
    /// Per turn start: (sequence, the switch's answer).
    #[cfg(test)]
    pub(in super::super) starts: Vec<(SeqId, bool)>,
    /// Per scheduled proposal: (sequence, anchor row, drafted).
    #[cfg(test)]
    pub(in super::super) proposals: Vec<(SeqId, usize, bool)>,
    /// Per proposal the draft made: (sequence, anchor row, proposals
    /// handed to the engine).
    #[cfg(test)]
    pub(in super::super) offered: Vec<(SeqId, usize, usize)>,
    /// Tests put their drawn proposals and rows in place of the draft's on
    /// a sampled turn, as [`DraftStepper::drawn`], from the anchor.
    #[cfg(test)]
    pub(in super::super) drawn: Option<Box<dyn FnMut(u32) -> (Vec<u32>, Vec<SparseDistribution>)>>,
    /// Tests make the proposal at this many proposals made fail.
    #[cfg(test)]
    pub(in super::super) fail_at: Option<usize>,
}

impl super::super::Inner {
    /// Attaches the draft or takes it off. A scheduled turn runs its cycle
    /// only when [`draft_turn`] says so at the turn's start.
    pub(in super::super) fn set_draft(&mut self, draft: Option<Arc<DFlash2Model>>) -> Result<()> {
        let Some(model) = draft else {
            self.draft = None;
            return self.decoder.set_hidden_taps(None);
        };
        model.validate_target(self.decoder.draft_target_shape())?;
        self.draft = Some(DraftSlot {
            context: DFlash2ContextCache::new(&model.config),
            model,
            seq: None,
            turn: None,
            record: DraftRecord::new(),
            #[cfg(test)]
            confidence: None,
            #[cfg(test)]
            chosen: Vec::new(),
            #[cfg(test)]
            forced: None,
            #[cfg(test)]
            starts: Vec::new(),
            #[cfg(test)]
            proposals: Vec::new(),
            #[cfg(test)]
            offered: Vec::new(),
            #[cfg(test)]
            drawn: None,
            #[cfg(test)]
            fail_at: None,
        });
        Ok(())
    }

    /// The bytes the draft holds beside its weights now: its context's ring
    /// and the taps kept for it.
    pub(in super::super) fn draft_bytes_held(&self) -> u64 {
        let Some(slot) = self.draft.as_ref() else {
            return 0;
        };
        let mut arrays = Vec::new();
        slot.context.collect_arrays(&mut arrays);
        arrays
            .iter()
            .map(|array| array.nbytes() as u64)
            .sum::<u64>()
            .saturating_add(self.decoder.hidden_tap_bytes())
    }

    /// The draft's stepper over the active sequence, `rows` rows a verify;
    /// `None` without a draft. Another sequence's context starts again.
    pub(in super::super) fn draft_stepper(
        &mut self,
        rows: usize,
    ) -> Result<Option<DraftStepper<'_>>> {
        let Some(slot) = self.draft.as_mut() else {
            return Ok(None);
        };
        if slot.seq != self.active_seq {
            slot.context.reset();
            slot.seq = self.active_seq;
        }
        let stepper = DraftStepper::new(&mut self.decoder, &slot.model, &mut slot.context, rows)?;
        #[cfg(test)]
        let stepper = match slot.forced.as_mut() {
            Some(forced) => DraftStepper {
                forced: Some(Box::new(move |position, ids: &[u32]| forced(position, ids))),
                ..stepper
            },
            None => stepper,
        };
        #[cfg(test)]
        let stepper = match slot.drawn.as_mut() {
            Some(drawn) => DraftStepper {
                drawn: Some(Box::new(drawn)),
                ..stepper
            },
            None => stepper,
        };
        Ok(Some(stepper))
    }

    /// A scheduled turn on `seq` starts: with [`draft_turn`]'s yes its
    /// decode runs the cycle and every pass from here on is tapped.
    pub(in super::super) fn begin_draft_turn(&mut self, seq: SeqId) -> Result<bool> {
        let open = 1
            + self.open_turns.iter().filter(|&&open| open != seq).count()
            + usize::from(self.exclusive_turn);
        let on = draft_turn(
            self.draft.is_some(),
            self.decoder.weights.plan.direct,
            !self.decoder.weights.direct_fill_pass(),
            open,
        );
        let Some(slot) = self.draft.as_mut() else {
            return Ok(false);
        };
        #[cfg(test)]
        slot.starts.push((seq, on));
        if !on {
            return Ok(false);
        }
        self.decoder.draft_taps_on(&slot.model)?;
        slot.turn = Some(seq);
        slot.record = DraftRecord::new();
        Ok(true)
    }

    /// Another turn opens beside a drafted one: from its next cycle the
    /// drafted turn verifies its anchor alone, and no pass is tapped.
    pub(in super::super) fn draft_peer_opened(&mut self, seq: SeqId) {
        if let Some(slot) = self.draft.as_mut()
            && slot.turn.is_some_and(|turn| turn != seq)
        {
            slot.turn = None;
            self.decoder.draft_taps_off();
        }
    }

    /// The drafted turn on `seq` ends: its last kept rows join the
    /// context, and the taps go off.
    pub(in super::super) fn end_draft_turn(&mut self, seq: SeqId) {
        let active = self.active_seq == Some(seq);
        let Some(slot) = self.draft.as_mut().filter(|slot| slot.turn == Some(seq)) else {
            return;
        };
        slot.turn = None;
        if active
            && slot.seq == Some(seq)
            && self
                .decoder
                .draft_follow(&slot.model, &mut slot.context)
                .is_err()
        {
            slot.context.reset();
        }
        self.decoder.draft_taps_off();
    }

    /// The draft's proposals for `seq`'s next cycle, at most `cap`: none
    /// for a turn that does not run the cycle (it verifies its anchor alone).
    /// A request that leaves the rows to the model (`mtp_adaptive_depth`)
    /// gets [`next_verify_rows`]' count; any other gets `cap`.
    pub(in super::super) fn propose_draft(
        &mut self,
        seq: SeqId,
        anchor: u32,
        cap: usize,
        params: &ChatParams,
        rng: &mut dyn rand::Rng,
    ) -> Result<DsparkProposal> {
        let drafting = self
            .draft
            .as_ref()
            .is_some_and(|slot| slot.turn == Some(seq));
        #[cfg(test)]
        if let Some(slot) = self.draft.as_mut() {
            slot.proposals
                .push((seq, self.decoder.history.len(), drafting));
        }
        let cap = cap.min(DRAFT_VERIFY_ROWS - 1);
        if !drafting || cap == 0 {
            return Ok(DsparkProposal {
                draft_ids: Vec::new(),
                device_draft_ids: None,
                device_draft_sparse: None,
                draft_dists: Vec::new(),
                draft_sparse_dists: Vec::new(),
                keep_probabilities: None,
            });
        }
        #[cfg(test)]
        if let Some(slot) = self.draft.as_mut()
            && slot.fail_at == Some(slot.offered.len())
        {
            return Err(Error::from_reason(
                "qwen4_exp DFlash2 draft: the proposal failed, so the turn fails: a test failure",
            ));
        }
        let rule = params.mtp_adaptive_depth;
        let row = self.decoder.history.len();
        if rule && let Some(slot) = self.draft.as_mut() {
            slot.record.observe(&self.decoder.history, anchor);
        }
        let failed = |error: Error| {
            Error::from_reason(format!(
                "qwen4_exp DFlash2 draft: the proposal failed, so the turn fails: {}",
                error.reason
            ))
        };
        let block = self
            .draft_stepper(cap + 1)
            .and_then(|stepper| {
                stepper
                    .ok_or_else(|| Error::from_reason("Qwen4 draft proposal without a draft"))?
                    .propose_block(anchor, params, rng, rule)
            })
            .map_err(failed)?;
        let slot = self
            .draft
            .as_mut()
            .ok_or_else(|| failed(Error::from_reason("the draft went away")))?;
        #[cfg(test)]
        let block = match slot.confidence.clone() {
            Some(confidence) => Proposed {
                confidence: Some(confidence),
                ..block
            },
            None => block,
        };
        let rows = if rule {
            let rows = next_verify_rows(block.confidence.as_deref(), &slot.record, cap + 1);
            #[cfg(test)]
            slot.chosen.push((
                seq,
                block.confidence.clone(),
                slot.record.kept,
                cap + 1,
                rows,
            ));
            slot.record.offer(row, anchor, &block.ids);
            rows
        } else {
            cap + 1
        };
        let proposal = block.proposal(rows - 1);
        if proposal.draft_ids.is_empty() {
            self.decoder
                .weights
                .live
                .draft_one_row_cycles
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        }
        #[cfg(test)]
        if let Some(slot) = self.draft.as_mut() {
            slot.offered
                .push((seq, self.decoder.history.len(), proposal.draft_ids.len()));
        }
        Ok(proposal)
    }

    /// The draft's model and context, for tests that read them.
    #[cfg(test)]
    pub(in super::super) fn draft_parts(&self) -> Option<(&DFlash2Model, &DFlash2ContextCache)> {
        self.draft
            .as_ref()
            .map(|slot| (slot.model.as_ref(), &slot.context))
    }

    /// The attached draft's slot, for tests that set or read its records.
    #[cfg(test)]
    pub(in super::super) fn draft_slot(&mut self) -> Option<&mut DraftSlot> {
        self.draft.as_mut()
    }
}
