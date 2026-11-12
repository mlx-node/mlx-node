//! Per-turn prefix accounting for DFlash2: a ledger of the last turns' block
//! hash chains, the reason a turn resumed shallower than the deepest state the
//! design keeps, and the ideal resume point the workload gates compare against.

use std::collections::VecDeque;
use std::fmt;

use crate::engine::backend::ResetScope;
use crate::models::qwen3_5::gdn_checkpoint_store::compute_paged_prefix_block_hashes;
use crate::models::qwen3_5::model::{TokenPrefixMismatchTrace, token_prefix_mismatch_trace};

pub(crate) const LEDGER_CAPACITY: usize = 32;

/// The draft window the offline designs plan with (Splash
/// `draftContextTokens` of the 27B draft, the minimum junction gain).
const DRAFT_WINDOW: u32 = 2048;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HitKind {
    Live,
    End,
    Capture,
    Miss,
}

impl HitKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            HitKind::Live => "live",
            HitKind::End => "E",
            HitKind::Capture => "C",
            HitKind::Miss => "none",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Hit {
    pub(crate) kind: HitKind,
    pub(crate) tokens: usize,
}

impl Hit {
    pub(crate) const MISS: Hit = Hit {
        kind: HitKind::Miss,
        tokens: 0,
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MissReason {
    /// No earlier turn shares a whole block with the prompt.
    ColdFirst,
    /// The deepest lineage is not the live one: another conversation ran in
    /// between, or the prompt only shares a header with the live history.
    CrossSession,
    /// The live lineage diverges at or after its replay point and before its
    /// end (a re-render of the previous generation).
    Rerender,
    /// The live history extends the prompt but the flat lane is not
    /// authoritative; carries the site that dirtied it when one was recorded.
    FlatDirty(Option<&'static str>),
    /// The live history extends the prompt but the retained draft context or
    /// the flat frontier does not match it.
    ContextMismatch,
    /// The prompt is too short to hold a whole block before its last token.
    BelowBlock,
    ReuseDisabled,
    Invalidated(&'static str),
    Evicted,
    /// An exact state deeper than the one used was available.
    StateExistedUnused,
    Unknown,
}

impl fmt::Display for MissReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MissReason::ColdFirst => f.write_str("cold_first"),
            MissReason::CrossSession => f.write_str("cross_session"),
            MissReason::Rerender => f.write_str("rerender"),
            MissReason::FlatDirty(None) => f.write_str("flat_dirty"),
            MissReason::FlatDirty(Some(site)) => write!(f, "flat_dirty:{site}"),
            MissReason::ContextMismatch => f.write_str("context_mismatch"),
            MissReason::BelowBlock => f.write_str("below_block"),
            MissReason::ReuseDisabled => f.write_str("reuse_disabled"),
            MissReason::Invalidated(site) => write!(f, "invalidated:{site}"),
            MissReason::Evicted => f.write_str("evicted"),
            MissReason::StateExistedUnused => f.write_str("state_existed_unused"),
            MissReason::Unknown => f.write_str("unknown"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DropKind {
    /// The site discarded the draft context and the flat lane's authority.
    Invalidated,
    /// The site ran a paged turn over the flat mirror.
    Dirtied,
}

/// The site that dropped the live state since the last DFlash2 turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LiveDrop {
    pub(crate) site: &'static str,
    pub(crate) kind: DropKind,
}

impl LiveDrop {
    fn reason(self) -> MissReason {
        match self.kind {
            DropKind::Invalidated => MissReason::Invalidated(self.site),
            DropKind::Dirtied => MissReason::FlatDirty(Some(self.site)),
        }
    }
}

/// The model's live flat state as the turn found it.
#[derive(Clone, Copy)]
pub(crate) struct LiveView<'a> {
    /// The cached token history the live state holds.
    pub(crate) tokens: &'a [u32],
    pub(crate) authoritative: bool,
    /// The retained draft context and the flat frontier both equal `tokens`.
    pub(crate) context_matches: bool,
}

/// What the state store holds for this prompt.
#[derive(Clone, Copy, Default)]
pub(crate) struct StoreView {
    /// The deepest exact entry `b < |P|` with `P[..b]` equal to its tokens.
    pub(crate) deepest: Option<Hit>,
    /// The deepest evicted entry that would have matched the prompt.
    pub(crate) evicted_depth: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct TurnObservation<'a> {
    pub(crate) prompt: &'a [u32],
    pub(crate) reuse_cache: bool,
    /// The state the turn actually resumed from.
    pub(crate) used: Hit,
    /// `None` when the model holds no live state.
    pub(crate) live: Option<LiveView<'a>>,
    pub(crate) dropped: Option<LiveDrop>,
    pub(crate) store: StoreView,
}

#[derive(Clone, Copy)]
pub(crate) struct TurnClassification {
    pub(crate) hit: Hit,
    /// `None` when the turn resumed at least as deep as the ideal.
    pub(crate) miss_reason: Option<MissReason>,
    /// Block-floored best prefix shared with any ledger turn, below `|P|`.
    pub(crate) shadow_hit: usize,
    /// The deepest state the design keeps for this prompt (`TurnLedger::ideal_resume`).
    pub(crate) ideal: usize,
    /// First prompt/live-history divergence; all `-1` without a live state.
    pub(crate) diverge: TokenPrefixMismatchTrace,
}

/// One finished turn as the ledger records it.
#[derive(Clone, Copy)]
pub(crate) struct LedgerTurn<'a> {
    /// Tokens the turn's end state covers: the prompt plus the generated rows.
    pub(crate) history: &'a [u32],
    pub(crate) prompt_len: usize,
    pub(crate) replay_point: Option<usize>,
    pub(crate) is_dflash2: bool,
    /// Block-aligned depths at which the turn captured a state.
    pub(crate) captures: &'a [usize],
}

struct LedgerEntry {
    block_hashes: Vec<u64>,
    /// Hash of the sub-block tail `history[floor_bs(E)..E]`, chained on the
    /// last block hash, so "the prompt extends this history" is exact.
    end_hash: u64,
    prompt_len: usize,
    replay_point: Option<usize>,
    end: usize,
    is_dflash2: bool,
    captures: Vec<usize>,
}

#[derive(Clone, Copy)]
struct EntryMatch {
    /// Block-floored common prefix with the prompt.
    lcp: usize,
    /// The prompt starts with the entry's whole history.
    extends: bool,
}

pub(crate) struct TurnLedger {
    block_size: usize,
    entries: VecDeque<LedgerEntry>,
}

impl TurnLedger {
    pub(crate) fn new(block_size: u32) -> Self {
        Self {
            block_size: (block_size as usize).max(1),
            entries: VecDeque::with_capacity(LEDGER_CAPACITY),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn block_size(&self) -> usize {
        self.block_size
    }

    fn floor(&self, len: usize) -> usize {
        len - len % self.block_size
    }

    /// The deepest block boundary a resume may use: strictly below `len`.
    fn resumable(&self, len: usize) -> usize {
        self.floor(len.saturating_sub(1))
    }

    fn chain(&self, tokens: &[u32]) -> Vec<u64> {
        let blocks = tokens.len() / self.block_size;
        if blocks == 0 {
            return Vec::new();
        }
        let extra_keys = vec![Vec::new(); blocks];
        compute_paged_prefix_block_hashes(
            tokens,
            (blocks * self.block_size) as u32,
            self.block_size as u32,
            &extra_keys,
            0,
        )
        .unwrap_or_default()
    }

    pub(crate) fn record(&mut self, turn: LedgerTurn<'_>) {
        let block_hashes = self.chain(turn.history);
        let tail = self.floor(turn.history.len());
        let end_hash = mlx_paged_attn::hash_tokens(
            &turn.history[tail..],
            block_hashes.last().copied().unwrap_or(0),
            &[],
        );
        if self.entries.len() == LEDGER_CAPACITY {
            self.entries.pop_front();
        }
        self.entries.push_back(LedgerEntry {
            block_hashes,
            end_hash,
            prompt_len: turn.prompt_len,
            replay_point: turn.replay_point,
            end: turn.history.len(),
            is_dflash2: turn.is_dflash2,
            captures: turn.captures.to_vec(),
        });
    }

    fn matches(&self, prompt: &[u32]) -> Vec<EntryMatch> {
        let chain = self.chain(prompt);
        self.entries
            .iter()
            .map(|entry| {
                let blocks = entry
                    .block_hashes
                    .iter()
                    .zip(&chain)
                    .take_while(|(a, b)| a == b)
                    .count();
                let lcp = blocks * self.block_size;
                let extends = blocks == entry.block_hashes.len()
                    && entry.end <= prompt.len()
                    && mlx_paged_attn::hash_tokens(
                        &prompt[lcp..entry.end],
                        entry.block_hashes.last().copied().unwrap_or(0),
                        &[],
                    ) == entry.end_hash;
                EntryMatch { lcp, extends }
            })
            .collect()
    }

    /// The deepest state this entry's turn left for a prompt of `len` tokens:
    /// its end state when the prompt extends its whole history (DFlash2 turns
    /// only), or a capture inside the shared prefix.
    fn reach(entry: &LedgerEntry, m: EntryMatch, len: usize) -> usize {
        let mut best = 0;
        if entry.is_dflash2 && m.extends && entry.end < len {
            best = entry.end;
        }
        for &c in &entry.captures {
            if c <= m.lcp && c < len {
                best = best.max(c);
            }
        }
        best
    }

    fn shadow_from(&self, prompt: &[u32], matches: &[EntryMatch]) -> usize {
        let cap = self.resumable(prompt.len());
        matches.iter().map(|m| m.lcp.min(cap)).max().unwrap_or(0)
    }

    fn ideal_from(&self, prompt: &[u32], matches: &[EntryMatch]) -> usize {
        self.entries
            .iter()
            .zip(matches)
            .map(|(entry, &m)| Self::reach(entry, m, prompt.len()))
            .max()
            .unwrap_or(0)
    }

    /// Block-floored best prefix the prompt shares with any ledger turn.
    pub(crate) fn shadow_hit(&self, prompt: &[u32]) -> usize {
        self.shadow_from(prompt, &self.matches(prompt))
    }

    /// The deepest state the design keeps for `prompt`, given the ledger: an
    /// earlier DFlash2 turn's end state the prompt extends, or a recorded
    /// capture inside a shared prefix.
    pub(crate) fn ideal_resume(&self, prompt: &[u32]) -> usize {
        self.ideal_from(prompt, &self.matches(prompt))
    }

    pub(crate) fn classify(&self, obs: &TurnObservation<'_>) -> TurnClassification {
        let matches = self.matches(obs.prompt);
        let shadow_hit = self.shadow_from(obs.prompt, &matches);
        let ideal = self.ideal_from(obs.prompt, &matches);
        let diverge = obs
            .live
            .map(|live| token_prefix_mismatch_trace(obs.prompt, live.tokens))
            .unwrap_or_default();
        TurnClassification {
            hit: obs.used,
            miss_reason: self.miss_reason(obs, &matches, shadow_hit, ideal),
            shadow_hit,
            ideal,
            diverge,
        }
    }

    fn miss_reason(
        &self,
        obs: &TurnObservation<'_>,
        matches: &[EntryMatch],
        shadow_hit: usize,
        ideal: usize,
    ) -> Option<MissReason> {
        if !obs.reuse_cache {
            return Some(MissReason::ReuseDisabled);
        }
        let n = obs.prompt.len();
        let used = obs.used.tokens;
        let live_extends = obs.live.filter(|live| {
            !live.tokens.is_empty() && live.tokens.len() < n && obs.prompt.starts_with(live.tokens)
        });
        let live_usable = live_extends
            .filter(|live| live.authoritative && live.context_matches)
            .map_or(0, |live| live.tokens.len());
        let store_best = obs.store.deepest.map_or(0, |hit| hit.tokens);
        if live_usable.max(store_best) > used {
            return Some(MissReason::StateExistedUnused);
        }
        if used >= ideal {
            if used > 0 {
                return None;
            }
            if self.resumable(n) == 0 {
                return Some(MissReason::BelowBlock);
            }
            if shadow_hit == 0 {
                return Some(MissReason::ColdFirst);
            }
        }

        if let Some(live) = live_extends
            && used < live.tokens.len()
        {
            return Some(if let Some(dropped) = obs.dropped {
                dropped.reason()
            } else if !live.authoritative {
                MissReason::FlatDirty(None)
            } else if !live.context_matches {
                MissReason::ContextMismatch
            } else {
                MissReason::Unknown
            });
        }

        // The DFlash2 lineage that agrees with the prompt the longest; ties go
        // to the most recent turn.
        let depth = |m: EntryMatch, entry: &LedgerEntry| if m.extends { entry.end } else { m.lcp };
        let mut best: Option<usize> = None;
        for (idx, (entry, &m)) in self.entries.iter().zip(matches).enumerate() {
            if !entry.is_dflash2 {
                continue;
            }
            if best.is_none_or(|b| depth(m, entry) >= depth(matches[b], &self.entries[b])) {
                best = Some(idx);
            }
        }
        let latest = self.entries.iter().rposition(|entry| entry.is_dflash2);
        let Some(x) = best else {
            return Some(if shadow_hit > 0 {
                MissReason::CrossSession
            } else {
                MissReason::Unknown
            });
        };
        let (entry, m) = (&self.entries[x], matches[x]);

        if Some(x) != latest {
            return Some(if obs.store.evicted_depth > used {
                MissReason::Evicted
            } else {
                MissReason::CrossSession
            });
        }
        if !m.extends {
            // Without a probed replay point the generated region still starts
            // at the prompt end.
            let replay = entry
                .replay_point
                .unwrap_or_else(|| self.floor(entry.prompt_len));
            return Some(if replay <= m.lcp {
                MissReason::Rerender
            } else if m.lcp > 0 {
                MissReason::CrossSession
            } else {
                MissReason::Unknown
            });
        }
        Some(match obs.dropped {
            Some(dropped) => dropped.reason(),
            None => MissReason::Unknown,
        })
    }
}

/// Per-model prefix accounting: the turn ledger plus the site that dropped
/// the live state since the last DFlash2 turn.
pub(crate) struct PrefixTracker {
    pub(crate) ledger: TurnLedger,
    dropped: Option<LiveDrop>,
}

impl PrefixTracker {
    pub(crate) fn new(block_size: u32) -> Self {
        Self {
            ledger: TurnLedger::new(block_size),
            dropped: None,
        }
    }

    /// Keeps the first site: a later one only finds the state already gone.
    pub(crate) fn note_drop(&mut self, site: &'static str, kind: DropKind) {
        self.dropped.get_or_insert(LiveDrop { site, kind });
    }

    pub(crate) fn take_drop(&mut self) -> Option<LiveDrop> {
        self.dropped.take()
    }
}

pub(crate) fn reset_drop_site(scope: ResetScope) -> &'static str {
    match scope {
        ResetScope::PrefixMiss => "reset_caches_prefix_miss",
        ResetScope::Command => "reset_caches_command",
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StateDesign {
    /// End states plus the replay-point capture C(R).
    Flat,
    /// Flat plus a junction capture at the paged KV hit.
    Paged,
}

/// The block-aligned depths a turn of a `prompt_len`-token prompt captures
/// under `design` when it resumes at `resume` with a KV prefix hit of
/// `kv_hit` tokens. The paged design plans the ordinary points only (Splash
/// `configureDraftStatePlan` without its rolling checkpoints); a prompt
/// without a known generation prompt replays from its last whole block
/// before its last token (Splash `promptReplayBoundary`, a one-token tail).
pub(crate) fn design_captures(
    design: StateDesign,
    block_size: u32,
    replay_point: Option<usize>,
    prompt_len: usize,
    resume: usize,
    kv_hit: usize,
) -> Vec<usize> {
    let block_size = (block_size as usize).max(1);
    let mut captures = Vec::new();
    if let Some(r) = replay_point
        && r > resume
    {
        captures.push(r);
    }
    if design == StateDesign::Paged {
        let to_u32 = |tokens: usize| u32::try_from(tokens).unwrap_or(u32::MAX);
        let tail_replay = prompt_len.saturating_sub(1) / block_size * block_size;
        let replay = to_u32(replay_point.unwrap_or(tail_replay));
        let plan = super::plan::configure_draft_state_plan(&super::plan::PlanInputs {
            resume: to_u32(resume),
            junction: to_u32(kv_hit - kv_hit % block_size),
            prompt_replay: replay,
            replay_state: replay,
            window: DRAFT_WINDOW,
            interval: 0,
            chunk: 0,
        });
        return plan
            .boundaries
            .iter()
            .filter(|point| !point.disposable)
            .map(|point| point.tokens as usize)
            .collect();
    }
    captures
}

#[derive(Clone, Copy)]
pub(crate) struct ReplayTurn<'a> {
    pub(crate) prompt: &'a [u32],
    /// The prompt plus the rows the turn generated.
    pub(crate) history: &'a [u32],
    pub(crate) replay_point: Option<usize>,
    pub(crate) is_dflash2: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct IdealTurn {
    pub(crate) resume: usize,
    pub(crate) prefill: usize,
    pub(crate) captures: Vec<usize>,
}

/// Replays `turns` in order through a fresh ledger: each turn resumes at the
/// deepest state the design kept so far and records what it captures.
pub(crate) fn ideal_prefill(
    design: StateDesign,
    block_size: u32,
    turns: &[ReplayTurn<'_>],
) -> Vec<IdealTurn> {
    let mut ledger = TurnLedger::new(block_size);
    turns
        .iter()
        .map(|turn| {
            let resume = ledger.ideal_resume(turn.prompt);
            let kv_hit = match design {
                StateDesign::Flat => resume,
                StateDesign::Paged => resume.max(ledger.shadow_hit(turn.prompt)),
            };
            let captures = design_captures(
                design,
                block_size,
                turn.replay_point,
                turn.prompt.len(),
                resume,
                kv_hit,
            );
            ledger.record(LedgerTurn {
                history: turn.history,
                prompt_len: turn.prompt.len(),
                replay_point: turn.replay_point,
                is_dflash2: turn.is_dflash2,
                captures: &captures,
            });
            IdealTurn {
                resume,
                prefill: turn.prompt.len() - resume,
                captures,
            }
        })
        .collect()
}

/// Collects the `dflash2_prefix` events emitted on this thread.
#[cfg(test)]
pub(crate) mod capture {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use tracing::field::{Field, Visit};
    use tracing_subscriber::layer::{Context, Layer};
    use tracing_subscriber::prelude::*;

    pub(crate) type PrefixEvent = HashMap<String, String>;

    #[derive(Clone, Default)]
    struct Collector(Arc<Mutex<Vec<PrefixEvent>>>);

    struct Fields(PrefixEvent);

    impl Visit for Fields {
        fn record_str(&mut self, field: &Field, value: &str) {
            self.0.insert(field.name().to_string(), value.to_string());
        }

        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            self.0
                .insert(field.name().to_string(), format!("{value:?}"));
        }
    }

    impl<S: tracing::Subscriber> Layer<S> for Collector {
        fn on_event(&self, event: &tracing::Event<'_>, _: Context<'_, S>) {
            if event.metadata().target() != "mlx_core::inference" {
                return;
            }
            let mut fields = Fields(HashMap::new());
            event.record(&mut fields);
            if fields.0.get("event").map(String::as_str) == Some("dflash2_prefix") {
                self.0.lock().unwrap().push(fields.0);
            }
        }
    }

    pub(crate) fn capture_prefix_events<R>(f: impl FnOnce() -> R) -> (R, Vec<PrefixEvent>) {
        let collector = Collector::default();
        let out = tracing::subscriber::with_default(
            tracing_subscriber::registry().with(collector.clone()),
            f,
        );
        let events = std::mem::take(&mut *collector.0.lock().unwrap());
        (out, events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BS: u32 = 16;
    const SUFFIX: [u32; 3] = [900, 901, 902];

    fn span(start: u32, len: usize) -> Vec<u32> {
        (start..start + len as u32).collect()
    }

    fn cat(parts: &[&[u32]]) -> Vec<u32> {
        parts.concat()
    }

    fn replay_point(prompt: &[u32]) -> Option<usize> {
        crate::engine::gen_prompt::replay_point(prompt, &SUFFIX, BS as usize)
    }

    /// A prompt that ends with the generation suffix.
    fn prompt(body: &[u32]) -> Vec<u32> {
        cat(&[body, &SUFFIX])
    }

    fn record(ledger: &mut TurnLedger, prompt: &[u32], history: &[u32], captures: &[usize]) {
        ledger.record(LedgerTurn {
            history,
            prompt_len: prompt.len(),
            replay_point: replay_point(prompt),
            is_dflash2: true,
            captures,
        });
    }

    fn live(tokens: &[u32]) -> LiveView<'_> {
        LiveView {
            tokens,
            authoritative: true,
            context_matches: true,
        }
    }

    fn obs<'a>(prompt: &'a [u32], live: Option<LiveView<'a>>) -> TurnObservation<'a> {
        TurnObservation {
            prompt,
            reuse_cache: true,
            used: Hit::MISS,
            live,
            dropped: None,
            store: StoreView::default(),
        }
    }

    /// One finished turn of session A: a 200-token body, the suffix, then
    /// 40 generated rows.
    struct A1 {
        prompt: Vec<u32>,
        history: Vec<u32>,
    }

    fn a1() -> A1 {
        let prompt = prompt(&span(10_000, 200));
        let history = cat(&[&prompt, &span(11_000, 40)]);
        A1 { prompt, history }
    }

    /// A follow-up prompt that extends `history` with a new user message.
    fn follow_up(history: &[u32], start: u32) -> Vec<u32> {
        prompt(&cat(&[history, &span(start, 30)]))
    }

    fn reason(ledger: &TurnLedger, o: &TurnObservation<'_>) -> Option<MissReason> {
        ledger.classify(o).miss_reason
    }

    #[test]
    fn cold_first_without_a_shared_block() {
        let ledger = TurnLedger::new(BS);
        let a = a1();
        let c = ledger.classify(&obs(&a.prompt, None));
        assert_eq!(c.miss_reason, Some(MissReason::ColdFirst));
        assert_eq!((c.shadow_hit, c.ideal), (0, 0));
        assert_eq!(c.diverge.index, -1);
        assert_eq!(MissReason::ColdFirst.to_string(), "cold_first");
    }

    #[test]
    fn below_block_when_the_prompt_holds_no_whole_block() {
        let ledger = TurnLedger::new(BS);
        let short = prompt(&span(10_000, 13));
        assert_eq!(short.len(), 16);
        assert_eq!(
            reason(&ledger, &obs(&short, None)),
            Some(MissReason::BelowBlock)
        );
        assert_eq!(MissReason::BelowBlock.to_string(), "below_block");
    }

    #[test]
    fn reuse_disabled_wins_over_every_other_rule() {
        let mut ledger = TurnLedger::new(BS);
        let a = a1();
        record(&mut ledger, &a.prompt, &a.history, &[]);
        let next = follow_up(&a.history, 12_000);
        let mut o = obs(&next, Some(live(&a.history)));
        o.reuse_cache = false;
        assert_eq!(reason(&ledger, &o), Some(MissReason::ReuseDisabled));
        assert_eq!(MissReason::ReuseDisabled.to_string(), "reuse_disabled");
    }

    #[test]
    fn live_hit_has_no_miss_reason() {
        let mut ledger = TurnLedger::new(BS);
        let a = a1();
        record(&mut ledger, &a.prompt, &a.history, &[]);
        let next = follow_up(&a.history, 12_000);
        let mut o = obs(&next, Some(live(&a.history)));
        o.used = Hit {
            kind: HitKind::Live,
            tokens: a.history.len(),
        };
        let c = ledger.classify(&o);
        assert_eq!(c.miss_reason, None);
        assert_eq!(c.ideal, a.history.len());
        assert_eq!(c.hit.kind.as_str(), "live");
        assert_eq!(c.diverge.index, a.history.len() as i64);
        assert_eq!(c.diverge.cached_token, -1);
    }

    #[test]
    fn flat_dirty_when_the_live_history_extends_but_is_not_authoritative() {
        let mut ledger = TurnLedger::new(BS);
        let a = a1();
        record(&mut ledger, &a.prompt, &a.history, &[]);
        let next = follow_up(&a.history, 12_000);
        let mut view = live(&a.history);
        view.authoritative = false;
        assert_eq!(
            reason(&ledger, &obs(&next, Some(view))),
            Some(MissReason::FlatDirty(None))
        );
        assert_eq!(MissReason::FlatDirty(None).to_string(), "flat_dirty");

        // A paged turn in between keeps the history, drops the draft context
        // and dirties the flat mirror.
        view.context_matches = false;
        let mut o = obs(&next, Some(view));
        o.dropped = Some(LiveDrop {
            site: "paged_turn",
            kind: DropKind::Dirtied,
        });
        let r = reason(&ledger, &o);
        assert_eq!(r, Some(MissReason::FlatDirty(Some("paged_turn"))));
        assert_eq!(r.unwrap().to_string(), "flat_dirty:paged_turn");
    }

    #[test]
    fn context_mismatch_when_the_draft_context_disagrees() {
        let mut ledger = TurnLedger::new(BS);
        let a = a1();
        record(&mut ledger, &a.prompt, &a.history, &[]);
        let next = follow_up(&a.history, 12_000);
        let mut view = live(&a.history);
        view.context_matches = false;
        assert_eq!(
            reason(&ledger, &obs(&next, Some(view))),
            Some(MissReason::ContextMismatch)
        );
        assert_eq!(MissReason::ContextMismatch.to_string(), "context_mismatch");
    }

    #[test]
    fn invalidated_names_the_site_that_dropped_the_live_state() {
        let mut ledger = TurnLedger::new(BS);
        let a = a1();
        record(&mut ledger, &a.prompt, &a.history, &[]);
        let next = follow_up(&a.history, 12_000);
        let owner_flip = Some(LiveDrop {
            site: "set_cache_owner_id",
            kind: DropKind::Invalidated,
        });
        let mut o = obs(&next, None);
        o.dropped = owner_flip;
        let r = reason(&ledger, &o);
        assert_eq!(r, Some(MissReason::Invalidated("set_cache_owner_id")));
        assert_eq!(r.unwrap().to_string(), "invalidated:set_cache_owner_id");

        // The MoE owner flip keeps the token history but drops the draft
        // context and the flat lane's authority.
        let mut view = live(&a.history);
        view.authoritative = false;
        view.context_matches = false;
        let mut kept_history = obs(&next, Some(view));
        kept_history.dropped = owner_flip;
        assert_eq!(
            reason(&ledger, &kept_history),
            Some(MissReason::Invalidated("set_cache_owner_id"))
        );
    }

    #[test]
    fn drop_site_names_a_live_state_replaced_by_another_lane() {
        let mut ledger = TurnLedger::new(BS);
        let a = a1();
        record(&mut ledger, &a.prompt, &a.history, &[]);
        // A paged turn replaced the live state with a history the prompt does
        // not extend, while the prompt still extends the DFlash2 turn's end.
        let replaced = follow_up(&a.history, 13_000);
        let next = follow_up(&a.history, 12_000);
        let mut view = live(&replaced);
        view.authoritative = false;
        view.context_matches = false;
        let mut o = obs(&next, Some(view));
        o.dropped = Some(LiveDrop {
            site: "paged_whole_turn",
            kind: DropKind::Dirtied,
        });
        assert_eq!(
            reason(&ledger, &o),
            Some(MissReason::FlatDirty(Some("paged_whole_turn")))
        );
    }

    #[test]
    fn unknown_when_the_live_state_vanished_without_a_site() {
        let mut ledger = TurnLedger::new(BS);
        let a = a1();
        record(&mut ledger, &a.prompt, &a.history, &[]);
        let next = follow_up(&a.history, 12_000);
        assert_eq!(
            reason(&ledger, &obs(&next, None)),
            Some(MissReason::Unknown)
        );
        assert_eq!(MissReason::Unknown.to_string(), "unknown");
    }

    #[test]
    fn evicted_when_an_older_lineage_lost_its_state() {
        let mut ledger = TurnLedger::new(BS);
        let a = a1();
        record(&mut ledger, &a.prompt, &a.history, &[]);
        let b_prompt = prompt(&span(20_000, 300));
        let b_history = cat(&[&b_prompt, &span(21_000, 20)]);
        record(&mut ledger, &b_prompt, &b_history, &[]);
        let next = follow_up(&a.history, 12_000);
        let mut o = obs(&next, Some(live(&b_history)));
        assert_eq!(reason(&ledger, &o), Some(MissReason::CrossSession));
        o.store.evicted_depth = a.history.len() / 16 * 16;
        assert_eq!(reason(&ledger, &o), Some(MissReason::Evicted));
        assert_eq!(MissReason::Evicted.to_string(), "evicted");
    }

    #[test]
    fn state_existed_unused_when_a_deeper_exact_state_was_skipped() {
        let mut ledger = TurnLedger::new(BS);
        let a = a1();
        record(&mut ledger, &a.prompt, &a.history, &[]);
        let next = follow_up(&a.history, 12_000);

        let mut from_store = obs(&next, None);
        from_store.store.deepest = Some(Hit {
            kind: HitKind::End,
            tokens: a.history.len(),
        });
        assert_eq!(
            reason(&ledger, &from_store),
            Some(MissReason::StateExistedUnused)
        );
        from_store.used = Hit {
            kind: HitKind::Capture,
            tokens: replay_point(&a.prompt).unwrap(),
        };
        assert_eq!(
            reason(&ledger, &from_store),
            Some(MissReason::StateExistedUnused)
        );

        let unused_live = obs(&next, Some(live(&a.history)));
        assert_eq!(
            reason(&ledger, &unused_live),
            Some(MissReason::StateExistedUnused)
        );
        assert_eq!(
            MissReason::StateExistedUnused.to_string(),
            "state_existed_unused"
        );
    }

    /// Re-render with reasoning stripped: the previous turn generated
    /// `<think> reasoning </think> answer`; the next render keeps only the
    /// answer, so the prompt diverges right after the old prompt.
    #[test]
    fn rerender_with_stripped_reasoning() {
        let mut ledger = TurnLedger::new(BS);
        let body = span(10_000, 200);
        let a_prompt = prompt(&body);
        let reasoning = span(13_000, 120);
        let answer = span(14_000, 25);
        let a_history = cat(&[&a_prompt, &[800], &reasoning, &[801], &answer]);
        let r = replay_point(&a_prompt).unwrap();
        assert_eq!(r, 192);
        record(&mut ledger, &a_prompt, &a_history, &[r]);

        let rerender = prompt(&cat(&[&a_prompt, &answer, &span(15_000, 30)]));
        let c = ledger.classify(&obs(&rerender, Some(live(&a_history))));
        assert_eq!(c.miss_reason, Some(MissReason::Rerender));
        assert_eq!(c.ideal, r);
        assert_eq!(c.diverge.index, a_prompt.len() as i64);
        assert_eq!(c.diverge.prompt_token, answer[0] as i64);
        assert_eq!(c.diverge.cached_token, 800);
        assert_eq!(MissReason::Rerender.to_string(), "rerender");

        // Resuming at C(R) is the ideal: no miss left to explain.
        let mut resumed = obs(&rerender, Some(live(&a_history)));
        resumed.used = Hit {
            kind: HitKind::Capture,
            tokens: r,
        };
        assert_eq!(reason(&ledger, &resumed), None);
    }

    #[test]
    fn a_shared_header_below_the_replay_point_is_cross_session() {
        let mut ledger = TurnLedger::new(BS);
        let header = span(20_000, 2000);
        let b_prompt = prompt(&cat(&[&header, &span(30_000, 300)]));
        let b_history = cat(&[&b_prompt, &span(31_000, 20)]);
        record(&mut ledger, &b_prompt, &b_history, &[]);
        let c_prompt = prompt(&cat(&[&header, &span(40_000, 300)]));
        let c = ledger.classify(&obs(&c_prompt, Some(live(&b_history))));
        assert_eq!(c.miss_reason, Some(MissReason::CrossSession));
        assert_eq!(c.shadow_hit, 2000);
        assert_eq!(c.ideal, 0);
        assert_eq!(MissReason::CrossSession.to_string(), "cross_session");
    }

    /// One main session A and three subagents B/C/D that share a header (A
    /// shares nothing with them), run the way the flat lane does today: one
    /// live state, whole-history prefix reuse.
    #[test]
    fn main_and_three_subagents_interleaved() {
        let header = span(20_000, 2000);
        let firsts: Vec<Vec<u32>> = vec![
            prompt(&span(10_000, 1500)),
            prompt(&cat(&[&header, &span(30_000, 400)])),
            prompt(&cat(&[&header, &span(40_000, 400)])),
            prompt(&cat(&[&header, &span(50_000, 400)])),
        ];
        let mut histories: Vec<Vec<u32>> = vec![Vec::new(); 4];
        let mut ledger = TurnLedger::new(BS);
        let mut live_history: Vec<u32> = Vec::new();
        let mut seq = 100_000u32;

        let mut run = |prompt: Vec<u32>, ledger: &mut TurnLedger| {
            let live_view = (!live_history.is_empty()).then(|| live(&live_history));
            let used = match live_view {
                Some(view)
                    if view.tokens.len() < prompt.len() && prompt.starts_with(view.tokens) =>
                {
                    Hit {
                        kind: HitKind::Live,
                        tokens: view.tokens.len(),
                    }
                }
                _ => Hit::MISS,
            };
            let mut o = obs(&prompt, live_view);
            o.used = used;
            let c = ledger.classify(&o);
            seq += 1000;
            let history = cat(&[&prompt, &span(seq, 64)]);
            let r = replay_point(&prompt);
            ledger.record(LedgerTurn {
                history: &history,
                prompt_len: prompt.len(),
                replay_point: r,
                is_dflash2: true,
                captures: &design_captures(
                    StateDesign::Flat,
                    BS,
                    r,
                    prompt.len(),
                    used.tokens,
                    used.tokens,
                ),
            });
            live_history = history.clone();
            (c, prompt.len(), history)
        };

        let expect_first = [
            MissReason::ColdFirst,
            MissReason::ColdFirst,
            MissReason::CrossSession,
            MissReason::CrossSession,
        ];
        for (session, first) in firsts.into_iter().enumerate() {
            let (c, _, history) = run(first, &mut ledger);
            assert_eq!(c.miss_reason, Some(expect_first[session]), "{session}");
            histories[session] = history;
        }
        for (session, slot) in histories.iter_mut().enumerate() {
            let next = follow_up(slot, 60_000 + session as u32 * 100);
            let (c, len, history) = run(next, &mut ledger);
            *slot = history;
            assert_eq!(c.miss_reason, Some(MissReason::CrossSession), "{session}");
            assert_eq!(c.hit, Hit::MISS);
            assert_eq!(c.ideal, len - 30 - SUFFIX.len());
            assert!(
                c.shadow_hit * 10 >= len * 9,
                "{session}: {}/{len}",
                c.shadow_hit
            );
        }
        // D again, right after D: the live state serves the whole history.
        let next = follow_up(&histories[3], 70_000);
        let (c, _, _) = run(next, &mut ledger);
        assert_eq!(c.miss_reason, None);
        assert_eq!(c.hit.kind, HitKind::Live);
        assert_eq!(ledger.len(), 9);
    }

    #[test]
    fn ledger_keeps_the_last_32_turns() {
        let mut ledger = TurnLedger::new(BS);
        let a = a1();
        record(&mut ledger, &a.prompt, &a.history, &[]);
        for i in 0..LEDGER_CAPACITY as u32 {
            let p = prompt(&span(200_000 + i * 1000, 100));
            record(&mut ledger, &p, &p, &[]);
        }
        assert_eq!(ledger.len(), LEDGER_CAPACITY);
        let next = follow_up(&a.history, 12_000);
        assert_eq!(ledger.ideal_resume(&next), 0);
        assert_eq!(ledger.shadow_hit(&next), 0);
    }

    #[test]
    fn block_size_32_floors_the_shadow_hit_to_32() {
        let mut ledger = TurnLedger::new(32);
        let a = a1();
        ledger.record(LedgerTurn {
            history: &a.history,
            prompt_len: a.prompt.len(),
            replay_point: None,
            is_dflash2: true,
            captures: &[],
        });
        let next = follow_up(&a.history, 12_000);
        assert_eq!(a.history.len(), 243);
        assert_eq!(ledger.shadow_hit(&next), 224);
        assert_eq!(ledger.ideal_resume(&next), 243);
    }

    #[test]
    fn extends_needs_the_whole_history_and_resumes_stay_below_the_prompt() {
        let mut ledger = TurnLedger::new(BS);
        let a = a1();
        assert_eq!(a.history.len(), 243);
        record(&mut ledger, &a.prompt, &a.history, &[]);

        // Every whole block matches, the sub-block tail does not.
        let mut tail_differs = follow_up(&a.history, 12_000);
        tail_differs[242] += 1;
        assert_eq!(ledger.ideal_resume(&tail_differs), 0);
        assert_eq!(ledger.shadow_hit(&tail_differs), 240);

        // A prompt equal to a block-aligned history resumes one block early.
        let aligned = a.history[..240].to_vec();
        record(&mut ledger, &a.prompt, &aligned, &[]);
        assert_eq!(ledger.ideal_resume(&aligned), 0);
        assert_eq!(ledger.shadow_hit(&aligned), 224);
    }

    /// A1, B1 and C1 (B/C share a 4160-token header), A2 extends A1, D1
    /// shares the header again. Flat keeps only E and C(R); paged also
    /// captures the junction C1 prefilled past.
    #[test]
    fn ideal_prefill_on_a_five_request_trace() {
        let header = span(20_000, 4160);
        let a1_prompt = prompt(&span(10_000, 100));
        let a1_history = cat(&[&a1_prompt, &span(11_000, 20)]);
        let b1_prompt = prompt(&cat(&[&header, &span(30_000, 40)]));
        let b1_history = cat(&[&b1_prompt, &span(31_000, 10)]);
        let c1_prompt = prompt(&cat(&[&header, &span(40_000, 50)]));
        let c1_history = cat(&[&c1_prompt, &span(41_000, 10)]);
        let a2_prompt = prompt(&cat(&[&a1_history, &span(12_000, 30)]));
        let a2_history = cat(&[&a2_prompt, &span(13_000, 10)]);
        let d1_prompt = prompt(&cat(&[&header, &span(50_000, 60)]));
        let d1_history = cat(&[&d1_prompt, &span(51_000, 10)]);
        let turns: Vec<ReplayTurn<'_>> = [
            (&a1_prompt, &a1_history),
            (&b1_prompt, &b1_history),
            (&c1_prompt, &c1_history),
            (&a2_prompt, &a2_history),
            (&d1_prompt, &d1_history),
        ]
        .into_iter()
        .map(|(p, h)| ReplayTurn {
            prompt: p,
            history: h,
            replay_point: replay_point(p),
            is_dflash2: true,
        })
        .collect();
        let lens: Vec<usize> = turns.iter().map(|t| t.prompt.len()).collect();
        assert_eq!(lens, [103, 4203, 4213, 156, 4223]);

        let flat = ideal_prefill(StateDesign::Flat, BS, &turns);
        let flat_resume: Vec<usize> = flat.iter().map(|t| t.resume).collect();
        assert_eq!(flat_resume, [0, 0, 0, 123, 0]);
        assert_eq!(flat[2].captures, [4208]);
        assert_eq!(flat.iter().map(|t| t.prefill).sum::<usize>(), 12_775);

        let paged = ideal_prefill(StateDesign::Paged, BS, &turns);
        let paged_resume: Vec<usize> = paged.iter().map(|t| t.resume).collect();
        assert_eq!(paged_resume, [0, 0, 0, 123, 4160]);
        assert_eq!(paged[2].captures, [4160, 4208]);
        assert_eq!(paged[4].captures, [4208]);
        assert_eq!(paged.iter().map(|t| t.prefill).sum::<usize>(), 8_615);
    }

    /// A paged turn whose prompt has no known generation prompt still
    /// captures its junction: its replay state is the last whole block
    /// before its last token (Splash `promptReplayBoundary` with a
    /// one-token tail), past the junction.
    /// Mutation TJ1 (the replay state is 0 without a replay point) → red
    /// (nothing captured).
    #[test]
    fn a_paged_turn_without_a_replay_point_keeps_its_junction() {
        let captures = design_captures(StateDesign::Paged, BS, None, 5_000, 0, 4_100);
        assert_eq!(captures, [4096, 4992]);
    }
}
