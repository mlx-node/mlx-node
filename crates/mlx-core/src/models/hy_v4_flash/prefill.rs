//! Chunked prefill of one sequence and prefix reuse across turns.
//!
//! ```text
//! verify:  prompt ─► longest common prefix with the sequence (< prompt end)
//!                    ─► deepest of: the sequence cut there │ a ladder entry ─► Plan.reuse
//! prefill: Plan ─► settle (cut │ restore │ empty; a refused point falls to a deeper one)
//!               ─► chunks on the grid ─► snapshots at state rows ─► last-row output
//! ```
//!
//! Chunk ends are absolute: every multiple of `C` past the start, every state
//! row, and the prompt end. State rows: every 8,192 rows below 32,768 and
//! every 32,768 from there, below the replay row; the replay row is the block
//! row before the generation prompt (the last assistant opening). A prompt
//! resumed anywhere cuts where a cold prefill of it cuts.
//!
//! A cancelled chunk leaves the sequence at its start (`Model::step` puts
//! it back): the finished chunks stay and a ladder entry marks the point.

use std::cell::Cell;
use std::ops::Range;

use napi::{Error, Result};

use super::cache::LayerCache;
use super::ladder::{Kind, Ladder, Snapshot};
use super::model::{Logits, Mark, Model, Output, Session, Want};
use super::moe::RowKind;

/// Rows of one KV block: every state row sits on one.
pub const BLOCK_ROWS: usize = 32;
/// Checkpoint spacing below `FAR_FROM`.
pub const NEAR_EVERY: usize = 8192;
/// From this row on, checkpoints are this many rows apart.
pub const FAR_FROM: usize = 32_768;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 prefill: {msg}"))
}

/// A sequence the driver grows, cuts back and snapshots.
pub trait Seq {
    type Snap: Snapshot;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn tokens(&self) -> &[u32];

    /// Most positions it may hold.
    fn cap(&self) -> usize;

    /// Drops the positions past `len`; refused with nothing changed.
    fn truncate(&mut self, len: usize) -> Result<()>;

    fn snap(&self) -> Self::Snap;

    /// Goes back to `s`; refused when its rows were rewritten since.
    fn restore(&mut self, s: &Self::Snap) -> Result<()>;

    /// Whether `restore(s)` would succeed.
    fn can_restore(&self, s: &Self::Snap) -> bool;
}

/// Runs prefill chunks on a `Seq`.
pub trait Runner<S: Seq> {
    type Out;

    /// Whether `seq.truncate(len)` would succeed.
    fn can_truncate(&self, seq: &S, len: usize) -> bool;

    /// Appends `ids` to `seq` as one prefill chunk; on error `seq` is as it
    /// was. `cancel(layer)` is asked before each layer.
    fn run(
        &self,
        seq: &mut S,
        ids: &[u32],
        want: Want,
        cancel: &dyn Fn(usize) -> bool,
    ) -> Result<Self::Out>;
}

impl Snapshot for Mark {
    fn len(&self) -> usize {
        Mark::len(self)
    }
}

impl Seq for Session {
    type Snap = Mark;

    fn len(&self) -> usize {
        Session::len(self)
    }

    fn tokens(&self) -> &[u32] {
        Session::tokens(self)
    }

    fn cap(&self) -> usize {
        Session::cap(self)
    }

    fn truncate(&mut self, len: usize) -> Result<()> {
        Session::truncate(self, len)
    }

    fn snap(&self) -> Mark {
        self.mark()
    }

    fn restore(&mut self, s: &Mark) -> Result<()> {
        Session::restore(self, s)
    }

    fn can_restore(&self, s: &Mark) -> bool {
        Session::can_restore(self, s)
    }
}

impl Runner<Session> for Model {
    type Out = Output;

    fn can_truncate(&self, seq: &Session, len: usize) -> bool {
        session_keeps(seq, len, self.cfg().indexer.kpool)
    }

    fn run(
        &self,
        seq: &mut Session,
        ids: &[u32],
        want: Want,
        cancel: &dyn Fn(usize) -> bool,
    ) -> Result<Output> {
        self.forward(seq, ids, RowKind::Prefill, want, Some(cancel), None)
    }
}

/// `WindowRing::truncate`'s rule: a ring of `now` positions holding `kept`
/// rows of a `window` can go back to `len`.
pub fn window_keeps(now: usize, kept: usize, window: usize, len: usize) -> bool {
    if len >= now {
        return len == now;
    }
    len == 0 || kept.saturating_sub(now - len) >= len.min(window)
}

/// `IndexerState::truncate`'s rule: an indexer of `now` positions holding
/// `held` open `tok_k` rows can go back to `len`.
pub fn index_keeps(now: usize, held: usize, kpool: usize, len: usize) -> bool {
    if len >= now {
        return len == now;
    }
    if kpool == 0 {
        return false;
    }
    let closed = len / kpool * kpool;
    len <= closed || closed >= now.saturating_sub(held)
}

/// Whether `Session::truncate(len)` would succeed.
pub fn session_keeps(sess: &Session, len: usize, kpool: usize) -> bool {
    let now = sess.len();
    if len >= now {
        return len == now;
    }
    sess.layers().iter().all(|st| {
        let window = match &st.cache {
            LayerCache::Window(r) => window_keeps(r.len(), r.kept(), r.window(), len),
            LayerCache::Latent(_) => true,
        };
        window
            && st
                .index
                .as_ref()
                .is_none_or(|ix| index_keeps(ix.len(), ix.held(), kpool, len))
    })
}

/// Whether a prefill takes a checkpoint at `row`.
pub fn is_checkpoint(row: usize) -> bool {
    row > 0
        && if row < FAR_FROM {
            row.is_multiple_of(NEAR_EVERY)
        } else {
            row.is_multiple_of(FAR_FROM)
        }
}

/// Checkpoint rows in `(start, below)`, ascending.
pub fn checkpoints(start: usize, below: usize) -> Vec<usize> {
    let mut rows = Vec::new();
    let mut r = (start / NEAR_EVERY + 1) * NEAR_EVERY;
    while r < below {
        if is_checkpoint(r) {
            rows.push(r);
        }
        r += NEAR_EVERY;
    }
    rows
}

/// The block row at or below `kv_end` (the rows with KV at a turn's end).
pub fn turn_row(kv_end: usize) -> usize {
    kv_end - kv_end % BLOCK_ROWS
}

/// A row where a prefill keeps a snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateRow {
    pub row: usize,
    pub kind: Kind,
}

/// Where a prompt's chunks end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grid {
    chunk: usize,
    /// Token ids of an assistant opening.
    open: Vec<u32>,
}

impl Grid {
    /// `chunk` (C): a positive multiple of `BLOCK_ROWS`.
    pub fn new(chunk: usize, open: Vec<u32>) -> Result<Self> {
        if chunk == 0 || !chunk.is_multiple_of(BLOCK_ROWS) {
            return Err(bad(format!(
                "chunk {chunk}: expected a positive multiple of {BLOCK_ROWS}"
            )));
        }
        Ok(Self { chunk, open })
    }

    pub fn chunk(&self) -> usize {
        self.chunk
    }

    /// The block row before the last assistant opening past position 0,
    /// else the block row before the prompt's last token.
    pub fn replay_row(&self, prompt: &[u32]) -> usize {
        let opening = if self.open.is_empty() {
            None
        } else {
            (1..prompt.len())
                .rev()
                .find(|&at| prompt[at..].starts_with(&self.open))
        };
        let at = opening.unwrap_or_else(|| prompt.len().saturating_sub(1));
        at - at % BLOCK_ROWS
    }

    /// State rows of a prefill of `prompt` from `start`: checkpoints below
    /// the replay row, then the replay row, all past `start`.
    pub fn states(&self, start: usize, prompt: &[u32]) -> Vec<StateRow> {
        let replay = self.replay_row(prompt);
        if replay <= start {
            return Vec::new();
        }
        let mut rows: Vec<StateRow> = checkpoints(start, replay)
            .into_iter()
            .map(|row| StateRow {
                row,
                kind: Kind::Checkpoint,
            })
            .collect();
        rows.push(StateRow {
            row: replay,
            kind: Kind::Prompt,
        });
        rows
    }

    /// Chunk ends of a prefill of `prompt` from `start`, ascending: the
    /// multiples of C past `start`, the state rows, the prompt end.
    pub fn ends(&self, start: usize, prompt: &[u32]) -> Vec<usize> {
        let len = prompt.len();
        if start >= len {
            return Vec::new();
        }
        let mut ends = Vec::new();
        let mut g = (start / self.chunk + 1) * self.chunk;
        while g < len {
            ends.push(g);
            g += self.chunk;
        }
        ends.extend(self.states(start, prompt).iter().map(|s| s.row));
        ends.push(len);
        ends.sort_unstable();
        ends.dedup();
        ends
    }
}

/// Length of the common prefix of `a` and `b`.
pub fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// What `verify` promised for a prompt: its rows below `reuse` are not run
/// again, unless the point is refused at prefill (then a deeper point is
/// used and the rows between are run).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    prompt: Vec<u32>,
    reuse: usize,
}

impl Plan {
    pub fn reuse(&self) -> usize {
        self.reuse
    }

    pub fn prompt(&self) -> &[u32] {
        &self.prompt
    }
}

/// How a prefill ended.
pub enum Prefilled<O> {
    /// The last chunk's output; rows from `from` were run.
    Done { out: O, from: usize },
    /// Cancelled; the sequence holds the chunks before `at`.
    Cancelled { at: usize },
}

/// The full prompt and the row the prefill input starts at: the plan's when
/// `tokens` is its tail, else `tokens` appended to the sequence.
fn resolve(plan: Option<Plan>, held: &[u32], tokens: &[u32]) -> (Vec<u32>, usize) {
    match plan {
        Some(p) if p.reuse + tokens.len() == p.prompt.len() && p.prompt[p.reuse..] == *tokens => {
            (p.prompt, p.reuse)
        }
        _ => {
            let mut prompt = Vec::with_capacity(held.len() + tokens.len());
            prompt.extend_from_slice(held);
            prompt.extend_from_slice(tokens);
            (prompt, held.len())
        }
    }
}

/// Puts `seq` at the deepest point `<= target` that holds `prompt`'s rows:
/// the sequence cut at `target`, else a ladder entry, else empty. Returns
/// that point.
fn settle<S: Seq>(
    seq: &mut S,
    ladder: &mut Ladder<S::Snap>,
    prompt: &[u32],
    target: usize,
) -> Result<usize> {
    let shared = common_prefix(seq.tokens(), prompt);
    if shared >= target && seq.truncate(target).is_ok() {
        ladder.drop_above(target);
        return Ok(target);
    }
    let limit = target.min(shared);
    while let Some(len) = ladder.deepest(limit, seq.len()) {
        let restored = ladder.snap(len).is_some_and(|s| seq.restore(s).is_ok());
        if restored {
            ladder.drop_above(len);
            return Ok(len);
        }
        ladder.remove(len);
    }
    seq.truncate(0)?;
    ladder.clear();
    Ok(0)
}

/// One sequence's prefill over a `Runner`.
pub struct Driver<'a, R> {
    pub runner: &'a R,
    pub grid: &'a Grid,
    /// Every chunk also returns every row's trunk output (the MTP input).
    pub hidden: bool,
}

impl<R> Driver<'_, R> {
    /// The reuse plan for `prompt` (`reuse` false: none).
    pub fn plan<S: Seq>(
        &self,
        seq: &S,
        ladder: &Ladder<S::Snap>,
        prompt: &[u32],
        reuse: bool,
    ) -> Plan
    where
        R: Runner<S>,
    {
        let reuse = if reuse {
            self.reach(seq, ladder, prompt)
        } else {
            0
        };
        Plan {
            prompt: prompt.to_vec(),
            reuse,
        }
    }

    fn reach<S: Seq>(&self, seq: &S, ladder: &Ladder<S::Snap>, prompt: &[u32]) -> usize
    where
        R: Runner<S>,
    {
        let limit = common_prefix(seq.tokens(), prompt).min(prompt.len().saturating_sub(1));
        let direct = if self.runner.can_truncate(seq, limit) {
            limit
        } else {
            0
        };
        let restorable = ladder.deepest_where(limit, seq.len(), |s| seq.can_restore(s));
        direct.max(restorable.unwrap_or(0))
    }

    /// Runs `tokens` (the prompt past `plan`'s reuse point, or past the
    /// sequence when the plan does not fit them) in chunks on the grid.
    /// `on_chunk` sees each finished chunk's rows and output.
    pub fn prefill<S: Seq>(
        &self,
        seq: &mut S,
        ladder: &mut Ladder<S::Snap>,
        plan: Option<Plan>,
        tokens: &[u32],
        cancel: &dyn Fn(usize) -> bool,
        on_chunk: &mut dyn FnMut(Range<usize>, &R::Out) -> Result<()>,
    ) -> Result<Prefilled<R::Out>>
    where
        R: Runner<S>,
    {
        if tokens.is_empty() {
            return Err(bad("nothing to prefill"));
        }
        let (prompt, target) = resolve(plan, seq.tokens(), tokens);
        if prompt.len() > seq.cap() {
            return Err(bad(format!(
                "a prompt of {} tokens exceeds the context of {}",
                prompt.len(),
                seq.cap()
            )));
        }
        let from = settle(seq, ladder, &prompt, target)?;
        self.drive(seq, ladder, &prompt, from, cancel, on_chunk)
    }

    fn drive<S: Seq>(
        &self,
        seq: &mut S,
        ladder: &mut Ladder<S::Snap>,
        prompt: &[u32],
        from: usize,
        cancel: &dyn Fn(usize) -> bool,
        on_chunk: &mut dyn FnMut(Range<usize>, &R::Out) -> Result<()>,
    ) -> Result<Prefilled<R::Out>>
    where
        R: Runner<S>,
    {
        let states = self.grid.states(from, prompt);
        let mut start = from;
        for end in self.grid.ends(from, prompt) {
            let last = end == prompt.len();
            let want = Want {
                logits: if last { Logits::Last } else { Logits::None },
                hidden: self.hidden,
            };
            let fired = Cell::new(false);
            let watch = |layer: usize| {
                let stop = cancel(layer);
                if stop {
                    fired.set(true);
                }
                stop
            };
            match self.runner.run(seq, &prompt[start..end], want, &watch) {
                Ok(out) => {
                    if let Some(s) = states.iter().find(|s| s.row == end) {
                        ladder.put(s.kind, seq.snap());
                    }
                    on_chunk(start..end, &out)?;
                    if last {
                        return Ok(Prefilled::Done { out, from });
                    }
                    start = end;
                }
                Err(_) if fired.get() && seq.len() == start => {
                    if start > 0 && !ladder.contains(start) {
                        ladder.put(Kind::Cancel, seq.snap());
                    }
                    return Ok(Prefilled::Cancelled { at: start });
                }
                Err(e) => return Err(e),
            }
        }
        Err(bad(format!(
            "no chunk ends past {from} for a prompt of {}",
            prompt.len()
        )))
    }
}
