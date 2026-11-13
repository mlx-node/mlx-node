//! Warm prefix reuse: the cell a sequence keeps outside the KV pool (GDN
//! convolution history, GDN recurrent matrices, PLE history, and on a load
//! with the MTP head the draft history) stored at rows of a prompt's lineage,
//! beside the prefix cache's KV blocks (Splash d8af1054 `StateCache`: a
//! replay point before the generation prompt and rolling checkpoints every
//! 4,096 tokens, docs/internals.md).
//!
//! A pass reads its own rows' K/V at BF16 and every earlier row from the
//! pool, so a prefill's bits depend on where its passes are cut. Every pass
//! end is a pure function of the prompt ([`pass_ends`]): the prefill window
//! from the previous end, cut at the block before every assistant message
//! opening, so a cold prefill of a prompt makes the passes every earlier turn
//! of its conversation made below its replay row. Each KV block's
//! prefix-cache key also binds the tokens and ends of its whole pass (see
//! [`pass_keys`]): two runs share a block only when both ran that same pass
//! over the same tokens, and a key match below a state's row proves the row
//! is an end of the new prompt.

use std::collections::HashMap;

use crate::array::MxArray;
use crate::models::qwen3_5::dflash2_state::plan::planned_checkpoints;
use crate::models::qwen3_5::dflash2_state::store::{
    EntryId, HybridStateStore, Lineage, StateBytes, StoreHit,
};
use crate::transformer::paged_kv_cache_adapter::{PagedKVCacheAdapter, SeqId};

/// Splash `kPrefillCheckpointTokens`.
pub(super) const CHECKPOINT_INTERVAL: usize = 4096;

/// Rows of one KV page: a state sits on a whole block.
pub(super) const BLOCK_ROWS: usize = 32;

/// One layer's part of a cell. A full-attention layer keeps only its count
/// of pooled indexer blocks: at a block row no indexer key waits for a
/// block, and the pooled keys live in the KV pages.
pub(super) struct LayerCell {
    pub(super) conv: Option<MxArray>,
    pub(super) recurrent: Option<MxArray>,
    pub(super) ple_conv: Option<MxArray>,
    pub(super) index_blocks: usize,
}

/// Draft rows `[kW, (k + 1)W)` of one draft pass: F32 K and V
/// (`[1, heads, W, dim]`) and the pooled indexer keys (`[W / ratio, dim]`).
/// Immutable, in storage of their own: the states of one lineage share them.
#[derive(Clone)]
pub(super) struct Segment {
    pub(super) keys: MxArray,
    pub(super) values: MxArray,
    pub(super) index: MxArray,
}

/// The MTP owner at a state's row `D`: the draft passes of its first
/// `c = floor((D - 1) / W) * W` rows, the target hidden rows waiting for the
/// next draft pass (`[1, D - 1 - c, width]`, their tokens are the lineage's
/// rows `c + 1 ..= D - 1`), and the seed hidden row of row `D - 1`.
#[derive(Clone)]
pub(super) struct DraftCell {
    pub(super) segments: Vec<Segment>,
    pub(super) waiting: Option<MxArray>,
    pub(super) seed: MxArray,
}

impl DraftCell {
    pub(super) fn arrays(&self) -> impl Iterator<Item = &MxArray> {
        self.segments
            .iter()
            .flat_map(|segment| [&segment.keys, &segment.values, &segment.index])
            .chain(self.waiting.iter())
            .chain(std::iter::once(&self.seed))
    }
}

/// A sequence's state outside the KV pool at one row, with the prefill
/// window of the load that produced it. Its arrays own their storage; a
/// load with the MTP head always carries the draft part.
pub(super) struct Cell {
    pub(super) grid: usize,
    pub(super) layers: Vec<LayerCell>,
    pub(super) draft: Option<DraftCell>,
}

impl Cell {
    pub(super) fn arrays(&self) -> impl Iterator<Item = &MxArray> {
        self.layers
            .iter()
            .flat_map(|layer| {
                layer
                    .conv
                    .iter()
                    .chain(layer.recurrent.iter())
                    .chain(layer.ple_conv.iter())
            })
            .chain(self.draft.iter().flat_map(DraftCell::arrays))
    }
}

impl StateBytes for Cell {
    fn state_bytes(&self) -> u64 {
        self.arrays().map(|array| array.nbytes() as u64).sum()
    }
}

/// A row where the turn publishes a state; a checkpoint is disposable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Point {
    pub(super) row: usize,
    pub(super) checkpoint: bool,
}

/// The block rows before every assistant message opening (`open`, the
/// token ids of `<|im_start|>assistant`) past the first block, ascending and
/// without repeats. Whether a row is one depends only on the tokens up to
/// the end of its block's opening.
pub(super) fn boundaries(tokens: &[u32], open: &[u32]) -> Vec<usize> {
    let Some(&first) = open.first() else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    for (at, &token) in tokens.iter().enumerate().skip(1) {
        if token != first || !tokens[at..].starts_with(open) {
            continue;
        }
        let row = at - at % BLOCK_ROWS;
        if row > 0 && rows.last() != Some(&row) {
            rows.push(row);
        }
    }
    rows
}

/// Every pass end of a `len`-token prompt: from each end the next is the
/// first of the window `rows_at(end)` admits there, the next boundary, and
/// `len`. A cold prefill and every warm resume of the prompt cut there.
pub(super) fn pass_ends(
    len: usize,
    boundaries: &[usize],
    rows_at: impl Fn(usize) -> usize,
) -> Vec<usize> {
    let mut ends = Vec::new();
    let mut next = 0;
    let mut at = 0;
    while at < len {
        while boundaries.get(next).is_some_and(|&row| row <= at) {
            next += 1;
        }
        let mut end = (at + rows_at(at).max(1)).min(len);
        if let Some(&row) = boundaries.get(next) {
            end = end.min(row);
        }
        ends.push(end);
        at = end;
    }
    ends
}

/// The per-block extra keys of a prompt with pass ends `ends` (prefill
/// window `grid`): block `j` carries one key of the tokens of the passes from
/// the last end at or before its first row to the first end at or after its
/// last, of those ends and of the ends between them. A row's bits can depend
/// on the later rows of its pass (kernel shapes, expert batches), so a block
/// is only the same as another run's when those passes were. Blocks of the
/// final pass get no key: it ends at the prompt's length, which a longer
/// prompt does not cut at, and a short key list stops the pool's lookup
/// there.
pub(super) fn pass_keys(tokens: &[u32], ends: &[usize], grid: usize) -> Vec<Vec<u64>> {
    let Some(&last) = ends.last() else {
        return Vec::new();
    };
    let mut keys = Vec::new();
    let mut before = 0;
    let mut after = 0;
    let mut memo: Option<(usize, usize, u64)> = None;
    for block in 0.. {
        let (low, high) = (block * BLOCK_ROWS, (block + 1) * BLOCK_ROWS);
        while ends.get(before).is_some_and(|&end| end <= low) {
            before += 1;
        }
        while ends.get(after).is_some_and(|&end| end < high) {
            after += 1;
        }
        let Some(&end) = ends.get(after) else {
            break;
        };
        if end == last {
            break;
        }
        let start = before.checked_sub(1).map_or(0, |index| ends[index]);
        let key = match memo {
            Some((s, e, key)) if s == start && e == end => key,
            _ => {
                let mut extra = vec![grid as u64, start as u64, end as u64];
                extra.extend(
                    ends[before..after]
                        .iter()
                        .filter(|&&row| row > start && row < end)
                        .map(|&row| row as u64),
                );
                let key = mlx_paged_attn::hash_tokens(&tokens[start..end], 0, &extra);
                memo = Some((start, end, key));
                key
            }
        };
        keys.push(vec![key]);
    }
    keys
}

/// The replay point: the block row before the last assistant opening (the
/// generation prompt, Splash `Engine::promptReplayBoundary`), else the last
/// block-aligned pass end before the prompt's last token.
pub(super) fn replay_row(boundaries: &[usize], ends: &[usize], len: usize) -> usize {
    if let Some(&row) = boundaries.last() {
        return row;
    }
    let body = len.saturating_sub(1);
    ends.iter()
        .rev()
        .copied()
        .find(|&end| end <= body && end.is_multiple_of(BLOCK_ROWS))
        .unwrap_or(0)
}

/// The turn's points in `(resume, replay]`: rolling checkpoints every
/// `interval` rows (Splash `plannedCheckpoints`), each moved to the first
/// block-aligned pass end at or after it that leaves one window before the
/// replay point, then the replay point. A window that is not whole KV blocks
/// plans nothing.
pub(super) fn plan_points(
    resume: usize,
    replay: usize,
    ends: &[usize],
    grid: usize,
    interval: usize,
) -> Vec<Point> {
    if grid == 0 || !grid.is_multiple_of(BLOCK_ROWS) || replay <= resume {
        return Vec::new();
    }
    let interval = interval.max(grid).next_multiple_of(grid);
    let (Ok(resume32), Ok(replay32), Ok(interval32), Ok(grid32)) = (
        u32::try_from(resume),
        u32::try_from(replay),
        u32::try_from(interval),
        u32::try_from(grid),
    ) else {
        return Vec::new();
    };
    let mut points: Vec<Point> = Vec::new();
    for row in planned_checkpoints(resume32, replay32, interval32, grid32) {
        let Some(end) = ends.iter().copied().find(|&end| {
            end >= row as usize && end.is_multiple_of(BLOCK_ROWS) && end + grid <= replay
        }) else {
            continue;
        };
        if points.last().map(|point| point.row) != Some(end) {
            points.push(Point {
                row: end,
                checkpoint: true,
            });
        }
    }
    points.push(Point {
        row: replay,
        checkpoint: false,
    });
    points
}

/// A text prompt's passes: the block rows before its assistant openings
/// ([`boundaries`]), every pass end ([`pass_ends`]) and, when it may reuse,
/// its blocks' keys ([`pass_keys`]).
pub(super) struct PassPlan {
    pub(super) cuts: Vec<usize>,
    pub(super) ends: Vec<usize>,
    pub(super) keys: Vec<Vec<u64>>,
}

/// What the store published, refused and dropped since load.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Counters {
    pub(super) restored: u64,
    pub(super) published: u64,
    pub(super) retired: u64,
    /// The room left none even after the oldest states went.
    pub(super) refused_room: u64,
    /// The host's free memory or MLX's active ceiling had no room.
    pub(super) refused_live: u64,
    /// The blocks below the point did not register.
    pub(super) refused_register: u64,
    /// A pass a cold prefill does not run came before the point.
    pub(super) refused_off_grid: u64,
    /// A load with the MTP head had no draft history of the cold run there.
    pub(super) refused_draft: u64,
    pub(super) dropped_kv: u64,
    pub(super) dropped_memory: u64,
}

/// The running turn's pass ends and points (Splash `Request::stateBoundaries`,
/// `stateBoundaryCursor`, `latestCheckpoint`).
pub(super) struct Turn {
    pub(super) salt: u64,
    pub(super) prompt_len: usize,
    /// The row the turn resumed at (0 when cold).
    pub(super) resume: usize,
    /// Every pass end of the prompt ([`pass_ends`]), ascending, the last at
    /// `prompt_len`.
    pub(super) ends: Vec<usize>,
    /// The prompt up to its last point.
    pub(super) lineage: Vec<u32>,
    /// The chain hashes of `lineage`'s whole blocks, computed once.
    pub(super) chain: Vec<u64>,
    /// [`pass_keys`] of `lineage`'s whole blocks.
    pub(super) keys: Vec<Vec<u64>>,
    pub(super) points: Vec<Point>,
    pub(super) cursor: usize,
    /// Every pass so far is the pass a cold prefill runs there.
    pub(super) canonical: bool,
    /// Blocks whose states this turn resumed from or published as its
    /// replay point, in use until it ends.
    pub(super) uses: Vec<u64>,
    /// The turn's rolling checkpoint.
    pub(super) checkpoint: Option<EntryId>,
    /// The turn's replay state and its row, which its end parks for the
    /// disk.
    pub(super) replay: Option<(EntryId, usize)>,
}

impl Turn {
    fn is_end(&self, row: usize) -> bool {
        self.ends.binary_search(&row).is_ok()
    }

    /// The first pass end past `row`.
    pub(super) fn next_end(&self, row: usize) -> Option<usize> {
        let at = self.ends.partition_point(|&end| end <= row);
        self.ends.get(at).copied()
    }

    /// Whether a pass over `[start, end)` is one a cold prefill runs.
    pub(super) fn cold_pass(&self, start: usize, end: usize) -> bool {
        (start == self.resume || start == 0 || self.is_end(start)) && self.is_end(end)
    }

    /// The passes over `[from, to)`, grouped into the runs a layer-major
    /// group may hold: consecutive cold passes of exactly `grid` rows, the
    /// last of a run possibly one of at least 256 rows, never past a group
    /// cut ([`Turn::group_cuts`]). Every other pass is a run of its own.
    pub(super) fn runs(&self, from: usize, to: usize, grid: usize) -> Vec<Vec<(usize, usize)>> {
        let cuts = self.group_cuts(from, to);
        let mut runs: Vec<Vec<(usize, usize)>> = Vec::new();
        let mut open = false;
        let mut at = from;
        while at < to {
            let end = self.next_end(at).map_or(to, |end| end.min(to));
            let rows = end - at;
            let fits = self.cold_pass(at, end) && (rows == grid || rows >= 256);
            match runs.last_mut() {
                Some(run) if open && fits => run.push((at, end)),
                _ => runs.push(vec![(at, end)]),
            }
            open = fits && rows == grid && !cuts.contains(&end);
            at = end;
        }
        runs
    }

    /// The point a pass `[start, end)` reached, past the points inside it;
    /// `cut_ok` tells whether the pass is the one a cold prefill runs there.
    pub(super) fn reach(&mut self, end: usize, cut_ok: bool) -> Option<Point> {
        self.canonical &= cut_ok;
        while self
            .points
            .get(self.cursor)
            .is_some_and(|point| point.row < end)
        {
            self.cursor += 1;
        }
        let point = self
            .points
            .get(self.cursor)
            .copied()
            .filter(|point| point.row == end)?;
        self.cursor += 1;
        Some(point)
    }

    /// The deepest checkpoint and the replay point inside `(start, end)`:
    /// the rows a layer-major group over `[start, end)` ends at, so that a
    /// group holds no captured cell of its own.
    pub(super) fn group_cuts(&self, start: usize, end: usize) -> Vec<usize> {
        let inside = |point: &&Point| point.row > start && point.row < end;
        let mut cuts: Vec<usize> = self
            .points
            .iter()
            .filter(inside)
            .filter(|point| point.checkpoint)
            .map(|point| point.row)
            .max()
            .into_iter()
            .chain(
                self.points
                    .iter()
                    .filter(inside)
                    .filter(|point| !point.checkpoint)
                    .map(|point| point.row),
            )
            .collect();
        cuts.sort_unstable();
        cuts.dedup();
        cuts
    }
}

/// One loaded model's stored states and the running turns' points. Model
/// thread only.
pub(super) struct States {
    pub(super) store: HybridStateStore<Cell>,
    pub(super) turns: HashMap<SeqId, Turn>,
    /// The token ids that open an assistant message (`<|im_start|>`
    /// `assistant`), which cut the passes and place the replay point; empty
    /// when the tokenizer has no single `<|im_start|>` token.
    pub(super) open: Vec<u32>,
    pub(super) interval: usize,
    pub(super) counters: Counters,
    #[cfg(test)]
    pub(super) room_override: Option<u64>,
}

impl States {
    pub(super) fn new(open: Vec<u32>) -> Self {
        Self {
            store: HybridStateStore::new(BLOCK_ROWS as u32),
            turns: HashMap::new(),
            open,
            interval: CHECKPOINT_INTERVAL,
            counters: Counters::default(),
            #[cfg(test)]
            room_override: None,
        }
    }

    pub(super) fn bytes(&self) -> u64 {
        self.store.bytes().state_bytes
    }

    /// Drops every state and turn (a command reset, a GPU reset).
    pub(super) fn clear(&mut self) {
        self.store.clear();
        self.turns.clear();
    }

    /// Drops every state with a block of its chain no longer cached, or made
    /// with another prefill window.
    pub(super) fn prune(&mut self, adapter: &PagedKVCacheAdapter, grid: usize) {
        let gone: Vec<EntryId> = self
            .store
            .entries()
            .filter(|entry| {
                entry.state().grid != grid
                    || !entry
                        .chain()
                        .iter()
                        .all(|&hash| adapter.prefix_cached(hash))
            })
            .map(|entry| entry.id())
            .collect();
        for id in gone {
            if self.store.remove(id) {
                self.counters.dropped_kv += 1;
            }
        }
    }

    /// The deepest state whose lineage is a strict prefix of `tokens`, whose
    /// blocks carry `keys` ([`pass_keys`]).
    pub(super) fn lookup(&self, tokens: &[u32], keys: &[Vec<u64>], salt: u64) -> Option<StoreHit> {
        self.store.lookup(Lineage {
            tokens,
            extra_keys: keys,
            salt,
        })
    }

    /// Ends `seq`'s turn: the states it used become the newest of their
    /// class (Splash `StateCache::unuse`). What it published stays.
    pub(super) fn end_turn(&mut self, seq: SeqId) {
        if let Some(turn) = self.turns.remove(&seq) {
            for block in turn.uses {
                self.store.end_use(block);
            }
        }
    }

    /// The state memory gives up first: a checkpoint (oldest publication),
    /// else the least recently used ordinary state; never one in use, nor
    /// one of `skip`.
    fn victim(&self, skip: &[EntryId]) -> Option<EntryId> {
        let free = || {
            self.store
                .entries()
                .filter(|entry| !self.store.in_use(entry.mark()) && !skip.contains(&entry.id()))
        };
        free()
            .filter(|entry| entry.checkpoint())
            .min_by_key(|entry| entry.publication())
            .or_else(|| free().min_by_key(|entry| entry.last_used()))
            .map(|entry| entry.id())
    }

    /// The states to drop so that `need` more bytes fit in `room`, and the
    /// bytes they hold; `None` when the states in use alone leave too
    /// little. Drops nothing.
    pub(super) fn room_for(&self, room: u64, need: u64) -> Option<(Vec<EntryId>, u64)> {
        let mut victims = Vec::new();
        let mut freed = 0u64;
        while self.bytes().saturating_sub(freed).saturating_add(need) > room {
            let id = self.victim(&victims)?;
            freed = freed.saturating_add(self.store.get(id).map_or(0, |entry| entry.state_bytes()));
            victims.push(id);
        }
        Some((victims, freed))
    }

    /// Drops states toward `shortfall` bytes; the bytes freed.
    pub(super) fn reclaim(&mut self, shortfall: u64) -> u64 {
        let before = self.bytes();
        while before.saturating_sub(self.bytes()) < shortfall {
            let Some(id) = self.victim(&[]) else {
                break;
            };
            self.store.remove(id);
            self.counters.dropped_memory += 1;
        }
        before.saturating_sub(self.bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(points: &[Point]) -> Vec<(usize, bool)> {
        points
            .iter()
            .map(|point| (point.row, point.checkpoint))
            .collect()
    }

    const OPEN: [u32; 2] = [9, 8];

    fn grid(window: usize) -> impl Fn(usize) -> usize {
        move |_| window
    }

    /// Pass ends follow the prompt: the plain grid without an opening; with
    /// openings at rows 40, 70 and 200 on a 64-row window the grid restarts
    /// at each block row before them; two openings in one block are one
    /// end; an opening in the first block is none; a window the scratch
    /// halves past a row halves the passes there.
    /// Mutation: the grid is not restarted at a boundary → red.
    #[test]
    fn pass_ends_follow_the_prompt() {
        let mut tokens = vec![1u32; 300];
        assert_eq!(boundaries(&tokens, &OPEN), Vec::<usize>::new());
        assert_eq!(
            pass_ends(300, &[], grid(64)),
            [64, 128, 192, 256, 300],
            "no opening: the plain grid"
        );
        for at in [40, 70, 200] {
            tokens[at..at + 2].copy_from_slice(&OPEN);
        }
        let cuts = boundaries(&tokens, &OPEN);
        assert_eq!(cuts, [32, 64, 192]);
        assert_eq!(
            pass_ends(300, &cuts, grid(64)),
            [32, 64, 128, 192, 256, 300]
        );
        assert_eq!(
            pass_ends(300, &[160], grid(64)),
            [64, 128, 160, 224, 288, 300],
            "the grid restarts at a boundary"
        );
        tokens[50..52].copy_from_slice(&OPEN);
        assert_eq!(
            boundaries(&tokens, &OPEN),
            [32, 64, 192],
            "openings at 40 and 50 share one block"
        );
        tokens[5..7].copy_from_slice(&OPEN);
        assert_eq!(
            boundaries(&tokens, &OPEN),
            [32, 64, 192],
            "an opening in the first block cuts nothing"
        );
        tokens[71] = 1;
        assert_eq!(
            boundaries(&tokens, &OPEN),
            [32, 192],
            "the role token is part of the opening"
        );
        let halved = |row: usize| if row >= 128 { 16 } else { 64 };
        assert_eq!(
            pass_ends(200, &[192], halved),
            [64, 128, 144, 160, 176, 192, 200]
        );
        assert!(pass_ends(0, &[], grid(64)).is_empty());
    }

    /// The replay point is the block row before the last opening; without an
    /// opening it is the last block-aligned end before the last token.
    #[test]
    fn the_replay_point_is_the_block_before_the_generation_prompt() {
        let ends = pass_ends(1030, &[], grid(1024));
        assert_eq!(replay_row(&[], &ends, 1030), 1024);
        assert_eq!(replay_row(&[], &ends, 1024), 0);
        let mut tokens = vec![1u32; 1030];
        tokens[1026..1028].copy_from_slice(&OPEN);
        let cuts = boundaries(&tokens, &OPEN);
        assert_eq!(cuts, [1024]);
        let ends = pass_ends(1030, &cuts, grid(1024));
        assert_eq!(replay_row(&cuts, &ends, 1030), 1024);
        tokens[1026] = 1;
        tokens[700..702].copy_from_slice(&OPEN);
        let cuts = boundaries(&tokens, &OPEN);
        let ends = pass_ends(1030, &cuts, grid(1024));
        assert_eq!(ends, [672, 1030]);
        assert_eq!(replay_row(&cuts, &ends, 1030), 672);
        let halved = pass_ends(100, &[], grid(16));
        assert_eq!(replay_row(&[], &halved, 100), 96);
        assert_eq!(replay_row(&[], &pass_ends(100, &[], grid(48)), 100), 96);
    }

    /// Points keep to the pass ends: checkpoints every interval with a
    /// one-window margin before the replay point, then the replay point; a
    /// resumed turn plans only past its resume row; a window that is not
    /// whole blocks or a replay point at the resume row plans nothing; a
    /// checkpoint moves to the first block-aligned end at or after its row.
    #[test]
    fn the_state_plan_keeps_to_the_pass_ends() {
        let plain = |len: usize, window: usize| pass_ends(len, &[], grid(window));
        assert_eq!(
            rows(&plan_points(0, 61_440, &plain(62_000, 1024), 1024, 4096)),
            [
                (4096, true),
                (8192, true),
                (12_288, true),
                (16_384, true),
                (20_480, true),
                (24_576, true),
                (28_672, true),
                (32_768, true),
                (36_864, true),
                (40_960, true),
                (45_056, true),
                (49_152, true),
                (53_248, true),
                (57_344, true),
                (61_440, false)
            ]
        );
        let ends = plain(6000, 1024);
        assert_eq!(
            rows(&plan_points(0, 4096, &ends, 1024, 4096)),
            [(4096, false)]
        );
        assert_eq!(
            rows(&plan_points(0, 5120, &ends, 1024, 4096)),
            [(4096, true), (5120, false)]
        );
        assert_eq!(
            rows(&plan_points(
                57_344,
                61_440,
                &plain(62_000, 1024),
                1024,
                4096
            )),
            [(61_440, false)]
        );
        let ends = plain(100, 32);
        assert_eq!(
            rows(&plan_points(0, 96, &ends, 32, 32)),
            [(32, true), (64, true), (96, false)]
        );
        assert!(plan_points(0, 96, &plain(100, 48), 48, 48).is_empty());
        assert!(plan_points(96, 96, &ends, 32, 32).is_empty());
        assert!(plan_points(0, 0, &ends, 1024, 4096).is_empty());
        let moved = pass_ends(9000, &[2976, 4192], grid(1024));
        assert_eq!(
            rows(&plan_points(0, 7264, &moved, 1024, 4096)),
            [(4192, true), (7264, false)],
            "the checkpoint at 4,096 moves to the next end"
        );
    }

    /// A block's key binds its pass: blocks of one pass share a key, the
    /// final pass has none, and a prompt cut elsewhere keys a block
    /// differently even over the same tokens.
    /// Mutation: the key leaves out the ends inside its passes → red.
    #[test]
    fn pass_keys_bind_the_pass() {
        let tokens: Vec<u32> = (0..250).map(|i| i % 50).collect();
        let ends = pass_ends(250, &[], grid(64));
        let keys = pass_keys(&tokens, &ends, 64);
        assert_eq!(
            keys.len(),
            6,
            "the block of the pass ending at 250 has no key"
        );
        assert_eq!(keys[0], keys[1]);
        assert_ne!(keys[1], keys[2]);
        let cut = pass_ends(250, &[32], grid(64));
        assert_eq!(cut, [32, 96, 160, 224, 250]);
        let other = pass_keys(&tokens, &cut, 64);
        assert_ne!(other[0], keys[0]);
        let short = pass_ends(250, &[], grid(16));
        let short_keys = pass_keys(&tokens, &short, 16);
        assert_eq!(short_keys.len(), 7);
        assert_ne!(short_keys[0], short_keys[1]);
        assert_ne!(
            pass_keys(&tokens, &[32, 64, 250], 32)[0],
            pass_keys(&tokens, &[16, 32, 64, 250], 32)[0],
            "a block over two passes is not the block of one"
        );
    }

    /// A pass that runs past points skips them; a cut off the ends ends the
    /// turn's publications; a group stops at the deepest checkpoint and the
    /// replay point inside it; runs hold whole windows only.
    #[test]
    fn a_turn_reaches_its_points_in_order() {
        let ends = pass_ends(100, &[], grid(32));
        let mut turn = Turn {
            salt: 0,
            prompt_len: 100,
            resume: 0,
            ends: ends.clone(),
            lineage: Vec::new(),
            chain: Vec::new(),
            keys: Vec::new(),
            points: plan_points(0, 96, &ends, 32, 32),
            cursor: 0,
            canonical: true,
            uses: Vec::new(),
            checkpoint: None,
            replay: None,
        };
        assert_eq!(turn.group_cuts(0, 100), [64, 96]);
        assert_eq!(turn.group_cuts(32, 96), [64]);
        assert_eq!(
            turn.runs(0, 100, 32),
            [vec![(0, 32), (32, 64)], vec![(64, 96)], vec![(96, 100)]]
        );
        assert_eq!(turn.next_end(40), Some(64));
        assert_eq!(turn.next_end(100), None);
        assert!(turn.cold_pass(32, 64) && !turn.cold_pass(32, 48));
        assert_eq!(
            turn.reach(64, true),
            Some(Point {
                row: 64,
                checkpoint: true
            })
        );
        assert!(turn.canonical);
        assert_eq!(turn.reach(80, false), None);
        assert!(!turn.canonical);
        assert_eq!(
            turn.reach(96, true),
            Some(Point {
                row: 96,
                checkpoint: false
            })
        );
        assert!(!turn.canonical);
    }
}
