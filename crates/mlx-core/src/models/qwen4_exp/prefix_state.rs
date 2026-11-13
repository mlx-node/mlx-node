//! Warm prefix reuse: the cell a sequence keeps outside the KV pool (GDN
//! convolution history, GDN recurrent matrices, PLE history) stored at rows
//! of a prompt's lineage, beside the prefix cache's KV blocks (Splash
//! d8af1054 `StateCache`: a replay point before the generation prompt and
//! rolling checkpoints every 4,096 tokens, docs/internals.md).
//!
//! A pass reads its own rows' K/V at BF16 and every earlier row from the
//! pool, so a prefill's bits depend on where its passes are cut. A state
//! sits only on a line of the pass grid (a multiple of the prefill slice),
//! where every cold prefill cuts, and a turn resumed from it runs the passes
//! a cold turn runs after it. Each KV block's prefix-cache key also binds the
//! tokens of its whole pass (see [`pass_keys`]): two runs share a block only
//! when both ran that same pass over the same tokens, so a state never reads
//! another run's bits for rows its own run computed differently.

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
/// of pooled indexer blocks: at a grid line no indexer key waits for a
/// block, and the pooled keys live in the KV pages.
pub(super) struct LayerCell {
    pub(super) conv: Option<MxArray>,
    pub(super) recurrent: Option<MxArray>,
    pub(super) ple_conv: Option<MxArray>,
    pub(super) index_blocks: usize,
}

/// A sequence's state outside the KV pool at one row, with the pass grid of
/// the load that produced it. Its arrays own their storage.
pub(super) struct Cell {
    pub(super) grid: usize,
    pub(super) layers: Vec<LayerCell>,
}

impl Cell {
    pub(super) fn arrays(&self) -> impl Iterator<Item = &MxArray> {
        self.layers.iter().flat_map(|layer| {
            layer
                .conv
                .iter()
                .chain(layer.recurrent.iter())
                .chain(layer.ple_conv.iter())
        })
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

/// The replay point: the last grid line at or before the generation prompt
/// (the last `turn_open` token after the first row), else the last grid line
/// before the prompt's last token (Splash `Engine::promptReplayBoundary`,
/// on the pass grid).
pub(super) fn replay_row(tokens: &[u32], turn_open: Option<u32>, grid: usize) -> usize {
    let grid = grid.max(1);
    let body = turn_open
        .and_then(|id| tokens.iter().rposition(|&token| token == id))
        .filter(|&opening| opening > 0)
        .unwrap_or(tokens.len().saturating_sub(1));
    body - body % grid
}

/// The turn's points in `(resume, replay]`: rolling checkpoints every
/// `interval` rows, the last at least one grid line before the replay point
/// (Splash `plannedCheckpoints`), then the replay point. A grid that is not
/// whole KV blocks plans nothing.
pub(super) fn plan_points(
    resume: usize,
    replay: usize,
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
    let mut points: Vec<Point> = planned_checkpoints(resume32, replay32, interval32, grid32)
        .into_iter()
        .map(|row| Point {
            row: row as usize,
            checkpoint: true,
        })
        .collect();
    points.push(Point {
        row: replay,
        checkpoint: false,
    });
    points
}

/// The per-block extra keys of `tokens`' whole grid cells: every block of a
/// cell carries one key of the cell's tokens and the grid. A row's bits can
/// depend on the later rows of its pass (kernel shapes, expert batches), so a
/// block is only the same as another run's when the whole pass was. Blocks of
/// a partial last cell get no key: no state sits on them, and a short key
/// list stops the pool's lookup there.
pub(super) fn pass_keys(tokens: &[u32], grid: usize) -> Vec<Vec<u64>> {
    if grid == 0 || !grid.is_multiple_of(BLOCK_ROWS) {
        return Vec::new();
    }
    tokens
        .chunks_exact(grid)
        .flat_map(|cell| {
            let key = mlx_paged_attn::hash_tokens(cell, 0, &[grid as u64]);
            std::iter::repeat_n(vec![key], grid / BLOCK_ROWS)
        })
        .collect()
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
    /// A pass off the cold turn's grid came before the point.
    pub(super) refused_off_grid: u64,
    pub(super) dropped_kv: u64,
    pub(super) dropped_memory: u64,
}

/// The running turn's points (Splash `Request::stateBoundaries`,
/// `stateBoundaryCursor`, `latestCheckpoint`).
pub(super) struct Turn {
    pub(super) salt: u64,
    pub(super) prompt_len: usize,
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
}

impl Turn {
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
    /// The token that opens a chat turn (`<|im_start|>`), which places the
    /// replay point.
    pub(super) turn_open: Option<u32>,
    pub(super) interval: usize,
    pub(super) counters: Counters,
    #[cfg(test)]
    pub(super) room_override: Option<u64>,
}

impl States {
    pub(super) fn new(turn_open: Option<u32>) -> Self {
        Self {
            store: HybridStateStore::new(BLOCK_ROWS as u32),
            turns: HashMap::new(),
            turn_open,
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
    /// on another pass grid.
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

    /// The replay point is the last grid line at or before the generation
    /// prompt's opening token, not a line inside the generation prompt; a
    /// prompt without that token ends one row before its last token.
    /// Mutation: the replay point ignores the opening token → red.
    #[test]
    fn the_replay_point_is_the_last_grid_line_before_the_generation_prompt() {
        const OPEN: u32 = 9;
        let mut tokens = vec![1u32; 1030];
        tokens[0] = OPEN;
        tokens[1020] = OPEN;
        assert_eq!(replay_row(&tokens, Some(OPEN), 1024), 0);
        assert_eq!(replay_row(&tokens, None, 1024), 1024);
        tokens[1020] = 1;
        tokens[1026] = OPEN;
        assert_eq!(replay_row(&tokens, Some(OPEN), 1024), 1024);
        assert_eq!(replay_row(&tokens, Some(OPEN), 32), 1024);
        assert_eq!(replay_row(&tokens[..1026], Some(OPEN), 32), 1024);
        tokens[1026] = 1;
        assert_eq!(
            replay_row(&tokens, Some(OPEN), 32),
            1024,
            "only an opening token after the first row places the point"
        );
        assert_eq!(replay_row(&[OPEN, 1, 2], Some(OPEN), 32), 0);
    }

    /// Points keep to the grid: checkpoints every interval with a one-line
    /// margin before the replay point, then the replay point; a resumed
    /// turn plans only past its resume row; a grid that is not whole
    /// blocks or a replay point at the resume row plans nothing.
    #[test]
    fn the_state_plan_keeps_to_the_grid() {
        assert_eq!(
            rows(&plan_points(0, 61_440, 1024, 4096)),
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
        assert_eq!(rows(&plan_points(0, 4096, 1024, 4096)), [(4096, false)]);
        assert_eq!(
            rows(&plan_points(0, 5120, 1024, 4096)),
            [(4096, true), (5120, false)]
        );
        assert_eq!(
            rows(&plan_points(57_344, 61_440, 1024, 4096)),
            [(61_440, false)]
        );
        assert_eq!(
            rows(&plan_points(0, 96, 32, 32)),
            [(32, true), (64, true), (96, false)]
        );
        assert!(plan_points(0, 96, 48, 48).is_empty());
        assert!(plan_points(96, 96, 32, 32).is_empty());
        assert!(plan_points(0, 0, 1024, 4096).is_empty());
    }

    /// A pass that runs past points skips them; a cut off the grid ends the
    /// turn's publications; a group stops at the deepest checkpoint and the
    /// replay point inside it.
    #[test]
    fn a_turn_reaches_its_points_in_order() {
        let mut turn = Turn {
            salt: 0,
            prompt_len: 100,
            lineage: Vec::new(),
            chain: Vec::new(),
            keys: Vec::new(),
            points: plan_points(0, 96, 32, 32),
            cursor: 0,
            canonical: true,
            uses: Vec::new(),
            checkpoint: None,
        };
        assert_eq!(turn.group_cuts(0, 100), [64, 96]);
        assert_eq!(turn.group_cuts(32, 96), [64]);
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
