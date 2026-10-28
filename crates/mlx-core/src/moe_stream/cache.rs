//! The global expert cache directory: `S` cache slots shared by every
//! layer. `dir[layer][expert] -> slot` and `occ[slot] -> (layer, expert)`
//! stay a bijection. A layer step asks [`Cache::plan`] where each wanted
//! expert is served from: a cache hit, a victim slot to read into, or its
//! staging slot. A decode miss whose staging slot still holds it is served
//! from there, and at the turn's end those copies move into victims
//! ([`Cache::promote_staged`]). A victim is a free slot, else the policy's pick
//! among slots that hold no expert of the step's wanted set and whose last
//! GPU use is complete. Pure host bookkeeping: the bank owns the bytes.

use std::cmp::Reverse;
use std::collections::BTreeSet;

const NONE: u32 = u32::MAX;

/// Layer steps of at most this many rows take the decode path (cache
/// victims); larger ones stage their misses.
pub const DECODE_ROWS: usize = 1;

pub fn decode_step(rows: usize) -> bool {
    rows <= DECODE_ROWS
}

/// Eviction order. `touch` on every use, `forget` when a slot is emptied.
pub trait Policy: Send {
    fn touch(&mut self, slot: u32);
    fn forget(&mut self, slot: u32);
    /// The first slot in eviction order that `ok` accepts.
    fn victim(&self, ok: &dyn Fn(u32) -> bool) -> Option<u32>;
}

/// Least recently used: one tick per use.
#[derive(Default)]
pub struct Lru {
    tick: u64,
    at: Vec<Option<u64>>,
    order: BTreeSet<(u64, u32)>,
}

impl Policy for Lru {
    fn touch(&mut self, slot: u32) {
        let s = slot as usize;
        if self.at.len() <= s {
            self.at.resize(s + 1, None);
        }
        if let Some(t) = self.at[s] {
            self.order.remove(&(t, slot));
        }
        self.tick += 1;
        self.at[s] = Some(self.tick);
        self.order.insert((self.tick, slot));
    }

    fn forget(&mut self, slot: u32) {
        if let Some(t) = self.at.get_mut(slot as usize).and_then(Option::take) {
            self.order.remove(&(t, slot));
        }
    }

    fn victim(&self, ok: &dyn Fn(u32) -> bool) -> Option<u32> {
        self.order.iter().map(|&(_, s)| s).find(|&s| ok(s))
    }
}

/// Least frequently used with exponential decay: each use adds 1 to the
/// slot's score, scores halve every `half_life` uses (all slots' uses
/// together); the victim is the lowest score (lowest slot on ties).
/// Scores are kept inflated by `2^((tick - base) / half_life)`, which keeps
/// their order; `base` moves forward before the factor overflows.
pub struct LfuDecay {
    half_life: f64,
    tick: u64,
    base: u64,
    score: Vec<Option<f64>>,
    order: BTreeSet<(u64, u32)>,
}

/// Inflation exponent at which every score is rescaled (2^512 stays far
/// from f64 overflow after any number of adds).
const LFU_RESCALE: f64 = 512.0;

impl LfuDecay {
    pub fn new(half_life: f64) -> Self {
        assert!(
            half_life.is_finite() && half_life > 0.0,
            "LfuDecay half-life {half_life}"
        );
        Self {
            half_life,
            tick: 0,
            base: 0,
            score: Vec::new(),
            order: BTreeSet::new(),
        }
    }

    /// Scores of positive f64 compare like their bit patterns.
    fn key(v: f64) -> u64 {
        v.to_bits()
    }

    fn rescale(&mut self) {
        let shift = (LFU_RESCALE * self.half_life).floor() as u64;
        self.base += shift;
        let f = (-(shift as f64) / self.half_life).exp2();
        self.order.clear();
        for (s, v) in self.score.iter_mut().enumerate() {
            if let Some(v) = v {
                *v *= f;
                self.order.insert((Self::key(*v), s as u32));
            }
        }
    }
}

impl Policy for LfuDecay {
    fn touch(&mut self, slot: u32) {
        self.tick += 1;
        if (self.tick - self.base) as f64 / self.half_life > LFU_RESCALE {
            self.rescale();
        }
        let s = slot as usize;
        if self.score.len() <= s {
            self.score.resize(s + 1, None);
        }
        let add = ((self.tick - self.base) as f64 / self.half_life).exp2();
        if let Some(v) = self.score[s] {
            self.order.remove(&(Self::key(v), slot));
        }
        let v = self.score[s].unwrap_or(0.0) + add;
        self.score[s] = Some(v);
        self.order.insert((Self::key(v), slot));
    }

    fn forget(&mut self, slot: u32) {
        if let Some(v) = self.score.get_mut(slot as usize).and_then(Option::take) {
            self.order.remove(&(Self::key(v), slot));
        }
    }

    fn victim(&self, ok: &dyn Fn(u32) -> bool) -> Option<u32> {
        self.order.iter().map(|&(_, s)| s).find(|&s| ok(s))
    }
}

/// Where one wanted expert of a layer step is served from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// Cache slot holding it.
    Hit(u32),
    /// Cache slot to read it into; unmapped until [`Cache::commit`].
    Victim(u32),
    /// Its staging slot.
    Staging,
}

pub struct Cache {
    n_experts: usize,
    n_layers: usize,
    dir: Vec<u32>,
    occ: Vec<Option<(u32, u32)>>,
    /// Epoch of each slot's last use.
    last_use: Vec<u64>,
    free: Vec<u32>,
    epoch: u64,
    completed: u64,
    policy: Box<dyn Policy>,
    /// Prefill misses placed in victims this turn, per layer.
    seeded: Vec<usize>,
    /// Per `(layer, expert)`: tick of its last decode use served from
    /// staging this turn (0 = none).
    staged_at: Vec<u64>,
    staged_tick: u64,
}

impl Cache {
    pub fn new(slots: usize, n_layers: usize, n_experts: usize, policy: Box<dyn Policy>) -> Self {
        Self {
            n_experts,
            n_layers,
            dir: vec![NONE; n_layers * n_experts],
            occ: vec![None; slots],
            last_use: vec![0; slots],
            free: (0..slots as u32).rev().collect(),
            epoch: 0,
            completed: 0,
            policy,
            seeded: vec![0; n_layers],
            staged_at: vec![0; n_layers * n_experts],
            staged_tick: 0,
        }
    }

    /// A new turn: every layer's prefill seeding quota is full again.
    pub fn new_turn(&mut self) {
        self.seeded.fill(0);
        self.staged_at.fill(0);
    }

    /// Turn end (all GPU work complete): every `(layer, e)` this turn
    /// served from staging that `valid` says its staging slot still holds and
    /// the cache does not, gets a victim, most recently used last (the most
    /// recent `slots` of them when more are valid). Returns
    /// `(slot, layer, e)` per placement, already mapped: the caller copies
    /// staging slot `e` into cache slot `slot` before any gather.
    pub fn promote_staged(&mut self, valid: &dyn Fn(usize, u32) -> bool) -> Vec<(u32, usize, u32)> {
        let mut used: Vec<(u64, usize)> = self
            .staged_at
            .iter()
            .enumerate()
            .filter(|(_, t)| **t > 0)
            .map(|(k, &t)| (t, k))
            .collect();
        self.staged_at.fill(0);
        let mut out = Vec::new();
        if self.occ.is_empty() || used.is_empty() {
            return out;
        }
        used.sort_unstable();
        let n = self.n_experts;
        used.retain(|&(_, k)| {
            self.lookup(k / n, (k % n) as u32).is_none() && valid(k / n, (k % n) as u32)
        });
        // each placement is a victim of this epoch: at most `slots` fit, so
        // keep the most recently used ones
        let drop = used.len().saturating_sub(self.occ.len());
        self.epoch += 1;
        let epoch = self.epoch;
        for (_, k) in used.drain(drop..) {
            let (layer, e) = (k / n, (k % n) as u32);
            let Some(v) = self.victim(layer, &[]) else {
                break;
            };
            self.unmap(v);
            self.last_use[v as usize] = epoch;
            self.commit(v, layer, e);
            out.push((v, layer, e));
        }
        // the copies are host work done before the next step
        self.completed = self.epoch;
        out
    }

    pub fn lru(slots: usize, n_layers: usize, n_experts: usize) -> Self {
        Self::new(slots, n_layers, n_experts, Box::new(Lru::default()))
    }

    /// [`LfuDecay`] with `half_life` in expert uses.
    pub fn lfu_decay(slots: usize, n_layers: usize, n_experts: usize, half_life: f64) -> Self {
        Self::new(
            slots,
            n_layers,
            n_experts,
            Box::new(LfuDecay::new(half_life)),
        )
    }

    pub fn slots(&self) -> usize {
        self.occ.len()
    }

    pub fn occupied(&self) -> usize {
        self.occ.iter().flatten().count()
    }

    fn key(&self, layer: usize, e: u32) -> usize {
        assert!(
            layer < self.n_layers && (e as usize) < self.n_experts,
            "moe_stream cache: (layer {layer}, expert {e}) outside {} x {}",
            self.n_layers,
            self.n_experts
        );
        layer * self.n_experts + e as usize
    }

    pub fn lookup(&self, layer: usize, e: u32) -> Option<u32> {
        let s = self.dir[self.key(layer, e)];
        (s != NONE).then_some(s)
    }

    pub fn occ(&self, slot: u32) -> Option<(usize, u32)> {
        self.occ
            .get(slot as usize)
            .copied()
            .flatten()
            .map(|(l, e)| (l as usize, e))
    }

    /// GPU work of every epoch up to `epoch` has finished.
    pub fn complete(&mut self, epoch: u64) {
        self.completed = self.completed.max(epoch.min(self.epoch));
    }

    /// Everything submitted so far has finished (after a device sync).
    pub fn complete_all(&mut self) {
        self.completed = self.epoch;
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    fn unmap(&mut self, slot: u32) {
        if let Some((l, e)) = self.occ[slot as usize].take() {
            let k = self.key(l as usize, e);
            self.dir[k] = NONE;
        }
        self.policy.forget(slot);
    }

    /// One layer step: a new epoch; every hit is stamped first, then each
    /// decode miss gets a victim, and a prefill step places up to
    /// `slots / layers` misses per layer and turn in victims (most-used rows
    /// first, then lowest id; a free slot or the same layer's resident, never
    /// another layer's); the rest, or a miss with no victim left, use the
    /// staging slot. `wanted` holds distinct expert ids.
    pub fn plan(&mut self, layer: usize, wanted: &[u32], decode: bool) -> (u64, Vec<Source>) {
        self.plan_step(layer, wanted, &[], decode, &|_| false)
    }

    /// `plan` with the seeding rules: `uses[i]` = rows of the step that use
    /// `wanted[i]` (prefill seeding order; empty = all equal), `staged(e)` =
    /// the staging slot of `e` holds this layer's copy.
    pub fn plan_step(
        &mut self,
        layer: usize,
        wanted: &[u32],
        uses: &[u32],
        decode: bool,
        staged: &dyn Fn(u32) -> bool,
    ) -> (u64, Vec<Source>) {
        debug_assert!(
            wanted.windows(2).all(|w| w[0] < w[1]),
            "moe_stream cache: wanted must be sorted and distinct"
        );
        assert!(
            uses.is_empty() || uses.len() == wanted.len(),
            "moe_stream cache: {} use counts for {} experts",
            uses.len(),
            wanted.len()
        );
        self.epoch += 1;
        let epoch = self.epoch;
        let mut out: Vec<Source> = wanted
            .iter()
            .map(|&e| match self.lookup(layer, e) {
                Some(s) => {
                    self.last_use[s as usize] = epoch;
                    self.policy.touch(s);
                    Source::Hit(s)
                }
                None => Source::Staging,
            })
            .collect();
        if self.occ.is_empty() {
            return (epoch, out);
        }
        let mut miss = Vec::new();
        for i in 0..wanted.len() {
            if out[i] != Source::Staging {
                continue;
            }
            if decode && staged(wanted[i]) {
                let k = self.key(layer, wanted[i]);
                self.staged_tick += 1;
                self.staged_at[k] = self.staged_tick;
            } else {
                miss.push(i);
            }
        }
        let take = if decode {
            miss.len()
        } else {
            miss.sort_by_key(|&i| (Reverse(uses.get(i).copied().unwrap_or(0)), i));
            let quota = self.occ.len() / self.n_layers.max(1);
            quota.saturating_sub(self.seeded[layer]).min(miss.len())
        };
        for &i in &miss[..take] {
            let Some(v) = self.victim_where(layer, wanted, !decode) else {
                break;
            };
            self.unmap(v);
            self.last_use[v as usize] = epoch;
            out[i] = Source::Victim(v);
            if !decode {
                self.seeded[layer] += 1;
            }
        }
        (epoch, out)
    }

    fn victim(&mut self, layer: usize, wanted: &[u32]) -> Option<u32> {
        self.victim_where(layer, wanted, false)
    }

    /// `own_layer`: an occupied victim must hold an expert of `layer` (a
    /// prefill seed never evicts another layer's resident, which this very
    /// prefill pass may still hit).
    fn victim_where(&mut self, layer: usize, wanted: &[u32], own_layer: bool) -> Option<u32> {
        let (epoch, completed) = (self.epoch, self.completed);
        // `wanted` is sorted (callers' contract, asserted in tests)
        let ok = |s: u32| {
            let s = s as usize;
            self.last_use[s] <= completed
                && self.last_use[s] < epoch
                && match self.occ[s] {
                    Some((l, e)) => {
                        l as usize == layer && wanted.binary_search(&e).is_err()
                            || l as usize != layer && !own_layer
                    }
                    None => true,
                }
        };
        if let Some(i) = self.free.iter().rposition(|&s| ok(s)) {
            return Some(self.free.remove(i));
        }
        self.policy.victim(&ok)
    }

    /// The read into victim `slot` succeeded: it now holds `(layer, e)`.
    pub fn commit(&mut self, slot: u32, layer: usize, e: u32) {
        let k = self.key(layer, e);
        assert!(
            self.occ[slot as usize].is_none() && self.dir[k] == NONE,
            "moe_stream cache: commit of (layer {layer}, expert {e}) into slot {slot} that is not a free victim"
        );
        self.occ[slot as usize] = Some((layer as u32, e));
        self.dir[k] = slot;
        self.policy.touch(slot);
    }

    /// The read into victim `slot` failed: the slot is free again.
    pub fn release(&mut self, slot: u32) {
        self.unmap(slot);
        if !self.free.contains(&slot) {
            self.free.push(slot);
        }
    }

    /// Every slot's contents are gone (bank `Empty`, or a new bank).
    pub fn clear(&mut self) {
        for s in 0..self.occ.len() as u32 {
            self.unmap(s);
        }
        self.free = (0..self.occ.len() as u32).rev().collect();
    }

    /// `dir` and `occ` are inverse maps; free slots hold nothing and are
    /// listed once.
    pub fn check(&self) -> Result<(), String> {
        for (s, o) in self.occ.iter().enumerate() {
            if let Some((l, e)) = *o
                && self.dir[self.key(l as usize, e)] != s as u32
            {
                return Err(format!("occ[{s}] = ({l}, {e}) but dir says otherwise"));
            }
        }
        for (k, &s) in self.dir.iter().enumerate() {
            let want = Some(((k / self.n_experts) as u32, (k % self.n_experts) as u32));
            if s != NONE && self.occ.get(s as usize).copied().flatten() != want {
                return Err(format!("dir[{k}] = {s} but occ disagrees"));
            }
        }
        let mut seen = BTreeSet::new();
        for &s in &self.free {
            if !seen.insert(s) || self.occ[s as usize].is_some() {
                return Err(format!("free slot {s} listed twice or occupied"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
