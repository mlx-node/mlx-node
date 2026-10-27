//! The global expert cache directory: `S` cache slots shared by every
//! layer. `dir[layer][expert] -> slot` and `occ[slot] -> (layer, expert)`
//! stay a bijection. A layer step asks [`Cache::plan`] where each wanted
//! expert is served from: a cache hit, a victim slot to read into (decode),
//! or its staging slot. A victim is a free slot, else the policy's pick
//! among slots that hold no expert of the step's wanted set and whose last
//! GPU use is complete. Pure host bookkeeping: the bank owns the bytes.

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
        }
    }

    pub fn lru(slots: usize, n_layers: usize, n_experts: usize) -> Self {
        Self::new(slots, n_layers, n_experts, Box::new(Lru::default()))
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
    /// miss gets a victim when `decode` (else, or with no victim left, its
    /// staging slot). `wanted` holds distinct expert ids.
    pub fn plan(&mut self, layer: usize, wanted: &[u32], decode: bool) -> (u64, Vec<Source>) {
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
        if !decode || self.occ.is_empty() {
            return (epoch, out);
        }
        for src in out.iter_mut().filter(|s| **s == Source::Staging) {
            let Some(v) = self.victim(layer, wanted) else {
                break;
            };
            self.unmap(v);
            self.last_use[v as usize] = epoch;
            *src = Source::Victim(v);
        }
        (epoch, out)
    }

    fn victim(&mut self, layer: usize, wanted: &[u32]) -> Option<u32> {
        let (epoch, completed) = (self.epoch, self.completed);
        let ok = |s: u32| {
            let s = s as usize;
            self.last_use[s] <= completed
                && self.last_use[s] < epoch
                && !matches!(self.occ[s], Some((l, e)) if l as usize == layer && wanted.contains(&e))
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
