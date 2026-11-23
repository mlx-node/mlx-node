//! Staging + cache + epoch bookkeeping state machine for streamed expert banks.
//!
//! Pure host bookkeeping shared by the MXFP4 bank (`moe_stream::bank`) and the
//! K-quant multi-class bank (`models::hy_v4_flash::bank`); each bank owns its
//! own backing buffers and I/O repacking.

use napi::{Error, Result};

use super::cache::{Cache, Source};

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("moe_stream directory: {msg}"))
}

/// Where one layer step's experts come from.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Plan {
    pub epoch: u64,
    /// Bank slot of each wanted expert.
    pub slots: Vec<u32>,
    /// Experts read into their staging slot.
    pub staging: Vec<u32>,
    /// `(expert, cache slot)` read into a victim.
    pub victims: Vec<(u32, u32)>,
    /// `(expert, cache slot)` copied from its staging slot, which holds it.
    pub copies: Vec<(u32, u32)>,
    pub hits: usize,
    /// A staging slot in `staging` is still used by an unsettled step.
    pub drain: bool,
}

impl Plan {
    pub fn reads(&self) -> usize {
        self.staging.len() + self.victims.len()
    }
}

/// Slot directory: staging tags, per-slot epoch stamps, and the global cache.
/// Pure host bookkeeping; the bank owns the bytes.
pub struct Directory {
    staging: usize,
    n_layers: usize,
    n_experts: usize,
    tags: Vec<Option<(usize, u32)>>,
    /// Per staging slot: epoch of the last step that uses it.
    stage_use: Vec<u64>,
    cache: Cache,
}

impl Directory {
    fn with_cache(staging: usize, n_layers: usize, n_experts: usize, cache: Cache) -> Result<Self> {
        if !(1..=2).contains(&staging) {
            return Err(bad(format!("{staging} staging layers (1 or 2)")));
        }
        let cache_slots = cache.slots();
        if n_experts == 0 || u32::try_from(staging * n_experts + cache_slots).is_err() {
            return Err(bad(format!(
                "{staging} x {n_experts} staging + {cache_slots} cache slots"
            )));
        }
        Ok(Self {
            staging,
            n_layers,
            n_experts,
            tags: vec![None; staging * n_experts],
            stage_use: vec![0; staging * n_experts],
            cache,
        })
    }

    /// `staging` = 1 or 2 staging layers with LRU eviction.
    pub fn new(
        staging: usize,
        n_layers: usize,
        n_experts: usize,
        cache_slots: usize,
    ) -> Result<Self> {
        Self::with_cache(
            staging,
            n_layers,
            n_experts,
            Cache::lru(cache_slots, n_layers, n_experts),
        )
    }

    /// `staging` = 1 or 2 staging layers with LFU-decay eviction.
    pub fn new_with_lfu(
        staging: usize,
        n_layers: usize,
        n_experts: usize,
        cache_slots: usize,
        half_life_uses: f64,
    ) -> Result<Self> {
        if !half_life_uses.is_finite() || half_life_uses <= 0.0 {
            return Err(bad(format!("LFU half-life {half_life_uses}")));
        }
        let cache = Cache::lfu_decay(cache_slots, n_layers, n_experts, half_life_uses);
        Self::with_cache(staging, n_layers, n_experts, cache)
    }

    pub fn has_staged(&self) -> bool {
        self.cache.has_staged()
    }

    pub fn has_unsettled(&self) -> bool {
        self.cache.completed() < self.cache.epoch()
    }

    pub fn staging_layers(&self) -> usize {
        self.staging
    }

    pub fn stage(&self, layer: usize, e: u32) -> usize {
        (layer % self.staging) * self.n_experts + e as usize
    }

    /// Bank index of cache slot 0.
    pub fn base(&self) -> u32 {
        (self.staging * self.n_experts) as u32
    }

    pub fn total_slots(&self) -> usize {
        self.base() as usize + self.cache.slots()
    }

    pub fn cache(&self) -> &Cache {
        &self.cache
    }

    pub fn cache_mut(&mut self) -> &mut Cache {
        &mut self.cache
    }

    pub fn tags(&self) -> &[Option<(usize, u32)>] {
        &self.tags
    }

    pub fn tag_of(&self, layer: usize, e: u32) -> Option<(usize, u32)> {
        self.tags.get(self.stage(layer, e)).copied().flatten()
    }

    pub fn is_staged(&self, layer: usize, e: u32) -> bool {
        self.tags[self.stage(layer, e)] == Some((layer, e))
    }

    pub fn is_resident(&self, layer: usize, e: u32) -> bool {
        self.cache.lookup(layer, e).is_some() || self.is_staged(layer, e)
    }

    pub fn set_tag(&mut self, layer: usize, e: u32) {
        let s = self.stage(layer, e);
        self.tags[s] = Some((layer, e));
    }

    pub fn clear_tag(&mut self, layer: usize, e: u32) {
        let s = self.stage(layer, e);
        self.tags[s] = None;
    }

    /// Highest epoch that used any of `experts`' staging slots for `layer`.
    pub fn max_stage_use(&self, layer: usize, experts: &[u32]) -> u64 {
        experts
            .iter()
            .map(|&e| self.stage_use[self.stage(layer, e)])
            .max()
            .unwrap_or(0)
    }

    /// What `(layer, expert)` bank slot `slot` holds for a query expecting `(layer, e)`.
    pub fn held_at(&self, layer: usize, e: u32, slot: u32) -> Option<(usize, u32)> {
        let base = self.base();
        if slot < base {
            (slot as usize == self.stage(layer, e))
                .then(|| self.tags[slot as usize])
                .flatten()
        } else {
            self.cache.occ(slot - base)
        }
    }

    /// Whether bank slot `slot` holds `(layer, e)`.
    pub fn holds(&self, layer: usize, e: u32, slot: u32) -> bool {
        self.held_at(layer, e, slot) == Some((layer, e))
    }

    /// `experts` sorted and distinct; `uses[i]` = rows using `experts[i]`
    /// (may be empty).
    pub fn check_step(&self, layer: usize, experts: &[u32], uses: &[u32]) -> Result<()> {
        if layer >= self.n_layers {
            return Err(bad(format!("layer {layer} of {}", self.n_layers)));
        }
        if let Some(&e) = experts.iter().find(|&&e| e as usize >= self.n_experts) {
            return Err(bad(format!("expert {e} of {}", self.n_experts)));
        }
        if experts.windows(2).any(|w| w[0] >= w[1]) {
            return Err(bad("experts must be sorted and distinct"));
        }
        if !uses.is_empty() && uses.len() != experts.len() {
            return Err(bad(format!(
                "{} use counts for {} experts",
                uses.len(),
                experts.len()
            )));
        }
        Ok(())
    }

    /// Hits in place; with `decode`, misses into cache victims (a miss whose
    /// staging slot holds it is served from there); a prefill seed whose
    /// staging copy is valid is copied, never read twice; the rest into
    /// staging slots that do not hold them. Every staging slot the step uses
    /// is stamped with its epoch. Call `check_step` first.
    pub fn plan(&mut self, layer: usize, experts: &[u32], uses: &[u32], decode: bool) -> Plan {
        self.plan_with_staging_copy(layer, experts, uses, decode, true)
    }

    /// Like [`Self::plan`], allowing callers to gate staging-to-cache victim
    /// copies (used by `StagingBank` which copies only with two staging layers
    /// on non-decode steps to match `replay_turns_staged`).
    pub fn plan_with_staging_copy(
        &mut self,
        layer: usize,
        experts: &[u32],
        uses: &[u32],
        decode: bool,
        copy_staged: bool,
    ) -> Plan {
        let (n, staging_n) = (self.n_experts, self.staging);
        let stage = |e: u32| (layer % staging_n) * n + e as usize;
        let tags = &self.tags;
        let (epoch, src) = self.cache.plan_step(layer, experts, uses, decode, &|e| {
            tags[stage(e)] == Some((layer, e))
        });
        let base = self.base();
        let mut plan = Plan {
            epoch,
            ..Plan::default()
        };
        for (&e, s) in experts.iter().zip(&src) {
            let valid = self.tags[stage(e)] == Some((layer, e));
            plan.slots.push(match *s {
                Source::Hit(c) => {
                    plan.hits += 1;
                    base + c
                }
                Source::Victim(c) => {
                    if copy_staged && valid {
                        plan.copies.push((e, c));
                    } else {
                        plan.victims.push((e, c));
                    }
                    base + c
                }
                Source::Staging => {
                    let s = stage(e);
                    if !valid {
                        plan.staging.push(e);
                        plan.drain |= self.stage_use[s] > self.cache.completed();
                    }
                    self.stage_use[s] = epoch;
                    s as u32
                }
            });
        }
        plan
    }

    /// Before the reads of `plan`: its staging slots lose their tags.
    pub fn untag(&mut self, layer: usize, plan: &Plan) {
        for &e in &plan.staging {
            let s = self.stage(layer, e);
            self.tags[s] = None;
        }
    }

    /// The copies of `plan` are written.
    pub fn commit_copies(&mut self, layer: usize, plan: &Plan) {
        for &(e, c) in &plan.copies {
            self.cache.commit(c, layer, e);
        }
    }

    /// Releases the cache victims reserved for `plan.copies` (e.g. if a copy failed).
    pub fn abandon_copies(&mut self, plan: &Plan) {
        for &(_, c) in &plan.copies {
            self.cache.release(c);
        }
    }

    /// Every read of `plan` is in.
    pub fn publish(&mut self, layer: usize, plan: &Plan) {
        for &e in &plan.staging {
            let s = self.stage(layer, e);
            self.tags[s] = Some((layer, e));
        }
        for &(e, c) in &plan.victims {
            self.cache.commit(c, layer, e);
        }
    }

    /// A read of `plan` failed or was never queued: its victims are free
    /// again, its staging slots stay untagged.
    pub fn abandon(&mut self, plan: &Plan) {
        for &(_, c) in &plan.victims {
            self.cache.release(c);
        }
    }

    pub fn complete(&mut self, epoch: u64) {
        self.cache.complete(epoch);
    }

    pub fn complete_all(&mut self) {
        self.cache.complete_all();
    }

    pub fn new_turn(&mut self) {
        self.cache.new_turn();
    }

    /// Every slot's contents are gone (no GPU work in flight).
    pub fn clear(&mut self) {
        self.tags.fill(None);
        self.stage_use.fill(0);
        self.cache.clear();
    }

    /// Turn end (all GPU work complete): `(from, to)` bank slots to copy,
    /// staging copies the cache now maps to `to`.
    pub fn promote(&mut self) -> Vec<(u32, u32)> {
        let (n, staging_n) = (self.n_experts, self.staging);
        let stage = |l: usize, e: u32| (l % staging_n) * n + e as usize;
        let tags = &self.tags;
        let moved = self
            .cache
            .promote_staged(&|l, e| tags[stage(l, e)] == Some((l, e)));
        let base = self.base();
        moved
            .into_iter()
            .map(|(slot, l, e)| (stage(l, e) as u32, base + slot))
            .collect()
    }

    /// Test hook: clears staging tags and releases cache slots matching `f`.
    #[cfg(test)]
    pub fn forget_where(&mut self, f: impl Fn(usize, u32) -> bool) {
        for t in self.tags.iter_mut() {
            if t.is_some_and(|(l, e)| f(l, e)) {
                *t = None;
            }
        }
        let gone: Vec<u32> = (0..self.cache.slots() as u32)
            .filter(|&s| self.cache.occ(s).is_some_and(|(l, e)| f(l, e)))
            .collect();
        for s in gone {
            self.cache.release(s);
        }
    }
}
