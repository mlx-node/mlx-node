//! The shared expert bank: per projection one `PurgeableBank` pair (`w`
//! uint32 E2M1 codes, `s` uint8 E8M0 scales) of `[slots, rows, k / 8]` /
//! `[slots, rows, k / 32]`, used by every layer. Slots `0..n_experts` are
//! staging (slot = expert id), tagged `(layer, expert)`: a tag is cleared
//! before a refill and set only after every projection's read finished.
//! Slots `n_experts..` are the global cache ([`Cache`] directory).
//!
//! Between turns the banks are `Volatile`; the first use of a turn makes
//! them `NonVolatile` again and, when Metal emptied any of them, forgets
//! every tag and cache entry before a gather can read the slots. A bank of
//! another cache size is allocated only after the old one is gone.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use napi::{Error, Result};

use super::cache::{Cache, CacheMode, Source};
use super::io::{Extent, IoPool, IoStats, Priority, Read, Ticket};
use super::repack::SlotRepack;
use crate::array::purgeable::{BankWatch, PurgeableBank, PurgeableState};
use crate::array::{DType, MxArray};
use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("moe_stream bank: {msg}"))
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One layer's routed-expert tensor of one projection (`[n, rows, k]` MXFP4).
#[derive(Clone)]
pub struct ExpertTensor {
    pub shard: usize,
    /// The shard's data section start.
    pub data_offset: u64,
    pub info: GgufTensorInfo,
}

struct Proj {
    w: PurgeableBank,
    s: PurgeableBank,
    repack: SlotRepack,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BankStats {
    /// Layer steps that read anything.
    pub fills: usize,
    /// Slots (experts) read, all projections of one expert counted once.
    pub slots: usize,
    pub bytes: u64,
    pub ns: u64,
    /// Expert uses (distinct experts per layer step).
    pub uses: usize,
    /// Uses served from a cache slot.
    pub hits: usize,
    /// Staging copies moved into cache slots at turn ends (no read).
    pub promoted: usize,
    pub promote_ns: u64,
    /// Time the caller was blocked waiting for reads (`ns` minus what
    /// overlapped GPU work in a split step).
    pub wait_ns: u64,
    /// Layer steps whose reads overlapped GPU work (`prepare_split`), and
    /// those among them whose overlapped pass also gathered ready experts.
    pub split_steps: usize,
    pub split_mixed: usize,
}

/// Where one layer step's experts are: `slots[i]` is the bank slot of
/// `experts[i]`; `arrays` are the bank's (codes, scales) per projection.
/// `reading` (sorted) = experts whose slots are still being read
/// (`prepare_split` before `Pending::finish`).
pub struct Prepared {
    pub layer: usize,
    pub experts: Vec<u32>,
    pub slots: Vec<u32>,
    pub arrays: Vec<(MxArray, MxArray)>,
    pub epoch: u64,
    pub reading: Vec<u32>,
}

impl Prepared {
    pub fn slot(&self, e: u32) -> Option<u32> {
        self.experts.binary_search(&e).ok().map(|i| self.slots[i])
    }

    /// `e`'s slot is being read (no gather may use it before `finish`).
    pub fn is_reading(&self, e: u32) -> bool {
        self.reading.binary_search(&e).is_ok()
    }
}

/// The reads of one `prepare_split` step. `finish` waits for them and
/// publishes the slots; dropping it unfinished waits for them too and
/// leaves the slots unpublished (staging tags cleared, victims released).
pub struct Pending<'a> {
    bank: &'a StagingBank,
    ticket: Option<Ticket>,
    layer: usize,
    staging: Vec<u32>,
    victims: Vec<(u32, u32)>,
}

impl Pending<'_> {
    /// Blocks until every read is in, then tags the staging slots and
    /// commits the victims. A failed read publishes nothing.
    pub fn finish(mut self) -> Result<()> {
        let Some(ticket) = self.ticket.take() else {
            return Ok(());
        };
        let t = Instant::now();
        let io = ticket.wait();
        let waited = t.elapsed().as_nanos() as u64;
        let bank = self.bank;
        let mut g = lock(&bank.inner);
        let io: IoStats = match io {
            Ok(io) => io,
            Err(err) => {
                self.victims.iter().for_each(|&(_, c)| g.cache.release(c));
                return Err(err);
            }
        };
        for &e in &self.staging {
            g.tags[e as usize] = Some((self.layer, e));
        }
        for &(e, c) in &self.victims {
            g.cache.commit(c, self.layer, e);
        }
        drop(g);
        let mut st = lock(&bank.stats);
        st.fills += 1;
        st.slots += self.staging.len() + self.victims.len();
        st.bytes += io.bytes;
        st.ns += io.ns;
        st.wait_ns += waited;
        Ok(())
    }
}

impl Drop for Pending<'_> {
    fn drop(&mut self) {
        if let Some(ticket) = self.ticket.take() {
            // the slots stay unpublished: wait so no read outlives the step
            drop(ticket);
            let mut g = lock(&self.bank.inner);
            self.victims.iter().for_each(|&(_, c)| g.cache.release(c));
        }
    }
}

struct Inner {
    proj: Vec<Proj>,
    tags: Vec<Option<(usize, u32)>>,
    cache: Cache,
    in_turn: bool,
    /// Buffers of banks dropped by `rebuild` that an array may still hold.
    retired: Vec<BankWatch>,
}

pub struct StagingBank {
    pool: IoPool,
    /// `[layer][projection]`
    layers: Vec<Vec<ExpertTensor>>,
    n_experts: usize,
    mode: CacheMode,
    split: AtomicBool,
    inner: Mutex<Inner>,
    stats: Mutex<BankStats>,
    /// Test hook: no reads (a gather then meets stale tags).
    pub skip_fill: AtomicBool,
    #[cfg(test)]
    force_empty: AtomicBool,
}

fn alloc(first: &[ExpertTensor], slots: i64) -> Result<Vec<Proj>> {
    let mut proj = Vec::with_capacity(first.len());
    for t in first {
        let shape = t.info.mlx_shape();
        let (rows, k) = (shape[1], shape[2]);
        let w = PurgeableBank::new(&[slots, rows, k / 8], DType::Uint32)?;
        let s = PurgeableBank::new(&[slots, rows, k / 32], DType::Uint8)?;
        // SAFETY: both banks hold exactly `slots * rows * k / 8` u32 and
        // `slots * rows * k / 32` bytes and live as long as `repack` (both in
        // `Proj`); slots are written only under the bank lock.
        let repack = unsafe {
            SlotRepack::mxfp4(
                w.data_ptr().0.cast(),
                s.data_ptr().0,
                slots as usize,
                rows as usize,
                k as usize,
            )
        }?;
        proj.push(Proj { w, s, repack });
    }
    Ok(proj)
}

impl StagingBank {
    /// Staging only (no cache slots).
    pub fn new(pool: IoPool, layers: Vec<Vec<ExpertTensor>>) -> Result<Self> {
        Self::with_cache(pool, layers, 0)
    }

    /// `with_mode` under `CacheMode::Lru` (S16a).
    pub fn with_cache(
        pool: IoPool,
        layers: Vec<Vec<ExpertTensor>>,
        cache_slots: usize,
    ) -> Result<Self> {
        Self::with_mode(pool, layers, cache_slots, CacheMode::Lru)
    }

    /// One bank per projection of `n_experts + cache_slots` slots for the
    /// expert tensors `layers[layer][projection]` (same shape in every layer).
    pub fn with_mode(
        pool: IoPool,
        layers: Vec<Vec<ExpertTensor>>,
        cache_slots: usize,
        mode: CacheMode,
    ) -> Result<Self> {
        let first = layers.first().ok_or_else(|| bad("no layers"))?;
        let mut n_experts = 0;
        for (p, t) in first.iter().enumerate() {
            let shape = t.info.mlx_shape();
            if t.info.tensor_type != GgufTensorType::MXFP4
                || shape.len() != 3
                || shape.iter().any(|&d| d <= 0)
            {
                return Err(bad(format!(
                    "{}: expected a 3-D MXFP4 tensor, got {} {shape:?}",
                    t.info.name,
                    t.info.tensor_type.name()
                )));
            }
            for l in &layers {
                let o = l
                    .get(p)
                    .ok_or_else(|| bad("layers differ in projections"))?;
                if l.len() != first.len()
                    || o.info.mlx_shape() != shape
                    || o.info.tensor_type != t.info.tensor_type
                {
                    return Err(bad(format!(
                        "{} does not match {}",
                        o.info.name, t.info.name
                    )));
                }
            }
            if p > 0 && shape[0] != n_experts {
                return Err(bad("projections differ in expert count"));
            }
            n_experts = shape[0];
        }
        let n = n_experts as usize;
        let proj = alloc(first, n_experts + cache_slots as i64)?;
        Ok(Self {
            pool,
            inner: Mutex::new(Inner {
                proj,
                tags: vec![None; n],
                cache: Cache::lru(cache_slots, layers.len(), n).with_mode(mode),
                in_turn: true,
                retired: Vec::new(),
            }),
            layers,
            n_experts: n,
            mode,
            split: AtomicBool::new(false),
            stats: Mutex::new(BankStats::default()),
            skip_fill: AtomicBool::new(false),
            #[cfg(test)]
            force_empty: AtomicBool::new(false),
        })
    }

    pub fn n_experts(&self) -> usize {
        self.n_experts
    }

    pub fn mode(&self) -> CacheMode {
        self.mode
    }

    pub fn cache_slots(&self) -> usize {
        lock(&self.inner).cache.slots()
    }

    /// Bytes of one slot of every projection (MLX layout).
    pub fn slot_bytes(&self) -> u64 {
        self.layers[0]
            .iter()
            .map(|t| t.info.data_size().unwrap_or(0) / self.n_experts as u64)
            .sum()
    }

    /// Bytes of every bank.
    pub fn nbytes(&self) -> u64 {
        lock(&self.inner)
            .proj
            .iter()
            .map(|p| (p.w.nbytes() + p.s.nbytes()) as u64)
            .sum()
    }

    /// Bytes of every bank resident in RAM now.
    pub fn resident_bytes(&self) -> Result<u64> {
        let g = lock(&self.inner);
        let mut n = 0;
        for p in &g.proj {
            n += p.w.resident_bytes()? + p.s.resident_bytes()?;
        }
        Ok(n)
    }

    /// Projection `p`'s (codes, scales) arrays.
    pub fn arrays(&self, p: usize) -> Result<(MxArray, MxArray)> {
        let g = lock(&self.inner);
        let pr = g
            .proj
            .get(p)
            .ok_or_else(|| bad(format!("projection {p} of {}", g.proj.len())))?;
        Ok((pr.w.array().clone(), pr.s.array().clone()))
    }

    pub fn tags(&self) -> Vec<Option<(usize, u32)>> {
        lock(&self.inner).tags.clone()
    }

    pub fn stats(&self) -> BankStats {
        *lock(&self.stats)
    }

    /// Runs `f` on the cache directory (diagnostics, tests).
    pub fn with_cache_dir<T>(&self, f: impl FnOnce(&Cache) -> T) -> T {
        f(&lock(&self.inner).cache)
    }

    fn begin(&self, g: &mut Inner) -> Result<bool> {
        let mut emptied = false;
        for p in &g.proj {
            for b in [&p.w, &p.s] {
                emptied |= b.set_purgeable(PurgeableState::NonVolatile)? == PurgeableState::Empty;
            }
        }
        if emptied {
            g.tags.fill(None);
            g.cache.clear();
        }
        g.cache.new_turn();
        g.in_turn = true;
        Ok(emptied)
    }

    /// Between `begin_turn` (or the first use) and `end_turn`: non-volatile.
    pub fn in_turn(&self) -> bool {
        lock(&self.inner).in_turn
    }

    /// Makes the banks `NonVolatile` for a turn. Returns whether Metal had
    /// discarded any of them (every tag and cache entry is then forgotten).
    pub fn begin_turn(&self) -> Result<bool> {
        let mut g = lock(&self.inner);
        if g.in_turn {
            return Ok(false);
        }
        self.begin(&mut g)
    }

    /// After the turn's last eval: the banks become `Volatile` (the system
    /// may reclaim them until the next turn).
    pub fn end_turn(&self) -> Result<()> {
        let mut g = lock(&self.inner);
        if !g.in_turn {
            return Ok(());
        }
        crate::array::synchronize();
        g.cache.complete_all();
        let n = self.n_experts;
        let total = n + g.cache.slots();
        let t0 = std::time::Instant::now();
        let inner = &mut *g;
        let tags = &inner.tags;
        let moved = inner
            .cache
            .promote_staged(&|l, e| tags[e as usize] == Some((l, e)));
        for &(slot, _, e) in &moved {
            for p in &inner.proj {
                for b in [&p.w, &p.s] {
                    let sb = b.nbytes() / total;
                    // SAFETY: both ranges lie inside the bank (slot e < n and
                    // n + slot < total, `sb` = one slot's bytes), they do not
                    // overlap, the GPU is idle (synchronize above) and the
                    // bank lock is held; the bank is non-volatile in a turn.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            b.data_ptr().0.add(e as usize * sb),
                            b.data_ptr().0.add((n + slot as usize) * sb),
                            sb,
                        );
                    }
                }
            }
        }
        if !moved.is_empty() {
            let mut st = lock(&self.stats);
            st.promoted += moved.len();
            st.promote_ns += t0.elapsed().as_nanos() as u64;
        }
        let state = if self.empty_between_turns() {
            PurgeableState::Empty
        } else {
            PurgeableState::Volatile
        };
        g.in_turn = false;
        for p in &g.proj {
            p.w.set_purgeable(state)?;
            p.s.set_purgeable(state)?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn empty_between_turns(&self) -> bool {
        self.force_empty.load(Ordering::Relaxed)
            || std::env::var_os("MLX_DSV41_TEST_FORCE_EMPTY").is_some_and(|v| v == "1")
    }

    #[cfg(not(test))]
    fn empty_between_turns(&self) -> bool {
        false
    }

    /// Test hook: `end_turn` empties the banks instead of making them volatile.
    #[cfg(test)]
    pub fn test_force_empty(&self, on: bool) {
        self.force_empty.store(on, Ordering::Relaxed);
    }

    /// Drops the banks and allocates new ones with `cache_slots` cache
    /// slots, after the old buffers are gone from MLX (refuses while any
    /// array still holds one; the bank then has no slots until a rebuild
    /// succeeds).
    pub fn rebuild(&self, cache_slots: usize) -> Result<()> {
        let mut g = lock(&self.inner);
        crate::array::synchronize();
        let watches: Vec<_> = g
            .proj
            .iter()
            .flat_map(|p| [p.w.watch(), p.s.watch()])
            .collect();
        // kept across failed rebuilds: every retired buffer must be gone
        // before any new bank is allocated
        g.retired.extend(watches);
        g.proj = Vec::new();
        g.tags.fill(None);
        g.cache = Cache::lru(0, self.layers.len(), self.n_experts);
        crate::array::clear_cache();
        g.retired.retain(|w| w.alive());
        if !g.retired.is_empty() {
            return Err(bad(
                "an array still holds the old bank; a second bank is not allocated",
            ));
        }
        g.proj = alloc(&self.layers[0], (self.n_experts + cache_slots) as i64)?;
        g.cache = Cache::lru(cache_slots, self.layers.len(), self.n_experts).with_mode(self.mode);
        g.in_turn = true;
        Ok(())
    }

    fn reads(
        &self,
        g: &Inner,
        layer: usize,
        e: u32,
        slot: usize,
        out: &mut Vec<Read>,
    ) -> Result<()> {
        for (t, p) in self.layers[layer].iter().zip(&g.proj) {
            let ext = Extent::expert(t.shard, t.data_offset, &t.info, e as usize)?;
            out.push(Read::slot(ext, &p.repack, slot));
        }
        Ok(())
    }

    fn submit(&self, reads: Vec<Read>) -> Result<super::io::IoStats> {
        // SAFETY: every slot read is untagged (staging) or an unmapped victim
        // (cache), so no gather passes the slot checks on it until the reads
        // are done; GPU work was drained and the bank lock is held.
        unsafe { self.pool.submit(reads, Priority::High) }?.wait()
    }

    fn check_step(&self, g: &Inner, layer: usize, experts: &[u32]) -> Result<()> {
        if layer >= self.layers.len() {
            return Err(bad(format!("layer {layer} of {}", self.layers.len())));
        }
        if g.proj.is_empty() {
            return Err(bad("no bank (a rebuild failed)"));
        }
        if let Some(&e) = experts.iter().find(|&&e| e as usize >= self.n_experts) {
            return Err(bad(format!("expert {e} of {}", self.n_experts)));
        }
        if experts.windows(2).any(|w| w[0] >= w[1]) {
            return Err(bad("experts must be sorted and distinct"));
        }
        Ok(())
    }

    /// Slots for one layer step's distinct, sorted `experts`: cache hits
    /// in place; with `decode`, misses read into cache victims; the rest
    /// into staging slots whose tag is not `(layer, e)`. `uses[i]` = rows
    /// using `experts[i]` (`CacheMode::Seeded` prefill order; may be empty).
    /// Drains the GPU before any CPU write. Graphs built on the returned
    /// arrays must be evaluated, then `done(epoch)`, before the next step may
    /// evict them.
    pub fn prepare(
        &self,
        layer: usize,
        experts: &[u32],
        uses: &[u32],
        decode: bool,
    ) -> Result<Prepared> {
        let (mut p, pending) = self.prepare_split(layer, experts, uses, decode)?;
        if let Some(pending) = pending {
            pending.finish()?;
        }
        p.reading.clear();
        Ok(p)
    }

    /// `prepare` that returns once the reads are queued: `Prepared.reading`
    /// lists the experts still being read and the `Pending` (None when
    /// nothing is read) must be finished before a gather uses their slots.
    /// Gathers on the other slots may run meanwhile: the reads write only
    /// slots no gather of this step or an unfinished earlier one uses.
    pub fn prepare_split(
        &self,
        layer: usize,
        experts: &[u32],
        uses: &[u32],
        decode: bool,
    ) -> Result<(Prepared, Option<Pending<'_>>)> {
        let mut g = lock(&self.inner);
        self.check_step(&g, layer, experts)?;
        if !uses.is_empty() && uses.len() != experts.len() {
            return Err(bad(format!(
                "{} use counts for {} experts",
                uses.len(),
                experts.len()
            )));
        }
        if !g.in_turn {
            self.begin(&mut g)?;
        }
        let n = self.n_experts as u32;
        let inner = &mut *g;
        let tags = &inner.tags;
        let (epoch, src) = inner.cache.plan_step(layer, experts, uses, decode, &|e| {
            tags[e as usize] == Some((layer, e))
        });
        let (mut slots, mut staging, mut victims, mut hits) =
            (Vec::new(), Vec::new(), Vec::new(), 0);
        for (&e, s) in experts.iter().zip(&src) {
            slots.push(match *s {
                Source::Hit(c) => {
                    hits += 1;
                    n + c
                }
                Source::Victim(c) => {
                    victims.push((e, c));
                    n + c
                }
                Source::Staging => {
                    if g.tags[e as usize] != Some((layer, e)) {
                        staging.push(e);
                    }
                    e
                }
            });
        }
        let mut pending = None;
        let mut reading = Vec::new();
        if self.skip_fill.load(Ordering::Relaxed) {
            victims.iter().for_each(|&(_, c)| g.cache.release(c));
        } else if !staging.is_empty() || !victims.is_empty() {
            crate::array::synchronize();
            g.cache.complete_all();
            let mut reads = Vec::new();
            for &e in &staging {
                g.tags[e as usize] = None;
                self.reads(&g, layer, e, e as usize, &mut reads)?;
            }
            for &(e, c) in &victims {
                self.reads(&g, layer, e, (n + c) as usize, &mut reads)?;
            }
            // SAFETY: every slot read is untagged (staging) or an unmapped
            // victim (cache), so no gather passes the slot checks on it until
            // `Pending::finish`; earlier GPU work was drained above, and a
            // `Pending` (finished or dropped) waits for the batch before the
            // bank can be used for another step.
            let ticket = match unsafe { self.pool.submit(reads, Priority::High) } {
                Ok(t) => t,
                Err(err) => {
                    victims.iter().for_each(|&(_, c)| g.cache.release(c));
                    return Err(err);
                }
            };
            reading = staging
                .iter()
                .chain(victims.iter().map(|(e, _)| e))
                .copied()
                .collect();
            reading.sort_unstable();
            pending = Some(Pending {
                bank: self,
                ticket: Some(ticket),
                layer,
                staging,
                victims,
            });
        }
        {
            let mut st = lock(&self.stats);
            st.uses += experts.len();
            st.hits += hits;
        }
        let p = Prepared {
            layer,
            experts: experts.to_vec(),
            slots,
            arrays: g
                .proj
                .iter()
                .map(|p| (p.w.array().clone(), p.s.array().clone()))
                .collect(),
            epoch,
            reading,
        };
        Ok((p, pending))
    }

    /// Counts one step whose reads overlapped GPU work (`mixed`: ready
    /// experts were gathered meanwhile, not only the shared expert).
    pub fn note_split(&self, mixed: bool) {
        let mut st = lock(&self.stats);
        st.split_steps += 1;
        st.split_mixed += usize::from(mixed);
    }

    /// Whether decode steps overlap their reads with the gathers of the
    /// slots they already hold (`MLX_DSV41_SPLIT`, stage P3).
    pub fn split(&self) -> bool {
        self.split.load(Ordering::Relaxed)
    }

    pub fn set_split(&self, on: bool) {
        self.split.store(on, Ordering::Relaxed);
    }

    /// Test hook: forgets every staging tag and cache entry (as an `Empty`
    /// bank), so the next steps read every expert again.
    #[cfg(test)]
    pub fn test_forget_all(&self) {
        self.test_forget_where(|_, _| true);
    }

    /// Test hook: forgets the staging tags and cache entries of the
    /// `(layer, expert)` pairs `f` picks (their next use is a read).
    #[cfg(test)]
    pub fn test_forget_where(&self, f: impl Fn(usize, u32) -> bool) {
        let mut g = lock(&self.inner);
        crate::array::synchronize();
        for t in g.tags.iter_mut() {
            if t.is_some_and(|(l, e)| f(l, e)) {
                *t = None;
            }
        }
        let gone: Vec<u32> = (0..g.cache.slots() as u32)
            .filter(|&s| g.cache.occ(s).is_some_and(|(l, e)| f(l, e)))
            .collect();
        for s in gone {
            g.cache.release(s);
        }
    }

    /// The gathers of step `epoch` (and every earlier one) are evaluated.
    pub fn done(&self, epoch: u64) {
        lock(&self.inner).cache.complete(epoch);
    }

    /// Panics (release builds too) unless every slot of `p` holds its
    /// expert of `p.layer`: a gather on a stale slot would compute with
    /// another layer's (or discarded) weights.
    pub fn assert_slots(&self, p: &Prepared) {
        self.assert_slots_where(p, |_| true);
    }

    /// `assert_slots` for the experts not in `p.reading`.
    pub fn assert_ready_slots(&self, p: &Prepared) {
        self.assert_slots_where(p, |e| !p.is_reading(e));
    }

    fn assert_slots_where(&self, p: &Prepared, check: impl Fn(u32) -> bool) {
        let n = self.n_experts as u32;
        let stale = {
            let g = lock(&self.inner);
            p.experts
                .iter()
                .zip(&p.slots)
                .filter(|(e, _)| check(**e))
                .find_map(|(&e, &s)| {
                    let held = if s < n {
                        (s == e).then(|| g.tags[s as usize]).flatten()
                    } else {
                        g.cache.occ(s - n)
                    };
                    (held != Some((p.layer, e))).then_some((e, s, held))
                })
        };
        assert!(
            stale.is_none(),
            "moe_stream bank: gather of layer {} on stale slot {stale:?}",
            p.layer
        );
    }

    /// Reads every expert of `experts` into its staging slot unless the
    /// slot holds `layer`'s copy (no cache). Waits for all GPU work first;
    /// graphs built on the bank must be evaluated before the next `fill`.
    pub fn fill(&self, layer: usize, experts: &[u32]) -> Result<()> {
        if self.skip_fill.load(Ordering::Relaxed) {
            return Ok(());
        }
        let mut want = experts.to_vec();
        want.sort_unstable();
        want.dedup();
        let mut g = lock(&self.inner);
        self.check_step(&g, layer, &want)?;
        if !g.in_turn {
            self.begin(&mut g)?;
        }
        want.retain(|&e| g.tags[e as usize] != Some((layer, e)));
        if want.is_empty() {
            return Ok(());
        }
        crate::array::synchronize();
        let mut reads = Vec::new();
        for &e in &want {
            g.tags[e as usize] = None;
            self.reads(&g, layer, e, e as usize, &mut reads)?;
        }
        let io = self.submit(reads)?;
        for &e in &want {
            g.tags[e as usize] = Some((layer, e));
        }
        let mut st = lock(&self.stats);
        st.fills += 1;
        st.slots += want.len();
        st.bytes += io.bytes;
        st.ns += io.ns;
        Ok(())
    }

    /// Panics unless every staging slot of `experts` holds `layer`'s expert.
    pub fn assert_tags(&self, layer: usize, experts: &[u32]) {
        let stale = {
            let g = lock(&self.inner);
            experts
                .iter()
                .find(|&&e| g.tags.get(e as usize).copied().flatten() != Some((layer, e)))
                .map(|&e| (e, g.tags.get(e as usize).copied().flatten()))
        };
        assert!(
            stale.is_none(),
            "moe_stream bank: gather of layer {layer} on stale slot {stale:?}"
        );
    }
}
