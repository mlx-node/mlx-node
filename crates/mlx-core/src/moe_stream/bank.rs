//! The shared expert bank: per projection one `PurgeableBank` pair (`w`
//! uint32 E2M1 codes, `s` uint8 E8M0 scales) of `[slots, rows, k / 8]` /
//! `[slots, rows, k / 32]`, used by every layer. Slots `0..n_experts` are
//! staging (slot = expert id). Each slot carries a `(layer, expert)` tag
//! that is cleared before a refill and set only after every projection's
//! read finished.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use napi::{Error, Result};

use super::io::{Extent, IoPool, Priority, Read};
use super::repack::SlotRepack;
use crate::array::purgeable::PurgeableBank;
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
    /// `fill` calls that read anything.
    pub fills: usize,
    /// Slots (experts) read, all projections of one expert counted once.
    pub slots: usize,
    pub bytes: u64,
    pub ns: u64,
}

pub struct StagingBank {
    pool: IoPool,
    proj: Vec<Proj>,
    /// `[layer][projection]`
    layers: Vec<Vec<ExpertTensor>>,
    tags: Mutex<Vec<Option<(usize, u32)>>>,
    stats: Mutex<BankStats>,
    /// Test hook: `fill` does nothing (a gather then meets stale tags).
    pub skip_fill: AtomicBool,
}

impl StagingBank {
    /// One bank per projection, `n_experts` staging slots, for the expert
    /// tensors `layers[layer][projection]` (same shape in every layer).
    pub fn new(pool: IoPool, layers: Vec<Vec<ExpertTensor>>) -> Result<Self> {
        let first = layers.first().ok_or_else(|| bad("no layers"))?;
        let mut proj = Vec::with_capacity(first.len());
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
            let (n, rows, k) = (shape[0], shape[1], shape[2]);
            if p > 0 && n != n_experts {
                return Err(bad("projections differ in expert count"));
            }
            n_experts = n;
            let w = PurgeableBank::new(&[n, rows, k / 8], DType::Uint32)?;
            let s = PurgeableBank::new(&[n, rows, k / 32], DType::Uint8)?;
            // SAFETY: both banks hold exactly `n * rows * k / 8` u32 and
            // `n * rows * k / 32` bytes and live as long as `repack` (both in
            // `Proj`); slots are written only by `fill` under the tag lock.
            let repack = unsafe {
                SlotRepack::mxfp4(
                    w.data_ptr().0.cast(),
                    s.data_ptr().0,
                    n as usize,
                    rows as usize,
                    k as usize,
                )
            }?;
            proj.push(Proj { w, s, repack });
        }
        Ok(Self {
            pool,
            proj,
            layers,
            tags: Mutex::new(vec![None; n_experts as usize]),
            stats: Mutex::new(BankStats::default()),
            skip_fill: AtomicBool::new(false),
        })
    }

    /// Bytes of every bank.
    pub fn nbytes(&self) -> u64 {
        self.proj
            .iter()
            .map(|p| (p.w.nbytes() + p.s.nbytes()) as u64)
            .sum()
    }

    /// Projection `p`'s (codes, scales) arrays.
    pub fn arrays(&self, p: usize) -> Result<(MxArray, MxArray)> {
        let pr = self
            .proj
            .get(p)
            .ok_or_else(|| bad(format!("projection {p} of {}", self.proj.len())))?;
        Ok((pr.w.array().clone(), pr.s.array().clone()))
    }

    pub fn tags(&self) -> Vec<Option<(usize, u32)>> {
        lock(&self.tags).clone()
    }

    pub fn stats(&self) -> BankStats {
        *lock(&self.stats)
    }

    /// Reads every expert of `experts` whose slot does not hold `layer`'s
    /// copy. Waits for all GPU work first, so no running kernel reads a slot
    /// being refilled; graphs built on the bank must be evaluated before the
    /// next `fill`. A failed read leaves the slots untagged.
    pub fn fill(&self, layer: usize, experts: &[u32]) -> Result<()> {
        if self.skip_fill.load(Ordering::Relaxed) {
            return Ok(());
        }
        let src = self
            .layers
            .get(layer)
            .ok_or_else(|| bad(format!("layer {layer} of {}", self.layers.len())))?;
        let mut tags = lock(&self.tags);
        let mut missing: Vec<u32> = Vec::new();
        for &e in experts {
            let slot = tags
                .get(e as usize)
                .ok_or_else(|| bad(format!("expert {e} of {}", tags.len())))?;
            if *slot != Some((layer, e)) && !missing.contains(&e) {
                missing.push(e);
            }
        }
        if missing.is_empty() {
            return Ok(());
        }
        crate::array::synchronize();
        let mut reads = Vec::with_capacity(missing.len() * src.len());
        for &e in &missing {
            tags[e as usize] = None;
            for (t, p) in src.iter().zip(&self.proj) {
                let ext = Extent::expert(t.shard, t.data_offset, &t.info, e as usize)?;
                reads.push(Read::slot(ext, &p.repack, e as usize));
            }
        }
        // SAFETY: the slots belong to this bank and are untagged, so no gather
        // passes `assert_tags` on them until the reads are done; GPU work was
        // drained above and the tag lock is held for the whole fill.
        let io = unsafe { self.pool.submit(reads, Priority::High) }?.wait()?;
        for &e in &missing {
            tags[e as usize] = Some((layer, e));
        }
        let mut st = lock(&self.stats);
        st.fills += 1;
        st.slots += missing.len();
        st.bytes += io.bytes;
        st.ns += io.ns;
        Ok(())
    }

    /// Panics (release builds too) unless every slot of `experts` holds
    /// `layer`'s expert: a gather on a stale slot would compute with another
    /// layer's weights.
    pub fn assert_tags(&self, layer: usize, experts: &[u32]) {
        let stale = {
            let tags = lock(&self.tags);
            experts
                .iter()
                .find(|&&e| tags.get(e as usize).copied().flatten() != Some((layer, e)))
                .map(|&e| (e, tags.get(e as usize).copied().flatten()))
        };
        assert!(
            stale.is_none(),
            "moe_stream bank: gather of layer {layer} on stale slot {stale:?}"
        );
    }
}
