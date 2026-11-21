//! The routed-expert bank of the trunk layers, one per expert class: A
//! (gate Q4_K, up Q4_K, down Q6_K) and B (all Q4_K). Per projection a
//! `PurgeableBank` triplet in the loader's K-quant form (`w` uint32 codes,
//! `s` uint8 or int8 sub-scales, `b` f16 `d` / `dmin`) of
//! `[slots, rows, cols]`. Slots `0..L * n_experts` are `L` staging layers
//! (class-local layer `l` stages expert `e` in `(l % L) * n_experts + e`);
//! the rest are the class's cache ([`Cache`], class-local layer ids). A miss
//! is read by the IO pool in row bands of about [`BAND_BYTES`] per
//! projection, each repacked on its reader thread by `kq_slot` (the bytes of
//! `cpu_kquant_band_into`), so a slot holds the bytes the whole-file loader
//! makes. A staging tag is cleared before a refill and set only after every
//! read of the step is in; `prepare` returns only when no read it queued is
//! in flight.
//!
//! Gathers stay lazy after `prepare`. A step's slots stay pinned until the
//! caller settles its epoch (`settle` after a host read that descends from
//! the step's gathers, `settle_all` after a wave's last eval): a cache slot
//! is a victim only once its last use settled, and a staging slot whose
//! last use has not settled is refilled only after a drain of the GPU.
//!
//! Between turns the banks are `Volatile`; the first use of a turn makes
//! them `NonVolatile` again and, when Metal emptied a class's banks, forgets
//! that class's tags and cache entries before a gather can read the slots.

use std::collections::{HashMap, VecDeque};
use std::ffi::CString;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use napi::{Error, Result};

use super::config::MoeConfig;
use super::kq_slot;
use super::scan::{ExpertClass, ExpertLayer, Located};
use crate::array::alloc::{DstPtrs, MetalResidencySet, OutLayout};
use crate::array::purgeable::{BankWatch, PurgeableBank, PurgeableState};
use crate::array::{DType, MxArray};
use crate::moe_stream::cache::{Cache, Source};
use crate::moe_stream::io::{BatchWatch, Extent, IoPool, IoStats, Priority, Read, Ticket};
use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};
use crate::utils::gguf_kquant::KQuantFormat;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 expert bank: {msg}"))
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// GGUF bytes one read of a projection's row band aims at: a 1536 x 1536
/// Q4_K projection reads as 3 bands, Q6_K as 4, so the reader threads
/// share one expert.
pub const BAND_BYTES: u64 = 512 << 10;

const PAGE_BYTES: u64 = 16 << 10;

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// Modular inverse of odd `a` modulo power-of-two `m`.
fn mod_inv_pow2(a: u64, m: u64) -> u64 {
    debug_assert!(m.is_power_of_two() && (a & 1) == 1);
    let mask = m - 1;
    let mut inv = 1u64;
    for _ in 0..6 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(a.wrapping_mul(inv))) & mask;
    }
    inv
}

/// Row stride for splitting a projection of `rows` rows (`row_bytes` each) into
/// ~`band_bytes` reads. Picks an exact 16 KiB-aligned multiple when one lies in
/// `[band_bytes / 2, band_bytes * 3 / 2]`.
pub(crate) fn band_rows(rows: usize, row_bytes: usize, band_bytes: u64) -> usize {
    if rows <= 1 || row_bytes == 0 {
        return rows;
    }
    let rb = row_bytes as u64;
    let total = (rows as u64).saturating_mul(rb);
    if band_bytes >= total {
        return rows;
    }
    if band_bytes >= PAGE_BYTES {
        let g = gcd_u64(rb, PAGE_BYTES);
        let align_rows = PAGE_BYTES / g;
        let align_bytes = align_rows.saturating_mul(rb);
        let lo = band_bytes / 2;
        let hi = band_bytes.saturating_add(band_bytes / 2);
        if align_bytes > 0 && align_bytes <= hi {
            let mut k = (band_bytes.saturating_add(align_bytes / 2) / align_bytes).max(1);
            if k.saturating_mul(align_bytes) > hi && k > 1 {
                k -= 1;
            }
            let step_bytes = k.saturating_mul(align_bytes);
            if (lo..=hi).contains(&step_bytes) {
                return k.saturating_mul(align_rows).min(rows as u64) as usize;
            }
        }
    }
    let bands = total.div_ceil(band_bytes.max(1)).clamp(1, rows as u64) as usize;
    rows.div_ceil(bands)
}

/// Disjoint `(row0, len)` ranges covering `0..rows`. When `band_rows` is an
/// exact 16 KiB multiple and `offset` permits page-aligned row boundaries,
/// places every internal cut on an exact 16 KiB page edge.
pub(crate) fn band_ranges(
    offset: u64,
    rows: usize,
    row_bytes: usize,
    band_bytes: u64,
) -> Vec<(usize, usize)> {
    if rows == 0 {
        return Vec::new();
    }
    let step = band_rows(rows, row_bytes, band_bytes).max(1);
    if step >= rows {
        return vec![(0, rows)];
    }
    let rb = row_bytes as u64;
    let g = gcd_u64(rb, PAGE_BYTES);
    if (step as u64).saturating_mul(rb).is_multiple_of(PAGE_BYTES) && offset.is_multiple_of(g) {
        let m = PAGE_BYTES / g;
        let phase = if m == 1 {
            0
        } else {
            let a = rb / g;
            let b = (offset / g) % m;
            (((m - b) % m) * mod_inv_pow2(a, m) % m) as usize
        };
        let min_band = (step / 2).max(1);
        let mut first = phase;
        while first < min_band {
            first += step;
        }
        if first < rows {
            let mut out = Vec::with_capacity(rows.div_ceil(step) + 1);
            let mut prev = 0usize;
            let mut cut = first;
            while cut < rows {
                if rows - cut <= step / 2 && prev > 0 {
                    break;
                }
                out.push((prev, cut - prev));
                prev = cut;
                cut += step;
            }
            out.push((prev, rows - prev));
            return out;
        }
    }
    (0..rows)
        .step_by(step)
        .map(|r0| (r0, step.min(rows - r0)))
        .collect()
}

pub(crate) fn sort_reads(reads: &mut [Read]) {
    reads.sort_unstable_by_key(|r| (r.extent.shard, r.extent.offset));
}

/// The streamed classes, in bank index order.
pub const CLASSES: [ExpertClass; 2] = [ExpertClass::Q6kDown, ExpertClass::Q4kDown];

/// Half-life of LFU-decay expert cache scores, in decode tokens.
pub const LFU_HALF_LIFE_TOKENS: f64 = 256.0;

fn use_lfu_policy() -> bool {
    std::env::var("MLX_HY4_CACHE_POLICY").is_ok_and(|v| v.eq_ignore_ascii_case("lfu"))
}

fn prefill_nax_enabled() -> bool {
    std::env::var("MLX_HY4_PREFILL_NAX").as_deref() != Ok("0")
}

pub fn decode_prefetch_enabled() -> bool {
    std::env::var("MLX_HY4_DECODE_PREFETCH").as_deref() != Ok("0")
}

fn lfu_half_life(n_layers: usize, n_used: usize) -> f64 {
    (LFU_HALF_LIFE_TOKENS * (n_layers.max(1) * n_used.max(1)) as f64).max(1.0)
}

fn make_directory(
    staging: usize,
    n_layers: usize,
    n_experts: usize,
    cache_slots: usize,
    n_used: usize,
) -> Result<Directory> {
    if use_lfu_policy() {
        Directory::new_with_lfu(
            staging,
            n_layers,
            n_experts,
            cache_slots,
            lfu_half_life(n_layers, n_used),
        )
    } else {
        Directory::new(staging, n_layers, n_experts, cache_slots)
    }
}

/// [`KqBank::residency`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Residency {
    pub resident: u64,
    /// Resident bytes of the banks the kernel holds volatile or empty.
    pub volatile: u64,
}

/// Whether the kernel holds `b`'s pages volatile or empty (`None` without a
/// reading).
#[cfg(target_os = "macos")]
fn kernel_volatile(b: &PurgeableBank) -> Option<bool> {
    const VM_PURGABLE_GET_STATE: libc::c_int = 1;
    const VM_PURGABLE_STATE_MASK: libc::c_int = 3;
    const VM_PURGABLE_NONVOLATILE: libc::c_int = 0;
    unsafe extern "C" {
        static mach_task_self_: libc::mach_port_t;
        fn vm_purgable_control(
            task: libc::mach_port_t,
            address: libc::uintptr_t,
            control: libc::c_int,
            state: *mut libc::c_int,
        ) -> libc::kern_return_t;
    }
    let mut state: libc::c_int = 0;
    // SAFETY: a query on this task's own live mapping; `state` is writable.
    let kr = unsafe {
        vm_purgable_control(
            mach_task_self_,
            b.data_ptr().0 as libc::uintptr_t,
            VM_PURGABLE_GET_STATE,
            &mut state,
        )
    };
    (kr == libc::KERN_SUCCESS).then_some(state & VM_PURGABLE_STATE_MASK != VM_PURGABLE_NONVOLATILE)
}

#[cfg(not(target_os = "macos"))]
fn kernel_volatile(_: &PurgeableBank) -> Option<bool> {
    None
}

pub fn class_index(class: ExpertClass) -> Option<usize> {
    CLASSES.iter().position(|&c| c == class)
}

/// GGUF type of gate, up and down for a streamed class.
pub fn class_types(class: ExpertClass) -> Option<[GgufTensorType; 3]> {
    use GgufTensorType::{Q4K, Q6K};
    match class {
        ExpertClass::Q6kDown => Some([Q4K, Q4K, Q6K]),
        ExpertClass::Q4kDown => Some([Q4K, Q4K, Q4K]),
        ExpertClass::Bf16 => None,
    }
}

/// One projection of one expert: `rows` outputs of `k` inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjGeo {
    pub format: KQuantFormat,
    pub rows: usize,
    pub k: usize,
}

impl ProjGeo {
    pub fn weight_cols(&self) -> usize {
        self.format.weight_cols(self.k)
    }

    pub fn scales_cols(&self) -> usize {
        self.format.scales_cols(self.k)
    }

    pub fn biases_cols(&self) -> usize {
        self.format.biases_cols(self.k)
    }

    pub fn scales_dtype(&self) -> DType {
        if self.format.scales_are_signed() {
            DType::Int8
        } else {
            DType::Uint8
        }
    }

    /// Bank bytes of one expert (MLX form).
    pub fn slot_bytes(&self) -> u64 {
        (self.rows * (self.weight_cols() * 4 + self.scales_cols() + self.biases_cols() * 2)) as u64
    }

    /// GGUF bytes of one expert.
    pub fn file_bytes(&self) -> u64 {
        (self.rows * self.format.row_bytes(self.k)) as u64
    }
}

/// Gate and up map `latent` inputs to `ff` outputs (ggml `{latent, ff, E}`),
/// down maps `ff` back to `latent` (`{ff, latent, E}`).
pub fn geometry(moe: &MoeConfig, class: ExpertClass) -> Result<[ProjGeo; 3]> {
    let types = class_types(class).ok_or_else(|| bad(format!("{class:?} is not streamed")))?;
    let rk = [
        (moe.ff, moe.latent),
        (moe.ff, moe.latent),
        (moe.latent, moe.ff),
    ];
    let mut out = [ProjGeo {
        format: KQuantFormat::Q4K,
        rows: 0,
        k: 0,
    }; 3];
    for (p, (ty, (rows, k))) in types.into_iter().zip(rk).enumerate() {
        let format = ty
            .k_quant_format()
            .ok_or_else(|| bad(format!("{} is not a K-quant", ty.name())))?;
        if rows == 0 || k == 0 || !k.is_multiple_of(format.block_size()) {
            return Err(bad(format!(
                "projection {p}: {rows} x {k} does not fit {} blocks",
                format.mlx_mode()
            )));
        }
        out[p] = ProjGeo { format, rows, k };
    }
    Ok(out)
}

/// Bank bytes of one expert of `class` (gate + up + down).
pub fn slot_bytes(moe: &MoeConfig, class: ExpertClass) -> Result<u64> {
    Ok(geometry(moe, class)?.iter().map(ProjGeo::slot_bytes).sum())
}

/// GGUF bytes one miss of `class` reads.
pub fn file_bytes(moe: &MoeConfig, class: ExpertClass) -> Result<u64> {
    Ok(geometry(moe, class)?.iter().map(ProjGeo::file_bytes).sum())
}

/// Refuses any tensor of layer `il` whose type or ggml dims are not exactly
/// what `geo` expects (`{k, rows, n_experts}`): with latent == ff a swapped
/// projection has the same shape, so only the per-projection dims catch it.
pub fn check_layer(
    il: usize,
    layer: &ExpertLayer,
    class: ExpertClass,
    geo: &[ProjGeo; 3],
    n_experts: usize,
) -> Result<[Arc<Located>; 3]> {
    if layer.class != class {
        return Err(bad(format!(
            "layer {il} is {:?}, the bank class is {class:?}",
            layer.class
        )));
    }
    let types = class_types(class).ok_or_else(|| bad(format!("{class:?} is not streamed")))?;
    let locs = [&layer.gate, &layer.up, &layer.down];
    for ((loc, g), ty) in locs.iter().zip(geo).zip(types) {
        let want = [g.k as u64, g.rows as u64, n_experts as u64];
        let info = &loc.info;
        if info.dims != want || info.tensor_type != ty {
            return Err(bad(format!(
                "'{}' is {} {:?}, expected {} {want:?}",
                info.name,
                info.tensor_type.name(),
                info.dims,
                ty.name()
            )));
        }
        let size = info.data_size()?;
        if n_experts == 0 || size != g.file_bytes() * n_experts as u64 {
            return Err(bad(format!(
                "'{}' holds {size} bytes, not {n_experts} experts of {}",
                info.name,
                g.file_bytes()
            )));
        }
    }
    Ok(locs.map(|l| Arc::new(l.clone())))
}

/// Which bank serves each layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassLayout {
    /// Per layer: (class index, class-local layer); `None` = dense FFN.
    pub of_layer: Vec<Option<(usize, usize)>>,
    /// Per class index: its layers, in class-local order.
    pub layers: [Vec<usize>; 2],
}

impl ClassLayout {
    pub fn new(classes: &[Option<ExpertClass>]) -> Result<Self> {
        let mut layers = [Vec::new(), Vec::new()];
        let mut of_layer = Vec::with_capacity(classes.len());
        for (il, c) in classes.iter().enumerate() {
            of_layer.push(match c {
                None => None,
                Some(c) => {
                    let ci = class_index(*c)
                        .ok_or_else(|| bad(format!("layer {il}: {c:?} is not streamed")))?;
                    layers[ci].push(il);
                    Some((ci, layers[ci].len() - 1))
                }
            });
        }
        Ok(Self { of_layer, layers })
    }
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

/// One class's slot directory: staging tags and the cache. Pure host
/// bookkeeping; the bank owns the bytes.
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

    /// `staging` = 1 or 2 staging layers.
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

    /// Whether bank slot `slot` holds `(layer, e)`.
    pub fn holds(&self, layer: usize, e: u32, slot: u32) -> bool {
        let base = self.base();
        if slot < base {
            slot as usize == self.stage(layer, e) && self.tags[slot as usize] == Some((layer, e))
        } else {
            self.cache.occ(slot - base) == Some((layer, e))
        }
    }

    /// `experts` sorted and distinct; `uses[i]` = rows using `experts[i]`
    /// (may be empty).
    pub fn check_step(&self, layer: usize, experts: &[u32], uses: &[u32]) -> Result<()> {
        if layer >= self.n_layers {
            return Err(bad(format!("class layer {layer} of {}", self.n_layers)));
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
                    if valid {
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
}

/// One projection's bank memory: `slots` experts of `rows` rows in the
/// K-quant form of `format` (`RepackDst::alloc` row-major arrays).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SlotDst {
    dst: DstPtrs,
    format: KQuantFormat,
    slots: usize,
    rows: usize,
}

impl SlotDst {
    /// # Safety
    /// `dst` addresses `dst.rows` rows of `.weight`, `.scales` and
    /// `.biases` of `format`'s K-quant contract with `k = dst.k`, all valid
    /// while any read into a slot is in flight, and no one else touches a
    /// slot while a read into it is in flight.
    pub(crate) unsafe fn new(dst: DstPtrs, format: KQuantFormat, rows: usize) -> Result<Self> {
        let (Some(s), Some(b)) = (dst.scales, dst.biases) else {
            return Err(bad("a K-quant slot needs scales and biases"));
        };
        if dst.weight.0.is_null() || s.0.is_null() || b.0.is_null() {
            return Err(bad("null slot destination"));
        }
        if dst.layout != OutLayout::RowMajor {
            return Err(bad("expert slots are row-major"));
        }
        if dst.k == 0 || !dst.k.is_multiple_of(format.block_size()) {
            return Err(bad(format!(
                "K {} is not a positive multiple of {}",
                dst.k,
                format.block_size()
            )));
        }
        if rows == 0 || dst.rows == 0 || !dst.rows.is_multiple_of(rows) {
            return Err(bad(format!(
                "{} rows do not split into slots of {rows}",
                dst.rows
            )));
        }
        Ok(Self {
            slots: dst.rows / rows,
            dst,
            format,
            rows,
        })
    }

    /// GGUF bytes of one slot.
    pub(crate) fn file_bytes(&self) -> u64 {
        (self.rows * self.format.row_bytes(self.dst.k)) as u64
    }

    /// Repacks one expert's GGUF blocks of `info` into `slot`.
    #[cfg(test)]
    pub(crate) fn write(&self, info: &GgufTensorInfo, slot: usize, blocks: &[u8]) -> Result<()> {
        self.write_rows(info, slot, 0, self.rows, blocks)
    }

    /// Repacks rows `[r0, r0 + rows)` of one expert (exactly their GGUF
    /// blocks) into those rows of `slot`.
    pub(crate) fn write_rows(
        &self,
        info: &GgufTensorInfo,
        slot: usize,
        r0: usize,
        rows: usize,
        blocks: &[u8],
    ) -> Result<()> {
        if slot >= self.slots {
            return Err(bad(format!("slot {slot} of {}", self.slots)));
        }
        if info.tensor_type.k_quant_format() != Some(self.format) {
            return Err(bad(format!(
                "'{}' is {}, the slot takes {}",
                info.name,
                info.tensor_type.name(),
                self.format.mlx_mode()
            )));
        }
        let row_bytes = self.format.row_bytes(self.dst.k);
        if rows == 0 || r0.checked_add(rows).is_none_or(|end| end > self.rows) {
            return Err(bad(format!(
                "'{}': rows {r0}+{rows} of a slot of {}",
                info.name, self.rows
            )));
        }
        if blocks.len() != rows * row_bytes {
            return Err(bad(format!(
                "'{}': {} bytes for {rows} rows of {row_bytes}",
                info.name,
                blocks.len()
            )));
        }
        kq_slot::rows_into(info, blocks, slot * self.rows + r0, rows, &self.dst)
    }

    /// Expert `e` of `t`, read and repacked into `slot` in row bands of
    /// about [`BAND_BYTES`], each on a reader thread.
    pub(crate) fn reads(&self, t: &Arc<Located>, e: u32, slot: usize) -> Result<Vec<Read>> {
        self.reads_banded(t, e, slot, BAND_BYTES)
    }

    pub(crate) fn reads_banded(
        &self,
        t: &Arc<Located>,
        e: u32,
        slot: usize,
        band_bytes: u64,
    ) -> Result<Vec<Read>> {
        if slot >= self.slots {
            return Err(bad(format!("slot {slot} of {}", self.slots)));
        }
        let ext = Extent::expert(t.shard, t.data_offset, &t.info, e as usize)?;
        if ext.len != self.file_bytes() {
            return Err(bad(format!(
                "'{}': expert of {} bytes, the slot takes {}",
                t.info.name,
                ext.len,
                self.file_bytes()
            )));
        }
        let row_bytes = self.format.row_bytes(self.dst.k);
        let ranges = band_ranges(ext.offset, self.rows, row_bytes, band_bytes);
        let mut out = Vec::with_capacity(ranges.len());
        for (r0, rows) in ranges {
            let band = Extent {
                shard: ext.shard,
                offset: ext.offset + r0 as u64 * row_bytes as u64,
                len: rows as u64 * row_bytes as u64,
            };
            let (dst, t) = (*self, Arc::clone(t));
            out.push(Read::repack(
                band,
                Arc::new(move |b: &[u8]| dst.write_rows(&t.info, slot, r0, rows, b)),
            ));
        }
        Ok(out)
    }
}

/// Exact BM=32 tile count and upper slot bound (`max_slot + 1`) for an
/// ascending slice of bank slot ids.
pub fn sorted_slot_bounds(sorted_slots: &[u32]) -> (i32, i32) {
    if sorted_slots.is_empty() {
        return (0, 0);
    }
    const BM: usize = 32;
    let mut tiles = 0usize;
    let mut run = 0usize;
    let mut prev = sorted_slots[0];
    for &s in sorted_slots {
        debug_assert!(s >= prev, "sorted_slot_bounds requires ascending slots");
        if s == prev {
            run += 1;
        } else {
            tiles += run.div_ceil(BM);
            prev = s;
            run = 1;
        }
    }
    tiles += run.div_ceil(BM);
    let max_group = (prev as usize + 1).min(i32::MAX as usize) as i32;
    let y_tiles = tiles.min(i32::MAX as usize) as i32;
    (y_tiles, max_group)
}

/// One projection's bank arrays and their K-quant mode.
#[derive(Clone)]
pub struct ProjArrays {
    pub w: MxArray,
    pub s: MxArray,
    pub b: MxArray,
    pub format: KQuantFormat,
}

/// One projection mapped read-only directly from a GGUF shard.
#[derive(Clone)]
pub struct MmapProj {
    pub buf: MxArray,
    pub base_offset: u64,
    pub format: KQuantFormat,
    pub rows: usize,
    pub k: usize,
}

/// Resident per-layer expert weights served without staging or cache lookup.
#[derive(Clone)]
pub enum DirectLayerArrays {
    Repacked([ProjArrays; 3]),
    Mmap([MmapProj; 3]),
}

impl ProjArrays {
    /// `x [rows, 1, 1, k]` (or `[rows, n, 1, k]`) through the bank slots
    /// `slots [rows, n]` (uint32) -> `[rows, n, 1, out]`, one matvec per
    /// (row, slot).
    pub fn gather(&self, x: &MxArray, slots: &MxArray) -> Result<MxArray> {
        let mode = CString::new(self.format.mlx_mode()).map_err(|e| bad(e.to_string()))?;
        // SAFETY: every handle is a live array owned by the caller for the
        // call; the bridge returns a new array or null on error.
        let h = unsafe {
            mlx_sys::mlx_gather_qmm(
                x.as_raw_ptr(),
                self.w.as_raw_ptr(),
                self.s.as_raw_ptr(),
                self.b.as_raw_ptr(),
                std::ptr::null_mut(),
                slots.as_raw_ptr(),
                true,
                self.format.group_size() as i32,
                self.format.bits() as i32,
                mode.as_ptr(),
                false,
            )
        };
        MxArray::from_handle(h, "hy_v4_flash gather_qmm")
    }

    /// `x [b, 1, k]` through the ascending bank slots `slots [b]` (uint32)
    /// -> `[b, 1, out]`: the sorted-rhs kernel, which reads a slot's weights
    /// once per run of rows on it.
    pub fn gather_sorted(&self, x: &MxArray, slots: &MxArray) -> Result<MxArray> {
        let mode = CString::new(self.format.mlx_mode()).map_err(|e| bad(e.to_string()))?;
        // SAFETY: every handle is a live array owned by the caller for the
        // call; the bridge returns a new array or null on error.
        let h = unsafe {
            mlx_sys::mlx_kquant_gather_qmm_sorted(
                x.as_raw_ptr(),
                self.w.as_raw_ptr(),
                self.s.as_raw_ptr(),
                self.b.as_raw_ptr(),
                slots.as_raw_ptr(),
                self.format.group_size() as i32,
                self.format.bits() as i32,
                mode.as_ptr(),
            )
        };
        MxArray::from_handle(h, "hy_v4_flash sorted gather_qmm")
    }

    /// Like `gather_sorted`, enabling per-op TF32 NAX on F32 activations
    /// unless `MLX_HY4_PREFILL_NAX=0`.
    pub fn gather_sorted_nax(&self, x: &MxArray, slots: &MxArray) -> Result<MxArray> {
        self.gather_sorted_nax_bounded(x, slots, 0, 0)
    }

    /// Like `gather_sorted_nax`, bounding the NAX grid and offset scan using
    /// CPU-known `max_y_tiles` (for fixed BM=32) and `max_group` (`max_slot + 1`).
    pub fn gather_sorted_nax_bounded(
        &self,
        x: &MxArray,
        slots: &MxArray,
        max_y_tiles: i32,
        max_group: i32,
    ) -> Result<MxArray> {
        if !prefill_nax_enabled() {
            return self.gather_sorted(x, slots);
        }
        let mode = CString::new(self.format.mlx_mode()).map_err(|e| bad(e.to_string()))?;
        // SAFETY: every handle is a live array owned by the caller for the
        // call; the bridge returns a new array or null on error.
        let h = unsafe {
            mlx_sys::mlx_kquant_gather_qmm_sorted_nax(
                x.as_raw_ptr(),
                self.w.as_raw_ptr(),
                self.s.as_raw_ptr(),
                self.b.as_raw_ptr(),
                slots.as_raw_ptr(),
                self.format.group_size() as i32,
                self.format.bits() as i32,
                mode.as_ptr(),
                max_y_tiles,
                max_group,
            )
        };
        MxArray::from_handle(h, "hy_v4_flash sorted gather_qmm nax")
    }
}

/// Where one layer step's experts are: `slots[i]` is the bank slot of
/// `experts[i]` in `arrays` (gate, up, down). `reading` (sorted) lists
/// experts whose slots are still being read and must not be gathered before
/// their pending reads finish.
pub struct Prepared {
    pub layer: usize,
    pub class: ExpertClass,
    pub experts: Vec<u32>,
    pub slots: Vec<u32>,
    pub arrays: [ProjArrays; 3],
    pub reading: Vec<u32>,
}

impl Prepared {
    pub fn slot(&self, e: u32) -> Option<u32> {
        self.experts.binary_search(&e).ok().map(|i| self.slots[i])
    }

    pub fn is_reading(&self, e: u32) -> bool {
        self.reading.binary_search(&e).is_ok()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KqStats {
    /// Layer steps that read anything.
    pub fills: usize,
    /// Experts read (all projections of one expert counted once).
    pub slots: usize,
    pub bytes: u64,
    /// Wall time of the reads, submit to last done.
    pub ns: u64,
    /// Summed over reads.
    pub read_ns: u64,
    pub repack_ns: u64,
    /// Expert uses (distinct experts per layer step).
    pub uses: usize,
    pub hits: usize,
    /// Staging copies moved into cache slots (no read).
    pub promoted: usize,
    pub promote_ns: u64,
    /// Steps that drained the GPU before a staging refill.
    pub drains: usize,
    /// Wall time the caller blocked waiting for reads.
    pub wait_ns: u64,
    pub split_steps: usize,
    pub split_mixed: usize,
    pub prefetch_slots: usize,
    pub prefetch_unused: usize,
}

/// In-flight reads for one `prepare_split` step. Dropping unfinished blocks on
/// the reads and abandons uncommitted victims so no IO outlives the step or
/// leaks cache slots.
pub struct Pending<'a> {
    bank: &'a KqBank,
    ci: usize,
    local: usize,
    plan: Plan,
    ticket: Option<Ticket>,
}

impl Pending<'_> {
    pub fn finish(mut self) -> Result<()> {
        let Some(ticket) = self.ticket.take() else {
            return Ok(());
        };
        let t = Instant::now();
        let io = ticket.wait();
        let waited = t.elapsed().as_nanos() as u64;
        let mut g = lock(&self.bank.inner);
        let io: IoStats = match io {
            Ok(io) => io,
            Err(err) => {
                g.classes[self.ci].dir.abandon(&self.plan);
                return Err(err);
            }
        };
        g.classes[self.ci].dir.publish(self.local, &self.plan);
        drop(g);
        let mut st = lock(&self.bank.stats);
        let s = &mut st[self.ci];
        s.fills += 1;
        s.slots += self.plan.reads();
        s.bytes += io.bytes;
        s.ns += io.ns;
        s.read_ns += io.read_ns;
        s.repack_ns += io.repack_ns;
        s.wait_ns += waited;
        Ok(())
    }
}

impl Drop for Pending<'_> {
    fn drop(&mut self) {
        if let Some(ticket) = self.ticket.take() {
            drop(ticket);
            let mut g = lock(&self.bank.inner);
            g.classes[self.ci].dir.abandon(&self.plan);
        }
    }
}

#[derive(Clone, Copy)]
enum MissTarget {
    Staging(u32),
    Victim(u32, u32),
}

/// Overlapped read schedule for one prefill FFN chunk's missing experts.
/// Dropping early waits on the active read batch and releases every
/// uncommitted cache victim so free slots never leak.
pub struct PrefillSubgroups<'a> {
    bank: &'a KqBank,
    ci: usize,
    local: usize,
    active: Option<(Vec<MissTarget>, Ticket)>,
    queued: VecDeque<Vec<MissTarget>>,
}

impl PrefillSubgroups<'_> {
    /// Waits for the active subgroup's reads, publishes its slots, submits the
    /// next subgroup at high priority, and returns the just-published experts.
    pub fn next_subgroup(&mut self) -> Result<Option<Vec<u32>>> {
        let Some((targets, ticket)) = self.active.take() else {
            return Ok(None);
        };
        let t = Instant::now();
        let io = ticket.wait();
        let waited = t.elapsed().as_nanos() as u64;
        let mut g = lock(&self.bank.inner);
        let io: IoStats = match io {
            Ok(io) => io,
            Err(err) => {
                let cache = &mut g.classes[self.ci].dir.cache;
                release_targets(cache, &targets);
                for group in self.queued.drain(..) {
                    release_targets(cache, &group);
                }
                return Err(err);
            }
        };
        let c = &mut g.classes[self.ci];
        let mut published = Vec::with_capacity(targets.len());
        for &tgt in &targets {
            match tgt {
                MissTarget::Staging(e) => {
                    let s = c.dir.stage(self.local, e);
                    c.dir.tags[s] = Some((self.local, e));
                    published.push(e);
                }
                MissTarget::Victim(e, slot) => {
                    c.dir.cache.commit(slot, self.local, e);
                    published.push(e);
                }
            }
        }
        {
            let mut st = lock(&self.bank.stats);
            let s = &mut st[self.ci];
            s.fills += 1;
            s.slots += targets.len();
            s.bytes += io.bytes;
            s.ns += io.ns;
            s.read_ns += io.read_ns;
            s.repack_ns += io.repack_ns;
            s.wait_ns += waited;
        }
        if let Some(next_targets) = self.queued.pop_front() {
            let reads = match g.classes[self.ci].reads_targets(self.local, &next_targets) {
                Ok(r) => r,
                Err(err) => {
                    let cache = &mut g.classes[self.ci].dir.cache;
                    release_targets(cache, &next_targets);
                    for group in self.queued.drain(..) {
                        release_targets(cache, &group);
                    }
                    return Err(err);
                }
            };
            // SAFETY: target staging slots were untagged upfront, victims are
            // unmapped, GPU drain (if any) ran before subgroup 0, and `inflight`
            // plus `self.active` keep the batch watched until completion.
            let next_ticket = match unsafe { self.bank.pool.submit(reads, Priority::High) } {
                Ok(t) => t,
                Err(err) => {
                    let cache = &mut g.classes[self.ci].dir.cache;
                    release_targets(cache, &next_targets);
                    for group in self.queued.drain(..) {
                        release_targets(cache, &group);
                    }
                    return Err(err);
                }
            };
            g.watch(&next_ticket);
            self.active = Some((next_targets, next_ticket));
        }
        Ok(Some(published))
    }
}

impl Drop for PrefillSubgroups<'_> {
    fn drop(&mut self) {
        let active_targets = self.active.take().map(|(targets, ticket)| {
            drop(ticket);
            targets
        });
        if active_targets.is_some() || !self.queued.is_empty() {
            let mut g = lock(&self.bank.inner);
            let cache = &mut g.classes[self.ci].dir.cache;
            if let Some(targets) = active_targets {
                release_targets(cache, &targets);
            }
            for group in self.queued.drain(..) {
                release_targets(cache, &group);
            }
        }
    }
}

fn release_targets(cache: &mut Cache, targets: &[MissTarget]) {
    for &tgt in targets {
        if let MissTarget::Victim(_, slot) = tgt {
            cache.release(slot);
        }
    }
}

struct ProjBank {
    w: PurgeableBank,
    s: PurgeableBank,
    b: PurgeableBank,
    dst: SlotDst,
}

impl ProjBank {
    fn banks(&self) -> [&PurgeableBank; 3] {
        [&self.w, &self.s, &self.b]
    }

    fn arrays(&self) -> ProjArrays {
        ProjArrays {
            w: self.w.array().clone(),
            s: self.s.array().clone(),
            b: self.b.array().clone(),
            format: self.dst.format,
        }
    }
}

fn alloc(geo: &[ProjGeo; 3], slots: usize) -> Result<Vec<ProjBank>> {
    let mut out = Vec::with_capacity(3);
    for g in geo {
        let shape = |cols: usize| [slots as i64, g.rows as i64, cols as i64];
        let w = PurgeableBank::new(&shape(g.weight_cols()), DType::Uint32)?;
        let s = PurgeableBank::new(&shape(g.scales_cols()), g.scales_dtype())?;
        let b = PurgeableBank::new(&shape(g.biases_cols()), DType::Float16)?;
        let ptrs = DstPtrs {
            weight: w.data_ptr(),
            scales: Some(s.data_ptr()),
            biases: Some(b.data_ptr()),
            rows: slots * g.rows,
            k: g.k,
            layout: OutLayout::RowMajor,
        };
        // SAFETY: the three banks hold `slots * rows` rows of `g`'s K-quant
        // arrays and live as long as `dst` (both in `ProjBank`); slots are
        // written only by reads that `prepare` waits for under the bank lock,
        // into slots no unfinished gather reads.
        let dst = unsafe { SlotDst::new(ptrs, g.format, g.rows) }?;
        out.push(ProjBank { w, s, b, dst });
    }
    Ok(out)
}

/// Copies bank slot `from` into slot `to` in every projection. The caller
/// holds the bank lock, `from` holds its expert and no unfinished gather
/// reads `to`.
fn copy_slot(proj: &[ProjBank], total: usize, from: u32, to: u32) -> Result<()> {
    let (from, to) = (from as usize, to as usize);
    if from >= total || to >= total || from == to {
        return Err(bad(format!("copy of slot {from} to {to} of {total}")));
    }
    for p in proj {
        for b in p.banks() {
            let sb = b.nbytes() / total;
            // SAFETY: both ranges lie inside the bank (`from`, `to` < total,
            // `sb` = one slot's bytes) and do not overlap (from != to); no read
            // writes them, no GPU work writes either or reads `to`, and the
            // bank is non-volatile.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    b.data_ptr().0.add(from * sb),
                    b.data_ptr().0.add(to * sb),
                    sb,
                );
            }
        }
    }
    Ok(())
}

struct ClassBank {
    class: ExpertClass,
    geo: [ProjGeo; 3],
    /// Per class-local layer: gate, up, down.
    tensors: Vec<[Arc<Located>; 3]>,
    /// Empty when the class has no layers or a rebuild failed.
    proj: Vec<ProjBank>,
    dir: Directory,
}

impl ClassBank {
    fn reads(&self, layer: usize, plan: &Plan) -> Result<Vec<Read>> {
        let base = self.dir.base() as usize;
        let mut out = Vec::with_capacity(plan.reads() * 12);
        let targets = plan
            .staging
            .iter()
            .map(|&e| (e, self.dir.stage(layer, e)))
            .chain(plan.victims.iter().map(|&(e, c)| (e, base + c as usize)));
        for (e, slot) in targets {
            for (t, p) in self.tensors[layer].iter().zip(&self.proj) {
                out.extend(p.dst.reads(t, e, slot)?);
            }
        }
        sort_reads(&mut out);
        Ok(out)
    }

    fn reads_targets(&self, layer: usize, targets: &[MissTarget]) -> Result<Vec<Read>> {
        let base = self.dir.base() as usize;
        let mut out = Vec::with_capacity(targets.len() * 12);
        for &tgt in targets {
            let (e, slot) = match tgt {
                MissTarget::Staging(e) => (e, self.dir.stage(layer, e)),
                MissTarget::Victim(e, c) => (e, base + c as usize),
            };
            for (t, p) in self.tensors[layer].iter().zip(&self.proj) {
                out.extend(p.dst.reads(t, e, slot)?);
            }
        }
        sort_reads(&mut out);
        Ok(out)
    }

    fn arrays(&self) -> Result<[ProjArrays; 3]> {
        match self.proj.as_slice() {
            [g, u, d] => Ok([g.arrays(), u.arrays(), d.arrays()]),
            _ => Err(bad(format!("{:?}: no bank (a rebuild failed)", self.class))),
        }
    }
}

struct Ahead {
    local: usize,
    experts: Vec<u32>,
    ticket: Ticket,
}

struct Inner {
    // Declared before `classes` so Drop waits for all in-flight reads before
    // any PurgeableBank buffer is released.
    ahead: [Vec<Ahead>; 2],
    inflight: Vec<BatchWatch>,
    ahead_left: [Vec<(usize, Vec<u32>)>; 2],
    classes: Vec<ClassBank>,
    in_turn: bool,
    /// Buffers of banks dropped by `rebuild` that an array may still hold.
    retired: Vec<BankWatch>,
}

impl Inner {
    fn drain(&mut self) {
        for w in self.inflight.drain(..) {
            w.wait();
        }
    }

    fn watch(&mut self, t: &Ticket) {
        self.inflight.retain(|w| !w.done());
        self.inflight.push(t.watch());
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        for list in &mut self.ahead {
            list.clear();
        }
        self.drain();
    }
}

const DIRECT_FILL_BATCH_EXPERTS: usize = 32;

struct DirectFillProgress {
    filled_layers: usize,
    error: Option<String>,
    stopped: bool,
}

struct DirectFillState {
    mu: Mutex<DirectFillProgress>,
    cv: Condvar,
}

struct DirectFillHandle {
    stop: Arc<AtomicBool>,
    state: Arc<DirectFillState>,
    worker: Option<JoinHandle<()>>,
}

impl Drop for DirectFillHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        {
            let mut st = lock(&self.state.mu);
            st.stopped = true;
            self.state.cv.notify_all();
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

struct DirectLayerJob {
    layer: usize,
    tensors: [Arc<Located>; 3],
    dsts: [SlotDst; 3],
    ready: Arc<AtomicBool>,
}

struct DirectLayer {
    ci: usize,
    class: ExpertClass,
    proj: [ProjBank; 3],
    arrays: [ProjArrays; 3],
    ready: Arc<AtomicBool>,
}

struct DirectBank {
    // Declared before `layers` so Drop cancels and joins the fill thread
    // before any direct PurgeableBank buffer is released.
    fill: DirectFillHandle,
    layers: Vec<DirectLayer>,
    layer_to_idx: Vec<Option<usize>>,
    uses: [AtomicUsize; 2],
    slot_bytes: [u64; 2],
    cache_slots: [usize; 2],
    in_turn: AtomicBool,
}

fn run_direct_fill(
    pool: Arc<IoPool>,
    n_experts: usize,
    jobs: Vec<DirectLayerJob>,
    stop: Arc<AtomicBool>,
    state: Arc<DirectFillState>,
) {
    struct ExitGuard {
        done: bool,
        stop: Arc<AtomicBool>,
        state: Arc<DirectFillState>,
    }
    impl Drop for ExitGuard {
        fn drop(&mut self) {
            if !self.done {
                let mut st = lock(&self.state.mu);
                if self.stop.load(Ordering::Acquire) {
                    st.stopped = true;
                } else if st.error.is_none() {
                    st.error = Some("direct expert fill thread exited unexpectedly".into());
                }
                self.state.cv.notify_all();
            }
        }
    }
    let mut guard = ExitGuard {
        done: false,
        stop: Arc::clone(&stop),
        state: Arc::clone(&state),
    };
    let total_layers = jobs.len();
    let t_all = Instant::now();
    for job in jobs {
        if stop.load(Ordering::Acquire) {
            let mut st = lock(&state.mu);
            st.stopped = true;
            state.cv.notify_all();
            guard.done = true;
            return;
        }
        for e0 in (0..n_experts).step_by(DIRECT_FILL_BATCH_EXPERTS) {
            if stop.load(Ordering::Acquire) {
                let mut st = lock(&state.mu);
                st.stopped = true;
                state.cv.notify_all();
                guard.done = true;
                return;
            }
            let e1 = (e0 + DIRECT_FILL_BATCH_EXPERTS).min(n_experts);
            let mut reads = Vec::with_capacity((e1 - e0) * 12);
            for e in e0..e1 {
                for (t, dst) in job.tensors.iter().zip(&job.dsts) {
                    match dst.reads(t, e as u32, e) {
                        Ok(r) => reads.extend(r),
                        Err(err) => {
                            let mut st = lock(&state.mu);
                            st.error =
                                Some(format!("layer {} expert {e}: {}", job.layer, err.reason));
                            state.cv.notify_all();
                            guard.done = true;
                            return;
                        }
                    }
                }
            }
            sort_reads(&mut reads);
            // SAFETY: `DirectFillHandle::drop` joins this thread before
            // `DirectBank::layers` frees any `PurgeableBank`. Each batch writes
            // disjoint slots `e0..e1` before `job.ready` publishes with Release.
            let ticket = match unsafe { pool.submit(reads, Priority::Low) } {
                Ok(t) => t,
                Err(err) => {
                    let mut st = lock(&state.mu);
                    st.error = Some(format!("layer {}: {}", job.layer, err.reason));
                    state.cv.notify_all();
                    guard.done = true;
                    return;
                }
            };
            if let Err(err) = ticket.wait() {
                let mut st = lock(&state.mu);
                if stop.load(Ordering::Acquire) {
                    st.stopped = true;
                } else {
                    st.error = Some(format!("layer {}: {}", job.layer, err.reason));
                }
                state.cv.notify_all();
                guard.done = true;
                return;
            }
        }
        let mut st = lock(&state.mu);
        job.ready.store(true, Ordering::Release);
        st.filled_layers += 1;
        state.cv.notify_all();
    }
    tracing::info!(
        "hy_v4_flash direct expert fill completed {total_layers} layers ({n_experts} experts/layer) in {:.2} s",
        t_all.elapsed().as_secs_f64()
    );
    guard.done = true;
}

impl DirectBank {
    fn wait_layer(&self, layer: usize) -> Result<&DirectLayer> {
        let idx = self
            .layer_to_idx
            .get(layer)
            .copied()
            .flatten()
            .ok_or_else(|| bad(format!("layer {layer} has no direct experts")))?;
        let dl = &self.layers[idx];
        if dl.ready.load(Ordering::Acquire) {
            return Ok(dl);
        }
        let mut st = lock(&self.fill.state.mu);
        loop {
            if dl.ready.load(Ordering::Acquire) {
                return Ok(dl);
            }
            if let Some(err) = &st.error {
                return Err(bad(err.clone()));
            }
            if st.stopped {
                return Err(bad("direct expert fill was stopped"));
            }
            st = self
                .fill
                .state
                .cv
                .wait(st)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn wait_all(&self) -> Result<()> {
        let total = self.layers.len();
        let mut st = lock(&self.fill.state.mu);
        loop {
            if st.filled_layers >= total {
                return Ok(());
            }
            if let Some(err) = &st.error {
                return Err(bad(err.clone()));
            }
            if st.stopped {
                return Err(bad("direct expert fill was stopped"));
            }
            st = self
                .fill
                .state
                .cv
                .wait(st)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn fill_complete(&self) -> bool {
        lock(&self.fill.state.mu).filled_layers >= self.layers.len()
    }
}

fn check_direct_step(n_experts: usize, experts: &[u32], uses: &[u32]) -> Result<()> {
    if let Some(&e) = experts.iter().find(|&&e| e as usize >= n_experts) {
        return Err(bad(format!("expert {e} of {n_experts}")));
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

#[cfg(test)]
thread_local! {
    /// Forces `build_direct_bank` to fail partway through layer allocation on
    /// this thread so error-recovery tests can verify partial bank restoration.
    pub(super) static TEST_FAIL_DIRECT_ALLOC_AT: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

fn build_direct_bank(
    pool: &Arc<IoPool>,
    n_experts: usize,
    layout: &ClassLayout,
    geos: &[[ProjGeo; 3]; 2],
    mut layer_tensors: impl FnMut(usize, usize, usize) -> Result<[Arc<Located>; 3]>,
    in_turn: bool,
) -> Result<DirectBank> {
    if n_experts == 0 || u32::try_from(n_experts).is_err() {
        return Err(bad(format!("invalid direct expert count {n_experts}")));
    }
    let slot_bytes = [
        geos[0].iter().map(ProjGeo::slot_bytes).sum(),
        geos[1].iter().map(ProjGeo::slot_bytes).sum(),
    ];
    let cache_slots = [
        layout.layers[0].len() * n_experts,
        layout.layers[1].len() * n_experts,
    ];
    let n_moe_layers = layout.layers[0].len() + layout.layers[1].len();
    let mut layers = Vec::with_capacity(n_moe_layers);
    let mut jobs = Vec::with_capacity(n_moe_layers);
    let mut layer_to_idx = vec![None; layout.of_layer.len()];
    for (il, opt) in layout.of_layer.iter().enumerate() {
        let Some((ci, local)) = *opt else {
            continue;
        };
        #[cfg(test)]
        if TEST_FAIL_DIRECT_ALLOC_AT.with(std::cell::Cell::get) == Some(layers.len()) {
            return Err(bad(format!(
                "injected direct allocation failure at layer {il}"
            )));
        }
        let class = CLASSES[ci];
        let geo = &geos[ci];
        let tensors = layer_tensors(il, ci, local)?;
        let proj_vec = alloc(geo, n_experts)?;
        let proj: [ProjBank; 3] = proj_vec
            .try_into()
            .map_err(|_| bad("expected 3 expert projections"))?;
        let arrays = [proj[0].arrays(), proj[1].arrays(), proj[2].arrays()];
        let dsts = [proj[0].dst, proj[1].dst, proj[2].dst];
        let ready = Arc::new(AtomicBool::new(false));
        let idx = layers.len();
        layer_to_idx[il] = Some(idx);
        layers.push(DirectLayer {
            ci,
            class,
            proj,
            arrays,
            ready: Arc::clone(&ready),
        });
        jobs.push(DirectLayerJob {
            layer: il,
            tensors,
            dsts,
            ready,
        });
    }
    let stop = Arc::new(AtomicBool::new(false));
    let state = Arc::new(DirectFillState {
        mu: Mutex::new(DirectFillProgress {
            filled_layers: 0,
            error: None,
            stopped: false,
        }),
        cv: Condvar::new(),
    });
    let worker = if jobs.is_empty() {
        None
    } else {
        let p = Arc::clone(pool);
        let s = Arc::clone(&stop);
        let st = Arc::clone(&state);
        Some(
            std::thread::Builder::new()
                .name("hy4-direct-fill".into())
                .spawn(move || run_direct_fill(p, n_experts, jobs, s, st))
                .map_err(|e| bad(format!("spawn direct fill thread: {e}")))?,
        )
    };
    Ok(DirectBank {
        fill: DirectFillHandle {
            stop,
            state,
            worker,
        },
        layers,
        layer_to_idx,
        uses: [AtomicUsize::new(0), AtomicUsize::new(0)],
        slot_bytes,
        cache_slots,
        in_turn: AtomicBool::new(in_turn),
    })
}

const MMAP_PREWARM_CHUNK: usize = 4 << 20;
const MMAP_PREWARM_MERGE_GAP: u64 = 4096;

#[cfg(unix)]
struct MmapShard {
    _file: std::fs::File,
    ptr: *mut u8,
    len: usize,
    mapped_len: usize,
    file_len: u64,
}

#[cfg(unix)]
// SAFETY: the mapping is `PROT_READ` and immutable for its lifetime; `munmap`
// runs only after the last `Arc<MmapShard>` (including all MLX buffers) drops.
unsafe impl Send for MmapShard {}
#[cfg(unix)]
unsafe impl Sync for MmapShard {}

#[cfg(unix)]
impl MmapShard {
    fn open(path: &std::path::Path, page: usize) -> Result<Self> {
        use std::os::unix::io::AsRawFd;

        let file = std::fs::File::open(path)
            .map_err(|e| bad(format!("open '{}' for mmap: {e}", path.display())))?;
        let file_len = file
            .metadata()
            .map_err(|e| bad(format!("stat '{}' for mmap: {e}", path.display())))?
            .len();
        if file_len == 0 {
            return Err(bad(format!("cannot mmap empty shard '{}'", path.display())));
        }
        let len = usize::try_from(file_len)
            .map_err(|_| bad(format!("shard '{}' length overflows usize", path.display())))?;
        let mapped_len = len
            .checked_add(page - 1)
            .map(|n| n & !(page - 1))
            .ok_or_else(|| {
                bad(format!(
                    "shard '{}' mapped length overflows",
                    path.display()
                ))
            })?;
        // SAFETY: `file` is an open read-only descriptor with positive `len`.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(bad(format!(
                "mmap '{}' ({file_len} bytes): {}",
                path.display(),
                std::io::Error::last_os_error()
            )));
        }
        Ok(Self {
            _file: file,
            ptr: ptr.cast::<u8>(),
            len,
            mapped_len,
            file_len,
        })
    }
}

#[cfg(unix)]
impl Drop for MmapShard {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` and `self.len` come from a successful `mmap` and
        // every MLX view holding an `Arc<MmapShard>` has already dropped.
        unsafe {
            libc::munmap(self.ptr.cast(), self.len);
        }
    }
}

#[cfg(unix)]
unsafe extern "C" fn release_mmap_shard(ctx: *mut std::ffi::c_void) {
    // SAFETY: `ctx` was produced by `Arc::into_raw` for `MmapShard` when
    // creating the MLX buffer; MLX invokes the deleter exactly once.
    unsafe {
        Arc::decrement_strong_count(ctx.cast::<MmapShard>());
    }
}

#[cfg(not(unix))]
struct MmapShard;

#[cfg(unix)]
fn os_page_size() -> Result<usize> {
    // SAFETY: `sysconf(_SC_PAGESIZE)` has no preconditions.
    let raw = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let page = usize::try_from(raw).unwrap_or(0);
    if page == 0 || !page.is_power_of_two() {
        return Err(bad(format!("invalid OS page size {raw}")));
    }
    Ok(page)
}

#[cfg(target_os = "macos")]
fn set_background_io_qos() {
    const QOS_CLASS_BACKGROUND: libc::c_uint = 0x09;
    const IOPOL_TYPE_DISK: libc::c_int = 0;
    const IOPOL_SCOPE_THREAD: libc::c_int = 1;
    const IOPOL_THROTTLE: libc::c_int = 3;
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(
            qos_class: libc::c_uint,
            relative_priority: libc::c_int,
        ) -> libc::c_int;
        fn setiopolicy_np(
            iotype: libc::c_int,
            scope: libc::c_int,
            policy: libc::c_int,
        ) -> libc::c_int;
    }
    // SAFETY: best-effort QoS/IO-priority hints affecting only the current thread.
    unsafe {
        let _ = pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);
        let _ = setiopolicy_np(IOPOL_TYPE_DISK, IOPOL_SCOPE_THREAD, IOPOL_THROTTLE);
    }
}

#[cfg(not(target_os = "macos"))]
fn set_background_io_qos() {}

struct MmapWarmHandle {
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl Drop for MmapWarmHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(unix)]
fn run_mmap_prewarm(
    spans: Vec<(PathBuf, u64, u64)>,
    budget: u64,
    stop: Arc<AtomicBool>,
    in_turn: Arc<AtomicBool>,
) {
    use std::os::unix::fs::FileExt;

    set_background_io_qos();
    let t0 = Instant::now();
    let mut buf = vec![0u8; MMAP_PREWARM_CHUNK];
    let mut remaining = budget;
    let mut warmed = 0u64;
    let mut open_path: Option<(PathBuf, std::fs::File)> = None;

    for (path, start, end) in spans {
        if remaining == 0 || stop.load(Ordering::Acquire) {
            break;
        }
        let reopen = open_path.as_ref().is_none_or(|(p, _)| p != &path);
        if reopen {
            match std::fs::File::open(&path) {
                Ok(f) => open_path = Some((path.clone(), f)),
                Err(e) => {
                    tracing::warn!("hy_v4_flash mmap prewarm open '{}': {e}", path.display());
                    continue;
                }
            }
        }
        let Some((_, file)) = open_path.as_ref() else {
            continue;
        };
        let mut off = start;
        while off < end && remaining > 0 {
            if stop.load(Ordering::Acquire) {
                return;
            }
            while in_turn.load(Ordering::Acquire) {
                if stop.load(Ordering::Acquire) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            let want = (end - off).min(remaining).min(buf.len() as u64) as usize;
            match file.read_at(&mut buf[..want], off) {
                Ok(0) => break,
                Ok(n) => {
                    off += n as u64;
                    warmed += n as u64;
                    remaining = remaining.saturating_sub(n as u64);
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    tracing::warn!("hy_v4_flash mmap prewarm read '{}': {e}", path.display());
                    return;
                }
            }
        }
    }
    if warmed > 0 {
        tracing::info!(
            "hy_v4_flash mmap prewarm warmed {:.2} GiB in {:.2} s",
            warmed as f64 / (1u64 << 30) as f64,
            t0.elapsed().as_secs_f64()
        );
    }
}

struct MmapLayer {
    ci: usize,
    projs: [MmapProj; 3],
}

struct MmapExpertBank {
    // Drop order is load-bearing: join pre-warm, release Metal residency set,
    // drop MTL::Buffer arrays (`layers`), then release shard Arcs (`_shards`).
    _warm: Option<MmapWarmHandle>,
    rset: Option<MetalResidencySet>,
    layers: Vec<MmapLayer>,
    _shards: Vec<Arc<MmapShard>>,
    layer_to_idx: Vec<Option<usize>>,
    uses: [AtomicUsize; 2],
    slot_bytes: [u64; 2],
    cache_slots: [usize; 2],
    mapped_bytes: u64,
    in_turn: Arc<AtomicBool>,
}

impl MmapExpertBank {
    fn set_in_turn(&self, active: bool) {
        self.in_turn.store(active, Ordering::Release);
        if active && let Some(rset) = &self.rset {
            rset.request();
        }
    }

    #[cfg(unix)]
    fn new(
        n_experts: usize,
        layout: &ClassLayout,
        geos: &[[ProjGeo; 3]; 2],
        experts: &[Option<ExpertLayer>],
        paths: &[PathBuf],
        warm_budget_bytes: u64,
    ) -> Result<Self> {
        if n_experts == 0 || u32::try_from(n_experts).is_err() {
            return Err(bad(format!("invalid mmap expert count {n_experts}")));
        }
        let page = os_page_size()?;
        let page_u64 = page as u64;
        let slot_bytes = [
            geos[0].iter().map(ProjGeo::file_bytes).sum(),
            geos[1].iter().map(ProjGeo::file_bytes).sum(),
        ];
        let cache_slots = [
            layout.layers[0].len() * n_experts,
            layout.layers[1].len() * n_experts,
        ];
        let n_moe_layers = layout.layers[0].len() + layout.layers[1].len();
        let mut shards: Vec<Arc<MmapShard>> = Vec::new();
        let mut shard_map: HashMap<usize, usize> = HashMap::new();
        let mut layers = Vec::with_capacity(n_moe_layers);
        let mut layer_to_idx = vec![None; layout.of_layer.len()];
        let mut raw_spans: Vec<(usize, u64, u64)> = Vec::with_capacity(n_moe_layers * 3);
        let mut mapped_bytes = 0u64;

        for (il, opt) in layout.of_layer.iter().enumerate() {
            let Some((ci, _)) = *opt else {
                continue;
            };
            let class = CLASSES[ci];
            let geo = &geos[ci];
            let exp = experts[il]
                .as_ref()
                .ok_or_else(|| bad(format!("layer {il} has no experts")))?;
            let tensors = check_layer(il, exp, class, geo, n_experts)?;
            let mut projs_vec = Vec::with_capacity(3);
            for (loc, g) in tensors.iter().zip(geo) {
                let shard_slot = match shard_map.get(&loc.shard) {
                    Some(&idx) => idx,
                    None => {
                        let path = paths
                            .get(loc.shard)
                            .ok_or_else(|| bad(format!("shard {} path missing", loc.shard)))?;
                        let opened = Arc::new(MmapShard::open(path, page)?);
                        let idx = shards.len();
                        shards.push(opened);
                        shard_map.insert(loc.shard, idx);
                        idx
                    }
                };
                let shard = &shards[shard_slot];
                let abs_off = loc.abs_offset()?;
                let tensor_bytes = loc.info.data_size()?;
                let end_off = abs_off
                    .checked_add(tensor_bytes)
                    .ok_or_else(|| bad(format!("'{}' end offset overflows", loc.info.name)))?;
                if end_off > shard.file_len {
                    return Err(bad(format!(
                        "'{}' ends at {end_off}, past shard length {}",
                        loc.info.name, shard.file_len
                    )));
                }
                let aligned_off = abs_off & !(page_u64 - 1);
                let intra_off = abs_off - aligned_off;
                let min_align = match g.format {
                    KQuantFormat::Q6K => 2u64,
                    _ => 4u64,
                };
                if !intra_off.is_multiple_of(min_align) {
                    return Err(bad(format!(
                        "'{}' offset {abs_off} is not {min_align}-byte aligned for {:?}",
                        loc.info.name, g.format
                    )));
                }
                let aligned_len_u64 = intra_off
                    .checked_add(tensor_bytes)
                    .and_then(|n| n.checked_add(page_u64 - 1))
                    .map(|n| n & !(page_u64 - 1))
                    .ok_or_else(|| bad(format!("'{}' aligned length overflows", loc.info.name)))?;
                let aligned_end = aligned_off
                    .checked_add(aligned_len_u64)
                    .ok_or_else(|| bad(format!("'{}' aligned end overflows", loc.info.name)))?;
                if aligned_end > shard.mapped_len as u64 {
                    return Err(bad(format!(
                        "'{}' aligned range ends at {aligned_end}, past mapping {}",
                        loc.info.name, shard.mapped_len
                    )));
                }
                let aligned_off_usize = usize::try_from(aligned_off)
                    .map_err(|_| bad(format!("'{}' offset overflows usize", loc.info.name)))?;
                let aligned_len = usize::try_from(aligned_len_u64)
                    .map_err(|_| bad(format!("'{}' length overflows usize", loc.info.name)))?;
                // SAFETY: `aligned_off + aligned_len <= shard.mapped_len`.
                let aligned_ptr = unsafe { shard.ptr.add(aligned_off_usize) };
                let raw_ctx = Arc::into_raw(Arc::clone(shard)) as *mut std::ffi::c_void;
                // SAFETY: `shard` owns a read-only `MAP_SHARED` mapping that stays
                // alive until `release_mmap_shard` decrements the refcount passed in
                // `raw_ctx`; on `Err`, `from_mmap_readonly` never calls `release_fn`.
                let buf = match unsafe {
                    MxArray::from_mmap_readonly(
                        aligned_ptr.cast(),
                        aligned_len,
                        release_mmap_shard,
                        raw_ctx,
                    )
                } {
                    Ok(arr) => arr,
                    Err(err) => {
                        // SAFETY: `raw_ctx` ownership was not taken on `Err`.
                        unsafe { drop(Arc::from_raw(raw_ctx.cast::<MmapShard>())) };
                        return Err(err);
                    }
                };
                mapped_bytes = mapped_bytes
                    .checked_add(tensor_bytes)
                    .ok_or_else(|| bad("mmap expert bytes overflow u64"))?;
                raw_spans.push((loc.shard, abs_off, end_off));
                projs_vec.push(MmapProj {
                    buf,
                    base_offset: intra_off,
                    format: g.format,
                    rows: g.rows,
                    k: g.k,
                });
            }
            let projs: [MmapProj; 3] = projs_vec
                .try_into()
                .map_err(|_| bad("expected 3 mmap expert projections"))?;
            let idx = layers.len();
            layer_to_idx[il] = Some(idx);
            layers.push(MmapLayer { ci, projs });
        }

        let in_turn = Arc::new(AtomicBool::new(false));
        let warm = if warm_budget_bytes > 0 && !raw_spans.is_empty() {
            raw_spans.sort_unstable_by_key(|&(s, start, _)| (s, start));
            let mut merged: Vec<(PathBuf, u64, u64)> = Vec::new();
            for (s, start, end) in raw_spans {
                let path = &paths[s];
                if let Some((prev_p, _, prev_end)) = merged.last_mut()
                    && prev_p == path
                    && start <= prev_end.saturating_add(MMAP_PREWARM_MERGE_GAP)
                {
                    *prev_end = (*prev_end).max(end);
                } else {
                    merged.push((path.clone(), start, end));
                }
            }
            let budget = warm_budget_bytes.min(mapped_bytes);
            let stop = Arc::new(AtomicBool::new(false));
            let stop_c = Arc::clone(&stop);
            let turn_c = Arc::clone(&in_turn);
            let worker = std::thread::Builder::new()
                .name("hy4-mmap-prewarm".into())
                .spawn(move || run_mmap_prewarm(merged, budget, stop_c, turn_c))
                .map_err(|e| bad(format!("spawn mmap prewarm thread: {e}")))?;
            Some(MmapWarmHandle {
                stop,
                worker: Some(worker),
            })
        } else {
            None
        };

        let buf_refs: Vec<&MxArray> = layers
            .iter()
            .flat_map(|l| l.projs.iter().map(|p| &p.buf))
            .collect();
        let rset = MetalResidencySet::create(&buf_refs);

        Ok(Self {
            _warm: warm,
            rset,
            layers,
            _shards: shards,
            layer_to_idx,
            uses: [AtomicUsize::new(0), AtomicUsize::new(0)],
            slot_bytes,
            cache_slots,
            mapped_bytes,
            in_turn,
        })
    }

    #[cfg(not(unix))]
    fn new(
        _n_experts: usize,
        _layout: &ClassLayout,
        _geos: &[[ProjGeo; 3]; 2],
        _experts: &[Option<ExpertLayer>],
        _paths: &[PathBuf],
        _warm_budget_bytes: u64,
    ) -> Result<Self> {
        Err(bad("mmap expert bank requires Unix"))
    }
}

enum ResidentBank {
    Direct(DirectBank),
    Mmap(MmapExpertBank),
}

pub struct KqBank {
    // Declared before `inner` and `pool` so direct fill, mmap pre-warm, and
    // streamed in-flight reads finish before `pool` shuts down and before any
    // PurgeableBank or mmap shard frees.
    direct: OnceLock<ResidentBank>,
    inner: Mutex<Inner>,
    pool: Arc<IoPool>,
    n_experts: usize,
    n_used: usize,
    layout: ClassLayout,
    split: AtomicBool,
    prefetch: AtomicBool,
    decode_prefetch: AtomicBool,
    stats: Mutex<[KqStats; 2]>,
}

impl KqBank {
    /// Banks for the trunk layers `experts` (`Scan::trunk_experts`), with
    /// `staging` staging layers and `cache_slots[c]` cache slots for class
    /// `CLASSES[c]`.
    pub fn new(
        pool: IoPool,
        moe: &MoeConfig,
        experts: &[Option<ExpertLayer>],
        cache_slots: [usize; 2],
        staging: usize,
    ) -> Result<Self> {
        let n = moe.n_experts;
        let classes: Vec<Option<ExpertClass>> = experts
            .iter()
            .map(|l| l.as_ref().map(|l| l.class))
            .collect();
        let layout = ClassLayout::new(&classes)?;
        let mut banks = Vec::with_capacity(CLASSES.len());
        for (ci, &class) in CLASSES.iter().enumerate() {
            let geo = geometry(moe, class)?;
            let layers = &layout.layers[ci];
            if layers.is_empty() && cache_slots[ci] > 0 {
                return Err(bad(format!("{class:?} has no layers but cache slots")));
            }
            let tensors = layers
                .iter()
                .map(|&il| match &experts[il] {
                    Some(l) => check_layer(il, l, class, &geo, n),
                    None => Err(bad(format!("layer {il} has no experts"))),
                })
                .collect::<Result<Vec<_>>>()?;
            let dir = make_directory(staging, layers.len(), n, cache_slots[ci], moe.n_used)?;
            let proj = if layers.is_empty() {
                Vec::new()
            } else {
                alloc(&geo, dir.total_slots())?
            };
            banks.push(ClassBank {
                class,
                geo,
                tensors,
                proj,
                dir,
            });
        }
        Ok(Self {
            direct: OnceLock::new(),
            inner: Mutex::new(Inner {
                ahead: [Vec::new(), Vec::new()],
                inflight: Vec::new(),
                ahead_left: [Vec::new(), Vec::new()],
                classes: banks,
                in_turn: true,
                retired: Vec::new(),
            }),
            pool: Arc::new(pool),
            n_experts: n,
            n_used: moe.n_used,
            layout,
            split: AtomicBool::new(true),
            prefetch: AtomicBool::new(true),
            decode_prefetch: AtomicBool::new(decode_prefetch_enabled()),
            stats: Mutex::new([KqStats::default(); 2]),
        })
    }

    /// Direct full-expert residency: allocates one NonVolatile `[n_experts, ...]`
    /// bank per MoE layer (`slot_id == expert_id`) and fills all layers in
    /// ascending layer order on a background thread using `pool`.
    pub fn new_direct(
        pool: IoPool,
        moe: &MoeConfig,
        experts: &[Option<ExpertLayer>],
    ) -> Result<Self> {
        let n = moe.n_experts;
        let classes: Vec<Option<ExpertClass>> = experts
            .iter()
            .map(|l| l.as_ref().map(|l| l.class))
            .collect();
        let layout = ClassLayout::new(&classes)?;
        let geos = [geometry(moe, CLASSES[0])?, geometry(moe, CLASSES[1])?];
        let pool = Arc::new(pool);
        let direct = build_direct_bank(
            &pool,
            n,
            &layout,
            &geos,
            |il, ci, _| match &experts[il] {
                Some(l) => check_layer(il, l, CLASSES[ci], &geos[ci], n),
                None => Err(bad(format!("layer {il} has no experts"))),
            },
            true,
        )?;
        Ok(Self {
            direct: OnceLock::from(ResidentBank::Direct(direct)),
            inner: Mutex::new(Inner {
                ahead: [Vec::new(), Vec::new()],
                inflight: Vec::new(),
                ahead_left: [Vec::new(), Vec::new()],
                classes: Vec::new(),
                in_turn: true,
                retired: Vec::new(),
            }),
            pool,
            n_experts: n,
            n_used: moe.n_used,
            layout,
            split: AtomicBool::new(true),
            prefetch: AtomicBool::new(true),
            decode_prefetch: AtomicBool::new(decode_prefetch_enabled()),
            stats: Mutex::new([KqStats::default(); 2]),
        })
    }

    /// Zero-copy read-only mmap expert bank: maps each shard once and wraps
    /// every trunk expert tensor as a page-aligned Metal buffer with 0 B
    /// anonymous RAM. When `warm_budget_bytes > 0`, spawns a low-priority
    /// background pre-warmer that pauses during active turns.
    pub fn new_mmap(
        pool: IoPool,
        moe: &MoeConfig,
        experts: &[Option<ExpertLayer>],
        paths: &[PathBuf],
        warm_budget_bytes: u64,
    ) -> Result<Self> {
        let n = moe.n_experts;
        let classes: Vec<Option<ExpertClass>> = experts
            .iter()
            .map(|l| l.as_ref().map(|l| l.class))
            .collect();
        let layout = ClassLayout::new(&classes)?;
        let geos = [geometry(moe, CLASSES[0])?, geometry(moe, CLASSES[1])?];
        let mmap = MmapExpertBank::new(n, &layout, &geos, experts, paths, warm_budget_bytes)?;
        Ok(Self {
            direct: OnceLock::from(ResidentBank::Mmap(mmap)),
            inner: Mutex::new(Inner {
                ahead: [Vec::new(), Vec::new()],
                inflight: Vec::new(),
                ahead_left: [Vec::new(), Vec::new()],
                classes: Vec::new(),
                in_turn: false,
                retired: Vec::new(),
            }),
            pool: Arc::new(pool),
            n_experts: n,
            n_used: moe.n_used,
            layout,
            split: AtomicBool::new(true),
            prefetch: AtomicBool::new(true),
            decode_prefetch: AtomicBool::new(decode_prefetch_enabled()),
            stats: Mutex::new([KqStats::default(); 2]),
        })
    }

    /// Transitions an idle streamed bank (`in_turn == false`, after GPU sync)
    /// to direct full-expert residency. Releases partial banks before
    /// allocating direct banks; if direct allocation fails, restores the
    /// partial banks before returning the error.
    pub fn promote_to_direct(&self) -> Result<()> {
        if self.direct.get().is_some() {
            return Ok(());
        }
        let mut g = lock(&self.inner);
        if self.direct.get().is_some() {
            return Ok(());
        }
        if g.in_turn {
            return Err(bad(
                "cannot promote expert bank to direct while a turn is active",
            ));
        }
        crate::array::synchronize();
        self.settle_all_ahead(&mut g);
        let inner = &mut *g;
        if inner.classes.len() != CLASSES.len() {
            return Err(bad("streamed class banks missing during direct promotion"));
        }
        for c in &mut inner.classes {
            inner.retired.extend(
                c.proj
                    .iter()
                    .flat_map(|p| p.banks().map(PurgeableBank::watch)),
            );
            c.proj = Vec::new();
            c.dir.clear();
        }
        crate::array::clear_cache();
        inner.retired.retain(BankWatch::alive);
        if !inner.retired.is_empty() {
            return Err(bad(
                "an array still holds the old bank; direct banks are not allocated",
            ));
        }
        let geos = [inner.classes[0].geo, inner.classes[1].geo];
        let built = build_direct_bank(
            &self.pool,
            self.n_experts,
            &self.layout,
            &geos,
            |il, ci, local| {
                inner.classes[ci]
                    .tensors
                    .get(local)
                    .cloned()
                    .ok_or_else(|| bad(format!("layer {il} missing from class {ci}")))
            },
            false,
        );
        match built {
            Ok(direct) => {
                if self.direct.set(ResidentBank::Direct(direct)).is_err() {
                    return Err(bad("direct bank already installed"));
                }
                inner.classes.clear();
                Ok(())
            }
            Err(err) => {
                // Free any partially allocated direct banks before restoring partial.
                crate::array::clear_cache();
                for c in &mut inner.classes {
                    if !c.tensors.is_empty() {
                        match alloc(&c.geo, c.dir.total_slots()) {
                            Ok(proj) => {
                                for p in &proj {
                                    for b in p.banks() {
                                        let _ = b.set_purgeable(PurgeableState::Volatile);
                                    }
                                }
                                c.proj = proj;
                            }
                            Err(restore_err) => {
                                tracing::warn!(
                                    "hy_v4_flash partial bank restore failed after direct alloc error: {restore_err}"
                                );
                                for prev in &mut inner.classes {
                                    prev.proj = Vec::new();
                                    prev.dir.clear();
                                }
                                crate::array::clear_cache();
                                break;
                            }
                        }
                    }
                }
                Err(err)
            }
        }
    }

    pub fn is_direct(&self) -> bool {
        self.direct.get().is_some()
    }

    pub fn is_mmap(&self) -> bool {
        matches!(self.direct.get(), Some(ResidentBank::Mmap(_)))
    }

    pub fn mapped_bytes(&self) -> u64 {
        match self.direct.get() {
            Some(ResidentBank::Mmap(m)) => m.mapped_bytes,
            _ => 0,
        }
    }

    /// Records `count` routed-expert uses for `layer` in direct or mmap mode.
    pub fn record_direct_use(&self, layer: usize, count: usize) {
        match self.direct.get() {
            Some(ResidentBank::Direct(d)) => {
                if let Some(idx) = d.layer_to_idx.get(layer).copied().flatten() {
                    d.uses[d.layers[idx].ci].fetch_add(count, Ordering::Relaxed);
                }
            }
            Some(ResidentBank::Mmap(m)) => {
                if let Some(idx) = m.layer_to_idx.get(layer).copied().flatten() {
                    m.uses[m.layers[idx].ci].fetch_add(count, Ordering::Relaxed);
                }
            }
            None => {}
        }
    }

    /// In direct or mmap mode, returns `layer`'s resident expert views
    /// (`slot_id == expert_id`) without taking the bank lock.
    /// Returns `Ok(None)` in streamed mode.
    pub fn direct_layer_arrays(&self, layer: usize) -> Result<Option<DirectLayerArrays>> {
        match self.direct.get() {
            None => Ok(None),
            Some(ResidentBank::Direct(direct)) => {
                let dl = direct.wait_layer(layer)?;
                direct.uses[dl.ci].fetch_add(self.n_used, Ordering::Relaxed);
                Ok(Some(DirectLayerArrays::Repacked(dl.arrays.clone())))
            }
            Some(ResidentBank::Mmap(mmap)) => {
                let idx = mmap
                    .layer_to_idx
                    .get(layer)
                    .copied()
                    .flatten()
                    .ok_or_else(|| bad(format!("layer {layer} has no mmap experts")))?;
                let ml = &mmap.layers[idx];
                mmap.uses[ml.ci].fetch_add(self.n_used, Ordering::Relaxed);
                Ok(Some(DirectLayerArrays::Mmap(ml.projs.clone())))
            }
        }
    }

    /// In mmap mode, returns `layer`'s `[MmapProj; 3]` and records `uses`
    /// distinct expert uses. Returns `Ok(None)` when not in mmap mode.
    pub(crate) fn mmap_layer(&self, layer: usize, uses: usize) -> Result<Option<[MmapProj; 3]>> {
        let Some(ResidentBank::Mmap(mmap)) = self.direct.get() else {
            return Ok(None);
        };
        let idx = mmap
            .layer_to_idx
            .get(layer)
            .copied()
            .flatten()
            .ok_or_else(|| bad(format!("layer {layer} has no mmap experts")))?;
        let ml = &mmap.layers[idx];
        mmap.uses[ml.ci].fetch_add(uses, Ordering::Relaxed);
        Ok(Some(ml.projs.clone()))
    }

    /// In repacked direct mode, waits until `layer` has finished filling and
    /// returns its `[ProjArrays; 3]` (`slot_id == expert_id`) without taking
    /// the bank lock. Returns `Ok(None)` in streamed mode.
    pub fn direct_layer(&self, layer: usize) -> Result<Option<[ProjArrays; 3]>> {
        match self.direct.get() {
            None => Ok(None),
            Some(ResidentBank::Direct(direct)) => {
                let dl = direct.wait_layer(layer)?;
                direct.uses[dl.ci].fetch_add(self.n_used, Ordering::Relaxed);
                Ok(Some(dl.arrays.clone()))
            }
            Some(ResidentBank::Mmap(_)) => Err(bad(format!(
                "layer {layer}: mmap bank uses raw GGUF buffers, not repacked ProjArrays"
            ))),
        }
    }

    /// Whether the background direct fill has populated every MoE layer.
    pub fn direct_fill_complete(&self) -> bool {
        match self.direct.get() {
            Some(ResidentBank::Direct(d)) => d.fill_complete(),
            Some(ResidentBank::Mmap(_)) => true,
            None => false,
        }
    }

    /// Blocks until the background direct fill finishes all layers (no-op in
    /// mmap or streamed mode).
    pub fn wait_direct_fill(&self) -> Result<()> {
        match self.direct.get() {
            Some(ResidentBank::Direct(direct)) => direct.wait_all(),
            Some(ResidentBank::Mmap(_)) | None => Ok(()),
        }
    }

    pub fn split(&self) -> bool {
        self.split.load(Ordering::Relaxed)
    }

    pub fn set_split(&self, on: bool) {
        self.split.store(on, Ordering::Relaxed);
    }

    pub fn prefetch(&self) -> bool {
        self.prefetch.load(Ordering::Relaxed)
    }

    pub fn set_prefetch(&self, on: bool) {
        self.prefetch.store(on, Ordering::Relaxed);
    }

    pub fn decode_prefetch(&self) -> bool {
        self.decode_prefetch.load(Ordering::Relaxed)
    }

    pub fn set_decode_prefetch(&self, on: bool) {
        self.decode_prefetch.store(on, Ordering::Relaxed);
    }

    pub fn reads_in_flight(&self) -> usize {
        if self.direct.get().is_some() {
            return 0;
        }
        lock(&self.inner)
            .inflight
            .iter()
            .filter(|w| !w.done())
            .count()
    }

    pub fn n_experts(&self) -> usize {
        self.n_experts
    }

    pub fn layout(&self) -> &ClassLayout {
        &self.layout
    }

    pub fn class_of(&self, layer: usize) -> Option<ExpertClass> {
        self.layout
            .of_layer
            .get(layer)
            .copied()
            .flatten()
            .map(|(ci, _)| CLASSES[ci])
    }

    pub fn cache_slots(&self) -> [usize; 2] {
        match self.direct.get() {
            Some(ResidentBank::Direct(d)) => return d.cache_slots,
            Some(ResidentBank::Mmap(m)) => return m.cache_slots,
            None => {}
        }
        let g = lock(&self.inner);
        [
            g.classes[0].dir.cache().slots(),
            g.classes[1].dir.cache().slots(),
        ]
    }

    pub fn staging_layers(&self) -> usize {
        if self.direct.get().is_some() {
            return 0;
        }
        lock(&self.inner).classes[0].dir.staging_layers()
    }

    /// Per class: bank bytes of one expert.
    pub fn slot_bytes(&self) -> [u64; 2] {
        match self.direct.get() {
            Some(ResidentBank::Direct(d)) => return d.slot_bytes,
            Some(ResidentBank::Mmap(m)) => return m.slot_bytes,
            None => {}
        }
        let g = lock(&self.inner);
        let one = |c: &ClassBank| c.geo.iter().map(ProjGeo::slot_bytes).sum();
        [one(&g.classes[0]), one(&g.classes[1])]
    }

    /// Anonymous bank bytes held in RAM (0 for mmap clean file-backed pages).
    pub fn nbytes(&self) -> u64 {
        match self.direct.get() {
            Some(ResidentBank::Direct(direct)) => {
                return direct
                    .layers
                    .iter()
                    .flat_map(|l| &l.proj)
                    .flat_map(ProjBank::banks)
                    .map(|b| b.nbytes() as u64)
                    .sum();
            }
            Some(ResidentBank::Mmap(_)) => return 0,
            None => {}
        }
        lock(&self.inner)
            .classes
            .iter()
            .flat_map(|c| &c.proj)
            .flat_map(ProjBank::banks)
            .map(|b| b.nbytes() as u64)
            .sum()
    }

    /// Anonymous bytes of every bank resident in RAM now (0 for mmap).
    pub fn resident_bytes(&self) -> Result<u64> {
        match self.direct.get() {
            Some(ResidentBank::Direct(direct)) => {
                let mut n = 0;
                for p in direct.layers.iter().flat_map(|l| &l.proj) {
                    for b in p.banks() {
                        n += b.resident_bytes()?;
                    }
                }
                return Ok(n);
            }
            Some(ResidentBank::Mmap(_)) => return Ok(0),
            None => {}
        }
        let g = lock(&self.inner);
        let mut n = 0;
        for p in g.classes.iter().flat_map(|c| &c.proj) {
            for b in p.banks() {
                n += b.resident_bytes()?;
            }
        }
        Ok(n)
    }

    /// Bytes of every bank resident in RAM now, and of those the bytes of
    /// banks the kernel holds volatile or empty, which the host counts
    /// purgeable. Without a kernel reading a bank is volatile outside a
    /// turn. Always `{0, 0}` for mmap because clean file-backed pages are
    /// excluded from `HostMemory::non_reclaimable()`.
    pub fn residency(&self) -> Result<Residency> {
        match self.direct.get() {
            Some(ResidentBank::Direct(direct)) => {
                let mut r = Residency::default();
                for p in direct.layers.iter().flat_map(|l| &l.proj) {
                    for b in p.banks() {
                        let n = b.resident_bytes()?;
                        r.resident += n;
                        if kernel_volatile(b).unwrap_or(false) {
                            r.volatile += n;
                        }
                    }
                }
                return Ok(r);
            }
            Some(ResidentBank::Mmap(_)) => return Ok(Residency::default()),
            None => {}
        }
        let g = lock(&self.inner);
        let mut r = Residency::default();
        for p in g.classes.iter().flat_map(|c| &c.proj) {
            for b in p.banks() {
                let n = b.resident_bytes()?;
                r.resident += n;
                if kernel_volatile(b).unwrap_or(!g.in_turn) {
                    r.volatile += n;
                }
            }
        }
        Ok(r)
    }

    /// Writes zeros over every bank so each of its pages is resident.
    #[cfg(test)]
    pub(crate) fn touch(&self) {
        match self.direct.get() {
            Some(ResidentBank::Direct(direct)) => {
                let _ = direct.wait_all();
                return;
            }
            Some(ResidentBank::Mmap(_)) => return,
            None => {}
        }
        let g = lock(&self.inner);
        for b in g
            .classes
            .iter()
            .flat_map(|c| &c.proj)
            .flat_map(ProjBank::banks)
        {
            // SAFETY: the whole of a live bank, no GPU work in flight (tests).
            unsafe { std::ptr::write_bytes(b.data_ptr().0, 0, b.nbytes()) };
        }
    }

    /// Per class index.
    pub fn stats(&self) -> [KqStats; 2] {
        let mut out = *lock(&self.stats);
        let uses = match self.direct.get() {
            Some(ResidentBank::Direct(d)) => Some(&d.uses),
            Some(ResidentBank::Mmap(m)) => Some(&m.uses),
            None => None,
        };
        if let Some(uses) = uses {
            for (ci, s) in out.iter_mut().enumerate() {
                let u = uses[ci].load(Ordering::Relaxed);
                s.uses += u;
                s.hits += u;
            }
        }
        out
    }

    fn begin(&self, g: &mut Inner) -> Result<bool> {
        let mut any = false;
        for c in &mut g.classes {
            let mut emptied = false;
            for p in &c.proj {
                for b in p.banks() {
                    emptied |=
                        b.set_purgeable(PurgeableState::NonVolatile)? == PurgeableState::Empty;
                }
            }
            if emptied {
                c.dir.clear();
            }
            c.dir.new_turn();
            any |= emptied;
        }
        g.in_turn = true;
        Ok(any)
    }

    pub fn in_turn(&self) -> bool {
        match self.direct.get() {
            Some(ResidentBank::Direct(d)) => d.in_turn.load(Ordering::Acquire),
            Some(ResidentBank::Mmap(m)) => m.in_turn.load(Ordering::Acquire),
            None => lock(&self.inner).in_turn,
        }
    }

    /// Updates the in-turn flag directly (pauses/resumes background mmap pre-warm).
    pub fn set_in_turn(&self, active: bool) {
        match self.direct.get() {
            Some(ResidentBank::Direct(d)) => d.in_turn.store(active, Ordering::Release),
            Some(ResidentBank::Mmap(m)) => m.set_in_turn(active),
            None => lock(&self.inner).in_turn = active,
        }
    }

    /// Makes the banks `NonVolatile` for a turn. Returns whether Metal had
    /// discarded any of them (that class's tags and cache are forgotten).
    pub fn begin_turn(&self) -> Result<bool> {
        match self.direct.get() {
            Some(ResidentBank::Direct(direct)) => {
                direct.in_turn.store(true, Ordering::Release);
                return Ok(false);
            }
            Some(ResidentBank::Mmap(mmap)) => {
                mmap.set_in_turn(true);
                return Ok(false);
            }
            None => {}
        }
        let mut g = lock(&self.inner);
        if g.in_turn {
            return Ok(false);
        }
        self.begin(&mut g)
    }

    fn settle_ahead(&self, g: &mut Inner, ci: usize, pick: impl Fn(usize) -> bool) {
        let (now, keep): (Vec<Ahead>, Vec<Ahead>) = std::mem::take(&mut g.ahead[ci])
            .into_iter()
            .partition(|a| pick(a.local));
        g.ahead[ci] = keep;
        for a in now {
            match a.ticket.wait() {
                Ok(io) => {
                    let c = &mut g.classes[ci];
                    for &e in &a.experts {
                        let s = c.dir.stage(a.local, e);
                        c.dir.tags[s] = Some((a.local, e));
                    }
                    {
                        let mut st = lock(&self.stats);
                        let s = &mut st[ci];
                        s.fills += 1;
                        s.slots += a.experts.len();
                        s.bytes += io.bytes;
                        s.ns += io.ns;
                        s.read_ns += io.read_ns;
                        s.repack_ns += io.repack_ns;
                    }
                    match g.ahead_left[ci].iter_mut().find(|(l, _)| *l == a.local) {
                        Some((_, left)) => {
                            left.extend(&a.experts);
                            left.sort_unstable();
                            left.dedup();
                        }
                        None => g.ahead_left[ci].push((a.local, a.experts)),
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        "hy_v4_flash prefetch of class {ci} local layer {} failed: {err}",
                        a.local
                    );
                }
            }
        }
    }

    fn close_ahead_locked(&self, g: &mut Inner, ci: usize, pick: impl Fn(usize) -> bool) {
        let (done, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut g.ahead_left[ci])
            .into_iter()
            .partition(|(l, _)| pick(*l));
        g.ahead_left[ci] = keep;
        let unused: usize = done.iter().map(|(_, left)| left.len()).sum();
        if unused > 0 {
            lock(&self.stats)[ci].prefetch_unused += unused;
        }
    }

    fn settle_all_ahead(&self, g: &mut Inner) {
        for ci in 0..CLASSES.len() {
            self.settle_ahead(g, ci, |_| true);
            self.close_ahead_locked(g, ci, |_| true);
        }
        g.drain();
    }

    /// After the turn's last eval: staging copies served this turn move into
    /// cache slots, then the banks become `Volatile`.
    pub fn end_turn(&self) -> Result<()> {
        match self.direct.get() {
            Some(ResidentBank::Direct(direct)) => {
                if direct.in_turn.swap(false, Ordering::AcqRel) {
                    crate::array::synchronize();
                }
                // Direct banks remain NonVolatile until KqBank drops.
                return Ok(());
            }
            Some(ResidentBank::Mmap(mmap)) => {
                if mmap.in_turn.swap(false, Ordering::AcqRel) {
                    crate::array::synchronize();
                }
                return Ok(());
            }
            None => {}
        }
        let mut g = lock(&self.inner);
        if !g.in_turn {
            return Ok(());
        }
        crate::array::synchronize();
        self.settle_all_ahead(&mut g);
        self.promote(&mut g)?;
        g.in_turn = false;
        for p in g.classes.iter().flat_map(|c| &c.proj) {
            for b in p.banks() {
                b.set_purgeable(PurgeableState::Volatile)?;
            }
        }
        Ok(())
    }

    /// Staging copies served since the last promote move into cache slots
    /// (the GPU is drained). The copies made.
    fn promote(&self, g: &mut Inner) -> Result<usize> {
        let mut total_moved = 0;
        for (ci, c) in g.classes.iter_mut().enumerate() {
            let t0 = Instant::now();
            c.dir.complete_all();
            let total = c.dir.total_slots();
            let moved = c.dir.promote();
            for &(from, to) in &moved {
                copy_slot(&c.proj, total, from, to)?;
            }
            if !moved.is_empty() {
                let mut st = lock(&self.stats);
                st[ci].promoted += moved.len();
                st[ci].promote_ns += t0.elapsed().as_nanos() as u64;
            }
            total_moved += moved.len();
        }
        Ok(total_moved)
    }

    /// Between two steps of one busy period, after every step so far was
    /// evaluated: the staging copies served so far move into cache slots as
    /// at a turn end, then with `seed_reset` the seed quota starts over as
    /// at a turn start. The banks stay non-volatile. The copies made.
    pub fn safe_point(&self, seed_reset: bool) -> Result<usize> {
        if self.direct.get().is_some() {
            return Ok(0);
        }
        let mut g = lock(&self.inner);
        if !g.in_turn {
            return Ok(0);
        }
        self.settle_all_ahead(&mut g);
        let any_staged = g.classes.iter().any(|c| c.dir.has_staged());
        let moved = if any_staged {
            if g.classes.iter().any(|c| c.dir.has_unsettled()) {
                crate::array::synchronize();
            }
            self.promote(&mut g)?
        } else {
            0
        };
        if seed_reset {
            for c in &mut g.classes {
                c.dir.new_turn();
            }
        }
        Ok(moved)
    }

    /// Drops the banks and allocates new ones with `staging` staging layers
    /// and `cache_slots` cache slots per class, after the old buffers are
    /// gone from MLX (refuses while any array still holds one; the bank then
    /// has no slots until a rebuild succeeds).
    pub fn rebuild(&self, cache_slots: [usize; 2], staging: usize) -> Result<()> {
        if self.direct.get().is_some() {
            return Err(bad("direct expert bank cannot be rebuilt"));
        }
        let mut g = lock(&self.inner);
        crate::array::synchronize();
        self.settle_all_ahead(&mut g);
        let n = self.n_experts;
        let mut dirs = Vec::with_capacity(CLASSES.len());
        for (ci, c) in g.classes.iter().enumerate() {
            if c.tensors.is_empty() && cache_slots[ci] > 0 {
                return Err(bad(format!("{:?} has no layers but cache slots", c.class)));
            }
            dirs.push(make_directory(
                staging,
                c.tensors.len(),
                n,
                cache_slots[ci],
                self.n_used,
            )?);
        }
        let inner = &mut *g;
        for c in &mut inner.classes {
            inner.retired.extend(
                c.proj
                    .iter()
                    .flat_map(|p| p.banks().map(PurgeableBank::watch)),
            );
            c.proj = Vec::new();
            c.dir.clear();
        }
        crate::array::clear_cache();
        inner.retired.retain(BankWatch::alive);
        if !inner.retired.is_empty() {
            return Err(bad(
                "an array still holds the old bank; a second bank is not allocated",
            ));
        }
        for (c, dir) in inner.classes.iter_mut().zip(dirs) {
            if !c.tensors.is_empty() {
                c.proj = alloc(&c.geo, dir.total_slots())?;
            }
            c.dir = dir;
        }
        inner.in_turn = true;
        Ok(())
    }

    /// Slots for one layer step's distinct, sorted `experts` (`uses[i]` =
    /// rows using `experts[i]`, may be empty); returns once every read is
    /// in. The slots stay pinned until the step is settled (`settle` /
    /// `settle_all`): only then may a later step evict or refill them.
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

    /// Like `prepare`, but returns immediately after submitting miss reads at
    /// high priority. `Prepared.reading` lists experts whose slots are in
    /// flight; the caller must call `Pending::finish` before gathering them.
    pub fn prepare_split(
        &self,
        layer: usize,
        experts: &[u32],
        uses: &[u32],
        decode: bool,
    ) -> Result<(Prepared, Option<Pending<'_>>)> {
        match self.direct.get() {
            Some(ResidentBank::Direct(direct)) => {
                check_direct_step(self.n_experts, experts, uses)?;
                let dl = direct.wait_layer(layer)?;
                direct.uses[dl.ci].fetch_add(experts.len(), Ordering::Relaxed);
                return Ok((
                    Prepared {
                        layer,
                        class: dl.class,
                        experts: experts.to_vec(),
                        slots: experts.to_vec(),
                        arrays: dl.arrays.clone(),
                        reading: Vec::new(),
                    },
                    None,
                ));
            }
            Some(ResidentBank::Mmap(_)) => {
                return Err(bad("mmap expert bank does not support repacked prepare"));
            }
            None => {}
        }
        let (ci, local) = self
            .layout
            .of_layer
            .get(layer)
            .copied()
            .flatten()
            .ok_or_else(|| bad(format!("layer {layer} has no streamed experts")))?;
        let mut g = lock(&self.inner);
        g.classes[ci].dir.check_step(local, experts, uses)?;
        let arrays = g.classes[ci].arrays()?;
        if !g.in_turn {
            self.begin(&mut g)?;
        }
        // Keep the next class-local layer's prefetch alive in the other
        // staging half; settle this layer's own prefetch and any stale ones.
        let t_ahead = Instant::now();
        let had_ahead = g.ahead[ci].iter().any(|a| a.local != local + 1);
        self.settle_ahead(&mut g, ci, |l| l != local + 1);
        let ahead_wait = if had_ahead {
            t_ahead.elapsed().as_nanos() as u64
        } else {
            0
        };
        g.drain();
        self.close_ahead_locked(&mut g, ci, |l| l < local);
        for (l, left) in g.ahead_left[ci].iter_mut() {
            if *l == local {
                left.retain(|e| experts.binary_search(e).is_err());
            }
        }
        let c = &mut g.classes[ci];
        let plan = c.dir.plan(local, experts, uses, decode);
        if !plan.copies.is_empty() {
            let t0 = Instant::now();
            let total = c.dir.total_slots();
            let base = c.dir.base();
            for &(e, slot) in &plan.copies {
                if let Err(err) =
                    copy_slot(&c.proj, total, c.dir.stage(local, e) as u32, base + slot)
                {
                    c.dir.abandon(&plan);
                    plan.copies
                        .iter()
                        .for_each(|&(_, s)| c.dir.cache.release(s));
                    return Err(err);
                }
            }
            c.dir.commit_copies(local, &plan);
            let mut st = lock(&self.stats);
            st[ci].promoted += plan.copies.len();
            st[ci].promote_ns += t0.elapsed().as_nanos() as u64;
        }
        let mut reading = Vec::new();
        let mut pending = None;
        if plan.reads() > 0 {
            let reads = match c.reads(local, &plan) {
                Ok(r) => r,
                Err(err) => {
                    c.dir.abandon(&plan);
                    return Err(err);
                }
            };
            if plan.drain {
                crate::array::synchronize();
                c.dir.complete(plan.epoch - 1);
            }
            c.dir.untag(local, &plan);
            // SAFETY: every slot read is untagged (staging) or an unmapped
            // victim (cache); no unfinished gather reads it (victims settled,
            // busy staging slots drained above); the batch is watched in
            // `inflight` and owned by `Pending` whose Drop waits on it.
            let ticket = match unsafe { self.pool.submit(reads, Priority::High) } {
                Ok(t) => t,
                Err(err) => {
                    c.dir.abandon(&plan);
                    return Err(err);
                }
            };
            reading = plan
                .staging
                .iter()
                .copied()
                .chain(plan.victims.iter().map(|&(e, _)| e))
                .collect();
            reading.sort_unstable();
            g.watch(&ticket);
            pending = Some(Pending {
                bank: self,
                ci,
                local,
                plan: Plan {
                    epoch: plan.epoch,
                    slots: Vec::new(),
                    staging: plan.staging,
                    victims: plan.victims,
                    copies: Vec::new(),
                    hits: plan.hits,
                    drain: plan.drain,
                },
                ticket: Some(ticket),
            });
        }
        {
            let mut st = lock(&self.stats);
            let s = &mut st[ci];
            s.uses += experts.len();
            s.hits += plan.hits;
            s.drains += usize::from(plan.drain);
            s.wait_ns += ahead_wait;
        }
        Ok((
            Prepared {
                layer,
                class: CLASSES[ci],
                experts: experts.to_vec(),
                slots: plan.slots,
                arrays,
                reading,
            },
            pending,
        ))
    }

    /// Plans one prefill FFN chunk once, executes copies and any required GPU
    /// drain, returns resident experts (`group_0`) ready for immediate gather,
    /// and starts reading missing experts in `subgroup_experts`-sized batches
    /// ordered by `last_use_order`.
    pub fn plan_prefill_subgroups(
        &self,
        layer: usize,
        used: &[u32],
        uses: &[u32],
        last_use_order: &[u32],
        subgroup_experts: usize,
    ) -> Result<(Prepared, Vec<u32>, PrefillSubgroups<'_>)> {
        match self.direct.get() {
            Some(ResidentBank::Direct(direct)) => {
                check_direct_step(self.n_experts, used, uses)?;
                let dl = direct.wait_layer(layer)?;
                direct.uses[dl.ci].fetch_add(used.len(), Ordering::Relaxed);
                return Ok((
                    Prepared {
                        layer,
                        class: dl.class,
                        experts: used.to_vec(),
                        slots: used.to_vec(),
                        arrays: dl.arrays.clone(),
                        reading: Vec::new(),
                    },
                    used.to_vec(),
                    PrefillSubgroups {
                        bank: self,
                        ci: dl.ci,
                        local: 0,
                        active: None,
                        queued: VecDeque::new(),
                    },
                ));
            }
            Some(ResidentBank::Mmap(_)) => {
                return Err(bad(
                    "mmap expert bank does not support repacked prefill subgroups",
                ));
            }
            None => {}
        }
        let (ci, local) = self
            .layout
            .of_layer
            .get(layer)
            .copied()
            .flatten()
            .ok_or_else(|| bad(format!("layer {layer} has no streamed experts")))?;
        let mut g = lock(&self.inner);
        g.classes[ci].dir.check_step(local, used, uses)?;
        let arrays = g.classes[ci].arrays()?;
        if !g.in_turn {
            self.begin(&mut g)?;
        }
        let t_ahead = Instant::now();
        let had_ahead = g.ahead[ci].iter().any(|a| a.local != local + 1);
        self.settle_ahead(&mut g, ci, |l| l != local + 1);
        let ahead_wait = if had_ahead {
            t_ahead.elapsed().as_nanos() as u64
        } else {
            0
        };
        g.drain();
        self.close_ahead_locked(&mut g, ci, |l| l < local);
        for (l, left) in g.ahead_left[ci].iter_mut() {
            if *l == local {
                left.retain(|e| used.binary_search(e).is_err());
            }
        }
        let c = &mut g.classes[ci];
        let plan = c.dir.plan(local, used, uses, false);
        if !plan.copies.is_empty() {
            let t0 = Instant::now();
            let total = c.dir.total_slots();
            let base = c.dir.base();
            for &(e, slot) in &plan.copies {
                if let Err(err) =
                    copy_slot(&c.proj, total, c.dir.stage(local, e) as u32, base + slot)
                {
                    c.dir.abandon(&plan);
                    plan.copies
                        .iter()
                        .for_each(|&(_, s)| c.dir.cache.release(s));
                    return Err(err);
                }
            }
            c.dir.commit_copies(local, &plan);
            let mut st = lock(&self.stats);
            st[ci].promoted += plan.copies.len();
            st[ci].promote_ns += t0.elapsed().as_nanos() as u64;
        }
        if plan.drain {
            crate::array::synchronize();
            c.dir.complete(plan.epoch - 1);
        }
        let mut miss_kind: Vec<Option<MissTarget>> = vec![None; self.n_experts];
        for &e in &plan.staging {
            miss_kind[e as usize] = Some(MissTarget::Staging(e));
        }
        for &(e, slot) in &plan.victims {
            miss_kind[e as usize] = Some(MissTarget::Victim(e, slot));
        }
        let group_0: Vec<u32> = used
            .iter()
            .copied()
            .filter(|&e| miss_kind[e as usize].is_none())
            .collect();
        let mut ordered_misses = Vec::with_capacity(plan.reads());
        for &e in last_use_order {
            if let Some(tgt) = miss_kind.get_mut(e as usize).and_then(Option::take) {
                ordered_misses.push(tgt);
            }
        }
        for tgt in miss_kind.into_iter().flatten() {
            ordered_misses.push(tgt);
        }
        let mut reading: Vec<u32> = plan
            .staging
            .iter()
            .copied()
            .chain(plan.victims.iter().map(|&(e, _)| e))
            .collect();
        reading.sort_unstable();

        let cap = subgroup_experts.max(1);
        let mut queued: VecDeque<Vec<MissTarget>> = ordered_misses
            .chunks(cap)
            .map(<[MissTarget]>::to_vec)
            .collect();

        let active = if let Some(first_targets) = queued.pop_front() {
            let reads = match c.reads_targets(local, &first_targets) {
                Ok(r) => r,
                Err(err) => {
                    c.dir.abandon(&plan);
                    return Err(err);
                }
            };
            c.dir.untag(local, &plan);
            // SAFETY: every miss staging slot is untagged above, victims are
            // unmapped, any busy staging slot drained above, and `inflight`
            // plus `PrefillSubgroups::Drop` ensure no IO outlives this step.
            let ticket = match unsafe { self.pool.submit(reads, Priority::High) } {
                Ok(t) => t,
                Err(err) => {
                    c.dir.abandon(&plan);
                    return Err(err);
                }
            };
            g.watch(&ticket);
            Some((first_targets, ticket))
        } else {
            None
        };

        {
            let mut st = lock(&self.stats);
            let s = &mut st[ci];
            s.uses += used.len();
            s.hits += plan.hits;
            s.drains += usize::from(plan.drain);
            s.wait_ns += ahead_wait;
        }
        Ok((
            Prepared {
                layer,
                class: CLASSES[ci],
                experts: used.to_vec(),
                slots: plan.slots,
                arrays,
                reading,
            },
            group_0,
            PrefillSubgroups {
                bank: self,
                ci,
                local,
                active,
                queued,
            },
        ))
    }

    /// Reads `layer`'s predicted `experts` (sorted, distinct) into its staging
    /// half at low priority when the bank has at least two staging layers.
    pub fn prefetch_layer(&self, layer: usize, experts: &[u32]) -> Result<usize> {
        if self.direct.get().is_some() || !self.prefetch() {
            return Ok(0);
        }
        let (ci, local) = self
            .layout
            .of_layer
            .get(layer)
            .copied()
            .flatten()
            .ok_or_else(|| bad(format!("layer {layer} has no streamed experts")))?;
        let mut g = lock(&self.inner);
        if g.classes[ci].dir.staging_layers() < 2 {
            return Ok(0);
        }
        g.classes[ci].dir.check_step(local, experts, &[])?;
        if g.classes[ci].proj.is_empty() {
            return Err(bad(format!(
                "{:?}: no bank (a rebuild failed)",
                g.classes[ci].class
            )));
        }
        if !g.in_turn {
            self.begin(&mut g)?;
        }
        // Keep the immediately preceding class-local layer's prefetch alive in
        // the other staging half; settle everything else in this class.
        self.settle_ahead(&mut g, ci, |l| l + 1 != local);
        let c = &mut g.classes[ci];
        let want: Vec<u32> = experts
            .iter()
            .copied()
            .filter(|&e| {
                c.dir.cache.lookup(local, e).is_none()
                    && c.dir.tags[c.dir.stage(local, e)] != Some((local, e))
            })
            .collect();
        if want.is_empty() {
            return Ok(0);
        }
        let max_use = want
            .iter()
            .map(|&e| c.dir.stage_use[c.dir.stage(local, e)])
            .max()
            .unwrap_or(0);
        let need_drain = max_use > c.dir.cache.completed();
        if need_drain {
            crate::array::synchronize();
            c.dir.complete(max_use);
        }
        let mut reads = Vec::with_capacity(want.len() * 12);
        for &e in &want {
            let s = c.dir.stage(local, e);
            for (t, p) in c.tensors[local].iter().zip(&c.proj) {
                reads.extend(p.dst.reads(t, e, s)?);
            }
        }
        sort_reads(&mut reads);
        for &e in &want {
            let s = c.dir.stage(local, e);
            c.dir.tags[s] = None;
        }
        // SAFETY: target staging slots are untagged, any pinned slot was
        // drained above, and `ahead[ci]` owns the ticket until settled.
        let ticket = unsafe { self.pool.submit(reads, Priority::Low) }?;
        let k = want.len();
        g.ahead[ci].push(Ahead {
            local,
            experts: want,
            ticket,
        });
        let mut st = lock(&self.stats);
        st[ci].prefetch_slots += k;
        st[ci].drains += usize::from(need_drain);
        Ok(k)
    }

    /// Counts unused prefetched experts for `layer` once every step of it ran.
    pub fn end_layer(&self, layer: usize) {
        if self.direct.get().is_some() {
            return;
        }
        let Some((ci, local)) = self.layout.of_layer.get(layer).copied().flatten() else {
            return;
        };
        let mut g = lock(&self.inner);
        self.close_ahead_locked(&mut g, ci, |l| l == local);
    }

    pub fn note_split(&self, layer: usize, mixed: bool) {
        let Some((ci, _)) = self.layout.of_layer.get(layer).copied().flatten() else {
            return;
        };
        let mut st = lock(&self.stats);
        st[ci].split_steps += 1;
        st[ci].split_mixed += usize::from(mixed);
    }

    /// Per class: the epoch of the last step planned so far.
    pub fn issued(&self) -> [u64; 2] {
        if self.direct.get().is_some() {
            return [0, 0];
        }
        let g = lock(&self.inner);
        [
            g.classes[0].dir.cache().epoch(),
            g.classes[1].dir.cache().epoch(),
        ]
    }

    /// The GPU finished every gather of the steps up to `issued` (taken
    /// before a host read that waits for them).
    pub fn settle(&self, issued: [u64; 2]) {
        if self.direct.get().is_some() {
            return;
        }
        let mut g = lock(&self.inner);
        for (c, e) in g.classes.iter_mut().zip(issued) {
            c.dir.complete(e);
        }
    }

    /// The GPU finished every gather planned so far.
    pub fn settle_all(&self) {
        if self.direct.get().is_some() {
            return;
        }
        for c in &mut lock(&self.inner).classes {
            c.dir.complete_all();
        }
    }

    /// Panics (release builds too) unless every slot of `p` holds its
    /// expert of `p.layer`: a gather on a stale slot would compute with
    /// another layer's (or discarded) weights.
    pub fn assert_slots(&self, p: &Prepared) {
        self.assert_slots_where(p, |_| true);
    }

    /// `assert_slots` restricted to experts not in `p.reading`.
    pub fn assert_ready_slots(&self, p: &Prepared) {
        self.assert_slots_where(p, |e| !p.is_reading(e));
    }

    /// `assert_slots` restricted to `experts`.
    pub fn assert_experts_slots(&self, p: &Prepared, experts: &[u32]) {
        if let Some(resident) = self.direct.get() {
            match resident {
                ResidentBank::Direct(direct) => {
                    let idx = direct.layer_to_idx.get(p.layer).copied().flatten();
                    assert!(
                        idx.is_some_and(|i| direct.layers[i].ready.load(Ordering::Acquire)),
                        "Hy4 expert bank: direct gather of layer {} before fill completed",
                        p.layer
                    );
                }
                ResidentBank::Mmap(mmap) => {
                    assert!(
                        mmap.layer_to_idx.get(p.layer).copied().flatten().is_some(),
                        "Hy4 expert bank: mmap gather of unmapped layer {}",
                        p.layer
                    );
                }
            }
            let stale = experts.iter().find_map(|&e| {
                let s = p.slot(e)?;
                (s != e || e as usize >= self.n_experts).then_some((e, s))
            });
            assert!(
                stale.is_none(),
                "Hy4 expert bank: direct gather of layer {} on non-identity slot {stale:?}",
                p.layer
            );
            return;
        }
        let stale = self
            .layout
            .of_layer
            .get(p.layer)
            .copied()
            .flatten()
            .and_then(|(ci, local)| {
                let g = lock(&self.inner);
                let dir = &g.classes[ci].dir;
                experts.iter().find_map(|&e| {
                    let s = p.slot(e)?;
                    (!dir.holds(local, e, s)).then_some((e, s))
                })
            });
        assert!(
            stale.is_none(),
            "Hy4 expert bank: gather of layer {} on stale slot {stale:?}",
            p.layer
        );
    }

    fn assert_slots_where(&self, p: &Prepared, check: impl Fn(u32) -> bool) {
        if let Some(resident) = self.direct.get() {
            match resident {
                ResidentBank::Direct(direct) => {
                    let idx = direct.layer_to_idx.get(p.layer).copied().flatten();
                    assert!(
                        idx.is_some_and(|i| direct.layers[i].ready.load(Ordering::Acquire)),
                        "Hy4 expert bank: direct gather of layer {} before fill completed",
                        p.layer
                    );
                }
                ResidentBank::Mmap(mmap) => {
                    assert!(
                        mmap.layer_to_idx.get(p.layer).copied().flatten().is_some(),
                        "Hy4 expert bank: mmap gather of unmapped layer {}",
                        p.layer
                    );
                }
            }
            let stale = p
                .experts
                .iter()
                .zip(&p.slots)
                .filter(|&(&e, _)| check(e))
                .find(|&(&e, &s)| s != e || e as usize >= self.n_experts)
                .map(|(&e, &s)| (e, s));
            assert!(
                stale.is_none(),
                "Hy4 expert bank: direct gather of layer {} on non-identity slot {stale:?}",
                p.layer
            );
            return;
        }
        let stale = self
            .layout
            .of_layer
            .get(p.layer)
            .copied()
            .flatten()
            .and_then(|(ci, local)| {
                let g = lock(&self.inner);
                let dir = &g.classes[ci].dir;
                p.experts
                    .iter()
                    .zip(&p.slots)
                    .filter(|&(&e, _)| check(e))
                    .find(|&(&e, &s)| !dir.holds(local, e, s))
                    .map(|(&e, &s)| (e, s))
            });
        assert!(
            stale.is_none(),
            "Hy4 expert bank: gather of layer {} on stale slot {stale:?}",
            p.layer
        );
    }

    #[cfg(test)]
    pub(crate) fn test_check_caches(&self) {
        if self.direct.get().is_some() {
            return;
        }
        let g = lock(&self.inner);
        for c in &g.classes {
            c.dir.cache().check().unwrap();
        }
    }

    #[cfg(test)]
    pub(crate) fn test_forget_where(&self, f: impl Fn(usize, u32) -> bool) {
        if self.direct.get().is_some() {
            return;
        }
        let mut g = lock(&self.inner);
        crate::array::synchronize();
        self.settle_all_ahead(&mut g);
        for (ci, c) in g.classes.iter_mut().enumerate() {
            let layers = &self.layout.layers[ci];
            for t in c.dir.tags.iter_mut() {
                if let Some((local, e)) = *t
                    && f(layers[local], e)
                {
                    *t = None;
                }
            }
            let gone: Vec<u32> = (0..c.dir.cache.slots() as u32)
                .filter(|&s| {
                    c.dir
                        .cache
                        .occ(s)
                        .is_some_and(|(local, e)| f(layers[local], e))
                })
                .collect();
            for s in gone {
                c.dir.cache.release(s);
            }
        }
    }
}

#[cfg(test)]
mod tests;
