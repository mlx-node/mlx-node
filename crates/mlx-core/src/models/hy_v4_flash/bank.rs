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

use std::collections::VecDeque;
use std::ffi::CString;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use napi::{Error, Result};

use super::config::MoeConfig;
use super::kq_slot;
use super::scan::{ExpertClass, ExpertLayer, Located};
use crate::array::alloc::{DstPtrs, OutLayout};
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

/// The streamed classes, in bank index order.
pub const CLASSES: [ExpertClass; 2] = [ExpertClass::Q6kDown, ExpertClass::Q4kDown];

/// Half-life of LFU-decay expert cache scores, in decode tokens.
pub const LFU_HALF_LIFE_TOKENS: f64 = 256.0;

fn use_lru_policy() -> bool {
    std::env::var("MLX_HY4_CACHE_POLICY").is_ok_and(|v| v.eq_ignore_ascii_case("lru"))
}

fn prefill_nax_enabled() -> bool {
    std::env::var("MLX_HY4_PREFILL_NAX").as_deref() != Ok("0")
}

fn lfu_half_life(n_layers: usize, n_used: usize) -> f64 {
    (LFU_HALF_LIFE_TOKENS * (n_layers.max(1) * n_used.max(1)) as f64).max(1.0)
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

    /// `staging` = 1 or 2 staging layers; uses LFU-decay unless
    /// `MLX_HY4_CACHE_POLICY=lru`.
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
        let cache = if use_lru_policy() {
            Cache::lru(cache_slots, n_layers, n_experts)
        } else {
            Cache::lfu_decay(cache_slots, n_layers, n_experts, half_life_uses)
        };
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
        let row_bytes = self.format.row_bytes(self.dst.k) as u64;
        let bands = ext
            .len
            .div_ceil(band_bytes.max(1))
            .clamp(1, self.rows as u64) as usize;
        let per = self.rows.div_ceil(bands);
        let mut out = Vec::with_capacity(bands);
        for r0 in (0..self.rows).step_by(per) {
            let rows = per.min(self.rows - r0);
            let band = Extent {
                shard: ext.shard,
                offset: ext.offset + r0 as u64 * row_bytes,
                len: rows as u64 * row_bytes,
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

/// One projection's bank arrays and their K-quant mode.
#[derive(Clone)]
pub struct ProjArrays {
    pub w: MxArray,
    pub s: MxArray,
    pub b: MxArray,
    pub format: KQuantFormat,
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

pub struct KqBank {
    pool: IoPool,
    n_experts: usize,
    n_used: usize,
    layout: ClassLayout,
    split: AtomicBool,
    prefetch: AtomicBool,
    inner: Mutex<Inner>,
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
            let dir = Directory::new_with_lfu(
                staging,
                layers.len(),
                n,
                cache_slots[ci],
                lfu_half_life(layers.len(), moe.n_used),
            )?;
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
            pool,
            n_experts: n,
            n_used: moe.n_used,
            layout,
            split: AtomicBool::new(true),
            prefetch: AtomicBool::new(true),
            inner: Mutex::new(Inner {
                ahead: [Vec::new(), Vec::new()],
                inflight: Vec::new(),
                ahead_left: [Vec::new(), Vec::new()],
                classes: banks,
                in_turn: true,
                retired: Vec::new(),
            }),
            stats: Mutex::new([KqStats::default(); 2]),
        })
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

    pub fn reads_in_flight(&self) -> usize {
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
        let g = lock(&self.inner);
        [
            g.classes[0].dir.cache().slots(),
            g.classes[1].dir.cache().slots(),
        ]
    }

    pub fn staging_layers(&self) -> usize {
        lock(&self.inner).classes[0].dir.staging_layers()
    }

    /// Per class: bank bytes of one expert.
    pub fn slot_bytes(&self) -> [u64; 2] {
        let g = lock(&self.inner);
        let one = |c: &ClassBank| c.geo.iter().map(ProjGeo::slot_bytes).sum();
        [one(&g.classes[0]), one(&g.classes[1])]
    }

    /// Bytes of every bank.
    pub fn nbytes(&self) -> u64 {
        lock(&self.inner)
            .classes
            .iter()
            .flat_map(|c| &c.proj)
            .flat_map(ProjBank::banks)
            .map(|b| b.nbytes() as u64)
            .sum()
    }

    /// Bytes of every bank resident in RAM now.
    pub fn resident_bytes(&self) -> Result<u64> {
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
    /// turn.
    pub fn residency(&self) -> Result<Residency> {
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
        *lock(&self.stats)
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
        lock(&self.inner).in_turn
    }

    /// Makes the banks `NonVolatile` for a turn. Returns whether Metal had
    /// discarded any of them (that class's tags and cache are forgotten).
    pub fn begin_turn(&self) -> Result<bool> {
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
        let mut g = lock(&self.inner);
        crate::array::synchronize();
        self.settle_all_ahead(&mut g);
        let n = self.n_experts;
        let mut dirs = Vec::with_capacity(CLASSES.len());
        for (ci, c) in g.classes.iter().enumerate() {
            if c.tensors.is_empty() && cache_slots[ci] > 0 {
                return Err(bad(format!("{:?} has no layers but cache slots", c.class)));
            }
            dirs.push(Directory::new_with_lfu(
                staging,
                c.tensors.len(),
                n,
                cache_slots[ci],
                lfu_half_life(c.tensors.len(), self.n_used),
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
        if !self.prefetch() {
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
        let need_drain = want
            .iter()
            .any(|&e| c.dir.stage_use[c.dir.stage(local, e)] > c.dir.cache.completed());
        if need_drain {
            crate::array::synchronize();
            c.dir.complete_all();
        }
        let mut reads = Vec::with_capacity(want.len() * 12);
        for &e in &want {
            let s = c.dir.stage(local, e);
            for (t, p) in c.tensors[local].iter().zip(&c.proj) {
                reads.extend(p.dst.reads(t, e, s)?);
            }
        }
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
        let g = lock(&self.inner);
        [
            g.classes[0].dir.cache().epoch(),
            g.classes[1].dir.cache().epoch(),
        ]
    }

    /// The GPU finished every gather of the steps up to `issued` (taken
    /// before a host read that waits for them).
    pub fn settle(&self, issued: [u64; 2]) {
        let mut g = lock(&self.inner);
        for (c, e) in g.classes.iter_mut().zip(issued) {
            c.dir.complete(e);
        }
    }

    /// The GPU finished every gather planned so far.
    pub fn settle_all(&self) {
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
        let g = lock(&self.inner);
        for c in &g.classes {
            c.dir.cache().check().unwrap();
        }
    }

    #[cfg(test)]
    pub(crate) fn test_forget_where(&self, f: impl Fn(usize, u32) -> bool) {
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
