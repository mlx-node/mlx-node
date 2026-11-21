//! Store: the Hy4 trunk from its GGUF shards. Resident weights are read once
//! (`weights::Resident`), routed experts stream through the two-class
//! K-quant bank, Engram rows come from the F8 table on shard 2.
//!
//! The admission picks the context and the prefill chunk at load:
//!
//! ```text
//! ctx = max_context, max_context / 2, ... >= MIN_CONTEXT
//!   chunk = the largest of CHUNKS admitted with cache >= second_staging_min_cache,
//!           else the smallest admitted, else the next ctx
//! cache bytes -> class slots in proportion to the classes' layers
//! ```
//!
//! A staging layer is charged 1024 slots (512 per class) at the larger
//! slot: 2 layers = 2 x 1024 x 4,663,296 B = 8.89 GiB, above the 8.35 GiB
//! the bank allocates.
//!
//! Lanes (sequences decoded together) share one KV allowance sized for the
//! admitted context; each lane past the first is charged its floor and its
//! draft's pooled keys (`lane_bytes`). The context and chunk are picked for
//! one lane; the load then admits `min(LANES_MAX, MLX_SCHED_MAX_NUM_SEQS,
//! row_cap)` lanes at them (one when no wave fits), fewer when only fewer
//! fit, so lanes never shrink the window or the chunk grid. A decode wave holds at most `row_cap`
//! rows (`row_cap`), the rows its Q8 matrices take through `qmv_wide`.
//!
//! A turn is `begin_turn` .. `end_turn` (`Lease`): the start re-admits at the
//! same context and chunk and may shrink the bank to what the host admits or
//! rebuild it, then holds a wired-limit lease of fixed + cache bytes until
//! the end, when the bank becomes volatile. A re-admission inside the turn
//! (`recheck`) shrinks the bank the same way but does not grow one that
//! holds cache slots.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use napi::{Error, Result};

use super::bank::{CLASSES, ClassLayout, KqBank, Residency, file_bytes, slot_bytes};
use super::cache::WINDOW_SLACK;
use super::config::{Attn, HyV4FlashConfig};
use super::model::Model;
use super::moe::ExpertSource;
use super::mtp::SEEDS;
use super::rows::{EngramRows, F8Table, PrefetchHandle};
use super::scan::{Role, Scan};
use super::weights::{
    Form, MmapDense, Resident, form, mmap_dense_saved_bytes, mmap_dense_spec, resident_bytes,
    role_resident,
};
use crate::cache_limit::{CacheLimitGuard, coordinator};
use crate::models::deepseek_v41::store::POOL_LIMIT;
use crate::moe_stream::budget::{
    Admission, Bound, DSV41, Limits, ModelSize, Own, Profile, Terms, Turn, admit_with,
    readmit_with, working_set,
};
use crate::moe_stream::io::{DEFAULT_THREADS, IoPool, ShardSet};
use crate::stream::WiredLimitContext;
use crate::utils::host::{HostMemory, host_memory};

pub const CACHE_ENV: &str = "MLX_HY4_EXPERT_CACHE_GB";
pub const RESERVE_ENV: &str = "MLX_HY4_ADMIT_RESERVE_GB";
/// Direct full-expert residency switch: `0`/`false`, `1`/`true`, or unset/`auto`.
pub const DIRECT_ENV: &str = "MLX_HY4_DIRECT";
/// Zero-copy mmap expert residency switch: `0`/`false`, `1`/`true`, or unset/`auto`.
pub const MMAP_ENV: &str = "MLX_HY4_MMAP";
/// Background page-cache pre-warm switch for mmap experts: `0`/`false`, `1`/`true`, or unset/`auto`.
pub const MMAP_PREWARM_ENV: &str = "MLX_HY4_MMAP_PREWARM";
/// `1`/`true` forces `F_NOCACHE` on expert shards too; default lets expert
/// shards use the OS page cache while keeping the Engram shard `F_NOCACHE`.
pub const SHARD1_NOCACHE_ENV: &str = "MLX_HY4_SHARD1_NOCACHE";
/// Lowers the admitted prefill chunk (a multiple of 256).
pub const CHUNK_ENV: &str = "MLX_HY4_PREFILL_CHUNK";
/// Overrides the mmap prefill chunk cap (a multiple of 256; default [`MMAP_PREFILL_CHUNK_MAX`]).
pub const MMAP_CHUNK_ENV: &str = "MLX_HY4_MMAP_PREFILL_CHUNK";
/// `0` leaves the MTP layer out of the admission.
pub const MTP_ENV: &str = "MLX_HY4_MTP";
/// Prefill chunk rows the admission may pick, largest first.
pub const CHUNKS: [usize; 3] = [8192, 4096, 2048];
/// Maximum prefill chunk rows for mmap expert residency so peak Metal scratch
/// stays below macOS's working set limit.
pub const MMAP_PREFILL_CHUNK_MAX: usize = 4096;
/// The context ladder's floor.
pub const MIN_CONTEXT: usize = 16_384;
/// Most lanes a load admits.
pub const LANES_MAX: usize = 4;
/// Overrides MLX's qmv batch limit for every shape (`quantized.cpp`).
pub const SPLITK_ENV: &str = "MLX_QMM_SPLITK_MIN_M";
const CHUNK_ALIGN: usize = 256;

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;
/// Minimum host physical memory to auto-select mmap expert residency.
pub const MMAP_MIN_PHYSICAL: u64 = 100 * GIB;
/// Minimum free host budget required to spawn the background mmap pre-warmer.
pub const MMAP_PREWARM_MIN_BUDGET: u64 = 8 * GIB;
/// Headroom reserved for other processes when evaluating direct full-expert residency.
pub const DIRECT_MARGIN: u64 = 4 * GIB;
/// Upfront KV token allowance for direct full-expert residency; the paged KV
/// pool still serves up to `max_ctx` dynamically as memory allows.
pub const DIRECT_KV_ALLOWANCE_TOKENS: usize = 65_536;
/// RAM ladder entries (window clones and tails).
pub const LADDER_BYTES: u64 = 150 * MIB;
const HY4_COMPUTE_BASE: u64 = 512 * MIB;
const HY4_COMPUTE_PER_ROW: u64 = 320 * 1024;
/// Gathered keys of the sparse attention core.
const SPARSE_BYTES: u64 = 256 * MIB;
/// Indexer score sub-chunks.
const INDEX_BYTES: u64 = 256 * MIB;
/// Per context token: index ids and scores of a prefill chunk.
const COMPUTE_PER_CTX: u64 = 128;
/// Disk-tier host staging and copy queue.
const IO_FIXED: u64 = 160 * MIB;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 store: {msg}"))
}

/// Hy4's admission constants at `kv_per_ctx` KV bytes per token.
pub fn profile(kv_per_ctx: u64) -> Profile {
    Profile {
        cache_env: CACHE_ENV,
        reserve_env: RESERVE_ENV,
        compute_base: HY4_COMPUTE_BASE,
        compute_per_row: HY4_COMPUTE_PER_ROW,
        compute_fixed: SPARSE_BYTES + INDEX_BYTES,
        compute_per_ctx: COMPUTE_PER_CTX,
        kv_base: DSV41.kv_base + LADDER_BYTES,
        kv_per_ctx,
        io_base: DSV41.io_base,
        io_per_row: DSV41.io_per_row,
        io_fixed: IO_FIXED,
        second_staging_min_cache: DSV41.second_staging_min_cache,
    }
}

/// KV bytes per context token: the f16 latent row and the f16 pooled index
/// keys of every indexed layer (the MTP layer's with `mtp`), and the u32 id.
pub fn kv_per_token(cfg: &HyV4FlashConfig, mtp: bool) -> u64 {
    let layers = if mtp { &cfg.layers[..] } else { cfg.trunk() };
    let n = layers.iter().filter(|l| l.attn == Attn::Indexed).count() as u64;
    let row = cfg.mla.kv_row() as u64;
    let dim = cfg.indexer.head_dim as u64;
    let kpool = cfg.indexer.kpool.max(1) as u64;
    n * row * 2 + n * dim * 2 / kpool + 4
}

/// `profile` with `lanes - 1` lane floors added to the KV base.
pub fn lane_profile(p: &Profile, lanes: usize, floor: u64) -> Profile {
    Profile {
        kv_base: p.kv_base + lanes.saturating_sub(1) as u64 * floor,
        ..*p
    }
}

/// Bytes a lane holds beside the shared KV allowance: the f16 window ring
/// (`window + WINDOW_SLACK` rows) of every sliding layer, the f32 indexer
/// tail (`kpool - 1 + WINDOW_SLACK` rows) of every indexed layer (the MTP
/// layer's with `mtp`) and, with `mtp`, the `SEEDS` f32 seed rows of the
/// live draft.
pub fn lane_floor(cfg: &HyV4FlashConfig, mtp: bool) -> u64 {
    let layers = if mtp { &cfg.layers[..] } else { cfg.trunk() };
    let count = |a: Attn| layers.iter().filter(|l| l.attn == a).count() as u64;
    let windows =
        count(Attn::Sliding) * (cfg.window + WINDOW_SLACK) as u64 * cfg.mla.kv_row() as u64 * 2;
    let tail = (cfg.indexer.kpool.saturating_sub(1) + WINDOW_SLACK) as u64;
    let tails = count(Attn::Indexed) * tail * cfg.indexer.head_dim as u64 * 4;
    let seeds = if mtp {
        SEEDS as u64 * cfg.dim as u64 * 4
    } else {
        0
    };
    windows + tails + seeds
}

/// `lane_floor` and, with `mtp`, the pooled index keys of the draft's MTP
/// layer for `ctx` positions, which its first write allocates whole: what
/// one lane past the first adds beside the shared KV allowance.
pub fn lane_bytes(cfg: &HyV4FlashConfig, mtp: bool, ctx: usize) -> u64 {
    let pooled = if mtp {
        (ctx / cfg.indexer.kpool.max(1)) as u64 * cfg.indexer.head_dim as u64 * 2
    } else {
        0
    };
    lane_floor(cfg, mtp) + pooled
}

/// The most lanes up to `want` (at least 1) whose admission `admit(lanes)`
/// passes, with that admission; one lane's refusal otherwise.
pub fn plan_lanes<T>(want: usize, mut admit: impl FnMut(usize) -> Result<T>) -> Result<(usize, T)> {
    let mut lanes = want.max(1);
    loop {
        match admit(lanes) {
            Ok(t) => return Ok((lanes, t)),
            Err(e) if lanes == 1 => return Err(e),
            Err(e) => {
                tracing::info!(
                    "hy_v4_flash admission refused {lanes} lanes ({}); trying {}",
                    e.reason,
                    lanes - 1
                );
                lanes -= 1;
            }
        }
    }
}

/// The most lanes up to `want` admitted (`admit(lanes)`) at the context and
/// chunk of `one`, the one-lane pick.
pub fn lanes_at(
    one: Pick,
    want: usize,
    mut admit: impl FnMut(usize) -> Result<Admission>,
) -> (usize, Pick) {
    plan_lanes(want, |n| match n {
        1 => Ok(one),
        _ => admit(n).map(|admission| Pick { admission, ..one }),
    })
    .unwrap_or((1, one))
}

/// MLX's `get_qmv_batch_limit(D, O)` (`backend/metal/quantized.cpp`) on the
/// Metal architecture `arch` (`applegpu_g16s`: generation from the two
/// digits before the size letter, as `metal/device.cpp`): from this many
/// rows a transposed `quantized_matmul` of `k` inputs and `n` outputs
/// leaves the matvec kernels.
pub fn qmv_batch_limit(arch: &str, k: usize, n: usize) -> usize {
    let b = arch.as_bytes();
    let digit = |back: usize| {
        b.len()
            .checked_sub(back)
            .and_then(|i| b.get(i))
            .filter(|c| c.is_ascii_digit())
            .map_or(0, |c| usize::from(c - b'0'))
    };
    let generation = if b.len() >= 3 {
        digit(3) * 10 + digit(2)
    } else {
        0
    };
    let size = b.last().copied().unwrap_or(0);
    let class = if k <= 2048 && n <= 2048 {
        0
    } else if k <= 4096 && n <= 4096 {
        1
    } else {
        2
    };
    let table: [usize; 3] = if generation >= 17 && size != b'd' {
        [33, 25, 13]
    } else if generation >= 15 && size != b'd' {
        [13, 15, 13]
    } else if size == b'd' {
        [32, 18, 12]
    } else if generation >= 13 {
        [
            14,
            10,
            if generation == 13 && size == b's' {
                7
            } else {
                6
            },
        ]
    } else {
        [18, 12, 10]
    };
    table[class]
}

/// `MLX_QMM_SPLITK_MIN_M` read as MLX does (`std::atoi`, used when > 0).
pub fn splitk_min_m(raw: Option<&str>) -> Option<usize> {
    let s = raw?.trim_start();
    let (neg, rest) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    let end = rest
        .bytes()
        .position(|c| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    let v = rest[..end].parse::<i32>().ok()?;
    (!neg && v > 0).then_some(v as usize)
}

/// `(inputs, outputs)` of every matrix the model runs through
/// `quantized_matmul`: Q8_0 held as `Dense::Qmm` (a rank-3 tensor per head
/// group), their load-time row-fused projections, and with `mtp` the MTP
/// layer's BF16 matrices, held the same way.
pub fn q8_shapes(scan: &Scan, mtp: bool) -> Result<Vec<(usize, usize)>> {
    let mut shapes = Vec::new();
    for (role, held) in [(Role::Dense, Form::Qmm), (Role::Mtp, Form::Bf16)] {
        if role == Role::Mtp && !mtp {
            continue;
        }
        for (_, loc) in scan.with_role(role) {
            if let [k, n, ..] = loc.info.dims[..]
                && form(role, &loc.info)? == held
            {
                shapes.push((k as usize, n as usize));
            }
        }
    }
    let qmm_2d = |role: Role, held: Form, name: &str| -> Result<Option<(usize, usize)>> {
        let Some(entry) = scan.tensors.get(name) else {
            return Ok(None);
        };
        if entry.role != role {
            return Ok(None);
        }
        if let [k, n] = entry.loc.info.dims[..]
            && form(role, &entry.loc.info)? == held
        {
            return Ok(Some((k as usize, n as usize)));
        }
        Ok(None)
    };
    let push_fused = |shapes: &mut Vec<(usize, usize)>,
                      role: Role,
                      held: Form,
                      il: usize,
                      names: &[&str]|
     -> Result<()> {
        let mut k_common = None;
        let mut n_sum = 0usize;
        for s in names {
            let name = format!("blk.{il}.{s}.weight");
            let Some((k, n)) = qmm_2d(role, held, &name)? else {
                return Ok(());
            };
            match k_common {
                None => k_common = Some(k),
                Some(prev) if prev != k => return Ok(()),
                _ => {}
            }
            n_sum += n;
        }
        if let Some(k) = k_common {
            shapes.push((k, n_sum));
        }
        Ok(())
    };
    for il in 0..scan.cfg.layers.len() {
        if il >= scan.cfg.n_layers && !mtp {
            continue;
        }
        let (role, held) = if il < scan.cfg.n_layers {
            (Role::Dense, Form::Qmm)
        } else {
            (Role::Mtp, Form::Bf16)
        };
        push_fused(&mut shapes, role, held, il, &["attn_q_a", "attn_kv_a_mqa"])?;
        push_fused(
            &mut shapes,
            role,
            held,
            il,
            &["attn_q_a", "attn_kv_a_mqa", "attn_gate"],
        )?;
        push_fused(
            &mut shapes,
            role,
            held,
            il,
            &["ffn_routed_down", "ffn_gate_shexp", "ffn_up_shexp"],
        )?;
        push_fused(&mut shapes, role, held, il, &["ffn_gate", "ffn_up"])?;
    }
    shapes.sort_unstable();
    shapes.dedup();
    Ok(shapes)
}

/// Rows a decode wave may hold: one below the smallest qmv batch limit of
/// `shapes` on `arch` (`splitk` overrides every limit), so every Q8 call of
/// 2..=cap rows runs `qmv_wide` on the generations that have it. Below 2
/// no wave fits.
pub fn row_cap(arch: &str, splitk: Option<usize>, shapes: &[(usize, usize)]) -> Result<usize> {
    shapes
        .iter()
        .map(|&(k, n)| splitk.unwrap_or_else(|| qmv_batch_limit(arch, k, n)))
        .min()
        .map(|m| m.saturating_sub(1))
        .ok_or_else(|| bad("no Q8 matrix to size the wave by"))
}

/// The Metal architecture name MLX's kernels route by (it honours
/// `MLX_METAL_GPU_ARCH`).
fn routing_arch() -> Result<String> {
    let mut buf = [0 as std::ffi::c_char; 64];
    // SAFETY: `buf` is writable for its length; the bridge writes a
    // NUL-terminated name into it or returns false.
    let ok = unsafe { mlx_sys::mlx_test_metal_architecture(false, buf.as_mut_ptr(), buf.len()) };
    if !ok {
        return Err(bad("no Metal architecture name"));
    }
    let name: Vec<u8> = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8(name).map_err(|e| bad(format!("Metal architecture name: {e}")))
}

/// Resident bytes of the MTP layer as it is held: routed experts affine-4
/// g64 (bf16 scales and biases), other BF16 matrices symmetric Q8_0 g32
/// (`Dense::SymQ8`), the rest as `weights::resident_bytes`, plus the optional
/// draft-only SymQ8 LM head when enabled. 0 without an MTP layer.
pub fn mtp_bytes(scan: &Scan) -> Result<u64> {
    let add = |acc: u64, b: u64| {
        acc.checked_add(b)
            .ok_or_else(|| bad("MTP bytes overflow u64"))
    };
    let mut n = 0u64;
    for (_, loc) in scan.with_role(Role::Mtp) {
        let b = match form(Role::Mtp, &loc.info)? {
            Form::Bf16 => {
                let v = loc.info.num_elements()?;
                v + v / 16
            }
            _ => resident_bytes(Role::Mtp, &loc.info)?,
        };
        n = add(n, b)?;
    }
    if let Some(l) = scan.mtp_experts() {
        for loc in [&l.gate, &l.up, &l.down] {
            let v = loc.info.num_elements()?;
            n = add(n, v / 2 + v / 16)?;
        }
    }
    if n > 0 {
        n = add(n, super::mtp::draft_head_bytes(scan)?)?;
    }
    Ok(n)
}

/// Layers and bank bytes of one expert per streamed class (`CLASSES` order).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Classes {
    pub layers: [usize; 2],
    pub slot: [u64; 2],
}

impl Classes {
    pub fn of(scan: &Scan) -> Result<Self> {
        let classes: Vec<_> = scan
            .trunk_experts()
            .iter()
            .map(|l| l.as_ref().map(|l| l.class))
            .collect();
        let layout = ClassLayout::new(&classes)?;
        let mut slot = [0u64; 2];
        for (s, &c) in slot.iter_mut().zip(&CLASSES) {
            *s = slot_bytes(&scan.cfg.moe, c)?;
            if *s == 0 {
                return Err(bad(format!("{c:?}: empty expert slot")));
            }
        }
        Ok(Self {
            layers: [layout.layers[0].len(), layout.layers[1].len()],
            slot,
        })
    }

    /// Cache slots per class for `bytes`, in proportion to the layer counts
    /// at each class's own slot size; the rest goes to the second class.
    pub fn split(&self, bytes: u64) -> [usize; 2] {
        let [la, lb] = self.layers.map(|l| l as u128);
        let [sa, sb] = self.slot;
        let w = la * u128::from(sa) + lb * u128::from(sb);
        if w == 0 || sa == 0 || sb == 0 {
            return [0, 0];
        }
        let a = (u128::from(bytes) * la / w) as u64;
        let b = if lb == 0 { 0 } else { (bytes - a * sa) / sb };
        [a as usize, b as usize]
    }

    /// Staging slots of every class with layers.
    fn staging_experts(&self, n_experts: usize) -> u64 {
        self.layers.iter().filter(|&&l| l > 0).count() as u64 * n_experts as u64
    }
}

/// What the admission charges, from the headers alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sizes {
    /// Resident trunk arrays (`Role::Dense`).
    pub dense_trunk: u64,
    /// The MTP layer as held (`mtp_bytes`).
    pub dense_mtp: u64,
    pub classes: Classes,
    pub n_experts: usize,
}

impl Sizes {
    pub fn of(scan: &Scan) -> Result<Self> {
        Ok(Self {
            dense_trunk: role_resident(scan, Role::Dense)?,
            dense_mtp: mtp_bytes(scan)?,
            classes: Classes::of(scan)?,
            n_experts: scan.cfg.moe.n_experts,
        })
    }

    /// One staging layer covers both classes (`n_experts` slots each) at the
    /// larger slot.
    pub fn model(&self, mtp: bool, ctx: usize) -> ModelSize {
        ModelSize {
            dense: self.dense_trunk + if mtp { self.dense_mtp } else { 0 },
            slot: self.classes.slot[0].max(self.classes.slot[1]),
            n_experts: self.classes.staging_experts(self.n_experts),
            ctx: ctx as u64,
        }
    }

    /// Exact MLX bytes of all trunk MoE layers held directly (`n_experts` slots per layer).
    pub fn direct_expert_bytes(&self) -> u64 {
        let n = self.n_experts as u64;
        self.classes.layers[0] as u64 * n * self.classes.slot[0]
            + self.classes.layers[1] as u64 * n * self.classes.slot[1]
    }

    /// Total routed experts per class across all trunk MoE layers.
    pub fn total_experts(&self) -> [usize; 2] {
        [
            self.classes.layers[0] * self.n_experts,
            self.classes.layers[1] * self.n_experts,
        ]
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DirectMode {
    Off,
    On,
    #[default]
    Auto,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MmapMode {
    Off,
    On,
    #[default]
    Auto,
}

fn parse_tri_mode(env_name: &str, raw: Option<&str>) -> Result<MmapMode> {
    let Some(v) = raw.map(str::trim).filter(|v| !v.is_empty()) else {
        return Ok(MmapMode::Auto);
    };
    if v.eq_ignore_ascii_case("auto") {
        Ok(MmapMode::Auto)
    } else if v == "1" || v.eq_ignore_ascii_case("true") {
        Ok(MmapMode::On)
    } else if v == "0" || v.eq_ignore_ascii_case("false") {
        Ok(MmapMode::Off)
    } else {
        Err(bad(format!(
            "{env_name}={v:?}: expected auto, 1/true, or 0/false"
        )))
    }
}

/// Parses `MLX_HY4_MMAP`: `0`/`false` disables mmap residency, `1`/`true`
/// requires it, unset/blank/`auto` selects it when host memory and Metal admit it.
pub fn mmap_mode(raw: Option<&str>) -> Result<MmapMode> {
    parse_tri_mode(MMAP_ENV, raw)
}

/// Parses `MLX_HY4_MMAP_PREWARM`: `0`/`false` disables background page pre-warm,
/// `1`/`true` or unset/blank/`auto` enables it when host headroom >= 8 GiB.
pub fn mmap_prewarm_mode(raw: Option<&str>) -> Result<MmapMode> {
    parse_tri_mode(MMAP_PREWARM_ENV, raw)
}

/// Parses `MLX_HY4_SHARD1_NOCACHE`: `1`/`true` forces `F_NOCACHE` on expert shards.
pub fn shard1_nocache(raw: Option<&str>) -> bool {
    raw.map(str::trim)
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// Per-shard OS page-cache flags for [`ShardSet::open_with_cache`]: expert
/// shards allow clean file-backed caching unless `nocache` is set, and the
/// Engram table shard always stays `F_NOCACHE`.
pub fn shard_page_cache(scan: &Scan, nocache: bool) -> Vec<bool> {
    shard_page_cache_for(
        scan.paths.len(),
        scan.trunk_experts(),
        scan.engram_table.shard,
        nocache,
    )
}

pub(crate) fn shard_page_cache_for(
    n_shards: usize,
    experts: &[Option<super::scan::ExpertLayer>],
    engram_shard: usize,
    nocache: bool,
) -> Vec<bool> {
    let mut allow = vec![false; n_shards];
    if !nocache {
        for layer in experts.iter().flatten() {
            for s in [layer.gate.shard, layer.up.shard, layer.down.shard] {
                if let Some(slot) = allow.get_mut(s) {
                    *slot = true;
                }
            }
        }
        if let Some(slot) = allow.get_mut(engram_shard) {
            *slot = false;
        }
    }
    allow
}

/// Parses `MLX_HY4_DIRECT`: `0`/`false` disables direct residency, `1`/`true`
/// requests it, unset/blank/`auto` enables it when host memory admits it.
pub fn direct_mode(raw: Option<&str>) -> Result<DirectMode> {
    let Some(v) = raw.map(str::trim).filter(|v| !v.is_empty()) else {
        return Ok(DirectMode::Auto);
    };
    if v.eq_ignore_ascii_case("auto") {
        Ok(DirectMode::Auto)
    } else if v == "1" || v.eq_ignore_ascii_case("true") {
        Ok(DirectMode::On)
    } else if v == "0" || v.eq_ignore_ascii_case("false") {
        Ok(DirectMode::Off)
    } else {
        Err(bad(format!(
            "{DIRECT_ENV}={v:?}: expected auto, 1/true, or 0/false"
        )))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectDecision {
    pub admitted: bool,
    pub available: u64,
    pub budget: u64,
    pub margin: u64,
    pub dense_bytes: u64,
    pub direct_expert_bytes: u64,
    pub kv_and_scratch: u64,
    pub scratch_bytes: u64,
    pub need: u64,
    pub physical: u64,
    pub host_used: u64,
    pub others: u64,
}

/// Parses an explicit GiB reserve override from `MLX_HY4_ADMIT_RESERVE_GB`.
pub fn parse_reserve_override(raw: Option<&str>) -> Option<u64> {
    let s = raw?.trim();
    if s.is_empty() {
        return None;
    }
    match s.parse::<f64>() {
        Ok(v) if (0.0..=1e6).contains(&v) => Some((v * GIB as f64) as u64),
        _ => None,
    }
}

/// Direct-mode host margin: `DIRECT_MARGIN` unless an explicit reserve override
/// raises it.
pub fn direct_margin(reserve_override: Option<u64>) -> u64 {
    reserve_override.map_or(DIRECT_MARGIN, |r| DIRECT_MARGIN.max(r))
}

/// Upfront KV allowance for direct residency: `min(max_ctx, DIRECT_KV_ALLOWANCE_TOKENS)`
/// tokens plus the profile's fixed KV base.
pub fn direct_kv_allowance(p: &Profile, max_ctx: usize) -> u64 {
    let tokens = max_ctx.min(DIRECT_KV_ALLOWANCE_TOKENS) as u64;
    p.kv_base
        .saturating_add(p.kv_per_ctx.saturating_mul(tokens))
}

/// Direct-mode prefill scratch allowance at `prefill_rows` and `ctx`, using the
/// 1.2x profile compute formula.
pub fn direct_scratch_bytes(p: &Profile, prefill_rows: usize, ctx: usize) -> u64 {
    let raw = p
        .compute_base
        .saturating_add(p.compute_per_row.saturating_mul(prefill_rows as u64))
        .saturating_add(p.compute_fixed)
        .saturating_add(p.compute_per_ctx.saturating_mul(ctx as u64));
    raw.saturating_add(raw / 5)
}

/// Caps the Metal wired-limit request at `working_set()` so macOS never
/// rejects or aborts when direct residency exceeds the Metal working set.
pub fn direct_wired_limit(wired_want: u64, ws: u64) -> u64 {
    wired_want.min(ws)
}

#[allow(clippy::too_many_arguments)]
pub fn direct_gate(
    mode: DirectMode,
    host: &HostMemory,
    own: &Own,
    dense_bytes: u64,
    direct_expert_bytes: u64,
    kv_allowance_initial: u64,
    scratch_bytes: u64,
    margin: u64,
    max_memory: Option<u64>,
    cache_cap: Option<u64>,
) -> DirectDecision {
    let host_used = host.non_reclaimable();
    let others = host_used.saturating_sub(own.bytes());
    let available = host.physical.saturating_sub(others).saturating_sub(margin);
    let budget = max_memory.map_or(available, |m| available.min(m));
    let kv_and_scratch = kv_allowance_initial.saturating_add(scratch_bytes);
    let need = dense_bytes
        .saturating_add(direct_expert_bytes)
        .saturating_add(kv_and_scratch);
    let cache_ok = cache_cap.is_none_or(|cap| cap >= direct_expert_bytes);
    let admitted = mode != DirectMode::Off && direct_expert_bytes > 0 && cache_ok && budget >= need;
    DirectDecision {
        admitted,
        available,
        budget,
        margin,
        dense_bytes,
        direct_expert_bytes,
        kv_and_scratch,
        scratch_bytes,
        need,
        physical: host.physical,
        host_used,
        others,
    }
}

/// Evaluates direct residency across `CHUNKS` (largest first, capped at `ctx`)
/// using [`direct_kv_allowance`] (`min(ctx, DIRECT_KV_ALLOWANCE_TOKENS)`).
/// Returns the largest admitted chunk and its decision, or the smallest chunk's
/// refused decision when none fits.
#[allow(clippy::too_many_arguments)]
pub fn pick_direct(
    mode: DirectMode,
    p: &Profile,
    ctx: usize,
    host: &HostMemory,
    own: &Own,
    dense_bytes: u64,
    direct_expert_bytes: u64,
    reserve_override: Option<u64>,
    max_memory: Option<u64>,
    cache_cap: Option<u64>,
) -> (usize, DirectDecision) {
    let margin = direct_margin(reserve_override);
    let kv_allowance_initial = direct_kv_allowance(p, ctx);
    let eval = |chunk: usize| {
        let rows = chunk.min(ctx);
        let scratch = direct_scratch_bytes(p, rows, ctx);
        let gate = direct_gate(
            mode,
            host,
            own,
            dense_bytes,
            direct_expert_bytes,
            kv_allowance_initial,
            scratch,
            margin,
            max_memory,
            cache_cap,
        );
        (rows, gate)
    };
    for &chunk in &CHUNKS[..CHUNKS.len() - 1] {
        let (rows, gate) = eval(chunk);
        if gate.admitted {
            return (rows, gate);
        }
    }
    eval(CHUNKS[CHUNKS.len() - 1])
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MmapDecision {
    pub admitted: bool,
    pub available: u64,
    pub budget: u64,
    pub margin: u64,
    pub dense_bytes: u64,
    pub mapped_bytes: u64,
    pub kv_and_scratch: u64,
    pub scratch_bytes: u64,
    pub need: u64,
    pub physical: u64,
    pub host_used: u64,
    pub others: u64,
}

#[allow(clippy::too_many_arguments)]
pub fn mmap_gate(
    mode: MmapMode,
    metal_available: bool,
    has_experts: bool,
    host: &HostMemory,
    own: &Own,
    dense_bytes: u64,
    mapped_bytes: u64,
    kv_allowance_initial: u64,
    scratch_bytes: u64,
    margin: u64,
    max_memory: Option<u64>,
) -> MmapDecision {
    let host_used = host.non_reclaimable();
    let others = host_used.saturating_sub(own.bytes());
    let available = host.physical.saturating_sub(others).saturating_sub(margin);
    let budget = max_memory.map_or(available, |m| available.min(m));
    let kv_and_scratch = kv_allowance_initial.saturating_add(scratch_bytes);
    let base_need = dense_bytes.saturating_add(kv_and_scratch);
    let need = match mode {
        MmapMode::Auto => base_need.saturating_add(mapped_bytes),
        MmapMode::On | MmapMode::Off => base_need,
    };
    let physical_ok = match mode {
        MmapMode::Off => false,
        MmapMode::On => true,
        MmapMode::Auto => host.physical >= MMAP_MIN_PHYSICAL,
    };
    let admitted = metal_available && has_experts && physical_ok && budget >= need;
    MmapDecision {
        admitted,
        available,
        budget,
        margin,
        dense_bytes,
        mapped_bytes,
        kv_and_scratch,
        scratch_bytes,
        need,
        physical: host.physical,
        host_used,
        others,
    }
}

/// Evaluates mmap residency across `CHUNKS` capped at
/// `ctx.min(MMAP_PREFILL_CHUNK_MAX)` using [`direct_kv_allowance`]
/// (`min(ctx, DIRECT_KV_ALLOWANCE_TOKENS)`). Returns the largest admitted chunk
/// and its decision, or the smallest evaluated chunk's refused decision.
#[allow(clippy::too_many_arguments)]
pub fn pick_mmap(
    mode: MmapMode,
    metal_available: bool,
    has_experts: bool,
    p: &Profile,
    ctx: usize,
    host: &HostMemory,
    own: &Own,
    dense_bytes: u64,
    mapped_bytes: u64,
    reserve_override: Option<u64>,
    max_memory: Option<u64>,
) -> (usize, MmapDecision) {
    pick_mmap_with_max(
        mode,
        metal_available,
        has_experts,
        p,
        ctx,
        host,
        own,
        dense_bytes,
        mapped_bytes,
        MMAP_PREFILL_CHUNK_MAX,
        reserve_override,
        max_memory,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn pick_mmap_with_max(
    mode: MmapMode,
    metal_available: bool,
    has_experts: bool,
    p: &Profile,
    ctx: usize,
    host: &HostMemory,
    own: &Own,
    dense_bytes: u64,
    mapped_bytes: u64,
    chunk_max: usize,
    reserve_override: Option<u64>,
    max_memory: Option<u64>,
) -> (usize, MmapDecision) {
    let margin = direct_margin(reserve_override);
    let kv_allowance_initial = direct_kv_allowance(p, ctx);
    let cap = ctx.min(chunk_max);
    let eval = |chunk: usize| {
        let rows = chunk.min(cap);
        let scratch = direct_scratch_bytes(p, rows, ctx);
        let gate = mmap_gate(
            mode,
            metal_available,
            has_experts,
            host,
            own,
            dense_bytes,
            mapped_bytes,
            kv_allowance_initial,
            scratch,
            margin,
            max_memory,
        );
        (rows, gate)
    };
    let mut cands = Vec::with_capacity(CHUNKS.len() + 1);
    if !CHUNKS.contains(&chunk_max) {
        cands.push(chunk_max);
    }
    for &chunk in &CHUNKS {
        if chunk <= chunk_max {
            cands.push(chunk);
        }
    }
    let mut last = None;
    for chunk in cands {
        let res = eval(chunk);
        if res.1.admitted {
            return res;
        }
        last = Some(res);
    }
    last.unwrap_or_else(|| eval(chunk_max))
}

/// Selected expert residency mode at load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertResidency {
    Mmap {
        chunk: usize,
        gate: MmapDecision,
    },
    Direct {
        chunk: usize,
        gate: DirectDecision,
    },
    Streamed {
        direct_refusal: Option<DirectDecision>,
    },
}

/// Resolves expert residency precedence between `MLX_HY4_MMAP` and
/// `MLX_HY4_DIRECT`. Fails loud only when `MLX_HY4_MMAP=1` cannot be admitted.
#[allow(clippy::too_many_arguments)]
pub fn select_expert_mode(
    mmap: MmapMode,
    direct: DirectMode,
    metal_available: bool,
    p: &Profile,
    ctx: usize,
    host: &HostMemory,
    own: &Own,
    dense_bytes: u64,
    direct_expert_bytes: u64,
    mapped_bytes: u64,
    reserve_override: Option<u64>,
    max_memory: Option<u64>,
    cache_cap: Option<u64>,
) -> Result<ExpertResidency> {
    select_expert_mode_for(
        mmap,
        direct,
        metal_available,
        p,
        ctx,
        host,
        own,
        dense_bytes,
        dense_bytes,
        direct_expert_bytes,
        mapped_bytes,
        MMAP_PREFILL_CHUNK_MAX,
        reserve_override,
        max_memory,
        cache_cap,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn select_expert_mode_for(
    mmap: MmapMode,
    direct: DirectMode,
    metal_available: bool,
    p: &Profile,
    ctx: usize,
    host: &HostMemory,
    own: &Own,
    dense_bytes: u64,
    mmap_dense_bytes: u64,
    direct_expert_bytes: u64,
    mapped_bytes: u64,
    mmap_chunk_max: usize,
    reserve_override: Option<u64>,
    max_memory: Option<u64>,
    cache_cap: Option<u64>,
) -> Result<ExpertResidency> {
    let has_experts = direct_expert_bytes > 0 && mapped_bytes > 0;
    let try_mmap = |mode: MmapMode| {
        pick_mmap_with_max(
            mode,
            metal_available,
            has_experts,
            p,
            ctx,
            host,
            own,
            mmap_dense_bytes,
            mapped_bytes,
            mmap_chunk_max,
            reserve_override,
            max_memory,
        )
    };
    let try_direct = |mode: DirectMode| {
        pick_direct(
            mode,
            p,
            ctx,
            host,
            own,
            dense_bytes,
            direct_expert_bytes,
            reserve_override,
            max_memory,
            cache_cap,
        )
    };

    match mmap {
        MmapMode::On => {
            if !metal_available {
                return Err(bad(format!("{MMAP_ENV}=1 requires Metal GPU support")));
            }
            if !has_experts {
                return Err(bad(format!("{MMAP_ENV}=1: model has no routed experts")));
            }
            let (chunk, gate) = try_mmap(MmapMode::On);
            if gate.admitted {
                Ok(ExpertResidency::Mmap { chunk, gate })
            } else {
                Err(bad(format!(
                    "{MMAP_ENV}=1 refused: need {:.2} GiB (dense {:.2} GiB + kv/scratch {:.2} GiB) \
                     vs budget {:.2} GiB (available {:.2} GiB, others {:.2} GiB, margin {:.2} GiB)",
                    gate.need as f64 / GIB as f64,
                    gate.dense_bytes as f64 / GIB as f64,
                    gate.kv_and_scratch as f64 / GIB as f64,
                    gate.budget as f64 / GIB as f64,
                    gate.available as f64 / GIB as f64,
                    gate.others as f64 / GIB as f64,
                    gate.margin as f64 / GIB as f64,
                )))
            }
        }
        MmapMode::Off => {
            let (chunk, gate) = try_direct(direct);
            if gate.admitted {
                Ok(ExpertResidency::Direct { chunk, gate })
            } else {
                Ok(ExpertResidency::Streamed {
                    direct_refusal: (direct == DirectMode::On).then_some(gate),
                })
            }
        }
        MmapMode::Auto => match direct {
            DirectMode::On => {
                let (d_chunk, d_gate) = try_direct(DirectMode::On);
                if d_gate.admitted {
                    return Ok(ExpertResidency::Direct {
                        chunk: d_chunk,
                        gate: d_gate,
                    });
                }
                let (m_chunk, m_gate) = try_mmap(MmapMode::Auto);
                if m_gate.admitted {
                    return Ok(ExpertResidency::Mmap {
                        chunk: m_chunk,
                        gate: m_gate,
                    });
                }
                Ok(ExpertResidency::Streamed {
                    direct_refusal: Some(d_gate),
                })
            }
            DirectMode::Auto => {
                let (m_chunk, m_gate) = try_mmap(MmapMode::Auto);
                if m_gate.admitted {
                    return Ok(ExpertResidency::Mmap {
                        chunk: m_chunk,
                        gate: m_gate,
                    });
                }
                let (d_chunk, d_gate) = try_direct(DirectMode::Auto);
                if d_gate.admitted {
                    return Ok(ExpertResidency::Direct {
                        chunk: d_chunk,
                        gate: d_gate,
                    });
                }
                Ok(ExpertResidency::Streamed {
                    direct_refusal: None,
                })
            }
            DirectMode::Off => {
                let (m_chunk, m_gate) = try_mmap(MmapMode::Auto);
                if m_gate.admitted {
                    return Ok(ExpertResidency::Mmap {
                        chunk: m_chunk,
                        gate: m_gate,
                    });
                }
                Ok(ExpertResidency::Streamed {
                    direct_refusal: None,
                })
            }
        },
    }
}

/// Computes the clean page-cache pre-warm budget for mmap experts; returns 0
/// when disabled or when headroom is below [`MMAP_PREWARM_MIN_BUDGET`].
pub fn mmap_warm_budget(
    mode: MmapMode,
    host: &HostMemory,
    dense_bytes: u64,
    kv_bytes: u64,
    margin: u64,
    mapped_bytes: u64,
) -> u64 {
    if mode == MmapMode::Off || mapped_bytes == 0 {
        return 0;
    }
    let budget = host
        .physical
        .saturating_sub(host.non_reclaimable())
        .saturating_sub(dense_bytes.saturating_add(kv_bytes).saturating_add(margin))
        .min(mapped_bytes);
    if budget < MMAP_PREWARM_MIN_BUDGET {
        0
    } else {
        budget
    }
}

/// Exact GGUF file bytes of all trunk routed-expert tensors in `scan`.
pub fn mmap_expert_bytes(scan: &Scan) -> Result<u64> {
    let classes = Classes::of(scan)?;
    let n = scan.cfg.moe.n_experts as u64;
    let f0 = file_bytes(&scan.cfg.moe, CLASSES[0])?;
    let f1 = file_bytes(&scan.cfg.moe, CLASSES[1])?;
    Ok((classes.layers[0] as u64)
        .saturating_mul(n)
        .saturating_mul(f0)
        .saturating_add(
            (classes.layers[1] as u64)
                .saturating_mul(n)
                .saturating_mul(f1),
        ))
}

/// The admitted context, prefill chunk and admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pick {
    pub ctx: usize,
    /// At most `ctx`.
    pub chunk: usize,
    pub admission: Admission,
}

/// The context ladder and the chunk rule over `admit_with`; a chunk is
/// charged at most `ctx` rows. Errs with the last refusal at the floor.
pub fn pick(
    p: &Profile,
    base: &ModelSize,
    max_ctx: usize,
    limits: &Limits,
    host: &HostMemory,
    own: &Own,
    ws: u64,
) -> Result<Pick> {
    let smallest = CHUNKS[CHUNKS.len() - 1];
    let mut ctx = max_ctx;
    loop {
        let m = ModelSize {
            ctx: ctx as u64,
            ..*base
        };
        let mut fallback = None;
        let mut refusal = None;
        for chunk in CHUNKS {
            match admit_with(p, &m, chunk.min(ctx) as u64, limits, host, own, ws) {
                Ok(a) if a.cache_bytes() >= p.second_staging_min_cache => {
                    return Ok(Pick {
                        ctx,
                        chunk: chunk.min(ctx),
                        admission: a,
                    });
                }
                Ok(a) if chunk == smallest => fallback = Some(a),
                Ok(_) => {}
                Err(e) => refusal = Some(e),
            }
        }
        if let Some(admission) = fallback {
            return Ok(Pick {
                ctx,
                chunk: smallest.min(ctx),
                admission,
            });
        }
        let why = refusal.map_or_else(String::new, |e| e.reason);
        let next = ctx / 2;
        if next < MIN_CONTEXT {
            return Err(bad(format!("refused at context {ctx}: {why}")));
        }
        tracing::info!("hy_v4_flash admission refused at context {ctx} ({why}); trying {next}");
        ctx = next;
    }
}

/// `MLX_HY4_PREFILL_CHUNK`: unset or blank = no cap.
pub fn chunk_cap(raw: Option<&str>) -> Result<Option<usize>> {
    let Some(v) = raw.map(str::trim).filter(|v| !v.is_empty()) else {
        return Ok(None);
    };
    match v.parse::<usize>() {
        Ok(c) if c >= CHUNK_ALIGN && c % CHUNK_ALIGN == 0 => Ok(Some(c)),
        _ => Err(bad(format!(
            "{CHUNK_ENV}={v:?}: expected a multiple of {CHUNK_ALIGN}"
        ))),
    }
}

/// `MLX_HY4_MMAP_PREFILL_CHUNK`: unset or blank = [`MMAP_PREFILL_CHUNK_MAX`].
pub fn mmap_prefill_chunk_max(raw: Option<&str>) -> Result<usize> {
    let Some(v) = raw.map(str::trim).filter(|v| !v.is_empty()) else {
        return Ok(MMAP_PREFILL_CHUNK_MAX);
    };
    match v.parse::<usize>() {
        Ok(c) if c >= CHUNK_ALIGN && c.is_multiple_of(CHUNK_ALIGN) => Ok(c),
        _ => Err(bad(format!(
            "{MMAP_CHUNK_ENV}={v:?}: expected a multiple of {CHUNK_ALIGN}"
        ))),
    }
}

/// How many cache slots the bank gets.
#[derive(Clone, Debug, PartialEq)]
pub enum CachePlan {
    /// Staging only, no admission.
    #[cfg(test)]
    Off,
    /// Exactly these cache slots and staging layers, no admission.
    #[cfg(test)]
    Slots { cache: [usize; 2], staging: usize },
    /// Admission at load and at every turn start, limits from the env.
    Admit,
    /// `Admit` with these limits.
    #[cfg(test)]
    AdmitWith(Limits),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    /// The load option `maxContextTokens`.
    pub max_context: Option<usize>,
    /// Charge the MTP layer (when the file has one).
    pub mtp: bool,
    pub chunk_cap: Option<usize>,
    /// Lanes wanted; the admission may give fewer.
    pub lanes: usize,
    pub plan: CachePlan,
    /// The load option `maxMemoryBytes`: the admission's budget stays
    /// within it.
    pub max_memory: Option<u64>,
    pub direct: DirectMode,
    pub mmap: MmapMode,
    pub mmap_prewarm: MmapMode,
    pub mmap_chunk_max: usize,
}

impl Options {
    /// `serial`: one sequence at a time (`MLX_SERVE_FORCE_SERIAL` at load),
    /// else `min(LANES_MAX, MLX_SCHED_MAX_NUM_SEQS)` lanes.
    pub fn from_env(max_context: Option<usize>, serial: bool) -> Result<Self> {
        let mtp = std::env::var(MTP_ENV).ok().is_none_or(|v| v.trim() != "0");
        let lanes = if serial {
            1
        } else {
            crate::engine::hybrid_scheduler::scheduler_max_num_seqs_for(LANES_MAX)
        };
        Ok(Self {
            max_context,
            mtp,
            chunk_cap: chunk_cap(std::env::var(CHUNK_ENV).ok().as_deref())?,
            lanes,
            plan: CachePlan::Admit,
            max_memory: None,
            direct: direct_mode(std::env::var(DIRECT_ENV).ok().as_deref())?,
            mmap: mmap_mode(std::env::var(MMAP_ENV).ok().as_deref())?,
            mmap_prewarm: mmap_prewarm_mode(std::env::var(MMAP_PREWARM_ENV).ok().as_deref())?,
            mmap_chunk_max: mmap_prefill_chunk_max(std::env::var(MMAP_CHUNK_ENV).ok().as_deref())?,
        })
    }
}

#[cfg(test)]
thread_local! {
    /// The host reading the admission sees on this thread instead of the
    /// real one.
    pub(super) static TEST_HOST: std::cell::Cell<Option<(HostMemory, u64)>> =
        const { std::cell::Cell::new(None) };
    /// The MLX free pool the re-admission sees on this thread instead of the
    /// real one.
    pub(super) static TEST_POOL: std::cell::Cell<Option<u64>> =
        const { std::cell::Cell::new(None) };
    /// The host's purgeable count a turn start reads before its flip to
    /// non-volatile on this thread instead of the real one.
    pub(super) static TEST_BEFORE_FLIP: std::cell::Cell<Option<u64>> =
        const { std::cell::Cell::new(None) };
}

/// The working set an admission may fill: `ws`, at most `max_memory`.
pub fn capped_ws(ws: u64, max_memory: Option<u64>) -> u64 {
    max_memory.map_or(ws, |m| m.min(ws))
}

/// Host memory and the Metal working set for an admission, capped at
/// `max_memory`.
fn host_reading(max_memory: Option<u64>) -> Result<(HostMemory, u64)> {
    #[cfg(test)]
    if let Some((h, ws)) = TEST_HOST.with(std::cell::Cell::get) {
        return Ok((h, capped_ws(ws, max_memory)));
    }
    let h = host_memory().ok_or_else(|| bad("no host memory reading for admission"))?;
    Ok((h, capped_ws(working_set(), max_memory)))
}

/// The host's purgeable count right before a turn start's flip.
fn purgeable_before_flip(max_memory: Option<u64>) -> Result<u64> {
    #[cfg(test)]
    if let Some(p) = TEST_BEFORE_FLIP.with(std::cell::Cell::get) {
        return Ok(p);
    }
    Ok(host_reading(max_memory)?.0.purgeable)
}

/// The MLX free-buffer pool (process-wide).
fn free_pool() -> u64 {
    #[cfg(test)]
    if let Some(f) = TEST_POOL.with(std::cell::Cell::get) {
        return f;
    }
    crate::array::get_cache_memory() as u64
}

/// This model's bytes inside `host_used`: resident arrays, the sessions'
/// buffers `kv` (made at their first write and kept, already charged in
/// `fixed`), the bank's resident pages (volatile between turns) and the MLX
/// free pool up to `pool`, the pool term the admission charges, so those
/// bytes are charged once whichever model of the process freed them.
/// Scratch and the rest of the process footprint stay in `others`: the
/// process-wide counters also hold other loaded models (conservative).
pub fn own(bank: &KqBank, dense: u64, kv: u64, pool: u64) -> Result<Own> {
    Ok(own_of(
        dense,
        kv,
        bank.residency()?.resident,
        bank.in_turn(),
        free_pool(),
        pool,
    ))
}

/// A host reading for a re-admission of this model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reading {
    /// [`settled`] for the bank.
    pub host: HostMemory,
    /// The host with every resident page of a turn's bank out of the
    /// purgeable count: other processes' volatile pages up to the bank's
    /// size count as used. Growth and the size of a shrink are read on it,
    /// so a purgeable count that still holds the bank without the kernel
    /// saying so never grows the bank.
    pub wary: HostMemory,
    pub own: Own,
}

/// [`own`] and the host reading `h` for a re-admission. `lag`: bank bytes
/// the turn start's flip left in the purgeable count ([`flip_lag`]; 0
/// inside a turn).
pub fn reading(
    bank: &KqBank,
    dense: u64,
    kv: u64,
    pool: u64,
    h: HostMemory,
    lag: u64,
) -> Result<Reading> {
    let r = bank.residency()?;
    let in_turn = bank.in_turn();
    let counted = Residency {
        volatile: r.volatile.saturating_add(lag).min(r.resident),
        ..r
    };
    let all = Residency {
        volatile: r.resident,
        ..r
    };
    Ok(Reading {
        host: settled(h, counted, in_turn),
        wary: settled(h, all, in_turn),
        own: own_of(dense, kv, r.resident, in_turn, free_pool(), pool),
    })
}

/// Bank bytes a turn start's flip to non-volatile left in the host's
/// purgeable count (measured: 55.5 GB right after the flip, the bank's
/// 54.9 plus the usual 0.5, with every in-turn reading at 0.3-0.5; 18
/// buffers held volatile for 30 s stayed in the count 0.1-1 s past the
/// flip while the kernel already listed them non-volatile). The
/// count `before` the flip held the `volatile` bank bytes it took out, so
/// what `after` holds past `before - volatile`, up to `volatile`, is the
/// bank's; others' pages that turned volatile in between count as used, at
/// most what they freed. A count below `volatile` did not hold the bank: 0.
pub fn flip_lag(before: u64, after: u64, volatile: u64) -> u64 {
    if before < volatile {
        return 0;
    }
    after.saturating_sub(before - volatile).min(volatile)
}

/// `h` with the pages of a turn's bank the host still counts purgeable
/// (`bank.volatile`) out of that count: the turn counts them as own, and
/// `host_used` leaves purgeable pages out (from the anonymous ones, so with
/// the bank wired other processes' pages vanished, with it anonymous the
/// bank did).
pub fn settled(h: HostMemory, bank: Residency, in_turn: bool) -> HostMemory {
    if !in_turn {
        return h;
    }
    HostMemory {
        purgeable: h.purgeable.saturating_sub(bank.volatile),
        ..h
    }
}

/// [`own`] from its parts.
pub fn own_of(dense: u64, kv: u64, resident: u64, in_turn: bool, free_pool: u64, pool: u64) -> Own {
    Own {
        active: dense + kv + resident,
        pool: free_pool.min(pool),
        volatile: if in_turn { 0 } else { resident },
    }
}

/// Policy parameters for idle-gap promotion from partial streaming to direct
/// full-expert residency when the last open turn closes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectPromotion {
    pub mode: DirectMode,
    pub reserve_override: Option<u64>,
    pub n_experts: usize,
}

impl DirectPromotion {
    fn direct_expert_bytes(&self, classes: &Classes) -> u64 {
        let n = self.n_experts as u64;
        (classes.layers[0] as u64)
            .saturating_mul(n)
            .saturating_mul(classes.slot[0])
            .saturating_add(
                (classes.layers[1] as u64)
                    .saturating_mul(n)
                    .saturating_mul(classes.slot[1]),
            )
    }

    fn total_experts(&self, classes: &Classes) -> [usize; 2] {
        [
            classes.layers[0] * self.n_experts,
            classes.layers[1] * self.n_experts,
        ]
    }
}

struct TurnState {
    admission: Option<Admission>,
    direct_budget: Option<u64>,
    wired: Option<WiredLimitContext>,
    guard: CacheLimitGuard,
    /// Bank switches: (cache slots, seconds to free the old bank and allocate).
    switches: Vec<(usize, f64)>,
    /// The last switch failed: the bank may hold no slots, so the next turn
    /// rebuilds even when the admission is kept.
    lost: bool,
}

/// The per-turn re-admission and wired-limit lease of one bank.
pub struct Lease {
    profile: Profile,
    size: ModelSize,
    rows: u64,
    classes: Classes,
    /// `None`: no admission (tests).
    limits: Option<Limits>,
    max_memory: Option<u64>,
    direct_promotion: Option<DirectPromotion>,
    turn: Mutex<TurnState>,
}

impl Lease {
    /// `admission` is the one `bank` was built for; `rows` the prefill rows
    /// it was charged at.
    pub fn new(
        profile: Profile,
        size: ModelSize,
        rows: u64,
        classes: Classes,
        limits: Option<Limits>,
        admission: Option<Admission>,
        bank: &KqBank,
    ) -> Self {
        Self {
            profile,
            size,
            rows,
            classes,
            limits,
            max_memory: None,
            direct_promotion: None,
            turn: Mutex::new(TurnState {
                admission,
                direct_budget: None,
                wired: None,
                guard: coordinator().register_with_cache_limit(bank.nbytes(), Some(POOL_LIMIT)),
                switches: Vec::new(),
                lost: false,
            }),
        }
    }

    /// Re-admissions stay within `max_memory` too.
    pub fn with_max_memory(mut self, max_memory: Option<u64>) -> Self {
        self.max_memory = max_memory;
        self
    }

    /// Records the host budget for mmap residency without enabling idle-gap
    /// promotion to anonymous `DirectBank`.
    pub fn with_direct_budget(self, direct_budget: Option<u64>) -> Self {
        self.turn().direct_budget = direct_budget;
        self
    }

    /// Enables idle-gap promotion from partial streaming to direct residency
    /// when `end_turn_idle` runs with 0 active turns.
    pub fn with_direct_promotion(
        mut self,
        promotion: DirectPromotion,
        direct_budget: Option<u64>,
    ) -> Self {
        self.direct_promotion = Some(promotion);
        self.turn().direct_budget = direct_budget;
        self
    }

    fn turn(&self) -> MutexGuard<'_, TurnState> {
        self.turn.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn admission(&self) -> Option<Admission> {
        self.turn().admission
    }

    pub fn direct_budget(&self) -> Option<u64> {
        self.turn().direct_budget
    }

    /// Every bank switch so far: (cache slots of both classes, seconds).
    pub fn switches(&self) -> Vec<(usize, f64)> {
        self.turn().switches.clone()
    }

    pub fn in_turn(&self) -> bool {
        self.turn().wired.is_some()
    }

    pub fn wired_limit(&self, bank: &KqBank) -> Option<u64> {
        let a = self.turn().admission?;
        if bank.is_direct() {
            Some(direct_wired_limit(
                a.fixed.saturating_add(bank.nbytes()),
                a.ws,
            ))
        } else {
            Some(a.wired_limit())
        }
    }

    fn switch(&self, t: &mut TurnState, bank: &KqBank, a: &Admission) -> Result<()> {
        let t0 = Instant::now();
        let slots = self.classes.split(a.cache_bytes());
        t.lost = true;
        bank.rebuild(slots, a.staging_layers as usize)?;
        t.lost = false;
        t.guard = coordinator().register_with_cache_limit(bank.nbytes(), Some(POOL_LIMIT));
        let s = t0.elapsed().as_secs_f64();
        let n = slots[0] + slots[1];
        tracing::info!("hy_v4_flash bank switch to {slots:?} cache slots in {s:.2} s");
        t.switches.push((n, s));
        Ok(())
    }

    /// Turn start: make the bank non-volatile, re-admit, switch the bank if
    /// the admission says so, take the wired-limit lease. `kv`: bytes of
    /// every live session (`Session::nbytes`). Errs (bank volatile again)
    /// when the fixed working set alone no longer fits.
    pub fn begin_turn(&self, bank: &KqBank, dense: u64, kv: u64) -> Result<()> {
        let mut t = self.turn();
        if bank.is_direct() {
            if self.limits.is_some()
                && t.wired.is_none()
                && let Some(a) = t.admission
            {
                let wired_bytes = direct_wired_limit(a.fixed.saturating_add(bank.nbytes()), a.ws);
                t.wired = Some(WiredLimitContext::bounded(
                    usize::try_from(wired_bytes).unwrap_or(usize::MAX),
                    Vec::new(),
                ));
            }
            bank.begin_turn()?;
            return Ok(());
        }
        if let Some(limits) = &self.limits
            && t.wired.is_none()
        {
            let cur = t.admission.ok_or_else(|| bad("no admission"))?;
            let before = purgeable_before_flip(self.max_memory)?;
            let flipped = bank.residency()?.volatile;
            // non-volatile first: its resident pages are then in `host_used`
            // and counted as own
            bank.begin_turn()?;
            let next = (|| {
                let (h, ws) = host_reading(self.max_memory)?;
                let lag = flip_lag(before, h.purgeable, flipped);
                let r = reading(bank, dense, kv, cur.terms.pool, h, lag)?;
                let turn = self.readmit(&cur, limits, &r, ws)?;
                let now = || {
                    admit_with(
                        &self.profile,
                        &self.size,
                        self.rows,
                        limits,
                        &r.wary,
                        &r.own,
                        ws,
                    )
                };
                match next_bank(turn, &cur, t.lost, false, now) {
                    None => Ok(cur),
                    Some(a) => {
                        self.switch(&mut t, bank, &a)?;
                        tracing::info!("hy_v4_flash re-admission: {a}");
                        Ok(a)
                    }
                }
            })();
            let next = match next {
                Ok(n) => n,
                Err(e) => {
                    let _ = bank.end_turn();
                    return Err(e);
                }
            };
            t.admission = Some(next);
            t.wired = Some(WiredLimitContext::bounded(
                usize::try_from(next.wired_limit()).unwrap_or(usize::MAX),
                Vec::new(),
            ));
        }
        bank.begin_turn()?;
        Ok(())
    }

    /// Inside a turn (between two evaluated steps): the turn start's
    /// re-admission again, for a host that changed since ([`next_bank`]).
    /// `true` when it switched; `false` outside a turn or without an
    /// admission.
    pub fn recheck(&self, bank: &KqBank, dense: u64, kv: u64) -> Result<bool> {
        if bank.is_direct() {
            return Ok(false);
        }
        let mut t = self.turn();
        let (Some(limits), Some(cur)) = (&self.limits, t.admission) else {
            return Ok(false);
        };
        if t.wired.is_none() {
            return Ok(false);
        }
        let (h, ws) = host_reading(self.max_memory)?;
        let r = reading(bank, dense, kv, cur.terms.pool, h, 0)?;
        let turn = self.readmit(&cur, limits, &r, ws)?;
        let now = || {
            admit_with(
                &self.profile,
                &self.size,
                self.rows,
                limits,
                &r.wary,
                &r.own,
                ws,
            )
        };
        let next = match next_bank(turn, &cur, t.lost, true, now) {
            Some(a) => a,
            None => return Ok(false),
        };
        crate::array::synchronize();
        self.switch(&mut t, bank, &next)?;
        tracing::info!("hy_v4_flash re-admission inside a turn: {next}");
        t.admission = Some(next);
        t.wired = Some(WiredLimitContext::bounded(
            usize::try_from(next.wired_limit()).unwrap_or(usize::MAX),
            Vec::new(),
        ));
        Ok(true)
    }

    /// The re-admission of `cur` on `r.host` ([`wary_growth`]).
    fn readmit(&self, cur: &Admission, limits: &Limits, r: &Reading, ws: u64) -> Result<Turn> {
        let at = |h: &HostMemory| {
            readmit_with(
                &self.profile,
                cur,
                &self.size,
                self.rows,
                limits,
                h,
                &r.own,
                ws,
            )
        };
        Ok(wary_growth(at(&r.host)?, || at(&r.wary)))
    }

    /// Turn end (after the turn's last eval): release the wired lease, make
    /// the bank volatile.
    pub fn end_turn(&self, bank: &KqBank) -> Result<()> {
        let mut t = self.turn();
        crate::array::synchronize();
        t.wired = None;
        bank.end_turn()
    }

    /// Turn end when 0 turns remain open: releases the wired lease, evaluates
    /// idle-gap promotion from partial streaming to direct full-expert
    /// residency, and transitions the bank when both `pick_direct` and live
    /// headroom admit the growth. Returns the admitted direct prefill chunk
    /// when promotion occurred.
    pub fn end_turn_idle(&self, bank: &KqBank, dense: u64, kv: u64) -> Result<Option<usize>> {
        let mut t = self.turn();
        crate::array::synchronize();
        t.wired = None;
        if bank.is_direct() {
            bank.end_turn()?;
            return Ok(None);
        }
        let Some(promo) = self.direct_promotion.filter(|p| p.mode != DirectMode::Off) else {
            bank.end_turn()?;
            return Ok(None);
        };
        let Some(limits) = &self.limits else {
            bank.end_turn()?;
            return Ok(None);
        };
        let direct_expert_bytes = promo.direct_expert_bytes(&self.classes);
        if direct_expert_bytes == 0
            || limits
                .cache_cap
                .is_some_and(|cap| cap < direct_expert_bytes)
        {
            bank.end_turn()?;
            return Ok(None);
        }
        let r = bank.residency()?;
        let current_bank_bytes = r.resident.saturating_sub(r.volatile);
        let ctx = self.size.ctx as usize;
        let eval = || -> Result<(u64, Own, usize, DirectDecision, bool)> {
            let (h, ws) = host_reading(self.max_memory)?;
            let own = Own {
                active: dense.saturating_add(current_bank_bytes).saturating_add(kv),
                pool: 0,
                volatile: 0,
            };
            let (direct_chunk, gate) = pick_direct(
                promo.mode,
                &self.profile,
                ctx,
                &h,
                &own,
                self.size.dense,
                direct_expert_bytes,
                promo.reserve_override,
                self.max_memory,
                limits.cache_cap,
            );
            let growth = direct_expert_bytes.saturating_sub(current_bank_bytes);
            let headroom_ok = gate.admitted
                && h.non_reclaimable()
                    .saturating_add(growth)
                    .saturating_add(gate.margin)
                    <= h.physical
                && self.max_memory.is_none_or(|cap| {
                    dense.saturating_add(direct_expert_bytes).saturating_add(kv) <= cap
                });
            Ok((ws, own, direct_chunk, gate, headroom_ok))
        };
        let evaluated = match eval() {
            Ok(v) => v,
            Err(err) => {
                bank.end_turn()?;
                tracing::warn!("hy_v4_flash idle direct promotion check failed: {err}");
                return Ok(None);
            }
        };
        let (ws, own, direct_chunk, gate, headroom_ok) =
            if evaluated.3.admitted && !evaluated.4 && crate::array::get_cache_memory() > 0.0 {
                crate::array::clear_cache();
                match eval() {
                    Ok(v) => v,
                    Err(err) => {
                        bank.end_turn()?;
                        tracing::warn!("hy_v4_flash idle direct promotion recheck failed: {err}");
                        return Ok(None);
                    }
                }
            } else {
                evaluated
            };
        if !headroom_ok {
            bank.end_turn()?;
            return Ok(None);
        }
        let t0 = Instant::now();
        bank.end_turn()?;
        if let Err(err) = bank.promote_to_direct() {
            if bank.nbytes() == 0 {
                t.lost = true;
            }
            tracing::warn!(
                "hy_v4_flash idle promotion to direct failed, keeping partial bank: {err}"
            );
            return Ok(None);
        }
        t.lost = false;
        t.guard = coordinator().register_with_cache_limit(bank.nbytes(), Some(POOL_LIMIT));
        let total_slots = promo.total_experts(&self.classes);
        let direct_kv = direct_kv_allowance(&self.profile, ctx);
        let direct_fixed = gate.dense_bytes.saturating_add(gate.kv_and_scratch);
        let wired_cap = direct_wired_limit(direct_fixed.saturating_add(direct_expert_bytes), ws);
        let a = Admission {
            physical: gate.physical,
            host_used: gate.host_used,
            own: own.bytes(),
            others: gate.others,
            reserve: gate.margin,
            live: gate.available,
            ws,
            terms: Terms {
                dense: gate.dense_bytes,
                staging_layer: 0,
                kv: direct_kv,
                compute: gate.scratch_bytes,
                io: 0,
                driver: 0,
                pool: 0,
            },
            staging_layers: 0,
            fixed: direct_fixed,
            slot: self.size.slot,
            cache_slots: (total_slots[0] + total_slots[1]) as u64,
            bound: if self.max_memory.is_some_and(|m| m < gate.available) {
                Bound::Env
            } else {
                Bound::Live
            },
        };
        t.admission = Some(a);
        t.direct_budget = Some(gate.budget);
        let elapsed = t0.elapsed().as_secs_f64();
        let n = total_slots[0] + total_slots[1];
        t.switches.push((n, elapsed));
        tracing::info!(
            "hy_v4_flash promoted partial bank to direct residency in {elapsed:.2} s: \
             chunk {direct_chunk}, direct experts {:.2} GiB ({n} slots), others {:.2} GiB, \
             available {:.2} GiB, margin {:.2} GiB, wired cap {:.2} GiB",
            direct_expert_bytes as f64 / GIB as f64,
            gate.others as f64 / GIB as f64,
            gate.available as f64 / GIB as f64,
            gate.margin as f64 / GIB as f64,
            wired_cap as f64 / GIB as f64,
        );
        Ok(Some(direct_chunk))
    }
}

/// `turn` with a rebuild kept unless the wary reading rebuilds too, at its
/// size.
pub fn wary_growth(turn: Turn, wary: impl FnOnce() -> Result<Turn>) -> Turn {
    match turn {
        Turn::Rebuild(_) => match wary() {
            Ok(Turn::Rebuild(a)) => Turn::Rebuild(a),
            _ => Turn::Keep,
        },
        t => t,
    }
}

/// The bank a re-admission's `turn` switches to (`None` keeps it). A bank
/// the host has no room for (`StagingOnly`) shrinks to what the host
/// admits now, `now` (on the wary reading), staging only when that admits
/// no cache or refuses. Growth that would empty a bank holding cache slots
/// waits for a turn start when `in_turn`.
pub fn next_bank(
    turn: Turn,
    cur: &Admission,
    lost: bool,
    in_turn: bool,
    now: impl FnOnce() -> Result<Admission>,
) -> Option<Admission> {
    match turn {
        Turn::Keep if lost => Some(*cur),
        Turn::Keep => None,
        Turn::Rebuild(_) if in_turn && cur.cache_slots > 0 && !lost => None,
        Turn::Rebuild(a) => Some(a),
        Turn::StagingOnly(s) => Some(match now() {
            Ok(a) if a.cache_slots > 0 => a,
            _ => s,
        }),
    }
}

/// Engram rows through a table the store keeps for its read counters.
struct TableRows(Arc<F8Table>);

impl EngramRows for TableRows {
    fn rows(&self, layer: usize, ids: &[u32]) -> Result<Vec<f32>> {
        self.0.rows(layer, ids)
    }

    fn prefetch(&self, layer: usize, ids: &[u32]) -> Result<Option<PrefetchHandle>> {
        self.0.prefetch(layer, ids)
    }

    fn take_prefetched(&self, handle: PrefetchHandle) -> Result<Vec<f32>> {
        self.0.take_prefetched(handle)
    }
}

pub struct Store {
    pub model: Model,
    pub bank: Arc<KqBank>,
    pub table: Arc<F8Table>,
    /// The headers the load read and the one descriptor per shard every
    /// reader of the load goes through.
    pub scan: Scan,
    pub shards: Arc<ShardSet>,
    /// GGUF bytes of the resident tensors read at load.
    pub dense_read: u64,
    /// Bytes of the resident arrays.
    pub dense_bytes: u64,
    pub sizes: Sizes,
    pub size: ModelSize,
    pub profile: Profile,
    /// Tokens one session may hold.
    pub ctx: usize,
    /// Rows of one prefill chunk.
    pub chunk: usize,
    /// Sequences the admission charged.
    pub lanes: usize,
    /// Rows a decode wave may hold (`row_cap`); below 2 no wave fits.
    pub row_cap: usize,
    /// The MTP layer is charged.
    pub mtp: bool,
    /// Seconds to allocate the bank at load (no page touched yet).
    pub bank_alloc_s: f64,
    /// The load option `maxMemoryBytes`.
    pub max_memory: Option<u64>,
    /// Host budget when direct full-expert residency was admitted.
    pub direct_budget: Option<u64>,
    pub lease: Lease,
}

impl Store {
    /// `paths` are the shards in order (shard 0 holds the metadata).
    pub fn open(paths: &[std::path::PathBuf], opts: &Options) -> Result<Self> {
        let scan = Scan::open(paths)?;
        let sizes = Sizes::of(&scan)?;
        let mtp = opts.mtp && scan.mtp.is_some();
        let max_ctx = scan.cfg.max_context(opts.max_context)?;
        let row_cap = row_cap(
            &routing_arch()?,
            splitk_min_m(std::env::var(SPLITK_ENV).ok().as_deref()),
            &q8_shapes(&scan, mtp)?,
        )?;
        let want = if super::sched::paged_mode(false, row_cap) {
            super::sched::lanes(opts.lanes, row_cap)
        } else {
            1
        };
        let one_lane = profile(kv_per_token(&scan.cfg, mtp));
        let base = sizes.model(mtp, max_ctx);
        let (limits, reserve_override) = match &opts.plan {
            CachePlan::Admit => {
                let l = Limits::from_env_with(&one_lane)?;
                let ov =
                    parse_reserve_override(std::env::var(one_lane.reserve_env).ok().as_deref());
                (Some(l), ov)
            }
            #[cfg(test)]
            CachePlan::AdmitWith(l) => {
                let ov = (l.reserve != crate::moe_stream::budget::RESERVE).then_some(l.reserve);
                (Some(*l), ov)
            }
            #[cfg(test)]
            CachePlan::Off | CachePlan::Slots { .. } => (None, None),
        };
        let smallest = CHUNKS[CHUNKS.len() - 1];
        enum BankInit {
            Mmap {
                warm_budget: u64,
                force_prewarm: bool,
            },
            Direct,
            Streamed {
                cache: [usize; 2],
                staging: usize,
            },
        }
        let (mmap_dense_saved, mmap_dense_file) = mmap_dense_saved_bytes(&scan)?;
        let mmap_dense_bytes = base.dense.saturating_sub(mmap_dense_saved);
        let (lanes, ctx, admitted, admission, direct_budget, bank_init) = match (
            &limits, &opts.plan,
        ) {
            (Some(l), _) => {
                let (h, ws) = host_reading(opts.max_memory)?;
                // nothing of this model exists yet: every byte in use is others'
                let own = Own::default();
                let direct_expert_bytes = sizes.direct_expert_bytes();
                let mapped_bytes = mmap_expert_bytes(&scan)?.saturating_add(mmap_dense_file);
                let kv_initial = direct_kv_allowance(&one_lane, max_ctx);
                // SAFETY: pure process-wide Metal availability probe.
                let metal_available = unsafe { mlx_sys::mlx_metal_is_available() };
                let selected = select_expert_mode_for(
                    opts.mmap,
                    opts.direct,
                    metal_available,
                    &one_lane,
                    max_ctx,
                    &h,
                    &own,
                    base.dense,
                    mmap_dense_bytes,
                    direct_expert_bytes,
                    mapped_bytes,
                    opts.mmap_chunk_max,
                    reserve_override,
                    opts.max_memory,
                    l.cache_cap,
                )?;
                match selected {
                    ExpertResidency::Mmap { chunk, gate } => {
                        let ctx = max_ctx;
                        let floor = lane_bytes(&scan.cfg, mtp, ctx);
                        let lanes = (1..=want)
                            .rev()
                            .find(|&n| {
                                n == 1
                                    || gate.need.saturating_add(n.saturating_sub(1) as u64 * floor)
                                        <= gate.budget
                            })
                            .unwrap_or(1);
                        let extra_lanes = lanes.saturating_sub(1) as u64 * floor;
                        let kv_total = kv_initial.saturating_add(extra_lanes);
                        let kv_scratch = gate.kv_and_scratch.saturating_add(extra_lanes);
                        let mmap_fixed = gate.dense_bytes.saturating_add(kv_scratch);
                        let total_slots = sizes.total_experts();
                        let warm_budget = mmap_warm_budget(
                            opts.mmap_prewarm,
                            &h,
                            gate.dense_bytes,
                            kv_total,
                            gate.margin,
                            mapped_bytes,
                        );
                        let wired_cap = direct_wired_limit(mmap_fixed, ws);
                        let a = Admission {
                            physical: gate.physical,
                            host_used: gate.host_used,
                            own: 0,
                            others: gate.others,
                            reserve: gate.margin,
                            live: gate.available,
                            ws,
                            terms: Terms {
                                dense: gate.dense_bytes,
                                staging_layer: 0,
                                kv: kv_total,
                                compute: gate.scratch_bytes,
                                io: 0,
                                driver: 0,
                                pool: 0,
                            },
                            staging_layers: 0,
                            fixed: mmap_fixed,
                            slot: base.slot,
                            cache_slots: (total_slots[0] + total_slots[1]) as u64,
                            bound: if opts.max_memory.is_some_and(|m| m < gate.available) {
                                Bound::Env
                            } else {
                                Bound::Live
                            },
                        };
                        tracing::info!(
                            "hy_v4_flash mmap residency at load: {lanes} lanes, context {ctx}, \
                             chunk {chunk}, wave rows {row_cap}: dense {:.2} GiB + mmap experts \
                             {:.2} GiB (nbytes=0, {} slots, prewarm {:.2} GiB) + kv/scratch {:.2} GiB \
                             vs available {:.2} GiB (others {:.2} GiB, margin {:.2} GiB); wired cap {:.2} GiB",
                            gate.dense_bytes as f64 / GIB as f64,
                            mapped_bytes as f64 / GIB as f64,
                            total_slots[0] + total_slots[1],
                            warm_budget as f64 / GIB as f64,
                            kv_scratch as f64 / GIB as f64,
                            gate.available as f64 / GIB as f64,
                            gate.others as f64 / GIB as f64,
                            gate.margin as f64 / GIB as f64,
                            wired_cap as f64 / GIB as f64,
                        );
                        (
                            lanes,
                            ctx,
                            chunk,
                            Some(a),
                            Some(gate.budget),
                            BankInit::Mmap {
                                warm_budget,
                                force_prewarm: opts.mmap_prewarm == MmapMode::On,
                            },
                        )
                    }
                    ExpertResidency::Direct { chunk, gate } => {
                        let ctx = max_ctx;
                        let floor = lane_bytes(&scan.cfg, mtp, ctx);
                        let lanes = (1..=want)
                            .rev()
                            .find(|&n| {
                                n == 1
                                    || gate.need.saturating_add(n.saturating_sub(1) as u64 * floor)
                                        <= gate.budget
                            })
                            .unwrap_or(1);
                        let extra_lanes = lanes.saturating_sub(1) as u64 * floor;
                        let kv_scratch = gate.kv_and_scratch.saturating_add(extra_lanes);
                        let direct_fixed = gate.dense_bytes.saturating_add(kv_scratch);
                        let total_slots = sizes.total_experts();
                        let wired_cap = direct_wired_limit(
                            direct_fixed.saturating_add(direct_expert_bytes),
                            ws,
                        );
                        let a = Admission {
                            physical: gate.physical,
                            host_used: gate.host_used,
                            own: 0,
                            others: gate.others,
                            reserve: gate.margin,
                            live: gate.available,
                            ws,
                            terms: Terms {
                                dense: gate.dense_bytes,
                                staging_layer: 0,
                                kv: kv_initial.saturating_add(extra_lanes),
                                compute: gate.scratch_bytes,
                                io: 0,
                                driver: 0,
                                pool: 0,
                            },
                            staging_layers: 0,
                            fixed: direct_fixed,
                            slot: base.slot,
                            cache_slots: (total_slots[0] + total_slots[1]) as u64,
                            bound: if opts.max_memory.is_some_and(|m| m < gate.available) {
                                Bound::Env
                            } else {
                                Bound::Live
                            },
                        };
                        tracing::info!(
                            "hy_v4_flash direct residency at load: {lanes} lanes, context {ctx}, \
                             chunk {chunk}, wave rows {row_cap}: dense {:.2} GiB + direct experts \
                             {:.2} GiB ({} slots) + kv/scratch {:.2} GiB vs available {:.2} GiB \
                             (others {:.2} GiB, margin {:.2} GiB); wired cap {:.2} GiB",
                            gate.dense_bytes as f64 / GIB as f64,
                            direct_expert_bytes as f64 / GIB as f64,
                            total_slots[0] + total_slots[1],
                            kv_scratch as f64 / GIB as f64,
                            gate.available as f64 / GIB as f64,
                            gate.others as f64 / GIB as f64,
                            gate.margin as f64 / GIB as f64,
                            wired_cap as f64 / GIB as f64,
                        );
                        (
                            lanes,
                            ctx,
                            chunk,
                            Some(a),
                            Some(gate.budget),
                            BankInit::Direct,
                        )
                    }
                    ExpertResidency::Streamed { direct_refusal } => {
                        if let Some(gate) = direct_refusal {
                            tracing::warn!(
                                "hy_v4_flash direct residency refused (need {:.2} GiB vs budget \
                                 {:.2} GiB, available {:.2} GiB, others {:.2} GiB, cache cap {:?}); \
                                 falling back to partial streaming",
                                gate.need as f64 / GIB as f64,
                                gate.budget as f64 / GIB as f64,
                                gate.available as f64 / GIB as f64,
                                gate.others as f64 / GIB as f64,
                                l.cache_cap,
                            );
                        }
                        let one = pick(&one_lane, &base, max_ctx, l, &h, &own, ws)?;
                        let at = ModelSize {
                            ctx: one.ctx as u64,
                            ..base
                        };
                        let floor = lane_bytes(&scan.cfg, mtp, one.ctx);
                        let (lanes, p) = lanes_at(one, want, |n| {
                            let lp = lane_profile(&one_lane, n, floor);
                            admit_with(&lp, &at, one.chunk as u64, l, &h, &own, ws)
                        });
                        tracing::info!(
                            "hy_v4_flash admission at load: {lanes} lanes, context {}, chunk {}, \
                             wave rows {row_cap}: {}",
                            p.ctx,
                            p.chunk,
                            p.admission
                        );
                        let a = p.admission;
                        let cache = sizes.classes.split(a.cache_bytes());
                        (
                            lanes,
                            p.ctx,
                            p.chunk,
                            Some(a),
                            None,
                            BankInit::Streamed {
                                cache,
                                staging: a.staging_layers as usize,
                            },
                        )
                    }
                }
            }
            #[cfg(test)]
            (None, CachePlan::Slots { cache, staging }) => (
                want,
                max_ctx,
                smallest.min(max_ctx),
                None,
                None,
                BankInit::Streamed {
                    cache: *cache,
                    staging: *staging,
                },
            ),
            _ => (
                want,
                max_ctx,
                smallest.min(max_ctx),
                None,
                None,
                BankInit::Streamed {
                    cache: [0, 0],
                    staging: 1,
                },
            ),
        };
        let profile = lane_profile(&one_lane, lanes, lane_bytes(&scan.cfg, mtp, ctx));
        // the re-admission charges the rows the load admitted, the env only
        // lowers the chunk the driver runs
        let rows = admitted as u64;
        let chunk = opts.chunk_cap.map_or(admitted, |c| admitted.min(c));
        let is_mmap = matches!(bank_init, BankInit::Mmap { .. });
        let size = ModelSize {
            ctx: ctx as u64,
            dense: if is_mmap {
                mmap_dense_bytes
            } else {
                base.dense
            },
            ..base
        };
        let allow_cache = shard_page_cache(
            &scan,
            shard1_nocache(std::env::var(SHARD1_NOCACHE_ENV).ok().as_deref()),
        );
        let shards = Arc::new(ShardSet::open_with_cache(&scan.paths, &allow_cache)?);
        let t_bank = Instant::now();
        let (bank, mmap_dense) = match bank_init {
            BankInit::Mmap {
                warm_budget,
                force_prewarm,
            } => {
                let dense_spec = mmap_dense_spec(&scan)?;
                let (b, d) = KqBank::new_mmap_with_dense(
                    IoPool::new(Arc::clone(&shards), DEFAULT_THREADS)?,
                    &scan.cfg.moe,
                    scan.trunk_experts(),
                    &scan.paths,
                    warm_budget,
                    force_prewarm,
                    &dense_spec,
                )?;
                (Arc::new(b), d)
            }
            BankInit::Direct => (
                Arc::new(KqBank::new_direct(
                    IoPool::new(Arc::clone(&shards), DEFAULT_THREADS)?,
                    &scan.cfg.moe,
                    scan.trunk_experts(),
                )?),
                MmapDense::default(),
            ),
            BankInit::Streamed { cache, staging } => (
                Arc::new(KqBank::new(
                    IoPool::new(Arc::clone(&shards), DEFAULT_THREADS)?,
                    &scan.cfg.moe,
                    scan.trunk_experts(),
                    cache,
                    staging,
                )?),
                MmapDense::default(),
            ),
        };
        let bank_alloc_s = t_bank.elapsed().as_secs_f64();
        let built = (|| -> Result<_> {
            let resident = Resident::load_with_mmap(&scan, &shards, mmap_dense)?;
            let (read, bytes) = (resident.read, resident.bytes);
            let table = Arc::new(F8Table::open_in(
                Arc::clone(&shards),
                &scan.engram_table,
                &scan.cfg.engram,
            )?);
            let b = Arc::clone(&bank);
            let model = Model::new(
                scan.cfg.clone(),
                Box::new(resident),
                move |_| Ok(ExpertSource::Bank(Arc::clone(&b))),
                Box::new(TableRows(Arc::clone(&table))),
            )?;
            Ok((model, table, read, bytes))
        })();
        let (model, table, dense_read, dense_bytes) = match built {
            Ok(v) => v,
            Err(e) => {
                // the partial model is gone; keep its buffers out of MLX's free pool
                drop(bank);
                crate::array::clear_cache();
                return Err(e);
            }
        };
        let lease = Lease::new(profile, size, rows, sizes.classes, limits, admission, &bank)
            .with_max_memory(opts.max_memory);
        let lease = if is_mmap {
            lease.with_direct_budget(direct_budget)
        } else {
            lease.with_direct_promotion(
                DirectPromotion {
                    mode: opts.direct,
                    reserve_override,
                    n_experts: sizes.n_experts,
                },
                direct_budget,
            )
        };
        Ok(Self {
            model,
            bank,
            table,
            scan,
            shards,
            dense_read,
            dense_bytes,
            sizes,
            size,
            profile,
            ctx,
            chunk,
            lanes,
            row_cap,
            mtp,
            bank_alloc_s,
            max_memory: opts.max_memory,
            direct_budget,
            lease,
        })
    }

    pub fn is_direct(&self) -> bool {
        self.bank.is_direct()
    }

    pub fn is_mmap(&self) -> bool {
        self.bank.is_mmap()
    }

    pub fn mapped_bytes(&self) -> u64 {
        self.bank.mapped_bytes()
    }

    pub fn wait_direct_fill(&self) -> Result<()> {
        self.bank.wait_direct_fill()
    }

    pub fn admission(&self) -> Option<Admission> {
        self.lease.admission()
    }

    pub fn wired_limit(&self) -> Option<u64> {
        self.lease.wired_limit(&self.bank)
    }

    /// Fails when a shard was replaced or changed since the load opened it.
    pub fn files_unchanged(&self) -> Result<()> {
        unchanged_since_open(&self.shards)
    }

    /// The memory governor's limit for this model: direct host budget when
    /// direct residency is active, otherwise the capped Metal working set.
    pub fn hard_budget(&self) -> u64 {
        self.lease
            .direct_budget()
            .or(self.direct_budget)
            .unwrap_or_else(|| capped_ws(working_set(), self.max_memory))
    }

    pub fn cache_slots(&self) -> [usize; 2] {
        self.bank.cache_slots()
    }

    /// `kv`: bytes of every live session (`Session::nbytes`).
    pub fn own(&self, kv: u64) -> Result<Own> {
        let pool = self.admission().map_or(0, |a| a.terms.pool);
        own(&self.bank, self.dense_bytes, kv, pool)
    }

    /// `kv`: bytes of every live session (`Session::nbytes`).
    pub fn begin_turn(&self, kv: u64) -> Result<()> {
        self.lease.begin_turn(&self.bank, self.dense_bytes, kv)
    }

    pub fn end_turn(&self) -> Result<()> {
        self.lease.end_turn(&self.bank)
    }

    /// Idle-gap turn end (0 active turns): releases the wired lease and
    /// promotes a partial bank to direct residency when host headroom admits it.
    pub fn end_turn_idle(&self, kv: u64) -> Result<Option<usize>> {
        self.lease.end_turn_idle(&self.bank, self.dense_bytes, kv)
    }

    /// `kv`: bytes of every live session (`Session::nbytes`).
    pub fn recheck(&self, kv: u64) -> Result<bool> {
        self.lease.recheck(&self.bank, self.dense_bytes, kv)
    }
}

pub(crate) fn unchanged_since_open(shards: &ShardSet) -> Result<()> {
    shards.recheck().map_err(|e| {
        Error::from_reason(format!(
            "Hy4 load: {}; the GGUF changed during the load; retry the load",
            e.reason
        ))
    })
}

#[cfg(test)]
mod tests;
