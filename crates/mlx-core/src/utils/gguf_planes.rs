//! The MDGG0001 planes plan of a GGUF load (Splash 1.6.1 d8af1054
//! runtime/model/GgufPreparation.cpp, runtime/ops/KernelSelection.cpp;
//! Apache-2.0, see THIRD_PARTY_NOTICES): which tensors also get Splash's
//! weight planes, and the kernel, K splits, pass and walk each 8-row call runs,
//! decided from the header and the device alone.
//!
//! The plan covers only what was measured: the Qwen3.8-27B shapes (hidden
//! 5120, 64 trunk blocks, FFN 17408, mixer width 6144), the trunk's MLP,
//! GDN out and attention o projections, and per format and role only the
//! kernels measured faster than the Tiled64 route on a 40-core M5 Max. A
//! device with fewer cores, or whose ledger budget (under the configured
//! memory cap) cannot hold the weights, the planes and one lane at the
//! trained window, plans nothing. A load plans only with a DFlash2 draft (the
//! verify the planes serve), never for a CLEF decision model.
//!
//! The plan is made for one lane (8 rows) and kept for 16 / 24 / 32 rows.
//! A load sums the split-K partials in Tiled64's order (the product walk):
//! the planes give the Tiled64 route's bits in both activations modes, so the
//! plan adds nothing to the state store identity. Splash's spread walk stays
//! a plan the tests reach.

use std::collections::HashMap;
use std::fmt::Write as _;

use napi::{Error, Result};

use sha2::{Digest, Sha256};

use crate::array::{DType, MxArray};
use crate::engine::memory_plan::{Margin, hard_budget_bytes};
use crate::utils::gguf::{GgufFile, GgufTensorInfo, GgufTensorType};

pub(crate) const FMT_Q4K: u32 = 0;
pub(crate) const FMT_IQ4XS: u32 = 1;
pub(crate) const FMT_IQ4NL: u32 = 2;
pub(crate) const FMT_Q5K: u32 = 3;
pub(crate) const FMT_Q6K: u32 = 4;
pub(crate) const FMT_Q3K: u32 = 5;
pub(crate) const FMT_Q8_0: u32 = 6;
pub(crate) const FMT_IQ3S: u32 = 7;

const ARCHITECTURE: &str = "qwen35";
const HIDDEN: u32 = 5120;
const FFN: u32 = 17408;
const MIXER: u32 = 6144;
const TRUNK_BLOCKS: u32 = 64;
pub(crate) const TILE_ROWS: u32 = 8;
/// One lane's bytes besides its KV that the floor keeps: the DFlash2 draft
/// (3.85 GB on disk), the lane's GDN cells, ring and window, the verify
/// transients and the plan's pipeline and runtime reserves.
const LANE_RESERVE_BYTES: u64 = 8 << 30;
/// The device class the plan's tables were measured on.
const MIN_CORES: u32 = 40;
/// sha256 of the planes route's sources, oldest first: everything under
/// `crates/mlx-sys/src/metal/splash_gguf` (its compile flags included) and
/// `crates/mlx-sys/src/mlx_gguf_planes.{h,cpp}`. A test fails while the last
/// entry is not the sources' hash: append the new hash, which bumps
/// [`ROUTE_VERSION`], so stored states never restore across two routes.
const ROUTE_SOURCES: [&str; 1] =
    ["65d35fd2f6cc9125e05d69643792ae77bb71e1d5f2957b977776d86a2989a1bf"];
/// Hashed into the identity.
const ROUTE_VERSION: u32 = ROUTE_SOURCES.len() as u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlanesDevice {
    /// Metal, an Apple10 GPU and tensor ops.
    pub planes: bool,
    pub cores: u32,
    /// Metal's recommended working set.
    pub working_set_bytes: u64,
    /// The load's configured `max_memory_bytes`: it only lowers the ledger
    /// budget.
    pub max_memory_bytes: Option<u64>,
}

impl PlanesDevice {
    pub(crate) fn this(max_memory_bytes: Option<u64>) -> Self {
        // SAFETY: plain queries that catch internally.
        let (planes, cores) = unsafe {
            (
                mlx_sys::mlx_gguf_planes_available(),
                mlx_sys::mlx_gpu_reported_core_count(),
            )
        };
        Self {
            planes,
            // Splash's DeviceCapabilities fallback when the IORegistry publishes no count.
            cores: u32::try_from(cores).ok().filter(|&c| c > 0).unwrap_or(32),
            working_set_bytes: crate::engine::memory_plan::recommended_working_set().unwrap_or(0),
            max_memory_bytes,
        }
    }
}

/// The order the planes' split-K partials sum in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Walk {
    /// Tiled64's order: the planes route gives the Tiled64 route's bits.
    #[default]
    Product,
    /// Splash's spread walk, where the plan's law spreads.
    Spread,
}

/// What a load adds to the header and device for its plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlanesLoad {
    /// The load installs a DFlash2 draft.
    pub draft: bool,
    pub clef: bool,
    /// GGUF files of the main checkpoint.
    pub files: usize,
    pub walk: Walk,
}

impl PlanesLoad {
    /// A load's planes inputs, in both activations modes on the product
    /// walk: the planes give the Tiled64 route's bits, so a state written
    /// under either mode restores under the other.
    pub(crate) fn of_load(draft: bool, clef: bool, files: usize) -> Self {
        Self {
            draft,
            clef,
            files,
            walk: Walk::Product,
        }
    }
}

/// A projection's place in the trunk block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Role {
    /// One merged `[2 x FFN, HIDDEN]` tensor of a same-format gate and up.
    GateUp,
    Gate,
    Up,
    Down,
    /// GDN `ssm_out`.
    Out,
    /// Attention `attn_output`.
    O,
}

/// The decode kernel of a planned tensor at 8 / 16 / 32 rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Kernel {
    /// `gguf_decode_<f>_m<r>_a`.
    Plain,
    /// `_r`: plus the residual.
    Residual,
    /// `_g`: swiglu of the gate output and this up projection.
    Gate,
    /// `_gate_up`: both halves of a merged tensor and their swiglu in one pass.
    GateUp,
}

impl Kernel {
    fn name(self) -> &'static str {
        match self {
            Kernel::Plain => "a",
            Kernel::Residual => "r",
            Kernel::Gate => "g",
            Kernel::GateUp => "gate_up",
        }
    }

    /// `mlx_gguf_planes_prepare` kind.
    pub(crate) fn prepare_kind(self) -> u32 {
        match self {
            Kernel::Plain => 0,
            Kernel::Residual => 1,
            Kernel::Gate => 2,
            Kernel::GateUp => 3,
        }
    }
}

/// One planes array: the GGUF tensors whose rows it holds (row offset each),
/// its format and shape, and how its 8-row call runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PlanesTensor {
    /// The import map key of the planes array.
    pub key: String,
    pub layer: u32,
    pub sources: Vec<(String, u32)>,
    pub format: u32,
    pub rows: u32,
    pub k: u32,
    pub role: Role,
    pub kernel: Kernel,
    pub splits: u32,
    pub one_pass: bool,
    pub walk: bool,
    pub tile: u32,
    pub bytes: u64,
}

impl PlanesTensor {
    fn line(&self) -> String {
        let mut s = format!(
            "{} fmt={} rows={} k={} role={:?} kernel={} splits={} one_pass={} walk={} tile={}",
            self.key,
            self.format,
            self.rows,
            self.k,
            self.role,
            self.kernel.name(),
            self.splits,
            u8::from(self.one_pass),
            u8::from(self.walk),
            self.tile
        );
        for (name, row0) in &self.sources {
            let _ = write!(s, " {name}@{row0}");
        }
        s
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct PlanesPlan {
    pub tensors: Vec<PlanesTensor>,
    pub walk: Walk,
    /// Why the load plans nothing, or why a trunk projection stays off the
    /// planes: `(what, reason)`.
    pub left_out: Vec<(String, &'static str)>,
}

impl PlanesPlan {
    pub(crate) fn bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.bytes).sum()
    }

    /// The planes id: a hash of the route version and every planned tensor's
    /// line, empty when nothing is planned.
    pub(crate) fn identity(&self) -> String {
        self.identity_at(ROUTE_VERSION)
    }

    fn identity_at(&self, version: u32) -> String {
        if self.tensors.is_empty() {
            return String::new();
        }
        let mut lines: Vec<String> = self.tensors.iter().map(PlanesTensor::line).collect();
        lines.sort();
        let mut h = Sha256::new();
        h.update(format!("route {version}\n").as_bytes());
        for line in &lines {
            h.update(line.as_bytes());
            h.update(b"\n");
        }
        let hex: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        format!("planes-route:{hex}")
    }

    /// The part the state store identity takes: the planes id under the
    /// spread walk; nothing under the product walk, whose bits are the
    /// Tiled64 route's.
    pub(crate) fn store_identity(&self) -> Option<String> {
        (self.walk == Walk::Spread && !self.tensors.is_empty()).then(|| self.identity())
    }

    pub(crate) fn describe(&self) -> String {
        let mut by_role = std::collections::BTreeMap::new();
        for t in &self.tensors {
            *by_role
                .entry((t.role, t.kernel.name(), t.splits))
                .or_insert(0u32) += 1;
        }
        let mut s = format!(
            "GGUF planes: {} tensors, {} bytes, {:?} walk",
            self.tensors.len(),
            self.bytes(),
            self.walk
        );
        for ((role, kernel, splits), n) in by_role {
            let _ = write!(s, "; {role:?} {kernel} splits {splits} x{n}");
        }
        if self.tensors.is_empty()
            && let Some((what, why)) = self.left_out.first()
        {
            let _ = write!(s, "; none: {what}: {why}");
        }
        s
    }

    fn none(what: impl Into<String>, why: &'static str) -> Self {
        Self {
            tensors: Vec::new(),
            walk: Walk::default(),
            left_out: vec![(what.into(), why)],
        }
    }

    /// The plan without every tensor `why` gives a reason for. A gate or up
    /// leaves with its pair partner: one half of a pair never runs.
    fn without(self, why: impl Fn(&PlanesTensor) -> Option<&'static str>) -> Self {
        let mut leaving: Vec<Option<&'static str>> = self.tensors.iter().map(why).collect();
        let pairs_out: Vec<u32> = self
            .tensors
            .iter()
            .zip(&leaving)
            .filter(|(t, why)| why.is_some() && matches!(t.role, Role::Gate | Role::Up))
            .map(|(t, _)| t.layer)
            .collect();
        for (t, why) in self.tensors.iter().zip(&mut leaving) {
            if why.is_none()
                && matches!(t.role, Role::Gate | Role::Up)
                && pairs_out.contains(&t.layer)
            {
                *why = Some("its gate|up partner left the plan");
            }
        }
        let mut out = Self {
            tensors: Vec::new(),
            walk: self.walk,
            left_out: self.left_out,
        };
        for (t, why) in self.tensors.into_iter().zip(leaving) {
            match why {
                Some(why) => out.left_out.push((t.key, why)),
                None => out.tensors.push(t),
            }
        }
        out
    }

    /// The plan with every tensor whose sources the import does not read in
    /// whole 64-row bands (`banded(name)` false) left out.
    pub(crate) fn keep_banded(self, banded: impl Fn(&str) -> bool) -> Self {
        self.without(|t| {
            (!t.sources.iter().all(|(name, _)| banded(name))).then_some("not read in 64-row bands")
        })
    }

    /// The plan with every tensor left out whose K split count is not the
    /// Tiled64 route's at its shape on `cores` GPU cores. The product walk
    /// gives the Tiled64 bits only at Tiled64's splits; the lane tier splits
    /// as the 8-row call does, and the planes keep their 8-row splits at 16
    /// and 32 rows, so one count covers alone and waves.
    pub(crate) fn keep_tiled_splits(self, cores: u32) -> Self {
        self.without(|t| {
            (tiled_splits(t.rows, t.k, cores) != Some(t.splits))
                .then_some("K splits differ from the Tiled64 route's")
        })
    }
}

/// K splits of the K-quant Tiled64 call of an `[n, k]` weight on `cores`
/// GPU cores; `None` without Metal.
pub(crate) fn tiled_splits(n: u32, k: u32, cores: u32) -> Option<u32> {
    let (n, k, cores) = (
        i32::try_from(n).ok()?,
        i32::try_from(k).ok()?,
        i32::try_from(cores).ok()?,
    );
    // SAFETY: pure arithmetic over its arguments.
    let splits = unsafe { mlx_sys::mlx_kquant_tiled_splits(n, k, cores) };
    u32::try_from(splits).ok().filter(|&s| s > 0)
}

fn format_of(ty: GgufTensorType) -> Option<u32> {
    // SAFETY: plain query.
    let f = unsafe { mlx_sys::mlx_gguf_planes_format_of(ty as u32) };
    (f < 8).then_some(f)
}

fn bits16(fmt: u32) -> u32 {
    // SAFETY: plain query.
    unsafe { mlx_sys::mlx_gguf_planes_bits16(fmt) }
}

pub(crate) fn planes_bytes(fmt: u32, n: u32, k: u32) -> u64 {
    // SAFETY: plain query; null parts.
    unsafe { mlx_sys::mlx_gguf_planes_bytes(fmt, n.into(), k.into(), std::ptr::null_mut()) }
}

/// Splash's one-lane plan of `[n, k]`; a gate/up pair names both formats.
fn law(n: u32, k: u32, cores: u32, fmt: u32, pair: Option<(u32, u32)>) -> mlx_sys::GgufPlanesPlan {
    let mut out = mlx_sys::GgufPlanesPlan::default();
    let (b16, one, gb, ub) = match pair {
        Some((g, u)) => ((bits16(g) + bits16(u)) / 2, g == u, bits16(g), bits16(u)),
        None => (bits16(fmt), false, 0, 0),
    };
    // SAFETY: `out` is a live struct; `cores` is nonzero, so no device query.
    let ok = unsafe { mlx_sys::mlx_gguf_planes_plan(n, k, b16, cores, one, gb, ub, &mut out) };
    if !ok {
        out.staged = 0;
    }
    out
}

/// The routes measured faster than Tiled64 (8 rows, 40-core M5 Max): the
/// formats of a merged gate|up whose one-pass kernel was, those whose `_a`
/// over both halves was, the split pairs `(gate, up)` whose `_a` + `_g` was,
/// and per residual role the formats whose `_r` was. IQ4_XS gates with a
/// Q5_K or Q3_K up were not faster and stay on Tiled64; Q4_K / IQ4_XS pairs,
/// either way round, were not faster on real tensors.
const MERGED_ONE_PASS: [u32; 3] = [FMT_Q5K, FMT_Q4K, FMT_Q6K];
const MERGED_PLAIN: [u32; 1] = [FMT_IQ4XS];
const PAIRS: [(u32, u32); 5] = [
    (FMT_Q4K, FMT_Q5K),
    (FMT_Q6K, FMT_Q5K),
    (FMT_IQ4NL, FMT_Q5K),
    (FMT_Q5K, FMT_IQ4NL),
    (FMT_IQ4NL, FMT_IQ4XS),
];
const DOWN: [u32; 5] = [FMT_Q5K, FMT_IQ4XS, FMT_Q4K, FMT_IQ4NL, FMT_IQ3S];
const OUT: [u32; 2] = [FMT_Q5K, FMT_Q4K];
const O: [u32; 2] = [FMT_Q5K, FMT_IQ4XS];

fn meta_u64(header: &GgufFile, key: &str) -> Option<u64> {
    header.metadata.get(key).and_then(|v| v.as_u64())
}

/// `blk.<i>.<role>.weight` -> `(i, role)`.
fn trunk_name(name: &str) -> Option<(u32, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let (index, rest) = rest.split_once('.')?;
    if index.is_empty() || !index.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let role = rest.strip_suffix(".weight")?;
    Some((index.parse().ok()?, role))
}

fn fits(t: &GgufTensorInfo, n: u32, k: u32) -> bool {
    shape(t) == Some((n, k))
}

/// `[n, k]` of a 2-D tensor.
fn shape(t: &GgufTensorInfo) -> Option<(u32, u32)> {
    match t.dims.as_slice() {
        [k, n] => Some((u32::try_from(*n).ok()?, u32::try_from(*k).ok()?)),
        _ => None,
    }
}

/// One lane's int8 K/V bytes at `tokens`: K and V of every full-attention
/// layer, one byte per value and an f16 scale per head.
fn lane_kv_bytes(header: &GgufFile, tokens: u64) -> Option<u64> {
    let blocks = meta_u64(header, "qwen35.block_count")?;
    let mtp = meta_u64(header, "qwen35.nextn_predict_layers").unwrap_or(0);
    let interval = meta_u64(header, "qwen35.full_attention_interval")?.max(1);
    let kv_heads = meta_u64(header, "qwen35.attention.head_count_kv")?;
    let key = meta_u64(header, "qwen35.attention.key_length")?;
    let value = meta_u64(header, "qwen35.attention.value_length")?;
    let layers = blocks.saturating_sub(mtp) / interval;
    Some(tokens * layers * kv_heads * (key + value + 4))
}

fn tensor(
    key: String,
    layer: u32,
    sources: Vec<(String, u32)>,
    (format, rows, k): (u32, u32, u32),
    role: Role,
    kernel: Kernel,
    law: mlx_sys::GgufPlanesPlan,
) -> PlanesTensor {
    PlanesTensor {
        key,
        layer,
        sources,
        format,
        rows,
        k,
        role,
        kernel,
        splits: law.splits,
        one_pass: law.one_pass == 1,
        walk: law.spread == 1,
        tile: TILE_ROWS,
        bytes: planes_bytes(format, rows, k),
    }
}

/// A layer's gate and up: one merged planes tensor of a same-format pair,
/// else one each.
fn plan_mlp(
    plan: &mut PlanesPlan,
    layer: u32,
    stem: &str,
    g: &GgufTensorInfo,
    u: &GgufTensorInfo,
    cores: u32,
) {
    let (Some(gf), Some(uf)) = (format_of(g.tensor_type), format_of(u.tensor_type)) else {
        plan.left_out
            .push((format!("{stem}.ffn_gate|up"), "no planes format"));
        return;
    };
    if gf == uf {
        let key = format!("{stem}.ffn_gate_up.planes");
        let sources = vec![(g.name.clone(), 0), (u.name.clone(), FFN)];
        let shape = (gf, 2 * FFN, HIDDEN);
        let pair = law(FFN, HIDDEN, cores, gf, Some((gf, uf)));
        if pair.staged == 1 && pair.one_pass == 1 && MERGED_ONE_PASS.contains(&gf) {
            plan.tensors.push(tensor(
                key,
                layer,
                sources,
                shape,
                Role::GateUp,
                Kernel::GateUp,
                pair,
            ));
            return;
        }
        let whole = law(2 * FFN, HIDDEN, cores, gf, None);
        if whole.staged == 1 && MERGED_PLAIN.contains(&gf) {
            plan.tensors.push(tensor(
                key,
                layer,
                sources,
                shape,
                Role::GateUp,
                Kernel::Plain,
                whole,
            ));
        } else {
            plan.left_out
                .push((key, "not a faster route on this device"));
        }
        return;
    }
    let pair = law(FFN, HIDDEN, cores, gf, Some((gf, uf)));
    if pair.staged != 1 || !PAIRS.contains(&(gf, uf)) {
        plan.left_out.push((
            format!("{stem}.ffn_gate|up"),
            "not a faster route on this device",
        ));
        return;
    }
    plan.tensors.push(tensor(
        format!("{stem}.ffn_gate.planes"),
        layer,
        vec![(g.name.clone(), 0)],
        (gf, FFN, HIDDEN),
        Role::Gate,
        Kernel::Plain,
        pair,
    ));
    plan.tensors.push(tensor(
        format!("{stem}.ffn_up.planes"),
        layer,
        vec![(u.name.clone(), 0)],
        (uf, FFN, HIDDEN),
        Role::Up,
        Kernel::Gate,
        pair,
    ));
}

/// The plan of `header` on `device`. Pure: the same header and device give
/// the same plan.
pub(crate) fn plan_planes(header: &GgufFile, device: &PlanesDevice) -> PlanesPlan {
    let arch = header
        .metadata
        .get("general.architecture")
        .and_then(|v| v.as_str());
    if arch != Some(ARCHITECTURE) {
        return PlanesPlan::none("general.architecture", "not qwen35");
    }
    let blocks = meta_u64(header, "qwen35.block_count").unwrap_or(0);
    let mtp = meta_u64(header, "qwen35.nextn_predict_layers").unwrap_or(0);
    let trunk = blocks.saturating_sub(mtp);
    if meta_u64(header, "qwen35.embedding_length") != Some(HIDDEN.into())
        || meta_u64(header, "qwen35.feed_forward_length") != Some(FFN.into())
        || trunk != u64::from(TRUNK_BLOCKS)
    {
        return PlanesPlan::none("shape", "not the measured 27B shapes");
    }
    if !device.planes || device.cores == 0 {
        return PlanesPlan::none("device", "no Apple10 GPU with tensor ops");
    }
    if device.cores < MIN_CORES {
        return PlanesPlan::none(
            format!("device of {} GPU cores", device.cores),
            "fewer cores than the measured device class",
        );
    }
    let mut by_layer: Vec<HashMap<&str, &GgufTensorInfo>> =
        (0..TRUNK_BLOCKS).map(|_| Default::default()).collect();
    for t in &header.tensors {
        if let Some((layer, role)) = trunk_name(&t.name)
            && layer < TRUNK_BLOCKS
        {
            by_layer[layer as usize].insert(role, t);
        }
    }
    let mut plan = PlanesPlan {
        walk: Walk::Spread,
        ..PlanesPlan::default()
    };
    let cores = device.cores;
    for (layer, roles) in by_layer.iter().enumerate() {
        let layer = layer as u32;
        let stem = format!("blk.{layer}");
        match (roles.get("ffn_gate"), roles.get("ffn_up")) {
            (Some(g), Some(u)) if fits(g, FFN, HIDDEN) && fits(u, FFN, HIDDEN) => {
                plan_mlp(&mut plan, layer, &stem, g, u, cores);
            }
            (None, None) => {}
            _ => plan
                .left_out
                .push((format!("{stem}.ffn_gate|up"), "not the 27B shape")),
        }
        for (gguf_role, role, n, k, formats) in [
            ("ffn_down", Role::Down, HIDDEN, FFN, &DOWN[..]),
            ("ssm_out", Role::Out, HIDDEN, MIXER, &OUT[..]),
            ("attn_output", Role::O, HIDDEN, MIXER, &O[..]),
        ] {
            let Some(t) = roles.get(gguf_role) else {
                continue;
            };
            let key = format!("{stem}.{gguf_role}.planes");
            let Some(fmt) = format_of(t.tensor_type).filter(|_| fits(t, n, k)) else {
                plan.left_out
                    .push((key, "not a planes tensor of the 27B shape"));
                continue;
            };
            if fmt == FMT_Q8_0 {
                plan.left_out.push((key, "Q8_0 has no Tiled64 bands"));
                continue;
            }
            let single = law(n, k, cores, fmt, None);
            if single.staged == 1 && formats.contains(&fmt) {
                let sources = vec![(t.name.clone(), 0)];
                plan.tensors.push(tensor(
                    key,
                    layer,
                    sources,
                    (fmt, n, k),
                    role,
                    Kernel::Residual,
                    single,
                ));
            } else {
                plan.left_out
                    .push((key, "not a faster route on this device"));
            }
        }
    }
    let trained = meta_u64(header, "qwen35.context_length").unwrap_or(0);
    let weights: u64 = header
        .tensors
        .iter()
        .map(|t| t.data_size().unwrap_or(u64::MAX))
        .fold(0u64, u64::saturating_add);
    let floor = weights
        .saturating_add(plan.bytes())
        .saturating_add(lane_kv_bytes(header, trained).unwrap_or(u64::MAX))
        .saturating_add(LANE_RESERVE_BYTES);
    let budget = hard_budget_bytes(
        device.working_set_bytes,
        device.max_memory_bytes,
        Margin::Ledger,
    );
    if budget < floor {
        return PlanesPlan::none(
            format!("ledger budget {budget} below the floor {floor}"),
            "weights, planes and one lane at the trained window",
        );
    }
    plan
}

/// A load's planes line: logged, and kept for a test to read back.
pub(crate) fn load_line(line: String) {
    tracing::info!("{line}");
    #[cfg(test)]
    LOAD_LINES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(line);
}

#[cfg(test)]
pub(crate) static LOAD_LINES: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The plan a load of `header` makes on `device`: [`plan_planes`] when the
/// load has a DFlash2 draft, is not CLEF and reads one GGUF file, every
/// tensor on `load`'s walk; on the product walk only the tensors at the
/// Tiled64 route's K splits ([`PlanesPlan::keep_tiled_splits`]).
pub(crate) fn plan_for_load(
    header: &GgufFile,
    device: &PlanesDevice,
    load: &PlanesLoad,
) -> PlanesPlan {
    if !load.draft {
        return PlanesPlan::none("load", "no DFlash2 draft");
    }
    if load.clef {
        return PlanesPlan::none("load", "a CLEF decision model");
    }
    if load.files != 1 {
        return PlanesPlan::none("load", "a split GGUF");
    }
    let mut plan = plan_planes(header, device);
    plan.walk = load.walk;
    if load.walk == Walk::Product {
        for t in &mut plan.tensors {
            t.walk = false;
        }
        plan = plan.keep_tiled_splits(device.cores);
    }
    plan
}

/// The plan with every tensor whose pipelines this device refuses left out.
pub(crate) fn prepare_planes(plan: PlanesPlan) -> PlanesPlan {
    plan.without(|t| {
        let fmts = [t.format];
        // SAFETY: one format in `fmts`.
        let built =
            unsafe { mlx_sys::mlx_gguf_planes_prepare(t.kernel.prepare_kind(), fmts.as_ptr(), 1) };
        (!built).then_some("pipeline refused")
    })
}

/// Rows `[row0, row0 + rows)` of an `n_total x k` tensor of `fmt` whose
/// native GGUF rows sit in `staging` from byte `src_offset`, repacked into
/// the tensor's planes `dst`; returns the lazy array sharing `dst`'s buffer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn repack_band(
    staging: &MxArray,
    src_offset: u64,
    fmt: u32,
    k: u32,
    n_total: u32,
    row0: u32,
    rows: u32,
    dst: &MxArray,
) -> Result<MxArray> {
    let mut out: *mut mlx_sys::mlx_array = std::ptr::null_mut();
    let mut err = [0 as std::ffi::c_char; 512];
    // SAFETY: live handles; `err` is writable for its length.
    let rc = unsafe {
        mlx_sys::mlx_gguf_planes_repack_band(
            staging.as_raw_ptr(),
            src_offset,
            fmt,
            k,
            n_total,
            row0,
            rows,
            dst.as_raw_ptr(),
            &mut out,
            err.as_mut_ptr(),
            err.len(),
        )
    };
    if rc != 0 {
        // SAFETY: the callee leaves a NUL-terminated message.
        let msg = unsafe { std::ffi::CStr::from_ptr(err.as_ptr()) };
        return Err(Error::from_reason(msg.to_string_lossy().into_owned()));
    }
    MxArray::from_handle(out, "gguf planes band")
}

struct PlanesDst {
    key: String,
    format: u32,
    rows: u32,
    k: u32,
    array: MxArray,
    bands_left: usize,
}

/// The import's second writer: every band of a planned tensor, read once into
/// the staging ring for its Tiled64 arrays, is also repacked into the planes
/// array of the plan, at the tensor's row offset there.
pub(crate) struct PlanesWriter {
    dsts: Vec<PlanesDst>,
    /// Header tensor index -> (planes array, row offset).
    of_tensor: HashMap<usize, (usize, u32)>,
}

impl PlanesWriter {
    /// `tiled(i)`: tensor `i` is read in whole 64-row bands; `bands[i]`: its
    /// band count.
    pub(crate) fn new(
        header: &GgufFile,
        plan: &PlanesPlan,
        tiled: impl Fn(usize) -> bool,
        bands: &[usize],
    ) -> Result<Self> {
        let index: HashMap<&str, usize> = header
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.as_str(), i))
            .collect();
        let mut dsts = Vec::new();
        let mut of_tensor = HashMap::new();
        for t in &plan.tensors {
            let (array, _, _) = MxArray::alloc_uninit(&[t.bytes as i64], DType::Uint8)?;
            let mut bands_left = 0;
            for (name, row0) in &t.sources {
                let &i = index.get(name.as_str()).ok_or_else(|| {
                    Error::from_reason(format!("planned tensor {name} is not in the GGUF"))
                })?;
                if !tiled(i) {
                    return Err(Error::from_reason(format!(
                        "planned tensor {name} is not read in 64-row bands"
                    )));
                }
                if of_tensor.insert(i, (dsts.len(), *row0)).is_some() {
                    return Err(Error::from_reason(format!("{name} is planned twice")));
                }
                bands_left += bands[i];
            }
            dsts.push(PlanesDst {
                key: t.key.clone(),
                format: t.format,
                rows: t.rows,
                k: t.k,
                array,
                bands_left,
            });
        }
        Ok(Self { dsts, of_tensor })
    }

    /// The band `[row0, row0 + rows)` of header tensor `tensor` at byte
    /// `src_off` of `staging`; the encoded array when the tensor is planned.
    pub(crate) fn band(
        &mut self,
        staging: &MxArray,
        tensor: usize,
        src_off: usize,
        row0: usize,
        rows: usize,
    ) -> Result<Option<MxArray>> {
        let Some(&(i, offset)) = self.of_tensor.get(&tensor) else {
            return Ok(None);
        };
        let d = &mut self.dsts[i];
        let u32_of = |v: usize| {
            u32::try_from(v).map_err(|_| Error::from_reason(format!("{}: {v} past 32 bits", d.key)))
        };
        let (row0, rows) = (u32_of(row0)?, u32_of(rows)?);
        let next = repack_band(
            staging,
            src_off as u64,
            d.format,
            d.k,
            d.rows,
            offset + row0,
            rows,
            &d.array,
        )?;
        d.array = next.clone();
        d.bands_left -= 1;
        Ok(Some(next))
    }

    /// `(key, planes)` of every planned tensor, each fully written.
    pub(crate) fn finish(self) -> Result<Vec<(String, MxArray)>> {
        self.dsts
            .into_iter()
            .map(|d| {
                if d.bands_left != 0 {
                    return Err(Error::from_reason(format!(
                        "{}: {} bands not written",
                        d.key, d.bands_left
                    )));
                }
                Ok((d.key, d.array))
            })
            .collect()
    }
}

/// The 27B UD-Q4_K_XL header from `testdata/gguf_planes/q4kxl.header.txt`.
#[cfg(test)]
pub(crate) fn fixture_header() -> GgufFile {
    use crate::utils::gguf::GgufMetaValue;
    let text = include_str!("testdata/gguf_planes/q4kxl.header.txt");
    let mut metadata = std::collections::HashMap::new();
    let mut tensors = Vec::new();
    for line in text.lines() {
        let fields: Vec<&str> = line.split(' ').collect();
        match fields.as_slice() {
            ["KV", key, value] => {
                let parsed = if let Ok(v) = value.parse::<u64>() {
                    GgufMetaValue::Uint64(v)
                } else if let Ok(v) = value.parse::<f32>() {
                    GgufMetaValue::Float32(v)
                } else {
                    GgufMetaValue::String((*value).to_string())
                };
                metadata.insert((*key).to_string(), parsed);
            }
            ["T", name, ty, dims, offset] => {
                let tensor_type = match *ty {
                    "F32" => GgufTensorType::F32,
                    "Q3_K" => GgufTensorType::Q3K,
                    "Q4_K" => GgufTensorType::Q4K,
                    "Q5_K" => GgufTensorType::Q5K,
                    "Q6_K" => GgufTensorType::Q6K,
                    "Q8_0" => GgufTensorType::Q8_0,
                    "IQ4_XS" => GgufTensorType::IQ4XS,
                    "IQ4_NL" => GgufTensorType::IQ4NL,
                    "IQ3_S" => GgufTensorType::IQ3S,
                    other => panic!("fixture type {other}"),
                };
                let dims: Vec<u64> = dims.split('x').map(|d| d.parse().unwrap()).collect();
                tensors.push(GgufTensorInfo {
                    name: (*name).to_string(),
                    n_dims: dims.len() as u32,
                    dims,
                    tensor_type,
                    offset: offset.parse().unwrap(),
                });
            }
            _ => {}
        }
    }
    GgufFile {
        version: 3,
        tensor_count: tensors.len() as u64,
        metadata,
        tensors,
        alignment: 32,
        data_offset: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    const GIB: u64 = 1 << 30;

    fn copy(h: &GgufFile) -> GgufFile {
        GgufFile {
            version: h.version,
            tensor_count: h.tensor_count,
            metadata: h.metadata.clone(),
            tensors: h.tensors.clone(),
            alignment: h.alignment,
            data_offset: h.data_offset,
        }
    }

    fn app_machine() -> PlanesDevice {
        PlanesDevice {
            planes: true,
            cores: 40,
            working_set_bytes: 107 * GIB,
            max_memory_bytes: None,
        }
    }

    fn counts(plan: &PlanesPlan) -> BTreeMap<(Role, Kernel, u32, u32), u32> {
        let mut out = BTreeMap::new();
        for t in &plan.tensors {
            *out.entry((t.role, t.kernel, t.format, t.splits))
                .or_insert(0) += 1;
        }
        out
    }

    /// The 27B header on a 40-core M5 Max, Splash's splits: 39 merged
    /// gate|up (Q5_K 18, Q4_K 3, Q6_K 2 in one pass; IQ4_XS 16 on `_a` over
    /// both halves), 12 split pairs (IQ4_XS gates with a Q5_K or Q3_K up left
    /// out: blk.0, 7, 14, 22, 48, 49; Q4_K / IQ4_XS pairs left out), 59 down
    /// (Q6_K left out), 34 GDN out (Q6_K and Q8_0 left out), 8 attention o
    /// (Q6_K left out); none from blk.64: 164 tensors, 9,885,491,200 bytes.
    #[test]
    fn plan_of_the_27b_header() {
        let header = fixture_header();
        let plan = plan_planes(&header, &app_machine());
        for line in plan.tensors.iter().map(PlanesTensor::line) {
            println!("PLANLINE {line}");
        }
        for (what, why) in &plan.left_out {
            println!("LEFTOUT {what}: {why}");
        }
        let c = counts(&plan);
        for (key, n) in &c {
            println!("PLANCOUNT {key:?} {n}");
        }
        let n = |role: Role, kernel: Kernel, fmt: u32, splits: u32| {
            c.get(&(role, kernel, fmt, splits)).copied().unwrap_or(0)
        };
        assert_eq!(n(Role::GateUp, Kernel::GateUp, FMT_Q5K, 1), 18);
        assert_eq!(n(Role::GateUp, Kernel::GateUp, FMT_Q4K, 1), 3);
        assert_eq!(n(Role::GateUp, Kernel::GateUp, FMT_Q6K, 1), 2);
        assert_eq!(n(Role::GateUp, Kernel::Plain, FMT_IQ4XS, 1), 16);
        let role_total = |role: Role| plan.tensors.iter().filter(|t| t.role == role).count();
        assert_eq!(role_total(Role::Gate), 12);
        assert_eq!(role_total(Role::Up), 12);
        assert_eq!(role_total(Role::Down), 59);
        assert_eq!(role_total(Role::Out), 34);
        assert_eq!(role_total(Role::O), 8);
        assert_eq!(plan.tensors.len(), 164);
        for pair in plan.tensors.windows(2) {
            if pair[0].role == Role::Gate {
                assert_eq!(pair[1].role, Role::Up, "{}", pair[0].key);
                let formats = (pair[0].format, pair[1].format);
                assert!(PAIRS.contains(&formats), "{} {formats:?}", pair[0].key);
                assert!(
                    !matches!(formats, (FMT_Q4K, FMT_IQ4XS) | (FMT_IQ4XS, FMT_Q4K)),
                    "{} pair {formats:?} stays on Tiled64",
                    pair[0].key
                );
            }
        }
        let pairs_out = plan
            .left_out
            .iter()
            .filter(|(what, _)| what.ends_with(".ffn_gate|up"))
            .count();
        println!("PAIRSOUT {pairs_out}");
        assert!(
            pairs_out >= 12,
            "6 IQ4_XS gate layers and 6 Q4_K / IQ4_XS pairs"
        );
        for layer in [0u32, 7, 14, 22, 48, 49] {
            assert!(
                plan.tensors
                    .iter()
                    .all(|t| t.layer != layer || !matches!(t.role, Role::Gate | Role::Up)),
                "blk.{layer} gate|up stays on Tiled64"
            );
            assert!(
                plan.left_out
                    .iter()
                    .any(|(what, _)| *what == format!("blk.{layer}.ffn_gate|up")),
                "blk.{layer} gate|up is named left out"
            );
        }
        for t in &plan.tensors {
            assert!(t.layer < 64 && !t.key.starts_with("blk.64."), "{}", t.key);
            assert!(t.walk && t.tile == 8, "{}", t.key);
            let want = match t.role {
                Role::Down | Role::Out | Role::O => 4,
                _ => 1,
            };
            assert_eq!(t.splits, want, "{} splits", t.key);
            assert_eq!(t.kernel == Kernel::GateUp, t.one_pass, "{} pass", t.key);
        }
        assert!(
            plan.left_out
                .iter()
                .any(|(what, _)| what == "blk.0.ssm_out.planes"),
            "the Q8_0 out of blk.0 stays off"
        );
        assert!(
            plan.tensors.iter().all(|t| t.format != FMT_Q8_0),
            "no Q8_0 tensor"
        );
        println!("PLANBYTES {}", plan.bytes());
        println!("PLANID {}", plan.identity());
        println!("PLANDESC {}", plan.describe());
        assert_eq!(plan.bytes(), 9_885_491_200);
    }

    /// Same header and device, same plan and identity; another format of
    /// one planned tensor, a device that changes a split, or another route
    /// version, another identity; nothing planned, no identity.
    #[test]
    fn identity_is_a_hash_of_the_plan() {
        let header = fixture_header();
        let a = plan_planes(&header, &app_machine());
        let b = plan_planes(&header, &app_machine());
        assert_eq!(a, b);
        assert_eq!(a.identity(), b.identity());
        assert!(a.identity().starts_with("planes-route:"));
        let mut other = copy(&header);
        let down = other
            .tensors
            .iter_mut()
            .find(|t| t.name == "blk.3.ffn_down.weight")
            .unwrap();
        assert_eq!(down.tensor_type, GgufTensorType::Q5K);
        down.tensor_type = GgufTensorType::IQ4XS;
        let c = plan_planes(&other, &app_machine());
        assert_eq!(c.tensors.len(), a.tensors.len());
        assert_ne!(
            c.identity(),
            a.identity(),
            "a swapped format keeps the count"
        );
        let more = plan_planes(
            &header,
            &PlanesDevice {
                cores: 80,
                ..app_machine()
            },
        );
        println!("PLANDESC cores=80 {}", more.describe());
        if more
            .tensors
            .iter()
            .map(|t| t.splits)
            .ne(a.tensors.iter().map(|t| t.splits))
        {
            assert_ne!(more.identity(), a.identity(), "another split");
        }
        assert_ne!(
            a.identity_at(ROUTE_VERSION + 1),
            a.identity(),
            "a route version bump with the same plan lines"
        );
        let none = plan_planes(
            &header,
            &PlanesDevice {
                planes: false,
                ..app_machine()
            },
        );
        assert!(none.tensors.is_empty());
        assert_eq!(none.identity(), "");
    }

    /// A machine whose ledger budget cannot hold weights, planes and one lane
    /// at 262,144 tokens plans nothing: a 36 GiB working set (48 GB M5 Pro
    /// class) and 24 GiB; 48 and 64 GiB plan. A configured memory cap below
    /// the floor plans nothing on the app machine; one above it plans.
    #[test]
    fn small_machine_plans_nothing() {
        let header = fixture_header();
        for (gib, planned) in [(24u64, false), (36, false), (48, true), (64, true)] {
            let plan = plan_planes(
                &header,
                &PlanesDevice {
                    working_set_bytes: gib * GIB,
                    ..app_machine()
                },
            );
            println!("SMALL working_set={gib}GiB {}", plan.describe());
            assert_eq!(!plan.tensors.is_empty(), planned, "{gib} GiB");
            if !planned {
                assert_eq!(plan.identity(), "");
                assert!(plan.left_out[0].0.contains("floor"));
            }
        }
        let full = plan_planes(&header, &app_machine());
        for (gib, planned) in [(32u64, false), (40, false), (48, true), (200, true)] {
            let plan = plan_planes(
                &header,
                &PlanesDevice {
                    max_memory_bytes: Some(gib * GIB),
                    ..app_machine()
                },
            );
            println!(
                "CAP working_set=107GiB max_memory={gib}GiB {}",
                plan.describe()
            );
            if planned {
                assert_eq!(plan, full, "a cap above the floor keeps the plan");
            } else {
                assert!(plan.tensors.is_empty(), "{gib} GiB cap");
                assert!(plan.left_out[0].0.contains("floor"));
            }
        }
    }

    /// Other shapes or another architecture plan nothing: a 4096-wide
    /// header, a 48-block trunk, a gemma4 header.
    #[test]
    fn other_shapes_plan_nothing() {
        use crate::utils::gguf::GgufMetaValue;
        let header = fixture_header();
        let mut narrow = copy(&header);
        narrow.metadata.insert(
            "qwen35.embedding_length".into(),
            GgufMetaValue::Uint64(4096),
        );
        let mut short = copy(&header);
        short
            .metadata
            .insert("qwen35.block_count".into(), GgufMetaValue::Uint64(49));
        let mut other = copy(&header);
        other.metadata.insert(
            "general.architecture".into(),
            GgufMetaValue::String("gemma4".into()),
        );
        for (what, h) in [("narrow", narrow), ("short", short), ("gemma4", other)] {
            let plan = plan_planes(&h, &app_machine());
            println!("OTHER {what} {}", plan.describe());
            assert!(plan.tensors.is_empty(), "{what}");
        }
        let mut odd = copy(&header);
        let t = odd
            .tensors
            .iter_mut()
            .find(|t| t.name == "blk.3.ffn_down.weight")
            .unwrap();
        t.dims = vec![17408, 4096];
        let plan = plan_planes(&odd, &app_machine());
        assert!(
            plan.tensors
                .iter()
                .all(|t| t.key != "blk.3.ffn_down.planes"),
            "a tensor of another shape stays off"
        );
    }

    fn app_load(walk: Walk) -> PlanesLoad {
        PlanesLoad {
            draft: true,
            clef: false,
            files: 1,
            walk,
        }
    }

    /// The load's conditions, each alone: no DFlash2 draft, a CLEF model, a
    /// split GGUF, a device of 20 or 32 cores, a 40 GiB memory cap: each
    /// plans nothing and names why. The app load plans the full plan.
    #[test]
    fn a_load_plans_only_under_every_condition() {
        let header = fixture_header();
        let full = plan_planes(&header, &app_machine());
        let app = plan_for_load(&header, &app_machine(), &app_load(Walk::Spread));
        assert_eq!(app.tensors, full.tensors);
        let cases: [(&str, PlanesDevice, PlanesLoad); 6] = [
            (
                "no draft",
                app_machine(),
                PlanesLoad {
                    draft: false,
                    ..app_load(Walk::Product)
                },
            ),
            (
                "clef",
                app_machine(),
                PlanesLoad {
                    clef: true,
                    ..app_load(Walk::Product)
                },
            ),
            (
                "split",
                app_machine(),
                PlanesLoad {
                    files: 2,
                    ..app_load(Walk::Product)
                },
            ),
            (
                "cores 20",
                PlanesDevice {
                    cores: 20,
                    ..app_machine()
                },
                app_load(Walk::Product),
            ),
            (
                "cores 32",
                PlanesDevice {
                    cores: 32,
                    ..app_machine()
                },
                app_load(Walk::Product),
            ),
            (
                "cap 40 GiB",
                PlanesDevice {
                    max_memory_bytes: Some(40 * GIB),
                    ..app_machine()
                },
                app_load(Walk::Product),
            ),
        ];
        for (what, device, load) in cases {
            let plan = plan_for_load(&header, &device, &load);
            println!("CONDITION {what}: {}", plan.describe());
            assert!(plan.tensors.is_empty(), "{what} plans nothing");
            assert_eq!(plan.store_identity(), None, "{what}");
            assert_eq!(plan.left_out.len(), 1, "{what} names one reason");
        }
    }

    /// A load plans on the product walk in both activations modes: the walk
    /// is off on every tensor and the plan adds nothing to the store
    /// identity; the 27B app load keeps 164 tensors. The spread walk (tests
    /// only) keeps the law's walk and
    /// adds the planes id of the plan.
    /// Mutation walk-spread (a load takes the spread walk) → red.
    #[test]
    fn a_load_takes_the_product_walk_and_adds_no_identity_part() {
        let header = fixture_header();
        let load = PlanesLoad::of_load(true, false, 1);
        assert_eq!(load.walk, Walk::Product, "the load's walk");
        let product = plan_for_load(&header, &app_machine(), &load);
        let spread = plan_for_load(&header, &app_machine(), &app_load(Walk::Spread));
        assert_eq!(product.tensors.len(), 164);
        assert!(product.tensors.iter().all(|t| !t.walk));
        assert!(spread.tensors.iter().all(|t| t.walk));
        assert_eq!(product.store_identity(), None);
        let id = spread
            .store_identity()
            .expect("the spread walk names its planes");
        assert_eq!(id, spread.identity());
        assert!(id.starts_with("planes-route:"));
        println!("PLANID spread {id}");
        println!("PLANDESC product {}", product.describe());
    }

    /// A planned tensor whose source the import does not read in whole
    /// 64-row bands leaves the plan with its reason; a merged gate|up leaves
    /// when either half is not banded.
    #[test]
    fn a_tensor_not_read_in_bands_leaves_the_plan() {
        let header = fixture_header();
        let plan = plan_for_load(&header, &app_machine(), &app_load(Walk::Product));
        let merged = plan
            .tensors
            .iter()
            .find(|t| t.role == Role::GateUp)
            .expect("a merged gate|up")
            .clone();
        let up_source = merged.sources[1].0.clone();
        let down = "blk.3.ffn_down.weight";
        let kept = plan
            .clone()
            .keep_banded(|name| name != up_source && name != down);
        assert_eq!(kept.tensors.len(), plan.tensors.len() - 2);
        assert!(kept.tensors.iter().all(|t| t.key != merged.key));
        assert!(
            kept.tensors
                .iter()
                .all(|t| t.key != "blk.3.ffn_down.planes")
        );
        for key in [merged.key.as_str(), "blk.3.ffn_down.planes"] {
            assert!(
                kept.left_out
                    .iter()
                    .any(|(what, why)| what == key && *why == "not read in 64-row bands"),
                "{key} is named"
            );
        }
        assert_eq!(kept.walk, plan.walk);
    }

    /// The product walk keeps a tensor only at the Tiled64 route's K splits
    /// for its shape on the device: a tensor planned at other splits leaves
    /// with its reason. On 40 cores the plan of record stays whole; on 48,
    /// 64 and 80 cores every kept tensor splits as Tiled64 does.
    /// Mutation split-check-off → red.
    #[test]
    fn the_product_walk_keeps_only_the_tiled64_splits() {
        let header = fixture_header();
        let plan = plan_for_load(&header, &app_machine(), &app_load(Walk::Product));
        let mut odd = plan.clone();
        let down = odd
            .tensors
            .iter_mut()
            .find(|t| t.role == Role::Down)
            .expect("a planned down");
        let key = down.key.clone();
        down.splits = 2;
        let kept = odd.keep_tiled_splits(40);
        assert!(
            kept.tensors.iter().all(|t| t.key != key)
                && kept.left_out.iter().any(|(what, why)| {
                    *what == key && *why == "K splits differ from the Tiled64 route's"
                }),
            "a tensor whose K splits differ from the Tiled64 route's leaves the plan"
        );
        assert_eq!(kept.tensors.len(), plan.tensors.len() - 1);
        assert_eq!(plan.tensors.len(), 164, "40 cores keep the plan of record");
        for cores in [40u32, 48, 64, 80] {
            let device = PlanesDevice {
                cores,
                ..app_machine()
            };
            let product = plan_for_load(&header, &device, &app_load(Walk::Product));
            let spread = plan_for_load(&header, &device, &app_load(Walk::Spread));
            for t in &product.tensors {
                assert_eq!(
                    Some(t.splits),
                    tiled_splits(t.rows, t.k, cores),
                    "{} on {cores} cores",
                    t.key
                );
            }
            let differ: Vec<&str> = product
                .left_out
                .iter()
                .filter(|(_, why)| {
                    matches!(
                        *why,
                        "K splits differ from the Tiled64 route's"
                            | "its gate|up partner left the plan"
                    )
                })
                .map(|(what, _)| what.as_str())
                .collect();
            println!(
                "SPLITPIN cores={cores} spread={} product={} left={} first={:?}",
                spread.tensors.len(),
                product.tensors.len(),
                differ.len(),
                differ.first()
            );
            println!("SPLITPIN cores={cores} {}", product.describe());
            assert_eq!(
                product.tensors.len() + differ.len(),
                spread.tensors.len(),
                "{cores} cores: every spread tensor is kept or named"
            );
        }
    }

    /// A gate or up that leaves the plan takes its pair partner with it (one
    /// half of a pair never runs): an up whose source is not banded drops
    /// the layer's gate too, named.
    /// Mutation pair-split → red.
    #[test]
    fn a_gate_up_pair_leaves_the_plan_together() {
        let plan = plan_for_load(&fixture_header(), &app_machine(), &app_load(Walk::Product));
        let up = plan
            .tensors
            .iter()
            .find(|t| t.role == Role::Up)
            .expect("a planned up")
            .clone();
        let gate = plan
            .tensors
            .iter()
            .find(|t| t.role == Role::Gate && t.layer == up.layer)
            .expect("its gate")
            .clone();
        let kept = plan.clone().keep_banded(|name| name != up.sources[0].0);
        assert!(
            kept.tensors.iter().all(|t| t.key != gate.key),
            "a gate/up pair leaves the plan together"
        );
        assert!(kept.tensors.iter().all(|t| t.key != up.key));
        assert_eq!(kept.tensors.len(), plan.tensors.len() - 2);
        assert!(
            kept.left_out
                .iter()
                .any(|(what, why)| *what == gate.key
                    && *why == "its gate|up partner left the plan"),
            "the partner is named"
        );
    }

    /// The pinned files, relative to `crates/mlx-sys/src`, in byte order of
    /// the path; hidden files are not part of the sources.
    fn route_source_files(sys_src: &std::path::Path) -> Vec<String> {
        fn walk(dir: &std::path::Path, rel: &str, out: &mut Vec<String>) {
            for entry in std::fs::read_dir(dir).expect("read the planes sources") {
                let entry = entry.expect("a directory entry");
                let name = entry.file_name().into_string().expect("a UTF-8 file name");
                if name.starts_with('.') {
                    continue;
                }
                let rel = format!("{rel}/{name}");
                if entry.file_type().expect("a file type").is_dir() {
                    walk(&entry.path(), &rel, out);
                } else {
                    out.push(rel);
                }
            }
        }
        let mut files = vec![
            "mlx_gguf_planes.h".to_string(),
            "mlx_gguf_planes.cpp".to_string(),
        ];
        walk(
            &sys_src.join("metal/splash_gguf"),
            "metal/splash_gguf",
            &mut files,
        );
        files.sort();
        files
    }

    /// sha256 over every pinned file: its path, a NUL, its length (u64 LE)
    /// and its bytes; `flip` = (file index, byte) inverts that byte's low bit.
    fn route_sources_hash(flip: Option<(usize, usize)>) -> String {
        let sys_src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../mlx-sys/src");
        let mut h = Sha256::new();
        for (i, rel) in route_source_files(&sys_src).iter().enumerate() {
            let mut bytes = std::fs::read(sys_src.join(rel)).expect("read a planes source");
            if let Some((file, at)) = flip
                && file == i
            {
                let at = at.min(bytes.len() - 1);
                bytes[at] ^= 1;
            }
            h.update(rel.as_bytes());
            h.update([0u8]);
            h.update((bytes.len() as u64).to_le_bytes());
            h.update(&bytes);
        }
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }

    fn route_pin(hash: &str) -> std::result::Result<(), String> {
        let pinned = ROUTE_SOURCES[ROUTE_SOURCES.len() - 1];
        if hash == pinned {
            Ok(())
        } else {
            Err(format!(
                "the planes route sources hash to {hash}, pinned {pinned} (version {ROUTE_VERSION}): \
                 append the new hash to ROUTE_SOURCES"
            ))
        }
    }

    #[test]
    fn route_version_pins_the_planes_sources() {
        let sys_src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../mlx-sys/src");
        let files = route_source_files(&sys_src);
        assert!(
            files
                .iter()
                .any(|f| f == "metal/splash_gguf/metal_flags.txt")
        );
        let hash = route_sources_hash(None);
        println!(
            "ROUTEPIN files={} version={ROUTE_VERSION} hash={hash}",
            files.len()
        );
        if let Err(e) = route_pin(&hash) {
            panic!("{e}");
        }
    }

    /// One byte changed in the first, a middle and the last pinned file
    /// fails the pin.
    #[test]
    fn a_changed_planes_source_byte_fails_the_route_pin() {
        let sys_src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../mlx-sys/src");
        let last = route_source_files(&sys_src).len() - 1;
        for flip in [(0, 0), (last / 2, 4096), (last, usize::MAX)] {
            let hash = route_sources_hash(Some(flip));
            assert!(
                route_pin(&hash).is_err(),
                "a flipped byte {flip:?} kept the pin"
            );
            println!("ROW mutation source_flip file={} red=1", flip.0);
        }
    }
}
