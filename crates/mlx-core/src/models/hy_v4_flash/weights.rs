//! Resident weights: every tensor of the scan except routed experts and the
//! Engram table, read once by aligned `F_NOCACHE` `pread`.
//!
//! | file | held as |
//! |---|---|
//! | Q8_0 `token_embd` | `Dense::Q8` codes + f16 scales (row gathers) |
//! | Q8_0 indexer compressor matrices | f32 array |
//! | other Q8_0 | `Dense::Qmm` (f32 scales, bias `-128·s`) |
//! | F32 | f32 array |
//! | BF16 (`output`, the MTP layer) | bf16 array (MTP compressor: f32) |
//!
//! Tensors of rank 3 (`attn_k_b` `[H, kv_lora, nope]`, `attn_v_b`
//! `[H, v, kv_lora]`) are held as 2-D `[H·rows, k]`, the layout
//! `Dense::grouped_qmm_t(x, H)` splits back into heads.

use napi::{Error, Result};

use super::config::{Attn, HyV4FlashConfig};
use super::scan::{Located, Role, Scan};
use crate::array::alloc::OutLayout;
use crate::array::{DType, MxArray};
use crate::cache_limit::{CacheLimitGuard, coordinator};
use crate::models::deepseek_v41::store::Dense;
use crate::moe_stream::io::{AlignedBuf, Extent, ShardSet};
use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};
use crate::utils::gguf_cpu_repack::decode_affine_band_into;
use crate::utils::gguf_gpu::RepackDst;

pub const TOKEN_EMBD: &str = "token_embd.weight";
pub const OUTPUT: &str = "output.weight";
const COMPRESSOR_MATS: [&str; 2] = [
    "indexer_compressor_value.weight",
    "indexer_compressor_gate.weight",
];
const Q8_0_VALUES: usize = 32;
const Q8_0_BYTES: usize = 34;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 weights: {msg}"))
}

/// What one resident tensor becomes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// Q8_0 as affine-8 codes + f16 scales (`Dense::Q8`).
    Codes,
    /// Q8_0 as affine-8 codes + f32 scales and biases (`Dense::Qmm`).
    Qmm,
    /// An f32 array from F32, Q8_0 or BF16 data.
    F32,
    /// A bf16 array from BF16 data.
    Bf16,
}

fn is_compressor_mat(name: &str) -> bool {
    COMPRESSOR_MATS
        .iter()
        .any(|s| name.strip_suffix(s).is_some_and(|p| p.ends_with('.')))
}

/// The form of a resident tensor; refuses experts, the Engram table and any
/// type the file does not use for that role.
pub fn form(role: Role, info: &GgufTensorInfo) -> Result<Form> {
    use GgufTensorType::{BF16, F32, Q8_0};
    let name = info.name.as_str();
    let mat = is_compressor_mat(name);
    Ok(match (role, info.tensor_type) {
        (Role::Dense, Q8_0) if name == TOKEN_EMBD => Form::Codes,
        (Role::Dense, Q8_0) if mat => Form::F32,
        (Role::Dense, Q8_0) => Form::Qmm,
        (Role::Dense | Role::Mtp, F32) => Form::F32,
        (Role::Dense, BF16) if name == OUTPUT => Form::Bf16,
        (Role::Mtp, BF16) if mat => Form::F32,
        (Role::Mtp, BF16) => Form::Bf16,
        (Role::Expert | Role::EngramTable, _) => {
            return Err(bad(format!("'{name}' is not a resident tensor")));
        }
        (_, t) => {
            return Err(bad(format!(
                "'{name}': type {} is not supported for a {role:?} tensor",
                t.name()
            )));
        }
    })
}

/// Bytes of the arrays `dense_from_raw` makes for this tensor.
pub fn resident_bytes(role: Role, info: &GgufTensorInfo) -> Result<u64> {
    let n = info.num_elements()?;
    let r = match form(role, info)? {
        Form::Codes => n.checked_add(n / 16),
        Form::Qmm => n.checked_add(n / 4),
        Form::F32 => n.checked_mul(4),
        Form::Bf16 => n.checked_mul(2),
    };
    r.ok_or_else(|| bad(format!("'{}': resident size overflows u64", info.name)))
}

/// Resident bytes of every tensor with `role`, from the headers alone.
pub fn role_resident(scan: &Scan, role: Role) -> Result<u64> {
    scan.with_role(role).try_fold(0u64, |acc, (_, loc)| {
        acc.checked_add(resident_bytes(role, &loc.info)?)
            .ok_or_else(|| bad("resident bytes overflow u64"))
    })
}

/// The held shape: the MLX shape, with every axis but the last folded into
/// rows for rank 3 and up.
pub fn held_shape(info: &GgufTensorInfo) -> Result<Vec<i64>> {
    let s = info.mlx_shape();
    if s.len() <= 2 {
        return Ok(s);
    }
    let (k, lead) = s
        .split_last()
        .ok_or_else(|| bad(format!("'{}' has no dims", info.name)))?;
    let rows = lead
        .iter()
        .try_fold(1i64, |a, &d| a.checked_mul(d))
        .ok_or_else(|| bad(format!("'{}': row count overflows", info.name)))?;
    Ok(vec![rows, *k])
}

fn check_lens(raw: usize, out: usize, width: usize, what: &str) -> Result<()> {
    if out.checked_mul(width) != Some(raw) {
        return Err(bad(format!(
            "{what}: {raw} bytes for {out} values of {width} B"
        )));
    }
    Ok(())
}

/// Little-endian u16 words.
pub fn le_u16_into(raw: &[u8], out: &mut [u16]) -> Result<()> {
    check_lens(raw.len(), out.len(), 2, "u16 words")?;
    for (o, c) in out.iter_mut().zip(raw.as_chunks::<2>().0) {
        *o = u16::from_le_bytes(*c);
    }
    Ok(())
}

/// Little-endian f32 values.
pub fn le_f32_into(raw: &[u8], out: &mut [f32]) -> Result<()> {
    check_lens(raw.len(), out.len(), 4, "f32 values")?;
    for (o, c) in out.iter_mut().zip(raw.as_chunks::<4>().0) {
        *o = f32::from_le_bytes(*c);
    }
    Ok(())
}

/// Little-endian bf16 values widened to f32 (exact).
pub fn bf16_into_f32(raw: &[u8], out: &mut [f32]) -> Result<()> {
    check_lens(raw.len(), out.len(), 2, "bf16 values")?;
    for (o, c) in out.iter_mut().zip(raw.as_chunks::<2>().0) {
        *o = f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16);
    }
    Ok(())
}

/// Q8_0 blocks (f16 `d`, 32 × i8 `q`) as `d·q` in f32 (exact).
pub fn q8_0_into_f32(raw: &[u8], out: &mut [f32]) -> Result<()> {
    let blocks = raw.len() / Q8_0_BYTES;
    if !raw.len().is_multiple_of(Q8_0_BYTES) || blocks.checked_mul(Q8_0_VALUES) != Some(out.len()) {
        return Err(bad(format!(
            "Q8_0: {} bytes for {} values",
            raw.len(),
            out.len()
        )));
    }
    for (b, o) in raw
        .as_chunks::<Q8_0_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<Q8_0_VALUES>().0)
    {
        let d = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
        for (v, &q) in o.iter_mut().zip(&b[2..]) {
            *v = d * f32::from(i8::from_le_bytes([q]));
        }
    }
    Ok(())
}

/// A fresh `shape` / `dtype` array of `n` values of `T`, written by `fill`.
/// Model thread only (`alloc_uninit`).
fn alloc_filled<T>(
    shape: &[i64],
    dtype: DType,
    n: usize,
    fill: impl FnOnce(&mut [T]) -> Result<()>,
) -> Result<MxArray> {
    let (a, ptr, bytes) = MxArray::alloc_uninit(shape, dtype)?;
    if n.checked_mul(size_of::<T>()) != Some(bytes) || ptr.0.align_offset(align_of::<T>()) != 0 {
        return Err(bad(format!(
            "destination of {bytes} B does not hold {n} aligned values"
        )));
    }
    // SAFETY: `alloc_uninit` returned `bytes` writable bytes at `ptr`, alive
    // while `a` is; size and alignment are checked above, and no other user
    // of the buffer exists before `a` is returned.
    let out = unsafe { std::slice::from_raw_parts_mut(ptr.0.cast::<T>(), n) };
    fill(out)?;
    Ok(a)
}

fn elements(info: &GgufTensorInfo) -> Result<usize> {
    usize::try_from(info.num_elements()?)
        .map_err(|_| bad(format!("'{}': element count exceeds usize", info.name)))
}

fn f32_array(info: &GgufTensorInfo, raw: &[u8], shape: &[i64]) -> Result<MxArray> {
    let decode: fn(&[u8], &mut [f32]) -> Result<()> = match info.tensor_type {
        GgufTensorType::F32 => le_f32_into,
        GgufTensorType::BF16 => bf16_into_f32,
        GgufTensorType::Q8_0 => q8_0_into_f32,
        t => {
            return Err(bad(format!(
                "'{}': no f32 decode for type {}",
                info.name,
                t.name()
            )));
        }
    };
    alloc_filled(shape, DType::Float32, elements(info)?, |out| {
        decode(raw, out)
    })
}

fn bf16_array(info: &GgufTensorInfo, raw: &[u8], shape: &[i64]) -> Result<MxArray> {
    if info.tensor_type != GgufTensorType::BF16 {
        return Err(bad(format!(
            "'{}': type {} is not BF16",
            info.name,
            info.tensor_type.name()
        )));
    }
    alloc_filled(shape, DType::BFloat16, elements(info)?, |out| {
        le_u16_into(raw, out)
    })
}

fn q8_dense(info: &GgufTensorInfo, raw: &[u8], form: Form) -> Result<Dense> {
    if info.tensor_type != GgufTensorType::Q8_0 {
        return Err(bad(format!(
            "'{}': {form:?} needs Q8_0, not {}",
            info.name,
            info.tensor_type.name()
        )));
    }
    let (dst, ptrs) = RepackDst::alloc(info, OutLayout::RowMajor)?;
    decode_affine_band_into(info.tensor_type, raw, 0, ptrs.rows, ptrs.k, &ptrs)?;
    let scales = dst
        .scales
        .ok_or_else(|| bad(format!("'{}': Q8_0 repack gave no scales", info.name)))?;
    let rows = i64::try_from(ptrs.rows).map_err(|_| bad("row count exceeds i64"))?;
    let fold = |a: &MxArray| -> Result<MxArray> {
        let nd = a.ndim()?;
        a.reshape(&[rows, a.shape_at(nd.saturating_sub(1))?])
    };
    let (w, scales) = (fold(&dst.weight)?, fold(&scales)?);
    if form == Form::Codes {
        MxArray::eval_arrays_with_context(&[&w, &scales], "hy_v4_flash q8 codes")?;
        return Ok(Dense::Q8 { w, scales });
    }
    let s = scales.astype(DType::Float32)?;
    let b = s.mul_scalar(-128.0)?;
    MxArray::eval_arrays_with_context(&[&w, &s, &b], "hy_v4_flash q8 scales")?;
    Ok(Dense::Qmm {
        w,
        scales: s,
        biases: b,
    })
}

/// One tensor's GGUF bytes as `form`. Model thread only (`alloc_uninit`).
pub fn dense_from_raw(info: &GgufTensorInfo, raw: &[u8], form: Form) -> Result<Dense> {
    let want = info.data_size()?;
    if raw.len() as u64 != want {
        return Err(bad(format!(
            "'{}': {} bytes, expected {want}",
            info.name,
            raw.len()
        )));
    }
    let shape = held_shape(info)?;
    Ok(match form {
        Form::Codes | Form::Qmm => q8_dense(info, raw, form)?,
        Form::F32 => Dense::Array(f32_array(info, raw, &shape)?),
        Form::Bf16 => Dense::Array(bf16_array(info, raw, &shape)?),
    })
}

fn extent(loc: &Located) -> Result<Extent> {
    Ok(Extent {
        shard: loc.shard,
        offset: loc.abs_offset()?,
        len: loc.info.data_size()?,
    })
}

/// Reads resident tensors of one scan, one aligned `pread` each.
pub struct Reader<'a> {
    scan: &'a Scan,
    shards: &'a ShardSet,
    buf: AlignedBuf,
    /// GGUF bytes read.
    pub read: u64,
    /// Bytes of the arrays made.
    pub bytes: u64,
    /// Tensors made.
    pub count: usize,
}

impl<'a> Reader<'a> {
    /// `shards` must be opened from `scan.paths`.
    pub fn new(scan: &'a Scan, shards: &'a ShardSet) -> Self {
        Self {
            scan,
            shards,
            buf: AlignedBuf::new(),
            read: 0,
            bytes: 0,
            count: 0,
        }
    }

    pub fn dense(&mut self, name: &str) -> Result<Dense> {
        let scan = self.scan;
        let entry = scan
            .tensors
            .get(name)
            .ok_or_else(|| bad(format!("tensor '{name}' missing")))?;
        let form = form(entry.role, &entry.loc.info)?;
        let e = extent(&entry.loc)?;
        let raw = self.shards.read_aligned(e, &mut self.buf)?;
        let d = dense_from_raw(&entry.loc.info, raw, form)?;
        self.read += e.len;
        self.bytes += d.nbytes() as u64;
        self.count += 1;
        Ok(d)
    }

    /// A tensor used as a plain array (norms, biases, HC, sinks, APE).
    pub fn plain(&mut self, name: &str) -> Result<MxArray> {
        self.dense(name)?.array()
    }

    pub fn opt_plain(&mut self, name: &str) -> Result<Option<MxArray>> {
        if self.scan.opt(name).is_none() {
            return Ok(None);
        }
        self.plain(name).map(Some)
    }

    /// Experts `lo..hi` of a BF16 expert tensor (ggml `{k, rows, E}`) as one
    /// bf16 `[hi - lo, rows, k]` array. Counts `read`, not `bytes`.
    pub fn bf16_experts(&mut self, loc: &Located, lo: usize, hi: usize) -> Result<MxArray> {
        let info = &loc.info;
        let &[k, rows, n] = info.dims.as_slice() else {
            return Err(bad(format!("'{}' is not a 3-D expert tensor", info.name)));
        };
        if info.tensor_type != GgufTensorType::BF16 {
            return Err(bad(format!("'{}' is not BF16", info.name)));
        }
        if lo >= hi || hi as u64 > n {
            return Err(bad(format!("'{}': experts {lo}..{hi} of {n}", info.name)));
        }
        let span = |x: usize| -> Result<u64> {
            (x as u64)
                .checked_mul(k)
                .and_then(|v| v.checked_mul(rows))
                .and_then(|v| v.checked_mul(2))
                .ok_or_else(|| bad(format!("'{}': expert span overflows", info.name)))
        };
        let e = Extent {
            shard: loc.shard,
            offset: loc
                .abs_offset()?
                .checked_add(span(lo)?)
                .ok_or_else(|| bad("expert offset overflows"))?,
            len: span(hi - lo)?,
        };
        let raw = self.shards.read_aligned(e, &mut self.buf)?;
        let dim = |x: u64| i64::try_from(x).map_err(|_| bad("expert dim exceeds i64"));
        let shape = [dim((hi - lo) as u64)?, dim(rows)?, dim(k)?];
        let n_vals = usize::try_from(e.len / 2).map_err(|_| bad("expert span exceeds usize"))?;
        let a = alloc_filled(&shape, DType::BFloat16, n_vals, |out| le_u16_into(raw, out))?;
        self.read += e.len;
        Ok(a)
    }

    fn check_all(&self, role: Role) -> Result<()> {
        let want = self.scan.with_role(role).count();
        let got = self.count;
        if got != want {
            return Err(bad(format!("loaded {got} of the {want} {role:?} tensors")));
        }
        Ok(())
    }
}

fn blk(il: usize, s: &str) -> String {
    format!("blk.{il}.{s}.weight")
}

/// `{prefix}_fn` `[2·hc or hc, hc·d]`, `{prefix}_base`, `{prefix}_scale`.
pub struct HcWeights {
    pub weight: MxArray,
    pub base: MxArray,
    pub scale: MxArray,
}

impl HcWeights {
    fn load(r: &mut Reader, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: r.plain(&format!("{prefix}_fn.weight"))?,
            base: r.plain(&format!("{prefix}_base.weight"))?,
            scale: r.plain(&format!("{prefix}_scale.weight"))?,
        })
    }
}

pub struct AttnWeights {
    pub norm: MxArray,
    pub sinks: MxArray,
    pub q_a: Dense,
    pub q_a_norm: MxArray,
    pub q_b: Dense,
    pub kv_a: Dense,
    pub kv_a_norm: MxArray,
    /// `[H·kv_lora, qk_nope]`.
    pub k_b: Dense,
    /// `[H·head_dim_v, kv_lora]`.
    pub v_b: Dense,
    pub output: Dense,
    pub gate: Dense,
    /// GGUF `post_attention_norm` (HF `post_proj_norm`).
    pub post_norm: MxArray,
}

impl AttnWeights {
    fn load(r: &mut Reader, il: usize) -> Result<Self> {
        Ok(Self {
            norm: r.plain(&blk(il, "attn_norm"))?,
            sinks: r.plain(&blk(il, "attn_sinks"))?,
            q_a: r.dense(&blk(il, "attn_q_a"))?,
            q_a_norm: r.plain(&blk(il, "attn_q_a_norm"))?,
            q_b: r.dense(&blk(il, "attn_q_b"))?,
            kv_a: r.dense(&blk(il, "attn_kv_a_mqa"))?,
            kv_a_norm: r.plain(&blk(il, "attn_kv_a_norm"))?,
            k_b: r.dense(&blk(il, "attn_k_b"))?,
            v_b: r.dense(&blk(il, "attn_v_b"))?,
            output: r.dense(&blk(il, "attn_output"))?,
            gate: r.dense(&blk(il, "attn_gate"))?,
            post_norm: r.plain(&blk(il, "post_attention_norm"))?,
        })
    }
}

pub struct IndexerWeights {
    pub attn_k: Dense,
    pub k_norm: MxArray,
    pub q_b: Dense,
    pub proj: Dense,
    /// f32 `[head_dim, head_dim]`.
    pub compressor_value: MxArray,
    /// f32 `[head_dim, head_dim]`.
    pub compressor_gate: MxArray,
    /// `[kpool, head_dim]`.
    pub compressor_ape: MxArray,
    pub compressor_norm: MxArray,
}

impl IndexerWeights {
    fn load(r: &mut Reader, il: usize) -> Result<Self> {
        Ok(Self {
            attn_k: r.dense(&blk(il, "indexer.attn_k"))?,
            k_norm: r.plain(&blk(il, "indexer.k_norm"))?,
            q_b: r.dense(&blk(il, "indexer.attn_q_b"))?,
            proj: r.dense(&blk(il, "indexer.proj"))?,
            compressor_value: r.plain(&blk(il, "indexer_compressor_value"))?,
            compressor_gate: r.plain(&blk(il, "indexer_compressor_gate"))?,
            compressor_ape: r.plain(&blk(il, "indexer_compressor_ape"))?,
            compressor_norm: r.plain(&blk(il, "indexer_compressor_norm"))?,
        })
    }
}

pub struct EngramWeights {
    /// `[d·(hc + 1), n_cols·head_dim]`: hc keys, then the value.
    pub kv: Dense,
    /// `[hc + 1, d]`.
    pub kv_b: MxArray,
    /// `[hc, d]`.
    pub gate: MxArray,
    /// `[n_cols, head_dim]`.
    pub lookup_norm: MxArray,
    pub pre_gamma: MxArray,
    /// `[hc, d]`.
    pub hc_scale: MxArray,
}

impl EngramWeights {
    fn load(r: &mut Reader, il: usize) -> Result<Self> {
        Ok(Self {
            kv: r.dense(&blk(il, "engram_kv"))?,
            kv_b: r.plain(&blk(il, "engram_kv_b"))?,
            gate: r.plain(&blk(il, "engram_gate"))?,
            lookup_norm: r.plain(&blk(il, "engram_lookup_norm"))?,
            pre_gamma: r.plain(&blk(il, "engram_pre_gamma"))?,
            hc_scale: r.plain(&blk(il, "engram_hc_scale"))?,
        })
    }
}

/// Non-expert weights of a routed-MoE layer.
pub struct MoeWeights {
    /// f32 `[n_experts, d]`.
    pub router: Dense,
    /// `exp_probs_b`, when the file has it.
    pub router_bias: Option<MxArray>,
    /// `[latent, d]`.
    pub routed_down: Dense,
    /// `[d, latent]`.
    pub routed_up: Dense,
    pub shared_gate: Dense,
    pub shared_up: Dense,
    pub shared_down: Dense,
}

impl MoeWeights {
    fn load(r: &mut Reader, il: usize) -> Result<Self> {
        Ok(Self {
            router: r.dense(&blk(il, "ffn_gate_inp"))?,
            router_bias: r.opt_plain(&format!("blk.{il}.exp_probs_b.bias"))?,
            routed_down: r.dense(&blk(il, "ffn_routed_down"))?,
            routed_up: r.dense(&blk(il, "ffn_routed_up"))?,
            shared_gate: r.dense(&blk(il, "ffn_gate_shexp"))?,
            shared_up: r.dense(&blk(il, "ffn_up_shexp"))?,
            shared_down: r.dense(&blk(il, "ffn_down_shexp"))?,
        })
    }
}

pub enum Ffn {
    /// The leading dense FFN.
    Mlp {
        gate: Dense,
        up: Dense,
        down: Dense,
    },
    Moe(MoeWeights),
}

/// The MTP layer's input side.
pub struct NextnWeights {
    pub enorm: MxArray,
    pub hnorm: MxArray,
    /// `[d, 2·d]` over `[rms(e)·enorm ; rms(h)·hnorm]`.
    pub eh_proj: Dense,
    pub shared_head_norm: MxArray,
}

impl NextnWeights {
    fn load(r: &mut Reader, il: usize) -> Result<Self> {
        Ok(Self {
            enorm: r.plain(&blk(il, "nextn.enorm"))?,
            hnorm: r.plain(&blk(il, "nextn.hnorm"))?,
            eh_proj: r.dense(&blk(il, "nextn.eh_proj"))?,
            shared_head_norm: r.plain(&blk(il, "nextn.shared_head_norm"))?,
        })
    }
}

/// How a layer joins the residual: trunk layers mix the HC streams, the
/// MTP layer has one stream fed by `eh_proj`.
pub enum Residual {
    Hc { attn: HcWeights, ffn: HcWeights },
    Mtp(NextnWeights),
}

pub struct LayerWeights {
    pub attn: AttnWeights,
    pub indexer: Option<IndexerWeights>,
    pub engram: Option<EngramWeights>,
    pub residual: Residual,
    pub ffn_norm: MxArray,
    pub post_ffw_norm: MxArray,
    pub ffn: Ffn,
}

impl LayerWeights {
    fn load(r: &mut Reader, cfg: &HyV4FlashConfig, il: usize) -> Result<Self> {
        let l = cfg
            .layers
            .get(il)
            .ok_or_else(|| bad(format!("layer {il} of {}", cfg.layers.len())))?;
        let attn = AttnWeights::load(r, il)?;
        let indexer = match l.attn {
            Attn::Indexed => Some(IndexerWeights::load(r, il)?),
            Attn::Sliding => None,
        };
        let engram = match l.engram {
            Some(_) => Some(EngramWeights::load(r, il)?),
            None => None,
        };
        let residual = if l.mtp {
            Residual::Mtp(NextnWeights::load(r, il)?)
        } else {
            Residual::Hc {
                attn: HcWeights::load(r, &format!("blk.{il}.hc_attn"))?,
                ffn: HcWeights::load(r, &format!("blk.{il}.hc_ffn"))?,
            }
        };
        let ffn = if l.dense_ffn {
            Ffn::Mlp {
                gate: r.dense(&blk(il, "ffn_gate"))?,
                up: r.dense(&blk(il, "ffn_up"))?,
                down: r.dense(&blk(il, "ffn_down"))?,
            }
        } else {
            Ffn::Moe(MoeWeights::load(r, il)?)
        };
        Ok(Self {
            attn,
            indexer,
            engram,
            residual,
            ffn_norm: r.plain(&blk(il, "ffn_norm"))?,
            post_ffw_norm: r.plain(&blk(il, "post_ffw_norm"))?,
            ffn,
        })
    }
}

/// Every resident tensor, registered with the cache-limit coordinator.
pub struct Resident {
    /// Q8_0 codes, rows only.
    pub token_embd: Dense,
    /// bf16 `[vocab, d]`.
    pub output: Dense,
    pub output_norm: MxArray,
    pub output_hc: HcWeights,
    /// Trunk layers.
    pub layers: Vec<LayerWeights>,
    /// MTP layers; empty until `load_mtp`.
    pub mtp: Vec<LayerWeights>,
    /// GGUF bytes read.
    pub read: u64,
    /// Bytes of the resident arrays.
    pub bytes: u64,
    guard: CacheLimitGuard,
}

impl Resident {
    /// Trunk tensors (`Role::Dense`). Model thread only.
    pub fn load(scan: &Scan, shards: &ShardSet) -> Result<Self> {
        let out = Self::read_trunk(scan, shards);
        if out.is_err() {
            // the partial set is gone; keep its buffers out of MLX's free pool
            crate::array::clear_cache();
        }
        out
    }

    fn read_trunk(scan: &Scan, shards: &ShardSet) -> Result<Self> {
        let cfg = &scan.cfg;
        let mut r = Reader::new(scan, shards);
        let token_embd = r.dense(TOKEN_EMBD)?;
        let output = r.dense(OUTPUT)?;
        let output_norm = r.plain("output_norm.weight")?;
        let output_hc = HcWeights::load(&mut r, "output_hc")?;
        let layers = (0..cfg.n_layers)
            .map(|il| LayerWeights::load(&mut r, cfg, il))
            .collect::<Result<Vec<_>>>()?;
        r.check_all(Role::Dense)?;
        Ok(Self {
            token_embd,
            output,
            output_norm,
            output_hc,
            layers,
            mtp: Vec::new(),
            read: r.read,
            bytes: r.bytes,
            guard: coordinator().register(r.bytes),
        })
    }

    /// MTP-layer tensors (`Role::Mtp`, routed experts excluded); a no-op
    /// once loaded. Model thread only.
    pub fn load_mtp(&mut self, scan: &Scan, shards: &ShardSet) -> Result<()> {
        if !self.mtp.is_empty() {
            return Ok(());
        }
        let cfg = &scan.cfg;
        let mut r = Reader::new(scan, shards);
        let mtp = (cfg.n_layers..cfg.layers.len())
            .map(|il| LayerWeights::load(&mut r, cfg, il))
            .collect::<Result<Vec<_>>>()
            .and_then(|m| r.check_all(Role::Mtp).map(|()| m));
        let mtp = match mtp {
            Ok(m) => m,
            Err(e) => {
                crate::array::clear_cache();
                return Err(e);
            }
        };
        self.mtp = mtp;
        self.read += r.read;
        self.bytes += r.bytes;
        self.guard = coordinator().register(self.bytes);
        Ok(())
    }
}
