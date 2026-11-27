//! Shard headers -> where every tensor is and what it is. Every tensor of the
//! file must be one the config expects, with the exact ggml dims of the PR
//! #32517 loader (`load_arch_tensors`): expert gate/up `{latent, ff, E}`,
//! down `{ff, latent, E}`. Checking by name and dims is what refuses a
//! swapped orientation; on Hy4 latent == ff, so shape alone cannot.

use std::collections::{HashMap, HashSet};
use std::io::BufReader;
use std::path::{Path, PathBuf};

use napi::{Error, Result};

use super::config::{Attn, HyV4FlashConfig};
use crate::moe_stream::io::open_nocache;
use crate::utils::gguf::{GgufFile, GgufTensorInfo, GgufTensorType, parse_gguf_reader};
use crate::utils::gguf_source::{check_split_parts, check_tensor_bounds, reject_split};

pub const ENGRAM_TABLE: &str = "per_layer_token_embd.weight";
const EXP_PROBS_B: &str = "exp_probs_b.bias";

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 GGUF: {msg}"))
}

#[derive(Debug, Clone)]
pub struct Located {
    pub shard: usize,
    /// The shard's data section start.
    pub data_offset: u64,
    pub info: GgufTensorInfo,
}

impl Located {
    pub fn abs_offset(&self) -> Result<u64> {
        self.data_offset
            .checked_add(self.info.offset)
            .ok_or_else(|| bad(format!("'{}' offset overflows u64", self.info.name)))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Trunk weights other than routed experts.
    Dense,
    /// `ffn_{gate,up,down}_exps` of any layer.
    Expert,
    /// MTP-layer weights other than routed experts.
    Mtp,
    EngramTable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertClass {
    /// gate Q4_K, up Q4_K, down Q6_K.
    Q6kDown,
    /// gate, up and down Q4_K.
    Q4kDown,
    /// gate, up and down NVFP4 (trunk SPARK-4Bit / SPARK-4Bit-Fast).
    Nvfp4,
    /// gate, up and down BF16 (MTP layer only).
    Bf16,
    /// gate, up and down Q8_0 (MTP layer only, SPARK-4Bit / SPARK-4Bit-Fast).
    Q8_0,
}

#[derive(Debug, Clone)]
pub struct ExpertLayer {
    pub class: ExpertClass,
    pub gate: Located,
    pub up: Located,
    pub down: Located,
}

/// The MTP layer `blk.{layer}`: scanned with the file, never a trunk layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtpLayer {
    /// Index into `cfg.layers` and `Scan::experts`.
    pub layer: usize,
    /// Stored bytes of every `blk.{layer}.*` tensor, routed experts included.
    pub bytes: u64,
}

#[derive(Debug, Clone)]
pub struct Entry {
    pub role: Role,
    pub loc: Located,
}

pub struct Scan {
    pub paths: Vec<PathBuf>,
    pub cfg: HyV4FlashConfig,
    pub tensors: HashMap<String, Entry>,
    /// One per layer (trunk then MTP); `None` on dense-FFN layers.
    pub experts: Vec<Option<ExpertLayer>>,
    pub engram_table: Located,
    /// `None` when the file has no `nextn_predict_layers`.
    pub mtp: Option<MtpLayer>,
}

impl Scan {
    pub fn get(&self, name: &str) -> Result<&Located> {
        self.tensors
            .get(name)
            .map(|e| &e.loc)
            .ok_or_else(|| bad(format!("tensor '{name}' missing")))
    }

    pub fn opt(&self, name: &str) -> Option<&Located> {
        self.tensors.get(name).map(|e| &e.loc)
    }

    pub fn trunk_experts(&self) -> &[Option<ExpertLayer>] {
        &self.experts[..self.cfg.n_layers]
    }

    pub fn mtp_experts(&self) -> Option<&ExpertLayer> {
        self.mtp
            .as_ref()
            .and_then(|m| self.experts.get(m.layer))
            .and_then(Option::as_ref)
    }

    pub fn with_role(&self, role: Role) -> impl Iterator<Item = (&str, &Located)> {
        self.tensors
            .iter()
            .filter(move |(_, e)| e.role == role)
            .map(|(n, e)| (n.as_str(), &e.loc))
    }

    /// Reads every shard header (`paths` in shard order, shard 0 first).
    pub fn open(paths: &[PathBuf]) -> Result<Self> {
        let headers = paths
            .iter()
            .map(|p| read_header(p))
            .collect::<Result<Vec<_>>>()?;
        Self::from_headers(paths, &headers)
    }

    pub fn from_headers(paths: &[PathBuf], headers: &[(GgufFile, u64)]) -> Result<Self> {
        if paths.is_empty() || paths.len() != headers.len() {
            return Err(bad("one header per shard path is required"));
        }
        let parts: Vec<(&Path, &GgufFile)> = paths
            .iter()
            .map(PathBuf::as_path)
            .zip(headers.iter().map(|(gguf, _)| gguf))
            .collect();
        if parts.len() == 1 {
            reject_split(parts[0].1, parts[0].0)?;
        } else {
            check_split_parts(&parts)?;
        }
        for (path, (gguf, file_len)) in paths.iter().zip(headers) {
            check_bounds(gguf, *file_len, path)?;
        }
        let cfg = HyV4FlashConfig::from_gguf(&headers[0].0)?;
        let mut found: HashMap<String, Located> = HashMap::new();
        for (shard, (gguf, _)) in headers.iter().enumerate() {
            for t in &gguf.tensors {
                found.insert(
                    t.name.clone(),
                    Located {
                        shard,
                        data_offset: gguf.data_offset,
                        info: t.clone(),
                    },
                );
            }
        }
        classify(paths.to_vec(), cfg, found)
    }
}

fn read_header(path: &Path) -> Result<(GgufFile, u64)> {
    let file = open_nocache(path)?;
    let file_len = file
        .metadata()
        .map_err(|e| bad(format!("'{}': {e}", path.display())))?
        .len();
    let gguf = parse_gguf_reader(BufReader::with_capacity(1 << 20, file))
        .map_err(|e| bad(format!("'{}': {}", path.display(), e.reason)))?;
    Ok((gguf, file_len))
}

/// `check_tensor_bounds` sizes type 43 as MXFP8 (33 B / 32), which the F8
/// Engram table is not; the table gets its own 1 B/value bound.
fn check_bounds(gguf: &GgufFile, file_len: u64, path: &Path) -> Result<()> {
    let rest = GgufFile {
        version: gguf.version,
        tensor_count: gguf.tensor_count,
        metadata: HashMap::new(),
        tensors: gguf
            .tensors
            .iter()
            .filter(|t| t.name != ENGRAM_TABLE)
            .cloned()
            .collect(),
        alignment: gguf.alignment,
        data_offset: gguf.data_offset,
    };
    check_tensor_bounds(&rest, file_len, path)?;
    if let Some(t) = gguf.tensors.iter().find(|t| t.name == ENGRAM_TABLE) {
        let size = match t.tensor_type {
            GgufTensorType::MXFP8 => t.num_elements()?,
            GgufTensorType::Q8_0 => t.data_size()?,
            _ => t.num_elements()?,
        };
        let end = size
            .checked_add(gguf.data_offset)
            .and_then(|x| x.checked_add(t.offset));
        if end.is_none_or(|end| end > file_len) {
            return Err(bad(format!(
                "'{}': the Engram table runs past the end of the file",
                path.display()
            )));
        }
    }
    Ok(())
}

/// One expected tensor: full name and ggml dims.
struct Want {
    name: String,
    dims: Vec<u64>,
    role: Role,
    required: bool,
}

fn want(name: String, dims: &[usize], role: Role) -> Want {
    Want {
        name,
        dims: dims.iter().map(|&d| d as u64).collect(),
        role,
        required: true,
    }
}

fn globals(cfg: &HyV4FlashConfig) -> Vec<Want> {
    let (d, v, hc) = (cfg.dim, cfg.vocab_size, cfg.hc.count);
    vec![
        want("token_embd.weight".into(), &[d, v], Role::Dense),
        want("output_norm.weight".into(), &[d], Role::Dense),
        want("output.weight".into(), &[d, v], Role::Dense),
        want("output_hc_fn.weight".into(), &[hc * d, hc], Role::Dense),
        want("output_hc_base.weight".into(), &[hc], Role::Dense),
        want("output_hc_scale.weight".into(), &[1], Role::Dense),
        Want {
            name: ENGRAM_TABLE.into(),
            dims: vec![cfg.engram.head_dim as u64, cfg.engram_rows()],
            role: Role::EngramTable,
            required: true,
        },
    ]
}

fn layer_wants(cfg: &HyV4FlashConfig, il: usize) -> Vec<Want> {
    let l = &cfg.layers[il];
    let role = if l.mtp { Role::Mtp } else { Role::Dense };
    let (d, h, m) = (cfg.dim, l.n_heads, &cfg.mla);
    let mut w: Vec<(&str, Vec<usize>)> = vec![
        ("attn_norm", vec![d]),
        ("attn_sinks", vec![h]),
        ("attn_q_a", vec![d, m.q_lora_rank]),
        ("attn_q_a_norm", vec![m.q_lora_rank]),
        ("attn_q_b", vec![m.q_lora_rank, h * m.head_dim_qk]),
        ("attn_kv_a_mqa", vec![d, m.kv_row()]),
        ("attn_kv_a_norm", vec![m.kv_lora_rank]),
        ("attn_k_b", vec![m.qk_nope_dim, m.kv_lora_rank, h]),
        ("attn_v_b", vec![m.kv_lora_rank, m.head_dim_v, h]),
        ("attn_output", vec![h * m.head_dim_v, d]),
        ("attn_gate", vec![d, h * m.head_dim_v]),
        ("post_attention_norm", vec![d]),
        ("ffn_norm", vec![d]),
        ("post_ffw_norm", vec![d]),
    ];
    let hc = cfg.hc.count;
    if l.mtp {
        w.extend([
            ("nextn.enorm", vec![d]),
            ("nextn.hnorm", vec![d]),
            ("nextn.eh_proj", vec![2 * d, d]),
            ("nextn.shared_head_norm", vec![d]),
        ]);
    } else {
        w.extend([
            ("hc_attn_fn", vec![hc * d, 2 * hc]),
            ("hc_attn_base", vec![2 * hc]),
            ("hc_attn_scale", vec![2]),
            ("hc_ffn_fn", vec![hc * d, 2 * hc]),
            ("hc_ffn_base", vec![2 * hc]),
            ("hc_ffn_scale", vec![2]),
        ]);
    }
    if l.attn == Attn::Indexed {
        let x = &cfg.indexer;
        w.extend([
            ("indexer.attn_k", vec![d, x.head_dim]),
            ("indexer.k_norm", vec![x.head_dim]),
            ("indexer_compressor_value", vec![x.head_dim, x.head_dim]),
            ("indexer_compressor_gate", vec![x.head_dim, x.head_dim]),
            ("indexer_compressor_ape", vec![x.head_dim, x.kpool]),
            ("indexer_compressor_norm", vec![x.head_dim]),
            (
                "indexer.attn_q_b",
                vec![m.q_lora_rank, x.head_dim * x.n_heads],
            ),
            ("indexer.proj", vec![d, x.n_heads]),
        ]);
    }
    if l.engram.is_some() {
        let e = &cfg.engram;
        w.extend([
            ("engram_kv", vec![e.n_cols() * e.head_dim, d * (hc + 1)]),
            ("engram_kv_b", vec![d, hc + 1]),
            ("engram_gate", vec![d, hc]),
            ("engram_lookup_norm", vec![e.head_dim, e.n_cols()]),
            ("engram_pre_gamma", vec![d]),
            ("engram_hc_scale", vec![d, hc]),
        ]);
    }
    let o = &cfg.moe;
    let shared = o.ff * o.n_shared;
    if l.dense_ffn {
        w.extend([
            ("ffn_gate", vec![d, cfg.dense_ff]),
            ("ffn_down", vec![cfg.dense_ff, d]),
            ("ffn_up", vec![d, cfg.dense_ff]),
        ]);
    } else {
        w.extend([
            ("ffn_gate_inp", vec![d, o.n_experts]),
            ("ffn_routed_down", vec![d, o.latent]),
            ("ffn_routed_up", vec![o.latent, d]),
            ("ffn_gate_shexp", vec![d, shared]),
            ("ffn_down_shexp", vec![shared, d]),
            ("ffn_up_shexp", vec![d, shared]),
        ]);
    }
    let mut out: Vec<Want> = w
        .into_iter()
        .map(|(n, dims)| want(format!("blk.{il}.{n}.weight"), &dims, role))
        .collect();
    if !l.dense_ffn {
        out.push(Want {
            required: false,
            ..want(format!("blk.{il}.{EXP_PROBS_B}"), &[o.n_experts], role)
        });
        for (p, dims) in expert_dims(cfg) {
            out.push(want(expert_name(il, p), &dims, Role::Expert));
        }
    }
    out
}

const PROJS: [&str; 3] = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];

fn expert_name(il: usize, proj: &str) -> String {
    format!("blk.{il}.{proj}.weight")
}

fn expert_dims(cfg: &HyV4FlashConfig) -> [(&'static str, [usize; 3]); 3] {
    let o = &cfg.moe;
    [
        (PROJS[0], [o.latent, o.ff, o.n_experts]),
        (PROJS[1], [o.latent, o.ff, o.n_experts]),
        (PROJS[2], [o.ff, o.latent, o.n_experts]),
    ]
}

fn classify(
    paths: Vec<PathBuf>,
    cfg: HyV4FlashConfig,
    mut found: HashMap<String, Located>,
) -> Result<Scan> {
    let mut wants = globals(&cfg);
    for il in 0..cfg.layers.len() {
        wants.extend(layer_wants(&cfg, il));
    }
    let mut names = HashSet::new();
    let mut tensors = HashMap::with_capacity(wants.len());
    for w in wants {
        if !names.insert(w.name.clone()) {
            return Err(bad(format!("tensor '{}' expected twice", w.name)));
        }
        let Some(loc) = found.remove(&w.name) else {
            if w.required {
                return Err(bad(format!("tensor '{}' missing", w.name)));
            }
            continue;
        };
        if loc.info.dims != w.dims {
            return Err(bad(format!(
                "tensor '{}' has ggml dims {:?}, expected {:?}",
                w.name, loc.info.dims, w.dims
            )));
        }
        tensors.insert(w.name, Entry { role: w.role, loc });
    }
    if let Some(name) = found.keys().min() {
        return Err(bad(format!(
            "tensor '{name}' is not part of the hy_v4_flash layout ({} unexpected)",
            found.len()
        )));
    }
    let engram_table = table(&tensors)?;
    let experts = (0..cfg.layers.len())
        .map(|il| expert_layer(&cfg, &tensors, il))
        .collect::<Result<Vec<_>>>()?;
    let mtp = mtp_layer(&cfg, &tensors)?;
    Ok(Scan {
        paths,
        cfg,
        tensors,
        experts,
        engram_table,
        mtp,
    })
}

fn mtp_layer(cfg: &HyV4FlashConfig, tensors: &HashMap<String, Entry>) -> Result<Option<MtpLayer>> {
    if cfg.mtp().is_empty() {
        return Ok(None);
    }
    let layer = cfg.n_layers;
    let prefix = format!("blk.{layer}.");
    let mut bytes = 0u64;
    for (name, e) in tensors {
        if name.starts_with(&prefix) {
            bytes = bytes
                .checked_add(e.loc.info.data_size()?)
                .ok_or_else(|| bad("MTP layer size overflows u64"))?;
        }
    }
    Ok(Some(MtpLayer { layer, bytes }))
}

/// Type 43 is parsed as MXFP8 (validated as F8_E4M3 by the row reader's
/// EOF extent proof); type 8 is Q8_0 (validated by `base + n_rows * row_bytes <= file_len`).
fn table(tensors: &HashMap<String, Entry>) -> Result<Located> {
    let loc = tensors
        .get(ENGRAM_TABLE)
        .map(|e| e.loc.clone())
        .ok_or_else(|| bad("Engram table missing"))?;
    if !matches!(
        loc.info.tensor_type,
        GgufTensorType::MXFP8 | GgufTensorType::Q8_0
    ) {
        return Err(bad(format!(
            "Engram table type {:?} is not GGUF type 43 (F8_E4M3) or Q8_0",
            loc.info.tensor_type
        )));
    }
    Ok(loc)
}

fn expert_layer(
    cfg: &HyV4FlashConfig,
    tensors: &HashMap<String, Entry>,
    il: usize,
) -> Result<Option<ExpertLayer>> {
    use GgufTensorType::{BF16, NVFP4, Q4K, Q6K, Q8_0};
    let l = &cfg.layers[il];
    if l.dense_ffn {
        return Ok(None);
    }
    let get = |p: &str| -> Result<Located> {
        let n = expert_name(il, p);
        tensors
            .get(&n)
            .map(|e| e.loc.clone())
            .ok_or_else(|| bad(format!("tensor '{n}' missing")))
    };
    let (gate, up, down) = (get(PROJS[0])?, get(PROJS[1])?, get(PROJS[2])?);
    let types = (
        gate.info.tensor_type,
        up.info.tensor_type,
        down.info.tensor_type,
    );
    let class = match (types, l.mtp) {
        ((Q4K, Q4K, Q6K), false) => ExpertClass::Q6kDown,
        ((Q4K, Q4K, Q4K), false) => ExpertClass::Q4kDown,
        ((NVFP4, NVFP4, NVFP4), false) => ExpertClass::Nvfp4,
        ((BF16, BF16, BF16), true) => ExpertClass::Bf16,
        ((Q8_0, Q8_0, Q8_0), true) => ExpertClass::Q8_0,
        (t, mtp) => {
            return Err(bad(format!(
                "layer {il}{}: expert types {t:?} (gate, up, down) are not supported",
                if mtp { " (MTP)" } else { "" }
            )));
        }
    };
    Ok(Some(ExpertLayer {
        class,
        gate,
        up,
        down,
    }))
}
