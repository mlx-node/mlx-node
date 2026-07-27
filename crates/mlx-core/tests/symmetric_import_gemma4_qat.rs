//! Acceptance gate for the symmetric affine import, on the checkpoint the
//! change exists for: Google's `gemma-4-12b-it-qat-q4_0.gguf`.
//!
//! 91.5% of that file is ggml Q4_0 — `w = d * (q - 8)`, one f16 scale per 32
//! weights and no stored offset. Importing it into MLX's `w = scale * q + bias`
//! used to materialise a `.biases` array whose every entry was `-8 * scale`,
//! which took the import from 6.50 GiB of source GGUF up to 7.12 GiB of output.
//! The array is no longer written; the loader rebuilds it from the scale.
//!
//! Dropping bytes is only allowed if the numbers do not move, so this compares a
//! conversion made BEFORE the change against one made AFTER it, on the real
//! bytes:
//!
//!   * every tensor the new file drops is a `.biases`, and nothing else changed
//!     name or appeared;
//!   * for all 328 affine tensors, `.weight` and `.scales` are byte-identical;
//!   * `f16(-8 * scales_new)` equals `biases_old` bitwise over every pair;
//!   * a sample of tensors decoded through `mlx_dequantize` with the stored bias
//!     and with the rebuilt one agree bit for bit — which also witnesses that
//!     the folded `q * scale + bias` form, and the signed-zero difference from
//!     ggml's `d * (q - 8)` that it produces, are properties of the affine
//!     decode itself and not of this change;
//!   * the Q6_K token embedding — weight, scales and the float16 super-block
//!     scale it keeps in `.biases` — is untouched;
//!   * the affine portion lands at exactly 4.5000 bpw, ggml's own Q4_0 figure.
//!
//! Fixtures are ~14 GiB in total and not checked in. Point `SYM_QAT_ROOT` at a
//! directory holding
//!
//!   $SYM_QAT_ROOT/gemma4-qat-mlx/   conversion from before the change
//!   $SYM_QAT_ROOT/gemma4-qat-sym/   conversion from after it
//!
//! These are `#[ignore]`d so an unattended run reports them as ignored rather
//! than passing on a comparison it never made — an acceptance gate that goes
//! green because its fixture is absent is worse than no gate.
//!
//! `SYM_QAT_ROOT` unset is the only skip; a half-present root fails.
//!
//! Run:
//!   SYM_QAT_ROOT=... cargo test -p mlx-core --release \
//!     --test symmetric_import_gemma4_qat -- --ignored --nocapture

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use mlx_core::array::MxArray;

/// ggml Q4_0 subtracts 8 from every code.
const ZERO_POINT: i32 = 8;
/// One f16 scale per 32 weights, and no second companion.
const EXPECTED_BPW: f64 = 4.5;

/// A safetensors file opened for raw byte reads: no MLX, no dtype
/// interpretation, just the header and the payload ranges it names.
struct RawSafetensors {
    path: PathBuf,
    /// name -> (dtype, shape, absolute start, absolute end)
    entries: BTreeMap<String, (String, Vec<usize>, u64, u64)>,
    header_len: u64,
    file_len: u64,
}

impl RawSafetensors {
    fn open(dir: &Path) -> Self {
        let path = dir.join("model.safetensors");
        let mut f = File::open(&path).expect("open model.safetensors");
        let mut len_bytes = [0u8; 8];
        f.read_exact(&mut len_bytes).expect("read header length");
        let header_len = u64::from_le_bytes(len_bytes);
        let mut header = vec![0u8; header_len as usize];
        f.read_exact(&mut header).expect("read header");
        let json: serde_json::Value = serde_json::from_slice(&header).expect("parse header");
        let base = 8 + header_len;

        let mut entries = BTreeMap::new();
        for (name, value) in json.as_object().expect("header object") {
            if name == "__metadata__" {
                continue;
            }
            let offsets = value["data_offsets"].as_array().expect("data_offsets");
            let start = offsets[0].as_u64().expect("start");
            let end = offsets[1].as_u64().expect("end");
            let shape = value["shape"]
                .as_array()
                .expect("shape")
                .iter()
                .map(|d| d.as_u64().expect("dim") as usize)
                .collect();
            entries.insert(
                name.clone(),
                (
                    value["dtype"].as_str().expect("dtype").to_string(),
                    shape,
                    base + start,
                    base + end,
                ),
            );
        }
        let file_len = f.metadata().expect("stat").len();
        Self {
            path,
            entries,
            header_len,
            file_len,
        }
    }

    fn names(&self) -> BTreeSet<&str> {
        self.entries.keys().map(String::as_str).collect()
    }

    fn byte_len(&self, name: &str) -> u64 {
        let (_, _, a, b) = &self.entries[name];
        b - a
    }

    fn read(&self, name: &str) -> Vec<u8> {
        let (_, _, a, b) = self.entries.get(name).unwrap_or_else(|| {
            panic!("{} has no tensor {name}", self.path.display());
        });
        let mut f = File::open(&self.path).expect("open for read");
        f.seek(SeekFrom::Start(*a)).expect("seek");
        let mut buf = vec![0u8; (b - a) as usize];
        f.read_exact(&mut buf).expect("read payload");
        buf
    }
}

/// The `.biases` entry the pre-symmetric converter wrote, recomputed from the
/// scale exactly as it did.
fn historical_bias_bits(scale_bits: u16) -> u16 {
    let scale = half::f16::from_bits(scale_bits).to_f32();
    half::f16::from_f32(-(ZERO_POINT as f32) * scale).to_bits()
}

fn u16_le(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// `config.json` names tensors with a `language_model.` wrapper the safetensors
/// keys drop.
fn resolve<'a>(raw: &RawSafetensors, config_key: &'a str) -> Option<&'a str> {
    for cand in [config_key, config_key.trim_start_matches("language_model.")] {
        if raw.entries.contains_key(&format!("{cand}.weight")) {
            return Some(cand);
        }
    }
    None
}

struct Fixtures {
    old: RawSafetensors,
    new: RawSafetensors,
    old_quant: serde_json::Value,
    new_quant: serde_json::Value,
}

/// `None` only when `SYM_QAT_ROOT` is unset. Once set, both conversions are
/// required — skipping a half-present root would report `ok` for a comparison
/// that never ran.
fn fixtures() -> Option<Fixtures> {
    let root = PathBuf::from(std::env::var("SYM_QAT_ROOT").ok()?);
    let old_dir = root.join("gemma4-qat-mlx");
    let new_dir = root.join("gemma4-qat-sym");
    let missing: Vec<String> = [&old_dir, &new_dir]
        .iter()
        .flat_map(|d| ["model.safetensors", "config.json"].map(|f| d.join(f)))
        .filter(|p| !p.is_file())
        .map(|p| p.display().to_string())
        .collect();
    assert!(
        missing.is_empty(),
        "SYM_QAT_ROOT is set but {} fixture file(s) are missing:\n  {}\n\
         Reconvert the QAT GGUF into both layouts (see the module docs) or \
         unset SYM_QAT_ROOT to skip.",
        missing.len(),
        missing.join("\n  ")
    );
    let quant = |d: &Path| -> serde_json::Value {
        let cfg: serde_json::Value =
            serde_json::from_reader(File::open(d.join("config.json")).expect("open config.json"))
                .expect("parse config.json");
        cfg["quantization"].clone()
    };
    Some(Fixtures {
        old_quant: quant(&old_dir),
        new_quant: quant(&new_dir),
        old: RawSafetensors::open(&old_dir),
        new: RawSafetensors::open(&new_dir),
    })
}

/// Every affine tensor the new config declares, as `(config key, tensor prefix)`.
fn affine_prefixes(f: &Fixtures) -> Vec<(String, String)> {
    f.new_quant
        .as_object()
        .expect("quantization object")
        .iter()
        .filter(|(_, v)| v.is_object() && v["mode"] == "affine")
        .map(|(k, _)| {
            let base = resolve(&f.new, k)
                .unwrap_or_else(|| panic!("config names {k} but no such tensor exists"));
            (k.clone(), base.to_string())
        })
        .collect()
}

#[test]
#[ignore = "needs SYM_QAT_ROOT pointing at both gemma4 QAT conversions"]
fn only_the_derived_bias_arrays_left_the_file() {
    let Some(f) = fixtures() else {
        eprintln!("SYM_QAT_ROOT unset — skipping gemma4 QAT acceptance");
        return;
    };
    // Guard against comparing two post-change conversions: the old fixture has
    // to be a genuinely pre-symmetric artifact, which is exactly what having no
    // marker anywhere means.
    assert!(
        f.old_quant["symmetric_zero_point"].is_null()
            && f.old_quant
                .as_object()
                .expect("quantization object")
                .values()
                .all(|v| !v.is_object() || v["symmetric_zero_point"].is_null()),
        "gemma4-qat-mlx already declares a zero point — it is not a pre-change conversion"
    );
    assert_eq!(f.new_quant["symmetric_zero_point"], i64::from(ZERO_POINT));

    let old_names = f.old.names();
    let new_names = f.new.names();

    let appeared: Vec<_> = new_names.difference(&old_names).collect();
    assert!(
        appeared.is_empty(),
        "the symmetric conversion invented tensors: {appeared:?}"
    );

    let dropped: Vec<&str> = old_names.difference(&new_names).copied().collect();
    let affine: BTreeSet<String> = affine_prefixes(&f)
        .into_iter()
        .map(|(_, base)| format!("{base}.biases"))
        .collect();
    assert_eq!(
        dropped
            .iter()
            .copied()
            .map(str::to_string)
            .collect::<BTreeSet<_>>(),
        affine,
        "the dropped set must be exactly the affine .biases companions"
    );
    assert!(
        !dropped.is_empty(),
        "nothing was dropped — did convert run?"
    );

    println!(
        "dropped {} tensors, all affine .biases; old {} tensors -> new {}",
        dropped.len(),
        old_names.len(),
        new_names.len()
    );
}

#[test]
#[ignore = "needs SYM_QAT_ROOT pointing at both gemma4 QAT conversions"]
fn weights_and_scales_are_byte_identical_and_the_bias_is_derivable() {
    let Some(f) = fixtures() else {
        eprintln!("SYM_QAT_ROOT unset — skipping gemma4 QAT acceptance");
        return;
    };
    let prefixes = affine_prefixes(&f);
    assert!(!prefixes.is_empty(), "no affine tensors to compare");

    let mut pairs = 0u64;
    let mut bias_mismatches = 0u64;
    let mut first: Option<String> = None;
    for (config_key, base) in &prefixes {
        assert_eq!(
            f.new_quant[config_key]["symmetric_zero_point"]
                .as_i64()
                .unwrap_or_else(|| panic!("{config_key} carries no zero point")),
            i64::from(ZERO_POINT)
        );
        assert!(
            !f.new.entries.contains_key(&format!("{base}.biases")),
            "{base}: the symmetric conversion still stores a .biases"
        );

        assert_eq!(
            f.old.read(&format!("{base}.weight")),
            f.new.read(&format!("{base}.weight")),
            "{base}.weight moved"
        );
        let old_scales = f.old.read(&format!("{base}.scales"));
        let new_scales = f.new.read(&format!("{base}.scales"));
        assert_eq!(old_scales, new_scales, "{base}.scales moved");

        let stored = u16_le(&f.old.read(&format!("{base}.biases")));
        let scales = u16_le(&new_scales);
        assert_eq!(
            stored.len(),
            scales.len(),
            "{base}: companion lengths differ"
        );
        for (i, (&s, &b)) in scales.iter().zip(&stored).enumerate() {
            if historical_bias_bits(s) != b {
                bias_mismatches += 1;
                first.get_or_insert_with(|| {
                    format!(
                        "{base}[{i}]: scale 0x{s:04x} derives 0x{:04x}, file stored 0x{b:04x}",
                        historical_bias_bits(s)
                    )
                });
            }
        }
        pairs += scales.len() as u64;
    }

    println!(
        "affine tensors {} | scale/bias pairs {pairs} | bias mismatches {bias_mismatches}",
        prefixes.len()
    );
    assert_eq!(bias_mismatches, 0, "first: {}", first.unwrap_or_default());
    assert!(pairs > 0, "compared nothing");
}

#[test]
#[ignore = "needs SYM_QAT_ROOT pointing at both gemma4 QAT conversions"]
fn the_decode_is_unchanged_with_the_rebuilt_bias() {
    let Some(f) = fixtures() else {
        eprintln!("SYM_QAT_ROOT unset — skipping gemma4 QAT acceptance");
        return;
    };
    let bits = f.new_quant["bits"].as_i64().expect("bits") as i32;
    let group_size = f.new_quant["group_size"].as_i64().expect("group_size") as i32;
    let mode = CString::new("affine").expect("mode cstring");

    // The whole file is 5.4 GiB of packed weights; a spread of tensors decoded
    // in full is enough to catch a decode that moved, and the bitwise companion
    // comparison above already covers every element of every tensor.
    let prefixes = affine_prefixes(&f);
    let step = (prefixes.len() / 12).max(1);
    let sample: Vec<_> = prefixes.iter().step_by(step).collect();
    assert!(sample.len() >= 8, "sample too small to be meaningful");

    let mut compared = 0u64;
    for (_, base) in &sample {
        let (_, w_shape, ..) = &f.new.entries[&format!("{base}.weight")];
        let (_, s_shape, ..) = &f.new.entries[&format!("{base}.scales")];
        let w_shape: Vec<i64> = w_shape.iter().map(|&d| d as i64).collect();
        let s_shape: Vec<i64> = s_shape.iter().map(|&d| d as i64).collect();

        let w_bytes = f.new.read(&format!("{base}.weight"));
        let words: Vec<u32> = w_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let weight = MxArray::from_uint32(&words, &w_shape).expect("weight array");
        let scales =
            MxArray::from_float16(&u16_le(&f.new.read(&format!("{base}.scales"))), &s_shape)
                .expect("scales array");
        let stored =
            MxArray::from_float16(&u16_le(&f.old.read(&format!("{base}.biases"))), &s_shape)
                .expect("stored bias array");
        let rebuilt = scales
            .mul_scalar(-f64::from(ZERO_POINT))
            .expect("rebuild bias");

        let decode = |b: &MxArray| {
            // SAFETY: the three handles outlive the call; out_dtype 0 is float32.
            let handle = unsafe {
                mlx_sys::mlx_dequantize(
                    weight.as_raw_ptr(),
                    scales.as_raw_ptr(),
                    b.as_raw_ptr(),
                    group_size,
                    bits,
                    0,
                    mode.as_ptr(),
                )
            };
            assert!(
                !handle.is_null(),
                "{base}: mlx_dequantize rejected the group"
            );
            let mut h = handle;
            // SAFETY: non-null handle from mlx_dequantize.
            let ok = unsafe { mlx_sys::mlx_eval(&mut h, 1) };
            assert!(ok, "{base}: eval failed");
            // SAFETY: non-null handle.
            let len = unsafe { mlx_sys::mlx_array_size(handle) };
            let mut out = vec![0f32; len];
            for (i, slot) in out.iter_mut().enumerate() {
                // SAFETY: i < len == array size.
                let ok = unsafe { mlx_sys::mlx_array_item_at_float32(handle, i, slot) };
                assert!(ok, "{base}: item_at_float32 failed at {i}");
            }
            // SAFETY: the handle is owned here and not used afterwards.
            unsafe { mlx_sys::mlx_array_delete(handle) };
            out
        };

        let a = decode(&stored);
        let b = decode(&rebuilt);
        assert_eq!(a.len(), b.len(), "{base}: element counts differ");
        for i in 0..a.len() {
            assert_eq!(
                a[i].to_bits(),
                b[i].to_bits(),
                "{base}[{i}]: stored-bias decode {:?} vs rebuilt-bias decode {:?}",
                a[i],
                b[i]
            );
        }
        compared += a.len() as u64;
    }
    println!(
        "decoded {} tensors, {compared} elements, all bitwise identical",
        sample.len()
    );
    assert!(compared > 0, "compared nothing");
}

#[test]
#[ignore = "needs SYM_QAT_ROOT pointing at both gemma4 QAT conversions"]
fn the_kquant_embedding_is_untouched() {
    let Some(f) = fixtures() else {
        eprintln!("SYM_QAT_ROOT unset — skipping gemma4 QAT acceptance");
        return;
    };
    let kquant: Vec<&String> = f
        .new_quant
        .as_object()
        .expect("quantization object")
        .iter()
        .filter(|(_, v)| v.is_object() && v["mode"] != "affine")
        .map(|(k, _)| k)
        .collect();
    assert_eq!(kquant.len(), 1, "expected one K-quant tensor: {kquant:?}");
    let base = resolve(&f.new, kquant[0]).expect("K-quant tensor missing");
    assert_eq!(f.new_quant[kquant[0]]["mode"], "q6k");
    assert!(
        f.new_quant[kquant[0]]["symmetric_zero_point"].is_null(),
        "a K-quant tensor must carry no zero point"
    );

    // `.biases` holds the ggml `d` super-block SCALE for a K-quant, not an
    // additive bias — the one array named `.biases` that must survive.
    for suffix in ["weight", "scales", "biases"] {
        let key = format!("{base}.{suffix}");
        assert_eq!(
            f.old.read(&key),
            f.new.read(&key),
            "{key} moved — the K-quant path must be untouched"
        );
    }
    println!("q6k {base}: weight, scales and super-block d all byte-identical");
}

#[test]
#[ignore = "needs SYM_QAT_ROOT pointing at both gemma4 QAT conversions"]
fn the_affine_portion_reaches_ggml_q4_0_density() {
    let Some(f) = fixtures() else {
        eprintln!("SYM_QAT_ROOT unset — skipping gemma4 QAT acceptance");
        return;
    };
    let prefixes = affine_prefixes(&f);
    let mut weight_bytes = 0u64;
    let mut scale_bytes = 0u64;
    let mut dropped_bias_bytes = 0u64;
    for (_, base) in &prefixes {
        weight_bytes += f.new.byte_len(&format!("{base}.weight"));
        scale_bytes += f.new.byte_len(&format!("{base}.scales"));
        dropped_bias_bytes += f.old.byte_len(&format!("{base}.biases"));
    }
    let elements = weight_bytes * 8 / 4; // 4-bit packing
    let bpw = (weight_bytes + scale_bytes) as f64 * 8.0 / elements as f64;
    let shrink = f.old.file_len - f.new.file_len;
    let header_shrink = f.old.header_len - f.new.header_len;

    println!(
        "old {} B (header {}) -> new {} B (header {})",
        f.old.file_len, f.old.header_len, f.new.file_len, f.new.header_len
    );
    println!(
        "shrink {shrink} B = {dropped_bias_bytes} payload + {header_shrink} header | \
         affine bpw {bpw:.4}"
    );

    assert_eq!(
        bpw, EXPECTED_BPW,
        "the affine portion must land on ggml's own Q4_0 density"
    );
    assert_eq!(
        shrink,
        dropped_bias_bytes + header_shrink,
        "the whole size change must be the dropped companions plus the header entries \
         naming them — anything else means a payload moved"
    );
}
