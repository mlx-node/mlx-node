//! Real-data K-quant parity: the weights we import out of a ggml GGUF, decoded
//! by MLX, against llama.cpp's OWN decode of the same file.
//!
//! The committed gate (`kquant_ggml_parity.rs`) proves MLX's decode matches
//! ggml's on synthesised vectors that sweep the whole code / sub-scale space.
//! This gate proves the same thing on the actual bytes of an actual model, end
//! to end through the shipped artifact:
//!
//!   pure.gguf --(`mlx convert --gguf-kquant`)--> model.safetensors
//!             --(`mlx_dequantize`, default = Metal)--> f32        [ours]
//!   pure.gguf --(`llama-quantize ... F32`)--> deq-f32.gguf        [llama.cpp]
//!
//! The reference side is produced by the llama.cpp binary itself: `llama-quantize
//! --allow-requantize <pure>.gguf <out>.gguf F32` walks every tensor and calls
//! `ggml_get_type_traits(t)->to_float`, i.e. `dequantize_row_q{4,5,6}_K`. No
//! transliteration, no vendored copy, no golden blob — the reference is a file
//! written by the same code path llama.cpp uses to run the model on CPU.
//!
//! Comparison is on raw IEEE-754 bit patterns, element for element, over every
//! quantized tensor in the file. Q4_K and Q5_K must be identical with no
//! exemption at all. Q6_K carries the one documented exemption from the
//! committed gate: ggml's `(d*sc) * (code - 32)` yields `-0.0` where the folded
//! `scale*code + bias` form yields `+0.0`. Those pairs are counted, and both
//! sides are asserted to be zero, so a real divergence cannot hide inside the
//! exemption.
//!
//! The Q8_0 case is the control — the ordinary affine import path, which this
//! branch does not touch. It does NOT come out bit-identical, and that is a
//! property of MLX rather than of the import: MLX's affine decode evaluates
//! `scale * code + bias` in the dtype of `scales` (float16 for a GGUF import)
//! while ggml evaluates `d * q` in float32. The control therefore asserts the
//! exact shape of that gap — every element equals llama.cpp's f32 value rounded
//! to float16 — which still pins down the scale, the code and the layout, and
//! makes it visible that the K-quant decode is the one running in full f32.
//!
//! Fixtures are not checked in (the GGUFs are ~0.5 GB each). Point
//! `KQ_PARITY_ROOT` at a directory laid out as
//!
//!   $KQ_PARITY_ROOT/gguf/qwen3-0.6b-pure-Q4_K_S.gguf      (source)
//!   $KQ_PARITY_ROOT/deq/q4k-deq-f32.gguf                  (llama-quantize F32)
//!   $KQ_PARITY_ROOT/qwen3-q4k-mlx2/                       (mlx convert output)
//!
//! and the test runs; leave it unset and it skips.
//!
//! Run:
//!   KQ_PARITY_ROOT=... cargo test -p mlx-core --release \
//!     --test kquant_realdata_llamacpp_parity -- --nocapture

use std::ffi::CString;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use mlx_core::utils::gguf::{gguf_name_to_hf, parse_gguf};
use mlx_core::utils::safetensors::SafeTensorsFile;

/// How closely MLX's decode is required to track llama.cpp's.
#[derive(PartialEq, Eq, Clone, Copy)]
enum Expect {
    /// Every element identical, bit for bit. No exemption whatsoever.
    BitExact,
    /// Bit-exact except where BOTH sides are zero and only the sign differs —
    /// the one documented Q6_K reassociation (see module docs).
    BitExactModuloSignedZero,
    /// MLX's *affine* decode evaluates `scale * code + bias` in the dtype of
    /// `scales` (float16 here), then widens, while ggml evaluates `d * q` in
    /// float32. So the two agree only to float16 precision. This is upstream
    /// MLX behaviour on a code path this branch does not touch, and it is
    /// asserted here rather than tolerated: every single element must equal
    /// llama.cpp's f32 value rounded to float16, which pins the divergence to
    /// round-off and rules out a wrong scale, a wrong code, or a wrong layout.
    F16RoundedDecode,
}

/// One (source GGUF, llama.cpp F32 dequantization, our conversion) triple.
struct Case {
    /// Label used in assertion messages.
    label: &'static str,
    /// The pure quantized GGUF `llama-quantize --pure` produced.
    source_gguf: &'static str,
    /// `llama-quantize --allow-requantize <source> <this> F32`.
    reference_gguf: &'static str,
    /// `mlx convert [--gguf-kquant] -i <source> -o <this>`.
    mlx_dir: &'static str,
    expect: Expect,
}

const CASES: &[Case] = &[
    Case {
        label: "q4k",
        source_gguf: "gguf/qwen3-0.6b-pure-Q4_K_S.gguf",
        reference_gguf: "deq/q4k-deq-f32.gguf",
        mlx_dir: "qwen3-q4k-mlx2",
        expect: Expect::BitExact,
    },
    Case {
        label: "q5k",
        source_gguf: "gguf/qwen3-0.6b-pure-Q5_K_S.gguf",
        reference_gguf: "deq/q5_k_s-deq-f32.gguf",
        mlx_dir: "qwen3-q5k-mlx",
        expect: Expect::BitExact,
    },
    Case {
        label: "q6k",
        source_gguf: "gguf/qwen3-0.6b-pure-Q6_K.gguf",
        reference_gguf: "deq/q6_k-deq-f32.gguf",
        mlx_dir: "qwen3-q6k-mlx",
        expect: Expect::BitExactModuloSignedZero,
    },
    Case {
        label: "Q8_0-affine-control",
        source_gguf: "gguf/qwen3-0.6b-pure-Q8_0.gguf",
        reference_gguf: "deq/q8_0-deq-f32.gguf",
        mlx_dir: "qwen3-q8-mlx2",
        expect: Expect::F16RoundedDecode,
    },
];

fn root() -> Option<PathBuf> {
    let raw = std::env::var("KQ_PARITY_ROOT").ok()?;
    let p = PathBuf::from(raw);
    p.is_dir().then_some(p)
}

/// Read a whole F32 tensor out of a GGUF at its recorded data offset.
fn read_f32_tensor(path: &Path, data_offset: u64, tensor_offset: u64, n: usize) -> Vec<f32> {
    let mut f = File::open(path).expect("open reference gguf");
    f.seek(SeekFrom::Start(data_offset + tensor_offset))
        .expect("seek to tensor");
    let mut bytes = vec![0u8; n * 4];
    f.read_exact(&mut bytes).expect("read tensor payload");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Read MLX's decode back element-by-element.
///
/// `mlx_array_item_at_float32` reads `arr.data<float>()[i]` with no arithmetic.
/// The bulk `mlx_array_to_float32` path cannot be used here: it materialises via
/// `add(arr, zeros)`, and `-0.0 + 0.0 == +0.0` would launder exactly the
/// signed-zero distinction this gate measures.
fn read_f32_exact(handle: *mut mlx_sys::mlx_array, len: usize) -> Vec<f32> {
    let mut h = handle;
    // SAFETY: `handle` is a live array produced by mlx_dequantize.
    let ok = unsafe { mlx_sys::mlx_eval(&mut h, 1) };
    assert!(ok, "eval of the MLX dequantize output failed");
    let mut out = vec![0f32; len];
    for (i, slot) in out.iter_mut().enumerate() {
        // SAFETY: i < len == array size, checked by the caller.
        let ok = unsafe { mlx_sys::mlx_array_item_at_float32(handle, i, slot) };
        assert!(ok, "mlx_array_item_at_float32 failed at index {i}");
    }
    out
}

struct Totals {
    tensors: usize,
    elements: u64,
    /// Differences the case's `Expect` does NOT account for. Must be zero.
    hard_diffs: u64,
    /// Both sides zero, signs differ (Q6_K's documented reassociation).
    zero_sign_diffs: u64,
    /// Ours equals llama.cpp's value rounded to float16 (affine decode).
    f16_round_diffs: u64,
    first_hard: Option<String>,
}

/// llama.cpp's f32 value as MLX's float16 affine decode would land on it.
fn round_to_f16(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

fn run_case(root: &Path, case: &Case) -> Totals {
    let source = root.join(case.source_gguf);
    let reference = root.join(case.reference_gguf);
    let mlx_dir = root.join(case.mlx_dir);
    assert!(source.is_file(), "missing source gguf {}", source.display());
    assert!(
        reference.is_file(),
        "missing reference gguf {}",
        reference.display()
    );

    // Which tensors were actually quantized in the source file.
    let src = parse_gguf(&source).expect("parse source gguf");
    let quantized: std::collections::BTreeMap<String, String> = src
        .tensors
        .iter()
        .filter(|t| {
            !matches!(
                format!("{:?}", t.tensor_type).as_str(),
                "F32" | "F16" | "BF16"
            )
        })
        .map(|t| (t.name.clone(), format!("{:?}", t.tensor_type)))
        .collect();
    assert!(
        !quantized.is_empty(),
        "{}: source gguf has no quantized tensors — --pure did not take",
        case.label
    );

    // llama.cpp's own dequantization of the same file.
    let refs = parse_gguf(&reference).expect("parse reference gguf");
    let ref_by_name: std::collections::HashMap<&str, &mlx_core::utils::gguf::GgufTensorInfo> =
        refs.tensors.iter().map(|t| (t.name.as_str(), t)).collect();

    // Our conversion: quantization parameters + the packed arrays.
    let config: serde_json::Value =
        serde_json::from_reader(File::open(mlx_dir.join("config.json")).expect("open config.json"))
            .expect("parse config.json");
    let qblock = &config["quantization"];
    let bits = qblock["bits"].as_i64().expect("bits") as i32;
    let group_size = qblock["group_size"].as_i64().expect("group_size") as i32;
    let mode = qblock["mode"].as_str().expect("mode").to_string();
    let mode_c = CString::new(mode.clone()).expect("mode cstring");

    let st_path = mlx_dir.join("model.safetensors");
    let st = SafeTensorsFile::load(&st_path).expect("load safetensors header");
    let tensors = st.load_tensors(&st_path).expect("load safetensors tensors");

    let mut totals = Totals {
        tensors: 0,
        elements: 0,
        hard_diffs: 0,
        zero_sign_diffs: 0,
        f16_round_diffs: 0,
        first_hard: None,
    };

    for (gguf_name, ggml_type) in &quantized {
        let hf = gguf_name_to_hf(gguf_name);
        let base = hf
            .strip_suffix(".weight")
            .unwrap_or_else(|| panic!("{}: unexpected mapped name {hf}", case.label));

        let w = tensors
            .get(&format!("{base}.weight"))
            .unwrap_or_else(|| panic!("{}: no {base}.weight in our checkpoint", case.label));
        let s = tensors
            .get(&format!("{base}.scales"))
            .unwrap_or_else(|| panic!("{}: no {base}.scales in our checkpoint", case.label));
        let b = tensors
            .get(&format!("{base}.biases"))
            .unwrap_or_else(|| panic!("{}: no {base}.biases in our checkpoint", case.label));

        // SAFETY: the three handles outlive the call; out_dtype 0 is float32.
        let handle = unsafe {
            mlx_sys::mlx_dequantize(
                w.as_raw_ptr(),
                s.as_raw_ptr(),
                b.as_raw_ptr(),
                group_size,
                bits,
                0,
                mode_c.as_ptr(),
            )
        };
        assert!(
            !handle.is_null(),
            "{}: mlx_dequantize rejected {base} (mode {mode})",
            case.label
        );
        // SAFETY: non-null handle from mlx_dequantize.
        let len = unsafe { mlx_sys::mlx_array_size(handle) };

        let info = ref_by_name
            .get(gguf_name.as_str())
            .unwrap_or_else(|| panic!("{}: {gguf_name} missing from reference", case.label));
        assert_eq!(
            format!("{:?}", info.tensor_type),
            "F32",
            "{}: reference tensor {gguf_name} was not dequantized (still {:?}) — \
             llama-quantize skipped it",
            case.label,
            info.tensor_type
        );
        let n = info.num_elements() as usize;
        assert_eq!(
            len, n,
            "{}: {gguf_name} element count differs (ours {len}, llama.cpp {n})",
            case.label
        );

        let ours = read_f32_exact(handle, len);
        // SAFETY: the handle is owned here and not referenced afterwards.
        unsafe { mlx_sys::mlx_array_delete(handle) };
        let theirs = read_f32_tensor(&reference, refs.data_offset, info.offset, n);

        for i in 0..n {
            let a = ours[i];
            let c = theirs[i];
            if a.to_bits() == c.to_bits() {
                continue;
            }
            if case.expect == Expect::BitExactModuloSignedZero && a == 0.0 && c == 0.0 {
                totals.zero_sign_diffs += 1;
                continue;
            }
            if case.expect == Expect::F16RoundedDecode && a.to_bits() == round_to_f16(c).to_bits() {
                totals.f16_round_diffs += 1;
                continue;
            }
            totals.hard_diffs += 1;
            if totals.first_hard.is_none() {
                totals.first_hard = Some(format!(
                    "{gguf_name} ({ggml_type}) idx {i}: ours {a:?} (0x{:08x}) vs llama.cpp {c:?} \
                     (0x{:08x})",
                    a.to_bits(),
                    c.to_bits()
                ));
            }
        }

        totals.tensors += 1;
        totals.elements += n as u64;
    }

    assert_eq!(
        totals.tensors,
        quantized.len(),
        "{}: not every quantized tensor was compared",
        case.label
    );
    totals
}

#[test]
fn imported_kquant_weights_match_llamacpp_dequantization() {
    let Some(root) = root() else {
        eprintln!("KQ_PARITY_ROOT unset or not a directory — skipping real-data parity");
        return;
    };

    let mut failures = Vec::new();
    for case in CASES {
        let t = run_case(&root, case);
        let identical = t.elements - t.hard_diffs - t.zero_sign_diffs - t.f16_round_diffs;
        println!(
            "[{}] tensors={} elements={} bit_identical={} hard_diffs={} zero_sign_diffs={} \
             f16_round_diffs={}",
            case.label,
            t.tensors,
            t.elements,
            identical,
            t.hard_diffs,
            t.zero_sign_diffs,
            t.f16_round_diffs
        );
        if let Some(first) = &t.first_hard {
            println!("[{}] first unexplained divergence: {first}", case.label);
        }
        if t.hard_diffs != 0 {
            failures.push(format!(
                "{}: {} of {} elements differ from llama.cpp beyond what this case allows",
                case.label, t.hard_diffs, t.elements
            ));
        }
        if case.expect == Expect::BitExact && t.elements != identical {
            failures.push(format!(
                "{}: expected strict bit-identity over all {} elements, got {}",
                case.label, t.elements, identical
            ));
        }
        assert!(t.elements > 0, "{}: compared nothing", case.label);
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
