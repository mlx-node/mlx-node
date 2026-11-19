//! The planes writer on the real Qwen3.8-27B UD-Q4_K_XL GGUF (ignored: they
//! read the 27B file): every imported array hashed after one import with the
//! writer off and with it on, and the planes the same import writes from the
//! same staged bands.
use std::path::Path;

use napi::{Error, Result};
use sha2::{Digest, Sha256};

use crate::array::{DType, MxArray};
use crate::engine::checkpoint_source::{native_import_plans, open_gguf_main};
use crate::utils::gguf::NativeGgufFamily;
use crate::utils::gguf_import::{ImportProbe, RawTensors, ReadConfig, RepackBackend};
use crate::utils::gguf_plan::ImportPlan;
use crate::utils::gguf_planes::{
    PlanesDevice, PlanesTensor, plan_planes, prepare_planes, repack_band,
};
use crate::utils::gguf_read::PageCacheMode;
use crate::utils::gguf_source::GgufSourceSet;

/// The Qwen3.8-27B UD-Q4_K_XL GGUF that `MLX_TEST_QWEN38_27B_GGUF` names.
pub(crate) fn real_gguf() -> String {
    std::env::var("MLX_TEST_QWEN38_27B_GGUF").unwrap_or_default()
}

pub(crate) fn real_gguf_present() -> bool {
    Path::new(&real_gguf()).is_file()
}

/// The bytes of an evaluated array, in memory order.
pub(crate) fn array_bytes(a: &MxArray) -> Result<Vec<u8>> {
    a.eval();
    Ok(match a.dtype()? {
        DType::Uint32 => a
            .to_uint32()?
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        DType::Uint8 => a.to_uint8()?,
        DType::Int8 => a.to_int8()?.into_iter().map(|v| v as u8).collect(),
        DType::Float16 | DType::BFloat16 => a
            .to_uint16_native()?
            .into_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        DType::Float32 => a
            .to_float32()?
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect(),
        DType::Int32 => a.to_int32()?.iter().flat_map(|v| v.to_le_bytes()).collect(),
    })
}

pub(crate) fn sha_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// One import of the real GGUF as the Qwen3.5 load plans it.
pub(crate) fn import_real(probe: &ImportProbe) -> Result<(RawTensors, ImportPlan)> {
    let main = open_gguf_main(&[real_gguf().into()])?;
    let (path, header, identity) = &main.parts[0];
    let (_, plan, _) = native_import_plans(NativeGgufFamily::Qwen35, (path, &main.header), None)?;
    let set = GgufSourceSet::open(&[(path, header.clone(), identity)], None)?;
    let handle = &set.main[0];
    let raw = crate::utils::gguf_import::read_gguf_tensors_with(
        &handle.file,
        handle,
        &plan,
        RepackBackend::Auto,
        PageCacheMode::Auto,
        &ReadConfig::default(),
        probe,
    )?;
    Ok((raw, plan))
}

/// `T64HASH <tensor> <sha256 of weight | scales | biases>` for every tiled
/// tensor, in name order.
pub(crate) fn tiled64_hashes(raw: &RawTensors, plan: &ImportPlan) -> Result<Vec<String>> {
    let mut lines = Vec::new();
    for name in &plan.tiled_gguf_names {
        let prefix = name.strip_suffix(".weight").unwrap_or(name);
        let mut bytes = Vec::new();
        for key in [
            name.clone(),
            format!("{prefix}.scales"),
            format!("{prefix}.biases"),
        ] {
            let a = raw
                .map
                .get(&key)
                .ok_or_else(|| Error::from_reason(format!("import has no {key}")))?;
            bytes.extend(array_bytes(a)?);
        }
        lines.push(format!("T64HASH {name} {}", sha_hex(&bytes)));
    }
    Ok(lines)
}

/// `ALLHASH <key> <dtype> <shape> <sha256>` for every imported array but the
/// planes, in key order.
pub(crate) fn all_hashes(raw: &RawTensors) -> Result<Vec<String>> {
    let mut keys: Vec<&String> = raw.map.keys().filter(|k| !k.ends_with(".planes")).collect();
    keys.sort();
    let mut lines = Vec::with_capacity(keys.len());
    for key in keys {
        let a = &raw.map[key];
        let shape = a.shape()?;
        let bytes = array_bytes(a)?;
        lines.push(format!(
            "ALLHASH {key} {:?} {:?} {}",
            a.dtype()?,
            &shape[..],
            sha_hex(&bytes)
        ));
    }
    Ok(lines)
}

/// The Tiled64 bytes of the real load hashed per tensor, and every imported
/// array, with the planes writer off (weights only, no inference).
#[test]
#[ignore = "reads the real 27B GGUF"]
fn real_gguf_tiled64_hashes() -> Result<()> {
    if !real_gguf_present() {
        println!("SKIP: MLX_TEST_QWEN38_27B_GGUF names no file");
        return Ok(());
    }
    let started = std::time::Instant::now();
    let (raw, plan) = import_real(&ImportProbe::default())?;
    println!(
        "IMPORT tensors={} tiled={} secs={:.1}",
        raw.map.len(),
        plan.tiled_gguf_names.len(),
        started.elapsed().as_secs_f64()
    );
    let lines = tiled64_hashes(&raw, &plan)?;
    for line in &lines {
        println!("{line}");
    }
    println!("T64COUNT {}", lines.len());
    let all = all_hashes(&raw)?;
    for line in &all {
        println!("{line}");
    }
    println!("ALLCOUNT {}", all.len());
    Ok(())
}

unsafe extern "C" {
    fn dequantize_row_q4_K(x: *const u8, y: *mut f32, k: i64);
    fn dequantize_row_q5_K(x: *const u8, y: *mut f32, k: i64);
    fn dequantize_row_q6_K(x: *const u8, y: *mut f32, k: i64);
    fn dequantize_row_q3_K(x: *const u8, y: *mut f32, k: i64);
    fn dequantize_row_iq4_xs(x: *const u8, y: *mut f32, k: i64);
    fn dequantize_row_iq4_nl(x: *const u8, y: *mut f32, k: i64);
    fn dequantize_row_iq3_s(x: *const u8, y: *mut f32, k: i64);
}

/// ggml's `dequantize_row_*` of one row of `k` values.
fn ggml_row(fmt: u32, src: &[u8], dst: &mut [f32]) {
    use crate::utils::gguf_planes::{
        FMT_IQ3S, FMT_IQ4NL, FMT_IQ4XS, FMT_Q3K, FMT_Q4K, FMT_Q5K, FMT_Q6K,
    };
    let k = dst.len() as i64;
    let (x, y) = (src.as_ptr(), dst.as_mut_ptr());
    // SAFETY: `src` holds one row of `k` values of `fmt`, `dst` `k` floats.
    unsafe {
        match fmt {
            FMT_Q4K => dequantize_row_q4_K(x, y, k),
            FMT_Q5K => dequantize_row_q5_K(x, y, k),
            FMT_Q6K => dequantize_row_q6_K(x, y, k),
            FMT_Q3K => dequantize_row_q3_K(x, y, k),
            FMT_IQ4XS => dequantize_row_iq4_xs(x, y, k),
            FMT_IQ4NL => dequantize_row_iq4_nl(x, y, k),
            FMT_IQ3S => dequantize_row_iq3_s(x, y, k),
            other => panic!("no ggml reference for format {other}"),
        }
    }
}

/// One fp16 ULP at |v|.
fn half_ulp(v: f32) -> f32 {
    let e = ((v.abs().to_bits() >> 23) & 0xff) as i32 - 127;
    2f32.powi(e.max(-14) - 10)
}

/// The worst distance in fp16 ULPs and the count above 0.5 of the decoded
/// halves `dec` against ggml's decode of the native rows `native`.
fn against_ggml(fmt: u32, native: &[u8], dec: &[u16], rows: usize, k: usize) -> (f32, u64) {
    let row_bytes = native.len() / rows;
    let threads = 12usize;
    let per = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        let jobs: Vec<_> = (0..threads)
            .map(|j| {
                scope.spawn(move || {
                    let mut reference = vec![0f32; k];
                    let (mut worst, mut over) = (0f32, 0u64);
                    for r in (j * per)..((j + 1) * per).min(rows) {
                        ggml_row(
                            fmt,
                            &native[r * row_bytes..(r + 1) * row_bytes],
                            &mut reference,
                        );
                        for (c, &want) in reference.iter().enumerate() {
                            let got = half::f16::from_bits(dec[r * k + c]).to_f32();
                            let e = (got - want).abs() / half_ulp(want);
                            if e.is_nan() || e > worst {
                                worst = if e.is_nan() { f32::INFINITY } else { e };
                            }
                            if e.is_nan() || e > 0.5 {
                                over += 1;
                            }
                        }
                    }
                    (worst, over)
                })
            })
            .collect();
        jobs.into_iter()
            .map(|h| h.join().expect("ggml compare thread"))
            .fold((0f32, 0u64), |(w, o), (w2, o2)| (w.max(w2), o + o2))
    })
}

/// The native bytes of every source of `t`, rows in planes order.
fn native_rows(
    file: &std::fs::File,
    header: &crate::utils::gguf::GgufFile,
    t: &PlanesTensor,
) -> Result<Vec<u8>> {
    use crate::utils::gguf_read::ReadAt;
    let mut out = Vec::new();
    for (name, _) in &t.sources {
        let info = header
            .tensors
            .iter()
            .find(|i| &i.name == name)
            .ok_or_else(|| Error::from_reason(format!("no {name}")))?;
        let len = info.data_size()? as usize;
        let at = out.len();
        out.resize(at + len, 0);
        file.read_exact_at(&mut out[at..], header.data_offset + info.offset)
            .map_err(|e| Error::from_reason(format!("{name}: {e}")))?;
    }
    Ok(out)
}

/// The real load with the planes writer on, one import (Tiled64 and planes
/// from the same staged bands): the plan line and identity; the Tiled64 and
/// every-array hashes; every planned planes array written band by band
/// equals the planes of the whole tensor repacked at once; and the planes'
/// GPU decode, in the fast-math and in the epilogue build, of every weight
/// of every planned tensor is within 1 fp16 ULP of ggml's dequantize_row_*.
#[test]
#[ignore = "reads the real 27B GGUF"]
fn real_gguf_planes() -> Result<()> {
    if !real_gguf_present() {
        println!("SKIP: MLX_TEST_QWEN38_27B_GGUF names no file");
        return Ok(());
    }
    let device = PlanesDevice::this(None);
    let main = open_gguf_main(&[real_gguf().into()])?;
    let plan = prepare_planes(plan_planes(&main.header, &device));
    println!("DEVICE {device:?}");
    println!("PLANDESC {}", plan.describe());
    println!("PLANID {}", plan.identity());
    for (what, why) in &plan.left_out {
        println!("LEFTOUT {what}: {why}");
    }
    assert!(!plan.tensors.is_empty(), "the app machine plans planes");
    let started = std::time::Instant::now();
    let probe = ImportProbe {
        planes: Some(plan.clone()),
        ..ImportProbe::default()
    };
    let (raw, import_plan) = import_real(&probe)?;
    println!(
        "IMPORT tensors={} tiled={} secs={:.1} max_alloc={}",
        raw.map.len(),
        import_plan.tiled_gguf_names.len(),
        started.elapsed().as_secs_f64(),
        probe
            .max_alloc_bytes
            .load(std::sync::atomic::Ordering::Relaxed)
    );
    let lines = tiled64_hashes(&raw, &import_plan)?;
    for line in &lines {
        println!("{line}");
    }
    println!("T64COUNT {}", lines.len());
    let all = all_hashes(&raw)?;
    for line in &all {
        println!("{line}");
    }
    println!("ALLCOUNT {}", all.len());

    mlx_sys::gguf_planes_use_test_library();
    let file = std::fs::File::open(real_gguf()).map_err(|e| Error::from_reason(e.to_string()))?;
    let (mut equal, mut within, mut weights) = (0usize, 0usize, 0u64);
    let mut failures = Vec::new();
    let mut by_format: std::collections::BTreeMap<(u32, u32), (f32, u64, u64)> = Default::default();
    for t in &plan.tensors {
        let planes = raw
            .map
            .get(&t.key)
            .ok_or_else(|| Error::from_reason(format!("import has no {}", t.key)))?;
        let banded = array_bytes(planes)?;
        let native = native_rows(&file, &main.header, t)?;
        let staging = MxArray::from_uint8(&native, &[native.len() as i64])?;
        let fresh = MxArray::zeros(&[t.bytes as i64], Some(DType::Uint8))?;
        let whole = repack_band(&staging, 0, t.format, t.k, t.rows, 0, t.rows, &fresh)?;
        let whole = array_bytes(&whole)?;
        let same = banded == whole;
        equal += usize::from(same);
        let mut ok = true;
        for build in [0u32, 1] {
            // SAFETY: a live planes array of `t`'s format and shape.
            let h = unsafe {
                mlx_sys::mlx_test_gguf_planes_dequant(
                    planes.as_raw_ptr(),
                    t.format,
                    t.rows,
                    t.k,
                    build,
                )
            };
            let dec = MxArray::from_handle(h, "planes dequant")?;
            dec.eval();
            let dec = dec.to_uint16_native()?;
            let (worst, over) =
                against_ggml(t.format, &native, &dec, t.rows as usize, t.k as usize);
            let e = by_format.entry((build, t.format)).or_insert((0.0, 0, 0));
            e.0 = e.0.max(worst);
            e.1 += over;
            e.2 += dec.len() as u64;
            if build == 0 {
                weights += dec.len() as u64;
            }
            println!(
                "ROW planes {} build={build} fmt={} rows={} k={} bands_eq_whole={} sha={} max_half_ulps={worst} over_half={over}",
                t.key,
                t.format,
                t.rows,
                t.k,
                u8::from(same),
                &sha_hex(&banded)[..16]
            );
            if !same || worst > 1.0 {
                ok = false;
                failures.push(format!(
                    "{} build {build}: bands_eq_whole {same}, {worst} ULPs",
                    t.key
                ));
            }
        }
        within += usize::from(ok);
    }
    for ((build, fmt), (worst, over, n)) in &by_format {
        println!(
            "ROW decode_vs_ggml build={build} fmt={fmt} weights={n} over_half_ulp={over} max_half_ulps={worst}"
        );
    }
    println!(
        "ROW planes_total tensors={} bands_eq_whole={equal} within_1_ulp={within} weights={weights} bytes={}",
        plan.tensors.len(),
        plan.bytes()
    );
    assert!(failures.is_empty(), "planes on the real load: {failures:?}");
    Ok(())
}

/// The band fault: one Tiled64 band written 64 rows down must change the
/// Tiled64 hashes.
#[test]
#[ignore = "reads the real 27B GGUF"]
fn real_gguf_band_shift() -> Result<()> {
    if !real_gguf_present() {
        println!("SKIP: MLX_TEST_QWEN38_27B_GGUF names no file");
        return Ok(());
    }
    let probe = ImportProbe::default();
    probe
        .band_shift
        .store(true, std::sync::atomic::Ordering::Relaxed);
    let (raw, plan) = import_real(&probe)?;
    println!(
        "SHIFT fired={}",
        !probe.band_shift.load(std::sync::atomic::Ordering::Relaxed)
    );
    for line in tiled64_hashes(&raw, &plan)? {
        println!("{line}");
    }
    Ok(())
}
