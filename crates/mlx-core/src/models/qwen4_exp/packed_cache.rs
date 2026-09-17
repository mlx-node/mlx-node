//! Optional, lossless SSD cache of already imported quantized matrix chunks.
//! Opening a Store only fingerprints source metadata. Entries are created on
//! demand, atomically, and contain at most one bounded row/expert chunk.
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::AsRawFd;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use napi::{Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::weights::{MAX_READ_BYTES, Weight};
use crate::array::{DType, MxArray};

const MAGIC: &[u8; 8] = b"Q4PACK02";
const MAX_HEADER: usize = 4096;
const DISK_BUDGET: u64 = 96 << 30;
const FREE_RESERVE: u64 = 16 << 30;

fn err(e: impl std::fmt::Display) -> Error {
    Error::from_reason(e.to_string())
}

#[derive(Serialize, Deserialize)]
struct ArrayInfo {
    dtype: String,
    shape: Vec<i64>,
    bytes: usize,
}
#[derive(Serialize, Deserialize)]
struct Header {
    key: String,
    group: i32,
    bits: i32,
    mode: String,
    // values, scales, optional biases
    arrays: Vec<ArrayInfo>,
}

/// Fully verified owned bytes; no file mapping or path-based trust survives a read.
pub(super) struct VerifiedChunk {
    header: Header,
    data: Vec<u8>,
    file_len: u64,
}

pub(super) struct PackedCache {
    root: PathBuf,
}

pub(super) fn chunk_key(
    path: &Path,
    offset: u64,
    bytes: u64,
    rows: usize,
    width: usize,
    ty: u32,
) -> String {
    let mut hash = Sha256::new();
    hash.update(path.as_os_str().as_bytes());
    for n in [offset, bytes, rows as u64, width as u64, ty as u64] {
        hash.update(n.to_le_bytes());
    }
    hash.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

impl PackedCache {
    pub fn new(source: &Path, files: BTreeSet<PathBuf>) -> Result<Self> {
        // The source checkpoint may live on a slow external volume. Keep the
        // reusable working set in the OS application cache by default.
        let root = std::env::var_os("MLX_QWEN4_PACKED_CACHE_DIR")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|home| {
                    PathBuf::from(home).join(if cfg!(target_os = "macos") {
                        "Library/Caches/mlx-node/qwen4-packed"
                    } else {
                        ".cache/mlx-node/qwen4-packed"
                    })
                })
            })
            .unwrap_or_else(|| source.parent().unwrap_or(Path::new(".")).to_path_buf());
        Self::new_at(&root, files)
    }

    pub(super) fn new_at(root: &Path, files: BTreeSet<PathBuf>) -> Result<Self> {
        let mut hash = Sha256::new();
        hash.update(MAGIC);
        for path in files {
            let canonical = path.canonicalize().map_err(err)?;
            let meta = fs::metadata(&canonical).map_err(err)?;
            hash.update(canonical.as_os_str().as_bytes());
            for n in [
                meta.dev(),
                meta.ino(),
                meta.len(),
                meta.mtime() as u64,
                meta.mtime_nsec() as u64,
                meta.ctime() as u64,
                meta.ctime_nsec() as u64,
            ] {
                hash.update(n.to_le_bytes());
            }
        }
        let identity: String = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
        let root = root.join(format!(".mlx-qwen4-packed-v2-{identity}"));
        Ok(Self { root })
    }

    fn path(&self, key: &str) -> PathBuf {
        self.root.join(format!("{key}.pack"))
    }

    pub fn read(&self, key: &str) -> Result<Option<(Weight, u64)>> {
        self.read_verified(key, MAX_READ_BYTES)?
            .map(VerifiedChunk::into_weight)
            .transpose()
    }

    /// IO and hashing run without MLX arrays or Store mutation. Bounds are
    /// validated before spawning workers and enforced again on each file.
    pub(super) fn read_batch(
        &self,
        requests: &[(String, u64)],
    ) -> Result<Vec<Option<VerifiedChunk>>> {
        let total = requests
            .iter()
            .try_fold(0u64, |n, (_, bytes)| n.checked_add(*bytes));
        if requests.len() > 24 || total.is_none_or(|n| n > MAX_READ_BYTES) {
            return Err(err("Packed batch exceeds staging budget"));
        }
        let workers = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(4)
            .min(requests.len());
        if workers <= 1 {
            return Ok(requests
                .iter()
                .map(|(key, bytes)| self.read_verified(key, *bytes).ok().flatten())
                .collect());
        }
        std::thread::scope(|scope| {
            let mut jobs = Vec::new();
            for worker in 0..workers {
                jobs.push(scope.spawn(move || {
                    (worker..requests.len())
                        .step_by(workers)
                        .map(|i| {
                            let (key, bytes) = &requests[i];
                            (i, self.read_verified(key, *bytes).ok().flatten())
                        })
                        .collect::<Vec<_>>()
                }));
            }
            let mut out: Vec<_> = (0..requests.len()).map(|_| None).collect();
            for job in jobs {
                let completed = job
                    .join()
                    .map_err(|_| err("Packed reader worker panicked"))?;
                for (i, chunk) in completed {
                    out[i] = chunk;
                }
            }
            Ok(out)
        })
    }

    fn read_verified(&self, key: &str, payload_limit: u64) -> Result<Option<VerifiedChunk>> {
        let mut file = match File::open(self.path(key)) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(err(e)),
        };
        let len = file.metadata().map_err(err)?.len();
        // All bounds and the checksum are checked before creating MLX arrays.
        if !(44..=44 + MAX_HEADER as u64 + MAX_READ_BYTES).contains(&len) {
            return Err(err("Packed chunk exceeds staging budget"));
        }
        let mut prefix = [0u8; 44];
        file.read_exact(&mut prefix).map_err(err)?;
        let header_len = u32::from_le_bytes(
            prefix[8..12]
                .try_into()
                .map_err(|_| err("Invalid packed chunk header length field"))?,
        ) as usize;
        if &prefix[..8] != MAGIC || header_len > MAX_HEADER || 44 + header_len as u64 > len {
            return Err(err("Invalid packed chunk header"));
        }
        let mut header_bytes = vec![0; header_len];
        file.read_exact(&mut header_bytes).map_err(err)?;
        let header: Header = serde_json::from_slice(&header_bytes).map_err(err)?;
        if header.key != key
            || !(2..=3).contains(&header.arrays.len())
            || !(2..=8).contains(&header.bits)
            || ![16, 32, 64, 128].contains(&header.group)
            || header.mode.len() > 16
        {
            return Err(err("Invalid packed chunk quantization metadata"));
        }
        let mut total = 0usize;
        for a in &header.arrays {
            let element_bytes = match a.dtype.as_str() {
                "U32" | "F32" => 4,
                "F16" | "BF16" => 2,
                "U8" | "I8" => 1,
                _ => return Err(err("Invalid packed chunk dtype")),
            };
            if a.shape.is_empty() || a.shape.len() > 4 || a.shape.iter().any(|&d| d <= 0) {
                return Err(err("Invalid packed chunk shape"));
            }
            let expected = a
                .shape
                .iter()
                .try_fold(element_bytes, |n: usize, &d| n.checked_mul(d as usize));
            if expected != Some(a.bytes) {
                return Err(err("Invalid packed chunk array length"));
            }
            total = total
                .checked_add(a.bytes)
                .ok_or_else(|| err("Packed chunk size overflow"))?;
        }
        if total as u64 > MAX_READ_BYTES.min(payload_limit)
            || total as u64 + header_len as u64 + 44 != len
        {
            return Err(err("Truncated or oversized packed chunk"));
        }
        let mut data = vec![0; total];
        file.read_exact(&mut data).map_err(err)?;
        let digest = {
            Sha256::new()
                .chain_update(&header_bytes)
                .chain_update(&data)
                .finalize()
        };
        if digest[..] != prefix[12..44] {
            return Err(err("Packed chunk checksum mismatch"));
        }
        Ok(Some(VerifiedChunk {
            header,
            data,
            file_len: len,
        }))
    }

    pub fn write(&self, key: &str, weight: &Weight) -> Result<()> {
        if weight.bytes()? > MAX_READ_BYTES || weight.scales.is_none() {
            return Ok(());
        }
        fs::create_dir_all(&self.root).map_err(err)?;
        // Serialize writers across processes. A busy writer never stalls inference.
        let mut ledger = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(self.root.join("budget"))
            .map_err(err)?;
        if unsafe { libc::flock(ledger.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Ok(());
        }
        // Closing ledger releases the lock on every return/error path.
        let mut n = [0u8; 8];
        let used = if ledger.metadata().map_err(err)?.len() == 8 {
            ledger.read_exact(&mut n).map_err(err)?;
            u64::from_le_bytes(n)
        } else {
            fs::read_dir(&self.root)
                .map_err(err)?
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|s| s == "pack"))
                .try_fold(0u64, |sum, e| {
                    sum.checked_add(e.metadata().map_err(err)?.len())
                        .ok_or_else(|| err("Packed cache budget overflow"))
                })?
        };
        let reserve = weight.bytes()? + MAX_HEADER as u64 + 44;
        if used.saturating_add(reserve) > DISK_BUDGET || !has_disk_space(&self.root, reserve) {
            return Ok(());
        }
        // Reserve first: an interrupted write can overcount, never overspend.
        ledger.seek(SeekFrom::Start(0)).map_err(err)?;
        ledger
            .write_all(&(used + reserve).to_le_bytes())
            .map_err(err)?;
        ledger.set_len(8).map_err(err)?;
        let mut header = Header {
            key: key.into(),
            group: weight.group,
            bits: weight.bits,
            mode: weight.mode.clone(),
            arrays: Vec::new(),
        };
        let mut data = Vec::with_capacity(weight.bytes()? as usize);
        for a in [
            Some(&weight.values),
            weight.scales.as_ref(),
            weight.biases.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            let (dtype, bytes) = encode_array(a)?;
            header.arrays.push(ArrayInfo {
                dtype: dtype.into(),
                shape: a.shape()?.to_vec(),
                bytes: bytes.len(),
            });
            data.extend_from_slice(&bytes);
        }
        let header = serde_json::to_vec(&header).map_err(err)?;
        if header.len() > MAX_HEADER || data.len() as u64 > MAX_READ_BYTES {
            return Err(err("Packed chunk exceeds staging budget"));
        }
        let digest = Sha256::new()
            .chain_update(&header)
            .chain_update(&data)
            .finalize();
        let temp = self.root.join(format!("{}.tmp", uuid::Uuid::new_v4()));
        let result = (|| {
            let mut f = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&temp)
                .map_err(err)?;
            f.write_all(MAGIC).map_err(err)?;
            f.write_all(&(header.len() as u32).to_le_bytes())
                .map_err(err)?;
            f.write_all(&digest).map_err(err)?;
            f.write_all(&header).map_err(err)?;
            f.write_all(&data).map_err(err)?;
            drop(f);
            fs::rename(&temp, self.path(key)).map_err(err)
        })();
        let _ = fs::remove_file(temp);
        result
    }
}

impl VerifiedChunk {
    pub(super) fn into_weight(self) -> Result<(Weight, u64)> {
        let Self {
            header,
            data,
            file_len,
        } = self;
        if !(2..=3).contains(&header.arrays.len()) {
            return Err(err(
                "Packed chunk requires values, scales and optional biases",
            ));
        }
        let mut arrays = Vec::with_capacity(3);
        let mut remaining = data.as_slice();
        for info in &header.arrays {
            let (bytes, tail) = remaining
                .split_at_checked(info.bytes)
                .ok_or_else(|| err("Truncated packed chunk array payload"))?;
            arrays.push(decode_array(info, bytes)?);
            remaining = tail;
        }
        if !remaining.is_empty() {
            return Err(err("Packed chunk has trailing array payload"));
        }
        let biases = if arrays.len() == 3 {
            arrays.pop()
        } else {
            None
        };
        let scales = arrays.pop();
        let values = arrays
            .pop()
            .ok_or_else(|| err("Packed chunk is missing its values array"))?;
        Ok((
            Weight {
                dense_bf16: None,
                values,
                scales,
                biases,
                group: header.group,
                bits: header.bits,
                mode: header.mode,
            },
            file_len,
        ))
    }
}

fn has_disk_space(path: &Path, needed: u64) -> bool {
    let Ok(path) = std::ffi::CString::new(path.as_os_str().as_bytes()) else {
        return false;
    };
    let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    if unsafe { libc::statvfs(path.as_ptr(), stat.as_mut_ptr()) } != 0 {
        return false;
    }
    let stat = unsafe { stat.assume_init() };
    (stat.f_bavail as u64).saturating_mul(stat.f_frsize) >= FREE_RESERVE + needed
}

fn encode_array(a: &MxArray) -> Result<(&'static str, Vec<u8>)> {
    Ok(match a.dtype()? {
        DType::Uint32 => (
            "U32",
            a.to_uint32()?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        ),
        DType::Float32 => (
            "F32",
            a.to_float32()?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        ),
        DType::Float16 | DType::BFloat16 => (
            if a.dtype()? == DType::Float16 {
                "F16"
            } else {
                "BF16"
            },
            a.to_uint16_native()?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        ),
        DType::Uint8 => ("U8", a.to_uint8()?),
        DType::Int8 => ("I8", a.to_int8()?.into_iter().map(|v| v as u8).collect()),
        _ => return Err(err("Unsupported packed cache dtype")),
    })
}
fn decode_array(a: &ArrayInfo, bytes: &[u8]) -> Result<MxArray> {
    match a.dtype.as_str() {
        "U32" => MxArray::from_uint32(
            &bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect::<Vec<_>>(),
            &a.shape,
        ),
        "F32" => MxArray::from_float32(
            &bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect::<Vec<_>>(),
            &a.shape,
        ),
        "F16" | "BF16" => {
            let data = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_le_bytes(*b))
                .collect::<Vec<_>>();
            if a.dtype == "F16" {
                MxArray::from_float16(&data, &a.shape)
            } else {
                MxArray::from_bfloat16(&data, &a.shape)
            }
        }
        "U8" => MxArray::from_uint8(bytes, &a.shape),
        "I8" => MxArray::from_int8(
            &bytes.iter().map(|&b| b as i8).collect::<Vec<_>>(),
            &a.shape,
        ),
        _ => Err(err("Unsupported packed cache dtype")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_verified_chunks_are_rejected_before_array_creation() {
        for (count, data) in [(0, Vec::new()), (1, Vec::new()), (2, vec![0; 3])] {
            let chunk = VerifiedChunk {
                header: Header {
                    key: "malformed".into(),
                    group: 32,
                    bits: 4,
                    mode: "affine".into(),
                    arrays: (0..count)
                        .map(|_| ArrayInfo {
                            dtype: "U32".into(),
                            shape: vec![1],
                            bytes: 4,
                        })
                        .collect(),
                },
                data,
                file_len: 0,
            };
            assert!(chunk.into_weight().is_err());
        }
    }

    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!("qwen4-packed-{}", uuid::Uuid::new_v4()));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn weight(mode: &str) -> Weight {
        let (bits, group, width, scales, bias) = match mode {
            "q6k" => (
                6,
                16,
                48,
                MxArray::from_int8(&[-7; 32], &[2, 16]).unwrap(),
                MxArray::from_float16(&[half::f16::from_f32(0.01).to_bits(); 2], &[2, 1]).unwrap(),
            ),
            "q4k" => (
                4,
                32,
                32,
                MxArray::from_uint8(&[3; 32], &[2, 16]).unwrap(),
                MxArray::from_float16(&[half::f16::from_f32(0.03).to_bits(); 4], &[2, 2]).unwrap(),
            ),
            _ => (
                5,
                32,
                40,
                MxArray::from_float16(&[half::f16::from_f32(0.25).to_bits(); 16], &[2, 8]).unwrap(),
                MxArray::from_float16(&[half::f16::from_f32(-2.0).to_bits(); 16], &[2, 8]).unwrap(),
            ),
        };
        Weight {
            dense_bf16: None,
            values: MxArray::from_uint32(&vec![0x11223344; 2 * width], &[2, width as i64]).unwrap(),
            scales: Some(scales),
            biases: Some(bias),
            bits,
            group,
            mode: mode.into(),
        }
    }

    #[test]
    fn packed_chunks_preserve_codes_signed_scales_and_quantized_outputs() {
        let temp = Temp::new();
        let cache = PackedCache {
            root: temp.0.join("cache"),
        };
        let x = MxArray::from_float32(
            &(0..256)
                .map(|i| (i as f32 * 0.07).sin())
                .collect::<Vec<_>>(),
            &[1, 256],
        )
        .unwrap();
        for mode in ["affine", "q4k", "q6k"] {
            let w = weight(mode);
            cache.write(mode, &w).unwrap();
            let (loaded, _) = cache.read(mode).unwrap().unwrap();
            for (a, b) in [
                (&w.values, &loaded.values),
                (w.scales.as_ref().unwrap(), loaded.scales.as_ref().unwrap()),
                (w.biases.as_ref().unwrap(), loaded.biases.as_ref().unwrap()),
            ] {
                assert_eq!(encode_array(a).unwrap(), encode_array(b).unwrap());
                assert_eq!(&*a.shape().unwrap(), &*b.shape().unwrap());
            }
            assert_eq!(
                w.linear(&x).unwrap().to_float32().unwrap().as_ref(),
                loaded.linear(&x).unwrap().to_float32().unwrap().as_ref()
            );
            let original = std::sync::Arc::new(w.clone());
            let originals = vec![original; 3];
            let stacked = Weight::stack(&originals).unwrap().unwrap();
            let actual = stacked
                .linear(
                    &x.reshape(&[1, 1, 256])
                        .unwrap()
                        .broadcast_to(&[3, 1, 256])
                        .unwrap(),
                )
                .unwrap()
                .to_float32()
                .unwrap();
            let expected = w.linear(&x).unwrap().to_float32().unwrap();
            for row in actual.chunks(2) {
                assert!(
                    row.iter()
                        .zip(expected.iter())
                        .all(|(a, b)| (a - b).abs() < 1e-4),
                    "{mode}: batched expert arithmetic changed"
                );
            }
        }
    }

    #[test]
    fn resident_gather_matches_selected_quantized_experts_for_prompt_and_decode() {
        use std::sync::Arc;
        for mode in ["affine", "q4k", "q6k"] {
            let experts: Vec<_> = (0..3)
                .map(|expert| {
                    let mut w = weight(mode);
                    let shape = w.values.shape().unwrap();
                    let codes = w
                        .values
                        .to_uint32()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .map(|(i, v)| v.wrapping_add((expert * 7919 + i * 101) as u32))
                        .collect::<Vec<_>>();
                    w.values = MxArray::from_uint32(&codes, &shape).unwrap();
                    Arc::new(w)
                })
                .collect();
            let bank = Weight::concatenate_rows(&experts).unwrap();
            let ids = [2u32, 0, 1, 1, 0, 2];
            let indices = MxArray::from_uint32(&ids, &[3, 2]).unwrap();
            for dtype in [DType::Float32, DType::BFloat16] {
                for fanout in [1, 2] {
                    let x = MxArray::from_float32(
                        &(0..3 * fanout * 256)
                            .map(|i| (i as f32 * 0.037).sin() * 0.1)
                            .collect::<Vec<_>>(),
                        &[3, fanout as i64, 1, 256],
                    )
                    .unwrap()
                    .astype(dtype)
                    .unwrap();
                    let got = bank
                        .gather_rows(&x, &indices, 3, false)
                        .unwrap()
                        .to_float32()
                        .unwrap();
                    for (j, &expert) in ids.iter().enumerate() {
                        let row = x
                            .reshape(&[(3 * fanout) as i64, 256])
                            .unwrap()
                            .slice_axis(
                                0,
                                ((j / 2) * fanout + if fanout == 1 { 0 } else { j % 2 }) as i64,
                                ((j / 2) * fanout + if fanout == 1 { 0 } else { j % 2 } + 1) as i64,
                            )
                            .unwrap();
                        let want = experts[expert as usize]
                            .linear(&row)
                            .unwrap()
                            .to_float32()
                            .unwrap();
                        for k in 0..2 {
                            let tolerance = if dtype == DType::Float32 { 1e-4 } else { 0.04 };
                            assert!(
                                (got[j * 2 + k] - want[k]).abs() < tolerance,
                                "mode={mode}, dtype={dtype:?}, fanout={fanout}, row={j}, got={}, want={}",
                                got[j * 2 + k],
                                want[k]
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn corrupt_or_oversized_cache_is_rejected_before_array_creation_and_can_be_repaired() {
        let temp = Temp::new();
        let cache = PackedCache {
            root: temp.0.join("cache"),
        };
        let w = weight("affine");
        cache.write("key", &w).unwrap();
        let path = cache.path("key");
        let original = fs::read(&path).unwrap();
        for bad in [
            original[..20].to_vec(),
            {
                let mut b = original.clone();
                *b.last_mut().unwrap() ^= 1;
                b
            },
            {
                let mut b = original.clone();
                b[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
                b
            },
        ] {
            fs::write(&path, bad).unwrap();
            assert!(cache.read("key").is_err());
        }
        let f = File::create(&path).unwrap();
        f.set_len(1 << 40).unwrap();
        assert!(cache.read("key").is_err());
        drop(f);
        cache.write("key", &w).unwrap();
        assert!(cache.read("key").unwrap().is_some());
        let mut ledger = File::create(cache.root.join("budget")).unwrap();
        ledger.write_all(&DISK_BUDGET.to_le_bytes()).unwrap();
        cache.write("full", &w).unwrap();
        assert!(cache.read("full").unwrap().is_none());
    }

    #[test]
    fn source_change_invalidates_prepared_chunks_without_reading_payloads() {
        let temp = Temp::new();
        let source = temp.0.join("source.gguf");
        fs::write(&source, b"first").unwrap();
        let a = PackedCache::new(&source, BTreeSet::from([source.clone()])).unwrap();
        assert!(!a.root.exists());
        fs::write(&source, b"replacement").unwrap();
        let b = PackedCache::new(&source, BTreeSet::from([source.clone()])).unwrap();
        assert_ne!(a.root, b.root);
    }

    #[test]
    fn parallel_reads_preserve_order_ownership_and_revalidate_changed_files() {
        let temp = Temp::new();
        let cache = PackedCache {
            root: temp.0.join("cache"),
        };
        for mode in ["affine", "q4k", "q6k"] {
            cache.write(mode, &weight(mode)).unwrap();
        }
        let requests: Vec<_> = ["q6k", "affine", "missing", "q4k", "q6k"]
            .into_iter()
            .map(|key| (key.to_string(), 4096))
            .collect();
        let chunks = cache.read_batch(&requests).unwrap();
        // Returned bytes own the verified generation even if the path changes
        // before main-thread import. No mmap or stale verification is reused.
        fs::write(cache.path("q6k"), b"truncated").unwrap();
        let mut corrupt = fs::read(cache.path("affine")).unwrap();
        *corrupt.last_mut().unwrap() ^= 1;
        fs::write(cache.path("affine"), corrupt).unwrap();
        for ((key, _), chunk) in requests.iter().zip(chunks) {
            if key == "missing" {
                assert!(chunk.is_none());
                continue;
            }
            let (got, _) = chunk.unwrap().into_weight().unwrap();
            let want = weight(key);
            assert_eq!(
                encode_array(&got.values).unwrap(),
                encode_array(&want.values).unwrap()
            );
            assert_eq!(
                encode_array(got.scales.as_ref().unwrap()).unwrap(),
                encode_array(want.scales.as_ref().unwrap()).unwrap()
            );
        }
        let changed = cache.read_batch(&requests).unwrap();
        assert!(
            changed[0].is_none()
                && changed[1].is_none()
                && changed[2].is_none()
                && changed[4].is_none()
        );
        assert!(changed[3].is_some());
        assert!(cache.read_batch(&[("q4k".into(), 1)]).unwrap()[0].is_none());
        assert!(
            cache
                .read_batch(&[("q4k".into(), MAX_READ_BYTES + 1)])
                .is_err()
        );
        assert!(cache.read_batch(&vec![("q4k".into(), 1); 25]).is_err());
    }
}
