//! Test-only readers for the tiny fixture: dense GGUF tensors as f32 and
//! numpy `.npz` dumps.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use crate::array::MxArray;
use crate::utils::gguf::{GgufFile, GgufTensorType, parse_gguf};

pub(super) fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/deepseek-v41")
        .join(name)
}

/// `tiny-q8_0.gguf` dense tensors (F32, Q8_0) dequantized to f32 on the
/// host (`d * q` is exact in f32), torch order.
pub(super) struct TinyGguf {
    path: PathBuf,
    pub gguf: GgufFile,
}

impl TinyGguf {
    pub fn open() -> Self {
        let path = fixture("tiny-q8_0.gguf");
        let gguf = parse_gguf(&path).unwrap();
        Self { path, gguf }
    }

    pub fn dense(&self, name: &str) -> napi::Result<MxArray> {
        let t = self
            .gguf
            .tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("{name} not in the tiny GGUF"));
        let mut f = std::fs::File::open(&self.path).unwrap();
        f.seek(SeekFrom::Start(self.gguf.data_offset + t.offset))
            .unwrap();
        let mut raw = vec![0u8; t.data_size()? as usize];
        f.read_exact(&mut raw).unwrap();
        let vals: Vec<f32> = match t.tensor_type {
            GgufTensorType::F32 => raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect(),
            GgufTensorType::Q8_0 => raw
                .as_chunks::<34>()
                .0
                .iter()
                .flat_map(|b| {
                    let d = half::f16::from_le_bytes([b[0], b[1]]).to_f32();
                    b[2..].iter().map(move |&q| d * f32::from(q as i8))
                })
                .collect(),
            other => panic!("{name}: {} is not a dense test type", other.name()),
        };
        MxArray::from_float32(&vals, &t.mlx_shape())
    }
}

/// An `.npz` written by `np.savez`: `<f4` and `<i4` arrays, C order.
pub(super) struct Npz(pub HashMap<String, (Vec<i64>, Vec<f32>)>);

impl Npz {
    pub fn open(name: &str) -> Self {
        let f = std::fs::File::open(fixture(name)).unwrap();
        let mut z = zip::ZipArchive::new(f).unwrap();
        let mut out = HashMap::new();
        for i in 0..z.len() {
            let mut e = z.by_index(i).unwrap();
            let key = e.name().unwrap().trim_end_matches(".npy").to_string();
            let mut b = Vec::new();
            e.read_to_end(&mut b).unwrap();
            assert_eq!(&b[..6], b"\x93NUMPY", "{key}");
            let (hlen, h0) = match b[6] {
                1 => (u16::from_le_bytes([b[8], b[9]]) as usize, 10),
                _ => (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12),
            };
            let header = std::str::from_utf8(&b[h0..h0 + hlen]).unwrap();
            assert!(header.contains("'fortran_order': False"), "{key}");
            let shape_txt = header
                .split("'shape': (")
                .nth(1)
                .and_then(|s| s.split(')').next())
                .unwrap();
            let shape: Vec<i64> = shape_txt
                .split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.trim().parse().unwrap())
                .collect();
            let data = &b[h0 + hlen..];
            let words = data.as_chunks::<4>().0.iter().copied();
            let vals: Vec<f32> = if header.contains("'descr': '<f4'") {
                words.map(f32::from_le_bytes).collect()
            } else if header.contains("'descr': '<i4'") {
                words.map(|w| i32::from_le_bytes(w) as f32).collect()
            } else {
                panic!("{key}: unsupported dtype in {header}")
            };
            assert_eq!(vals.len() as i64, shape.iter().product::<i64>(), "{key}");
            out.insert(key, (shape, vals));
        }
        Self(out)
    }

    pub fn get(&self, key: &str) -> &(Vec<i64>, Vec<f32>) {
        self.0
            .get(key)
            .unwrap_or_else(|| panic!("{key} not in the npz"))
    }

    /// Rows `lo..hi` of a 2-D array as an MLX array.
    pub fn rows(&self, key: &str, lo: usize, hi: usize) -> MxArray {
        let (shape, v) = self.get(key);
        let w = shape[1] as usize;
        MxArray::from_float32(&v[lo * w..hi * w], &[(hi - lo) as i64, w as i64]).unwrap()
    }
}

/// `||a - b|| / ||b||` in f64.
pub(super) fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut num, mut den) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        num += (f64::from(x) - f64::from(y)).powi(2);
        den += f64::from(y).powi(2);
    }
    (num / den.max(1e-300)).sqrt()
}

pub(super) fn read(x: &MxArray) -> Vec<f32> {
    x.astype(crate::array::DType::Float32)
        .unwrap()
        .to_float32()
        .unwrap()
        .as_ref()
        .to_vec()
}
