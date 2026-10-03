//! Golden output digests for bridge kernels whose bits were proven equal to a
//! reference. A fixture under `tests/golden/` holds one line per case, in the
//! order the gate runs its cases: the first 6 bytes of SHA-256 over the
//! output's dtype, shape and raw element bytes (base64url), or a plain value.
//! The header pins the case count and a hash of the ordered case keys, so a
//! mismatch is named by the key the test itself built. Metal and CPU digests
//! are hardware specific: a fixture records the machine it was captured on,
//! and checking it anywhere else fails. There is no capture mode: the
//! fork-oracle tests wrote the fixtures on MLX pin 053e43fec, and a new
//! fixture needs a new proof.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;

use mlx_core::array::DType;
use sha2::{Digest, Sha256};

/// Env vars that change kernel routing or reduction order.
const ROUTING_ENV: [&str; 3] = ["MLX_SDPA_BLOCKS", "MLX_QMM_SPLITK_MIN_M", "MLX_ENABLE_TF32"];

pub fn metal_architecture(hardware: bool) -> Option<String> {
    let mut buf = [0 as std::ffi::c_char; 128];
    // SAFETY: `buf` is the NUL-terminated output sink of `buf.len()` bytes.
    let ok = unsafe { mlx_sys::mlx_test_metal_architecture(hardware, buf.as_mut_ptr(), buf.len()) };
    ok.then(|| {
        // SAFETY: on success the FFI wrote a NUL-terminated string into `buf`.
        unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    })
}

fn element_bytes(dtype: DType) -> usize {
    match dtype {
        DType::Float32 => 4,
        DType::Float16 | DType::BFloat16 => 2,
        other => panic!("no golden digest for dtype {other:?}"),
    }
}

fn base64url(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    assert_eq!(bytes.len() % 3, 0);
    bytes
        .chunks(3)
        .flat_map(|c| {
            let n = (u32::from(c[0]) << 16) | (u32::from(c[1]) << 8) | u32::from(c[2]);
            [18, 12, 6, 0].map(|shift| A[((n >> shift) & 63) as usize] as char)
        })
        .collect()
}

/// First 6 bytes of SHA-256 over dtype, rank, dims and the element bytes
/// (little-endian, at the dtype's width), as 8 base64url characters.
pub fn digest(shape: &[i64], dtype: DType, bits: &[u32]) -> String {
    let width = element_bytes(dtype);
    let mut h = Sha256::new();
    h.update(format!("{dtype:?}").as_bytes());
    h.update((shape.len() as u64).to_le_bytes());
    for d in shape {
        h.update(d.to_le_bytes());
    }
    for b in bits {
        h.update(&b.to_le_bytes()[..width]);
    }
    base64url(&h.finalize()[..6])
}

/// First 8 bytes of SHA-256 over the case keys in order, one per line.
pub fn keys_hash<'a>(keys: impl IntoIterator<Item = &'a String>) -> String {
    let mut h = Sha256::new();
    for key in keys {
        h.update(key.as_bytes());
        h.update(b"\n");
    }
    h.finalize()[..8]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

pub struct Golden {
    path: PathBuf,
    expected: Vec<String>,
    expected_keys: String,
    cases: Vec<(String, String)>,
    index: HashMap<String, usize>,
}

impl Golden {
    /// Metal outputs: one fixture per routing architecture
    /// (`<stem>.<arch>.txt`), valid only on the captured hardware.
    pub fn metal(stem: &str) -> Golden {
        let route = metal_architecture(false).expect("golden Metal fixtures need a Metal device");
        // SAFETY: nullary predicate.
        let nax = unsafe { mlx_sys::mlx_metal_is_nax_available() };
        Golden::open(
            &format!("{stem}.{route}"),
            vec![
                ("machine", hardware()),
                ("route", route),
                ("nax", nax.to_string()),
            ],
        )
    }

    /// CPU outputs: valid only on the captured machine.
    pub fn cpu(stem: &str) -> Golden {
        Golden::open(stem, vec![("machine", hardware())])
    }

    /// Pure functions: valid on any machine.
    pub fn pure(stem: &str) -> Golden {
        Golden::open(stem, vec![("machine", "any".into())])
    }

    fn open(name: &str, header: Vec<(&'static str, String)>) -> Golden {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/golden")
            .join(format!("{name}.txt"));
        let pure = header.iter().any(|(f, v)| *f == "machine" && v == "any");
        for var in ROUTING_ENV.iter().filter(|_| !pure) {
            assert!(
                std::env::var_os(var).is_none(),
                "{var} is set: golden digests assume MLX's default routing"
            );
        }
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "no golden fixture {} ({e}); the captured routes are the `{}.*.txt` files \
                 in tests/golden",
                path.display(),
                name.split('.').next().unwrap_or(name)
            )
        });
        let mut fixture_header = HashMap::new();
        let mut expected = Vec::new();
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            match line.split_once(" = ") {
                Some((value, field)) => {
                    fixture_header.insert(field.to_string(), value.to_string());
                }
                None => expected.push(line.to_string()),
            }
        }
        for (field, now) in &header {
            let captured = fixture_header
                .get(*field)
                .unwrap_or_else(|| panic!("{}: no `{field}` header", path.display()));
            if *field == "machine" {
                assert_eq!(
                    captured,
                    now,
                    "{}: fixture captured on {captured}, this machine is {now}; the digests \
                     are specific to the capture hardware, run this gate on a {captured} machine",
                    path.display()
                );
            } else {
                assert_eq!(
                    captured,
                    now,
                    "{}: fixture captured with {field} {captured}, this run has {now}",
                    path.display()
                );
            }
        }
        let field = |f: &str| {
            fixture_header
                .get(f)
                .cloned()
                .unwrap_or_else(|| panic!("{}: no `{f}` header", path.display()))
        };
        assert_eq!(
            field("cases").parse::<usize>().ok(),
            Some(expected.len()),
            "{}: case count",
            path.display()
        );
        let expected_keys = field("keys");
        Golden {
            path,
            expected,
            expected_keys,
            cases: Vec::new(),
            index: HashMap::new(),
        }
    }

    pub fn record_bits(&mut self, key: &str, shape: &[i64], dtype: DType, bits: &[u32]) {
        self.record_value(key, &digest(shape, dtype, bits));
    }

    /// Cases are matched to fixture lines by order; a repeated key must
    /// repeat its value.
    pub fn record_value(&mut self, key: &str, value: &str) {
        if let Some(&i) = self.index.get(key) {
            assert_eq!(
                self.cases[i].1, value,
                "{key}: the same case gave different bits"
            );
            return;
        }
        if let Some(want) = self.expected.get(self.cases.len()).filter(|w| *w != value) {
            eprintln!("golden mismatch: {key}: fixture {want}, now {value}");
        }
        self.index.insert(key.to_string(), self.cases.len());
        self.cases.push((key.to_string(), value.to_string()));
    }

    /// The case list must be the captured one, then every value must equal
    /// its fixture line.
    pub fn finish(self) {
        let path = self.path.display();
        assert_eq!(
            self.cases.len(),
            self.expected.len(),
            "{path}: the gate ran {} cases, the fixture has {}; the case matrix changed",
            self.cases.len(),
            self.expected.len()
        );
        assert_eq!(
            keys_hash(self.cases.iter().map(|(k, _)| k)),
            self.expected_keys,
            "{path}: the case keys or their order changed since the capture"
        );
        let mismatches: Vec<_> = self
            .cases
            .iter()
            .zip(&self.expected)
            .filter(|((_, now), want)| now != *want)
            .collect();
        assert!(
            mismatches.is_empty(),
            "{path}: {} of {} cases differ from the golden digests (listed above)",
            mismatches.len(),
            self.cases.len()
        );
        eprintln!(
            "{path}: {} cases equal the golden digests",
            self.cases.len()
        );
    }
}

fn hardware() -> String {
    metal_architecture(true).unwrap_or_else(|| "no-metal".into())
}
