//! Golden output digests for bridge kernels whose bits were proven equal to a
//! reference. A fixture under `tests/golden/` maps each case key to the first
//! 8 bytes of SHA-256 over the output's dtype, shape and raw element bytes
//! (or to a plain value). Metal and CPU digests are hardware specific: a
//! fixture records the machine it was captured on, and checking it anywhere
//! else fails. There is no capture mode: the fork-oracle tests wrote the
//! fixtures on MLX pin 053e43fec, and a new fixture needs a new proof.

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

/// First 8 bytes of SHA-256 over dtype, rank, dims and the element bytes
/// (little-endian, at the dtype's width), as 16 hex digits.
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
    h.finalize()[..8]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

pub struct Golden {
    path: PathBuf,
    expected: HashMap<String, String>,
    values: HashMap<String, String>,
    mismatches: usize,
}

impl Golden {
    /// Metal outputs: one fixture per routing architecture
    /// (`<stem>.<arch>.txt`), valid only on the captured hardware.
    pub fn metal(stem: &str) -> Golden {
        let route = metal_architecture(false).expect("golden Metal fixtures need a Metal device");
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
        let mut expected = HashMap::new();
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let (value, key) = line
                .split_once(' ')
                .unwrap_or_else(|| panic!("{}: bad line {line:?}", path.display()));
            if let Some(field) = key.strip_prefix("= ") {
                fixture_header.insert(field.to_string(), value.to_string());
            } else {
                expected.insert(key.to_string(), value.to_string());
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
        let cases: usize = fixture_header
            .get("cases")
            .and_then(|c| c.parse().ok())
            .unwrap_or_else(|| panic!("{}: no `cases` header", path.display()));
        assert_eq!(cases, expected.len(), "{}: case count", path.display());
        Golden {
            path,
            expected,
            values: HashMap::new(),
            mismatches: 0,
        }
    }

    pub fn record_bits(&mut self, key: &str, shape: &[i64], dtype: DType, bits: &[u32]) {
        self.record_value(key, &digest(shape, dtype, bits));
    }

    pub fn record_value(&mut self, key: &str, value: &str) {
        if let Some(previous) = self.values.get(key) {
            assert_eq!(previous, value, "{key}: the same case gave different bits");
            return;
        }
        match self.expected.get(key) {
            None => panic!("{key}: no such case in {}", self.path.display()),
            Some(want) if want != value => {
                self.mismatches += 1;
                if self.mismatches <= 20 {
                    eprintln!("golden mismatch: {key}: fixture {want}, now {value}");
                }
            }
            Some(_) => {}
        }
        self.values.insert(key.to_string(), value.to_string());
    }

    /// Every fixture case must have run with its recorded value.
    pub fn finish(self) {
        let missing: Vec<&String> = self
            .expected
            .keys()
            .filter(|k| !self.values.contains_key(*k))
            .collect();
        assert!(
            missing.is_empty(),
            "{}: {} fixture cases never ran, e.g. {:?}",
            self.path.display(),
            missing.len(),
            &missing[..missing.len().min(5)]
        );
        assert_eq!(
            self.mismatches,
            0,
            "{}: {} of {} cases differ from the golden digests (first ones above)",
            self.path.display(),
            self.mismatches,
            self.values.len()
        );
        eprintln!(
            "{}: {} cases equal the golden digests",
            self.path.display(),
            self.values.len()
        );
    }
}

fn hardware() -> String {
    metal_architecture(true).unwrap_or_else(|| "no-metal".into())
}
