use std::path::{Path, PathBuf};
use std::sync::Arc;

use napi::Result;
use serde_json::Value;

use super::f8::{decode, decode_f8_rows, is_nan_code};
use super::rows::{EngramRows, F8Table, ResidentF8Table, prove_f8_extent};
use super::scan::{ENGRAM_TABLE, Located, Scan};
use super::store::unchanged_since_open;
use crate::models::deepseek_v41::EngramConfig;
use crate::moe_stream::io::ShardSet;
use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};

const HD: usize = 64;
/// Rows of each (layer, column): 2 Engram layers × 4 columns, 300 in all.
const SIZES: [u64; 8] = [40, 35, 30, 45, 50, 25, 40, 35];
/// Row 0 at 16300, so row 1 straddles the first 16 KiB page boundary.
const DATA_OFFSET: u64 = 16000;
const REL_OFFSET: u64 = 300;

/// OCP E4M3 fn written out: sign bit 7, exponent bits 6..3 (bias 7),
/// mantissa bits 2..0; exponent 0 is m·2⁻⁹; 0x7F / 0xFF are NaN.
fn formula(c: u8) -> Option<f32> {
    let (e, m) = (i32::from((c >> 3) & 0xF), f64::from(c & 7));
    if e == 15 && m == 7.0 {
        return None;
    }
    let v = if e == 0 {
        m * 2f64.powi(-9)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    };
    Some(if c & 0x80 != 0 { -v as f32 } else { v as f32 })
}

fn err<T>(r: Result<T>) -> String {
    match r {
        Ok(_) => panic!("accepted what it must refuse"),
        Err(e) => e.reason,
    }
}

fn fixture(name: &str) -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/hy-v4-flash")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap()
}

fn hex_bits(v: &Value) -> u32 {
    u32::from_str_radix(v.as_str().unwrap().trim_start_matches("0x"), 16).unwrap()
}

#[test]
fn lut_matches_the_ggml_formula() {
    let lut = fixture("f8-rows.json")["lut"].as_array().unwrap().clone();
    assert_eq!(lut.len(), 256);
    for c in 0..=255u8 {
        let entry = &lut[usize::from(c)];
        assert_eq!(entry["code"].as_u64(), Some(u64::from(c)));
        match formula(c) {
            None => {
                assert!(decode(c).is_nan(), "code {c:#04x}");
                assert!(is_nan_code(c));
                assert_eq!(entry["nan"].as_bool(), Some(true));
            }
            Some(v) => {
                assert_eq!(decode(c).to_bits(), v.to_bits(), "code {c:#04x}");
                assert!(!is_nan_code(c));
                assert_eq!(
                    decode(c).to_bits(),
                    hex_bits(&entry["bits"]),
                    "code {c:#04x}"
                );
            }
        }
    }
    assert_eq!(decode(0x7E), 448.0);
    assert_eq!(decode(0xFE), -448.0);
    assert_eq!(decode(0x01), 2f32.powi(-9));
    assert_eq!(decode(0x08), 2f32.powi(-6));
    assert_eq!(decode(0x80).to_bits(), (-0.0f32).to_bits());
    assert!(decode(0x7F).is_nan() && decode(0xFF).is_nan());
}

#[test]
fn decode_refuses_nan_codes_and_length_mismatch() {
    let mut out = [0f32; 8];
    decode_f8_rows(&[0x00, 0x01, 0x7E, 0xFE, 0x80, 0x38, 0x08, 0x07], &mut out).unwrap();
    assert_eq!(out[2], 448.0);
    assert_eq!(out[5], 1.0);
    let msg = err(decode_f8_rows(&[0, 0, 0, 0, 0, 0xFF, 0, 0], &mut out));
    assert!(msg.contains("value 5"), "{msg}");
    let msg = err(decode_f8_rows(&[0, 0x7F, 0, 0, 0, 0, 0, 0], &mut out));
    assert!(msg.contains("value 1"), "{msg}");
    err(decode_f8_rows(&[0; 7], &mut out));
}

#[test]
fn size_proof_accepts_f8_only() {
    let p = &fixture("f8-rows.json")["size_proof"];
    let n = p["n_rows"].as_u64().unwrap();
    let hd = p["head_dim"].as_u64().unwrap() as usize;
    let f8 = p["f8_bytes"].as_u64().unwrap();
    let mxfp8 = p["mxfp8_bytes"].as_u64().unwrap();
    assert_eq!((n, hd, f8), (660_006_416, 96, 63_360_615_936));
    assert_eq!(p["extent_to_eof"].as_u64(), Some(f8));
    prove_f8_extent(f8, n, hd).unwrap();
    let msg = err(prove_f8_extent(mxfp8, n, hd));
    assert!(msg.contains("MXFP8"), "{msg}");
    err(prove_f8_extent(f8 + 1, n, hd));
    err(prove_f8_extent(f8 - 1, n, hd));
    err(prove_f8_extent(0, 0, hd));
    err(prove_f8_extent(0, n, 0));
    prove_f8_extent(10 * 40, 10, 40).unwrap();
    err(prove_f8_extent(10 * 40 + 1, 10, 40));
}

fn cfg() -> EngramConfig {
    let mut offsets = Vec::new();
    let mut at = 0;
    for s in SIZES {
        offsets.push(at);
        at += s;
    }
    EngramConfig {
        layers: vec![1, 3],
        max_ngram: 3,
        n_heads: 2,
        head_dim: HD,
        multipliers: vec![1, 3, 5, 7, 9, 11],
        head_offsets: offsets,
        head_vocab_sizes: SIZES.to_vec(),
        pad_token_id: 0,
        token_map: (0..16).collect(),
        compressed_vocab_size: 16,
        vocab_size: None,
    }
}

fn n_rows() -> u64 {
    SIZES.iter().sum()
}

/// Random non-NaN codes for every row.
fn table_codes(seed: u64) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n_rows() as usize * HD)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let c = (x >> 24) as u8;
            if is_nan_code(c) { c ^ 1 } else { c }
        })
        .collect()
}

fn located(dims: Vec<u64>, ty: GgufTensorType, name: &str) -> Located {
    Located {
        shard: 0,
        data_offset: DATA_OFFSET,
        info: GgufTensorInfo {
            name: name.into(),
            n_dims: dims.len() as u32,
            dims,
            tensor_type: ty,
            offset: REL_OFFSET,
        },
    }
}

fn table_loc() -> Located {
    located(
        vec![HD as u64, n_rows()],
        GgufTensorType::MXFP8,
        ENGRAM_TABLE,
    )
}

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("hy4-rows-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    /// A shard whose data section holds junk, then `table` to the end.
    fn shard(&self, table: &[u8]) -> PathBuf {
        let path = self.0.join("table.gguf");
        let mut bytes = vec![0xA5u8; (DATA_OFFSET + REL_OFFSET) as usize];
        bytes.extend_from_slice(table);
        std::fs::write(&path, bytes).unwrap();
        path
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn expect_rows(codes: &[u8], ids: &[u32]) -> Vec<u32> {
    ids.iter()
        .flat_map(|&id| {
            let at = id as usize * HD;
            codes[at..at + HD]
                .iter()
                .map(|&c| formula(c).unwrap().to_bits())
        })
        .collect()
}

fn to_bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Two positions × 4 columns of layer `e`, with a repeated row.
fn layer_ids(e: usize) -> Vec<u32> {
    let c = cfg();
    let col = |k: usize, j: u64| (c.head_offsets[4 * e + k] + j) as u32;
    let last = |k: usize| col(k, c.head_vocab_sizes[4 * e + k] - 1);
    vec![
        col(0, 0),
        col(1, 3),
        col(2, 7),
        last(3),
        col(0, 0),
        last(1),
        col(2, 1),
        col(3, 0),
    ]
}

#[test]
fn file_table_reads_each_layer_and_column() {
    let tmp = Temp::new();
    let codes = table_codes(7);
    let path = tmp.shard(&codes);
    let t = F8Table::open_at(&path, &table_loc(), &cfg()).unwrap();
    assert_eq!(t.n_rows(), n_rows());
    for e in 0..2 {
        let ids = layer_ids(e);
        assert_eq!(
            to_bits(&t.rows(e, &ids).unwrap()),
            expect_rows(&codes, &ids)
        );
    }
    let all: Vec<u32> = (0..n_rows() as u32).rev().collect();
    assert_eq!(
        to_bits(&t.read_rows(&all).unwrap()),
        expect_rows(&codes, &all)
    );
    let (bytes, _) = t.read_stats();
    assert!(bytes > 0);

    let mut ids = layer_ids(0);
    ids[1] = ids[0];
    let msg = err(t.rows(0, &ids));
    assert!(msg.contains("column 1"), "{msg}");
    err(t.rows(0, &layer_ids(1)));
    err(t.rows(0, &layer_ids(0)[..6]));
    let msg = err(t.rows(2, &layer_ids(0)));
    assert!(msg.contains("no column"), "{msg}");
    err(t.read_rows(&[n_rows() as u32]));
}

#[test]
fn file_table_refuses_a_bad_size_or_header() {
    let tmp = Temp::new();
    let codes = table_codes(11);
    let open = |table: &[u8], loc: &Located, cfg: &EngramConfig| {
        F8Table::open_at(&tmp.shard(table), loc, cfg).map(|_| ())
    };
    open(&codes, &table_loc(), &cfg()).unwrap();
    let mut longer = codes.clone();
    longer.push(0);
    err(open(&longer, &table_loc(), &cfg()));
    err(open(&codes[..codes.len() - 1], &table_loc(), &cfg()));
    let mxfp8 = vec![0u8; n_rows() as usize * HD / 32 * 33];
    let msg = err(open(&mxfp8, &table_loc(), &cfg()));
    assert!(msg.contains("MXFP8"), "{msg}");

    let wrong_name = located(
        vec![HD as u64, n_rows()],
        GgufTensorType::MXFP8,
        "token_embd.weight",
    );
    err(open(&codes, &wrong_name, &cfg()));
    let wrong_type = located(
        vec![HD as u64, n_rows()],
        GgufTensorType::Q8_0,
        ENGRAM_TABLE,
    );
    err(open(&codes, &wrong_type, &cfg()));
    let swapped = located(
        vec![n_rows(), HD as u64],
        GgufTensorType::MXFP8,
        ENGRAM_TABLE,
    );
    err(open(&codes, &swapped, &cfg()));
    let mut fewer = cfg();
    fewer.head_vocab_sizes[7] -= 1;
    err(open(&codes, &table_loc(), &fewer));
    let mut short = cfg();
    short.head_offsets.truncate(7);
    err(open(&codes, &table_loc(), &short));
}

/// The table reads its own shard of the load's shard set, keeps the bytes
/// of the inode the load opened, and the load sees an atomic replace.
#[test]
fn file_table_reads_the_load_shard_set() {
    let tmp = Temp::new();
    let codes = table_codes(19);
    let first = tmp.0.join("first.gguf");
    std::fs::write(&first, vec![0x3Cu8; 70_000]).unwrap();
    let path = tmp.shard(&codes);
    let shards = Arc::new(ShardSet::open(&[&first, &path]).unwrap());
    let loc = Located {
        shard: 1,
        ..table_loc()
    };
    let t = F8Table::open_in(Arc::clone(&shards), &loc, &cfg()).unwrap();
    let ids = layer_ids(1);
    assert_eq!(
        to_bits(&t.rows(1, &ids).unwrap()),
        expect_rows(&codes, &ids)
    );
    unchanged_since_open(&shards).unwrap();

    let next = tmp.0.join("next.gguf");
    let mut bytes = std::fs::read(&path).unwrap();
    let at = (DATA_OFFSET + REL_OFFSET) as usize;
    bytes[at..].copy_from_slice(&table_codes(23));
    std::fs::write(&next, bytes).unwrap();
    std::fs::rename(&next, &path).unwrap();
    let msg = err(unchanged_since_open(&shards));
    assert!(msg.contains("retry the load"), "{msg}");
    assert_eq!(
        to_bits(&t.rows(1, &ids).unwrap()),
        expect_rows(&codes, &ids)
    );
    err(F8Table::open_in(
        shards,
        &Located {
            shard: 2,
            ..table_loc()
        },
        &cfg(),
    ));
}

#[test]
fn nan_code_in_a_read_row_is_refused_by_row() {
    let tmp = Temp::new();
    let mut codes = table_codes(13);
    let bad_row = cfg().head_offsets[2] as usize + 5;
    codes[bad_row * HD + 5] = 0xFF;
    let t = F8Table::open_at(&tmp.shard(&codes), &table_loc(), &cfg()).unwrap();
    let ids = layer_ids(0);
    assert!(!ids.contains(&(bad_row as u32)));
    t.rows(0, &ids).unwrap();
    let mut ids = layer_ids(0);
    ids[6] = bad_row as u32;
    let msg = err(t.rows(0, &ids));
    assert!(msg.contains(&format!("row {bad_row}")), "{msg}");

    let r = ResidentF8Table::new(codes, &cfg()).unwrap();
    let msg = err(r.rows(0, &ids));
    assert!(msg.contains(&format!("row {bad_row}")), "{msg}");
}

#[test]
fn resident_table_matches_the_file_table() {
    let tmp = Temp::new();
    let codes = table_codes(17);
    let file = F8Table::open_at(&tmp.shard(&codes), &table_loc(), &cfg()).unwrap();
    let mem = ResidentF8Table::new(codes.clone(), &cfg()).unwrap();
    for e in 0..2 {
        let ids = layer_ids(e);
        assert_eq!(
            to_bits(&mem.rows(e, &ids).unwrap()),
            to_bits(&file.rows(e, &ids).unwrap())
        );
    }
    err(mem.rows(0, &layer_ids(1)));
    err(ResidentF8Table::new(codes[1..].to_vec(), &cfg()));
    err(ResidentF8Table::new(
        vec![0u8; n_rows() as usize * HD / 32 * 33],
        &cfg(),
    ));
}

fn real_scan() -> Scan {
    let dir = std::env::var("HY4_DIR").expect("HY4_DIR = directory of the two Hy4 shards");
    let paths = crate::models::deepseek_v41::chat::shard_paths(Path::new(&dir)).unwrap();
    assert_eq!(paths.len(), 2);
    Scan::open(&paths).unwrap()
}

#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards"]
fn real_rows_match_the_oracle() {
    let want = fixture("f8-rows.json");
    let scan = real_scan();
    let t = F8Table::open(&scan).unwrap();
    let p = &want["size_proof"];
    assert_eq!(Some(t.n_rows()), p["n_rows"].as_u64());
    assert_eq!(scan.engram_table.shard, 1);
    let base = scan.engram_table.abs_offset().unwrap();
    assert_eq!(Some(base), p["abs_offset"].as_u64());

    let mut by_row = std::collections::HashMap::new();
    for r in want["rows"].as_array().unwrap() {
        let row = r["row"].as_u64().unwrap();
        assert_eq!(Some(base + row * 96), r["abs_offset"].as_u64());
        let bits: Vec<u32> = r["bits"].as_array().unwrap().iter().map(hex_bits).collect();
        assert_eq!(bits.len(), 96);
        let got = t.read_rows(&[row as u32]).unwrap();
        assert_eq!(to_bits(&got), bits, "row {row}");
        by_row.insert(row, bits);
    }

    let e = &scan.cfg.engram;
    let cols = e.n_cols();
    let bases = want["layer_base_rows"].as_array().unwrap();
    for (layer, b) in bases.iter().enumerate() {
        let b = b.as_u64().unwrap();
        assert_eq!(e.head_offsets[layer * cols], b);
        let ids: Vec<u32> = e.head_offsets[layer * cols..(layer + 1) * cols]
            .iter()
            .map(|&o| o as u32)
            .collect();
        let got = t.rows(layer, &ids).unwrap();
        assert_eq!(to_bits(&got[..96]), by_row[&b], "layer {layer}");
    }
    let last = e.head_offsets[cols - 1] + e.head_vocab_sizes[cols - 1] - 1;
    let mut ids: Vec<u32> = e.head_offsets[..cols].iter().map(|&o| o as u32).collect();
    ids[cols - 1] = last as u32;
    let got = t.rows(0, &ids).unwrap();
    assert_eq!(to_bits(&got[(cols - 1) * 96..]), by_row[&last]);
    ids[0] = last as u32;
    err(t.rows(0, &ids));
}
