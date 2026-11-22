use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use napi::Result;
use serde_json::Value;

use super::f8::{decode, decode_f8_rows, is_nan_code};
use super::rows::{
    DEFAULT_ENGRAM_CACHE_MB, DEFAULT_ENGRAM_THREADS, EngramRows, F8RowCache, F8Table,
    MAX_COALESCE_BYTES, PageSpan, ResidentF8Table, coalesce_spans, parse_cache_mb, prove_f8_extent,
};
use super::scan::{ENGRAM_TABLE, Located};
use super::store::unchanged_since_open;
use super::test_support::{Rng, Temp, fixture, j_u64, j_usize, real_scan};
use crate::models::deepseek_v41::EngramConfig;
use crate::moe_stream::io::ShardSet;
use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};

const HD: usize = 64;
/// Rows of each (layer, column): 2 Engram layers × 4 columns, 300 in all.
const SIZES: [u64; 8] = [40, 35, 30, 45, 50, 25, 40, 35];
const N_ROWS: u64 = 300;
/// Row 0 at 16300, so row 1 straddles the first 16 KiB page boundary.
const DATA_OFFSET: u64 = 16000;
const REL_OFFSET: u64 = 300;
const PAGE: u64 = 16 << 10;

/// OCP E4M3 fn: sign bit 7, exp 6..3 (bias 7), mant 2..0; exp 0 -> m·2⁻⁹; 0x7F/0xFF -> NaN.
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

fn hex_bits(v: &Value) -> u32 {
    u32::from_str_radix(v.as_str().unwrap().trim_start_matches("0x"), 16).unwrap()
}

fn engram_cfg(layers: &[usize], sizes: &[u64]) -> EngramConfig {
    let mut acc = 0u64;
    let offsets = sizes
        .iter()
        .map(|&s| {
            let o = acc;
            acc += s;
            o
        })
        .collect();
    EngramConfig {
        layers: layers.to_vec(),
        max_ngram: 3,
        n_heads: 2,
        head_dim: HD,
        multipliers: (0..layers.len() * 3).map(|i| 2 * i as u64 + 1).collect(),
        head_offsets: offsets,
        head_vocab_sizes: sizes.to_vec(),
        pad_token_id: 0,
        token_map: (0..16).collect(),
        compressed_vocab_size: 16,
        vocab_size: None,
    }
}

fn cfg() -> EngramConfig {
    engram_cfg(&[1, 3], &SIZES)
}

fn codes_for(rows: u64, seed: u64) -> Vec<u8> {
    let raw = Rng::new(seed).bytes(rows as usize * HD);
    raw.into_iter()
        .map(|c| if is_nan_code(c) { c ^ 1 } else { c })
        .collect()
}

fn located(dims: &[u64], ty: GgufTensorType, name: &str, off: (u64, u64)) -> Located {
    let info = GgufTensorInfo {
        name: name.into(),
        n_dims: dims.len() as u32,
        dims: dims.to_vec(),
        tensor_type: ty,
        offset: off.1,
    };
    Located {
        shard: 0,
        data_offset: off.0,
        info,
    }
}

fn table_loc() -> Located {
    let off = (DATA_OFFSET, REL_OFFSET);
    located(
        &[HD as u64, N_ROWS],
        GgufTensorType::MXFP8,
        ENGRAM_TABLE,
        off,
    )
}

fn write_shard(tmp: &Temp, table: &[u8]) -> PathBuf {
    let mut bytes = vec![0xA5u8; (DATA_OFFSET + REL_OFFSET) as usize];
    bytes.extend_from_slice(table);
    tmp.write("table.gguf", &bytes)
}

fn open_tmp(seed: u64) -> (Temp, Vec<u8>, PathBuf, F8Table) {
    let (tmp, codes) = (Temp::new("rows"), codes_for(N_ROWS, seed));
    let path = write_shard(&tmp, &codes);
    let t = F8Table::open_at(&path, &table_loc(), &cfg()).unwrap();
    (tmp, codes, path, t)
}

fn make_f8_table(
    path: &Path,
    loc: &Located,
    c: &EngramConfig,
    threads: usize,
    cap: usize,
) -> (F8Table, Arc<ShardSet>) {
    let shards = Arc::new(ShardSet::open(&[path]).unwrap());
    let t = F8Table::open_with_options(Arc::clone(&shards), 0, loc, c, threads, cap).unwrap();
    (t, shards)
}

fn to_bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn assert_eq_rows(got: &[f32], codes: &[u8], ids: &[u32]) {
    let want: Vec<u32> = ids
        .iter()
        .flat_map(|&id| codes[id as usize * HD..(id as usize + 1) * HD].iter())
        .map(|&c| formula(c).unwrap().to_bits())
        .collect();
    assert_eq!(to_bits(got), want);
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

fn span(start: u64, end: u64, rows: &[u32]) -> PageSpan {
    PageSpan {
        start,
        end,
        rows: rows.to_vec(),
    }
}

#[test]
fn lut_matches_the_ggml_formula() {
    let lut = fixture("f8-rows.json")["lut"].as_array().unwrap().clone();
    assert_eq!(lut.len(), 256);
    for c in 0..=255u8 {
        let e = &lut[usize::from(c)];
        assert_eq!(e["code"].as_u64(), Some(u64::from(c)));
        match formula(c) {
            None => {
                assert!(decode(c).is_nan() && is_nan_code(c), "code {c:#04x}");
                assert_eq!(e["nan"].as_bool(), Some(true));
            }
            Some(v) => {
                let b = decode(c).to_bits();
                assert!(!is_nan_code(c) && b == v.to_bits(), "{c:#x}");
                assert_eq!(b, hex_bits(&e["bits"]), "{c:#x}");
            }
        }
    }
    assert_eq!((decode(0x7E), decode(0xFE)), (448.0, -448.0));
    assert_eq!((decode(0x01), decode(0x08)), (2f32.powi(-9), 2f32.powi(-6)));
    assert_eq!(decode(0x80).to_bits(), (-0.0f32).to_bits());
    assert!(decode(0x7F).is_nan() && decode(0xFF).is_nan());
}

#[test]
fn decode_refuses_nan_codes_and_length_mismatch() {
    let mut out = [0f32; 8];
    decode_f8_rows(&[0x00, 0x01, 0x7E, 0xFE, 0x80, 0x38, 0x08, 0x07], &mut out).unwrap();
    assert_eq!((out[2], out[5]), (448.0, 1.0));
    assert!(err(decode_f8_rows(&[0, 0, 0, 0, 0, 0xFF, 0, 0], &mut out)).contains("value 5"));
    assert!(err(decode_f8_rows(&[0, 0x7F, 0, 0, 0, 0, 0, 0], &mut out)).contains("value 1"));
    err(decode_f8_rows(&[0; 7], &mut out));
}

#[test]
fn size_proof_accepts_f8_only() {
    let p = &fixture("f8-rows.json")["size_proof"];
    let (n, hd) = (j_u64(&p["n_rows"]), j_usize(&p["head_dim"]));
    let (f8, mxfp8) = (j_u64(&p["f8_bytes"]), j_u64(&p["mxfp8_bytes"]));
    assert_eq!((n, hd, f8), (660_006_416, 96, 63_360_615_936));
    assert_eq!(p["extent_to_eof"].as_u64(), Some(f8));
    prove_f8_extent(f8, n, hd).unwrap();
    prove_f8_extent(400, 10, 40).unwrap();
    assert!(err(prove_f8_extent(mxfp8, n, hd)).contains("MXFP8"));
    for (ext, r, d) in [
        (f8 + 1, n, hd),
        (f8 - 1, n, hd),
        (0, 0, hd),
        (0, n, 0),
        (401, 10, 40),
    ] {
        err(prove_f8_extent(ext, r, d));
    }
}

#[test]
fn file_table_reads_each_layer_and_column() {
    let (_tmp, codes, _, t) = open_tmp(7);
    assert_eq!((t.n_rows(), SIZES.iter().sum::<u64>()), (N_ROWS, N_ROWS));
    for e in 0..2 {
        let ids = layer_ids(e);
        assert_eq_rows(&t.rows(e, &ids).unwrap(), &codes, &ids);
    }
    let all: Vec<u32> = (0..N_ROWS as u32).rev().collect();
    assert_eq_rows(&t.read_rows(&all).unwrap(), &codes, &all);
    assert!(t.read_stats().0 > 0);

    let mut ids = layer_ids(0);
    ids[1] = ids[0];
    assert!(err(t.rows(0, &ids)).contains("column 1"));
    err(t.rows(0, &layer_ids(1)));
    err(t.rows(0, &layer_ids(0)[..6]));
    assert!(err(t.rows(2, &layer_ids(0))).contains("no column"));
    err(t.read_rows(&[N_ROWS as u32]));
}

#[test]
fn file_table_refuses_a_bad_size_or_header() {
    use GgufTensorType::{MXFP8, Q8_0};
    let (tmp, codes) = (Temp::new("rows"), codes_for(N_ROWS, 11));
    let open = |tab: &[u8], loc: &Located, c: &EngramConfig| {
        F8Table::open_at(&write_shard(&tmp, tab), loc, c).map(|_| ())
    };
    let (loc, c) = (table_loc(), cfg());
    open(&codes, &loc, &c).unwrap();
    let mut longer = codes.clone();
    longer.push(0);
    err(open(&longer, &loc, &c));
    err(open(&codes[..codes.len() - 1], &loc, &c));
    let mxfp8 = vec![0u8; N_ROWS as usize * HD / 32 * 33];
    assert!(err(open(&mxfp8, &loc, &c)).contains("MXFP8"));
    let off = (DATA_OFFSET, REL_OFFSET);
    for (dims, ty, name) in [
        ([HD as u64, N_ROWS], MXFP8, "token_embd.weight"),
        ([HD as u64, N_ROWS], Q8_0, ENGRAM_TABLE),
        ([N_ROWS, HD as u64], MXFP8, ENGRAM_TABLE),
    ] {
        err(open(&codes, &located(&dims, ty, name, off), &c));
    }
    let (mut fewer, mut short) = (cfg(), cfg());
    fewer.head_vocab_sizes[7] -= 1;
    short.head_offsets.truncate(7);
    err(open(&codes, &loc, &fewer));
    err(open(&codes, &loc, &short));
}

#[test]
fn file_table_reads_the_load_shard_set() {
    let (tmp, codes) = (Temp::new("rows"), codes_for(N_ROWS, 19));
    let pad = vec![0x3Cu8; 70_000];
    let (first, path) = (tmp.write("first.gguf", &pad), write_shard(&tmp, &codes));
    let shards = Arc::new(ShardSet::open(&[&first, &path]).unwrap());
    let loc = Located {
        shard: 1,
        ..table_loc()
    };
    let t = F8Table::open_in(Arc::clone(&shards), &loc, &cfg()).unwrap();
    let ids = layer_ids(1);
    assert_eq_rows(&t.rows(1, &ids).unwrap(), &codes, &ids);
    unchanged_since_open(&shards).unwrap();

    let mut bytes = std::fs::read(&path).unwrap();
    let start = (DATA_OFFSET + REL_OFFSET) as usize;
    bytes[start..].copy_from_slice(&codes_for(N_ROWS, 23));
    std::fs::rename(tmp.write("next.gguf", &bytes), &path).unwrap();
    assert!(err(unchanged_since_open(&shards)).contains("retry the load"));
    assert_eq_rows(&t.rows(1, &ids).unwrap(), &codes, &ids);
    let bad = Located {
        shard: 2,
        ..table_loc()
    };
    err(F8Table::open_in(shards, &bad, &cfg()));
}

#[test]
fn nan_code_in_a_read_row_is_refused_by_row() {
    let (tmp, mut codes) = (Temp::new("rows"), codes_for(N_ROWS, 13));
    let bad = cfg().head_offsets[2] as usize + 5;
    codes[bad * HD + 5] = 0xFF;
    let t = F8Table::open_at(&write_shard(&tmp, &codes), &table_loc(), &cfg()).unwrap();
    let mut ids = layer_ids(0);
    assert!(!ids.contains(&(bad as u32)));
    t.rows(0, &ids).unwrap();
    ids[6] = bad as u32;
    let want = format!("row {bad}");
    assert!(err(t.rows(0, &ids)).contains(&want));
    let r = ResidentF8Table::new(codes, &cfg()).unwrap();
    assert!(err(r.rows(0, &ids)).contains(&want));
}

#[test]
fn resident_table_matches_the_file_table() {
    let (_tmp, codes, _, file) = open_tmp(17);
    let mem = ResidentF8Table::new(codes.clone(), &cfg()).unwrap();
    for e in 0..2 {
        let ids = layer_ids(e);
        let (a, b) = (mem.rows(e, &ids).unwrap(), file.rows(e, &ids).unwrap());
        assert_eq!(to_bits(&a), to_bits(&b));
    }
    err(mem.rows(0, &layer_ids(1)));
    err(ResidentF8Table::new(codes[1..].to_vec(), &cfg()));
    let mxfp8 = vec![0u8; N_ROWS as usize * HD / 32 * 33];
    err(ResidentF8Table::new(mxfp8, &cfg()));
}

#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards"]
fn real_rows_match_the_oracle() {
    let (want, scan) = (fixture("f8-rows.json"), real_scan());
    let t = F8Table::open(&scan).unwrap();
    let (p, base) = (&want["size_proof"], scan.engram_table.abs_offset().unwrap());
    assert_eq!(Some(t.n_rows()), p["n_rows"].as_u64());
    assert_eq!(scan.engram_table.shard, 1);
    assert_eq!(Some(base), p["abs_offset"].as_u64());

    let mut by_row = std::collections::HashMap::new();
    for r in want["rows"].as_array().unwrap() {
        let row = j_u64(&r["row"]);
        assert_eq!(Some(base + row * 96), r["abs_offset"].as_u64());
        let bits: Vec<u32> = r["bits"].as_array().unwrap().iter().map(hex_bits).collect();
        assert_eq!(bits.len(), 96);
        let got = to_bits(&t.read_rows(&[row as u32]).unwrap());
        assert_eq!(got, bits, "row {row}");
        by_row.insert(row, bits);
    }

    let (e, cols) = (&scan.cfg.engram, scan.cfg.engram.n_cols());
    for (layer, b) in want["layer_base_rows"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let b = j_u64(b);
        assert_eq!(e.head_offsets[layer * cols], b);
        let offs = &e.head_offsets[layer * cols..(layer + 1) * cols];
        let ids: Vec<u32> = offs.iter().map(|&o| o as u32).collect();
        let got = to_bits(&t.rows(layer, &ids).unwrap()[..96]);
        assert_eq!(got, by_row[&b], "layer {layer}");
    }
    let last = e.head_offsets[cols - 1] + e.head_vocab_sizes[cols - 1] - 1;
    let mut ids: Vec<u32> = e.head_offsets[..cols].iter().map(|&o| o as u32).collect();
    ids[cols - 1] = last as u32;
    let got = to_bits(&t.rows(0, &ids).unwrap()[(cols - 1) * 96..]);
    assert_eq!(got, by_row[&last]);
    ids[0] = last as u32;
    err(t.rows(0, &ids));
}

#[test]
fn coalesce_spans_merges_across_16k_straddling_and_splits_at_max_bytes() {
    const P: u64 = PAGE;
    const M: u64 = MAX_COALESCE_BYTES;
    assert!(coalesce_spans(&[], 0, HD, 1 << 20).is_empty());

    let base = DATA_OFFSET + REL_OFFSET;
    let s = coalesce_spans(&[2, 0, 1, 2, 0], base, HD, base + 512 * 1024);
    assert_eq!(s, [span(0, 2 * P, &[0, 1, 2])]);

    let rpp = (P / HD as u64) as u32;
    let s = coalesce_spans(&[3 * rpp, rpp, 0], 0, HD, 8 * P);
    assert_eq!(
        s,
        [span(0, 2 * P, &[0, rpp]), span(3 * P, 4 * P, &[3 * rpp])]
    );

    let p17: Vec<u32> = (0..17).map(|p| p * rpp).collect();
    let s = coalesce_spans(&p17, 0, HD, 32 * P);
    assert_eq!(
        (s.len(), s[0].start, s[0].end, s[0].rows.len()),
        (2, 0, M, 16)
    );
    assert_eq!(s[1], span(M, M + P, &[16 * rpp]));

    let s_base = 20u64;
    let s_row = ((M - s_base) / HD as u64) as u32;
    let s_off = s_base + u64::from(s_row) * HD as u64;
    assert!(s_off < M && s_off + HD as u64 > M);
    let mut ids: Vec<u32> = (0..16).map(|p| p * rpp).collect();
    ids.push(s_row);
    let s = coalesce_spans(&ids, s_base, HD, 32 * P);
    assert_eq!((s.len(), s[0].start, s[0].end), (2, 0, M));
    assert_eq!(s[1], span(M - P, M + P, &[s_row]));

    let eof = 2 * P + 500;
    let last = ((eof - HD as u64) / HD as u64) as u32;
    assert_eq!(
        coalesce_spans(&[last], 0, HD, eof),
        [span(2 * P, eof, &[last])]
    );

    let (_tmp, codes, path, _) = open_tmp(31);
    let (t, _) = make_f8_table(&path, &table_loc(), &cfg(), 4, 0);
    assert_eq_rows(&t.read_rows(&[0, 1, 2]).unwrap(), &codes, &[0, 1, 2]);
    assert_eq!(t.read_stats().0, 2 * P);
}

#[test]
fn f8_row_cache_lru_hits_eviction_and_zero_capacity() {
    for (raw, want) in [
        (None, Some(DEFAULT_ENGRAM_CACHE_MB)),
        (Some(""), Some(DEFAULT_ENGRAM_CACHE_MB)),
        (Some("  "), Some(DEFAULT_ENGRAM_CACHE_MB)),
        (Some("0"), Some(0)),
        (Some("64"), Some(64)),
        (Some("-1"), None),
        (Some("abc"), None),
    ] {
        assert_eq!(parse_cache_mb(raw).ok(), want, "{raw:?}");
    }
    let def = F8RowCache::from_budget_bytes(DEFAULT_ENGRAM_CACHE_MB << 20, 96);
    assert_eq!(def.capacity(), 2_097_152);

    let mut off = F8RowCache::from_budget_bytes(0, 4);
    off.insert(7, &[1, 2, 3, 4]);
    assert!((off.capacity(), off.len()) == (0, 0) && off.get(7).is_none());

    let mut one = F8RowCache::new(1, 4);
    one.insert(10, &[1; 4]);
    assert_eq!(one.get(10), Some(&[1; 4][..]));
    one.insert(20, &[2; 4]);
    assert_eq!((one.len(), one.contains(10)), (1, false));
    assert_eq!(one.get(20), Some(&[2; 4][..]));

    let mut lru = F8RowCache::new(3, 4);
    (1..=3u32).for_each(|k| lru.insert(k, &[k as u8 * 10; 4]));
    assert!(lru.len() == 3 && lru.get(1) == Some(&[10; 4][..]));
    lru.insert(4, &[40; 4]);
    assert!(lru.len() == 3 && !lru.contains(2));
    assert!(lru.contains(1) && lru.contains(3) && lru.contains(4));
    lru.insert(3, &[30; 4]);
    assert_eq!(lru.len(), 3);
    lru.insert(5, &[50; 4]);
    assert!(lru.len() == 3 && !lru.contains(1));
    for (k, v) in [(3, [30; 4]), (4, [40; 4]), (5, [50; 4])] {
        assert_eq!(lru.get(k), Some(&v[..]));
    }
}

#[test]
fn f8_table_lru_hits_eviction_nan_rejection_and_zero_capacity() {
    let (tmp, mut codes) = (Temp::new("rows"), codes_for(N_ROWS, 41));
    let bad = 90u32;
    codes[bad as usize * HD + 7] = 0x7F;
    let (path, loc, c) = (write_shard(&tmp, &codes), table_loc(), cfg());

    let (t, _) = make_f8_table(&path, &loc, &c, 4, 4);
    assert_eq_rows(&t.read_rows(&[0, 1, 2, 3]).unwrap(), &codes, &[0, 1, 2, 3]);
    let b_fill = t.read_stats().0;
    assert!(t.cache_len() == 4 && b_fill > 0);
    assert_eq_rows(&t.read_rows(&[0]).unwrap(), &codes, &[0]);
    assert_eq!(t.read_stats().0, b_fill);
    assert_eq_rows(&t.read_rows(&[4]).unwrap(), &codes, &[4]);
    let b_ev = t.read_stats().0;
    assert!(t.cache_len() == 4 && b_ev > b_fill);
    assert_eq_rows(&t.read_rows(&[0]).unwrap(), &codes, &[0]);
    assert_eq!(t.read_stats().0, b_ev);
    assert_eq_rows(&t.read_rows(&[1]).unwrap(), &codes, &[1]);
    assert!(t.read_stats().0 > b_ev);

    let (t_nan, _) = make_f8_table(&path, &loc, &c, 4, 16);
    let msg = err(t_nan.read_rows(&[10, bad, 11]));
    assert!(msg.contains(&format!("row {bad}")) && msg.contains("value 7"));
    assert_eq!(t_nan.cache_len(), 0);
    assert!(err(t_nan.read_rows(&[bad])).contains("value 7") && t_nan.cache_len() == 0);
    assert_eq_rows(&t_nan.read_rows(&[10, 11]).unwrap(), &codes, &[10, 11]);
    assert_eq!(t_nan.cache_len(), 2);

    let (t_off, _) = make_f8_table(&path, &loc, &c, 4, 0);
    assert_eq!(t_off.cache_capacity(), 0);
    let a = t_off.read_rows(&[0, 1]).unwrap();
    let b1 = t_off.read_stats().0;
    let b = t_off.read_rows(&[0, 1]).unwrap();
    assert!(b1 > 0 && t_off.read_stats().0 > b1 && t_off.cache_len() == 0);
    assert_eq!(to_bits(&a), to_bits(&b));
}

#[test]
fn prefetch_take_matches_rows_cancel_drop_and_4_layer_fifo() {
    let (tmp, codes, _, t) = open_tmp(53);
    let resident = ResidentF8Table::new(codes.clone(), &cfg()).unwrap();
    assert!(resident.prefetch(0, &layer_ids(0)).unwrap().is_none());

    for e in 0..2 {
        let ids = layer_ids(e);
        let h = t.prefetch(e, &ids).unwrap().expect("F8Table prefetches");
        let got = t.take_prefetched(h).unwrap();
        assert_eq!(to_bits(&got), to_bits(&t.rows(e, &ids).unwrap()));
        assert_eq_rows(&got, &codes, &ids);
    }
    let uniq = [0u32, 41, 80, 110];
    let h = t.prefetch(0, &uniq).unwrap().unwrap();
    assert_eq!(
        to_bits(&t.take_prefetched(h).unwrap()),
        to_bits(&t.rows(0, &uniq).unwrap())
    );
    assert!(t.prefetch_rows(&[]).unwrap().wait().unwrap().is_empty());

    // 4-Engram-layer table spanning 64 distinct 16 KiB pages (1 MiB) for cancel drop + FIFO.
    let stride = 2 * (PAGE / HD as u64);
    let cfg_4 = engram_cfg(&[1, 11, 21, 31], &[stride; 16]);
    let (n4, codes_4) = (16 * stride, codes_for(16 * stride, 77));
    let path_4 = tmp.write("table4.gguf", &codes_4);
    let loc_4 = located(
        &[HD as u64, n4],
        GgufTensorType::MXFP8,
        ENGRAM_TABLE,
        (0, 0),
    );

    let (t_c, shards_c) = make_f8_table(&path_4, &loc_4, &cfg_4, 1, 0);
    shards_c.delay_ms.store(20, Ordering::Relaxed);
    let spans: Vec<u32> = cfg_4.head_offsets.iter().map(|&o| o as u32).collect();
    let handle = t_c.prefetch_rows(&spans).unwrap();
    let t0 = Instant::now();
    drop(handle);
    assert!(t0.elapsed() < Duration::from_millis(50), "drop blocked");
    shards_c.delay_ms.store(0, Ordering::Relaxed);
    let id0 = [cfg_4.head_offsets[0] as u32];
    assert_eq_rows(&t_c.read_rows(&id0).unwrap(), &codes_4, &id0);
    assert!(
        t_c.read_stats().0 < 16 * PAGE,
        "cancelled spans must skip pread"
    );

    let lids: Vec<Vec<u32>> = (0..4)
        .map(|e| {
            cfg_4.head_offsets[4 * e..4 * (e + 1)]
                .iter()
                .map(|&o| o as u32)
                .collect()
        })
        .collect();
    for (threads, cap, check_fifo) in [(1, 0, true), (DEFAULT_ENGRAM_THREADS, 64, false)] {
        let (t4, _) = make_f8_table(&path_4, &loc_4, &cfg_4, threads, cap);
        let hs: Vec<_> = (0..4)
            .map(|e| t4.prefetch(e, &lids[e]).unwrap().unwrap())
            .collect();
        if check_fifo {
            let seqs: Vec<u64> = hs.iter().map(|h| h.done_seq()).collect();
            assert_eq!(seqs, [1, 2, 3, 4]);
        }
        for (e, h) in hs.into_iter().enumerate() {
            let got = t4.take_prefetched(h).unwrap();
            if !check_fifo {
                assert_eq!(to_bits(&got), to_bits(&t4.rows(e, &lids[e]).unwrap()));
            }
            assert_eq_rows(&got, &codes_4, &lids[e]);
        }
        if !check_fifo {
            assert_eq!(t4.cache_len(), 16);
        }
    }
}
