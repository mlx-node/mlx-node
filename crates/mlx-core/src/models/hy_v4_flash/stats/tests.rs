use super::*;

fn class(fills: usize, slots: usize, uses: usize, hits: usize) -> KqStats {
    KqStats {
        fills,
        slots,
        bytes: slots as u64 * 1_000_000,
        ns: slots as u64 * 1_000,
        read_ns: slots as u64 * 600,
        repack_ns: slots as u64 * 300,
        uses,
        hits,
        promoted: hits / 2,
        promote_ns: 7,
    }
}

#[test]
fn counters_sum_the_classes_and_subtract() {
    let c = Counters::of(&[class(2, 10, 40, 20), class(3, 6, 30, 12)], (512, 9), 2);
    assert_eq!(
        c,
        Counters {
            fills: 5,
            fill_slots: 16,
            fill_bytes: 16_000_000,
            fill_ns: 16_000,
            read_ns: 9_600,
            repack_ns: 4_800,
            uses: 70,
            hits: 32,
            promoted: 16,
            promote_ns: 14,
            engram_bytes: 512,
            engram_ns: 9,
            staging_layers: 2,
        }
    );
    let before = Counters::of(&[class(1, 4, 10, 5), class(1, 2, 10, 2)], (500, 4), 1);
    let d = c.since(&before);
    assert_eq!(
        (
            d.fills,
            d.fill_slots,
            d.uses,
            d.hits,
            d.engram_bytes,
            d.engram_ns
        ),
        (3, 10, 50, 25, 12, 5)
    );
    // the staging layers are the current value, not a difference
    assert_eq!(d.staging_layers, 2);
    // a counter that went back (a fresh bank) never underflows
    assert_eq!(before.since(&c).fills, 0);
}

fn line(counters: Counters, decode_s: &[f64]) -> Value {
    Line {
        stage: "chat",
        prompt_tok: 100,
        gen_tok: 1 + decode_s.len(),
        ttft_s: 2.0,
        decode_s,
        counters,
        wall_s: 4.0,
        cache_slots: [190, 220],
        ctx: 262_144,
        chunk: 8192,
        peak_footprint: 3 << 30,
        mlx_cache_bytes: (1u64 << 29) as f64,
        swapout_delta: 1 << 21,
        memstatus_min: Some(40),
    }
    .json()
}

#[test]
fn line_fields() {
    let c = Counters {
        fill_slots: 10,
        fill_bytes: 4_000_000_000,
        fill_ns: 1_000_000_000,
        read_ns: 600_000_000,
        repack_ns: 300_000_000,
        uses: 40,
        hits: 30,
        engram_bytes: 1_000,
        engram_ns: 500_000_000,
        staging_layers: 2,
        ..Counters::default()
    };
    let v = line(c, &[0.5, 0.25, 0.2, 0.3]);
    let f = |k: &str| v[k].as_f64().unwrap_or_else(|| panic!("{k}: {}", v[k]));
    let close = |k: &str, want: f64| assert!((f(k) - want).abs() < 1e-9, "{k}: {} vs {want}", f(k));
    assert_eq!(v["stage"], "chat");
    assert_eq!(v["gen_tok"], 5);
    close("prefill_tok_s", 50.0);
    // median of [0.2, 0.25, 0.3, 0.5] at index 2
    close("decode_tok_s_median", 1.0 / 0.3);
    close("hit_rate_uses", 0.75);
    close("hit_rate_bytes", 0.75);
    close("miss_gb_per_tok", 0.8);
    close("read_gbps", 4.0);
    close("t_c_ms", (4.0 - 1.0 - 0.5) * 1e3 / 5.0);
    close("t_io_ms", 200.0);
    close("t_read_ms", 120.0);
    close("t_repack_ms", 60.0);
    close("t_engram_ms", 100.0);
    assert_eq!(v["bytes_read_total"], 4_000_001_000u64);
    assert_eq!(v["cache_slots"], 410);
    assert_eq!(v["cache_slots_by_class"], json!([190, 220]));
    assert_eq!(v["staging_layers"], 2);
    assert_eq!(
        (v["ctx"].as_u64(), v["prefill_chunk"].as_u64()),
        (Some(262_144), Some(8192))
    );
    close("peak_footprint_gib", 3.0);
    close("mlx_cache_gib", 0.5);
    close("swapout_delta_mib", 2.0);
    assert_eq!(v["memstatus_min"], 40);
    assert_eq!(v["cold"], false);
}

#[test]
fn line_without_decode_or_uses() {
    let v = line(Counters::default(), &[]);
    assert_eq!(v["decode_tok_s_median"].as_f64(), Some(0.0));
    assert_eq!(v["hit_rate_uses"].as_f64(), Some(0.0));
    assert_eq!(v["hit_rate_bytes"].as_f64(), Some(1.0));
    assert_eq!(v["cold"], true);
    assert!(v["memstatus_min"].as_u64().is_some());
}

#[test]
fn emit_appends_one_line_per_call() {
    let dir = std::env::temp_dir().join(format!("hy4-stats-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("stats.jsonl");
    emit_to(None, &json!({"a": 1})).unwrap();
    assert!(!path.exists());
    emit_to(Some(path.as_os_str()), &json!({"a": 1})).unwrap();
    emit_to(Some(path.as_os_str()), &json!({"b": [2, 3]})).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let lines: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines, vec![json!({"a": 1}), json!({"b": [2, 3]})]);
}

#[test]
fn phase_names_are_distinct() {
    let mut names: Vec<_> = PHASES.iter().map(|p| p.name()).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), PHASES.len());
    assert_eq!(phase_seconds().len(), PHASES.len());
    assert!(PHASES.iter().enumerate().all(|(i, &p)| p as usize == i));
}

fn wave_counters(uses: u64, reads: u64) -> Counters {
    Counters {
        uses,
        fill_slots: reads,
        fill_bytes: reads * 1_000_000,
        fill_ns: reads * 1_000_000,
        engram_bytes: 64,
        ..Counters::default()
    }
}

fn wave<'a>(owners: &'a [(u32, usize)], uses: u64, reads: u64, evals: u64) -> WaveSample<'a> {
    WaveSample {
        owners,
        picks_per_row: 8 * 41,
        counters: wave_counters(uses, reads),
        evals,
        secs: 0.25,
    }
}

#[test]
fn waves_count_rows_union_and_lanes() {
    let mut log = WaveLog::default();
    log.record(&wave(&[(1, 1)], 300, 60, 54));
    log.record(&wave(&[(1, 1), (2, 1)], 500, 50, 108));
    log.record(&wave(&[(1, 4), (2, 4), (3, 4), (4, 4)], 900, 90, 216));
    let w = log.waves;
    assert_eq!(w.decode, 3);
    // rows 1, 2 and 16 (the last bucket takes the wave past 12)
    let mut hist = [0; WAVE_ROWS];
    hist[0] = 1;
    hist[1] = 1;
    hist[WAVE_ROWS - 1] = 1;
    assert_eq!(w.rows, hist);
    assert_eq!(w.union_experts, 1_700);
    assert_eq!(w.row_experts, 19 * 8 * 41);
    assert_eq!(w.miss_bytes, 200_000_000);
    assert_eq!(w.tokens, 19);
    assert!((w.evals_per_wave() - 126.0).abs() < 1e-9);
    let by = w.hit_rate_by_lanes();
    assert!((by[0] - 0.8).abs() < 1e-9, "{by:?}");
    assert!((by[1] - 0.9).abs() < 1e-9, "{by:?}");
    assert_eq!(by[2], 0.0);
    assert!((by[3] - 0.9).abs() < 1e-9, "{by:?}");
    assert!((w.miss_bytes_per_token() - 200_000_000.0 / 19.0).abs() < 1e-6);
}

#[test]
fn cuts_take_back_only_the_last_wave_rows() {
    let mut log = WaveLog::default();
    log.record(&wave(&[(7, 1)], 10, 1, 1));
    log.record(&wave(&[(7, 4), (8, 1)], 10, 1, 1));
    assert_eq!(log.waves.tokens, 6);
    // a verify of 4 rows keeps 2: two rows emitted no token
    log.cut(7, 2);
    assert_eq!(log.waves.tokens, 4);
    // a regenerate cut deeper than the wave takes back only its rest
    log.cut(7, 50);
    assert_eq!(log.waves.tokens, 2);
    log.cut(7, 1);
    assert_eq!(log.waves.tokens, 2);
    // an owner with no wave, and a forgotten one, take nothing back
    log.cut(9, 3);
    log.forget(8);
    log.cut(8, 1);
    assert_eq!(log.waves.tokens, 2);
    log.stalled(5);
    log.stalled(7);
    assert_eq!(log.waves.prefill_stall_ns, 12);
}

#[test]
fn empty_waves_read_zero() {
    let w = Waves::default();
    assert_eq!(w.miss_bytes_per_token(), 0.0);
    assert_eq!(w.evals_per_wave(), 0.0);
    assert_eq!(w.hit_rate_by_lanes(), [0.0; LANES_MAX]);
    assert_eq!(served_rate(0, 0), 0.0);
    assert_eq!(served_rate(4, 9), 0.0);
}

#[test]
fn wave_line_fields() {
    let owners = [(1, 1), (2, 3)];
    let v = wave(&owners, 400, 100, 60).json();
    assert_eq!(v["stage"], "wave");
    assert_eq!(v["owners"], 2);
    assert_eq!(v["rows"], 4);
    assert_eq!(v["union_experts"], 400);
    assert_eq!(v["row_experts"], 4 * 8 * 41);
    assert_eq!(v["miss_bytes"], 100_000_000u64);
    assert_eq!(v["engram_bytes"], 64);
    assert_eq!(v["evals"], 60);
    let f = |k: &str| v[k].as_f64().unwrap_or_else(|| panic!("{k}: {}", v[k]));
    assert!((f("served_rate") - 0.75).abs() < 1e-9);
    assert!((f("read_gbps") - 1.0).abs() < 1e-9);
    assert!((f("wave_ms") - 250.0).abs() < 1e-9);
}

#[test]
fn counted_syncs_move_the_counter() {
    let before = evals();
    synced();
    synced();
    assert!(evals() >= before + 2);
}
