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
