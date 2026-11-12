use super::*;

const G: u64 = 1 << 30;
const SPARE_RESERVE: u64 = 4 * G;

/// A host whose headroom above the spare reserve is `headroom`.
fn host(headroom: u64) -> HostMemory {
    HostMemory {
        available: SPARE_RESERVE + headroom,
        spare: SPARE_RESERVE + headroom,
    }
}

fn rule_at(headroom: u64, system: Pressure, hysteresis: u64, rule: &mut HostRule) {
    rule.sample(
        Some(&host(headroom)),
        SPARE_RESERVE,
        0,
        system,
        hysteresis,
        0,
        false,
    );
}

/// Splash `HostRule::admits` at a headroom that keeps the margin, by
/// pressure and band: InService, Start, Optional, Growth, Floor, each
/// without and with the host's exemption.
#[test]
fn host_rule_follows_splash() {
    const CLASSES: [ClaimClass; 5] = [
        ClaimClass::InService,
        ClaimClass::Start,
        ClaimClass::Optional,
        ClaimClass::Growth,
        ClaimClass::Floor,
    ];
    let table: [(Pressure, bool, &str); 6] = [
        (Pressure::Normal, false, "11 11 11 11 11"),
        (Pressure::Normal, true, "11 00 00 01 11"),
        (Pressure::Warning, false, "01 00 00 01 11"),
        (Pressure::Warning, true, "01 00 00 01 11"),
        (Pressure::Critical, false, "00 00 00 00 00"),
        (Pressure::Critical, true, "00 00 00 00 00"),
    ];
    for (system, band, row) in table {
        let mut rule = HostRule::default();
        if band {
            // In at 0.5 GiB; 1.2 GiB stays inside a 0.5 GiB reach.
            rule_at(G / 2, system, G / 2, &mut rule);
            rule_at(G + G / 5, system, G / 2, &mut rule);
        } else {
            rule_at(4 * G, system, 0, &mut rule);
        }
        assert_eq!(rule.constrained(), band, "{system:?} band");
        let expected: Vec<bool> = row
            .chars()
            .filter(|c| *c != ' ')
            .map(|c| c == '1')
            .collect();
        let mut at = 0;
        for cls in CLASSES {
            for exempt in [false, true] {
                assert_eq!(
                    rule.admits(cls, 0, exempt),
                    expected[at],
                    "{system:?} band={band} {cls:?} exempt={exempt}"
                );
                at += 1;
            }
        }
    }
    // The 03-PLAN gate: under macOS's warning an exempt lane passes.
    let mut rule = HostRule::default();
    rule_at(4 * G, Pressure::Warning, 0, &mut rule);
    assert!(rule.admits(ClaimClass::InService, G, true));
    // Without the margin beyond the bytes, an InService claim is refused.
    rule_at(G + G / 2, Pressure::Normal, 0, &mut rule);
    assert!(rule.admits(ClaimClass::InService, G / 2, false));
    assert!(!rule.admits(ClaimClass::InService, G / 2 + 1, false));
    assert_eq!(Pressure::Warning.name(), "warning");
}

#[test]
fn band_enters_below_the_margin_and_leaves_by_reach() {
    let mut rule = HostRule::default();
    let hysteresis = G / 2;
    rule_at(G / 2, Pressure::Normal, hysteresis, &mut rule);
    assert!(rule.constrained(), "0.5 GiB enters the band");
    rule_at(G + G / 5, Pressure::Normal, hysteresis, &mut rule);
    assert!(rule.constrained(), "1.2 GiB stays inside the reach");
    rule_at(G + 3 * G / 5, Pressure::Normal, hysteresis, &mut rule);
    assert!(!rule.constrained(), "1.6 GiB clears M by the reach");

    // Production: no growth pools, so the band leaves at M.
    let mut rule = HostRule::default();
    rule_at(G / 2, Pressure::Normal, 0, &mut rule);
    assert!(rule.constrained());
    rule_at(G, Pressure::Normal, 0, &mut rule);
    assert!(!rule.constrained(), "1.0 GiB leaves with no hysteresis");
    assert!(rule.measured());
    assert_eq!(rule.headroom(), G);
    assert_eq!(rule.level(), Pressure::Normal);

    // Unmeasured: in the band, no deficit.
    rule.sample(None, SPARE_RESERVE, 0, Pressure::Normal, 0, 0, false);
    assert!(rule.constrained());
    assert_eq!(rule.deficit_bytes(G, 0), 0);
    // A host pass that found nothing waives the band.
    rule_at(G / 2, Pressure::Normal, hysteresis, &mut rule);
    rule_at(G + G / 5, Pressure::Normal, hysteresis, &mut rule);
    assert!(rule.constrained());
    rule.sample(
        Some(&host(G + G / 5)),
        SPARE_RESERVE,
        0,
        Pressure::Normal,
        hysteresis,
        0,
        true,
    );
    assert!(!rule.constrained());
    // Deficit: M + refused - h - returning.
    rule_at(G / 2, Pressure::Normal, 0, &mut rule);
    assert_eq!(rule.deficit_bytes(G / 4, G / 8), G / 2 + G / 4 - G / 8);
}

#[test]
fn host_estimate_matches_splash() {
    const PAGE: u64 = 16_384;
    let base = HostPages {
        free: 1000,
        speculative: 100,
        file_backed: 300,
        purgeable: 50,
        anonymous: 600,
        compressor: 0,
        compressed: 0,
    };
    // 900 free + 300 file + 50 purgeable = 1250 pages.
    // Compressor empty: credit = 600 / 2.
    let empty = host_memory_at(&base, PAGE, Some(Pressure::Normal));
    assert_eq!(
        empty,
        HostMemory {
            available: (1250 + 300) * PAGE,
            spare: (1250 + 150) * PAGE,
        }
    );
    // Ratio 3 (300 held in 100): at most half, 600 / 2.
    let ratio = HostPages {
        compressor: 100,
        compressed: 300,
        ..base
    };
    assert_eq!(host_memory_at(&ratio, PAGE, Some(Pressure::Warning)), empty);
    // Ratio 1.5 (300 held in 200): credit = 600 - 600 * 200 / 300.
    let ratio_low = HostPages {
        compressor: 200,
        compressed: 300,
        ..base
    };
    assert_eq!(
        host_memory_at(&ratio_low, PAGE, Some(Pressure::Warning)),
        HostMemory {
            available: (1250 + 200) * PAGE,
            spare: (1250 + 100) * PAGE,
        }
    );
    // Ratio 1 or below: nothing saved.
    let ratio_one = HostPages {
        compressor: 300,
        compressed: 300,
        ..base
    };
    assert_eq!(compression_savings(&ratio_one), 0);
    // Critical (or unread) pressure: no credit.
    let critical = HostMemory {
        available: 1250 * PAGE,
        spare: 1250 * PAGE,
    };
    assert_eq!(
        host_memory_at(&ratio, PAGE, Some(Pressure::Critical)),
        critical
    );
    assert_eq!(host_memory_at(&ratio, PAGE, None), critical);
    // Counts that do not add up give nothing.
    let broken = HostPages {
        speculative: 2000,
        ..base
    };
    assert_eq!(host_memory_at(&broken, PAGE, None), HostMemory::default());

    assert_eq!(spare_reserve_bytes(24 * G), 2 * G);
    assert_eq!(spare_reserve_bytes(64 * G), 64 * G / 100 * 7);
    assert_eq!(spare_reserve_bytes(256 * G), 7 * G);
    assert_eq!(available_reserve_bytes(16 * G), 16 * G / 10);
    assert_eq!(available_reserve_bytes(64 * G), 2 * G);
    if cfg!(target_os = "macos") {
        let live = query_host_memory(query_pressure()).expect("macOS reports its pages");
        assert!(live.available >= live.spare);
    }
}
