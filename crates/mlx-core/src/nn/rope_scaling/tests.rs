use super::*;

const THETA: f32 = 1e7;
const NATIVE: u32 = 262_144;

/// Splash `rotaryInverseFrequencies` at 9124ff2c built with its `-O3` flags.
const SPLASH_400K: [u32; 32] = [
    0x3f800000, 0x3f1ab32b, 0x3ebaf81a, 0x3e61f836, 0x3e088d77, 0x3da50957, 0x3d47763f, 0x3cf11176,
    0x3c91ad39, 0x3c301052, 0x3bd4ca14, 0x3b80967d, 0x3b1b690d, 0x3abbd3ec, 0x3a6301e2, 0x3a0301b3,
    0x3996df78, 0x392d53b1, 0x38c69572, 0x3862d5f8, 0x38011de6, 0x37926e1e, 0x3725592e, 0x36c7d6bd,
    0x36718616, 0x3611f3b2, 0x35b0657f, 0x35553105, 0x3500d4b2, 0x349bb43b, 0x343c2eca, 0x33e36fb3,
];
const SPLASH_FACTOR_4: [u32; 32] = [
    0x3f800000, 0x3f1ab32b, 0x3ebaf81a, 0x3e61f836, 0x3e088d77, 0x3da50957, 0x3d47763f, 0x3cf11176,
    0x3c91ad39, 0x3c301052, 0x3bd4ca14, 0x3b80967d, 0x3b1b690d, 0x3abbd3ec, 0x3a6301e2, 0x39f8a364,
    0x3986b53d, 0x3910058b, 0x38975c0e, 0x381b7e07, 0x379ac367, 0x3712f6ed, 0x36812dac, 0x361c1fc4,
    0x35bcb0c1, 0x35640cc6, 0x3509cf4b, 0x34a68e4c, 0x34494c56, 0x33f3499c, 0x3393048e, 0x3331af44,
];

fn plain() -> Vec<f32> {
    (0..32)
        .map(|pair| THETA.powf(-(pair as f32) / 32.0))
        .collect()
}

fn scaling(factor: f64) -> RopeScaling {
    RopeScaling {
        factor,
        original_context_tokens: NATIVE,
    }
}

fn bits(table: &[f32]) -> Vec<u32> {
    table.iter().map(|value| value.to_bits()).collect()
}

fn assert_ramp(table: &[f32], factor: f64) {
    let plain = plain();
    for pair in 0..=14 {
        assert_eq!(table[pair].to_bits(), plain[pair].to_bits(), "pair {pair}");
    }
    for pair in 22..32 {
        assert_eq!(
            table[pair].to_bits(),
            (plain[pair] / factor as f32).to_bits(),
            "pair {pair}"
        );
    }
}

/// Splash `testYarn` at factor 4: the table bit for bit, pairs 0..=14
/// plain, 22..32 divided by the factor. Mutation: the correction range over
/// the 256-dim head instead of the 64 rotated dims → red.
#[test]
fn yarn_table_matches_splash() {
    let factor_4 = rotary_inverse_frequencies(THETA, 32, &scaling(4.0)).unwrap();
    assert_eq!(bits(&factor_4), SPLASH_FACTOR_4);
    assert_ramp(&factor_4, 4.0);
    for (pair, expected) in [
        (15, 4.7423981595784426e-04),
        (18, 7.217386882985011e-05),
        (21, 8.759770025790203e-06),
    ] {
        let value = f64::from(factor_4[pair]);
        assert!(
            ((value - expected) / expected).abs() < 1e-6,
            "pair {pair}: {value}"
        );
    }
    assert_eq!(scaling(4.0).attention_factor(), 1.138_629_4_f32);
    let unscaled = rotary_inverse_frequencies(THETA, 32, &RopeScaling::default()).unwrap();
    assert_eq!(bits(&unscaled), bits(&plain()));
}

/// Gate: 400K gives factor 1.5625, its attention factor and Splash's table.
/// Mutation: `attention_factor` returns 1 → red.
#[test]
fn four_hundred_k_gives_the_formula() {
    let served =
        RopeScaling::served_for("Qwen3.5", NATIVE, RopeScaling::default(), Some(409_600)).unwrap();
    assert_eq!(served.factor, 1.5625);
    assert_eq!(served.original_context_tokens, NATIVE);
    assert_eq!(served.attention_factor().to_bits(), 0x3f85b665);
    assert_eq!(served.served_context_tokens(NATIVE), 409_600);
    let table = rotary_inverse_frequencies(THETA, 32, &served).unwrap();
    assert_eq!(bits(&table), SPLASH_400K);
    assert_ramp(&table, 1.5625);

    let served =
        RopeScaling::served_for("Qwen3.5", NATIVE, RopeScaling::default(), Some(400_000)).unwrap();
    assert_eq!(served.factor, 1.52587890625);
    assert_eq!(served.attention_factor().to_bits(), 0x3f8568ae);
    assert_eq!(served.served_context_tokens(NATIVE), 400_000);
    let million =
        RopeScaling::served_for("Qwen3.5", NATIVE, RopeScaling::default(), Some(1_000_000))
            .unwrap();
    assert_eq!(million.attention_factor().to_bits(), 0x3f91232e);
}

/// Splash `servedFor` and `parse_max_context`'s range. Mutation: drop the
/// already-scaled refusal in `served_for` → red.
#[test]
fn served_for_bounds() {
    let none = RopeScaling::default();
    for request in [None, Some(1), Some(102_400), Some(NATIVE)] {
        let served = RopeScaling::served_for("Qwen3.5", NATIVE, none, request).unwrap();
        assert!(!served.yarn(), "{request:?}");
        assert_eq!(served.served_context_tokens(NATIVE), NATIVE);
    }
    for (tokens, factor) in [
        (262_145, 1.000_003_814_697_265_6),
        (300_000, 1.1444091796875),
        (524_288, 2.0),
        (1_000_000, 1_000_000.0 / 262_144.0),
        (1_010_000, 1_010_000.0 / 262_144.0),
    ] {
        let served = RopeScaling::served_for("Qwen3.5", NATIVE, none, Some(tokens)).unwrap();
        assert_eq!(served.factor, factor, "{tokens}");
        assert_eq!(served.served_context_tokens(NATIVE), tokens);
        let error = RopeScaling::served_for("Qwen3.5", NATIVE, served, Some(tokens)).unwrap_err();
        assert_eq!(error.reason, "Qwen3.5: the target's RoPE is already scaled");
    }
    let error = RopeScaling::served_for("Qwen3.5", NATIVE, none, Some(1_010_001)).unwrap_err();
    assert_eq!(
        error.reason,
        "Qwen3.5 serves at most 1010000 tokens of context"
    );
    let error = RopeScaling::served_for("Tiny", 256, none, Some(257)).unwrap_err();
    assert_eq!(error.reason, "Tiny serves at most 256 tokens of context");
    assert!(
        !RopeScaling::served_for("Tiny", 256, none, Some(256))
            .unwrap()
            .yarn()
    );

    assert_eq!(max_context_option("Qwen3.5", None).unwrap(), None);
    assert_eq!(max_context_option("Qwen3.5", Some(1)).unwrap(), Some(1));
    assert_eq!(
        max_context_option("Qwen3.5", Some(1_010_000)).unwrap(),
        Some(1_010_000)
    );
    for bad in [0, -1, 1_010_001, (1 << 32) + 100] {
        let error = max_context_option("Qwen3.5", Some(bad)).unwrap_err();
        assert_eq!(
            error.reason,
            "Qwen3.5 load: maxContextTokens must be a token count from 1 to 1010000"
        );
    }
}

/// Splash `persistentCacheNamespace` writes the factor at `max_digits10`.
/// Mutation: `{:?}` for the factor in `namespace_line` → red ("2.0").
#[test]
fn namespace_line_takes_seventeen_digits() {
    assert_eq!(RopeScaling::default().namespace_line(), None);
    assert_eq!(
        scaling(1.5625).namespace_line().as_deref(),
        Some("rope yarn 1.5625 262144\n")
    );
    assert_eq!(
        scaling(2.0).namespace_line().as_deref(),
        Some("rope yarn 2 262144\n")
    );
    assert_eq!(
        scaling(524_291.0 / 262_144.0).namespace_line().as_deref(),
        Some("rope yarn 2.0000114440917969 262144\n")
    );
    assert_eq!(format_general(1.1444091796875, 6), "1.14441");
    assert_eq!(format_general(3.814697265625, 6), "3.8147");
    assert_eq!(format_general(1.0000038146972656, 6), "1");
    assert_eq!(format_general(0.0001234, 3), "0.000123");
    assert_eq!(format_general(0.00001234, 3), "1.23e-05");
    assert_eq!(format_general(1234567.0, 3), "1.23e+06");
    assert_eq!(format_general(9.9999995, 6), "10");
}
