//! Arbitrary-precision decimal helpers matching Python's `decimal.Decimal` semantics.
//!
//! Supports scientific notation up to at least `e±308`, precision beyond 310 digits,
//! exact `ROUND_HALF_UP` rounding, base-10 scaling (`scaleb`), plain fixed-point
//! formatting, and JSON number conversion.

use bigdecimal::{
    BigDecimal, RoundingMode, Zero,
    num_bigint::{BigInt, Sign},
    num_traits::ToPrimitive,
};
use serde_json::{Number, Value};
use std::str::FromStr;

/// Maximum exponent magnitude accepted for decimal parsing/scaling to prevent
/// pathological allocations when expanding to plain strings.
const MAX_SAFE_EXPONENT: i64 = 10_000;

/// Maximum precision allowed when rounding to a requested decimal scale.
const MAX_SAFE_ROUND_DIGITS: i64 = 10_000;

/// Parse a decimal string (including leading `+`, `-`, and scientific notation `e`/`E`).
///
/// Rejects `NaN`, `Inf`, `Infinity`, empty strings, and non-decimal syntax.
pub fn parse_decimal(input: &str) -> Option<BigDecimal> {
    let trimmed = input.trim();
    if trimmed.is_empty() || !is_valid_decimal_literal(trimmed) {
        return None;
    }
    let parsed = BigDecimal::from_str(trimmed).ok()?;
    let (_, scale) = parsed.as_bigint_and_exponent();
    if scale.abs() > MAX_SAFE_EXPONENT {
        return None;
    }
    Some(normalize_zero(parsed))
}

/// Convert a finite `f64` into an exact decimal using its shortest decimal string
/// representation, matching Python's `Decimal(str(value))`.
pub fn from_f64(value: f64) -> Option<BigDecimal> {
    if !value.is_finite() {
        return None;
    }
    if value == 0.0 {
        return Some(BigDecimal::from(0));
    }
    parse_decimal(&value.to_string())
}

/// Parse a JSON number or decimal string into a `BigDecimal`.
///
/// Rejects JSON booleans, nulls, arrays, objects, and non-finite/invalid strings.
pub fn from_json_value(value: &Value) -> Option<BigDecimal> {
    match value {
        Value::Number(num) => parse_decimal(&num.to_string()),
        Value::String(s) => parse_decimal(s),
        _ => None,
    }
}

/// Multiply `value` by `10^exp` (exact port of Python `Decimal.scaleb(exp)`).
pub fn scaleb(value: &BigDecimal, exp: i64) -> BigDecimal {
    let (digits, scale) = value.as_bigint_and_exponent();
    if digits.is_zero() {
        let new_scale = scale.saturating_sub(exp);
        return BigDecimal::new(BigInt::ZERO, new_scale);
    }
    let new_scale = scale.saturating_sub(exp);
    BigDecimal::new(digits, new_scale)
}

/// Return the adjusted exponent of a non-zero decimal, matching Python `Decimal.adjusted()`.
///
/// For `1.23e4` (`12300`), `adjusted() == 4`. For `0.005`, `adjusted() == -3`. For `0`, returns `0`.
pub fn adjusted_exponent(value: &BigDecimal) -> i64 {
    if value.is_zero() {
        return 0;
    }
    let digits = value.digits() as i64;
    let scale = value.fractional_digit_count();
    digits - 1 - scale
}

/// Round `value` to `decimals` digits after the decimal point using exact `ROUND_HALF_UP`
/// (ties away from zero, matching Python `decimal.ROUND_HALF_UP`).
pub fn round_half_up(value: &BigDecimal, decimals: i64) -> BigDecimal {
    if value.is_zero() {
        return BigDecimal::new(BigInt::ZERO, decimals);
    }
    let rounded = value.with_scale_round(decimals, RoundingMode::HalfUp);
    normalize_zero(rounded)
}

/// Checked variant of [`round_half_up`] that rejects numbers whose representation at
/// `decimals` places would exceed [`MAX_SAFE_ROUND_DIGITS`].
pub fn round_half_up_checked(value: &BigDecimal, decimals: i64) -> Option<BigDecimal> {
    if value.is_zero() {
        return Some(BigDecimal::new(BigInt::ZERO, decimals));
    }
    let needed = adjusted_exponent(value).saturating_add(decimals);
    if needed > MAX_SAFE_ROUND_DIGITS || decimals.abs() > MAX_SAFE_ROUND_DIGITS {
        return None;
    }
    Some(round_half_up(value, decimals))
}

/// Return `true` iff `value` has no non-zero fractional part.
pub fn is_integral(value: &BigDecimal) -> bool {
    value.is_integer()
}

/// Format `value` as a plain fixed-point string without scientific notation,
/// matching Python `format(d, "f")`.
///
/// Normalizes negative zero to non-negative zero.
pub fn to_plain_string(value: &BigDecimal) -> String {
    if value.is_zero() {
        let scale = value.fractional_digit_count();
        if scale > 0 {
            let mut out = String::with_capacity(2 + scale as usize);
            out.push_str("0.");
            for _ in 0..scale {
                out.push('0');
            }
            return out;
        }
        return "0".to_string();
    }
    value.to_plain_string()
}

/// Format `value` as a stripped plain number string matching `model.py::plain_number`.
///
/// Strips trailing fractional zeros and trailing decimal point (e.g. `12.50 -> "12.5"`,
/// `12.00 -> "12"`, `-0.0 -> "0"`).
pub fn to_stripped_plain_string(value: &BigDecimal) -> String {
    let text = to_plain_string(value);
    if text.contains('.') {
        let trimmed = text.trim_end_matches('0').trim_end_matches('.');
        if trimmed == "-0" || trimmed.is_empty() {
            "0".to_string()
        } else {
            trimmed.to_string()
        }
    } else if text == "-0" {
        "0".to_string()
    } else {
        text
    }
}

/// Canonical normalized decimal string representation matching Python `str(d.normalize())`.
pub fn normalize_decimal_str(value: &BigDecimal) -> String {
    if value.is_zero() {
        return "0".to_string();
    }
    let norm = value.normalized();
    let (digits, scale) = norm.as_bigint_and_exponent();
    if scale <= 0 {
        let plain = norm.to_plain_string();
        if plain.len() <= 32 {
            return plain;
        }
    }
    let _ = digits;
    norm.to_string()
}

/// Return the normalized significant decimal digits of `|value|` (without leading/trailing
/// zeros), matching `abs(d).normalize().as_tuple().digits` in Python.
pub fn significant_digits(value: &BigDecimal) -> Vec<u8> {
    if value.is_zero() {
        return vec![0];
    }
    let norm = value.abs().normalized();
    let (bigint, _) = norm.into_bigint_and_exponent();
    let s = bigint.to_str_radix(10);
    s.bytes()
        .filter(|b| b.is_ascii_digit())
        .map(|b| b - b'0')
        .collect()
}

/// Convert `value` into a `serde_json::Value::Number`.
///
/// Integral values that fit in `i64` or `u64` are emitted as exact JSON integers;
/// otherwise finite `f64` JSON numbers are produced. Returns `None` if `value`
/// overflows `f64` (non-finite).
pub fn to_json_number(value: &BigDecimal) -> Option<Value> {
    if value.is_zero() {
        return Some(Value::Number(Number::from(0)));
    }
    if value.is_integer() {
        let trunc = value.with_scale(0);
        if let Some(i) = trunc.to_i64() {
            return Some(Value::Number(Number::from(i)));
        }
        if let Some(u) = trunc.to_u64() {
            return Some(Value::Number(Number::from(u)));
        }
    }
    let f = value.to_f64()?;
    if !f.is_finite() {
        return None;
    }
    let clean = if f == 0.0 { 0.0 } else { f };
    Number::from_f64(clean).map(Value::Number)
}

/// Convert `value` to JSON number following Clairic `decode_number` rules:
/// integral if `force_integer || (is_integral && |value| < 2^53)`, else finite `f64`.
pub fn to_decoded_json_number(value: &BigDecimal, force_integer: bool) -> Option<Value> {
    if value.is_zero() {
        return Some(Value::Number(Number::from(0)));
    }
    if force_integer {
        let trunc = value.with_scale(0);
        if let Some(i) = trunc.to_i64() {
            return Some(Value::Number(Number::from(i)));
        }
        if let Some(u) = trunc.to_u64() {
            return Some(Value::Number(Number::from(u)));
        }
    } else if value.is_integer() {
        let trunc = value.with_scale(0);
        if let Some(i) = trunc.to_i64()
            && i.unsigned_abs() < (1u64 << 53)
        {
            return Some(Value::Number(Number::from(i)));
        }
    }
    let f = value.to_f64()?;
    if !f.is_finite() {
        return None;
    }
    let clean = if f == 0.0 { 0.0 } else { f };
    Number::from_f64(clean).map(Value::Number)
}

fn normalize_zero(d: BigDecimal) -> BigDecimal {
    let (bigint, scale) = d.into_bigint_and_exponent();
    if bigint.sign() == Sign::NoSign {
        BigDecimal::new(BigInt::ZERO, scale)
    } else {
        BigDecimal::new(bigint, scale)
    }
}

fn is_valid_decimal_literal(s: &str) -> bool {
    let bytes = s.as_bytes();
    let mut i = 0;
    let n = bytes.len();
    if i < n && (bytes[i] == b'+' || bytes[i] == b'-') {
        i += 1;
    }
    let mut saw_digit = false;
    while i < n && bytes[i].is_ascii_digit() {
        saw_digit = true;
        i += 1;
    }
    if i < n && bytes[i] == b'.' {
        i += 1;
        while i < n && bytes[i].is_ascii_digit() {
            saw_digit = true;
            i += 1;
        }
    }
    if !saw_digit {
        return false;
    }
    if i < n && (bytes[i] == b'e' || bytes[i] == b'E') {
        i += 1;
        if i < n && (bytes[i] == b'+' || bytes[i] == b'-') {
            i += 1;
        }
        let mut saw_exp = false;
        while i < n && bytes[i].is_ascii_digit() {
            saw_exp = true;
            i += 1;
        }
        if !saw_exp {
            return false;
        }
    }
    i == n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_decimal_handles_scientific_and_rejects_nonfinite() {
        let huge = parse_decimal("1.7976931348623157e+308").unwrap();
        assert_eq!(adjusted_exponent(&huge), 308);
        assert!(to_plain_string(&huge).starts_with("17976931348623157"));
        assert_eq!(to_plain_string(&huge).len(), 309);

        let tiny = parse_decimal("+2.2250738585072014e-308").unwrap();
        assert_eq!(adjusted_exponent(&tiny), -308);
        assert!(to_plain_string(&tiny).starts_with("0.00000"));

        assert_eq!(parse_decimal("0"), Some(BigDecimal::from(0)));
        assert_eq!(parse_decimal("  -42.50  "), parse_decimal("-42.50"));
        assert_eq!(parse_decimal(".25"), parse_decimal("0.25"));
        assert_eq!(parse_decimal("5."), parse_decimal("5"));

        assert!(parse_decimal("").is_none());
        assert!(parse_decimal("NaN").is_none());
        assert!(parse_decimal("inf").is_none());
        assert!(parse_decimal("-Infinity").is_none());
        assert!(parse_decimal("1_000").is_none());
        assert!(parse_decimal("1e").is_none());
    }

    #[test]
    fn round_half_up_ties_and_negative_zero() {
        assert_eq!(
            to_plain_string(&round_half_up(&parse_decimal("2.5").unwrap(), 0)),
            "3"
        );
        assert_eq!(
            to_plain_string(&round_half_up(&parse_decimal("3.5").unwrap(), 0)),
            "4"
        );
        assert_eq!(
            to_plain_string(&round_half_up(&parse_decimal("-2.5").unwrap(), 0)),
            "-3"
        );
        assert_eq!(
            to_plain_string(&round_half_up(&parse_decimal("1.005").unwrap(), 2)),
            "1.01"
        );
        assert_eq!(
            to_plain_string(&round_half_up(&parse_decimal("-0.004").unwrap(), 2)),
            "0.00"
        );
        assert_eq!(
            to_stripped_plain_string(&round_half_up(&parse_decimal("-0.004").unwrap(), 0)),
            "0"
        );
        assert_eq!(
            to_stripped_plain_string(&round_half_up(&parse_decimal("1e308").unwrap(), 2)),
            to_plain_string(&parse_decimal("1e308").unwrap())
        );
    }

    #[test]
    fn scaleb_and_plain_string_and_json_number() {
        let d = parse_decimal("12.34").unwrap();
        assert_eq!(to_plain_string(&scaleb(&d, 3)), "12340");
        assert_eq!(to_plain_string(&scaleb(&d, -2)), "0.1234");
        assert_eq!(
            to_stripped_plain_string(&parse_decimal("12.3400").unwrap()),
            "12.34"
        );
        assert_eq!(
            to_stripped_plain_string(&parse_decimal("100.00").unwrap()),
            "100"
        );
        assert_eq!(
            to_stripped_plain_string(&parse_decimal("-0.0").unwrap()),
            "0"
        );

        assert!(is_integral(&parse_decimal("42.000").unwrap()));
        assert!(!is_integral(&parse_decimal("42.001").unwrap()));

        assert_eq!(
            to_json_number(&parse_decimal("42.00").unwrap()),
            Some(serde_json::json!(42))
        );
        assert_eq!(
            to_json_number(&parse_decimal("42.25").unwrap()),
            Some(serde_json::json!(42.25))
        );
        assert_eq!(
            to_json_number(&parse_decimal("-0.0").unwrap()),
            Some(serde_json::json!(0))
        );
        assert_eq!(
            significant_digits(&parse_decimal("1200.00").unwrap()),
            vec![1, 2]
        );
        assert_eq!(
            significant_digits(&parse_decimal("0.012").unwrap()),
            vec![1, 2]
        );
    }

    #[test]
    fn normalize_decimal_str_cases() {
        assert_eq!(normalize_decimal_str(&parse_decimal("-0.00").unwrap()), "0");
        assert_eq!(
            normalize_decimal_str(&parse_decimal("12.500").unwrap()),
            "12.5"
        );
        assert_eq!(
            normalize_decimal_str(&parse_decimal("100.00").unwrap()),
            "100"
        );
    }
}
