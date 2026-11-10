//! YaRN, the RoPE scaling that stretches a Qwen target's native window
//! (Splash `RopeScaling`, `rotaryInverseFrequencies`, `servedFor`).

use napi::bindgen_prelude::*;

/// Splash `QwenTargetDimensions::yarnContextTokens` of the families YaRN
/// stretches; Splash `parse_max_context` takes no more.
pub(crate) const MAX_YARN_CONTEXT_TOKENS: u32 = 1_010_000;

/// Splash `nativeContextTokens` of every layout with a YaRN window.
const YARN_NATIVE_CONTEXT_TOKENS: u32 = 262_144;

/// Splash `yarnContextTokens` for a target of `native` tokens: the families
/// Splash stretches all have a native window of 262,144 (`sizeDifference`
/// refuses any other), so another native has no YaRN window.
pub(crate) fn yarn_context_tokens(native: u32) -> u32 {
    if native == YARN_NATIVE_CONTEXT_TOKENS {
        MAX_YARN_CONTEXT_TOKENS
    } else {
        0
    }
}

/// Splash `RopeScaling`: factor 1 and original 0 without YaRN.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeScaling {
    pub factor: f64,
    pub original_context_tokens: u32,
}

impl Default for RopeScaling {
    fn default() -> Self {
        Self {
            factor: 1.0,
            original_context_tokens: 0,
        }
    }
}

/// A config's `rope_scaling` never crosses NAPI: a load sets it from
/// `maxContextTokens`, and the field is hidden from TypeScript.
impl ToNapiValue for RopeScaling {
    unsafe fn to_napi_value(env: napi::sys::napi_env, _: Self) -> Result<napi::sys::napi_value> {
        unsafe { <() as ToNapiValue>::to_napi_value(env, ()) }
    }
}

impl FromNapiValue for RopeScaling {
    unsafe fn from_napi_value(_: napi::sys::napi_env, _: napi::sys::napi_value) -> Result<Self> {
        Err(Error::from_reason(
            "rope_scaling comes from the maxContextTokens load option",
        ))
    }
}

impl RopeScaling {
    pub(crate) fn yarn(&self) -> bool {
        self.factor != 1.0
    }

    pub(crate) fn attention_factor(&self) -> f32 {
        if self.yarn() {
            0.1f64.mul_add(self.factor.ln(), 1.0) as f32
        } else {
            1.0
        }
    }

    /// Splash `QwenHybridLayout::servedContextTokens`.
    pub(crate) fn served_context_tokens(&self, native: u32) -> u32 {
        if self.yarn() {
            (f64::from(native) * self.factor).round() as u32
        } else {
            native
        }
    }

    /// Splash `servedFor` of a target whose native window is `native`.
    pub(crate) fn served_for(
        label: &str,
        native: u32,
        already: RopeScaling,
        max_context: Option<u32>,
    ) -> Result<Self> {
        if already.yarn() {
            return Err(Error::from_reason(format!(
                "{label}: the target's RoPE is already scaled"
            )));
        }
        let Some(tokens) = max_context.filter(|&tokens| tokens > native) else {
            return Ok(Self::default());
        };
        let yarn = yarn_context_tokens(native);
        if tokens > yarn {
            return Err(Error::from_reason(format!(
                "{label} serves at most {} tokens of context",
                native.max(yarn)
            )));
        }
        Ok(Self {
            factor: f64::from(tokens) / f64::from(native),
            original_context_tokens: native,
        })
    }

    /// Splash `persistentCacheNamespace`'s YaRN line, only with YaRN.
    pub(crate) fn namespace_line(&self) -> Option<String> {
        self.yarn().then(|| {
            format!(
                "rope yarn {} {}\n",
                format_general(self.factor, 17),
                self.original_context_tokens
            )
        })
    }

    /// Splash `RuntimeResources`' startup line of a YaRN target.
    pub(crate) fn startup_line(&self, served: u32) -> String {
        let native = self.original_context_tokens;
        format!(
            "YaRN is on: the target's RoPE is scaled by {} for {served} tokens of context, past \
             the native window of {native}. Static YaRN changes answers at every length, so pass \
             a --max-context above {native} only when a request needs it.",
            format_general(self.factor, 6)
        )
    }
}

/// Splash `rotaryInverseFrequencies` over `pairs` rotated pairs of base
/// `theta`. The blend is one fused multiply-add, as Splash's `-O3` build
/// contracts it.
pub(crate) fn rotary_inverse_frequencies(
    theta: f32,
    pairs: u32,
    scaling: &RopeScaling,
) -> Result<Vec<f32>> {
    if pairs == 0 {
        return Err(Error::from_reason("invalid rotary inverse frequency table"));
    }
    let mut frequencies: Vec<f32> = (0..pairs)
        .map(|pair| theta.powf(-(pair as f32) / pairs as f32))
        .collect();
    if !scaling.yarn() {
        return Ok(frequencies);
    }
    if scaling.factor.partial_cmp(&1.0) != Some(std::cmp::Ordering::Greater)
        || scaling.original_context_tokens == 0
    {
        return Err(Error::from_reason("invalid YaRN scaling"));
    }
    let dimensions = 2.0 * f64::from(pairs);
    let original = f64::from(scaling.original_context_tokens);
    let correction = |rotations: f64| {
        dimensions * (original / (rotations * 2.0 * std::f64::consts::PI)).ln()
            / (2.0 * f64::from(theta).ln())
    };
    let low = correction(32.0).floor().max(0.0);
    let mut high = correction(1.0).ceil().min(dimensions - 1.0);
    if low == high {
        high += 0.001;
    }
    let factor = scaling.factor as f32;
    for (pair, frequency) in frequencies.iter_mut().enumerate() {
        let ramp = (((pair as f64) - low) / (high - low)) as f32;
        let ramp = ramp.clamp(0.0, 1.0);
        let extrapolation = *frequency;
        let interpolation = extrapolation / factor;
        *frequency = interpolation.mul_add(ramp, extrapolation * (1.0 - ramp));
    }
    Ok(frequencies)
}

/// Splash `validateTextConfig`'s rope type: `rope_type` when present, else
/// the older `type`, must be `default`. An absent type loads.
pub(crate) fn refuse_scaled_rope(label: &str, rope: Option<&serde_json::Value>) -> Result<()> {
    let Some(rope) = rope.and_then(|rope| rope.as_object()) else {
        return Ok(());
    };
    let key = if rope.contains_key("rope_type") {
        "rope_type"
    } else {
        "type"
    };
    match rope.get(key) {
        None => Ok(()),
        Some(value) if value.as_str() == Some("default") => Ok(()),
        Some(value) => Err(Error::from_reason(format!(
            "{label} config: rope_parameters.{key} {value}: the target's RoPE is already scaled"
        ))),
    }
}

/// The `maxContextTokens` load option (Splash `parse_max_context`'s range).
pub(crate) fn max_context_option(label: &str, value: Option<i64>) -> Result<Option<u32>> {
    match value {
        None => Ok(None),
        Some(tokens) if (1..=i64::from(MAX_YARN_CONTEXT_TOKENS)).contains(&tokens) => {
            Ok(Some(tokens as u32))
        }
        Some(_) => Err(Error::from_reason(format!(
            "{label} load: maxContextTokens must be a token count from 1 to \
             {MAX_YARN_CONTEXT_TOKENS}"
        ))),
    }
}

/// C's `%.{precision}g`: `precision` significant digits, no trailing zeros.
pub(crate) fn format_general(value: f64, precision: usize) -> String {
    let precision = precision.max(1);
    if value == 0.0 || !value.is_finite() {
        return format!("{value}");
    }
    let scientific = format!("{:.*e}", precision - 1, value);
    let Some((mantissa, exponent)) = scientific.split_once('e') else {
        return scientific;
    };
    let Ok(exponent) = exponent.parse::<i32>() else {
        return scientific;
    };
    if exponent < -4 || exponent >= precision as i32 {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!(
            "{}e{sign}{:02}",
            trim_fraction(mantissa),
            exponent.unsigned_abs()
        )
    } else {
        let decimals = (precision as i32 - 1 - exponent) as usize;
        trim_fraction(&format!("{value:.decimals$}")).to_string()
    }
}

fn trim_fraction(text: &str) -> &str {
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.')
    } else {
        text
    }
}

#[cfg(test)]
mod tests;
