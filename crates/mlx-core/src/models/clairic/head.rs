//! Inference-only port of Cloudflare's `ClairicHead` / MLX `TypedHead` (lexical readout).
//!
//! Matches upstream `clairic_runtime/mlx_engine.py::TypedHead` and
//! `clairic_runtime/model.py::ClairicHead` (`forward_lexical` and `answer_slot_states`).

use crate::{
    array::{DType, MxArray, scaled_dot_product_attention},
    nn::{Activations, LayerNorm},
};
use napi::{Either, Error, Result};
use serde::Deserialize;
use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

/// Compiled-graph id for the fused `gelu_exact` used by Clairic head blocks.
/// Pure elementwise function with no captured weights, so a single process-wide
/// id is safe and nothing needs to be erased on drop.
const GELU_EXACT_FN_ID: u64 = 0xC1A1_21C0_0000_0001;

pub const MAX_SPAN: i64 = 64;
pub const TRIM_CLASSES: i64 = 8;
pub const BINS: i64 = 256;
pub const POINTER_WIDTH: i64 = 256;
pub const NUMBER_DIGITS: i64 = 12;
pub const MIN_EXPONENT: i64 = -12;
pub const MAX_EXPONENT: i64 = 15;
pub const EXPONENT_CLASSES: i64 = MAX_EXPONENT - MIN_EXPONENT + 2; // 29
pub const YEAR_DIGITS: i64 = 4;
pub const YEAR_LOW: i32 = 1800;
pub const YEAR_HIGH: i32 = 2199;
pub const YEAR_OUTSIDE: i64 = (YEAR_HIGH - YEAR_LOW + 1) as i64; // 400
pub const YEAR_CLASSES: i64 = YEAR_OUTSIDE + 1; // 401
pub const LEXICAL_DIGITS: i64 = 4;
pub const ESTIMATE_SPREAD_LIMIT: f64 = 2.0;
pub const NUMBER_WIDTH: i64 = 2 + EXPONENT_CLASSES + NUMBER_DIGITS * 10; // 151
pub const GROUP_DIGITS: i64 = 3;
pub const GROUP_CLASSES: i64 = 1000;
pub const GROUP_LOW: i64 = -4; // MIN_EXPONENT / GROUP_DIGITS
pub const GROUP_HIGH: i64 = 5; // MAX_EXPONENT / GROUP_DIGITS
pub const NUMBER_GROUPS: i64 = GROUP_HIGH - GROUP_LOW + 1; // 10
pub const GROUP_RANK: i64 = 256;
pub const DATE_WIDTH: i64 = YEAR_DIGITS * 10 + YEAR_CLASSES + 12 + 31; // 484
pub const TIME_WIDTH: i64 = 24 + 60 + 60; // 144

pub const DIGIT_SLOTS: i64 = NUMBER_DIGITS + YEAR_DIGITS + 2 + 2 + 2 + 2; // 24
pub const YEAR_SLOT: i64 = NUMBER_DIGITS; // 12
pub const DAY_SLOT: i64 = NUMBER_DIGITS + YEAR_DIGITS; // 16
pub const HOUR_SLOT: i64 = NUMBER_DIGITS + YEAR_DIGITS + 2; // 18
pub const MINUTE_SLOT: i64 = NUMBER_DIGITS + YEAR_DIGITS + 4; // 20
pub const SECOND_SLOT: i64 = NUMBER_DIGITS + YEAR_DIGITS + 6; // 22

pub const ROUTE_KINDS: [&str; 8] = [
    "text",
    "tags",
    "number",
    "date",
    "time",
    "estimate",
    "number_list",
    "date_list",
];
pub const LIST_KINDS: [&str; 2] = ["number_list", "date_list"];
pub const ROUTE_KIND_COUNT: i64 = 8;
pub const LIST_KIND_COUNT: i64 = 2;
pub const KIND_EMBEDDING_ROWS: i64 = ROUTE_KIND_COUNT + LIST_KIND_COUNT; // 10

/// Configuration deserialized from `clair_head_config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct ClairicHeadConfig {
    pub hidden_size: i64,
    pub bins: i64,
    pub pointer_width: i64,
    pub readout: String,
    pub route_width: i64,
    pub route_layers: usize,
    pub route_heads: i64,
    pub slot_rank: i64,
    #[serde(default)]
    pub estimate_lexical: bool,
    #[serde(default)]
    pub span_trim: bool,
    #[serde(default)]
    pub loop_feedback: bool,
    #[serde(default)]
    pub answer_slots: bool,
    #[serde(default)]
    pub answer_slot_routing: bool,
    #[serde(default)]
    pub number_groups: bool,
    #[serde(default)]
    pub number_pointer: bool,
}

impl ClairicHeadConfig {
    /// Load `clair_head_config.json` from a checkpoint directory.
    pub fn load(dir: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(dir.join("clair_head_config.json")).map_err(|e| {
            Error::from_reason(format!("Failed to read clair_head_config.json: {e}"))
        })?;
        serde_json::from_str(&raw)
            .map_err(|e| Error::from_reason(format!("Invalid clair_head_config.json: {e}")))
    }

    /// Validate that the configuration matches the supported lexical typed head contract
    /// (`mlx_engine.py::TypedHead.REQUIRED`) and expected backbone hidden size.
    pub fn validate(&self, expected_hidden_size: i64) -> Result<()> {
        if self.readout != "lexical" {
            return Err(Error::from_reason(
                "unsupported typed head: readout must be 'lexical'",
            ));
        }
        for (name, flag) in [
            ("estimate_lexical", self.estimate_lexical),
            ("span_trim", self.span_trim),
            ("answer_slots", self.answer_slots),
            ("answer_slot_routing", self.answer_slot_routing),
            ("number_groups", self.number_groups),
            ("number_pointer", self.number_pointer),
        ] {
            if !flag {
                return Err(Error::from_reason(format!(
                    "unsupported typed head: {name} must be true"
                )));
            }
        }
        if self.loop_feedback {
            return Err(Error::from_reason(
                "unsupported typed head: loop_feedback is not supported",
            ));
        }
        if self.hidden_size != expected_hidden_size
            || self.hidden_size <= 0
            || self.bins <= 0
            || self.pointer_width <= 0
            || self.route_width <= 0
            || self.route_heads <= 0
            || self.route_width % self.route_heads != 0
            || self.route_layers == 0
            || self.route_layers > 32
            || self.slot_rank <= 0
        {
            return Err(Error::from_reason(
                "Invalid Clairic head configuration or backbone hidden size",
            ));
        }
        Ok(())
    }
}

/// Index views describing which query positions and list-item slots belong to each readout group.
#[derive(Debug, Clone, Default)]
pub struct PredictionViews {
    /// Token positions in `hidden` for each field query (`[Q]`).
    pub query_positions: Vec<i64>,
    /// Row indices into `query_positions` for `text` span fields.
    pub text_rows: Vec<i64>,
    /// Row indices into `query_positions` for `tags` (`text_list`) BIO fields.
    pub tags_rows: Vec<i64>,
    /// Row indices into `query_positions` for `number` fields.
    pub number_rows: Vec<i64>,
    /// Row indices into `query_positions` for `date` fields.
    pub date_rows: Vec<i64>,
    /// Row indices into `query_positions` for `time` fields.
    pub time_rows: Vec<i64>,
    /// Row indices into `query_positions` for `estimate` fields.
    pub estimate_rows: Vec<i64>,
    /// Row indices into `query_positions` for `number_list` fields.
    pub number_list_rows: Vec<i64>,
    /// Row indices into `query_positions` for `date_list` fields.
    pub date_list_rows: Vec<i64>,
    /// Token positions in `hidden` for `number_list` item cues (`[number_list_rows.len(), list_slots]`).
    pub number_list_items: Vec<Vec<i64>>,
    /// Token positions in `hidden` for `date_list` item cues (`[date_list_rows.len(), list_slots]`).
    pub date_list_items: Vec<Vec<i64>>,
}

/// Outputs produced by [`ClairicHead::forward`], matching the keys of `TypedHead.forward`.
pub struct HeadOutputs {
    /// Null logits for every field query: `[Q]`.
    pub null: MxArray,
    /// Boundary trim logits for input tokens: `[T, 2, 8]`.
    pub trim: Option<MxArray>,
    /// Span pointer scores for text fields: `[N_text, T, 64]`.
    pub text: Option<MxArray>,
    /// BIO tag scores for `text_list` fields: `[N_tags, T, 3]`.
    pub tags: Option<MxArray>,
    /// Lexical number logits: `[N_num, 151]`.
    pub number: Option<MxArray>,
    /// 3-digit group logits for number fields: `[N_num, 10, 1000]`.
    pub number_groups: Option<MxArray>,
    /// Span copy pointer scores for number fields: `[N_num, T, 64]`.
    pub number_copy: Option<MxArray>,
    /// Copy gate logits for number fields: `[N_num]`.
    pub number_copy_gate: Option<MxArray>,
    /// Lexical date logits: `[N_date, 484]`.
    pub date: Option<MxArray>,
    /// Lexical time logits: `[N_time, 144]`.
    pub time: Option<MxArray>,
    /// Estimate histogram bin logits: `[N_est, bins]`.
    pub estimate: Option<MxArray>,
    /// Estimate within-bin offset logits: `[N_est, bins]`.
    pub estimate_offset: Option<MxArray>,
    /// Estimate below/above overflow logits: `[N_est, 2]`.
    pub estimate_overflow: Option<MxArray>,
    /// Untempered lexical number logits for estimates: `[N_est, 151]`.
    pub estimate_number_raw: Option<MxArray>,
    /// Spread-tempered lexical number logits for estimates: `[N_est, 151]`.
    pub estimate_number: Option<MxArray>,
    /// Lexical vs histogram blend gate logits for estimates: `[N_est]`.
    pub estimate_gate: Option<MxArray>,
    /// Lexical number logits for `number_list` items: `[N_nl, S_nl, 151]`.
    pub number_list: Option<MxArray>,
    /// End-of-list logits for `number_list` items: `[N_nl, S_nl]`.
    pub number_list_end: Option<MxArray>,
    /// 3-digit group logits for `number_list` items: `[N_nl, S_nl, 10, 1000]`.
    pub number_list_groups: Option<MxArray>,
    /// Lexical date logits for `date_list` items: `[N_dl, S_dl, 484]`.
    pub date_list: Option<MxArray>,
    /// End-of-list logits for `date_list` items: `[N_dl, S_dl]`.
    pub date_list_end: Option<MxArray>,
}

/// Loaded Clairic lexical typed head evaluated in FP32 on MLX.
pub struct ClairicHead {
    config: ClairicHeadConfig,
    weights: HashMap<String, MxArray>,
    /// Weight-derived tensors and static index tables baked once at load time.
    derived: HashMap<String, MxArray>,
}

struct TypedOutputs {
    number: MxArray,
    date: MxArray,
    time: MxArray,
}

impl ClairicHead {
    /// Load `clair_head_config.json` and `clair_head.safetensors` from `dir`.
    pub fn load(dir: &Path, expected_hidden_size: i64, round_bf16: bool) -> Result<Self> {
        let config = ClairicHeadConfig::load(dir)?;
        let weights =
            crate::utils::safetensors::load_safetensors_lazy(dir.join("clair_head.safetensors"))?;
        Self::from_weights(config, weights, expected_hidden_size, round_bf16)
    }

    /// Construct and validate a [`ClairicHead`] from an explicit config and tensor map.
    ///
    /// When `round_bf16` is `true`, mirrors upstream `TypedHead.__init__`: every float
    /// weight except `length_bias` is rounded through `BFloat16 -> Float32`, while
    /// `length_bias` is widened directly to `Float32`.
    pub fn from_weights(
        config: ClairicHeadConfig,
        weights: HashMap<String, MxArray>,
        expected_hidden_size: i64,
        round_bf16: bool,
    ) -> Result<Self> {
        config.validate(expected_hidden_size)?;
        let shapes = Self::expected_shapes(&config);
        let expected: HashSet<&str> = shapes.iter().map(|(k, _)| k.as_str()).collect();

        for (name, shape) in &shapes {
            let weight = weights.get(name).ok_or_else(|| {
                Error::from_reason(format!("Missing Clairic head tensor: {name}"))
            })?;
            let actual_shape = weight.shape()?;
            let dtype = weight.dtype()?;
            if actual_shape.as_ref() != shape.as_slice()
                || !matches!(dtype, DType::Float32 | DType::Float16 | DType::BFloat16)
            {
                return Err(Error::from_reason(format!(
                    "Invalid Clairic head tensor shape or dtype for {name}: got shape {:?}, dtype {:?}, expected {:?}",
                    actual_shape.as_ref(),
                    dtype,
                    shape
                )));
            }
        }
        if let Some(extra) = weights.keys().find(|k| !expected.contains(k.as_str())) {
            return Err(Error::from_reason(format!(
                "Unexpected Clairic head tensor: {extra}"
            )));
        }

        let mut converted: HashMap<String, MxArray> = HashMap::with_capacity(weights.len());
        for (name, tensor) in weights {
            let widened = if round_bf16 && name != "length_bias" {
                tensor.astype(DType::BFloat16)?.astype(DType::Float32)?
            } else {
                tensor.astype(DType::Float32)?
            };
            converted.insert(name, widened);
        }
        {
            let refs: Vec<&MxArray> = converted.values().collect();
            MxArray::eval_arrays(&refs)?;
        }

        let mut head = Self {
            config,
            weights: converted,
            derived: HashMap::new(),
        };
        head.bake_derived()?;
        Ok(head)
    }

    /// Return the parsed head configuration.
    pub fn config(&self) -> &ClairicHeadConfig {
        &self.config
    }

    /// Total byte size of loaded head weights and baked derived tensors.
    pub fn nbytes(&self) -> u64 {
        self.weights
            .values()
            .chain(self.derived.values())
            .map(|a| a.nbytes() as u64)
            .sum()
    }

    fn expected_shapes(c: &ClairicHeadConfig) -> Vec<(String, Vec<i64>)> {
        let mut out: Vec<(String, Vec<i64>)> = Vec::new();
        fn norm(out: &mut Vec<(String, Vec<i64>)>, prefix: &str, width: i64) {
            out.push((format!("{prefix}.weight"), vec![width]));
            out.push((format!("{prefix}.bias"), vec![width]));
        }
        fn linear(
            out: &mut Vec<(String, Vec<i64>)>,
            prefix: &str,
            input: i64,
            output: i64,
            bias: bool,
        ) {
            out.push((format!("{prefix}.weight"), vec![output, input]));
            if bias {
                out.push((format!("{prefix}.bias"), vec![output]));
            }
        }
        fn attention(out: &mut Vec<(String, Vec<i64>)>, prefix: &str, width: i64) {
            out.push((format!("{prefix}.in_proj_weight"), vec![3 * width, width]));
            out.push((format!("{prefix}.in_proj_bias"), vec![3 * width]));
            linear(out, &format!("{prefix}.out_proj"), width, width, true);
        }

        let h = c.hidden_size;
        let w = c.route_width;
        let p = c.pointer_width;
        let b = c.bins;
        let r = c.slot_rank;

        out.push(("slot_route_scale".into(), vec![ROUTE_KIND_COUNT]));
        norm(&mut out, "norm", h);
        norm(&mut out, "token_norm", h);
        linear(&mut out, "null", h, 1, true);
        linear(&mut out, "span_query", h, 2 * p, true);
        linear(&mut out, "span_key", h, 2 * p, true);
        out.push(("length_bias".into(), vec![MAX_SPAN]));
        linear(&mut out, "tag_query", h, 2 * p, true);
        linear(&mut out, "tag_key", h, p, true);
        out.push(("tag_bias".into(), vec![2]));
        linear(&mut out, "trim", h, 2 * TRIM_CLASSES, true);

        linear(&mut out, "route_in", h, w, false);
        out.push(("kind_embedding.weight".into(), vec![KIND_EMBEDDING_ROWS, w]));
        linear(&mut out, "memory_in", h, w, false);
        for i in 0..c.route_layers {
            let prefix = format!("route_layers.{i}");
            for n in ["norm1", "norm2", "norm3"] {
                norm(&mut out, &format!("{prefix}.{n}"), w);
            }
            attention(&mut out, &format!("{prefix}.self_attn"), w);
            attention(&mut out, &format!("{prefix}.multihead_attn"), w);
            linear(&mut out, &format!("{prefix}.linear1"), w, 4 * w, true);
            linear(&mut out, &format!("{prefix}.linear2"), 4 * w, w, true);
        }
        norm(&mut out, "route_norm", w);
        linear(&mut out, "route_out", w, h, true);
        out.push(("route_gate".into(), vec![]));

        out.push(("digit_rows".into(), vec![10, h]));
        out.push(("month_rows".into(), vec![12, h]));
        out.push(("sign_rows".into(), vec![2, h]));
        out.push(("lexical_scale".into(), vec![3]));
        linear(&mut out, "slot_down", h, (DIGIT_SLOTS + 2) * r, false);
        out.push(("slot_up".into(), vec![DIGIT_SLOTS, r, 10]));
        out.push(("month_up".into(), vec![r, 12]));
        out.push(("sign_up".into(), vec![r, 2]));
        linear(
            &mut out,
            "slot_residual",
            w,
            DIGIT_SLOTS * 10 + 12 + 2,
            true,
        );
        out.push(("residual_gate".into(), vec![]));
        linear(&mut out, "exponent", h, EXPONENT_CLASSES, true);

        linear(&mut out, "item_end", h, 1, true);
        linear(&mut out, "bins", h, b, true);
        linear(&mut out, "offset", h, b, true);
        linear(&mut out, "overflow", h, 2, true);
        linear(&mut out, "estimate_gate", h, 1, true);
        linear(&mut out, "estimate_spread", h, 1 + LEXICAL_DIGITS, true);

        linear(&mut out, "number_group_down", h, GROUP_RANK, true);
        linear(
            &mut out,
            "number_group_up",
            GROUP_RANK,
            NUMBER_GROUPS * GROUP_CLASSES,
            true,
        );
        linear(&mut out, "copy_query", h, 2 * p, true);
        linear(&mut out, "copy_gate", h, 1, true);

        out
    }

    /// Pre-slice and transpose attention Q/K/V projections, pre-normalize lexical
    /// rows, bake scalar gates/scales, and materialize static calendar index tables.
    fn bake_derived(&mut self) -> Result<()> {
        let mut inserts: Vec<(String, MxArray)> = Vec::new();
        let w = self.config.route_width;

        inserts.push((
            "route_gate_sigmoid".into(),
            Activations::sigmoid(self.w("route_gate")?)?,
        ));
        inserts.push((
            "residual_gate_sigmoid".into(),
            Activations::sigmoid(self.w("residual_gate")?)?,
        ));
        let scale = self
            .w("lexical_scale")?
            .clip(None, Some(100.0_f64.ln()))?
            .exp()?;
        inserts.push(("lexical_scale_0".into(), scale.slice_axis(0, 0, 1)?));
        inserts.push(("lexical_scale_1".into(), scale.slice_axis(0, 1, 2)?));
        inserts.push(("lexical_scale_2".into(), scale.slice_axis(0, 2, 3)?));

        inserts.push((
            "digit_rows_unit_t".into(),
            Self::normalize(self.w("digit_rows")?)?.transpose(None)?,
        ));
        inserts.push((
            "month_rows_unit_t".into(),
            Self::normalize(self.w("month_rows")?)?.transpose(None)?,
        ));
        inserts.push((
            "sign_rows_unit_t".into(),
            Self::normalize(self.w("sign_rows")?)?.transpose(None)?,
        ));
        inserts.push((
            "slot_route_scale_tanh".into(),
            self.w("slot_route_scale")?.tanh()?,
        ));

        for i in 0..self.config.route_layers {
            for attn in ["self_attn", "multihead_attn"] {
                let prefix = format!("route_layers.{i}.{attn}");
                let weight = self.w(&format!("{prefix}.in_proj_weight"))?;
                let bias = self.w(&format!("{prefix}.in_proj_bias"))?;
                for (part, tag) in [(0i64, "q"), (1i64, "k"), (2i64, "v")] {
                    let start = part * w;
                    let end = (part + 1) * w;
                    let pw = weight.slice_axis(0, start, end)?.transpose(None)?;
                    let pb = bias.slice_axis(0, start, end)?;
                    inserts.push((format!("{prefix}.{tag}_w"), pw));
                    inserts.push((format!("{prefix}.{tag}_b"), pb));
                }
            }
        }

        // Static calendar composition indices matching `model.py:1027-1049`.
        for pos in 0..YEAR_DIGITS as usize {
            let divisor = 10_i32.pow((YEAR_DIGITS as u32) - 1 - (pos as u32));
            let idx: Vec<i32> = (YEAR_LOW..=YEAR_HIGH)
                .map(|year| (year / divisor) % 10)
                .collect();
            inserts.push((
                format!("year_index.{pos}"),
                MxArray::from_int32(&idx, &[idx.len() as i64])?,
            ));
        }
        for (name, range) in [("day", 1i32..32), ("hour", 0i32..24), ("sixty", 0i32..60)] {
            let tens: Vec<i32> = range.clone().map(|v| v / 10).collect();
            let ones: Vec<i32> = range.map(|v| v % 10).collect();
            inserts.push((
                format!("{name}_tens"),
                MxArray::from_int32(&tens, &[tens.len() as i64])?,
            ));
            inserts.push((
                format!("{name}_ones"),
                MxArray::from_int32(&ones, &[ones.len() as i64])?,
            ));
        }

        let refs: Vec<&MxArray> = inserts.iter().map(|(_, a)| a).collect();
        MxArray::eval_arrays(&refs)?;
        self.derived.extend(inserts);
        Ok(())
    }

    fn w(&self, name: &str) -> Result<&MxArray> {
        self.weights
            .get(name)
            .ok_or_else(|| Error::from_reason(format!("Missing Clairic head weight: {name}")))
    }

    fn d(&self, name: &str) -> Result<&MxArray> {
        self.derived
            .get(name)
            .ok_or_else(|| Error::from_reason(format!("Missing Clairic derived tensor: {name}")))
    }

    fn linear(&self, name: &str, x: &MxArray) -> Result<MxArray> {
        let weight = self.w(&format!("{name}.weight"))?;
        let y = x.matmul(&weight.transpose(None)?)?;
        match self.weights.get(&format!("{name}.bias")) {
            Some(bias) => y.add(bias),
            None => Ok(y),
        }
    }

    fn norm(&self, name: &str, x: &MxArray) -> Result<MxArray> {
        LayerNorm::from_weights(
            self.w(&format!("{name}.weight"))?,
            Some(self.w(&format!("{name}.bias"))?),
            Some(1e-5),
        )?
        .forward(x)
    }

    fn normalize(x: &MxArray) -> Result<MxArray> {
        let denom = x
            .mul(x)?
            .sum(Some(&[-1]), Some(true))?
            .sqrt()?
            .clip(Some(1e-12), None)?;
        x.div(&denom)
    }

    fn gelu_exact(x: &MxArray) -> Result<MxArray> {
        let mut builder = |inputs: &[MxArray]| -> Result<Vec<MxArray>> {
            let x = &inputs[0];
            let erf = x.mul_scalar(std::f64::consts::FRAC_1_SQRT_2)?.erf()?;
            Ok(vec![x.mul(&erf.add_scalar(1.0)?)?.mul_scalar(0.5)?])
        };
        match crate::compiled_graph::invoke_compiled_graph(
            GELU_EXACT_FN_ID,
            &[x],
            1,
            true,
            &mut builder,
        )? {
            Some(mut out) => Ok(out.remove(0)),
            None => Activations::gelu_exact(x),
        }
    }

    fn project_part(&self, prefix: &str, tag: &str, x: &MxArray) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq = x.shape_at(1)?;
        let heads = self.config.route_heads;
        let head_dim = self.config.route_width / heads;
        let y = x
            .matmul(self.d(&format!("{prefix}.{tag}_w"))?)?
            .add(self.d(&format!("{prefix}.{tag}_b"))?)?;
        y.reshape(&[batch, seq, heads, head_dim])?
            .transpose(Some(&[0, 2, 1, 3]))
    }

    fn attention(
        &self,
        prefix: &str,
        queries: &MxArray,
        keys: &MxArray,
        values: &MxArray,
        mask: Option<&MxArray>,
    ) -> Result<MxArray> {
        let batch = queries.shape_at(0)?;
        let q_len = queries.shape_at(2)?;
        let head_dim = queries.shape_at(3)?;
        let k_batch = keys.shape_at(0)?;
        let (k_bcast, v_bcast) = if k_batch != batch {
            let k_shape = keys.shape()?;
            let v_shape = values.shape()?;
            let k_target = [batch, k_shape[1], k_shape[2], k_shape[3]];
            let v_target = [batch, v_shape[1], v_shape[2], v_shape[3]];
            (
                Some(keys.broadcast_to(&k_target)?),
                Some(values.broadcast_to(&v_target)?),
            )
        } else {
            (None, None)
        };
        let keys_ref = k_bcast.as_ref().unwrap_or(keys);
        let values_ref = v_bcast.as_ref().unwrap_or(values);
        let scale = 1.0 / (head_dim as f64).sqrt();
        let out = scaled_dot_product_attention(queries, keys_ref, values_ref, scale, mask)?;
        let out = out.transpose(Some(&[0, 2, 1, 3]))?.reshape(&[
            batch,
            q_len,
            self.config.route_width,
        ])?;
        self.linear(&format!("{prefix}.out_proj"), &out)
    }

    /// Compute the prompt cross-attention memory tensor `[1, stop, route_width]` in FP32.
    ///
    /// Matches `model.py:1156` (`memory_in(token_norm(hidden[:stop].float())).unsqueeze(0)`)
    /// and the input stage of `mlx_engine.py::_memory`.
    pub fn memory(&self, hidden: &MxArray, stop: i64) -> Result<MxArray> {
        if hidden.ndim()? != 2 || hidden.shape_at(1)? != self.config.hidden_size {
            return Err(Error::from_reason(
                "Invalid hidden tensor shape for ClairicHead::memory",
            ));
        }
        let tokens = hidden.shape_at(0)?;
        if stop <= 0 || stop > tokens {
            return Err(Error::from_reason(format!(
                "Invalid Clairic memory stop={stop} for hidden length {tokens}"
            )));
        }
        let prefix = hidden.slice_axis(0, 0, stop)?.astype(DType::Float32)?;
        let normed = self.norm("token_norm", &prefix)?;
        let mem = self.linear("memory_in", &normed)?.expand_dims(0)?;
        mem.eval();
        Ok(mem)
    }

    fn normalize_memory_arg(&self, memory: &MxArray, stop: i64) -> Result<MxArray> {
        let mem = match memory.ndim()? {
            2 => memory.expand_dims(0)?,
            3 => memory.clone(),
            other => {
                return Err(Error::from_reason(format!(
                    "Invalid Clairic memory rank {other}; expected 2 or 3"
                )));
            }
        };
        if mem.shape_at(0)? != 1
            || mem.shape_at(1)? != stop
            || mem.shape_at(2)? != self.config.route_width
        {
            return Err(Error::from_reason(format!(
                "Invalid Clairic memory shape {:?}; expected [1, {stop}, {}]",
                mem.shape()?.as_ref(),
                self.config.route_width
            )));
        }
        if mem.dtype()? != DType::Float32 {
            mem.astype(DType::Float32)
        } else {
            Ok(mem)
        }
    }

    fn project_memory_kv(&self, memory: &MxArray) -> Result<Vec<(MxArray, MxArray)>> {
        let mut projected = Vec::with_capacity(self.config.route_layers);
        for layer in 0..self.config.route_layers {
            let prefix = format!("route_layers.{layer}.multihead_attn");
            let k = self.project_part(&prefix, "k", memory)?;
            let v = self.project_part(&prefix, "v", memory)?;
            projected.push((k, v));
        }
        Ok(projected)
    }

    fn route_layers(
        &self,
        mut queries: MxArray,
        memory_kv: &[(MxArray, MxArray)],
        self_attn_mask: Option<&MxArray>,
    ) -> Result<MxArray> {
        for (layer, (keys, values)) in memory_kv.iter().enumerate() {
            let prefix = format!("route_layers.{layer}");
            let x = self.norm(&format!("{prefix}.norm1"), &queries)?;
            let sa_prefix = format!("{prefix}.self_attn");
            let sq = self.project_part(&sa_prefix, "q", &x)?;
            let sk = self.project_part(&sa_prefix, "k", &x)?;
            let sv = self.project_part(&sa_prefix, "v", &x)?;
            queries = queries.add(&self.attention(&sa_prefix, &sq, &sk, &sv, self_attn_mask)?)?;

            let x = self.norm(&format!("{prefix}.norm2"), &queries)?;
            let ca_prefix = format!("{prefix}.multihead_attn");
            let cq = self.project_part(&ca_prefix, "q", &x)?;
            queries = queries.add(&self.attention(&ca_prefix, &cq, keys, values, None)?)?;

            let x = self.norm(&format!("{prefix}.norm3"), &queries)?;
            let ff = Self::gelu_exact(&self.linear(&format!("{prefix}.linear1"), &x)?)?;
            queries = queries.add(&self.linear(&format!("{prefix}.linear2"), &ff)?)?;
        }
        Ok(queries)
    }

    fn span_scores(start: &MxArray, end: &MxArray, length_bias: &MxArray) -> Result<MxArray> {
        let q = start.shape_at(0)?;
        let tokens = end.shape_at(1)?;
        let width = length_bias.shape_at(0)?;
        if tokens == 0 {
            return MxArray::zeros(&[q, 0, width], Some(DType::Float32));
        }
        let neg_inf = MxArray::full(
            &[q, width - 1],
            Either::A(f64::NEG_INFINITY),
            Some(DType::Float32),
        )?;
        let padded = MxArray::concatenate(end, &neg_inf, 1)?;
        let row_idx = MxArray::arange(0.0, tokens as f64, Some(1.0), Some(DType::Int32))?
            .reshape(&[tokens, 1])?;
        let col_idx = MxArray::arange(0.0, width as f64, Some(1.0), Some(DType::Int32))?
            .reshape(&[1, width])?;
        let index = row_idx.add(&col_idx)?;
        let gathered_end = padded.take(&index, 1)?;
        start.expand_dims(2)?.add(&gathered_end)?.add(length_bias)
    }

    fn two_digits(&self, slots: &MxArray, name: &str) -> Result<MxArray> {
        let s0 = slots.slice_axis(1, 0, 1)?.squeeze(Some(&[1]))?;
        let s1 = slots.slice_axis(1, 1, 2)?.squeeze(Some(&[1]))?;
        let tens = s0.take(self.d(&format!("{name}_tens"))?, 1)?;
        let ones = s1.take(self.d(&format!("{name}_ones"))?, 1)?;
        tens.add(&ones)
    }

    fn compose_years(&self, year_slots: &MxArray) -> Result<MxArray> {
        let mut inside: Option<MxArray> = None;
        for pos in 0..YEAR_DIGITS {
            let slot = year_slots
                .slice_axis(1, pos, pos + 1)?
                .squeeze(Some(&[1]))?;
            let term = slot.take(self.d(&format!("year_index.{pos}"))?, 1)?;
            inside = Some(match inside {
                Some(acc) => acc.add(&term)?,
                None => term,
            });
        }
        let inside = inside.ok_or_else(|| Error::from_reason("YEAR_DIGITS must be > 0"))?;
        let total = year_slots
            .logsumexp(Some(&[-1]), Some(false))?
            .sum(Some(&[-1]), Some(false))?;
        let gap = inside
            .logsumexp(Some(&[-1]), Some(false))?
            .sub(&total)?
            .clip(None, Some(-1e-6))?;
        // Accurate `-expm1(gap)` on `(-inf, -1e-6]`:
        // For `gap > -0.05`, the 4-term Taylor series `-gap * (1 + gap*(1/2 + gap*(1/6 + gap/24)))`
        // has truncation error < 3e-9 (well below f32 ULP) and avoids cancellation in `1 - exp(gap)`.
        // For `gap <= -0.05`, `1.0 - exp(gap)` has full f32 precision.
        let threshold = MxArray::scalar_float(-0.05)?;
        let small = gap.greater(&threshold)?;
        let poly = gap
            .div_scalar(24.0)?
            .add_scalar(1.0 / 6.0)?
            .mul(&gap)?
            .add_scalar(0.5)?
            .mul(&gap)?
            .add_scalar(1.0)?
            .mul(&gap.negative()?)?;
        let one_minus_exp = gap.exp()?.negative()?.add_scalar(1.0)?;
        let neg_expm1 = small.where_(&poly, &one_minus_exp)?;
        let outside = total.add(&neg_expm1.log()?)?.expand_dims(1)?;
        MxArray::concatenate(&inside, &outside, -1)
    }

    fn typed_logits(&self, raw: &MxArray, routed: &MxArray) -> Result<(MxArray, MxArray, MxArray)> {
        let count = raw.shape_at(0)?;
        let r = self.config.slot_rank;
        let unit = Self::normalize(raw)?;
        let down = raw
            .matmul(&self.w("slot_down.weight")?.transpose(None)?)?
            .reshape(&[count, DIGIT_SLOTS + 2, r])?;

        let digit_prior = unit.matmul(self.d("digit_rows_unit_t")?)?.expand_dims(1)?;
        let month_prior = unit.matmul(self.d("month_rows_unit_t")?)?;
        let sign_prior = unit.matmul(self.d("sign_rows_unit_t")?)?;

        let residual = self
            .d("residual_gate_sigmoid")?
            .mul(&self.linear("slot_residual", routed)?)?;
        let mut res_parts = residual
            .split_sections(&[DIGIT_SLOTS * 10, DIGIT_SLOTS * 10 + 12], -1)?
            .into_iter();
        let next = |it: &mut std::vec::IntoIter<MxArray>| -> Result<MxArray> {
            it.next()
                .ok_or_else(|| Error::from_reason("split_sections under-produced"))
        };
        let digit_residual = next(&mut res_parts)?.reshape(&[count, DIGIT_SLOTS, 10])?;
        let month_residual = next(&mut res_parts)?;
        let sign_residual = next(&mut res_parts)?;

        // einsum("nsr,srd->nsd", down[:, :DIGIT_SLOTS], slot_up)
        let down_digits = down
            .slice_axis(1, 0, DIGIT_SLOTS)?
            .transpose(Some(&[1, 0, 2]))?;
        let digit_low_rank = down_digits
            .matmul(self.w("slot_up")?)?
            .transpose(Some(&[1, 0, 2]))?;

        let down_month = down
            .slice_axis(1, DIGIT_SLOTS, DIGIT_SLOTS + 1)?
            .squeeze(Some(&[1]))?;
        let month_low_rank = down_month.matmul(self.w("month_up")?)?;

        let down_sign = down
            .slice_axis(1, DIGIT_SLOTS + 1, DIGIT_SLOTS + 2)?
            .squeeze(Some(&[1]))?;
        let sign_low_rank = down_sign.matmul(self.w("sign_up")?)?;

        let digits = self
            .d("lexical_scale_0")?
            .mul(&digit_prior)?
            .add(&digit_low_rank)?
            .add(&digit_residual)?;
        let months = self
            .d("lexical_scale_1")?
            .mul(&month_prior)?
            .add(&month_low_rank)?
            .add(&month_residual)?;
        let signs = self
            .d("lexical_scale_2")?
            .mul(&sign_prior)?
            .add(&sign_low_rank)?
            .add(&sign_residual)?;

        Ok((digits, months, signs))
    }

    fn typed_outputs(
        &self,
        raw: &MxArray,
        states: &MxArray,
        routed: &MxArray,
    ) -> Result<TypedOutputs> {
        let (digits, months, signs) = self.typed_logits(raw, routed)?;
        let count = raw.shape_at(0)?;

        let num_digits = digits
            .slice_axis(1, 0, NUMBER_DIGITS)?
            .reshape(&[count, NUMBER_DIGITS * 10])?;
        let exp_logits = self.linear("exponent", states)?;
        let number = MxArray::concatenate_many(vec![&signs, &exp_logits, &num_digits], Some(-1))?;

        let year_slots = digits.slice_axis(1, YEAR_SLOT, YEAR_SLOT + YEAR_DIGITS)?;
        let year_digits_flat = year_slots.reshape(&[count, YEAR_DIGITS * 10])?;
        let year_composed = self.compose_years(&year_slots)?;
        let day_slots = digits.slice_axis(1, DAY_SLOT, DAY_SLOT + 2)?;
        let days = self.two_digits(&day_slots, "day")?;
        let date = MxArray::concatenate_many(
            vec![&year_digits_flat, &year_composed, &months, &days],
            Some(-1),
        )?;

        let hour_slots = digits.slice_axis(1, HOUR_SLOT, HOUR_SLOT + 2)?;
        let min_slots = digits.slice_axis(1, MINUTE_SLOT, MINUTE_SLOT + 2)?;
        let sec_slots = digits.slice_axis(1, SECOND_SLOT, SECOND_SLOT + 2)?;
        let hours = self.two_digits(&hour_slots, "hour")?;
        let mins = self.two_digits(&min_slots, "sixty")?;
        let secs = self.two_digits(&sec_slots, "sixty")?;
        let time = MxArray::concatenate_many(vec![&hours, &mins, &secs], Some(-1))?;

        Ok(TypedOutputs { number, date, time })
    }

    fn group_scores(&self, states: &MxArray) -> Result<MxArray> {
        let count = states.shape_at(0)?;
        let hidden = Self::gelu_exact(&self.linear("number_group_down", states)?)?;
        self.linear("number_group_up", &hidden)?
            .reshape(&[count, NUMBER_GROUPS, GROUP_CLASSES])
    }

    fn temper_number(logits: &MxArray, spread: &MxArray) -> Result<MxArray> {
        let count = logits.shape_at(0)?;
        let sign = logits.slice_axis(1, 0, 2)?;
        let exponent = logits.slice_axis(1, 2, 2 + EXPONENT_CLASSES)?;
        let slots = logits
            .slice_axis(1, 2 + EXPONENT_CLASSES, NUMBER_WIDTH)?
            .reshape(&[count, NUMBER_DIGITS, 10])?;
        let temperature = spread
            .div_scalar(ESTIMATE_SPREAD_LIMIT)?
            .tanh()?
            .mul_scalar(ESTIMATE_SPREAD_LIMIT)?
            .exp()?;
        let exp_temp = temperature.slice_axis(1, 0, 1)?;
        let digit_temp = temperature
            .slice_axis(1, 1, 1 + LEXICAL_DIGITS)?
            .expand_dims(2)?;
        let head_slots = slots.slice_axis(1, 0, LEXICAL_DIGITS)?.div(&digit_temp)?;
        let tail_slots = slots.slice_axis(1, LEXICAL_DIGITS, NUMBER_DIGITS)?;
        let tempered_slots = MxArray::concatenate(&head_slots, &tail_slots, 1)?
            .reshape(&[count, NUMBER_DIGITS * 10])?;
        let tempered_exp = exponent.div(&exp_temp)?;
        MxArray::concatenate_many(vec![&sign, &tempered_exp, &tempered_slots], Some(-1))
    }

    fn take_rows(values: &MxArray, rows: &[i64]) -> Result<MxArray> {
        let idx_i32: Vec<i32> = rows.iter().map(|&r| r as i32).collect();
        let idx = MxArray::from_int32(&idx_i32, &[idx_i32.len() as i64])?;
        values.take(&idx, 0)
    }

    fn pointer_scores(
        &self,
        query_name: &str,
        selected: &MxArray,
        tokens: &MxArray,
    ) -> Result<MxArray> {
        let p = self.config.pointer_width;
        let scale = 1.0 / (p as f64).sqrt();
        let mut q_parts = self
            .linear(query_name, selected)?
            .split_sections(&[p], -1)?
            .into_iter();
        let mut k_parts = self
            .linear("span_key", tokens)?
            .split_sections(&[p], -1)?
            .into_iter();
        let next = |it: &mut std::vec::IntoIter<MxArray>| -> Result<MxArray> {
            it.next()
                .ok_or_else(|| Error::from_reason("split_sections under-produced"))
        };
        let q_start = next(&mut q_parts)?;
        let q_end = next(&mut q_parts)?;
        let k_start = next(&mut k_parts)?;
        let k_end = next(&mut k_parts)?;
        let start = q_start
            .matmul(&k_start.transpose(None)?)?
            .mul_scalar(scale)?;
        let end = q_end.matmul(&k_end.transpose(None)?)?.mul_scalar(scale)?;
        Self::span_scores(&start, &end, self.w("length_bias")?)
    }

    /// Run the full lexical typed-head forward pass (`mlx_engine.py::TypedHead.forward`).
    pub fn forward(
        &self,
        hidden: &MxArray,
        input_start: i64,
        stop: i64,
        views: &PredictionViews,
        memory: &MxArray,
    ) -> Result<HeadOutputs> {
        if hidden.ndim()? != 2 || hidden.shape_at(1)? != self.config.hidden_size {
            return Err(Error::from_reason(
                "Invalid hidden tensor shape for ClairicHead::forward",
            ));
        }
        let tokens_len = hidden.shape_at(0)?;
        if stop <= 0 || stop > tokens_len || input_start < 0 || input_start > stop {
            return Err(Error::from_reason(format!(
                "Invalid Clairic token range start={input_start}, stop={stop}, hidden_len={tokens_len}"
            )));
        }
        let count = views.query_positions.len();
        if count == 0 {
            return Err(Error::from_reason(
                "ClairicHead::forward requires at least one query position",
            ));
        }
        let mem = self.normalize_memory_arg(memory, stop)?;
        let memory_kv = self.project_memory_kv(&mem)?;

        // Validate row indices and list item shapes before building flat positions/kinds.
        let group_rows: [&[i64]; 8] = [
            &views.text_rows,
            &views.tags_rows,
            &views.number_rows,
            &views.date_rows,
            &views.time_rows,
            &views.estimate_rows,
            &views.number_list_rows,
            &views.date_list_rows,
        ];
        for (group_idx, rows) in group_rows.iter().enumerate() {
            for &r in *rows {
                if r < 0 || (r as usize) >= count {
                    return Err(Error::from_reason(format!(
                        "Clairic view row index {r} out of range for group {} (queries={count})",
                        ROUTE_KINDS[group_idx]
                    )));
                }
            }
        }

        let validate_items = |label: &str, rows_len: usize, items: &[Vec<i64>]| -> Result<usize> {
            if rows_len == 0 {
                return Ok(0);
            }
            if items.len() != rows_len {
                return Err(Error::from_reason(format!(
                    "{label}_items row count {} does not match {label}_rows length {rows_len}",
                    items.len()
                )));
            }
            let width = items[0].len();
            if width == 0 {
                return Err(Error::from_reason(format!(
                    "{label}_items must have at least 1 slot per row"
                )));
            }
            for row in items {
                if row.len() != width {
                    return Err(Error::from_reason(format!(
                        "{label}_items rows must be rectangular (expected {width}, got {})",
                        row.len()
                    )));
                }
            }
            Ok(width)
        };
        let nl_slots = validate_items(
            "number_list",
            views.number_list_rows.len(),
            &views.number_list_items,
        )?;
        let dl_slots = validate_items(
            "date_list",
            views.date_list_rows.len(),
            &views.date_list_items,
        )?;

        let nl_size = views.number_list_rows.len() * nl_slots;
        let dl_size = views.date_list_rows.len() * dl_slots;
        let total_positions = count + nl_size + dl_size;

        let mut positions: Vec<i32> = Vec::with_capacity(total_positions);
        for &p in &views.query_positions {
            if p < 0 || p >= tokens_len {
                return Err(Error::from_reason(format!(
                    "Clairic query position {p} out of range [0, {tokens_len})"
                )));
            }
            positions.push(p as i32);
        }
        if !views.number_list_rows.is_empty() {
            for row in &views.number_list_items {
                for &p in row {
                    if p < 0 || p >= tokens_len {
                        return Err(Error::from_reason(format!(
                            "Clairic number_list item position {p} out of range [0, {tokens_len})"
                        )));
                    }
                    positions.push(p as i32);
                }
            }
        }
        if !views.date_list_rows.is_empty() {
            for row in &views.date_list_items {
                for &p in row {
                    if p < 0 || p >= tokens_len {
                        return Err(Error::from_reason(format!(
                            "Clairic date_list item position {p} out of range [0, {tokens_len})"
                        )));
                    }
                    positions.push(p as i32);
                }
            }
        }

        let mut kinds: Vec<i32> = vec![0; total_positions];
        for (group_idx, rows) in group_rows.iter().enumerate() {
            for &r in *rows {
                kinds[r as usize] = group_idx as i32;
            }
        }
        let mut kind_offset = count;
        if nl_size > 0 {
            let nl_kind = ROUTE_KIND_COUNT as i32;
            for k in &mut kinds[kind_offset..kind_offset + nl_size] {
                *k = nl_kind;
            }
            kind_offset += nl_size;
        }
        if dl_size > 0 {
            let dl_kind = (ROUTE_KIND_COUNT + 1) as i32;
            for k in &mut kinds[kind_offset..kind_offset + dl_size] {
                *k = dl_kind;
            }
        }

        let hidden_f32 = if hidden.dtype()? == DType::Float32 {
            hidden.clone()
        } else {
            hidden.astype(DType::Float32)?
        };

        let pos_arr = MxArray::from_int32(&positions, &[total_positions as i64])?;
        let kind_arr = MxArray::from_int32(&kinds, &[total_positions as i64])?;

        let raw = hidden_f32.take(&pos_arr, 0)?;
        let normed_raw = self.norm("norm", &raw)?;
        let kind_emb = self.w("kind_embedding.weight")?.take(&kind_arr, 0)?;
        let route_queries = self
            .linear("route_in", &normed_raw)?
            .add(&kind_emb)?
            .expand_dims(0)?;
        let routed = self
            .route_layers(route_queries, &memory_kv, None)?
            .squeeze(Some(&[0]))?;
        let routed = self.norm("route_norm", &routed)?;
        let states = normed_raw.add(
            &self
                .d("route_gate_sigmoid")?
                .mul(&self.linear("route_out", &routed)?)?,
        )?;

        let count_i64 = count as i64;
        let field_states = states.slice_axis(0, 0, count_i64)?;
        let field_raw = raw.slice_axis(0, 0, count_i64)?;
        let field_routed = routed.slice_axis(0, 0, count_i64)?;

        let null = self.linear("null", &field_states)?.squeeze(Some(&[-1]))?;

        let has_text = !views.text_rows.is_empty();
        let has_tags = !views.tags_rows.is_empty();
        let has_numbers = !views.number_rows.is_empty();

        let mut trim = None;
        let tokens = if has_text || has_tags || has_numbers {
            let sliced = hidden_f32.slice_axis(0, input_start, stop)?;
            let tok = self.norm("token_norm", &sliced)?;
            if has_text || has_tags {
                let tok_len = tok.shape_at(0)?;
                trim = Some(
                    self.linear("trim", &tok)?
                        .reshape(&[tok_len, 2, TRIM_CLASSES])?,
                );
            }
            Some(tok)
        } else {
            None
        };

        let (number_copy, number_copy_gate) = if has_numbers {
            let tok = tokens
                .as_ref()
                .ok_or_else(|| Error::from_reason("Internal error: missing token_norm states"))?;
            let number_states = Self::take_rows(&field_states, &views.number_rows)?;
            let copy = self.pointer_scores("copy_query", &number_states, tok)?;
            let gate = self
                .linear("copy_gate", &number_states)?
                .squeeze(Some(&[-1]))?;
            (Some(copy), Some(gate))
        } else {
            (None, None)
        };

        let text = if has_text {
            let tok = tokens
                .as_ref()
                .ok_or_else(|| Error::from_reason("Internal error: missing token_norm states"))?;
            let text_states = Self::take_rows(&field_states, &views.text_rows)?;
            Some(self.pointer_scores("span_query", &text_states, tok)?)
        } else {
            None
        };

        let tags = if has_tags {
            let tok = tokens
                .as_ref()
                .ok_or_else(|| Error::from_reason("Internal error: missing token_norm states"))?;
            let p = self.config.pointer_width;
            let scale = 1.0 / (p as f64).sqrt();
            let tag_states = Self::take_rows(&field_states, &views.tags_rows)?;
            let mut q_parts = self
                .linear("tag_query", &tag_states)?
                .split_sections(&[p], -1)?
                .into_iter();
            let q_begin = q_parts
                .next()
                .ok_or_else(|| Error::from_reason("split_sections under-produced"))?;
            let q_inside = q_parts
                .next()
                .ok_or_else(|| Error::from_reason("split_sections under-produced"))?;
            let keys_t = self.linear("tag_key", tok)?.transpose(None)?;
            let tag_bias = self.w("tag_bias")?;
            let b0 = tag_bias.slice_axis(0, 0, 1)?;
            let b1 = tag_bias.slice_axis(0, 1, 2)?;
            let begin = q_begin.matmul(&keys_t)?.mul_scalar(scale)?.add(&b0)?;
            let inside = q_inside.matmul(&keys_t)?.mul_scalar(scale)?.add(&b1)?;
            let outside = MxArray::zeros(begin.shape()?.as_ref(), Some(DType::Float32))?;
            Some(MxArray::stack(vec![&outside, &begin, &inside], Some(-1))?)
        } else {
            None
        };

        let mut number = None;
        let mut number_groups = None;
        let mut date = None;
        let mut time = None;
        let typed_total = views.number_rows.len() + views.date_rows.len() + views.time_rows.len();
        if typed_total > 0 {
            let mut selected: Vec<i64> = Vec::with_capacity(typed_total);
            selected.extend_from_slice(&views.number_rows);
            selected.extend_from_slice(&views.date_rows);
            selected.extend_from_slice(&views.time_rows);
            let typed = self.typed_outputs(
                &Self::take_rows(&field_raw, &selected)?,
                &Self::take_rows(&field_states, &selected)?,
                &Self::take_rows(&field_routed, &selected)?,
            )?;
            let mut off = 0i64;
            let n_len = views.number_rows.len() as i64;
            if n_len > 0 {
                number = Some(typed.number.slice_axis(0, off, off + n_len)?);
                number_groups =
                    Some(self.group_scores(&Self::take_rows(&field_states, &views.number_rows)?)?);
                off += n_len;
            }
            let d_len = views.date_rows.len() as i64;
            if d_len > 0 {
                date = Some(typed.date.slice_axis(0, off, off + d_len)?);
                off += d_len;
            }
            let t_len = views.time_rows.len() as i64;
            if t_len > 0 {
                time = Some(typed.time.slice_axis(0, off, off + t_len)?);
            }
        }

        let (
            estimate,
            estimate_offset,
            estimate_overflow,
            estimate_number_raw,
            estimate_number,
            estimate_gate,
        ) = if !views.estimate_rows.is_empty() {
            let est_states = Self::take_rows(&field_states, &views.estimate_rows)?;
            let est_raw = Self::take_rows(&field_raw, &views.estimate_rows)?;
            let est_routed = Self::take_rows(&field_routed, &views.estimate_rows)?;
            let est_bins = self.linear("bins", &est_states)?;
            let est_off = self.linear("offset", &est_states)?;
            let est_ovf = self.linear("overflow", &est_states)?;
            let num_raw = self
                .typed_outputs(&est_raw, &est_states, &est_routed)?
                .number;
            let spread = self.linear("estimate_spread", &est_states)?;
            let num_tempered = Self::temper_number(&num_raw, &spread)?;
            let gate = self
                .linear("estimate_gate", &est_states)?
                .squeeze(Some(&[-1]))?;
            (
                Some(est_bins),
                Some(est_off),
                Some(est_ovf),
                Some(num_raw),
                Some(num_tempered),
                Some(gate),
            )
        } else {
            (None, None, None, None, None, None)
        };

        let mut list_offset = count_i64;
        let (number_list, number_list_end, number_list_groups) = if nl_size > 0 {
            let rows = views.number_list_rows.len() as i64;
            let slots = nl_slots as i64;
            let size = nl_size as i64;
            let item_states = states.slice_axis(0, list_offset, list_offset + size)?;
            let item_raw = raw.slice_axis(0, list_offset, list_offset + size)?;
            let item_routed = routed.slice_axis(0, list_offset, list_offset + size)?;
            list_offset += size;
            let end_logits = self
                .linear("item_end", &item_states)?
                .squeeze(Some(&[-1]))?
                .reshape(&[rows, slots])?;
            let num_logits = self
                .typed_outputs(&item_raw, &item_states, &item_routed)?
                .number
                .reshape(&[rows, slots, NUMBER_WIDTH])?;
            let grp_logits = self.group_scores(&item_states)?.reshape(&[
                rows,
                slots,
                NUMBER_GROUPS,
                GROUP_CLASSES,
            ])?;
            (Some(num_logits), Some(end_logits), Some(grp_logits))
        } else {
            (None, None, None)
        };

        let (date_list, date_list_end) = if dl_size > 0 {
            let rows = views.date_list_rows.len() as i64;
            let slots = dl_slots as i64;
            let size = dl_size as i64;
            let item_states = states.slice_axis(0, list_offset, list_offset + size)?;
            let item_raw = raw.slice_axis(0, list_offset, list_offset + size)?;
            let item_routed = routed.slice_axis(0, list_offset, list_offset + size)?;
            let end_logits = self
                .linear("item_end", &item_states)?
                .squeeze(Some(&[-1]))?
                .reshape(&[rows, slots])?;
            let date_logits = self
                .typed_outputs(&item_raw, &item_states, &item_routed)?
                .date
                .reshape(&[rows, slots, DATE_WIDTH])?;
            (Some(date_logits), Some(end_logits))
        } else {
            (None, None)
        };

        Ok(HeadOutputs {
            null,
            trim,
            text,
            tags,
            number,
            number_groups,
            number_copy,
            number_copy_gate,
            date,
            time,
            estimate,
            estimate_offset,
            estimate_overflow,
            estimate_number_raw,
            estimate_number,
            estimate_gate,
            number_list,
            number_list_end,
            number_list_groups,
            date_list,
            date_list_end,
        })
    }

    /// Correct each answer-slot hidden state by evidence routed from the prompt.
    ///
    /// Matches `mlx_engine.py::TypedHead.answer_slot_states` (`L326-353`) and
    /// `model.py::ClairicHead.answer_slot_states` (`L1162-1246`).
    #[allow(clippy::too_many_arguments)]
    pub fn answer_slot_states(
        &self,
        hidden: &MxArray,
        field_queries: &[i64],
        slot_kinds: &[i64],
        slot_positions: &[Vec<i64>],
        slot_lengths: &[i64],
        stop: i64,
        memory: &MxArray,
    ) -> Result<MxArray> {
        const CHUNK: usize = 256;

        if hidden.ndim()? != 2 || hidden.shape_at(1)? != self.config.hidden_size {
            return Err(Error::from_reason("invalid answer-slot routing layout"));
        }
        let tokens = hidden.shape_at(0)?;
        let n = field_queries.len();
        if slot_kinds.len() != n || slot_positions.len() != n || slot_lengths.len() != n {
            return Err(Error::from_reason(
                "answer-slot fields, kinds and lengths must align",
            ));
        }
        if stop <= 0 || stop > tokens {
            return Err(Error::from_reason(
                "invalid answer-slot lengths or evidence boundary",
            ));
        }
        if n == 0 {
            return MxArray::zeros(&[0, 0, self.config.hidden_size], Some(DType::Float32));
        }
        let width = slot_positions[0].len();
        if width < 1 {
            return Err(Error::from_reason(
                "invalid answer-slot lengths or evidence boundary",
            ));
        }
        for i in 0..n {
            if slot_positions[i].len() != width {
                return Err(Error::from_reason("invalid answer-slot routing layout"));
            }
            if slot_lengths[i] < 1 || slot_lengths[i] > width as i64 {
                return Err(Error::from_reason(
                    "invalid answer-slot lengths or evidence boundary",
                ));
            }
            if field_queries[i] < 0 || field_queries[i] >= tokens {
                return Err(Error::from_reason(
                    "answer-slot query, token or kind out of range",
                ));
            }
            if slot_kinds[i] < 0 || slot_kinds[i] >= ROUTE_KIND_COUNT {
                return Err(Error::from_reason(
                    "answer-slot query, token or kind out of range",
                ));
            }
            for &p in &slot_positions[i] {
                if p < 0 || p >= tokens {
                    return Err(Error::from_reason(
                        "answer-slot query, token or kind out of range",
                    ));
                }
            }
        }

        let mem = self.normalize_memory_arg(memory, stop)?;
        let memory_kv = self.project_memory_kv(&mem)?;
        let hidden_f32 = if hidden.dtype()? == DType::Float32 {
            hidden.clone()
        } else {
            hidden.astype(DType::Float32)?
        };
        let gate = self.d("route_gate_sigmoid")?;
        let slot_route_scale_tanh = self.d("slot_route_scale_tanh")?;

        let field_batch = (CHUNK / (width + 1))
            .min(262_144usize / (stop as usize))
            .max(1);
        let mut chunks: Vec<MxArray> = Vec::with_capacity(n.div_ceil(field_batch));

        let width_i64 = width as i64;
        for first in (0..n).step_by(field_batch) {
            let last = (first + field_batch).min(n);
            let batch = last - first;
            let batch_i64 = batch as i64;

            let mut slot_pos_flat: Vec<i32> = Vec::with_capacity(batch * width);
            let mut query_pos_flat: Vec<i32> = Vec::with_capacity(batch * (width + 1));
            let mut row_kinds: Vec<i32> = Vec::with_capacity(batch);
            let mut valid_mask: Vec<f32> = Vec::with_capacity(batch * (width + 1));
            let mut gather_idx: Vec<i32> = Vec::with_capacity(batch * width);

            for row in first..last {
                let len = slot_lengths[row] as usize;
                row_kinds.push(slot_kinds[row] as i32);
                query_pos_flat.push(field_queries[row] as i32);
                for &p in &slot_positions[row] {
                    slot_pos_flat.push(p as i32);
                    query_pos_flat.push(p as i32);
                }
                // Additive attention mask matching `padding = arange(width + 1) > length`:
                // 0.0 for valid keys (`j <= len`), -inf for padded keys (`j > len`).
                for j in 0..=width {
                    valid_mask.push(if j <= len { 0.0 } else { f32::NEG_INFINITY });
                }
                let max_idx = (len - 1) as i32;
                for j in 0..width {
                    gather_idx.push((j as i32).min(max_idx));
                }
            }

            let slot_pos_arr = MxArray::from_int32(&slot_pos_flat, &[batch_i64, width_i64])?;
            let query_pos_arr = MxArray::from_int32(&query_pos_flat, &[batch_i64, width_i64 + 1])?;
            let row_kinds_arr = MxArray::from_int32(&row_kinds, &[batch_i64])?;
            let attn_mask = MxArray::from_float32(&valid_mask, &[batch_i64, 1, 1, width_i64 + 1])?;
            let gather_arr = MxArray::from_int32(&gather_idx, &[batch_i64, width_i64, 1])?;

            let gathered_q = hidden_f32.take(&query_pos_arr, 0)?;
            let queries_in = self.linear("route_in", &self.norm("norm", &gathered_q)?)?;
            let kind_emb = self
                .w("kind_embedding.weight")?
                .take(&row_kinds_arr, 0)?
                .expand_dims(1)?;
            let routed =
                self.route_layers(queries_in.add(&kind_emb)?, &memory_kv, Some(&attn_mask))?;
            let slot_routed = routed.slice_axis(1, 1, width_i64 + 1)?;
            let evidence = self.linear("route_out", &self.norm("route_norm", &slot_routed)?)?;
            let scale = slot_route_scale_tanh
                .take(&row_kinds_arr, 0)?
                .reshape(&[batch_i64, 1, 1])?;
            let raw_slots = hidden_f32.take(&slot_pos_arr, 0)?;
            let corrected = raw_slots.add(&scale.mul(gate)?.mul(&evidence)?)?;
            let clamped = corrected.take_along_axis(&gather_arr, 1)?;
            clamped.eval();
            chunks.push(clamped);
        }

        if chunks.len() == 1 {
            Ok(chunks.remove(0))
        } else {
            let refs: Vec<&MxArray> = chunks.iter().collect();
            MxArray::concatenate_many(refs, Some(0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_config() -> ClairicHeadConfig {
        ClairicHeadConfig {
            hidden_size: 64,
            bins: 16,
            pointer_width: 16,
            readout: "lexical".to_string(),
            route_width: 32,
            route_layers: 2,
            route_heads: 4,
            slot_rank: 8,
            estimate_lexical: true,
            span_trim: true,
            loop_feedback: false,
            answer_slots: true,
            answer_slot_routing: true,
            number_groups: true,
            number_pointer: true,
        }
    }

    fn det_data(seed: u64, len: usize, scale: f32) -> Vec<f32> {
        let mut x = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let u = ((x >> 40) as f32) / ((1u32 << 24) as f32);
            out.push((u * 2.0 - 1.0) * scale);
        }
        out
    }

    fn synthetic_weights(cfg: &ClairicHeadConfig) -> HashMap<String, MxArray> {
        let shapes = ClairicHead::expected_shapes(cfg);
        let mut map = HashMap::with_capacity(shapes.len());
        for (i, (name, shape)) in shapes.into_iter().enumerate() {
            let numel: usize = shape.iter().map(|&d| d as usize).product();
            let numel = numel.max(1);
            let data = if name == "length_bias" {
                (0..numel)
                    .map(|j| -0.05 * (j as f32) + 0.000_123_4)
                    .collect()
            } else if name.ends_with(".weight") && shape.len() == 1 {
                det_data((i as u64) + 100, numel, 0.2)
                    .into_iter()
                    .map(|v| 1.0 + v)
                    .collect()
            } else if name == "lexical_scale" {
                vec![std::f32::consts::LN_10, 2.0, 1.5]
            } else {
                det_data((i as u64) + 1, numel, 0.25)
            };
            let arr = MxArray::from_float32(&data, &shape).unwrap();
            map.insert(name, arr);
        }
        map
    }

    fn assert_all_finite(name: &str, arr: &MxArray) {
        let vals = arr.to_float32().unwrap();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "{name} contained non-finite value"
        );
    }

    #[test]
    fn from_weights_validates_keys_shapes_and_config() {
        let cfg = synthetic_config();
        let weights = synthetic_weights(&cfg);
        let head = ClairicHead::from_weights(cfg.clone(), weights, 64, false).unwrap();
        assert!(head.nbytes() > 0);

        // Wrong hidden_size
        let err = ClairicHead::from_weights(cfg.clone(), synthetic_weights(&cfg), 128, false);
        assert!(err.is_err());

        // Missing required key
        let mut missing = synthetic_weights(&cfg);
        missing.remove("route_layers.1.norm2.bias");
        assert!(ClairicHead::from_weights(cfg.clone(), missing, 64, false).is_err());

        // Unexpected extra key
        let mut extra = synthetic_weights(&cfg);
        extra.insert(
            "unexpected_tensor".into(),
            MxArray::zeros(&[4], Some(DType::Float32)).unwrap(),
        );
        assert!(ClairicHead::from_weights(cfg.clone(), extra, 64, false).is_err());

        // Wrong shape
        let mut bad_shape = synthetic_weights(&cfg);
        bad_shape.insert(
            "length_bias".into(),
            MxArray::zeros(&[32], Some(DType::Float32)).unwrap(),
        );
        assert!(ClairicHead::from_weights(cfg.clone(), bad_shape, 64, false).is_err());

        // Unsupported config flag (loop_feedback = true)
        let mut bad_cfg = cfg.clone();
        bad_cfg.loop_feedback = true;
        assert!(ClairicHead::from_weights(bad_cfg, synthetic_weights(&cfg), 64, false).is_err());
    }

    #[test]
    fn round_bf16_preserves_length_bias_and_rounds_other_weights() {
        let cfg = synthetic_config();
        let mut weights = synthetic_weights(&cfg);
        // Set exact f32 values that change under bf16 rounding.
        let lb: Vec<f32> = (0..MAX_SPAN as usize)
            .map(|j| -0.05 * (j as f32) + 0.000_123_45)
            .collect();
        let tb = vec![-2.000_123_5_f32, -1.999_876_5_f32];
        weights.insert(
            "length_bias".into(),
            MxArray::from_float32(&lb, &[MAX_SPAN]).unwrap(),
        );
        weights.insert("tag_bias".into(), MxArray::from_float32(&tb, &[2]).unwrap());

        let head = ClairicHead::from_weights(cfg, weights, 64, true).unwrap();
        let got_lb: Vec<f32> = head
            .w("length_bias")
            .unwrap()
            .to_float32()
            .unwrap()
            .to_vec();
        assert_eq!(got_lb, lb, "length_bias must remain exact f32");

        let expected_tb: Vec<f32> = MxArray::from_float32(&tb, &[2])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap()
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap()
            .to_vec();
        let got_tb: Vec<f32> = head.w("tag_bias").unwrap().to_float32().unwrap().to_vec();
        assert_eq!(got_tb, expected_tb);
        assert_ne!(got_tb, tb, "tag_bias should be bf16-rounded");
    }

    #[test]
    fn forward_covers_all_eight_route_kinds_and_answer_slot_states() {
        let cfg = synthetic_config();
        let head =
            ClairicHead::from_weights(cfg.clone(), synthetic_weights(&cfg), 64, false).unwrap();

        let tokens = 30i64;
        let input_start = 4i64;
        let stop = 16i64;
        let span_tokens = stop - input_start; // 12
        let hidden = MxArray::from_float32(
            &det_data(42, (tokens * cfg.hidden_size) as usize, 0.5),
            &[tokens, cfg.hidden_size],
        )
        .unwrap();
        let memory = head.memory(&hidden, stop).unwrap();
        assert_eq!(
            memory.shape().unwrap().as_ref(),
            &[1, stop, cfg.route_width]
        );

        // 8 queries, one per route kind, plus 2 list-item rows with 3 slots each.
        let views = PredictionViews {
            query_positions: vec![16, 17, 18, 19, 20, 21, 22, 23],
            text_rows: vec![0],
            tags_rows: vec![1],
            number_rows: vec![2],
            date_rows: vec![3],
            time_rows: vec![4],
            estimate_rows: vec![5],
            number_list_rows: vec![6],
            date_list_rows: vec![7],
            number_list_items: vec![vec![24, 25, 26]],
            date_list_items: vec![vec![27, 28, 29]],
        };

        let out = head
            .forward(&hidden, input_start, stop, &views, &memory)
            .unwrap();
        assert_eq!(out.null.shape().unwrap().as_ref(), &[8]);
        assert_all_finite("null", &out.null);

        let trim = out.trim.as_ref().unwrap();
        assert_eq!(
            trim.shape().unwrap().as_ref(),
            &[span_tokens, 2, TRIM_CLASSES]
        );
        assert_all_finite("trim", trim);

        let text = out.text.as_ref().unwrap();
        assert_eq!(text.shape().unwrap().as_ref(), &[1, span_tokens, MAX_SPAN]);
        // In-bounds spans finite, out-of-bounds spans -inf
        let text_vals = text.to_float32().unwrap();
        assert!(text_vals[0].is_finite());
        assert!(text_vals[(span_tokens as usize - 1) * (MAX_SPAN as usize) + 1].is_infinite());

        let tags = out.tags.as_ref().unwrap();
        assert_eq!(tags.shape().unwrap().as_ref(), &[1, span_tokens, 3]);
        assert_all_finite("tags", tags);

        let number = out.number.as_ref().unwrap();
        assert_eq!(number.shape().unwrap().as_ref(), &[1, NUMBER_WIDTH]);
        assert_all_finite("number", number);

        let number_groups = out.number_groups.as_ref().unwrap();
        assert_eq!(
            number_groups.shape().unwrap().as_ref(),
            &[1, NUMBER_GROUPS, GROUP_CLASSES]
        );
        assert_all_finite("number_groups", number_groups);

        let number_copy = out.number_copy.as_ref().unwrap();
        assert_eq!(
            number_copy.shape().unwrap().as_ref(),
            &[1, span_tokens, MAX_SPAN]
        );

        let number_copy_gate = out.number_copy_gate.as_ref().unwrap();
        assert_eq!(number_copy_gate.shape().unwrap().as_ref(), &[1]);
        assert_all_finite("number_copy_gate", number_copy_gate);

        let date = out.date.as_ref().unwrap();
        assert_eq!(date.shape().unwrap().as_ref(), &[1, DATE_WIDTH]);
        assert_all_finite("date", date);

        let time = out.time.as_ref().unwrap();
        assert_eq!(time.shape().unwrap().as_ref(), &[1, TIME_WIDTH]);
        assert_all_finite("time", time);

        let est = out.estimate.as_ref().unwrap();
        assert_eq!(est.shape().unwrap().as_ref(), &[1, cfg.bins]);
        assert_all_finite("estimate", est);
        assert_eq!(
            out.estimate_offset
                .as_ref()
                .unwrap()
                .shape()
                .unwrap()
                .as_ref(),
            &[1, cfg.bins]
        );
        assert_eq!(
            out.estimate_overflow
                .as_ref()
                .unwrap()
                .shape()
                .unwrap()
                .as_ref(),
            &[1, 2]
        );
        assert_eq!(
            out.estimate_number_raw
                .as_ref()
                .unwrap()
                .shape()
                .unwrap()
                .as_ref(),
            &[1, NUMBER_WIDTH]
        );
        assert_eq!(
            out.estimate_number
                .as_ref()
                .unwrap()
                .shape()
                .unwrap()
                .as_ref(),
            &[1, NUMBER_WIDTH]
        );
        assert_eq!(
            out.estimate_gate
                .as_ref()
                .unwrap()
                .shape()
                .unwrap()
                .as_ref(),
            &[1]
        );
        assert_all_finite("estimate_number", out.estimate_number.as_ref().unwrap());
        assert_all_finite("estimate_gate", out.estimate_gate.as_ref().unwrap());

        assert_eq!(
            out.number_list.as_ref().unwrap().shape().unwrap().as_ref(),
            &[1, 3, NUMBER_WIDTH]
        );
        assert_eq!(
            out.number_list_end
                .as_ref()
                .unwrap()
                .shape()
                .unwrap()
                .as_ref(),
            &[1, 3]
        );
        assert_eq!(
            out.number_list_groups
                .as_ref()
                .unwrap()
                .shape()
                .unwrap()
                .as_ref(),
            &[1, 3, NUMBER_GROUPS, GROUP_CLASSES]
        );
        assert_eq!(
            out.date_list.as_ref().unwrap().shape().unwrap().as_ref(),
            &[1, 3, DATE_WIDTH]
        );
        assert_eq!(
            out.date_list_end
                .as_ref()
                .unwrap()
                .shape()
                .unwrap()
                .as_ref(),
            &[1, 3]
        );
        assert_all_finite("number_list", out.number_list.as_ref().unwrap());
        assert_all_finite("date_list", out.date_list.as_ref().unwrap());

        // answer_slot_states with variable valid lengths (testing padding mask + tail clamp)
        let states = head
            .answer_slot_states(
                &hidden,
                &[16, 17, 18],
                &[0, 1, 2],
                &[
                    vec![24, 25, 26, 27],
                    vec![25, 26, 27, 28],
                    vec![26, 27, 28, 29],
                ],
                &[2, 4, 1],
                stop,
                &memory,
            )
            .unwrap();
        assert_eq!(states.shape().unwrap().as_ref(), &[3, 4, cfg.hidden_size]);
        assert_all_finite("answer_slot_states", &states);
        // For row 0 (length = 2), slots 2 and 3 must equal slot 1 (clamped tail).
        let flat = states.to_float32().unwrap();
        let h = cfg.hidden_size as usize;
        let row0_slot1 = &flat[h..2 * h];
        let row0_slot2 = &flat[2 * h..3 * h];
        let row0_slot3 = &flat[3 * h..4 * h];
        assert_eq!(row0_slot1, row0_slot2);
        assert_eq!(row0_slot1, row0_slot3);
    }

    #[test]
    fn golden_parity_against_upstream_pytorch_when_env_set() {
        let Ok(root) = std::env::var("CLAIRIC_GOLDEN_DIR") else {
            return;
        };
        let use_cpu = std::env::var("CLAIRIC_GOLDEN_CPU").as_deref() == Ok("1");
        let prev_dev = unsafe { mlx_sys::mlx_default_device() };
        if use_cpu {
            unsafe { mlx_sys::mlx_set_default_device(0) };
        }
        let golden_file = if use_cpu {
            "golden_torch.json"
        } else {
            "golden.json"
        };
        let tol = if use_cpu { 1e-5 } else { 2e-4 };
        let root = Path::new(&root);
        for (subdir, round_bf16) in [("fp32", false), ("bf16", true)] {
            let dir = root.join(subdir);
            let head = ClairicHead::load(&dir, 64, round_bf16).unwrap();
            let golden: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(dir.join(golden_file)).unwrap())
                    .unwrap();

            let h_shape: Vec<i64> = golden["hidden"]["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect();
            let h_data: Vec<f32> = golden["hidden"]["data"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let hidden = MxArray::from_float32(&h_data, &h_shape).unwrap();
            let input_start = golden["input_start"].as_i64().unwrap();
            let stop = golden["stop"].as_i64().unwrap();
            let memory = head.memory(&hidden, stop).unwrap();

            let views = PredictionViews {
                query_positions: vec![16, 17, 18, 19, 20, 21, 22, 23],
                text_rows: vec![0],
                tags_rows: vec![1],
                number_rows: vec![2],
                date_rows: vec![3],
                time_rows: vec![4],
                estimate_rows: vec![5],
                number_list_rows: vec![6],
                date_list_rows: vec![7],
                number_list_items: vec![vec![24, 25, 26]],
                date_list_items: vec![vec![27, 28, 29]],
            };
            let out = head
                .forward(&hidden, input_start, stop, &views, &memory)
                .unwrap();

            let mut worst_label = String::new();
            let mut worst_err = 0.0f64;
            let mut check = |label: &str, actual: &MxArray, expected: &serde_json::Value| {
                let exp_shape: Vec<i64> = expected["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_i64().unwrap())
                    .collect();
                assert_eq!(
                    actual.shape().unwrap().as_ref(),
                    exp_shape.as_slice(),
                    "{subdir}/{label} shape mismatch"
                );
                let act_vals = actual.to_float32().unwrap();
                let exp_arr = expected["data"].as_array().unwrap();
                assert_eq!(
                    act_vals.len(),
                    exp_arr.len(),
                    "{subdir}/{label} len mismatch"
                );
                let mut max_err = 0.0f64;
                for (idx, (&a, e)) in act_vals.iter().zip(exp_arr.iter()).enumerate() {
                    match e.as_f64() {
                        None => assert!(
                            !a.is_finite(),
                            "{subdir}/{label}[{idx}] expected non-finite, got {a}"
                        ),
                        Some(ev) => {
                            assert!(
                                a.is_finite(),
                                "{subdir}/{label}[{idx}] expected {ev}, got {a}"
                            );
                            let err = (f64::from(a) - ev).abs();
                            if err > max_err {
                                max_err = err;
                            }
                        }
                    }
                }
                eprintln!("golden parity [{subdir}] {label}: max_abs_err = {max_err:.3e}");
                if max_err > worst_err {
                    worst_err = max_err;
                    worst_label = format!("{subdir}/{label}");
                }
            };

            let outputs = &golden["outputs"];
            check("null", &out.null, &outputs["null"]);
            check("trim", out.trim.as_ref().unwrap(), &outputs["trim"]);
            check("text", out.text.as_ref().unwrap(), &outputs["text"]);
            check("tags", out.tags.as_ref().unwrap(), &outputs["tags"]);
            check("number", out.number.as_ref().unwrap(), &outputs["number"]);
            check(
                "number_groups",
                out.number_groups.as_ref().unwrap(),
                &outputs["number_groups"],
            );
            check(
                "number_copy",
                out.number_copy.as_ref().unwrap(),
                &outputs["number_copy"],
            );
            check(
                "number_copy_gate",
                out.number_copy_gate.as_ref().unwrap(),
                &outputs["number_copy_gate"],
            );
            check("date", out.date.as_ref().unwrap(), &outputs["date"]);
            check("time", out.time.as_ref().unwrap(), &outputs["time"]);
            check(
                "estimate",
                out.estimate.as_ref().unwrap(),
                &outputs["estimate"],
            );
            check(
                "estimate_offset",
                out.estimate_offset.as_ref().unwrap(),
                &outputs["estimate_offset"],
            );
            check(
                "estimate_overflow",
                out.estimate_overflow.as_ref().unwrap(),
                &outputs["estimate_overflow"],
            );
            check(
                "estimate_number_raw",
                out.estimate_number_raw.as_ref().unwrap(),
                &outputs["estimate_number_raw"],
            );
            check(
                "estimate_number",
                out.estimate_number.as_ref().unwrap(),
                &outputs["estimate_number"],
            );
            check(
                "estimate_gate",
                out.estimate_gate.as_ref().unwrap(),
                &outputs["estimate_gate"],
            );
            check(
                "number_list",
                out.number_list.as_ref().unwrap(),
                &outputs["number_list"],
            );
            check(
                "number_list_end",
                out.number_list_end.as_ref().unwrap(),
                &outputs["number_list_end"],
            );
            check(
                "number_list_groups",
                out.number_list_groups.as_ref().unwrap(),
                &outputs["number_list_groups"],
            );
            check(
                "date_list",
                out.date_list.as_ref().unwrap(),
                &outputs["date_list"],
            );
            check(
                "date_list_end",
                out.date_list_end.as_ref().unwrap(),
                &outputs["date_list_end"],
            );

            let slot_states = head
                .answer_slot_states(
                    &hidden,
                    &[16, 17, 18],
                    &[0, 1, 2],
                    &[
                        vec![24, 25, 26, 27],
                        vec![25, 26, 27, 28],
                        vec![26, 27, 28, 29],
                    ],
                    &[2, 4, 1],
                    stop,
                    &memory,
                )
                .unwrap();
            check(
                "answer_slot_states",
                &slot_states,
                &golden["answer_slot_states"],
            );
            assert!(
                worst_err < tol,
                "{worst_label} max_abs_err {worst_err:.3e} >= {tol:.1e}"
            );
        }
        if use_cpu {
            unsafe { mlx_sys::mlx_set_default_device(prev_dev) };
        }
    }
}
