//! Inference-only port of Cloudflare JointSchemaHead (Apache-2.0).
use super::encoding::EncodedRecord;
use crate::{
    array::{DType, MxArray, scaled_dot_product_attention},
    nn::{Activations, LayerNorm},
};
use napi::{Error, Result};
use serde::Deserialize;
use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

#[derive(Deserialize)]
pub(crate) struct HeadConfig {
    pub hidden_size: i64,
    pub width: i64,
    pub routing_layers: usize,
    pub layers: usize,
    pub heads: i64,
    pub feedforward: i64,
}

pub(crate) struct JointHead {
    config: HeadConfig,
    weights: HashMap<String, MxArray>,
}

impl JointHead {
    pub fn load(path: &Path, hidden_size: i64) -> Result<Self> {
        let config: HeadConfig = serde_json::from_str(
            &std::fs::read_to_string(path.join("joint_head_config.json"))
                .map_err(|e| Error::from_reason(e.to_string()))?,
        )
        .map_err(|e| Error::from_reason(e.to_string()))?;
        let weights =
            crate::utils::safetensors::load_safetensors_lazy(path.join("joint_head.safetensors"))?;
        Self::from_weights(config, weights, hidden_size)
    }
    pub(crate) fn from_weights(
        config: HeadConfig,
        weights: HashMap<String, MxArray>,
        hidden_size: i64,
    ) -> Result<Self> {
        let c = &config;
        if c.hidden_size != hidden_size
            || c.width <= 0
            || c.heads <= 0
            || c.width % c.heads != 0
            || c.feedforward <= 0
            || c.layers > 32
            || c.routing_layers > 32
        {
            return Err(Error::from_reason(
                "Invalid CLEF head configuration or backbone hidden size",
            ));
        }
        let mut shapes: Vec<(String, Vec<i64>)> = Vec::new();
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
        fn attention(out: &mut Vec<(String, Vec<i64>)>, prefix: &str, w: i64) {
            out.push((format!("{prefix}.in_proj_weight"), vec![3 * w, w]));
            out.push((format!("{prefix}.in_proj_bias"), vec![3 * w]));
            linear(out, &format!("{prefix}.out_proj"), w, w, true);
        }
        norm(&mut shapes, "hidden_norm", c.hidden_size);
        for name in [
            "memory_projection",
            "question_projection",
            "option_question_projection",
            "global_projection",
            "option_context_projection",
            "option_lexical_projection",
        ] {
            linear(&mut shapes, name, c.hidden_size, c.width, false);
        }
        shapes.push(("type_embedding.weight".into(), vec![3, c.width]));
        for name in ["option_summary_norm", "field_norm", "option_norm"] {
            norm(&mut shapes, name, c.width);
        }
        for i in 0..c.routing_layers {
            let p = format!("evidence_layers.{i}");
            for name in ["query_norm", "memory_norm", "feedforward_norm"] {
                norm(&mut shapes, &format!("{p}.{name}"), c.width);
            }
            attention(&mut shapes, &format!("{p}.attention"), c.width);
            linear(
                &mut shapes,
                &format!("{p}.feedforward.0"),
                c.width,
                c.feedforward,
                true,
            );
            linear(
                &mut shapes,
                &format!("{p}.feedforward.3"),
                c.feedforward,
                c.width,
                true,
            );
        }
        for i in 0..c.layers {
            let p = format!("layers.{i}");
            for name in ["norm1", "norm2", "norm3"] {
                norm(&mut shapes, &format!("{p}.{name}"), c.width);
            }
            for name in ["self_attn", "multihead_attn"] {
                attention(&mut shapes, &format!("{p}.{name}"), c.width);
            }
            linear(
                &mut shapes,
                &format!("{p}.linear1"),
                c.width,
                c.feedforward,
                true,
            );
            linear(
                &mut shapes,
                &format!("{p}.linear2"),
                c.feedforward,
                c.width,
                true,
            );
        }
        linear(&mut shapes, "residual_scorer.0", c.width * 4, c.width, true);
        linear(&mut shapes, "residual_scorer.3", c.width, 1, true);
        for name in ["prior_logit_scale", "joint_logit_scale", "residual_gate"] {
            shapes.push((name.into(), vec![]));
        }
        let expected: HashSet<_> = shapes.iter().map(|(k, _)| k.as_str()).collect();
        for (name, shape) in &shapes {
            let weight = weights
                .get(name)
                .ok_or_else(|| Error::from_reason(format!("Missing CLEF head tensor: {name}")))?;
            if weight.shape()?.as_ref() != shape.as_slice()
                || !matches!(
                    weight.dtype()?,
                    DType::Float32 | DType::Float16 | DType::BFloat16
                )
            {
                return Err(Error::from_reason(format!(
                    "Invalid CLEF head tensor shape or dtype: {name}"
                )));
            }
        }
        if let Some(key) = weights.keys().find(|k| !expected.contains(k.as_str())) {
            return Err(Error::from_reason(format!(
                "Unexpected CLEF head tensor: {key}"
            )));
        }
        crate::array::memory::materialize_weights(&weights.values().collect::<Vec<_>>())?;
        Ok(Self { config, weights })
    }
    pub fn nbytes(&self) -> u64 {
        self.weights.values().map(|a| a.nbytes() as u64).sum()
    }
    fn w(&self, name: &str) -> &MxArray {
        &self.weights[name]
    }
    fn linear(&self, x: &MxArray, prefix: &str) -> Result<MxArray> {
        let weight = self.w(&format!("{prefix}.weight"));
        let y = x.matmul(&weight.transpose(None)?)?;
        match self.weights.get(&format!("{prefix}.bias")) {
            Some(b) => y.add(b),
            None => Ok(y),
        }
    }
    fn norm(&self, x: &MxArray, prefix: &str) -> Result<MxArray> {
        LayerNorm::from_weights(
            self.w(&format!("{prefix}.weight")),
            Some(self.w(&format!("{prefix}.bias"))),
            Some(1e-5),
        )?
        .forward(x)
    }
    fn attention(&self, q: &MxArray, memory: &MxArray, prefix: &str) -> Result<MxArray> {
        let w = self.config.width;
        let heads = self.config.heads;
        let weight = self.w(&format!("{prefix}.in_proj_weight"));
        let bias = self.w(&format!("{prefix}.in_proj_bias"));
        let project = |x: &MxArray, i: i64| -> Result<MxArray> {
            x.matmul(&weight.slice_axis(0, i * w, (i + 1) * w)?.transpose(None)?)?
                .add(&bias.slice_axis(0, i * w, (i + 1) * w)?)?
                .reshape(&[1, -1, heads, w / heads])?
                .transpose(Some(&[0, 2, 1, 3]))
        };
        let output = scaled_dot_product_attention(
            &project(q, 0)?,
            &project(memory, 1)?,
            &project(memory, 2)?,
            1.0 / ((w / heads) as f64).sqrt(),
            None,
        )?;
        self.linear(
            &output.transpose(Some(&[0, 2, 1, 3]))?.reshape(&[-1, w])?,
            &format!("{prefix}.out_proj"),
        )
    }
    /// Row-wise mean over `spans` of `x` in ONE gather pair: an fp32 exclusive
    /// prefix sum indexed at (start, end) beats one slice+mean op per span.
    /// fp32 accumulation also matches MLX's own reduction accumulation, so the
    /// astype back to the weight dtype lands within a ulp of slice.mean().
    fn span_means(x: &MxArray, dtype: DType, spans: &[(usize, usize)]) -> Result<MxArray> {
        let zero = MxArray::zeros(&[1, x.shape_at(1)?], Some(DType::Float32))?;
        let cumsum = MxArray::concatenate(&zero, &x.astype(DType::Float32)?.cumsum(0)?, 0)?;
        let starts: Vec<u32> = spans.iter().map(|s| s.0 as u32).collect();
        let ends: Vec<u32> = spans.iter().map(|s| s.1 as u32).collect();
        let lengths: Vec<f32> = spans.iter().map(|s| (s.1 - s.0) as f32).collect();
        let sums = cumsum
            .take(&MxArray::from_uint32(&ends, &[ends.len() as i64])?, 0)?
            .sub(&cumsum.take(&MxArray::from_uint32(&starts, &[starts.len() as i64])?, 0)?)?;
        sums.div(&MxArray::from_float32(
            &lengths,
            &[lengths.len() as i64, 1],
        )?)?
        .astype(dtype)
    }
    /// Mean-embed every (possibly ragged) contiguous id group in one take:
    /// gather all rows once, then span-mean over the flat layout.
    fn group_mean_embeds(
        &self,
        output_weight: &MxArray,
        ids: &[u32],
        lengths: &[usize],
        dtype: DType,
    ) -> Result<MxArray> {
        let embeds = output_weight.take(&MxArray::from_uint32(ids, &[ids.len() as i64])?, 0)?;
        let mut offset = 0;
        let spans: Vec<(usize, usize)> = lengths
            .iter()
            .map(|&len| {
                let span = (offset, offset + len);
                offset += len;
                span
            })
            .collect();
        Self::span_means(&embeds, dtype, &spans)
    }
    fn unit(x: &MxArray, eps: f64) -> Result<MxArray> {
        x.div(
            &x.mul(x)?
                .sum(Some(&[-1]), Some(true))?
                .sqrt()?
                .clip(Some(eps), None)?,
        )
    }
    pub fn forward(
        &self,
        hidden: &MxArray,
        record: &EncodedRecord,
        output_weight: &MxArray,
    ) -> Result<Vec<Vec<f64>>> {
        let hidden = hidden.astype(self.w("hidden_norm.weight").dtype()?)?;
        let h = self
            .norm(&hidden, "hidden_norm")?
            .reshape(&[-1, self.config.hidden_size])?;
        let dtype = h.dtype()?;
        let memory = self.linear(&h, "memory_projection")?;
        let global = h.slice_axis(0, h.shape_at(0)? - 1, h.shape_at(0)?)?;
        // Batched span means: question vectors [Q,H] and option contexts [O,H].
        let q_spans: Vec<(usize, usize)> = record.questions.iter().map(|q| q.span).collect();
        let questions = Self::span_means(&h, dtype, &q_spans)?;
        let option_spans: Vec<(usize, usize)> = record
            .questions
            .iter()
            .flat_map(|q| q.option_spans.iter().copied())
            .collect();
        let contexts = Self::span_means(&h, dtype, &option_spans)?;
        // Lexical embeddings: one take over every option token, one mean pass.
        let mut option_ids_flat: Vec<u32> = Vec::new();
        let mut option_lengths: Vec<usize> = Vec::new();
        for q in &record.questions {
            for &(start, end) in &q.option_spans {
                option_ids_flat.extend_from_slice(&record.input_ids[start..end]);
                option_lengths.push(end - start);
            }
        }
        let lexical =
            self.group_mean_embeds(output_weight, &option_ids_flat, &option_lengths, dtype)?;
        // Option→question gather index, reused by the projection and scoring.
        let mut question_index: Vec<u32> = Vec::with_capacity(option_spans.len());
        let counts: Vec<u32> = record
            .questions
            .iter()
            .map(|q| q.option_ids.len() as u32)
            .collect();
        for (i, &count) in counts.iter().enumerate() {
            question_index.extend(std::iter::repeat_n(i as u32, count as usize));
        }
        let option_count = option_spans.len() as i64;
        let option_of_question = MxArray::from_uint32(&question_index, &[option_count])?;
        let mut routed = self
            .linear(&contexts, "option_context_projection")?
            .add(&self.linear(&lexical, "option_lexical_projection")?)?
            .add(
                &self
                    .linear(&questions, "option_question_projection")?
                    .take(&option_of_question, 0)?,
            )?;
        for i in 0..self.config.routing_layers {
            let p = format!("evidence_layers.{i}");
            routed = routed.add(&self.attention(
                &self.norm(&routed, &format!("{p}.query_norm"))?,
                &self.norm(&memory, &format!("{p}.memory_norm"))?,
                &format!("{p}.attention"),
            )?)?;
            let ff = Activations::gelu_exact(&self.linear(
                &self.norm(&routed, &format!("{p}.feedforward_norm"))?,
                &format!("{p}.feedforward.0"),
            )?)?;
            routed = routed.add(&self.linear(&ff, &format!("{p}.feedforward.3"))?)?;
        }
        let base = self.linear(&questions, "question_projection")?;
        // Per-question option softmax, padded to [Q, max_options]: index O_tot
        // addresses a zero row, masked out before the softmax either way.
        let question_count = record.questions.len() as i64;
        let max_options = counts.iter().copied().max().unwrap_or(0) as i64;
        let mut padded_indices: Vec<u32> =
            Vec::with_capacity((question_count * max_options) as usize);
        let mut offset = 0u32;
        for &count in &counts {
            for j in 0..max_options as u32 {
                padded_indices.push(if j < count {
                    offset + j
                } else {
                    option_count as u32
                });
            }
            offset += count;
        }
        let routed_padded = MxArray::concatenate(
            &routed,
            &MxArray::zeros(&[1, self.config.width], Some(dtype))?,
            0,
        )?;
        let options_padded = routed_padded.take(
            &MxArray::from_uint32(&padded_indices, &[question_count, max_options])?,
            0,
        )?;
        let valid = MxArray::arange(0.0, max_options as f64, Some(1.0), None)?
            .reshape(&[1, max_options])?
            .less(&MxArray::from_uint32(&counts, &[question_count, 1])?.astype(DType::Float32)?)?;
        let neg_inf = MxArray::full(
            &[question_count, max_options],
            napi::Either::A(f64::NEG_INFINITY),
            Some(dtype),
        )?;
        let scores = base
            .expand_dims(1)?
            .matmul(&options_padded.transpose(Some(&[0, 2, 1]))?)?
            .reshape(&[question_count, max_options])?
            .div_scalar((self.config.width as f64).sqrt())?;
        let option_weights = Activations::softmax(&valid.where_(&scores, &neg_inf)?, Some(-1))?;
        let summaries = option_weights
            .expand_dims(-1)?
            .mul(&options_padded)?
            .sum(Some(&[1]), Some(false))?;
        let types: Vec<_> = record.questions.iter().map(|q| q.kind).collect();
        let types = MxArray::from_uint32(&types, &[types.len() as i64])?;
        let mut fields = base
            .add(&self.norm(&summaries, "option_summary_norm")?)?
            .add(&self.linear(&global, "global_projection")?)?
            .add(&self.w("type_embedding.weight").take(&types, 0)?)?;
        for i in 0..self.config.layers {
            let p = format!("layers.{i}");
            let norm = self.norm(&fields, &format!("{p}.norm1"))?;
            fields = fields.add(&self.attention(&norm, &norm, &format!("{p}.self_attn"))?)?;
            fields = fields.add(&self.attention(
                &self.norm(&fields, &format!("{p}.norm2"))?,
                &memory,
                &format!("{p}.multihead_attn"),
            )?)?;
            let ff = Activations::gelu_exact(&self.linear(
                &self.norm(&fields, &format!("{p}.norm3"))?,
                &format!("{p}.linear1"),
            )?)?;
            fields = fields.add(&self.linear(&ff, &format!("{p}.linear2"))?)?;
        }
        fields = self.norm(&fields, "field_norm")?;
        // Scoring flattened over every (question, option) pair; one device
        // readback at the end instead of a GPU sync per question.
        let prior_scale = self
            .w("prior_logit_scale")
            .clip(None, Some(100f64.ln()))?
            .exp()?;
        let joint_scale = self
            .w("joint_logit_scale")
            .clip(None, Some(100f64.ln()))?
            .exp()?;
        let gate = Activations::sigmoid(self.w("residual_gate"))?;
        let anchors = Self::unit(&questions.add(&global)?, 1e-12)?.take(&option_of_question, 0)?;
        let prior = Self::unit(&lexical, 1e-12)?
            .mul(&anchors)?
            .sum(Some(&[-1]), Some(false))?
            .mul(&prior_scale)?;
        let options = self.norm(&routed, "option_norm")?;
        let field = fields.take(&option_of_question, 0)?;
        let cosine = Self::unit(&field, 1e-8)?
            .mul(&Self::unit(&options, 1e-8)?)?
            .sum(Some(&[-1]), Some(false))?;
        let product = field.mul(&options)?;
        let distance = field.sub(&options)?.abs()?;
        let features =
            MxArray::concatenate_many(vec![&field, &options, &product, &distance], Some(-1))?;
        let residual = self
            .linear(
                &Activations::gelu_exact(&self.linear(&features, "residual_scorer.0")?)?,
                "residual_scorer.3",
            )?
            .squeeze(Some(&[-1]))?;
        let joint = cosine.mul(&joint_scale)?.add(&residual)?;
        let logits_flat = prior.add(&joint.mul(&gate)?)?;
        // Scatter flat logits into [Q, max_options]; pad slots read index
        // O_tot (a -inf row) so the masked softmax is exact per question.
        let logits_padded = MxArray::concatenate(
            &logits_flat,
            &MxArray::full(&[1], napi::Either::A(f64::NEG_INFINITY), Some(dtype))?,
            0,
        )?;
        let logits = logits_padded.take(
            &MxArray::from_uint32(&padded_indices, &[question_count, max_options])?,
            0,
        )?;
        let probabilities_padded = Activations::softmax_precise(
            &valid.where_(&logits, &neg_inf)?.astype(DType::Float32)?,
            Some(-1),
        )?;
        let flat: Vec<f32> = probabilities_padded.to_float32()?.to_vec();
        let mut results = Vec::with_capacity(record.questions.len());
        for (i, &count) in counts.iter().enumerate() {
            let row = &flat[i * max_options as usize..(i + 1) * max_options as usize];
            let probabilities: Vec<f64> = row[..count as usize]
                .iter()
                .map(|&v| f64::from(v))
                .collect();
            if probabilities.iter().any(|p| !p.is_finite()) {
                return Err(Error::from_reason("CLEF produced non-finite probabilities"));
            }
            results.push(probabilities);
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::clef::encoding::EncodedQuestion;
    #[test]
    fn joint_head_matches_independent_pytorch_fixture() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../__test__/fixtures/clef");
        let head = JointHead::load(&root, 8).unwrap();
        let fixture: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("head.json")).unwrap())
                .unwrap();
        let array = |name: &str, shape: &[i64]| {
            let data: Vec<f32> = fixture[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            MxArray::from_float32(&data, shape).unwrap()
        };
        let record = EncodedRecord {
            input_ids: (0..24).collect(),
            questions: vec![
                EncodedQuestion {
                    id: "10".into(),
                    kind: 1,
                    span: (1, 3),
                    option_spans: vec![(3, 5), (5, 7)],
                    option_ids: vec!["a".into(), "b".into()],
                },
                EncodedQuestion {
                    id: "2".into(),
                    kind: 2,
                    span: (7, 9),
                    option_spans: vec![(9, 11), (11, 13), (13, 15)],
                    option_ids: vec!["0".into(), "1".into(), "2".into()],
                },
                EncodedQuestion {
                    id: "flag".into(),
                    kind: 0,
                    span: (15, 17),
                    option_spans: vec![(17, 19), (19, 21)],
                    option_ids: vec!["true".into(), "false".into()],
                },
            ],
        };
        let actual = head
            .forward(
                &array("hidden", &[1, 24, 8]),
                &record,
                &array("output", &[32, 8]),
            )
            .unwrap();
        let max_error = actual
            .iter()
            .enumerate()
            .flat_map(|(q, values)| values.iter().enumerate().map(move |(i, &v)| (q, i, v)))
            .map(|(q, i, v)| (v - fixture["probabilities"][q][i].as_f64().unwrap()).abs())
            .fold(0.0f64, f64::max);
        eprintln!("CLEF Metal probability max absolute error: {max_error}");
        assert!(max_error < 1e-4, "Metal head drift: {max_error}");
        assert!(JointHead::load(&root, 16).is_err());
        let mut weights =
            crate::utils::safetensors::load_safetensors_lazy(root.join("joint_head.safetensors"))
                .unwrap();
        weights.remove("layers.0.norm1.bias");
        let config = serde_json::from_str(
            &std::fs::read_to_string(root.join("joint_head_config.json")).unwrap(),
        )
        .unwrap();
        assert!(JointHead::from_weights(config, weights, 8).is_err());
    }
}
