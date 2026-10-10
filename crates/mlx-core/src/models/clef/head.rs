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
    fn mean_span(x: &MxArray, span: (usize, usize)) -> Result<MxArray> {
        x.slice_axis(0, span.0 as i64, span.1 as i64)?
            .mean(Some(&[0]), Some(true))
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
        let memory = self.linear(&h, "memory_projection")?;
        let global = h.slice_axis(0, h.shape_at(0)? - 1, h.shape_at(0)?)?;
        let qvectors = record
            .questions
            .iter()
            .map(|q| Self::mean_span(&h, q.span))
            .collect::<Result<Vec<_>>>()?;
        let questions = MxArray::concatenate_many(qvectors.iter().collect(), Some(0))?;
        let mut lexical = Vec::new();
        let mut option_queries = Vec::new();
        for (i, q) in record.questions.iter().enumerate() {
            let mut contexts = Vec::new();
            let mut words = Vec::new();
            for &span in &q.option_spans {
                contexts.push(Self::mean_span(&h, span)?);
                let ids = &record.input_ids[span.0..span.1];
                let ids = MxArray::from_uint32(ids, &[ids.len() as i64])?;
                words.push(
                    output_weight
                        .take(&ids, 0)?
                        .mean(Some(&[0]), Some(true))?
                        .astype(h.dtype()?)?,
                );
            }
            let contexts = MxArray::concatenate_many(contexts.iter().collect(), Some(0))?;
            let words = MxArray::concatenate_many(words.iter().collect(), Some(0))?;
            option_queries.push(
                self.linear(&contexts, "option_context_projection")?
                    .add(&self.linear(&words, "option_lexical_projection")?)?
                    .add(&self.linear(&qvectors[i], "option_question_projection")?)?,
            );
            lexical.push(words);
        }
        let mut routed = MxArray::concatenate_many(option_queries.iter().collect(), Some(0))?;
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
        let mut offset = 0;
        let mut splits = Vec::new();
        let mut summaries = Vec::new();
        for (i, q) in record.questions.iter().enumerate() {
            let count = q.option_ids.len() as i64;
            let options = routed.slice_axis(0, offset, offset + count)?;
            offset += count;
            let weights = Activations::softmax(
                &options
                    .matmul(
                        &base
                            .slice_axis(0, i as i64, i as i64 + 1)?
                            .transpose(None)?,
                    )?
                    .div_scalar((self.config.width as f64).sqrt())?,
                Some(0),
            )?;
            summaries.push(weights.mul(&options)?.sum(Some(&[0]), Some(true))?);
            splits.push(options);
        }
        let types: Vec<_> = record.questions.iter().map(|q| q.kind).collect();
        let types = MxArray::from_uint32(&types, &[types.len() as i64])?;
        let summaries = MxArray::concatenate_many(summaries.iter().collect(), Some(0))?;
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
        let mut results = Vec::new();
        for (i, options) in splits.iter().enumerate() {
            let anchor = Self::unit(&qvectors[i].add(&global)?, 1e-12)?;
            let prior = Self::unit(&lexical[i], 1e-12)?
                .matmul(&anchor.transpose(None)?)?
                .squeeze(Some(&[-1]))?
                .mul(
                    &self
                        .w("prior_logit_scale")
                        .clip(None, Some(100f64.ln()))?
                        .exp()?,
                )?;
            let options = self.norm(options, "option_norm")?;
            let field = fields
                .slice_axis(0, i as i64, i as i64 + 1)?
                .broadcast_to(options.shape()?.as_ref())?;
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
            let joint = cosine
                .mul(
                    &self
                        .w("joint_logit_scale")
                        .clip(None, Some(100f64.ln()))?
                        .exp()?,
                )?
                .add(&residual)?;
            let logits = prior.add(&joint.mul(&Activations::sigmoid(self.w("residual_gate"))?)?)?;
            let probabilities =
                Activations::softmax_precise(&logits.astype(DType::Float32)?, Some(-1))?
                    .to_float32()?
                    .iter()
                    .map(|v| f64::from(*v))
                    .collect::<Vec<_>>();
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
