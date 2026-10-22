use super::{
    config::{Config, FAMILY, Options, PAIR_SPECIAL_TOKENS, PROFILE, Policy, invalid},
    weights::Weights,
};
use crate::{
    array::{self, DType, MxArray, attention::scaled_dot_product_attention},
    models::{
        quantized_linear::LinearProj,
        rerank_runtime::{Backend, Request},
    },
    nn::{Embedding, LayerNorm},
};
use napi::{Error, Result};
use serde_json::{Value, json};
use std::{
    path::Path,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};
use tokenizers::{Encoding, Tokenizer};

struct Layer {
    q: LinearProj,
    k: LinearProj,
    v: LinearProj,
    o: LinearProj,
    attention_norm: LayerNorm,
    up: LinearProj,
    down: LinearProj,
    output_norm: LayerNorm,
}
pub(super) struct Model {
    config: Config,
    policy: Policy,
    tokenizer: Tokenizer,
    compute_dtype: DType,
    words: Embedding,
    positions: Embedding,
    types: Embedding,
    embedding_norm: LayerNorm,
    layers: Vec<Layer>,
    head: LinearProj,
    output: LinearProj,
    gelu: ExactGelu,
    pub info: Value,
    _cache_limit_guard: crate::cache_limit::CacheLimitGuard,
}
struct Pair {
    index: usize,
    encoding: Encoding,
}
struct Prepared {
    pairs: Vec<Pair>,
    input_tokens: usize,
    prompt_tokens: usize,
    truncated_query: bool,
    truncated_documents: Vec<usize>,
}
fn check_cancel(cancel: &AtomicBool, shutdown: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Acquire) || shutdown.load(Ordering::Acquire) {
        Err(Error::from_reason("rerank_cancelled: request cancelled"))
    } else {
        Ok(())
    }
}
fn too_large(message: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("rerank_input_too_large: {message}"))
}
fn prefix(e: &Encoding, n: usize) -> Encoding {
    // Copy only retained tokens, without creating overflow encodings or decoding.
    Encoding::new(
        e.get_ids()[..n].to_vec(),
        e.get_type_ids()[..n].to_vec(),
        e.get_tokens()[..n].to_vec(),
        e.get_word_ids()[..n].to_vec(),
        e.get_offsets()[..n].to_vec(),
        e.get_special_tokens_mask()[..n].to_vec(),
        e.get_attention_mask()[..n].to_vec(),
        vec![],
        Default::default(),
    )
}
pub(super) fn sigmoid(logit: f32) -> Result<f64> {
    if !logit.is_finite() {
        return Err(Error::from_reason(
            "rerank_numerical_error: non-finite classification output",
        ));
    }
    let x = f64::from(logit);
    Ok(if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    })
}
// XLM-R's MLX reference divides by sqrt(2) before erf. Multiplication by
// a dtype-rounded reciprocal is not equivalent in BF16. Keep this composition
// local: changing the shared GELU would alter other model families.
fn gelu_eager(x: &MxArray) -> Result<MxArray> {
    x.mul(
        &x.div_scalar(std::f64::consts::SQRT_2)?
            .erf()?
            .add_scalar(1.0)?,
    )?
    .div_scalar(2.0)
}
// Reuse MLX's compiler bridge to match nn.gelu's fusion and rounding. Each
// model owns its tape; no weights are captured and unloading erases the entry.
struct ExactGelu(u64);
impl ExactGelu {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(0xB6E0_0000_0000_0000 | NEXT.fetch_add(1, Ordering::Relaxed))
    }
    fn forward(&self, x: &MxArray) -> Result<MxArray> {
        let mut builder = |inputs: &[MxArray]| Ok(vec![gelu_eager(&inputs[0])?]);
        match crate::compiled_graph::invoke_compiled_graph(self.0, &[x], 1, true, &mut builder)? {
            Some(mut outputs) => Ok(outputs.remove(0)),
            None => Err(Error::from_reason(
                "rerank_worker_failed: GELU graph execution failed",
            )),
        }
    }
}
impl Drop for ExactGelu {
    fn drop(&mut self) {
        crate::compiled_graph::erase_compiled_graphs_matching(u64::MAX, self.0);
    }
}
fn position_ids(ids: &[u32], pad: u32) -> Vec<u32> {
    let mut position = pad;
    ids.iter()
        .map(|&id| {
            if id == pad {
                pad
            } else {
                position += 1;
                position
            }
        })
        .collect()
}
fn batch_end(pairs: &[Pair], start: usize, policy: &Policy) -> usize {
    let longest = pairs[start].encoding.len();
    let count = policy
        .max_batch_size
        .min(policy.max_batch_tokens / longest)
        .min(pairs.len() - start);
    start + count
}
impl Model {
    pub fn load(path: &Path, options: Options) -> Result<Self> {
        let raw: Value = serde_json::from_str(
            &std::fs::read_to_string(path.join("config.json")).map_err(invalid)?,
        )
        .map_err(invalid)?;
        let config = Config::parse(&raw)?;
        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json")).map_err(invalid)?;
        if tokenizer.get_truncation().is_some() || tokenizer.get_padding().is_some() {
            return Err(invalid(
                "tokenizer must not have automatic padding or truncation",
            ));
        }
        if tokenizer
            .get_vocab(true)
            .values()
            .any(|&id| i64::from(id) >= config.vocab_size)
        {
            return Err(invalid("tokenizer vocabulary exceeds embedding dimensions"));
        }
        for (name, id) in [
            ("<s>", config.bos_token_id),
            ("</s>", config.eos_token_id),
            ("<pad>", config.pad_token_id),
        ] {
            if tokenizer.token_to_id(name) != Some(id) {
                return Err(invalid("tokenizer special IDs conflict with config"));
            }
        }
        // Validate the actual post-processor, including the empty second sequence.
        for text in ["", "probe"] {
            let q = tokenizer.encode("query", false).map_err(invalid)?;
            let d = tokenizer.encode(text, false).map_err(invalid)?;
            let pair = tokenizer
                .post_process(q.clone(), Some(d.clone()), true)
                .map_err(invalid)?;
            let mut expected = vec![config.bos_token_id];
            expected.extend_from_slice(q.get_ids());
            expected.extend([config.eos_token_id, config.eos_token_id]);
            expected.extend_from_slice(d.get_ids());
            expected.push(config.eos_token_id);
            if pair.get_ids() != expected
                || pair.get_type_ids().iter().any(|&v| v != 0)
                || pair.get_attention_mask().iter().any(|&v| v != 1)
            {
                return Err(invalid("unsupported XLM-R pair post-processor"));
            }
        }
        let tc: Value = serde_json::from_str(
            &std::fs::read_to_string(path.join("tokenizer_config.json")).map_err(invalid)?,
        )
        .map_err(invalid)?;
        let context = tc["model_max_length"]
            .as_u64()
            .filter(|&n| n > 0)
            .map(|n| n.min(config.context() as u64) as usize)
            .unwrap_or(config.context());
        let policy = Policy::resolve(options, context)?;
        let weights = Weights::new(path, &raw)?;
        weights.validate_keys(config.num_hidden_layers)?;
        let compute_dtype = weights.compute_dtype(&raw)?;
        let h = config.hidden_size;
        let words = weights.embedding(
            "roberta.embeddings.word_embeddings",
            config.vocab_size,
            h,
            true,
        )?;
        let positions = weights.embedding(
            "roberta.embeddings.position_embeddings",
            config.max_position_embeddings as i64,
            h,
            false,
        )?;
        let types = weights.embedding(
            "roberta.embeddings.token_type_embeddings",
            config.type_vocab_size,
            h,
            false,
        )?;
        let embedding_norm =
            weights.norm("roberta.embeddings.LayerNorm", h, config.layer_norm_eps)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let base = format!("roberta.encoder.layer.{i}");
            let linear =
                |name: &str, rows, cols| weights.linear(&format!("{base}.{name}"), rows, cols);
            let norm =
                |name: &str| weights.norm(&format!("{base}.{name}"), h, config.layer_norm_eps);
            layers.push(Layer {
                q: linear("attention.self.query", h, h)?,
                k: linear("attention.self.key", h, h)?,
                v: linear("attention.self.value", h, h)?,
                o: linear("attention.output.dense", h, h)?,
                attention_norm: norm("attention.output.LayerNorm")?,
                up: linear("intermediate.dense", config.intermediate_size, h)?,
                down: linear("output.dense", h, config.intermediate_size)?,
                output_norm: norm("output.LayerNorm")?,
            });
        }
        let head = weights.linear("classifier.dense", h, h)?;
        let output = weights.linear("classifier.out_proj", 1, h)?;
        let info = json!({"family":FAMILY,"profile":PROFILE,"execution":"cross-encoder","scoreType":"sigmoid","contextWindow":context,"policy":policy,"quantized":weights.tensors.keys().any(|k|k.ends_with(".scales")),"weightFormats":weights.formats()});
        let bytes = weights.tensors.values().map(|w| w.nbytes() as u64).sum();
        let guard = crate::cache_limit::coordinator().register(bytes);
        for w in weights.tensors.values() {
            w.eval();
        }
        array::synchronize();
        Ok(Self {
            config,
            policy,
            tokenizer,
            compute_dtype,
            words,
            positions,
            types,
            embedding_norm,
            layers,
            head,
            output,
            gelu: ExactGelu::new(),
            info,
            _cache_limit_guard: guard,
        })
    }
    fn prepare(
        &self,
        request: &Request,
        cancel: &AtomicBool,
        shutdown: &AtomicBool,
    ) -> Result<Prepared> {
        if request.query.trim().is_empty() || request.top_n == Some(0) {
            return Err(Error::from_reason(
                "rerank_invalid_request: nonempty query and positive topN required",
            ));
        }
        if request.documents.len() > self.policy.max_documents {
            return Err(too_large("candidate limit exceeded"));
        }
        let q = self
            .tokenizer
            .encode(request.query.as_str(), false)
            .map_err(invalid)?;
        let mut input_tokens = q.len();
        if input_tokens > self.policy.max_input_tokens {
            return Err(too_large("total input token limit exceeded"));
        }
        let truncated_query = q.len() > self.policy.max_query_tokens;
        if truncated_query && !request.truncate {
            return Err(too_large("query token limit exceeded"));
        }
        let q = prefix(&q, q.len().min(self.policy.max_query_tokens));
        let doc_limit = self
            .policy
            .max_document_tokens
            .min(self.policy.max_pair_tokens - q.len() - PAIR_SPECIAL_TOKENS);
        let mut pairs = Vec::with_capacity(request.documents.len());
        let mut truncated_documents = Vec::new();
        let mut prompt_tokens = 0;
        for (index, text) in request.documents.iter().enumerate() {
            check_cancel(cancel, shutdown)?;
            let d = self
                .tokenizer
                .encode(text.as_str(), false)
                .map_err(invalid)?;
            input_tokens = input_tokens
                .checked_add(d.len())
                .ok_or_else(|| too_large("token count overflow"))?;
            if input_tokens > self.policy.max_input_tokens {
                return Err(too_large("total input token limit exceeded"));
            }
            if d.len() > doc_limit {
                if !request.truncate {
                    return Err(too_large(format!(
                        "document {index} exceeds document or pair token limit"
                    )));
                }
                truncated_documents.push(index);
            }
            let d = prefix(&d, d.len().min(doc_limit));
            let encoding = self
                .tokenizer
                .post_process(q.clone(), Some(d), true)
                .map_err(invalid)?;
            if encoding.len() > self.policy.max_pair_tokens {
                return Err(invalid("post-processor exceeded validated pair overhead"));
            }
            prompt_tokens += encoding.len();
            pairs.push(Pair { index, encoding });
        }
        pairs.sort_by(|a, b| {
            b.encoding
                .len()
                .cmp(&a.encoding.len())
                .then(a.index.cmp(&b.index))
        });
        Ok(Prepared {
            pairs,
            input_tokens,
            prompt_tokens,
            truncated_query,
            truncated_documents,
        })
    }
    fn forward(
        &self,
        pairs: &[Pair],
        cancel: &AtomicBool,
        shutdown: &AtomicBool,
    ) -> Result<Vec<f32>> {
        self.forward_observed(pairs, cancel, shutdown, |_| Ok(()))
    }
    fn forward_observed(
        &self,
        pairs: &[Pair],
        cancel: &AtomicBool,
        shutdown: &AtomicBool,
        mut observe: impl FnMut(&MxArray) -> Result<()>,
    ) -> Result<Vec<f32>> {
        check_cancel(cancel, shutdown)?;
        let batch = pairs.len() as i64;
        let length = pairs
            .iter()
            .map(|p| p.encoding.len())
            .max()
            .ok_or_else(|| invalid("empty batch"))?;
        let mut ids = Vec::with_capacity(pairs.len() * length);
        let mut pos = Vec::with_capacity(ids.capacity());
        let mut types = Vec::with_capacity(ids.capacity());
        let mut mask = Vec::with_capacity(ids.capacity());
        for p in pairs {
            ids.extend_from_slice(p.encoding.get_ids());
            pos.extend(position_ids(p.encoding.get_ids(), self.config.pad_token_id));
            types.extend_from_slice(p.encoding.get_type_ids());
            mask.extend(
                p.encoding
                    .get_attention_mask()
                    .iter()
                    .map(|&v| if v == 0 { f32::NEG_INFINITY } else { 0.0 }),
            );
            let pad = length - p.encoding.len();
            ids.extend(std::iter::repeat_n(self.config.pad_token_id, pad));
            pos.extend(std::iter::repeat_n(self.config.pad_token_id, pad));
            types.extend(std::iter::repeat_n(0, pad));
            mask.extend(std::iter::repeat_n(f32::NEG_INFINITY, pad));
        }
        let shape = [batch, length as i64];
        // Match HF's addition order: word + token type, then position.
        let mut x = self.embedding_norm.forward(
            &self
                .words
                .forward(&MxArray::from_uint32(&ids, &shape)?)?
                .astype(self.compute_dtype)?
                .add(&self.types.forward(&MxArray::from_uint32(&types, &shape)?)?)?
                .add(
                    &self
                        .positions
                        .forward(&MxArray::from_uint32(&pos, &shape)?)?,
                )?,
        )?;
        let mask =
            MxArray::from_float32(&mask, &[batch, 1, 1, length as i64])?.astype(x.dtype()?)?;
        let heads = self.config.num_attention_heads;
        let dim = self.config.hidden_size / heads;
        let project = |proj: &LinearProj, x: &MxArray| {
            proj.forward(x)?
                .reshape(&[batch, length as i64, heads, dim])?
                .transpose(Some(&[0, 2, 1, 3]))
        };
        x.eval();
        check_cancel(cancel, shutdown)?;
        observe(&x)?;
        for layer in &self.layers {
            let q = project(&layer.q, &x)?;
            let k = project(&layer.k, &x)?;
            let v = project(&layer.v, &x)?;
            let attended =
                scaled_dot_product_attention(&q, &k, &v, 1.0 / (dim as f64).sqrt(), Some(&mask))?
                    .transpose(Some(&[0, 2, 1, 3]))?
                    .reshape(&[batch, length as i64, self.config.hidden_size])?;
            x = layer
                .attention_norm
                .forward(&layer.o.forward(&attended)?.add(&x)?)?;
            x = layer.output_norm.forward(
                &layer
                    .down
                    .forward(&self.gelu.forward(&layer.up.forward(&x)?)?)?
                    .add(&x)?,
            )?;
            x.eval();
            check_cancel(cancel, shutdown)?;
            observe(&x)?;
        }
        let cls = x
            .slice(&[0, 0, 0], &[batch, 1, self.config.hidden_size])?
            .reshape(&[batch, self.config.hidden_size])?;
        let logits = self
            .output
            .forward(&self.head.forward(&cls)?.tanh()?)?
            .astype(DType::Float32)?;
        logits.eval();
        check_cancel(cancel, shutdown)?;
        Ok(logits.to_float32()?.to_vec())
    }
}
impl Backend for Model {
    fn info(&self) -> &Value {
        &self.info
    }
    fn rerank(
        &self,
        request: Request,
        cancel: &AtomicBool,
        shutdown: &AtomicBool,
    ) -> Result<Value> {
        check_cancel(cancel, shutdown)?;
        let prepared = self.prepare(&request, cancel, shutdown)?;
        let mut results = Vec::with_capacity(prepared.pairs.len());
        let mut batches = 0;
        let mut padded_tokens = 0;
        let mut start = 0;
        while start < prepared.pairs.len() {
            let end = batch_end(&prepared.pairs, start, &self.policy);
            let pairs = &prepared.pairs[start..end];
            padded_tokens += pairs.len() * pairs[0].encoding.len();
            for (pair, logit) in pairs.iter().zip(self.forward(pairs, cancel, shutdown)?) {
                results.push((pair.index, sigmoid(logit)?));
            }
            batches += 1;
            start = end;
        }
        results.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        results.truncate(request.top_n.unwrap_or(results.len()).min(results.len()));
        Ok(
            json!({"results":results.into_iter().map(|(index,score)|json!({"index":index,"relevanceScore":score})).collect::<Vec<_>>(),"usage":{"inputTokens":prepared.input_tokens,"promptTokens":prepared.prompt_tokens},"meta":{"profile":PROFILE,"processedDocuments":prepared.pairs.len(),"batches":batches,"maxPairTokens":self.policy.max_pair_tokens,"maxBatchSize":self.policy.max_batch_size,"maxBatchTokens":self.policy.max_batch_tokens,"paddedTokens":padded_tokens,"paddingTokens":padded_tokens-prepared.prompt_tokens,"truncatedQuery":prepared.truncated_query,"truncatedDocumentIndices":prepared.truncated_documents}}),
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sigmoid_is_stable_and_rejects_nonfinite() {
        assert_eq!(sigmoid(0.0).unwrap(), 0.5);
        assert_eq!(sigmoid(f32::MAX).unwrap(), 1.0);
        assert_eq!(sigmoid(-f32::MAX).unwrap(), 0.0);
        assert!(sigmoid(20.0).unwrap() < 1.0);
        for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(sigmoid(x).is_err());
        }
        for i in -100..100 {
            assert!(sigmoid(i as f32).unwrap() <= sigmoid((i + 1) as f32).unwrap());
        }
    }
    #[test]
    fn literal_pad_only_affects_positions() {
        assert_eq!(position_ids(&[0, 52, 1, 2, 1], 1), [2, 3, 1, 4, 1]);
    }
}

#[cfg(test)]
mod golden_tests {
    use super::*;
    #[test]
    #[ignore = "requires pinned BGE checkpoint and independent reference artifacts; see scripts/rerank/bge/README.md"]
    fn golden_checkpoint_logits() {
        let path = std::env::var("MLX_BGE_GOLDEN_MODEL").expect("model path");
        let reference = std::env::var("MLX_BGE_GOLDEN_REFERENCE").expect("reference JSON");
        let reference: Value =
            serde_json::from_str(&std::fs::read_to_string(reference).unwrap()).unwrap();
        let cases_path = std::env::var("MLX_BGE_GOLDEN_CASES").ok();
        let cases_text = cases_path
            .map(|p| std::fs::read_to_string(p).unwrap())
            .unwrap_or_else(|| {
                include_str!("../../../../../fixtures/rerank/bge/cases.json").into()
            });
        let cases: Value = serde_json::from_str(&cases_text).unwrap();
        let manifest: Value = serde_json::from_str(include_str!(
            "../../../../../fixtures/rerank/bge/manifest.json"
        ))
        .unwrap();
        let threshold = &manifest["thresholds"][if reference["backend"] == "torch" {
            "fp32"
        } else {
            "sameCheckpoint"
        }];
        let cancelled = AtomicBool::new(false);
        assert_eq!(
            cases.as_array().unwrap().len(),
            reference["cases"].as_array().unwrap().len()
        );
        for (case, expected) in cases
            .as_array()
            .unwrap()
            .iter()
            .zip(reference["cases"].as_array().unwrap())
        {
            assert_eq!(case["name"], expected["name"]);
            let model = Model::load(
                Path::new(&path),
                serde_json::from_value(case.get("policy").cloned().unwrap_or(json!({}))).unwrap(),
            )
            .unwrap();
            let request = Request {
                query: case["query"].as_str().unwrap().into(),
                documents: serde_json::from_value(case["documents"].clone()).unwrap(),
                top_n: None,
                truncate: case["truncate"].as_bool().unwrap_or(false),
            };
            let prepared = model.prepare(&request, &cancelled, &cancelled).unwrap();
            assert_eq!(
                json!(prepared.input_tokens),
                expected["usage"]["inputTokens"]
            );
            assert_eq!(
                json!(prepared.prompt_tokens),
                expected["usage"]["promptTokens"]
            );
            assert_eq!(json!(prepared.truncated_query), expected["truncatedQuery"]);
            assert_eq!(
                json!(prepared.truncated_documents),
                expected["truncatedDocumentIndices"]
            );
            let mut logits = Vec::new();
            let mut start = 0;
            let mut padded_tokens = 0;
            let mut stage_max = 0.0f64;
            while start < prepared.pairs.len() {
                let end = batch_end(&prepared.pairs, start, &model.policy);
                let pairs = &prepared.pairs[start..end];
                let mut stage = 0;
                let output = model
                    .forward_observed(pairs, &cancelled, &cancelled, |x| {
                        let cls = x
                            .slice(
                                &[0, 0, 0],
                                &[pairs.len() as i64, 1, model.config.hidden_size],
                            )?
                            .astype(DType::Float32)?
                            .to_float32()?;
                        for (i, row) in cls.chunks(model.config.hidden_size as usize).enumerate() {
                            for (v, e) in row.iter().zip(
                                expected["pairs"][start + i]["stages"][stage]
                                    .as_array()
                                    .unwrap(),
                            ) {
                                stage_max =
                                    stage_max.max((f64::from(*v) - e.as_f64().unwrap()).abs());
                            }
                        }
                        stage += 1;
                        Ok(())
                    })
                    .unwrap();
                assert_eq!(stage, model.layers.len() + 1);
                for (i, pair) in pairs.iter().enumerate() {
                    let e = &expected["pairs"][start + i];
                    assert_eq!(json!(pair.index), e["index"]);
                    assert_eq!(json!(pair.encoding.get_ids()), e["tokens"]);
                    assert_eq!(json!(pair.encoding.get_attention_mask()), e["mask"]);
                    assert_eq!(json!(pair.encoding.get_type_ids()), e["types"]);
                    assert_eq!(
                        json!(position_ids(
                            pair.encoding.get_ids(),
                            model.config.pad_token_id
                        )),
                        e["positions"]
                    );
                }
                padded_tokens += pairs.len() * pairs[0].encoding.len();
                logits.extend(output);
                start = end;
            }
            assert_eq!(json!(padded_tokens), expected["paddedTokens"]);
            let expected = expected["pairs"].as_array().unwrap();
            let mut max_logit = 0.0f64;
            let mut sum_logit = 0.0;
            let mut max_score = 0.0f64;
            let mut sum_score = 0.0;
            for (i, &actual) in logits.iter().enumerate() {
                let error = (f64::from(actual) - expected[i]["logit"].as_f64().unwrap()).abs();
                max_logit = max_logit.max(error);
                sum_logit += error;
                let error =
                    (sigmoid(actual).unwrap() - expected[i]["score"].as_f64().unwrap()).abs();
                max_score = max_score.max(error);
                sum_score += error;
                for j in 0..i {
                    let margin = expected[i]["logit"].as_f64().unwrap()
                        - expected[j]["logit"].as_f64().unwrap();
                    if margin.abs() > 2.0 * threshold["logitMaxAbs"].as_f64().unwrap() {
                        assert_eq!(
                            margin.signum(),
                            (f64::from(actual) - f64::from(logits[j])).signum()
                        );
                    }
                }
            }
            let mean_logit = sum_logit / logits.len().max(1) as f64;
            let mean_score = sum_score / logits.len().max(1) as f64;
            println!(
                "{}",
                json!({"case":case["name"],"maxLogitError":max_logit,"meanLogitError":mean_logit,"maxScoreError":max_score,"meanScoreError":mean_score,"maxStageError":stage_max})
            );
            assert!(max_logit <= threshold["logitMaxAbs"].as_f64().unwrap());
            assert!(mean_logit <= threshold["logitMeanAbs"].as_f64().unwrap());
            assert!(max_score <= threshold["scoreMaxAbs"].as_f64().unwrap());
            assert!(mean_score <= threshold["scoreMeanAbs"].as_f64().unwrap());
        }
        array::synchronize();
    }
}
