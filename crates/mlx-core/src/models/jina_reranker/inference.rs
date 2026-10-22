use super::{
    config::{Config, Options, Policy, Profile, invalid},
    weights::Weights,
};
use crate::models::rerank_runtime::Request;
use crate::{
    array::{
        self, DType, MxArray,
        attention::{scaled_dot_product_attention, scaled_dot_product_attention_causal},
        mask::create_causal_mask,
    },
    models::quantized_linear::LinearProj,
    nn::{Activations, Embedding, RMSNorm, RoPE},
    transformer::{KVCache, RotatingKVCache},
};
use napi::{Error, Result};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};
use tokenizers::Tokenizer;

struct Layer {
    q: LinearProj,
    k: LinearProj,
    v: LinearProj,
    o: LinearProj,
    gate: LinearProj,
    up: LinearProj,
    down: LinearProj,
    input_norm: RMSNorm,
    post_norm: RMSNorm,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    window: Option<i32>,
}
enum Cache {
    Full(KVCache),
    Sliding(RotatingKVCache),
}
impl Cache {
    fn update(&mut self, k: &MxArray, v: &MxArray) -> Result<(MxArray, MxArray)> {
        match self {
            Self::Full(c) => c.update_and_fetch(k, v),
            Self::Sliding(c) => {
                let mut kv = c.update_and_fetch(k, v)?.into_iter();
                Ok((
                    kv.next().ok_or_else(|| invalid("missing cached keys"))?,
                    kv.next().ok_or_else(|| invalid("missing cached values"))?,
                ))
            }
        }
    }
}

pub(super) struct Model {
    config: Config,
    profile: Profile,
    policy: Policy,
    tokenizer: Tokenizer,
    embedding: Embedding,
    layers: Vec<Layer>,
    norm: RMSNorm,
    rope: RoPE,
    projector1: LinearProj,
    projector2: LinearProj,
    doc_token: u32,
    query_token: u32,
    pub info: Value,
    _cache_limit_guard: crate::cache_limit::CacheLimitGuard,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ResultItem {
    index: usize,
    relevance_score: f32,
}
fn rank_order(a: &ResultItem, b: &ResultItem) -> std::cmp::Ordering {
    // Signed zeros are equal scores; total_cmp alone would rank +0 above -0.
    if a.relevance_score == b.relevance_score {
        a.index.cmp(&b.index)
    } else {
        b.relevance_score.total_cmp(&a.relevance_score)
    }
}
struct Block {
    tokens: Vec<u32>,
    positions: Vec<usize>,
    query_position: usize,
}

const RANKING_MARKERS: [&str; 3] = ["<|embed_token|>", "<|rerank_token|>", "<|score_token|>"];

fn sanitize_text(text: &str) -> String {
    let mut clean = String::with_capacity(text.len());
    for character in text.chars() {
        clean.push(character);
        // The accumulated prefix is marker-free. Remove a newly completed
        // suffix immediately, including markers formed around earlier removals.
        // Fixed-length suffix checks keep deeply nested inputs linear-time.
        for marker in RANKING_MARKERS {
            if clean.ends_with(marker) {
                clean.truncate(clean.len() - marker.len());
                break;
            }
        }
    }
    clean
}

pub(super) fn check_cancel(cancel: &AtomicBool, shutdown: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Acquire) || shutdown.load(Ordering::Acquire) {
        Err(Error::from_reason("rerank_cancelled: request cancelled"))
    } else {
        Ok(())
    }
}

impl Model {
    pub fn load(path: &Path, options: Options) -> Result<Self> {
        let raw: Value = serde_json::from_str(
            &std::fs::read_to_string(path.join("config.json")).map_err(invalid)?,
        )
        .map_err(invalid)?;
        let (config, profile) = Config::parse(&raw)?;
        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json")).map_err(invalid)?;
        if tokenizer.get_truncation().is_some() || tokenizer.get_padding().is_some() {
            return Err(invalid(
                "reranker tokenizer must not have automatic truncation or padding",
            ));
        }
        if tokenizer
            .get_vocab(true)
            .values()
            .any(|&id| u64::from(id) >= config.vocab_size as u64)
        {
            return Err(invalid("tokenizer vocabulary exceeds embedding dimensions"));
        }
        let doc_token = tokenizer
            .token_to_id("<|embed_token|>")
            .ok_or_else(|| invalid("missing document marker"))?;
        let query_token = tokenizer
            .token_to_id("<|rerank_token|>")
            .ok_or_else(|| invalid("missing query marker"))?;
        if doc_token == query_token
            || u64::from(doc_token.max(query_token)) >= config.vocab_size as u64
        {
            return Err(invalid("invalid marker IDs"));
        }
        let tokenizer_config: Value = serde_json::from_str(
            &std::fs::read_to_string(path.join("tokenizer_config.json")).map_err(invalid)?,
        )
        .map_err(invalid)?;
        let context = tokenizer_config["model_max_length"]
            .as_u64()
            .filter(|&v| v > 0)
            .map(|v| v.min(config.max_position_embeddings as u64) as usize)
            .unwrap_or(config.max_position_embeddings);
        let policy = Policy::resolve(options, profile, context)?;
        let weights = Weights::new(path, &raw)?;
        weights.validate_config(&config)?;
        let embedding = weights.embedding(config.vocab_size, config.hidden_size)?;
        let h = config.hidden_size;
        let d = config.head_dim;
        let inter = config.intermediate_size;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("layers.{i}");
            let linear =
                |suffix: &str, rows, cols| weights.linear(&format!("{p}.{suffix}"), rows, cols);
            let norm = |suffix: &str, dim| {
                weights.norm(&format!("{p}.{suffix}"), dim, config.rms_norm_eps)
            };
            layers.push(Layer {
                q: linear("self_attn.q_proj", config.num_attention_heads * d, h)?,
                k: linear("self_attn.k_proj", config.num_key_value_heads * d, h)?,
                v: linear("self_attn.v_proj", config.num_key_value_heads * d, h)?,
                o: linear("self_attn.o_proj", h, config.num_attention_heads * d)?,
                gate: linear("mlp.gate_proj", inter, h)?,
                up: linear("mlp.up_proj", inter, h)?,
                down: linear("mlp.down_proj", h, inter)?,
                input_norm: norm("input_layernorm", h)?,
                post_norm: norm("post_attention_layernorm", h)?,
                q_norm: norm("self_attn.q_norm", d)?,
                k_norm: norm("self_attn.k_norm", d)?,
                window: if config.layer_types[i] == "sliding_attention" {
                    config.sliding_window
                } else {
                    None
                },
            });
        }
        let pd = weights.output_dim("projector.0")?;
        let output = weights.output_dim("projector.2")?;
        let projector1 = weights.linear("projector.0", pd, h)?;
        let projector2 = weights.linear("projector.2", output, pd)?;
        let norm = weights.norm("norm", h, config.rms_norm_eps)?;
        let rope = RoPE::new(d as i32, Some(false), Some(config.rope_theta), None);
        let quantized = weights.tensors.keys().any(|k| k.ends_with(".scales"));
        let info = json!({"family":"jina_reranker","profile":profile,"execution":"listwise","scoreType":"cosine","contextWindow":context,"policy":policy,"quantized":quantized,"weightFormats":weights.formats()});
        // Count checkpoint-owned bytes, never a process-wide before/after
        // allocation delta: chat/audio may be active in the same process.
        let weight_bytes = weights.tensors.values().map(|w| w.nbytes() as u64).sum();
        let cache_limit_guard = crate::cache_limit::coordinator().register(weight_bytes);
        // Materialize on the owning thread: no disk/Metal work escapes load admission.
        for w in weights.tensors.values() {
            w.eval();
        }
        array::synchronize();
        Ok(Self {
            config,
            profile,
            policy,
            tokenizer,
            embedding,
            layers,
            norm,
            rope,
            projector1,
            projector2,
            doc_token,
            query_token,
            info,
            _cache_limit_guard: cache_limit_guard,
        })
    }
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(invalid)?
            .get_ids()
            .to_vec())
    }
    fn text(
        &self,
        text: &str,
        limit: usize,
        truncate: bool,
        label: &str,
    ) -> Result<(String, usize, bool)> {
        let clean = sanitize_text(text);
        let ids = self.encode(&clean)?;
        if ids.len() <= limit {
            return Ok((clean, ids.len(), false));
        }
        if !truncate {
            return Err(Error::from_reason(format!(
                "rerank_input_too_large: {label} exceeds {limit} tokens"
            )));
        }
        let shortened = self
            .tokenizer
            .decode(&ids[..limit], false)
            .map_err(invalid)?;
        // A tokenizer decoder may reconstruct text differently from its input.
        // Preserve the same marker-free invariant after truncation as well.
        Ok((sanitize_text(&shortened), ids.len(), true))
    }
    fn block(&self, query: &str, docs: &[String]) -> Result<Block> {
        let tokens = self.encode(&format_prompt(self.profile, query, docs))?;
        let positions = tokens
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (t == self.doc_token).then_some(i))
            .collect::<Vec<_>>();
        let queries = tokens
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (t == self.query_token).then_some(i))
            .collect::<Vec<_>>();
        let expected = if self.profile == Profile::V3 { 1 } else { 2 };
        if positions.len() != docs.len() || queries.len() != expected {
            return Err(invalid("tokenizer did not preserve ranking markers"));
        }
        Ok(Block {
            tokens,
            positions,
            query_position: queries[expected - 1],
        })
    }
    fn forward(
        &self,
        block: &Block,
        cancel: &AtomicBool,
        shutdown: &AtomicBool,
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
        let c = &self.config;
        let mut caches: Vec<_> = self
            .layers
            .iter()
            .map(|l| match l.window {
                Some(w) => Cache::Sliding(RotatingKVCache::new(w, Some(0))),
                None => Cache::Full(KVCache::new()),
            })
            .collect();
        let mut gathered = Vec::new();
        let mut query = None;
        for (chunk_index, ids) in block
            .tokens
            .chunks(self.policy.prefill_chunk_size)
            .enumerate()
        {
            check_cancel(cancel, shutdown)?;
            let offset = chunk_index * self.policy.prefill_chunk_size;
            let n = ids.len() as i64;
            let input = MxArray::from_uint32(ids, &[1, n])?;
            let mut x = self.embedding.forward(&input)?;
            for (layer, cache) in self.layers.iter().zip(&mut caches) {
                check_cancel(cancel, shutdown)?;
                let normalized = layer.input_norm.forward(&x)?;
                let q = layer
                    .q_norm
                    .forward(&layer.q.forward(&normalized)?.reshape(&[
                        1,
                        n,
                        c.num_attention_heads,
                        c.head_dim,
                    ])?)?
                    .transpose(Some(&[0, 2, 1, 3]))?;
                let k = layer
                    .k_norm
                    .forward(&layer.k.forward(&normalized)?.reshape(&[
                        1,
                        n,
                        c.num_key_value_heads,
                        c.head_dim,
                    ])?)?
                    .transpose(Some(&[0, 2, 1, 3]))?;
                let v = layer
                    .v
                    .forward(&normalized)?
                    .reshape(&[1, n, c.num_key_value_heads, c.head_dim])?
                    .transpose(Some(&[0, 2, 1, 3]))?;
                let q = self.rope.forward(&q, Some(offset as i32))?;
                let k = self.rope.forward(&k, Some(offset as i32))?;
                let (k, v) = cache.update(&k, &v)?;
                let scale = 1.0 / (c.head_dim as f64).sqrt();
                let attended = if let Some(window) = layer.window {
                    let mask = create_causal_mask(
                        n as i32,
                        Some((k.shape_at(2)? - n) as i32),
                        Some(window),
                    )?;
                    scaled_dot_product_attention(&q, &k, &v, scale, Some(&mask))?
                } else {
                    scaled_dot_product_attention_causal(&q, &k, &v, scale)?
                };
                x = x.add(&layer.o.forward(
                    &attended.transpose(Some(&[0, 2, 1, 3]))?.reshape(&[
                        1,
                        n,
                        c.num_attention_heads * c.head_dim,
                    ])?,
                )?)?;
                let normalized = layer.post_norm.forward(&x)?;
                let mlp = Activations::silu(&layer.gate.forward(&normalized)?)?
                    .mul(&layer.up.forward(&normalized)?)?;
                x = x.add(&layer.down.forward(&mlp)?)?;
                x.eval();
            }
            let hidden = self.norm.forward(&x)?.reshape(&[n, c.hidden_size])?;
            for &position in block
                .positions
                .iter()
                .chain(std::iter::once(&block.query_position))
            {
                if position < offset || position >= offset + ids.len() {
                    continue;
                }
                let row = hidden.slice(
                    &[(position - offset) as i64, 0],
                    &[(position - offset + 1) as i64, c.hidden_size],
                )?;
                let projected = self
                    .projector2
                    .forward(&Activations::relu(&self.projector1.forward(&row)?)?)?
                    .astype(DType::Float32)?;
                let vector = projected.to_float32()?.to_vec();
                if position == block.query_position {
                    query = Some(vector);
                } else {
                    gathered.push(vector);
                }
            }
        }
        check_cancel(cancel, shutdown)?;
        if gathered.len() != block.positions.len() {
            return Err(invalid("missing document features"));
        }
        Ok((
            gathered,
            query.ok_or_else(|| invalid("missing query features"))?,
        ))
    }
    pub fn rerank(
        &self,
        request: Request,
        cancel: &AtomicBool,
        shutdown: &AtomicBool,
    ) -> Result<Value> {
        check_cancel(cancel, shutdown)?;
        if request.query.trim().is_empty() || request.top_n == Some(0) {
            return Err(Error::from_reason(
                "rerank_invalid_request: query must be nonempty and topN must be positive",
            ));
        }
        if request.documents.len() > self.policy.max_documents {
            return Err(Error::from_reason(
                "rerank_input_too_large: too many documents",
            ));
        }
        let (query, mut input_tokens, query_truncated) = self.text(
            &request.query,
            self.policy.max_query_tokens,
            request.truncate,
            "query",
        )?;
        if query.trim().is_empty() {
            return Err(Error::from_reason(
                "rerank_invalid_request: query contains only reserved markers",
            ));
        }
        let mut docs = Vec::with_capacity(request.documents.len());
        let mut truncated = Vec::new();
        for (i, doc) in request.documents.iter().enumerate() {
            check_cancel(cancel, shutdown)?;
            let (text, count, cut) = self.text(
                doc,
                self.policy.max_document_tokens,
                request.truncate,
                &format!("document {i}"),
            )?;
            input_tokens = input_tokens
                .checked_add(count)
                .ok_or_else(|| invalid("token count overflow"))?;
            if input_tokens > self.policy.max_input_tokens {
                return Err(Error::from_reason(
                    "rerank_input_too_large: total input token limit exceeded",
                ));
            }
            if cut {
                truncated.push(i);
            }
            docs.push(text);
        }
        if input_tokens > self.policy.max_input_tokens {
            return Err(Error::from_reason(
                "rerank_input_too_large: total input token limit exceeded",
            ));
        }
        let mut all_docs = Vec::new();
        let mut queries = Vec::new();
        let mut weights = Vec::new();
        let mut prompt_tokens = 0usize;
        let mut start = 0;
        while start < docs.len() {
            // Binary search the exact formatted token budget, bounded by candidate count.
            let mut low = start + 1;
            let mut high = (start + self.policy.max_documents_per_block).min(docs.len());
            let mut selected = None;
            while low <= high {
                check_cancel(cancel, shutdown)?;
                let end = low + (high - low) / 2;
                let block = self.block(&query, &docs[start..end])?;
                if block.tokens.len() <= self.policy.max_block_tokens {
                    selected = Some((end, block));
                    low = end + 1;
                } else {
                    high = end - 1;
                }
            }
            let (end, block) = selected.ok_or_else(|| {
                Error::from_reason(
                    "rerank_input_too_large: one document plus query/template exceeds block budget",
                )
            })?;
            prompt_tokens += block.tokens.len();
            let (embeddings, q) = self.forward(&block, cancel, shutdown)?;
            let weight = embeddings
                .iter()
                .map(|d| cosine(&q, d).map(|s| (1.0 + s) * 0.5))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .fold(0.0f32, f32::max);
            all_docs.extend(embeddings);
            queries.push(q);
            weights.push(weight);
            start = end;
        }
        let mut results = Vec::new();
        if !all_docs.is_empty() {
            let fused = fuse(&queries, &weights)?;
            for (index, d) in all_docs.iter().enumerate() {
                results.push(ResultItem {
                    index,
                    relevance_score: cosine(&fused, d)?,
                });
            }
            results.sort_by(rank_order);
            results.truncate(request.top_n.unwrap_or(results.len()).min(results.len()));
        }
        Ok(
            json!({"results":results,"usage":{"inputTokens":input_tokens,"promptTokens":prompt_tokens},"meta":{"profile":self.profile,"blocks":queries.len(),"processedDocuments":docs.len(),"maxBlockTokens":self.policy.max_block_tokens,"truncatedQuery":query_truncated,"truncatedDocumentIndices":truncated}}),
        )
    }
}
fn format_prompt(profile: Profile, query: &str, docs: &[String]) -> String {
    let system = "<|im_start|>system\nYou are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. If the query is a question, how relevant a passage is depends on how well it answers the question. If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. If an instruction is provided, you should follow the instruction when determining the ranking.<|im_end|>\n<|im_start|>user\n";
    let early = if profile == Profile::V35 {
        "<|rerank_token|>"
    } else {
        ""
    };
    let mut prompt = format!(
        "{system}I will provide you with {} passages, each indicated by a numerical identifier. Rank the passages based on their relevance to query: {query}{early}\n",
        docs.len()
    );
    for (i, doc) in docs.iter().enumerate() {
        if i > 0 {
            prompt.push('\n');
        }
        prompt.push_str(&format!(
            "<passage id=\"{i}\">\n{doc}<|embed_token|>\n</passage>"
        ));
    }
    prompt.push_str(&format!("\n<query>\n{query}<|rerank_token|>\n</query>"));
    if profile == Profile::V35 {
        prompt.push_str("\nPlease provide the ranking of all passages based on their relevance to the search query, in descending order of relevance, with each label enclosed in square brackets (e.g., [2] > [1] > [3] > [0]).");
    }
    prompt.push_str("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    prompt
}
fn cosine(q: &[f32], d: &[f32]) -> Result<f32> {
    if q.len() != d.len() {
        return Err(invalid("embedding dimension mismatch"));
    }
    let dot = q.iter().zip(d).map(|(a, b)| a * b).sum::<f32>();
    let norm =
        q.iter().map(|v| v * v).sum::<f32>().sqrt() * d.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm <= 0.0 || !norm.is_finite() || !dot.is_finite() {
        return Err(Error::from_reason(
            "rerank_numerical_error: invalid embedding norm",
        ));
    }
    Ok((dot / norm).clamp(-1.0, 1.0))
}
fn fuse(queries: &[Vec<f32>], weights: &[f32]) -> Result<Vec<f32>> {
    let sum = weights.iter().sum::<f32>();
    if sum <= 0.0 || !sum.is_finite() {
        return Err(Error::from_reason(
            "rerank_numerical_error: invalid fusion weights",
        ));
    }
    let Some(first) = queries.first() else {
        return Err(invalid("empty query fusion"));
    };
    let mut result = vec![0.0; first.len()];
    for (q, w) in queries.iter().zip(weights) {
        for (out, v) in result.iter_mut().zip(q) {
            *out += v * (w / sum);
        }
    }
    Ok(result)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nested_markers_are_removed_at_every_boundary_without_changing_unicode() {
        for outer in RANKING_MARKERS {
            for inner in RANKING_MARKERS {
                for boundary in 0..=outer.len() {
                    let text = format!(
                        "你好🌍{}{}{}café",
                        &outer[..boundary],
                        inner,
                        &outer[boundary..]
                    );
                    assert_eq!(sanitize_text(&text), "你好🌍café");
                }
            }
        }
        let ordinary = "你好🌍 <passage>café</passage> <|embed_token| incomplete";
        assert_eq!(sanitize_text(ordinary), ordinary);
        assert_eq!(sanitize_text(""), "");
    }
    #[test]
    fn deeply_nested_markers_do_not_require_repeated_whole_input_scans() {
        let text = format!(
            "a{}<|score_token|>{}b",
            "<|embed".repeat(4096),
            "_token|>".repeat(4096)
        );
        assert_eq!(sanitize_text(&text), "ab");
    }
    #[test]
    fn equal_scores_including_signed_zero_keep_original_order() {
        let mut results = [
            ResultItem {
                index: 1,
                relevance_score: 0.0,
            },
            ResultItem {
                index: 2,
                relevance_score: -0.5,
            },
            ResultItem {
                index: 0,
                relevance_score: -0.0,
            },
            ResultItem {
                index: 3,
                relevance_score: 0.5,
            },
        ];
        results.sort_by(rank_order);
        assert_eq!(results.map(|r| r.index), [3, 0, 1, 2]);
    }
    #[test]
    fn fusion_preserves_unnormalized_queries() {
        let q = fuse(&[vec![2., 0.], vec![0., 1.]], &[1., 1.]).unwrap();
        assert_eq!(q, vec![1., 0.5]);
        assert!((cosine(&q, &[1., 0.]).unwrap() - 0.8944272).abs() < 1e-6);
        assert!(cosine(&[0., 0.], &[1., 0.]).is_err());
        assert!(fuse(&[vec![1.]], &[0.]).is_err());
    }
    #[test]
    fn profile_marker_counts_and_tail_differ() {
        let docs = vec!["document".to_string()];
        let a = format_prompt(Profile::V3, "query", &docs);
        let b = format_prompt(Profile::V35, "query", &docs);
        assert_eq!(a.matches("<|rerank_token|>").count(), 1);
        assert_eq!(b.matches("<|rerank_token|>").count(), 2);
        assert!(!a.contains("Please provide"));
        assert!(b.contains("Please provide"));
    }
}

#[cfg(test)]
mod golden_tests {
    use super::*;
    /// This opt-in check compares intermediate features, not just final ordering.
    #[test]
    #[ignore = "requires pinned local checkpoints and independent reference artifacts; see scripts/rerank/README.md"]
    fn golden_checkpoint_features() {
        let path = std::env::var("MLX_RERANK_GOLDEN_MODEL").expect("model directory");
        let reference = std::env::var("MLX_RERANK_GOLDEN_REFERENCE").expect("reference JSON");
        let cases: Value =
            serde_json::from_str(include_str!("../../../../../scripts/rerank/cases.json")).unwrap();
        let references: Value =
            serde_json::from_str(&std::fs::read_to_string(reference).unwrap()).unwrap();
        for (case, reference) in cases
            .as_array()
            .unwrap()
            .iter()
            .zip(references.as_array().unwrap())
        {
            let options: Options =
                serde_json::from_value(case.get("policy").cloned().unwrap_or(json!({}))).unwrap();
            let model = Model::load(Path::new(&path), options).unwrap();
            let (query, _, _) = model
                .text(
                    case["query"].as_str().unwrap(),
                    model.policy.max_query_tokens,
                    false,
                    "query",
                )
                .unwrap();
            let docs: Vec<_> = case["documents"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| {
                    model
                        .text(
                            d.as_str().unwrap(),
                            model.policy.max_document_tokens,
                            false,
                            "document",
                        )
                        .unwrap()
                        .0
                })
                .collect();
            let mut at = 0;
            for block in reference["blocks"].as_array().unwrap() {
                let count = block["documents"].as_array().unwrap().len();
                let actual = model.block(&query, &docs[at..at + count]).unwrap();
                at += count;
                assert_eq!(
                    json!(actual.tokens),
                    block["tokens"],
                    "token IDs: {}",
                    case["name"]
                );
                assert_eq!(json!(actual.positions), block["positions"]);
                assert_eq!(json!(actual.query_position), block["queryPosition"]);
                let (d, q) = model
                    .forward(&actual, &AtomicBool::new(false), &AtomicBool::new(false))
                    .unwrap();
                let weight = d
                    .iter()
                    .map(|v| (1.0 + cosine(&q, v).unwrap()) * 0.5)
                    .fold(0.0f32, f32::max);
                assert!(
                    (f64::from(weight) - block["weight"].as_f64().unwrap()).abs() <= 0.005,
                    "block fusion weight: {}",
                    case["name"]
                );
                let expected: Vec<Vec<f32>> =
                    serde_json::from_value(block["documents"].clone()).unwrap();
                let eq: Vec<f32> = serde_json::from_value(block["query"].clone()).unwrap();
                for (actual, expected) in d
                    .iter()
                    .chain(std::iter::once(&q))
                    .zip(expected.iter().chain(std::iter::once(&eq)))
                {
                    assert_eq!(actual.len(), expected.len());
                    let error = actual
                        .iter()
                        .zip(expected)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f32>()
                        .sqrt();
                    let norm = expected.iter().map(|v| v * v).sum::<f32>().sqrt();
                    assert!(
                        error / norm < 0.03,
                        "projected features diverged: {} relative L2 {}",
                        case["name"],
                        error / norm
                    );
                }
            }
        }
    }
}
