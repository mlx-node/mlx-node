//! Inference-only port of Cloudflare JointSchemaHead (Apache-2.0).
use super::encoding::EncodedRecord;
use crate::{
    array::{DType, MxArray, scaled_dot_product_attention},
    models::quantized_linear::LinearProj,
    nn::Activations,
};
use napi::{Error, Result};
use serde::Deserialize;
use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

/// Compiled-graph id for the fused `gelu_exact` used by every clef head.
/// A pure elementwise function with no captured weights, so the tape needs
/// no per-model keying and nothing to erase on drop.
const GELU_EXACT_FN_ID: u64 = 0xDFC7_0000_0000_0000;

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
    /// Weight-derived tensors materialized once at load: baked scalars
    /// (`prior_scale`/`joint_scale`/`residual_gate_sigmoid`), pre-transposed
    /// linear weights (`{prefix}.weight_t`) and self-attention projections
    /// (`layers.{i}.self_attn.in_proj_t`), and the pre-split, pre-transposed
    /// cross-attention projections (`{prefix}.q` / `{prefix}.kv` /
    /// `{prefix}.q_bias` / `{prefix}.kv_bias`). Derived views that stayed lazy
    /// would re-issue their producer nodes inside every decide's forward graph,
    /// so they are evaluated here.
    derived: HashMap<String, MxArray>,
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
        mut weights: HashMap<String, MxArray>,
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
        let mut has_f32 = false;
        let mut has_f16 = false;
        for (name, shape) in &shapes {
            let weight = weights
                .get(name)
                .ok_or_else(|| Error::from_reason(format!("Missing CLEF head tensor: {name}")))?;
            let dtype = weight.dtype()?;
            if weight.shape()?.as_ref() != shape.as_slice()
                || !matches!(dtype, DType::Float32 | DType::Float16 | DType::BFloat16)
            {
                return Err(Error::from_reason(format!(
                    "Invalid CLEF head tensor shape or dtype: {name}"
                )));
            }
            has_f32 |= dtype == DType::Float32;
            has_f16 |= dtype == DType::Float16;
        }
        if let Some(key) = weights.keys().find(|k| !expected.contains(k.as_str())) {
            return Err(Error::from_reason(format!(
                "Unexpected CLEF head tensor: {key}"
            )));
        }
        // GGUF head extraction leaves norms, biases, and scalars in Float32 while
        // Q8_0 linear/attention weights decode to Float16. Because `forward` casts
        // `hidden` to `hidden_norm.weight`'s dtype, mixed F32+F16 would force the
        // entire head into FP32 + per-call F16->F32 promotions on CUDA. Unify the
        // mixed GGUF signature to BFloat16 (matching HF `joint_head.safetensors`),
        // preserving the raw FP32 scalars for exact `clip+exp`/`sigmoid` in
        // `bake_derived()`, while leaving all-Float32 and all-BFloat16 checkpoints
        // untouched.
        let raw_scalars = if crate::accelerator::cuda_ops_available() && has_f32 && has_f16 {
            let saved = (
                weights["prior_logit_scale"].clone(),
                weights["joint_logit_scale"].clone(),
                weights["residual_gate"].clone(),
            );
            for tensor in weights.values_mut() {
                if tensor.dtype()? != DType::BFloat16 {
                    *tensor = tensor.astype(DType::BFloat16)?;
                }
            }
            let refs: Vec<&MxArray> = weights.values().collect();
            MxArray::eval_arrays(&refs)?;
            Some(saved)
        } else {
            None
        };
        let mut head = Self {
            config,
            weights,
            derived: HashMap::new(),
        };
        head.bake_derived(raw_scalars.as_ref())?;
        Ok(head)
    }
    /// Materialize weight-derived tensors once so forward graphs carry zero
    /// weight-transform chains.
    ///
    /// Scalars: `prior_scale`, `joint_scale` (clip(.; ln 100).exp) and
    /// `residual_gate_sigmoid` — computed in FP32 for exact precision and cast
    /// to `hidden_norm.weight`'s dtype so scoring multiplications stay in the
    /// head's working dtype without promoting back to FP32.
    ///
    /// Linears and self-attention: every 2D linear `.weight` is pre-transposed
    /// into `{prefix}.weight_t`, and `layers.{i}.self_attn.in_proj_weight` into
    /// `layers.{i}.self_attn.in_proj_t`.
    ///
    /// Cross-attention: every cross-attention block projects its three roles
    /// out of one `in_proj_weight [3W,W]` / `in_proj_bias [3W]`. For
    /// cross-attention (`evidence_layers.{i}.attention` and
    /// `layers.{i}.multihead_attn`), `[0,W)` / `[W,3W)` slices + the
    /// pre-transposed Q/KV mats are baked into
    /// `{prefix}.q`/`{prefix}.kv`/`{prefix}.q_bias`/`{prefix}.kv_bias`.
    /// Slicing then stacking rows back is a row permutation of the same GEMM
    /// operands, so outputs are bit-identical to three sliced matmuls.
    fn bake_derived(&mut self, raw_scalars: Option<&(MxArray, MxArray, MxArray)>) -> Result<()> {
        let mut inserts: Vec<(String, MxArray)> = Vec::new();
        let w = self.config.width;
        let target_dtype = self.w("hidden_norm.weight").dtype()?;
        let mut insert = |name: String, value: MxArray| inserts.push((name, value));
        let (prior_raw, joint_raw, gate_raw) = match raw_scalars {
            Some((prior, joint, gate)) => (prior, joint, gate),
            None => (
                self.w("prior_logit_scale"),
                self.w("joint_logit_scale"),
                self.w("residual_gate"),
            ),
        };
        let to_f32 = |a: &MxArray| -> Result<MxArray> {
            if a.dtype()? == DType::Float32 {
                Ok(a.clone())
            } else {
                a.astype(DType::Float32)
            }
        };
        let to_target = |a: MxArray| -> Result<MxArray> {
            if a.dtype()? == target_dtype {
                Ok(a)
            } else {
                a.astype(target_dtype)
            }
        };
        insert(
            "prior_scale".into(),
            to_target(to_f32(prior_raw)?.clip(None, Some(100f64.ln()))?.exp()?)?,
        );
        insert(
            "joint_scale".into(),
            to_target(to_f32(joint_raw)?.clip(None, Some(100f64.ln()))?.exp()?)?,
        );
        insert(
            "residual_gate_sigmoid".into(),
            to_target(Activations::sigmoid(&to_f32(gate_raw)?)?)?,
        );
        let bake_linear = |insert: &mut dyn FnMut(String, MxArray), prefix: &str| -> Result<()> {
            let weight = self.w(&format!("{prefix}.weight"));
            insert(format!("{prefix}.weight_t"), weight.transpose(None)?);
            Ok(())
        };
        let bake_cross_attention =
            |insert: &mut dyn FnMut(String, MxArray), prefix: &str| -> Result<()> {
                let weight = self.w(&format!("{prefix}.in_proj_weight"));
                let bias = self.w(&format!("{prefix}.in_proj_bias"));
                // Q rows [0,W) of in_proj weight/bias; KV rows [W,3W).
                let q_w = weight.slice_axis(0, 0, w)?.transpose(None)?;
                let kv_w = weight.slice_axis(0, w, 3 * w)?.transpose(None)?;
                let q_b = bias.slice_axis(0, 0, w)?;
                let kv_b = bias.slice_axis(0, w, 3 * w)?;
                insert(format!("{prefix}.q"), q_w);
                insert(format!("{prefix}.kv"), kv_w);
                insert(format!("{prefix}.q_bias"), q_b);
                insert(format!("{prefix}.kv_bias"), kv_b);
                Ok(())
            };
        for name in [
            "memory_projection",
            "question_projection",
            "option_question_projection",
            "global_projection",
            "option_context_projection",
            "option_lexical_projection",
            "residual_scorer.0",
            "residual_scorer.3",
        ] {
            bake_linear(&mut insert, name)?;
        }
        for i in 0..self.config.routing_layers {
            let p = format!("evidence_layers.{i}");
            bake_cross_attention(&mut insert, &format!("{p}.attention"))?;
            bake_linear(&mut insert, &format!("{p}.attention.out_proj"))?;
            bake_linear(&mut insert, &format!("{p}.feedforward.0"))?;
            bake_linear(&mut insert, &format!("{p}.feedforward.3"))?;
        }
        for i in 0..self.config.layers {
            let p = format!("layers.{i}");
            let self_attn = format!("{p}.self_attn");
            insert(
                format!("{self_attn}.in_proj_t"),
                self.w(&format!("{self_attn}.in_proj_weight"))
                    .transpose(None)?,
            );
            bake_linear(&mut insert, &format!("{self_attn}.out_proj"))?;
            bake_cross_attention(&mut insert, &format!("{p}.multihead_attn"))?;
            bake_linear(&mut insert, &format!("{p}.multihead_attn.out_proj"))?;
            bake_linear(&mut insert, &format!("{p}.linear1"))?;
            bake_linear(&mut insert, &format!("{p}.linear2"))?;
        }
        let refs: Vec<&MxArray> = inserts.iter().map(|(_, a)| a).collect();
        MxArray::eval_arrays(&refs)?;
        self.derived.extend(inserts);
        Ok(())
    }
    pub fn nbytes(&self) -> u64 {
        self.weights
            .values()
            .chain(self.derived.values())
            .map(|a| a.nbytes() as u64)
            .sum()
    }
    fn w(&self, name: &str) -> &MxArray {
        &self.weights[name]
    }
    fn d(&self, name: &str) -> &MxArray {
        &self.derived[name]
    }
    fn linear(&self, x: &MxArray, prefix: &str) -> Result<MxArray> {
        let y = x.matmul(self.d(&format!("{prefix}.weight_t")))?;
        match self.weights.get(&format!("{prefix}.bias")) {
            Some(b) => y.add(b),
            None => Ok(y),
        }
    }
    fn norm(&self, x: &MxArray, prefix: &str) -> Result<MxArray> {
        let weight = self.w(&format!("{prefix}.weight"));
        let bias = self.w(&format!("{prefix}.bias"));
        // SAFETY: `x`, `weight`, and `bias` are live `MxArray`s validated at load
        // time (1D weight and bias matching `x`'s last dim), and `from_handle`
        // checks for null.
        let handle = unsafe {
            mlx_sys::mlx_fast_layer_norm(
                x.as_raw_ptr(),
                weight.as_raw_ptr(),
                bias.as_raw_ptr(),
                1e-5_f32,
            )
        };
        MxArray::from_handle(handle, "clef_head_layer_norm")
    }
    /// One attention block out of the checkpoint's fused
    /// `in_proj_weight [3W,W]` / `in_proj_bias [3W]` layout.
    ///
    /// `self_attention` (Q == K/V input): a single `[rows,3W]` GEMM split
    /// into Q|K|V — one matmul + one shared-buffer split where the old form
    /// ran three slice+matmul pairs. Row-fused GEMM outputs are
    /// element-identical: each output column reduces over the same K dim.
    ///
    /// Cross-attention (Q != K/V input): the Q projection uses the baked
    /// `{prefix}.q` rows and the K/V pair comes from one `{prefix}.kv`
    /// projection + split — two matmuls instead of three.
    fn attention(
        &self,
        q: &MxArray,
        memory: &MxArray,
        prefix: &str,
        self_attention: bool,
    ) -> Result<MxArray> {
        let w = self.config.width;
        let heads = self.config.heads;
        let to_heads = |x: &MxArray| -> Result<MxArray> {
            x.reshape(&[1, -1, heads, w / heads])?
                .transpose(Some(&[0, 2, 1, 3]))
        };
        // split_sections already validates its output count; the
        // `ok_or_else` only satisfies the no-unwrap lint.
        let next_part = |parts: &mut std::vec::IntoIter<MxArray>| -> Result<MxArray> {
            parts
                .next()
                .ok_or_else(|| Error::from_reason("split_sections under-produced"))
        };
        let (queries, keys, values) = if self_attention {
            debug_assert!(std::ptr::eq(q, memory));
            let bias = self.w(&format!("{prefix}.in_proj_bias"));
            let fused = q
                .matmul(self.d(&format!("{prefix}.in_proj_t")))?
                .add(bias)?;
            let mut parts = fused.split_sections(&[w, 2 * w], 1)?.into_iter();
            let (q, k, v) = (
                next_part(&mut parts)?,
                next_part(&mut parts)?,
                next_part(&mut parts)?,
            );
            (to_heads(&q)?, to_heads(&k)?, to_heads(&v)?)
        } else {
            let queries = q
                .matmul(self.d(&format!("{prefix}.q")))?
                .add(self.d(&format!("{prefix}.q_bias")))?;
            let kv = memory
                .matmul(self.d(&format!("{prefix}.kv")))?
                .add(self.d(&format!("{prefix}.kv_bias")))?;
            let mut parts = kv.split_sections(&[w], 1)?.into_iter();
            let (k, v) = (next_part(&mut parts)?, next_part(&mut parts)?);
            (to_heads(&queries)?, to_heads(&k)?, to_heads(&v)?)
        };
        let scale = 1.0 / ((w / heads) as f64).sqrt();
        let output = if crate::accelerator::cuda_ops_available()
            && (queries.shape_at(2)?.max(0) as u64).saturating_mul(keys.shape_at(2)?.max(0) as u64)
                <= 2_000_000
        {
            // Avoid cuDNN delay-load (~3.6 s) and per-T SDPA graph cache misses
            // on CUDA: Clef's head runs small [1, H, O|Q, T] attentions where
            // precompiled cuBLASLt matmuls + FP32-accumulating `softmax_precise`
            // match MLX's own `fast.cpp` SDPA fallback with zero JIT/cuDNN overhead.
            let q_scaled = queries.mul_scalar(scale)?;
            let scores = q_scaled.matmul(&keys.transpose(Some(&[0, 1, 3, 2]))?)?;
            let probs = Activations::softmax_precise(&scores, Some(-1))?;
            probs.matmul(&values)?
        } else {
            scaled_dot_product_attention(&queries, &keys, &values, scale, None)?
        };
        self.linear(
            &output.transpose(Some(&[0, 2, 1, 3]))?.reshape(&[-1, w])?,
            &format!("{prefix}.out_proj"),
        )
    }
    /// fp32 exclusive prefix sum over `x` used by span-mean pooling.
    fn span_cumsum(x: &MxArray) -> Result<MxArray> {
        let zero = MxArray::zeros(&[1, x.shape_at(1)?], Some(DType::Float32))?;
        let f32_x = if x.dtype()? == DType::Float32 {
            x.clone()
        } else {
            x.astype(DType::Float32)?
        };
        MxArray::concatenate(&zero, &f32_x.cumsum(0)?, 0)
    }
    /// Row-wise mean over `spans` from a shared cumsum: an fp32 exclusive
    /// prefix sum indexed at (start, end) beats one slice+mean op per span.
    /// fp32 accumulation also matches MLX's own reduction accumulation, so
    /// the astype back to the weight dtype lands within a ulp of
    /// slice.mean().
    fn span_means_from_cumsum(
        cumsum: &MxArray,
        dtype: DType,
        spans: &[(usize, usize)],
    ) -> Result<MxArray> {
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
    /// Row-wise mean over `spans` of `x` (`[T, H]`).
    ///
    /// When `spans` is non-empty and covers less than half of `T`
    /// (`total_len > 0 && total_len * 2 < t`), gathers only the referenced
    /// rows in ONE `take` (`[total_len, H]`) before running the FP32 exclusive
    /// prefix sum over the compacted matrix with remapped contiguous spans.
    /// This avoids a full `[T, H]` FP32 cast + cumsum on medium/long prompts
    /// and prevents long-prefix FP32 cancellation. Very short prompts
    /// (`total_len * 2 >= t`), already-compacted inputs, and empty span lists
    /// keep the direct `span_cumsum` path.
    fn span_means(x: &MxArray, dtype: DType, spans: &[(usize, usize)]) -> Result<MxArray> {
        let t = usize::try_from(x.shape_at(0)?)
            .map_err(|_| Error::from_reason("span_means: negative sequence length"))?;
        if !spans.is_empty() {
            let total_len: usize = spans.iter().try_fold(0usize, |acc, &(s, e)| {
                if s > e || e > t {
                    return Err(Error::from_reason(format!(
                        "Invalid span ({s}, {e}) for sequence length {t}"
                    )));
                }
                acc.checked_add(e - s)
                    .ok_or_else(|| Error::from_reason("span_means: total span length overflow"))
            })?;
            if total_len > 0 && total_len.saturating_mul(2) < t {
                let mut flat_indices: Vec<u32> = Vec::with_capacity(total_len);
                let mut remapped: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
                let mut offset = 0usize;
                for &(s, e) in spans {
                    flat_indices.extend((s..e).map(|idx| idx as u32));
                    let len = e - s;
                    remapped.push((offset, offset + len));
                    offset += len;
                }
                let idx_arr = MxArray::from_uint32(&flat_indices, &[total_len as i64])?;
                let compacted = x.take(&idx_arr, 0)?;
                let cumsum = Self::span_cumsum(&compacted)?;
                return Self::span_means_from_cumsum(&cumsum, dtype, &remapped);
            }
        }
        Self::span_means_from_cumsum(&Self::span_cumsum(x)?, dtype, spans)
    }
    /// Mean-embed every (possibly ragged) contiguous id group in one take:
    /// gather all rows once, then span-mean over the flat layout. The gather
    /// runs on the packed projection (`take_dequantized_rows`), so a packed
    /// lm_head dequantizes only the option-token rows instead of the whole
    /// `[vocab, hidden]` table.
    fn group_mean_embeds(
        &self,
        output: &LinearProj,
        ids: &[u32],
        lengths: &[usize],
        dtype: DType,
    ) -> Result<MxArray> {
        let embeds = output.take_dequantized_rows(ids)?;
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
    /// Row-wise L2 normalize. Batching callers concatenate rows first, so the
    /// mul/sum/sqrt/clip/div chain dispatches once per epsilon group; every
    /// op is row-local, so a normalized row equals its standalone result.
    fn unit(x: &MxArray, eps: f64) -> Result<MxArray> {
        x.div(
            &x.mul(x)?
                .sum(Some(&[-1]), Some(true))?
                .sqrt()?
                .clip(Some(eps), None)?,
        )
    }
    /// `gelu_exact` fused by MLX's own `compile` machinery via
    /// [`crate::compiled_graph::invoke_compiled_graph`]: one replayed tape
    /// replaces the five-node elementwise chain. The traced builder is a
    /// pure function of its input (weights are never captured), so a single
    /// process-wide `fn_id` serves every loaded clef head and nothing is
    /// erased on drop. Falls back to the eager chain when the compiled
    /// backend is unavailable.
    fn gelu_exact(x: &MxArray) -> Result<MxArray> {
        // On CUDA, avoid `invoke_compiled_graph` (which NVRTC-compiles BF16
        // templates on the first run) and run the identical eager chain over
        // precompiled CUDA ops (`mul_scalar` / `add_scalar` / `erf` all
        // preserve floating input dtype).
        if crate::accelerator::cuda_ops_available() {
            let erf = x.mul_scalar(std::f64::consts::FRAC_1_SQRT_2)?.erf()?;
            return x.mul(&erf.add_scalar(1.0)?)?.mul_scalar(0.5);
        }
        let mut builder = |inputs: &[MxArray]| -> Result<Vec<MxArray>> {
            // 0.5 * x * (1 + erf(x / sqrt(2))): identical expression and
            // operation order to `Activations::gelu_exact`'s C++ chain, so a
            // compile-disabled replay computes the same value.
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
    pub fn forward(
        &self,
        hidden: &MxArray,
        record: &EncodedRecord,
        output: &LinearProj,
    ) -> Result<Vec<Vec<f64>>> {
        let target_dtype = self.w("hidden_norm.weight").dtype()?;
        let hidden = if hidden.dtype()? == target_dtype {
            hidden.clone()
        } else {
            hidden.astype(target_dtype)?
        };
        let h = self
            .norm(&hidden, "hidden_norm")?
            .reshape(&[-1, self.config.hidden_size])?;
        let dtype = h.dtype()?;
        let memory = self.linear(&h, "memory_projection")?;
        let global = h.slice_axis(0, h.shape_at(0)? - 1, h.shape_at(0)?)?;
        // Batched span means: question vectors [Q,H] and option contexts
        // [O,H], computed via ONE shared compacted (or direct) fp32 cumsum
        // over h and split back into questions and contexts.
        let q_spans: Vec<(usize, usize)> = record.questions.iter().map(|q| q.span).collect();
        let option_spans: Vec<(usize, usize)> = record
            .questions
            .iter()
            .flat_map(|q| q.option_spans.iter().copied())
            .collect();
        let question_count = q_spans.len() as i64;
        let mut all_spans = Vec::with_capacity(q_spans.len() + option_spans.len());
        all_spans.extend_from_slice(&q_spans);
        all_spans.extend_from_slice(&option_spans);
        let mut span_parts = Self::span_means(&h, dtype, &all_spans)?
            .split_sections(&[question_count], 0)?
            .into_iter();
        let questions = span_parts
            .next()
            .ok_or_else(|| Error::from_reason("split_sections under-produced"))?;
        let contexts = span_parts
            .next()
            .ok_or_else(|| Error::from_reason("split_sections under-produced"))?;
        // Lexical embeddings: one take over every option token, one mean pass.
        let mut option_ids_flat: Vec<u32> = Vec::new();
        let mut option_lengths: Vec<usize> = Vec::new();
        for q in &record.questions {
            for &(start, end) in &q.option_spans {
                option_ids_flat.extend_from_slice(&record.input_ids[start..end]);
                option_lengths.push(end - start);
            }
        }
        let lexical = self.group_mean_embeds(output, &option_ids_flat, &option_lengths, dtype)?;
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
                false,
            )?)?;
            let ff = Self::gelu_exact(&self.linear(
                &self.norm(&routed, &format!("{p}.feedforward_norm"))?,
                &format!("{p}.feedforward.0"),
            )?)?;
            routed = routed.add(&self.linear(&ff, &format!("{p}.feedforward.3"))?)?;
        }
        let base = self.linear(&questions, "question_projection")?;
        // Per-question option softmax, padded to [Q, max_options]: index O_tot
        // addresses a zero row, masked out before the softmax either way.
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
        let padded_idx = MxArray::from_uint32(&padded_indices, &[question_count, max_options])?;
        let routed_padded = MxArray::concatenate(
            &routed,
            &MxArray::zeros(&[1, self.config.width], Some(dtype))?,
            0,
        )?;
        let options_padded = routed_padded.take(&padded_idx, 0)?;
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
        // Weighted option summaries as one batched matmul — the same
        // contraction as the expand/mul/sum chain it replaces.
        let summaries = option_weights
            .expand_dims(1)?
            .matmul(&options_padded)?
            .squeeze(Some(&[1]))?;
        let types: Vec<_> = record.questions.iter().map(|q| q.kind).collect();
        let types = MxArray::from_uint32(&types, &[types.len() as i64])?;
        let mut fields = base
            .add(&self.norm(&summaries, "option_summary_norm")?)?
            .add(&self.linear(&global, "global_projection")?)?
            .add(&self.w("type_embedding.weight").take(&types, 0)?)?;
        for i in 0..self.config.layers {
            let p = format!("layers.{i}");
            let norm = self.norm(&fields, &format!("{p}.norm1"))?;
            fields =
                fields.add(&self.attention(&norm, &norm, &format!("{p}.self_attn"), true)?)?;
            fields = fields.add(&self.attention(
                &self.norm(&fields, &format!("{p}.norm2"))?,
                &memory,
                &format!("{p}.multihead_attn"),
                false,
            )?)?;
            let ff = Self::gelu_exact(&self.linear(
                &self.norm(&fields, &format!("{p}.norm3"))?,
                &format!("{p}.linear1"),
            )?)?;
            fields = fields.add(&self.linear(&ff, &format!("{p}.linear2"))?)?;
        }
        fields = self.norm(&fields, "field_norm")?;
        // Scoring flattened over every (question, option) pair; one device
        // readback at the end instead of a GPU sync per question.
        // prior_scale / joint_scale / gate are baked at load: the clip+exp /
        // sigmoid chains no longer dispatch per decide.
        let prior_scale = self.d("prior_scale");
        let joint_scale = self.d("joint_scale");
        let gate = self.d("residual_gate_sigmoid");
        // eps-grouped unit norms: anchors+lexical share 1e-12, field+options
        // share 1e-8. Concat then split issues the whole normalize chain
        // once per group.
        let anchors = questions.add(&global)?;
        let next_normed = |parts: &mut std::vec::IntoIter<MxArray>| -> Result<MxArray> {
            parts
                .next()
                .ok_or_else(|| Error::from_reason("split_sections under-produced"))
        };
        let mut normed12 = Self::unit(&MxArray::concatenate(&anchors, &lexical, 0)?, 1e-12)?
            .split_sections(&[question_count], 0)?
            .into_iter();
        let anchors = next_normed(&mut normed12)?.take(&option_of_question, 0)?;
        let lexical_u = next_normed(&mut normed12)?;
        let prior = lexical_u
            .mul(&anchors)?
            .sum(Some(&[-1]), Some(false))?
            .mul(prior_scale)?;
        let options = self.norm(&routed, "option_norm")?;
        let field = fields.take(&option_of_question, 0)?;
        let mut normed8 = Self::unit(&MxArray::concatenate(&field, &options, 0)?, 1e-8)?
            .split_sections(&[option_count], 0)?
            .into_iter();
        let field_u = next_normed(&mut normed8)?;
        let options_u = next_normed(&mut normed8)?;
        let cosine = field_u.mul(&options_u)?.sum(Some(&[-1]), Some(false))?;
        let product = field.mul(&options)?;
        let distance = field.sub(&options)?.abs()?;
        let features =
            MxArray::concatenate_many(vec![&field, &options, &product, &distance], Some(-1))?;
        let residual = self
            .linear(
                &Self::gelu_exact(&self.linear(&features, "residual_scorer.0")?)?,
                "residual_scorer.3",
            )?
            .squeeze(Some(&[-1]))?;
        let joint = cosine.mul(joint_scale)?.add(&residual)?;
        let logits_flat = prior.add(&joint.mul(gate)?)?;
        // Scatter flat logits into [Q, max_options]; pad slots read index
        // O_tot (a -inf row) so the masked softmax is exact per question.
        let logits_padded = MxArray::concatenate(
            &logits_flat,
            &MxArray::full(&[1], napi::Either::A(f64::NEG_INFINITY), Some(dtype))?,
            0,
        )?;
        let logits = logits_padded.take(&padded_idx, 0)?;
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
        let output_proj = LinearProj::Standard(
            crate::nn::Linear::from_weights(&array("output", &[32, 8]), None).unwrap(),
        );
        let actual = head
            .forward(&array("hidden", &[1, 24, 8]), &record, &output_proj)
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

    #[test]
    fn mixed_f32_f16_head_unifies_to_bf16_on_cuda_and_matches_reference() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../__test__/fixtures/clef");
        let ref_head = JointHead::load(&root, 8).unwrap();
        let config: HeadConfig = serde_json::from_str(
            &std::fs::read_to_string(root.join("joint_head_config.json")).unwrap(),
        )
        .unwrap();
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
        let fp32_weights =
            crate::utils::safetensors::load_safetensors_lazy(root.join("joint_head.safetensors"))
                .unwrap();
        // Mirror GGUF `extract_clef_head_assets`: 2D linear/attention weights
        // decode to Float16 while 1D norms/biases and 0D scalars stay Float32.
        let mut mixed_weights: HashMap<String, MxArray> =
            HashMap::with_capacity(fp32_weights.len());
        for (name, tensor) in fp32_weights {
            let converted = if tensor.ndim().unwrap() == 2 && name != "type_embedding.weight" {
                tensor.astype(DType::Float16).unwrap()
            } else {
                tensor
            };
            mixed_weights.insert(name, converted);
        }
        let mixed_head = JointHead::from_weights(config, mixed_weights, 8).unwrap();

        // All-Float32 checkpoints must remain untouched even on CUDA, and
        // `self_attn` must only bake `in_proj_t` (not unused cross-attn slices).
        assert_eq!(
            ref_head.w("hidden_norm.weight").dtype().unwrap(),
            DType::Float32
        );
        assert_eq!(
            ref_head.d("memory_projection.weight_t").dtype().unwrap(),
            DType::Float32
        );
        assert_eq!(
            ref_head.d("layers.0.self_attn.in_proj_t").dtype().unwrap(),
            DType::Float32
        );
        assert!(
            !ref_head.derived.contains_key("layers.0.self_attn.q"),
            "self_attn should not bake unused cross-attention q/kv slices"
        );

        if crate::accelerator::cuda_ops_available() {
            for (name, tensor) in &mixed_head.weights {
                assert_eq!(
                    tensor.dtype().unwrap(),
                    DType::BFloat16,
                    "expected unified BFloat16 weight on CUDA for {name}"
                );
            }
            for (name, tensor) in &mixed_head.derived {
                assert_eq!(
                    tensor.dtype().unwrap(),
                    DType::BFloat16,
                    "expected unified BFloat16 derived tensor on CUDA for {name}"
                );
            }
        }

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
        let hidden = array("hidden", &[1, 24, 8]);
        let output_proj = LinearProj::Standard(
            crate::nn::Linear::from_weights(&array("output", &[32, 8]), None).unwrap(),
        );
        let expected = ref_head.forward(&hidden, &record, &output_proj).unwrap();
        let actual = mixed_head.forward(&hidden, &record, &output_proj).unwrap();
        let max_error = actual
            .iter()
            .zip(&expected)
            .flat_map(|(a, b)| a.iter().zip(b))
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_error < 5e-3,
            "mixed F32+F16 head drifted from FP32 reference: max_error={max_error}"
        );
        let hidden_bf16 = hidden.astype(DType::BFloat16).unwrap();
        let actual_bf16 = mixed_head
            .forward(&hidden_bf16, &record, &output_proj)
            .unwrap();
        let max_error_bf16 = actual_bf16
            .iter()
            .zip(&expected)
            .flat_map(|(a, b)| a.iter().zip(b))
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_error_bf16 < 5e-3,
            "mixed F32+F16 head with BF16 hidden drifted from FP32 reference: max_error={max_error_bf16}"
        );
    }

    #[test]
    fn gelu_exact_preserves_floating_dtype_and_matches_reference() {
        let vals: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.25).collect();
        let x_f32 = MxArray::from_float32(&vals, &[vals.len() as i64]).unwrap();
        let ref_out = JointHead::gelu_exact(&x_f32).unwrap();
        assert_eq!(ref_out.dtype().unwrap(), DType::Float32);
        let ref_vals = ref_out.to_float32().unwrap();
        for (dtype, tol) in [(DType::Float16, 2e-2_f32), (DType::BFloat16, 5e-2_f32)] {
            let x = x_f32.astype(dtype).unwrap();
            let y = JointHead::gelu_exact(&x).unwrap();
            assert_eq!(y.dtype().unwrap(), dtype, "gelu_exact changed {dtype:?}");
            let got = y.astype(DType::Float32).unwrap().to_float32().unwrap();
            let max_err = got
                .iter()
                .zip(ref_vals.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < tol,
                "gelu_exact drift for {dtype:?}: max_err={max_err}"
            );
        }
    }

    #[test]
    fn span_means_compacted_path_matches_full_cumsum() {
        // Long sequence (T = 256, H = 16) where sparse spans cover well under
        // half of T (triggering the compacted take + cumsum path), including
        // non-contiguous, out-of-order, single-token, and overlapping spans.
        let t = 256usize;
        let h = 16usize;
        let data: Vec<f32> = (0..t * h)
            .map(|i| {
                let row = (i / h) as f32;
                let col = (i % h) as f32;
                (row * 0.13 + col * 0.37).sin() * 2.5 + (row * 0.07 - col * 0.19).cos() * 0.75
            })
            .collect();
        let x_f32 = MxArray::from_float32(&data, &[t as i64, h as i64]).unwrap();
        let spans: Vec<(usize, usize)> = vec![
            (3, 8),     // early span
            (5, 11),    // overlaps (3, 8)
            (120, 121), // single-token span
            (240, 255), // near-end span (where full cumsum accumulates 255 rows)
            (10, 15),   // out-of-order, overlaps (5, 11)
            (200, 212), // non-contiguous mid-late span
        ];
        let total_len: usize = spans.iter().map(|&(s, e)| e - s).sum();
        assert!(total_len * 2 < t, "test must exercise the compacted path");

        // FP32 input -> FP32 output must match full cumsum within < 1e-5.
        let full_f32_ref = JointHead::span_means_from_cumsum(
            &JointHead::span_cumsum(&x_f32).unwrap(),
            DType::Float32,
            &spans,
        )
        .unwrap()
        .to_float32()
        .unwrap();
        let compact_f32_arr = JointHead::span_means(&x_f32, DType::Float32, &spans).unwrap();
        assert_eq!(compact_f32_arr.dtype().unwrap(), DType::Float32);
        assert_eq!(
            compact_f32_arr.shape().unwrap().as_ref(),
            &[spans.len() as i64, h as i64]
        );
        let compact_f32 = compact_f32_arr.to_float32().unwrap();
        let f32_err = compact_f32
            .iter()
            .zip(full_f32_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            f32_err < 1e-5,
            "compacted span_means (FP32) drifted from full cumsum: max_err={f32_err}"
        );

        // BF16 input -> FP32 output (no BF16 output quantization noise) must also
        // match full cumsum within < 1e-5, and BF16 output must preserve dtype/shape.
        let x_bf16 = x_f32.astype(DType::BFloat16).unwrap();
        let full_bf16_f32 = JointHead::span_means_from_cumsum(
            &JointHead::span_cumsum(&x_bf16).unwrap(),
            DType::Float32,
            &spans,
        )
        .unwrap()
        .to_float32()
        .unwrap();
        let compact_bf16_f32 = JointHead::span_means(&x_bf16, DType::Float32, &spans)
            .unwrap()
            .to_float32()
            .unwrap();
        let bf16_f32_err = compact_bf16_f32
            .iter()
            .zip(full_bf16_f32.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            bf16_f32_err < 1e-5,
            "compacted span_means (BF16 -> FP32) drifted from full cumsum: max_err={bf16_f32_err}"
        );
        let compact_bf16 = JointHead::span_means(&x_bf16, DType::BFloat16, &spans).unwrap();
        assert_eq!(compact_bf16.dtype().unwrap(), DType::BFloat16);
        assert_eq!(
            compact_bf16.shape().unwrap().as_ref(),
            &[spans.len() as i64, h as i64]
        );

        // Dense coverage (total_len * 2 >= t) exercises the direct cumsum path.
        let dense_spans = vec![(0, 140), (120, 256)];
        let dense_total: usize = dense_spans.iter().map(|&(s, e)| e - s).sum();
        assert!(dense_total * 2 >= t);
        let dense_expected = JointHead::span_means_from_cumsum(
            &JointHead::span_cumsum(&x_f32).unwrap(),
            DType::Float32,
            &dense_spans,
        )
        .unwrap()
        .to_float32()
        .unwrap();
        let dense_actual = JointHead::span_means(&x_f32, DType::Float32, &dense_spans)
            .unwrap()
            .to_float32()
            .unwrap();
        assert_eq!(dense_actual.as_ref(), dense_expected.as_ref());

        // Out-of-bounds or inverted spans must fail cleanly.
        assert!(JointHead::span_means(&x_f32, DType::Float32, &[(10, 5)]).is_err());
        assert!(JointHead::span_means(&x_f32, DType::Float32, &[(0, t + 1)]).is_err());
    }
}
