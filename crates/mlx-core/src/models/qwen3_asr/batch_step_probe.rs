//! Batched Qwen3-ASR decode step: the fused dense step with `[B,1]` ids and
//! left padding (`static`) against `forward_paged_adapter_batched` over a
//! `PagedKVCacheAdapter` (`paged`), on real prefills. Dense checkpoints only.
//!
//! ```text
//! ASR_BATCH_MODEL=<dir> ASR_BATCH_CLIPS=<file, one mono 16-bit wav per line> \
//!   cargo test -p mlx-core --release --lib batch_step_probe -- --ignored --nocapture --test-threads=1
//! ```

use super::*;
use crate::array::{
    clear_cache, get_active_memory, get_peak_memory, reset_peak_memory, synchronize,
};
use crate::transformer::paged_kv_cache_adapter::{PagedKVCacheAdapter, PagedRaggedRow, SeqId};
use crate::transformer::paged_policy::record_decode_wave;
use std::time::Duration;

const MIB: f64 = 1024.0 * 1024.0;
const BLOCK_SIZE: u32 = 16;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().parse().unwrap())
        .unwrap_or(default)
}

fn env_list(name: &str, default: &str) -> Vec<usize> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(|value| value.trim().parse().unwrap())
        .collect()
}

fn env_words(name: &str, default: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(|value| value.trim().to_string())
        .collect()
}

fn read_wav(path: &str) -> (Vec<f32>, u32) {
    let bytes = fs::read(path).unwrap();
    assert!(
        &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "{path}: not a WAV"
    );
    let (mut rate, mut channels, mut bits_per_sample, mut data) = (0u32, 0u16, 0u16, None);
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = &bytes[pos + 8..(pos + 8 + size).min(bytes.len())];
        if id == b"fmt " {
            channels = u16::from_le_bytes([body[2], body[3]]);
            rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
            bits_per_sample = u16::from_le_bytes([body[14], body[15]]);
        } else if id == b"data" {
            data = Some(body);
        }
        pos += 8 + size + (size & 1);
    }
    assert!(
        channels == 1 && bits_per_sample == 16,
        "{path}: need mono 16-bit"
    );
    let samples = data
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| i16::from_le_bytes(*pair) as f32 / 32768.0)
        .collect();
    (samples, rate)
}

fn quantile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

fn bits(array: &MxArray) -> Vec<u32> {
    let wide = array.astype(DType::Float32).unwrap();
    wide.eval();
    wide.to_float32()
        .unwrap()
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn greedy_token(logits: &MxArray) -> u32 {
    let token = logits
        .argmax(-1, Some(false))
        .unwrap()
        .astype(DType::Uint32)
        .unwrap();
    token.eval();
    token.item_at_uint32(0).unwrap()
}

fn argmax_ids(logits: &MxArray, rows: i64) -> MxArray {
    logits
        .argmax(-1, Some(false))
        .unwrap()
        .astype(DType::Uint32)
        .unwrap()
        .reshape(&[rows, 1])
        .unwrap()
}

fn host_ids(ids: &MxArray) -> Vec<u32> {
    ids.eval();
    ids.to_uint32().unwrap().to_vec()
}

fn mark(label: &str) {
    synchronize();
    std::thread::sleep(Duration::from_millis(30));
    eprintln!("[asr-batch-case] {label}");
}

struct Clip {
    name: String,
    len: i32,
    /// Per layer `[1, kv_heads, cap0, head_dim]`; only rows `[0, len)` are valid.
    keys: Vec<MxArray>,
    values: Vec<MxArray>,
    /// `reference[s + 1]` is the `decode_fused` argmax after feeding `reference[s]`.
    reference: Vec<u32>,
    reference_bits: Vec<Vec<u32>>,
    /// Top-1 minus top-2 logit per step; 0 is an exact tie any reordering may flip.
    reference_margin: Vec<f32>,
}

fn top2_margin(logits: &[u32]) -> f32 {
    let (mut first, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for value in logits.iter().map(|bits| f32::from_bits(*bits)) {
        if value > first {
            second = first;
            first = value;
        } else if value > second {
            second = value;
        }
    }
    first - second
}

struct StaticKv {
    keys: Vec<Option<MxArray>>,
    values: Vec<Option<MxArray>>,
    cache_idx: i32,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Route {
    Static,
    Paged,
}

impl Route {
    fn name(self) -> &'static str {
        match self {
            Route::Static => "static",
            Route::Paged => "paged",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    /// Step N+1 is submitted before token N is read, as `greedy_decode` does.
    Pipe,
    Sync,
}

struct RunOut {
    total_ms: f64,
    intervals: Vec<f64>,
    host_ms: f64,
    tokens: Vec<Vec<u32>>,
}

impl RunOut {
    fn per_step(&self) -> f64 {
        self.total_ms / self.intervals.len().max(1) as f64
    }
    /// The first interval also covers the first step's graph build.
    fn steady(&self) -> &[f64] {
        if self.intervals.len() > 1 {
            &self.intervals[1..]
        } else {
            &self.intervals
        }
    }
}

fn run_steps(
    mode: Mode,
    steps: usize,
    rows: i64,
    ids0: MxArray,
    mut step: impl FnMut(&MxArray) -> MxArray,
) -> RunOut {
    let mut intervals = Vec::with_capacity(steps);
    let mut tokens = Vec::with_capacity(steps);
    let mut host = Duration::ZERO;
    let start = Instant::now();
    match mode {
        Mode::Pipe => {
            let built = Instant::now();
            let mut next = argmax_ids(&step(&ids0), rows);
            host += built.elapsed();
            MxArray::async_eval_arrays(&[&next]);
            let mut last = start;
            for index in 0..steps {
                let scheduled = (index + 1 < steps).then(|| {
                    let built = Instant::now();
                    let following = argmax_ids(&step(&next), rows);
                    host += built.elapsed();
                    MxArray::async_eval_arrays(&[&following]);
                    following
                });
                tokens.push(host_ids(&next));
                let now = Instant::now();
                intervals.push(ms(now - last));
                last = now;
                match scheduled {
                    Some(following) => next = following,
                    None => break,
                }
            }
        }
        Mode::Sync => {
            let mut ids = ids0;
            for _ in 0..steps {
                let begin = Instant::now();
                let next = argmax_ids(&step(&ids), rows);
                host += begin.elapsed();
                tokens.push(host_ids(&next));
                intervals.push(ms(begin.elapsed()));
                ids = next;
            }
        }
    }
    synchronize();
    RunOut {
        total_ms: ms(start.elapsed()),
        intervals,
        host_ms: ms(host),
        tokens,
    }
}

struct Probe {
    inner: Qwen3AsrInner,
    kv_heads: i64,
    head_dim: i64,
    vocab: i64,
    layers: usize,
    language: Option<String>,
}

impl Probe {
    fn load() -> Self {
        let model = std::env::var("ASR_BATCH_MODEL").expect("ASR_BATCH_MODEL");
        let language = resolve_language(Some(
            std::env::var("ASR_BATCH_LANGUAGE")
                .unwrap_or_else(|_| "en".to_string())
                .as_str(),
        ))
        .unwrap();
        let (inner, _) = load_inner(Path::new(&model)).unwrap();
        assert!(
            inner.text_decoder.packed.is_none(),
            "the batch-step probe supports dense checkpoints only"
        );
        let config = &inner.text_decoder.config;
        let probe = Self {
            kv_heads: config.num_key_value_heads as i64,
            head_dim: config.head_dim as i64,
            vocab: config.vocab_size as i64,
            layers: inner.text_decoder.layers.len(),
            language,
            inner,
        };
        eprintln!(
            "[asr-batch-model] path={model} layers={} kv_heads={} head_dim={} vocab={} language={:?} kv_bytes_per_token={}",
            probe.layers,
            probe.kv_heads,
            probe.head_dim,
            probe.vocab,
            probe.language,
            probe.kv_bytes_per_token()
        );
        probe
    }

    fn kv_bytes_per_token(&self) -> i64 {
        2 * self.layers as i64 * self.kv_heads * self.head_dim * 2
    }

    fn clip_paths(limit: usize) -> Vec<String> {
        let list = std::env::var("ASR_BATCH_CLIPS").expect("ASR_BATCH_CLIPS");
        fs::read_to_string(&list)
            .unwrap()
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .take(limit)
            .map(str::to_string)
            .collect()
    }

    fn prefill_clip(&self, path: &str, ref_steps: usize, keep_bits: bool) -> Clip {
        let inner = &self.inner;
        let decoder = &inner.text_decoder;
        let (samples, rate) = read_wav(path);
        let native = resample_mono(&samples, rate, inner.feature_extractor.sample_rate());
        let features = inner.feature_extractor.extract(&native).unwrap();
        let audio = inner.audio_tower.forward(features).unwrap();
        audio.eval();
        let embeddings = decoder
            .embed_prompt(&inner.tokenizer, &audio, None, self.language.as_deref())
            .unwrap();
        let mut caches: Vec<KVCache> = decoder.layers.iter().map(|_| KVCache::new()).collect();
        let logits = decoder.prefill(&embeddings, &mut caches).unwrap();
        let first = greedy_token(&logits);
        let taken = FusedKv::take(&mut caches).unwrap();
        let len = taken.cache_idx;
        let keys: Vec<MxArray> = taken.keys.into_iter().map(Option::unwrap).collect();
        let values: Vec<MxArray> = taken.values.into_iter().map(Option::unwrap).collect();
        let all: Vec<&MxArray> = keys.iter().chain(&values).collect();
        MxArray::eval_arrays(&all).unwrap();

        // The clip keeps its own handles, so the reference decode below never
        // donates (and so never overwrites) the prefill buffers.
        let mut fused = FusedKv {
            keys: keys.iter().cloned().map(Some).collect(),
            values: values.iter().cloned().map(Some).collect(),
            cache_idx: len,
        };
        let mut reference = vec![first];
        let mut reference_bits = Vec::new();
        let mut reference_margin = Vec::new();
        for _ in 0..ref_steps {
            let token = MxArray::from_uint32(&[*reference.last().unwrap()], &[1]).unwrap();
            let logits = decoder.decode_fused(&token, &mut fused).unwrap();
            logits.eval();
            let logit_bits = bits(&logits);
            reference_margin.push(top2_margin(&logit_bits));
            if keep_bits {
                reference_bits.push(logit_bits);
            }
            reference.push(greedy_token(&logits));
        }
        synchronize();
        Clip {
            name: Path::new(path)
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string(),
            len,
            keys,
            values,
            reference,
            reference_bits,
            reference_margin,
        }
    }

    fn prepare_clips(&self, count: usize, ref_steps: usize, byte_clips: usize) -> Vec<Clip> {
        let paths = Self::clip_paths(count);
        assert!(!paths.is_empty(), "ASR_BATCH_CLIPS lists no clip");
        let start = Instant::now();
        let clips: Vec<Clip> = paths
            .iter()
            .enumerate()
            .map(|(index, path)| self.prefill_clip(path, ref_steps, index < byte_clips))
            .collect();
        let eos = &self.inner.config.eos_token_id;
        for (index, clip) in clips.iter().enumerate() {
            let tokens_to_eos = clip
                .reference
                .iter()
                .position(|token| eos.contains(token))
                .map(|position| (position + 1).to_string())
                .unwrap_or_else(|| "none".to_string());
            eprintln!(
                "[asr-batch-clip] index={index} name={} len={} tokens_to_eos={tokens_to_eos} first_ids={:?}",
                clip.name,
                clip.len,
                &clip.reference[..clip.reference.len().min(6)]
            );
        }
        eprintln!(
            "[asr-batch-clips] count={} ref_steps={ref_steps} prepare_ms={:.0}",
            clips.len(),
            ms(start.elapsed())
        );
        clips
    }

    /// The row's `len` valid positions land in columns `[width - len, width)`.
    fn padded_row(&self, source: &MxArray, len: i32, width: i32, cap: i32) -> MxArray {
        source
            .slice(
                &[0, 0, 0, 0],
                &[1, self.kv_heads, len as i64, self.head_dim],
            )
            .unwrap()
            .pad(&[0, 0, 0, 0, width - len, cap - width, 0, 0], 0.0)
            .unwrap()
    }

    fn stack_rows(
        &self,
        rows: &[&Clip],
        layer: usize,
        keys: bool,
        width: i32,
        cap: i32,
    ) -> MxArray {
        let parts: Vec<MxArray> = rows
            .iter()
            .map(|clip| {
                let source = if keys {
                    &clip.keys[layer]
                } else {
                    &clip.values[layer]
                };
                self.padded_row(source, clip.len, width, cap)
            })
            .collect();
        MxArray::concatenate_many(parts.iter().collect(), Some(0)).unwrap()
    }

    fn static_build(&self, rows: &[&Clip], cap: i32) -> StaticKv {
        let width = rows.iter().map(|clip| clip.len).max().unwrap();
        assert!(width < cap, "static width {width} does not fit cap {cap}");
        StaticKv {
            keys: (0..self.layers)
                .map(|layer| Some(self.stack_rows(rows, layer, true, width, cap)))
                .collect(),
            values: (0..self.layers)
                .map(|layer| Some(self.stack_rows(rows, layer, false, width, cap)))
                .collect(),
            cache_idx: width,
        }
    }

    fn eval_static(kv: &StaticKv) {
        let all: Vec<&MxArray> = kv.keys.iter().chain(&kv.values).flatten().collect();
        MxArray::eval_arrays(&all).unwrap();
    }

    /// At B=1 with offset `cache_idx` and padding 0 this is the `decode_fused` graph.
    fn static_step(&self, ids: &MxArray, kv: &mut StaticKv, rope: &[i32], pad: &[i32]) -> MxArray {
        use mlx_sys as sys;
        use std::ptr;

        let decoder = &self.inner.text_decoder;
        let rows = rope.len() as i64;
        assert_eq!(rope.len(), pad.len());
        let rope_offsets = MxArray::from_int32(rope, &[rows]).unwrap();
        let left_padding = pad
            .iter()
            .any(|&pad| pad != 0)
            .then(|| MxArray::from_int32(pad, &[rows]).unwrap());
        let mut layer_weights: Vec<*mut sys::mlx_array> =
            Vec::with_capacity(decoder.layers.len() * 11);
        for layer in &decoder.layers {
            layer_weights.push(layer.get_input_layernorm_weight().handle.0);
            layer_weights.push(layer.get_post_attention_layernorm_weight().handle.0);
            layer_weights.extend([
                layer.self_attn.get_q_proj_weight().handle.0,
                layer.self_attn.get_k_proj_weight().handle.0,
                layer.self_attn.get_v_proj_weight().handle.0,
                layer.self_attn.get_o_proj_weight().handle.0,
            ]);
            layer_weights.push(
                layer
                    .self_attn
                    .get_q_norm_weight()
                    .map(|array| array.handle.0)
                    .unwrap_or(ptr::null_mut()),
            );
            layer_weights.push(
                layer
                    .self_attn
                    .get_k_norm_weight()
                    .map(|array| array.handle.0)
                    .unwrap_or(ptr::null_mut()),
            );
            layer_weights.extend([
                layer.mlp.get_gate_proj_weight().handle.0,
                layer.mlp.get_up_proj_weight().handle.0,
                layer.mlp.get_down_proj_weight().handle.0,
            ]);
        }
        let key_ptrs: Vec<_> = kv
            .keys
            .iter()
            .map(|value| {
                value
                    .as_ref()
                    .map_or(ptr::null_mut(), |array| array.handle.0)
            })
            .collect();
        let value_ptrs: Vec<_> = kv
            .values
            .iter()
            .map(|value| {
                value
                    .as_ref()
                    .map_or(ptr::null_mut(), |array| array.handle.0)
            })
            .collect();
        let mut logits_ptr = ptr::null_mut();
        let mut key_outputs = vec![ptr::null_mut(); decoder.layers.len()];
        let mut value_outputs = vec![ptr::null_mut(); decoder.layers.len()];
        let mut next_cache_idx = 0;
        let final_norm = decoder.final_norm.get_weight();
        let embedding = decoder.embedding.get_weight();
        let mode = std::ffi::CString::new("affine").unwrap();
        unsafe {
            sys::mlx_qwen3_forward_step(
                ids.handle.0,
                embedding.handle.0,
                layer_weights.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                0,
                mode.as_ptr(),
                decoder.layers.len() as i32,
                final_norm.handle.0,
                ptr::null_mut(),
                true,
                decoder.config.hidden_size as i32,
                decoder.config.num_attention_heads as i32,
                decoder.config.num_key_value_heads as i32,
                decoder.config.head_dim as i32,
                decoder.config.rope_parameters.rope_theta as f32,
                decoder.config.rms_norm_eps as f32,
                key_ptrs.as_ptr(),
                value_ptrs.as_ptr(),
                kv.cache_idx,
                rope_offsets.handle.0,
                left_padding
                    .as_ref()
                    .map_or(ptr::null_mut(), |array| array.handle.0),
                &mut logits_ptr,
                key_outputs.as_mut_ptr(),
                value_outputs.as_mut_ptr(),
                &mut next_cache_idx,
            );
        }
        assert!(
            !logits_ptr.is_null()
                && key_outputs.iter().all(|pointer| !pointer.is_null())
                && value_outputs.iter().all(|pointer| !pointer.is_null()),
            "fused step returned a null output"
        );
        // Same handle reuse as `decode_fused`: the previous buffer is only
        // referenced by the new graph, so slice_update can donate it.
        for (slot, pointer) in kv.keys.iter_mut().zip(key_outputs) {
            if let Some(array) = slot
                && let Some(handle) = Arc::get_mut(&mut array.handle)
            {
                unsafe { handle.overwrite(pointer) };
                continue;
            }
            *slot = Some(MxArray::from_handle(pointer, "batch_probe_static_key").unwrap());
        }
        for (slot, pointer) in kv.values.iter_mut().zip(value_outputs) {
            if let Some(array) = slot
                && let Some(handle) = Arc::get_mut(&mut array.handle)
            {
                unsafe { handle.overwrite(pointer) };
                continue;
            }
            *slot = Some(MxArray::from_handle(pointer, "batch_probe_static_value").unwrap());
        }
        kv.cache_idx = next_cache_idx;
        MxArray::from_handle(logits_ptr, "batch_probe_static_logits")
            .unwrap()
            .reshape(&[rows, self.vocab])
            .unwrap()
    }

    fn new_adapter(&self, gpu_memory_mb: u32) -> (PagedKVCacheAdapter, u32) {
        let config = &self.inner.text_decoder.config;
        let pa_config = mlx_paged_attn::PagedAttentionConfig {
            block_size: BLOCK_SIZE,
            gpu_memory_mb,
            head_size: config.head_dim as u32,
            num_kv_heads: config.num_key_value_heads as u32,
            num_layers: config.num_hidden_layers as u32,
            max_seq_len: Some(config.max_position_embeddings as u32),
            max_batch_size: Some(64),
        };
        let cache_dtype = mlx_paged_attn::metal::MetalDtype::BFloat16;
        let num_blocks = pa_config.calculate_num_blocks(cache_dtype);
        let allocator = Arc::new(std::sync::Mutex::new(mlx_paged_attn::BlockAllocator::new(
            num_blocks, num_blocks, BLOCK_SIZE,
        )));
        let pool = mlx_paged_attn::LayerKVPool::new(pa_config, num_blocks, num_blocks, cache_dtype)
            .unwrap();
        let adapter = PagedKVCacheAdapter::new(allocator, Arc::new(pool), BLOCK_SIZE).unwrap();
        (adapter, num_blocks)
    }

    fn token_major(&self, source: &MxArray, len: i32) -> MxArray {
        source
            .slice(
                &[0, 0, 0, 0],
                &[1, self.kv_heads, len as i64, self.head_dim],
            )
            .unwrap()
            .reshape(&[self.kv_heads, len as i64, self.head_dim])
            .unwrap()
            .transpose(Some(&[1, 0, 2]))
            .unwrap()
    }

    /// The native write is lazy; the next step's graph depends on it.
    fn paged_admit(&self, adapter: &mut PagedKVCacheAdapter, rows: &[(SeqId, &Clip)], write: bool) {
        for (seq, clip) in rows {
            adapter.begin_request(*seq).unwrap();
            adapter
                .record_tokens_for(*seq, &vec![0; clip.len as usize])
                .unwrap();
        }
        if !write || rows.is_empty() {
            return;
        }
        let meta: Vec<PagedRaggedRow> = rows
            .iter()
            .map(|(seq, clip)| PagedRaggedRow {
                seq_id: *seq,
                first_logical_position: 0,
                query_len: clip.len as u32,
            })
            .collect();
        for layer in 0..self.layers {
            let keys: Vec<MxArray> = rows
                .iter()
                .map(|(_, clip)| self.token_major(&clip.keys[layer], clip.len))
                .collect();
            let values: Vec<MxArray> = rows
                .iter()
                .map(|(_, clip)| self.token_major(&clip.values[layer], clip.len))
                .collect();
            let keys = MxArray::concatenate_many(keys.iter().collect(), Some(0)).unwrap();
            let values = MxArray::concatenate_many(values.iter().collect(), Some(0)).unwrap();
            adapter
                .update_keys_values_native_ragged(layer as u32, &keys, &values, &meta)
                .unwrap();
        }
    }

    fn paged_step(
        &self,
        adapter: &mut PagedKVCacheAdapter,
        ids: &MxArray,
        seqs: &[SeqId],
    ) -> MxArray {
        let decoder = &self.inner.text_decoder;
        let rows: Vec<(SeqId, u32)> = seqs.iter().map(|&seq| (seq, 0)).collect();
        let planned = record_decode_wave(adapter, &rows, "qwen3_asr_batch_probe").unwrap();
        let mut hidden = decoder.embedding.forward(ids).unwrap();
        for (index, layer) in decoder.layers.iter().enumerate() {
            hidden = layer
                .forward_paged_adapter_batched(&hidden, adapter, index as u32, &planned)
                .unwrap();
        }
        let normed = decoder.final_norm.forward(&hidden).unwrap();
        decoder
            .embedding
            .as_linear(&normed)
            .unwrap()
            .reshape(&[seqs.len() as i64, self.vocab])
            .unwrap()
    }

    fn paged_release(adapter: &mut PagedKVCacheAdapter, seqs: &[SeqId]) {
        for &seq in seqs {
            adapter.release_request_for(seq).unwrap();
        }
    }

    fn eos(&self) -> &[u32] {
        &self.inner.config.eos_token_id
    }

    /// `tokens[s][row]` must equal `reference[s + 1]` of that row's clip.
    fn compare_rows(&self, rows: &[&Clip], tokens: &[Vec<u32>]) -> (usize, usize, Vec<String>) {
        let (mut to_eos, mut all) = (0, 0);
        let mut diverged = Vec::new();
        for (row, clip) in rows.iter().enumerate() {
            let produced: Vec<u32> = tokens.iter().map(|step| step[row]).collect();
            let expected = &clip.reference[1..=produced.len()];
            let first_diff = produced.iter().zip(expected).position(|(a, b)| a != b);
            let eos_at = expected
                .iter()
                .position(|token| self.eos().contains(token))
                .unwrap_or(expected.len() - 1);
            if first_diff.is_none_or(|diff| diff > eos_at) {
                to_eos += 1;
            } else {
                let step = first_diff.unwrap();
                diverged.push(format!(
                    "{row}:{}@{step}:margin={}",
                    clip.name, clip.reference_margin[step]
                ));
            }
            if first_diff.is_none() {
                all += 1;
            }
        }
        (to_eos, all, diverged)
    }
}

/// Teacher-forced B=1 logits against `decode_fused`: bytes for the static
/// step on a rebuilt buffer, max difference and argmax for the paged step.
fn byte_check(probe: &Probe, clips: &[Clip], steps: usize, adapter: &mut PagedKVCacheAdapter) {
    for clip in clips.iter().filter(|clip| !clip.reference_bits.is_empty()) {
        let cap = (((clip.len as usize + steps + 1).div_ceil(256)) * 256) as i32;
        let mut kv = probe.static_build(&[clip], cap);
        Probe::eval_static(&kv);
        let mut mismatched = 0;
        let mut first_mismatch = None;
        for step in 0..steps {
            let ids = MxArray::from_uint32(&[clip.reference[step]], &[1, 1]).unwrap();
            let rope = [clip.len + step as i32];
            let logits = probe.static_step(&ids, &mut kv, &rope, &[0]);
            logits.eval();
            if bits(&logits) != clip.reference_bits[step] {
                mismatched += 1;
                first_mismatch.get_or_insert(step);
            }
        }
        eprintln!(
            "[asr-batch-b1-bytes] clip={} route=static steps={steps} cap={cap} mismatched_steps={mismatched} first_mismatch={first_mismatch:?} equal={}",
            clip.name,
            mismatched == 0
        );

        let seq: SeqId = 900_000;
        probe.paged_admit(adapter, &[(seq, clip)], true);
        let (mut argmax_equal, mut max_abs, mut bytes_equal) = (0, 0.0f32, 0);
        for step in 0..steps {
            let ids = MxArray::from_uint32(&[clip.reference[step]], &[1, 1]).unwrap();
            let logits = probe.paged_step(adapter, &ids, &[seq]);
            let got = bits(&logits);
            if got == clip.reference_bits[step] {
                bytes_equal += 1;
            }
            let worst = got
                .iter()
                .zip(&clip.reference_bits[step])
                .map(|(a, b)| (f32::from_bits(*a) - f32::from_bits(*b)).abs())
                .fold(0.0f32, f32::max);
            max_abs = max_abs.max(worst);
            if greedy_token(&logits) == clip.reference[step + 1] {
                argmax_equal += 1;
            }
        }
        Probe::paged_release(adapter, &[seq]);
        synchronize();
        eprintln!(
            "[asr-batch-b1-bytes] clip={} route=paged steps={steps} bytes_equal_steps={bytes_equal} argmax_equal_steps={argmax_equal} max_abs_logit_diff={max_abs:e}",
            clip.name
        );
    }
}

#[test]
#[ignore = "needs ASR_BATCH_MODEL and ASR_BATCH_CLIPS"]
fn decode_step_sweep() {
    let probe = Probe::load();
    let batch_sizes = env_list("ASR_BATCH_BS", "1,2,4,5,6,8,10,11,12,15,16,20,24,28,32");
    let steps = env_usize("ASR_BATCH_STEPS", 50);
    let rounds = env_usize("ASR_BATCH_ROUNDS", 2);
    let byte_clips = env_usize("ASR_BATCH_BYTE_CLIPS", 4);
    let pool_mb = env_usize("ASR_BATCH_POOL_MB", 2048) as u32;
    let modes: Vec<Mode> = env_words("ASR_BATCH_MODES", "pipe,sync")
        .iter()
        .map(|mode| match mode.as_str() {
            "pipe" => Mode::Pipe,
            "sync" => Mode::Sync,
            other => panic!("unknown mode {other}"),
        })
        .collect();
    let max_b = *batch_sizes.iter().max().unwrap();
    let clips = probe.prepare_clips(max_b, steps, byte_clips);
    let (mut adapter, num_blocks) = probe.new_adapter(pool_mb);
    eprintln!(
        "[asr-batch-pool] gpu_memory_mb={pool_mb} blocks={num_blocks} block_size={BLOCK_SIZE} pool_mib={:.1}",
        adapter.pool_allocated_bytes().unwrap() as f64 / MIB
    );

    byte_check(&probe, &clips, steps, &mut adapter);

    let mut next_seq: SeqId = 1;
    for round in 0..=rounds {
        let mut order = batch_sizes.clone();
        if round % 2 == 1 {
            order.reverse();
        }
        for (slot, &batch) in order.iter().enumerate() {
            let routes = if (round + slot).is_multiple_of(2) {
                [Route::Static, Route::Paged]
            } else {
                [Route::Paged, Route::Static]
            };
            let rows: Vec<&Clip> = (0..batch)
                .map(|index| &clips[index % clips.len()])
                .collect();
            let firsts: Vec<u32> = rows.iter().map(|clip| clip.reference[0]).collect();
            let width = rows.iter().map(|clip| clip.len).max().unwrap();
            let lens: Vec<i32> = rows.iter().map(|clip| clip.len).collect();
            for route in routes {
                for &mode in &modes {
                    synchronize();
                    clear_cache();
                    reset_peak_memory();
                    let active_before = get_active_memory();
                    let rows_i64 = batch as i64;
                    let ids0 = MxArray::from_uint32(&firsts, &[rows_i64, 1]).unwrap();
                    let setup = Instant::now();
                    let (out, kv_bytes, cap) = match route {
                        Route::Static => {
                            let cap = (((width as usize + steps + 1).div_ceil(256)) * 256) as i32;
                            let mut kv = probe.static_build(&rows, cap);
                            Probe::eval_static(&kv);
                            synchronize();
                            let setup_ms = ms(setup.elapsed());
                            let mut real: Vec<i32> = lens.clone();
                            let out = run_steps(mode, steps, rows_i64, ids0, |ids| {
                                let ci = kv.cache_idx;
                                let pad: Vec<i32> = real.iter().map(|len| ci - len).collect();
                                let logits = probe.static_step(ids, &mut kv, &real, &pad);
                                for len in &mut real {
                                    *len += 1;
                                }
                                logits
                            });
                            let kv_bytes = batch as i64 * cap as i64 * probe.kv_bytes_per_token();
                            ((out, setup_ms), kv_bytes, cap)
                        }
                        Route::Paged => {
                            let seqs: Vec<SeqId> =
                                (0..batch as u32).map(|i| next_seq + i).collect();
                            next_seq += batch as u32;
                            let admitted: Vec<(SeqId, &Clip)> =
                                seqs.iter().copied().zip(rows.iter().copied()).collect();
                            probe.paged_admit(&mut adapter, &admitted, true);
                            adapter.eval_pending_pool_writes().unwrap();
                            synchronize();
                            let setup_ms = ms(setup.elapsed());
                            let out = run_steps(mode, steps, rows_i64, ids0, |ids| {
                                probe.paged_step(&mut adapter, ids, &seqs)
                            });
                            Probe::paged_release(&mut adapter, &seqs);
                            let blocks: u64 = lens
                                .iter()
                                .map(|len| (*len as u64 + steps as u64).div_ceil(BLOCK_SIZE as u64))
                                .sum();
                            let kv_bytes =
                                blocks as i64 * BLOCK_SIZE as i64 * probe.kv_bytes_per_token();
                            ((out, setup_ms), kv_bytes, 0)
                        }
                    };
                    let (out, setup_ms) = out;
                    synchronize();
                    let peak = get_peak_memory();
                    let steady = out.steady();
                    let mode_name = match mode {
                        Mode::Pipe => "pipe",
                        Mode::Sync => "sync",
                    };
                    eprintln!(
                        "[asr-batch-step] round={round} route={} mode={mode_name} b={batch} steps={steps} per_step_ms={:.4} median_ms={:.4} p10_ms={:.4} p90_ms={:.4} tok_per_s={:.1} host_ms_per_step={:.4} setup_ms={:.2} width={width} cap={cap} kv_mib={:.1} active_before_mib={:.1} peak_mib={:.1} peak_extra_mib={:.1}",
                        route.name(),
                        out.per_step(),
                        quantile(steady, 0.5),
                        quantile(steady, 0.1),
                        quantile(steady, 0.9),
                        batch as f64 * 1e3 / quantile(steady, 0.5),
                        out.host_ms / steps as f64,
                        setup_ms,
                        kv_bytes as f64 / MIB,
                        active_before / MIB,
                        peak / MIB,
                        (peak - active_before) / MIB,
                    );
                    if round == 1 {
                        let (to_eos, all, diverged) = probe.compare_rows(&rows, &out.tokens);
                        eprintln!(
                            "[asr-batch-ids] route={} mode={mode_name} b={batch} rows_equal_to_eos={to_eos}/{batch} rows_equal_all_steps={all}/{batch} diverged_before_eos={diverged:?}",
                            route.name()
                        );
                    }
                }
            }
        }
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn life(&mut self) -> u32 {
        5 + (self.next() % 11) as u32
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Rule {
    /// Static: rebuild every row (pad + concat) at each step with a leave or join.
    Full,
    /// Static: a joining row overwrites the leaving row's slot; all rows shift
    /// together when the new row is longer than the write column, or when the
    /// shared padding reaches the threshold (0 = only at the buffer end).
    Slot(i32),
    /// Paged: release the leaving request; one native write admits the new one.
    Paged,
}

impl Rule {
    fn parse(word: &str) -> Self {
        match word {
            "full" => Rule::Full,
            "paged" => Rule::Paged,
            other if other.starts_with("slot") => Rule::Slot(other[4..].parse().unwrap()),
            other => panic!("unknown churn rule {other}"),
        }
    }
    fn name(self) -> String {
        match self {
            Rule::Full => "full".to_string(),
            Rule::Slot(threshold) => format!("slot{threshold}"),
            Rule::Paged => "paged".to_string(),
        }
    }
}

struct ChurnRow {
    clip: usize,
    real_len: i32,
    left: u32,
    seq: SeqId,
    produced: Vec<u32>,
}

struct ChurnOut {
    per_step_ms: f64,
    event_host_ms: f64,
    joins: usize,
    shifts_left: usize,
    shifts_right: usize,
    max_width: i32,
    rows_checked: usize,
    rows_equal: usize,
    diverged: Vec<String>,
    peak_extra_mib: f64,
}

/// Moves columns `[from, to)` of every row to start at column `at`.
fn shift(
    probe: &Probe,
    buffer: &MxArray,
    rows: i64,
    from: i32,
    to: i32,
    at: i32,
    cap: i32,
) -> MxArray {
    buffer
        .slice(
            &[0, 0, from as i64, 0],
            &[rows, probe.kv_heads, to as i64, probe.head_dim],
        )
        .unwrap()
        .pad(&[0, 0, 0, 0, at, cap - at - (to - from), 0, 0], 0.0)
        .unwrap()
}

#[allow(clippy::too_many_arguments)]
fn run_churn(
    probe: &Probe,
    clips: &[Clip],
    adapter: &mut PagedKVCacheAdapter,
    next_seq: &mut SeqId,
    rule: Rule,
    batch: usize,
    steps: usize,
    copies: bool,
    seed: u64,
    cap: i32,
) -> ChurnOut {
    let mut rng = Rng(seed);
    let mut rows: Vec<ChurnRow> = Vec::new();
    let mut next_clip = 0usize;
    let mut new_row = |rows: &mut Vec<ChurnRow>, rng: &mut Rng, next_seq: &mut SeqId| {
        let clip = next_clip % clips.len();
        next_clip += 1;
        *next_seq += 1;
        rows.push(ChurnRow {
            clip,
            real_len: clips[clip].len,
            left: rng.life(),
            seq: *next_seq,
            produced: Vec::new(),
        });
        rows.len() - 1
    };
    // `batch_rows[position]` = row id in batch order (the slot for Rule::Slot).
    let mut batch_rows: Vec<usize> = (0..batch)
        .map(|_| new_row(&mut rows, &mut rng, next_seq))
        .collect();
    let rows_i64 = batch as i64;

    synchronize();
    clear_cache();
    let mut kv = None;
    match rule {
        Rule::Paged => {
            let admitted: Vec<(SeqId, &Clip)> = batch_rows
                .iter()
                .map(|&id| (rows[id].seq, &clips[rows[id].clip]))
                .collect();
            probe.paged_admit(adapter, &admitted, true);
            adapter.eval_pending_pool_writes().unwrap();
        }
        Rule::Full | Rule::Slot(_) => {
            let initial: Vec<&Clip> = batch_rows.iter().map(|&id| &clips[rows[id].clip]).collect();
            let built = probe.static_build(&initial, cap);
            Probe::eval_static(&built);
            kv = Some(built);
        }
    }
    synchronize();
    reset_peak_memory();
    let active_before = get_active_memory();

    let (mut joins, mut shifts_left, mut shifts_right, mut max_width) = (0, 0, 0, 0);
    let mut event_host = Duration::ZERO;
    let mut previous: Option<(MxArray, Vec<usize>)> = None;
    let start = Instant::now();
    for step in 0..steps {
        let event_start = Instant::now();
        let ids = match &previous {
            None => {
                let firsts: Vec<u32> = batch_rows
                    .iter()
                    .map(|&id| clips[rows[id].clip].reference[0])
                    .collect();
                MxArray::from_uint32(&firsts, &[rows_i64, 1]).unwrap()
            }
            Some((last_out, _)) => {
                let leaving: Vec<usize> = (0..batch)
                    .filter(|&position| rows[batch_rows[position]].left == 0)
                    .collect();
                if leaving.is_empty() {
                    last_out.clone()
                } else {
                    joins += leaving.len();
                    match rule {
                        Rule::Slot(_) => {
                            let kv = kv.as_mut().unwrap();
                            let mut keep = vec![1u32; batch];
                            let mut fresh = vec![0u32; batch];
                            for &position in &leaving {
                                let id = new_row(&mut rows, &mut rng, next_seq);
                                batch_rows[position] = id;
                                keep[position] = 0;
                                fresh[position] = clips[rows[id].clip].reference[0];
                            }
                            let longest_new = leaving
                                .iter()
                                .map(|&position| rows[batch_rows[position]].real_len)
                                .max()
                                .unwrap();
                            let ci = kv.cache_idx;
                            if longest_new > ci {
                                let gap = longest_new - ci;
                                shifts_right += 1;
                                if copies {
                                    for buffer in kv.keys.iter_mut().chain(kv.values.iter_mut()) {
                                        let moved = shift(
                                            probe,
                                            buffer.as_ref().unwrap(),
                                            rows_i64,
                                            0,
                                            ci,
                                            gap,
                                            cap,
                                        );
                                        *buffer = Some(moved);
                                    }
                                }
                                kv.cache_idx = longest_new;
                            }
                            if copies {
                                let ci = kv.cache_idx;
                                for &position in &leaving {
                                    let clip = &clips[rows[batch_rows[position]].clip];
                                    for layer in 0..probe.layers {
                                        let key =
                                            probe.padded_row(&clip.keys[layer], clip.len, ci, cap);
                                        let value = probe.padded_row(
                                            &clip.values[layer],
                                            clip.len,
                                            ci,
                                            cap,
                                        );
                                        kv.keys[layer]
                                            .as_mut()
                                            .unwrap()
                                            .slice_assign_axis_inplace(
                                                0,
                                                position as i64,
                                                position as i64 + 1,
                                                &key,
                                            )
                                            .unwrap();
                                        kv.values[layer]
                                            .as_mut()
                                            .unwrap()
                                            .slice_assign_axis_inplace(
                                                0,
                                                position as i64,
                                                position as i64 + 1,
                                                &value,
                                            )
                                            .unwrap();
                                    }
                                }
                            }
                            let keep = MxArray::from_uint32(&keep, &[rows_i64, 1]).unwrap();
                            let fresh = MxArray::from_uint32(&fresh, &[rows_i64, 1]).unwrap();
                            last_out.mul(&keep).unwrap().add(&fresh).unwrap()
                        }
                        Rule::Full | Rule::Paged => {
                            let survivors: Vec<usize> = (0..batch)
                                .filter(|position| !leaving.contains(position))
                                .collect();
                            let mut order: Vec<usize> = survivors
                                .iter()
                                .map(|&position| batch_rows[position])
                                .collect();
                            let mut fresh_ids = Vec::new();
                            for _ in &leaving {
                                let id = new_row(&mut rows, &mut rng, next_seq);
                                fresh_ids.push(id);
                            }
                            order.extend(&fresh_ids);
                            let fresh_tokens: Vec<u32> = fresh_ids
                                .iter()
                                .map(|&id| clips[rows[id].clip].reference[0])
                                .collect();
                            let fresh_tokens =
                                MxArray::from_uint32(&fresh_tokens, &[fresh_ids.len() as i64, 1])
                                    .unwrap();
                            let ids = if survivors.is_empty() {
                                fresh_tokens
                            } else {
                                let index: Vec<i32> =
                                    survivors.iter().map(|&position| position as i32).collect();
                                let index =
                                    MxArray::from_int32(&index, &[index.len() as i64]).unwrap();
                                let kept = last_out.take(&index, 0).unwrap();
                                MxArray::concatenate_many(vec![&kept, &fresh_tokens], Some(0))
                                    .unwrap()
                            };
                            if rule == Rule::Paged {
                                let leaving_seqs: Vec<SeqId> = leaving
                                    .iter()
                                    .map(|&position| rows[batch_rows[position]].seq)
                                    .collect();
                                Probe::paged_release(adapter, &leaving_seqs);
                                let admitted: Vec<(SeqId, &Clip)> = fresh_ids
                                    .iter()
                                    .map(|&id| (rows[id].seq, &clips[rows[id].clip]))
                                    .collect();
                                probe.paged_admit(adapter, &admitted, copies);
                            } else {
                                let kv = kv.as_mut().unwrap();
                                let ci = kv.cache_idx;
                                let width =
                                    order.iter().map(|&id| rows[id].real_len).max().unwrap();
                                assert!(width < cap, "full rebuild width {width} >= cap {cap}");
                                if copies {
                                    for layer in 0..probe.layers {
                                        for keys_side in [true, false] {
                                            let old = if keys_side {
                                                kv.keys[layer].as_ref().unwrap()
                                            } else {
                                                kv.values[layer].as_ref().unwrap()
                                            };
                                            let mut parts: Vec<MxArray> = survivors
                                                .iter()
                                                .map(|&position| {
                                                    let real = rows[batch_rows[position]].real_len;
                                                    old.slice(
                                                        &[
                                                            position as i64,
                                                            0,
                                                            (ci - real) as i64,
                                                            0,
                                                        ],
                                                        &[
                                                            position as i64 + 1,
                                                            probe.kv_heads,
                                                            ci as i64,
                                                            probe.head_dim,
                                                        ],
                                                    )
                                                    .unwrap()
                                                    .pad(
                                                        &[
                                                            0,
                                                            0,
                                                            0,
                                                            0,
                                                            width - real,
                                                            cap - width,
                                                            0,
                                                            0,
                                                        ],
                                                        0.0,
                                                    )
                                                    .unwrap()
                                                })
                                                .collect();
                                            for &id in &fresh_ids {
                                                let clip = &clips[rows[id].clip];
                                                let source = if keys_side {
                                                    &clip.keys[layer]
                                                } else {
                                                    &clip.values[layer]
                                                };
                                                parts.push(
                                                    probe.padded_row(source, clip.len, width, cap),
                                                );
                                            }
                                            let rebuilt = MxArray::concatenate_many(
                                                parts.iter().collect(),
                                                Some(0),
                                            )
                                            .unwrap();
                                            if keys_side {
                                                kv.keys[layer] = Some(rebuilt);
                                            } else {
                                                kv.values[layer] = Some(rebuilt);
                                            }
                                        }
                                    }
                                }
                                kv.cache_idx = width;
                            }
                            batch_rows = order;
                            ids
                        }
                    }
                }
            }
        };
        // Drop leading columns that are padding for every row.
        if let Some(kv) = kv.as_mut() {
            let ci = kv.cache_idx;
            let longest = batch_rows
                .iter()
                .map(|&id| rows[id].real_len)
                .max()
                .unwrap();
            let unused = ci - longest;
            let threshold = match rule {
                Rule::Slot(threshold) if threshold > 0 => threshold,
                _ => i32::MAX,
            };
            if unused > 0 && (unused >= threshold || ci + 1 >= cap) {
                shifts_left += 1;
                if copies {
                    for buffer in kv.keys.iter_mut().chain(kv.values.iter_mut()) {
                        let moved = shift(
                            probe,
                            buffer.as_ref().unwrap(),
                            rows_i64,
                            unused,
                            ci,
                            0,
                            cap,
                        );
                        *buffer = Some(moved);
                    }
                }
                kv.cache_idx = longest;
            }
            assert!(
                kv.cache_idx + 1 < cap,
                "static width {} reached cap {cap}",
                kv.cache_idx
            );
            max_width = max_width.max(kv.cache_idx);
        }
        event_host += event_start.elapsed();

        let logits = match rule {
            Rule::Paged => {
                let seqs: Vec<SeqId> = batch_rows.iter().map(|&id| rows[id].seq).collect();
                probe.paged_step(adapter, &ids, &seqs)
            }
            Rule::Full | Rule::Slot(_) => {
                let kv = kv.as_mut().unwrap();
                let ci = kv.cache_idx;
                let rope: Vec<i32> = batch_rows.iter().map(|&id| rows[id].real_len).collect();
                let pad: Vec<i32> = rope.iter().map(|real| ci - real).collect();
                probe.static_step(&ids, kv, &rope, &pad)
            }
        };
        let out = argmax_ids(&logits, rows_i64);
        MxArray::async_eval_arrays(&[&out]);
        if let Some((last_out, members)) = previous.take() {
            let tokens = host_ids(&last_out);
            for (position, &id) in members.iter().enumerate() {
                rows[id].produced.push(tokens[position]);
            }
        }
        for &id in &batch_rows {
            rows[id].left -= 1;
            rows[id].real_len += 1;
        }
        previous = Some((out, batch_rows.clone()));
        if step + 1 == steps
            && let Some((last_out, members)) = previous.take()
        {
            let tokens = host_ids(&last_out);
            for (position, &id) in members.iter().enumerate() {
                rows[id].produced.push(tokens[position]);
            }
        }
    }
    synchronize();
    let total = ms(start.elapsed());
    let peak = get_peak_memory();
    if rule == Rule::Paged {
        let live: Vec<SeqId> = batch_rows.iter().map(|&id| rows[id].seq).collect();
        Probe::paged_release(adapter, &live);
    }

    let (mut rows_checked, mut rows_equal) = (0, 0);
    let mut diverged = Vec::new();
    if copies {
        for row in &rows {
            let reference = &clips[row.clip].reference;
            let span = row.produced.len().min(reference.len() - 1);
            if span == 0 {
                continue;
            }
            let expected = &reference[1..=span];
            let stop = expected
                .iter()
                .position(|token| probe.eos().contains(token))
                .map_or(span, |position| position + 1);
            rows_checked += 1;
            match row.produced[..stop]
                .iter()
                .zip(&expected[..stop])
                .position(|(a, b)| a != b)
            {
                None => rows_equal += 1,
                Some(step) => diverged.push(format!(
                    "{}@{step}:margin={}",
                    clips[row.clip].name, clips[row.clip].reference_margin[step]
                )),
            }
        }
    }
    ChurnOut {
        per_step_ms: total / steps as f64,
        event_host_ms: ms(event_host) / steps as f64,
        joins,
        shifts_left,
        shifts_right,
        max_width,
        rows_checked,
        rows_equal,
        diverged,
        peak_extra_mib: (peak - active_before) / MIB,
    }
}

#[test]
#[ignore = "needs ASR_BATCH_MODEL and ASR_BATCH_CLIPS"]
fn decode_step_churn() {
    let probe = Probe::load();
    let batch_sizes = env_list("ASR_BATCH_CHURN_BS", "16,32");
    let steps = env_usize("ASR_BATCH_CHURN_STEPS", 120);
    let rounds = env_usize("ASR_BATCH_ROUNDS", 2);
    let clip_count = env_usize("ASR_BATCH_CHURN_CLIPS", 48);
    let cap = env_usize("ASR_BATCH_CAP", 256) as i32;
    let pool_mb = env_usize("ASR_BATCH_POOL_MB", 2048) as u32;
    let seed = env_usize("ASR_BATCH_SEED", 7) as u64;
    let rules: Vec<Rule> = env_words("ASR_BATCH_RULES", "full,slot32,slot0,paged")
        .iter()
        .map(|word| Rule::parse(word))
        .collect();
    // Rows live at most 15 steps, so 16 reference tokens cover every row.
    let clips = probe.prepare_clips(clip_count, 16, 0);
    let (mut adapter, _) = probe.new_adapter(pool_mb);
    let mut next_seq: SeqId = 1_000;
    for round in 0..=rounds {
        for &batch in &batch_sizes {
            let mut order = rules.clone();
            if round % 2 == 1 {
                order.reverse();
            }
            for (slot, &rule) in order.iter().enumerate() {
                let arms = if (round + slot).is_multiple_of(2) {
                    [true, false]
                } else {
                    [false, true]
                };
                for copies in arms {
                    let out = run_churn(
                        &probe,
                        &clips,
                        &mut adapter,
                        &mut next_seq,
                        rule,
                        batch,
                        steps,
                        copies,
                        seed,
                        cap,
                    );
                    eprintln!(
                        "[asr-batch-churn] round={round} rule={} b={batch} arm={} steps={steps} per_step_ms={:.4} event_host_ms_per_step={:.4} joins={} shifts_left={} shifts_right={} max_width={} rows_equal={}/{} peak_extra_mib={:.1} diverged={:?}",
                        rule.name(),
                        if copies { "churn" } else { "control" },
                        out.per_step_ms,
                        out.event_host_ms,
                        out.joins,
                        out.shifts_left,
                        out.shifts_right,
                        out.max_width,
                        out.rows_equal,
                        out.rows_checked,
                        out.peak_extra_mib,
                        out.diverged,
                    );
                }
            }
        }
    }
}

/// One marked pipelined segment per route and B, for `MLX_METAL_COMMAND_TRACE=1`.
#[test]
#[ignore = "needs ASR_BATCH_MODEL, ASR_BATCH_CLIPS and MLX_METAL_COMMAND_TRACE=1"]
fn decode_step_trace() {
    let probe = Probe::load();
    let batch_sizes = env_list("ASR_BATCH_TRACE_BS", "1,8,15,16,32");
    let steps = env_usize("ASR_BATCH_TRACE_STEPS", 20);
    let rounds = env_usize("ASR_BATCH_TRACE_ROUNDS", 1);
    let max_b = *batch_sizes.iter().max().unwrap();
    let clips = probe.prepare_clips(max_b, 0, 0);
    let (mut adapter, _) = probe.new_adapter(2048);
    let mut next_seq: SeqId = 500_000;
    for round in 0..rounds {
        let mut order = batch_sizes.clone();
        if round % 2 == 1 {
            order.reverse();
        }
        for (slot, &batch) in order.iter().enumerate() {
            let routes = if (round + slot).is_multiple_of(2) {
                [Route::Static, Route::Paged]
            } else {
                [Route::Paged, Route::Static]
            };
            for route in routes {
                let rows: Vec<&Clip> = (0..batch)
                    .map(|index| &clips[index % clips.len()])
                    .collect();
                let firsts: Vec<u32> = rows.iter().map(|clip| clip.reference[0]).collect();
                let rows_i64 = batch as i64;
                let lens: Vec<i32> = rows.iter().map(|clip| clip.len).collect();
                let seqs: Vec<SeqId> = (0..batch as u32).map(|i| next_seq + i).collect();
                next_seq += batch as u32;
                let mut kv = None;
                match route {
                    Route::Static => {
                        let built = probe.static_build(&rows, 256);
                        Probe::eval_static(&built);
                        kv = Some(built);
                    }
                    Route::Paged => {
                        let admitted: Vec<(SeqId, &Clip)> =
                            seqs.iter().copied().zip(rows.iter().copied()).collect();
                        probe.paged_admit(&mut adapter, &admitted, true);
                        adapter.eval_pending_pool_writes().unwrap();
                    }
                }
                let mut real = lens.clone();
                let mut step = |ids: &MxArray| match route {
                    Route::Static => {
                        let kv = kv.as_mut().unwrap();
                        let ci = kv.cache_idx;
                        let pad: Vec<i32> = real.iter().map(|len| ci - len).collect();
                        let logits = probe.static_step(ids, kv, &real, &pad);
                        for len in &mut real {
                            *len += 1;
                        }
                        logits
                    }
                    Route::Paged => probe.paged_step(&mut adapter, ids, &seqs),
                };
                let ids0 = MxArray::from_uint32(&firsts, &[rows_i64, 1]).unwrap();
                let warm = run_steps(Mode::Sync, 3, rows_i64, ids0, &mut step);
                let ids1 =
                    MxArray::from_uint32(warm.tokens.last().unwrap(), &[rows_i64, 1]).unwrap();
                mark(&format!(
                    "{} b{batch} steps{steps} round{round}",
                    route.name()
                ));
                let out = run_steps(Mode::Pipe, steps, rows_i64, ids1, &mut step);
                mark("__end__");
                eprintln!(
                    "[asr-batch-trace-wall] round={round} route={} b={batch} steps={steps} per_step_ms={:.4} median_ms={:.4}",
                    route.name(),
                    out.per_step(),
                    quantile(out.steady(), 0.5)
                );
                if route == Route::Paged {
                    Probe::paged_release(&mut adapter, &seqs);
                }
            }
        }
    }
}
