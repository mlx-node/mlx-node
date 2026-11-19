use super::*;
use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
use crate::models::qwen4_exp::live::{Fallback, Live, Residency};
use crate::utils::gguf::{GgufMetaValue, GgufTensorType};

/// The Flash Next MoE shape the device route admits (hidden 2560, expert
/// width 640, top 10 and a 640-wide Q8 shared expert), with head_dim 128 so
/// the pages are int8, and the paged tiny fixture's small size everywhere
/// else: 12 experts, 4 layers (GDN 0-2, attention 3, PLE at 1), `hc_count` 2.
const MOE_SHAPE: [(&str, u32); 10] = [
    ("embedding_length", 2560),
    ("attention.key_length", 128),
    ("rope.dimension_count", 8),
    ("expert_count", 12),
    ("expert_used_count", 10),
    ("expert_feed_forward_length", 640),
    ("expert_shared_feed_forward_length", 640),
    ("context_length", 256),
    ("attention.head_count", 4),
    ("attention.head_count_kv", 2),
];

/// Deterministic payload bytes (splitmix64).
struct Payload(u64);

impl Payload {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn bytes(&mut self, out: &mut Vec<u8>, count: usize) {
        for _ in 0..count.div_ceil(8) {
            out.extend(self.next().to_le_bytes());
        }
        out.truncate(out.len() - (count.div_ceil(8) * 8 - count));
    }

    /// Uniform in `[-bound, bound)`.
    fn uniform(&mut self, bound: f32) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * bound
    }

    fn bf16(&mut self, count: usize, bound: f32) -> Vec<u8> {
        (0..count)
            .flat_map(|_| {
                half::bf16::from_f32(self.uniform(bound))
                    .to_bits()
                    .to_le_bytes()
            })
            .collect()
    }

    /// `rows` rows of `width` values in `[-bound, bound]` as `ty` blocks.
    fn quantized(&mut self, ty: GgufTensorType, rows: usize, width: usize, bound: f32) -> Vec<u8> {
        let f16 = |v: f32| half::f16::from_f32(v).to_bits().to_le_bytes();
        // Every sub-block scale and min is 1: value = d * q - dmin.
        let k_scales = [1, 1, 1, 1, 1, 1, 1, 1, 0x11, 0x11, 0x11, 0x11];
        let mut out = Vec::new();
        for _ in 0..rows {
            match ty {
                GgufTensorType::Q4K | GgufTensorType::Q5K => {
                    let levels = if ty == GgufTensorType::Q4K {
                        15.0
                    } else {
                        31.0
                    };
                    for _ in 0..width / 256 {
                        out.extend(f16(2.0 * bound / levels));
                        out.extend(f16(bound));
                        out.extend(k_scales);
                        if ty == GgufTensorType::Q5K {
                            self.bytes(&mut out, 32);
                        }
                        self.bytes(&mut out, 128);
                    }
                }
                GgufTensorType::Q5_1 => {
                    for _ in 0..width / 32 {
                        out.extend(f16(2.0 * bound / 31.0));
                        out.extend(f16(-bound));
                        self.bytes(&mut out, 20);
                    }
                }
                GgufTensorType::Q8_0 => {
                    for _ in 0..width / 32 {
                        out.extend(f16(bound / 127.0));
                        out.extend((0..32).map(|_| ((self.next() % 255) as i32 - 127) as i8 as u8));
                    }
                }
                other => panic!("no fixture blocks for {other:?}"),
            }
        }
        out
    }
}

/// Writes the [`MOE_SHAPE`] checkpoint to `path` from the paged tiny
/// fixture: its metadata with the shape keys replaced, each tensor at the
/// new shape. Projections are uniform in `±1/sqrt(width)` like the tiny
/// fixture's; norms are 1; GDN tensors that do not touch the hidden width
/// keep the tiny bytes. Routed gate/up are Q4_K (Q5_K in layer 2), routed
/// down Q5_1 (Q8_0 in layer 3), the shared expert Q8_0, the rest BF16/F32.
/// `repeat` stacks the 4 layers that many times with new weights (layer
/// 4k + j has layer j's kind and formats; PLE stays at layer 1 only).
fn write_moe_shape_gguf(path: &Path, repeat: usize) -> u64 {
    use crate::utils::gguf::parse_gguf;
    use crate::utils::gguf_test_fixtures::build_minimal_gguf;

    let source_path = fixture().join("bf16/paged/model.gguf");
    let source = parse_gguf(&source_path).unwrap();
    let source_bytes = std::fs::read(&source_path).unwrap();
    let mut metadata: Vec<(String, GgufMetaValue)> = source
        .metadata
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    for (key, value) in MOE_SHAPE {
        let key = format!("qwen4exp.{key}");
        let (_, current) = metadata
            .iter_mut()
            .find(|(name, _)| *name == key)
            .unwrap_or_else(|| panic!("the tiny fixture has no {key}"));
        *current = GgufMetaValue::Uint32(value);
    }
    for (name, value) in metadata.iter_mut().filter(|_| repeat > 1) {
        if name == "qwen4exp.block_count" {
            *value = GgufMetaValue::Uint32(4 * repeat as u32);
        }
        if name == "qwen4exp.attention.compress_ratios" {
            *value = match &*value {
                GgufMetaValue::ArrayU32(v) => GgufMetaValue::ArrayU32(v.repeat(repeat)),
                GgufMetaValue::ArrayI32(v) => GgufMetaValue::ArrayI32(v.repeat(repeat)),
                GgufMetaValue::ArrayU64(v) => GgufMetaValue::ArrayU64(v.repeat(repeat)),
                GgufMetaValue::ArrayI64(v) => GgufMetaValue::ArrayI64(v.repeat(repeat)),
                other => panic!("no repeat rule for compress ratios {other:?}"),
            };
        }
    }
    let meta = |key: &str| {
        metadata
            .iter()
            .find(|(name, _)| name == &format!("qwen4exp.{key}"))
            .and_then(|(_, value)| value.as_u64())
            .unwrap() as usize
    };
    let h = meta("embedding_length");
    let hc = h * meta("hyper_connection.count");
    let lowrank = meta("hyper_connection.low_rank");
    let (experts, width) = (meta("expert_count"), meta("expert_feed_forward_length"));
    let shared = meta("expert_shared_feed_forward_length");
    let (heads, kv_heads) = (
        meta("attention.head_count"),
        meta("attention.head_count_kv"),
    );
    let head_dim = meta("attention.key_length");
    let mut payload = Payload(0x5EED_F1A5_4E07);
    let mut tensors: Vec<(String, Vec<u64>, GgufTensorType, Vec<u8>)> = Vec::new();
    let stacked = (0..repeat).flat_map(|rep| source.tensors.iter().map(move |t| (rep, t)));
    for (rep, tensor) in stacked {
        let (layer, local) = match tensor.name.strip_prefix("blk.") {
            Some(rest) => {
                let (layer, local) = rest.split_once('.').unwrap();
                (layer.parse::<usize>().ok(), local)
            }
            None => (None, tensor.name.as_str()),
        };
        if rep > 0 && (layer.is_none() || local.starts_with("ple_")) {
            continue;
        }
        let name = match layer {
            Some(layer) if rep > 0 => format!("blk.{}.{local}", layer + 4 * rep),
            _ => tensor.name.clone(),
        };
        // The tiny shape in row-major order: the leading dims are rows.
        let tiny: Vec<usize> = tensor.dims.iter().rev().map(|&d| d as usize).collect();
        let hc_wide = local.starts_with("hc_") || local.starts_with("output_hc_");
        let hc_tensor = |part: &str| hc_wide && local.ends_with(part);
        let (shape, ty, data) = if matches!(
            local,
            "ssm_conv1d.weight"
                | "ssm_dt.bias"
                | "ssm_a"
                | "ssm_norm.weight"
                | "indexer.q_norm.weight"
                | "indexer.k_norm.weight"
                | "per_layer_token_embd.weight"
        ) {
            let start = (source.data_offset + tensor.offset) as usize;
            let end = start + tensor.data_size().unwrap() as usize;
            (tiny, tensor.tensor_type, source_bytes[start..end].to_vec())
        } else if local == "ffn_gate_exps.weight" || local == "ffn_up_exps.weight" {
            let ty = if layer == Some(2) {
                GgufTensorType::Q5K
            } else {
                GgufTensorType::Q4K
            };
            let data = payload.quantized(ty, experts * width, h, 1.0 / (h as f32).sqrt());
            (vec![experts, width, h], ty, data)
        } else if local == "ffn_down_exps.weight" {
            let ty = if layer == Some(3) {
                GgufTensorType::Q8_0
            } else {
                GgufTensorType::Q5_1
            };
            let data = payload.quantized(ty, experts * h, width, 1.0 / (width as f32).sqrt());
            (vec![experts, h, width], ty, data)
        } else if local == "ffn_gate_shexp.weight" || local == "ffn_up_shexp.weight" {
            let ty = GgufTensorType::Q8_0;
            let data = payload.quantized(ty, shared, h, 1.0 / (h as f32).sqrt());
            (vec![shared, h], ty, data)
        } else if local == "ffn_down_shexp.weight" {
            let ty = GgufTensorType::Q8_0;
            let data = payload.quantized(ty, h, shared, 1.0 / (shared as f32).sqrt());
            (vec![h, shared], ty, data)
        } else if tensor.tensor_type == GgufTensorType::F32 && tiny.len() == 1 {
            let len = if hc_wide || local.starts_with("ple_norm_") {
                hc
            } else if local == "attn_q_norm.weight" || local == "attn_k_norm.weight" {
                head_dim
            } else {
                panic!("no moe-shape rule for {}", tensor.name)
            };
            let data = (0..len).flat_map(|_| 1f32.to_le_bytes()).collect();
            (vec![len], GgufTensorType::F32, data)
        } else {
            let shape = match local {
                "token_embd.weight" | "output.weight" => vec![tiny[0], h],
                "attn_qkv.weight"
                | "attn_gate.weight"
                | "ssm_beta.weight"
                | "ssm_alpha.weight"
                | "indexer.q_proj.weight"
                | "indexer.k_proj.weight" => vec![tiny[0], h],
                "ssm_out.weight" => vec![h, tiny[1]],
                "ffn_gate_inp.weight" => vec![experts, h],
                "ffn_gate_inp_shexp.weight" => vec![h],
                "attn_q.weight" => vec![2 * heads * head_dim, h],
                "attn_k.weight" | "attn_v.weight" => vec![kv_heads * head_dim, h],
                "attn_output.weight" => vec![h, heads * head_dim],
                "ple_key.weight" => vec![hc, tiny[1]],
                "ple_value.weight" => vec![h, tiny[1]],
                "ple_conv1d.weight" => vec![hc, tiny[1]],
                _ if hc_tensor("_down.weight") => vec![lowrank, hc],
                _ if hc_tensor("_up.weight") => vec![hc, lowrank],
                _ if hc_tensor("_inject.weight") => vec![tiny[0], hc],
                _ => panic!("no moe-shape rule for {}", tensor.name),
            };
            assert_eq!(tensor.tensor_type, GgufTensorType::BF16, "{}", tensor.name);
            let bound = match local {
                "token_embd.weight" => 0.3,
                "ple_conv1d.weight" => 0.0625,
                _ => 1.0 / (*shape.last().unwrap() as f32).sqrt(),
            };
            let data = payload.bf16(shape.iter().product(), bound);
            (shape, GgufTensorType::BF16, data)
        };
        let dims = shape.iter().rev().map(|&d| d as u64).collect();
        tensors.push((name, dims, ty, data));
    }
    let metadata: Vec<_> = metadata
        .iter()
        .map(|(key, value)| (key.as_str(), value.clone()))
        .collect();
    let descriptors: Vec<_> = tensors
        .iter()
        .map(|(name, dims, ty, data)| (name.as_str(), dims.as_slice(), *ty, data.as_slice()))
        .collect();
    let file = build_minimal_gguf(&metadata, &descriptors);
    std::fs::write(path, &file).unwrap();
    file.len() as u64
}

/// The [`MOE_SHAPE`] checkpoint, written once per test process into the
/// temp dir. Earlier processes' copies are removed once they have exited.
pub(super) fn moe_shape_fixture() -> &'static Path {
    static FIXTURE: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    FIXTURE.get_or_init(|| write_moe_shape_fixture("model.gguf", 1))
}

/// [`moe_shape_fixture`] with 12 layers: attention at 3, 7 and 11, and a
/// decode pass submits route groups after layers 1, 4, 7 and 10.
pub(super) fn deep_moe_shape_fixture() -> &'static Path {
    static FIXTURE: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    FIXTURE.get_or_init(|| write_moe_shape_fixture("deep.gguf", 3))
}

fn write_moe_shape_fixture(file: &str, repeat: usize) -> PathBuf {
    const PREFIX: &str = "qwen4-moe-shape-";
    let root = std::env::temp_dir();
    for entry in std::fs::read_dir(&root).into_iter().flatten().flatten() {
        let name = entry.file_name();
        let Some(pid) = name
            .to_str()
            .and_then(|name| name.strip_prefix(PREFIX))
            .and_then(|pid| pid.parse::<libc::pid_t>().ok())
        else {
            continue;
        };
        let exited = unsafe { libc::kill(pid, 0) } != 0
            && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH);
        if exited {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
    let dir = root.join(format!("{PREFIX}{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(file);
    let started = std::time::Instant::now();
    let bytes = write_moe_shape_gguf(&path, repeat);
    eprintln!(
        "moe-shape fixture: {bytes} bytes written in {:?} to {}",
        started.elapsed(),
        path.display()
    );
    path
}

/// A decoder over `fixture` (a [`moe_shape_fixture`]) with int8 pages and
/// no store plan yet.
pub(super) fn moe_shape_decoder(fixture: &Path) -> (decoder::Decoder, media::Vision) {
    let store = weights::Store::open_fixture(fixture, None).unwrap();
    let mut c = gguf::config(&store).unwrap();
    c.kv_format = load_plan::resolve_kv_format("moe-shape fixture", None, &c).unwrap();
    let vision = media::Vision::metadata(&serde_json::Value::Null, &store, &c).unwrap();
    let mut decoder = decoder::Decoder::new(c.clone(), store).unwrap();
    decoder.paged = Some(paged::create_fixture(&c).unwrap());
    (decoder, vision)
}

/// The [`moe_shape_fixture`] served like the app's partial Flash Next
/// load: a streamed store of `capacity` slots per layer (all empty), int8
/// pages, no MTP head. Needs a scoped test host for the slot admission.
pub(super) fn moe_shape_inner(capacity: usize) -> Inner {
    moe_shape_inner_at(moe_shape_fixture(), capacity)
}

fn moe_shape_inner_at(fixture: &Path, capacity: usize) -> Inner {
    let (mut decoder, vision) = moe_shape_decoder(fixture);
    decoder.weights.stream_fixture_store(capacity).unwrap();
    decoder.register_cache_limit();
    decoder.weights.publish_live();
    decoder.residency_changed();
    inner_over(decoder, vision, false)
}

/// Loads `experts` into every layer's slots, as the host path would on its
/// first routes.
pub(super) fn fill_slots(decoder: &mut decoder::Decoder, experts: &[u32]) {
    let c = decoder.config.clone();
    for layer in 0..c.num_hidden_layers {
        decoder
            .weights
            .expert_slots(layer, c.num_experts, c.num_experts_per_tok, experts)
            .unwrap()
            .expect("the fixture's experts fit the slots");
    }
}

/// On the real MoE shape every decode token of a filled streamed store
/// takes the device route: one try per token and no replay, no layer reads
/// its routes back, the fused routed+shared kernel runs every layer, routes
/// change between tokens, logits stay finite and the pages are int8.
/// MUTATION: `forward_with_device_routes` refuses the device gate → red (no
/// tries, read-backs).
/// MUTATION: `tentative_shared_experts` returns `None` → red (no fused layer;
/// the pass still takes the device route through `tentative_experts`).
#[test]
fn fixture_takes_the_device_route_path() {
    let _host = memory::scoped_fixture_host();
    let mut inner = moe_shape_inner(12);
    let c = inner.decoder.config.clone();
    assert_eq!(c.kv_format, crate::transformer::KvFormat::Int8);
    inner.activate_paged_seq(1).unwrap();
    let all: Vec<u32> = (0..c.num_experts as u32).collect();
    fill_slots(&mut inner.decoder, &all);
    for layer in 0..c.num_hidden_layers {
        let mut held = inner.decoder.weights.slot_occupants(layer).unwrap();
        held.sort_unstable();
        assert_eq!(held, all.iter().copied().map(Some).collect::<Vec<_>>());
    }
    let live = Arc::clone(&inner.decoder.weights.live);
    let tries = live.route_tries.load(Ordering::Relaxed);
    let replays = live.route_replays.load(Ordering::Relaxed);
    let readbacks = inner.decoder.route_readbacks;
    const STEPS: u64 = 8;
    let mut token = 3;
    for step in 0..STEPS {
        let logits = inner.decoder.step(token).unwrap();
        assert!(
            logits.to_float32().unwrap().iter().all(|v| v.is_finite()),
            "step {step}"
        );
        token = argmax(&logits);
    }
    assert_eq!(live.route_tries.load(Ordering::Relaxed) - tries, STEPS);
    assert_eq!(live.route_replays.load(Ordering::Relaxed) - replays, 0);
    assert_eq!(inner.decoder.route_readbacks - readbacks, 0);
    let layers = c.num_hidden_layers as u64;
    assert_eq!(inner.decoder.fused_shared_layers, STEPS * layers);
    let tapes = &inner.decoder.route_tapes;
    assert_eq!(tapes.len() as u64, STEPS);
    let mut sets = std::collections::HashSet::new();
    for tape in tapes {
        assert_eq!(
            tape.iter().map(|(layer, _)| *layer).collect::<Vec<_>>(),
            (0..c.num_hidden_layers).collect::<Vec<_>>()
        );
        sets.insert(
            tape.iter()
                .map(|(_, ids)| {
                    let mut ids = ids.clone();
                    ids.sort_unstable();
                    ids
                })
                .collect::<Vec<_>>(),
        );
    }
    assert!(sets.len() >= 2, "the routes never changed: {sets:?}");
    let pool = inner.decoder.paged.as_ref().unwrap().key_pool_array(0);
    assert_eq!(pool.unwrap().dtype().unwrap(), crate::array::DType::Int8);
}

/// The bits a decode pass leaves behind are the same in `a` and `b`:
/// history, positions, every layer's state arrays, the last chunk's hidden
/// rows, and per attention layer its index blocks, K/V pages and scales.
pub(super) fn assert_same_decoder_state(
    a: &mut decoder::Decoder,
    b: &mut decoder::Decoder,
    what: &str,
) {
    assert_eq!(a.history, b.history, "{what}: history");
    assert_eq!(a.positions, b.positions, "{what}: positions");
    for (layer, (x, y)) in a.caches.iter().zip(&b.caches).enumerate() {
        let (x, y): (Vec<_>, Vec<_>) = (x.arrays().collect(), y.arrays().collect());
        assert_eq!(x.len(), y.len(), "{what}: layer {layer} state count");
        for (k, (x, y)) in x.iter().zip(&y).enumerate() {
            assert_same_bits(x, y, &format!("{what}: layer {layer} state {k}"));
        }
    }
    assert_eq!(
        a.last_chunk_hidden.len(),
        b.last_chunk_hidden.len(),
        "{what}: last chunk"
    );
    for (x, y) in a.last_chunk_hidden.iter().zip(&b.last_chunk_hidden) {
        assert_same_bits(x, y, &format!("{what}: last chunk hidden"));
    }
    let c = a.config.clone();
    let attention = (0..c.num_hidden_layers).filter(|&layer| !c.linear(layer));
    for (pool, layer) in attention.enumerate() {
        let pool = pool as u32;
        match (a.index_keys(layer), b.index_keys(layer)) {
            (Some(x), Some(y)) => assert_same_bits(&x, &y, &format!("{what}: index {layer}")),
            (None, None) => {}
            _ => panic!("{what}: layer {layer} holds index blocks on one side only"),
        }
        let (pa, pb) = match (a.paged.as_mut(), b.paged.as_mut()) {
            (Some(pa), Some(pb)) => (pa, pb),
            (None, None) => continue,
            _ => panic!("{what}: pages on one side only"),
        };
        assert_eq!(pa.request_tokens(), pb.request_tokens(), "{what}: pages");
        let planes = [
            ("keys", pa.key_pool_array(pool), pb.key_pool_array(pool)),
            (
                "values",
                pa.value_pool_array(pool),
                pb.value_pool_array(pool),
            ),
            ("key scales", pa.k_scale_array(pool), pb.k_scale_array(pool)),
            (
                "value scales",
                pa.v_scale_array(pool),
                pb.v_scale_array(pool),
            ),
        ];
        for (name, x, y) in planes {
            let (x, y) = (x.unwrap(), y.unwrap());
            assert_eq!(x.dtype().unwrap(), y.dtype().unwrap(), "{what}: {name}");
            assert_same_bits(&x, &y, &format!("{what}: layer {layer} {name}"));
        }
    }
}

/// On the real MoE shape a device decode token (the fused routed+shared
/// kernel over the slots) leaves the same bits as a host-route token (routes
/// read back, the slot kernel, then the shared expert): every layer's
/// output, every state array, the int8 pages and scales, the index blocks,
/// history and logits, over 64 greedy steps that cross page blocks and take
/// a route miss whose replay rewrites the token's int8 row.
/// MUTATION: swap the first two scores of the fused call → red.
/// MUTATION: the host pin is ignored → red (the reference takes the device
/// route too).
#[test]
fn fixture_device_token_equals_host_token() {
    let _host = memory::scoped_fixture_host();
    let mut device = moe_shape_inner(12);
    let mut host = moe_shape_inner(12);
    host.decoder.route_mode_pin = Some(decoder::RouteMode::Host);
    let c = device.decoder.config.clone();
    let layers = c.num_hidden_layers;
    let all: Vec<u32> = (0..c.num_experts as u32).collect();
    for inner in [&mut device, &mut host] {
        inner.activate_paged_seq(1).unwrap();
        fill_slots(&mut inner.decoder, &all);
        inner.decoder.layer_outputs = Some(Vec::new());
    }
    let counts = |inner: &Inner| {
        let live = &inner.decoder.weights.live;
        [
            &live.route_tries,
            &live.route_replays,
            &live.replayed_layers,
        ]
        .map(|n| n.load(Ordering::Relaxed))
    };
    let (device_start, host_start) = (counts(&device), counts(&host));
    let host_readbacks = host.decoder.route_readbacks;
    const STEPS: usize = 64;
    const MISS_AT: usize = 8;
    let mut token = 3;
    for step in 0..STEPS {
        if step == MISS_AT {
            // Nine resident experts in layer 0 cannot serve a top-10 route.
            for inner in [&mut device, &mut host] {
                for expert in [9, 10, 11] {
                    inner.decoder.weights.evict_for_test(0, expert).unwrap();
                }
            }
        }
        let replays = counts(&device)[1];
        let got = device.decoder.step(token).unwrap();
        let want = host.decoder.step(token).unwrap();
        if step == MISS_AT {
            assert_eq!(counts(&device)[1], replays + 1, "the miss replays its row");
        }
        let what = format!("step {step} token {token}");
        let pass = |inner: &mut Inner| {
            let outputs = inner.decoder.layer_outputs.replace(Vec::new()).unwrap();
            outputs[outputs.len() - layers..].to_vec()
        };
        let (got_layers, want_layers) = (pass(&mut device), pass(&mut host));
        for (layer, ((i, x), (j, y))) in got_layers.iter().zip(&want_layers).enumerate() {
            assert_eq!([*i, *j], [layer, layer], "{what}: layer order");
            assert_same_bits(x, y, &format!("{what}: layer {layer} output"));
        }
        assert_same_bits(&got, &want, &format!("{what}: logits"));
        assert_same_decoder_state(&mut device.decoder, &mut host.decoder, &what);
        token = argmax(&got);
    }
    let [tries, replays, replayed] = [0, 1, 2].map(|k| counts(&device)[k] - device_start[k]);
    assert!(replays >= 1);
    assert!(tries > replays, "tries {tries}, replays {replays}");
    assert_eq!(
        device.decoder.fused_shared_layers,
        tries * layers as u64 + replayed,
        "every device pass, a replay's too, runs the fused kernel"
    );
    assert_eq!(counts(&host), host_start, "the host reference never tries");
    assert_eq!(host.decoder.fused_shared_layers, 0);
    assert_eq!(
        host.decoder.route_readbacks - host_readbacks,
        (STEPS * layers) as u64
    );
    let sets: std::collections::HashSet<_> = device
        .decoder
        .route_tapes
        .iter()
        .map(|tape| {
            tape.iter()
                .map(|(_, ids)| {
                    let mut ids = ids.clone();
                    ids.sort_unstable();
                    ids
                })
                .collect::<Vec<_>>()
        })
        .collect();
    assert!(sets.len() >= 2, "the routes never changed");
    let block = device.decoder.paged.as_ref().unwrap().block_size() as usize;
    assert!(STEPS > block, "the steps cross a page block of {block}");
}

/// A direct store (every expert in identity banks) keeps the fused
/// routed+shared kernel for its decode tokens, and keeps no layer inputs.
/// MUTATION: `tentative_shared_experts` returns `None` → red.
/// MUTATION: `keeps_layer_inputs` ignores `plan.direct` → red.
#[test]
fn fixture_direct_store_keeps_the_fused_kernel() {
    let _host = memory::scoped_fixture_host();
    let (mut decoder, vision) = moe_shape_decoder(moe_shape_fixture());
    go_live(&mut decoder, 0);
    assert!(
        decoder.weights.plan.direct,
        "the fixture host admits every bank"
    );
    let layers = decoder.config.num_hidden_layers;
    let probe = MxArray::from_uint32(&[0], &[1]).unwrap();
    for layer in 0..layers {
        // Waits for the layer's direct fill.
        decoder
            .weights
            .device_expert_slots(layer, &probe)
            .unwrap()
            .expect("a filled direct layer");
    }
    let mut inner = inner_over(decoder, vision, false);
    inner.activate_paged_seq(1).unwrap();
    let live = Arc::clone(&inner.decoder.weights.live);
    let tries = live.route_tries.load(Ordering::Relaxed);
    let readbacks = inner.decoder.route_readbacks;
    const STEPS: u64 = 8;
    let mut token = 3;
    for _ in 0..STEPS {
        token = argmax(&inner.decoder.step(token).unwrap());
    }
    assert_eq!(live.route_tries.load(Ordering::Relaxed) - tries, STEPS);
    assert_eq!(live.route_replays.load(Ordering::Relaxed), 0);
    assert_eq!(inner.decoder.route_readbacks - readbacks, 0);
    assert_eq!(inner.decoder.fused_shared_layers, STEPS * layers as u64);
    assert_eq!(
        inner.decoder.kept_layer_inputs, 0,
        "a direct store keeps no layer inputs"
    );
}

/// A device decode token counts one route try; only a miss counts a replay,
/// and the layers from the missed one on, which run again on device routes,
/// count as replayed layers and none as host tail layers; a hit counts
/// neither. The route guard weighs the hit at 0 and the miss at the layers
/// it ran again.
/// MUTATION: a miss counts every layer as replayed → red.
#[test]
fn records_replay_counts_tries_and_layers() {
    let _host = memory::scoped_fixture_host();
    let mut inner = moe_shape_inner(12);
    let c = inner.decoder.config.clone();
    inner.activate_paged_seq(1).unwrap();
    let all: Vec<u32> = (0..c.num_experts as u32).collect();
    fill_slots(&mut inner.decoder, &all);
    let live = Arc::clone(&inner.decoder.weights.live);
    let counts = |live: &Live| {
        [
            &live.route_tries,
            &live.route_replays,
            &live.replayed_layers,
            &live.host_tail_layers,
            &live.route_guard_trips,
        ]
        .map(|n| n.load(Ordering::Relaxed))
    };
    let start = counts(&live);
    let token = argmax(&inner.decoder.step(3).unwrap());
    let hit = counts(&live);
    assert_eq!(
        std::array::from_fn::<u64, 5, _>(|k| hit[k] - start[k]),
        [1, 0, 0, 0, 0],
        "a hit counts one try and nothing replayed"
    );
    const MISSED: usize = 2;
    // Nine resident experts in this layer cannot serve a top-10 route.
    for expert in [9, 10, 11] {
        let slot = inner
            .decoder
            .weights
            .evict_for_test(MISSED, expert)
            .unwrap();
        assert_ne!(slot, 0, "slot 0 keeps the device pass admitted");
    }
    inner.decoder.step(token).unwrap();
    let miss = counts(&live);
    let rerun = (c.num_hidden_layers - MISSED) as u64;
    assert_eq!(
        std::array::from_fn::<u64, 5, _>(|k| miss[k] - hit[k]),
        [1, 1, rerun, 0, 0],
        "a miss counts one try, one replay and the layers from the missed one on"
    );
    let mut guard = decoder::RouteGuard::default();
    assert!(!guard.record(0) && !guard.record(rerun));
    assert_eq!(inner.decoder.route_guard, guard);
}

/// `residencyInfo()` reads each record from its own counter.
/// MUTATION: the getter swaps `routeTries` and `replayedLayers` → red.
/// MUTATION: the getter reads `prefillFillMicros` from `fill_micros` → red.
#[test]
fn records_residency_info_reads_each_counter() {
    let (decoder, _vision) = tiny_decoder();
    let live = Arc::clone(&decoder.weights.live);
    live.set_fallback(Fallback::StorePlan {
        bank_bytes: 1 << 30,
        cap: 1 << 29,
    });
    let counters = [
        (&live.route_tries, 101),
        (&live.replayed_layers, 102),
        (&live.host_tail_layers, 103),
        (&live.route_guard_trips, 104),
        (&live.route_replays, 105),
        (&live.anonymous_need, 106),
        (&live.host_wired, 107),
        (&live.host_compressor, 108),
        (&live.host_anonymous, 109),
        (&live.host_purgeable, 110),
        (&live.self_mlx_bytes, 111),
        (&live.fill_calls, 112),
        (&live.fill_micros, 113),
        (&live.fill_read_bytes, 114),
        (&live.prefill_fill_calls, 115),
        (&live.prefill_fill_micros, 116),
        (&live.prefill_fill_read_bytes, 117),
        (&live.prefill_slot_misses, 118),
        (&live.prefill_bytes_read, 119),
        (&live.prefill_route_replays, 120),
        (&live.lane_steps, 121),
    ];
    for (counter, value) in counters {
        counter.store(value, Ordering::Release);
    }
    let (thread, _ready) = ModelThread::spawn_with_init(
        || Ok(((), ())),
        |_: &mut (), _: crate::engine::model_command::ModelCommand| {},
    );
    let model = Qwen4ExpModel {
        thread,
        config: (*decoder.config).clone(),
        assets_path: String::new(),
        has_mtp: false,
        has_draft: false,
        context_limits: Qwen4ExpContextLimits {
            effective_window_tokens: 0,
            paged_block_capacity: 0,
            paged_block_size: 0,
            trained_window_tokens: 0,
            rest_of_window_output_budget: false,
            omitted_budget_is_rest_of_window: false,
            kv_format: "bf16".into(),
        },
        residency: decoder.weights.plan.clone(),
        live: Arc::clone(&live),
        lookup: Arc::default(),
        supports_images: false,
        prefix: Arc::default(),
        prefix_closer: None,
        state_tier: Default::default(),
    };
    let info = model.residency_info();
    model.thread.shutdown_and_join().unwrap();
    let read = [
        ("routeTries", info.route_tries),
        ("replayedLayers", info.replayed_layers),
        ("hostTailLayers", info.host_tail_layers),
        ("routeGuardTrips", info.route_guard_trips),
        ("routeReplays", info.route_replays),
        ("anonymousNeedBytes", info.anonymous_need_bytes),
        ("hostWiredBytes", info.host_wired_bytes),
        ("hostCompressorBytes", info.host_compressor_bytes),
        ("hostAnonymousBytes", info.host_anonymous_bytes),
        ("hostPurgeableBytes", info.host_purgeable_bytes),
        ("selfMlxBytes", info.self_mlx_bytes),
        ("fillCalls", info.fill_calls),
        ("fillMicros", info.fill_micros),
        ("fillReadBytes", info.fill_read_bytes),
        ("prefillFillCalls", info.prefill_fill_calls),
        ("prefillFillMicros", info.prefill_fill_micros),
        ("prefillFillReadBytes", info.prefill_fill_read_bytes),
        ("prefillSlotMisses", info.prefill_slot_misses),
        (
            "prefillCheckpointBytesRead",
            info.prefill_checkpoint_bytes_read,
        ),
        ("prefillRouteReplays", info.prefill_route_replays),
        ("laneSteps", info.lane_steps),
    ];
    for ((name, got), (_, want)) in read.iter().zip(counters) {
        assert_eq!(*got, want as f64, "{name}");
    }
    assert_eq!(info.residency, "store-plan");
    assert_eq!(info.expert_count, decoder.config.num_experts as u32);
}

/// The host split `residencyInfo()` reports beside `liveNonReclaimableBytes`.
fn recorded_host(live: &Live) -> memory::HostMemory {
    memory::HostMemory {
        wired: live.host_wired.load(Ordering::Acquire),
        compressor: live.host_compressor.load(Ordering::Acquire),
        anonymous: live.host_anonymous.load(Ordering::Acquire),
        purgeable: live.host_purgeable.load(Ordering::Acquire),
        ..Default::default()
    }
}

/// A streamed store plan names its reason and records the host it was
/// chosen beside, with the bank budget the host gate would admit.
/// MUTATION: drop the `set_fallback` of the plan branch in
/// `select_direct_residency_with` → red (`residency` reads `rows`).
#[test]
fn records_plan_branch_names_reason_and_host() {
    let temp = Temp::new();
    let path = temp.0.join("plan-branch.gguf");
    let config = write_direct_gguf_fixture(&path);
    let host = memory::HostMemory {
        physical: 16 << 30,
        free: 6 << 30,
        file: 2 << 30,
        anonymous: 5 << 30,
        purgeable: 1 << 30,
        wired: 2 << 30,
        compressor: 1 << 30,
    };
    let _host = memory::scoped_test_host(host);
    let mut store = weights::Store::open_fixture(&path, Some(&temp.0.join("cache"))).unwrap();
    store.plan.physical_bytes = Some(host.physical);
    store.set_store_plan(
        crate::engine::memory_plan::StorePlan {
            whole: false,
            expert_cap_bytes: 1 << 40,
            row_cache_bytes: 2 << 30,
        },
        1 << 30,
    );
    let need = 3u64 << 30;
    assert!(
        store
            .select_direct_residency_with(
                config.num_experts,
                need,
                &mut memory::direct_gate,
                &mut |_| panic!("a streamed store allocates no full banks"),
            )
            .unwrap()
    );
    let live = &store.live;
    assert_eq!(live.residency(), Residency::StorePlan);
    assert_eq!(live.residency().as_str(), "store-plan");
    let Some(Fallback::StorePlan { bank_bytes, cap }) = live.fallback() else {
        panic!("plan branch reason: {:?}", live.fallback());
    };
    assert!(bank_bytes > 0);
    assert!(cap > 0);
    assert!(
        live.fallback()
            .unwrap()
            .to_string()
            .starts_with("store plan: experts stream through a ")
    );
    assert_eq!(store.plan.host, Some(host));
    assert_eq!(
        live.live_non_reclaimable.load(Ordering::Acquire),
        host.non_reclaimable()
    );
    assert_eq!(
        live.bank_budget.load(Ordering::Acquire),
        memory::bank_gate(&host, need, memory::DIRECT_MARGIN).bank_budget
    );
    assert_eq!(
        recorded_host(live).non_reclaimable(),
        host.non_reclaimable()
    );
    assert_eq!(live.host_anonymous.load(Ordering::Acquire), host.anonymous);
    assert_eq!(live.anonymous_need.load(Ordering::Acquire), need);
    assert_eq!(
        live.self_mlx_bytes.load(Ordering::Acquire),
        store.plan.self_mlx_bytes
    );
}

/// A host-gate refusal records the very host read the gate decided on: its
/// split adds up to `liveNonReclaimableBytes` even when the host moves
/// between reads.
/// MUTATION: keep a host read taken before `direct_gate` samples its own
/// → red (the split no longer adds up).
#[test]
fn records_gate_branch_keeps_the_host_split() {
    let temp = Temp::new();
    let path = temp.0.join("gate-branch.gguf");
    let config = write_direct_gguf_fixture(&path);
    let host = memory::HostMemory {
        physical: 16 << 30,
        free: 1 << 30,
        file: 1 << 30,
        anonymous: 9 << 30,
        purgeable: 1 << 30,
        wired: 3 << 30,
        compressor: 2 << 30,
    };
    let mut store = weights::Store::open_fixture(&path, Some(&temp.0.join("cache"))).unwrap();
    store.plan_fixture_store().unwrap();
    store.plan.physical_bytes = Some(host.physical);
    let _host = memory::scoped_drifting_test_host(host, 1 << 30);
    let need = 1u64 << 30;
    assert!(
        store
            .select_direct_residency_with(
                config.num_experts,
                need,
                &mut memory::direct_gate,
                &mut |_| panic!("a refused gate must not allocate"),
            )
            .unwrap()
    );
    let live = &store.live;
    assert_eq!(live.residency(), Residency::HostGate);
    assert!(matches!(live.fallback(), Some(Fallback::LoadGate { .. })));
    let recorded = recorded_host(live);
    assert!(recorded.anonymous >= host.anonymous);
    assert_eq!(
        recorded.non_reclaimable(),
        live.live_non_reclaimable.load(Ordering::Acquire)
    );
    assert_eq!(live.anonymous_need.load(Ordering::Acquire), need);
}

/// A tiny-fixture decoder whose decode tokens take the paged or the flat
/// cache path.
fn tiny(paged: bool) -> Inner {
    let mut inner = tiny_inner();
    if paged {
        inner.activate_paged_seq(1).unwrap();
    } else {
        inner.decoder.paged = None;
    }
    inner
}

/// Runs `pass` inside a decode pass, as `step` runs a token.
fn in_decode_pass<R>(
    decoder: &mut decoder::Decoder,
    pass: impl FnOnce(&mut decoder::Decoder) -> R,
) -> R {
    let prior = std::mem::replace(&mut decoder.decode_pass, true);
    let out = pass(decoder);
    decoder.decode_pass = prior;
    out
}

/// The token's output of a pass that ran to the end.
fn done(pass: decoder::LayerPass) -> MxArray {
    match pass {
        decoder::LayerPass::Done(out) => out,
        decoder::LayerPass::Stopped { at } => panic!("the pass stopped before layer {at}"),
    }
}

/// A kept checkpoint holds, for every layer, the very stream a decoder
/// that keeps none feeds that layer (the embedding for layer 0, the
/// previous layer's output after it), and keeping it changes no bit.
/// MUTATION: keep each layer's output in place of its input → red.
#[test]
fn layers_checkpoint_holds_each_layer_input() {
    for paged in [false, true] {
        let mut kept = tiny(paged);
        let mut reference = tiny(paged);
        kept.decoder.layer_inputs_pin = Some(true);
        reference.decoder.layer_outputs = Some(Vec::new());
        let layers = kept.decoder.config.num_hidden_layers;
        let hc = kept.decoder.config.hc_count as i32;
        const STEPS: usize = 20;
        for step in 0..STEPS {
            let token = 3 + (step % 10) as u32;
            let got = kept.decoder.step(token).unwrap();
            let want = reference.decoder.step(token).unwrap();
            let what = format!("paged={paged} step {step}");
            assert_same_bits(&got, &want, &format!("{what}: logits"));
            assert_same_decoder_state(&mut kept.decoder, &mut reference.decoder, &what);
            let inputs = kept
                .decoder
                .layer_inputs
                .clone()
                .expect("kept layer inputs");
            assert_eq!(inputs.len(), layers, "{what}: kept inputs");
            let outputs = reference.decoder.layer_outputs.replace(Vec::new()).unwrap();
            assert_eq!(outputs.len(), layers, "{what}: layer outputs");
            let embedding =
                MxArray::tile(&reference.decoder.embed_token(token).unwrap(), &[1, 1, hc]).unwrap();
            for (layer, input) in inputs.iter().enumerate() {
                let fed = match layer.checked_sub(1) {
                    None => &embedding,
                    Some(before) => {
                        assert_eq!(outputs[before].0, before);
                        &outputs[before].1
                    }
                };
                assert_same_bits(input, fed, &format!("{what}: layer {layer} input"));
            }
        }
        assert!(reference.decoder.layer_inputs.is_none());
        assert_eq!(kept.decoder.kept_layer_inputs, (STEPS * layers) as u64);
    }
}

/// `decode_layers` from layer m on the kept input of layer m finishes the
/// token as one whole pass does: after a pass stopped before m, and after a
/// whole pass whose layers m.. got their caches back and `history` its
/// length (the pages keep the cursor and take the row again). Logits, every
/// state array, the pages and 3 more tokens are bit-equal, for m = 1 (PLE),
/// 2 (GDN) and 3 (attention), paged and flat, at a block end and after it.
/// MUTATION: continue from the input of layer m - 1 → red.
/// MUTATION: the stop takes layer m's cache out before it returns → red.
#[test]
fn layers_split_run_equals_one_run() {
    for paged in [false, true] {
        for warm in [15, 16] {
            for from in 1..=3 {
                for rerun in [false, true] {
                    let mut split = tiny(paged);
                    let mut whole = tiny(paged);
                    let layers = split.decoder.config.num_hidden_layers;
                    assert!(from < layers);
                    for i in 0..warm {
                        let token = 3 + (i % 10) as u32;
                        split.decoder.step(token).unwrap();
                        whole.decoder.step(token).unwrap();
                    }
                    let what = format!("paged={paged} warm={warm} from={from} rerun={rerun}");
                    let token = 7;
                    let want = whole.decoder.step(token).unwrap();
                    let history = split.decoder.history.clone();
                    let caches = split.decoder.caches.clone();
                    let got = in_decode_pass(&mut split.decoder, |d| {
                        let x = d.begin_decode_token(token, &[], None, paged).unwrap();
                        d.layer_inputs = Some(Vec::new());
                        if rerun {
                            done(d.decode_layers(0, x, token, true).unwrap());
                            d.caches[from..].clone_from_slice(&caches[from..]);
                            d.history.clone_from(&history);
                        } else {
                            d.stop_before = Some(from);
                            let pass = d.decode_layers(0, x, token, true).unwrap();
                            let decoder::LayerPass::Stopped { at } = pass else {
                                panic!("{what}: the pass did not stop");
                            };
                            assert_eq!(at, from, "{what}");
                            d.stop_before = None;
                            assert_eq!(d.history, history, "{what}: history after the stop");
                            for layer in from..layers {
                                let (x, y): (Vec<_>, Vec<_>) = (
                                    d.caches[layer].arrays().collect(),
                                    caches[layer].arrays().collect(),
                                );
                                assert_eq!(
                                    x.len(),
                                    y.len(),
                                    "{what}: layer {layer} after the stop"
                                );
                                for (x, y) in x.iter().zip(&y) {
                                    assert_same_bits(
                                        x,
                                        y,
                                        &format!("{what}: layer {layer} after the stop"),
                                    );
                                }
                            }
                        }
                        let inputs = d.layer_inputs.as_ref().unwrap();
                        assert_eq!(
                            inputs.len(),
                            if rerun { layers } else { from + 1 },
                            "{what}"
                        );
                        let x = inputs[from].clone();
                        let out = done(d.decode_layers(from, x, token, true).unwrap());
                        assert_eq!(d.layer_inputs.as_ref().unwrap().len(), layers, "{what}");
                        out
                    });
                    assert_same_bits(&got, &want, &format!("{what}: logits"));
                    assert_same_decoder_state(&mut split.decoder, &mut whole.decoder, &what);
                    for token in [9, 4, 12] {
                        let got = split.decoder.step(token).unwrap();
                        let want = whole.decoder.step(token).unwrap();
                        let what = format!("{what} then {token}");
                        assert_same_bits(&got, &want, &format!("{what}: logits"));
                        assert_same_decoder_state(&mut split.decoder, &mut whole.decoder, &what);
                    }
                }
            }
        }
    }
}

/// On the real MoE shape a partial store's tentative pass keeps every
/// layer's input and that changes no bit against a pass that keeps none:
/// layer outputs, logits and every state array over 64 greedy hit steps (a
/// miss replays from the kept inputs, so a pass that keeps none cannot
/// miss; the `tail_` and `device_replay_` tests cover misses).
/// MUTATION: `keeps_layer_inputs` answers false on a partial store → red
/// (nothing kept).
#[test]
fn layers_checkpoint_on_the_moe_fixture_changes_no_bit() {
    let _host = memory::scoped_fixture_host();
    let mut kept = moe_shape_inner(12);
    let mut plain = moe_shape_inner(12);
    plain.decoder.layer_inputs_pin = Some(false);
    let c = kept.decoder.config.clone();
    let layers = c.num_hidden_layers;
    let all: Vec<u32> = (0..c.num_experts as u32).collect();
    for inner in [&mut kept, &mut plain] {
        inner.activate_paged_seq(1).unwrap();
        fill_slots(&mut inner.decoder, &all);
        inner.decoder.layer_outputs = Some(Vec::new());
    }
    let counts = |inner: &Inner| {
        let live = &inner.decoder.weights.live;
        [&live.route_tries, &live.route_replays].map(|n| n.load(Ordering::Relaxed))
    };
    let (kept_start, plain_start) = (counts(&kept), counts(&plain));
    const STEPS: usize = 64;
    let mut token = 3;
    for step in 0..STEPS {
        let got = kept.decoder.step(token).unwrap();
        let want = plain.decoder.step(token).unwrap();
        let what = format!("step {step} token {token}");
        let pass = |inner: &mut Inner| {
            let outputs = inner.decoder.layer_outputs.replace(Vec::new()).unwrap();
            outputs[outputs.len() - layers..].to_vec()
        };
        for ((i, x), (j, y)) in pass(&mut kept).iter().zip(&pass(&mut plain)) {
            assert_eq!(i, j, "{what}: layer order");
            assert_same_bits(x, y, &format!("{what}: layer {i} output"));
        }
        assert_same_bits(&got, &want, &format!("{what}: logits"));
        assert_same_decoder_state(&mut kept.decoder, &mut plain.decoder, &what);
        assert!(
            kept.decoder.layer_inputs.is_none(),
            "{what}: the pass drops its inputs"
        );
        token = argmax(&got);
    }
    let [tries, replays] = [0, 1].map(|k| counts(&kept)[k] - kept_start[k]);
    assert_eq!([tries, replays], [STEPS as u64, 0]);
    assert_eq!(
        [0, 1].map(|k| counts(&plain)[k] - plain_start[k]),
        [tries, replays]
    );
    assert_eq!(kept.decoder.kept_layer_inputs, tries * layers as u64);
    assert_eq!(plain.decoder.kept_layer_inputs, 0);
}

/// Each layer's output from the last pass that ran it, since the last call:
/// a replay from layer m runs layers m.. again after the tentative pass.
fn final_layer_outputs(decoder: &mut decoder::Decoder) -> Vec<MxArray> {
    let mut last: Vec<Option<MxArray>> = vec![None; decoder.config.num_hidden_layers];
    for (layer, output) in decoder.layer_outputs.replace(Vec::new()).unwrap() {
        last[layer] = Some(output);
    }
    last.into_iter()
        .enumerate()
        .map(|(layer, output)| output.unwrap_or_else(|| panic!("layer {layer} ran in no pass")))
        .collect()
}

/// A tentative pass of `token` as a partial store runs it, every layer
/// input kept, with the MoE output of layer `from` zeroed (a stand-in for
/// a skipped expert). Returns the state before the token, the pass's
/// logits and its kept inputs.
fn poisoned_pass(
    decoder: &mut decoder::Decoder,
    token: u32,
    from: usize,
) -> (decoder::DecoderState, MxArray, Option<Vec<MxArray>>) {
    let base = decoder.snapshot();
    decoder.layer_inputs_pin = Some(true);
    decoder.poison_moe_at = Some(from);
    let logits = decoder.step(token).unwrap();
    decoder.layer_inputs_pin = None;
    assert_eq!(decoder.poison_moe_at, None, "the pass ran layer {from}");
    (base, logits, decoder.layer_inputs.take())
}

/// `replay_from_layer` inside a decode pass, as `step` runs it.
fn replay(
    decoder: &mut decoder::Decoder,
    base: decoder::DecoderState,
    from: usize,
    inputs: Option<Vec<MxArray>>,
    token: u32,
) -> napi::Result<MxArray> {
    in_decode_pass(decoder, |d| {
        d.replay_from_layer(&base, from, inputs, token, true)
    })
}

/// The tiny fixture's first `count` warm-up tokens.
fn warm_prefix(count: usize) -> Vec<u32> {
    (0..count).map(|i| 3 + (i % 10) as u32).collect()
}

/// A miss at layer m replays the token from m on host routes: layers before
/// m keep the tentative pass's state and layers m.. run again on the kept
/// input of m. Logits, history, the page cursor, every state array, the
/// pages and 3 more tokens are bit-equal to a host token, for m = 0, 1
/// (PLE), 2 (GDN) and 3 (attention), after 15, 16, 31 and 32 tokens, on
/// flat caches, bf16 pages and int8 pages (head_dim 128).
/// MUTATION: the replay gives caches back from m + 1 → red.
/// MUTATION: the replay skips the attention layer's page write → red (the
/// tentative row stays).
#[test]
fn tail_replay_is_exact_at_every_layer_kind() {
    use crate::transformer::KvFormat;
    let wide = Temp::new();
    write_wide_head_fixture(&wide.0);
    let int8 = || {
        let mut inner = wide_head_inner(&wide.0, KvFormat::Int8);
        inner.activate_paged_seq(1).unwrap();
        inner
    };
    let cases: [(&str, &dyn Fn() -> Inner); 3] = [
        ("flat", &|| tiny(false)),
        ("bf16 pages", &|| tiny(true)),
        ("int8 pages", &int8),
    ];
    for (name, make) in cases {
        for boundary in [15, 16, 31, 32] {
            for from in 0..4 {
                let mut replayed = make();
                let mut reference = make();
                let c = replayed.decoder.config.clone();
                assert_eq!(
                    (0..c.num_hidden_layers)
                        .map(|i| (c.linear(i), c.ple_layer_ids.contains(&(i + 1))))
                        .collect::<Vec<_>>(),
                    [(true, false), (true, true), (true, false), (false, false)],
                    "layer kinds"
                );
                if name == "int8 pages" {
                    let pool = replayed.decoder.paged.as_ref().unwrap().key_pool_array(0);
                    assert_eq!(pool.unwrap().dtype().unwrap(), crate::array::DType::Int8);
                }
                let prefix = warm_prefix(boundary);
                for inner in [&mut replayed, &mut reference] {
                    inner.decoder.prefill_chunk(&prefix, None, false).unwrap();
                }
                let what = format!("{name} after {boundary} from {from}");
                let token = 7;
                let want = reference.decoder.step(token).unwrap();
                let (base, tentative, inputs) = poisoned_pass(&mut replayed.decoder, token, from);
                assert_ne!(
                    f32_bits(&tentative),
                    f32_bits(&want),
                    "{what}: the poison changed the tentative pass"
                );
                let got = replay(&mut replayed.decoder, base, from, inputs, token).unwrap();
                assert_same_bits(&got, &want, &format!("{what}: logits"));
                assert_same_decoder_state(&mut replayed.decoder, &mut reference.decoder, &what);
                for token in [9, 4, 12] {
                    let got = replayed.decoder.step(token).unwrap();
                    let want = reference.decoder.step(token).unwrap();
                    let what = format!("{what} then {token}");
                    assert_same_bits(&got, &want, &format!("{what}: logits"));
                    assert_same_decoder_state(&mut replayed.decoder, &mut reference.decoder, &what);
                }
            }
        }
    }
}

/// The replay keeps the token on the page cursor: the tentative pass
/// recorded it, the replay writes its rows again at the same position, and
/// the cursor, history, logits and the next token match a host token.
/// MUTATION: the replay rolls the cursor back one token and does not record
/// it again → red (the row is outside the write range).
#[test]
fn tail_keeps_the_cursor() {
    let mut replayed = tiny(true);
    let mut reference = tiny(true);
    let prefix = warm_prefix(16);
    for inner in [&mut replayed, &mut reference] {
        inner.decoder.prefill_chunk(&prefix, None, false).unwrap();
    }
    let cursor = |inner: &Inner| {
        inner
            .decoder
            .paged
            .as_ref()
            .unwrap()
            .request_tokens()
            .to_vec()
    };
    let (token, from) = (7, 3);
    let want = reference.decoder.step(token).unwrap();
    let (base, _, inputs) = poisoned_pass(&mut replayed.decoder, token, from);
    assert_eq!(
        cursor(&replayed),
        cursor(&reference),
        "the pass recorded the token"
    );
    let got = replay(&mut replayed.decoder, base, from, inputs, token)
        .unwrap_or_else(|error| panic!("the replay failed: {}", error.reason));
    assert_eq!(
        cursor(&replayed),
        cursor(&reference),
        "the replay kept the cursor"
    );
    assert_eq!(replayed.decoder.history, reference.decoder.history);
    assert_same_bits(&got, &want, "logits");
    assert_same_bits(
        &replayed.decoder.step(9).unwrap(),
        &reference.decoder.step(9).unwrap(),
        "next logits",
    );
}

/// After a replay the last hidden rows (read by the MTP head and the stored
/// states) are the replay's, not the tentative pass's.
/// MUTATION: the replay keeps the tentative pass's `last_chunk_hidden` → red.
#[test]
fn tail_last_hidden_is_the_replays() {
    let mut replayed = tiny(true);
    let mut reference = tiny(true);
    let prefix = warm_prefix(16);
    for inner in [&mut replayed, &mut reference] {
        inner.decoder.prefill_chunk(&prefix, None, false).unwrap();
    }
    let (token, from) = (7, 1);
    reference.decoder.step(token).unwrap();
    let (base, _, inputs) = poisoned_pass(&mut replayed.decoder, token, from);
    let tentative = replayed.decoder.last_chunk_hidden.clone();
    replay(&mut replayed.decoder, base, from, inputs, token).unwrap();
    let (got, want) = (
        &replayed.decoder.last_chunk_hidden,
        &reference.decoder.last_chunk_hidden,
    );
    assert_eq!([tentative.len(), got.len(), want.len()], [1, 1, 1]);
    assert_ne!(
        f32_bits(&tentative[0]),
        f32_bits(&want[0]),
        "the poison changed the tentative hidden"
    );
    assert_same_bits(&got[0], &want[0], "last hidden");
}

/// What a [`moe_replay_run`] counted on its device side.
struct MoeRun {
    /// [tries, replays, replayed layers, host tail layers] over the run.
    total: [u64; 4],
    /// The same counts for each planned step, in plan order.
    planned: Vec<[u64; 4]>,
    /// [slot hits, slot misses] of each planned step, device then host.
    planned_slots: Vec<[[u64; 2]; 2]>,
    /// The device decoder's `preload_checks` and `device_rerun_peak`.
    checks: Vec<(usize, bool)>,
    peak: usize,
    /// Whether each step tried device routes, and the guard's trips.
    tried: Vec<bool>,
    trips: u64,
    /// Device passes a route read stopped early, and the route reads that
    /// found an entry in flight.
    stops: u64,
    unready_polls: u64,
}

/// The knobs of a [`moe_replay_run`].
#[derive(Default)]
struct RunKnobs<'a> {
    /// Hints name experts the layer did not route.
    wrong_hints: bool,
    /// Each layer's slot book and the slot counts equal the host's.
    book: bool,
    /// Costs the device side's route guard records first.
    prime: &'a [u64],
    /// The device side's `after_group_submit` pin.
    poll: Option<decoder::GroupPoll>,
    /// Both sides run flat caches in place of int8 pages.
    flat: bool,
    /// Both sides run the 12-layer [`deep_moe_shape_fixture`].
    deep: bool,
}

fn book() -> RunKnobs<'static> {
    RunKnobs {
        book: true,
        ..RunKnobs::default()
    }
}

/// A greedy run of `steps` tokens of the MoE-shape fixture against a
/// host-route reference. Both hold `capacity` slots per layer filled with
/// experts 0..; at each planned step both lose experts 9, 10 and 11 of each
/// planned layer, so the device pass misses there, and get them back after
/// the step. Every step's layer outputs, logits and decoder state are
/// bit-equal, and the device side reads back routes only on tokens it did
/// not try and on host tail layers. With `book`, each layer's slot
/// occupants, ages and tick and the slot hit and miss counts are equal too.
/// The device side's route guard first records the costs in `prime`; after
/// every step it equals a guard that recorded each tried token's device
/// layers run again plus 3 per host layer, and a step tries device routes
/// unless that guard sends it to host routes. A step counts one missed
/// token when it ran any replay pass, however many.
/// MUTATION: `route_miss_tokens` counts each replay pass → red in every run
/// with a token that replays twice (`early_cut_is_exact` capacity 10).
fn moe_replay_run(
    capacity: usize,
    steps: usize,
    planned: &[(usize, &[usize])],
    knobs: &RunKnobs,
) -> MoeRun {
    let RunKnobs {
        wrong_hints,
        book,
        prime,
        poll,
        flat,
        deep,
    } = *knobs;
    let fixture = if deep {
        deep_moe_shape_fixture()
    } else {
        moe_shape_fixture()
    };
    let mut device = moe_shape_inner_at(fixture, capacity);
    let mut host = moe_shape_inner_at(fixture, capacity);
    host.decoder.route_mode_pin = Some(decoder::RouteMode::Host);
    device.decoder.wrong_hints = wrong_hints;
    device.decoder.after_group_submit = poll;
    let layers = device.decoder.config.num_hidden_layers;
    let filled: Vec<u32> = (0..capacity as u32).collect();
    for inner in [&mut device, &mut host] {
        if flat {
            inner.decoder.paged = None;
        } else {
            inner.activate_paged_seq(1).unwrap();
        }
        fill_slots(&mut inner.decoder, &filled);
        inner.decoder.layer_outputs = Some(Vec::new());
    }
    let counts = |inner: &Inner| {
        let live = &inner.decoder.weights.live;
        [
            &live.route_tries,
            &live.route_replays,
            &live.replayed_layers,
            &live.host_tail_layers,
        ]
        .map(|n| n.load(Ordering::Relaxed))
    };
    let miss_tokens = |inner: &Inner| {
        let live = &inner.decoder.weights.live;
        live.route_miss_tokens.load(Ordering::Relaxed)
    };
    let slot_counts = |inner: &Inner| {
        let live = &inner.decoder.weights.live;
        [&live.slot_hits, &live.slot_misses].map(|n| n.load(Ordering::Relaxed))
    };
    let (start, host_start) = (counts(&device), counts(&host));
    let readbacks = device.decoder.route_readbacks;
    let trips = |inner: &Inner| {
        inner
            .decoder
            .weights
            .live
            .route_guard_trips
            .load(Ordering::Relaxed)
    };
    let trips_start = trips(&device);
    for &cost in prime {
        assert!(!device.decoder.route_guard.record(cost), "prime {prime:?}");
    }
    let mut guard = device.decoder.route_guard.clone();
    let mut want_trips = 0;
    let mut run = MoeRun {
        total: [0; 4],
        planned: Vec::new(),
        planned_slots: Vec::new(),
        checks: Vec::new(),
        peak: 0,
        tried: Vec::new(),
        trips: 0,
        stops: 0,
        unready_polls: 0,
    };
    let mut token = 3;
    for step in 0..steps {
        let what = format!("capacity {capacity} step {step} token {token}");
        let missed = planned
            .iter()
            .find(|(at, _)| *at == step)
            .map(|&(_, layers)| layers);
        for &layer in missed.unwrap_or_default() {
            for inner in [&mut device, &mut host] {
                for expert in [9, 10, 11] {
                    let slot = inner.decoder.weights.evict_for_test(layer, expert).unwrap();
                    assert_ne!(slot, 0, "{what}: slot 0 keeps the device pass admitted");
                }
            }
        }
        let host_token = guard.host_token();
        let before = counts(&device);
        let missed_before = miss_tokens(&device);
        let slots_before = [slot_counts(&device), slot_counts(&host)];
        let got = device.decoder.step(token).unwrap();
        let want = host.decoder.step(token).unwrap();
        let [tries, replays, replayed, host_tail] = {
            let after = counts(&device);
            std::array::from_fn::<u64, 4, _>(|k| after[k] - before[k])
        };
        assert_eq!(tries, u64::from(!host_token), "{what}: device try");
        assert_eq!(
            miss_tokens(&device) - missed_before,
            u64::from(replays > 0),
            "{what}: a token that ran {replays} replay passes missed once"
        );
        if tries == 1 && guard.record(replayed + 3 * host_tail) {
            want_trips += 1;
        }
        assert_eq!(device.decoder.route_guard, guard, "{what}: route guard");
        assert_eq!(
            trips(&device) - trips_start,
            want_trips,
            "{what}: guard trips"
        );
        run.tried.push(tries == 1);
        if missed.is_some() {
            let after = counts(&device);
            run.planned
                .push(std::array::from_fn(|k| after[k] - before[k]));
            let slots_after = [slot_counts(&device), slot_counts(&host)];
            run.planned_slots.push(std::array::from_fn(|side| {
                std::array::from_fn(|k| slots_after[side][k] - slots_before[side][k])
            }));
            for inner in [&mut device, &mut host] {
                fill_slots(&mut inner.decoder, &filled);
            }
        }
        let (got_layers, want_layers) = (
            final_layer_outputs(&mut device.decoder),
            final_layer_outputs(&mut host.decoder),
        );
        for (layer, (x, y)) in got_layers.iter().zip(&want_layers).enumerate() {
            assert_same_bits(x, y, &format!("{what}: layer {layer} output"));
        }
        assert_same_bits(&got, &want, &format!("{what}: logits"));
        assert_same_decoder_state(&mut device.decoder, &mut host.decoder, &what);
        if book {
            for layer in 0..layers {
                let (a, b) = (&device.decoder.weights, &host.decoder.weights);
                assert_eq!(
                    a.slot_occupants(layer),
                    b.slot_occupants(layer),
                    "{what}: layer {layer} occupants"
                );
                assert_eq!(
                    a.slot_ages(layer),
                    b.slot_ages(layer),
                    "{what}: layer {layer} ages"
                );
            }
            assert_eq!(
                slot_counts(&device),
                slot_counts(&host),
                "{what}: slot hits and misses"
            );
        }
        token = argmax(&got);
    }
    assert_eq!(counts(&host), host_start, "the host reference never tries");
    let end = counts(&device);
    run.total = std::array::from_fn(|k| end[k] - start[k]);
    let [tries, _, _, host_tail] = run.total;
    assert_eq!(
        device.decoder.route_readbacks - readbacks,
        (steps as u64 - tries) * layers as u64 + host_tail,
        "capacity {capacity}: a device pass reads no route back"
    );
    assert!(
        device.decoder.fused_shared_layers > 0,
        "capacity {capacity}"
    );
    run.checks = std::mem::take(&mut device.decoder.preload_checks);
    run.peak = device.decoder.device_rerun_peak;
    run.trips = want_trips;
    run.stops = device.decoder.early_stops;
    run.unready_polls = device.decoder.unready_polls.get();
    run
}

/// On the real MoE shape with int8 pages, a miss reads the missed experts
/// and runs its token again from the missed layer on device routes, and the
/// run stays bit-equal to host routes: misses placed at layers 1, 2, 3 and
/// 0 at capacity 12 (the slots also end each step as the host's do), and
/// the misses capacity 10 and 11 run into on every token. A device replay
/// reads no route back, and those misses never trip the route guard.
/// MUTATION: the replay starts from the input of layer m + 1 → red.
/// MUTATION: the preload keeps the layer's old device map → red (the kernel
/// skips the loaded experts while the host check finds them).
#[test]
fn device_replay_is_exact() {
    let _host = memory::scoped_fixture_host();
    const LAYERS: usize = 4;
    let planned: [(usize, &[usize]); 4] = [(4, &[1]), (22, &[2]), (40, &[3]), (58, &[0])];
    let run = moe_replay_run(12, 64, &planned, &book());
    for (&(step, layers), delta) in planned.iter().zip(&run.planned) {
        let rerun = (LAYERS - layers[0]) as u64;
        assert_eq!(
            *delta,
            [1, 1, rerun, 0],
            "step {step}: one device replay from layer {}",
            layers[0]
        );
    }
    let [tries, replays, replayed, host_tail] = run.total;
    assert!(tries > replays, "tries {tries}, replays {replays}");
    assert_eq!(replays, planned.len() as u64);
    let rerun: u64 = planned.iter().map(|&(_, m)| (LAYERS - m[0]) as u64).sum();
    assert_eq!([replayed, host_tail], [rerun, 0]);
    for capacity in [10, 11] {
        let run = moe_replay_run(capacity, 64, &[], &RunKnobs::default());
        eprintln!(
            "capacity {capacity}: [tries, replays, replayed, host tail] {:?}, peak {}",
            run.total, run.peak
        );
        let [tries, replays, replayed, _] = run.total;
        assert!(
            tries > 0 && replays > 0 && replayed > 0,
            "capacity {capacity}: tries {tries}, replays {replays}, replayed {replayed}"
        );
        assert!(run.peak <= LAYERS, "capacity {capacity}: peak {}", run.peak);
        assert_eq!(run.trips, 0, "capacity {capacity}");
    }
}

/// Hints that name experts the layer did not route leave its routes
/// missing: the device replay misses again. Misses at layers 1 and 3: the
/// second device pass starts at layer 3 from the stream the first replay
/// fed it. Misses at layers 0 and 2: the first replay re-ran a whole pass,
/// so layers 2.. end on host routes. Bit-equal to host routes throughout.
/// MUTATION: no per-token budget → red (a third device pass, more than one
/// whole pass of device layers in one token).
/// MUTATION: the second replay starts from the tentative pass's input of
/// its layer → red.
#[test]
fn device_replay_wrong_hints_end_on_host_routes() {
    let _host = memory::scoped_fixture_host();
    const LAYERS: usize = 4;
    let planned: [(usize, &[usize]); 2] = [(4, &[1, 3]), (22, &[0, 2])];
    let run = moe_replay_run(
        12,
        40,
        &planned,
        &RunKnobs {
            wrong_hints: true,
            ..RunKnobs::default()
        },
    );
    assert_eq!(
        run.planned,
        [[1, 2, 3 + 1, 0], [1, 2, LAYERS as u64, 2]],
        "two device passes; then one device pass and a host tail"
    );
    assert!(run.total[3] > 0, "a token ended on host routes");
    assert_eq!(run.peak, LAYERS, "one token re-ran a whole pass");
}

/// The fills before a device replay keep every id the replay will use:
/// after them each id read for the missed layer, and for each hinted layer,
/// holds a slot.
/// MUTATION: a preload protects only the ids it reads → red (a resident
/// routed id of the same layer is evicted).
#[test]
fn device_replay_fills_protect_the_layers_ids() {
    let _host = memory::scoped_fixture_host();
    for capacity in [10, 11] {
        let run = moe_replay_run(capacity, 64, &[], &RunKnobs::default());
        assert!(!run.checks.is_empty(), "capacity {capacity}: no preload");
        let unmapped: Vec<_> = run.checks.iter().filter(|(_, mapped)| !mapped).collect();
        assert!(
            unmapped.is_empty(),
            "capacity {capacity}: preloaded ids without a slot in layers {unmapped:?}"
        );
    }
}

/// A routed expert a preload read counts once, as a miss: the replay's
/// commit leaves it out of its hits. Hits and misses of the token equal
/// the host route's, which reads it inline.
/// MUTATION: the preload counts the ids it found resident as hits → red.
#[test]
fn device_replay_counts_a_loaded_pick_once() {
    let _host = memory::scoped_fixture_host();
    let run = moe_replay_run(12, 8, &[(4, &[2])], &book());
    assert_eq!(run.planned, [[1, 1, 2, 0]]);
    let [[device_hits, device_misses], host] = run.planned_slots[0];
    assert!(device_misses > 0, "the preload read the missed experts");
    assert_eq!([device_hits, device_misses], host);
}

/// With no cooldown every decode token of a partial store tries device
/// routes, the tokens after a miss too, and stays bit-equal to host routes:
/// at capacity 11 nearly every token misses, and the guard never trips.
/// MUTATION: a miss sends the next 16 tokens to host routes → red (tries
/// below the tokens).
/// MUTATION: a host-route layer weighs 1 in the guard → red (the guard
/// differs from the counted cost).
#[test]
fn guard_every_token_tries_the_device_path() {
    let _host = memory::scoped_fixture_host();
    const STEPS: usize = 64;
    let run = moe_replay_run(11, STEPS, &[], &RunKnobs::default());
    let [tries, replays, _, host_tail] = run.total;
    assert_eq!(tries, STEPS as u64);
    assert!(run.tried.iter().all(|&tried| tried));
    assert!(replays > 0, "the run misses");
    assert!(host_tail > 0, "a token ends on host routes");
    assert_eq!(run.trips, 0);
}

/// The guard trips only when the last 32 tokens cost more than 64 each on
/// average: 32 tokens of 65 trip it, the next 16 tokens run on host routes,
/// then a token tries device routes again on an empty window; tokens of 64
/// never trip it. On the MoE-shape fixture a guard one unit under the
/// limit trips on a miss: one trip counted, the next 16 tokens skip device
/// routes, the one after tries again, bit-equal to host routes throughout.
/// MUTATION: the guard trips at `>=` → red (64 per token trips).
/// MUTATION: a trip keeps the window → red (the first token after the host
/// tokens trips again).
/// MUTATION: a trip is not counted → red.
#[test]
fn guard_trips_and_recovers() {
    let mut guard = decoder::RouteGuard::default();
    for round in 0..2 {
        for token in 0..31 {
            assert!(!guard.record(65), "round {round} token {token}");
        }
        assert!(guard.record(65), "round {round}: 32 tokens of 65 trip");
        for token in 0..16 {
            assert!(guard.host_token(), "round {round}: host token {token}");
        }
        assert!(!guard.host_token(), "round {round}: then a device try");
    }
    let mut guard = decoder::RouteGuard::default();
    for token in 0..96 {
        assert!(!guard.record(64), "token {token} of 64");
        assert!(!guard.host_token());
    }

    let _host = memory::scoped_fixture_host();
    // Two zeros, then 2048 in all: a hit leaves the sum at the limit and a
    // miss after it passes the limit.
    let mut prime = vec![0, 0];
    prime.extend([64; 29]);
    prime.push(192);
    const STEPS: usize = 20;
    let run = moe_replay_run(
        12,
        STEPS,
        &[(1, &[1])],
        &RunKnobs {
            prime: &prime,
            ..RunKnobs::default()
        },
    );
    assert_eq!(run.planned, [[1, 1, 3, 0]], "step 1 misses at layer 1");
    assert_eq!(run.trips, 1);
    let mut tried = vec![true; STEPS];
    tried[2..18].fill(false);
    assert_eq!(run.tried, tried);
}

/// A request keeps its route guard across row switches: B starts with an
/// empty guard while A is parked, and A resumes with its own tripped guard
/// and counts a host token on its next step. Every replay, of a decode token
/// or a prefill window, counts once in the live counters; host tail layers
/// count the decode layers run again on host routes and replayed layers none
/// of them; tries count decode tokens only, and a token the device gate
/// refuses is no try.
/// MUTATION: `take_state` parks a row with an empty guard → red.
/// MUTATION: count a try before the guard and the device gate → red.
#[test]
fn guard_stays_with_its_request_across_row_switches() {
    let mut inner = tiny_inner();
    let replays = |inner: &Inner| {
        inner
            .decoder
            .weights
            .live
            .route_replays
            .load(Ordering::Relaxed)
    };
    inner.activate_paged_seq(1).unwrap();
    for token in [3, 4, 5] {
        inner.decoder.step(token).unwrap();
    }
    let from = 2;
    let (base, _, inputs) = poisoned_pass(&mut inner.decoder, 7, from);
    replay(&mut inner.decoder, base, from, inputs, 7).unwrap();
    assert_eq!(replays(&inner), 1);
    let live = Arc::clone(&inner.decoder.weights.live);
    let layers = (inner.decoder.config.num_hidden_layers - from) as u64;
    assert_eq!(live.replayed_layers.load(Ordering::Relaxed), 0);
    assert_eq!(live.host_tail_layers.load(Ordering::Relaxed), layers);
    for _ in 0..31 {
        assert!(!inner.decoder.route_guard.record(65));
    }
    assert!(inner.decoder.route_guard.record(65));
    let tripped = inner.decoder.route_guard.clone();
    inner.activate_paged_seq(2).unwrap();
    let empty = decoder::RouteGuard::default();
    assert_eq!(inner.decoder.route_guard, empty, "B starts with none");
    inner.decoder.step(6).unwrap();
    assert_eq!(inner.decoder.route_guard, empty);
    inner.activate_paged_seq(1).unwrap();
    assert_eq!(
        inner.decoder.route_guard, tripped,
        "A resumes with its own guard"
    );
    inner.decoder.step(8).unwrap();
    let mut want = tripped;
    assert!(want.host_token());
    assert_eq!(inner.decoder.route_guard, want);
    let base = inner.decoder.snapshot();
    inner.decoder.prefill_chunk(&[11; 5], None, true).unwrap();
    inner
        .decoder
        .replay_device_prefill(base, &[3, 4, 5, 6, 7], None, true)
        .unwrap();
    assert_eq!(replays(&inner), 2);
    assert_eq!(inner.decoder.route_guard, want);
    assert_eq!(
        live.replayed_layers.load(Ordering::Relaxed),
        0,
        "a prefill window replay adds no decode layers"
    );
    assert_eq!(
        live.route_tries.load(Ordering::Relaxed),
        0,
        "the tiny fixture never passes the device gate"
    );
    assert_eq!(live.host_tail_layers.load(Ordering::Relaxed), layers);
    assert_eq!(live.route_guard_trips.load(Ordering::Relaxed), 0);
}

/// A route entry is read only once its evaluation has finished, and the
/// read never waits for the GPU: an unscheduled 1-D u32 entry and an entry
/// queued behind a long GPU job give `None` and stay in flight; a finished
/// `[1, 1, 10]` array gives `None`; the finished entries a device pass
/// submits (the singleton route kernel's ids and an argpartition slice, each
/// reshaped to 1-D) give their ids while the long job is still in flight,
/// so reading them queued no GPU work behind it.
/// MUTATION: `ready_ids` reads before its check → red (the unscheduled
/// entry turns available).
/// MUTATION: `ready_ids` reads through a GPU cast → red (the long job
/// completes).
#[test]
fn early_ready_ids_never_waits() {
    use crate::array::DType;
    let reads = crate::array::unready_id_reads();
    let lazy = MxArray::arange(0., 10., None, Some(DType::Uint32)).unwrap();
    assert!(!lazy.is_available(), "arange is lazy");
    assert_eq!(lazy.ready_ids().unwrap(), None);
    assert!(
        !lazy.is_available(),
        "the unscheduled entry stays unscheduled"
    );
    let cube = lazy.reshape(&[1, 1, 10]).unwrap();
    cube.eval();
    assert!(cube.is_available());
    assert_eq!(cube.ready_ids().unwrap(), None, "a [1, 1, 10] array");

    let values: Vec<f32> = (0..512).map(|i| ((i * 37) % 512) as f32 / 512.0).collect();
    let logits = MxArray::from_float32(&values, &[1, 1, 512])
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap();
    let (singleton, _) = math::singleton_routes(&logits, 10)
        .unwrap()
        .expect("the singleton route kernel");
    let partition = MxArray::from_float32(&values[..12], &[1, 1, 12])
        .unwrap()
        .argpartition(-10, Some(-1))
        .unwrap()
        .slice_axis(2, 2, 12)
        .unwrap();
    let entries = [
        singleton.reshape(&[-1]).unwrap(),
        partition.reshape(&[-1]).unwrap(),
    ];
    MxArray::eval_arrays(&entries.iter().collect::<Vec<_>>()).unwrap();
    let want: Vec<Vec<u32>> = entries
        .iter()
        .map(|entry| entry.to_uint32().unwrap().to_vec())
        .collect();

    let scale = MxArray::ones(&[4096, 4096], Some(DType::Float32))
        .unwrap()
        .mul_scalar(1.0 / 4096.0)
        .unwrap();
    let mut job = scale.clone();
    for _ in 0..8 {
        job = job.matmul(&scale).unwrap();
    }
    let behind = job
        .reshape(&[1, 1, -1])
        .unwrap()
        .slice_axis(2, 0, 12)
        .unwrap()
        .argpartition(-10, Some(-1))
        .unwrap()
        .slice_axis(2, 2, 12)
        .unwrap()
        .reshape(&[-1])
        .unwrap();
    MxArray::async_eval_arrays(&[&job, &behind]);
    for (entry, want) in entries.iter().zip(&want) {
        assert_eq!(entry.ready_ids().unwrap().as_ref(), Some(want));
    }
    assert_eq!(behind.ready_ids().unwrap(), None, "an entry in flight");
    assert!(!job.is_available(), "a read waited for the GPU queue");
    assert!(!behind.is_available(), "the entry stays in flight");
    behind.eval();
    assert_eq!(behind.ready_ids().unwrap().map(|ids| ids.len()), Some(10));
    assert_eq!(
        crate::array::unready_id_reads(),
        reads,
        "reads of an unfinished entry"
    );
}

/// With each group completed before its routes are read, a device pass
/// stops at the group submit after its miss, reads the missed experts and
/// runs again from the missed layer, bit-equal to host routes, with int8
/// pages and flat: misses placed at layers 1, 0, 2 and 3 at capacity 12
/// (the slots also end each step as the host's do) build 1, 2, 2 and 1
/// device layers twice (the 4-layer fixture submits one group, after layer
/// 1, so misses at 2 and 3 wait for the pass end), and the misses capacity
/// 10 and 11 run into build fewer layers twice per replay than with no
/// group read.
/// MUTATION: the restart after a stop keeps the stopped pass's cache of
/// the missed layer → red.
/// MUTATION: a group submit leaves out its route entries → red (no pass
/// stops).
#[test]
fn early_cut_is_exact() {
    let _host = memory::scoped_fixture_host();
    let ready = Some(decoder::GroupPoll::Ready);
    for flat in [false, true] {
        let planned: [(usize, &[usize]); 4] = [(4, &[1]), (22, &[0]), (40, &[2]), (58, &[3])];
        let run = moe_replay_run(
            12,
            64,
            &planned,
            &RunKnobs {
                book: true,
                poll: ready,
                flat,
                ..RunKnobs::default()
            },
        );
        assert_eq!(
            run.planned,
            [[1, 1, 1, 0], [1, 1, 2, 0], [1, 1, 2, 0], [1, 1, 1, 0]],
            "flat={flat}: one device replay each"
        );
        assert_eq!(run.stops, 2, "flat={flat}: misses at layers 1 and 0 stop");
        assert_eq!(run.total[1..], [4, 6, 0], "flat={flat}");
        for capacity in [10, 11] {
            let cut = moe_replay_run(
                capacity,
                64,
                &[],
                &RunKnobs {
                    poll: ready,
                    flat,
                    ..RunKnobs::default()
                },
            );
            let whole = moe_replay_run(
                capacity,
                64,
                &[],
                &RunKnobs {
                    flat,
                    ..RunKnobs::default()
                },
            );
            eprintln!(
                "flat={flat} capacity {capacity}: [tries, replays, replayed, host tail] \
                 cut {:?} stops {} peak {}, whole {:?} peak {}",
                cut.total, cut.stops, cut.peak, whole.total, whole.peak
            );
            let what = format!("flat={flat} capacity {capacity}");
            assert!(cut.stops > 0, "{what}: no pass stopped");
            assert_eq!(whole.stops, 0, "{what}");
            let [_, cut_replays, cut_replayed, _] = cut.total;
            let [_, whole_replays, whole_replayed, _] = whole.total;
            assert!(cut_replays > 0 && whole_replays > 0, "{what}");
            assert!(
                cut_replayed * whole_replays < whole_replayed * cut_replays,
                "{what}: layers built twice per replay {cut_replayed}/{cut_replays} \
                 vs {whole_replayed}/{whole_replays}"
            );
            assert!(cut.peak <= 4, "{what}: peak {}", cut.peak);
            assert_eq!([cut.trips, whole.trips], [0, 0], "{what}");
        }
    }
}

/// On a store that holds every routed expert, reading each group's routes
/// as soon as it is submitted changes nothing: every step's submit log,
/// layer outputs, logits and decoder state equal those of passes that
/// submit and read no route entry, with int8 pages and flat; no pass
/// stops, and the reads found entries still in flight and read none of
/// them.
/// MUTATION: `ready_ids` reads before its check → red (unfinished entries
/// read).
/// MUTATION: the route read waits for each entry → red (no read finds an
/// entry in flight).
#[test]
fn early_cut_leaves_the_hit_path_alone() {
    let _host = memory::scoped_fixture_host();
    for flat in [false, true] {
        let mut polled = moe_shape_inner(12);
        let mut plain = moe_shape_inner(12);
        polled.decoder.after_group_submit = Some(decoder::GroupPoll::InFlight);
        plain.decoder.after_group_submit = Some(decoder::GroupPoll::Never);
        let c = polled.decoder.config.clone();
        let all: Vec<u32> = (0..c.num_experts as u32).collect();
        for inner in [&mut polled, &mut plain] {
            if flat {
                inner.decoder.paged = None;
            } else {
                inner.activate_paged_seq(1).unwrap();
            }
            fill_slots(&mut inner.decoder, &all);
            inner.decoder.layer_outputs = Some(Vec::new());
        }
        let counts = |inner: &Inner| {
            let live = &inner.decoder.weights.live;
            [&live.route_tries, &live.route_replays].map(|n| n.load(Ordering::Relaxed))
        };
        let (polled_start, plain_start) = (counts(&polled), counts(&plain));
        let reads = crate::array::unready_id_reads();
        const STEPS: usize = 64;
        let mut token = 3;
        for step in 0..STEPS {
            let what = format!("flat={flat} step {step} token {token}");
            for inner in [&mut polled, &mut plain] {
                inner.decoder.submit_log.borrow_mut().clear();
            }
            let got = polled.decoder.step(token).unwrap();
            let want = plain.decoder.step(token).unwrap();
            let log = polled.decoder.submit_log.borrow().clone();
            assert!(
                log.iter().any(|event| event == "qwen4::decoder::next"),
                "{what}: a group submit {log:?}"
            );
            assert_eq!(
                log,
                *plain.decoder.submit_log.borrow(),
                "{what}: submit log"
            );
            let (got_layers, want_layers) = (
                final_layer_outputs(&mut polled.decoder),
                final_layer_outputs(&mut plain.decoder),
            );
            for (layer, (x, y)) in got_layers.iter().zip(&want_layers).enumerate() {
                assert_same_bits(x, y, &format!("{what}: layer {layer} output"));
            }
            assert_same_bits(&got, &want, &format!("{what}: logits"));
            assert_same_decoder_state(&mut polled.decoder, &mut plain.decoder, &what);
            token = argmax(&got);
        }
        let steps = STEPS as u64;
        assert_eq!(
            [0, 1].map(|k| counts(&polled)[k] - polled_start[k]),
            [steps, 0],
            "flat={flat}"
        );
        assert_eq!(
            [0, 1].map(|k| counts(&plain)[k] - plain_start[k]),
            [steps, 0],
            "flat={flat}"
        );
        assert_eq!(polled.decoder.early_stops, 0, "flat={flat}");
        assert_eq!(
            crate::array::unready_id_reads(),
            reads,
            "flat={flat}: reads of an unfinished entry"
        );
        assert!(
            polled.decoder.unready_polls.get() > 0,
            "flat={flat}: no read found an entry in flight"
        );
        assert_eq!(plain.decoder.unready_polls.get(), 0, "flat={flat}");
    }
}

/// On 12 layers (attention at 3, 7 and 11; route groups submitted after
/// layers 1, 4, 7 and 10) the live rule stops a pass while the group it
/// just submitted is in flight, and the run stays bit-equal to host routes,
/// with int8 pages and flat, the slot book equal every step. A miss at
/// layer m is read at the submit after the next group: the pass stops
/// before layer 5, 8 or 11, or (m >= 8) at its end, so a replay builds
/// those layers again; the replay keeps the attention layers below m. A
/// token that misses at 0 and 3, or at 2 and 6, stops once and reads the
/// second miss's experts before its single replay. With the plain live
/// rule (the GPU decides which groups are finished) the run is bit-equal
/// too, and so are the misses capacity 10 runs into.
/// MUTATION: the preload drops a stopped pass's unread entries → red (the
/// 0 + 3 token replays twice).
/// MUTATION: only a pass's first group submit carries its route entries →
/// red (a miss at layer 2 waits for the pass end).
#[test]
fn early_cut_with_groups_in_flight_is_exact() {
    let _host = memory::scoped_fixture_host();
    const LAYERS: usize = 12;
    let single: Vec<[usize; 1]> = (0..LAYERS).map(|m| [m]).collect();
    let mut planned: Vec<(usize, &[usize])> = single
        .iter()
        .enumerate()
        .map(|(m, layers)| (2 + 3 * m, &layers[..]))
        .collect();
    planned.extend([(38, &[0, 3][..]), (41, &[2, 6][..])]);
    let stop = |m: usize| match m {
        0..2 => 5,
        2..5 => 8,
        5..8 => 11,
        _ => LAYERS,
    };
    let mut want: Vec<[u64; 4]> = (0..LAYERS)
        .map(|m| [1, 1, (stop(m) - m) as u64, 0])
        .collect();
    want.extend([[1, 1, 5, 0], [1, 1, 6, 0]]);
    for flat in [false, true] {
        let run = moe_replay_run(
            12,
            44,
            &planned,
            &RunKnobs {
                book: true,
                poll: Some(decoder::GroupPoll::Older),
                flat,
                deep: true,
                ..RunKnobs::default()
            },
        );
        assert_eq!(run.planned, want, "flat={flat}: one replay each");
        assert_eq!(run.stops, 10, "flat={flat}: misses at layers 0..8 stop");
        assert_eq!(run.unready_polls, 0, "flat={flat}");
        assert_eq!(run.total[1], planned.len() as u64, "flat={flat}");
    }
    let live = moe_replay_run(
        12,
        44,
        &planned,
        &RunKnobs {
            book: true,
            deep: true,
            ..RunKnobs::default()
        },
    );
    let natural = moe_replay_run(
        10,
        32,
        &[],
        &RunKnobs {
            deep: true,
            ..RunKnobs::default()
        },
    );
    for (what, run) in [("placed", &live), ("capacity 10", &natural)] {
        eprintln!(
            "live rule {what}: [tries, replays, replayed, host tail] {:?} stops {} \
             unready {} peak {}",
            run.total, run.stops, run.unready_polls, run.peak
        );
        assert!(run.peak <= LAYERS, "{what}: peak {}", run.peak);
    }
    assert!(
        live.planned.iter().all(|delta| delta[1] > 0),
        "every placed miss replays"
    );
}
