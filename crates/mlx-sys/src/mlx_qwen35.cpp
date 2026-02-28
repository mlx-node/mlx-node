#include "mlx_common.h"

#include <string>
#include <unordered_map>
#include <mutex>

// =============================================================================
// Qwen3.5 Fused Forward Pass
//
// Implements the entire Qwen3.5 model forward pass (single-token decode) in one
// FFI call, replacing ~3700 individual Rust→C++ calls with 1.
//
// Weight storage: All model weights are stored in a C++ unordered_map keyed by
// name (e.g. "layers.0.input_layernorm.weight"). Rust calls store/clear once
// at model load time. The forward step reads weights by name.
// =============================================================================

namespace {

// Global weight storage
static std::unordered_map<std::string, array> g_weights;
static std::mutex g_weights_mutex;

// Cached transposes for weight matrices (computed lazily)
static std::unordered_map<std::string, array> g_weight_transposes;

const array& get_weight(const std::string& name) {
  auto it = g_weights.find(name);
  if (it == g_weights.end()) {
    throw std::runtime_error("Weight not found: " + name);
  }
  return it->second;
}

// Get a transposed weight, caching the result
const array& get_weight_t(const std::string& name) {
  auto it = g_weight_transposes.find(name);
  if (it != g_weight_transposes.end()) {
    return it->second;
  }
  const auto& w = get_weight(name);
  auto wt = transpose(w);
  auto [inserted_it, _] = g_weight_transposes.emplace(name, std::move(wt));
  return inserted_it->second;
}

// Check if a weight exists
bool has_weight(const std::string& name) {
  return g_weights.count(name) > 0;
}

// RMS norm without learnable weight (for q/k norm in GatedDeltaNet)
array rms_norm_no_weight(const array& x, float eps) {
  return fast::rms_norm(x, std::nullopt, eps);
}

// SiLU activation: x * sigmoid(x)
array silu(const array& x) {
  return x * sigmoid(x);
}

// SwiGLU: silu(gate) * up — compiled for kernel fusion (called 64x per step)
static std::vector<array> swiglu_impl(const std::vector<array>& inputs) {
  const auto& gate = inputs[0];
  const auto& up = inputs[1];
  return {sigmoid(gate) * gate * up};
}

static auto& compiled_swiglu() {
  static auto fn = mlx::core::compile(swiglu_impl, /*shapeless=*/true);
  return fn;
}

array swiglu(const array& gate, const array& up) {
  return compiled_swiglu()({gate, up})[0];
}

// Softplus: log(1 + exp(x)) — preserves input dtype
array softplus(const array& x) {
  return log(mlx::core::add(exp(x), array(1.0f, x.dtype())));
}

// Fused compute_g — compiled for kernel fusion (called 48x per step)
// Matches Python: exp(-exp(A_log.astype(f32)) * softplus(a + dt_bias)).astype(A_log.dtype)
static std::vector<array> compute_g_impl(const std::vector<array>& inputs) {
  const auto& a_log = inputs[0];
  const auto& a = inputs[1];
  const auto& dt_bias = inputs[2];
  auto orig_dt = a_log.dtype();
  auto A = exp(astype(a_log, mlx::core::float32));
  auto sp = softplus(mlx::core::add(a, dt_bias));
  return {astype(exp(negative(A * sp)), orig_dt)};
}

static auto& compiled_compute_g() {
  static auto fn = mlx::core::compile(compute_g_impl, /*shapeless=*/true);
  return fn;
}

array fused_compute_g(const array& a_log, const array& a, const array& dt_bias) {
  return compiled_compute_g()({a_log, a, dt_bias})[0];
}

// Metal kernel for gated delta recurrence (single dispatch, runs entirely on GPU)
// This replaces the 22-op ops-based implementation with a single Metal kernel dispatch.
// The kernel handles GQA internally (maps Hk -> Hv via hv_idx / (Hv / Hk)).
static std::mutex qwen35_kernel_mutex;
static std::unique_ptr<mlx::core::fast::CustomKernelFunction> qwen35_gd_kernel;

static void ensure_gated_delta_kernel() {
  std::lock_guard<std::mutex> lock(qwen35_kernel_mutex);
  if (qwen35_gd_kernel) return;

  // Non-vectorized, non-masked kernel (Qwen3.5 single-token decode)
  std::string source = R"(
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    y += b_idx * T * Hv * Dv + hv_idx * Dv;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;

    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = static_cast<float>(i_state[s_idx]);
    }
    // g: [B, T, Hv]
    auto g_ = g + b_idx * T * Hv;
    auto beta_ = beta + b_idx * T * Hv;

    for (int t = 0; t < T; ++t) {
      if (true) {
        float kv_mem = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = state[i] * g_[hv_idx];
          kv_mem += state[i] * k_[s_idx];
        }
        kv_mem = simd_sum(kv_mem);

        auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

        float out = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = state[i] + k_[s_idx] * delta;
          out += state[i] * q_[s_idx];
        }
        out = simd_sum(out);
        if (thread_index_in_simdgroup == 0) {
          y[dv_idx] = static_cast<InT>(out);
        }
      }
      q_ += Hk * Dk;
      k_ += Hk * Dk;
      v_ += Hv * Dv;
      y += Hv * Dv;
      g_ += Hv;
      beta_ += Hv;
    }
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      o_state[s_idx] = static_cast<InT>(state[i]);
    }
  )";

  auto kernel = fast::metal_kernel(
      "gated_delta_step",
      {"q", "k", "v", "g", "beta", "state_in", "T"},
      {"y", "state_out"},
      source
  );
  qwen35_gd_kernel = std::make_unique<mlx::core::fast::CustomKernelFunction>(std::move(kernel));
}

// Call the Metal kernel for gated delta recurrence
// q: [B, T, Hk, Dk], k: [B, T, Hk, Dk], v: [B, T, Hv, Dv]
// g: [B, T, Hv], beta: [B, T, Hv], state: [B, Hv, Dv, Dk]
// Returns: (y [B, T, Hv, Dv], new_state [B, Hv, Dv, Dk])
std::pair<array, array> gated_delta_kernel_call(
    const array& q, const array& k, const array& v,
    const array& g, const array& beta_arr,
    const array& state) {
  ensure_gated_delta_kernel();

  int B = q.shape(0);
  int T = q.shape(1);
  int Hk = q.shape(2);
  int Dk = q.shape(3);
  int Hv = v.shape(2);
  int Dv = v.shape(3);
  auto input_type = q.dtype();
  auto T_arr = array(T, mlx::core::int32);

  auto results = (*qwen35_gd_kernel)(
      {q, k, v, g, beta_arr, state, T_arr},
      {Shape{B, T, Hv, Dv}, state.shape()},
      {input_type, input_type},
      std::make_tuple(32, Dv, B * Hv),
      std::make_tuple(32, 4, 1),
      {{"InT", input_type}, {"Dk", Dk}, {"Dv", Dv}, {"Hk", Hk}, {"Hv", Hv}},
      std::nullopt, false,
      mlx::core::default_stream(mlx::core::Device::gpu)
  );

  return std::pair<array, array>(std::move(results[0]), std::move(results[1]));
}

}  // namespace

// =============================================================================
// Compiled forward pass infrastructure
// =============================================================================
// Uses mlx::core::compile() to cache the computation graph across decode steps.
// Key design:
//   1. KV caches pre-allocated to max_kv_len (no growth during decode)
//   2. fast::rope uses array-valued offset (compilable)
//   3. slice_update uses array-valued start index (compilable)
//   4. Full-buffer SDPA with additive mask (fixed shapes)
//   5. Entire layer loop wrapped in mlx::core::compile() → graph cached after step 1
// =============================================================================

namespace {

// Config for the compiled path (set once at init time)
struct CompileConfig {
  int num_layers;
  int hidden_size;
  int num_heads;
  int num_kv_heads;
  int head_dim;
  float rope_theta;
  int rope_dims;
  float rms_norm_eps;
  int full_attention_interval;
  int linear_num_k_heads;
  int linear_num_v_heads;
  int linear_key_head_dim;
  int linear_value_head_dim;
  int linear_conv_kernel_dim;
  bool tie_word_embeddings;
  int max_kv_len; // Pre-allocated KV cache length (e.g. 4096)
  int batch_size; // Usually 1
};

static CompileConfig g_compile_config{};
static std::vector<array> g_compiled_caches;         // num_layers * 2 arrays
static std::optional<array> g_compiled_offset;       // scalar int32 (current decode position)
static int g_offset_int = 0;                         // C++ int mirror of g_compiled_offset
static bool g_compile_inited = false;

// SiLU activation fused with multiply: silu(x) * y = sigmoid(x) * x * y
// Compiled to avoid 2 separate Metal dispatches per GDN layer
static std::vector<array> silu_mul_impl(const std::vector<array>& inputs) {
  const auto& x = inputs[0];
  const auto& y = inputs[1];
  return {sigmoid(x) * x * y};
}
static auto& compiled_silu_mul() {
  static auto fn = mlx::core::compile(silu_mul_impl, /*shapeless=*/true);
  return fn;
}

// silu(x): sigmoid(x) * x — compiled to avoid 2 dispatches
static std::vector<array> silu_impl2(const std::vector<array>& inputs) {
  const auto& x = inputs[0];
  return {sigmoid(x) * x};
}
static auto& compiled_silu() {
  static auto fn = mlx::core::compile(silu_impl2, /*shapeless=*/true);
  return fn;
}

// attn_gate: attn_out * sigmoid(gate) — compiled to avoid 2 dispatches
static std::vector<array> attn_gate_impl(const std::vector<array>& inputs) {
  return {inputs[0] * sigmoid(inputs[1])};
}
static auto& compiled_attn_gate() {
  static auto fn = mlx::core::compile(attn_gate_impl, /*shapeless=*/true);
  return fn;
}

// =============================================================================
// Pure GDN forward (for compiled path)
// x is 2D [B, hidden] — reshapes to 3D/4D only for conv1d and Metal kernel.
// =============================================================================
struct GDNPureResult { array output, conv_state, recurrent_state; };

static GDNPureResult gdn_pure_fn(
    const array& x,              // [B, hidden] — 2D
    int layer_idx,
    const array& conv_state,      // [B, kernel-1, conv_dim]
    const array& recurrent_state, // [B, Hv, Dv, Dk]
    const CompileConfig& cfg) {
  int B = x.shape(0);
  int key_dim = cfg.linear_num_k_heads * cfg.linear_key_head_dim;
  int value_dim = cfg.linear_num_v_heads * cfg.linear_value_head_dim;
  int conv_dim = key_dim * 2 + value_dim;

  std::string pfx = "layers." + std::to_string(layer_idx) + ".linear_attn.";

  // Projections — 2D×2D matmul (no Flatten/Unflatten)
  auto qkvz = matmul(x, get_weight_t(pfx + "in_proj_qkvz.weight"));  // [B, qkvz_dim]
  auto ba   = matmul(x, get_weight_t(pfx + "in_proj_ba.weight"));    // [B, 2*Hv]

  // Split ba → b, a (2D slices)
  auto b = slice(ba, {0, 0},                         {B, cfg.linear_num_v_heads});
  auto a = slice(ba, {0, cfg.linear_num_v_heads},     {B, cfg.linear_num_v_heads * 2});

  // Split qkvz → qkv (conv_dim), z (value_dim) (2D slices)
  auto qkv = slice(qkvz, {0, 0},          {B, conv_dim});
  auto z   = slice(qkvz, {0, conv_dim},   {B, key_dim * 2 + value_dim * 2});

  // Reshape qkv to 3D for conv1d: [B, conv_dim] → [B, 1, conv_dim]
  auto qkv_3d = reshape(qkv, {B, 1, conv_dim});

  // Conv state is always pre-initialized — no null check
  auto conv_input = concatenate({conv_state, qkv_3d}, 1);  // [B, kernel, conv_dim]

  // Save new conv_state (last kernel-1 timesteps)
  int total_len = cfg.linear_conv_kernel_dim;
  int keep = cfg.linear_conv_kernel_dim - 1;
  auto new_conv_state = slice(conv_input, {0, total_len - keep, 0}, {B, total_len, conv_dim});

  // Depthwise conv1d (requires 3D input)
  const auto& conv_w = get_weight(pfx + "conv1d.weight");
  auto conv_out = mlx::core::conv1d(conv_input, conv_w, 1, 0, 1, conv_dim);
  // conv_out: [B, 1, conv_dim]

  // SiLU — compiled for fusion
  conv_out = compiled_silu()({conv_out})[0];

  // Reshape conv_out to 4D for Metal kernel: [B, 1, conv_dim] → split → [B, 1, H, D]
  auto q = slice(conv_out, {0, 0, 0},              {B, 1, key_dim});
  auto k = slice(conv_out, {0, 0, key_dim},        {B, 1, key_dim * 2});
  auto v = slice(conv_out, {0, 0, key_dim * 2},    {B, 1, conv_dim});

  // Reshape to heads (4D for Metal kernel)
  q = reshape(q, {B, 1, cfg.linear_num_k_heads, cfg.linear_key_head_dim});
  k = reshape(k, {B, 1, cfg.linear_num_k_heads, cfg.linear_key_head_dim});
  v = reshape(v, {B, 1, cfg.linear_num_v_heads, cfg.linear_value_head_dim});

  // RMS norm + scaling (no learnable weight)
  float inv_s = std::pow((float)cfg.linear_key_head_dim, -0.5f);
  auto q_dt = q.dtype();
  q = rms_norm_no_weight(q, 1e-6f) * array(inv_s * inv_s, q_dt);
  k = rms_norm_no_weight(k, 1e-6f) * array(inv_s, q_dt);

  // Beta = sigmoid(b); g = fused_compute_g
  // b, a are 2D [B, Hv] — reshape to 3D [B, 1, Hv] for kernel
  auto beta_3d = reshape(sigmoid(b), {B, 1, cfg.linear_num_v_heads});
  const auto& a_log  = get_weight(pfx + "A_log");
  const auto& dt_b   = get_weight(pfx + "dt_bias");
  auto g_3d = reshape(fused_compute_g(a_log, a, dt_b), {B, 1, cfg.linear_num_v_heads});

  // Gated delta recurrence (custom Metal kernel — requires 4D q/k/v, 3D g/beta)
  auto [y, new_recurrent_state] = gated_delta_kernel_call(q, k, v, g_3d, beta_3d, recurrent_state);
  // y: [B, 1, Hv, Dv]

  // RMSNorm gating: silu(z) * rms_norm(y)
  // z is 2D [B, value_dim], y is 4D [B, 1, Hv, Dv] — reshape z to match y for norm
  auto z_h = reshape(z, {B, 1, cfg.linear_num_v_heads, cfg.linear_value_head_dim});
  const auto& nw = get_weight(pfx + "norm.weight");
  auto y_normed = fast::rms_norm(y, nw, cfg.rms_norm_eps);
  // silu(z_h) * y_normed — compiled for fusion
  y_normed = compiled_silu_mul()({z_h, y_normed})[0];

  // Output: reshape to 2D [B, value_dim] for matmul
  auto y_flat = reshape(y_normed, {B, value_dim});
  auto output = matmul(y_flat, get_weight_t(pfx + "out_proj.weight"));  // [B, hidden]

  return {output, new_conv_state, new_recurrent_state};
}

// =============================================================================
// Pure attention forward (for compiled path)
// x is 2D [B, hidden] — reshapes to 4D only for rope + SDPA.
// =============================================================================
struct AttnPureResult { array output, keys, values; };

static AttnPureResult attn_pure_fn(
    const array& x,          // [B, hidden] — 2D
    int layer_idx,
    const array& kv_keys,    // [B, Hkv, max_kv_len, D]
    const array& kv_values,  // [B, Hkv, max_kv_len, D]
    const array& attn_mask,  // [1, 1, 1, max_kv_len] additive mask
    int offset,              // integer offset (current decode position)
    const CompileConfig& cfg) {
  int B = x.shape(0);
  std::string pfx = "layers." + std::to_string(layer_idx) + ".self_attn.";

  // Q projection (2x width for per-head gating) — 2D×2D matmul
  auto q_proj = matmul(x, get_weight_t(pfx + "q_proj.weight"));  // [B, 2*H*D]
  if (has_weight(pfx + "q_proj.bias")) q_proj = q_proj + get_weight(pfx + "q_proj.bias");

  // Split Q and gate — reshape to 4D for rope, keep gate as 2D
  auto qph    = reshape(q_proj, {B, 1, cfg.num_heads, cfg.head_dim * 2});
  auto queries = slice(qph, {0, 0, 0, 0},           {B, 1, cfg.num_heads, cfg.head_dim});
  auto gate    = slice(qph, {0, 0, 0, cfg.head_dim}, {B, 1, cfg.num_heads, cfg.head_dim * 2});
  gate = reshape(gate, {B, cfg.num_heads * cfg.head_dim});  // 2D for elementwise gate

  // K, V projections — 2D×2D matmul, then reshape to 4D for rope
  auto keys   = matmul(x, get_weight_t(pfx + "k_proj.weight"));  // [B, Hkv*D]
  auto values = matmul(x, get_weight_t(pfx + "v_proj.weight"));  // [B, Hkv*D]
  if (has_weight(pfx + "k_proj.bias")) keys   = keys   + get_weight(pfx + "k_proj.bias");
  if (has_weight(pfx + "v_proj.bias")) values = values + get_weight(pfx + "v_proj.bias");

  keys   = reshape(keys,   {B, 1, cfg.num_kv_heads, cfg.head_dim});
  values = reshape(values, {B, 1, cfg.num_kv_heads, cfg.head_dim});

  // QK normalization (4D — rms_norm on last dim)
  queries = fast::rms_norm(queries, get_weight(pfx + "q_norm.weight"), cfg.rms_norm_eps);
  keys    = fast::rms_norm(keys,    get_weight(pfx + "k_norm.weight"), cfg.rms_norm_eps);

  // RoPE — integer offset (more efficient kernel path than array offset)
  queries = fast::rope(queries, cfg.rope_dims, false, cfg.rope_theta, 1.0f, offset);
  keys    = fast::rope(keys,    cfg.rope_dims, false, cfg.rope_theta, 1.0f, offset);

  // Transpose to [B, H, 1, D] for SDPA
  queries = transpose(queries, {0, 2, 1, 3});
  keys    = transpose(keys,    {0, 2, 1, 3});
  values  = transpose(values,  {0, 2, 1, 3});

  // Dynamic KV cache update via slice_update (array offset for compile compat)
  auto offset_1d = reshape(array(offset, mlx::core::int32), {1});
  auto new_kv_keys   = slice_update(kv_keys,   keys,   offset_1d, {2});
  auto new_kv_values = slice_update(kv_values, values, offset_1d, {2});

  // Scaled dot-product attention with additive mask (full buffer, mask hides padding)
  float scale = std::pow((float)cfg.head_dim, -0.5f);
  auto attn_out = fast::scaled_dot_product_attention(
      queries, new_kv_keys, new_kv_values, scale, "", attn_mask, {});

  // Transpose back + reshape to 2D [B, H*D]
  attn_out = transpose(attn_out, {0, 2, 1, 3});
  attn_out = reshape(attn_out, {B, cfg.num_heads * cfg.head_dim});

  // Gate: attn_out * sigmoid(gate) — both 2D, compiled for fusion
  attn_out = compiled_attn_gate()({attn_out, gate})[0];

  // Output projection — 2D×2D matmul
  auto output = matmul(attn_out, get_weight_t(pfx + "o_proj.weight"));  // [B, hidden]
  if (has_weight(pfx + "o_proj.bias")) output = output + get_weight(pfx + "o_proj.bias");

  return {output, new_kv_keys, new_kv_values};
}

// =============================================================================
// The compilable forward function
// inputs: [h, offset_arr, cache[0].a, cache[0].b, ..., cache[N-1].a, cache[N-1].b]
// outputs: [logits, new_offset, new_cache[0].a, new_cache[0].b, ...]
// =============================================================================
static std::vector<array> qwen35_decode_fn(const std::vector<array>& inputs) {
  const auto& cfg = g_compile_config;
  auto h = inputs[0];
  // Use integer offset for RoPE (efficient kernel path).
  // Build additive attention mask from integer offset for SDPA (fixed shape → no recompilation).
  int offset = g_offset_int;

  // Attention mask: [1, 1, 1, max_kv_len] — 0 for valid positions, -inf for padding.
  // This keeps SDPA shapes fixed across steps (no variable-length slice).
  // Find first full-attention layer to get max_kv_len from its KV cache shape.
  int first_fa = cfg.full_attention_interval - 1;  // e.g., layer 3 for interval=4
  int max_kv_len = inputs[2 + first_fa * 2].shape(2);  // KV cache dim 2 = max_kv_len
  auto positions = arange(0, max_kv_len, mlx::core::int32);
  auto valid_mask = less_equal(positions, array(offset, mlx::core::int32));
  // Use bfloat16 for mask to match model dtype and avoid f32 promotion in SDPA
  auto attn_mask = where(valid_mask,
      array(0.0f, mlx::core::bfloat16),
      array(-std::numeric_limits<float>::infinity(), mlx::core::bfloat16));
  attn_mask = reshape(attn_mask, {1, 1, 1, max_kv_len});

  // Pre-fill with bf16 zeros (every element is overwritten in the loop below)
  std::vector<array> new_caches;
  new_caches.reserve(cfg.num_layers * 2);
  for (int i = 0; i < cfg.num_layers * 2; i++) {
    new_caches.push_back(zeros({}, mlx::core::bfloat16));
  }

  for (int i = 0; i < cfg.num_layers; i++) {
    bool is_linear = !((i + 1) % cfg.full_attention_interval == 0);
    std::string lp = "layers." + std::to_string(i);

    // Pre-norm
    auto normed = fast::rms_norm(h, get_weight(lp + ".input_layernorm.weight"), cfg.rms_norm_eps);

    // Attention block (linear or full)
    array layer_out = zeros({}, mlx::core::bfloat16); // placeholder; overwritten immediately below
    if (is_linear) {
      const auto& cs = inputs[2 + i * 2];     // conv_state
      const auto& rs = inputs[2 + i * 2 + 1]; // recurrent_state
      auto res = gdn_pure_fn(normed, i, cs, rs, cfg);
      layer_out = std::move(res.output);
      new_caches[i * 2]     = std::move(res.conv_state);
      new_caches[i * 2 + 1] = std::move(res.recurrent_state);
    } else {
      const auto& kk = inputs[2 + i * 2];     // kv_keys
      const auto& kv = inputs[2 + i * 2 + 1]; // kv_values
      auto res = attn_pure_fn(normed, i, kk, kv, attn_mask, offset, cfg);
      layer_out = std::move(res.output);
      new_caches[i * 2]     = std::move(res.keys);
      new_caches[i * 2 + 1] = std::move(res.values);
    }
    h = h + layer_out;

    // MLP (SwiGLU)
    std::string mp = lp + ".mlp.";
    auto mlp_in  = fast::rms_norm(h, get_weight(lp + ".post_attention_layernorm.weight"), cfg.rms_norm_eps);
    auto gate    = matmul(mlp_in, get_weight_t(mp + "gate_proj.weight"));
    auto up      = matmul(mlp_in, get_weight_t(mp + "up_proj.weight"));
    auto mlp_out = matmul(swiglu(gate, up), get_weight_t(mp + "down_proj.weight"));
    h = h + mlp_out;
  }

  // Final norm + LM head
  h = fast::rms_norm(h, get_weight("final_norm.weight"), cfg.rms_norm_eps);
  if (cfg.tie_word_embeddings) {
    h = matmul(h, get_weight_t("embedding.weight"));
  } else {
    h = matmul(h, get_weight_t("lm_head.weight"));
  }

  // Increment offset (integer — avoids lazy add-chain)
  auto new_offset = array(offset + 1, mlx::core::int32);

  // Return [logits, new_offset, new_cache_0_a, new_cache_0_b, ...]
  std::vector<array> result;
  result.reserve(2 + cfg.num_layers * 2);
  result.push_back(std::move(h));
  result.push_back(std::move(new_offset));
  for (auto& c : new_caches) result.push_back(std::move(c));
  return result;
}

static auto& compiled_qwen35_decode() {
  static auto fn = mlx::core::compile(qwen35_decode_fn);
  return fn;
}

} // namespace (compiled helpers)

// =============================================================================
// Public FFI functions
// =============================================================================

extern "C" {

void mlx_qwen35_store_weight(const char* name, mlx_array* weight) {
  std::lock_guard<std::mutex> lock(g_weights_mutex);
  auto& arr = *reinterpret_cast<array*>(weight);
  g_weights.insert_or_assign(std::string(name), arr);
}

void mlx_qwen35_clear_weights() {
  std::lock_guard<std::mutex> lock(g_weights_mutex);
  g_weights.clear();
  g_weight_transposes.clear();
}

size_t mlx_qwen35_weight_count() {
  std::lock_guard<std::mutex> lock(g_weights_mutex);
  return g_weights.size();
}

// =============================================================================
// Compiled forward pass FFI — initialize from post-prefill caches
// =============================================================================
// Call once after prefill completes, before the decode loop.
// Imports the per-layer caches and sets up the compiled state.
//
// cache_arrays[i*2]   = conv_state (GDN) or kv_keys (full-attn)
// cache_arrays[i*2+1] = recurrent_state (GDN) or kv_values (full-attn)
// All pointers must be non-null.
void mlx_qwen35_compiled_init_from_prefill(
    int num_layers,
    int hidden_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float rope_theta,
    int rope_dims,
    float rms_norm_eps,
    int full_attention_interval,
    int linear_num_k_heads,
    int linear_num_v_heads,
    int linear_key_head_dim,
    int linear_value_head_dim,
    int linear_conv_kernel_dim,
    int tie_word_embeddings,
    int max_kv_len,
    int batch_size,
    mlx_array** cache_arrays,
    int prefill_offset
) {
  g_compile_config = CompileConfig{
    num_layers, hidden_size, num_heads, num_kv_heads, head_dim,
    rope_theta, rope_dims, rms_norm_eps, full_attention_interval,
    linear_num_k_heads, linear_num_v_heads, linear_key_head_dim,
    linear_value_head_dim, linear_conv_kernel_dim,
    tie_word_embeddings != 0,
    max_kv_len, batch_size
  };

  // Build g_compiled_caches via push_back (array has no default constructor,
  // so resize/vector(N) won't work — we must construct each element explicitly).
  g_compiled_caches.clear();
  g_compiled_caches.reserve(num_layers * 2);

  for (int i = 0; i < num_layers; i++) {
    bool is_linear = !((i + 1) % full_attention_interval == 0);

    if (is_linear) {
      g_compiled_caches.push_back(*reinterpret_cast<array*>(cache_arrays[i * 2]));
      g_compiled_caches.push_back(*reinterpret_cast<array*>(cache_arrays[i * 2 + 1]));
    } else {
      auto& kk = *reinterpret_cast<array*>(cache_arrays[i * 2]);
      auto& kv = *reinterpret_cast<array*>(cache_arrays[i * 2 + 1]);
      int current_cap = kk.shape(2);
      if (current_cap < max_kv_len) {
        int pad_len = max_kv_len - current_cap;
        auto kpad = zeros({batch_size, num_kv_heads, pad_len, head_dim}, kk.dtype());
        auto vpad = zeros({batch_size, num_kv_heads, pad_len, head_dim}, kv.dtype());
        g_compiled_caches.push_back(concatenate({kk, kpad}, 2));
        g_compiled_caches.push_back(concatenate({kv, vpad}, 2));
      } else {
        g_compiled_caches.push_back(kk);
        g_compiled_caches.push_back(kv);
      }
    }
  }

  g_compiled_offset = array(prefill_offset, mlx::core::int32);
  g_offset_int = prefill_offset;
  g_compile_inited = true;

  // CRITICAL: Evaluate the global RNG key state to break the lazy split chain.
  // During model loading, KeySequence::next() is called 400+ times, each creating
  // a lazy split(key). Without evaluation, the decode step's categorical() call
  // pulls the entire 400+ node chain into the eval tape (~452 RandomBits + 452 Split).
  // Evaluating the key here materializes it, so decode steps see a concrete key.
  mlx::core::random::KeySequence::default_().eval_key();
}

// =============================================================================
// Per-function compiled decode step (Python-style)
// =============================================================================
// Calls qwen35_decode_fn directly WITHOUT a parent mlx::core::compile() wrapper.
//
// WHY NOT A PARENT COMPILE:
// A parent compile() causes nested compiled sub-functions (fused_compute_g,
// compiled_swiglu, etc.) to receive tracer inputs, which makes them fall through
// to their uncompiled implementations — inlining all ops into the parent trace.
// For 64 layers this produces ~5408 scheduler tasks vs Python's ~1620.
//
// Python mlx-lm uses per-function @mx.compile on small hot functions (compute_g,
// gated_delta_step, etc.) with NO parent compile on the decode loop. Each compiled
// sub-function becomes a single "Compiled" primitive node (1 task). We do the same.
//
// GRAPH ACCUMULATION PREVENTION:
// Without a parent compile, each decode step builds a fresh computation graph.
// New cache arrays depend on the previous step's outputs (chained). To prevent
// O(N) accumulation, mlx_qwen35_eval_token_and_compiled_caches calls async_eval
// on token + all caches after each step. This marks caches as "scheduled", so
// eval_impl treats them as leaves (it only recurses into unscheduled arrays).
//
// output_logits receives a new heap-allocated array (caller must free via mlx_array_free).
// cache_offset_out (optional) receives the updated decode position.
void mlx_qwen35_forward_compiled(
    mlx_array* input_ids_ptr,
    mlx_array* embedding_weight_ptr,
    mlx_array** output_logits,
    int* cache_offset_out
) {
  if (!g_compile_inited) {
    *output_logits = nullptr;
    return;
  }
  const auto& cfg = g_compile_config;

  try {
    auto& input_ids      = *reinterpret_cast<array*>(input_ids_ptr);
    auto& embedding_weight = *reinterpret_cast<array*>(embedding_weight_ptr);

    // Embedding lookup: [B, 1] int → [B, hidden] (2D to avoid Flatten/Unflatten in matmul)
    auto flat_ids = reshape(input_ids, {-1});
    auto h = take(embedding_weight, flat_ids, 0);
    // Keep h as 2D [B, hidden] — matches Python's convention for single-token decode.
    // 2D×2D matmul avoids Flatten+Unflatten wrapper primitives (~802 tasks saved).

    // Build inputs vector: [h, offset_arr, cache[0], cache[1], ..., cache[N*2-1]]
    // Pass a fresh scalar array for offset each step — avoids accumulating a lazy
    // add-chain (new_offset = add(add(add(..., 1), 1), 1)) across decode steps.
    std::vector<array> inputs;
    inputs.reserve(2 + cfg.num_layers * 2);
    inputs.push_back(std::move(h));
    inputs.push_back(array(g_offset_int, mlx::core::int32));
    for (const auto& c : g_compiled_caches) {
      inputs.push_back(c);
    }

    // Run forward pass directly (no parent compile wrapper — see comment above).
    // Sub-compiled functions (fused_compute_g, compiled_swiglu, etc.) will do their
    // own compile-trace on first call and compile-replace on subsequent calls,
    // each producing 1 Compiled primitive node (1 GPU task) instead of N inlined ops.
    auto outputs = qwen35_decode_fn(inputs);

    // outputs[0] = logits [B, vocab] (2D)
    // outputs[1] = new_offset (scalar int32, not used — we track in g_offset_int)
    // outputs[2..] = new_caches
    *output_logits = reinterpret_cast<mlx_array*>(new array(outputs[0]));
    g_offset_int++;
    for (int i = 0; i < cfg.num_layers * 2; i++) {
      g_compiled_caches[i] = outputs[2 + i];
    }

    if (cache_offset_out) {
      *cache_offset_out = g_offset_int;
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in mlx_qwen35_forward_compiled: %s\n", e.what());
    fflush(stderr);
    *output_logits = nullptr;
  } catch (...) {
    fprintf(stderr, "[MLX] Unknown exception in mlx_qwen35_forward_compiled\n");
    fflush(stderr);
    *output_logits = nullptr;
  }
}

// =============================================================================
// Async-eval token + caches after each decode step
// =============================================================================
// Analogous to Python mlx-lm's `mx.async_eval(next_y, cache)`.
//
// WHY async_eval token + ALL caches:
//   Each decode step builds a fresh computation graph where new cache arrays
//   depend on the previous step's caches. Without this call, eval_impl would
//   recurse into the unscheduled previous-step caches → their entire forward
//   pass graph → building an O(N) chain that grows with every step.
//
//   async_eval marks arrays as "scheduled" (GPU kernel dispatched).
//   eval_impl only recurses into Status::unscheduled arrays — scheduled arrays
//   are treated as leaves. This bounds the per-step graph size to O(1).
//
// WHY async_eval (not sync eval):
//   Non-blocking. While the CPU builds step N+1's graph, the GPU executes step N.
//   This is the standard pipelining pattern in mlx-lm.
void mlx_qwen35_eval_token_and_compiled_caches(mlx_array* next_token_ptr) {
  try {
    std::vector<array> to_eval;
    to_eval.reserve(1 + g_compiled_caches.size());
    to_eval.push_back(*reinterpret_cast<array*>(next_token_ptr));
    for (const auto& c : g_compiled_caches) {
      to_eval.push_back(c);
    }
    mlx::core::async_eval(std::move(to_eval));
  } catch (const std::exception& e) {
    fprintf(stderr, "[MLX] Exception in eval_token_and_compiled_caches: %s\n", e.what());
    fflush(stderr);
  } catch (...) {
    fprintf(stderr, "[MLX] Unknown exception in eval_token_and_compiled_caches\n");
    fflush(stderr);
  }
}

// Reset compiled state (call on model reset/new conversation).
void mlx_qwen35_compiled_reset() {
  g_compiled_caches.clear();
  g_compiled_offset = std::nullopt;
  g_offset_int = 0;
  g_compile_inited = false;
}

}  // extern "C"
