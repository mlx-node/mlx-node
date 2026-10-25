// Grouped K-quant MoE execution, following Splash's ops/MoE.cpp.
// Apache-2.0 provenance is recorded in THIRD_PARTY_NOTICES.
#include "mlx_common.h"
#include "mlx/allocator.h"
#include "mlx/primitives.h"
#include "mlx/transforms_impl.h"
#include "mlx_kquant.h"
#include "mlx_test_counters.h"
#include "metal/splash_moe/abi.h"

#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx_paged_metallib.h"
#include "mlx_splash_moe_workspace.h"
#endif

#include <array>
#include <cstddef>
#include <cstdio>
#include <limits>

namespace mlx::core::splash_moe {

// The source router and selector consume weights instead of logits. Keep
// this operand contract separate from the unchanged60-byte expert Metal ABI.
inline bool source_router_inputs(const std::vector<array>& a) {
  return a.size()>=3 && a[0].shape()==Shape{8,2048} && a[0].dtype()==bfloat16 &&
      a[1].shape()==Shape{256,2048} && a[1].dtype()==bfloat16 &&
      a[2].shape()==Shape{1,2048} && a[2].dtype()==bfloat16;
}
inline bool source_routing_inputs(const std::vector<array>& a, const SplashMoeParams& p) {
  if(!source_router_inputs(a) || p.rows!=8 || p.hidden!=2048 || p.intermediate!=512 ||
     p.experts!=256 || p.top_k!=8 || !(p.normalize&1) || !p.source_math ||
     a.size()!=size_t(21+((p.normalize&8)?1:0)+p.source_residual)) return false;
  for(int j=0;j<6;++j) {
    uint32_t f=p.formats[j],bits=f&255;
    if((f&~511u) || (j%3==2 ? bits!=4&&bits!=6 : bits!=4)) return false;
    int N=j%3==2?2048:512,K=j%3==2?512:2048;
    const auto& w=a[3+3*j];const auto& sc=a[4+3*j];const auto& bias=a[5+3*j];
    Shape ws=j<3?Shape{256,N,K*int(bits)/32}:Shape{N,K*int(bits)/32};
    Shape ss=j<3?Shape{256,N,K/16}:Shape{N,K/16};
    Shape bs=j<3?Shape{256,N,K/256*(bits==6?1:2)}:Shape{N,K/256*(bits==6?1:2)};
    if(w.shape()!=ws || w.dtype()!=uint32 || sc.shape()!=ss || sc.dtype()!=(bits==6?int8:uint8) ||
       bias.shape()!=bs || bias.dtype()!=float16) return false;
  }
  if((p.normalize&8) && (a[21].dtype()!=int32 || a[21].size()!=1)) return false;
  return !p.source_residual || (a.back().shape()==a[0].shape() && a.back().dtype()==bfloat16);
}
#ifdef MLX_NODE_METAL_ENABLED
// Exact copy of the proven source arithmetic, deliberately isolated from
// the native expert metallib's safe-math compilation. Safe routing changes
// coefficient bits; MathMode::Fast matches the original Splash kernels.
struct SourceFloatParams { uint32_t rows,input_size,output_size[2],out_stride[2],out_offset[2]; };
struct SourceRouteParams { uint32_t rows,input_size,experts,top_k; };
static_assert(sizeof(SourceFloatParams)==32 && sizeof(SourceRouteParams)==16,"source routing ABI");
void encode_source_routing(metal::Device& d, metal::CommandEncoder& enc,
    const array& x,const array& router,const array& shared,
    array& scores,size_t score_offset,array& selected,size_t selected_offset,
    array& weights,size_t weights_offset) {
  CompileOptions options;options.math_mode=MathMode::Fast;
  auto* library=d.get_library("mlx_node_source_routing_m8_e256_fast_v1",options,[]{
    return std::string(
#include "metal/common/source_routing.metal.inc"
    );
  });
  auto* project=d.get_kernel("mlx_splash_source_router",library);
  auto* route=d.get_kernel("mlx_splash_source_route",library);
  for(auto* pipeline:{project,route}) {
    if(pipeline->threadExecutionWidth()!=32 || pipeline->maxTotalThreadsPerThreadgroup()<256 ||
       pipeline->staticThreadgroupMemoryLength()>d.mtl_device()->maxThreadgroupMemoryLength())
      throw std::runtime_error("source routing threadgroup unsupported");
  }
  SourceFloatParams fp{8,2048,{256,0},{256,256},{0,0}};
  enc.set_compute_pipeline_state(project);
  enc.set_input_array(x,0);enc.set_input_array(router,1);
  enc.set_output_array(scores,2,score_offset);
  // The second segment has zero columns and is never accessed.
  enc.set_input_array(router,3);enc.set_output_array(scores,4,score_offset);enc.set_bytes(fp,5);
  enc.dispatch_threadgroups(MTL::Size(32,1,1),MTL::Size(256,1,1));
  SourceRouteParams rp{8,2048,256,8};
  enc.set_compute_pipeline_state(route);
  enc.set_input_array(scores,0,score_offset);enc.set_input_array(x,1);enc.set_input_array(shared,2);
  enc.set_output_array(selected,3,selected_offset);enc.set_output_array(weights,4,weights_offset);enc.set_bytes(rp,5);
  enc.dispatch_threadgroups(MTL::Size(8,1,1),MTL::Size(256,1,1));
  bridge_testing::record("splash_moe_source_routing");
}
#endif

class GroupedMoe : public UnaryPrimitive {
 public:
  GroupedMoe(Stream s, SplashMoeParams params, bool source_routing=false, bool workspace=false)
      : UnaryPrimitive(s), p_(params), source_routing_(source_routing), workspace_(workspace) {}
  DEFINE_NAME(GroupedMoe)
  bool is_equivalent(const Primitive &other) const override {
    const auto& rhs=static_cast<const GroupedMoe&>(other);
    return workspace_==rhs.workspace_ && source_routing_==rhs.source_routing_ && std::memcmp(&p_, &rhs.p_, sizeof(p_)) == 0;
  }
  std::vector<Shape> output_shapes(const std::vector<array> &inputs) override {
    if(source_routing_ && !source_routing_inputs(inputs,p_))
      throw std::invalid_argument("source routing replay changed its measured input contract");
    if (p_.normalize >> 16) return {{1}};
    if ((p_.normalize >> 8)&255) return {{inputs[0].shape(0), int(p_.top_k+1), int((p_.normalize>>8)==4 ? p_.hidden : p_.intermediate)}};
    return {inputs[0].shape()};
  }
  void eval_cpu(const std::vector<array> &, array &) override {
    throw std::runtime_error("grouped GGUF MoE requires a Metal tensor-op device");
  }
  void eval_gpu(const std::vector<array> &inputs, array &out) override;
 private:
  SplashMoeParams p_;
  bool source_routing_;
  bool workspace_;
};

#ifdef MLX_NODE_METAL_ENABLED
void GroupedMoe::eval_gpu(const std::vector<array> &inputs, array &out) {
  // A shapeless compile may replay a different token count. The weight
  // geometry stays fixed, but rows, tiers and workspaces are replay-local.
  SplashMoeParams p = p_;
  if(source_routing_ && (!source_routing_inputs(inputs,p) ||
     metal::device(stream().device).get_architecture_gen()<17))
    throw std::invalid_argument("source routing replay changed its measured input contract");
  if (inputs[0].ndim() != 2 || inputs[0].shape(1) != int(p.hidden)) {
    throw std::invalid_argument("grouped GGUF MoE replay changed its hidden dimension");
  }
  if ((p.normalize&8) && (inputs.size()!=size_t(22+p.source_residual) || inputs[21].dtype()!=int32 || inputs[21].size()!=1))
    throw std::invalid_argument("grouped GGUF MoE replay changed its budget shape/type");
  if (p.source_residual &&
      (inputs.size()!=size_t(22+((p.normalize&8)?1:0)) ||
       inputs.back().dtype()!=bfloat16 || inputs.back().shape()!=inputs[0].shape()))
    throw std::invalid_argument("source MoE residual replay changed shape/dtype");
  p.rows = inputs[0].shape(0);
  if (p.rows < 1 || p.rows > 2048) {
    throw std::invalid_argument("grouped GGUF MoE replay exceeds its token budget");
  }
  p.tile_rows = p.rows * p.top_k <= p.experts ? 8 : 32;
  if (!source_routing_ && (inputs[1].size() != size_t(p.rows) * p.experts || inputs[2].size() != p.rows)) {
    throw std::invalid_argument("grouped GGUF MoE replay changed its routing shape");
  }
  for (int j = 0; j < 6; ++j) {
    size_t elements = size_t(j < 3 ? p.experts : 1) * p.hidden * p.intermediate;
    uint32_t bits = p.formats[j] & 255;
    if (inputs[3 + j * 3].size() != elements * bits / 32 ||
        inputs[4 + j * 3].size() != elements / 16 ||
        inputs[5 + j * 3].size() != elements / 256 * (bits == 6 ? 1 : 2)) {
      throw std::invalid_argument("grouped GGUF MoE replay changed its weight geometry");
    }
  }
  // The legacy body-only diagnostic may replay native fallback widths and
  // retains native routing. The separately validated source-routing contract
  // is M8-only and uses original source FP32 scores/selection.
  p.source_math = p.source_math && p.rows==8 && p.hidden==2048 &&
      p.intermediate==512 && p.experts==256 && p.top_k==8 &&
      (p.normalize&6)==6 &&
      metal::device(stream().device).get_architecture_gen() >= 17 &&
      (p.formats[0]&255)==4 && (p.formats[1]&255)==4 &&
      ((p.formats[2]&255)==4 || (p.formats[2]&255)==6) &&
      (p.formats[3]&255)==4 && (p.formats[4]&255)==4 &&
      ((p.formats[5]&255)==4 || (p.formats[5]&255)==6);
  if (p.source_math) {
    p.normalize &= ~(16u|32u|64u); // source staged schedule has precedence
    bridge_testing::record("splash_moe_source_math");
    bridge_testing::record("splash_moe_source_column");
  }
  const bool register_experts = (p.normalize & 16) && p.rows == 8 &&
      metal::device(stream().device).get_architecture_gen() >= 17 &&
      (p.formats[0]&255)==4 && (p.formats[1]&255)==4 &&
      ((p.formats[2]&255)==4 || (p.formats[2]&255)==6) &&
      (p.formats[3]&255)==4 && (p.formats[4]&255)==4 &&
      ((p.formats[5]&255)==4 || (p.formats[5]&255)==6);
  // A distinct expert occurs at most once per physical row, so M8 always
  // fits eight rows per expert even when the generic prefill plan chose32.
  if (register_experts) p.tile_rows = 8;
  // Match the existing projection families' weight precision. Routed
  // vector gather decodes in F32, grouped gather stages BF16. A dense
  // shared projection uses the small-M half tier only at M=8 before its
  // normal GEMM cutoff; the GEMM stages BF16.
  const bool routed_bf16 = p.rows*p.top_k >= 64 && p.rows*p.top_k/p.experts >= 4;
  for (int j=0;j<6;++j) {
    uint32_t precision=0; // 0 F32, 1 half, 2 BF16 decoded weights.
    if(j<3) precision=routed_bf16 ? 2u : 0u;
    else {
      int N=j%3==2 ? int(p.hidden) : int(p.intermediate);
      int K=j%3==2 ? int(p.intermediate) : int(p.hidden);
      if(j!=5 && (p.formats[3]&511)==(p.formats[4]&511)) N*=2;
      if(int(p.rows)>=kquant::tiled_qmv_vector_limit(K,N)) precision=2;
      else if(p.rows==8 && (p.formats[j]&256)) precision=1;
    }
    if(p.source_math) precision=1; // both column and staged source weights are half
    p.formats[j] |= precision<<9;
  }
  auto &s = stream();
  auto &d = metal::device(s.device);
  auto &enc = metal::get_command_encoder(s);
  std::vector<array> a;
  a.reserve(inputs.size());
  for (const auto &input : inputs) {
    // A contiguous slice can still start inside a Metal allocation. The
    // activation, payload and subscale vector loads require uint4 alignment.
    if (input.flags().row_contiguous && input.offset() % 16 == 0) a.push_back(input);
    else {
      array contiguous(input.shape(), input.dtype(), nullptr, {});
      contiguous.set_data(allocator::malloc(contiguous.nbytes()));
      // General copy has no rank-zero Metal specialization (g0). Scalar
      // always allocates fresh storage; Vector may donate the misaligned
      // input and would defeat this alignment normalization.
      copy_gpu(input, contiguous,
               input.ndim() == 0 ? CopyType::Scalar : CopyType::General, s);
      enc.add_temporary(contiguous);
      a.push_back(std::move(contiguous));
    }
  }
  out.set_data(allocator::malloc(out.nbytes()));
  const uint32_t route_count = p.rows * (p.top_k + 1);
  const uint32_t routed = p.rows * p.top_k;
  // Sum_e ceil(count_e / tile_rows) <= ceil(routes/tile_rows)+E-1,
  // also bounded by the number of routes. No CPU routing readback.
  const uint32_t max_tiles = std::min(routed, (routed + p.tile_rows - 1) / p.tile_rows + p.experts - 1) +
                            (p.rows + p.tile_rows - 1) / p.tile_rows;
  const int grouped_rows = int(max_tiles * p.tile_rows);
  auto temporary = [&](Shape shape, Dtype dtype) {
    array a(std::move(shape), dtype, nullptr, {});
    a.set_data(allocator::malloc(a.nbytes()));
    enc.add_temporary(a);
    return a;
  };
  const uint32_t probe = (p.normalize >> 8)&255;
  const uint32_t profile = p.normalize >> 16;
  const std::vector<array>* workspace = nullptr;
  if (workspace_ && p.rows==8 && p.hidden==2048 && p.intermediate==512 &&
      p.experts==256 && p.top_k==8 && p.tile_rows==8 &&
      !register_experts && !probe && !profile) {
    // Borrow immutable descriptors, then copy handles locally for ordinary
    // tracked bindings. Never donate, expose, replace Data, or add_temporary.
    static const std::vector<persistent_workspace::Spec> specs = {
      {{72},uint32},{{72},float32},{{65,2},uint32},{{7},uint32},
      {{520},uint32},{{72},uint32},{{520,2048},bfloat16},
      {{520,512},bfloat16},{{520,2048},bfloat16}};
    static_assert(sizeof(SplashMoeTileCount)==7*sizeof(uint32_t));
    workspace=persistent_workspace::try_workspace(s,enc,0x53504c4153480001ULL,specs);
    if(workspace) bridge_testing::record("splash_moe_workspace");
  }
  auto scratch = [&](size_t slot, Shape shape, Dtype dtype) {
    return workspace ? (*workspace)[slot] : temporary(std::move(shape),dtype);
  };
  auto selected = scratch(0,{int(route_count)}, uint32);
  auto weights = scratch(1,{int(route_count)}, float32);
  auto tiles = scratch(2,{int(max_tiles), 2}, uint32);
  auto count = scratch(3,{int(sizeof(SplashMoeTileCount) / sizeof(uint32_t))}, uint32);
  auto routes = scratch(4,{grouped_rows}, uint32);
  auto inverse = scratch(5,{int(route_count)}, uint32);
  auto gathered = scratch(6,{grouped_rows, int(p.hidden)}, bfloat16);
  auto hidden = scratch(7,{grouped_rows, int(p.intermediate)}, bfloat16);
  auto projected = scratch(8,{grouped_rows, int(p.hidden)}, bfloat16);
  auto input_sums = register_experts ? temporary({int(max_tiles),int(p.hidden/32*8)},float32) : gathered;
  auto hidden_packed = register_experts ? temporary({grouped_rows,int(p.intermediate)},bfloat16) : hidden;
  // Q6 has two independent16-value sums per32-value span; Q4 uses the
  // first half of each tile's reserved capacity.
  auto hidden_sums = register_experts ? temporary({int(max_tiles),int(p.intermediate/16*8)},float32) : hidden;

  // Captured in primitive equivalence. This counts an opted-in launch, not
  // the device-dependent number of experts taking the dense F32 tile.
  if ((p.normalize & 32) && !register_experts)
    bridge_testing::record("splash_moe_paired_stage_enabled");
  auto profile_done = [&](uint32_t stage, const array &value) {
    if (profile != stage) return false;
    // The initialized first selected id or tile count is a real scalar
    // output; no uninitialized workspace elements escape the diagnostic.
    out.copy_shared_buffer(value, {1}, out.flags(), 1);
    return true;
  };
  auto diagnostic = probe==1 || probe==2 ? temporary({grouped_rows,int(p.intermediate)},bfloat16) : hidden;
  auto kernel = [&](const std::string &name) {
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(d, "splash_moe", "grouped", name));
  };
  const char *indirect_flag = std::getenv("MLX_SPLASH_MOE_INDIRECT");
  const bool indirect = !indirect_flag || std::strcmp(indirect_flag, "1") == 0;
  if (indirect) bridge_testing::record("splash_moe_indirect");
  auto expert_grid = [&](uint32_t columns, size_t offset) {
    if (indirect) {
      // Both expert kernels bind count as an input. MLX therefore tracks
      // the grouping write, barrier and retained argument-buffer storage.
      enc.dispatch_threadgroups_indirect(
          static_cast<const MTL::Buffer *>(count.buffer().ptr()), count.offset() + offset,
          MTL::Size(register_experts ? 256 : 64, 1, 1));
    } else {
      enc.dispatch_threadgroups(MTL::Size(columns, max_tiles, 1), MTL::Size(register_experts ? 256 : 64, 1, 1));
    }
  };
  const uint32_t expert_bound = std::min(p.experts, p.rows * p.top_k);
  auto bind_routed = [&](const array& weight, int index) {
    // T64 and row-major layouts keep complete, equally sized expert blocks.
    // Never guess a budget for strided/suballocation/shared-view descriptors.
    // The bound uses physical rows, not the device-resident adaptive budget.
    const bool bounded = (p.normalize & 128) && expert_bound < p.experts &&
        weight.ndim() == 3 && weight.shape(0) == int(p.experts) &&
        weight.flags().row_contiguous && weight.offset() == 0 &&
        weight.data_size() == weight.size() && weight.strides(0) > 0 &&
        size_t(weight.strides(0)) * p.experts == weight.data_size();
    if (bounded) {
      enc.set_input_array_with_budget(
          weight, index, size_t(weight.strides(0)) * expert_bound);
    } else {
      enc.set_input_array(weight, index);
    }
  };
  if(source_routing_) {
    // Scores occupy the initial8*256*sizeof(float) bytes of gathered until
    // selection finishes. The following gather overwrites it only after the
    // encoder's resource dependency barrier; no extra resident score buffer.
    if(gathered.nbytes()<size_t(8*256*sizeof(float))) throw std::logic_error("source score scratch extent");
    encode_source_routing(d,enc,a[0],a[1],a[2],gathered,0,selected,0,weights,0);
  } else {
    kernel("splash_moe_route");
    enc.set_input_array(a[1], 0); enc.set_input_array(a[2], 1);
    enc.set_output_array(selected, 2); enc.set_output_array(weights, 3);
    enc.set_bytes(p, 4); enc.set_input_array((p.normalize&8) ? a[21] : a[0],5);
    enc.dispatch_threadgroups(MTL::Size(p.rows, 1, 1), MTL::Size(64, 1, 1));
  }
  if (profile_done(1, selected)) return;

  kernel("splash_moe_group");
  enc.set_input_array(selected, 0); enc.set_output_array(tiles, 1);
  enc.set_output_array(count, 2); enc.set_output_array(routes, 3);
  enc.set_output_array(inverse, 4); enc.set_bytes(p, 5);
  enc.set_input_array((p.normalize&8) ? a[21] : a[0],6);
  enc.dispatch_threadgroups(MTL::Size(1, 1, 1), MTL::Size(256, 1, 1));
  if (profile_done(2, count)) return;

  kernel(register_experts ? "splash_moe_gather_sg8" : "splash_moe_gather");
  enc.set_input_array(a[0], 0); enc.set_input_array(routes, 1);
  enc.set_input_array(tiles, 2); enc.set_input_array(count, 3);
  enc.set_output_array(gathered, 4); enc.set_bytes(p, 5);
  if (register_experts) {
    enc.set_output_array(input_sums,6);
    enc.dispatch_threadgroups(MTL::Size(max_tiles,p.hidden/128,1),MTL::Size(128,1,1));
  } else enc.dispatch_threadgroups(MTL::Size(max_tiles, p.hidden / 256, 1), MTL::Size(256, 1, 1));
  if (profile_done(3, count)) return;

  kernel(p.source_math ? "splash_moe_gate_up_source_8" : register_experts ? "splash_moe_gate_up_sg8" : "splash_moe_gate_up_" + std::to_string(p.tile_rows));
  enc.set_input_array(gathered, 0); enc.set_input_array(tiles, 1); enc.set_input_array(count, 2);
  // Routed gate/up and shared gate/up, allowing all four quant formats to differ.
  for (int i=0; i<6; ++i) bind_routed(a[3+i], 3+i);
  for (int i=0; i<6; ++i) enc.set_input_array(a[12+i], 9+i);
  enc.set_output_array(hidden, 15); enc.set_bytes(p, 16);
  enc.set_output_array(diagnostic, 17);
  if (register_experts) {
    enc.set_input_array(input_sums,18); enc.set_output_array(hidden_packed,19); enc.set_output_array(hidden_sums,20);
    bridge_testing::record("splash_moe_register_gate_up");
    if (p.normalize & 64) bridge_testing::record("splash_moe_register_hybrid");
  }
  expert_grid(p.intermediate / 32, offsetof(SplashMoeTileCount, gate_up_grid));
  if (profile_done(4, count)) return;

  kernel(p.source_math ? "splash_moe_down_source_8" : register_experts ? "splash_moe_down_sg8" : "splash_moe_down_" + std::to_string(p.tile_rows));
  enc.set_input_array(register_experts ? hidden_packed : hidden, 0); enc.set_input_array(tiles, 1); enc.set_input_array(count, 2);
  for (int i=0; i<3; ++i) bind_routed(a[9+i], 3+i);
  for (int i=0; i<3; ++i) enc.set_input_array(a[18+i], 6+i);
  enc.set_output_array(projected, 9); enc.set_bytes(p, 10);
  if (register_experts) { enc.set_input_array(hidden_sums,11); bridge_testing::record("splash_moe_register_down"); }
  expert_grid(p.hidden / 64, offsetof(SplashMoeTileCount, down_grid));
  if (profile_done(5, count)) return;

  if (probe) {
    kernel("splash_moe_probe_scatter");
    enc.set_input_array(probe==4 ? projected : probe==3 ? hidden : diagnostic,0);
    enc.set_input_array(inverse,1); enc.set_output_array(out,2); enc.set_bytes(p,3);
    enc.dispatch_threadgroups(MTL::Size(route_count,(probe==4 ? p.hidden : p.intermediate)/256,1),MTL::Size(256,1,1));
    return;
  }
  auto combined = profile ? temporary({int(p.rows),int(p.hidden)},(p.normalize&6)==6 ? bfloat16 : float32) : out;
  kernel("splash_moe_combine");
  enc.set_input_array(projected, 0); enc.set_input_array(inverse, 1);
  enc.set_input_array(weights, 2); enc.set_output_array(combined, 3); enc.set_bytes(p, 4);
  enc.set_input_array((p.normalize&8) ? a[21] : a[0],5);
  enc.set_input_array(p.source_residual ? a.back() : a[0],6);
  enc.dispatch_threadgroups(MTL::Size(p.rows, p.hidden / 256, 1), MTL::Size(256, 1, 1));
  if (profile) profile_done(6, count);
}
#else
void GroupedMoe::eval_gpu(const std::vector<array> &, array &) {
  throw std::runtime_error("grouped GGUF MoE requires Metal");
}
#endif

bool available() {
#ifdef MLX_NODE_METAL_ENABLED
  const char *enabled = std::getenv("MLX_SPLASH_MOE");
  return !(enabled && std::strcmp(enabled, "0") == 0) && !detail::in_grad_tracing() &&
         default_device() == Device::gpu && is_available(Device::gpu) && metal::is_nax_available();
#else
  return false;
#endif
}

bool source_residual_available() {
#ifdef MLX_NODE_METAL_ENABLED
  return available() && metal::device(Device::gpu).get_architecture_gen()>=17;
#else
  return false;
#endif
}

array grouped(std::vector<array> a, const uint32_t *formats, int top_k, bool normalize, int probe=0, int profile=0, int source_math_policy=-1, const array* residual=nullptr, bool source_routing=false) {
  if(source_math_policy < -1 || source_math_policy > 1) throw std::invalid_argument("invalid source MoE arithmetic policy");
  if (profile<0 || profile>6 || (profile && probe)) throw std::invalid_argument("invalid grouped MoE profile cutoff");
  if (probe<0 || probe>4) throw std::invalid_argument("invalid grouped MoE probe stage");
  if (!available()) throw std::invalid_argument("grouped GGUF MoE is unavailable");
  if(source_routing && ((a.size()!=21 && a.size()!=22) || !source_router_inputs(a) ||
     !normalize || top_k!=8 || !source_residual_available() || probe || profile))
    throw std::invalid_argument("source routing requires measured M8 BF16 profile");
  if (!source_routing && ((a.size()!=21 && a.size()!=22) || a[0].ndim()!=2 || a[0].dtype()!=bfloat16 || a[1].ndim()!=2 ||
      (a[1].dtype()!=float32 && a[1].dtype()!=bfloat16) || (a[2].dtype()!=float32 && a[2].dtype()!=bfloat16) || a[0].shape(0)<1 || a[0].shape(0)>2048 ||
      a[1].shape(0)!=a[0].shape(0) || a[2].size()!=a[0].shape(0) ||
      top_k<1 || top_k>8 || a[1].shape(1)<top_k || a[1].shape(1)>256))
    throw std::invalid_argument("invalid grouped GGUF MoE activation/router inputs");
  if(a.size()==22 && (a[21].dtype()!=int32 || a[21].size()!=1))
    throw std::invalid_argument("grouped GGUF MoE row budget must be scalar int32");
  // The diagnostic scatter visits every route slot, including inactive slots
  // whose inverse indices a budget intentionally leaves uninitialized.
  if (probe && a.size()==22)
    throw std::invalid_argument("grouped MoE stage probes do not support a row budget");
  uint32_t rows=a[0].shape(0), H=a[0].shape(1), E=source_routing?256:a[1].shape(1);
  if (a[3].ndim()!=3 || a[3].shape(0)!=int(E)) throw std::invalid_argument("invalid grouped GGUF expert stack");
  uint32_t I=a[3].shape(1);
  if (H%256 || I%256 || H<256 || I<256) throw std::invalid_argument("grouped GGUF MoE needs whole 256-column blocks");
  SplashMoeParams p{};
  p.rows=rows; p.hidden=H; p.intermediate=I; p.experts=E; p.top_k=top_k;
  p.tile_rows=rows*uint32_t(top_k)<=E ? 8 : 32;
  p.normalize=uint32_t(normalize) | (a[1].dtype()==bfloat16 ? 2u : 0u) | (a[2].dtype()==bfloat16 ? 4u : 0u);
  for (int j=0;j<6;++j) {
    uint32_t format=formats[j], bits=format&255;
    if ((format&~511u) || bits<4 || bits>6) throw std::invalid_argument("unsupported grouped GGUF quant format");
    uint32_t N=j%3==2?H:I, K=j%3==2?I:H, experts=j<3?E:1;
    const auto &w=a[3+j*3], &sc=a[4+j*3], &b=a[5+j*3];
    Shape ws=j<3?Shape{int(experts),int(N),int(K*bits/32)}:Shape{int(N),int(K*bits/32)};
    Shape ss=j<3?Shape{int(experts),int(N),int(K/256*16)}:Shape{int(N),int(K/256*16)};
    Shape bs=j<3?Shape{int(experts),int(N),int(K/256*(bits==6?1:2))}:Shape{int(N),int(K/256*(bits==6?1:2))};
    if (w.shape()!=ws || w.dtype()!=uint32 || sc.shape()!=ss || sc.dtype()!=(bits==6?int8:uint8) || b.shape()!=bs || b.dtype()!=float16)
      throw std::invalid_argument("malformed grouped GGUF quantized projection "+std::to_string(j));
    p.formats[j]=format;
  }
  const char* register_flag=std::getenv("MLX_SPLASH_MOE_REGISTER");
  if(register_flag && std::strcmp(register_flag,"1")==0) {
    // Diagnostic-only format selector is captured with the graph. It narrows
    // the existing opt-in by routed down format; shared format stays guarded
    // independently by eval_gpu. No new format/shape/device is admitted.
    const char* down_bits=std::getenv("MLX_SPLASH_MOE_REGISTER_DOWN_BITS");
    const bool selected=!down_bits ||
        (std::strcmp(down_bits,"4")==0 && (p.formats[2]&255)==4) ||
        (std::strcmp(down_bits,"6")==0 && (p.formats[2]&255)==6);
    if(selected) {
      p.normalize |=16;
      const char* hybrid=std::getenv("MLX_SPLASH_MOE_REGISTER_HYBRID");
      if(hybrid && std::strcmp(hybrid,"1")==0) p.normalize |=64;
    }
  }
  const char* paired_flag=std::getenv("MLX_SPLASH_MOE_PAIRED_STAGE");
  if(paired_flag && std::strcmp(paired_flag,"1")==0) p.normalize |=32;
  const char* budget_flag=std::getenv("MLX_SPLASH_MOE_LOGICAL_BUDGET");
  if(budget_flag && std::strcmp(budget_flag,"1")==0) p.normalize |=128;
  const char* source_math=std::getenv("MLX_SPLASH_MOE_SOURCE_MATH");
  p.source_math=source_math_policy>=0 ? uint32_t(source_math_policy)
      : uint32_t(source_math && std::strcmp(source_math,"1")==0);
  p.normalize |= (a.size()==22 ? 8u : 0u) | uint32_t(probe)<<8 | uint32_t(profile)<<16;
  if (residual) {
    if (probe || profile || residual->dtype()!=bfloat16 || residual->shape()!=a[0].shape() ||
        H!=2048 || I!=512 || E!=256 || top_k!=8 || (p.normalize&6)!=6 ||
        !source_residual_available() ||
        (p.formats[0]&255)!=4 || (p.formats[1]&255)!=4 ||
        ((p.formats[2]&255)!=4 && (p.formats[2]&255)!=6) ||
        (p.formats[3]&255)!=4 || (p.formats[4]&255)!=4 ||
        ((p.formats[5]&255)!=4 && (p.formats[5]&255)!=6))
      throw std::invalid_argument("source MoE residual requires measured BF16 geometry/formats");
    p.source_math=1; p.source_residual=1;
    a.push_back(*residual);
  }
  if(source_routing) {
    p.source_math=1;
    if(!source_routing_inputs(a,p)) throw std::invalid_argument("source routing requires measured source expert formats");
  }
  const char* workspace_flag=std::getenv("MLX_SPLASH_MOE_WORKSPACE");
  const bool workspace=workspace_flag && std::strcmp(workspace_flag,"1")==0;
  auto shape=profile ? Shape{1} : probe ? Shape{int(rows),top_k+1,int(probe==4?H:I)} : a[0].shape();
  return array(shape,profile ? uint32 : probe || (p.normalize&6)==6 ? bfloat16 : float32,std::make_shared<GroupedMoe>(to_stream(Device::gpu),p,source_routing,workspace),std::move(a));
}

// Test-only view of the actual route/group kernels. It permits checking the
// maximum token/expert geometry without constructing any expert weights.
class RouteGroupProbe : public UnaryPrimitive {
 public:
  RouteGroupProbe(Stream s, SplashMoeParams p) : UnaryPrimitive(s), p_(p) {}
  DEFINE_NAME(RouteGroupProbe)
  void eval_cpu(const std::vector<array> &, array &) override {
    throw std::runtime_error("route/group probe requires Metal");
  }
  void eval_gpu(const std::vector<array> &a, array &out) override {
#ifdef MLX_NODE_METAL_ENABLED
    auto &s = stream(); auto &d = metal::device(s.device); auto &enc = metal::get_command_encoder(s);
    out.set_data(allocator::malloc(out.nbytes()));
    const size_t R = p_.rows * (p_.top_k + 1);
    const size_t T = std::min(p_.rows*p_.top_k,(p_.rows*p_.top_k+p_.tile_rows-1)/p_.tile_rows+p_.experts-1) +
                     (p_.rows+p_.tile_rows-1)/p_.tile_rows;
    const size_t tiles = 2*R, count = tiles+2*T,
                 routes=count+sizeof(SplashMoeTileCount)/sizeof(uint32_t), inverse=routes+T*p_.tile_rows;
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(d,"splash_moe","probe","splash_moe_route"));
    enc.set_input_array(a[0],0); enc.set_input_array(a[1],1);
    enc.set_output_array(out,2); enc.set_output_array(out,3,R*sizeof(uint32_t)); enc.set_bytes(p_,4);
    enc.set_input_array(a[0],5);
    enc.dispatch_threadgroups(MTL::Size(p_.rows,1,1),MTL::Size(64,1,1));
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(d,"splash_moe","probe","splash_moe_group"));
    enc.set_input_array(out,0); enc.set_output_array(out,1,tiles*sizeof(uint32_t));
    enc.set_output_array(out,2,count*sizeof(uint32_t)); enc.set_output_array(out,3,routes*sizeof(uint32_t));
    enc.set_output_array(out,4,inverse*sizeof(uint32_t)); enc.set_bytes(p_,5);
    enc.set_input_array(a[0],6);
    enc.dispatch_threadgroups(MTL::Size(1,1,1),MTL::Size(256,1,1));
#else
    throw std::runtime_error("route/group probe requires Metal");
#endif
  }
 private:
  SplashMoeParams p_;
};

array route_group_probe(const array &logits, const array &shared, int top_k, bool normalize) {
  if (!available() || logits.ndim()!=2 || (logits.dtype()!=float32 && logits.dtype()!=bfloat16) || (shared.dtype()!=float32 && shared.dtype()!=bfloat16) ||
      !logits.flags().row_contiguous || !shared.flags().row_contiguous || logits.shape(0)<1 || logits.shape(0)>2048 ||
      logits.shape(1)<top_k || logits.shape(1)>256 || top_k<1 || top_k>8 || shared.size()!=logits.shape(0))
    throw std::invalid_argument("invalid route/group probe input");
  SplashMoeParams p{}; p.rows=logits.shape(0); p.experts=logits.shape(1); p.top_k=top_k;
  p.hidden=1024; p.intermediate=512; // Fixed probe geometry for the indirect grids.
  p.normalize=uint32_t(normalize) | (logits.dtype()==bfloat16 ? 2u : 0u) | (shared.dtype()==bfloat16 ? 4u : 0u);
  p.tile_rows=p.rows*p.top_k<=p.experts?8:32;
  int R=p.rows*(p.top_k+1);
  int T=std::min(p.rows*p.top_k,(p.rows*p.top_k+p.tile_rows-1)/p.tile_rows+p.experts-1)+(p.rows+p.tile_rows-1)/p.tile_rows;
  return array({3*R+2*T+int(sizeof(SplashMoeTileCount)/sizeof(uint32_t))+T*int(p.tile_rows)},uint32,
      std::make_shared<RouteGroupProbe>(to_stream(Device::gpu),p),{logits,shared});
}
// The standalone observer invokes the identical encoder helper used by the
// production source-routed MoE. Uint32 bits pack scores, selected IDs, weights.
class SourceRoutingProbe final : public UnaryPrimitive {
 public:
  explicit SourceRoutingProbe(Stream s):UnaryPrimitive(s){}
  DEFINE_NAME(SourceRoutingProbe)
  bool is_equivalent(const Primitive&) const override { return true; }
  std::vector<Shape> output_shapes(const std::vector<array>& a) override {
    if(a.size()!=3 || !source_router_inputs(a)) throw std::invalid_argument("source routing probe replay contract");
    return {{8*256+2*8*9}};
  }
  void eval_cpu(const std::vector<array>&,array&) override {throw std::runtime_error("source routing probe requires Metal");}
  void eval_gpu(const std::vector<array>& inputs,array& out) override {
#ifdef MLX_NODE_METAL_ENABLED
    if(inputs.size()!=3 || !source_router_inputs(inputs)) throw std::invalid_argument("source routing probe replay contract");
    auto s=stream();auto& d=metal::device(s.device);auto& enc=metal::get_command_encoder(s);
    if(d.get_architecture_gen()<17) throw std::runtime_error("source routing requires M5");
    std::vector<array>a;
    for(const auto& input:inputs) {
      if(input.flags().row_contiguous && input.offset()%16==0) a.push_back(input);
      else {
        array c(input.shape(),input.dtype(),nullptr,{});c.set_data(allocator::malloc(c.nbytes()));
        copy_gpu(input,c,CopyType::General,s);enc.add_temporary(c);a.push_back(std::move(c));
      }
    }
    out.set_data(allocator::malloc(out.nbytes()));
    encode_source_routing(d,enc,a[0],a[1],a[2],out,0,out,8*256*sizeof(uint32_t),out,(8*256+8*9)*sizeof(uint32_t));
#else
    throw std::runtime_error("source routing probe requires Metal");
#endif
  }
};
} // namespace mlx::core::splash_moe

extern "C" bool mlx_splash_moe_available() { return mlx::core::splash_moe::available(); }

extern "C" mlx_array *mlx_test_splash_moe_route_group(mlx_array *logits, mlx_array *shared,
                                                       int top_k, bool normalize) {
  try {
    if (!logits || !shared) throw std::invalid_argument("null route/group probe input");
    auto out=mlx::core::splash_moe::route_group_probe(*reinterpret_cast<mlx::core::array *>(logits),
        *reinterpret_cast<mlx::core::array *>(shared),top_k,normalize);
    return reinterpret_cast<mlx_array *>(new mlx::core::array(std::move(out)));
  } catch (const std::exception &e) {
    std::cerr << "mlx_test_splash_moe_route_group: " << e.what() << std::endl;
    return nullptr;
  }
}

extern "C" mlx_array *mlx_splash_moe(mlx_array *const *inputs, size_t count,
                                     const uint32_t *formats, int top_k, bool normalize) {
  try {
    if (!inputs || !formats || (count!=21 && count!=22)) throw std::invalid_argument("grouped GGUF MoE requires 21 arrays and an optional row budget");
    std::vector<mlx::core::array> a; a.reserve(count);
    for (size_t i=0;i<count;++i) {
      if (!inputs[i]) throw std::invalid_argument("null grouped GGUF MoE input");
      a.push_back(*reinterpret_cast<mlx::core::array *>(inputs[i]));
    }
    return reinterpret_cast<mlx_array *>(new mlx::core::array(mlx::core::splash_moe::grouped(std::move(a),formats,top_k,normalize)));
  } catch (const std::exception &e) {
    std::cerr << "mlx_splash_moe: " << e.what() << std::endl;
    return nullptr;
  }
}

// Production uses a load-captured policy. The original entrypoint above
// remains environment-driven only for standalone operator diagnostics.
extern "C" mlx_array *mlx_splash_moe_with_policy(mlx_array *const *inputs,size_t count,
    const uint32_t *formats,int top_k,bool normalize,bool source_math) {
  try {
    if(!inputs||!formats||(count!=21&&count!=22))throw std::invalid_argument("invalid grouped MoE policy arguments");
    std::vector<mlx::core::array>a;a.reserve(count);
    for(size_t i=0;i<count;++i){
      if(!inputs[i])throw std::invalid_argument("null grouped MoE policy input");
      a.push_back(*reinterpret_cast<mlx::core::array*>(inputs[i]));
    }
    return reinterpret_cast<mlx_array*>(new mlx::core::array(
        mlx::core::splash_moe::grouped(std::move(a),formats,top_k,normalize,0,0,source_math?1:0)));
  }catch(const std::exception&e){std::cerr<<"mlx_splash_moe_with_policy: "<<e.what()<<std::endl;return nullptr;}
}

// Separate residual operand preserves the existing21/22-array ABI.
extern "C" mlx_array *mlx_splash_moe_with_residual(mlx_array *const *inputs, size_t count,
    const uint32_t *formats, int top_k, bool normalize, mlx_array *residual) {
  try {
    if(!inputs || !formats || !residual || (count!=21 && count!=22))
      throw std::invalid_argument("invalid residual-inclusive grouped MoE arguments");
    std::vector<mlx::core::array> a; a.reserve(count+1);
    for(size_t i=0;i<count;++i) {
      if(!inputs[i]) throw std::invalid_argument("null residual-inclusive grouped input");
      a.push_back(*reinterpret_cast<mlx::core::array *>(inputs[i]));
    }
    const auto& r=*reinterpret_cast<mlx::core::array *>(residual);
    return reinterpret_cast<mlx_array *>(new mlx::core::array(
        mlx::core::splash_moe::grouped(std::move(a),formats,top_k,normalize,0,0,1,&r)));
  } catch(const std::exception& e) {
    std::cerr << "mlx_splash_moe_with_residual: " << e.what() << std::endl;
    return nullptr;
  }
}

// Test-only stage view; the production primitive has no diagnostic output.
extern "C" mlx_array *mlx_test_splash_moe_stage(mlx_array *const *inputs, size_t count,
    const uint32_t *formats, int top_k, bool normalize, int stage) {
  try {
    if (!inputs || !formats || count!=21 || stage<1 || stage>4)
      throw std::invalid_argument("invalid grouped MoE stage probe");
    std::vector<mlx::core::array> a; a.reserve(count);
    for(size_t i=0;i<count;++i) {
      if (!inputs[i]) throw std::invalid_argument("null grouped MoE stage input");
      a.push_back(*reinterpret_cast<mlx::core::array *>(inputs[i]));
    }
    return reinterpret_cast<mlx_array *>(new mlx::core::array(
      mlx::core::splash_moe::grouped(std::move(a),formats,top_k,normalize,stage)));
  } catch(const std::exception &e) {
    std::cerr << "mlx_test_splash_moe_stage: " << e.what() << std::endl;
    return nullptr;
  }
}

// Research-only prefix timing. Every cutoff returns one initialized uint32;
// the encoder retains all submitted production work and its temporary owners.
extern "C" mlx_array *mlx_test_splash_moe_profile(mlx_array *const *inputs, size_t count,
    const uint32_t *formats, int top_k, bool normalize, int stage) {
  try {
    if (!inputs || !formats || (count!=21 && count!=22) || stage<1 || stage>6)
      throw std::invalid_argument("invalid grouped MoE profile input");
    std::vector<mlx::core::array> a; a.reserve(count);
    for(size_t i=0;i<count;++i) {
      if (!inputs[i]) throw std::invalid_argument("null grouped MoE profile input");
      a.push_back(*reinterpret_cast<mlx::core::array *>(inputs[i]));
    }
    return reinterpret_cast<mlx_array *>(new mlx::core::array(
      mlx::core::splash_moe::grouped(std::move(a),formats,top_k,normalize,0,stage)));
  } catch(const std::exception &e) {
    std::cerr << "mlx_test_splash_moe_profile: " << e.what() << std::endl;
    return nullptr;
  }
}

// Test-only raw packing observer. The producer and standalone oracle wrapper
// invoke the exact same TG8x32 pack function used by the fused gate/up tail.
namespace mlx::core::splash_moe {
class Sg8PackProbe final : public UnaryPrimitive {
 public:
  Sg8PackProbe(Stream stream,int bits):UnaryPrimitive(stream),bits_(bits){}
  DEFINE_NAME(Sg8PackProbe)
  bool is_equivalent(const Primitive& other) const override {
    return bits_==static_cast<const Sg8PackProbe&>(other).bits_;
  }
  std::vector<Shape> output_shapes(const std::vector<array>& a) override {
    return {{a[0].shape(1)*4+a[0].shape(1)/(bits_==6?16:32)*8}};
  }
  void eval_cpu(const std::vector<array>&,array&) override { throw std::runtime_error("SG8 packing probe requires Metal"); }
  void eval_gpu(const std::vector<array>& a,array& out) override {
#ifdef MLX_NODE_METAL_ENABLED
    if(a[0].shape(0)!=8 || a[0].dtype()!=bfloat16 || a[0].shape(1)%256)
      throw std::invalid_argument("SG8 packing probe replay shape");
    auto s=stream();auto& d=metal::device(s.device);auto& enc=metal::get_command_encoder(s);
    array x=a[0];
    if(!x.flags().row_contiguous || x.offset()%16) {
      array copy(x.shape(),x.dtype(),nullptr,{});copy.set_data(allocator::malloc(copy.nbytes()));
      copy_gpu(x,copy,CopyType::General,s);enc.add_temporary(copy);x=std::move(copy);
    }
    int K=x.shape(1);out.set_data(allocator::malloc(out.nbytes()));
    if(bits_==0) {
      if(a.size()!=3 || a[1].dtype()!=uint32 || a[1].shape()!=Shape{8} || a[2].dtype()!=uint32 || a[2].size()!=1)
        throw std::invalid_argument("grouped gather probe replay inputs");
      SplashMoeParams p{};p.rows=8;p.hidden=K;p.tile_rows=8;p.top_k=0;
      enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(d,"splash_moe","grouped","splash_moe_gather_sg8"));
      enc.set_input_array(x,0);enc.set_input_array(a[1],1);enc.set_input_array(a[1],2);enc.set_input_array(a[2],3);
      enc.set_output_array(out,4);enc.set_bytes(p,5);enc.set_output_array(out,6,size_t(K)*4*sizeof(uint32_t));
      enc.dispatch_threadgroups(MTL::Size(1,K/128,1),MTL::Size(128,1,1));return;
    }
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(d,"splash_moe","grouped","splash_moe_sg8_pack_probe"));
    enc.set_input_array(x,0);enc.set_output_array(out,1);enc.set_output_array(out,2,size_t(K)*4*sizeof(uint32_t));
    enc.set_bytes(K,3);enc.set_bytes(bits_,4);
    enc.dispatch_threadgroups(MTL::Size(K/32,1,1),MTL::Size(256,1,1));
#else
    throw std::runtime_error("SG8 packing probe requires Metal");
#endif
  }
 private:int bits_;
};
}
extern "C" mlx_array* mlx_test_splash_moe_sg8_pack(mlx_array* input,int bits) {
 try {
  using namespace mlx::core;
  if(!input || (bits!=4 && bits!=6) || !splash_moe::available()) return nullptr;
  auto x=*reinterpret_cast<array*>(input);
  if(x.ndim()!=2 || x.shape(0)!=8 || x.shape(1)<256 || x.shape(1)>8192 || x.shape(1)%256 || x.dtype()!=bfloat16) return nullptr;
  int K=x.shape(1);
  return reinterpret_cast<mlx_array*>(new array(Shape{K*4+K/(bits==6?16:32)*8},uint32,
      std::make_shared<splash_moe::Sg8PackProbe>(to_stream(Device::gpu),bits),{x}));
 } catch(const std::exception& e) {std::cerr<<"SG8 packing probe: "<<e.what()<<std::endl;return nullptr;}
}

extern "C" mlx_array* mlx_test_splash_moe_sg8_gather(mlx_array* input,mlx_array* routes,mlx_array* count) {
 try {
  using namespace mlx::core;
  if(!input || !routes || !count || !splash_moe::available()) return nullptr;
  auto x=*reinterpret_cast<array*>(input),r=*reinterpret_cast<array*>(routes),c=*reinterpret_cast<array*>(count);
  if(x.ndim()!=2 || x.shape(0)!=8 || x.shape(1)<256 || x.shape(1)>8192 || x.shape(1)%256 || x.dtype()!=bfloat16) return nullptr;
  int K=x.shape(1);
  return reinterpret_cast<mlx_array*>(new array(Shape{K*4+K/32*8},uint32,
      std::make_shared<splash_moe::Sg8PackProbe>(to_stream(Device::gpu),0),{x,r,c}));
 } catch(const std::exception& e) {std::cerr<<"SG8 gather probe: "<<e.what()<<std::endl;return nullptr;}
}

// Test-only real encoder checks. Run in a fresh process with
// MLX_MAX_MB_PER_BUFFER=1 and MLX_METAL_COMMAND_TRACE=1; no kernels read these
// bounded, uninitialized test allocations. Each synchronization emits the
// actual resource-byte trace to compare with the independently recorded size.
extern "C" bool mlx_test_sparse_input_budget_encoder() {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    using namespace mlx::core;
    auto &device = metal::device(Device::gpu);
    if (std::get<1>(device.get_max_ops_mb_per_buffer()) != 1)
      throw std::runtime_error("sparse budget test requires MLX_MAX_MB_PER_BUFFER=1 before process initialization");
    constexpr size_t Mi = size_t(1) << 20;
    constexpr size_t full = 4 * Mi;
    array a({int(full)}, uint8, nullptr, {});
    array b({int(full)}, uint8, nullptr, {});
    a.set_data(allocator::malloc(a.nbytes()));
    b.set_data(allocator::malloc(b.nbytes()));
    array alias({int(2 * Mi)}, uint8, nullptr, {});
    alias.copy_shared_buffer(a, {1}, alias.flags(), alias.size(), 16);
    const auto *buffer_a = static_cast<const MTL::Buffer *>(a.buffer().ptr());
    const auto *buffer_b = static_cast<const MTL::Buffer *>(b.buffer().ptr());
    const auto bytes_a = buffer_a->length(), bytes_b = buffer_b->length();
    auto expect = [](metal::CommandEncoder &enc, bool size, const char *step) {
      if (enc.needs_commit() != size ||
          std::strcmp(enc.commit_reason(), size ? "size" : "none") != 0)
        throw std::runtime_error(std::string("sparse budget threshold mismatch: ") + step);
    };
    auto flush = [](metal::CommandEncoder &enc, Stream stream, const char *name,
                    size_t bytes, size_t resources) {
      std::fprintf(stderr,
          "[sparse-budget-case] {\"name\":\"%s\",\"stream\":%d,"
          "\"expectedResourceBytes\":%zu,\"expectedResources\":%zu}\n",
          name, stream.index, bytes, resources);
      enc.synchronize();
    };
    auto run = [&](const char *name, size_t resources, auto check) {
      const auto stream = new_stream(Device::gpu);
      auto &enc = metal::get_command_encoder(stream);
      // Keep the whole actual allocations alive through command completion,
      // independently of the scheduling estimate.
      enc.add_temporary(a);
      if (resources == 2) enc.add_temporary(b);
      try {
        check(enc);
        flush(enc, stream, name, bytes_a + (resources == 2 ? bytes_b : 0), resources);
      } catch (...) {
        try { enc.synchronize(); } catch (...) {}
        throw;
      }
    };
    run("first", 1, [&](auto &enc) {
      enc.set_input_array_with_budget(a, 0, Mi);
      expect(enc, false, "first bounded binding");
    });
    run("repeated_budget", 1, [&](auto &enc) {
      enc.set_input_array_with_budget(a, 0, Mi);
      expect(enc, false, "before repeated budget");
      enc.set_input_array_with_budget(a, 1, 0);
      expect(enc, true, "repeated budget restores full");
      enc.set_input_array_with_budget(a, 2, 0);
      expect(enc, true, "third budget cannot discount again");
    });
    run("ordinary_after_budget", 1, [&](auto &enc) {
      enc.set_input_array_with_budget(a, 0, Mi);
      enc.set_input_array(a, 1);
      expect(enc, true, "ordinary input restores full");
    });
    run("budget_after_ordinary", 1, [&](auto &enc) {
      enc.set_input_array(a, 0);
      enc.set_input_array_with_budget(a, 1, 0);
      expect(enc, true, "already-bound ordinary input cannot be discounted");
    });
    run("raw_after_budget", 1, [&](auto &enc) {
      enc.set_input_array_with_budget(a, 0, Mi);
      enc.set_buffer(buffer_a, 1);
      expect(enc, true, "raw buffer restores full");
    });
    run("output_after_budget", 1, [&](auto &enc) {
      enc.set_input_array_with_budget(a, 0, Mi);
      enc.set_output_array(a, 1);
      expect(enc, true, "output binding restores full");
    });
    run("output_alias_registration", 1, [&](auto &enc) {
      enc.set_input_array_with_budget(a, 0, Mi);
      enc.register_output_array(alias);
      expect(enc, true, "aliased output registration restores full");
    });
    run("budget_after_output_alias", 1, [&](auto &enc) {
      enc.register_output_array(alias);
      enc.set_input_array_with_budget(a, 0, 0);
      expect(enc, true, "a previously registered output prevents input discount");
    });
    run("shared_view_budget", 1, [&](auto &enc) {
      enc.set_input_array_with_budget(a, 0, Mi);
      enc.set_input_array_with_budget(alias, 1, 0);
      expect(enc, true, "offset shared view restores first full charge");
    });
    run("zero", 1, [&](auto &enc) {
      enc.set_input_array_with_budget(a, 0, 0);
      expect(enc, false, "zero estimate does not wrap");
      enc.set_input_array(alias, 1);
      expect(enc, true, "zero estimate still tracks whole shared allocation");
    });
    run("oversized", 1, [&](auto &enc) {
      enc.set_input_array_with_budget(a, 0, std::numeric_limits<size_t>::max());
      expect(enc, true, "oversized estimate keeps full without overflow");
    });
    run("full", 1, [&](auto &enc) {
      enc.set_input_array_with_budget(a, 0, full);
      expect(enc, true, "full prefill estimate unchanged");
    });
    run("independent_buffers", 2, [&](auto &enc) {
      enc.set_input_array_with_budget(a, 0, Mi);
      expect(enc, false, "first independent buffer");
      enc.set_input_array_with_budget(b, 1, Mi);
      expect(enc, true, "independent estimates add");
    });
    // Encoder and command boundaries reset different scopes: a new encoder
    // charges its first use again, while a new command starts from zero.
    const auto stream = new_stream(Device::gpu);
    auto &enc = metal::get_command_encoder(stream);
    try {
      enc.add_temporary(a);
      enc.set_input_array_with_budget(a, 0, Mi);
      expect(enc, false, "reset case first encoder");
      enc.end_encoding();
      enc.set_input_array_with_budget(a, 0, Mi);
      expect(enc, true, "encoder reset retains aggregate command estimate");
      flush(enc, stream, "encoder_reset", bytes_a, 1);
      enc.add_temporary(a);
      enc.set_input_array_with_budget(a, 0, Mi);
      expect(enc, false, "new command estimate starts from zero");
      flush(enc, stream, "commit_reset", bytes_a, 1);
    } catch (...) {
      try { enc.synchronize(); } catch (...) {}
      throw;
    }
    return true;
  } catch (const std::exception &error) {
    std::cerr << "sparse budget encoder test: " << error.what() << '\n';
    return false;
  }
#else
  return false;
#endif
}

// Test-only direct view of the production combine kernel. Every output is
// initialized, and no source/expert arithmetic is replaced by this adapter.
namespace mlx::core::splash_moe {
class SourceResidualCombineProbe : public UnaryPrimitive {
 public:
  explicit SourceResidualCombineProbe(Stream s):UnaryPrimitive(s){}
  DEFINE_NAME(SourceResidualCombineProbe)
  std::vector<Shape> output_shapes(const std::vector<array>& a) override {return {a[3].shape()};}
  void eval_cpu(const std::vector<array>&,array&) override {throw std::runtime_error("Metal required");}
  void eval_gpu(const std::vector<array>& a,array& out) override {
#ifdef MLX_NODE_METAL_ENABLED
    if(a.size()!=5 || a[3].ndim()!=2 || a[3].dtype()!=bfloat16 ||
       a[0].dtype()!=bfloat16 || a[1].dtype()!=uint32 || a[2].dtype()!=float32 ||
       a[4].dtype()!=int32 || a[4].size()!=1 || a[3].shape(0)<1 || a[3].shape(0)>33 ||
       a[3].shape(1)!=2048 || a[0].shape()!=Shape{a[3].shape(0)*9,2048} ||
       a[1].shape()!=Shape{a[3].shape(0),9} || a[2].shape()!=a[1].shape())
      throw std::invalid_argument("source residual combine probe shape/dtype changed");
    for (const auto& input:a)
      if (!input.flags().row_contiguous)
        throw std::invalid_argument("source residual combine probe requires contiguous inputs");
    SplashMoeParams p{};p.rows=a[3].shape(0);p.hidden=2048;p.top_k=8;
    p.normalize=1|2|4|8;p.source_math=p.rows==8;p.source_residual=1;
    out.set_data(allocator::malloc(out.nbytes()));
    auto& s=stream();auto& d=metal::device(s.device);auto& enc=metal::get_command_encoder(s);
    enc.set_compute_pipeline_state(fast::paged::get_prebuilt_kernel(d,"splash_moe","probe","splash_moe_combine"));
    for(int i=0;i<3;++i)enc.set_input_array(a[i],i);
    enc.set_output_array(out,3);enc.set_bytes(p,4);enc.set_input_array(a[4],5);enc.set_input_array(a[3],6);
    enc.dispatch_threadgroups(MTL::Size(p.rows,8,1),MTL::Size(256,1,1));
#else
    throw std::runtime_error("Metal required");
#endif
  }
};
}
extern "C" mlx_array* mlx_test_splash_moe_source_residual_combine(mlx_array* experts,
    mlx_array* inverse,mlx_array* weights,mlx_array* residual,mlx_array* budget) {
  try {
    using namespace mlx::core;
    if(!experts||!inverse||!weights||!residual||!budget||!splash_moe::available())return nullptr;
    std::vector<array>a;
    for(auto* p:{experts,inverse,weights,residual,budget})a.push_back(*reinterpret_cast<array*>(p));
    auto shape=a[3].shape();
    return reinterpret_cast<mlx_array*>(new array(shape,bfloat16,
        std::make_shared<splash_moe::SourceResidualCombineProbe>(to_stream(Device::gpu)),std::move(a)));
  }catch(const std::exception& e){std::cerr<<"source residual combine probe: "<<e.what()<<std::endl;return nullptr;}
}

extern "C" bool mlx_splash_moe_source_residual_available(){return mlx::core::splash_moe::source_residual_available();}

extern "C" mlx_array* mlx_splash_moe_with_router(mlx_array* const* inputs,size_t count,
    const uint32_t* formats,int top_k,bool normalize,mlx_array* residual) {
  try {
    if(!inputs || !formats || (count!=21 && count!=22)) throw std::invalid_argument("invalid source-routed MoE arguments");
    std::vector<mlx::core::array>a;a.reserve(count+1);
    for(size_t i=0;i<count;++i) {
      if(!inputs[i]) throw std::invalid_argument("null source-routed MoE input");
      a.push_back(*reinterpret_cast<mlx::core::array*>(inputs[i]));
    }
    const auto* r=residual?reinterpret_cast<mlx::core::array*>(residual):nullptr;
    return reinterpret_cast<mlx_array*>(new mlx::core::array(
        mlx::core::splash_moe::grouped(std::move(a),formats,top_k,normalize,0,0,1,r,true)));
  }catch(const std::exception&e){std::cerr<<"mlx_splash_moe_with_router: "<<e.what()<<std::endl;return nullptr;}
}
extern "C" mlx_array* mlx_test_splash_moe_source_routing(mlx_array* input,mlx_array* router,mlx_array* shared) {
  try {
    using namespace mlx::core;
    if(!input || !router || !shared || !splash_moe::source_residual_available()) return nullptr;
    std::vector<array>a{*reinterpret_cast<array*>(input),*reinterpret_cast<array*>(router),*reinterpret_cast<array*>(shared)};
    if(!splash_moe::source_router_inputs(a)) return nullptr;
    return reinterpret_cast<mlx_array*>(new array(Shape{8*256+2*8*9},uint32,
        std::make_shared<splash_moe::SourceRoutingProbe>(to_stream(Device::gpu)),std::move(a)));
  }catch(const std::exception&e){std::cerr<<"source routing probe: "<<e.what()<<std::endl;return nullptr;}
}
