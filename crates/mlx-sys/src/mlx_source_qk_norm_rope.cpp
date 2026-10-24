// Strict source Q/K RMS+RoPE diagnostic. Normal model paths remain separate.
#include "mlx_common.h"
#include "mlx/allocator.h"
#include "mlx/fast_primitives.h"
#include "mlx/transforms_impl.h"
#include "mlx_test_counters.h"
#include <limits>
#include <memory>
#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/metal/device.h"
#endif
extern "C" int mlx_gpu_core_count();
namespace mlx::core::source_qk {
bool available() {
#ifdef MLX_NODE_METAL_ENABLED
  return !detail::in_grad_tracing() && default_device()==Device::gpu &&
      is_available(Device::gpu) && metal::device(Device::gpu).get_architecture_gen()>=17 &&
      mlx_gpu_core_count()==40;
#else
  return false;
#endif
}
bool shapes(const std::vector<array>& a) {
  return a.size()==6 && a[0].shape()==Shape{1,8,16,256} && a[1].shape()==Shape{1,8,2,256} &&
      a[0].dtype()==bfloat16 && a[1].dtype()==bfloat16 &&
      a[2].shape()==Shape{256} && a[3].shape()==Shape{256} &&
      a[2].dtype()==float32 && a[3].dtype()==float32 &&
      a[4].shape()==Shape{32} && a[4].dtype()==float32 &&
      a[5].shape()==Shape{1} && a[5].dtype()==int32;
}
bool layouts(const std::vector<array>& a) {
  for(int j=0;j<2;++j)
    if(a[j].strides(1)<=0 || a[j].strides(2)<=0 || a[j].strides(3)!=1) return false;
  for(int j=2;j<5;++j) if(a[j].strides(0)<=0) return false;
  return true;
}
#ifdef MLX_NODE_METAL_ENABLED
// Metadata-only extent validation after inputs materialize. Positive strided
// views can contain holes; verify their last scalar load lies in the owner.
bool extent(const array& a) {
  size_t elements=1;
  for(int axis=0;axis<a.ndim();++axis) {
    if(a.shape(axis)<=1)continue;
    if(a.strides(axis)<=0)return false;
    const size_t stride=a.strides(axis),count=size_t(a.shape(axis)-1);
    if(stride>(std::numeric_limits<size_t>::max()-elements)/count)return false;
    elements+=stride*count;
  }
  return a.offset()%a.itemsize()==0 && a.offset()<=a.buffer_size() &&
      elements<=(a.buffer_size()-a.offset())/a.itemsize();
}
struct Params {uint64_t q_row,q_head,k_row,k_head,wq,wk,frequency;};
static_assert(sizeof(Params)==56);
#endif
class SourceQk final:public fast::Custom {
 public:
  explicit SourceQk(Stream s):Custom(s,[](std::vector<array>)->std::vector<array>{
    throw std::runtime_error("source QK diagnostic does not support transforms");}){}
  DEFINE_NAME(SourceQk)
  bool is_equivalent(const Primitive&)const override{return true;}
  std::vector<Shape> output_shapes(const std::vector<array>& a)override {
    if(!shapes(a))throw std::invalid_argument("source QK replay shape/dtype contract");
    return {{1,16,8,256},{1,2,8,256}};
  }
  void eval_cpu(const std::vector<array>&,std::vector<array>&)override {
    throw std::runtime_error("source QK diagnostic requires measured Metal device");
  }
  void eval_gpu(const std::vector<array>& a,std::vector<array>& out)override {
#ifdef MLX_NODE_METAL_ENABLED
    if(!shapes(a)||!layouts(a)||!available())throw std::invalid_argument("source QK replay contract/device");
    for(const auto& x:a)if(!extent(x))throw std::invalid_argument("source QK view exceeds buffer extent");
    auto s=stream();auto& d=metal::device(s.device);auto& enc=metal::get_command_encoder(s);
    CompileOptions options;options.math_mode=MathMode::Fast;
    auto* library=d.get_library("mlx_source_qk_norm_rope_m8_fast_v1",options,[]{return std::string(
#include "metal/common/source_qk_norm_rope.metal.inc"
    );});
    auto* pipeline=d.get_kernel("source_qk_norm_rope_m8",library);
    if(pipeline->threadExecutionWidth()!=32 || pipeline->maxTotalThreadsPerThreadgroup()<256 ||
        pipeline->staticThreadgroupMemoryLength()>d.mtl_device()->maxThreadgroupMemoryLength())
      throw std::runtime_error("source QK threadgroup unsupported");
    for(auto& x:out)x.set_data(allocator::malloc(x.nbytes()));
    enc.set_compute_pipeline_state(pipeline);
    for(int i=0;i<6;++i)enc.set_input_array(a[i],i);
    enc.set_output_array(out[0],6);enc.set_output_array(out[1],7);
    const Params p{uint64_t(a[0].strides(1)),uint64_t(a[0].strides(2)),
      uint64_t(a[1].strides(1)),uint64_t(a[1].strides(2)),uint64_t(a[2].strides(0)),
      uint64_t(a[3].strides(0)),uint64_t(a[4].strides(0))};
    enc.set_bytes(p,8);enc.dispatch_threadgroups(MTL::Size(8*18,1,1),MTL::Size(256,1,1));
    bridge_testing::record("source_qk_norm_rope_m8");
#else
    throw std::runtime_error("source QK diagnostic requires Metal");
#endif
  }
};
}
extern "C" bool mlx_source_qk_norm_rope_available(){try{return mlx::core::source_qk::available();}catch(...){return false;}}
extern "C" bool mlx_source_qk_norm_rope(mlx_array* q,mlx_array* k,mlx_array* wq,mlx_array* wk,
    mlx_array* inverse_frequencies,mlx_array* offsets,mlx_array** out_q,mlx_array** out_k) {
  if(out_q)*out_q=nullptr;if(out_k)*out_k=nullptr;
  try {
    using namespace mlx::core;
    if(!q||!k||!wq||!wk||!inverse_frequencies||!offsets||!out_q||!out_k||out_q==out_k||!source_qk::available())return false;
    std::vector<array>a{*reinterpret_cast<array*>(q),*reinterpret_cast<array*>(k),*reinterpret_cast<array*>(wq),
      *reinterpret_cast<array*>(wk),*reinterpret_cast<array*>(inverse_frequencies),*reinterpret_cast<array*>(offsets)};
    if(!source_qk::shapes(a)||!source_qk::layouts(a))return false;
    auto outputs=array::make_arrays({Shape{1,16,8,256},Shape{1,2,8,256}},
        {bfloat16,bfloat16},std::make_shared<source_qk::SourceQk>(to_stream(Device::gpu)),std::move(a));
    auto oq=std::make_unique<array>(std::move(outputs[0]));auto ok=std::make_unique<array>(std::move(outputs[1]));
    *out_q=reinterpret_cast<mlx_array*>(oq.release());*out_k=reinterpret_cast<mlx_array*>(ok.release());return true;
  }catch(const std::exception&e){std::cerr<<"source QK: "<<e.what()<<'\n';return false;}
}
