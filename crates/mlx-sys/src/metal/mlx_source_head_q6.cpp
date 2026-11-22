// Strict source Q6 vocabulary-head diagnostic. See THIRD_PARTY_NOTICES.
#include "mlx_common.h"
#include "mlx/allocator.h"
#include "mlx/primitives.h"
#include "mlx/transforms_impl.h"
#include "mlx_test_counters.h"
#include <limits>
#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#endif
extern "C" int mlx_gpu_core_count();
namespace mlx::core::source_head_q6 {
constexpr int K=2048,N=248320,Rows=8;
bool available() {
#ifdef MLX_NODE_METAL_ENABLED
  return !detail::in_grad_tracing() && default_device()==Device::gpu &&
      is_available(Device::gpu) && metal::device(Device::gpu).get_architecture_gen()>=17 &&
      mlx_gpu_core_count()==40;
#else
  return false;
#endif
}
// A lazy view does not have an owner allocation yet. Check its metadata span
// without multiplying until bounds are proved, then recheck the materialized
// owner in eval_gpu before copies, allocation or vector-buffer bindings.
bool byte_span(const array& a,size_t& bytes) {
  size_t elements=1;
  for(int axis=0;axis<a.ndim();++axis) {
    if(a.shape(axis)<=1)continue;
    if(a.strides(axis)<=0)return false;
    const size_t stride=a.strides(axis),count=size_t(a.shape(axis)-1);
    if(stride>(std::numeric_limits<size_t>::max()-elements)/count)return false;
    elements+=stride*count;
  }
  if(elements>std::numeric_limits<size_t>::max()/a.itemsize())return false;
  bytes=elements*a.itemsize();
  return a.offset()%a.itemsize()==0 &&
      a.offset()<=std::numeric_limits<size_t>::max()-bytes;
}
bool supported(const std::vector<array>& a) {
  const bool shape=a.size()==4 && (a[0].shape()==Shape{Rows,K} || a[0].shape()==Shape{1,Rows,K}) &&
      a[0].dtype()==bfloat16 && a[1].shape()==Shape{N,K*6/32} && a[1].dtype()==uint32 &&
      a[2].shape()==Shape{N,K/16} && a[2].dtype()==int8 &&
      a[3].shape()==Shape{N,K/256} && a[3].dtype()==float16;
  if(!shape)return false;
  // These are opaque flat T64 streams, not logical row-major [N,K] weights.
  // Never reinterpret a padded/strided packed view as another T64 layout.
  for(unsigned i=1;i<4;++i)if(!a[i].flags().row_contiguous)return false;
  for(const auto& input:a){size_t bytes;if(!byte_span(input,bytes))return false;}
  return true;
}
Shape output_shape(const array& x) {auto shape=x.shape();shape.back()=N;return shape;}
#ifdef MLX_NODE_METAL_ENABLED
MTL::ComputePipelineState* kernel(metal::Device& d) {
  CompileOptions options;options.math_mode=MathMode::Fast;
  auto* lib=d.get_library("mlx_source_head_q6_248320x2048_fast_v1",options,[]{return std::string(
#include "metal/common/source_head_q6.metal.inc"
  );});
  auto* p=d.get_kernel("source_head_q6_m8_f32",lib);
  if(p->threadExecutionWidth()!=32 || p->maxTotalThreadsPerThreadgroup()<64 ||
      p->staticThreadgroupMemoryLength()>d.mtl_device()->maxThreadgroupMemoryLength())
    throw std::runtime_error("source Q6 head threadgroup unsupported");
  return p;
}
std::vector<array> normalized(const std::vector<array>& inputs,Stream s,metal::CommandEncoder& enc) {
  std::vector<array> a;a.reserve(inputs.size());
  for(const auto& input:inputs) {
    // Lazy views acquire their actual offsets at eval; check here, not only
    // when the primitive is selected. General copy preserves logical striding.
    if(input.flags().row_contiguous && input.offset()%16==0)a.push_back(input);
    else {
      array copy(input.shape(),input.dtype(),nullptr,{});copy.set_data(allocator::malloc(copy.nbytes()));
      copy_gpu(input,copy,input.flags().row_contiguous?CopyType::Vector:CopyType::General,s);
      enc.add_temporary(copy);a.push_back(std::move(copy));
    }
  }
  return a;
}
#endif
class SourceHeadQ6 final:public UnaryPrimitive {
 public:
  explicit SourceHeadQ6(Stream s):UnaryPrimitive(s){}
  DEFINE_NAME(SourceHeadQ6)
  bool is_equivalent(const Primitive&)const override{return true;}
  std::vector<Shape> output_shapes(const std::vector<array>& a)override {
    if(!supported(a))throw std::invalid_argument("source Q6 head replay contract");
    return {output_shape(a[0])};
  }
  void eval_cpu(const std::vector<array>&,array&)override {
    throw std::runtime_error("source Q6 head requires measured Metal device");
  }
  void eval_gpu(const std::vector<array>& inputs,array& out)override {
#ifdef MLX_NODE_METAL_ENABLED
    if(!supported(inputs)||!available())throw std::invalid_argument("source Q6 head replay contract/device");
    for(const auto& input:inputs) {
      size_t bytes;
      if(!byte_span(input,bytes) || input.offset()>input.buffer_size() ||
          bytes>input.buffer_size()-input.offset())
        throw std::invalid_argument("source Q6 head view exceeds buffer extent");
    }
    auto s=stream();auto& d=metal::device(s.device);auto& enc=metal::get_command_encoder(s);
    auto a=normalized(inputs,s,enc);
    out.set_data(allocator::malloc(out.nbytes()));
    enc.set_compute_pipeline_state(kernel(d));
    for(unsigned i=0;i<4;++i)enc.set_input_array(a[i],i);
    enc.set_output_array(out,4);
    enc.dispatch_threadgroups(MTL::Size(N/64,1,1),MTL::Size(64,1,1));
    bridge_testing::record("source_head_q6_m8_f32");
#else
    throw std::runtime_error("source Q6 head requires Metal");
#endif
  }
};
}
extern "C" bool mlx_source_head_q6_available(){try{return mlx::core::source_head_q6::available();}catch(...){return false;}}
extern "C" mlx_array* mlx_source_head_q6(mlx_array* x,mlx_array* w,mlx_array* scales,mlx_array* bases) {
  try {
    using namespace mlx::core;
    if(!x||!w||!scales||!bases||!source_head_q6::available())return nullptr;
    std::vector<array> a{*reinterpret_cast<array*>(x),*reinterpret_cast<array*>(w),
        *reinterpret_cast<array*>(scales),*reinterpret_cast<array*>(bases)};
    if(!source_head_q6::supported(a))return nullptr;
    auto shape=source_head_q6::output_shape(a[0]);
    return reinterpret_cast<mlx_array*>(new array(std::move(shape),float32,
        std::make_shared<source_head_q6::SourceHeadQ6>(to_stream(Device::gpu)),std::move(a)));
  }catch(const std::exception& e){std::cerr<<"source Q6 head: "<<e.what()<<'\n';return nullptr;}
}
