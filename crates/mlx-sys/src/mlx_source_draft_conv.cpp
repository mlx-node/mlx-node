// Source arithmetic for the measured DFlash2 companion, separate from the
// native per-operation BF16 convolution. See THIRD_PARTY_NOTICES.
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
namespace mlx::core::source_draft_conv {
bool available() {
#ifdef MLX_NODE_METAL_ENABLED
  return !detail::in_grad_tracing() && default_device()==Device::gpu &&
      is_available(Device::gpu) && metal::device(Device::gpu).get_architecture_gen()>=17 &&
      mlx_gpu_core_count()==40;
#else
  return false;
#endif
}
bool shapes(const std::vector<array>& a,int side) {
  if(side<0 || side>1 || a.size()!=size_t(side?4:3) ||
      a[0].shape()!=Shape{1,8,2048} || a[1].shape()!=Shape{1,8,2,2,128} ||
      a[2].shape()!=Shape{2,2,2048} || (side && a[3].shape()!=a[0].shape()))return false;
  for(const auto& x:a)if(x.dtype()!=bfloat16)return false;
  return true;
}
bool layouts(const std::vector<array>& a) {
  for(const auto& x:a) {
    if(x.strides(-1)!=1)return false;
    for(int axis=0;axis<x.ndim();++axis)
      if(x.shape(axis)>1 && x.strides(axis)<=0)return false;
  }
  return true;
}
#ifdef MLX_NODE_METAL_ENABLED
bool extent(const array& a) {
  size_t elements=1;
  for(int axis=0;axis<a.ndim();++axis) {
    if(a.shape(axis)<=1)continue;
    const size_t stride=a.strides(axis),count=size_t(a.shape(axis)-1);
    if(stride>(std::numeric_limits<size_t>::max()-elements)/count)return false;
    elements+=stride*count;
  }
  return a.offset()%a.itemsize()==0 && a.offset()<=a.buffer_size() &&
      elements<=(a.buffer_size()-a.offset())/a.itemsize();
}
struct Params {uint64_t x_row,d_row,d_side,d_tap,b_side,b_tap,residual_row;uint32_t side,pad;};
static_assert(sizeof(Params)==64);
#endif
class SourceDraftConv final:public fast::Custom {
 public:
  SourceDraftConv(Stream s,int side):Custom(s,[](std::vector<array>)->std::vector<array>{
    throw std::runtime_error("source draft convolution does not support transforms");}),side_(side){}
  DEFINE_NAME(SourceDraftConv)
  bool is_equivalent(const Primitive& rhs)const override{return side_==static_cast<const SourceDraftConv&>(rhs).side_;}
  std::vector<Shape> output_shapes(const std::vector<array>& a)override {
    if(!shapes(a,side_))throw std::invalid_argument("source draft convolution replay shape/dtype contract");
    return {{1,8,2048}};
  }
  void eval_cpu(const std::vector<array>&,std::vector<array>&)override {
    throw std::runtime_error("source draft convolution requires measured Metal device");
  }
  void eval_gpu(const std::vector<array>& a,std::vector<array>& outputs)override {
#ifdef MLX_NODE_METAL_ENABLED
    if(!shapes(a,side_)||!layouts(a)||!available())throw std::invalid_argument("source draft convolution replay contract/device");
    for(const auto& x:a)if(!extent(x))throw std::invalid_argument("source draft convolution view exceeds owner extent");
    auto s=stream();auto& d=metal::device(s.device);auto& enc=metal::get_command_encoder(s);
    CompileOptions options;options.math_mode=MathMode::Fast;
    auto* library=d.get_library("mlx_source_draft_conv_m8_fast_v1",options,[]{return std::string(
#include "metal/common/source_draft_conv.metal.inc"
    );});
    auto* pipeline=d.get_kernel("source_draft_conv_m8",library);
    if(pipeline->threadExecutionWidth()!=32 || pipeline->maxTotalThreadsPerThreadgroup()<256)
      throw std::runtime_error("source draft convolution launch unsupported");
    auto& out=outputs[0];out.set_data(allocator::malloc(out.nbytes()));
    enc.set_compute_pipeline_state(pipeline);
    for(int j=0;j<3;++j)enc.set_input_array(a[j],j);
    enc.set_input_array(a[side_?3:0],3);enc.set_output_array(out,4);
    const Params p{uint64_t(a[0].strides(1)),uint64_t(a[1].strides(1)),uint64_t(a[1].strides(2)),
      uint64_t(a[1].strides(3)),uint64_t(a[2].strides(0)),uint64_t(a[2].strides(1)),
      uint64_t(a[side_?3:0].strides(1)),uint32_t(side_),0};
    enc.set_bytes(p,5);enc.dispatch_threadgroups(MTL::Size(64,1,1),MTL::Size(256,1,1));
    bridge_testing::record(side_?"source_draft_conv_finish":"source_draft_conv_prepare");
#else
    throw std::runtime_error("source draft convolution requires Metal");
#endif
  }
 private:int side_;
};
}
extern "C" bool mlx_source_dflash2_conv_available(){try{return mlx::core::source_draft_conv::available();}catch(...){return false;}}
extern "C" mlx_array* mlx_source_dflash2_conv(mlx_array* x,mlx_array* dynamic,mlx_array* base,int side,mlx_array* residual) {
  try {
    using namespace mlx::core;
    if(!x||!dynamic||!base||side<0||side>1||bool(residual)!=(side==1)||!source_draft_conv::available())return nullptr;
    std::vector<array>a{*reinterpret_cast<array*>(x),*reinterpret_cast<array*>(dynamic),*reinterpret_cast<array*>(base)};
    if(residual)a.push_back(*reinterpret_cast<array*>(residual));
    if(!source_draft_conv::shapes(a,side)||!source_draft_conv::layouts(a))return nullptr;
    auto out=array({1,8,2048},bfloat16,std::make_shared<source_draft_conv::SourceDraftConv>(to_stream(Device::gpu),side),std::move(a));
    return reinterpret_cast<mlx_array*>(new array(std::move(out)));
  }catch(const std::exception& e){std::cerr<<"source draft convolution: "<<e.what()<<'\n';return nullptr;}
}
