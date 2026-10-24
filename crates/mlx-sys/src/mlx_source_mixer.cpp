// Exact-width source Q4 mixer diagnostic; normal models do not call this.
#include "mlx_common.h"
#include "mlx/allocator.h"
#include "mlx/primitives.h"
#include "mlx/transforms_impl.h"
#include "mlx_test_counters.h"
#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#endif
extern "C" int mlx_gpu_core_count();
namespace mlx::core::source_mixer {
constexpr int K=4096,N=2048,Rows=8,Splits=8,TableWords=K*4,SumWords=3072,Counters=32;
bool available() {
#ifdef MLX_NODE_METAL_ENABLED
  return !detail::in_grad_tracing() && default_device()==Device::gpu &&
      is_available(Device::gpu) && metal::device(Device::gpu).get_architecture_gen()>=17 &&
      mlx_gpu_core_count()==40;
#else
  return false;
#endif
}
bool supported(const std::vector<array>& a,bool residual) {
  return a.size()==size_t(4+residual) && a[0].shape()==Shape{Rows,K} && a[0].dtype()==bfloat16 &&
      a[1].shape()==Shape{N,K/8} && a[1].dtype()==uint32 &&
      a[2].shape()==Shape{N,K/16} && a[2].dtype()==uint8 &&
      a[3].shape()==Shape{N,K/128} && a[3].dtype()==float16 &&
      (!residual || (a[4].shape()==Shape{Rows,N} && a[4].dtype()==bfloat16));
}
#ifdef MLX_NODE_METAL_ENABLED
MTL::ComputePipelineState* kernel(metal::Device& d,const char* name) {
  CompileOptions options;options.math_mode=MathMode::Fast;
  auto* lib=d.get_library("mlx_source_mixer_q4_2048x4096_fast_v1",options,[]{return std::string(
#include "metal/common/source_mixer.metal.inc"
  );});
  auto* p=d.get_kernel(name,lib);
  if(p->threadExecutionWidth()!=32 || p->maxTotalThreadsPerThreadgroup()<128 ||
     p->staticThreadgroupMemoryLength()>d.mtl_device()->maxThreadgroupMemoryLength())
    throw std::runtime_error("source mixer threadgroup unsupported");
  return p;
}
std::vector<array> contiguous(const std::vector<array>& inputs,Stream s,metal::CommandEncoder& enc) {
  std::vector<array>a;a.reserve(inputs.size());
  for(const auto& input:inputs) {
    if(input.flags().row_contiguous && input.offset()%16==0)a.push_back(input);
    else {
      array copy(input.shape(),input.dtype(),nullptr,{});copy.set_data(allocator::malloc(copy.nbytes()));
      copy_gpu(input,copy,CopyType::General,s);enc.add_temporary(copy);a.push_back(std::move(copy));
    }
  }
  return a;
}
void prepare(metal::Device& d,metal::CommandEncoder& enc,const array& x,array& table,size_t table_offset,
    array& sums,size_t sums_offset,array& counters,size_t counter_offset) {
  enc.set_compute_pipeline_state(kernel(d,"source_mixer_q4_prepare"));
  enc.set_input_array(x,0);enc.set_output_array(table,1,table_offset);enc.set_output_array(sums,2,sums_offset);
  enc.set_output_array(counters,3,counter_offset);
  enc.dispatch_threadgroups(MTL::Size(K/32,1,1),MTL::Size(128,1,1));
  bridge_testing::record("source_mixer_q4_prepare");
}
#endif
class SourceMixer final:public UnaryPrimitive {
 public:
  SourceMixer(Stream stream,bool residual):UnaryPrimitive(stream),residual_(residual){}
  DEFINE_NAME(SourceMixer)
  bool is_equivalent(const Primitive& other)const override{return residual_==static_cast<const SourceMixer&>(other).residual_;}
  std::vector<Shape> output_shapes(const std::vector<array>& a)override {
    if(!supported(a,residual_))throw std::invalid_argument("source mixer replay contract");
    return {{Rows,N}};
  }
  void eval_cpu(const std::vector<array>&,array&)override{throw std::runtime_error("source mixer requires measured Metal device");}
  void eval_gpu(const std::vector<array>& inputs,array& out)override {
#ifdef MLX_NODE_METAL_ENABLED
    // Preserve output meaning on every replay; unsupported views are copied,
    // unsupported shapes/dtypes fail before allocation or GPU writes.
    if(!supported(inputs,residual_) || !available())throw std::invalid_argument("source mixer replay contract/device");
    auto s=stream();auto& d=metal::device(s.device);auto& enc=metal::get_command_encoder(s);
    auto a=contiguous(inputs,s,enc);
    auto temporary=[&](Shape shape,Dtype dtype){array x(std::move(shape),dtype,nullptr,{});x.set_data(allocator::malloc(x.nbytes()));enc.add_temporary(x);return x;};
    auto table=temporary({Rows,K},bfloat16),sums=temporary({SumWords},float32),
         counters=temporary({Counters},uint32),partials=temporary({Splits,Rows,N},float32);
    out.set_data(allocator::malloc(out.nbytes()));
    prepare(d,enc,a[0],table,0,sums,0,counters,0);
    enc.set_compute_pipeline_state(kernel(d,residual_?"source_mixer_q4_residual":"source_mixer_q4_plain"));
    enc.set_input_array(table,0);enc.set_input_array(sums,1);enc.set_input_array(a[1],2);
    enc.set_input_array(a[2],3);enc.set_input_array(a[3],4);enc.set_output_array(out,5);
    enc.set_output_array(partials,6);enc.set_output_array(counters,7);enc.set_input_array(residual_?a[4]:a[0],8);
    enc.dispatch_threadgroups(MTL::Size(N/64,Splits,1),MTL::Size(128,1,1));
    bridge_testing::record(residual_?"source_mixer_q4_residual":"source_mixer_q4_plain");
#else
    throw std::runtime_error("source mixer requires Metal");
#endif
  }
 private:bool residual_;
};
class PrepareProbe final:public UnaryPrimitive {
 public:
  explicit PrepareProbe(Stream s):UnaryPrimitive(s){}
  DEFINE_NAME(PrepareProbe)
  bool is_equivalent(const Primitive&)const override{return true;}
  static bool valid(const std::vector<array>& a){return a.size()==1 && a[0].shape()==Shape{Rows,K} && a[0].dtype()==bfloat16;}
  std::vector<Shape> output_shapes(const std::vector<array>& a)override{
    if(!valid(a))throw std::invalid_argument("source mixer prepare replay contract");return {{TableWords+SumWords+Counters}};
  }
  void eval_cpu(const std::vector<array>&,array&)override{throw std::runtime_error("source mixer prepare requires Metal");}
  void eval_gpu(const std::vector<array>& inputs,array& out)override {
#ifdef MLX_NODE_METAL_ENABLED
    if(!valid(inputs)||!available())throw std::invalid_argument("source mixer prepare replay contract/device");
    auto s=stream();auto& d=metal::device(s.device);auto& enc=metal::get_command_encoder(s);auto a=contiguous(inputs,s,enc);
    out.set_data(allocator::malloc(out.nbytes()));
    prepare(d,enc,a[0],out,0,out,TableWords*sizeof(uint32_t),out,(TableWords+SumWords)*sizeof(uint32_t));
#else
    throw std::runtime_error("source mixer prepare requires Metal");
#endif
  }
};
}
extern "C" bool mlx_source_mixer_q4_available(){try{return mlx::core::source_mixer::available();}catch(...){return false;}}
extern "C" mlx_array* mlx_source_mixer_q4(mlx_array* x,mlx_array* w,mlx_array* scales,mlx_array* bases,mlx_array* residual){
  try{
    using namespace mlx::core;
    if(!x||!w||!scales||!bases||!source_mixer::available())return nullptr;
    std::vector<array>a{*reinterpret_cast<array*>(x),*reinterpret_cast<array*>(w),*reinterpret_cast<array*>(scales),*reinterpret_cast<array*>(bases)};
    if(residual)a.push_back(*reinterpret_cast<array*>(residual));
    if(!source_mixer::supported(a,residual!=nullptr))return nullptr;
    return reinterpret_cast<mlx_array*>(new array(Shape{8,2048},bfloat16,
        std::make_shared<source_mixer::SourceMixer>(to_stream(Device::gpu),residual!=nullptr),std::move(a)));
  }catch(const std::exception& e){std::cerr<<"source mixer: "<<e.what()<<'\n';return nullptr;}
}
extern "C" mlx_array* mlx_test_source_mixer_q4_prepare(mlx_array* x){
  try{
    using namespace mlx::core;if(!x||!source_mixer::available())return nullptr;std::vector<array>a{*reinterpret_cast<array*>(x)};
    if(!source_mixer::PrepareProbe::valid(a))return nullptr;
    return reinterpret_cast<mlx_array*>(new array(Shape{source_mixer::TableWords+source_mixer::SumWords+source_mixer::Counters},uint32,
        std::make_shared<source_mixer::PrepareProbe>(to_stream(Device::gpu)),std::move(a)));
  }catch(const std::exception& e){std::cerr<<"source mixer prepare: "<<e.what()<<'\n';return nullptr;}
}
