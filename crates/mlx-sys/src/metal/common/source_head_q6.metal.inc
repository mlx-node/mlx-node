R"SOURCE_HEAD_Q6(
// Adapted from Splash (Apache-2.0), common/gguf_staged{,_tile}.h and
// shared/gguf_linear.metal, commit 1482f6bbc2abf12375332725302019594150b6cb.
// Exact measured head only: M8/N248320/K2048, source signed-half decode,
// source spread walk and device-A MPP operand, F32 output. Native T64 planes.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
#pragma clang fp reassociate(off)

constant constexpr uint HeadK=2048,HeadN=248320,HeadRows=8,HeadStep=32;
struct HeadQ6Codes { uint4 a; uint2 b; };
inline HeadQ6Codes head_codes(device const uint* base,uint unit) {
  device const uint2* p=reinterpret_cast<device const uint2*>(base+unit*64*6);
  return {uint4(p[0],p[1]),p[2]};
}
inline uint2 head_bytes8(HeadQ6Codes w,ushort j) {
  uint lo,hi;
  if(j==0){lo=w.a.x;hi=w.a.y;}
  else if(j==1){lo=(w.a.y>>16)|(w.a.z<<16);hi=w.a.z>>16;}
  else if(j==2){lo=w.a.w;hi=w.b.x;}
  else {lo=(w.b.x>>16)|(w.b.y<<16);hi=w.b.y>>16;}
  uint even=(lo&0x3Fu)|((lo>>4)&0x3F00u)|((lo>>8)&0x3F0000u)|((hi<<20)&0x3F000000u);
  uint odd=((lo>>6)&0x3Fu)|((lo>>10)&0x3F00u)|((lo>>14)&0x030000u)|((hi&15u)<<18)|((hi<<14)&0x3F000000u);
  return uint2(even,odd);
}
struct HeadQ6Meta { uint4 scales; float base; };
inline HeadQ6Meta head_meta(device const char* scales,device const half* bases,uint n,uint unit) {
  uint tile=n/64,row=n%64,sb=unit/8;
  return {*reinterpret_cast<device const uint4*>(scales+(tile*8*64+sb*64+row)*16),
      float(bases[tile*8*64+sb*64+row])};
}
inline half2 head_signed_pair(uint pair,float scale) {
  float2 code=float2(as_type<half2>(pair|0x64006400u)-half2(half(1056)));
  return half2(code*scale);
}
inline void head_stage(HeadQ6Codes codes,HeadQ6Meta meta,uint unit,threadgroup half* dst) {
  uint j=unit%8;
  uint pair=meta.scales[j/2]>>(16*(j%2));
  float2 scale=float2(meta.base*float(as_type<char>(uchar(pair&255u))),
      meta.base*float(as_type<char>(uchar(pair>>8))));
#pragma unroll
  for(ushort c=0;c<4;++c) {
    uint2 q=head_bytes8(codes,c);float s=c<2?scale.x:scale.y;
    half2 p0=head_signed_pair((q.x&255u)|((q.y&255u)<<16),s);
    half2 p1=head_signed_pair(((q.x>>8)&255u)|(((q.y>>8)&255u)<<16),s);
    half2 p2=head_signed_pair(((q.x>>16)&255u)|(((q.y>>16)&255u)<<16),s);
    half2 p3=head_signed_pair((q.x>>24)|((q.y>>24)<<16),s);
    *reinterpret_cast<threadgroup half4*>(dst+8*c)=half4(p0,p1);
    *reinterpret_cast<threadgroup half4*>(dst+8*c+4)=half4(p2,p3);
  }
}
kernel void source_head_q6_m8_f32(
    device bfloat* input [[buffer(0)]],device const uint* weights [[buffer(1)]],
    device const char* scales [[buffer(2)]],device const half* bases [[buffer(3)]],
    device float* output [[buffer(4)]],uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup half stage[2*2*32*32];
  threadgroup half* my=stage+sg*2*32*32;
  const uint origin=tile*64+sg*32,n=origin+lane;
  device const uint* row=weights+((n/64)*64*64+n%64)*6;
  const uint first=uint(ulong((tile*40503u)&0xFFFFu)*64>>16);
  auto a=tensor(input,dextents<int,2>{HeadK,HeadRows},array<int,2>{1,HeadK});
  constexpr auto descriptor=matmul2d_descriptor(8,32,32,false,true,false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor,execution_simdgroups<1>> operation;
  auto a0=a.template slice<32,8>(0,0);
  tensor<threadgroup half,dextents<int,2>,tensor_inline> bt0(my,dextents<int,2>{32,32},array<int,2>{1,32});
  tensor<threadgroup half,dextents<int,2>,tensor_inline> bt1(my+32*32,dextents<int,2>{32,32},array<int,2>{1,32});
  auto b0=bt0.slice<32,32>(0,0),b1=bt1.slice<32,32>(0,0);
  auto acc=operation.template get_destination_cooperative_tensor<decltype(a0),decltype(b0),float>();
#pragma unroll
  for(ushort i=0;i<acc.get_capacity();++i)acc[i]=0.0f;
  HeadQ6Codes packed=head_codes(row,first);HeadQ6Meta meta=head_meta(scales,bases,n,first);
  const auto run_step=[&](uint step,uint next,bool more,uint buffer) __attribute__((always_inline)) {
    head_stage(packed,meta,step,my+buffer*32*32+lane*32);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if(more) {
      packed=head_codes(row,next);
      if(next/8!=step/8)meta=head_meta(scales,bases,n,next);
    }
    auto ax=a.template slice<32,8>(step*32,0);
    if(buffer)operation.run(ax,b1,acc);else operation.run(ax,b0,acc);
  };
  if(first==0) {
    for(uint step=0;step<64;++step)run_step(step,step+1,step+1<64,step&1);
  } else {
    uint walked=0;
    for(uint run=0;run<2;++run) {
      uint from=run?0:first,to=run?first:64;
      for(uint step=from;step<to;++step,++walked)
        run_step(step,step+1<to?step+1:0,walked+1<64,walked&1);
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
  for(ushort i=0;i<acc.get_capacity();++i) {
    if(!acc.is_valid_element(i))continue;
    auto index=acc.get_multidimensional_index(i);
    output[uint(index[1])*HeadN+origin+uint(index[0])]=float(acc[i]);
  }
}

)SOURCE_HEAD_Q6"
