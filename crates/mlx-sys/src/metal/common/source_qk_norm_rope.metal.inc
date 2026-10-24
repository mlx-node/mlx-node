R"METAL(
// Adapted from Splash1482f6b attention_qkv_prepare.h, rms_inverse.h, target_rope.h. Apache-2.0; off-tree research.
#include <metal_stdlib>
using namespace metal;

constant constexpr float kRmsEpsilon = 1e-6;

// The inverse RMS of `width` values from each thread's sum of squares,
// reduced per simdgroup and then over the first Simdgroups simdgroups, in
// order, by thread 0, and returned to every thread. `reductions` holds a
// partial per simdgroup. This and row_squares keep the source order of their
// float additions (reassociate(off), scoped to their bodies), whichever file
// includes them.
template <ushort Simdgroups = 8>
inline float rms_inverse_of_sums(float sum, uint width, threadgroup float *reductions,
                                 uint thread_index, uint lane, uint simd_group) {
#pragma clang fp reassociate(off)
  sum = simd_sum(sum);
  if (lane == 0)
    reductions[simd_group] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float total = 0.0f;
    for (uint i = 0; i < Simdgroups; ++i)
      total += reductions[i];
    reductions[0] = rsqrt(total / width + kRmsEpsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return reductions[0];
}

// A thread's squares of columns thread_index + 256 i of a row in device or
// threadgroup memory, in column order.
template <class Row>
inline float row_squares(Row row, uint width, uint thread_index) {
#pragma clang fp reassociate(off)
  float sum = 0.0f;
  for (uint column = thread_index; column < width; column += 256) {
    float value = float(row[column]);
    sum += value * value;
  }
  return sum;
}

// Inverse RMS of one row, reduced across the 256-thread group and returned to
// every thread.
inline float rms_inverse(device const bfloat *row_input, uint width,
                         threadgroup float *reductions, uint thread_index,
                         uint lane, uint simd_group) {
  return rms_inverse_of_sums(row_squares(row_input, width, thread_index), width, reductions,
                             thread_index, lane, simd_group);
}


// Replay-local strides are element counts; encoder bindings include each
// array's byte offset. No host position readback or trace-time baked offset.
struct SourceQkParams {
 ulong q_row,q_head,k_row,k_head,wq,wk,frequency;
};
kernel void source_qk_norm_rope_m8(
 device const bfloat* q_input [[buffer(0)]], device const bfloat* k_input [[buffer(1)]],
 device const float* wq [[buffer(2)]], device const float* wk [[buffer(3)]],
 device const float* inverse_frequencies [[buffer(4)]], device const int* offsets [[buffer(5)]],
 device bfloat* q [[buffer(6)]], device bfloat* k [[buffer(7)]],
 constant SourceQkParams& p [[buffer(8)]], uint task [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
 const bool query=task<8*16;
 const uint local=query?task:task-8*16;
 const uint heads=query?16:2,row=local/heads,head=local%heads;
 const device bfloat* src=query?q_input+ulong(row)*p.q_row+ulong(head)*p.q_head
     :k_input+ulong(row)*p.k_row+ulong(head)*p.k_head;
 const device float* weight=query?wq:wk;
 const ulong weight_stride=query?p.wq:p.wk;
 device bfloat* dst=(query?q:k)+(ulong(head)*8+row)*256;
 threadgroup float sums[8];threadgroup bfloat normalized[256];
 const float x=float(src[tid]);
 const float inv=rms_inverse_of_sums(x*x,256,sums,tid,lane,sg);
 normalized[tid]=bfloat(x*inv*weight[tid*weight_stride]);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(tid<32){
  const float first=float(normalized[tid]),second=float(normalized[tid+32]);
  // Source uses unsigned position+row. Product logical offsets are nonnegative.
  const float angle=float(uint(*offsets)+row)*inverse_frequencies[tid*p.frequency];
  const float c=cos(angle),s=sin(angle);
  dst[tid]=bfloat(first*c-second*s);dst[tid+32]=bfloat(second*c+first*s);
 } else if(tid>=64)dst[tid]=normalized[tid];
}

)METAL"
