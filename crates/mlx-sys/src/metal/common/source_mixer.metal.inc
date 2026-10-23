R"METAL(
// Source Q4 mixer diagnostic, adapted from Splash 1482f6bbc2abf12375332725302019594150b6cb.
// Apache-2.0. Original register/Table16/split arithmetic, native lossless T64 planes.
#pragma clang fp reassociate(off)
#include <metal_stdlib>
using namespace metal;
#define GGUF_TABLE16_SPAN_INPUTS 64u
#define GGUF_TABLE16_SPAN_VALUES 512u
#define GGUF_TABLE16_SPAN_SEEDS 32u
#define GGUF_TABLE16_SPAN_SUMS 16u
inline ulong table16_sums32_offset(uint width) { return ulong(width/64)*32; }
inline ulong table16_sums_per_tile(uint width) { return ulong(width/64)*48; }
using namespace metal;

// The 8x8 simdgroup_matrix MMA on operands held in plain registers, shared by
// the register matrix kernels (decode/linear_gguf_sgmatrix.metal,
// shared/gguf_float.metal).
namespace sgmatrix {
// simdgroup_matrix lane -> element mapping (verified by the driver's probe):
// a lane's thread_elements() are M[fm][fn] and M[fm][fn + 1].
struct Lane {
  ushort fm;
  ushort fn;
};
inline Lane lane_map(uint lane) {
  const uint qid = lane >> 2;
  Lane l;
  l.fm = ushort((qid & 4) | ((lane >> 1) & 3));
  l.fn = ushort(((qid & 2) << 1) | ((lane & 1) << 1));
  return l;
}

template <typename T>
__attribute__((always_inline)) inline thread vec<T, 2> &te(thread simdgroup_matrix<T, 8, 8> &m) {
  return reinterpret_cast<thread vec<T, 2> &>(m.thread_elements());
}

// One 8x8x8 MMA on plain-register operands: c += a x b. The simdgroup_matrix
// objects live only inside this call (the MLX steel pattern), so the
// persistent accumulator is an ordinary float2 the compiler keeps in
// registers; a persistent simdgroup_matrix read through thread_elements()
// each group was kept in thread memory instead.
template <typename T>
__attribute__((always_inline)) inline void
mma_acc(thread float2 &c, vec<T, 2> a, vec<T, 2> b) {
  simdgroup_matrix<T, 8, 8> A, B;
  simdgroup_matrix<float, 8, 8> C, D;
  te(A) = a;
  te(B) = b;
  te(C) = c;
  simdgroup_multiply_accumulate(D, A, B, C);
  c = te(D);
}
} // namespace sgmatrix
using namespace metal;

// The eight-row X^T table the Apple9 GGUF register kernel reads
// (LinearInput::Table16, laid out as metal/abi/Gguf.h states). A 64-element
// span occupies 512 bfloat in the xt_offset order; its fragments follow the chunk order of the GGUF
// image (metal/abi/QuantFormat.h): fragment 4q + f holds pair f of every
// chunk of the span's 32-element group q, so fragments 4q + 2h and
// 4q + 2h + 1 cover its 16-element group h. Per row, the sum of every 32
// inputs (formats with a min) and the chain seed of every 16 inputs (formats
// with a zero point) are stored row-minor, so a lane reads its two rows at
// once.
namespace gguf_sg {

// Formats with a zero point (Q6_K, Q3_K) enter the MMA as 160 + code - zero,
// exact in bf16 (160 = 128 + 32, Q6_K's zero point), so each 16-input chain
// starts from -160 times the sum of its inputs: the seed the table stores.
constant constexpr uint kZeroPointOffset = 160;

// The rows of a table: one lane's verify rows.
constant constexpr uint kRows = 8;
static_assert(GGUF_TABLE16_SPAN_VALUES == GGUF_TABLE16_SPAN_INPUTS * kRows, "a span holds every row's inputs");

// A span's values laid out as quads, so the lane (fm, fn) reads fragments
// 4 jq .. 4 jq + 3 for (k' = fm, m = fn, fn + 1) as one vec<T, 8>:
//   offset(j, k', m) = (((j >> 2) 8 + k') 4 + (m >> 1)) 8 + (j & 3) 2 + (m & 1)
inline uint xt_offset(uint j, uint kp, uint m) {
  return (((j >> 2) * 8 + kp) * 4 + (m >> 1)) * 8 + (j & 3) * 2 + (m & 1);
}

// Physical k inside a span -> (fragment j, row k').
inline uint2 klogical(uint k) {
  const uint q = k >> 5, h = (k >> 4) & 1, a = (k >> 2) & 3, b = (k >> 1) & 1, e = k & 1;
  return uint2(4 * q + 2 * h + b, 2 * a + e);
}

// One simdgroup writes one span of one row; the lane holds elements
// 2 lane, 2 lane + 1.
inline void write_input(device bfloat *table, device float *sums, uint width, uint span,
                        uint row, uint lane, bfloat a, bfloat b) {
  const uint2 l = klogical(2 * lane);
  table[span * GGUF_TABLE16_SPAN_VALUES + xt_offset(l.x, l.y, row)] = a;
  table[span * GGUF_TABLE16_SPAN_VALUES + xt_offset(l.x, l.y + 1, row)] = b;
  float s = float(a) + float(b);
  s += simd_shuffle_xor(s, 1u);
  s += simd_shuffle_xor(s, 2u);
  s += simd_shuffle_xor(s, 4u);
  if ((lane & 7) == 0) sums[(span * (GGUF_TABLE16_SPAN_SEEDS / kRows) + lane / 8) * kRows + row] = -float(kZeroPointOffset) * s;
  const float s32 = s + simd_shuffle_xor(s, 8u);
  if ((lane & 15) == 0) sums[table16_sums32_offset(width) + (span * (GGUF_TABLE16_SPAN_SUMS / kRows) + lane / 16) * kRows + row] = s32;
}

struct Table16 {
  static ulong sums_per_tile(uint width) { return table16_sums_per_tile(width); }
  static void write(device bfloat *table, device float *sums, uint width, uint span, uint row,
                    uint lane, bfloat a, bfloat b) {
    write_input(table, sums, width, span, row, lane, a, b);
  }
};

} // namespace gguf_sg
using namespace metal;

// Cross-threadgroup reduction of a K split, shared by the split decode
// kernels. Each of the `splits` threadgroups of one output tile stores its
// fp32 partials, in its kernel's own layout, through a device
// coherent(device) pointer and then arrives at the tile's counter. The last
// one to arrive adds the splits in split order, its own from registers, and
// returns the counter to zero for the next dispatch. The result does not
// depend on scheduling, and no threadgroup waits for another.

// True in every thread of the threadgroup that arrives last.
inline bool split_arrive_last(device atomic_uint *counter, uint splits,
                              uint thread_index, threadgroup uint *arrival) {
  // Every writer publishes its partials before thread zero signals arrival.
  // The device-scope fences pair through the counter.
  threadgroup_barrier(mem_flags::mem_device);
  if (thread_index == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                        thread_scope_device);
    *arrival = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                        thread_scope_device);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  return *arrival == splits - 1;
}

// The splits' partials added in split order: `own` for split `self`,
// `partial(s)` for every other split s.
template <typename T, typename Partial>
inline T split_sum(T own, uint self, uint splits, Partial partial) {
#pragma clang fp reassociate(off)
  T total = T(0);
  for (uint s = 0; s < splits; ++s)
    total += s == self ? own : partial(s);
  return total;
}

// Returns the tile's counter to zero; called by the reducing threadgroup.
inline void split_release(device atomic_uint *counter, uint thread_index) {
  if (thread_index == 0)
    atomic_store_explicit(counter, 0u, memory_order_relaxed);
}
// Source prepare, with a disjoint counter initialization in the same dispatch.
// Packing/sum arithmetic is unchanged; no native SG8 layout is consumed.
kernel void source_mixer_q4_prepare(device const bfloat* input [[buffer(0)]],
    device bfloat* table [[buffer(1)]],device float* sums [[buffer(2)]],
    device atomic_uint* counters [[buffer(3)]],uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint K=4096;
  const uint span=(tg*4+sg)/8,row=(tg*4+sg)%8,k=span*64+2*lane;
  gguf_sg::write_input(table,sums,K,span,row,lane,input[row*K+k],input[row*K+k+1]);
  if(tg==0 && tid<32) atomic_store_explicit(counters+tid,0u,memory_order_relaxed);
}
namespace source_mixer {
// Native companion layout [N/64][K/256][64][16 byte pairs or2 half bases].
// The expanded byte pairs have exactly the values of source quant_k4_coef.
struct CoefSource { half2 d; uchar2 sm; };
inline CoefSource coefficient_source(device const uchar* scales,device const half* bases,
    uint column,uint unit,uint j) {
  const ulong at=(ulong(column/64)*16+unit)*64+column%64;
  return {*reinterpret_cast<device const half2*>(bases+at*2),
          *reinterpret_cast<device const uchar2*>(scales+at*16+2*j)};
}
inline float2 coefficient(CoefSource src) {
  const float scale=float(src.d.x)*float(src.sm.x);
  const float minimum=-float(src.d.y)*float(src.sm.y);
  return float2(scale,fma(-128.0f,scale,minimum));
}
inline uint chunk(device const uint* w,uint column,uint group,uint c) {
  const ulong at=((ulong(column/64)*128+group)*64+column%64)*4;
  // Native words contain consecutive codes. Source chunk c owns four from
  // each 16-input half, pair-interleaved by quant_slot/packPairs.
  const uint lo=(w[at+c/2]>>(16*(c%2)))&0xFFFFu;
  const uint hi=(w[at+2+c/2]>>(16*(c%2)))&0xFFFFu;
  const uint v=lo|(hi<<16);
  const uint even=(v&0xFu)|((v>>4)&0xF0u)|((v>>8)&0xF00u)|((v>>12)&0xF000u);
  const uint odd=((v>>4)&0xFu)|((v>>8)&0xF0u)|((v>>12)&0xF00u)|((v>>16)&0xF000u);
  return even|(odd<<16);
}
inline bfloat2 operand(uint word,uint f) {
  const uint pair=(word>>(4*f))&0x000F000Fu;
  return as_type<bfloat2>(pair+0x43004300u);
}
// Source gguf_sg::decode<FmtQ4K,1,Ep,bfloat>, retaining64 columns,
//8 K splits, original Table16, FMA order and ordered last-arrival reduction.
template<bool Residual>
inline void decode(device const bfloat* table,device const float* sums,
    device const uint* w,device const uchar* scales,device const half* bases,
    device bfloat* out,device coherent(device) float* partials,
    device atomic_uint* counters,device const bfloat* residual,
    uint2 tg,uint tid,uint sg,uint lane,threadgroup float2* coefs,threadgroup uint* arrival) {
  constexpr uint K=4096,N=2048,Splits=8,NC=16,UnitSpans=4,J=8;
  const uint u0=tg.y*2,u1=(tg.y+1)*2;
  const sgmatrix::Lane l=sgmatrix::lane_map(lane);
  const uint fm=l.fm,fn=l.fn,c=fn/2,base=tg.x*64+sg*NC;
  threadgroup float2* cu=coefs+sg*NC*J;
  uint group=2*u0*UnitSpans;
  uint cur[2][2],nxt[2][2];
  const auto load=[&](thread uint (&ch)[2][2]) __attribute__((always_inline)) {
#pragma unroll
    for(uint nf=0;nf<2;++nf) {
      ch[nf][0]=chunk(w,base+nf*8+fm,group,c);
      ch[nf][1]=chunk(w,base+nf*8+fm,group+1,c);
    }
  };
  device const vec<bfloat,8>* xt=reinterpret_cast<device const vec<bfloat,8>*>(table+ulong(u0*UnitSpans)*512);
  device const float* s32=sums+table16_sums32_offset(K)+ulong(u0*UnitSpans)*16;
  float2 acc[2]={float2(0),float2(0)};
  load(cur);
  for(uint u=u0;u<u1;++u) {
    CoefSource src[4];
#pragma unroll
    for(uint i=0;i<4;++i) {
      const uint e=lane+32*i;
      src[i]=coefficient_source(scales,bases,base+e%NC,u,e/NC);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for(uint i=0;i<4;++i)cu[lane+32*i]=coefficient(src[i]);
    simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for(uint us=0;us<UnitSpans;++us) {
      const bool more=us+1<UnitSpans||u+1<u1;
      group+=2;
      if(more)load(nxt);
#pragma unroll
      for(uint q=0;q<2;++q) {
        bfloat2 a[2][4];float2 cs[2];
#pragma unroll
        for(uint nf=0;nf<2;++nf) {
#pragma unroll
          for(uint f=0;f<4;++f)a[nf][f]=operand(cur[nf][q],f);
          cs[nf]=cu[(us*2+q)*NC+nf*8+fm];
        }
        const vec<bfloat,8> bq=xt[(8*q+fm)*4+c];
        const float2 sum=*reinterpret_cast<device const float2*>(s32+q*8+fn);
#pragma unroll
        for(uint nf=0;nf<2;++nf) {
          float2 dot(0);
#pragma unroll
          for(uint f=0;f<4;++f)sgmatrix::mma_acc<bfloat>(dot,a[nf][f],reinterpret_cast<thread const bfloat2*>(&bq)[f]);
          acc[nf]=fma(dot,cs[nf].x,acc[nf]);
          acc[nf]=fma(sum,cs[nf].y,acc[nf]);
        }
      }
      xt+=512/8;s32+=16;
      if(more) {
#pragma unroll
        for(uint nf=0;nf<2;++nf)cur[nf][0]=nxt[nf][0],cur[nf][1]=nxt[nf][1];
      }
    }
  }
  device atomic_uint* counter=counters+(base/64);
#pragma unroll
  for(uint nf=0;nf<2;++nf) {
    device coherent(device) float* slot=partials+(ulong(tg.y)*8+fn)*N+base+nf*8+fm;
    slot[0]=acc[nf].x;slot[N]=acc[nf].y;
  }
  if(!split_arrive_last(counter,Splits,tid,arrival))return;
#pragma unroll
  for(uint nf=0;nf<2;++nf)acc[nf]=split_sum(acc[nf],tg.y,Splits,[&](uint s) {
    device coherent(device) float* slot=partials+(ulong(s)*8+fn)*N+base+nf*8+fm;
    return float2(slot[0],slot[N]);
  });
  split_release(counter,tid);
#pragma unroll
  for(uint nf=0;nf<2;++nf) {
    const uint column=base+nf*8+fm;
#pragma unroll
    for(uint i=0;i<2;++i) {
      const ulong at=ulong(fn+i)*N+column;
      float value=acc[nf][i];if constexpr(Residual)value+=float(residual[at]);out[at]=bfloat(value);
    }
  }
}
}
template<bool Residual>
kernel void source_mixer_q4(device const bfloat* table [[buffer(0)]],device const float* sums [[buffer(1)]],
    device const uint* w [[buffer(2)]],device const uchar* scales [[buffer(3)]],device const half* bases [[buffer(4)]],
    device bfloat* out [[buffer(5)]],device coherent(device) float* partials [[buffer(6)]],
    device atomic_uint* counters [[buffer(7)]],device const bfloat* residual [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float2 coefs[64*8];threadgroup uint arrival;
  source_mixer::decode<Residual>(table,sums,w,scales,bases,out,partials,counters,residual,tg,tid,sg,lane,coefs,&arrival);
}
template [[host_name("source_mixer_q4_plain")]] kernel decltype(source_mixer_q4<false>) source_mixer_q4<false>;
template [[host_name("source_mixer_q4_residual")]] kernel decltype(source_mixer_q4<true>) source_mixer_q4<true>;

)METAL"
